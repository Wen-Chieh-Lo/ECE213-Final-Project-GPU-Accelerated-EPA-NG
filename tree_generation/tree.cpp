#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include <CLI/CLI.hpp>
#ifdef MLIPPER_USE_VENDOR_PLL
#include "pll.h"
#else
#include <libpll/pll.h>
#endif
#include <cuda_runtime.h>

#include "precision.hpp"
#include "root_likelihood.cuh"
#include "tree.hpp"
#include "tree_placement.cuh"
#include "parse_file.hpp"
#include "seq_preproc.hpp"
#include "../mlipper_util.h"

namespace {

// ----- File and path helpers -----

std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void validate_newick_with_pll(
    const std::string& tree_text,
    const std::string& option_name)
{
    if (tree_text.empty()) {
        throw CLI::ValidationError(option_name, "tree is empty");
    }

    std::filesystem::path tree_path_template =
        std::filesystem::temp_directory_path() / "mlipper-validate-XXXXXX.nwk";
    std::string tree_path_string = tree_path_template.string();
    std::vector<char> tree_path_buffer(
        tree_path_string.begin(),
        tree_path_string.end());
    tree_path_buffer.push_back('\0');

    const int tree_fd = mkstemps(tree_path_buffer.data(), 4);
    if (tree_fd < 0) {
        throw CLI::ValidationError(option_name, "failed to create temporary file for tree validation");
    }
    close(tree_fd);

    try {
        std::ofstream ofs(tree_path_buffer.data(), std::ios::trunc);
        if (!ofs) {
            throw CLI::ValidationError(option_name, "failed to open temporary file for tree validation");
        }
        ofs << tree_text;
        ofs.close();

        pll_rtree_t* rtree = pll_rtree_parse_newick(tree_path_buffer.data());
        std::remove(tree_path_buffer.data());
        if (!rtree) {
            throw CLI::ValidationError(option_name, "invalid Newick syntax");
        }
        pll_rtree_destroy(rtree, nullptr);
    } catch (...) {
        std::remove(tree_path_buffer.data());
        throw;
    }
}

struct ClvDumpHeader {
    char magic[8];
    uint32_t version;
    uint32_t states;
    uint32_t rate_cats;
    uint32_t sites;
    uint32_t per_rate_scaling;
    uint32_t record_count;
};

static uint64_t fnv1a_update(uint64_t hash, const char* data, size_t len) {
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    if (hash == 0) hash = kOffset;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
        hash *= kPrime;
    }
    return hash;
}

static uint64_t hash_leaf_list(std::vector<std::string> leaves) {
    std::sort(leaves.begin(), leaves.end());
    uint64_t h = 0;
    for (const auto& name : leaves) {
        h = fnv1a_update(h, name.data(), name.size());
        const char sep = '\n';
        h = fnv1a_update(h, &sep, 1);
    }
    return h;
}

static void collect_leaf_sets(
    const TreeBuildResult& tree,
    int node_id,
    std::vector<std::vector<std::string>>& leaf_sets)
{
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) return;
    if (!leaf_sets[node_id].empty()) return;
    const TreeNode& node = tree.nodes[(size_t)node_id];
    if (node.is_tip) {
        const std::string name = node.name.empty()
            ? ("tip_" + std::to_string(node.id))
            : node.name;
        leaf_sets[node_id].push_back(name);
        return;
    }
    if (node.left >= 0) {
        collect_leaf_sets(tree, node.left, leaf_sets);
        leaf_sets[node_id].insert(
            leaf_sets[node_id].end(),
            leaf_sets[node.left].begin(),
            leaf_sets[node.left].end());
    }
    if (node.right >= 0) {
        collect_leaf_sets(tree, node.right, leaf_sets);
        leaf_sets[node_id].insert(
            leaf_sets[node_id].end(),
            leaf_sets[node.right].begin(),
            leaf_sets[node.right].end());
    }
    std::sort(leaf_sets[node_id].begin(), leaf_sets[node_id].end());
}

static void write_u32(std::ofstream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_i32(std::ofstream& out, int32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_u64(std::ofstream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_f64(std::ofstream& out, double value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_clv_dump(
    const std::string& path,
    const TreeBuildResult& tree,
    const DeviceTree& D,
    const std::vector<fp_t>& clv_up,
    const std::vector<unsigned>& scaler_up)
{
    if (tree.nodes.empty() || D.N <= 0) {
        throw std::runtime_error("write_clv_dump: empty tree or device.");
    }
    const size_t per_node = D.per_node_elems();
    if (per_node == 0) {
        throw std::runtime_error("write_clv_dump: invalid per-node size.");
    }

    const size_t scaler_span = D.per_rate_scaling
        ? static_cast<size_t>(D.sites) * static_cast<size_t>(D.rate_cats)
        : static_cast<size_t>(D.sites);

    std::vector<std::vector<std::string>> leaf_sets(tree.nodes.size());
    collect_leaf_sets(tree, tree.root_id, leaf_sets);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("write_clv_dump: cannot open output " + path);
    }

    ClvDumpHeader header{};
    std::memcpy(header.magic, "CLVDUMP1", 8);
    header.version = 1;
    header.states = static_cast<uint32_t>(D.states);
    header.rate_cats = static_cast<uint32_t>(D.rate_cats);
    header.sites = static_cast<uint32_t>(D.sites);
    header.per_rate_scaling = D.per_rate_scaling ? 1u : 0u;
    header.record_count = static_cast<uint32_t>(tree.nodes.size());
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const size_t clv_len = per_node;
    const size_t scaler_len = scaler_span;

    for (size_t node_id = 0; node_id < tree.nodes.size(); ++node_id) {
        const TreeNode& node = tree.nodes[node_id];
        const std::vector<std::string>& leaves = leaf_sets[node_id];
        const uint64_t leaf_hash = hash_leaf_list(leaves);
        const uint32_t leaf_count = static_cast<uint32_t>(leaves.size());
        const bool is_tip = node.is_tip;
        const std::string name = is_tip
            ? (node.name.empty() ? ("tip_" + std::to_string(node.id)) : node.name)
            : std::string();

        write_u64(out, leaf_hash);
        write_u32(out, leaf_count);
        write_u32(out, static_cast<uint32_t>(node_id));
        write_i32(out, static_cast<int32_t>(node_id));
        write_i32(out, static_cast<int32_t>(node_id));
        write_u32(out, is_tip ? 1u : 0u);
        write_u32(out, 0u); // pool=0 (up)
        write_u32(out, static_cast<uint32_t>(name.size()));
        write_u32(out, static_cast<uint32_t>(clv_len));
        write_u32(out, static_cast<uint32_t>(scaler_len));
        if (!name.empty()) {
            out.write(name.data(), static_cast<std::streamsize>(name.size()));
        }

        const size_t clv_off = node_id * per_node;
        for (size_t i = 0; i < per_node; ++i) {
            write_f64(out, static_cast<double>(clv_up[clv_off + i]));
        }
        const size_t scaler_off = node_id * scaler_span;
        for (size_t i = 0; i < scaler_span; ++i) {
            const unsigned v = scaler_up.empty() ? 0u : scaler_up[scaler_off + i];
            write_u32(out, v);
        }
    }

    out.flush();
}

std::string resolve_path(const std::filesystem::path& base, const std::string& p) {
    std::filesystem::path candidate(p);
    if (candidate.empty()) return {};
    if (candidate.is_absolute()) return candidate.string();
    return (base / candidate).string();
}

std::string uppercase_ascii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string preview_name_list(const std::vector<std::string>& names, size_t limit = 5) {
    std::ostringstream oss;
    const size_t count = std::min(limit, names.size());
    for (size_t idx = 0; idx < count; ++idx) {
        if (idx) oss << ", ";
        oss << names[idx];
    }
    if (names.size() > limit) {
        oss << " ... (" << names.size() << " total)";
    }
    return oss.str();
}

bool is_valid_dna4_alignment_char(char c) {
    switch (static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(c)))) {
        case 'A':
        case 'C':
        case 'G':
        case 'T':
        case 'U':
        case 'R':
        case 'Y':
        case 'S':
        case 'W':
        case 'K':
        case 'M':
        case 'B':
        case 'D':
        case 'H':
        case 'V':
        case 'N':
        case '-':
        case '.':
        case '?':
            return true;
        default:
            return false;
    }
}

void validate_alignment_names(const parse::Alignment& alignment, const std::string& option_name) {
    if (alignment.names.size() != alignment.sequences.size()) {
        throw CLI::ValidationError(option_name, "name/sequence count mismatch");
    }

    std::unordered_set<std::string> seen;
    std::vector<std::string> duplicates;
    seen.reserve(alignment.names.size() * 2);
    for (const std::string& name : alignment.names) {
        if (name.empty()) {
            throw CLI::ValidationError(option_name, "contains an empty sequence name");
        }
        if (!seen.insert(name).second) {
            duplicates.push_back(name);
        }
    }
    if (!duplicates.empty()) {
        std::sort(duplicates.begin(), duplicates.end());
        duplicates.erase(std::unique(duplicates.begin(), duplicates.end()), duplicates.end());
        throw CLI::ValidationError(
            option_name,
            "contains duplicate sequence names: " + preview_name_list(duplicates));
    }
}

void validate_alignment_symbols(
    const parse::Alignment& alignment,
    int states,
    const std::string& option_name)
{
    if (states != 4) return;

    for (size_t seq_idx = 0; seq_idx < alignment.sequences.size(); ++seq_idx) {
        const std::string& name = alignment.names[seq_idx];
        const std::string& seq = alignment.sequences[seq_idx];
        for (size_t site_idx = 0; site_idx < seq.size(); ++site_idx) {
            const char c = seq[site_idx];
            if (is_valid_dna4_alignment_char(c)) continue;
            std::ostringstream oss;
            oss << "sequence '" << name << "' has unsupported DNA symbol '"
                << c << "' at site " << (site_idx + 1);
            throw CLI::ValidationError(option_name, oss.str());
        }
    }
}

void validate_positive_vector(
    const std::vector<double>& values,
    const std::string& option_name,
    const char* what)
{
    double sum = 0.0;
    for (double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw CLI::ValidationError(option_name, std::string(what) + " must be finite and > 0");
        }
        sum += value;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw CLI::ValidationError(option_name, std::string(what) + " must sum to a positive finite value");
    }
}

void validate_model_inputs(const parse::ModelConfig& model) {
    constexpr int kMaxRateCats = 8;
    if (model.states != 4) {
        throw CLI::ValidationError("--states", "currently only 4-state DNA input is supported");
    }
    if (model.ncat <= 0 || model.ncat > kMaxRateCats) {
        throw CLI::ValidationError(
            "--ncat",
            "must be between 1 and " + std::to_string(kMaxRateCats));
    }
    if (uppercase_ascii(model.subst_model) != "GTR") {
        throw CLI::ValidationError("--subst-model", "currently only GTR is supported");
    }
    if (!std::isfinite(model.alpha) || model.alpha <= 0.0) {
        throw CLI::ValidationError("--alpha", "must be finite and > 0");
    }
    if (!std::isfinite(model.pinv) || model.pinv < 0.0 || model.pinv > 1.0) {
        throw CLI::ValidationError("--pinv", "must be finite and between 0 and 1");
    }
    if (!model.freqs.empty()) {
        if (static_cast<int>(model.freqs.size()) != model.states) {
            throw CLI::ValidationError(
                "--freqs",
                "must have exactly " + std::to_string(model.states) + " values (states)");
        }
        validate_positive_vector(model.freqs, "--freqs", "equilibrium frequencies");
    }
    if (model.rates.size() != 6) {
        throw CLI::ValidationError("--rates", "must have exactly 6 values for 4-state GTR");
    }
    validate_positive_vector(model.rates, "--rates", "GTR rates");
    if (!model.rate_weights.empty()) {
        if (static_cast<int>(model.rate_weights.size()) != model.ncat) {
            throw CLI::ValidationError(
                "--rate-weights",
                "must have exactly " + std::to_string(model.ncat) + " values (ncat)");
        }
        validate_positive_vector(model.rate_weights, "--rate-weights", "rate weights");
    }
}

void validate_query_reference_name_overlap(
    const parse::Alignment& tree_alignment,
    const parse::Alignment& query_alignment,
    const std::string& option_name)
{
    std::unordered_set<std::string> tree_names;
    tree_names.reserve(tree_alignment.names.size() * 2);
    for (const std::string& name : tree_alignment.names) {
        tree_names.insert(name);
    }

    std::vector<std::string> overlaps;
    for (const std::string& name : query_alignment.names) {
        if (tree_names.count(name)) {
            overlaps.push_back(name);
        }
    }
    if (!overlaps.empty()) {
        std::sort(overlaps.begin(), overlaps.end());
        overlaps.erase(std::unique(overlaps.begin(), overlaps.end()), overlaps.end());
        throw CLI::ValidationError(
            option_name,
            "query names overlap reference tip names, which would cause ambiguous committed tips: " +
                preview_name_list(overlaps));
    }
}

std::filesystem::path normalize_cli_path(
    const std::filesystem::path& base,
    const std::string& raw_path)
{
    if (raw_path.empty()) return {};
    std::filesystem::path path = std::filesystem::path(resolve_path(base, raw_path));
    if (!path.is_absolute()) {
        path = std::filesystem::absolute(path);
    }
    return path.lexically_normal();
}

void validate_output_path(
    const std::filesystem::path& base,
    const std::string& option_name,
    const std::string& raw_path)
{
    const std::filesystem::path path = normalize_cli_path(base, raw_path);
    if (path.empty() || path.filename().empty()) {
        throw CLI::ValidationError(option_name, "output path must name a file");
    }

    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        if (ec) {
            throw CLI::ValidationError(option_name, "failed to inspect output path");
        }
        if (std::filesystem::is_directory(path, ec)) {
            throw CLI::ValidationError(option_name, "output path points to a directory");
        }
        if (ec) {
            throw CLI::ValidationError(option_name, "failed to inspect output path");
        }
    }

    std::filesystem::path ancestor = path.parent_path();
    while (!ancestor.empty()) {
        const bool exists = std::filesystem::exists(ancestor, ec);
        if (ec) {
            throw CLI::ValidationError(option_name, "failed to inspect output parent path");
        }
        if (exists) {
            if (!std::filesystem::is_directory(ancestor, ec) || ec) {
                throw CLI::ValidationError(
                    option_name,
                    "output parent path is not a directory: " + ancestor.string());
            }
            break;
        }
        ancestor = ancestor.parent_path();
    }
}

std::string find_any_tip_name_not_in(
    const TreeBuildResult& tree,
    int node_id,
    const std::unordered_set<std::string>& blocked)
{
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) return {};
    const TreeNode& node = tree.nodes[node_id];
    if (node.is_tip) {
        return blocked.count(node.name) ? std::string{} : node.name;
    }
    std::vector<int> queue;
    queue.push_back(node_id);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        const TreeNode& cur_node = tree.nodes[cur];
        if (cur_node.is_tip) {
            if (!blocked.count(cur_node.name)) return cur_node.name;
            continue;
        }
        if (cur_node.left >= 0) queue.push_back(cur_node.left);
        if (cur_node.right >= 0) queue.push_back(cur_node.right);
    }
    return {};
}

std::vector<int> bfs_distances(const TreeBuildResult& tree, int start);

struct PruneInfo {
    int pruned_id = -1;
    int free_internal_id = -1;
    int sibling_id = -1;
    int grandparent_id = -1;
    fp_t pruned_branch_length = fp_t(0);
};

struct LocalSprInsertionAnchor {
    std::string query_name;
    int tip_id = -1;
    int anchor_id = -1;
};

struct LocalSprRepairUnit {
    int unit_id = -1;
    std::vector<std::string> query_names;
    std::vector<int> anchor_ids;
    std::vector<char> envelope_mask;
    std::vector<int> envelope_nodes;
};

struct LocalSprCandidateMove {
    int repair_unit_id = -1;
    int prune_root_id = -1;
    int regraft_child_id = -1;
    int regraft_parent_id = -1;
    int old_parent_id = -1;
    double approx_gain = -std::numeric_limits<double>::infinity();
    double pendant_length = 0.0;
    double proximal_length = 0.0;
    std::vector<int> subtree_nodes;
    std::vector<int> regraft_path_nodes;
};

struct LocalSprSearchSummary {
    size_t unit_count = 0;
    size_t enumerated_candidates = 0;
    size_t retained_candidates = 0;
    size_t selected_candidates = 0;
};

struct LocalSprPruneRootWorkItem {
    int prune_root_id = -1;
    int skeleton_distance = std::numeric_limits<int>::max();
    int legal_inner_edge_count = 0;
    int subtree_size = 0;
};

struct IntDisjointSet {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit IntDisjointSet(int n) : parent((size_t)n), rank((size_t)n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[(size_t)x] != x) {
            parent[(size_t)x] = find(parent[(size_t)x]);
        }
        return parent[(size_t)x];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rank[(size_t)ra] < rank[(size_t)rb]) std::swap(ra, rb);
        parent[(size_t)rb] = ra;
        if (rank[(size_t)ra] == rank[(size_t)rb]) {
            ++rank[(size_t)ra];
        }
    }
};

static void rebuild_traversals(TreeBuildResult& T) {
    T.preorder.clear();
    T.postorder.clear();
    if (T.root_id < 0 || T.root_id >= static_cast<int>(T.nodes.size())) return;

    std::vector<int> stack;
    stack.push_back(T.root_id);
    while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        T.preorder.push_back(id);
        const TreeNode& node = T.nodes[static_cast<size_t>(id)];
        if (!node.is_tip) {
            if (node.right >= 0) stack.push_back(node.right);
            if (node.left >= 0) stack.push_back(node.left);
        }
    }

    std::vector<std::pair<int, bool>> st;
    st.emplace_back(T.root_id, false);
    while (!st.empty()) {
        const auto cur = st.back();
        st.pop_back();
        const int id = cur.first;
        const bool visited = cur.second;
        if (id < 0) continue;
        const TreeNode& node = T.nodes[static_cast<size_t>(id)];
        if (visited) {
            T.postorder.push_back(id);
        } else {
            st.emplace_back(id, true);
            if (!node.is_tip) {
                if (node.right >= 0) st.emplace_back(node.right, false);
                if (node.left >= 0) st.emplace_back(node.left, false);
            }
        }
    }
}

[[noreturn]] void local_spr_fail(const std::string& message)
{
    throw std::runtime_error("Local SPR subtree assertion failed: " + message);
}

void local_spr_assert(bool condition, const std::string& message)
{
    if (!condition) {
        local_spr_fail(message);
    }
}

void collect_subtree_node_ids(
    const TreeBuildResult& tree,
    int node_id,
    std::vector<int>& node_ids)
{
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) return;
    node_ids.clear();
    std::vector<int> stack;
    stack.push_back(node_id);
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        if (cur < 0 || cur >= static_cast<int>(tree.nodes.size())) continue;
        node_ids.push_back(cur);
        const TreeNode& node = tree.nodes[(size_t)cur];
        if (node.right >= 0) stack.push_back(node.right);
        if (node.left >= 0) stack.push_back(node.left);
    }
}

bool subtree_fully_inside_mask(
    const TreeBuildResult& tree,
    int node_id,
    const std::vector<char>& mask,
    std::vector<int>* node_ids_out = nullptr)
{
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) return false;
    if (mask.empty()) return false;

    std::vector<int> node_ids;
    std::vector<int>* sink = node_ids_out ? node_ids_out : &node_ids;
    collect_subtree_node_ids(tree, node_id, *sink);
    for (int cur : *sink) {
        if (cur < 0 || cur >= static_cast<int>(mask.size()) || !mask[(size_t)cur]) {
            return false;
        }
    }
    return true;
}

void rebuild_host_topology_from_tree_local(
    const TreeBuildResult& tree,
    HostPacking& host)
{
    host.postorder = tree.postorder;
    host.preorder = tree.preorder;
    host.parent.resize(tree.nodes.size(), -1);
    host.left.resize(tree.nodes.size(), -1);
    host.right.resize(tree.nodes.size(), -1);
    host.is_tip.resize(tree.nodes.size(), 0);
    host.blen.resize(tree.nodes.size(), fp_t(0));
    for (size_t node_idx = 0; node_idx < tree.nodes.size(); ++node_idx) {
        const TreeNode& node = tree.nodes[node_idx];
        host.parent[node_idx] = node.parent;
        host.left[node_idx] = node.left;
        host.right[node_idx] = node.right;
        host.is_tip[node_idx] = node.is_tip ? 1 : 0;
        host.blen[node_idx] = node.branch_length_to_parent;
    }
}

static inline void append_selected_downward_op(
    std::vector<NodeOpInfo>& ops,
    int parent_id,
    int left_id,
    int right_id,
    bool left_is_tip,
    bool right_is_tip,
    const std::vector<int>& node_to_tip,
    uint8_t dir_tag)
{
    NodeOpInfo op{};
    op.parent_id = parent_id;
    op.left_id = left_id;
    op.right_id = right_id;
    op.left_tip_index = left_is_tip ? node_to_tip[static_cast<size_t>(left_id)] : -1;
    op.right_tip_index = right_is_tip ? node_to_tip[static_cast<size_t>(right_id)] : -1;
    op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
    op.dir_tag = dir_tag;

    const bool target_is_left = (dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_tip = target_is_left ? left_is_tip : right_is_tip;
    const bool sibling_is_tip = target_is_left ? right_is_tip : left_is_tip;
    if (target_is_tip && sibling_is_tip) {
        op.op_type = static_cast<int>(OP_DOWN_TIP_TIP);
    } else if (target_is_tip) {
        op.op_type = static_cast<int>(OP_DOWN_TIP_INNER);
    } else if (sibling_is_tip) {
        op.op_type = static_cast<int>(OP_DOWN_INNER_TIP);
    } else {
        op.op_type = static_cast<int>(OP_DOWN_INNER_INNER);
    }
    ops.push_back(op);
}

void build_selected_downward_ops(
    const TreeBuildResult& tree,
    const HostPacking& host,
    const std::vector<int>& target_child_ids,
    std::vector<NodeOpInfo>& host_ops)
{
    std::vector<int> node_to_tip(tree.nodes.size(), -1);
    for (int tip_idx = 0; tip_idx < static_cast<int>(host.tip_node_ids.size()); ++tip_idx) {
        const int node_id = host.tip_node_ids[static_cast<size_t>(tip_idx)];
        if (node_id >= 0 && node_id < static_cast<int>(node_to_tip.size())) {
            node_to_tip[static_cast<size_t>(node_id)] = tip_idx;
        }
    }

    host_ops.clear();
    host_ops.reserve(target_child_ids.size());
    std::unordered_set<int> seen;
    for (int target_child_id : target_child_ids) {
        if (!seen.insert(target_child_id).second) continue;
        if (target_child_id < 0 || target_child_id >= static_cast<int>(tree.nodes.size())) continue;
        const int parent_id = tree.nodes[static_cast<size_t>(target_child_id)].parent;
        if (parent_id < 0 || parent_id >= static_cast<int>(tree.nodes.size())) continue;
        const TreeNode& parent = tree.nodes[static_cast<size_t>(parent_id)];
        const int left_id = parent.left;
        const int right_id = parent.right;
        if (left_id < 0 || right_id < 0) continue;

        const bool left_is_tip = tree.nodes[static_cast<size_t>(left_id)].is_tip;
        const bool right_is_tip = tree.nodes[static_cast<size_t>(right_id)].is_tip;
        if (left_id == target_child_id) {
            append_selected_downward_op(
                host_ops,
                parent_id,
                left_id,
                right_id,
                left_is_tip,
                right_is_tip,
                node_to_tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        } else if (right_id == target_child_id) {
            append_selected_downward_op(
                host_ops,
                parent_id,
                left_id,
                right_id,
                left_is_tip,
                right_is_tip,
                node_to_tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        } else {
            local_spr_fail("target child does not hang under its recorded parent");
        }
    }
}

void build_selected_downward_closure_ops(
    const TreeBuildResult& tree,
    const HostPacking& host,
    const std::vector<int>& target_child_ids,
    std::vector<NodeOpInfo>& host_ops)
{
    std::vector<int> node_to_tip(tree.nodes.size(), -1);
    for (int tip_idx = 0; tip_idx < static_cast<int>(host.tip_node_ids.size()); ++tip_idx) {
        const int node_id = host.tip_node_ids[static_cast<size_t>(tip_idx)];
        if (node_id >= 0 && node_id < static_cast<int>(node_to_tip.size())) {
            node_to_tip[static_cast<size_t>(node_id)] = tip_idx;
        }
    }

    std::vector<char> closure_mask(tree.nodes.size(), 0);
    for (int target_child_id : target_child_ids) {
        if (target_child_id < 0 || target_child_id >= static_cast<int>(tree.nodes.size())) continue;
        for (int node_id = target_child_id;
             node_id >= 0 && node_id < static_cast<int>(tree.nodes.size());
             node_id = tree.nodes[static_cast<size_t>(node_id)].parent) {
            const int parent_id = tree.nodes[static_cast<size_t>(node_id)].parent;
            if (parent_id < 0) break;
            closure_mask[static_cast<size_t>(node_id)] = 1;
        }
    }

    host_ops.clear();
    host_ops.reserve(target_child_ids.size() * 4);
    for (int node_id : tree.preorder) {
        if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) continue;
        if (!closure_mask[static_cast<size_t>(node_id)]) continue;

        const int parent_id = tree.nodes[static_cast<size_t>(node_id)].parent;
        if (parent_id < 0 || parent_id >= static_cast<int>(tree.nodes.size())) continue;

        const TreeNode& parent = tree.nodes[static_cast<size_t>(parent_id)];
        const int left_id = parent.left;
        const int right_id = parent.right;
        if (left_id < 0 || right_id < 0) continue;

        const bool left_is_tip = tree.nodes[static_cast<size_t>(left_id)].is_tip;
        const bool right_is_tip = tree.nodes[static_cast<size_t>(right_id)].is_tip;
        if (left_id == node_id) {
            append_selected_downward_op(
                host_ops,
                parent_id,
                left_id,
                right_id,
                left_is_tip,
                right_is_tip,
                node_to_tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        } else if (right_id == node_id) {
            append_selected_downward_op(
                host_ops,
                parent_id,
                left_id,
                right_id,
                left_is_tip,
                right_is_tip,
                node_to_tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        } else {
            local_spr_fail("closure node does not hang under its recorded parent");
        }
    }
}

struct LocalSprNodeOpKey {
    int parent_id = -1;
    int left_id = -1;
    int right_id = -1;
    int left_tip_index = -1;
    int right_tip_index = -1;
    int op_type = OP_TIP_TIP;
    uint8_t clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
    uint8_t dir_tag = static_cast<uint8_t>(CLV_DIR_UP);

    bool operator==(const LocalSprNodeOpKey& other) const
    {
        return parent_id == other.parent_id &&
               left_id == other.left_id &&
               right_id == other.right_id &&
               left_tip_index == other.left_tip_index &&
               right_tip_index == other.right_tip_index &&
               op_type == other.op_type &&
               clv_pool == other.clv_pool &&
               dir_tag == other.dir_tag;
    }
};

struct LocalSprNodeOpKeyHash {
    size_t operator()(const LocalSprNodeOpKey& key) const
    {
        size_t h = static_cast<size_t>(key.parent_id + 0x9e3779b9);
        auto mix = [&](size_t value) {
            h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<size_t>(key.left_id));
        mix(static_cast<size_t>(key.right_id));
        mix(static_cast<size_t>(key.left_tip_index));
        mix(static_cast<size_t>(key.right_tip_index));
        mix(static_cast<size_t>(key.op_type));
        mix(static_cast<size_t>(key.clv_pool));
        mix(static_cast<size_t>(key.dir_tag));
        return h;
    }
};

static LocalSprNodeOpKey make_local_spr_node_op_key(const NodeOpInfo& op)
{
    LocalSprNodeOpKey key;
    key.parent_id = op.parent_id;
    key.left_id = op.left_id;
    key.right_id = op.right_id;
    key.left_tip_index = op.left_tip_index;
    key.right_tip_index = op.right_tip_index;
    key.op_type = op.op_type;
    key.clv_pool = op.clv_pool;
    key.dir_tag = op.dir_tag;
    return key;
}

static std::vector<NodeOpInfo> filter_local_spr_new_ops(
    const std::vector<NodeOpInfo>& ops,
    const std::vector<NodeOpInfo>& already_computed_ops)
{
    std::unordered_set<LocalSprNodeOpKey, LocalSprNodeOpKeyHash> seen;
    seen.reserve(already_computed_ops.size() * 2 + ops.size());
    for (const NodeOpInfo& op : already_computed_ops) {
        seen.insert(make_local_spr_node_op_key(op));
    }

    std::vector<NodeOpInfo> filtered_ops;
    filtered_ops.reserve(ops.size());
    for (const NodeOpInfo& op : ops) {
        const LocalSprNodeOpKey key = make_local_spr_node_op_key(op);
        if (!seen.insert(key).second) {
            continue;
        }
        filtered_ops.push_back(op);
    }
    return filtered_ops;
}

void local_spr_assert_candidate_legal(
    const TreeBuildResult& tree,
    const LocalSprRepairUnit& unit,
    int prune_root_id,
    const std::vector<int>& subtree_nodes,
    const std::vector<int>& legal_edges,
    int regraft_child_id)
{
    local_spr_assert(
        prune_root_id >= 0 && prune_root_id < static_cast<int>(tree.nodes.size()),
        "prune root out of range");
    local_spr_assert(
        prune_root_id != tree.root_id,
        "prune root cannot be the tree root");
    local_spr_assert(
        tree.nodes[static_cast<size_t>(prune_root_id)].parent >= 0,
        "prune root must have a parent");

    std::vector<int> subtree_nodes_check;
    local_spr_assert(
        subtree_fully_inside_mask(tree, prune_root_id, unit.envelope_mask, &subtree_nodes_check),
        "committed subtree is not fully contained in its repair envelope");
    local_spr_assert(
        subtree_nodes_check == subtree_nodes,
        "subtree node set drifted between enumeration and commit");

    local_spr_assert(
        regraft_child_id >= 0 && regraft_child_id < static_cast<int>(tree.nodes.size()),
        "regraft child is out of range");
    const int regraft_parent_id = tree.nodes[static_cast<size_t>(regraft_child_id)].parent;
    local_spr_assert(regraft_parent_id >= 0, "regraft edge has no parent");
    local_spr_assert(
        unit.envelope_mask[static_cast<size_t>(regraft_child_id)] &&
        unit.envelope_mask[static_cast<size_t>(regraft_parent_id)],
        "regraft edge escaped the repair envelope");

    std::vector<char> subtree_mask(tree.nodes.size(), 0);
    for (int node_id : subtree_nodes) {
        if (node_id >= 0 && node_id < static_cast<int>(subtree_mask.size())) {
            subtree_mask[static_cast<size_t>(node_id)] = 1;
        }
    }
    local_spr_assert(
        !subtree_mask[static_cast<size_t>(regraft_child_id)] &&
        !subtree_mask[static_cast<size_t>(regraft_parent_id)],
        "self-regraft detected: target edge lies inside pruned subtree");
    local_spr_assert(
        std::find(legal_edges.begin(), legal_edges.end(), regraft_child_id) != legal_edges.end(),
        "selected regraft edge is not legal under the bounded-radius search");
}

void local_spr_assert_tree_integrity(
    const TreeBuildResult& tree,
    const std::string& label)
{
    const int node_count = static_cast<int>(tree.nodes.size());
    local_spr_assert(node_count > 0, label + ": tree is empty");
    local_spr_assert(
        tree.root_id >= 0 && tree.root_id < node_count,
        label + ": invalid root id");

    std::vector<int> parent_ref_count(static_cast<size_t>(node_count), 0);
    for (int node_id = 0; node_id < node_count; ++node_id) {
        const TreeNode& node = tree.nodes[static_cast<size_t>(node_id)];
        if (node.parent < 0) {
            local_spr_assert(node_id == tree.root_id, label + ": found non-root node with parent=-1");
        } else {
            local_spr_assert(
                node.parent >= 0 && node.parent < node_count,
                label + ": node parent out of range");
            ++parent_ref_count[static_cast<size_t>(node_id)];
        }
        if (node.is_tip) {
            local_spr_assert(node.left < 0 && node.right < 0, label + ": tip node has children");
            if (!node.name.empty()) {
                auto it = tree.tip_node_by_name.find(node.name);
                local_spr_assert(
                    it != tree.tip_node_by_name.end() && it->second == node_id,
                    label + ": tip name map is inconsistent");
            }
            continue;
        }
        local_spr_assert(
            node.left >= 0 && node.left < node_count &&
            node.right >= 0 && node.right < node_count &&
            node.left != node.right,
            label + ": internal node has invalid children");
        local_spr_assert(
            tree.nodes[static_cast<size_t>(node.left)].parent == node_id,
            label + ": left child parent pointer mismatch");
        local_spr_assert(
            tree.nodes[static_cast<size_t>(node.right)].parent == node_id,
            label + ": right child parent pointer mismatch");
    }

    for (int node_id = 0; node_id < node_count; ++node_id) {
        const int expected = (node_id == tree.root_id) ? 0 : 1;
        local_spr_assert(
            parent_ref_count[static_cast<size_t>(node_id)] == expected,
            label + ": node parent reference count mismatch");
    }

    std::vector<char> visited(static_cast<size_t>(node_count), 0);
    std::vector<int> stack;
    stack.push_back(tree.root_id);
    int visited_count = 0;
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        local_spr_assert(cur >= 0 && cur < node_count, label + ": DFS visited out-of-range node");
        local_spr_assert(!visited[static_cast<size_t>(cur)], label + ": cycle or duplicate child visit detected");
        visited[static_cast<size_t>(cur)] = 1;
        ++visited_count;
        const TreeNode& node = tree.nodes[static_cast<size_t>(cur)];
        if (!node.is_tip) {
            stack.push_back(node.right);
            stack.push_back(node.left);
        }
    }
    local_spr_assert(
        visited_count == node_count,
        label + ": tree is disconnected after subtree commit");
}

std::vector<int> build_tree_path_nodes(const TreeBuildResult& tree, int start, int end) {
    std::vector<int> path_a;
    std::unordered_map<int, int> pos_a;
    int cur = start;
    while (cur >= 0) {
        pos_a[cur] = static_cast<int>(path_a.size());
        path_a.push_back(cur);
        cur = tree.nodes[(size_t)cur].parent;
    }

    std::vector<int> path_b;
    int lca = -1;
    cur = end;
    while (cur >= 0) {
        auto it = pos_a.find(cur);
        if (it != pos_a.end()) {
            lca = cur;
            break;
        }
        path_b.push_back(cur);
        cur = tree.nodes[(size_t)cur].parent;
    }

    std::vector<int> out;
    if (lca < 0) return out;
    const int lca_pos = pos_a[lca];
    out.insert(out.end(), path_a.begin(), path_a.begin() + lca_pos + 1);
    std::reverse(path_b.begin(), path_b.end());
    out.insert(out.end(), path_b.begin(), path_b.end());
    return out;
}

std::vector<LocalSprInsertionAnchor> build_local_spr_insertion_anchors(
    const TreeBuildResult& tree,
    const std::vector<std::string>& inserted_names)
{
    std::vector<LocalSprInsertionAnchor> anchors;
    anchors.reserve(inserted_names.size());
    for (const std::string& name : inserted_names) {
        auto tip_it = tree.tip_node_by_name.find(name);
        if (tip_it == tree.tip_node_by_name.end()) {
            throw std::runtime_error("Local SPR could not find inserted tip in tree: " + name);
        }
        const int tip_id = tip_it->second;
        if (tip_id < 0 || tip_id >= static_cast<int>(tree.nodes.size())) {
            throw std::runtime_error("Local SPR tip id out of range for: " + name);
        }
        const int anchor_id = tree.nodes[(size_t)tip_id].parent;
        if (anchor_id < 0 || anchor_id >= static_cast<int>(tree.nodes.size())) {
            throw std::runtime_error("Local SPR inserted tip is missing an anchor parent: " + name);
        }
        anchors.push_back(LocalSprInsertionAnchor{name, tip_id, anchor_id});
    }
    return anchors;
}

std::vector<LocalSprRepairUnit> build_local_spr_repair_units(
    const TreeBuildResult& tree,
    const std::vector<LocalSprInsertionAnchor>& anchors,
    int cluster_threshold,
    int envelope_radius)
{
    std::vector<LocalSprRepairUnit> units;
    const int anchor_count = static_cast<int>(anchors.size());
    if (anchor_count == 0) return units;

    cluster_threshold = std::max(0, cluster_threshold);
    IntDisjointSet dsu(anchor_count);
    std::vector<std::vector<int>> anchor_dist((size_t)anchor_count);
    for (int i = 0; i < anchor_count; ++i) {
        anchor_dist[(size_t)i] = bfs_distances(tree, anchors[(size_t)i].anchor_id);
    }

    for (int i = 0; i < anchor_count; ++i) {
        for (int j = i + 1; j < anchor_count; ++j) {
            const int d = anchor_dist[(size_t)i][(size_t)anchors[(size_t)j].anchor_id];
            if (d >= 0 && d <= cluster_threshold) {
                dsu.unite(i, j);
            }
        }
    }

    std::unordered_map<int, int> root_to_unit;
    for (int i = 0; i < anchor_count; ++i) {
        const int root = dsu.find(i);
        auto [it, inserted] = root_to_unit.emplace(root, static_cast<int>(units.size()));
        if (inserted) {
            LocalSprRepairUnit unit;
            unit.unit_id = static_cast<int>(units.size());
            unit.envelope_mask.assign(tree.nodes.size(), 0);
            units.push_back(std::move(unit));
        }
        LocalSprRepairUnit& unit = units[(size_t)it->second];
        unit.query_names.push_back(anchors[(size_t)i].query_name);
        unit.anchor_ids.push_back(anchors[(size_t)i].anchor_id);
    }

    for (LocalSprRepairUnit& unit : units) {
        for (int anchor_id : unit.anchor_ids) {
            const std::vector<int> dist = bfs_distances(tree, anchor_id);
            for (int node_id = 0; node_id < static_cast<int>(tree.nodes.size()); ++node_id) {
                if (dist[(size_t)node_id] >= 0 && dist[(size_t)node_id] <= envelope_radius) {
                    unit.envelope_mask[(size_t)node_id] = 1;
                }
            }
        }
        for (int node_id = 0; node_id < static_cast<int>(tree.nodes.size()); ++node_id) {
            if (unit.envelope_mask[(size_t)node_id]) {
                unit.envelope_nodes.push_back(node_id);
            }
        }
    }

    return units;
}

void maybe_keep_local_spr_topk(
    std::vector<LocalSprCandidateMove>& topk,
    LocalSprCandidateMove candidate,
    int topk_limit)
{
    topk.push_back(std::move(candidate));
    std::sort(
        topk.begin(),
        topk.end(),
        [](const LocalSprCandidateMove& lhs, const LocalSprCandidateMove& rhs) {
            return lhs.approx_gain > rhs.approx_gain;
        });
    if (static_cast<int>(topk.size()) > topk_limit) {
        topk.resize((size_t)topk_limit);
    }
}

std::vector<int> select_local_spr_seed_edges(
    const PlacementResult& placement_result,
    const TreeBuildResult& tree,
    const std::vector<int>& center_dist,
    double baseline_logL,
    int seed_limit,
    int required_edge_radius)
{
    std::vector<std::pair<double, int>> ranked_edges;
    ranked_edges.reserve(placement_result.top_placements.size());
    for (const PlacementResult::RankedPlacement& placement :
         placement_result.top_placements) {
        const int edge_child_id = placement.target_id;
        if (edge_child_id < 0 || edge_child_id >= static_cast<int>(tree.nodes.size())) {
            continue;
        }
        const int edge_parent_id = tree.nodes[(size_t)edge_child_id].parent;
        if (edge_parent_id < 0) {
            continue;
        }
        if (required_edge_radius >= 0) {
            const int edge_child_dist = center_dist[(size_t)edge_child_id];
            const int edge_parent_dist = center_dist[(size_t)edge_parent_id];
            int edge_radius = -1;
            if (edge_child_dist < 0) {
                edge_radius = edge_parent_dist;
            } else if (edge_parent_dist < 0) {
                edge_radius = edge_child_dist;
            } else {
                edge_radius = std::min(edge_child_dist, edge_parent_dist);
            }
            if (edge_radius != required_edge_radius) {
                continue;
            }
        }
        const double approx_gain = placement.loglikelihood - baseline_logL;
        if (!(approx_gain > 0.0)) {
            continue;
        }
        ranked_edges.emplace_back(approx_gain, edge_child_id);
    }

    std::sort(
        ranked_edges.begin(),
        ranked_edges.end(),
        [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
            if (lhs.first == rhs.first) return lhs.second < rhs.second;
            return lhs.first > rhs.first;
        });

    std::vector<int> seed_edges;
    seed_edges.reserve(ranked_edges.size());
    std::unordered_set<int> seen;
    for (const auto& [gain, edge_child_id] : ranked_edges) {
        (void)gain;
        if (!seen.insert(edge_child_id).second) continue;
        seed_edges.push_back(edge_child_id);
        if (seed_limit > 0 && static_cast<int>(seed_edges.size()) >= seed_limit) {
            break;
        }
    }
    return seed_edges;
}

std::vector<LocalSprCandidateMove> select_local_spr_candidates(
    const std::vector<LocalSprCandidateMove>& ranked_candidates,
    int node_count)
{
    std::vector<LocalSprCandidateMove> selected;
    std::vector<char> selected_subtree_mask((size_t)std::max(0, node_count), 0);
    std::vector<char> selected_path_mask((size_t)std::max(0, node_count), 0);
    std::unordered_set<int> used_units;

    for (const LocalSprCandidateMove& candidate : ranked_candidates) {
        if (!used_units.insert(candidate.repair_unit_id).second) {
            continue;
        }

        bool conflict = false;
        for (int node_id : candidate.subtree_nodes) {
            if (node_id >= 0 &&
                node_id < node_count &&
                selected_subtree_mask[(size_t)node_id]) {
                conflict = true;
                break;
            }
        }
        if (conflict) {
            used_units.erase(candidate.repair_unit_id);
            continue;
        }
        if (candidate.regraft_child_id >= 0 &&
            candidate.regraft_child_id < node_count &&
            selected_subtree_mask[(size_t)candidate.regraft_child_id]) {
            used_units.erase(candidate.repair_unit_id);
            continue;
        }
        if (candidate.regraft_parent_id >= 0 &&
            candidate.regraft_parent_id < node_count &&
            selected_subtree_mask[(size_t)candidate.regraft_parent_id]) {
            used_units.erase(candidate.repair_unit_id);
            continue;
        }
        for (int node_id : candidate.regraft_path_nodes) {
            if (node_id >= 0 &&
                node_id < node_count &&
                selected_path_mask[(size_t)node_id]) {
                conflict = true;
                break;
            }
        }
        if (conflict) {
            used_units.erase(candidate.repair_unit_id);
            continue;
        }

        selected.push_back(candidate);
        for (int node_id : candidate.subtree_nodes) {
            if (node_id >= 0 && node_id < node_count) {
                selected_subtree_mask[(size_t)node_id] = 1;
            }
        }
        for (int node_id : candidate.regraft_path_nodes) {
            if (node_id >= 0 && node_id < node_count) {
                selected_path_mask[(size_t)node_id] = 1;
            }
        }
    }

    return selected;
}

bool local_spr_candidate_still_legal(
    const TreeBuildResult& tree,
    const LocalSprRepairUnit& unit,
    const LocalSprCandidateMove& candidate,
    std::vector<int>* current_subtree_nodes_out = nullptr)
{
    if (candidate.prune_root_id < 0 ||
        candidate.prune_root_id >= static_cast<int>(tree.nodes.size()) ||
        candidate.prune_root_id == tree.root_id) {
        return false;
    }
    const int current_parent =
        tree.nodes[(size_t)candidate.prune_root_id].parent;
    if (current_parent < 0 ||
        (candidate.old_parent_id >= 0 && current_parent != candidate.old_parent_id)) {
        return false;
    }

    std::vector<int> current_subtree_nodes_storage;
    std::vector<int>* current_subtree_nodes =
        current_subtree_nodes_out != nullptr
            ? current_subtree_nodes_out
            : &current_subtree_nodes_storage;
    if (!subtree_fully_inside_mask(
            tree,
            candidate.prune_root_id,
            unit.envelope_mask,
            current_subtree_nodes)) {
        return false;
    }

    if (candidate.regraft_child_id < 0 ||
        candidate.regraft_child_id >= static_cast<int>(tree.nodes.size())) {
        return false;
    }
    const int current_regraft_parent =
        tree.nodes[(size_t)candidate.regraft_child_id].parent;
    if (current_regraft_parent < 0 ||
        (candidate.regraft_parent_id >= 0 &&
         current_regraft_parent != candidate.regraft_parent_id)) {
        return false;
    }
    if (!unit.envelope_mask[(size_t)candidate.regraft_child_id] ||
        !unit.envelope_mask[(size_t)current_regraft_parent]) {
        return false;
    }

    std::vector<char> current_subtree_mask(tree.nodes.size(), 0);
    for (int node_id : *current_subtree_nodes) {
        if (node_id >= 0 &&
            node_id < static_cast<int>(current_subtree_mask.size())) {
            current_subtree_mask[(size_t)node_id] = 1;
        }
    }
    if (current_subtree_mask[(size_t)candidate.regraft_child_id] ||
        current_subtree_mask[(size_t)current_regraft_parent]) {
        return false;
    }

    return true;
}

bool prune_subtree_for_spr(TreeBuildResult& tree, int pruned_id, PruneInfo& info) {
    if (pruned_id < 0 || pruned_id >= static_cast<int>(tree.nodes.size())) return false;
    if (pruned_id == tree.root_id) return false;
    const int parent_id = tree.nodes[pruned_id].parent;
    if (parent_id < 0 || parent_id >= static_cast<int>(tree.nodes.size())) return false;
    const TreeNode& parent = tree.nodes[parent_id];
    const int sibling_id = (parent.left == pruned_id) ? parent.right :
                           (parent.right == pruned_id ? parent.left : -1);
    if (sibling_id < 0 || sibling_id >= static_cast<int>(tree.nodes.size())) return false;

    const int grandparent_id = parent.parent;
    info.pruned_id = pruned_id;
    info.free_internal_id = parent_id;
    info.sibling_id = sibling_id;
    info.grandparent_id = grandparent_id;
    info.pruned_branch_length = tree.nodes[pruned_id].branch_length_to_parent;

    tree.nodes[pruned_id].parent = -1;

    if (grandparent_id >= 0) {
        TreeNode& grandparent = tree.nodes[grandparent_id];
        if (grandparent.left == parent_id) {
            grandparent.left = sibling_id;
        } else if (grandparent.right == parent_id) {
            grandparent.right = sibling_id;
        } else {
            return false;
        }
        TreeNode& sibling = tree.nodes[sibling_id];
        sibling.parent = grandparent_id;
        sibling.branch_length_to_parent += parent.branch_length_to_parent;
    } else {
        tree.root_id = sibling_id;
        TreeNode& sibling = tree.nodes[sibling_id];
        sibling.parent = -1;
        sibling.branch_length_to_parent = fp_t(0);
    }

    TreeNode& free_internal = tree.nodes[parent_id];
    free_internal.left = -1;
    free_internal.right = -1;
    free_internal.parent = -1;
    free_internal.is_tip = false;
    free_internal.name.clear();
    free_internal.branch_length_to_parent = fp_t(0);
    rebuild_traversals(tree);
    return true;
}

bool prune_leaf_for_spr(TreeBuildResult& tree, int tip_id, PruneInfo& info) {
    if (tip_id < 0 || tip_id >= static_cast<int>(tree.nodes.size())) return false;
    if (!tree.nodes[(size_t)tip_id].is_tip) return false;
    return prune_subtree_for_spr(tree, tip_id, info);
}

std::vector<int> bfs_distances(const TreeBuildResult& tree, int start) {
    const int node_count = static_cast<int>(tree.nodes.size());
    std::vector<int> dist((size_t)node_count, -1);
    if (start < 0 || start >= node_count) return dist;
    std::vector<int> queue;
    queue.reserve((size_t)node_count);
    dist[(size_t)start] = 0;
    queue.push_back(start);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        const int next_dist = dist[(size_t)cur] + 1;
        const TreeNode& node = tree.nodes[cur];
        const int neighbors[3] = {node.parent, node.left, node.right};
        for (int neighbor : neighbors) {
            if (neighbor < 0 || neighbor >= node_count) continue;
            if (dist[(size_t)neighbor] >= 0) continue;
            dist[(size_t)neighbor] = next_dist;
            queue.push_back(neighbor);
        }
    }
    return dist;
}

std::vector<char> build_unit_anchor_skeleton_mask(
    const TreeBuildResult& tree,
    const std::vector<int>& anchor_ids)
{
    std::vector<char> mask(tree.nodes.size(), 0);
    if (anchor_ids.empty()) return mask;

    const int root_anchor = anchor_ids.front();
    if (root_anchor >= 0 && root_anchor < static_cast<int>(tree.nodes.size())) {
        mask[(size_t)root_anchor] = 1;
    }

    for (int anchor_id : anchor_ids) {
        if (anchor_id < 0 || anchor_id >= static_cast<int>(tree.nodes.size())) continue;
        const std::vector<int> path_nodes =
            build_tree_path_nodes(tree, root_anchor, anchor_id);
        for (int node_id : path_nodes) {
            if (node_id < 0 || node_id >= static_cast<int>(mask.size())) continue;
            mask[(size_t)node_id] = 1;
        }
    }
    return mask;
}

std::vector<int> multi_source_bfs_distances(
    const TreeBuildResult& tree,
    const std::vector<char>& source_mask)
{
    const int node_count = static_cast<int>(tree.nodes.size());
    std::vector<int> dist((size_t)node_count, -1);
    if (static_cast<int>(source_mask.size()) != node_count) return dist;

    std::vector<int> queue;
    queue.reserve((size_t)node_count);
    for (int node_id = 0; node_id < node_count; ++node_id) {
        if (!source_mask[(size_t)node_id]) continue;
        dist[(size_t)node_id] = 0;
        queue.push_back(node_id);
    }

    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        const int next_dist = dist[(size_t)cur] + 1;
        const TreeNode& node = tree.nodes[(size_t)cur];
        const int neighbors[3] = {node.parent, node.left, node.right};
        for (int neighbor : neighbors) {
            if (neighbor < 0 || neighbor >= node_count) continue;
            if (dist[(size_t)neighbor] >= 0) continue;
            dist[(size_t)neighbor] = next_dist;
            queue.push_back(neighbor);
        }
    }
    return dist;
}

int edge_distance_from_endpoint(const std::vector<int>& dist, int child_id, int parent_id) {
    if (child_id < 0 || parent_id < 0) return -1;
    const int d_child = dist[(size_t)child_id];
    const int d_parent = dist[(size_t)parent_id];
    if (d_child < 0) return d_parent;
    if (d_parent < 0) return d_child;
    return std::min(d_child, d_parent);
}

std::vector<int> collect_candidate_edges(
    const TreeBuildResult& tree,
    int endpoint_a,
    int endpoint_b,
    int radius,
    int exclude_a,
    int exclude_b) {
    std::vector<int> edges;
    if (radius < 0) return edges;
    const int node_count = static_cast<int>(tree.nodes.size());
    std::vector<int> dist_a = bfs_distances(tree, endpoint_a);
    std::vector<int> dist_b;
    if (endpoint_b >= 0 && endpoint_b != endpoint_a) {
        dist_b = bfs_distances(tree, endpoint_b);
    }
    for (int node_id = 0; node_id < node_count; ++node_id) {
        const int parent_id = tree.nodes[node_id].parent;
        if (parent_id < 0) continue;
        if (node_id == exclude_a || node_id == exclude_b) continue;
        if (parent_id == exclude_a || parent_id == exclude_b) continue;
        int best = edge_distance_from_endpoint(dist_a, node_id, parent_id);
        if (!dist_b.empty()) {
            const int alt = edge_distance_from_endpoint(dist_b, node_id, parent_id);
            if (best < 0 || (alt >= 0 && alt < best)) best = alt;
        }
        if (best >= 0 && best <= radius) {
            edges.push_back(node_id);
        }
    }
    return edges;
}

std::vector<int> compute_center_distances(
    const TreeBuildResult& tree,
    int endpoint_a,
    int endpoint_b)
{
    std::vector<int> best = bfs_distances(tree, endpoint_a);
    if (endpoint_b >= 0 && endpoint_b != endpoint_a) {
        const std::vector<int> alt = bfs_distances(tree, endpoint_b);
        if (best.size() != alt.size()) {
            throw std::runtime_error("compute_center_distances produced mismatched BFS arrays.");
        }
        for (size_t i = 0; i < best.size(); ++i) {
            if (best[i] < 0) {
                best[i] = alt[i];
            } else if (alt[i] >= 0 && alt[i] < best[i]) {
                best[i] = alt[i];
            }
        }
    }
    return best;
}

int edge_distance_from_center(
    const std::vector<int>& center_dist,
    int child_id,
    int parent_id)
{
    if (child_id < 0 || parent_id < 0) return -1;
    const int d_child = center_dist[(size_t)child_id];
    const int d_parent = center_dist[(size_t)parent_id];
    if (d_child < 0) return d_parent;
    if (d_parent < 0) return d_child;
    return std::min(d_child, d_parent);
}

std::vector<int> collect_outward_edges_from_seed(
    const TreeBuildResult& tree,
    const std::vector<int>& center_dist,
    int seed_edge_child_id,
    int min_radius_exclusive,
    int max_radius,
    int exclude_a,
    int exclude_b)
{
    std::vector<int> edges;
    if (max_radius <= min_radius_exclusive) return edges;
    if (seed_edge_child_id < 0 ||
        seed_edge_child_id >= static_cast<int>(tree.nodes.size())) {
        return edges;
    }

    const int seed_parent_id = tree.nodes[(size_t)seed_edge_child_id].parent;
    if (seed_parent_id < 0 ||
        seed_parent_id >= static_cast<int>(tree.nodes.size())) {
        return edges;
    }

    const int child_dist = center_dist[(size_t)seed_edge_child_id];
    const int parent_dist = center_dist[(size_t)seed_parent_id];
    if (child_dist < 0 || parent_dist < 0 || child_dist == parent_dist) {
        return edges;
    }

    struct FrontierState {
        int node_id = -1;
        int prev_id = -1;
    };

    std::vector<FrontierState> queue;
    queue.reserve(tree.nodes.size());
    std::vector<char> visited(tree.nodes.size(), 0);
    std::vector<char> edge_seen(tree.nodes.size(), 0);

    auto push_start = [&](int node_id, int prev_id) {
        if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) return;
        if (visited[(size_t)node_id]) return;
        visited[(size_t)node_id] = 1;
        queue.push_back(FrontierState{node_id, prev_id});
    };

    if (child_dist > parent_dist) {
        push_start(seed_edge_child_id, seed_parent_id);
    } else {
        push_start(seed_parent_id, seed_edge_child_id);
    }

    auto maybe_record_edge = [&](int node_a, int node_b) {
        int edge_child_id = -1;
        int edge_parent_id = -1;
        if (tree.nodes[(size_t)node_a].parent == node_b) {
            edge_child_id = node_a;
            edge_parent_id = node_b;
        } else if (tree.nodes[(size_t)node_b].parent == node_a) {
            edge_child_id = node_b;
            edge_parent_id = node_a;
        } else {
            return;
        }

        if (edge_child_id == exclude_a || edge_child_id == exclude_b) return;
        if (edge_parent_id == exclude_a || edge_parent_id == exclude_b) return;
        if (edge_seen[(size_t)edge_child_id]) return;

        const int edge_radius =
            edge_distance_from_center(center_dist, edge_child_id, edge_parent_id);
        if (edge_radius <= min_radius_exclusive || edge_radius > max_radius) return;

        edge_seen[(size_t)edge_child_id] = 1;
        edges.push_back(edge_child_id);
    };

    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const FrontierState state = queue[qi];
        const int cur = state.node_id;
        const int cur_dist = center_dist[(size_t)cur];
        const TreeNode& node = tree.nodes[(size_t)cur];
        const int neighbors[3] = {node.parent, node.left, node.right};
        for (int neighbor : neighbors) {
            if (neighbor < 0 || neighbor == state.prev_id) continue;
            if (neighbor >= static_cast<int>(tree.nodes.size())) continue;
            const int neighbor_dist = center_dist[(size_t)neighbor];
            if (neighbor_dist < 0 || neighbor_dist <= cur_dist) continue;

            maybe_record_edge(cur, neighbor);
            if (!visited[(size_t)neighbor]) {
                visited[(size_t)neighbor] = 1;
                queue.push_back(FrontierState{neighbor, cur});
            }
        }
    }

    return edges;
}

void regraft_subtree_for_spr(
    TreeBuildResult& tree,
    const PruneInfo& info,
    int target_child_id,
    bool use_lengths,
    double pendant_length,
    double proximal_length)
{
    const int parent_id = tree.nodes[target_child_id].parent;
    if (parent_id < 0) {
        throw std::runtime_error("SPR regraft target has no parent edge.");
    }
    const int free_internal_id = info.free_internal_id;
    const int pruned_id = info.pruned_id;

    TreeNode& parent = tree.nodes[parent_id];
    if (parent.left == target_child_id) {
        parent.left = free_internal_id;
    } else if (parent.right == target_child_id) {
        parent.right = free_internal_id;
    } else {
        throw std::runtime_error("SPR regraft target not found under its parent.");
    }

    TreeNode& internal = tree.nodes[free_internal_id];
    internal.parent = parent_id;
    internal.left = target_child_id;
    internal.right = pruned_id;
    internal.is_tip = false;
    internal.name.clear();

    TreeNode& target_child = tree.nodes[target_child_id];
    const fp_t total_length = target_child.branch_length_to_parent;
    fp_t proximal = total_length * fp_t(0.5);
    fp_t distal = total_length - proximal;
    if (use_lengths) {
        double prox = 0.0;
        double dist = 0.0;
        normalize_split_branch_lengths(
            static_cast<double>(total_length),
            proximal_length,
            OPT_BRANCH_LEN_MIN,
            prox,
            dist);
        proximal = static_cast<fp_t>(prox);
        distal = static_cast<fp_t>(dist);
    }
    internal.branch_length_to_parent = proximal;
    target_child.branch_length_to_parent = distal;
    target_child.parent = free_internal_id;

    TreeNode& pruned = tree.nodes[pruned_id];
    pruned.parent = free_internal_id;
    pruned.branch_length_to_parent =
        use_lengths
            ? static_cast<fp_t>(sanitize_branch_length(pendant_length))
            : info.pruned_branch_length;
    rebuild_traversals(tree);
}

void regraft_subtree_for_spr(TreeBuildResult& tree, const PruneInfo& info, int target_child_id) {
    regraft_subtree_for_spr(tree, info, target_child_id, false, 0.0, 0.0);
}

void maybe_expand_cuda_printf_fifo() {
}

// ----- Model input normalization helpers -----

void normalize_vector(std::vector<double>& vec) {
    double sum = 0.0;
    for (double v : vec) sum += v;
    if (sum <= 0.0) return;
    for (double& v : vec) v /= sum;
}

std::vector<double> estimate_empirical_pi(const parse::Alignment& alignment, int states) {
    if (states != 4 && states != 5) {
        throw std::runtime_error(
            "Empirical frequency estimation currently supports only 4-state or 5-state DNA data.");
    }

    std::vector<double> counts(states, 0.0);
    double informative_weight = 0.0;
    for (const std::string& seq : alignment.sequences) {
        for (char c : seq) {
            if (states == 4) {
                const uint8_t mask = encode_state_DNA4_mask(c);
                int bit_count = 0;
                for (int state = 0; state < 4; ++state) {
                    if (mask & (1u << state)) ++bit_count;
                }
                if (bit_count == 0) continue;
                const double share = 1.0 / static_cast<double>(bit_count);
                for (int state = 0; state < 4; ++state) {
                    if (mask & (1u << state)) counts[(size_t)state] += share;
                }
                informative_weight += 1.0;
                continue;
            }

            switch (c) {
                case 'A': case 'a': counts[0] += 1.0; break;
                case 'C': case 'c': counts[1] += 1.0; break;
                case 'G': case 'g': counts[2] += 1.0; break;
                case 'T': case 't':
                case 'U': case 'u': counts[3] += 1.0; break;
                case '-':
                case '.': counts[4] += 1.0; break;
                default: continue;
            }
            informative_weight += 1.0;
        }
    }
    if (informative_weight <= 0.0) {
        throw std::runtime_error(
            "Cannot estimate empirical frequencies: alignment contains no informative states.");
    }
    normalize_vector(counts);
    return counts;
}

std::vector<double> ensure_normalized_pi(std::vector<double> pi, int states) {
    if ((int)pi.size() != states) pi.assign(states, 1.0 / states);
    normalize_vector(pi);
    return pi;
}

// ----- Environment helpers -----

void set_int_env_if_specified(const char* name, int value) {
    if (value < 0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

void set_double_env_if_specified(const char* name, double value) {
    if (value < 0.0) return;
    setenv(name, std::to_string(value).c_str(), 1);
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] && std::string(value) != "0";
}

bool repetitive_column_compression_enabled() {
    if (env_flag_enabled("MLIPPER_DISABLE_REPETITIVE_COLUMNS")) {
        return false;
    }
    const char* legacy_enable = std::getenv("MLIPPER_ENABLE_REPETITIVE_COLUMNS");
    if (legacy_enable) {
        return legacy_enable[0] && std::string(legacy_enable) != "0";
    }
    return true;
}

void print_tree_rec(const TreeBuildResult& T, int node_id, int depth)
{
    if (node_id < 0) return;
    const TreeNode &nd = T.nodes[node_id];

    for (int i = 0; i < depth; ++i) std::cout << "  ";

    std::cout << "[" << nd.id << "]";
    if (nd.is_tip) {
        std::cout << " (tip: " << nd.name << ")";
    } else {
        std::cout << " (inner)";
    }

    if (nd.parent >= 0) {
        std::cout << "  len=" << nd.branch_length_to_parent
                  << "  parent=" << nd.parent;
    } else {
        std::cout << "  <ROOT>";
    }
    std::cout << "\n";

    if (!nd.is_tip) {
        if (nd.left  >= 0) print_tree_rec(T, nd.left,  depth + 1);
        if (nd.right >= 0) print_tree_rec(T, nd.right, depth + 1);
    }
}

void print_tree_structure(const TreeBuildResult& T)
{
    std::cout << "==== Tree structure (indented) ====\n";
    if (T.root_id < 0) {
        std::cout << "No root_id set!\n";
        return;
    }
    print_tree_rec(T, T.root_id, 0);
    std::cout << "===================================\n";
}

bool newick_name_requires_quotes(const std::string& name) {
    if (name.empty()) return true;
    for (char ch : name) {
        switch (ch) {
            case '(': case ')': case '[': case ']': case ':': case ';': case ',':
            case '\'': case ' ': case '\t': case '\n': case '\r':
                return true;
            default:
                break;
        }
    }
    return false;
}

std::string format_newick_name(const std::string& name) {
    if (!newick_name_requires_quotes(name)) return name;
    std::string quoted;
    quoted.reserve(name.size() + 2);
    quoted.push_back('\'');
    for (char ch : name) {
        quoted.push_back(ch);
        if (ch == '\'') quoted.push_back('\'');
    }
    quoted.push_back('\'');
    return quoted;
}

struct OutputTreeNode {
    int source_node_id = -1;
    bool is_tip = false;
    double branch_length_to_parent = 0.0;
    std::string name;
    std::vector<OutputTreeNode> children;
};

OutputTreeNode build_output_subtree(const TreeBuildResult& tree, int node_id) {
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) {
        throw std::runtime_error("Invalid node id while preparing Newick output tree.");
    }

    const TreeNode& node = tree.nodes[node_id];
    OutputTreeNode out;
    out.source_node_id = node.id;
    out.is_tip = node.is_tip;

    if (node.parent >= 0) {
        out.branch_length_to_parent = static_cast<double>(node.branch_length_to_parent);
        if (out.branch_length_to_parent < 0.0) {
            throw std::runtime_error(
                "Negative branch length while preparing Newick output tree for node " +
                std::to_string(node.id));
        }
    }

    if (node.is_tip) {
        out.name = node.name.empty() ? ("tip_" + std::to_string(node.id)) : node.name;
        return out;
    }

    if (node.left < 0 || node.right < 0) {
        throw std::runtime_error("Internal node missing child while preparing Newick output tree.");
    }

    out.children.reserve(2);
    out.children.push_back(build_output_subtree(tree, node.left));
    out.children.push_back(build_output_subtree(tree, node.right));
    return out;
}

size_t collapse_short_internal_output_branches(OutputTreeNode& node, double epsilon) {
    if (node.is_tip) return 0;

    size_t collapsed = 0;
    for (auto& child : node.children) {
        collapsed += collapse_short_internal_output_branches(child, epsilon);
    }

    std::vector<OutputTreeNode> rewritten_children;
    rewritten_children.reserve(node.children.size());
    for (auto& child : node.children) {
        const bool collapse_child =
            !child.is_tip &&
            child.branch_length_to_parent <= epsilon;
        if (!collapse_child) {
            rewritten_children.push_back(std::move(child));
            continue;
        }

        const double collapsed_length = child.branch_length_to_parent;
        for (auto& grandchild : child.children) {
            grandchild.branch_length_to_parent += collapsed_length;
            rewritten_children.push_back(std::move(grandchild));
        }
        ++collapsed;
    }

    node.children = std::move(rewritten_children);
    return collapsed;
}

void write_newick_subtree(const TreeBuildResult& tree, int node_id, std::ostream& os);
void write_newick_subtree(const OutputTreeNode& node, bool is_root, std::ostream& os);

std::string write_tree_to_newick_string(const TreeBuildResult& tree) {
    if (tree.root_id < 0) {
        throw std::runtime_error("Cannot serialize Newick tree: invalid root_id.");
    }
    std::ostringstream oss;
    write_newick_subtree(tree, tree.root_id, oss);
    oss << ';';
    return oss.str();
}

std::string write_tree_to_output_newick_string(
    const TreeBuildResult& tree,
    double collapse_internal_epsilon,
    size_t* collapsed_internal_branches_out = nullptr) {
    if (collapsed_internal_branches_out) *collapsed_internal_branches_out = 0;
    if (collapse_internal_epsilon < 0.0) {
        return write_tree_to_newick_string(tree);
    }
    if (!std::isfinite(collapse_internal_epsilon)) {
        throw std::runtime_error("Newick output collapse epsilon must be finite.");
    }
    if (tree.root_id < 0) {
        throw std::runtime_error("Cannot serialize Newick tree: invalid root_id.");
    }

    OutputTreeNode output_root = build_output_subtree(tree, tree.root_id);
    const size_t collapsed = collapse_short_internal_output_branches(
        output_root,
        collapse_internal_epsilon);
    if (collapsed_internal_branches_out) *collapsed_internal_branches_out = collapsed;

    std::ostringstream oss;
    write_newick_subtree(output_root, true, oss);
    oss << ';';
    return oss.str();
}

void write_newick_subtree(const TreeBuildResult& tree, int node_id, std::ostream& os) {
    if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) {
        throw std::runtime_error("Invalid node id while writing Newick tree.");
    }

    const TreeNode& node = tree.nodes[node_id];
    if (node.is_tip) {
        os << format_newick_name(node.name.empty() ? ("tip_" + std::to_string(node.id)) : node.name);
    } else {
        if (node.left < 0 || node.right < 0) {
            throw std::runtime_error("Internal node missing child while writing Newick tree.");
        }
        os << '(';
        write_newick_subtree(tree, node.left, os);
        os << ',';
        write_newick_subtree(tree, node.right, os);
        os << ')';
    }

    if (node.parent >= 0) {
        const double branch_length = static_cast<double>(node.branch_length_to_parent);
        if (branch_length < 0.0) {
            throw std::runtime_error(
                "Negative branch length while writing Newick tree for node " + std::to_string(node.id));
        }
        os << ':' << std::setprecision(17) << branch_length;
    }
}

void write_newick_subtree(const OutputTreeNode& node, bool is_root, std::ostream& os) {
    if (node.is_tip) {
        os << format_newick_name(node.name.empty() ? ("tip_" + std::to_string(node.source_node_id)) : node.name);
    } else {
        if (node.children.size() < 2) {
            throw std::runtime_error(
                "Internal output node collapsed below arity 2 while writing Newick tree.");
        }
        os << '(';
        for (size_t i = 0; i < node.children.size(); ++i) {
            if (i > 0) os << ',';
            write_newick_subtree(node.children[i], false, os);
        }
        os << ')';
    }

    if (!is_root) {
        if (node.branch_length_to_parent < 0.0) {
            throw std::runtime_error(
                "Negative branch length while writing collapsed Newick tree for node " +
                std::to_string(node.source_node_id));
        }
        os << ':' << std::setprecision(17) << node.branch_length_to_parent;
    }
}

size_t write_tree_to_newick_file(
    const TreeBuildResult& tree,
    const std::string& path,
    double collapse_internal_epsilon = -1.0) {
    if (tree.root_id < 0) {
        throw std::runtime_error("Cannot write Newick tree: invalid root_id.");
    }

    std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Cannot open Newick output path: " + path);
    }
    size_t collapsed_internal_branches = 0;
    const std::string newick = write_tree_to_output_newick_string(
        tree,
        collapse_internal_epsilon,
        &collapsed_internal_branches);
    ofs << newick << '\n';
    return collapsed_internal_branches;
}

} // namespace

int main(int argc, char** argv) {
    auto start_gpu = std::chrono::steady_clock::time_point{};
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    float gpu_ms_kernel = 0.0f;
    BuildToGpuResult res{};
    PlacementOpBuffer placement_ops{};

    CLI::App app{"MLIPPER"};
    app.get_formatter()->column_width(40);

    // Single config object filled directly by CLI11 options.
    parse::RunConfig config;

    // ---- Input (files/tree) ----
    std::string tree_newick;
    std::string jplace_out;
    std::string commit_tree_out;
    // Internal-only output/control knobs, not exposed via CLI.
    double commit_collapse_internal_epsilon = 1e-6;
    bool commit_to_tree = false;

    auto* opt_tree_alignment = app.add_option("--tree-alignment", config.files.tree_alignment, "Reference alignment (tree MSA)")
                                  ->group("Input")
                                  ->check(CLI::ExistingFile);
    auto* opt_query_alignment = app.add_option("--query-alignment", config.files.query_alignment,
                                               "Query alignment for placement (optional; defaults to --tree-alignment)")
                                   ->group("Input")
                                   ->check(CLI::ExistingFile);
    auto* opt_tree_file = app.add_option("--tree", config.files.tree, "Reference tree topology (Newick file)")
                              ->group("Input")
                              ->check(CLI::ExistingFile);
    auto* opt_tree_newick = app.add_option("--tree-newick", tree_newick, "Reference tree topology (Newick string)")
                                ->group("Input");
    auto* opt_jplace_out =
        app.add_option("--jplace-out", jplace_out, "Optional output path for a top-k placement jplace file")
            ->group("Output");
    app.add_option("--commit-to-tree", commit_tree_out,
                   "Commit query placements to reference tree and write the final tree in Newick (.nwk) format to this path")
        ->group("Output");
    opt_tree_file->excludes(opt_tree_newick);
    opt_tree_newick->excludes(opt_tree_file);

    // ---- Model ----
    // Defaults match the previous `config.yaml` defaults (pre-CLI refactor).
    config.model.states = 4;
    config.model.subst_model = "GTR";
    config.model.ncat = 4;
    config.model.alpha = 0.3;
    config.model.pinv = 0.0;
    config.model.rates = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    config.model.per_rate_scaling = true;
    bool no_per_rate_scaling = false;
    bool empirical_freqs = false;
    int full_opt_passes = -1;
    int refine_global_passes = -1;
    int refine_extra_passes = -1;
    int refine_detect_topk = -1;
    int refine_topk = -1;
    double refine_gap_top2 = -1.0;
    double refine_gap_top5 = -1.0;
    double refine_converged_loglk_eps = -1.0;
    double refine_converged_length_eps = -1.0;
    bool placement_fast = false;
    bool local_spr = false;
    bool local_spr_fast = false;
    bool fast_mode = false;
    // Internal-only local SPR tuning knobs not exposed via CLI.
    int batch_insert_size = 0;
    int local_spr_radius = 2;
    int local_spr_cluster_threshold = 3;
    int local_spr_topk_per_unit = 8;
    bool local_spr_dynamic_validation_conflicts = false;
    int local_spr_rounds = 1;

    app.add_option("--states", config.model.states, "Number of states")->group("Model");
    app.add_option("--subst-model", config.model.subst_model, "Substitution model")->group("Model");
    app.add_option("--ncat", config.model.ncat, "Number of rate categories")->group("Model");
    app.add_option("--alpha", config.model.alpha, "Gamma shape alpha")->group("Model");
    app.add_option("--pinv", config.model.pinv, "Proportion of invariant sites")->group("Model");
    auto* opt_freqs = app.add_option("--freqs", config.model.freqs, "Equilibrium freqs (list)")
                          ->group("Model")
                          ->delimiter(',');
    auto* opt_empirical_freqs = app.add_flag(
        "--empirical-freqs",
        empirical_freqs,
        "Estimate equilibrium freqs from --tree-alignment (distributes ambiguous DNA symbols across represented states)")
                                    ->group("Model");
    opt_freqs->excludes(opt_empirical_freqs);
    opt_empirical_freqs->excludes(opt_freqs);
    app.add_option("--rates", config.model.rates, "GTR rates rAC,rAG,rAT,rCG,rCT,rGT (list)")->group("Model")->delimiter(',');
    app.add_option("--rate-weights", config.model.rate_weights, "Rate category weights (list)")->group("Model")->delimiter(',');
    app.add_flag("--no-per-rate-scaling", no_per_rate_scaling, "Disable per-rate scaling")->group("Model");

    app.add_flag("--placement-fast", placement_fast,
                 "Use fast placement scoring (1 full optimization pass instead of baseline 4)")
        ->group("Placement");
    auto* opt_local_spr =
        app.add_flag("--local-spr", local_spr,
                     "Enable local subtree SPR after each batch insert")
            ->group("Placement");
    auto* opt_local_spr_fast =
        app.add_flag("--local-spr-fast", local_spr_fast,
                     "Use fast local SPR scoring (1 full optimization pass instead of baseline 4)")
            ->group("Placement");
    auto* opt_fast =
        app.add_flag("--fast", fast_mode,
                     "Enable both --placement-fast and --local-spr-fast")
            ->group("Placement");
    auto* opt_batch_insert_size =
        app.add_option("--batch-insert-size", batch_insert_size,
                       "Insert+commit query batches of size N (0 = all at once)")
            ->group("Placement");
    auto* opt_local_spr_radius =
        app.add_option("--local-spr-radius", local_spr_radius,
                       "Local SPR radius (filters candidate edges and defines subtree neighborhood)")
            ->group("Placement");
    auto* opt_local_spr_cluster_threshold =
        app.add_option("--local-spr-cluster-threshold", local_spr_cluster_threshold,
                       "Anchor-distance threshold for grouping inserted queries into local SPR repair units")
            ->group("Placement");
    auto* opt_local_spr_rounds =
        app.add_option("--local-spr-rounds", local_spr_rounds,
                       "Run up to N rounds of local SPR, rebuilding between accepted rounds")
            ->group("Placement");
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (batch_insert_size < 0) {
        return app.exit(CLI::ValidationError("--batch-insert-size", "must be >= 0"));
    }
    if (local_spr_radius < 0) {
        return app.exit(CLI::ValidationError("--local-spr-radius", "must be >= 0"));
    }
    if (local_spr_cluster_threshold < 0) {
        return app.exit(CLI::ValidationError("--local-spr-cluster-threshold", "must be >= 0"));
    }
    if (local_spr_rounds <= 0) {
        return app.exit(CLI::ValidationError("--local-spr-rounds", "must be >= 1"));
    }
    if (fast_mode) {
        placement_fast = true;
        local_spr_fast = true;
    }
    if (local_spr) {
        if (batch_insert_size <= 0) {
            batch_insert_size = 5;
        }
    }
    commit_to_tree = !commit_tree_out.empty();

    const bool local_spr_tuning_requested =
        opt_local_spr_radius->count() > 0 ||
        opt_local_spr_cluster_threshold->count() > 0 ||
        opt_local_spr_rounds->count() > 0 ||
        opt_local_spr_fast->count() > 0;
    if (local_spr_tuning_requested && !local_spr) {
        return app.exit(CLI::ValidationError(
            "--local-spr",
            "local SPR tuning flags require --local-spr"));
    }
    if (local_spr && !commit_to_tree) {
        return app.exit(CLI::ValidationError(
            "--local-spr",
            "--local-spr requires --commit-to-tree"));
    }
    if (opt_batch_insert_size->count() > 0 && batch_insert_size > 0 && !commit_to_tree) {
        return app.exit(CLI::ValidationError(
            "--batch-insert-size",
            "batch insert mode requires --commit-to-tree"));
    }
    if (batch_insert_size > 0 && commit_to_tree && opt_jplace_out->count() > 0) {
        return app.exit(CLI::ValidationError(
            "--jplace-out",
            "batch insert mode does not support --jplace-out"));
    }
    (void)opt_local_spr;
    (void)opt_fast;

    const std::filesystem::path config_base = std::filesystem::current_path();

    parse::RunInputs inputs;
    try {
        if (no_per_rate_scaling) config.model.per_rate_scaling = false;
        if (config.files.tree_alignment.empty()) throw CLI::RequiredError("--tree-alignment");
        if (config.files.query_alignment.empty()) {
            config.files.query_alignment = config.files.tree_alignment;
        }
        if (tree_newick.empty() && config.files.tree.empty()) throw CLI::RequiredError("one of [--tree, --tree-newick]");

        if (!commit_tree_out.empty()) {
            validate_output_path(config_base, "--commit-to-tree", commit_tree_out);
        }
        if (!jplace_out.empty()) {
            validate_output_path(config_base, "--jplace-out", jplace_out);
        }
        if (!commit_tree_out.empty() && !jplace_out.empty()) {
            const std::filesystem::path commit_path =
                normalize_cli_path(config_base, commit_tree_out);
            const std::filesystem::path jplace_path =
                normalize_cli_path(config_base, jplace_out);
            if (commit_path == jplace_path) {
                throw CLI::ValidationError(
                    "--jplace-out",
                    "must not be the same path as --commit-to-tree");
            }
        }

        validate_model_inputs(config.model);

        parse::Alignment tree_alignment;
        try {
            tree_alignment = parse::read_alignment_file(resolve_path(config_base, config.files.tree_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--tree-alignment", e.what());
        }

        parse::Alignment query_alignment;
        try {
            query_alignment = parse::read_alignment_file(resolve_path(config_base, config.files.query_alignment));
        } catch (const std::exception& e) {
            throw CLI::ValidationError("--query-alignment", e.what());
        }

        std::string tree_text;
        if (tree_newick.empty()) {
            try {
                tree_text = read_file_to_string(resolve_path(config_base, config.files.tree));
            } catch (const std::exception& e) {
                throw CLI::ValidationError("--tree", e.what());
            }
        } else {
            tree_text = tree_newick;
        }
        tree_text = parse::normalize_newick(tree_text);
        validate_newick_with_pll(
            tree_text,
            tree_newick.empty() ? "--tree" : "--tree-newick");

        inputs = parse::RunInputs{
            std::move(config),
            std::move(tree_alignment),
            std::move(query_alignment),
            std::move(tree_text)};

        if (inputs.tree_alignment.names.empty())
            throw CLI::ValidationError("--tree-alignment", "contains no sequences");
        if (inputs.tree_alignment.sites == 0)
            throw CLI::ValidationError("--tree-alignment", "contains zero sites");

        if (inputs.query_alignment.names.empty())
            throw CLI::ValidationError("--query-alignment", "contains no sequences");
        if (inputs.query_alignment.sites == 0)
            throw CLI::ValidationError("--query-alignment", "contains zero sites");
        if (inputs.query_alignment.sites != inputs.tree_alignment.sites) {
            throw CLI::ValidationError("--query-alignment",
                                       "sites mismatch with --tree-alignment (" +
                                           std::to_string(inputs.query_alignment.sites) + " vs " +
                                           std::to_string(inputs.tree_alignment.sites) + ")");
        }

        validate_alignment_names(inputs.tree_alignment, "--tree-alignment");
        validate_alignment_names(inputs.query_alignment, "--query-alignment");
        validate_alignment_symbols(inputs.tree_alignment, inputs.config.model.states, "--tree-alignment");
        validate_alignment_symbols(inputs.query_alignment, inputs.config.model.states, "--query-alignment");
        if (commit_to_tree) {
            validate_query_reference_name_overlap(
                inputs.tree_alignment,
                inputs.query_alignment,
                "--query-alignment");
        }
    } catch (const CLI::Error& e) {
        return app.exit(e);
    }

    try {
    start_gpu = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));

    const auto& alignment = inputs.tree_alignment;

    const auto& msa_names = alignment.names;
    std::vector<std::string> rows = alignment.sequences;
    size_t sites = alignment.sites;
    const std::string& newick = inputs.tree;
    const auto& model = inputs.config.model;
    int states = model.states;
    int rate_cats = model.ncat;
    bool per_rate_scaling = model.per_rate_scaling;
    std::vector<unsigned> pattern_weights(sites, 1u);

    std::vector<double> pi =
        empirical_freqs ? estimate_empirical_pi(alignment, states)
                        : ensure_normalized_pi(model.freqs, states);
    std::vector<double> rate_weights = build_mixture_weights(model, rate_cats);
    std::vector<double> rate_multipliers = build_gamma_rate_categories(model.alpha, rate_cats);
    std::vector<double> Q = build_gtr_q_matrix(states, model, pi);

    std::cout << "Equilibrium frequencies ("
              << (empirical_freqs ? "empirical" : (model.freqs.empty() ? "uniform" : "manual"))
              << ") =";
    for (double value : pi) {
        std::cout << ' ' << std::fixed << std::setprecision(8) << value;
    }
    std::cout << "\n";

    std::vector<NewPlacementQuery> placement_queries =
        build_placement_query(inputs.query_alignment.names, inputs.query_alignment.sequences);
    if (repetitive_column_compression_enabled()) {
        remove_repetitive_columns(rows, placement_queries, pattern_weights, sites);
        if (sites == 0) {
            throw std::runtime_error("All columns were removed after repetitive-column compression.");
        }
    }
    const bool disable_pattern_weights = env_flag_enabled("MLIPPER_DISABLE_PATTERN_WEIGHTS");
    const std::vector<unsigned> no_pattern_weights;
    const std::vector<unsigned>& pattern_weights_arg =
        disable_pattern_weights ? no_pattern_weights : pattern_weights;

    maybe_expand_cuda_printf_fifo();

    set_int_env_if_specified("MLIPPER_FULL_OPT_PASSES", full_opt_passes);
    set_int_env_if_specified("MLIPPER_REFINE_GLOBAL_PASSES", refine_global_passes);
    set_int_env_if_specified("MLIPPER_REFINE_EXTRA_PASSES", refine_extra_passes);
    set_int_env_if_specified("MLIPPER_REFINE_DETECT_TOPK", refine_detect_topk);
    set_int_env_if_specified("MLIPPER_REFINE_TOPK", refine_topk);
    set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP2", refine_gap_top2);
    set_double_env_if_specified("MLIPPER_REFINE_GAP_TOP5", refine_gap_top5);
    set_double_env_if_specified("MLIPPER_REFINE_CONVERGED_LOGLK_EPS", refine_converged_loglk_eps);
    set_double_env_if_specified("MLIPPER_REFINE_CONVERGED_LENGTH_EPS", refine_converged_length_eps);
    if (placement_fast) {
        if (full_opt_passes < 0) {
            setenv("MLIPPER_FULL_OPT_PASSES", "1", 1);
        }
        if (refine_global_passes < 0) {
            setenv("MLIPPER_REFINE_GLOBAL_PASSES", "0", 1);
        }
        if (refine_extra_passes < 0) {
            setenv("MLIPPER_REFINE_EXTRA_PASSES", "0", 1);
        }
        if (refine_detect_topk < 0) {
            setenv("MLIPPER_REFINE_DETECT_TOPK", "0", 1);
        }
        if (refine_topk < 0) {
            setenv("MLIPPER_REFINE_TOPK", "0", 1);
        }
    }

    printf("Precision mode: %s\n", FP_MODE_NAME);
    // const auto start_gpu = std::chrono::steady_clock::now();
    std::vector<PlacementResult> placement_results;
    std::vector<std::string> committed_query_names(placement_queries.size());

    if (commit_to_tree && batch_insert_size > 0 && !placement_queries.empty()) {
        if (!jplace_out.empty()) {
            throw std::runtime_error("Batch insert mode does not support --jplace-out.");
        }
        const bool profile_batch_timing = []() {
            const char* env = std::getenv("MLIPPER_PROFILE_COMMIT_TIMING");
            return env && std::atoi(env) != 0;
        }();
        struct BatchCommitTimingStats {
            double free_prev_ms = 0.0;
            double build_ms = 0.0;
            double initial_update_ms = 0.0;
            double evaluate_ms = 0.0;
            double append_ms = 0.0;
            double newick_ms = 0.0;
            double local_spr_ms = 0.0;
            int batches = 0;
            int queries = 0;
            CommitTimingStats commit{};
        };
        using BatchCommitClock = std::chrono::steady_clock;
        auto batch_commit_elapsed_ms = [](const BatchCommitClock::time_point& start) {
            return std::chrono::duration<double, std::milli>(
                BatchCommitClock::now() - start).count();
        };
        auto accumulate_commit_timing = [](CommitTimingStats& dst, const CommitTimingStats& src) {
            dst.initial_upward_host_ms += src.initial_upward_host_ms;
            dst.initial_downward_host_ms += src.initial_downward_host_ms;
            dst.initial_upward_stage_ms += src.initial_upward_stage_ms;
            dst.initial_downward_stage_ms += src.initial_downward_stage_ms;
            dst.query_reset_stage_ms += src.query_reset_stage_ms;
            dst.query_build_clv_stage_ms += src.query_build_clv_stage_ms;
            dst.query_kernel_total_ms += src.query_kernel_total_ms;
            dst.insertion_pre_clv_ms += src.insertion_pre_clv_ms;
            dst.insertion_upward_host_ms += src.insertion_upward_host_ms;
            dst.insertion_downward_host_ms += src.insertion_downward_host_ms;
            dst.insertion_upward_stage_ms += src.insertion_upward_stage_ms;
            dst.insertion_downward_stage_ms += src.insertion_downward_stage_ms;
            dst.initial_upward_ops += src.initial_upward_ops;
            dst.initial_downward_ops += src.initial_downward_ops;
            dst.insertion_upward_ops += src.insertion_upward_ops;
            dst.insertion_downward_ops += src.insertion_downward_ops;
            dst.initial_updates += src.initial_updates;
            dst.query_evals += src.query_evals;
            dst.insertion_updates += src.insertion_updates;
        };
        BatchCommitTimingStats batch_commit_timing;
        cudaEventRecord(start);
        std::vector<std::string> current_names = msa_names;
        std::vector<std::string> current_rows = rows;
        std::string current_tree_newick = newick;
        const int total_queries = static_cast<int>(placement_queries.size());
        placement_results.resize(placement_queries.size());

        for (int batch_start = 0; batch_start < total_queries; batch_start += batch_insert_size) {
            const int batch_end = std::min(batch_start + batch_insert_size, total_queries);
            ++batch_commit_timing.batches;
            std::vector<NewPlacementQuery> batch_queries;
            std::vector<std::string> batch_query_names;
            std::vector<int> batch_indices;
            batch_queries.reserve(batch_end - batch_start);
            batch_query_names.reserve(batch_end - batch_start);
            batch_indices.reserve(batch_end - batch_start);
            for (int idx = batch_start; idx < batch_end; ++idx) {
                batch_queries.push_back(placement_queries[(size_t)idx]);
                batch_query_names.push_back(placement_queries[(size_t)idx].msa_name);
                batch_indices.push_back(idx);
            }
            batch_commit_timing.queries += static_cast<int>(batch_queries.size());

            if (res.dev.N != 0) {
                const auto free_prev_start = BatchCommitClock::now();
                free_placement_op_buffer(placement_ops, stream);
                cudaStreamSynchronize(stream);
                free_device_tree(res.dev);
                batch_commit_timing.free_prev_ms += batch_commit_elapsed_ms(free_prev_start);
            }
            const auto build_start = BatchCommitClock::now();
            res = BuildAllToGPU(
                current_names,
                current_rows,
                current_tree_newick,
                Q,
                pi,
                rate_multipliers,
                rate_weights,
                pattern_weights_arg,
                sites,
                states,
                rate_cats,
                per_rate_scaling,
                batch_queries,
                true);
            batch_commit_timing.build_ms += batch_commit_elapsed_ms(build_start);
            if (res.tree.nodes.empty() || res.dev.N == 0) {
                throw std::runtime_error("BuildAllToGPU returned empty tree/device structures.");
            }
            if (res.tree.root_id < 0) {
                throw std::runtime_error("BuildAllToGPU produced tree with invalid root_id.");
            }

            placement_ops = PlacementOpBuffer{};
            placement_ops.profile_commit_timing = profile_batch_timing;
            const auto initial_update_start = BatchCommitClock::now();
            UpdateTreeClvs(
                res.dev,
                res.tree,
                res.hostPack,
                placement_ops,
                stream);
            batch_commit_timing.initial_update_ms +=
                batch_commit_elapsed_ms(initial_update_start);
            std::vector<PlacementResult> batch_results;
            std::vector<std::string> inserted_names(batch_queries.size());
            PlacementCommitContext batch_ctx;
            batch_ctx.tree = &res.tree;
            batch_ctx.host = &res.hostPack;
            batch_ctx.queries = &res.queries;
            batch_ctx.placement_ops = &placement_ops;
            batch_ctx.query_names = &batch_query_names;
            batch_ctx.inserted_query_names = &inserted_names;
            const auto evaluate_start = BatchCommitClock::now();
            EvaluatePlacementQueries(
                res.dev,
                res.eig,
                rate_multipliers,
                batch_ctx,
                &batch_results,
                1,
                true,
                stream);
            batch_commit_timing.evaluate_ms += batch_commit_elapsed_ms(evaluate_start);
            if (profile_batch_timing) {
                accumulate_commit_timing(batch_commit_timing.commit, placement_ops.timing);
            }

            const auto append_start = BatchCommitClock::now();
            for (size_t i = 0; i < batch_indices.size(); ++i) {
                const int qidx = batch_indices[i];
                committed_query_names[(size_t)qidx] = inserted_names[i];
                if (i < batch_results.size()) {
                    placement_results[(size_t)qidx] = batch_results[i];
                }
                current_names.push_back(inserted_names[i]);
                current_rows.push_back(placement_queries[(size_t)qidx].msa);
            }
            batch_commit_timing.append_ms += batch_commit_elapsed_ms(append_start);

            const auto newick_start = BatchCommitClock::now();
            current_tree_newick = write_tree_to_newick_string(res.tree);
            batch_commit_timing.newick_ms += batch_commit_elapsed_ms(newick_start);

            if (local_spr) {
                const std::vector<std::string>& local_spr_inserted_names = inserted_names;
                const auto local_spr_batch_start = BatchCommitClock::now();
                if (local_spr_inserted_names.empty()) {
                    std::cout << "Local SPR skipped: no inserted placements in this batch.\n";
                    batch_commit_timing.local_spr_ms +=
                        batch_commit_elapsed_ms(local_spr_batch_start);
                    continue;
                }
                struct EnvSnapshot {
                    const char* key;
                    std::string value;
                    bool has_value;
                };
                struct LocalSprTimingStats {
                    double total_ms = 0.0;
                    double prep_ms = 0.0;
                    double hot_filter_ms = 0.0;
                    double anchor_unit_ms = 0.0;
                    double search_ms = 0.0;
                    double score_host_prep_ms = 0.0;
                    double score_filter_align_ms = 0.0;
                    double score_pack_host_ms = 0.0;
                    double score_fill_pmat_ms = 0.0;
                    double score_upload_ms = 0.0;
                    double score_reload_branch_copy_ms = 0.0;
                    double score_reload_branch_reset_ms = 0.0;
                    double score_reload_tipchar_copy_ms = 0.0;
                    double score_reload_clv_reset_ms = 0.0;
                    double score_reload_root_seed_ms = 0.0;
                    double score_reload_pmat_copy_ms = 0.0;
                    double score_reload_query_copy_ms = 0.0;
                    double score_reload_pattern_copy_ms = 0.0;
                    double score_update_ms = 0.0;
                    double score_kernel_ms = 0.0;
                    double score_cleanup_ms = 0.0;
                    double score_query_clv_ms = 0.0;
                    double score_build_ops_ms = 0.0;
                    double score_upload_ops_ms = 0.0;
                    double selection_ms = 0.0;
                    double validation_ms = 0.0;
                    double validation_tree_edit_ms = 0.0;
                    double validation_eval_ms = 0.0;
                    double eval_build_ms = 0.0;
                    double eval_update_ms = 0.0;
                    double eval_logl_ms = 0.0;
                    double eval_cleanup_ms = 0.0;
                    double rebuild_ms = 0.0;
                    double rebuild_build_ms = 0.0;
                    double rebuild_update_ms = 0.0;
                    int scoring_instances = 0;
                    int validation_candidates = 0;
                    int eval_calls = 0;
                    int accepted_candidates = 0;
                };
                using LocalSprClock = std::chrono::steady_clock;
                auto local_spr_elapsed_ms = [](const LocalSprClock::time_point& start) {
                    return std::chrono::duration<double, std::milli>(
                        LocalSprClock::now() - start).count();
                };
                auto local_spr_time_stream_stage_ms = [&](const std::function<void()>& launch_fn) {
                    cudaEvent_t stage_start = nullptr;
                    cudaEvent_t stage_stop = nullptr;
                    CUDA_CHECK(cudaEventCreate(&stage_start));
                    CUDA_CHECK(cudaEventCreate(&stage_stop));
                    CUDA_CHECK(cudaEventRecord(stage_start, stream));
                    launch_fn();
                    CUDA_CHECK(cudaEventRecord(stage_stop, stream));
                    CUDA_CHECK(cudaEventSynchronize(stage_stop));
                    float stage_ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&stage_ms, stage_start, stage_stop));
                    CUDA_CHECK(cudaEventDestroy(stage_start));
                    CUDA_CHECK(cudaEventDestroy(stage_stop));
                    return static_cast<double>(stage_ms);
                };
                auto snapshot_env = [](const char* key) {
                    const char* value = std::getenv(key);
                    if (value) {
                        return EnvSnapshot{key, std::string(value), true};
                    }
                    return EnvSnapshot{key, std::string(), false};
                };
                auto restore_env = [](const EnvSnapshot& snap) {
                    if (snap.has_value) {
                        setenv(snap.key, snap.value.c_str(), 1);
                    } else {
                        unsetenv(snap.key);
                    }
                };

                const EnvSnapshot prev_full_opt = snapshot_env("MLIPPER_FULL_OPT_PASSES");
                const EnvSnapshot prev_refine_global = snapshot_env("MLIPPER_REFINE_GLOBAL_PASSES");
                const EnvSnapshot prev_refine_extra = snapshot_env("MLIPPER_REFINE_EXTRA_PASSES");
                const EnvSnapshot prev_refine_detect = snapshot_env("MLIPPER_REFINE_DETECT_TOPK");
                const EnvSnapshot prev_refine_topk = snapshot_env("MLIPPER_REFINE_TOPK");
                const EnvSnapshot prev_export_topk = snapshot_env("MLIPPER_EXPORT_PLACEMENT_TOPK");
                const EnvSnapshot prev_local_child_refine = snapshot_env("MLIPPER_LOCAL_CHILD_REFINE");
                const EnvSnapshot prev_double_rerank = snapshot_env("MLIPPER_DOUBLE_RERANK");

                setenv("MLIPPER_FULL_OPT_PASSES", local_spr_fast ? "1" : "4", 1);
                setenv("MLIPPER_REFINE_GLOBAL_PASSES", "0", 1);
                setenv("MLIPPER_REFINE_EXTRA_PASSES", "0", 1);
                setenv("MLIPPER_REFINE_DETECT_TOPK", "0", 1);
                setenv("MLIPPER_REFINE_TOPK", "0", 1);
                setenv(
                    "MLIPPER_EXPORT_PLACEMENT_TOPK",
                    std::to_string(std::max(1, static_cast<int>(res.tree.nodes.size()) * 2)).c_str(),
                    1);
                setenv("MLIPPER_LOCAL_CHILD_REFINE", "0", 1);
                setenv("MLIPPER_DOUBLE_RERANK", "0", 1);

                LocalSprTimingStats local_spr_timing;
                const auto local_spr_total_start = LocalSprClock::now();
                const double logL_accept_eps = 0.0;
                int local_spr_rounds_executed = 0;
                auto build_tree_host_packing = [&](const TreeBuildResult& tree) -> HostPacking {
                    HostPacking host_pack = pack_host_arrays_from_tree_and_msa(
                        tree,
                        current_names,
                        current_rows,
                        sites,
                        states);
                    host_pack.pattern_weights = pattern_weights_arg;
                    fill_pmats_in_host_packing(
                        tree,
                        host_pack,
                        res.eig,
                        rate_multipliers,
                        states,
                        rate_cats);
                    return host_pack;
                };
                BuildToGpuResult eval_res{};
                PlacementOpBuffer eval_ops{};
                bool eval_workspace_initialized = false;
                auto release_eval_workspace = [&]() {
                    if (!eval_workspace_initialized) {
                        return;
                    }
                    const auto eval_cleanup_start = LocalSprClock::now();
                    free_placement_op_buffer(eval_ops, stream);
                    cudaStreamSynchronize(stream);
                    free_device_tree(eval_res.dev);
                    eval_res = BuildToGpuResult{};
                    eval_workspace_initialized = false;
                    local_spr_timing.eval_cleanup_ms +=
                        local_spr_elapsed_ms(eval_cleanup_start);
                };
                auto eval_logL = [&](const TreeBuildResult& candidate_tree) -> double {
                    ++local_spr_timing.eval_calls;
                    const auto eval_build_start = LocalSprClock::now();
                    HostPacking candidate_host_pack = build_tree_host_packing(candidate_tree);
                    if (!eval_workspace_initialized) {
                        eval_res.tree = candidate_tree;
                        eval_res.hostPack = std::move(candidate_host_pack);
                        eval_res.eig = res.eig;
                        eval_res.queries = PlacementQueryBatch{};
                        eval_res.dev = upload_to_gpu(
                            eval_res.tree,
                            eval_res.hostPack,
                            eval_res.eig,
                            rate_weights,
                            rate_multipliers,
                            pi,
                            sites,
                            states,
                            rate_cats,
                            per_rate_scaling,
                            nullptr,
                            false);
                        eval_workspace_initialized = true;
                    } else {
                        eval_res.tree = candidate_tree;
                        eval_res.hostPack = std::move(candidate_host_pack);
                        reload_device_tree_live_data(
                            eval_res.dev,
                            eval_res.tree,
                            eval_res.hostPack,
                            nullptr,
                            stream);
                    }
                    local_spr_timing.eval_build_ms += local_spr_elapsed_ms(eval_build_start);
                    if (eval_res.tree.nodes.empty() || eval_res.dev.N == 0) {
                        throw std::runtime_error("Local SPR eval produced empty tree/device structures.");
                    }
                    if (eval_res.tree.root_id < 0) {
                        throw std::runtime_error("Local SPR eval produced tree with invalid root_id.");
                    }
                    const auto eval_update_start = LocalSprClock::now();
                    UpdateTreeClvs(
                        eval_res.dev,
                        eval_res.tree,
                        eval_res.hostPack,
                        eval_ops,
                        stream);
                    local_spr_timing.eval_update_ms += local_spr_elapsed_ms(eval_update_start);
                    const auto eval_logl_start = LocalSprClock::now();
                    const double logL = root_likelihood::compute_root_loglikelihood_total(
                        eval_res.dev,
                        eval_res.tree.root_id,
                        eval_res.dev.d_pattern_weights_u,
                        nullptr,
                        0.0,
                        0);
                    local_spr_timing.eval_logl_ms += local_spr_elapsed_ms(eval_logl_start);
                    return logL;
                };

                for (int local_spr_round = 1;
                     local_spr_round <= local_spr_rounds;
                     ++local_spr_round) {
                    ++local_spr_rounds_executed;
                    const auto local_spr_prep_start = LocalSprClock::now();
                    double current_logL = root_likelihood::compute_root_loglikelihood_total(
                        res.dev,
                        res.tree.root_id,
                        res.dev.d_pattern_weights_u,
                        nullptr,
                        0.0,
                        0);
                    TreeBuildResult base_tree = res.tree;

                const auto anchor_unit_start = LocalSprClock::now();
                const std::vector<LocalSprInsertionAnchor> anchors =
                    build_local_spr_insertion_anchors(base_tree, local_spr_inserted_names);
                const std::vector<LocalSprRepairUnit> repair_units =
                    build_local_spr_repair_units(
                        base_tree,
                        anchors,
                        local_spr_cluster_threshold,
                        local_spr_radius);
                local_spr_timing.anchor_unit_ms += local_spr_elapsed_ms(anchor_unit_start);
                local_spr_timing.prep_ms += local_spr_elapsed_ms(local_spr_prep_start);

                LocalSprSearchSummary search_summary;
                search_summary.unit_count = repair_units.size();
                std::vector<LocalSprCandidateMove> ranked_candidates;
                PlacementQueryBatch subtree_query_batch;
                subtree_query_batch.count = 1;
                subtree_query_batch.branch_lengths.assign(1, fp_t(0.5));
                subtree_query_batch.query_chars.assign(
                    sites,
                    static_cast<uint8_t>(states == 4 ? 15 : 4));
                DeviceTree local_spr_scoring_dev{};
                PlacementOpBuffer local_spr_tree_ops{};
                PlacementOpBuffer local_spr_candidate_ops{};
                int local_spr_prev_main_pmat_node = -1;
                auto release_local_spr_scoring_workspace = [&]() {
                    const auto cleanup_start = LocalSprClock::now();
                    free_placement_op_buffer(local_spr_candidate_ops, stream);
                    free_placement_op_buffer(local_spr_tree_ops, stream);
                    cudaStreamSynchronize(stream);
                    free_device_tree(local_spr_scoring_dev);
                    local_spr_timing.score_cleanup_ms +=
                        local_spr_elapsed_ms(cleanup_start);
                };

                const auto search_start = LocalSprClock::now();
                try {
                    const auto score_workspace_init_start = LocalSprClock::now();
                    local_spr_scoring_dev = upload_to_gpu(
                        base_tree,
                        res.hostPack,
                        res.eig,
                        rate_weights,
                        rate_multipliers,
                        pi,
                        sites,
                        states,
                        rate_cats,
                        per_rate_scaling,
                        &subtree_query_batch,
                        false);
                    local_spr_timing.score_upload_ms +=
                        local_spr_elapsed_ms(score_workspace_init_start);

                    const int inner_search_radius = std::min(local_spr_radius, 2);
                    const bool staged_radius_expansion =
                        local_spr_radius > inner_search_radius;
                    const int local_spr_candidate_pool_limit = local_spr_topk_per_unit;
                    const int local_spr_outer_seed_limit = local_spr_topk_per_unit;
                    const bool local_spr_use_tier_early_stop = true;
                    for (const LocalSprRepairUnit& unit : repair_units) {
                        std::vector<LocalSprCandidateMove> unit_topk;
                        const std::vector<char> anchor_skeleton_mask =
                            build_unit_anchor_skeleton_mask(base_tree, unit.anchor_ids);
                        const std::vector<int> skeleton_distance_by_node =
                            multi_source_bfs_distances(base_tree, anchor_skeleton_mask);
                        std::vector<LocalSprPruneRootWorkItem> prune_root_work_items;
                        prune_root_work_items.reserve(unit.envelope_nodes.size());

                        for (int candidate_prune_root_id : unit.envelope_nodes) {
                            if (candidate_prune_root_id < 0 ||
                                candidate_prune_root_id >= static_cast<int>(base_tree.nodes.size()) ||
                                candidate_prune_root_id == base_tree.root_id ||
                                base_tree.nodes[(size_t)candidate_prune_root_id].parent < 0) {
                                continue;
                            }

                            std::vector<int> subtree_nodes;
                            if (!subtree_fully_inside_mask(
                                    base_tree,
                                    candidate_prune_root_id,
                                    unit.envelope_mask,
                                    &subtree_nodes)) {
                                continue;
                            }

                            std::vector<char> subtree_mask(base_tree.nodes.size(), 0);
                            for (int node_id : subtree_nodes) {
                                if (node_id >= 0 &&
                                    node_id < static_cast<int>(subtree_mask.size())) {
                                    subtree_mask[(size_t)node_id] = 1;
                                }
                            }

                            TreeBuildResult pruned_tree = base_tree;
                            PruneInfo prune_info;
                            if (!prune_subtree_for_spr(
                                    pruned_tree,
                                    candidate_prune_root_id,
                                    prune_info)) {
                                continue;
                            }

                            std::vector<int> inner_candidate_edges = collect_candidate_edges(
                                pruned_tree,
                                prune_info.sibling_id,
                                prune_info.grandparent_id,
                                inner_search_radius,
                                prune_info.pruned_id,
                                prune_info.free_internal_id);
                            int legal_inner_edge_count = 0;
                            for (int edge_child : inner_candidate_edges) {
                                if (edge_child < 0 ||
                                    edge_child >= static_cast<int>(pruned_tree.nodes.size())) {
                                    continue;
                                }
                                const int edge_parent =
                                    pruned_tree.nodes[(size_t)edge_child].parent;
                                if (edge_parent < 0) continue;
                                if (!unit.envelope_mask[(size_t)edge_child] ||
                                    !unit.envelope_mask[(size_t)edge_parent]) {
                                    continue;
                                }
                                if (subtree_mask[(size_t)edge_child] ||
                                    subtree_mask[(size_t)edge_parent]) {
                                    continue;
                                }
                                ++legal_inner_edge_count;
                            }
                            if (legal_inner_edge_count <= 0) {
                                continue;
                            }

                            int skeleton_distance = std::numeric_limits<int>::max();
                            if (candidate_prune_root_id <
                                static_cast<int>(skeleton_distance_by_node.size())) {
                                const int dist =
                                    skeleton_distance_by_node[(size_t)candidate_prune_root_id];
                                if (dist >= 0) {
                                    skeleton_distance = dist;
                                }
                            }

                            LocalSprPruneRootWorkItem work_item;
                            work_item.prune_root_id = candidate_prune_root_id;
                            work_item.skeleton_distance = skeleton_distance;
                            work_item.legal_inner_edge_count = legal_inner_edge_count;
                            work_item.subtree_size = static_cast<int>(subtree_nodes.size());
                            prune_root_work_items.push_back(std::move(work_item));
                        }

                        std::sort(
                            prune_root_work_items.begin(),
                            prune_root_work_items.end(),
                            [](const LocalSprPruneRootWorkItem& lhs,
                               const LocalSprPruneRootWorkItem& rhs) {
                                if (lhs.skeleton_distance != rhs.skeleton_distance) {
                                    return lhs.skeleton_distance < rhs.skeleton_distance;
                                }
                                if (lhs.legal_inner_edge_count != rhs.legal_inner_edge_count) {
                                    return lhs.legal_inner_edge_count > rhs.legal_inner_edge_count;
                                }
                                if (lhs.subtree_size != rhs.subtree_size) {
                                    return lhs.subtree_size < rhs.subtree_size;
                                }
                                return lhs.prune_root_id < rhs.prune_root_id;
                            });
                        if (prune_root_work_items.empty()) {
                            continue;
                        }

                        auto evaluate_prune_root = [&](int prune_root_id) {
                            if (prune_root_id < 0 ||
                                prune_root_id >= static_cast<int>(base_tree.nodes.size()) ||
                                prune_root_id == base_tree.root_id ||
                                base_tree.nodes[(size_t)prune_root_id].parent < 0) {
                                return;
                            }

                            std::vector<int> subtree_nodes;
                            if (!subtree_fully_inside_mask(
                                    base_tree,
                                    prune_root_id,
                                    unit.envelope_mask,
                                    &subtree_nodes)) {
                                return;
                            }

                            std::vector<char> subtree_mask(base_tree.nodes.size(), 0);
                            for (int node_id : subtree_nodes) {
                                if (node_id >= 0 &&
                                    node_id < static_cast<int>(subtree_mask.size())) {
                                    subtree_mask[(size_t)node_id] = 1;
                                }
                            }

                            TreeBuildResult pruned_tree = base_tree;
                            PruneInfo prune_info;
                            if (!prune_subtree_for_spr(pruned_tree, prune_root_id, prune_info)) {
                                return;
                            }

                            const std::vector<int> center_dist = compute_center_distances(
                                pruned_tree,
                                prune_info.sibling_id,
                                prune_info.grandparent_id);
                            std::vector<int> candidate_edges = collect_candidate_edges(
                                pruned_tree,
                                prune_info.sibling_id,
                                prune_info.grandparent_id,
                                inner_search_radius,
                                prune_info.pruned_id,
                                prune_info.free_internal_id);
                            auto filter_legal_candidate_edges =
                                [&](const std::vector<int>& edge_candidates) {
                                    std::vector<int> legal_candidate_edges;
                                    legal_candidate_edges.reserve(edge_candidates.size());
                                    for (int edge_child : edge_candidates) {
                                        if (edge_child < 0 ||
                                            edge_child >= static_cast<int>(pruned_tree.nodes.size())) {
                                            continue;
                                        }
                                        const int edge_parent =
                                            pruned_tree.nodes[(size_t)edge_child].parent;
                                        if (edge_parent < 0) continue;
                                        if (!unit.envelope_mask[(size_t)edge_child] ||
                                            !unit.envelope_mask[(size_t)edge_parent]) {
                                            continue;
                                        }
                                        if (subtree_mask[(size_t)edge_child] ||
                                            subtree_mask[(size_t)edge_parent]) {
                                            continue;
                                        }
                                        legal_candidate_edges.push_back(edge_child);
                                    }
                                    return legal_candidate_edges;
                                };
                            std::vector<int> legal_candidate_edges =
                                filter_legal_candidate_edges(candidate_edges);
                            search_summary.enumerated_candidates += legal_candidate_edges.size();
                            if (legal_candidate_edges.empty()) {
                                return;
                            }

                            ++local_spr_timing.scoring_instances;
                            const auto score_host_prep_start = LocalSprClock::now();
                            const auto score_pack_host_start = LocalSprClock::now();
                            HostPacking pruned_host = res.hostPack;
                            rebuild_host_topology_from_tree_local(pruned_tree, pruned_host);
                            local_spr_timing.score_pack_host_ms +=
                                local_spr_elapsed_ms(score_pack_host_start);
                            pruned_host.pattern_weights = pattern_weights_arg;

                            const auto score_fill_pmat_start = LocalSprClock::now();
                            int changed_nodes[1] = { prune_info.sibling_id };
                            fill_pmats_in_host_packing(
                                pruned_tree,
                                pruned_host,
                                res.eig,
                                rate_multipliers,
                                states,
                                rate_cats,
                                changed_nodes,
                                1);
                            local_spr_timing.score_fill_pmat_ms +=
                                local_spr_elapsed_ms(score_fill_pmat_start);
                            local_spr_timing.score_host_prep_ms +=
                                local_spr_elapsed_ms(score_host_prep_start);

                            local_spr_timing.score_upload_ms +=
                                local_spr_time_stream_stage_ms([&]() {
                                    reload_device_tree_live_data_local_spr(
                                        local_spr_scoring_dev,
                                        pruned_tree,
                                        pruned_host,
                                        res.hostPack,
                                        prune_info.sibling_id,
                                        local_spr_prev_main_pmat_node,
                                        nullptr,
                                        stream,
                                        nullptr);
                                });

                            const auto score_build_ops_start = LocalSprClock::now();
                            std::vector<NodeOpInfo> blo_host_ops;
                            std::vector<NodeOpInfo> blo_closure_ops;
                            build_selected_downward_ops(
                                pruned_tree,
                                pruned_host,
                                legal_candidate_edges,
                                blo_host_ops);
                            build_selected_downward_closure_ops(
                                pruned_tree,
                                pruned_host,
                                legal_candidate_edges,
                                blo_closure_ops);
                            local_spr_assert(
                                !blo_host_ops.empty(),
                                "subtree BLO received zero candidate ops");
                            local_spr_assert(
                                !blo_closure_ops.empty(),
                                "subtree BLO received zero closure ops");
                            local_spr_timing.score_build_ops_ms +=
                                local_spr_elapsed_ms(score_build_ops_start);

                            const int upward_start_node = prune_info.grandparent_id;
                            local_spr_timing.score_update_ms +=
                                local_spr_time_stream_stage_ms([&]() {
                                    copy_upward_state(
                                        res.dev,
                                        local_spr_scoring_dev,
                                        stream);
                                    UpdateTreeClvsAfterPrune(
                                        local_spr_scoring_dev,
                                        pruned_tree,
                                        pruned_host,
                                        local_spr_tree_ops,
                                        upward_start_node,
                                        blo_closure_ops,
                                        stream);
                                });

                            local_spr_timing.score_query_clv_ms +=
                                local_spr_time_stream_stage_ms([&]() {
                                    copy_unscaled_up_clv_to_query_slot(
                                        res.dev,
                                        prune_root_id,
                                        local_spr_scoring_dev,
                                        0,
                                        stream);
                                });

                            local_spr_timing.score_upload_ops_ms +=
                                local_spr_time_stream_stage_ms([&]() {
                                    UploadPlacementOps(local_spr_candidate_ops, blo_host_ops, stream);
                                });

                            DeviceTree query_view = make_query_view(local_spr_scoring_dev, 0);
                            PlacementResult blo_result;
                            local_spr_timing.score_kernel_ms +=
                                local_spr_time_stream_stage_ms([&]() {
                                    blo_result = PlacementEvaluationKernel(
                                        query_view,
                                        res.eig,
                                        rate_multipliers,
                                        local_spr_candidate_ops.d_ops,
                                        local_spr_candidate_ops.num_ops,
                                        1,
                                        stream);
                                });

                            local_spr_assert(
                                blo_result.top_placements.size() == blo_host_ops.size(),
                                "subtree BLO did not return a score for every legal regraft edge");

                            double baseline_logL = -std::numeric_limits<double>::infinity();
                            if (prune_info.grandparent_id >= 0) {
                                for (const PlacementResult::RankedPlacement& placement :
                                     blo_result.top_placements) {
                                    if (placement.target_id == prune_info.sibling_id) {
                                        baseline_logL = placement.loglikelihood;
                                        break;
                                    }
                                }
                            }
                            if (!std::isfinite(baseline_logL)) {
                                return;
                            }

                            if (staged_radius_expansion) {
                                const std::vector<int> outer_seed_edges =
                                    select_local_spr_seed_edges(
                                        blo_result,
                                        pruned_tree,
                                        center_dist,
                                        baseline_logL,
                                        local_spr_outer_seed_limit,
                                        inner_search_radius);
                                if (!outer_seed_edges.empty()) {
                                    std::vector<int> outer_candidate_edges;
                                    outer_candidate_edges.reserve(
                                        outer_seed_edges.size() *
                                        std::max(1, local_spr_radius - inner_search_radius));
                                    std::unordered_set<int> seen_outer_edges;
                                    for (int seed_edge_child_id : outer_seed_edges) {
                                        const std::vector<int> expanded_edges =
                                            collect_outward_edges_from_seed(
                                                pruned_tree,
                                                center_dist,
                                                seed_edge_child_id,
                                                inner_search_radius,
                                                local_spr_radius,
                                                prune_info.pruned_id,
                                                prune_info.free_internal_id);
                                        for (int edge_child_id : expanded_edges) {
                                            if (!seen_outer_edges.insert(edge_child_id).second) {
                                                continue;
                                            }
                                            outer_candidate_edges.push_back(edge_child_id);
                                        }
                                    }

                                    const std::vector<int> outer_legal_candidate_edges =
                                        filter_legal_candidate_edges(outer_candidate_edges);
                                    search_summary.enumerated_candidates +=
                                        outer_legal_candidate_edges.size();
                                    if (!outer_legal_candidate_edges.empty()) {
                                        ++local_spr_timing.scoring_instances;
                                        const auto outer_build_ops_start = LocalSprClock::now();
                                        std::vector<NodeOpInfo> outer_host_ops;
                                        std::vector<NodeOpInfo> outer_closure_ops;
                                        build_selected_downward_ops(
                                            pruned_tree,
                                            pruned_host,
                                            outer_legal_candidate_edges,
                                            outer_host_ops);
                                        build_selected_downward_closure_ops(
                                            pruned_tree,
                                            pruned_host,
                                            outer_legal_candidate_edges,
                                            outer_closure_ops);
                                        local_spr_assert(
                                            !outer_host_ops.empty(),
                                            "outer local SPR scoring produced zero candidate ops");
                                        local_spr_assert(
                                            !outer_closure_ops.empty(),
                                            "outer local SPR scoring produced zero closure ops");
                                        std::vector<NodeOpInfo> outer_new_closure_ops =
                                            filter_local_spr_new_ops(
                                                outer_closure_ops,
                                                blo_closure_ops);
                                        local_spr_timing.score_build_ops_ms +=
                                            local_spr_elapsed_ms(outer_build_ops_start);

                                        if (!outer_new_closure_ops.empty()) {
                                            local_spr_timing.score_update_ms +=
                                                local_spr_time_stream_stage_ms([&]() {
                                                    UpdateTreeClvsAfterPrune(
                                                        local_spr_scoring_dev,
                                                        pruned_tree,
                                                        pruned_host,
                                                        local_spr_tree_ops,
                                                        -1,
                                                        outer_new_closure_ops,
                                                        stream);
                                                });
                                        }

                                        local_spr_timing.score_query_clv_ms +=
                                            local_spr_time_stream_stage_ms([&]() {
                                                copy_unscaled_up_clv_to_query_slot(
                                                    res.dev,
                                                    prune_root_id,
                                                    local_spr_scoring_dev,
                                                    0,
                                                    stream);
                                            });

                                        local_spr_timing.score_upload_ops_ms +=
                                            local_spr_time_stream_stage_ms([&]() {
                                                UploadPlacementOps(
                                                    local_spr_candidate_ops,
                                                    outer_host_ops,
                                                    stream);
                                            });

                                        DeviceTree outer_query_view =
                                            make_query_view(local_spr_scoring_dev, 0);
                                        PlacementResult outer_result;
                                        local_spr_timing.score_kernel_ms +=
                                            local_spr_time_stream_stage_ms([&]() {
                                                outer_result = PlacementEvaluationKernel(
                                                    outer_query_view,
                                                    res.eig,
                                                    rate_multipliers,
                                                    local_spr_candidate_ops.d_ops,
                                                    local_spr_candidate_ops.num_ops,
                                                    1,
                                                    stream);
                                            });
                                        local_spr_assert(
                                            outer_result.top_placements.size() ==
                                                outer_host_ops.size(),
                                            "outer local SPR scoring did not return a score for every edge");
                                        blo_result.top_placements.insert(
                                            blo_result.top_placements.end(),
                                            std::make_move_iterator(
                                                outer_result.top_placements.begin()),
                                            std::make_move_iterator(
                                                outer_result.top_placements.end()));
                                        std::sort(
                                            blo_result.top_placements.begin(),
                                            blo_result.top_placements.end(),
                                            [](const PlacementResult::RankedPlacement& lhs,
                                               const PlacementResult::RankedPlacement& rhs) {
                                                return lhs.loglikelihood > rhs.loglikelihood;
                                            });
                                    }
                                }
                            }

                            for (const PlacementResult::RankedPlacement& placement :
                                 blo_result.top_placements) {
                                if (placement.target_id < 0 ||
                                    placement.target_id >= static_cast<int>(pruned_tree.nodes.size())) {
                                    continue;
                                }
                                const int edge_child = placement.target_id;
                                const int edge_parent =
                                    pruned_tree.nodes[(size_t)edge_child].parent;
                                if (edge_parent < 0) continue;

                                const double approx_gain =
                                    placement.loglikelihood - baseline_logL;
                                if (!(approx_gain > 0.0)) {
                                    continue;
                                }

                                LocalSprCandidateMove candidate;
                                candidate.repair_unit_id = unit.unit_id;
                                candidate.prune_root_id = prune_root_id;
                                candidate.regraft_child_id = edge_child;
                                candidate.regraft_parent_id = edge_parent;
                                candidate.old_parent_id =
                                    base_tree.nodes[(size_t)prune_root_id].parent;
                                candidate.approx_gain = approx_gain;
                                candidate.pendant_length = placement.pendant_length;
                                candidate.proximal_length =
                                    static_cast<double>(
                                        pruned_tree.nodes[(size_t)edge_child].branch_length_to_parent) -
                                    placement.proximal_length;
                                candidate.subtree_nodes = subtree_nodes;
                                const int path_start =
                                    candidate.old_parent_id >= 0
                                        ? candidate.old_parent_id
                                        : prune_root_id;
                                candidate.regraft_path_nodes =
                                    build_tree_path_nodes(base_tree, path_start, edge_child);
                                maybe_keep_local_spr_topk(
                                    unit_topk,
                                    std::move(candidate),
                                    local_spr_candidate_pool_limit);
                            }
                        };

                        bool stop_after_next_tier = false;
                        for (size_t work_idx = 0; work_idx < prune_root_work_items.size();) {
                            const int tier_distance =
                                prune_root_work_items[work_idx].skeleton_distance;
                            size_t tier_end = work_idx;
                            while (tier_end < prune_root_work_items.size() &&
                                   prune_root_work_items[tier_end].skeleton_distance ==
                                       tier_distance) {
                                ++tier_end;
                            }

                            const size_t prev_topk_size = unit_topk.size();
                            for (; work_idx < tier_end; ++work_idx) {
                                evaluate_prune_root(
                                    prune_root_work_items[work_idx].prune_root_id);
                            }

                            const bool tier_found_positive =
                                unit_topk.size() > prev_topk_size;
                            if (local_spr_use_tier_early_stop && stop_after_next_tier) {
                                break;
                            }
                            if (local_spr_use_tier_early_stop && tier_found_positive) {
                                stop_after_next_tier = true;
                            }
                        }
                        search_summary.retained_candidates += unit_topk.size();
                        ranked_candidates.insert(
                            ranked_candidates.end(),
                            std::make_move_iterator(unit_topk.begin()),
                            std::make_move_iterator(unit_topk.end()));
                    }
                } catch (...) {
                    release_local_spr_scoring_workspace();
                    throw;
                }
                release_local_spr_scoring_workspace();
                local_spr_timing.search_ms += local_spr_elapsed_ms(search_start);

                const auto selection_start = LocalSprClock::now();
                std::sort(
                    ranked_candidates.begin(),
                    ranked_candidates.end(),
                    [](const LocalSprCandidateMove& lhs, const LocalSprCandidateMove& rhs) {
                        return lhs.approx_gain > rhs.approx_gain;
                    });

                std::vector<LocalSprCandidateMove> validation_candidates;
                if (local_spr_dynamic_validation_conflicts) {
                    validation_candidates = ranked_candidates;
                } else {
                    validation_candidates = select_local_spr_candidates(
                        ranked_candidates,
                        static_cast<int>(base_tree.nodes.size()));
                }
                search_summary.selected_candidates = validation_candidates.size();
                local_spr_timing.selection_ms += local_spr_elapsed_ms(selection_start);

                std::cout << "Local SPR round " << local_spr_round
                          << "/" << local_spr_rounds
                          << " subtree repair units: " << search_summary.unit_count
                          << ", enumerated candidates: " << search_summary.enumerated_candidates
                          << ", retained candidates: " << search_summary.retained_candidates
                          << ", selected: " << search_summary.selected_candidates
                          << " (radius=" << local_spr_radius
                          << ", cluster=" << local_spr_cluster_threshold
                          << ", topk=" << local_spr_topk_per_unit
                          << ", selection="
                          << (local_spr_dynamic_validation_conflicts
                                  ? "dynamic-validation"
                                  : "one-per-unit")
                          << ", pool=per-unit-topk"
                          << ")\n";

                const int accepted_before_round = local_spr_timing.accepted_candidates;
                const auto validation_start = LocalSprClock::now();
                for (size_t candidate_idx = 0;
                     candidate_idx < validation_candidates.size();
                     ++candidate_idx) {
                    const LocalSprCandidateMove candidate =
                        validation_candidates[candidate_idx];
                    if (candidate.repair_unit_id < 0 ||
                        candidate.repair_unit_id >= static_cast<int>(repair_units.size())) {
                        continue;
                    }

                    const LocalSprRepairUnit& unit =
                        repair_units[(size_t)candidate.repair_unit_id];
                    if (local_spr_dynamic_validation_conflicts &&
                        !local_spr_candidate_still_legal(
                            base_tree,
                            unit,
                            candidate,
                            nullptr)) {
                        continue;
                    }

                    const auto candidate_tree_edit_start = LocalSprClock::now();
                    local_spr_assert_tree_integrity(base_tree, "before subtree commit");
                    TreeBuildResult candidate_tree = base_tree;
                    PruneInfo prune_info;
                    if (!prune_subtree_for_spr(candidate_tree, candidate.prune_root_id, prune_info)) {
                        continue;
                    }

                    std::vector<int> current_subtree_nodes;
                    collect_subtree_node_ids(candidate_tree, prune_info.pruned_id, current_subtree_nodes);
                    std::vector<char> current_subtree_mask(candidate_tree.nodes.size(), 0);
                    for (int node_id : current_subtree_nodes) {
                        if (node_id >= 0 &&
                            node_id < static_cast<int>(current_subtree_mask.size())) {
                            current_subtree_mask[(size_t)node_id] = 1;
                        }
                    }
                    if (candidate.regraft_child_id < 0 ||
                        candidate.regraft_child_id >= static_cast<int>(candidate_tree.nodes.size())) {
                        continue;
                    }
                    const int regraft_parent =
                        candidate_tree.nodes[(size_t)candidate.regraft_child_id].parent;
                    if (regraft_parent < 0) continue;
                    if (current_subtree_mask[(size_t)candidate.regraft_child_id] ||
                        current_subtree_mask[(size_t)regraft_parent]) {
                        continue;
                    }
                    if (!unit.envelope_mask[(size_t)candidate.regraft_child_id] ||
                        !unit.envelope_mask[(size_t)regraft_parent]) {
                        continue;
                    }

                    std::vector<int> legal_edges = collect_candidate_edges(
                        candidate_tree,
                        prune_info.sibling_id,
                        prune_info.grandparent_id,
                        local_spr_radius,
                        prune_info.pruned_id,
                        prune_info.free_internal_id);
                    if (!local_spr_dynamic_validation_conflicts) {
                        local_spr_assert_candidate_legal(
                            base_tree,
                            unit,
                            candidate.prune_root_id,
                            candidate.subtree_nodes,
                            legal_edges,
                            candidate.regraft_child_id);
                    }

                    regraft_subtree_for_spr(
                        candidate_tree,
                        prune_info,
                        candidate.regraft_child_id,
                        true,
                        candidate.pendant_length,
                        candidate.proximal_length);
                    local_spr_assert_tree_integrity(candidate_tree, "after subtree regraft");
                    local_spr_assert(
                        subtree_fully_inside_mask(
                            candidate_tree,
                            prune_info.pruned_id,
                            unit.envelope_mask),
                        "committed subtree escaped repair envelope after regraft");
                    local_spr_timing.validation_tree_edit_ms +=
                        local_spr_elapsed_ms(candidate_tree_edit_start);
                    ++local_spr_timing.validation_candidates;
                    const auto candidate_eval_start = LocalSprClock::now();
                    const double candidate_logL = eval_logL(candidate_tree);
                    local_spr_timing.validation_eval_ms +=
                        local_spr_elapsed_ms(candidate_eval_start);
                    if (candidate_logL > current_logL + logL_accept_eps) {
                        current_logL = candidate_logL;
                        base_tree = std::move(candidate_tree);
                        ++local_spr_timing.accepted_candidates;
                        if (local_spr_dynamic_validation_conflicts &&
                            candidate_idx + 1 < validation_candidates.size()) {
                            auto keep_begin =
                                validation_candidates.begin() +
                                static_cast<std::ptrdiff_t>(candidate_idx + 1);
                            keep_begin = std::remove_if(
                                keep_begin,
                                validation_candidates.end(),
                                [&](const LocalSprCandidateMove& pending_candidate) {
                                    if (pending_candidate.repair_unit_id < 0 ||
                                        pending_candidate.repair_unit_id >=
                                            static_cast<int>(repair_units.size())) {
                                        return true;
                                    }
                                    return !local_spr_candidate_still_legal(
                                        base_tree,
                                        repair_units[(size_t)pending_candidate.repair_unit_id],
                                        pending_candidate,
                                        nullptr);
                                });
                            validation_candidates.erase(
                                keep_begin,
                                validation_candidates.end());
                        }
                    }
                }
                local_spr_timing.validation_ms += local_spr_elapsed_ms(validation_start);

                const int accepted_this_round =
                    local_spr_timing.accepted_candidates - accepted_before_round;
                if (accepted_this_round <= 0) {
                    std::cout << "Local SPR round " << local_spr_round
                              << " accepted no candidates; stopping.\n";
                    break;
                }
                std::cout << "Local SPR round " << local_spr_round
                          << " accepted " << accepted_this_round
                          << " candidate(s).\n";
                current_tree_newick = write_tree_to_newick_string(base_tree);

                const auto rebuild_start = LocalSprClock::now();
                free_placement_op_buffer(placement_ops, stream);
                cudaStreamSynchronize(stream);

                placement_ops = PlacementOpBuffer{};
                placement_ops.profile_commit_timing = profile_batch_timing;
                const PlacementQueryBatch empty_queries;
                const auto rebuild_build_start = LocalSprClock::now();
                HostPacking rebuilt_host_pack = build_tree_host_packing(base_tree);
                res.tree = base_tree;
                res.hostPack = std::move(rebuilt_host_pack);
                res.queries = PlacementQueryBatch{};
                reload_device_tree_live_data(
                    res.dev,
                    res.tree,
                    res.hostPack,
                    &empty_queries,
                    stream);
                local_spr_timing.rebuild_build_ms += local_spr_elapsed_ms(rebuild_build_start);
                const auto rebuild_update_start = LocalSprClock::now();
                UpdateTreeClvs(
                    res.dev,
                    res.tree,
                    res.hostPack,
                    placement_ops,
                    stream);
                local_spr_timing.rebuild_update_ms += local_spr_elapsed_ms(rebuild_update_start);
                local_spr_timing.rebuild_ms += local_spr_elapsed_ms(rebuild_start);
                }

                release_eval_workspace();
                restore_env(prev_refine_topk);
                restore_env(prev_refine_detect);
                restore_env(prev_refine_extra);
                restore_env(prev_refine_global);
                restore_env(prev_full_opt);
                restore_env(prev_double_rerank);
                restore_env(prev_local_child_refine);
                restore_env(prev_export_topk);

                local_spr_timing.total_ms = local_spr_elapsed_ms(local_spr_total_start);
                std::cout << "Local SPR joint refinement done (radius=" << local_spr_radius
                          << ", rounds=" << local_spr_rounds_executed
                          << "/" << local_spr_rounds << ").\n";
                std::printf(
                    "Local SPR timing summary: total=%.3f ms prep=%.3f ms hot=%.3f ms "
                    "anchors=%.3f ms search=%.3f ms select=%.3f ms validate=%.3f ms rebuild=%.3f ms\n",
                    local_spr_timing.total_ms,
                    local_spr_timing.prep_ms,
                    local_spr_timing.hot_filter_ms,
                    local_spr_timing.anchor_unit_ms,
                    local_spr_timing.search_ms,
                    local_spr_timing.selection_ms,
                    local_spr_timing.validation_ms,
                    local_spr_timing.rebuild_ms);
                std::printf(
                    "Local SPR scoring timing: jobs=%d host_prep=%.3f ms "
                    "[filter=%.3f ms host_stage=%.3f ms fill_pmat=%.3f ms] "
                    "reload=%.3f ms query_clv=%.3f ms build_ops=%.3f ms upload_ops=%.3f ms "
                    "update=%.3f ms kernel=%.3f ms cleanup=%.3f ms\n",
                    local_spr_timing.scoring_instances,
                    local_spr_timing.score_host_prep_ms,
                    local_spr_timing.score_filter_align_ms,
                    local_spr_timing.score_pack_host_ms,
                    local_spr_timing.score_fill_pmat_ms,
                    local_spr_timing.score_upload_ms,
                    local_spr_timing.score_query_clv_ms,
                    local_spr_timing.score_build_ops_ms,
                    local_spr_timing.score_upload_ops_ms,
                    local_spr_timing.score_update_ms,
                    local_spr_timing.score_kernel_ms,
                    local_spr_timing.score_cleanup_ms);
                std::printf(
                    "Local SPR validation timing: evaluated=%d accepted=%d eval_calls=%d "
                    "tree_edit=%.3f ms eval_total=%.3f ms "
                    "[build=%.3f ms update=%.3f ms logL=%.3f ms cleanup=%.3f ms]\n",
                    local_spr_timing.validation_candidates,
                    local_spr_timing.accepted_candidates,
                    local_spr_timing.eval_calls,
                    local_spr_timing.validation_tree_edit_ms,
                    local_spr_timing.validation_eval_ms,
                    local_spr_timing.eval_build_ms,
                    local_spr_timing.eval_update_ms,
                    local_spr_timing.eval_logl_ms,
                    local_spr_timing.eval_cleanup_ms);
                batch_commit_timing.local_spr_ms +=
                    batch_commit_elapsed_ms(local_spr_batch_start);
            }
        }
        if (profile_batch_timing) {
            const CommitTimingStats& stats = batch_commit_timing.commit;
            std::printf(
                "Batch commit timing: batches=%d queries=%d free_prev=%.3f ms build=%.3f ms "
                "initial_update=%.3f ms evaluate=%.3f ms append=%.3f ms newick=%.3f ms local_spr=%.3f ms\n",
                batch_commit_timing.batches,
                batch_commit_timing.queries,
                batch_commit_timing.free_prev_ms,
                batch_commit_timing.build_ms,
                batch_commit_timing.initial_update_ms,
                batch_commit_timing.evaluate_ms,
                batch_commit_timing.append_ms,
                batch_commit_timing.newick_ms,
                batch_commit_timing.local_spr_ms);
            std::printf(
                "Batch query timing: evals=%d reset=%.3f ms build_query_clv=%.3f ms "
                "placement_kernel_total=%.3f ms\n",
                stats.query_evals,
                stats.query_reset_stage_ms,
                stats.query_build_clv_stage_ms,
                stats.query_kernel_total_ms);
            std::printf(
                "Batch insertion timing: initial_updates=%d insertion_updates=%d "
                "initial_up_host=%.3f ms initial_up_stage=%.3f ms initial_down_host=%.3f ms initial_down_stage=%.3f ms "
                "insert_pre_clv=%.3f ms insert_up_host=%.3f ms insert_up_stage=%.3f ms insert_down_host=%.3f ms insert_down_stage=%.3f ms\n",
                stats.initial_updates,
                stats.insertion_updates,
                stats.initial_upward_host_ms,
                stats.initial_upward_stage_ms,
                stats.initial_downward_host_ms,
                stats.initial_downward_stage_ms,
                stats.insertion_pre_clv_ms,
                stats.insertion_upward_host_ms,
                stats.insertion_upward_stage_ms,
                stats.insertion_downward_host_ms,
                stats.insertion_downward_stage_ms);
            std::printf(
                "Batch op summary: initial_up_ops=%lld initial_down_ops=%lld "
                "insert_up_ops=%lld insert_down_ops=%lld\n",
                stats.initial_upward_ops,
                stats.initial_downward_ops,
                stats.insertion_upward_ops,
                stats.insertion_downward_ops);
        }
    } else {
        res = BuildAllToGPU(
            msa_names,
            rows,
            newick,
            Q,
            pi,
            rate_multipliers,
            rate_weights,
            pattern_weights_arg,
            sites,
            states,
            rate_cats,
            per_rate_scaling,
            placement_queries,
            commit_to_tree);
        if (res.tree.nodes.empty() || res.dev.N == 0) {
            throw std::runtime_error("BuildAllToGPU returned empty tree/device structures.");
        }
        if (res.tree.root_id < 0) {
            throw std::runtime_error("BuildAllToGPU produced tree with invalid root_id.");
        }

        std::cout << "Uploaded. N=" << res.dev.N << ", tips=" << res.dev.tips
                    << ", per_node_elems=" << res.dev.per_node_elems() << "\n";

        if (env_flag_enabled("MLIPPER_DEBUG_TREE_STRUCTURE")) {
            print_tree_structure(res.tree);
        }

        cudaEventRecord(start);
        placement_ops.profile_commit_timing = []() {
            const char* env = std::getenv("MLIPPER_PROFILE_COMMIT_TIMING");
            return env && std::atoi(env) != 0;
        }();
        UpdateTreeClvs(
            res.dev,
            res.tree,
            res.hostPack,
            placement_ops,
            stream);
        double logL = root_likelihood::compute_root_loglikelihood_total(
            res.dev,
            res.tree.root_id,
            res.dev.d_pattern_weights_u,
            nullptr,
            0.0,
            0);
        printf("Initial tree log-likelihood = %.12f\n", logL);
        std::vector<std::string> placement_query_names;
        if (commit_to_tree) {
            placement_query_names.reserve(placement_queries.size());
            for (const NewPlacementQuery& query : placement_queries) {
                placement_query_names.push_back(query.msa_name);
            }
        }
        PlacementCommitContext commit_ctx;
        commit_ctx.placement_ops = &placement_ops;

        bool actual_commit_to_tree = commit_to_tree;
        if (commit_to_tree) {
            commit_ctx.tree = &res.tree;
            commit_ctx.host = &res.hostPack;
            commit_ctx.queries = &res.queries;
            commit_ctx.query_names = &placement_query_names;
            commit_ctx.inserted_query_names = &committed_query_names;
        }
        EvaluatePlacementQueries(
            res.dev,
            res.eig,
            rate_multipliers,
            commit_ctx,
            &placement_results,
            1,
            actual_commit_to_tree,
            stream);
    }
    if (commit_to_tree) {
        const double committed_logL = root_likelihood::compute_root_loglikelihood_total(
            res.dev,
            res.tree.root_id,
            res.dev.d_pattern_weights_u,
            nullptr,
            0.0,
            0);
        printf("Committed tree log-likelihood = %.12f\n", committed_logL);
        if (placement_ops.profile_commit_timing) {
            const CommitTimingStats& stats = placement_ops.timing;
            printf(
                "Commit timing summary: initial_updates=%d insertion_updates=%d "
                "initial_up_host=%.3f ms initial_up_stage=%.3f ms initial_down_host=%.3f ms initial_down_stage=%.3f ms "
                "query_reset=%.3f ms query_build_clv=%.3f ms query_kernel_total=%.3f ms "
                "insert_pre_clv=%.3f ms insert_up_host=%.3f ms insert_up_stage=%.3f ms insert_down_host=%.3f ms insert_down_stage=%.3f ms\n",
                stats.initial_updates,
                stats.insertion_updates,
                stats.initial_upward_host_ms,
                stats.initial_upward_stage_ms,
                stats.initial_downward_host_ms,
                stats.initial_downward_stage_ms,
                stats.query_reset_stage_ms,
                stats.query_build_clv_stage_ms,
                stats.query_kernel_total_ms,
                stats.insertion_pre_clv_ms,
                stats.insertion_upward_host_ms,
                stats.insertion_upward_stage_ms,
                stats.insertion_downward_host_ms,
                stats.insertion_downward_stage_ms);
            printf(
                "Commit op summary: query_evals=%d initial_up_ops=%lld initial_down_ops=%lld "
                "insert_up_ops=%lld insert_down_ops=%lld\n",
                stats.query_evals,
                stats.initial_upward_ops,
                stats.initial_downward_ops,
                stats.insertion_upward_ops,
                stats.insertion_downward_ops);
        }

    }

    const double final_logL = root_likelihood::compute_root_loglikelihood_total(
        res.dev,
        res.tree.root_id,
        res.dev.d_pattern_weights_u,
        nullptr,
        0.0,
        0);
    printf("Final tree log-likelihood = %.12f\n", final_logL);
    if (env_flag_enabled("MLIPPER_DEBUG_TREE_STRUCTURE")) {
        print_tree_structure(res.tree);
    }

    free_placement_op_buffer(placement_ops, stream);
    cudaStreamSynchronize(stream);

    if (!commit_tree_out.empty()) {
        const size_t collapsed_internal_branches = write_tree_to_newick_file(
            res.tree,
            commit_tree_out,
            commit_collapse_internal_epsilon);
        std::cout << "Wrote Newick tree to " << commit_tree_out;
        if (commit_collapse_internal_epsilon >= 0.0) {
            std::cout << " after collapsing " << collapsed_internal_branches
                      << " internal branches <= " << commit_collapse_internal_epsilon;
        }
        std::cout << "\n";
    }

    if (!jplace_out.empty()) {
        if (placement_results.size() != placement_queries.size()) {
            std::cerr << "placement result count mismatch before jplace export: got "
                      << placement_results.size() << ", expected " << placement_queries.size()
                      << ". Exporting available prefix only.\n";
        }

        const JplaceTreeExport jplace_tree = build_jplace_tree_export(res.tree);
        std::vector<JplacePlacementRecord> jplace_records;
        const size_t export_count = std::min(placement_results.size(), placement_queries.size());
        jplace_records.reserve(export_count);
        for (size_t i = 0; i < export_count; ++i) {
            const PlacementResult& pres = placement_results[i];
            JplacePlacementRecord rec;
            rec.query_name = placement_queries[i].msa_name.empty()
                ? ("query_" + std::to_string(i))
                : placement_queries[i].msa_name;

            auto append_row = [&](int target_id,
                                  double loglikelihood,
                                  double like_weight_ratio,
                                  double proximal_length,
                                  double pendant_length) {
                if (target_id < 0 || target_id >= static_cast<int>(res.tree.nodes.size())) {
                    return;
                }
                const TreeNode& target = res.tree.nodes[target_id];
                if (target.parent < 0) {
                    return;
                }
                const int edge_num = jplace_tree.edge_num_by_node[target_id];
                if (edge_num < 0) {
                    return;
                }

                JplacePlacementRow row;
                row.edge_num = edge_num;
                row.likelihood = loglikelihood;
                row.like_weight_ratio = like_weight_ratio;
                // PlacementResult stores the jplace distal coordinate here.
                row.distal_length = proximal_length;
                row.pendant_length = pendant_length;
                rec.rows.push_back(std::move(row));
            };

            if (!pres.top_placements.empty()) {
                for (const PlacementResult::RankedPlacement& candidate : pres.top_placements) {
                    append_row(
                        candidate.target_id,
                        candidate.loglikelihood,
                        candidate.like_weight_ratio,
                        candidate.proximal_length,
                        candidate.pendant_length);
                }
            } else {
                append_row(
                    pres.target_id,
                    pres.loglikelihood,
                    1.0,
                    pres.proximal_length,
                    pres.pendant_length);
            }

            if (rec.rows.empty()) {
                throw std::runtime_error("Could not export any placement rows for jplace.");
            }
            jplace_records.push_back(std::move(rec));
        }

        std::ostringstream invocation;
        for (int i = 0; i < argc; ++i) {
            if (i) invocation << ' ';
            invocation << argv[i];
        }
        write_jplace(jplace_out, jplace_tree.tree, jplace_records, invocation.str());
        std::cout << "Wrote jplace to " << jplace_out << "\n";
    }

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time (milliseconds)
    cudaEventElapsedTime(&gpu_ms_kernel, start, stop);
    const auto end_gpu = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    printf("GPU kernel time = %.3f ms\n", gpu_ms_kernel);
    printf("GPU Wall Clock time = %.3f ms\n", gpu_ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    free_device_tree(res.dev);

    return 0;
    } catch (const std::exception& e) {
        free_device_tree(res.dev);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        std::cout.flush();
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
