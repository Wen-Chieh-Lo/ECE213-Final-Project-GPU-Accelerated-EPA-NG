#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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

std::string resolve_path(const std::filesystem::path& base, const std::string& p) {
    std::filesystem::path candidate(p);
    if (candidate.empty()) return {};
    if (candidate.is_absolute()) return candidate.string();
    return (base / candidate).string();
}

std::vector<std::string> parse_name_list(const std::string& text) {
    std::vector<std::string> out;
    std::string token;
    auto flush = [&]() {
        if (!token.empty()) {
            out.push_back(token);
            token.clear();
        }
    };
    for (char ch : text) {
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch))) {
            flush();
        } else {
            token.push_back(ch);
        }
    }
    flush();
    return out;
}

std::vector<std::string> read_name_list_file(const std::string& path) {
    if (path.empty()) return {};
    return parse_name_list(read_file_to_string(path));
}

bool env_flag_enabled_local(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] && std::string(value) != "0";
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
    const auto start_gpu = std::chrono::steady_clock::now();
    cudaEvent_t start, stop;
    cudaStream_t stream;
    float gpu_ms_kernel = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    CLI::App app{"MLIPPER"};
    app.get_formatter()->column_width(40);

    // Single config object filled directly by CLI11 options.
    parse::RunConfig config;

    // ---- Input (files/tree) ----
    std::string tree_newick;
    std::string jplace_out;
    std::string commit_tree_out;
    std::string prune_tree_in;
    std::string prune_tips_alignment;
    std::string prune_tree_out;
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
    app.add_option("--jplace-out", jplace_out, "Optional output path for a top-k placement jplace file")
        ->group("Output");
    app.add_option("--commit-to-tree", commit_tree_out,
                   "Commit query placements to reference tree and write the final tree in Newick (.nwk) format to this path")
        ->group("Output");
    app.add_option(
           "--commit-collapse-internal-epsilon",
           commit_collapse_internal_epsilon,
           "When writing --commit-to-tree output, collapse internal branches with length <= epsilon into polytomies (-1 disables; default 1e-6)")
        ->group("Output");
    app.add_option("--prune-tree-in", prune_tree_in,
                   "Input Newick tree to prune tip taxa from")
        ->group("Utility");
    app.add_option("--prune-tips-alignment", prune_tips_alignment,
                   "Alignment/Fasta whose sequence names should be removed from --prune-tree-in")
        ->group("Utility");
    app.add_option("--prune-tree-out", prune_tree_out,
                   "Output path for the pruned Newick tree")
        ->group("Utility");
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
    int reinsert_pass_count = 0;
    int reinsert_last_k = 0;
    int reinsert_chunk_size = 0;
    std::string reinsert_names_file;
    bool epa_ng_like = false;

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

    app.add_option("--full-opt-passes", full_opt_passes,
                   "Full optimization passes when refine is disabled")
        ->group("Placement");
    app.add_option("--refine-global-passes", refine_global_passes,
                   "Full-op passes before refine selection")
        ->group("Placement");
    app.add_option("--refine-extra-passes", refine_extra_passes,
                   "Maximum extra passes after refine top-K selection")
        ->group("Placement");
    app.add_option("--refine-detect-topk", refine_detect_topk,
                   "Top-K candidates examined when deciding whether to refine")
        ->group("Placement");
    app.add_option("--refine-topk", refine_topk,
                   "Top-K candidates retained for refine passes")
        ->group("Placement");
    app.add_option("--refine-gap-top2", refine_gap_top2,
                   "Ambiguity threshold for top1-top2 log-likelihood gap")
        ->group("Placement");
    app.add_option("--refine-gap-top5", refine_gap_top5,
                   "Ambiguity threshold for top1-top5 log-likelihood gap")
        ->group("Placement");
    app.add_option("--refine-converged-loglk-eps", refine_converged_loglk_eps,
                   "Per-op refine early-stop threshold for log-likelihood change")
        ->group("Placement");
    app.add_option("--refine-converged-length-eps", refine_converged_length_eps,
                   "Per-op refine early-stop threshold for branch-length change")
        ->group("Placement");
    app.add_option("--reinsert-pass-count", reinsert_pass_count,
                   "Number of post-commit prune+reinsert passes")
        ->group("Placement");
    app.add_option("--reinsert-last-k", reinsert_last_k,
                   "Reinsert the K most suspicious committed queries in each post-commit pass")
        ->group("Placement");
    app.add_option("--reinsert-chunk-size", reinsert_chunk_size,
                   "Process suspicious reinsertion in small iterative chunks (0 = all selected queries at once)")
        ->group("Placement");
    app.add_option("--reinsert-names-file", reinsert_names_file,
                   "Explicit list of committed query names to prune+reinsert (overrides gap-based selection)")
        ->group("Placement");
    app.add_flag("--epa-ng-like", epa_ng_like,
                 "Disable commit/reinsert and extra refine to mimic EPA-ng placement-only behavior")
        ->group("Placement");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (!std::isfinite(commit_collapse_internal_epsilon)) {
        return app.exit(CLI::ValidationError(
            "--commit-collapse-internal-epsilon",
            "must be finite (use a negative value such as -1 to disable output collapsing)"));
    }
    if (epa_ng_like) {
        // Keep placement-only behavior, but match EPA-ng's longer branch-length
        // smoothing schedule instead of collapsing to a single pass.
        full_opt_passes = 32;
        refine_global_passes = 0;
        refine_extra_passes = 0;
        refine_detect_topk = 0;
        refine_topk = 0;
        refine_gap_top2 = 0.0;
        refine_gap_top5 = 0.0;
        refine_converged_loglk_eps = 1e-5;
        refine_converged_length_eps = 1e-5;
        reinsert_pass_count = 0;
        reinsert_last_k = 0;
        reinsert_chunk_size = 0;
        setenv("MLIPPER_ENABLE_CONVERGENCE_CHECK", "1", 1);
        if (!commit_tree_out.empty()) {
            std::cerr << "EPA-NG-like mode disables --commit-to-tree output; ignoring "
                      << commit_tree_out << "\n";
        }
        commit_tree_out.clear();
    }
    commit_to_tree = !commit_tree_out.empty();

    const std::filesystem::path config_base = std::filesystem::current_path();

    if (!prune_tree_out.empty()) {
        if (prune_tree_in.empty()) throw CLI::RequiredError("--prune-tree-in");
        if (prune_tips_alignment.empty()) throw CLI::RequiredError("--prune-tips-alignment");

        const std::string tree_text = read_file_to_string(resolve_path(config_base, prune_tree_in));
        const parse::Alignment prune_alignment =
            parse::read_alignment_file(resolve_path(config_base, prune_tips_alignment));
        std::unordered_set<std::string> names_to_remove;
        names_to_remove.reserve(prune_alignment.names.size() * 2);
        for (const std::string& name : prune_alignment.names) {
            names_to_remove.insert(name);
        }

        const std::string pruned_tree = parse::prune_newick_tips(tree_text, names_to_remove);
        std::ofstream ofs(resolve_path(config_base, prune_tree_out));
        if (!ofs) {
            throw std::runtime_error("Cannot open prune output path: " + resolve_path(config_base, prune_tree_out));
        }
        ofs << pruned_tree << '\n';
        std::cout << "Pruned " << names_to_remove.size() << " tip names and wrote tree to "
                  << resolve_path(config_base, prune_tree_out) << "\n";
        return 0;
    }

    parse::RunInputs inputs;
    try {
        if (no_per_rate_scaling) config.model.per_rate_scaling = false;
        if (config.files.tree_alignment.empty()) throw CLI::RequiredError("--tree-alignment");
        if (tree_newick.empty() && config.files.tree.empty()) throw CLI::RequiredError("one of [--tree, --tree-newick]");

        if (config.model.states <= 0) throw CLI::ValidationError("--states", "must be positive");
        if (config.model.ncat <= 0) throw CLI::ValidationError("--ncat", "must be positive");

        if (!config.model.freqs.empty() && (int)config.model.freqs.size() != config.model.states) {
            throw CLI::ValidationError("--freqs",
                                       "must have exactly " + std::to_string(config.model.states) + " values (states)");
        }
        if (!config.model.rates.empty() && config.model.rates.size() != 6) {
            throw CLI::ValidationError("--rates", "must have exactly 6 values for 4-state GTR");
        }
        if (!config.model.rate_weights.empty() && (int)config.model.rate_weights.size() != config.model.ncat) {
            throw CLI::ValidationError("--rate-weights",
                                       "must have exactly " + std::to_string(config.model.ncat) + " values (ncat)");
        }

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

        inputs = parse::RunInputs{std::move(config), std::move(tree_alignment), std::move(tree_text)};

        if (inputs.tree_alignment.names.empty())
            throw CLI::ValidationError("--tree-alignment", "contains no sequences");
        if (inputs.tree_alignment.sites == 0)
            throw CLI::ValidationError("--tree-alignment", "contains zero sites");

        if (query_alignment.names.empty())
            throw CLI::ValidationError("--query-alignment", "contains no sequences");
        if (query_alignment.sites == 0)
            throw CLI::ValidationError("--query-alignment", "contains zero sites");
        if (query_alignment.sites != inputs.tree_alignment.sites) {
            throw CLI::ValidationError("--query-alignment",
                                       "sites mismatch with --tree-alignment (" +
                                           std::to_string(query_alignment.sites) + " vs " +
                                           std::to_string(inputs.tree_alignment.sites) + ")");
        }
    } catch (const CLI::Error& e) {
        return app.exit(e);
    }

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

    const std::string& query_alignment_cfg = inputs.config.files.query_alignment;
    std::vector<NewPlacementQuery> placement_queries =
        build_placement_query(resolve_path(config_base, query_alignment_cfg));
    if (env_flag_enabled("MLIPPER_ENABLE_REPETITIVE_COLUMNS")) {
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

    printf("Precision mode: %s\n", FP_MODE_NAME);
    // const auto start_gpu = std::chrono::steady_clock::now();
    auto res = BuildAllToGPU(
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

    print_tree_structure(res.tree);

    cudaEventRecord(start);
    PlacementOpBuffer placement_ops;
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
    std::vector<PlacementResult> placement_results;
    std::vector<std::string> placement_query_names;
    if (commit_to_tree) {
        placement_query_names.reserve(placement_queries.size());
        for (const NewPlacementQuery& query : placement_queries) {
            placement_query_names.push_back(query.msa_name);
        }
    }
    PlacementCommitContext commit_ctx;
    commit_ctx.placement_ops = &placement_ops;
    
    // If commit_to_tree is enabled, populate the remaining context fields.
    bool actual_commit_to_tree = commit_to_tree;
    std::vector<std::string> committed_query_names(placement_queries.size());
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
            printf(
                "Commit op summary: initial_up_ops=%lld initial_down_ops=%lld "
                "insert_up_ops=%lld insert_down_ops=%lld\n",
                stats.initial_upward_ops,
                stats.initial_downward_ops,
                stats.insertion_upward_ops,
                stats.insertion_downward_ops);
        }
    }

    if (commit_to_tree && reinsert_pass_count > 0) {
        if (!jplace_out.empty()) {
            throw std::runtime_error("Reinsert pass is not supported together with --jplace-out.");
        }
        const bool use_explicit_reinsert = !reinsert_names_file.empty();
        if (!use_explicit_reinsert && reinsert_last_k <= 0) {
            throw std::runtime_error("Reinsert pass requires --reinsert-last-k > 0.");
        }

        const int total_queries = static_cast<int>(placement_queries.size());
        std::vector<int> explicit_reinsert_indices;
        if (use_explicit_reinsert) {
            const std::vector<std::string> wanted_names = read_name_list_file(reinsert_names_file);
            if (wanted_names.empty()) {
                throw std::runtime_error("Reinsert names file is empty.");
            }
            std::unordered_map<std::string, int> name_to_index;
            name_to_index.reserve(committed_query_names.size());
            for (int idx = 0; idx < total_queries; ++idx) {
                const std::string& cname = committed_query_names[(size_t)idx];
                if (!cname.empty()) {
                    name_to_index[cname] = idx;
                }
            }
            for (const std::string& name : wanted_names) {
                auto it = name_to_index.find(name);
                if (it == name_to_index.end()) {
                    throw std::runtime_error("Reinsert names file contains unknown committed query: " + name);
                }
                explicit_reinsert_indices.push_back(it->second);
            }
            std::sort(explicit_reinsert_indices.begin(), explicit_reinsert_indices.end());
            explicit_reinsert_indices.erase(
                std::unique(explicit_reinsert_indices.begin(), explicit_reinsert_indices.end()),
                explicit_reinsert_indices.end());
        }
        const int actual_reinsert_k = use_explicit_reinsert
            ? static_cast<int>(explicit_reinsert_indices.size())
            : std::min(reinsert_last_k, total_queries);
        const bool profile_commit_timing = placement_ops.profile_commit_timing;
        for (int pass = 0; pass < reinsert_pass_count; ++pass) {
            std::unordered_set<int> reinserted_this_pass;
            const int chunk_limit_default =
                (reinsert_chunk_size > 0) ? reinsert_chunk_size : actual_reinsert_k;

            while (static_cast<int>(reinserted_this_pass.size()) < actual_reinsert_k) {
                const int remaining_budget = actual_reinsert_k - static_cast<int>(reinserted_this_pass.size());
                const int chunk_limit = std::min(chunk_limit_default, remaining_budget);

                std::vector<int> candidates;
                if (use_explicit_reinsert) {
                    candidates.reserve(explicit_reinsert_indices.size());
                    for (int idx : explicit_reinsert_indices) {
                        if (reinserted_this_pass.count(idx)) continue;
                        if (idx < 0 || idx >= static_cast<int>(committed_query_names.size())) continue;
                        if (committed_query_names[(size_t)idx].empty()) continue;
                        candidates.push_back(idx);
                    }
                } else {
                    candidates.reserve((size_t)total_queries);
                    for (int idx = 0; idx < total_queries; ++idx) {
                        if (reinserted_this_pass.count(idx)) continue;
                        if (idx < 0 || idx >= static_cast<int>(committed_query_names.size())) continue;
                        if (committed_query_names[(size_t)idx].empty()) continue;
                        candidates.push_back(idx);
                    }
                }
                if (candidates.empty()) break;

                const int actual_chunk = std::min(chunk_limit, static_cast<int>(candidates.size()));
                if (!use_explicit_reinsert) {
                    std::partial_sort(
                        candidates.begin(),
                        candidates.begin() + actual_chunk,
                        candidates.end(),
                        [&](int lhs, int rhs) {
                            const double lhs_gap = (lhs >= 0 && lhs < static_cast<int>(placement_results.size()))
                                ? placement_results[(size_t)lhs].gap_top2
                                : std::numeric_limits<double>::infinity();
                            const double rhs_gap = (rhs >= 0 && rhs < static_cast<int>(placement_results.size()))
                                ? placement_results[(size_t)rhs].gap_top2
                                : std::numeric_limits<double>::infinity();
                            if (lhs_gap != rhs_gap) return lhs_gap < rhs_gap;
                            const double lhs_gap5 = (lhs >= 0 && lhs < static_cast<int>(placement_results.size()))
                                ? placement_results[(size_t)lhs].gap_top5
                                : std::numeric_limits<double>::infinity();
                            const double rhs_gap5 = (rhs >= 0 && rhs < static_cast<int>(placement_results.size()))
                                ? placement_results[(size_t)rhs].gap_top5
                                : std::numeric_limits<double>::infinity();
                            if (lhs_gap5 != rhs_gap5) return lhs_gap5 < rhs_gap5;
                            return lhs < rhs;
                        });
                }

                std::vector<int> reinsert_indices;
                reinsert_indices.reserve((size_t)actual_chunk);
                std::unordered_set<int> reinsert_index_set;
                for (int rank = 0; rank < actual_chunk; ++rank) {
                    const int idx = candidates[(size_t)rank];
                    reinsert_index_set.insert(idx);
                    reinsert_indices.push_back(idx);
                }
                std::sort(reinsert_indices.begin(), reinsert_indices.end());

                std::unordered_set<std::string> names_to_remove;
                names_to_remove.reserve(reinsert_indices.size() * 2);
                for (int idx : reinsert_indices) {
                    if (!committed_query_names[(size_t)idx].empty()) {
                        names_to_remove.insert(committed_query_names[(size_t)idx]);
                    }
                }
                if (names_to_remove.empty()) {
                    throw std::runtime_error("Reinsert pass could not resolve committed query names to remove.");
                }

                const std::string current_tree_newick = write_tree_to_newick_string(res.tree);
                const std::string pruned_tree_newick = parse::prune_newick_tips(current_tree_newick, names_to_remove);

                std::vector<std::string> retained_names = msa_names;
                std::vector<std::string> retained_rows = rows;
                retained_names.reserve(msa_names.size() + placement_queries.size() - reinsert_indices.size());
                retained_rows.reserve(rows.size() + placement_queries.size() - reinsert_indices.size());
                for (int idx = 0; idx < total_queries; ++idx) {
                    if (reinsert_index_set.count(idx)) continue;
                    retained_names.push_back(committed_query_names[(size_t)idx]);
                    retained_rows.push_back(placement_queries[(size_t)idx].msa);
                }

                std::vector<NewPlacementQuery> reinsert_queries;
                std::vector<std::string> reinsert_query_names;
                reinsert_queries.reserve(reinsert_indices.size());
                reinsert_query_names.reserve(reinsert_indices.size());
                for (int idx : reinsert_indices) {
                    NewPlacementQuery q = placement_queries[(size_t)idx];
                    q.msa_name = committed_query_names[(size_t)idx];
                    reinsert_query_names.push_back(q.msa_name);
                    reinsert_queries.push_back(std::move(q));
                }

                free_placement_op_buffer(placement_ops, stream);
                cudaStreamSynchronize(stream);
                free_device_tree(res.dev);

                auto next_res = BuildAllToGPU(
                    retained_names,
                    retained_rows,
                    pruned_tree_newick,
                    Q,
                    pi,
                    rate_multipliers,
                    rate_weights,
                    pattern_weights_arg,
                    sites,
                    states,
                    rate_cats,
                    per_rate_scaling,
                    reinsert_queries,
                    true);

                PlacementOpBuffer next_ops;
                next_ops.profile_commit_timing = profile_commit_timing;
                UpdateTreeClvs(
                    next_res.dev,
                    next_res.tree,
                    next_res.hostPack,
                    next_ops,
                    stream);

                std::vector<PlacementResult> next_results;
                std::vector<std::string> next_inserted_query_names(reinsert_queries.size());
                PlacementCommitContext next_commit_ctx;
                next_commit_ctx.tree = &next_res.tree;
                next_commit_ctx.host = &next_res.hostPack;
                next_commit_ctx.queries = &next_res.queries;
                next_commit_ctx.placement_ops = &next_ops;
                next_commit_ctx.query_names = &reinsert_query_names;
                next_commit_ctx.inserted_query_names = &next_inserted_query_names;

                EvaluatePlacementQueries(
                    next_res.dev,
                    next_res.eig,
                    rate_multipliers,
                    next_commit_ctx,
                    &next_results,
                    1,
                    true,
                    stream);

                for (size_t i = 0; i < reinsert_indices.size(); ++i) {
                    const size_t query_slot = static_cast<size_t>(reinsert_indices[i]);
                    committed_query_names[query_slot] = next_inserted_query_names[i];
                    if (query_slot < placement_results.size() && i < next_results.size()) {
                        placement_results[query_slot] = next_results[i];
                    }
                    reinserted_this_pass.insert(reinsert_indices[i]);
                }

                res = std::move(next_res);
                placement_ops = std::move(next_ops);
            }

            const double reinsert_logL = root_likelihood::compute_root_loglikelihood_total(
                res.dev,
                res.tree.root_id,
                res.dev.d_pattern_weights_u,
                nullptr,
                0.0,
                0);
            printf("Reinsert pass %d committed tree log-likelihood = %.12f\n", pass + 1, reinsert_logL);
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
    print_tree_structure(res.tree);

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
        const char* debug_edge_env = std::getenv("MLIPPER_DEBUG_EDGE_NUM");
        if (debug_edge_env && debug_edge_env[0]) {
            const int wanted_edge = std::atoi(debug_edge_env);
            bool found_edge = false;
            for (size_t node_idx = 0; node_idx < jplace_tree.edge_num_by_node.size(); ++node_idx) {
                if (jplace_tree.edge_num_by_node[node_idx] != wanted_edge) {
                    continue;
                }
                const TreeNode& node = res.tree.nodes[node_idx];
                std::fprintf(
                    stderr,
                    "[MLIPPER-EDGE] edge_num=%d target_id=%zu parent=%d left=%d right=%d is_tip=%d name=%s\n",
                    wanted_edge,
                    node_idx,
                    node.parent,
                    node.left,
                    node.right,
                    node.is_tip ? 1 : 0,
                    node.name.empty() ? "<inner>" : node.name.c_str());
                found_edge = true;
            }
            if (!found_edge) {
                std::fprintf(
                    stderr,
                    "[MLIPPER-EDGE] edge_num=%d not_found\n",
                    wanted_edge);
            }
        }
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
}
