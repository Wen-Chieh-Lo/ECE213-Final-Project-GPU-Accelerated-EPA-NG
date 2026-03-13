#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>

#include "../mlipper_util.h"
#include "tree.hpp"
#include "pmat.h"
#include "core_likelihood.cuh"
#include "partial_likelihood.cuh"
#include "parse_file.hpp"

static void throw_if(bool cond, const char* msg) {
    if (cond) throw std::runtime_error(msg);
}

std::vector<double> build_mixture_weights(const parse::ModelConfig& model, int rate_cats) {
    std::vector<double> weights;
    if (rate_cats <= 0) return weights;
    if (static_cast<int>(model.rate_weights.size()) == rate_cats) {
        weights = model.rate_weights;
    } else {
        weights.assign(rate_cats, 1.0 / rate_cats);
    }

    double sum = 0.0;
    for (double value : weights) sum += value;
    if (sum <= 0.0) {
        weights.assign(rate_cats, 1.0 / rate_cats);
        return weights;
    }
    for (double& value : weights) value /= sum;
    return weights;
}

std::vector<double> build_gamma_rate_categories(double alpha, int rate_cats) {
    std::vector<double> rates(rate_cats, 1.0);
    if (rate_cats <= 1 || alpha <= 0.0) return rates;

    std::vector<double> gamma_tmp(rate_cats);
    const int status = pll_compute_gamma_cats(
        alpha,
        rate_cats,
        gamma_tmp.data(),
        PLL_GAMMA_RATES_MEAN);
    if (status != PLL_SUCCESS) {
        throw std::runtime_error("pll_compute_gamma_cats failed.");
    }

    for (int rate_idx = 0; rate_idx < rate_cats; ++rate_idx) {
        rates[rate_idx] = gamma_tmp[rate_idx];
    }
    return rates;
}

std::vector<double> build_gtr_q_matrix(
    int states,
    const parse::ModelConfig& model,
    const std::vector<double>& pi)
{
    std::vector<double> q_matrix(states * states, 0.0);
    if (states != 4 || model.rates.size() < 6 || pi.size() != 4) {
        return q_matrix;
    }

    auto set_pair = [&](int row, int col, double rate) {
        q_matrix[row * states + col] = rate * pi[col];
        q_matrix[col * states + row] = rate * pi[row];
    };

    set_pair(0, 1, model.rates[0]);
    set_pair(0, 2, model.rates[1]);
    set_pair(0, 3, model.rates[2]);
    set_pair(1, 2, model.rates[3]);
    set_pair(1, 3, model.rates[4]);
    set_pair(2, 3, model.rates[5]);

    for (int row = 0; row < states; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < states; ++col) {
            if (row != col) row_sum += q_matrix[row * states + col];
        }
        q_matrix[row * states + row] = -row_sum;
    }

    double mu = 0.0;
    for (int row = 0; row < states; ++row) {
        mu -= pi[row] * q_matrix[row * states + row];
    }

    for (double& entry : q_matrix) {
        entry /= mu;
    }

    return q_matrix;
}

static PlacementQueryBatch make_query_batch(
    const std::vector<NewPlacementQuery>& placement_queries,
    size_t sites,
    int states,
    int rate_cats,
    const EigResult& eig,
    const std::vector<double>& rate_multipliers)
{
    PlacementQueryBatch batch;
    batch.count = placement_queries.size();
    if (batch.empty()) return batch;

    if ((int)rate_multipliers.size() != rate_cats) {
        throw std::runtime_error("rate_multipliers size mismatch for queries.");
    }
    const size_t qcount = batch.count;
    batch.branch_lengths.assign(qcount, fp_t(0.5));
    batch.query_chars.resize(qcount * sites, 4);
    for (size_t qi = 0; qi < qcount; ++qi) {
        const auto& q = placement_queries[qi];
        if (q.pendant > fp_t(0)) batch.branch_lengths[qi] = q.pendant;
        if (q.msa.size() != sites) {
            throw std::runtime_error("Query sequence length mismatch.");
        }
        for (size_t s = 0; s < sites; ++s) {
            batch.query_chars[qi * sites + s] = encode_state_DNA5(q.msa[s]);
        }
    }
    return batch;
}



// Parse Newick text and build a rooted TreeBuildResult with topology and offsets.
TreeBuildResult build_tree_from_newick_with_pll(
    const std::vector<std::string>& msa_tip_names,
    const std::string& newick_text,
    size_t sites,
    int states,
    int rate_cats,
    bool per_rate_scaling)
{
    TreeBuildResult out;
    // 1) Parse Newick with libpll (rooted tree)
    const char* filename = "./tree.nwk";
    {
        std::ofstream ofs(filename);
        ofs << newick_text;
    }

    pll_rtree_t* rtree = pll_rtree_parse_newick(filename);

    throw_if(!rtree, "pll_rtree_parse_newick failed (check Newick syntax).");

    // 2) Collect all nodes (libpll provides nodes array and tip/inner counts)
    const unsigned int num_tips   = rtree->tip_count;
    const unsigned int num_inners = rtree->inner_count;
    const unsigned int num_nodes  = num_tips + num_inners; // rooted: includes the root itself

    out.nodes.resize(num_nodes);

    // 3) Name alignment: MSA tip name -> index
    std::unordered_map<std::string,int> msa_idx;
    msa_idx.reserve(msa_tip_names.size()*2);
    for (int i = 0; i < (int)msa_tip_names.size(); ++i) {
        msa_idx[msa_tip_names[i]] = i;
    }

    // 4) Build node_id mapping: libpll rooted trees usually have tips first, inners later; take order from traversal
    //    Do one postorder traversal to decide child/parent relations and ids.
    std::vector<pll_rnode_t*> postorder_nodes(num_nodes, nullptr);
    unsigned int count = 0;

    // Define callback: accepts a single node
    auto cb = [](pll_rnode_t *node) -> int {
        return PLL_SUCCESS;
    };

    // Let libpll fill nodes into the outbuffer in order
    int rc = pll_rtree_traverse(rtree->root,
                                PLL_TREE_TRAVERSE_POSTORDER,
                                cb,                              // callback
                                postorder_nodes.data(),           // outbuffer
                                &count);
    throw_if(rc != PLL_SUCCESS, "pll_rtree_traverse (POSTORDER) failed.");
    throw_if(count != (int)num_nodes, "postorder node count mismatch.");

    // Build a rnode* → id lookup
    std::unordered_map<pll_rnode_t*, int> id_of;
    id_of.reserve(num_nodes*2);

    // Assign ids by postorder: 0..N-1
    for (int i = 0; i < num_nodes; ++i) {
        id_of[ postorder_nodes[i] ] = i;
        out.nodes[i].id = i;
    }

    // 5) Fill left/right/parent/is_tip/name/branch_length_to_parent
    //    Note: root node->length is the length to its parent; root has no parent → set to 0
    for (int i = 0; i < num_nodes; ++i) {
        pll_rnode_t* nd = postorder_nodes[i];
        TreeNode &dst = out.nodes[i];

        // Tip check: libpll tips have no children; internals have left/right
        const bool is_tip = (nd->left == nullptr && nd->right == nullptr);
        dst.is_tip = is_tip;

        // name (tips only)
        if (is_tip && nd->label) dst.name = std::string(nd->label);
        else dst.name.clear();

        // parent and branch length
        if (nd->parent) {
            dst.parent = id_of[ nd->parent ];
            dst.branch_length_to_parent = nd->length; // Number after the colon in Newick
        } else {
            dst.parent = -1; // root
            dst.branch_length_to_parent = fp_t(0);
            out.root_id = i;
        }

        // children (internal nodes only)
        if (!is_tip) {
            throw_if(!nd->left || !nd->right, "Internal node missing children (non-binary?).");
            dst.left  = id_of[ nd->left ];
            dst.right = id_of[ nd->right ];
        }
    }

    // 6) Align MSA names: build tip name -> node id; also ensure every tip exists in the MSA
    out.tip_node_by_name.reserve(num_tips*2);
    unsigned int found_tips = 0;
    for (const TreeNode& tn : out.nodes) {
        if (!tn.is_tip) continue;
        if (tn.name.empty()) {
            throw std::runtime_error("Encountered a tip with empty name.");
        }
        out.tip_node_by_name[tn.name] = tn.id;
        if (msa_idx.find(tn.name) != msa_idx.end()) ++found_tips;
    }
    throw_if(found_tips != num_tips, "Some tips in Newick not found in MSA names.");

    // 7) Produce postorder over the whole tree (as ids): children before parent
    out.postorder.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) out.postorder[i] = out.nodes[i].id; // ids already follow postorder
    out.preorder.resize(num_nodes);
    // Manual preorder (parent -> left -> right) using the root pointer and id_of map.
    {
        std::vector<pll_rnode_t*> stack;
        stack.reserve(num_nodes);
        stack.push_back(rtree->root);
        int idx = 0;
        while (!stack.empty()) {
            pll_rnode_t* nd = stack.back();
            stack.pop_back();
            out.preorder[idx++] = id_of[nd];
            // push right then left so left is processed first
            if (nd->right) stack.push_back(nd->right);
            if (nd->left)  stack.push_back(nd->left);
        }
        throw_if(idx != (int)num_nodes, "preorder node count mismatch.");
    }

    // 8) Set each node's scaler offsets (in elements, not bytes)
    const size_t scaler_count_per_node = per_rate_scaling
        ? (sites * (size_t)rate_cats)
        : (sites);

    for (auto &tn : out.nodes) {
        tn.scaler_offset = (size_t)tn.id * scaler_count_per_node;  // d_scaler_pool + tn.scaler_offset
    }

    // 9) Cleanup libpll structures
    pll_rtree_destroy(rtree, nullptr);

    return out;
}

// End-to-end pipeline: build tree, pack host arrays, compute matrices, upload to GPU.
BuildToGpuResult BuildAllToGPU(
    const std::vector<std::string>& msa_tip_names,
    const std::vector<std::string>& msa_rows,
    const std::string& newick_text,
    const std::vector<double>& Q_rowmajor,   // size = states*states
    const std::vector<double>& pi,           // size = states
    const std::vector<double>& rate_multipliers,   // size = rate_cats
    const std::vector<double>& rate_weights, // size = rate_cats
    const std::vector<unsigned>& pattern_weights,
    size_t sites, int states, int rate_cats, bool per_rate_scaling,
    const std::vector<NewPlacementQuery>& placement_queries)
{
    throw_if(states > 64, "states exceeds MAX_STATES (64).");
    throw_if(rate_cats > 8, "rate_cats exceeds MAX_RATECATS (8).");
    if (Q_rowmajor.size() != (size_t)states*(size_t)states)
        throw std::runtime_error("Q size mismatch.");
    if (pi.size() != (size_t)states)
        throw std::runtime_error("pi size mismatch.");

    // 1) Newick → TreeBuildResult (postorder, topology, offsets)
    TreeBuildResult T = build_tree_from_newick_with_pll(
        msa_tip_names, newick_text, sites, states, rate_cats, per_rate_scaling);

    // 2) Align MSA → HostPacking (topology arrays, tipchars, offsets, scaler)
    HostPacking H = pack_host_arrays_from_tree_and_msa(
        T, msa_tip_names, msa_rows, sites, states);
    if (!pattern_weights.empty() && pattern_weights.size() != sites) {
        throw std::runtime_error("pattern_weights size mismatch.");
    }
    H.pattern_weights = pattern_weights;

    // 3) Host-side GTR decomposition (V/Vinv/λ)
    EigResult Eigen = gtr_eigendecomp_cpu(Q_rowmajor.data(), pi.data(), states);

    fill_pmats_in_host_packing(T, H, Eigen, rate_multipliers, states, rate_cats);

    // Placement queries staged separately from HostPacking.
    PlacementQueryBatch Q = make_query_batch(
        placement_queries,
        sites,
        states,
        rate_cats,
        Eigen,
        rate_multipliers);

    // 4) Upload everything to GPU (including model constants)
    DeviceTree D = upload_to_gpu(
        T,
        H,
        Eigen,
        rate_weights,
        rate_multipliers,
        pi,
        sites,
        states,
        rate_cats,
        per_rate_scaling,
        Q.empty() ? nullptr : &Q);

    return BuildToGpuResult{ std::move(D), std::move(T), std::move(H), std::move(Eigen), std::move(Q) };
}

std::vector<NewPlacementQuery> build_placement_query(const std::string& alignment_path)
{
    parse::Alignment aln = parse::read_alignment_file(alignment_path);
    return build_placement_query(aln.names, aln.sequences);
}

std::vector<NewPlacementQuery> build_placement_query(
    const std::vector<std::string>& msa_tip_names,
    const std::vector<std::string>& msa_rows)
{
    if (msa_tip_names.size() != msa_rows.size()) {
        throw std::runtime_error("Alignment names/sequences size mismatch.");
    }

    std::vector<NewPlacementQuery> out_queries;
    out_queries.reserve(msa_tip_names.size());

    for (std::size_t i = 0; i < msa_tip_names.size(); ++i) {
        NewPlacementQuery q;
        q.node_id_pair = {-1, -1};
        q.msa_name = msa_tip_names[i];
        q.msa = msa_rows[i];
        out_queries.emplace_back(std::move(q));
    }
    return out_queries;
}
