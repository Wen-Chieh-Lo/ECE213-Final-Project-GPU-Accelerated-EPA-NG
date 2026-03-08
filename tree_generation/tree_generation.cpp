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

double EvaluateTreeLogLikelihood_device(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream);

static void throw_if(bool cond, const char* msg) {
    if (cond) throw std::runtime_error(msg);
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
    batch.branch_lengths.assign(qcount, 0.5);
    batch.query_chars.resize(qcount * sites, 4);
    printf("Preparing %zu placement queries on GPU (sites=%zu)\n", qcount, sites);
    for (size_t qi = 0; qi < qcount; ++qi) {
        const auto& q = placement_queries[qi];
        printf("Q size: %zu\n", q.msa.size());
        if (q.pendant > 0.0) batch.branch_lengths[qi] = q.pendant;
        if (q.msa.size() != sites) {
            throw std::runtime_error("Query sequence length mismatch.");
        }
        for (size_t s = 0; s < sites; ++s) {
            batch.query_chars[qi * sites + s] = encode_state_DNA5(q.msa[s]);
        }
    }
    // fill_query_pmats(batch, eig, rate_multipliers, states, rate_cats);
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
            dst.branch_length_to_parent = 0.0;
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

    // 3) Host-side GTR decomposition (V/Vinv/λ)
    EigResult Eigen = gtr_eigendecomp_cpu(Q_rowmajor.data(), pi.data(), states);

    fill_pmats_in_host_packing(T, H, Eigen, pi, rate_multipliers, states, rate_cats);

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

// Evaluate tree log-likelihood using prepared DeviceTree/HostPacking and model vectors.
double EvaluateTreeLogLikelihood(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream = 0)
{
    double loglk = EvaluateTreeLogLikelihood_device(D, T, H, pi, rate_weights, stream);
    return loglk;
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
