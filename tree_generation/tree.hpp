#pragma once
#include <cuda_runtime.h>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <libpll/pll.h>

#include "../pmatrix/pmat.h"
#include "precision.hpp"

struct PlacementResult;

// ===== CUDA error helper =====
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr);  \
    if (_e != cudaSuccess) {  \
      throw std::runtime_error(std::string("[CUDA] ") + cudaGetErrorString(_e)); \
    }                         \
} while(0)
#endif

inline uint8_t encode_state_DNA5(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't':
        case 'U': case 'u': return 3;
        case '-':           return 4; // gap
        default:            return 4; // Treat as gap for now; switch to bitmask if you need IUPAC
    }
}

struct NewPlacementQuery{
    std::pair<int,int> node_id_pair{-1, -1}; // node where to insert {first = parent, second = child}
    fp_t pendant = fp_t(0);                  // branch length of new insertion
    fp_t distal = fp_t(0);                   // branch length to new insertion
    std::string msa;                         // sequence
    std::string msa_name;                    // sequence name
};


// GPU layout assumes every node has the same CLV size (sites * rate_cats * states)
struct TreeNode {
    int   id = -1;
    bool  is_tip = false;
    int   left = -1;    // child id
    int   right = -1;   // child id
    int   parent = -1;  // parent id (root = -1)
    fp_t branch_length_to_parent = fp_t(0); // Number after the colon in Newick
    fp_t branch_length_to_insert = fp_t(0); // for insertion operations
    std::string name;   // only for tips
    // GPU offsets: use these directly with scaler pool
    size_t scaler_offset = 0;  // elements-based offset (per-site or per-rate)
};

struct TreeBuildResult {
    std::vector<TreeNode> nodes;     // 0..N-1
    int root_id = -1;
    std::vector<int> postorder;      // Postorder (children -> parent)
    std::vector<int> preorder;   // Preorder (children -> parent)
    // Tip name -> node id map (so MSA tip CLVs can be placed directly)
    std::unordered_map<std::string,int> tip_node_by_name;
};

TreeBuildResult build_tree_from_newick_with_pll(
    const std::vector<std::string>& msa_tip_names,
    const std::string& newick_text,
    size_t sites,
    int states,
    int rate_cats,
    bool per_rate_scaling);

struct DeviceTree {
    // sizes
    int     N = 0;            // nodes = tips + inners
    int     tips = 0;
    int     inners = 0;
    int     placement_queries = 0; // number of placement queries staged on device
    // capacities (allocated sizes). These can be larger than N/tips to support appending query placements
    // without re-allocating large buffers (CLVs/PMATs/topology).
    int     capacity_N = 0;
    int     capacity_tips = 0;
    int     query_capacity = 0;
    int     root_id = -1;
    size_t  sites = 0;
    int     states = 0;
    int     rate_cats = 0;
    // Cached helper for tipmap indexing (ceil_log2(states+1)).
    unsigned int log2_stride = 0;
    bool    per_rate_scaling = false;
    bool    force_generic_downward = false;
    bool    force_generic_upward = false;

    fp_t   *d_lambdas  = nullptr;  // [rate_cats * states]
    fp_t   *d_V        = nullptr;  // [states*states]  row-major
    fp_t   *d_Vinv     = nullptr;  // [states*states]  row-major
    fp_t   *d_U        = nullptr;  // [states*states]  (optional; use directly if kernels need U)
    fp_t   *d_rate_w   = nullptr;  // [rate_cats]      Discrete Gamma or other rate categories
    fp_t   *d_frequencies = nullptr;  // [states]       Base frequencies (pi)
    fp_t   *d_rate_weights = nullptr; // [rate_cats]    Discrete rate weights (copy of host)

    // topology (device)
    // NOTE: this pipeline uses host-built NodeOpInfo lists to drive kernels, so we do not store
    // full tree topology arrays on device (postorder/preorder/parent/left/right).
    fp_t   *d_blen      = nullptr; // [N] branch length to parent
    fp_t   *d_new_pendant_length = nullptr;  // [N] updated pendant branch lengths (derivative output)
    fp_t   *d_new_proximal_length = nullptr; // [N] updated proximal branch lengths (derivative output)
    fp_t   *d_prev_pendant_length = nullptr;  // [N] previous pendant branch lengths (smoothing rollback)
    fp_t   *d_prev_proximal_length = nullptr; // [N] previous proximal branch lengths (smoothing rollback)

    // tips
    // tip indices 0..tips-1 follow the postorder traversal order encountered in TreeBuildResult.nodes
    uint8_t *d_tipchars     = nullptr;          // [tips * sites], DNA5: 0..4 (including gap)

    // CLV buffers: up/down split (can be two allocations or two slices of one big allocation).
    // layout per segment: contiguous by node: node i at [i * per_node_elems ..)
    fp_t    *d_clv_up       = nullptr;          // up/passive CLV (postorder result)
    fp_t    *d_clv_down     = nullptr;          // down/alternative CLV (optional second half of a big pool)
    // When using a single allocation cut in two, clv_down_offset_elems is the base offset into that buffer.
    size_t   clv_down_offset_elems = 0;
    // Midpoint CLV scratch (per-node sized, reused per branch)
    fp_t    *d_clv_mid      = nullptr;          // [sites * rate_cats * states]
    // Cached parent_down * sibling_up products for midpoint reuse.
    fp_t    *d_clv_mid_base = nullptr;          // [sites * rate_cats * states]
    // Persistent workspace for downward convergence updates (same span as one CLV pool).
    fp_t    *d_downward_scratch = nullptr;      // [capacity_N * sites * rate_cats * states]
    // Midpoint debug accumulator
    fp_t    *d_placement_clv = nullptr;
    // Workspace for derivative/sumtable calculations reused across placements.
    fp_t    *d_sumtable = nullptr;              // [sumtable_capacity_ops * sites * rate_cats * states]
    fp_t    *d_likelihoods = nullptr;           // [likelihood_capacity_ops]
    size_t   sumtable_capacity_ops = 0;
    size_t   likelihood_capacity_ops = 0;

    // scaler pool storage.
    // `down` and `mid_base` share one history; `mid` diverges after midpoint/query updates.
    unsigned *d_site_scaler = nullptr;          // legacy alias for UP/root likelihood path
    unsigned *d_site_scaler_storage = nullptr;  // base allocation
    unsigned *d_site_scaler_up = nullptr;
    unsigned *d_site_scaler_down = nullptr;
    unsigned *d_site_scaler_mid = nullptr;
    unsigned *d_site_scaler_mid_base = nullptr;

    fp_t* d_pmat = nullptr;
    fp_t* d_pmat_mid = nullptr;             // half-branch pmats for midpoint calculations
    fp_t* d_pmat_mid_prox = nullptr;        // proximal branch pmats for midpoint calculations
    fp_t* d_pmat_mid_dist = nullptr;        // distal branch pmats for midpoint calculations
    unsigned int* d_tipmap = nullptr;

    // Ready-to-place query buffers
    uint8_t* d_query_chars = nullptr;            // [num_queries * sites] encoded DNA5
    fp_t   *d_query_clv  = nullptr;              // [sites * rate_cats * states] working buffer
    fp_t   *d_query_pmat = nullptr;              // base pointer for query pmats

    // Helpers for calculating sizes
    size_t per_node_elems() const {
        return sites * (size_t)rate_cats * (size_t)states;
    }
    size_t clv_pool_elems() const {
        return (size_t)N * per_node_elems();
    }
    size_t scaler_elems() const {
        return per_rate_scaling ? (sites * (size_t)rate_cats) : sites;
    }
    size_t scaler_pool_elems() const {
        return (size_t)capacity_N * scaler_elems();
    }
    size_t scaler_storage_elems() const {
        return scaler_pool_elems() * 3;
    }
    size_t pmat_per_node_elems() const {
        return (size_t)rate_cats * (size_t)states * (size_t)states;
    }
};

struct HostPacking {
    // Host-side staging buffers for cudaMemcpy
    std::vector<int>     postorder, preorder, parent, left, right;
    std::vector<uint8_t> is_tip;
    std::vector<fp_t>    blen;

    std::vector<int>     tip_node_ids;      // size = tips
    std::vector<uint8_t> tipchars;          // size = tips * sites

    std::vector<unsigned> site_scaler;      // size = sites or sites*rate
    std::vector<fp_t>     pmats;
    std::vector<fp_t>     pmats_mid;        // half-branch pmats for midpoint calculations
    std::vector<fp_t>     pmats_mid_prox;   // proximal branch pmats for midpoint calculations
    std::vector<fp_t>     pmats_mid_dist;   // distal branch pmats for midpoint calculations
};

// Placement queries are staged separately from HostPacking to avoid coupling
// tree staging with per-query data.
struct PlacementQueryBatch {
    size_t count = 0;                            // number of placement queries
    std::vector<fp_t> branch_lengths;            // per-query pendant length
    std::vector<uint8_t> query_chars;            // [count * sites] encoded DNA5
    std::vector<fp_t> query_pmats;               // [count * rate_cats * states * states] half-branch pmats

    bool empty() const { return count == 0; }
    size_t size() const { return count; }
};


HostPacking pack_host_arrays_from_tree_and_msa(
    const TreeBuildResult& T,
    const std::vector<std::string>& msa_tip_names,  // len = tips
    const std::vector<std::string>& msa_rows,       // len = tips, each row length = sites
    size_t sites,
    int states
);
void fill_pmats_in_host_packing(
    const TreeBuildResult&       T,
    HostPacking&                 H,
    const EigResult&             er,
    const std::vector<double>&   pi,           // len = states
    const std::vector<double>&   rate_multipliers,   // len = rate_cats (per-category rate multipliers)
    int states,
    int rate_cats,
    // Optional: update only these node IDs (edge pmats for node->parent). When omitted, rebuild for all nodes.
    const int* changed_nodes = nullptr,
    int num_changed_nodes = 0
);
void fill_query_pmats(
    PlacementQueryBatch&         Q,
    const EigResult&             er,
    const std::vector<double>&   rate_multipliers,
    int states,
    int rate_cats);
DeviceTree make_query_view(const DeviceTree& D, int query_idx);
void build_query_clv_on_device(
    const DeviceTree& D,
    int query_idx,
    cudaStream_t stream = 0);

DeviceTree upload_to_gpu(
    const TreeBuildResult& T,
    const HostPacking& H,
    const EigResult& er,
    const std::vector<double>& rate_weights,
    const std::vector<double>& rate_multipliers,
    const std::vector<double>& pi,
    size_t sites, int states, int rate_cats, bool per_rate_scaling,
    const PlacementQueryBatch* queries = nullptr
);

struct BuildToGpuResult{
    DeviceTree dev;
    TreeBuildResult tree;
    HostPacking hostPack;
    EigResult eig;
    PlacementQueryBatch queries;
};

BuildToGpuResult BuildAllToGPU(
    const std::vector<std::string>& msa_tip_names,
    const std::vector<std::string>& msa_rows,
    const std::string& newick_text,
    const std::vector<double>& Q_rowmajor,   // size = states*states
    const std::vector<double>& pi,           // size = states
    const std::vector<double>& rate_multipliers,   // size = rate_cats
    const std::vector<double>& rate_weights, // size = rate_cats
    size_t sites, int states, int rate_cats, bool per_rate_scaling,
    const std::vector<NewPlacementQuery>& placement_queries);

void free_device_tree(DeviceTree& D);
static void throw_if(bool cond, const char* msg);

void launch_init_tip_clv(const DeviceTree& D);

double eval_root_loglikelihood(
    const DeviceTree& D,
    int root_id,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream
);

double EvaluateTreeLogLikelihood(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream
);

// Device-side site-parallel likelihood evaluation (upward pass)
double EvaluateTreeLogLikelihood_device(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream);

// Preorder downward CLV propagation launcher
void LaunchPreorderDownwardClv(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    cudaStream_t           stream = 0);

// Downward pass CLV update for full tree (preorder-based)
void UpdateTreeLogLikelihood_device(
    DeviceTree&            D,
    TreeBuildResult&       T,
    HostPacking&           H,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    std::vector<struct PlacementResult>* placement_results_out = nullptr,
    int smoothing = 1,
    cudaStream_t           stream = 0);

// Debug helper: pretty-print tree structure (expects root_id to be set)
void print_tree_structure(const TreeBuildResult& T);

std::vector<NewPlacementQuery> build_placement_query(const std::string& msa_path);
std::vector<NewPlacementQuery> build_placement_query(
    const std::vector<std::string>& msa_tip_names,
    const std::vector<std::string>& msa_rows);
