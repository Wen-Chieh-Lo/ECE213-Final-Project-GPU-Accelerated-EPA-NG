#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "../mlipper_util.h"
#include "tree.hpp" 
#include "pmat.h"
#include "root_likelihood.cuh"
#include "tree_placement.cuh"
#include "partial_likelihood.cuh"
#include <tbb/parallel_for.h>

namespace {

// ----- Downward-op construction helpers -----

static inline int down_op_type_for_target(bool target_is_tip, bool sibling_is_tip) {
    if (target_is_tip && sibling_is_tip) return static_cast<int>(OP_DOWN_TIP_TIP);
    if (target_is_tip) return static_cast<int>(OP_DOWN_TIP_INNER);
    if (sibling_is_tip) return static_cast<int>(OP_DOWN_INNER_TIP);
    return static_cast<int>(OP_DOWN_INNER_INNER);
}

static inline void append_downward_op(
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
    op.left_id   = left_id;
    op.right_id  = right_id;
    op.left_tip_index  = left_is_tip  ? node_to_tip[left_id]  : -1;
    op.right_tip_index = right_is_tip ? node_to_tip[right_id] : -1;
    op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
    op.dir_tag  = dir_tag;

    const bool target_is_left = (dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_tip  = target_is_left ? left_is_tip : right_is_tip;
    const bool sibling_is_tip = target_is_left ? right_is_tip : left_is_tip;
    op.op_type = down_op_type_for_target(target_is_tip, sibling_is_tip);
    ops.push_back(op);
}

} // namespace

// ----- Host-side PMAT staging -----

// Build transition probability matrices for each node/rate category on host.
void fill_pmats_in_host_packing(
    const TreeBuildResult&       T,
    HostPacking&                 H,
    const EigResult&             er,
    const std::vector<double>&   rate_multipliers,   // len = rate_cats (per-category rate multipliers)
    int states,
    int rate_cats,
    const int* changed_nodes,
    int num_changed_nodes)
{
    const int N = (int)T.nodes.size();
    const size_t per_node = (size_t)rate_cats * states * states;

    const size_t required = (size_t)N * per_node;
    const bool want_incremental = (changed_nodes && num_changed_nodes > 0);

    auto reset_all = [&]() {
        H.pmats.assign(required, fp_t(0));
        H.pmats_mid.assign(required, fp_t(0));
        H.pmats_mid_prox.assign(required, fp_t(0));
        H.pmats_mid_dist.assign(required, fp_t(0));
    };

    auto buffers_look_compatible = [&]() -> bool {
        if (per_node == 0) return false;
        if (H.pmats.empty()) return false;
        if (H.pmats.size() % per_node != 0) return false;
        if (!H.pmats_mid.empty() && H.pmats_mid.size() % per_node != 0) return false;
        if (!H.pmats_mid_prox.empty() && H.pmats_mid_prox.size() % per_node != 0) return false;
        if (!H.pmats_mid_dist.empty() && H.pmats_mid_dist.size() % per_node != 0) return false;
        return true;
    };

    if (!want_incremental) {
        reset_all();
    } else if (!buffers_look_compatible()) {
        // No existing buffers to update incrementally; fall back to full rebuild.
        reset_all();
    } else {
        // Preserve existing pmats for unchanged nodes; extend buffers for newly added nodes.
        H.pmats.resize(required, fp_t(0));
        H.pmats_mid.resize(required, fp_t(0));
        if (!H.pmats_mid_prox.empty()) H.pmats_mid_prox.resize(required, fp_t(0));
        if (!H.pmats_mid_dist.empty()) H.pmats_mid_dist.resize(required, fp_t(0));
    }

    auto compute_node_pmats = [&](int nid) {
        if (nid < 0 || nid >= N) return;
        const TreeNode& nd = T.nodes[nid];
        if (nd.parent < 0) return;
        fp_t* base = H.pmats.data() + (size_t)nid * per_node;
        fp_t* base_mid = H.pmats_mid.data() + (size_t)nid * per_node;
        fp_t* base_mid_prox = H.pmats_mid_prox.empty() ? nullptr : (H.pmats_mid_prox.data() + (size_t)nid * per_node);
        fp_t* base_mid_dist = H.pmats_mid_dist.empty() ? nullptr : (H.pmats_mid_dist.data() + (size_t)nid * per_node);
        const double blen  = static_cast<double>(nd.branch_length_to_parent);
        std::vector<double> pbuf((size_t)states * (size_t)states);
        std::vector<double> pbuf_mid((size_t)states * (size_t)states);

        for (int rc = 0; rc < rate_cats; ++rc) {
            double r = rate_multipliers[rc];  // rate category multiplier
            double t = blen;                  // branch length
            double p = 0;

            fp_t* P = base + (size_t)rc * states * states;
            fp_t* Pmid = base_mid + (size_t)rc * states * states;
            fp_t* Pprox = base_mid_prox ? (base_mid_prox + (size_t)rc * states * states) : nullptr;
            fp_t* Pdist = base_mid_dist ? (base_mid_dist + (size_t)rc * states * states) : nullptr;
            pmatrix_from_triple(
                er.Vinv.data(), er.V.data(), er.lambdas.data(),
                            r, t, p, pbuf.data(), states);
            // half-branch PMAT for midpoint
            pmatrix_from_triple(
                er.Vinv.data(), er.V.data(), er.lambdas.data(),
                            r, t * 0.5, p, pbuf_mid.data(), states);
            for (size_t idx = 0; idx < pbuf.size(); ++idx) {
                P[idx] = static_cast<fp_t>(pbuf[idx]);
                Pmid[idx] = static_cast<fp_t>(pbuf_mid[idx]);
            }
            if (Pprox) std::copy(Pmid, Pmid + pbuf.size(), Pprox);
            if (Pdist) std::copy(Pmid, Pmid + pbuf.size(), Pdist);
        }
    };

    if (!want_incremental || !changed_nodes || num_changed_nodes <= 0) {
        tbb::parallel_for(0, N, [&](int nid) { compute_node_pmats(nid);});
    } else {
        for (int i = 0; i < num_changed_nodes; ++i) compute_node_pmats(changed_nodes[i]);
    }
}

// ----- Per-query CLV staging -----

// Select view for a specific query's PMAT chunk.
DeviceTree make_query_view(const DeviceTree& D, int query_idx) {
    DeviceTree view = D;
    if (query_idx < 0 || query_idx >= D.placement_queries) return view;
    const size_t clv_span = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    if (D.d_query_pmat) {
        // Shared query PMAT buffer across queries; rebuilt per-query.
        view.d_query_pmat = D.d_query_pmat;
    }
    if (D.d_query_clv) {
        view.d_query_clv = D.d_query_clv + (size_t)query_idx * clv_span;
    }
    return view;
}

__global__ void BuildQueryClvKernel(
    DeviceTree D,
    const uint8_t* query_chars,
    int query_idx)
{
    const unsigned site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= D.sites) return;
    const size_t base_char = (size_t)query_idx * D.sites + site;
    const uint8_t enc = query_chars ? query_chars[base_char] : 4;

    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    const size_t clv_span = (size_t)D.sites * per_site;
    fp_t* out = D.d_query_clv + (size_t)query_idx * clv_span + (size_t)site * per_site;
    for (int rc = 0; rc < D.rate_cats; ++rc) {
        fp_t* row = out + (size_t)rc * D.states;
        for (int s = 0; s < D.states; ++s) {
            row[s] = (enc < D.states) ? (s == enc ? fp_t(1) : fp_t(0)) : fp_t(1);
        }
    }
}

void build_query_clv(
    const DeviceTree& D,
    int query_idx,
    cudaStream_t stream)
{
    if (!D.d_query_clv || !D.d_query_chars) return;
    if (query_idx < 0 || query_idx >= D.placement_queries) return;
    dim3 block(256);
    dim3 grid((unsigned)((D.sites + block.x - 1) / block.x));
    BuildQueryClvKernel<<<grid, block, 0, stream>>>(D, D.d_query_chars, query_idx);
    CUDA_CHECK(cudaGetLastError());
}

// Pack topology and tip encodings from tree/MSA into HostPacking.
HostPacking pack_host_arrays_from_tree_and_msa(
        const TreeBuildResult& T,
        const std::vector<std::string>& msa_tip_names,  // len = tips
        const std::vector<std::string>& msa_rows,       // len = tips, each row length = sites
        size_t sites,
        int states)
{
    if (msa_rows.size() != msa_tip_names.size())
        throw std::runtime_error("MSA rows/names size mismatch.");
    if (msa_rows.empty()) throw std::runtime_error("Empty MSA.");
    if (msa_rows[0].size() != sites)
        throw std::runtime_error("Sites mismatch.");

    const int N = (int)T.nodes.size();

    // Topology
    HostPacking H;
    H.postorder = T.postorder;
    H.preorder  = T.preorder;
    H.parent.resize(N, -1);
    H.left.resize(N, -1);
    H.right.resize(N, -1);
    H.is_tip.resize(N, 0);
    H.blen.resize(N, fp_t(0));

    for (int i = 0; i < N; ++i) {
        const auto& nd = T.nodes[i];
        H.parent[i] = nd.parent;
        H.left[i]   = nd.left;
        H.right[i]  = nd.right;
        H.is_tip[i] = nd.is_tip ? 1 : 0;
        H.blen[i]   = nd.branch_length_to_parent;
    }
    
    // Build tip name -> MSA row lookup
    std::unordered_map<std::string,int> name_to_row;
    name_to_row.reserve(msa_tip_names.size()*2);
    for (int row_idx = 0; row_idx < (int)msa_tip_names.size(); ++row_idx) {
        name_to_row[msa_tip_names[row_idx]] = row_idx;
    }

    // Collect tip order in postorder and the corresponding node ids
    std::vector<int> tip_node_ids_host;
    tip_node_ids_host.reserve(msa_tip_names.size());
    for (int id : T.postorder) {
        if (T.nodes[id].is_tip) tip_node_ids_host.push_back(id);
    }
    const int tips = (int)tip_node_ids_host.size();

    if (tips != (int)msa_tip_names.size())
        throw std::runtime_error("Tip count in tree != MSA names.");

    H.tip_node_ids = tip_node_ids_host;

    // Encode MSA rows into tipchars following the fixed tipIndex order (0..tips-1)
    H.tipchars.resize((size_t)tips * sites);

    if (states == 4 || states == 5) {
        for (int t = 0; t < tips; ++t) {
            const int node_id = tip_node_ids_host[t];
            const auto& name  = T.nodes[node_id].name;
            auto it = name_to_row.find(name);
            if (it == name_to_row.end())
                throw std::runtime_error("Tip not found in MSA: " + name);
            const std::string& row = msa_rows[it->second];
            for (size_t s = 0; s < sites; ++s)
                H.tipchars[(size_t)t * sites + s] = encode_state_DNA5(row[s]);
        }
    } 
    // else if (states == 20) {
    //     auto encAA = [](char c)->uint8_t {
    //         // TODO: fill in your protein mapping; temporarily treat unknown as 19
    //         switch (c) {
    //             case 'A': case 'a': return 0;
    //             // ... remaining 18
    //             default: return 19;
    //         }
    //     };
    //     for (int t = 0; t < tips; ++t) {
    //         const int node_id = tip_node_ids_host[t];
    //         const auto& name  = T.nodes[node_id].name;
    //         auto it = name2row.find(name);
    //         if (it == name2row.end())
    //             throw std::runtime_error("Tip not found in MSA: " + name);
    //         const std::string& row = msa_rows[it->second];
    //         for (size_t s = 0; s < sites; ++s)
    //             H.tipchars[(size_t)t * sites + s] = encAA(row[s]);
    //     }
    // } else {
    //     throw std::runtime_error("Unsupported states; expect 4/5/20.");
    // }

    return H;
}

namespace {

static inline std::vector<fp_t> cast_to_fp(const std::vector<double>& src) {
    std::vector<fp_t> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = static_cast<fp_t>(src[i]);
    }
    return out;
}

} // namespace

// ===== Copy host packing to GPU and build DeviceTree =====
// Upload host packing and model parameters to GPU, constructing DeviceTree.
DeviceTree upload_to_gpu(
    const TreeBuildResult& T,
    const HostPacking& H,
    const EigResult& er,
    const std::vector<double>& rate_weights,
    const std::vector<double>& rate_multipliers,
    const std::vector<double>& pi,
    size_t sites, int states, int rate_cats, bool per_rate_scaling,
    const PlacementQueryBatch* queries)
{
    DeviceTree D;
    const int node_count = static_cast<int>(T.nodes.size());
    const int tip_count = static_cast<int>(H.tip_node_ids.size());
    const int query_count = queries ? static_cast<int>(queries->size()) : 0;
    const size_t site_count = sites;
    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t matrix_elems = state_count * state_count;

    D.N = node_count;
    D.tips = tip_count;
    D.inners = D.N - D.tips;
    D.placement_queries = query_count;
    D.root_id = T.root_id;

    // Reserve space for future insertions: each committed placement adds 2 nodes (1 internal + 1 tip).
    // This preallocates large CLV/PMAT/topology buffers so the tree can grow without reupload_to_gpu().
    const int reserve_inserts = D.placement_queries;
    D.capacity_N = D.N + 2 * reserve_inserts;
    D.capacity_tips = D.tips + reserve_inserts;
    D.query_capacity = D.placement_queries;
    D.sites = sites;
    D.states = states;
    D.rate_cats = rate_cats;
    D.log2_stride = ceil_log2_u32((unsigned int)(D.states + 1));
    D.per_rate_scaling = per_rate_scaling;

    if (static_cast<int>(rate_weights.size()) != rate_cats) {
        throw std::runtime_error("rate_weights size mismatch.");
    }
    if (static_cast<int>(pi.size()) != states) {
        throw std::runtime_error("pi size mismatch.");
    }

    // Host-side staging for model uploads.
    std::vector<fp_t> lambdas_scaled(rate_count * state_count, fp_t(0));
    for (int rate_idx = 0; rate_idx < D.rate_cats; ++rate_idx) {
        const double rate_multiplier = rate_multipliers[rate_idx];
        for (int state_idx = 0; state_idx < D.states; ++state_idx) {
            lambdas_scaled[static_cast<size_t>(rate_idx) * state_count + static_cast<size_t>(state_idx)] =
                static_cast<fp_t>(er.lambdas[state_idx] * rate_multiplier);
        }
    }

    const auto V_fp = cast_to_fp(er.V);
    const auto Vinv_fp = cast_to_fp(er.Vinv);
    const auto U_fp = cast_to_fp(er.U);
    const auto rate_weights_fp = cast_to_fp(rate_weights);
    const auto pi_fp = cast_to_fp(pi);

    const size_t model_matrix_bytes = sizeof(fp_t) * matrix_elems;
    const size_t lambdas_bytes = sizeof(fp_t) * lambdas_scaled.size();
    const size_t rate_weights_bytes = sizeof(fp_t) * rate_count;
    const size_t frequencies_bytes = sizeof(fp_t) * state_count;

    // --- model parameters ---
    CUDA_CHECK(cudaMalloc(&D.d_lambdas, lambdas_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_V, model_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_Vinv, model_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_U, model_matrix_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_rate_weights, rate_weights_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_frequencies, frequencies_bytes));

    CUDA_CHECK(cudaMemcpy(D.d_lambdas, lambdas_scaled.data(), lambdas_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_V, V_fp.data(), model_matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_Vinv, Vinv_fp.data(), model_matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_U, U_fp.data(), model_matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_rate_weights, rate_weights_fp.data(), rate_weights_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_frequencies, pi_fp.data(), frequencies_bytes, cudaMemcpyHostToDevice));

    // --- branch lengths and tip characters ---
    const size_t capacity_node_bytes = sizeof(fp_t) * static_cast<size_t>(D.capacity_N);
    const size_t live_node_bytes = sizeof(fp_t) * static_cast<size_t>(D.N);
    const size_t tipchar_capacity_bytes = sizeof(uint8_t) * static_cast<size_t>(D.capacity_tips) * site_count;
    const size_t tipchar_live_bytes = sizeof(uint8_t) * static_cast<size_t>(D.tips) * site_count;

    CUDA_CHECK(cudaMalloc(&D.d_blen, live_node_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_new_pendant_length, capacity_node_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_new_proximal_length, capacity_node_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_prev_pendant_length, capacity_node_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_prev_proximal_length, capacity_node_bytes));
    CUDA_CHECK(cudaMemcpy(D.d_blen, H.blen.data(), live_node_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(D.d_new_pendant_length, 0, capacity_node_bytes));
    CUDA_CHECK(cudaMemset(D.d_new_proximal_length, 0, capacity_node_bytes));
    CUDA_CHECK(cudaMemset(D.d_prev_pendant_length, 0, capacity_node_bytes));
    CUDA_CHECK(cudaMemset(D.d_prev_proximal_length, 0, capacity_node_bytes));

    CUDA_CHECK(cudaMalloc(&D.d_tipchars, tipchar_capacity_bytes));
    CUDA_CHECK(cudaMemcpy(D.d_tipchars, H.tipchars.data(), tipchar_live_bytes, cudaMemcpyHostToDevice));

    // --- CLV and derivative workspaces ---
    const size_t per_node = site_count * rate_count * state_count;
    const size_t clv_capacity_elems = static_cast<size_t>(D.capacity_N) * per_node;
    const size_t clv_total = clv_capacity_elems * 2; // up + down
    const size_t clv_total_bytes = sizeof(fp_t) * clv_total;
    const size_t clv_capacity_bytes = sizeof(fp_t) * clv_capacity_elems;
    const size_t placement_clv_bytes = sizeof(fp_t) * site_count;

    CUDA_CHECK(cudaMalloc(&D.d_clv_up, clv_total_bytes));
    D.d_clv_down = D.d_clv_up + clv_capacity_elems;
    D.clv_down_offset_elems = clv_capacity_elems;

    CUDA_CHECK(cudaMalloc(&D.d_clv_mid, clv_capacity_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_clv_mid_base, clv_capacity_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_downward_scratch, clv_capacity_bytes));
    CUDA_CHECK(cudaMalloc(&D.d_placement_clv, placement_clv_bytes));
    CUDA_CHECK(cudaMemset(D.d_placement_clv, 0, placement_clv_bytes));

    const size_t sumtable_stride = site_count * rate_count * state_count;
    const size_t max_ops = static_cast<size_t>(D.capacity_N) * 2;
    D.sumtable_capacity_ops = max_ops;
    D.likelihood_capacity_ops = max_ops;
    if (sumtable_stride > 0 && max_ops > 0) {
        CUDA_CHECK(cudaMalloc(&D.d_sumtable, sizeof(fp_t) * sumtable_stride * max_ops));
    }
    if (max_ops > 0) {
        CUDA_CHECK(cudaMalloc(&D.d_likelihoods, sizeof(fp_t) * max_ops));
    }

    // Optionally zero out the CLV pool (or let kernels overwrite)
    CUDA_CHECK(cudaMemset(D.d_clv_up, 0, clv_total_bytes));
    CUDA_CHECK(cudaMemset(D.d_clv_mid, 0, clv_capacity_bytes));
    CUDA_CHECK(cudaMemset(D.d_clv_mid_base, 0, clv_capacity_bytes));
    CUDA_CHECK(cudaMemset(D.d_downward_scratch, 0, clv_capacity_bytes));

    // Initialize only the root slice of down pool to exact 1.0; others remain 0 until overwritten.
    if (per_node > 0) {
        std::vector<fp_t> ones(per_node, fp_t(1));
        CUDA_CHECK(cudaMemcpy(D.d_clv_down + (size_t)T.root_id * per_node,
                              ones.data(),
                              per_node * sizeof(fp_t),
                              cudaMemcpyHostToDevice));
    }

    // --- PMAT buffers ---
    const size_t pmat_elems_cur = static_cast<size_t>(D.N) * rate_count * matrix_elems;
    const size_t pmat_elems_cap = static_cast<size_t>(D.capacity_N) * rate_count * matrix_elems;
    const size_t pmat_bytes_cur = sizeof(fp_t) * pmat_elems_cur;
    const size_t pmat_bytes_cap = sizeof(fp_t) * pmat_elems_cap;
    CUDA_CHECK(cudaMalloc(&D.d_pmat, pmat_bytes_cap));
    CUDA_CHECK(cudaMemset(D.d_pmat, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat, H.pmats.data(), pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid, pmat_bytes_cap));
    CUDA_CHECK(cudaMemset(D.d_pmat_mid, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid, H.pmats_mid.data(), pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid_prox, pmat_bytes_cap));
    if (!H.pmats_mid_prox.empty() && H.pmats_mid_prox.size() != pmat_elems_cur) {
        throw std::runtime_error("pmats_mid_prox size mismatch.");
    }
    const fp_t* pmat_mid_prox_src = H.pmats_mid_prox.empty() ? H.pmats_mid.data() : H.pmats_mid_prox.data();
    CUDA_CHECK(cudaMemset(D.d_pmat_mid_prox, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid_prox, pmat_mid_prox_src, pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid_dist, pmat_bytes_cap));
    if (!H.pmats_mid_dist.empty() && H.pmats_mid_dist.size() != pmat_elems_cur) {
        throw std::runtime_error("pmats_mid_dist size mismatch.");
    }
    const fp_t* pmat_mid_dist_src = H.pmats_mid_dist.empty() ? H.pmats_mid.data() : H.pmats_mid_dist.data();
    CUDA_CHECK(cudaMemset(D.d_pmat_mid_dist, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid_dist, pmat_mid_dist_src, pmat_bytes_cur, cudaMemcpyHostToDevice));

    // Allocate query PMAT buffer sized for up to ~2*N placement ops (edges).
    {
        const size_t per_query = rate_count * matrix_elems;
        const size_t query_slots = static_cast<size_t>(D.capacity_N) * 2;
        const size_t query_pmat_bytes = sizeof(fp_t) * per_query * query_slots;
        CUDA_CHECK(cudaMalloc(&D.d_query_pmat, query_pmat_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_pmat, 0, query_pmat_bytes));
    }

    // --- optional pattern weights ---
    if (!H.pattern_weights.empty()) {
        if (H.pattern_weights.size() != sites) {
            throw std::runtime_error("pattern_weights size mismatch.");
        }
        std::vector<fp_t> pattern_weights_fp(H.pattern_weights.size(), fp_t(1));
        for (size_t i = 0; i < H.pattern_weights.size(); ++i) {
            pattern_weights_fp[i] = static_cast<fp_t>(H.pattern_weights[i]);
        }
        CUDA_CHECK(cudaMalloc(&D.d_pattern_weights_u, sizeof(unsigned) * sites));
        CUDA_CHECK(cudaMemcpy(
            D.d_pattern_weights_u,
            H.pattern_weights.data(),
            sizeof(unsigned) * sites,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&D.d_pattern_weights, sizeof(fp_t) * sites));
        CUDA_CHECK(cudaMemcpy(
            D.d_pattern_weights,
            pattern_weights_fp.data(),
            sizeof(fp_t) * sites,
            cudaMemcpyHostToDevice));
    }

    // --- optional query buffers ---
    if (queries && !queries->empty()) {
        const size_t qcount = queries->size();
        const size_t qcap = (size_t)D.query_capacity;
        const size_t chars_bytes = sizeof(uint8_t) * qcap * site_count;
        CUDA_CHECK(cudaMalloc(&D.d_query_chars, chars_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_chars, 0, chars_bytes));
        CUDA_CHECK(cudaMemcpy(D.d_query_chars, queries->query_chars.data(), sizeof(uint8_t) * qcount * site_count, cudaMemcpyHostToDevice));

        const size_t query_clv_elems = qcap * site_count * rate_count * state_count;
        const size_t query_clv_bytes = sizeof(fp_t) * query_clv_elems;
        CUDA_CHECK(cudaMalloc(&D.d_query_clv, query_clv_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_clv, 0, query_clv_bytes));
    }

    // --- scaler buffers ---
    const size_t scaler_span = per_rate_scaling ? (site_count * rate_count) : site_count;
    const size_t scaler_pool = static_cast<size_t>(D.capacity_N) * scaler_span;
    CUDA_CHECK(cudaMalloc(&D.d_site_scaler_storage, sizeof(unsigned) * scaler_pool * 3));
    CUDA_CHECK(cudaMemset(D.d_site_scaler_storage, 0, sizeof(unsigned) * scaler_pool * 3));
    D.d_site_scaler_up = D.d_site_scaler_storage;
    D.d_site_scaler_down = D.d_site_scaler_storage + scaler_pool;
    D.d_site_scaler_mid = D.d_site_scaler_storage + scaler_pool * 2;
    D.d_site_scaler_mid_base = D.d_site_scaler_down;
    D.d_site_scaler = D.d_site_scaler_up;

    // --- tip bitmask lookup ---
    const unsigned int tipmap_size = D.states + 1;
    std::vector<unsigned int> tipmap(tipmap_size);
    for (unsigned int j = 0; j < tipmap_size; ++j) {
        if (j == static_cast<unsigned int>(D.states)) {
            tipmap[j] = 15;
        } else {
            tipmap[j] = 1u << j;
        }
    }
    CUDA_CHECK(cudaMalloc(&D.d_tipmap, sizeof(unsigned) * tipmap_size));
    CUDA_CHECK(cudaMemcpy(D.d_tipmap, tipmap.data(), tipmap_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    return D;
}

// ===== Release GPU resources =====
// Release all device buffers held by DeviceTree.
void free_device_tree(DeviceTree& D) {
    auto F = [](void* p){ if(p) cudaFree(p); };
    F(D.d_blen);
    F(D.d_new_pendant_length); F(D.d_new_proximal_length);
    F(D.d_prev_pendant_length); F(D.d_prev_proximal_length);
    F(D.d_tipchars);
    // d_clv_down is an offset into d_clv_up; free only the base allocation.
    F(D.d_clv_up);
    F(D.d_clv_mid);
    F(D.d_clv_mid_base);
    F(D.d_downward_scratch);
    F(D.d_site_scaler_storage);
    F(D.d_lambdas); F(D.d_V); F(D.d_Vinv); F(D.d_U); F(D.d_rate_weights); F(D.d_frequencies);
    F(D.d_pmat); F(D.d_pmat_mid); F(D.d_pmat_mid_prox); F(D.d_pmat_mid_dist);
    F(D.d_placement_clv);
    F(D.d_sumtable);
    F(D.d_likelihoods);
    F(D.d_pattern_weights_u);
    F(D.d_pattern_weights);
    F(D.d_query_clv); F(D.d_query_chars);
    F(D.d_query_pmat);
    F(D.d_tipmap);
    D = DeviceTree{};
}


static void rebuild_node_to_tip_map(
    const TreeBuildResult& tree,
    const HostPacking& host,
    std::vector<int>& node_to_tip)
{
    const int node_count = static_cast<int>(tree.nodes.size());
    node_to_tip.assign(static_cast<size_t>(node_count), -1);
    for (int tip_idx = 0; tip_idx < static_cast<int>(host.tip_node_ids.size()); ++tip_idx) {
        const int node_id = host.tip_node_ids[tip_idx];
        if (node_id >= 0 && node_id < node_count) {
            node_to_tip[node_id] = tip_idx;
        }
    }
}

static void build_upward_ops_host(
    const TreeBuildResult& tree,
    const std::vector<int>& node_to_tip,
    std::vector<NodeOpInfo>& upward_ops_host)
{
    const int node_count = static_cast<int>(tree.nodes.size());
    upward_ops_host.clear();
    upward_ops_host.reserve(static_cast<size_t>(node_count));

    for (int node_id : tree.postorder) {
        if (node_id < 0 || node_id >= node_count) continue;
        const TreeNode& node = tree.nodes[node_id];
        if (node.is_tip) continue;

        const int left_id = node.left;
        const int right_id = node.right;
        if (left_id < 0 || right_id < 0) continue;

        const bool left_is_tip = tree.nodes[left_id].is_tip;
        const bool right_is_tip = tree.nodes[right_id].is_tip;
        const int left_tip_idx = left_is_tip ? node_to_tip[left_id] : -1;
        const int right_tip_idx = right_is_tip ? node_to_tip[right_id] : -1;

        NodeOpType op_type = OP_INNER_INNER;
        if (left_is_tip && right_is_tip) {
            op_type = OP_TIP_TIP;
        } else if (left_is_tip || right_is_tip) {
            op_type = OP_TIP_INNER;
        }

        NodeOpInfo op{};
        op.parent_id = node_id;
        op.left_id = left_id;
        op.right_id = right_id;
        op.left_tip_index = left_tip_idx;
        op.right_tip_index = right_tip_idx;
        op.op_type = static_cast<int>(op_type);
        op.clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
        op.dir_tag = static_cast<uint8_t>(CLV_DIR_UP);
        upward_ops_host.push_back(op);
    }
}

static void build_downward_ops_host(
    const TreeBuildResult& tree,
    const std::vector<int>& node_to_tip,
    std::vector<NodeOpInfo>& downward_ops_host)
{
    const int node_count = static_cast<int>(tree.nodes.size());
    downward_ops_host.clear();
    downward_ops_host.reserve(static_cast<size_t>(node_count) * 2);

    for (int parent_id : tree.preorder) {
        const TreeNode& node = tree.nodes[parent_id];
        if (node.is_tip) continue;

        const int left_id = node.left;
        const int right_id = node.right;
        const bool left_is_tip = (left_id >= 0 && tree.nodes[left_id].is_tip);
        const bool right_is_tip = (right_id >= 0 && tree.nodes[right_id].is_tip);

        append_downward_op(
            downward_ops_host,
            parent_id,
            left_id,
            right_id,
            left_is_tip,
            right_is_tip,
            node_to_tip,
            static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));

        append_downward_op(
            downward_ops_host,
            parent_id,
            left_id,
            right_id,
            left_is_tip,
            right_is_tip,
            node_to_tip,
            static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    }
}

static void launch_upward_clv_update(
    const DeviceTree& D,
    const std::vector<NodeOpInfo>& upward_ops_host,
    int num_sms,
    cudaStream_t stream)
{
    const int upward_op_count = static_cast<int>(upward_ops_host.size());
    if (upward_op_count <= 0) {
        throw std::runtime_error("No upward ops to update");
    }

    NodeOpInfo* d_upward_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_upward_ops, sizeof(NodeOpInfo) * static_cast<size_t>(upward_op_count)));
    CUDA_CHECK(cudaMemcpyAsync(
        d_upward_ops,
        upward_ops_host.data(),
        sizeof(NodeOpInfo) * static_cast<size_t>(upward_op_count),
        cudaMemcpyHostToDevice,
        stream));

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, Rtree_Likelihood_Site_Parallel_Upward_Kernel));
    int block = 256;
    if (attr.maxThreadsPerBlock > 0 && block > attr.maxThreadsPerBlock) {
        block = attr.maxThreadsPerBlock;
    }

    int max_blocks_per_sm = 4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        Rtree_Likelihood_Site_Parallel_Upward_Kernel,
        block,
        0));
    int max_blocks = num_sms * max_blocks_per_sm;
    int grid = static_cast<int>((D.sites + static_cast<size_t>(block) - 1) / static_cast<size_t>(block));
    if (max_blocks > 0 && grid > max_blocks) {
        grid = max_blocks;
    }

    Rtree_Likelihood_Site_Parallel_Upward_Kernel<<<grid, block, 0, stream>>>(
        D,
        d_upward_ops,
        upward_op_count);
    CHECK_CUDA_LAST();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_upward_ops));
}

static void launch_downward_clv_update(
    const DeviceTree& D,
    NodeOpInfo* d_ops,
    int num_ops,
    int num_sms,
    cudaStream_t stream)
{
    if (num_ops <= 0 || !d_ops) return;

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, Rtree_Likelihood_Site_Parallel_Downward_Kernel));
    int block = 256;
    if (attr.maxThreadsPerBlock > 0 && block > attr.maxThreadsPerBlock) {
        block = attr.maxThreadsPerBlock;
    }

    int max_blocks_per_sm = 4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        Rtree_Likelihood_Site_Parallel_Downward_Kernel,
        block,
        0));
    int max_blocks = num_sms * max_blocks_per_sm;
    int grid = static_cast<int>((D.sites + static_cast<size_t>(block) - 1) / static_cast<size_t>(block));
    if (max_blocks > 0 && grid > max_blocks) {
        grid = max_blocks;
    }

    Rtree_Likelihood_Site_Parallel_Downward_Kernel<<<grid, block, 0, stream>>>(D, d_ops, num_ops);
    CHECK_CUDA_LAST();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void free_placement_op_buffer(
    PlacementOpBuffer& placement_ops,
    cudaStream_t stream)
{
    if (placement_ops.d_ops) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(placement_ops.d_ops));
        placement_ops.d_ops = nullptr;
    }
    placement_ops.num_ops = 0;
}

// ----- Tree CLV update pipeline -----

void UpdateTreeClvs(
    DeviceTree& D,
    TreeBuildResult& T,
    HostPacking& H,
    PlacementOpBuffer& placement_ops,
    cudaStream_t stream)
{
    std::vector<NodeOpInfo> downward_ops_host;
    std::vector<NodeOpInfo> upward_ops_host;
    std::vector<int> node_to_tip(T.nodes.size(), -1);

    cudaDeviceProp device_props{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, 0));
    const int sm_count = device_props.multiProcessorCount;

    rebuild_node_to_tip_map(T, H, node_to_tip);
    build_upward_ops_host(T, node_to_tip, upward_ops_host);
    launch_upward_clv_update(D, upward_ops_host, sm_count, stream);

    build_downward_ops_host(T, node_to_tip, downward_ops_host);
    free_placement_op_buffer(placement_ops, stream);
    placement_ops.num_ops = static_cast<int>(downward_ops_host.size());
    if (placement_ops.num_ops > 0) {
        CUDA_CHECK(cudaMalloc(
            &placement_ops.d_ops,
            sizeof(NodeOpInfo) * static_cast<size_t>(placement_ops.num_ops)));
        CUDA_CHECK(cudaMemcpyAsync(
            placement_ops.d_ops,
            downward_ops_host.data(),
            sizeof(NodeOpInfo) * static_cast<size_t>(placement_ops.num_ops),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    launch_downward_clv_update(D, placement_ops.d_ops, placement_ops.num_ops, sm_count, stream);
}

// ----- Query placement evaluation -----

void EvaluatePlacementQueries(
    DeviceTree& D,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    const PlacementOpBuffer& placement_ops,
    std::vector<PlacementResult>* placement_results_out,
    int smoothing,
    cudaStream_t stream)
{
    if (placement_results_out) {
        placement_results_out->clear();
        placement_results_out->reserve(static_cast<size_t>(D.placement_queries));
    }

    int query_count = D.placement_queries;
    if (const char* env_max_queries = std::getenv("MLIPPER_MAX_QUERIES")) {
        const int parsed = std::atoi(env_max_queries);
        if (parsed > 0) {
            query_count = std::min(D.placement_queries, parsed);
        }
    }

    for (int query_idx = 0; query_idx < query_count; ++query_idx) {
        CUDA_CHECK(cudaMemset(D.d_new_pendant_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_new_proximal_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_prev_pendant_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_prev_proximal_length, 0, sizeof(double) * D.N));

        build_query_clv(D, query_idx, stream);
        CHECK_CUDA_LAST();

        DeviceTree query_view = make_query_view(D, query_idx);
        PlacementResult placement = PlacementEvaluationKernel(
            query_view,
            er,
            rate_multipliers,
            placement_ops.d_ops,
            placement_ops.num_ops,
            smoothing,
            stream);
        if (placement_results_out) {
            placement_results_out->push_back(placement);
        }
    }
}
