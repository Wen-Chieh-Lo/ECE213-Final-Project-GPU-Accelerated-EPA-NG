#include <vector>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cassert>
#include <tuple>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include "tree_placement.cuh"
#include "../mlipper_util.h"
#include "../pmatrix/pmat.h"
#include "../pmatrix/pmat_gpu.cuh"
#include "tree.hpp"
#include "../partial_CUDA/partial_likelihood.cuh"
#include "root_likelihood.cuh"
#include "derivative.cuh"

// Debug helper: validate that a pointer lives on device memory.
static void check_device_ptr(const void* ptr, const char* name) {
    cudaPointerAttributes attr{};
    cudaError_t st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        fprintf(stderr, "[dptr-check] %s invalid (ptr=%p, err=%s, type=%d)\n",
                name, ptr, cudaGetErrorString(st), (st == cudaSuccess) ? attr.type : -1);
        throw std::runtime_error(std::string("Invalid device pointer: ") + name);
    }
}

// Fetch a single NodeOpInfo from device (host-side helper).
static NodeOpInfo fetch_op_from_device(const NodeOpInfo* d_ops, int op_idx) {
    NodeOpInfo op{};
    if (!d_ops || op_idx < 0) return op;
    CUDA_CHECK(cudaMemcpy(&op, d_ops + op_idx, sizeof(NodeOpInfo), cudaMemcpyDeviceToHost));
    return op;
}

// Dump one query PMAT (host-side) for a given op index.
static void print_query_pmat(const DeviceTree& D, int op_idx, cudaStream_t stream, const char* tag) {
    if (!D.d_query_pmat) {
        fprintf(stderr, "[MLIPPER query_pmat %s] d_query_pmat is null\n", tag ? tag : "");
        return;
    }
    const size_t per_query = (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
    if (op_idx < 0) op_idx = 0;
    const size_t offset = (size_t)op_idx * per_query;
    std::vector<double> host(per_query, 0.0);
    CUDA_CHECK(cudaMemcpyAsync(
        host.data(),
        D.d_query_pmat + offset,
        sizeof(double) * per_query,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int rc = 0; rc < D.rate_cats; ++rc) {
        for (int i = 0; i < D.states; ++i) {
            fprintf(stderr, "[MLIPPER query_pmat %s] rc=%d row=%d", tag ? tag : "", rc, i);
            for (int j = 0; j < D.states; ++j) {
                const size_t idx = (size_t)rc * (size_t)D.states * (size_t)D.states
                                 + (size_t)i * (size_t)D.states
                                 + (size_t)j;
                fprintf(stderr, " %.12e", host[idx]);
            }
            fprintf(stderr, "\n");
        }
    }
}

int get_env_int_or(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    return std::atoi(raw);
}

double get_env_double_or(const char* name, double fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    return std::atof(raw);
}

void dump_node_scaler_and_clv_snapshot(
    const DeviceTree& D,
    int node_id,
    int max_sites,
    cudaStream_t stream,
    const char* tag)
{
    if (node_id < 0 || node_id >= D.N) {
        std::fprintf(stderr, "[scaler-debug %s] invalid node_id=%d\n", tag ? tag : "", node_id);
        return;
    }
    const int site_count = std::max(0, std::min<int>(D.sites, max_sites));
    if (site_count <= 0) return;

    const size_t scaler_site_width = D.per_rate_scaling ? (size_t)D.rate_cats : 1u;
    const size_t scaler_count = (size_t)site_count * scaler_site_width;
    const size_t clv_count = (size_t)site_count * (size_t)D.rate_cats * (size_t)D.states;
    const size_t scaler_node_offset = (size_t)node_id * D.scaler_elems();
    const size_t clv_node_offset = (size_t)node_id * D.per_node_elems();

    std::vector<unsigned> up_scaler(scaler_count, 0);
    std::vector<unsigned> down_scaler(scaler_count, 0);
    std::vector<unsigned> mid_scaler(scaler_count, 0);
    std::vector<unsigned> mid_base_scaler(scaler_count, 0);
    std::vector<double> up_clv(clv_count, 0.0);
    std::vector<double> down_clv(clv_count, 0.0);
    std::vector<double> mid_clv(clv_count, 0.0);
    std::vector<double> mid_base_clv(clv_count, 0.0);
    std::vector<double> query_clv(clv_count, 0.0);

    auto copy_scaler = [&](unsigned* src, std::vector<unsigned>& dst) {
        if (!src || dst.empty()) return;
        CUDA_CHECK(cudaMemcpyAsync(
            dst.data(),
            src + scaler_node_offset,
            sizeof(unsigned) * dst.size(),
            cudaMemcpyDeviceToHost,
            stream));
    };
    auto copy_clv = [&](double* src, std::vector<double>& dst, bool per_node) {
        if (!src || dst.empty()) return;
        const double* base = per_node ? (src + clv_node_offset) : src;
        CUDA_CHECK(cudaMemcpyAsync(
            dst.data(),
            base,
            sizeof(double) * dst.size(),
            cudaMemcpyDeviceToHost,
            stream));
    };

    copy_scaler(D.d_site_scaler_up, up_scaler);
    copy_scaler(D.d_site_scaler_down, down_scaler);
    copy_scaler(D.d_site_scaler_mid, mid_scaler);
    copy_scaler(D.d_site_scaler_mid_base, mid_base_scaler);
    copy_clv(D.d_clv_up, up_clv, true);
    copy_clv(D.d_clv_down, down_clv, true);
    copy_clv(D.d_clv_mid, mid_clv, true);
    copy_clv(D.d_clv_mid_base, mid_base_clv, true);
    copy_clv(D.d_query_clv, query_clv, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto scaler_at = [&](const std::vector<unsigned>& buf, int site, int rc) -> unsigned {
        if (buf.empty()) return 0;
        if (D.per_rate_scaling) return buf[(size_t)site * (size_t)D.rate_cats + (size_t)rc];
        return buf[(size_t)site];
    };
    auto clv_max_at = [&](const std::vector<double>& buf, int site, int rc) -> double {
        if (buf.empty()) return 0.0;
        const size_t base = ((size_t)site * (size_t)D.rate_cats + (size_t)rc) * (size_t)D.states;
        double vmax = 0.0;
        for (int s = 0; s < D.states; ++s) {
            const double v = buf[base + (size_t)s];
            if (v > vmax) vmax = v;
        }
        return vmax;
    };

    std::fprintf(stderr,
        "[scaler-debug %s] node=%d sites=%d rate_cats=%d states=%d per_rate=%d\n",
        tag ? tag : "", node_id, site_count, D.rate_cats, D.states, D.per_rate_scaling ? 1 : 0);
    for (int site = 0; site < site_count; ++site) {
        std::fprintf(stderr, "[scaler-debug %s] site=%d", tag ? tag : "", site);
        for (int rc = 0; rc < D.rate_cats; ++rc) {
            std::fprintf(stderr,
                " rc=%d up=%u down=%u mid=%u midb=%u qmax=%.3e upmax=%.3e downmax=%.3e midmax=%.3e midbmax=%.3e",
                rc,
                scaler_at(up_scaler, site, rc),
                scaler_at(down_scaler, site, rc),
                scaler_at(mid_scaler, site, rc),
                scaler_at(mid_base_scaler, site, rc),
                clv_max_at(query_clv, site, rc),
                clv_max_at(up_clv, site, rc),
                clv_max_at(down_clv, site, rc),
                clv_max_at(mid_clv, site, rc),
                clv_max_at(mid_base_clv, site, rc));
        }
        std::fprintf(stderr, "\n");
    }
}

static void print_top_ops_summary(
    const std::vector<NodeOpInfo>& h_ops,
    const std::vector<double>& h_loglk,
    const std::vector<double>& h_pendant,
    const std::vector<double>& h_proximal,
    int query_idx,
    int pass,
    int topk)
{
    if (topk <= 0 || h_ops.empty() || h_loglk.empty()) return;
    std::vector<std::tuple<double, int, int>> ranked;
    ranked.reserve(h_loglk.size());
    for (int op_i = 0; op_i < (int)h_loglk.size(); ++op_i) {
        const NodeOpInfo& op = h_ops[(size_t)op_i];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        const int target_id = target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
        ranked.emplace_back(h_loglk[(size_t)op_i], op_i, target_id);
    }
    std::partial_sort(
        ranked.begin(),
        ranked.begin() + std::min<int>(topk, ranked.size()),
        ranked.end(),
        [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    for (int rank = 0; rank < std::min<int>(topk, ranked.size()); ++rank) {
        const double ll = std::get<0>(ranked[(size_t)rank]);
        const int op_i = std::get<1>(ranked[(size_t)rank]);
        const int target_id = std::get<2>(ranked[(size_t)rank]);
        double pendant = 0.0;
        double proximal = 0.0;
        if (target_id >= 0 && target_id < (int)h_pendant.size()) pendant = h_pendant[(size_t)target_id];
        if (target_id >= 0 && target_id < (int)h_proximal.size()) proximal = h_proximal[(size_t)target_id];
        std::fprintf(stderr,
            "[topops] query=%d pass=%d rank=%d op=%d target=%d ll=%.12f pendant=%.12f proximal=%.12f\n",
            query_idx, pass, rank + 1, op_i, target_id, ll, pendant, proximal);
    }
}

static void print_selected_op_summary(
    const std::vector<NodeOpInfo>& h_ops,
    const std::vector<double>& h_loglk,
    const std::vector<double>& h_pendant,
    const std::vector<double>& h_proximal,
    int query_idx,
    int pass,
    int op_idx)
{
    if (op_idx < 0 || op_idx >= (int)h_ops.size() || op_idx >= (int)h_loglk.size()) return;
    const NodeOpInfo& op = h_ops[(size_t)op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int target_id = target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    const double pendant =
        (target_id >= 0 && target_id < (int)h_pendant.size()) ? h_pendant[(size_t)target_id] : 0.0;
    const double proximal =
        (target_id >= 0 && target_id < (int)h_proximal.size()) ? h_proximal[(size_t)target_id] : 0.0;
    std::fprintf(stderr,
        "[op-debug] query=%d pass=%d op=%d target=%d ll=%.12f pendant=%.12f proximal=%.12f dir=%u type=%d parent=%d left=%d right=%d\n",
        query_idx,
        pass,
        op_idx,
        target_id,
        h_loglk[(size_t)op_idx],
        pendant,
        proximal,
        static_cast<unsigned>(op.dir_tag),
        op.op_type,
        op.parent_id,
        op.left_id,
        op.right_id);
}

static void print_selected_op_transition(
    const std::vector<NodeOpInfo>& h_ops,
    const std::vector<double>& h_prev_loglk,
    const std::vector<double>& h_curr_loglk,
    int query_idx,
    int pass,
    int op_idx)
{
    if (op_idx < 0 || op_idx >= (int)h_ops.size()) return;
    const NodeOpInfo& op = h_ops[(size_t)op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int target_id = target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    const double prev = (op_idx < (int)h_prev_loglk.size()) ? h_prev_loglk[(size_t)op_idx] : 0.0;
    const double curr = (op_idx < (int)h_curr_loglk.size()) ? h_curr_loglk[(size_t)op_idx] : 0.0;
    std::fprintf(stderr,
        "[op-transition] query=%d pass=%d op=%d target=%d prev=%.12f curr=%.12f delta=%.12f dir=%u type=%d parent=%d left=%d right=%d\n",
        query_idx,
        pass,
        op_idx,
        target_id,
        prev,
        curr,
        curr - prev,
        static_cast<unsigned>(op.dir_tag),
        op.op_type,
        op.parent_id,
        op.left_id,
        op.right_id);
}

__global__ void BuildOpPendantLengthsKernel(
    const NodeOpInfo* ops,
    const double* node_lengths,
    double* op_lengths,
    int num_ops,
    int total_nodes,
    double min_len,
    double max_len,
    double default_len)
{
    const int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !op_lengths) return;

    double t = default_len;
    if (node_lengths) {
        const NodeOpInfo op = ops[op_idx];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            t = node_lengths[target_id];
        }
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;
    op_lengths[op_idx] = t;
}

__global__ void BuildOpDistalLengthsKernel(
    const NodeOpInfo* ops,
    const double* total_lengths,
    const double* proximal_lengths,
    double* op_lengths,
    int num_ops,
    int total_nodes,
    double min_len,
    double max_len,
    double default_len)
{
    const int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !op_lengths) return;

    double t = default_len;
    if (total_lengths && proximal_lengths) {
        const NodeOpInfo op = ops[op_idx];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            t = total_lengths[target_id] - proximal_lengths[target_id];
        }
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;
    op_lengths[op_idx] = t;
}

__global__ void BuildNodePendantLengthsKernel(
    const double* node_lengths,
    double* out_lengths,
    int total_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < 0 || node_idx >= total_nodes) return;
    if (!out_lengths) return;

    double t = default_len;
    if (node_lengths) {
        t = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = default_len;
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;
    out_lengths[node_idx] = t;
}

__global__ void BuildNodeDistalLengthsKernel(
    const double* total_lengths,
    const double* proximal_lengths,
    double* out_lengths,
    int total_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < 0 || node_idx >= total_nodes) return;
    if (!out_lengths) return;

    double t = default_len;
    if (total_lengths && proximal_lengths) {
        t = total_lengths[node_idx] - proximal_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = default_len;
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;
    out_lengths[node_idx] = t;
}

// Keep per-op best log-likelihood; rollback branch lengths if current pass is worse.
__global__ void KeepBestBranchLengthsKernel(
    const NodeOpInfo* ops,
    double* curr_loglk,
    double* prev_loglk,
    double* curr_pendant,
    double* curr_proximal,
    double* prev_pendant,
    double* prev_proximal,
    int* active_ops,
    int num_ops,
    int total_nodes)
{
    
    const int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !curr_loglk || !prev_loglk ||
        !curr_pendant || !curr_proximal || !prev_pendant || !prev_proximal) return;
    
    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;
    // if(threadIdx.x == 6 && blockIdx.x == 0)
    // printf("Ops = %d target = %d prevloglk = %f currloglk = %f pendant = %f proximal = %f\n", op_idx, target_id, prev_loglk[op_idx], curr_loglk[op_idx], curr_pendant[target_id], curr_proximal[target_id]);
    const double curr = curr_loglk[op_idx];
    const double prev = prev_loglk[op_idx];
    if (curr < prev) {
        curr_loglk[op_idx] = prev;
        curr_pendant[target_id] = prev_pendant[target_id];
        curr_proximal[target_id] = prev_proximal[target_id];
        if (active_ops) active_ops[op_idx] = 0; // TEMP: freeze rejected ops for later passes
    } else {
        prev_loglk[op_idx] = curr;
        prev_pendant[target_id] = curr_pendant[target_id];
        prev_proximal[target_id] = curr_proximal[target_id];
    }
}

// Fused per-op pendant PMAT builder: computes branch length and writes PMAT without staging.
__global__ void BuildPendantPMATPerOpKernel(
    const NodeOpInfo* ops,
    const double* node_lengths,
    const double* Vinv,
    const double* V,
    const double* lambdas,
    double p,
    double* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    double min_len,
    double max_len,
    double default_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_ops * rate_cats;
    if (idx < 0 || idx >= total) return;

    const int op_idx = idx / rate_cats;
    const int rc = idx - op_idx * rate_cats;
    if (!ops || op_idx < 0 || op_idx >= num_ops || rc < 0 || rc >= rate_cats) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;

    double t = default_len;
    if (node_lengths && target_id >= 0 && target_id < total_nodes) {
        t = node_lengths[target_id];
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;

    const double* lamb = lambdas + (size_t)rc * (size_t)states;
    double* out = P + (size_t)idx * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, 1.0, t, p, out, states);
}

// Fused per-node PMAT builder for proximal branches.
__global__ void BuildNodeProximalPMATKernel(
    const double* node_lengths,
    const double* Vinv,
    const double* V,
    const double* lambdas,
    double p,
    double* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_nodes * rate_cats;
    if (idx < 0 || idx >= total) return;

    const int node_idx = idx / rate_cats;
    const int rc = idx - node_idx * rate_cats;
    if (node_idx < 0 || node_idx >= num_nodes || rc < 0 || rc >= rate_cats) return;

    double t = default_len;
    if (node_lengths) {
        t = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = default_len;
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;

    const double* lamb = lambdas + (size_t)rc * (size_t)states;
    double* out = P + ((size_t)node_idx * (size_t)rate_cats + (size_t)rc)
        * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, 1.0, t, p, out, states);
}

// Fused per-node PMAT builder for distal branches (total - proximal).
__global__ void BuildNodeDistalPMATKernel(
    const double* total_lengths,
    const double* proximal_lengths,
    const double* Vinv,
    const double* V,
    const double* lambdas,
    double p,
    double* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_nodes * rate_cats;
    if (idx < 0 || idx >= total) return;

    const int node_idx = idx / rate_cats;
    const int rc = idx - node_idx * rate_cats;
    if (node_idx < 0 || node_idx >= num_nodes || rc < 0 || rc >= rate_cats) return;
    if (!total_lengths || !proximal_lengths) return;

    double t = total_lengths[node_idx] - proximal_lengths[node_idx];
    if (node_idx == root_id) {
        t = default_len;
    }
    if (t < min_len) t = min_len;
    if (t > max_len) t = max_len;

    const double* lamb = lambdas + (size_t)rc * (size_t)states;
    double* out = P + ((size_t)node_idx * (size_t)rate_cats + (size_t)rc)
        * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, 1.0, t, p, out, states);
}

// Per-site placement kernel: build midpoint CLV for placement.
__global__ void BuildMidpointForPlacementKernel(
    DeviceTree D,
    const NodeOpInfo* d_ops,
    int op_offset,
    int num_ops,
    bool proximal_mode)
{
    const int op_idx = op_offset + (int)blockIdx.y;
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_ops || op_idx < 0 || op_idx >= num_ops || tid >= D.sites) return;
    const NodeOpInfo op = d_ops[op_idx];
    // 先在 midpoint 上建立 CLV，再計算對數似然並寫回 midpoint 專用緩衝。
    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                compute_midpoint_inner_inner_ratecat<1>(D, op, tid, proximal_mode, op_idx);
                break;
            case 4:
                compute_midpoint_inner_inner_ratecat<4>(D, op, tid, proximal_mode, op_idx);
                break;
            case 8:
                compute_midpoint_inner_inner_ratecat<8>(D, op, tid, proximal_mode, op_idx);
                break;
            default:
                // Generic version not implemented for midpoint helper.
                break;
        }
    }
}

// Per-site root likelihood kernel: assumes midpoint CLV already computed.
__global__ void ComputeRootLikelihoodKernel(
    DeviceTree D,
    const NodeOpInfo* d_ops,
    int op_idx)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_ops || op_idx < 0 || tid >= D.sites) return;
    const NodeOpInfo op = d_ops[op_idx];
    root_likelihood::compute_root_loglikelihood_at_site(
        D,
        op,
        D.d_frequencies,
        D.d_rate_weights,
        nullptr,  // pattern_w
        nullptr,  // invar_indices
        0.0,      // invar_proportion
        tid);
}


PlacementResult PlacementEvaluationKernel (
    const DeviceTree& D,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    const NodeOpInfo* d_ops,
    int num_ops,
    int smoothing,
    cudaStream_t stream,
    int debug_query_idx
){
    PlacementResult result;
    // Debug: enforce device-pointer sanity before running placement reductions.
    check_device_ptr(d_ops, "d_ops");
    check_device_ptr(D.d_placement_clv, "d_placement_clv");
    assert(num_ops > 0 && "num_ops must be positive");
    if (num_ops <= 0) return result;
    assert(smoothing > 0 && "smoothing must be positive");
    

    const size_t sumtable_stride = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    if ((size_t)num_ops > D.sumtable_capacity_ops || (size_t)num_ops > D.likelihood_capacity_ops) {
        throw std::runtime_error("DeviceTree buffers too small for num_ops.");
    }
    double* d_likelihoods = D.d_likelihoods;
    double* d_sumtable = D.d_sumtable;

    const size_t diag_shared = (size_t)D.rate_cats * (size_t)D.states * 4;
    size_t shmem_bytes = sizeof(double) * diag_shared; // diag table only (df/ddf/loglk reduced via warp shuffle)

    // Choose a block size that fits the device based on occupancy.
    int block_threads = 512;
    int max_blocks_per_sm = 0;
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeKernel));
    while (block_threads >= 32) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            LikelihoodDerivativeKernel,
            block_threads,
            shmem_bytes));
        if (max_blocks_per_sm > 0) break;
        block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativeKernel on this GPU.");
    }

    dim3 placement_block(block_threads);
    dim3 midpoint_grid((D.sites + placement_block.x - 1) / placement_block.x, (unsigned)num_ops);
    dim3 deriv_grid((unsigned)num_ops);

    double* d_prev_loglk = nullptr;
    int* d_active_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev_loglk, sizeof(double) * (size_t)num_ops));
    CUDA_CHECK(cudaMalloc(&d_active_ops, sizeof(int) * (size_t)num_ops));
    CUDA_CHECK(cudaMemset(d_active_ops, 1, sizeof(int) * (size_t)num_ops));
    {
        const size_t total_nodes = (size_t)D.N;
        std::vector<double> h_prev_pendant(total_nodes, DEFAULT_BRANCH_LENGTH);
        std::vector<double> h_prev_proximal(total_nodes, DEFAULT_BRANCH_LENGTH);
        if (D.d_blen && total_nodes > 0) {
            std::vector<double> h_blen(total_nodes, DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaMemcpyAsync(
                h_blen.data(),
                D.d_blen,
                sizeof(double) * total_nodes,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (size_t i = 0; i < total_nodes; ++i) {
                double proximal = (static_cast<int>(i) == D.root_id)
                    ? DEFAULT_BRANCH_LENGTH
                    : 0.5 * h_blen[i];
                if (proximal < OPT_BRANCH_LEN_MIN) proximal = OPT_BRANCH_LEN_MIN;
                if (proximal > OPT_BRANCH_LEN_MAX) proximal = OPT_BRANCH_LEN_MAX;
                h_prev_proximal[i] = proximal;
            }
        } else {
            for (size_t i = 0; i < total_nodes; ++i) {
                double proximal = DEFAULT_BRANCH_LENGTH;
                if (proximal < OPT_BRANCH_LEN_MIN) proximal = OPT_BRANCH_LEN_MIN;
                if (proximal > OPT_BRANCH_LEN_MAX) proximal = OPT_BRANCH_LEN_MAX;
                h_prev_proximal[i] = proximal;
            }
        }
        for (size_t i = 0; i < total_nodes; ++i) {
            double pendant = h_prev_pendant[i];
            if (pendant < OPT_BRANCH_LEN_MIN) pendant = OPT_BRANCH_LEN_MIN;
            if (pendant > OPT_BRANCH_LEN_MAX) pendant = OPT_BRANCH_LEN_MAX;
            h_prev_pendant[i] = pendant;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_prev_pendant_length,
            h_prev_pendant.data(),
            sizeof(double) * total_nodes,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_prev_proximal_length,
            h_prev_proximal.data(),
            sizeof(double) * total_nodes,
            cudaMemcpyHostToDevice,
            stream));
    }

    // Baseline: build PMATs using prev lengths and compute initial per-op loglik.
    {
        dim3 pmat_block(128);
        dim3 pmat_grid((unsigned)((num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildPendantPMATPerOpKernel<<<pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            D.d_prev_pendant_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_query_pmat,
            D.states,
            D.rate_cats,
            num_ops,
            D.N,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        CUDA_CHECK(cudaGetLastError());

        dim3 node_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildNodeProximalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
            D.d_prev_proximal_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_pmat_mid_prox,
            D.states,
            D.rate_cats,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        CUDA_CHECK(cudaGetLastError());

        BuildNodeDistalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
            D.d_blen,
            D.d_prev_proximal_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_pmat_mid_dist,
            D.states,
            D.rate_cats,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        CUDA_CHECK(cudaGetLastError());

        BuildMidpointForPlacementKernel<<<midpoint_grid, placement_block, 0, stream>>>(
            D,
            d_ops,
            0,
            num_ops,
            false);
        CUDA_CHECK(cudaGetLastError());

        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_prev_loglk,
            stream);
    }

    std::vector<NodeOpInfo> h_ops((size_t)num_ops);
    CUDA_CHECK(cudaMemcpyAsync(
        h_ops.data(),
        d_ops,
        sizeof(NodeOpInfo) * (size_t)num_ops,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const bool debug_passes = (debug_query_idx >= 0);
    const int debug_topk = get_env_int_or("MLIPPER_DEBUG_TOPK", 5);
    const int debug_selected_op = get_env_int_or("MLIPPER_DEBUG_OP", -1);
    const int debug_selected_op2 = get_env_int_or("MLIPPER_DEBUG_OP2", -1);
    const int debug_midpoint_score_op = get_env_int_or("MLIPPER_DEBUG_MIDPOINT_SCORE_OP", -1);
    const int debug_midpoint_score_sites = get_env_int_or("MLIPPER_DEBUG_MIDPOINT_SCORE_SITES", 0);
    const int debug_shift_site = get_env_int_or("MLIPPER_DEBUG_SHIFT_SITE", -1);
    const int debug_fixed_op = get_env_int_or("MLIPPER_DEBUG_FIXED_OP", -1);
    const double debug_fixed_pendant = get_env_double_or("MLIPPER_DEBUG_FIXED_PENDANT", -1.0);
    const double debug_fixed_proximal = get_env_double_or("MLIPPER_DEBUG_FIXED_PROXIMAL", -1.0);
    const bool disable_opt = (get_env_int_or("MLIPPER_DISABLE_OPT", 0) != 0);
    const int opt_passes = std::max(0, get_env_int_or("MLIPPER_OPT_PASSES", std::max(4, smoothing)));
    if (debug_fixed_op >= 0 &&
        debug_fixed_op < num_ops &&
        debug_fixed_pendant >= 0.0 &&
        debug_fixed_proximal >= 0.0) {
        const NodeOpInfo fixed_op = h_ops[(size_t)debug_fixed_op];
        const bool target_is_left =
            (fixed_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const bool target_is_right =
            (fixed_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        const int target_id = target_is_left
            ? fixed_op.left_id
            : (target_is_right ? fixed_op.right_id : fixed_op.parent_id);
        if (target_id >= 0 && target_id < D.N) {
            std::vector<double> h_dbg_pendant((size_t)D.N, DEFAULT_BRANCH_LENGTH);
            std::vector<double> h_dbg_proximal((size_t)D.N, DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaMemcpyAsync(
                h_dbg_pendant.data(),
                D.d_prev_pendant_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                h_dbg_proximal.data(),
                D.d_prev_proximal_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            h_dbg_pendant[(size_t)target_id] = debug_fixed_pendant;
            h_dbg_proximal[(size_t)target_id] = debug_fixed_proximal;

            CUDA_CHECK(cudaMemcpyAsync(
                D.d_new_pendant_length,
                h_dbg_pendant.data(),
                sizeof(double) * (size_t)D.N,
                cudaMemcpyHostToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                D.d_new_proximal_length,
                h_dbg_proximal.data(),
                sizeof(double) * (size_t)D.N,
                cudaMemcpyHostToDevice,
                stream));

            dim3 pmat_block(128);
            dim3 pmat_grid((unsigned)((num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
            BuildPendantPMATPerOpKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                d_ops,
                D.d_new_pendant_length,
                D.d_Vinv,
                D.d_V,
                D.d_lambdas,
                0.0,
                D.d_query_pmat,
                D.states,
                D.rate_cats,
                num_ops,
                D.N,
                OPT_BRANCH_LEN_MIN,
                OPT_BRANCH_LEN_MAX,
                DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaGetLastError());

            dim3 node_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
            BuildNodeProximalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
                D.d_new_proximal_length,
                D.d_Vinv,
                D.d_V,
                D.d_lambdas,
                0.0,
                D.d_pmat_mid_prox,
                D.states,
                D.rate_cats,
                D.N,
                D.root_id,
                OPT_BRANCH_LEN_MIN,
                OPT_BRANCH_LEN_MAX,
                DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaGetLastError());

            BuildNodeDistalPMATKernel<<<node_grid, pmat_block, 0, stream>>>(
                D.d_blen,
                D.d_new_proximal_length,
                D.d_Vinv,
                D.d_V,
                D.d_lambdas,
                0.0,
                D.d_pmat_mid_dist,
                D.states,
                D.rate_cats,
                D.N,
                D.root_id,
                OPT_BRANCH_LEN_MIN,
                OPT_BRANCH_LEN_MAX,
                DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaGetLastError());

            root_likelihood::compute_combined_loglik_per_op_device(
                D,
                d_ops,
                num_ops,
                D.d_query_pmat,
                D.d_pmat_mid_dist,
                D.d_pmat_mid_prox,
                d_likelihoods,
                stream);
            std::vector<double> h_fixed_loglk((size_t)num_ops, 0.0);
            CUDA_CHECK(cudaMemcpyAsync(
                h_fixed_loglk.data(),
                d_likelihoods,
                sizeof(double) * (size_t)num_ops,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::fprintf(stderr,
                "[fixed-branch-debug] op=%d target=%d pendant=%.12f proximal=%.12f ll=%.12f\n",
                debug_fixed_op,
                target_id,
                debug_fixed_pendant,
                debug_fixed_proximal,
                h_fixed_loglk[(size_t)debug_fixed_op]);
        }
    }
    if (debug_midpoint_score_op >= 0 && debug_midpoint_score_op < num_ops) {
        const NodeOpInfo debug_op = h_ops[(size_t)debug_midpoint_score_op];
        const bool target_is_left =
            (debug_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const bool target_is_right =
            (debug_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        const int target_id = target_is_left
            ? debug_op.left_id
            : (target_is_right ? debug_op.right_id : debug_op.parent_id);
        DeviceTree Dmid = D;
        if (target_id >= 0 && target_id < D.N && D.d_site_scaler_mid) {
            Dmid.d_site_scaler = D.d_site_scaler_mid + (size_t)target_id * D.scaler_elems();
        } else {
            Dmid.d_site_scaler = nullptr;
        }
        CUDA_CHECK(cudaMemsetAsync(Dmid.d_placement_clv, 0, sizeof(double) * Dmid.sites, stream));
        dim3 dbg_grid((unsigned)((D.sites + placement_block.x - 1) / placement_block.x));
        UpdateMidpointWithQueryKernel<<<dbg_grid, placement_block, 0, stream>>>(
            Dmid,
            d_ops,
            debug_midpoint_score_op);
        CUDA_CHECK(cudaGetLastError());
        ComputeRootLikelihoodKernel<<<dbg_grid, placement_block, 0, stream>>>(
            Dmid,
            d_ops,
            debug_midpoint_score_op);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> h_midpoint_site((size_t)Dmid.sites, 0.0);
        std::vector<double> h_combined((size_t)num_ops, 0.0);
        CUDA_CHECK(cudaMemcpyAsync(
            h_midpoint_site.data(),
            Dmid.d_placement_clv,
            sizeof(double) * (size_t)Dmid.sites,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            h_combined.data(),
            d_prev_loglk,
            sizeof(double) * (size_t)num_ops,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double midpoint_total = 0.0;
        for (double v : h_midpoint_site) midpoint_total += v;
        std::fprintf(stderr,
            "[midpoint-score-debug] op=%d combined=%.12f midpoint_root=%.12f delta=%.12f\n",
            debug_midpoint_score_op,
            h_combined[(size_t)debug_midpoint_score_op],
            midpoint_total,
            midpoint_total - h_combined[(size_t)debug_midpoint_score_op]);
        if (debug_shift_site >= 0 && debug_shift_site < (int)D.sites) {
            if (target_id >= 0 && target_id < D.N && D.per_rate_scaling) {
                const size_t per_node_scalers = D.sites * (size_t)D.rate_cats;
                const size_t site_off = (size_t)debug_shift_site * (size_t)D.rate_cats;
                std::vector<unsigned> h_down((size_t)D.rate_cats, 0u);
                std::vector<unsigned> h_up((size_t)D.rate_cats, 0u);
                std::vector<unsigned> h_mid((size_t)D.rate_cats, 0u);
                CUDA_CHECK(cudaMemcpyAsync(
                    h_down.data(),
                    D.d_site_scaler_down + (size_t)target_id * per_node_scalers + site_off,
                    sizeof(unsigned) * (size_t)D.rate_cats,
                    cudaMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_up.data(),
                    D.d_site_scaler_up + (size_t)target_id * per_node_scalers + site_off,
                    sizeof(unsigned) * (size_t)D.rate_cats,
                    cudaMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    h_mid.data(),
                    D.d_site_scaler_mid + (size_t)target_id * per_node_scalers + site_off,
                    sizeof(unsigned) * (size_t)D.rate_cats,
                    cudaMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::fprintf(stderr,
                    "[midpoint-shift-debug] op=%d target=%d site=%d",
                    debug_midpoint_score_op,
                    target_id,
                    debug_shift_site);
                for (int rc = 0; rc < D.rate_cats; ++rc) {
                    std::fprintf(stderr,
                        " rc%d(down=%u up=%u sum=%u mid=%u)",
                        rc,
                        h_down[(size_t)rc],
                        h_up[(size_t)rc],
                        h_down[(size_t)rc] + h_up[(size_t)rc],
                        h_mid[(size_t)rc]);
                }
                std::fprintf(stderr, "\n");
            }
        }
        if (debug_midpoint_score_sites > 0) {
            const int limit = std::min<int>(debug_midpoint_score_sites, (int)h_midpoint_site.size());
            for (int site_i = 0; site_i < limit; ++site_i) {
                std::fprintf(stderr,
                    "[midpoint-site-debug] op=%d site=%d log=%.12f\n",
                    debug_midpoint_score_op,
                    site_i,
                    h_midpoint_site[(size_t)site_i]);
            }
        }
    }
    if (disable_opt) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_likelihoods,
            d_prev_loglk,
            sizeof(double) * (size_t)num_ops,
            cudaMemcpyDeviceToDevice,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (debug_passes) {
            std::vector<double> h_best_loglk((size_t)num_ops, 0.0);
            std::vector<double> h_prev_pendant((size_t)D.N, 0.0);
            std::vector<double> h_prev_proximal((size_t)D.N, 0.0);
            CUDA_CHECK(cudaMemcpy(
                h_best_loglk.data(),
                d_prev_loglk,
                sizeof(double) * (size_t)num_ops,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_prev_pendant.data(),
                D.d_prev_pendant_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_prev_proximal.data(),
                D.d_prev_proximal_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost));
            print_top_ops_summary(
                h_ops,
                h_best_loglk,
                h_prev_pendant,
                h_prev_proximal,
                debug_query_idx,
                0,
                debug_topk);
            if (debug_selected_op >= 0) {
                print_selected_op_summary(
                    h_ops,
                    h_best_loglk,
                    h_prev_pendant,
                    h_prev_proximal,
                    debug_query_idx,
                    0,
                    debug_selected_op);
            }
            if (debug_selected_op2 >= 0) {
                print_selected_op_summary(
                    h_ops,
                    h_best_loglk,
                    h_prev_pendant,
                    h_prev_proximal,
                    debug_query_idx,
                    0,
                    debug_selected_op2);
            }
        }
    }
    for (int pass = 0; !disable_opt && pass < opt_passes; ++pass) {
        // Reset shared workspaces before each pendant pass.

        // Pendant branch: build midpoints for all placements then optimize per block.
        BuildMidpointForPlacementKernel<<<midpoint_grid, placement_block, 0, stream>>>(
            D,
            d_ops,
            0,
            num_ops,
            false);
        CUDA_CHECK(cudaGetLastError());

        // Use previous pass's pendant lengths as the initial guess after pass 0.
        

        LikelihoodDerivativeKernel<<<deriv_grid, placement_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            nullptr,
            30,
            D.d_new_pendant_length,
            false,
            sumtable_stride,
            nullptr,
            D.d_prev_pendant_length,
            d_active_ops);
        CUDA_CHECK(cudaGetLastError());

        // Derivative kernel writes updated pendant lengths into D.d_new_pendant_length.
        // Update query PMAT with per-op pendant lengths on GPU (buffer allocated during upload).
        dim3 pmat_block(128);
        dim3 pmat_grid((unsigned)((num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildPendantPMATPerOpKernel<<<pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            D.d_new_pendant_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_query_pmat,
            D.states,
            D.rate_cats,
            num_ops,
            D.N,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        CUDA_CHECK(cudaGetLastError());
        // Ensure all pendant work is finished before proceeding.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        BuildMidpointForPlacementKernel<<<midpoint_grid, placement_block, 0, stream>>>(
            D,
            d_ops,
            0,
            num_ops,
            true);
        CUDA_CHECK(cudaGetLastError());

        // Use previous pass's proximal lengths as the initial guess after pass 0.
        

        LikelihoodDerivativeKernel<<<deriv_grid, placement_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            nullptr,
            30,
            D.d_new_proximal_length,
            true,
            sumtable_stride,
            nullptr,
            D.d_prev_proximal_length,
            d_active_ops);
        CUDA_CHECK(cudaGetLastError());

        // Derivative kernel writes updated proximal lengths into D.d_new_proximal_length.
        
        // Ensure proximal computations are finished before proceeding.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Proximal branch PMATs
        {
            dim3 pmat_block(128);
            dim3 pmat_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
            BuildNodeProximalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                D.d_new_proximal_length,
                D.d_Vinv,
                D.d_V,
                D.d_lambdas,
                0.0,
                D.d_pmat_mid_prox,
                D.states,
                D.rate_cats,
                D.N,
                D.root_id,
                OPT_BRANCH_LEN_MIN,
                OPT_BRANCH_LEN_MAX,
                DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaGetLastError());
        }

        // Distal branch PMATs (total length from D.d_blen)
        {
            BuildNodeDistalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                D.d_blen,
                D.d_new_proximal_length,
                D.d_Vinv,
                D.d_V,
                D.d_lambdas,
                0.0,
                D.d_pmat_mid_dist,
                D.states,
                D.rate_cats,
                D.N,
                D.root_id,
                OPT_BRANCH_LEN_MIN,
                OPT_BRANCH_LEN_MAX,
                DEFAULT_BRANCH_LENGTH);
            CUDA_CHECK(cudaGetLastError());
        }
        // Combined likelihood per placement op (parallel across ops) for this pass.
        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_likelihoods,
            stream);

        if (debug_passes && (debug_selected_op >= 0 || debug_selected_op2 >= 0)) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<double> h_curr_loglk((size_t)num_ops, 0.0);
            std::vector<double> h_prev_before_keep((size_t)num_ops, 0.0);
            CUDA_CHECK(cudaMemcpy(
                h_curr_loglk.data(),
                d_likelihoods,
                sizeof(double) * (size_t)num_ops,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_prev_before_keep.data(),
                d_prev_loglk,
                sizeof(double) * (size_t)num_ops,
                cudaMemcpyDeviceToHost));
            if (debug_selected_op >= 0) {
                print_selected_op_transition(
                    h_ops,
                    h_prev_before_keep,
                    h_curr_loglk,
                    debug_query_idx,
                    pass + 1,
                    debug_selected_op);
            }
            if (debug_selected_op2 >= 0) {
                print_selected_op_transition(
                    h_ops,
                    h_prev_before_keep,
                    h_curr_loglk,
                    debug_query_idx,
                    pass + 1,
                    debug_selected_op2);
            }
        }

        dim3 keep_block(256);
        dim3 keep_grid((unsigned)((num_ops + keep_block.x - 1) / keep_block.x));
        KeepBestBranchLengthsKernel<<<keep_grid, keep_block, 0, stream>>>(
            d_ops,
            d_likelihoods,
            d_prev_loglk,
            D.d_new_pendant_length,
            D.d_new_proximal_length,
            D.d_prev_pendant_length,
            D.d_prev_proximal_length,
            d_active_ops,
            num_ops,
            D.N);
        CUDA_CHECK(cudaGetLastError());

        if (debug_passes) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<double> h_best_loglk((size_t)num_ops, 0.0);
            std::vector<int> h_active((size_t)num_ops, 0);
            std::vector<double> h_prev_pendant((size_t)D.N, 0.0);
            std::vector<double> h_prev_proximal((size_t)D.N, 0.0);
            CUDA_CHECK(cudaMemcpy(
                h_best_loglk.data(),
                d_prev_loglk,
                sizeof(double) * (size_t)num_ops,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_active.data(),
                d_active_ops,
                sizeof(int) * (size_t)num_ops,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_prev_pendant.data(),
                D.d_prev_pendant_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                h_prev_proximal.data(),
                D.d_prev_proximal_length,
                sizeof(double) * (size_t)D.N,
                cudaMemcpyDeviceToHost));

            int active_count = 0;
            int best_op_idx = -1;
            int best_target_id = -1;
            double best_ll = -std::numeric_limits<double>::infinity();
            double best_pendant = 0.0;
            double best_proximal = 0.0;
            for (int op_i = 0; op_i < num_ops; ++op_i) {
                if (h_active[(size_t)op_i] != 0) ++active_count;
                const double ll = h_best_loglk[(size_t)op_i];
                if (ll > best_ll) {
                    best_ll = ll;
                    best_op_idx = op_i;
                    const NodeOpInfo& op = h_ops[(size_t)op_i];
                    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
                    best_target_id = target_is_left ? op.left_id : op.right_id;
                    if (best_target_id >= 0 && best_target_id < D.N) {
                        best_pendant = h_prev_pendant[(size_t)best_target_id];
                        best_proximal = h_prev_proximal[(size_t)best_target_id];
                    }
                }
            }
            std::printf(
                "[pass-debug] query=%d batch_ops=%d pass=%d active_ops=%d best_op=%d target=%d best_ll=%.12f pendant=%.12f proximal=%.12f\n",
                debug_query_idx,
                num_ops,
                pass + 1,
                active_count,
                best_op_idx,
                best_target_id,
                best_ll,
                best_pendant,
                best_proximal);
            print_top_ops_summary(
                h_ops,
                h_best_loglk,
                h_prev_pendant,
                h_prev_proximal,
                debug_query_idx,
                pass + 1,
                debug_topk);
            if (debug_selected_op >= 0) {
                print_selected_op_summary(
                    h_ops,
                    h_best_loglk,
                    h_prev_pendant,
                    h_prev_proximal,
                    debug_query_idx,
                    pass + 1,
                    debug_selected_op);
            }
        }
    }
    
    // Argmax on device to get best op index.
    using Pair = cub::KeyValuePair<int, double>;
    Pair* d_best = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(Pair)));
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::ArgMax(
        d_temp,
        temp_bytes,
        d_likelihoods,
        d_best,
        num_ops,
        stream));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    CUDA_CHECK(cub::DeviceReduce::ArgMax(
        d_temp,
        temp_bytes,
        d_likelihoods,
        d_best,
        num_ops,
        stream));
    Pair h_best{};
    CUDA_CHECK(cudaMemcpyAsync(&h_best, d_best, sizeof(Pair), cudaMemcpyDeviceToHost, stream));

    NodeOpInfo h_op{};
    CUDA_CHECK(cudaMemcpyAsync(
        &h_op,
        d_ops + h_best.key,
        sizeof(NodeOpInfo),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const bool target_is_left = (h_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (h_op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int op_target_id = target_is_left ? h_op.left_id
                             : (target_is_right ? h_op.right_id : h_op.parent_id);

    double pendant_length = 0.0;
    double proximal_length = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(
        &pendant_length,
        D.d_prev_pendant_length + op_target_id,
        sizeof(double),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &proximal_length,
        D.d_prev_proximal_length + op_target_id,
        sizeof(double),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_best));
    CUDA_CHECK(cudaFree(d_prev_loglk));
    CUDA_CHECK(cudaFree(d_active_ops));
    result.target_id = op_target_id;
    result.loglikelihood = h_best.value;
    result.proximal_length = proximal_length;
    result.pendant_length = pendant_length;
    return result;
}

PlacementResult PlacementEvaluationKernelPreorderPruned(
    const DeviceTree& D,
    const TreeBuildResult& T,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    const NodeOpInfo* d_ops,
    int num_ops,
    int smoothing,
    const PlacementPruneConfig& prune_cfg,
    cudaStream_t stream,
    int pseudo_root_id,
    int debug_query_idx)
{
    if (!prune_cfg.enable_pruning || T.root_id < 0 || T.preorder.empty()) {
        return PlacementEvaluationKernel(
            D,
            er,
            rate_multipliers,
            d_ops,
            num_ops,
            smoothing,
            stream,
            debug_query_idx);
    }
    if (!d_ops || num_ops <= 0) {
        return PlacementResult{};
    }

    std::vector<NodeOpInfo> host_ops((size_t)num_ops);
    CUDA_CHECK(cudaMemcpyAsync(
        host_ops.data(),
        d_ops,
        sizeof(NodeOpInfo) * (size_t)num_ops,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int num_parents = std::max(0, D.N);
    std::vector<int> parent_counts((size_t)num_parents, 0);
    for (int i = 0; i < num_ops; ++i) {
        const int pid = host_ops[(size_t)i].parent_id;
        if (pid >= 0 && pid < num_parents) {
            ++parent_counts[(size_t)pid];
        }
    }
    std::vector<int> parent_offsets((size_t)num_parents + 1, 0);
    for (int p = 0; p < num_parents; ++p) {
        parent_offsets[(size_t)p + 1] = parent_offsets[(size_t)p] + parent_counts[(size_t)p];
    }
    std::vector<int> parent_cursor = parent_offsets;
    std::vector<int> child_op_indices((size_t)num_ops, -1);
    for (int i = 0; i < num_ops; ++i) {
        const int pid = host_ops[(size_t)i].parent_id;
        if (pid >= 0 && pid < num_parents) {
            const int pos = parent_cursor[(size_t)pid]++;
            child_op_indices[(size_t)pos] = i;
        }
    }

    NodeOpInfo* d_batch_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_batch_ops, sizeof(NodeOpInfo) * (size_t)num_ops));

    auto target_id_of = [](const NodeOpInfo& op) -> int {
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        return target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    };

    struct SearchState {
        int op_idx = -1;
        double parent_ll = -std::numeric_limits<double>::infinity();
        int drop_streak = 0;
    };

    std::vector<SearchState> frontier;
    int start_node_id = T.root_id;
    if (pseudo_root_id >= 0 &&
        pseudo_root_id < (int)T.nodes.size() &&
        pseudo_root_id < D.N &&
        !T.nodes[(size_t)pseudo_root_id].is_tip) {
        start_node_id = pseudo_root_id;
    }
    if (start_node_id >= 0 && start_node_id < D.N) {
        const int start_begin = parent_offsets[(size_t)start_node_id];
        const int start_end = parent_offsets[(size_t)start_node_id + 1];
        frontier.reserve((size_t)std::max(0, start_end - start_begin));
        for (int pos = start_begin; pos < start_end; ++pos) {
            frontier.push_back(SearchState{
                child_op_indices[(size_t)pos],
                -std::numeric_limits<double>::infinity(),
                0});
        }
    }

    PlacementResult best{};
    best.loglikelihood = -std::numeric_limits<double>::infinity();
    const int max_consecutive_drops = std::max(1, prune_cfg.max_consecutive_drops);
    std::vector<SearchState> next_frontier;
    std::vector<SearchState> eval_states;
    std::vector<NodeOpInfo> batch_ops;
    std::vector<double> batch_ll;
    eval_states.reserve((size_t)num_ops);
    batch_ops.reserve((size_t)num_ops);
    batch_ll.reserve((size_t)num_ops);
    next_frontier.reserve((size_t)num_ops);

    while (!frontier.empty()) {        
        eval_states.clear();
        batch_ops.clear();
        for (size_t i = 0; i < frontier.size(); ++i) {
            const SearchState s = frontier[i];
            const int op_idx = s.op_idx;
            if (op_idx < 0 || op_idx >= num_ops) continue;
            eval_states.push_back(s);
            batch_ops.push_back(host_ops[(size_t)op_idx]);
        }
        const int batch_n = (int)eval_states.size();
        if (batch_n <= 0) break;
        if (prune_cfg.enable_small_frontier_fallback &&
            prune_cfg.small_frontier_threshold > 0 &&
            batch_n < prune_cfg.small_frontier_threshold) {
            CUDA_CHECK(cudaFree(d_batch_ops));
            return PlacementEvaluationKernel(
                D,
                er,
                rate_multipliers,
                d_ops,
                num_ops,
                smoothing,
                stream,
                debug_query_idx);
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_batch_ops,
            batch_ops.data(),
            sizeof(NodeOpInfo) * (size_t)batch_n,
            cudaMemcpyHostToDevice,
            stream));

        PlacementEvaluationKernel(
            D,
            er,
            rate_multipliers,
            d_batch_ops,
            batch_n,
            smoothing,
            stream,
            debug_query_idx);

        batch_ll.resize((size_t)batch_n);
        CUDA_CHECK(cudaMemcpyAsync(
            batch_ll.data(),
            D.d_likelihoods,
            sizeof(double) * (size_t)batch_n,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        next_frontier.clear();
        next_frontier.reserve((size_t)batch_n * 2);

        for (int batch_i = 0; batch_i < batch_n; ++batch_i) {
            const SearchState cur_state = eval_states[(size_t)batch_i];
            const NodeOpInfo& op = host_ops[(size_t)cur_state.op_idx];
            const int target_id = target_id_of(op);
            const double cur_ll = batch_ll[(size_t)batch_i];

            if (target_id >= 0 && target_id < D.N && cur_ll > best.loglikelihood) {
                best.target_id = target_id;
                best.loglikelihood = cur_ll;
            }

            int streak = 0;
            if (cur_state.parent_ll > -std::numeric_limits<double>::infinity() && cur_ll < cur_state.parent_ll) {
                streak = cur_state.drop_streak + 1;
            }
            const bool below_best_threshold = (cur_ll < (best.loglikelihood - prune_cfg.drop_threshold));
            const bool should_prune = (streak >= max_consecutive_drops) && below_best_threshold;
            if (should_prune) continue;

            if (target_id < 0 || target_id >= (int)T.nodes.size()) continue;
            if (T.nodes[(size_t)target_id].is_tip) continue;
            if (target_id < 0 || target_id >= D.N) continue;

            const int child_begin = parent_offsets[(size_t)target_id];
            const int child_end = parent_offsets[(size_t)target_id + 1];
            for (int pos = child_begin; pos < child_end; ++pos) {
                next_frontier.push_back(SearchState{child_op_indices[(size_t)pos], cur_ll, streak});
            }
        }
        frontier.swap(next_frontier);
    }

    CUDA_CHECK(cudaFree(d_batch_ops));
    if (best.loglikelihood == -std::numeric_limits<double>::infinity()) {
        return PlacementEvaluationKernel(
            D,
            er,
            rate_multipliers,
            d_ops,
            num_ops,
            smoothing,
            stream,
            debug_query_idx);
    }
    if (best.target_id >= 0 && best.target_id < D.N) {
        CUDA_CHECK(cudaMemcpyAsync(
            &best.pendant_length,
            D.d_prev_pendant_length + best.target_id,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &best.proximal_length,
            D.d_prev_proximal_length + best.target_id,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return best;
}
