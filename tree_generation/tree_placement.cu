#include <vector>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "tree_placement.cuh"
#include "../mlipper_util.h"
#include "../pmatrix/pmat.h"
#include "../pmatrix/pmat_gpu.cuh"
#include "tree.hpp"
#include "../partial_CUDA/partial_likelihood.cuh"
#include "root_likelihood.cuh"
#include "derivative.cuh"


namespace {
constexpr int kDefaultFullOptPasses = 4;
constexpr int kDefaultRefineGlobalPasses = 0;
constexpr int kDefaultRefineExtraPasses = 0;
constexpr int kDefaultDetectTopK = 0;
constexpr int kDefaultRefineTopK = 0;
constexpr double kRefineGapTop2 = 0.25;
constexpr double kRefineGapTop5 = 1.0;
constexpr double kRefineConvergedLoglkEps = 1e-2;
constexpr double kRefineConvergedLengthEps = 1e-4;

struct RefineConfig {
    int full_opt_passes = kDefaultFullOptPasses;
    int global_opt_passes = kDefaultRefineGlobalPasses;
    int refine_extra_passes = kDefaultRefineExtraPasses;
    int detect_topk_limit = kDefaultDetectTopK;
    int refine_topk_limit = kDefaultRefineTopK;
    double gap_top2 = kRefineGapTop2;
    double gap_top5 = kRefineGapTop5;
    double converged_loglk_eps = kRefineConvergedLoglkEps;
    double converged_length_eps = kRefineConvergedLengthEps;
};

static int getenv_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::max(0, std::atoi(value));
}

static double getenv_double_or_default(const char* name, double default_value) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::atof(value);
}

static RefineConfig load_refine_config() {
    RefineConfig cfg;
    cfg.full_opt_passes =
        getenv_int_or_default("MLIPPER_FULL_OPT_PASSES", cfg.full_opt_passes);
    cfg.global_opt_passes =
        getenv_int_or_default("MLIPPER_REFINE_GLOBAL_PASSES", cfg.global_opt_passes);
    cfg.refine_extra_passes =
        getenv_int_or_default("MLIPPER_REFINE_EXTRA_PASSES", cfg.refine_extra_passes);
    cfg.detect_topk_limit =
        getenv_int_or_default("MLIPPER_REFINE_DETECT_TOPK", cfg.detect_topk_limit);
    cfg.refine_topk_limit =
        getenv_int_or_default("MLIPPER_REFINE_TOPK", cfg.refine_topk_limit);
    cfg.gap_top2 =
        getenv_double_or_default("MLIPPER_REFINE_GAP_TOP2", cfg.gap_top2);
    cfg.gap_top5 =
        getenv_double_or_default("MLIPPER_REFINE_GAP_TOP5", cfg.gap_top5);
    cfg.converged_loglk_eps =
        getenv_double_or_default("MLIPPER_REFINE_CONVERGED_LOGLK_EPS", cfg.converged_loglk_eps);
    cfg.converged_length_eps =
        getenv_double_or_default("MLIPPER_REFINE_CONVERGED_LENGTH_EPS", cfg.converged_length_eps);
    return cfg;
}
}

__global__ void BuildOpPendantLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* node_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        const NodeOpInfo op = ops[op_local];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            branch_length = node_lengths[target_id];
        }
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    op_lengths[op_local] = branch_length;
}

__global__ void BuildOpDistalLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t branch_length = default_len;
    if (total_lengths && proximal_lengths) {
        const NodeOpInfo op = ops[op_local];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            branch_length = total_lengths[target_id] - proximal_lengths[target_id];
        }
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    op_lengths[op_local] = branch_length;
}

__global__ void BuildNodePendantLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

__global__ void BuildInitialProximalLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = static_cast<fp_t>(0.5) * node_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

__global__ void FillSequentialIndicesKernel(
    int* out_indices,
    int count)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_index >= count) return;
    if (!out_indices) return;
    out_indices[flat_index] = flat_index;
}

__global__ void BuildNodeDistalLengthsKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= total_nodes) return;
    if (!out_lengths) return;

    fp_t branch_length = default_len;
    if (total_lengths && proximal_lengths) {
        branch_length = total_lengths[node_id] - proximal_lengths[node_id];
    }
    if (node_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;
    out_lengths[node_id] = branch_length;
}

// Keep per-op best log-likelihood; rollback branch lengths if current pass is worse.
__global__ void KeepBestBranchLengthsKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    fp_t* curr_loglk,
    fp_t* prev_loglk,
    fp_t* curr_pendant,
    fp_t* curr_proximal,
    fp_t* prev_pendant,
    fp_t* prev_proximal,
    int* active_ops,
    int num_ops,
    int total_nodes)
{
    
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !curr_loglk || !prev_loglk ||
        !curr_pendant || !curr_proximal || !prev_pendant || !prev_proximal) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;
    const fp_t curr = curr_loglk[op_idx];
    const fp_t prev = prev_loglk[op_idx];
    if (curr < prev) {
        curr_loglk[op_idx] = prev;
        curr_pendant[target_id] = prev_pendant[target_id];
        curr_proximal[target_id] = prev_proximal[target_id];
        if (active_ops) active_ops[op_local] = 0;
    } else {
        prev_loglk[op_idx] = curr;
        prev_pendant[target_id] = curr_pendant[target_id];
        prev_proximal[target_id] = curr_proximal[target_id];
    }
}

__global__ void UpdateActiveOpsByConvergenceKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* curr_loglk,
    const fp_t* prev_loglk,
    const fp_t* curr_pendant,
    const fp_t* prev_pendant,
    const fp_t* curr_proximal,
    const fp_t* prev_proximal,
    int* active_ops,
    int num_ops,
    int total_nodes,
    double loglk_eps,
    double length_eps)
{
    const int op_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_local >= num_ops) return;
    if (!ops || !curr_loglk || !prev_loglk ||
        !curr_pendant || !prev_pendant || !curr_proximal || !prev_proximal || !active_ops) {
        return;
    }
    if (active_ops[op_local] == 0) return;

    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;
    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;

    const double d_ll = fabs(static_cast<double>(curr_loglk[op_idx]) - static_cast<double>(prev_loglk[op_idx]));
    const double d_pendant = fabs(static_cast<double>(curr_pendant[target_id]) - static_cast<double>(prev_pendant[target_id]));
    const double d_proximal = fabs(static_cast<double>(curr_proximal[target_id]) - static_cast<double>(prev_proximal[target_id]));

    if (d_ll < loglk_eps && d_pendant < length_eps && d_proximal < length_eps) {
        active_ops[op_local] = 0;
    }
}

// Build per-placement pendant PMATs directly from the target branch lengths.
__global__ void BuildPendantPMATPerOpKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;

    fp_t branch_length = default_len;
    if (node_lengths && target_id >= 0 && target_id < total_nodes) {
        branch_length = node_lengths[target_id];
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + static_cast<size_t>(flat_index) * matrix_span;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Build proximal PMATs for every node from the current proximal branch lengths.
__global__ void BuildNodeProximalPMATKernel(
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_nodes * rate_cats;
    if (flat_index >= total_entries) return;

    const int node_idx = flat_index / rate_cats;
    const int rate_idx = flat_index - node_idx * rate_cats;
    if (node_idx >= num_nodes || rate_idx >= rate_cats) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(node_idx) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Build distal PMATs for every node from total branch length minus proximal length.
__global__ void BuildNodeDistalPMATKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_nodes * rate_cats;
    if (flat_index >= total_entries) return;

    const int node_idx = flat_index / rate_cats;
    const int rate_idx = flat_index - node_idx * rate_cats;
    if (node_idx >= num_nodes || rate_idx >= rate_cats) return;
    if (!total_lengths || !proximal_lengths) return;

    fp_t branch_length = total_lengths[node_idx] - proximal_lengths[node_idx];
    if (node_idx == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(node_idx) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Refresh proximal PMATs only for the currently selected placement targets.
__global__ void BuildSelectedNodeProximalPMATKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;

    fp_t branch_length = default_len;
    if (node_lengths) {
        branch_length = node_lengths[target_id];
    }
    if (target_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(target_id) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Refresh distal PMATs only for the currently selected placement targets.
__global__ void BuildSelectedNodeDistalPMATKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    fp_t p,
    fp_t* P,
    int states,
    int rate_cats,
    int num_ops,
    int total_nodes,
    int root_id,
    fp_t min_len,
    fp_t max_len,
    fp_t default_len)
{
    const int flat_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = num_ops * rate_cats;
    if (flat_index >= total_entries) return;

    const int op_local = flat_index / rate_cats;
    const int rate_idx = flat_index - op_local * rate_cats;
    if (!ops || op_local >= num_ops || rate_idx >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= total_nodes) return;
    if (!total_lengths || !proximal_lengths) return;

    fp_t branch_length = total_lengths[target_id] - proximal_lengths[target_id];
    if (target_id == root_id) {
        branch_length = default_len;
    }
    if (branch_length < min_len) branch_length = min_len;
    if (branch_length > max_len) branch_length = max_len;

    const size_t state_count = static_cast<size_t>(states);
    const size_t rate_count = static_cast<size_t>(rate_cats);
    const size_t rate_offset = static_cast<size_t>(rate_idx) * state_count;
    const size_t matrix_span = state_count * state_count;
    const size_t output_base =
        (static_cast<size_t>(target_id) * rate_count + static_cast<size_t>(rate_idx)) * matrix_span;
    const fp_t* rate_lambdas = lambdas + rate_offset;
    fp_t* out_pmat = P + output_base;
    pmatrix_from_triple_device(Vinv, V, rate_lambdas, fp_t(1.0), branch_length, p, out_pmat, states);
}

// Per-site placement kernel: build midpoint CLV for placement.
__global__ void BuildMidpointForPlacementKernel(
    DeviceTree D,
    const NodeOpInfo* d_ops,
    const int* d_op_indices,
    int op_offset,
    int num_ops,
    bool proximal_mode)
{
    const int op_local = op_offset + (int)blockIdx.y;
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_ops || op_local >= num_ops) return;
    const int op_idx = d_op_indices ? d_op_indices[op_local] : op_local;
    if (op_idx < 0) return;
    const NodeOpInfo op = d_ops[op_idx];
    const bool active_thread = (tid < D.sites);
    __shared__ fp_t shared_target_mat[8 * 16];
    __shared__ fp_t shared_parent_mat[8 * 16];
    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                compute_midpoint_inner_inner_ratecat<1>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
                break;
            case 4:
                compute_midpoint_inner_inner_ratecat<4>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
                break;
            case 8:
                compute_midpoint_inner_inner_ratecat<8>(
                    D,
                    op,
                    tid,
                    proximal_mode,
                    op_idx,
                    active_thread,
                    shared_target_mat,
                    shared_parent_mat);
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
        D.d_pattern_weights_u,
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
    cudaStream_t stream
){
    PlacementResult result;
    assert(num_ops > 0 && "num_ops must be positive");
    if (num_ops <= 0) return result;
    assert(smoothing > 0 && "smoothing must be positive");
    

    const size_t sumtable_stride = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    if ((size_t)num_ops > D.sumtable_capacity_ops || (size_t)num_ops > D.likelihood_capacity_ops) {
        throw std::runtime_error("DeviceTree buffers too small for num_ops.");
    }
    auto check_launch = [&](const char* stage) {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
        }
    };
    fp_t* d_likelihoods = D.d_likelihoods;
    fp_t* d_sumtable = D.d_sumtable;

    const size_t diag_shared = (size_t)D.rate_cats * (size_t)D.states * 4;
    const RefineConfig refine_cfg = load_refine_config();
    const bool use_selective_refine =
        refine_cfg.refine_extra_passes > 0 &&
        refine_cfg.detect_topk_limit > 0 &&
        refine_cfg.refine_topk_limit > 0 &&
        refine_cfg.global_opt_passes > 0;
    const size_t midpoint_pmat_shared = (size_t)D.rate_cats * 16 * 2;
    size_t shmem_bytes = sizeof(fp_t) * diag_shared;
    shmem_bytes += sizeof(fp_t) * midpoint_pmat_shared;

    // Pendant and proximal derivative kernels have different register pressure.
    // Size them independently so one kernel does not inherit an invalid launch shape.
    int pendant_block_threads = 512;
    int max_blocks_per_sm = 0;
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativePendantKernel));
    while (pendant_block_threads >= 32) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            LikelihoodDerivativePendantKernel,
            pendant_block_threads,
            shmem_bytes));
        if (max_blocks_per_sm > 0) break;
        pendant_block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativePendantKernel on this GPU.");
    }

    int proximal_block_threads = 512;
    max_blocks_per_sm = 0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeProximalKernel));
    while (proximal_block_threads >= 32) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            LikelihoodDerivativeProximalKernel,
            proximal_block_threads,
            shmem_bytes));
        if (max_blocks_per_sm > 0) break;
        proximal_block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativeProximalKernel on this GPU.");
    }

    const int midpoint_block_threads = 256;
    const int pmat_block_threads = 128;
    dim3 pendant_block(pendant_block_threads);
    dim3 proximal_block(proximal_block_threads);
    dim3 midpoint_block(midpoint_block_threads);
    dim3 pmat_block(pmat_block_threads);
    dim3 midpoint_grid((D.sites + midpoint_block.x - 1) / midpoint_block.x, (unsigned)num_ops);
    dim3 deriv_grid((unsigned)num_ops);

    fp_t* d_prev_loglk = nullptr;
    fp_t* d_last_loglk = nullptr;
    int* d_active_ops = nullptr;
    int* d_refine_op_indices = nullptr;
    int* d_refine_active_ops = nullptr;
    int* d_any_active_flag = nullptr;
    void* d_any_active_temp = nullptr;
    size_t any_active_temp_bytes = 0;
    CUDA_CHECK(cudaMalloc(&d_prev_loglk, sizeof(fp_t) * (size_t)num_ops));
    CUDA_CHECK(cudaMalloc(&d_last_loglk, sizeof(fp_t) * (size_t)num_ops));
    CUDA_CHECK(cudaMalloc(&d_active_ops, sizeof(int) * (size_t)num_ops));
    CUDA_CHECK(cudaMemset(d_active_ops, 1, sizeof(int) * (size_t)num_ops));
    {
        const size_t total_nodes = (size_t)D.N;
        dim3 init_block(256);
        dim3 init_grid((unsigned)((total_nodes + init_block.x - 1) / init_block.x));
        BuildNodePendantLengthsKernel<<<init_grid, init_block, 0, stream>>>(
            nullptr,
            D.d_prev_pendant_length,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildNodePendantLengthsKernel");
        BuildInitialProximalLengthsKernel<<<init_grid, init_block, 0, stream>>>(
            D.d_blen,
            D.d_prev_proximal_length,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildInitialProximalLengthsKernel");
    }

    fp_t* d_last_pendant_length = nullptr;
    fp_t* d_last_proximal_length = nullptr;
    std::vector<fp_t> host_loglk_cache;
    std::vector<int> host_order_cache;
    CUDA_CHECK(cudaMalloc(&d_last_pendant_length, sizeof(fp_t) * (size_t)D.N));
    CUDA_CHECK(cudaMalloc(&d_last_proximal_length, sizeof(fp_t) * (size_t)D.N));
    if (use_selective_refine) {
        CUDA_CHECK(cudaMalloc(&d_any_active_flag, sizeof(int)));
        CUDA_CHECK(cub::DeviceReduce::Max(
            d_any_active_temp,
            any_active_temp_bytes,
            d_active_ops,
            d_any_active_flag,
            num_ops,
            stream));
        CUDA_CHECK(cudaMalloc(&d_any_active_temp, any_active_temp_bytes));
    }

    auto fetch_topk_loglk =
        [&](int topk, std::vector<int>& top_indices, std::vector<fp_t>& top_values) {
            if (topk <= 0) return;
            host_loglk_cache.resize((size_t)num_ops);
            CUDA_CHECK(cudaMemcpyAsync(
                host_loglk_cache.data(),
                d_prev_loglk,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const int actual_topk = std::min(num_ops, topk);
            host_order_cache.resize((size_t)num_ops);
            std::iota(host_order_cache.begin(), host_order_cache.end(), 0);
            std::partial_sort(
                host_order_cache.begin(),
                host_order_cache.begin() + actual_topk,
                host_order_cache.end(),
                [&](int lhs, int rhs) {
                    return host_loglk_cache[(size_t)lhs] > host_loglk_cache[(size_t)rhs];
                });
            top_indices.resize((size_t)actual_topk);
            top_values.resize((size_t)actual_topk);
            for (int i = 0; i < actual_topk; ++i) {
                const int op_idx = host_order_cache[(size_t)i];
                top_indices[(size_t)i] = op_idx;
                top_values[(size_t)i] = host_loglk_cache[(size_t)op_idx];
            }
        };
    auto any_active_on_device =
        [&](const int* d_active_mask, int count) {
            if (!d_active_mask || count <= 0) return false;
            int h_any_active = 0;
            CUDA_CHECK(cub::DeviceReduce::Max(
                d_any_active_temp,
                any_active_temp_bytes,
                d_active_mask,
                d_any_active_flag,
                count,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                &h_any_active,
                d_any_active_flag,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return h_any_active != 0;
        };

    // Baseline: build PMATs using prev lengths and compute initial per-op loglik.
    {
        dim3 pmat_grid((unsigned)((num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        BuildPendantPMATPerOpKernel<<<pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            nullptr,
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
        check_launch("BuildPendantPMATPerOpKernel baseline");

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
        check_launch("BuildNodeProximalPMATKernel baseline");

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
        check_launch("BuildNodeDistalPMATKernel baseline");

        BuildMidpointForPlacementKernel<<<midpoint_grid, midpoint_block, 0, stream>>>(
            D,
            d_ops,
            nullptr,
            0,
            num_ops,
            false);
        check_launch("BuildMidpointForPlacementKernel baseline");

        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            nullptr,
            num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_prev_loglk,
            stream);
    }

    const int opt_passes = use_selective_refine
        ? std::max(refine_cfg.global_opt_passes + refine_cfg.refine_extra_passes, smoothing)
        : std::max(refine_cfg.full_opt_passes, smoothing);
    bool restrict_to_refine_topk = false;
    int best_op_after_global_pass1 = -1;
    int refine_op_count = 0;
    for (int pass = 0; pass < opt_passes; ++pass) {
        if (use_selective_refine && pass >= refine_cfg.global_opt_passes && !restrict_to_refine_topk) {
            break;
        }
        const bool use_compact_refine =
            use_selective_refine && restrict_to_refine_topk && refine_op_count > 0;
        const int current_num_ops = use_compact_refine ? refine_op_count : num_ops;
        const int* current_op_indices = use_compact_refine ? d_refine_op_indices : nullptr;
        int* current_active_ops = use_compact_refine ? d_refine_active_ops : d_active_ops;
        dim3 current_midpoint_grid((D.sites + midpoint_block.x - 1) / midpoint_block.x, (unsigned)current_num_ops);
        dim3 current_deriv_grid((unsigned)current_num_ops);
        dim3 current_pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        if (use_selective_refine &&
            restrict_to_refine_topk &&
            pass >= refine_cfg.global_opt_passes) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_last_loglk,
                d_prev_loglk,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_last_pendant_length,
                D.d_prev_pendant_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_last_proximal_length,
                D.d_prev_proximal_length,
                sizeof(fp_t) * (size_t)D.N,
                cudaMemcpyDeviceToDevice,
                stream));
        }
        LikelihoodDerivativePendantKernel<<<current_deriv_grid, pendant_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            current_op_indices,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            D.d_pattern_weights_u,
            30,
            D.d_new_pendant_length,
            sumtable_stride,
            D.d_prev_pendant_length,
            current_active_ops);
        check_launch("LikelihoodDerivativePendantKernel");

        // Rebuild query-side PMATs from the updated pendant lengths.
        BuildPendantPMATPerOpKernel<<<current_pmat_grid, pmat_block, 0, stream>>>(
            d_ops,
            current_op_indices,
            D.d_new_pendant_length,
            D.d_Vinv,
            D.d_V,
            D.d_lambdas,
            0.0,
            D.d_query_pmat,
            D.states,
            D.rate_cats,
            current_num_ops,
            D.N,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        check_launch("BuildPendantPMATPerOpKernel refine");
        LikelihoodDerivativeProximalKernel<<<current_deriv_grid, proximal_block, shmem_bytes, stream>>>(
            D,
            d_ops,
            0,
            current_op_indices,
            nullptr,
            nullptr,
            0.0,
            d_sumtable,
            D.d_pattern_weights_u,
            30,
            D.d_new_proximal_length,
            sumtable_stride,
            D.d_prev_proximal_length,
            current_active_ops);
        check_launch("LikelihoodDerivativeProximalKernel");

        // Rebuild midpoint PMATs from the updated proximal lengths.
        {
            if (use_compact_refine) {
                dim3 pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildSelectedNodeProximalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    d_ops,
                    current_op_indices,
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_prox,
                    D.states,
                    D.rate_cats,
                    current_num_ops,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildSelectedNodeProximalPMATKernel refine");
            } else {
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
                check_launch("BuildNodeProximalPMATKernel refine");
            }
        }

        // Rebuild distal PMATs from total branch length minus proximal length.
        {
            if (use_compact_refine) {
                dim3 pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
                BuildSelectedNodeDistalPMATKernel<<<pmat_grid, pmat_block, 0, stream>>>(
                    d_ops,
                    current_op_indices,
                    D.d_blen,
                    D.d_new_proximal_length,
                    D.d_Vinv,
                    D.d_V,
                    D.d_lambdas,
                    0.0,
                    D.d_pmat_mid_dist,
                    D.states,
                    D.rate_cats,
                    current_num_ops,
                    D.N,
                    D.root_id,
                    OPT_BRANCH_LEN_MIN,
                    OPT_BRANCH_LEN_MAX,
                    DEFAULT_BRANCH_LENGTH);
                check_launch("BuildSelectedNodeDistalPMATKernel refine");
            } else {
                dim3 pmat_grid((unsigned)((D.N * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
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
                check_launch("BuildNodeDistalPMATKernel refine");
            }
        }

        // Score each placement op after the pendant/proximal updates.
        root_likelihood::compute_combined_loglik_per_op_device(
            D,
            d_ops,
            current_op_indices,
            current_num_ops,
            D.d_query_pmat,
            D.d_pmat_mid_dist,
            D.d_pmat_mid_prox,
            d_likelihoods,
            stream);

        dim3 keep_block(256);
        dim3 keep_grid((unsigned)((current_num_ops + keep_block.x - 1) / keep_block.x));
        KeepBestBranchLengthsKernel<<<keep_grid, keep_block, 0, stream>>>(
            d_ops,
            current_op_indices,
            d_likelihoods,
            d_prev_loglk,
            D.d_new_pendant_length,
            D.d_new_proximal_length,
            D.d_prev_pendant_length,
            D.d_prev_proximal_length,
            current_active_ops,
            current_num_ops,
            D.N);
        check_launch("KeepBestBranchLengthsKernel");

        if (use_selective_refine && pass + 1 == 1 && refine_cfg.global_opt_passes > 1) {
            std::vector<int> top_idx;
            std::vector<fp_t> top_vals;
            fetch_topk_loglk(1, top_idx, top_vals);
            best_op_after_global_pass1 = top_idx.empty() ? -1 : top_idx[0];
        }

        if (use_selective_refine && pass + 1 == refine_cfg.global_opt_passes) {
            if (refine_cfg.detect_topk_limit <= 0 || refine_cfg.refine_topk_limit <= 0) {
                break;
            }
            const int topk = std::min(num_ops, refine_cfg.detect_topk_limit);
            std::vector<int> order;
            std::vector<fp_t> host_topk_ll;
            fetch_topk_loglk(topk, order, host_topk_ll);
            if (best_op_after_global_pass1 < 0 && !order.empty()) {
                best_op_after_global_pass1 = order[0];
            }

            const double best_ll = (topk > 0)
                ? static_cast<double>(host_topk_ll[0])
                : -std::numeric_limits<double>::infinity();
            const int best_op_after_global_pass2 = (topk > 0) ? order[0] : -1;
            const double gap12 = (topk > 1)
                ? (best_ll - static_cast<double>(host_topk_ll[1]))
                : std::numeric_limits<double>::infinity();
            const int top5 = std::min(topk, 5);
            const double gap15 = (top5 > 1)
                ? (best_ll - static_cast<double>(host_topk_ll[top5 - 1]))
                : std::numeric_limits<double>::infinity();

            const bool ambiguous =
                (gap12 < refine_cfg.gap_top2) ||
                (gap15 < refine_cfg.gap_top5) ||
                (best_op_after_global_pass1 != best_op_after_global_pass2);

            if (!ambiguous) {
                break;
            }

            const int refine_topk = std::min(num_ops, refine_cfg.refine_topk_limit);
            std::vector<int> refine_indices((size_t)refine_topk, 0);
            std::vector<int> refine_active((size_t)refine_topk, 1);
            for (int rank = 0; rank < refine_topk; ++rank) {
                refine_indices[(size_t)rank] = order[(size_t)rank];
            }
            if (!d_refine_op_indices || refine_op_count < refine_topk) {
                if (d_refine_op_indices) CUDA_CHECK(cudaFree(d_refine_op_indices));
                if (d_refine_active_ops) CUDA_CHECK(cudaFree(d_refine_active_ops));
                CUDA_CHECK(cudaMalloc(&d_refine_op_indices, sizeof(int) * (size_t)refine_topk));
                CUDA_CHECK(cudaMalloc(&d_refine_active_ops, sizeof(int) * (size_t)refine_topk));
            }
            CUDA_CHECK(cudaMemcpy(
                d_refine_op_indices,
                refine_indices.data(),
                sizeof(int) * (size_t)refine_topk,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_refine_active_ops,
                refine_active.data(),
                sizeof(int) * (size_t)refine_topk,
                cudaMemcpyHostToDevice));
            refine_op_count = refine_topk;
            restrict_to_refine_topk = true;
            continue;
        }

        if (use_selective_refine &&
            restrict_to_refine_topk &&
            pass + 1 > refine_cfg.global_opt_passes) {
            dim3 conv_block(256);
            dim3 conv_grid((unsigned)((refine_op_count + conv_block.x - 1) / conv_block.x));
            UpdateActiveOpsByConvergenceKernel<<<conv_grid, conv_block, 0, stream>>>(
                d_ops,
                d_refine_op_indices,
                d_prev_loglk,
                d_last_loglk,
                D.d_prev_pendant_length,
                d_last_pendant_length,
                D.d_prev_proximal_length,
                d_last_proximal_length,
                d_refine_active_ops,
                refine_op_count,
                D.N,
                refine_cfg.converged_loglk_eps,
                refine_cfg.converged_length_eps);
            CUDA_CHECK(cudaGetLastError());

            if (!any_active_on_device(d_refine_active_ops, refine_op_count)) {
                break;
            }
        }
    }
    
    // Argmax on device to get best op index.
    using Pair = cub::KeyValuePair<int, fp_t>;
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

    fp_t pendant_length = fp_t(0);
    fp_t proximal_length = fp_t(0);
    CUDA_CHECK(cudaMemcpyAsync(
        &pendant_length,
        D.d_prev_pendant_length + op_target_id,
        sizeof(fp_t),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        &proximal_length,
        D.d_prev_proximal_length + op_target_id,
        sizeof(fp_t),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_best));
    CUDA_CHECK(cudaFree(d_last_proximal_length));
    CUDA_CHECK(cudaFree(d_last_pendant_length));
    CUDA_CHECK(cudaFree(d_last_loglk));
    CUDA_CHECK(cudaFree(d_prev_loglk));
    if (d_any_active_temp) {
        CUDA_CHECK(cudaFree(d_any_active_temp));
    }
    if (d_any_active_flag) {
        CUDA_CHECK(cudaFree(d_any_active_flag));
    }
    if (d_refine_active_ops) {
        CUDA_CHECK(cudaFree(d_refine_active_ops));
    }
    if (d_refine_op_indices) {
        CUDA_CHECK(cudaFree(d_refine_op_indices));
    }
    CUDA_CHECK(cudaFree(d_active_ops));
    result.target_id = op_target_id;
    result.loglikelihood = static_cast<double>(h_best.value);
    result.proximal_length = static_cast<double>(proximal_length);
    result.pendant_length = static_cast<double>(pendant_length);
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
    int pseudo_root_id)
{
    if (!prune_cfg.enable_pruning || T.root_id < 0 || T.preorder.empty()) {
        return PlacementEvaluationKernel(
            D,
            er,
            rate_multipliers,
            d_ops,
            num_ops,
            smoothing,
            stream);
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
    std::vector<fp_t> batch_ll;
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
                stream);
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
            stream);

        batch_ll.resize((size_t)batch_n);
        CUDA_CHECK(cudaMemcpyAsync(
            batch_ll.data(),
            D.d_likelihoods,
            sizeof(fp_t) * (size_t)batch_n,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        next_frontier.clear();
        next_frontier.reserve((size_t)batch_n * 2);

        for (int batch_i = 0; batch_i < batch_n; ++batch_i) {
            const SearchState cur_state = eval_states[(size_t)batch_i];
            const NodeOpInfo& op = host_ops[(size_t)cur_state.op_idx];
            const int target_id = target_id_of(op);
            const double cur_ll = static_cast<double>(batch_ll[(size_t)batch_i]);

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
            stream);
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
