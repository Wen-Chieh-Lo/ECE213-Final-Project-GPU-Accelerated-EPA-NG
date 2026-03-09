#include <vector>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
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
constexpr int kGlobalOptPasses = 2;
constexpr int kRefineExtraPasses = 2;
constexpr int kDetectTopK = 16;
constexpr int kRefineTopK = 16;
constexpr int kFastGlobalOptPasses = 1;
constexpr int kFastRefineExtraPasses = 1;
constexpr int kFastDetectTopK = 8;
constexpr int kFastRefineTopK = 8;
constexpr int kUltraFastGlobalOptPasses = 1;
constexpr int kUltraFastRefineExtraPasses = 0;
constexpr int kUltraFastDetectTopK = 0;
constexpr int kUltraFastRefineTopK = 0;
constexpr double kRefineGapTop2 = 0.25;
constexpr double kRefineGapTop5 = 1.0;
constexpr double kRefineConvergedLoglkEps = 1e-2;
constexpr double kRefineConvergedLengthEps = 1e-4;
constexpr int kTiledDerivativeDefaultTileSites = 32;
constexpr int kTiledDerivativeDefaultLocalIters = 4;
constexpr int kTiledDerivativeDefaultFinalGlobalIters = 1;
constexpr double kTiledDerivativeDefaultDamping = 1.0;
constexpr int kTiledDerivativeDefaultWarmupPasses = kGlobalOptPasses;
constexpr int kTiledDerivativeDefaultWarmupTopK = 0;

struct RefineModeConfig {
    bool use_adaptive_refine = false;
    bool use_fast_refine = false;
    bool use_ultra_fast_refine = false;
    bool use_fused_fast_derivative = false;
    int global_opt_passes = kGlobalOptPasses;
    int refine_extra_passes = kRefineExtraPasses;
    int detect_topk_limit = kDetectTopK;
    int refine_topk_limit = kRefineTopK;
};

struct TiledDerivativeConfig {
    bool use_tiled_derivative = false;
    bool use_tiled_derivative_global_curvature = false;
    int tile_sites = kTiledDerivativeDefaultTileSites;
    int local_iters = kTiledDerivativeDefaultLocalIters;
    int final_global_iters = kTiledDerivativeDefaultFinalGlobalIters;
    int warmup_passes = kTiledDerivativeDefaultWarmupPasses;
    int warmup_topk = kTiledDerivativeDefaultWarmupTopK;
    fp_t local_damping = static_cast<fp_t>(kTiledDerivativeDefaultDamping);
};

static RefineModeConfig load_refine_mode_config() {
    RefineModeConfig cfg;
    cfg.use_adaptive_refine = (std::getenv("MLIPPER_USE_ADAPTIVE_REFINE") != nullptr);
    cfg.use_fast_refine = (std::getenv("MLIPPER_USE_FAST_REFINE") != nullptr);
    cfg.use_ultra_fast_refine = (std::getenv("MLIPPER_USE_ULTRA_FAST_REFINE") != nullptr);
    cfg.use_fused_fast_derivative = cfg.use_adaptive_refine && cfg.use_fast_refine;
    if (cfg.use_ultra_fast_refine) {
        cfg.global_opt_passes = kUltraFastGlobalOptPasses;
        cfg.refine_extra_passes = kUltraFastRefineExtraPasses;
        cfg.detect_topk_limit = kUltraFastDetectTopK;
        cfg.refine_topk_limit = kUltraFastRefineTopK;
    } else if (cfg.use_fast_refine) {
        cfg.global_opt_passes = kFastGlobalOptPasses;
        cfg.refine_extra_passes = kFastRefineExtraPasses;
        cfg.detect_topk_limit = kFastDetectTopK;
        cfg.refine_topk_limit = kFastRefineTopK;
    }
    return cfg;
}

static TiledDerivativeConfig load_tiled_derivative_config() {
    TiledDerivativeConfig cfg;
    cfg.use_tiled_derivative = (std::getenv("MLIPPER_USE_TILED_DERIVATIVE") != nullptr);
    cfg.use_tiled_derivative_global_curvature =
        (std::getenv("MLIPPER_USE_TILED_DERIVATIVE_GLOBAL_CURVATURE") != nullptr);
    cfg.tile_sites = std::getenv("MLIPPER_TILED_DERIV_TILE_SITES")
        ? std::max(1, std::atoi(std::getenv("MLIPPER_TILED_DERIV_TILE_SITES")))
        : kTiledDerivativeDefaultTileSites;
    cfg.local_iters = std::getenv("MLIPPER_TILED_DERIV_LOCAL_ITERS")
        ? std::max(1, std::atoi(std::getenv("MLIPPER_TILED_DERIV_LOCAL_ITERS")))
        : kTiledDerivativeDefaultLocalIters;
    cfg.final_global_iters = std::getenv("MLIPPER_TILED_DERIV_FINAL_GLOBAL_ITERS")
        ? std::max(0, std::atoi(std::getenv("MLIPPER_TILED_DERIV_FINAL_GLOBAL_ITERS")))
        : kTiledDerivativeDefaultFinalGlobalIters;
    cfg.warmup_passes = std::getenv("MLIPPER_TILED_DERIV_WARMUP_PASSES")
        ? std::max(0, std::atoi(std::getenv("MLIPPER_TILED_DERIV_WARMUP_PASSES")))
        : kTiledDerivativeDefaultWarmupPasses;
    cfg.warmup_topk = std::getenv("MLIPPER_TILED_DERIV_WARMUP_TOPK")
        ? std::max(0, std::atoi(std::getenv("MLIPPER_TILED_DERIV_WARMUP_TOPK")))
        : kTiledDerivativeDefaultWarmupTopK;
    cfg.local_damping = std::getenv("MLIPPER_TILED_DERIV_DAMPING")
        ? static_cast<fp_t>(std::max(0.0, std::atof(std::getenv("MLIPPER_TILED_DERIV_DAMPING"))))
        : static_cast<fp_t>(kTiledDerivativeDefaultDamping);
    return cfg;
}
}

__global__ void BuildOpPendantLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* node_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    double min_len,
    double max_len,
    double default_len)
{
    const int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t t = static_cast<fp_t>(default_len);
    if (node_lengths) {
        const NodeOpInfo op = ops[op_idx];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            t = node_lengths[target_id];
        }
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);
    op_lengths[op_idx] = t;
}

__global__ void BuildOpDistalLengthsKernel(
    const NodeOpInfo* ops,
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* op_lengths,
    int num_ops,
    int total_nodes,
    double min_len,
    double max_len,
    double default_len)
{
    const int op_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !op_lengths) return;

    fp_t t = static_cast<fp_t>(default_len);
    if (total_lengths && proximal_lengths) {
        const NodeOpInfo op = ops[op_idx];
        const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
        const int target_id = target_is_left ? op.left_id : op.right_id;
        if (target_id >= 0 && target_id < total_nodes) {
            t = total_lengths[target_id] - proximal_lengths[target_id];
        }
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);
    op_lengths[op_idx] = t;
}

__global__ void BuildNodePendantLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < 0 || node_idx >= total_nodes) return;
    if (!out_lengths) return;

    fp_t t = static_cast<fp_t>(default_len);
    if (node_lengths) {
        t = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = static_cast<fp_t>(default_len);
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);
    out_lengths[node_idx] = t;
}

__global__ void BuildInitialProximalLengthsKernel(
    const fp_t* node_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < 0 || node_idx >= total_nodes) return;
    if (!out_lengths) return;

    fp_t t = static_cast<fp_t>(default_len);
    if (node_lengths) {
        t = static_cast<fp_t>(0.5) * node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = static_cast<fp_t>(default_len);
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);
    out_lengths[node_idx] = t;
}

__global__ void FillSequentialIndicesKernel(
    int* out_indices,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 0 || idx >= count) return;
    if (!out_indices) return;
    out_indices[idx] = idx;
}

__global__ void BuildNodeDistalLengthsKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    fp_t* out_lengths,
    int total_nodes,
    int root_id,
    double min_len,
    double max_len,
    double default_len)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < 0 || node_idx >= total_nodes) return;
    if (!out_lengths) return;

    fp_t t = static_cast<fp_t>(default_len);
    if (total_lengths && proximal_lengths) {
        t = total_lengths[node_idx] - proximal_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = static_cast<fp_t>(default_len);
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);
    out_lengths[node_idx] = t;
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
    if (op_local < 0 || op_local >= num_ops) return;
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
    if (op_local < 0 || op_local >= num_ops) return;
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

// Fused per-op pendant PMAT builder: computes branch length and writes PMAT without staging.
__global__ void BuildPendantPMATPerOpKernel(
    const NodeOpInfo* ops,
    const int* op_indices,
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    double p,
    fp_t* P,
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

    const int op_local = idx / rate_cats;
    const int rc = idx - op_local * rate_cats;
    if (!ops || op_local < 0 || op_local >= num_ops || rc < 0 || rc >= rate_cats) return;
    const int op_idx = op_indices ? op_indices[op_local] : op_local;
    if (op_idx < 0) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;

    fp_t t = static_cast<fp_t>(default_len);
    if (node_lengths && target_id >= 0 && target_id < total_nodes) {
        t = node_lengths[target_id];
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);

    const fp_t* lamb = lambdas + (size_t)rc * (size_t)states;
    fp_t* out = P + (size_t)idx * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, fp_t(1.0), t, static_cast<fp_t>(p), out, states);
}

// Fused per-node PMAT builder for proximal branches.
__global__ void BuildNodeProximalPMATKernel(
    const fp_t* node_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    double p,
    fp_t* P,
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

    fp_t t = static_cast<fp_t>(default_len);
    if (node_lengths) {
        t = node_lengths[node_idx];
    }
    if (node_idx == root_id) {
        t = static_cast<fp_t>(default_len);
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);

    const fp_t* lamb = lambdas + (size_t)rc * (size_t)states;
    fp_t* out = P + ((size_t)node_idx * (size_t)rate_cats + (size_t)rc)
        * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, fp_t(1.0), t, static_cast<fp_t>(p), out, states);
}

// Fused per-node PMAT builder for distal branches (total - proximal).
__global__ void BuildNodeDistalPMATKernel(
    const fp_t* total_lengths,
    const fp_t* proximal_lengths,
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    double p,
    fp_t* P,
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

    fp_t t = total_lengths[node_idx] - proximal_lengths[node_idx];
    if (node_idx == root_id) {
        t = static_cast<fp_t>(default_len);
    }
    if (t < static_cast<fp_t>(min_len)) t = static_cast<fp_t>(min_len);
    if (t > static_cast<fp_t>(max_len)) t = static_cast<fp_t>(max_len);

    const fp_t* lamb = lambdas + (size_t)rc * (size_t)states;
    fp_t* out = P + ((size_t)node_idx * (size_t)rate_cats + (size_t)rc)
        * (size_t)states * (size_t)states;
    pmatrix_from_triple_device(Vinv, V, lamb, fp_t(1.0), t, static_cast<fp_t>(p), out, states);
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
    if (!d_ops || op_local < 0 || op_local >= num_ops) return;
    const int op_idx = d_op_indices ? d_op_indices[op_local] : op_local;
    if (op_idx < 0) return;
    const NodeOpInfo op = d_ops[op_idx];
    const bool active_thread = (tid < D.sites);
    __shared__ fp_t shared_target_mat[8 * 16];
    __shared__ fp_t shared_parent_mat[8 * 16];
    // 先在 midpoint 上建立 CLV，再計算對數似然並寫回 midpoint 專用緩衝。
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
    fp_t* d_likelihoods = D.d_likelihoods;
    fp_t* d_sumtable = D.d_sumtable;

    const size_t diag_shared = (size_t)D.rate_cats * (size_t)D.states * 4;
    const RefineModeConfig refine_cfg = load_refine_mode_config();
    const TiledDerivativeConfig tiled_cfg = load_tiled_derivative_config();
    size_t shmem_bytes = sizeof(fp_t) * diag_shared;
    if (tiled_cfg.use_tiled_derivative || tiled_cfg.use_tiled_derivative_global_curvature) {
        const size_t tile_shared = (size_t)tiled_cfg.tile_sites * (size_t)D.rate_cats * (size_t)D.states;
        shmem_bytes += sizeof(fp_t) * tile_shared;
    } else if (refine_cfg.use_fused_fast_derivative) {
        const size_t midpoint_pmat_shared = (size_t)D.rate_cats * 16 * 2;
        shmem_bytes += sizeof(fp_t) * midpoint_pmat_shared;
    }

    // Choose a block size that fits the device based on occupancy.
    int block_threads = 512;
    int max_blocks_per_sm = 0;
    cudaFuncAttributes attr{};
    if (tiled_cfg.use_tiled_derivative_global_curvature) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeTiledGlobalCurvatureKernel));
    } else if (tiled_cfg.use_tiled_derivative) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeTiledKernel));
    } else if (refine_cfg.use_fused_fast_derivative) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeFusedPendantKernel));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attr, LikelihoodDerivativeKernel));
    }
    while (block_threads >= 32) {
        if (tiled_cfg.use_tiled_derivative_global_curvature) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                LikelihoodDerivativeTiledGlobalCurvatureKernel,
                block_threads,
                shmem_bytes));
        } else if (tiled_cfg.use_tiled_derivative) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                LikelihoodDerivativeTiledKernel,
                block_threads,
                shmem_bytes));
        } else if (refine_cfg.use_fused_fast_derivative) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                LikelihoodDerivativeFusedPendantKernel,
                block_threads,
                shmem_bytes));
        } else {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                LikelihoodDerivativeKernel,
                block_threads,
                shmem_bytes));
        }
        if (max_blocks_per_sm > 0) break;
        block_threads /= 2;
    }
    if (max_blocks_per_sm == 0) {
        throw std::runtime_error("No valid block size for LikelihoodDerivativeKernel on this GPU.");
    }

    const int midpoint_block_threads = 256;
    const int pmat_block_threads = 128;
    dim3 placement_block(block_threads);
    dim3 midpoint_block(midpoint_block_threads);
    dim3 pmat_block(pmat_block_threads);
    dim3 midpoint_grid((D.sites + midpoint_block.x - 1) / midpoint_block.x, (unsigned)num_ops);
    dim3 deriv_grid((unsigned)num_ops);

    fp_t* d_prev_loglk = nullptr;
    fp_t* d_last_loglk = nullptr;
    int* d_active_ops = nullptr;
    int* d_refine_op_indices = nullptr;
    int* d_refine_active_ops = nullptr;
    int* d_hybrid_warmup_ops = nullptr;
    fp_t* d_sort_keys_in = nullptr;
    fp_t* d_sort_keys_out = nullptr;
    int* d_sort_indices_in = nullptr;
    int* d_sort_indices_out = nullptr;
    int* d_any_active_flag = nullptr;
    void* d_sort_temp = nullptr;
    void* d_any_active_temp = nullptr;
    size_t sort_temp_bytes = 0;
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
        CUDA_CHECK(cudaGetLastError());
        BuildInitialProximalLengthsKernel<<<init_grid, init_block, 0, stream>>>(
            D.d_blen,
            D.d_prev_proximal_length,
            D.N,
            D.root_id,
            OPT_BRANCH_LEN_MIN,
            OPT_BRANCH_LEN_MAX,
            DEFAULT_BRANCH_LENGTH);
        CUDA_CHECK(cudaGetLastError());
    }

    fp_t* d_last_pendant_length = nullptr;
    fp_t* d_last_proximal_length = nullptr;
    CUDA_CHECK(cudaMalloc(&d_last_pendant_length, sizeof(fp_t) * (size_t)D.N));
    CUDA_CHECK(cudaMalloc(&d_last_proximal_length, sizeof(fp_t) * (size_t)D.N));
    if (refine_cfg.use_adaptive_refine || tiled_cfg.use_tiled_derivative_global_curvature) {
        CUDA_CHECK(cudaMalloc(&d_sort_keys_in, sizeof(fp_t) * (size_t)num_ops));
        CUDA_CHECK(cudaMalloc(&d_sort_keys_out, sizeof(fp_t) * (size_t)num_ops));
        CUDA_CHECK(cudaMalloc(&d_sort_indices_in, sizeof(int) * (size_t)num_ops));
        CUDA_CHECK(cudaMalloc(&d_sort_indices_out, sizeof(int) * (size_t)num_ops));
        CUDA_CHECK(cudaMalloc(&d_any_active_flag, sizeof(int)));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            d_sort_temp,
            sort_temp_bytes,
            d_sort_keys_in,
            d_sort_keys_out,
            d_sort_indices_in,
            d_sort_indices_out,
            num_ops,
            0,
            sizeof(fp_t) * 8,
            stream));
        CUDA_CHECK(cudaMalloc(&d_sort_temp, sort_temp_bytes));
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
            dim3 fill_block(256);
            dim3 fill_grid((unsigned)((num_ops + fill_block.x - 1) / fill_block.x));
            CUDA_CHECK(cudaMemcpyAsync(
                d_sort_keys_in,
                d_prev_loglk,
                sizeof(fp_t) * (size_t)num_ops,
                cudaMemcpyDeviceToDevice,
                stream));
            FillSequentialIndicesKernel<<<fill_grid, fill_block, 0, stream>>>(
                d_sort_indices_in,
                num_ops);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
                d_sort_temp,
                sort_temp_bytes,
                d_sort_keys_in,
                d_sort_keys_out,
                d_sort_indices_in,
                d_sort_indices_out,
                num_ops,
                0,
                sizeof(fp_t) * 8,
                stream));
            top_indices.resize((size_t)topk);
            top_values.resize((size_t)topk);
            CUDA_CHECK(cudaMemcpyAsync(
                top_indices.data(),
                d_sort_indices_out,
                sizeof(int) * (size_t)topk,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                top_values.data(),
                d_sort_keys_out,
                sizeof(fp_t) * (size_t)topk,
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
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

        BuildMidpointForPlacementKernel<<<midpoint_grid, midpoint_block, 0, stream>>>(
            D,
            d_ops,
            nullptr,
            0,
            num_ops,
            false);
        CUDA_CHECK(cudaGetLastError());

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

        if (tiled_cfg.use_tiled_derivative_global_curvature && tiled_cfg.warmup_topk > 0) {
            const int warmup_topk = std::min(num_ops, tiled_cfg.warmup_topk);
            std::vector<int> order;
            std::vector<fp_t> host_topk_ll;
            fetch_topk_loglk(warmup_topk, order, host_topk_ll);
            std::vector<int> warmup_mask((size_t)num_ops, 0);
            for (int rank = 0; rank < warmup_topk; ++rank) {
                warmup_mask[(size_t)order[(size_t)rank]] = 1;
            }
            CUDA_CHECK(cudaMalloc(&d_hybrid_warmup_ops, sizeof(int) * (size_t)num_ops));
            CUDA_CHECK(cudaMemcpy(
                d_hybrid_warmup_ops,
                warmup_mask.data(),
                sizeof(int) * (size_t)num_ops,
                cudaMemcpyHostToDevice));
        }
    }

    const int opt_passes = refine_cfg.use_adaptive_refine
        ? std::max(refine_cfg.global_opt_passes + refine_cfg.refine_extra_passes, smoothing)
        : std::max(4, smoothing);
    bool restrict_to_refine_topk = false;
    int best_op_after_global_pass1 = -1;
    int refine_op_count = 0;
    auto build_midpoint_if_needed =
        [&](const dim3& grid, const int* op_indices, int current_ops, bool proximal_mode) {
            if (!refine_cfg.use_fused_fast_derivative) {
                BuildMidpointForPlacementKernel<<<grid, midpoint_block, 0, stream>>>(
                    D,
                    d_ops,
                    op_indices,
                    0,
                    current_ops,
                    proximal_mode);
                CUDA_CHECK(cudaGetLastError());
            }
        };
    auto launch_derivative =
        [&](const dim3& grid,
            const int* op_indices,
            const int* active_ops,
            fp_t* new_lengths,
            const fp_t* prev_lengths,
            bool proximal_mode,
            const int* hybrid_active_ops,
            bool use_hybrid_now) {
            if (use_hybrid_now) {
                CUDA_CHECK(cudaMemcpyAsync(
                    new_lengths,
                    prev_lengths,
                    sizeof(fp_t) * (size_t)D.N,
                    cudaMemcpyDeviceToDevice,
                    stream));
                LikelihoodDerivativeTiledGlobalCurvatureKernel<<<grid, placement_block, shmem_bytes, stream>>>(
                    D,
                    d_ops,
                    0,
                    op_indices,
                    nullptr,
                    nullptr,
                    0.0,
                    d_sumtable,
                    nullptr,
                    30,
                    new_lengths,
                    proximal_mode,
                    sumtable_stride,
                    nullptr,
                    prev_lengths,
                    hybrid_active_ops,
                    tiled_cfg.tile_sites,
                    tiled_cfg.local_iters,
                    tiled_cfg.final_global_iters,
                    tiled_cfg.local_damping);
            } else if (tiled_cfg.use_tiled_derivative) {
                LikelihoodDerivativeTiledKernel<<<grid, placement_block, shmem_bytes, stream>>>(
                    D,
                    d_ops,
                    0,
                    op_indices,
                    nullptr,
                    nullptr,
                    0.0,
                    d_sumtable,
                    nullptr,
                    30,
                    new_lengths,
                    proximal_mode,
                    sumtable_stride,
                    nullptr,
                    prev_lengths,
                    active_ops,
                    tiled_cfg.tile_sites,
                    tiled_cfg.local_iters,
                    tiled_cfg.final_global_iters,
                    tiled_cfg.local_damping);
            } else if (refine_cfg.use_fused_fast_derivative) {
                if (proximal_mode) {
                    LikelihoodDerivativeFusedProximalKernel<<<grid, placement_block, shmem_bytes, stream>>>(
                        D,
                        d_ops,
                        0,
                        op_indices,
                        nullptr,
                        nullptr,
                        0.0,
                        d_sumtable,
                        nullptr,
                        30,
                        new_lengths,
                        sumtable_stride,
                        nullptr,
                        prev_lengths,
                        active_ops);
                } else {
                    LikelihoodDerivativeFusedPendantKernel<<<grid, placement_block, shmem_bytes, stream>>>(
                        D,
                        d_ops,
                        0,
                        op_indices,
                        nullptr,
                        nullptr,
                        0.0,
                        d_sumtable,
                        nullptr,
                        30,
                        new_lengths,
                        sumtable_stride,
                        nullptr,
                        prev_lengths,
                        active_ops);
                }
            } else {
                LikelihoodDerivativeKernel<<<grid, placement_block, shmem_bytes, stream>>>(
                    D,
                    d_ops,
                    0,
                    op_indices,
                    nullptr,
                    nullptr,
                    0.0,
                    d_sumtable,
                    nullptr,
                    30,
                    new_lengths,
                    proximal_mode,
                    sumtable_stride,
                    nullptr,
                    prev_lengths,
                    active_ops);
            }
            CUDA_CHECK(cudaGetLastError());
        };
    for (int pass = 0; pass < opt_passes; ++pass) {
        if (refine_cfg.use_adaptive_refine && pass >= refine_cfg.global_opt_passes && !restrict_to_refine_topk) {
            break;
        }
        const bool use_compact_refine =
            refine_cfg.use_adaptive_refine && restrict_to_refine_topk && refine_op_count > 0;
        const int current_num_ops = use_compact_refine ? refine_op_count : num_ops;
        const int* current_op_indices = use_compact_refine ? d_refine_op_indices : nullptr;
        int* current_active_ops = use_compact_refine ? d_refine_active_ops : d_active_ops;
        dim3 current_midpoint_grid((D.sites + midpoint_block.x - 1) / midpoint_block.x, (unsigned)current_num_ops);
        dim3 current_deriv_grid((unsigned)current_num_ops);
        dim3 current_pmat_grid((unsigned)((current_num_ops * D.rate_cats + pmat_block.x - 1) / pmat_block.x));
        const bool use_hybrid_derivative_now =
            tiled_cfg.use_tiled_derivative_global_curvature &&
            (pass < tiled_cfg.warmup_passes) &&
            !restrict_to_refine_topk;
        const int* deriv_active_ops =
            (use_hybrid_derivative_now && d_hybrid_warmup_ops)
                ? d_hybrid_warmup_ops
                : current_active_ops;
        if (refine_cfg.use_adaptive_refine &&
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
        build_midpoint_if_needed(current_midpoint_grid, current_op_indices, current_num_ops, false);
        launch_derivative(
            current_deriv_grid,
            current_op_indices,
            current_active_ops,
            D.d_new_pendant_length,
            D.d_prev_pendant_length,
            false,
            deriv_active_ops,
            use_hybrid_derivative_now);

        // Derivative kernel writes updated pendant lengths into D.d_new_pendant_length.
        // Update query PMAT with per-op pendant lengths on GPU (buffer allocated during upload).
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
        CUDA_CHECK(cudaGetLastError());
        // Ensure all pendant work is finished before proceeding.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        build_midpoint_if_needed(current_midpoint_grid, current_op_indices, current_num_ops, true);
        launch_derivative(
            current_deriv_grid,
            current_op_indices,
            current_active_ops,
            D.d_new_proximal_length,
            D.d_prev_proximal_length,
            true,
            deriv_active_ops,
            use_hybrid_derivative_now);

        // Derivative kernel writes updated proximal lengths into D.d_new_proximal_length.
        
        // Ensure proximal computations are finished before proceeding.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Proximal branch PMATs
        {
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
            CUDA_CHECK(cudaGetLastError());
        }
        // Combined likelihood per placement op (parallel across ops) for this pass.
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
        CUDA_CHECK(cudaGetLastError());

        if (refine_cfg.use_adaptive_refine && pass + 1 == 1) {
            std::vector<int> top_idx;
            std::vector<fp_t> top_vals;
            fetch_topk_loglk(1, top_idx, top_vals);
            best_op_after_global_pass1 = top_idx.empty() ? -1 : top_idx[0];
        }

        if (refine_cfg.use_adaptive_refine && pass + 1 == refine_cfg.global_opt_passes) {
            if (refine_cfg.detect_topk_limit <= 0 || refine_cfg.refine_topk_limit <= 0) {
                break;
            }
            const int topk = std::min(num_ops, refine_cfg.detect_topk_limit);
            std::vector<int> order;
            std::vector<fp_t> host_topk_ll;
            fetch_topk_loglk(topk, order, host_topk_ll);

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
                (gap12 < kRefineGapTop2) ||
                (gap15 < kRefineGapTop5) ||
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

        if (refine_cfg.use_adaptive_refine &&
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
                kRefineConvergedLoglkEps,
                kRefineConvergedLengthEps);
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
    if (d_sort_temp) {
        CUDA_CHECK(cudaFree(d_sort_temp));
    }
    if (d_any_active_flag) {
        CUDA_CHECK(cudaFree(d_any_active_flag));
    }
    if (d_sort_indices_out) {
        CUDA_CHECK(cudaFree(d_sort_indices_out));
    }
    if (d_sort_indices_in) {
        CUDA_CHECK(cudaFree(d_sort_indices_in));
    }
    if (d_sort_keys_out) {
        CUDA_CHECK(cudaFree(d_sort_keys_out));
    }
    if (d_sort_keys_in) {
        CUDA_CHECK(cudaFree(d_sort_keys_in));
    }
    if (d_hybrid_warmup_ops) {
        CUDA_CHECK(cudaFree(d_hybrid_warmup_ops));
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
