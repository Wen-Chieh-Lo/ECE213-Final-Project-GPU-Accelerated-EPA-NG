#pragma once
#ifndef TREE_GENERATION_TREE_PLACEMENT_CUH
#define TREE_GENERATION_TREE_PLACEMENT_CUH

#include <cuda_runtime.h>
#include "tree.hpp"
#include "../partial_CUDA/partial_likelihood.cuh"
#include "root_likelihood.cuh"

// Host entry points for evaluating placement likelihoods while keeping ops on device.
void InsertLikelihoodEvaluationKernel(
    const DeviceTree& D,
    PlacementQueryBatch& Q,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    const NodeOpInfo* d_ops,
    int op_idx,
    double* d_likelihoods,
    void* d_temp_reduce,
    size_t temp_bytes_reduce,
    cudaStream_t stream);

struct PlacementResult {
    int target_id = -1;
    double loglikelihood = 0.0;
    double proximal_length = 0.0;
    double pendant_length = 0.0;
};

struct PlacementPruneConfig {
    bool enable_pruning = false;
    int max_consecutive_drops = 2;
    double drop_threshold = 0.0;
    bool enable_small_frontier_fallback = false;
    int small_frontier_threshold = 0;
};

PlacementResult PlacementEvaluationKernel(
    const DeviceTree& D,
    const EigResult& er,
    const std::vector<double>& rate_multipliers,
    const NodeOpInfo* d_ops,
    int num_ops,
    int smoothing,
    cudaStream_t stream,
    int debug_query_idx = -1);

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
    int pseudo_root_id = -1,
    int debug_query_idx = -1);

int get_env_int_or(const char* name, int fallback);

void dump_node_scaler_and_clv_snapshot(
    const DeviceTree& D,
    int node_id,
    int max_sites,
    cudaStream_t stream,
    const char* tag);

#endif // TREE_GENERATION_TREE_PLACEMENT_CUH
