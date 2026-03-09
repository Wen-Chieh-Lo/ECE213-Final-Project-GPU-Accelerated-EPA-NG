/**
 * Derivative kernel declarations.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "tree.hpp"

// Evaluate per-site derivatives and accumulate df/ddf (single-block launch assumed).
__global__ void LikelihoodDerivativeKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    bool proximal_mode,
    size_t sumtable_stride,
    fp_t* placement_clv_base,
    const fp_t* prev_branch_lengths,
    const int* active_ops);

// Fused pendant derivative kernel used by the fast refine path. It computes
// midpoint CLVs on-the-fly while building the derivative sumtable, avoiding
// the midpoint global-memory round-trip.
__global__ void LikelihoodDerivativeFusedPendantKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    size_t sumtable_stride,
    fp_t* placement_clv_base,
    const fp_t* prev_branch_lengths,
    const int* active_ops);

// Fused proximal derivative kernel used by the fast refine path. It computes
// proximal-mode midpoint CLVs on-the-fly while building the derivative
// sumtable, avoiding the midpoint global-memory round-trip.
__global__ void LikelihoodDerivativeFusedProximalKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    size_t sumtable_stride,
    fp_t* placement_clv_base,
    const fp_t* prev_branch_lengths,
    const int* active_ops);

// Experimental tiled derivative kernel. This intentionally uses a different
// update schedule from LikelihoodDerivativeKernel and is kept as an explicit
// A/B path rather than replacing the original implementation.
__global__ void LikelihoodDerivativeTiledKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    bool proximal_mode,
    size_t sumtable_stride,
    fp_t* placement_clv_base,
    const fp_t* prev_branch_lengths,
    const int* active_ops,
    int tile_sites,
    int tile_local_iters,
    int final_global_iters,
    fp_t local_damping);

// Experimental hybrid tiled kernel: local tiles contribute only gradient signal,
// while curvature is estimated from a global pass.
__global__ void LikelihoodDerivativeTiledGlobalCurvatureKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int* op_indices,
    const int*    __restrict__ invariant_site,
    const fp_t* __restrict__ invar_proportion,
    fp_t invar_scalar,
    fp_t* __restrict__ sumtable,
    const fp_t* __restrict__ pattern_weights,
    int max_iter,
    fp_t* new_branch_length,
    bool proximal_mode,
    size_t sumtable_stride,
    fp_t* placement_clv_base,
    const fp_t* prev_branch_lengths,
    const int* active_ops,
    int tile_sites,
    int outer_passes,
    int final_global_iters,
    fp_t local_damping);
