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
    const int*    __restrict__ invariant_site,
    const double* __restrict__ invar_proportion,
    double invar_scalar,
    double* __restrict__ sumtable,
    const double* __restrict__ pattern_weights,
    int max_iter,
    double* new_branch_length,
    bool proximal_mode,
    size_t sumtable_stride,
    double* placement_clv_base,
    const double* prev_branch_lengths,
    const int* active_ops);
