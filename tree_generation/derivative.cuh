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
