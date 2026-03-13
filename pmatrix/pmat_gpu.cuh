#pragma once

#include <cuda_runtime.h>
#include "../tree_generation/precision.hpp"

// Build a transition matrix from the eigendecomposition uploaded to GPU.
// Assumes a small state space (for example DNA with 4 states) and supports
// up to 16 states.
__device__ void pmatrix_from_triple_device(
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* rate_eigenvalues,
    fp_t rate_scale,
    fp_t branch_length,
    fp_t pinv,
    fp_t* out_pmat,
    int state_count);

// Build PMATs for a flat item array.
// Output layout: [item_count * rate_cats * state_count * state_count].
__global__ void pmatrix_from_triple_kernel(
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,        // [rate_cats * state_count] scaled by per-rate multipliers
    const fp_t* branch_lengths, // [item_count]
    fp_t p,
    fp_t* P,
    int n,
    int rate_cats,
    int item_count);
