#pragma once

#include <cuda_runtime.h>

// GPU version of pmatrix_from_triple.
// Assumes small n (e.g., 4); supports up to 16 states.
__device__ void pmatrix_from_triple_device(
    const double* Vinv,
    const double* V,
    const double* lamb,
    double r,
    double t,
    double p,
    double* P,
    int n);

// Compute per-op PMATs for query pendant branches.
// P layout: [num_ops * rate_cats * n * n], contiguous.
__global__ void pmatrix_from_triple_kernel_per_op(
    const double* Vinv,
    const double* V,
    const double* lambdas,        // [rate_cats * n] already scaled by rate multipliers
    const double* branch_lengths, // [num_ops] per-op branch lengths
    double p,
    double* P,
    int n,
    int rate_cats,
    int num_ops);

// Compute per-node PMATs for a branch-length array indexed by node id.
// P layout: [num_nodes * rate_cats * n * n], contiguous by node.
__global__ void pmatrix_from_triple_kernel_per_node(
    const double* Vinv,
    const double* V,
    const double* lambdas,        // [rate_cats * n] already scaled by rate multipliers
    const double* branch_lengths, // [num_nodes] per-node branch lengths
    double p,
    double* P,
    int n,
    int rate_cats,
    int num_nodes);
