#include "pmat_gpu.cuh"

#include <cmath>

namespace {
constexpr int kMaxStates = 16;
}

__device__ void pmatrix_from_triple_device(
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lamb,
    fp_t r,
    fp_t t,
    fp_t p,
    fp_t* P,
    int n)
{
    if (!Vinv || !V || !lamb || !P) return;
    if (n <= 0 || n > kMaxStates) return;

    fp_t D[kMaxStates];
    for (int j = 0; j < n; ++j) {
        D[j] = fp_expm1(lamb[j] * r * t);
    }

    // V and Vinv are uploaded as transposed matrices (column-major view),
    // so index them accordingly to recover the original row-major math.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fp_t acc = fp_t(0);
            for (int k = 0; k < n; ++k) {
                const fp_t vik   = V[k * n + i];     // original V[i,k]
                const fp_t vinvj = Vinv[j * n + k];  // original Vinv[k,j]
                acc += vik * D[k] * vinvj;
            }
            P[i * n + j] = acc;
        }
        P[i * n + i] += fp_t(1.0);
    }

    if (p > 0.0) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fp_t I = (i == j) ? fp_t(1.0) : fp_t(0.0);
                P[i * n + j] = (fp_t(1.0) - p) * P[i * n + j] + p * I;
            }
        }
    }

    // Clamp tiny negatives and renormalize rows.
    for (int i = 0; i < n; ++i) {
        fp_t s = fp_t(0);
        for (int j = 0; j < n; ++j) {
            fp_t& x = P[i * n + j];
            if (x < 0.0 && x > fp_t(-1e-7f)) x = 0.0;
            s += x;
        }
        if (s != 0.0) {
            for (int j = 0; j < n; ++j) {
                P[i * n + j] /= s;
            }
        }
    }
}

__global__ void pmatrix_from_triple_kernel_per_op(
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    const fp_t* branch_lengths,
    fp_t p,
    fp_t* P,
    int n,
    int rate_cats,
    int num_ops)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_ops * rate_cats;
    if (idx < 0 || idx >= total) return;

    const int op_idx = idx / rate_cats;
    const int rc = idx - op_idx * rate_cats;
    if (op_idx < 0 || op_idx >= num_ops || rc < 0 || rc >= rate_cats) return;

    if (!branch_lengths) return;
    const fp_t t = branch_lengths[op_idx];

    const fp_t* lamb = lambdas + (size_t)rc * (size_t)n;
    fp_t* out = P + (size_t)idx * (size_t)n * (size_t)n;

    pmatrix_from_triple_device(Vinv, V, lamb, fp_t(1.0), t, p, out, n);
}

__global__ void pmatrix_from_triple_kernel_per_node(
    const fp_t* Vinv,
    const fp_t* V,
    const fp_t* lambdas,
    const fp_t* branch_lengths,
    fp_t p,
    fp_t* P,
    int n,
    int rate_cats,
    int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_nodes * rate_cats;
    if (idx < 0 || idx >= total) return;

    const int node_idx = idx / rate_cats;
    const int rc = idx - node_idx * rate_cats;
    if (node_idx < 0 || node_idx >= num_nodes || rc < 0 || rc >= rate_cats) return;

    if (!branch_lengths) return;
    const fp_t t = branch_lengths[node_idx];

    const fp_t* lamb = lambdas + (size_t)rc * (size_t)n;
    fp_t* out = P + ((size_t)node_idx * (size_t)rate_cats + (size_t)rc)
        * (size_t)n * (size_t)n;

    pmatrix_from_triple_device(Vinv, V, lamb, fp_t(1.0), t, p, out, n);
}
