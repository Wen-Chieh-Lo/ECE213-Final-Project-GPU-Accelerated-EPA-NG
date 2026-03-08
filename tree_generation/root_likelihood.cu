#include "root_likelihood.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "tree.hpp"
#include "../partial_CUDA/partial_likelihood.cuh"
#include "../core_CUDA/reduction.cuh"

namespace root_likelihood {

__device__ __constant__ int c_debug_combined_op;
__device__ __constant__ int c_debug_combined_site;
constexpr double kLn2 = 0.69314718055994530942;

__device__ __forceinline__ unsigned int combined_scaler_shift_at(
    const unsigned* scaler_pool,
    const DeviceTree& D,
    int node_id,
    unsigned int site_idx,
    int rate_idx)
{
    if (!scaler_pool || node_id < 0) return 0u;
    const size_t per_node = D.per_rate_scaling
        ? (D.sites * (size_t)D.rate_cats)
        : D.sites;
    const size_t base = (size_t)node_id * per_node;
    if (D.per_rate_scaling) {
        return scaler_pool[base + (size_t)site_idx * (size_t)D.rate_cats + (size_t)rate_idx];
    }
    return scaler_pool[base + (size_t)site_idx];
}

// Specialized device helpers for common state/rate counts.
template<int RC>
__device__ __forceinline__ void compute_root_loglikelihood_states4(
    const DeviceTree& D,
    const double* clv_site,
    const double* freqs, const double* rate_weights,
    unsigned int site_idx,
    const unsigned* pattern_w,
    const int* invar_indices,
    double invar_proportion,
    double * total_likelihood)
{
    const double pi0 = freqs[0];
    const double pi1 = freqs[1];
    const double pi2 = freqs[2];
    const double pi3 = freqs[3];

    double sum_rate = 0.0;
    #pragma unroll
    for (int r = 0; r < RC; ++r) {
        const double4 a = reinterpret_cast<const double4*>(clv_site)[r];
        double val = fma(a.x, pi0, fma(a.y, pi1, fma(a.z, pi2, a.w * pi3)));
        if (D.d_site_scaler) {
            unsigned int shift = D.per_rate_scaling
                ? D.d_site_scaler[(size_t)site_idx * (size_t)RC + r]
                : D.d_site_scaler[site_idx];
            if (shift) val = ldexp(val, -static_cast<int>(shift));
        }
        sum_rate = fma(rate_weights[r], val, sum_rate);
    }

    double site_sum = (1.0 - invar_proportion) * sum_rate;
    if (invar_indices) {
        int inv_idx = invar_indices[site_idx];
        if (inv_idx >= 0) site_sum += invar_proportion * freqs[inv_idx];
    }
    const double eps = 1e-300;
    double loglk = log(site_sum > eps ? site_sum : eps);
    if (pattern_w) loglk *= static_cast<double>(pattern_w[site_idx]);
    total_likelihood[site_idx] = loglk;
}

template<int RC>
__device__ __forceinline__ void compute_root_loglikelihood_states5(
    const DeviceTree& D,
    const double* clv_site,
    const double* freqs, const double* rate_weights,
    unsigned int site_idx,
    const unsigned* pattern_w,
    const int* invar_indices,
    double invar_proportion,
    double * total_likelihood)
{
    const double pi0 = freqs[0];
    const double pi1 = freqs[1];
    const double pi2 = freqs[2];
    const double pi3 = freqs[3];
    const double pi4 = freqs[4];

    double sum_rate = 0.0;
    #pragma unroll
    for (int r = 0; r < RC; ++r) {
        const double* cr = clv_site + (size_t)r * 5;
        double val = cr[0]*pi0 + cr[1]*pi1 + cr[2]*pi2 + cr[3]*pi3 + cr[4]*pi4;
        if (D.d_site_scaler) {
            unsigned int shift = D.per_rate_scaling
                ? D.d_site_scaler[(size_t)site_idx * (size_t)RC + r]
                : D.d_site_scaler[site_idx];
            if (shift) val = ldexp(val, -static_cast<int>(shift));
        }
        sum_rate = fma(rate_weights[r], val, sum_rate);
    }

    double site_sum = (1.0 - invar_proportion) * sum_rate;
    if (invar_indices) {
        int inv_idx = invar_indices[site_idx];
        if (inv_idx >= 0) site_sum += invar_proportion * freqs[inv_idx];
    }
    const double eps = 1e-300;
    double loglk = log(site_sum > eps ? site_sum : eps);
    if (pattern_w) loglk *= static_cast<double>(pattern_w[site_idx]);
    total_likelihood[site_idx] = loglk;
}

// Generic device root log-likelihood for any state/rate counts (fallback).
__device__ __forceinline__ void compute_root_loglikelihood_generic(
    const DeviceTree& D,
    const double* clv_site,
    const double* freqs, const double* rate_weights,
    unsigned int site_idx, const unsigned* pattern_w,
    const int* invar_indices, double invar_proportion,
    double * total_likelihood)
{
    double sum_rate = 0.0;
    for (unsigned int r = 0; r < (unsigned)D.rate_cats; ++r) {
        const double* cr = clv_site + (size_t)r * (size_t)D.states;
        double val = 0.0;
        for (unsigned int s = 0; s < (unsigned)D.states; ++s) {
            val = fma(cr[s], freqs[s], val);
        }
        if (D.d_site_scaler) {
            unsigned int shift = D.per_rate_scaling
                ? D.d_site_scaler[(size_t)site_idx * (size_t)D.rate_cats + r]
                : D.d_site_scaler[site_idx];
            if (shift) val = ldexp(val, -static_cast<int>(shift));
        }
        sum_rate = fma(rate_weights[r], val, sum_rate);
    }
    double site_sum = (1.0 - invar_proportion) * sum_rate;
    if (invar_indices) {
        int inv_idx = invar_indices[site_idx];
        if (inv_idx >= 0) site_sum += invar_proportion * freqs[inv_idx];
    }
    const double eps = 1e-300;
    double loglk = log(site_sum > eps ? site_sum : eps);
    if (pattern_w) loglk *= static_cast<double>(pattern_w[site_idx]);
    total_likelihood[site_idx] = loglk;
}

// Device helper that allows explicit site index (usable from arbitrary kernels).
__device__ void compute_root_loglikelihood_at_site(
    const DeviceTree& D,
    const NodeOpInfo& op,
    const double* freqs,
    const double* rate_weights,
    const unsigned* pattern_w,
    const int* invar_indices,
    double invar_proportion,
    unsigned int site_idx)
{
    if (site_idx >= D.sites) return;
    double *placement_clv = D.d_placement_clv;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int target_id  = target_is_left ? op.left_id  : (target_is_right ? op.right_id : op.parent_id);

    const size_t per_node = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    const double* clv_pool = D.d_clv_mid;
    if (!clv_pool || target_id < 0 || target_id >= D.N || !placement_clv) return;
    const double* clv_site = clv_pool + (size_t)target_id * per_node + (size_t)site_idx * (size_t)D.rate_cats * (size_t)D.states;
    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:  compute_root_loglikelihood_states4<1>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            case 4:  compute_root_loglikelihood_states4<4>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            case 8:  compute_root_loglikelihood_states4<8>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            default: compute_root_loglikelihood_generic(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
        }
    } else if (D.states == 5) {
        switch (D.rate_cats) {
            case 1:  compute_root_loglikelihood_states5<1>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            case 4:  compute_root_loglikelihood_states5<4>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            case 8:  compute_root_loglikelihood_states5<8>(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
            default: compute_root_loglikelihood_generic(D, clv_site, freqs, rate_weights, site_idx, pattern_w, invar_indices, invar_proportion, placement_clv); break;
        }
    }

}

__global__ void RootLikelihoodPerSiteKernel(
    DeviceTree D,
    NodeOpInfo op,
    const unsigned* __restrict__ d_pattern_w,
    const int* __restrict__ d_invar_indices,
    double invar_proportion)
{
    const unsigned int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= D.sites) return;
    compute_root_loglikelihood_at_site(
        D,
        op,
        D.d_frequencies,
        D.d_rate_weights,
        d_pattern_w,
        d_invar_indices,
        invar_proportion,
        site);
}

double compute_root_loglikelihood_total(
    const DeviceTree& D,
    int root_id,
    const unsigned* d_pattern_w,
    const int* d_invar_indices,
    double invar_proportion,
    cudaStream_t stream)
{
    if (root_id < 0 || root_id >= D.N) {
        throw std::runtime_error("Invalid root id.");
    }
    if (!D.d_frequencies || !D.d_rate_weights) {
        throw std::runtime_error("Device frequencies or rate weights are not initialized.");
    }
    if (!D.d_clv_mid || !D.d_clv_up) {
        throw std::runtime_error("Device CLV buffers are not initialized.");
    }
    if (!D.d_placement_clv) {
        throw std::runtime_error("Device placement_clv buffer is not initialized.");
    }

    const size_t per_node = D.per_node_elems();
    if (per_node > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_clv_mid,
            D.d_clv_up + (size_t)root_id * per_node,
            sizeof(double) * per_node,
            cudaMemcpyDeviceToDevice,
            stream));
        // std::vector<double> zero_buf(per_node, 0.0);
        // CUDA_CHECK(cudaMemcpyAsync(
        //     zero_buf.data(),
        //     D.d_clv_mid,
        //     sizeof(double) * per_node,
        //     cudaMemcpyDeviceToHost,
        //     stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // printf("[root-lk] copied root CLV for node %d, first value = %f\n", root_id, zero_buf[0]);
        
    }

    NodeOpInfo root_op{};
    root_op.parent_id = 0;
    root_op.dir_tag = static_cast<uint8_t>(CLV_DIR_UP);
    root_op.clv_pool = static_cast<uint8_t>(CLV_POOL_UP);

    dim3 block(256);
    dim3 grid((unsigned)((D.sites + block.x - 1) / block.x));
    RootLikelihoodPerSiteKernel<<<grid, block, 0, stream>>>(
        D,
        root_op,
        d_pattern_w,
        d_invar_indices,
        invar_proportion);
    
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<double> host_lk(D.sites, 0.0);
    if (D.sites > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            host_lk.data(),
            D.d_placement_clv,
            sizeof(double) * D.sites,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    // for(auto i =0; i < 10; i++) {
    //     if (host_lk[i] < 0.0) {
    //         fprintf(stderr, "[root-lk] site %d has negative log-likelihood: %f\n", i, host_lk[i]);
    //     }
    // }

    double total = 0.0;
    for (double v : host_lk) total += v;
    return total;
}

// Per-op combined placement log-likelihood kernel (states=4, rate_cats templated).
template<int RATE_CATS>
__global__ void CombinedPlacementLoglikPerOpKernelStates4(
    DeviceTree D,
    const NodeOpInfo* __restrict__ d_ops,
    const double* __restrict__ d_pendant_pmats,
    const double* __restrict__ d_distal_pmats,
    const double* __restrict__ d_proximal_pmats,
    size_t per_query,
    size_t per_node_pmat,
    double* __restrict__ d_out)
{
    const int op_idx = (int)blockIdx.y;
    if (!d_ops || op_idx < 0 || op_idx >= D.N) return;
    const NodeOpInfo op = d_ops[op_idx];
    const bool target_is_left  = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int target_id = target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    if (target_id < 0 || target_id >= D.N) return;

    const double* pendant_pmat = d_pendant_pmats ? d_pendant_pmats + (size_t)op_idx * per_query : nullptr;
    const double* distal_pmat  = d_distal_pmats ? d_distal_pmats + (size_t)target_id * per_node_pmat : nullptr;
    const double* prox_pmat    = d_proximal_pmats ? d_proximal_pmats + (size_t)target_id * per_node_pmat : nullptr;
    if (!pendant_pmat || !distal_pmat || !prox_pmat) return;

    if (!D.d_query_clv || !D.d_clv_mid_base || !D.d_clv_up || !D.d_rate_weights || !D.d_frequencies) return;
    const size_t per_site = (size_t)RATE_CATS * 4;
    const size_t per_node = (size_t)D.sites * per_site;

    double local_sum = 0.0;
    for (unsigned int site = threadIdx.x; site < D.sites; site += blockDim.x) {
        const bool debug_exact_site =
            (op_idx == c_debug_combined_op) && (c_debug_combined_site >= 0) && ((int)site == c_debug_combined_site);
        const bool debug_all_sites =
            (op_idx == c_debug_combined_op) && (c_debug_combined_site == -2);
        const double* query_clv = D.d_query_clv + (size_t)site * per_site;
        const double* distal_clv = D.d_clv_mid_base + (size_t)target_id * per_node + (size_t)site * per_site;
        const double* prox_clv = D.d_clv_up + (size_t)target_id * per_node + (size_t)site * per_site;
        const double* mid_clv = D.d_clv_mid
            ? (D.d_clv_mid + (size_t)target_id * per_node + (size_t)site * per_site)
            : nullptr;

        double rate_vals[RATE_CATS];
        unsigned int rate_shifts[RATE_CATS];
        unsigned int site_min_shift = 0u;
        bool have_positive = false;
        #pragma unroll
        for (int rc = 0; rc < RATE_CATS; ++rc) {
            const double4 q = reinterpret_cast<const double4*>(query_clv  + (size_t)rc * 4)[0];
            const double4 d = reinterpret_cast<const double4*>(distal_clv + (size_t)rc * 4)[0];
            const double4 p = reinterpret_cast<const double4*>(prox_clv   + (size_t)rc * 4)[0];
            const unsigned int distal_shift =
                combined_scaler_shift_at(D.d_site_scaler_mid_base, D, target_id, site, rc);
            const unsigned int prox_shift =
                combined_scaler_shift_at(D.d_site_scaler_up, D, target_id, site, rc);

            const double4* p_pendant = reinterpret_cast<const double4*>(pendant_pmat + (size_t)rc * 16);
            const double4* p_distal  = reinterpret_cast<const double4*>(distal_pmat  + (size_t)rc * 16);
            const double4* p_prox    = reinterpret_cast<const double4*>(prox_pmat    + (size_t)rc * 16);

            const double acc_pend0 = p_pendant[0].x * q.x + p_pendant[0].y * q.y + p_pendant[0].z * q.z + p_pendant[0].w * q.w;
            const double acc_pend1 = p_pendant[1].x * q.x + p_pendant[1].y * q.y + p_pendant[1].z * q.z + p_pendant[1].w * q.w;
            const double acc_pend2 = p_pendant[2].x * q.x + p_pendant[2].y * q.y + p_pendant[2].z * q.z + p_pendant[2].w * q.w;
            const double acc_pend3 = p_pendant[3].x * q.x + p_pendant[3].y * q.y + p_pendant[3].z * q.z + p_pendant[3].w * q.w;

            const double acc_dist0 = p_distal[0].x * d.x + p_distal[0].y * d.y + p_distal[0].z * d.z + p_distal[0].w * d.w;
            const double acc_dist1 = p_distal[1].x * d.x + p_distal[1].y * d.y + p_distal[1].z * d.z + p_distal[1].w * d.w;
            const double acc_dist2 = p_distal[2].x * d.x + p_distal[2].y * d.y + p_distal[2].z * d.z + p_distal[2].w * d.w;
            const double acc_dist3 = p_distal[3].x * d.x + p_distal[3].y * d.y + p_distal[3].z * d.z + p_distal[3].w * d.w;

            const double acc_prox0 = p_prox[0].x * p.x + p_prox[0].y * p.y + p_prox[0].z * p.z + p_prox[0].w * p.w;
            const double acc_prox1 = p_prox[1].x * p.x + p_prox[1].y * p.y + p_prox[1].z * p.z + p_prox[1].w * p.w;
            const double acc_prox2 = p_prox[2].x * p.x + p_prox[2].y * p.y + p_prox[2].z * p.z + p_prox[2].w * p.w;
            const double acc_prox3 = p_prox[3].x * p.x + p_prox[3].y * p.y + p_prox[3].z * p.z + p_prox[3].w * p.w;

            const double* freqs = D.d_frequencies;
            const double v0 = acc_pend0 * acc_dist0 * acc_prox0 * freqs[0];
            const double v1 = acc_pend1 * acc_dist1 * acc_prox1 * freqs[1];
            const double v2 = acc_pend2 * acc_dist2 * acc_prox2 * freqs[2];
            const double v3 = acc_pend3 * acc_dist3 * acc_prox3 * freqs[3];

            double val = 0.0;
            if (v0 > 0.0) val += v0;
            if (v1 > 0.0) val += v1;
            if (v2 > 0.0) val += v2;
            if (v3 > 0.0) val += v3;
            const int total_shift = (int)distal_shift + (int)prox_shift;
            rate_vals[rc] = val;
            rate_shifts[rc] = (unsigned int)((total_shift > 0) ? total_shift : 0);
            if (val > 0.0) {
                if (!have_positive || rate_shifts[rc] < site_min_shift) {
                    site_min_shift = rate_shifts[rc];
                }
                have_positive = true;
            }
            if (debug_exact_site) {
                const double4 mid = mid_clv
                    ? reinterpret_cast<const double4*>(mid_clv + (size_t)rc * 4)[0]
                    : make_double4(0.0, 0.0, 0.0, 0.0);
                const double recon0 = acc_dist0 * acc_prox0;
                const double recon1 = acc_dist1 * acc_prox1;
                const double recon2 = acc_dist2 * acc_prox2;
                const double recon3 = acc_dist3 * acc_prox3;
                printf(
                    "[combined-debug] op=%d target=%d site=%u rc=%d "
                    "q=(%.3e %.3e %.3e %.3e) "
                    "d=(%.3e %.3e %.3e %.3e) "
                    "p=(%.3e %.3e %.3e %.3e) "
                    "mid=(%.3e %.3e %.3e %.3e) "
                    "recon=(%.3e %.3e %.3e %.3e) "
                    "pend=(%.3e %.3e %.3e %.3e) "
                    "dist=(%.3e %.3e %.3e %.3e) "
                    "prox=(%.3e %.3e %.3e %.3e) "
                    "v=(%.3e %.3e %.3e %.3e) val=%.3e dshift=%u pshift=%u tshift=%u\n",
                    op_idx, target_id, site, rc,
                    q.x, q.y, q.z, q.w,
                    d.x, d.y, d.z, d.w,
                    p.x, p.y, p.z, p.w,
                    mid.x, mid.y, mid.z, mid.w,
                    recon0, recon1, recon2, recon3,
                    acc_pend0, acc_pend1, acc_pend2, acc_pend3,
                    acc_dist0, acc_dist1, acc_dist2, acc_dist3,
                    acc_prox0, acc_prox1, acc_prox2, acc_prox3,
                    v0, v1, v2, v3, val, distal_shift, prox_shift, rate_shifts[rc]);
            }
        }
        double site_lk = 0.0;
        #pragma unroll
        for (int rc = 0; rc < RATE_CATS; ++rc) {
            const double rate_w = D.d_rate_weights[rc];
            double val = rate_vals[rc];
            if (val > 0.0) {
                const int diff = (int)rate_shifts[rc] - (int)site_min_shift;
                if (diff > 0) val = ldexp(val, -diff);
                site_lk += rate_w * val;
            }
        }
        const double eps = 1e-300;
        if (debug_exact_site || debug_all_sites) {
            printf("[combined-debug] op=%d target=%d site=%u site_lk=%.12e min_shift=%u log=%.12e\n",
                   op_idx, target_id, site, site_lk, site_min_shift,
                   log(site_lk > eps ? site_lk : eps) - (double)site_min_shift * kLn2);
        }
        local_sum += log(site_lk > eps ? site_lk : eps) - (double)site_min_shift * kLn2;
    }

    // Block reduction (warp then block) to avoid atomics.
    __shared__ double warp_sum[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    double v = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (lane == 0) warp_sum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double block_sum = (lane < ((blockDim.x + 31) >> 5)) ? warp_sum[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) d_out[op_idx] = block_sum;
    }
}

// Generic kernel (fallback): block reduction, runtime rate_cats/states.
__global__ void CombinedPlacementLoglikPerOpKernelGeneric(
    DeviceTree D,
    const NodeOpInfo* __restrict__ d_ops,
    const double* __restrict__ d_pendant_pmats,
    const double* __restrict__ d_distal_pmats,
    const double* __restrict__ d_proximal_pmats,
    size_t per_query,
    size_t per_node_pmat,
    double* __restrict__ d_out)
{
    const int op_idx = (int)blockIdx.y;
    if (!d_ops || op_idx < 0 || op_idx >= D.N) return;
    const NodeOpInfo op = d_ops[op_idx];
    const bool target_is_left  = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_right = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
    const int target_id = target_is_left ? op.left_id : (target_is_right ? op.right_id : op.parent_id);
    if (target_id < 0 || target_id >= D.N) return;

    const double* pendant_pmat = d_pendant_pmats ? d_pendant_pmats + (size_t)op_idx * per_query : nullptr;
    const double* distal_pmat  = d_distal_pmats ? d_distal_pmats + (size_t)target_id * per_node_pmat : nullptr;
    const double* prox_pmat    = d_proximal_pmats ? d_proximal_pmats + (size_t)target_id * per_node_pmat : nullptr;
    if (!pendant_pmat || !distal_pmat || !prox_pmat) return;

    if (!D.d_query_clv || !D.d_clv_mid_base || !D.d_clv_up || !D.d_rate_weights || !D.d_frequencies) return;
    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    const size_t per_node = (size_t)D.sites * per_site;

    double local_sum = 0.0;
    for (unsigned int site = threadIdx.x; site < D.sites; site += blockDim.x) {
        const bool debug_all_sites =
            (op_idx == c_debug_combined_op) && (c_debug_combined_site == -2);
        const double* query_clv = D.d_query_clv + (size_t)site * per_site;
        const double* distal_clv = D.d_clv_mid_base + (size_t)target_id * per_node + (size_t)site * per_site;
        const double* prox_clv = D.d_clv_up + (size_t)target_id * per_node + (size_t)site * per_site;

        double rate_vals[64];
        unsigned int rate_shifts[64];
        unsigned int site_min_shift = 0u;
        bool have_positive = false;
        for (int rc = 0; rc < D.rate_cats; ++rc) {
            const unsigned int distal_shift =
                combined_scaler_shift_at(D.d_site_scaler_mid_base, D, target_id, site, rc);
            const unsigned int prox_shift =
                combined_scaler_shift_at(D.d_site_scaler_up, D, target_id, site, rc);
            const double* p_pendant = pendant_pmat + (size_t)rc * D.states * D.states;
            const double* p_distal  = distal_pmat  + (size_t)rc * D.states * D.states;
            const double* p_prox    = prox_pmat    + (size_t)rc * D.states * D.states;

            double pend_vec[64];
            double distal_vec[64];
            double prox_vec[64];
            for (int s = 0; s < D.states; ++s) {
                double acc_pend = 0.0, acc_dist = 0.0, acc_prox = 0.0;
                const double* qrow = query_clv + (size_t)rc * D.states;
                const double* drow = distal_clv + (size_t)rc * D.states;
                const double* prow = prox_clv   + (size_t)rc * D.states;
                for (int k = 0; k < D.states; ++k) {
                    acc_pend += p_pendant[(size_t)s * D.states + k] * qrow[k];
                    acc_dist += p_distal[(size_t)s * D.states + k]  * drow[k];
                    acc_prox += p_prox[(size_t)s * D.states + k]    * prow[k];
                }
                pend_vec[s] = acc_pend;
                distal_vec[s] = acc_dist;
                prox_vec[s] = acc_prox;
            }
            double rate_sum = 0.0;
            for (int s = 0; s < D.states; ++s) {
                double val = pend_vec[s] * distal_vec[s] * prox_vec[s] * D.d_frequencies[s];
                if (val > 0.0) rate_sum += val;
            }
            const int total_shift = (int)distal_shift + (int)prox_shift;
            rate_vals[rc] = rate_sum;
            rate_shifts[rc] = (unsigned int)((total_shift > 0) ? total_shift : 0);
            if (rate_sum > 0.0) {
                if (!have_positive || rate_shifts[rc] < site_min_shift) {
                    site_min_shift = rate_shifts[rc];
                }
                have_positive = true;
            }
        }
        double site_lk = 0.0;
        for (int rc = 0; rc < D.rate_cats; ++rc) {
            const double rate_w = D.d_rate_weights[rc];
            double rate_sum = rate_vals[rc];
            if (rate_sum > 0.0) {
                const int diff = (int)rate_shifts[rc] - (int)site_min_shift;
                if (diff > 0) rate_sum = ldexp(rate_sum, -diff);
                site_lk += rate_w * rate_sum;
            }
        }
        const double eps = 1e-300;
        if (debug_all_sites) {
            printf("[combined-debug] op=%d target=%d site=%u site_lk=%.12e min_shift=%u log=%.12e\n",
                   op_idx, target_id, site, site_lk, site_min_shift,
                   log(site_lk > eps ? site_lk : eps) - (double)site_min_shift * kLn2);
        }
        local_sum += log(site_lk > eps ? site_lk : eps) - (double)site_min_shift * kLn2;
    }

    __shared__ double warp_sum[32];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    double v = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (lane == 0) warp_sum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double block_sum = (lane < ((blockDim.x + 31) >> 5)) ? warp_sum[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) d_out[op_idx] = block_sum;
    }
}

void compute_combined_loglik_per_op(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    int num_ops,
    const double* d_pendant_pmats,
    const double* d_distal_pmats,
    const double* d_proximal_pmats,
    double * d_likelihoods,
    cudaStream_t stream)
{
    if (!d_ops || !d_pendant_pmats || !d_distal_pmats || !d_proximal_pmats) {
        throw std::runtime_error("Missing PMAT or ops pointers for combined loglik.");
    }
    if (!D.d_query_clv || !D.d_clv_mid_base || !D.d_clv_up || !D.d_rate_weights || !D.d_frequencies) {
        throw std::runtime_error("Placement buffers not initialized.");
    }
    compute_combined_loglik_per_op_device(
        D,
        d_ops,
        num_ops,
        d_pendant_pmats,
        d_distal_pmats,
        d_proximal_pmats,
        d_likelihoods,
        stream);
}

void compute_combined_loglik_per_op_device(
    const DeviceTree& D,
    const NodeOpInfo* d_ops,
    int num_ops,
    const double* d_pendant_pmats,
    const double* d_distal_pmats,
    const double* d_proximal_pmats,
    double* d_out,
    cudaStream_t stream)
{
    if (num_ops <= 0) return;
    if (!d_ops || !d_pendant_pmats || !d_distal_pmats || !d_proximal_pmats) {
        throw std::runtime_error("Missing PMAT or ops pointers for combined loglik.");
    }
    if (!d_out) {
        throw std::runtime_error("Missing output buffer for combined loglik.");
    }
    if (!D.d_query_clv || !D.d_clv_mid_base || !D.d_clv_up || !D.d_rate_weights || !D.d_frequencies) {
        throw std::runtime_error("Placement buffers not initialized.");
    }

    int debug_op = -1;
    int debug_site = -1;
    if (const char* raw = std::getenv("MLIPPER_DEBUG_COMBINED_OP")) debug_op = std::atoi(raw);
    if (const char* raw = std::getenv("MLIPPER_DEBUG_COMBINED_SITE")) debug_site = std::atoi(raw);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_debug_combined_op, &debug_op, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_debug_combined_site, &debug_site, sizeof(int), 0, cudaMemcpyHostToDevice, stream));

    const size_t per_query = (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
    const size_t per_node_pmat = per_query;

    dim3 block(256);
    dim3 grid(1, (unsigned)num_ops);
    if (D.states == 4) {
        if (D.rate_cats == 4) {
            CombinedPlacementLoglikPerOpKernelStates4<4><<<grid, block, 0, stream>>>(
                D,
                d_ops,
                d_pendant_pmats,
                d_distal_pmats,
                d_proximal_pmats,
                per_query,
                per_node_pmat,
                d_out);
        } else if (D.rate_cats == 1) {
            CombinedPlacementLoglikPerOpKernelStates4<1><<<grid, block, 0, stream>>>(
                D,
                d_ops,
                d_pendant_pmats,
                d_distal_pmats,
                d_proximal_pmats,
                per_query,
                per_node_pmat,
                d_out);
        } else if (D.rate_cats == 8) {
            CombinedPlacementLoglikPerOpKernelStates4<8><<<grid, block, 0, stream>>>(
                D,
                d_ops,
                d_pendant_pmats,
                d_distal_pmats,
                d_proximal_pmats,
                per_query,
                per_node_pmat,
                d_out);
        } else {
            CombinedPlacementLoglikPerOpKernelGeneric<<<grid, block, 0, stream>>>(
                D,
                d_ops,
                d_pendant_pmats,
                d_distal_pmats,
                d_proximal_pmats,
                per_query,
                per_node_pmat,
                d_out);
        }
    } else {
        CombinedPlacementLoglikPerOpKernelGeneric<<<grid, block, 0, stream>>>(
            D,
            d_ops,
            d_pendant_pmats,
            d_distal_pmats,
            d_proximal_pmats,
            per_query,
            per_node_pmat,
            d_out);
    }
    CUDA_CHECK(cudaGetLastError());
}



} // namespace root_likelihood
