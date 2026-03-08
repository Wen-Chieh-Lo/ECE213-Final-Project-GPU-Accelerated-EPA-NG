#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cub/cub.cuh>
#include "tree_placement.cuh"
#include "../mlipper_util.h"
#include "tree.hpp"
#include "../partial_CUDA/partial_likelihood.cuh"
#include "root_likelihood.cuh"
#include "derivative.cuh"

// Build exp(lambda*t) and its first/second derivatives for a branch into caller-provided buffer.
static __device__ __forceinline__ void build_diagtable_for_branch(
    const DeviceTree& D,
    const NodeOpInfo& op,
    fp_t branch_length,
    fp_t* diag_out)
{
    if (!diag_out || !D.d_lambdas) return;
    const unsigned int total = (unsigned int)(D.rate_cats * D.states);
    const unsigned int idx = threadIdx.x;
    if (idx >= total) return;
    const int rc = idx / D.states;
    const int st = idx - rc * D.states;
    const fp_t lambda = D.d_lambdas[(size_t)rc * D.states + st];
    const fp_t e = fp_exp(lambda * branch_length);
    // Debug: print lambda for each state/rate category to verify values.
    // printf("[diagtable] op(p=%d l=%d r=%d) idx=%u rc=%d st=%d lambda=%e branch=%f exp=%e\n",
    //        op.parent_id, op.left_id, op.right_id, idx, rc, st, lambda, branch_length, e);
    const size_t base = (size_t)idx * 4;
    diag_out[base + 0] = e;
    diag_out[base + 1] = lambda * e;
    diag_out[base + 2] = lambda * lambda * e;
    diag_out[base + 3] = fp_t(0);
}

template<int RATE_CATS>
__device__ void LikelihoodSumtableUpdateKernel(
    DeviceTree D,
    const fp_t* __restrict__ left_clv_base,
    const fp_t* __restrict__ right_clv_base,
    const unsigned* __restrict__ left_scaler_base,
    const unsigned* __restrict__ right_scaler_base,
    unsigned int site_idx,
    fp_t* sumtable);

static __device__ __forceinline__
unsigned int scaler_shift_at_site(
    const DeviceTree& D,
    const unsigned* __restrict__ scaler_base,
    unsigned int site_idx,
    int rate_idx)
{
    if (!scaler_base) return 0u;
    if (D.per_rate_scaling) {
        return scaler_base[(size_t)site_idx * (size_t)D.rate_cats + (size_t)rate_idx];
    }
    return scaler_base[(size_t)site_idx];
}

static __device__ __forceinline__
void core_site_likelihood_derivatives_site(
    const DeviceTree D,
    const int*    __restrict__ invariant_site, // [n_sites], per-site invariant index (or -1)
    const fp_t* __restrict__ invar_proportion,     // [rate_cats] (may be nullptr -> treated as invar_scalar)
    fp_t invar_scalar,
    const fp_t* __restrict__ sumtable,      // base of sumtable: [sites * rate_cats * states]
    const fp_t* __restrict__ lambdas,     // [rate_cats * states * 4]
    const fp_t* __restrict__ pattern_weights,// [sites] or nullptr
    size_t site,
    fp_t* placement_clv, // optional: write per-site log-likelihood
    double& d1_out,
    double& d2_out)
{
    fp_t site_lk0 = fp_t(0);
    fp_t site_lk1 = fp_t(0);
    fp_t site_lk2 = fp_t(0);

    const int inv = invariant_site ? invariant_site[site] : -1;
    const size_t site_stride = (size_t)D.rate_cats * (size_t)D.states;

    // Loop over rate categories
    for (int i = 0; i < D.rate_cats; ++i) {
        const fp_t* sum  = sumtable + site * site_stride + (size_t)i * D.states; // [states]
        const fp_t* diag = lambdas + ((size_t)i * D.states) * 4;              // [states * 4]
        const fp_t* t_freqs = D.d_frequencies;                                   // [states]

        fp_t cat0 = fp_t(0);
        fp_t cat1 = fp_t(0);
        fp_t cat2 = fp_t(0);

        // Loop over states (NOT padded)
        for (int j = 0; j < D.states; ++j) {
            const fp_t s = sum[j];
            cat0 = fp_fma(s, diag[0], cat0);
            cat1 = fp_fma(s, diag[1], cat1);
            cat2 = fp_fma(s, diag[2], cat2);
            diag += 4;
        }

        // account for invariant sites
        const fp_t pinv = invar_proportion ? invar_proportion[i] : invar_scalar;
        if (pinv > fp_t(0)) {
            const fp_t inv_site_lk = (inv < 0) ? fp_t(0) : (t_freqs[inv] * pinv);
            const fp_t non_pinv = fp_t(1) - pinv;
            cat0 = cat0 * non_pinv + inv_site_lk;
            cat1 = cat1 * non_pinv;
            cat2 = cat2 * non_pinv;
        }

        // Use rate weights if available (fallback to 1.0).
        const fp_t w = D.d_rate_weights ? D.d_rate_weights[i] : fp_t(1);
        site_lk0 += cat0 * w;
        site_lk1 += cat1 * w;
        site_lk2 += cat2 * w;
    }

    const fp_t inv_lk0 = (site_lk0 != fp_t(0)) ? (fp_t(1) / site_lk0) : fp_t(0);
    const fp_t d1 = -site_lk1 * inv_lk0;
    const fp_t d2 = d1 * d1 - (site_lk2 * inv_lk0);
    
    const fp_t weight = pattern_weights ? pattern_weights[site] : fp_t(1);
    d1_out = static_cast<double>(weight * d1);
    d2_out = static_cast<double>(weight * d2);
    if (placement_clv) {
        placement_clv[site] = fp_log(site_lk0 > FP_EPS ? site_lk0 : FP_EPS) * weight;
    }
}

template<int RATE_CATS>
__device__ void LikelihoodSumtableUpdateKernel(
        DeviceTree D,
        const fp_t* __restrict__ left_clv_base,
        const fp_t* __restrict__ right_clv_base,
        unsigned int site_idx,
        fp_t* sumtable);


__global__ void LikelihoodDerivativeKernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int op_idx,
    const int*    __restrict__ invariant_site,   // [sites] or nullptr
    const fp_t* __restrict__ invar_proportion, // [rate_cats] or nullptr
    fp_t invar_scalar,                         // used when invar_proportion == nullptr
    fp_t* __restrict__ sumtable,                 // [sites * rate_cats * states] (built in-kernel)
    const fp_t* __restrict__ pattern_weights,  // [sites] or nullptr
    int max_iter,                                // number of Newton updates
    fp_t* new_branch_length,                     // output updated branch length (Newton step; can be per-node array)
    bool proximal_mode,                          // whether this call targets proximal branch optimization
    size_t sumtable_stride,                      // stride between per-op sumtable slices
    fp_t* placement_clv_base,                    // optional per-op placement CLV buffer
    const fp_t* prev_branch_lengths,             // optional per-node initial branch lengths
    const int* active_ops)                       // optional per-op active mask (1=run, 0=skip)
{
    
    if (!sumtable || !ops || op_idx < 0) {
        return;
    }

    const int op_global = op_idx + (int)blockIdx.x;
    if (op_global < 0 || op_global >= D.N) return;
    if (active_ops && active_ops[op_global] == 0) return; // TEMP: skip rejected ops
    const NodeOpInfo op = ops[op_global];
    const bool target_is_left  = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= D.N) return;

    // Pendant mode: left=query, right=midpoint (parent_down * sibling_up).
    // Proximal mode: left=target up (clv_up), right=midpoint rebuilt with query/parent/child.
    const size_t clv_span = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    const size_t scaler_span = D.per_rate_scaling
        ? (D.sites * (size_t)D.rate_cats)
        : D.sites;
    const fp_t* left_base = proximal_mode
        ? (D.d_clv_up ? D.d_clv_up + (size_t)target_id * clv_span : nullptr)
        : D.d_query_clv;
    const fp_t* right_base = D.d_clv_mid
        ? D.d_clv_mid + (size_t)target_id * clv_span
        : nullptr;
    const unsigned* left_scaler_base = proximal_mode
        ? (D.d_site_scaler_up ? (D.d_site_scaler_up + (size_t)target_id * scaler_span) : nullptr)
        : nullptr;
    const unsigned* right_scaler_base = D.d_site_scaler_mid
        ? (D.d_site_scaler_mid + (size_t)target_id * scaler_span)
        : nullptr;

    if (!left_base || !right_base) return;

    fp_t* sumtable_op = sumtable ? sumtable + (size_t)op_global * sumtable_stride : nullptr;
    fp_t* placement_clv = placement_clv_base
        ? placement_clv_base + (size_t)op_global * D.sites
        : nullptr;

    // Shared branch accumulator so all threads use the same value each iteration.
    __shared__ double branch_shared;
    __shared__ int stop_updates;
    __shared__ double bracket_low_shared;
    __shared__ double bracket_high_shared;
    __shared__ double dxmax_shared;
    double init_branch = OPT_BRANCH_LEN_MIN;
    if (prev_branch_lengths) {
        // Use per-node lengths directly (target side).
        init_branch = static_cast<double>(prev_branch_lengths[target_id]);
    } else {
        // Fallback: use half of the current branch length for both pendant/proximal.
        if(proximal_mode) {
            init_branch = 0.5 * static_cast<double>(D.d_blen[target_id]);
        } else{
            init_branch = DEFAULT_BRANCH_LENGTH;
        }
    }
    if (threadIdx.x == 0) {
        if (init_branch < OPT_BRANCH_LEN_MIN) init_branch = OPT_BRANCH_LEN_MIN;
        if (init_branch > OPT_BRANCH_LEN_MAX) init_branch = OPT_BRANCH_LEN_MAX;
        branch_shared = init_branch;
        bracket_low_shared = OPT_BRANCH_LEN_MIN;
        
        if(proximal_mode){
            dxmax_shared = (static_cast<double>(D.d_blen[target_id]) - OPT_BRANCH_XTOL) / static_cast<double>(max_iter);
            // Proximal split can slide across the full target edge; clamping to half
            // the branch length prevents valid optima on the parent-side half.
            bracket_high_shared = static_cast<double>(D.d_blen[target_id]) - OPT_BRANCH_LEN_MIN/10;
        }
        else{
            dxmax_shared = OPT_BRANCH_LEN_MAX / static_cast<double>(max_iter);
            bracket_high_shared = OPT_BRANCH_LEN_MAX;
        }
        
    }
    if (threadIdx.x == 0) stop_updates = 0;
    __syncthreads();

    // Each block handles a single placement op; no cross-block reduction needed.
    extern __shared__ fp_t diag_shared[]; // [rate_cats * states * 4]
    __shared__ double warp_df[32];
    __shared__ double warp_ddf[32];
    double df_local  = 0.0;
    double ddf_local = 0.0;

    const unsigned int tid = threadIdx.x;
    const unsigned int step = blockDim.x;

    // Build sumtable once before derivative iterations.
    for (size_t site = tid; site < D.sites; site += step) {
        const unsigned int site_idx = static_cast<unsigned int>(site);
        if (D.rate_cats == 1) {
            LikelihoodSumtableUpdateKernel<1>(
                D,
                left_base,
                right_base,
                left_scaler_base,
                right_scaler_base,
                site_idx,
                sumtable_op);
        } else if (D.rate_cats == 4) {
            LikelihoodSumtableUpdateKernel<4>(
                D,
                left_base,
                right_base,
                left_scaler_base,
                right_scaler_base,
                site_idx,
                sumtable_op);
        } else if (D.rate_cats == 8) {
            LikelihoodSumtableUpdateKernel<8>(
                D,
                left_base,
                right_base,
                left_scaler_base,
                right_scaler_base,
                site_idx,
                sumtable_op);
        }
    }
    __syncthreads();

    double last_df = 0.0;
    double last_ddf = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (stop_updates) break;
        // Refresh diag table for current branch length (shared for the block).
        const fp_t branch = static_cast<fp_t>(branch_shared);
        build_diagtable_for_branch(D, op, branch, diag_shared);
        __syncthreads();

        df_local  = 0.0;
        ddf_local = 0.0;

        for (size_t site = tid; site < D.sites; site += step) {
            double d1 = 0.0;
            double d2 = 0.0;
            core_site_likelihood_derivatives_site(
                D,
                invariant_site,
                invar_proportion,
                invar_scalar,
                sumtable_op,
                diag_shared,
                pattern_weights,
                site,
                placement_clv,
                d1,
                d2);
            df_local  += d1;
            ddf_local += d2;
        }

        // Warp-level reduction for df/ddf/lk (blockDim.x assumed multiple of 32).
        const unsigned int lane = threadIdx.x & 31;
        const unsigned int warp = threadIdx.x >> 5;
        const unsigned int warp_count = (blockDim.x + 31) >> 5;

        for (int offset = 16; offset > 0; offset >>= 1) {
            df_local  += __shfl_down_sync(0xffffffff, df_local,  offset);
            ddf_local += __shfl_down_sync(0xffffffff, ddf_local, offset);
        }
        if (lane == 0) {
            warp_df[warp]  = df_local;
            warp_ddf[warp] = ddf_local;
        }
        __syncthreads();

        if (warp == 0) {
            double df_block  = (lane < warp_count) ? warp_df[lane]  : 0.0;
            double ddf_block = (lane < warp_count) ? warp_ddf[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                df_block  += __shfl_down_sync(0xffffffff, df_block,  offset);
                ddf_block += __shfl_down_sync(0xffffffff, ddf_block, offset);
            }
            if (lane == 0) {
                last_df    = df_block;
                last_ddf   = ddf_block;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const double tolerance = OPT_BRANCH_XTOL;
            const double ddf_eps = 1e-12;
            if (fabs(last_ddf) < ddf_eps) {
                stop_updates = 1;
            } else if (fabs(last_df) < tolerance) {
                stop_updates = 1;
            } else {
                double dx = 0.0;
                if (last_ddf > 0.0) {
                    if (last_df < 0.0) {
                        bracket_low_shared = branch_shared;
                    } else {
                        bracket_high_shared = branch_shared;
                    }
                    dx = -last_df / last_ddf;
                } else {
                    dx = -last_df / fabs(last_ddf);
                }

                dx = max(min(dx, dxmax_shared), -dxmax_shared);

                if (branch_shared + dx < bracket_low_shared) dx = bracket_low_shared - branch_shared;
                if (branch_shared + dx > bracket_high_shared) dx = bracket_high_shared - branch_shared;

                if (fabs(dx) < tolerance) {
                    stop_updates = 1;
                } else {
                    // if(threadIdx.x == 0 && blockIdx.x == 6)
                    // printf("[DerivativeKernel] op=%d target=%d iter=%d branch=%f next_branch=%f  df=%f ddf=%f upped: lower=%f upper=%f\n",
                    //     op_global, target_id, iter, branch_shared, branch_shared + dx, last_df, last_ddf, bracket_low_shared, bracket_high_shared);
                    double proposed_branch = branch_shared + dx;
                    if (proposed_branch < OPT_BRANCH_LEN_MIN) proposed_branch = OPT_BRANCH_LEN_MIN;
                    if (proposed_branch > OPT_BRANCH_LEN_MAX) proposed_branch = OPT_BRANCH_LEN_MAX;
                    branch_shared = proposed_branch;
                    
                }
            }
        }
        __syncthreads();
        
        if (stop_updates) break;
    }

    if (threadIdx.x == 0) {
        new_branch_length[target_id] = static_cast<fp_t>(branch_shared);
    }

}

template<int RATE_CATS>
__device__ void LikelihoodSumtableUpdateKernel(
        DeviceTree D,
        const fp_t* __restrict__ left_clv_base,
        const fp_t* __restrict__ right_clv_base,
        const unsigned* __restrict__ left_scaler_base,
        const unsigned* __restrict__ right_scaler_base,
        unsigned int site_idx,
        fp_t* sumtable
    ){
        // Generic sumtable builder: multiplies a left/right CLV pair for a given site.
        if (!sumtable || !left_clv_base || !right_clv_base) return;
        if (site_idx >= D.sites) return;

        const size_t span     = (size_t)D.rate_cats * (size_t)D.states;
        const size_t site_off = (size_t)site_idx * span;

        const fp_t* left_clv  = left_clv_base  + site_off;
        const fp_t* right_clv = right_clv_base + site_off;
        fp_t* sumtable_ptr = sumtable + site_off;
        unsigned int rate_shift[RATE_CATS];
        bool rate_has_signal[RATE_CATS];
        unsigned int site_min_shift = 0u;
        bool have_signal = false;

        #pragma unroll
        for(int r = 0; r < RATE_CATS; ++r){
            const fp4_t Qclv = reinterpret_cast<const fp4_t*>(left_clv  + (size_t)r * 4)[0];
            const fp4_t Pclv = reinterpret_cast<const fp4_t*>(right_clv + (size_t)r * 4)[0];
            fp_t* sumtable_row = sumtable_ptr  + (size_t)r * 4;
            
            const fp_t* Vinv = D.d_Vinv;
            const fp_t* V    = D.d_V;
            const fp_t* pi   = D.d_frequencies;

            // query side (tip): Vinv * (pi .* Q)
            const fp4_t piq = make_fp4(pi[0] * Qclv.x, pi[1] * Qclv.y, pi[2] * Qclv.z, pi[3] * Qclv.w);
            const fp_t l0 = fp_dot4(make_fp4(Vinv[0], Vinv[4], Vinv[8], Vinv[12]), piq);
            const fp_t l1 = fp_dot4(make_fp4(Vinv[1], Vinv[5], Vinv[9], Vinv[13]), piq);
            const fp_t l2 = fp_dot4(make_fp4(Vinv[2], Vinv[6], Vinv[10], Vinv[14]), piq);
            const fp_t l3 = fp_dot4(make_fp4(Vinv[3], Vinv[7], Vinv[11], Vinv[15]), piq);

            const fp_t r0 = fp_dot4(make_fp4(V[0], V[1], V[2], V[3]), Pclv);
            const fp_t r1 = fp_dot4(make_fp4(V[4], V[5], V[6], V[7]), Pclv);
            const fp_t r2 = fp_dot4(make_fp4(V[8], V[9], V[10], V[11]), Pclv);
            const fp_t r3 = fp_dot4(make_fp4(V[12], V[13], V[14], V[15]), Pclv);

            sumtable_row[0] = l0 * r0;
            sumtable_row[1] = l1 * r1;
            sumtable_row[2] = l2 * r2;
            sumtable_row[3] = l3 * r3;
            const fp_t row_max = fp_fmax(
                fp_fmax(sumtable_row[0], sumtable_row[1]),
                fp_fmax(sumtable_row[2], sumtable_row[3]));
            const unsigned int total_shift =
                scaler_shift_at_site(D, left_scaler_base, site_idx, r) +
                scaler_shift_at_site(D, right_scaler_base, site_idx, r);
            rate_shift[r] = total_shift;
            rate_has_signal[r] = (row_max > fp_t(0));
            if (rate_has_signal[r]) {
                if (!have_signal || total_shift < site_min_shift) {
                    site_min_shift = total_shift;
                }
                have_signal = true;
            }
        }

        if (!have_signal) return;

        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            if (!rate_has_signal[r]) continue;
            const int diff = (int)rate_shift[r] - (int)site_min_shift;
            if (diff <= 0) continue;
            fp_t* sumtable_row = sumtable_ptr + (size_t)r * 4;
            sumtable_row[0] = fp_ldexp(sumtable_row[0], -diff);
            sumtable_row[1] = fp_ldexp(sumtable_row[1], -diff);
            sumtable_row[2] = fp_ldexp(sumtable_row[2], -diff);
            sumtable_row[3] = fp_ldexp(sumtable_row[3], -diff);
        }
    } 

// Explicit instantiations for the common rate category counts.
template __device__ void LikelihoodSumtableUpdateKernel<1>(DeviceTree, const fp_t*, const fp_t*, const unsigned*, const unsigned*, unsigned int, fp_t*);
template __device__ void LikelihoodSumtableUpdateKernel<4>(DeviceTree, const fp_t*, const fp_t*, const unsigned*, const unsigned*, unsigned int, fp_t*);
template __device__ void LikelihoodSumtableUpdateKernel<8>(DeviceTree, const fp_t*, const fp_t*, const unsigned*, const unsigned*, unsigned int, fp_t*);
