#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "../mlipper_util.h"
#include "partial_likelihood.cuh"
#include "../tree_generation/root_likelihood.cuh"

__device__ inline void scale_double_pow2(double &x, int shift) {
    long long bits = __double_as_longlong(x);
    bits += ((long long)shift << 52);   // exponent += shift
    x = __longlong_as_double(bits);
}

__device__ __forceinline__ unsigned int scaler_slot(
    const DeviceTree& D,
    unsigned int rate_idx)
{
    return D.per_rate_scaling ? rate_idx : 0u;
}

__device__ __forceinline__ unsigned int read_scaler_shift(
    const DeviceTree& D,
    const unsigned int* scaler,
    unsigned int rate_idx)
{
    if (!scaler) return 0u;
    return scaler[scaler_slot(D, rate_idx)];
}

__device__ __forceinline__ void write_scaler_shift(
    const DeviceTree& D,
    unsigned int* scaler,
    unsigned int rate_idx,
    unsigned int value)
{
    if (!scaler) return;
    scaler[scaler_slot(D, rate_idx)] = value;
}

__device__ __forceinline__ unsigned int* scaler_ptr_for_pool(
    const DeviceTree& D,
    uint8_t clv_pool,
    int node_id,
    unsigned int site)
{
    if (clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN)) {
        return down_scaler_ptr(D, node_id, site);
    }
    return up_scaler_ptr(D, node_id, site);
}

__device__ __forceinline__ void add_scaler_shift(
    const DeviceTree& D,
    unsigned int* scaler,
    unsigned int rate_idx,
    unsigned int shift)
{
    if (!scaler || shift == 0) return;
    scaler[scaler_slot(D, rate_idx)] += shift;
}

// ---- Downward per-case helpers (states arbitrary) ----
__device__ __forceinline__ void compute_downward_inner_inner_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    const int sibling_id = target_is_left ? op.right_id : op.left_id;
    if (target_id < 0 || sibling_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const double* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !sibling_up || !target_down) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id  * rate_cats * states * states;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)sibling_id * rate_cats * states * states;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const double* Tmat = target_mat  + (size_t)r * states * states;
        const double* Thalf= target_mat_half + (size_t)r * states * states;
        const double* Smat = sibling_mat + (size_t)r * states * states;
        const double* Ppar = parent_down + (size_t)r * states;
        const double* Psib = sibling_up  + (size_t)r * states;
        double*       Pout = target_down + (size_t)r * states;
        double*       Pmid  = (target_mid && target_up) ? (target_mid + (size_t)r * states) : nullptr;
        const double* Pup   = target_up ? (target_up + (size_t)r * states) : nullptr;
        double*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const double* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k) acc += row[k] * Psib[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const double* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        if (Pmid) {
            // Cache parent_down * sibling_up (after sibling branch matrix) per state.
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const double* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                const double val = par_acc * tgt_acc;
                Pmid[i] = val;
            }
        }

        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            const unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            for (unsigned int j = 0; j < states; ++j)
                scale_double_pow2(Pout[j], shift);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pbase[j], shift);
            }
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pmid[j], shift);
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_inner_tip_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id       = target_is_left ? op.left_id  : op.right_id;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    if (target_id < 0 || sibling_tip_idx < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id * rate_cats * states * states;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * rate_cats * states * states;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const double* Tmat = target_mat  + (size_t)r * states * states;
        const double* Thalf= target_mat_half + (size_t)r * states * states;
        const double* Smat = sibling_mat + (size_t)r * states * states;
        const double* Ppar = parent_down + (size_t)r * states;
        const double* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        double*       Pout = target_down + (size_t)r * states;
        double*       Pmid = (target_up && D.d_clv_mid)
            ? (D.d_clv_mid + (size_t)target_id * per_node + site_off + (size_t)r * states)
            : nullptr;
        double*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const double* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k)
                if (mask & (1u << k)) acc += row[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const double* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const double* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = par_acc * tgt_acc;
            }
        }

        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            const unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            for (unsigned int j = 0; j < states; ++j)
                scale_double_pow2(Pout[j], shift);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pbase[j], shift);
            }
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pmid[j], shift);
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_tip_tip_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    if (target_tip_idx < 0 || sibling_tip_idx < 0 || target_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id * rate_cats * states * states;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * rate_cats * states * states)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * rate_cats * states * states;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const double* Tmat = target_mat  + (size_t)r * states * states;
        const double* Thalf= target_mat_half + (size_t)r * states * states;
        const double* Smat = sibling_mat + (size_t)r * states * states;
        const double* Ppar = parent_down + (size_t)r * states;
        const double* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        double*       Pout = target_down + (size_t)r * states;
        double*       Pmid = (target_up && target_mid) ? (target_mid + (size_t)r * states) : nullptr;
        double*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const double* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k)
                if (mask & (1u << k)) acc += row[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const double* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = acc;
            if (acc > col_scale_max_val) col_scale_max_val = acc;
        }

        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const double* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = par_acc * tgt_acc;
            }
        }

        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            const unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            for (unsigned int j = 0; j < states; ++j)
                scale_double_pow2(Pout[j], shift);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pbase[j], shift);
            }
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pmid[j], shift);
            }
        }
    }
}

__device__ __forceinline__ void compute_downward_tip_inner_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_id      = target_is_left ? op.right_id : op.left_id;
    if (target_tip_idx < 0 || sibling_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const double* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off;
    const double* target_up   = D.d_clv_up   + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off;
    if (!parent_down || !sibling_up || !target_down) return;

    const double* target_mat  = D.d_pmat + (size_t)(target_is_left ? op.left_id : op.right_id) * rate_cats * states * states;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)(target_is_left ? op.left_id : op.right_id) * rate_cats * states * states)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)sibling_id * rate_cats * states * states;
    double*       target_mid  = D.d_clv_mid
        ? (D.d_clv_mid + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off)
        : nullptr;
    double*       mid_base    = D.d_clv_mid_base
        ? (D.d_clv_mid_base + (size_t)(target_is_left ? op.left_id : op.right_id) * per_node + site_off)
        : nullptr;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)target_tip_idx * D.sites;
    const unsigned int tmask = D.d_tipmap[tipchars[site]];

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const double* Tmat = target_mat  + (size_t)r * states * states;
        const double* Thalf= target_mat_half + (size_t)r * states * states;
        const double* Smat = sibling_mat + (size_t)r * states * states;
        const double* Ppar = parent_down + (size_t)r * states;
        const double* Psib = sibling_up  + (size_t)r * states;
        const double* Pup  = target_up ? (target_up + (size_t)r * states) : nullptr;
        double*       Pout = target_down + (size_t)r * states;
        double*       Pmid = (target_up && target_mid) ? (target_mid + (size_t)r * states) : nullptr;
        double*       Pbase = mid_base ? (mid_base + (size_t)r * states) : nullptr;

        double sib_to_parent[64];
        for (unsigned int j = 0; j < states; ++j) {
            const double* row = Smat + j * states;
            double acc = 0.0;
            for (unsigned int k = 0; k < states; ++k) acc += row[k] * Psib[k];
            sib_to_parent[j] = acc;
        }

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            const double* Tcol = Tmat + i;
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j)
                acc += Tcol[j * states] * (Ppar[j] * sib_to_parent[j]);
            Pout[i] = (tmask & (1u << i)) ? acc : 0.0;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
        }

        if (Pmid) {
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j) {
                    Pbase[j] = Ppar[j] * sib_to_parent[j];
                }
            }
            for (unsigned int i = 0; i < states; ++i) {
                const double* Throw = Thalf + i * states;
                double par_acc = 0.0, tgt_acc = 0.0;
                for (unsigned int j = 0; j < states; ++j) {
                    const double pj = Pbase ? Pbase[j] : (Ppar[j] * sib_to_parent[j]);
                    par_acc += Throw[j] * pj;
                    tgt_acc += Throw[j] * Pup[j];
                }
                Pmid[i] = (tmask & (1u << i)) ? (par_acc * tgt_acc) : 0.0;
            }
        }

        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            const unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            for (unsigned int j = 0; j < states; ++j)
                scale_double_pow2(Pout[j], shift);
            if (Pbase) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pbase[j], shift);
            }
            if (Pmid) {
                for (unsigned int j = 0; j < states; ++j)
                    scale_double_pow2(Pmid[j], shift);
            }
        }
    }
}

// ---- Downward specializations for states=4, templated by rate cats ----
template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_inner_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    if (op.left_tip_index >= 0 || op.right_tip_index >= 0) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    const int sibling_id = target_is_left ? op.right_id : op.left_id;
    if (target_id < 0 || sibling_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const double* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !sibling_up || !target_up) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id  * (size_t)RATE_CATS * 16;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)sibling_id * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const double* Tmat = target_mat  + (size_t)r * 16;
        const double* Thalf= target_mat_half + (size_t)r * 16;
        const double* Smat = sibling_mat + (size_t)r * 16;
        const double4 Ppar = reinterpret_cast<const double4*>(parent_down)[r];
        const double4 Psib = reinterpret_cast<const double4*>(sibling_up)[r];
        const double4 Pup  = reinterpret_cast<const double4*>(target_up)[r];
        double*       Pout = target_down + (size_t)r * 4;

        const double sib0 = Smat[0]*Psib.x + Smat[1]*Psib.y + Smat[2]*Psib.z + Smat[3]*Psib.w;
        const double sib1 = Smat[4]*Psib.x + Smat[5]*Psib.y + Smat[6]*Psib.z + Smat[7]*Psib.w;
        const double sib2 = Smat[8]*Psib.x + Smat[9]*Psib.y + Smat[10]*Psib.z+ Smat[11]*Psib.w;
        const double sib3 = Smat[12]*Psib.x+ Smat[13]*Psib.y+ Smat[14]*Psib.z+ Smat[15]*Psib.w;

        const double p0 = Ppar.x * sib0;
        const double p1 = Ppar.y * sib1;
        const double p2 = Ppar.z * sib2;
        const double p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (mid_base) {
            double* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            double* Pmid = target_mid + (size_t)r * 4;
            const double par0 = Thalf[0] * p0 + Thalf[4] * p1 + Thalf[8]  * p2 + Thalf[12] * p3;
            const double par1 = Thalf[1] * p0 + Thalf[5] * p1 + Thalf[9]  * p2 + Thalf[13] * p3;
            const double par2 = Thalf[2] * p0 + Thalf[6] * p1 + Thalf[10] * p2 + Thalf[14] * p3;
            const double par3 = Thalf[3] * p0 + Thalf[7] * p1 + Thalf[11] * p2 + Thalf[15] * p3;

            const double tgt0 = Thalf[0]*Pup.x + Thalf[4]*Pup.y + Thalf[8]*Pup.z  + Thalf[12]*Pup.w;
            const double tgt1 = Thalf[1]*Pup.x + Thalf[5]*Pup.y + Thalf[9]*Pup.z  + Thalf[13]*Pup.w;
            const double tgt2 = Thalf[2]*Pup.x + Thalf[6]*Pup.y + Thalf[10]*Pup.z + Thalf[14]*Pup.w;
            const double tgt3 = Thalf[3]*Pup.x + Thalf[7]*Pup.y + Thalf[11]*Pup.z + Thalf[15]*Pup.w;

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        double col_scale_max_val = fmax(fmax(Pout[0], Pout[1]), fmax(Pout[2], Pout[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_double_pow2(Pout[j], shift);
            }
            if (mid_base) {
                double* Pbase = mid_base + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pbase[j], shift);
                }
            }
            if (target_mid) {
                double* Pmid = target_mid + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pmid[j], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_inner_tip_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id       = target_is_left ? op.left_id  : op.right_id;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    if (target_id < 0 || sibling_tip_idx < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !target_up) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const double* Tmat = target_mat  + (size_t)r * 16;
        const double* Thalf= target_mat_half + (size_t)r * 16;
        const double* Smat = sibling_mat + (size_t)r * 16;
        const double4 Ppar = reinterpret_cast<const double4*>(parent_down)[r];
        const double4 Pup  = reinterpret_cast<const double4*>(target_up)[r];
        double*       Pout = target_down + (size_t)r * 4;

        const double sib0 = ((mask & 1u) ? Smat[0]  : 0.0) + ((mask & 2u) ? Smat[1]  : 0.0) + ((mask & 4u) ? Smat[2]  : 0.0) + ((mask & 8u) ? Smat[3]  : 0.0);
        const double sib1 = ((mask & 1u) ? Smat[4]  : 0.0) + ((mask & 2u) ? Smat[5]  : 0.0) + ((mask & 4u) ? Smat[6]  : 0.0) + ((mask & 8u) ? Smat[7]  : 0.0);
        const double sib2 = ((mask & 1u) ? Smat[8]  : 0.0) + ((mask & 2u) ? Smat[9]  : 0.0) + ((mask & 4u) ? Smat[10] : 0.0) + ((mask & 8u) ? Smat[11] : 0.0);
        const double sib3 = ((mask & 1u) ? Smat[12] : 0.0) + ((mask & 2u) ? Smat[13] : 0.0) + ((mask & 4u) ? Smat[14] : 0.0) + ((mask & 8u) ? Smat[15] : 0.0);

        const double p0 = Ppar.x * sib0;
        const double p1 = Ppar.y * sib1;
        const double p2 = Ppar.z * sib2;
        const double p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (mid_base) {
            double* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            double* Pmid = target_mid + (size_t)r * 4;
            const double par0 = Thalf[0] * p0 + Thalf[4] * p1 + Thalf[8]  * p2 + Thalf[12] * p3;
            const double par1 = Thalf[1] * p0 + Thalf[5] * p1 + Thalf[9]  * p2 + Thalf[13] * p3;
            const double par2 = Thalf[2] * p0 + Thalf[6] * p1 + Thalf[10] * p2 + Thalf[14] * p3;
            const double par3 = Thalf[3] * p0 + Thalf[7] * p1 + Thalf[11] * p2 + Thalf[15] * p3;

            const double tgt0 = Thalf[0]*Pup.x + Thalf[4]*Pup.y + Thalf[8]*Pup.z  + Thalf[12]*Pup.w;
            const double tgt1 = Thalf[1]*Pup.x + Thalf[5]*Pup.y + Thalf[9]*Pup.z  + Thalf[13]*Pup.w;
            const double tgt2 = Thalf[2]*Pup.x + Thalf[6]*Pup.y + Thalf[10]*Pup.z + Thalf[14]*Pup.w;
            const double tgt3 = Thalf[3]*Pup.x + Thalf[7]*Pup.y + Thalf[11]*Pup.z + Thalf[15]*Pup.w;

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        double col_scale_max_val = fmax(fmax(Pout[0], Pout[1]), fmax(Pout[2], Pout[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_double_pow2(Pout[j], shift);
            }
            if (mid_base) {
                double* Pbase = mid_base + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pbase[j], shift);
                }
            }
            if (target_mid) {
                double* Pmid = target_mid + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pmid[j], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_tip_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    const int sibling_id      = target_is_left ? op.right_id : op.left_id;
    if (target_tip_idx < 0 || sibling_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    const double* sibling_up  = D.d_clv_up   + (size_t)sibling_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !sibling_up || !target_up) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)sibling_id * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, sibling_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)target_tip_idx * D.sites;
    const unsigned int tmask = D.d_tipmap[tipchars[site]];

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const double* Tmat  = target_mat  + (size_t)r * 16;
        const double* Thalf = target_mat_half + (size_t)r * 16;
        const double* Smat  = sibling_mat + (size_t)r * 16;
        const double4 Ppar = reinterpret_cast<const double4*>(parent_down)[r];
        const double4 Psib = reinterpret_cast<const double4*>(sibling_up)[r];
        const double4 Pup  = reinterpret_cast<const double4*>(target_up)[r];
        double*       Pout = target_down + (size_t)r * 4;

        const double s0 = Psib.x, s1 = Psib.y, s2 = Psib.z, s3 = Psib.w;
        const double sib0 = Smat[0]*s0 + Smat[1]*s1 + Smat[2]*s2 + Smat[3]*s3;
        const double sib1 = Smat[4]*s0 + Smat[5]*s1 + Smat[6]*s2 + Smat[7]*s3;
        const double sib2 = Smat[8]*s0 + Smat[9]*s1 + Smat[10]*s2 + Smat[11]*s3;
        const double sib3 = Smat[12]*s0+ Smat[13]*s1+ Smat[14]*s2+ Smat[15]*s3;

        const double p0 = Ppar.x * sib0;
        const double p1 = Ppar.y * sib1;
        const double p2 = Ppar.z * sib2;
        const double p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;
        if (!(tmask & 1u)) Pout[0] = 0.0;
        if (!(tmask & 2u)) Pout[1] = 0.0;
        if (!(tmask & 4u)) Pout[2] = 0.0;
        if (!(tmask & 8u)) Pout[3] = 0.0;

        if (mid_base) {
            double* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            double* Pmid = target_mid + (size_t)r * 4;
            const double par0 = Thalf[0] * p0 + Thalf[4] * p1 + Thalf[8]  * p2 + Thalf[12] * p3;
            const double par1 = Thalf[1] * p0 + Thalf[5] * p1 + Thalf[9]  * p2 + Thalf[13] * p3;
            const double par2 = Thalf[2] * p0 + Thalf[6] * p1 + Thalf[10] * p2 + Thalf[14] * p3;
            const double par3 = Thalf[3] * p0 + Thalf[7] * p1 + Thalf[11] * p2 + Thalf[15] * p3;

            const double tgt0 = Thalf[0]*Pup.x + Thalf[4]*Pup.y + Thalf[8]*Pup.z  + Thalf[12]*Pup.w;
            const double tgt1 = Thalf[1]*Pup.x + Thalf[5]*Pup.y + Thalf[9]*Pup.z  + Thalf[13]*Pup.w;
            const double tgt2 = Thalf[2]*Pup.x + Thalf[6]*Pup.y + Thalf[10]*Pup.z + Thalf[14]*Pup.w;
            const double tgt3 = Thalf[3]*Pup.x + Thalf[7]*Pup.y + Thalf[11]*Pup.z + Thalf[15]*Pup.w;

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
            if (!(tmask & 1u)) Pmid[0] = 0.0;
            if (!(tmask & 2u)) Pmid[1] = 0.0;
            if (!(tmask & 4u)) Pmid[2] = 0.0;
            if (!(tmask & 8u)) Pmid[3] = 0.0;
        }

        double col_scale_max_val = fmax(fmax(Pout[0], Pout[1]), fmax(Pout[2], Pout[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_double_pow2(Pout[j], shift);
            }
            if (mid_base) {
                double* Pbase = mid_base + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pbase[j], shift);
                }
            }
            if (target_mid) {
                double* Pmid = target_mid + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pmid[j], shift);
                }
            }
        }
    }
}

// target tip, sibling tip (states=4, rate-specific)
template<int RATE_CATS>
__device__ __forceinline__ void compute_downward_tip_tip_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_tip_idx  = target_is_left ? op.left_tip_index : op.right_tip_index;
    const int sibling_tip_idx = target_is_left ? op.right_tip_index : op.left_tip_index;
    const int target_id       = target_is_left ? op.left_id : op.right_id;
    if (target_tip_idx < 0 || sibling_tip_idx < 0 || target_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    const double* parent_down = D.d_clv_down + (size_t)op.parent_id * per_node + site_off;
    double*       target_down = D.d_clv_down + (size_t)target_id * per_node + site_off;
    double*       target_mid  = D.d_clv_mid ? (D.d_clv_mid + (size_t)target_id * per_node + site_off) : nullptr;
    double*       mid_base    = D.d_clv_mid_base ? (D.d_clv_mid_base + (size_t)target_id * per_node + site_off) : nullptr;
    const double* target_up   = D.d_clv_up   + (size_t)target_id * per_node + site_off;
    if (!parent_down || !target_down || !target_up) return;

    const double* target_mat  = D.d_pmat + (size_t)target_id * (size_t)RATE_CATS * 16;
    const double* target_mat_half = D.d_pmat_mid
        ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16)
        : target_mat;
    const double* sibling_mat = D.d_pmat + (size_t)(target_is_left ? op.right_id : op.left_id) * (size_t)RATE_CATS * 16;
    unsigned int* parent_scaler = down_scaler_ptr(D, op.parent_id, site);
    unsigned int* sibling_scaler = up_scaler_ptr(D, target_is_left ? op.right_id : op.left_id, site);
    unsigned int* target_up_scaler = up_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    const unsigned char* tipchars = D.d_tipchars + (size_t)sibling_tip_idx * D.sites;

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const unsigned int down_inherited =
            read_scaler_shift(D, parent_scaler, r) +
            read_scaler_shift(D, sibling_scaler, r);
        const unsigned int mid_inherited =
            down_inherited + read_scaler_shift(D, target_up_scaler, r);
        write_scaler_shift(D, down_scaler, r, down_inherited);
        write_scaler_shift(D, mid_scaler, r, mid_inherited);

        const unsigned int mask = D.d_tipmap[tipchars[site]];
        const double* Tmat  = target_mat  + (size_t)r * 16;
        const double* Thalf = target_mat_half + (size_t)r * 16;
        const double* Smat  = sibling_mat + (size_t)r * 16;
        const double4 Ppar = reinterpret_cast<const double4*>(parent_down)[r];
        const double4 Pup  = reinterpret_cast<const double4*>(target_up)[r];
        double*       Pout = target_down + (size_t)r * 4;

        const double sib0 = ((mask & 1u) ? Smat[0]  : 0.0) + ((mask & 2u) ? Smat[1]  : 0.0) + ((mask & 4u) ? Smat[2]  : 0.0) + ((mask & 8u) ? Smat[3]  : 0.0);
        const double sib1 = ((mask & 1u) ? Smat[4]  : 0.0) + ((mask & 2u) ? Smat[5]  : 0.0) + ((mask & 4u) ? Smat[6]  : 0.0) + ((mask & 8u) ? Smat[7]  : 0.0);
        const double sib2 = ((mask & 1u) ? Smat[8]  : 0.0) + ((mask & 2u) ? Smat[9]  : 0.0) + ((mask & 4u) ? Smat[10] : 0.0) + ((mask & 8u) ? Smat[11] : 0.0);
        const double sib3 = ((mask & 1u) ? Smat[12] : 0.0) + ((mask & 2u) ? Smat[13] : 0.0) + ((mask & 4u) ? Smat[14] : 0.0) + ((mask & 8u) ? Smat[15] : 0.0);

        const double p0 = Ppar.x * sib0;
        const double p1 = Ppar.y * sib1;
        const double p2 = Ppar.z * sib2;
        const double p3 = Ppar.w * sib3;

        Pout[0] = Tmat[0] * p0 + Tmat[4] * p1 + Tmat[8]  * p2 + Tmat[12] * p3;
        Pout[1] = Tmat[1] * p0 + Tmat[5] * p1 + Tmat[9]  * p2 + Tmat[13] * p3;
        Pout[2] = Tmat[2] * p0 + Tmat[6] * p1 + Tmat[10] * p2 + Tmat[14] * p3;
        Pout[3] = Tmat[3] * p0 + Tmat[7] * p1 + Tmat[11] * p2 + Tmat[15] * p3;

        if (mid_base) {
            double* Pbase = mid_base + (size_t)r * 4;
            Pbase[0] = p0;
            Pbase[1] = p1;
            Pbase[2] = p2;
            Pbase[3] = p3;
        }

        if (target_mid) {
            double* Pmid = target_mid + (size_t)r * 4;
            const double par0 = Thalf[0] * p0 + Thalf[4] * p1 + Thalf[8]  * p2 + Thalf[12] * p3;
            const double par1 = Thalf[1] * p0 + Thalf[5] * p1 + Thalf[9]  * p2 + Thalf[13] * p3;
            const double par2 = Thalf[2] * p0 + Thalf[6] * p1 + Thalf[10] * p2 + Thalf[14] * p3;
            const double par3 = Thalf[3] * p0 + Thalf[7] * p1 + Thalf[11] * p2 + Thalf[15] * p3;

            const double tgt0 = Thalf[0]*Pup.x + Thalf[4]*Pup.y + Thalf[8]*Pup.z  + Thalf[12]*Pup.w;
            const double tgt1 = Thalf[1]*Pup.x + Thalf[5]*Pup.y + Thalf[9]*Pup.z  + Thalf[13]*Pup.w;
            const double tgt2 = Thalf[2]*Pup.x + Thalf[6]*Pup.y + Thalf[10]*Pup.z + Thalf[14]*Pup.w;
            const double tgt3 = Thalf[3]*Pup.x + Thalf[7]*Pup.y + Thalf[11]*Pup.z + Thalf[15]*Pup.w;

            Pmid[0] = par0 * tgt0;
            Pmid[1] = par1 * tgt1;
            Pmid[2] = par2 * tgt2;
            Pmid[3] = par3 * tgt3;
        }

        double col_scale_max_val = fmax(fmax(Pout[0], Pout[1]), fmax(Pout[2], Pout[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, down_scaler, r, shift);
            add_scaler_shift(D, mid_scaler, r, shift);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scale_double_pow2(Pout[j], shift);
            }
            if (mid_base) {
                double* Pbase = mid_base + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pbase[j], shift);
                }
            }
            if (target_mid) {
                double* Pmid = target_mid + (size_t)r * 4;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pmid[j], shift);
                }
            }
        }
    }
}

// ===== Per-site computations =====
template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_tip_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    // Compute parent CLV from two tips directly (no lookup).
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int lmask = D.d_tipmap[left_tip[site]];
    const unsigned int rmask = D.d_tipmap[right_tip[site]];

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        double* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            double* out = lclv + site_off + (size_t)r * 4;
            out[0] = (lmask & 1u) ? 1.0 : 0.0;
            out[1] = (lmask & 2u) ? 1.0 : 0.0;
            out[2] = (lmask & 4u) ? 1.0 : 0.0;
            out[3] = (lmask & 8u) ? 1.0 : 0.0;
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        double* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            double* out = rclv + site_off + (size_t)r * 4;
            out[0] = (rmask & 1u) ? 1.0 : 0.0;
            out[1] = (rmask & 2u) ? 1.0 : 0.0;
            out[2] = (rmask & 4u) ? 1.0 : 0.0;
            out[3] = (rmask & 8u) ? 1.0 : 0.0;
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    const double* Lbase = D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const double* Rbase = D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;
    double* parent_pool = clv_write_pool_base<double>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const double* Lmat = Lbase + (size_t)r * 16;
        const double* Rmat = Rbase + (size_t)r * 16;
        double* pout = parent_pool + parent_off + (size_t)r * 4;

        double maxv = 0.0;
        // parent state j
        for (int j = 0; j < 4; ++j) {
            double left_term = 0.0;
            double right_term = 0.0;
            // sum over allowed tip states
            if (lmask & 1u) left_term  += Lmat[j * 4 + 0];
            if (lmask & 2u) left_term  += Lmat[j * 4 + 1];
            if (lmask & 4u) left_term  += Lmat[j * 4 + 2];
            if (lmask & 8u) left_term  += Lmat[j * 4 + 3];

            if (rmask & 1u) right_term += Rmat[j * 4 + 0];
            if (rmask & 2u) right_term += Rmat[j * 4 + 1];
            if (rmask & 4u) right_term += Rmat[j * 4 + 2];
            if (rmask & 8u) right_term += Rmat[j * 4 + 3];

            double v = left_term * right_term;
            pout[j] = v;
            if (v > maxv) maxv = v;
        }

        if (site_scaler_ptr) {
            int expv;
            frexp(maxv, &expv);
            if (expv < SCALE_THRESHOLD_EXPONENT) {
                unsigned int shift = SCALE_THRESHOLD_EXPONENT - expv;
                add_scaler_shift(D, site_scaler_ptr, r, shift);

                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    scale_double_pow2(pout[s], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_tip_site_ratecat_nolookup(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    // on-the-fly helper (no lookup table)
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int j = (unsigned int)left_tip[site];
    const unsigned int k = (unsigned int)right_tip[site];

    const unsigned int jmask_base = D.d_tipmap[j];
    const unsigned int kmask_base = D.d_tipmap[k];

    const double* __restrict__ jmat_base =
        D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const double* __restrict__ kmat_base =
        D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        double* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            double* out = lclv + site_off + (size_t)r * 4;
            unsigned int m = jmask_base;
            out[0] = (m & 1u) ? 1.0 : 0.0;
            out[1] = (m & 2u) ? 1.0 : 0.0;
            out[2] = (m & 4u) ? 1.0 : 0.0;
            out[3] = (m & 8u) ? 1.0 : 0.0;
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        double* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            double* out = rclv + site_off + (size_t)r * 4;
            unsigned int m = kmask_base;
            out[0] = (m & 1u) ? 1.0 : 0.0;
            out[1] = (m & 2u) ? 1.0 : 0.0;
            out[2] = (m & 4u) ? 1.0 : 0.0;
            out[3] = (m & 8u) ? 1.0 : 0.0;
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    double* parent_pool = clv_write_pool_base<double>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    double* __restrict__ dst = parent_pool + parent_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const double* __restrict__ jmat = jmat_base + (size_t)r * 4 * 4;
        const double* __restrict__ kmat = kmat_base + (size_t)r * 4 * 4;
        double* __restrict__ Pout = dst + (size_t)r * 4;

        double col_scale_max_val = 0.0;

        const double* Lrow = jmat;
        const double* Rrow = kmat;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            double termj = 0.0;
            double termk = 0.0;
            unsigned int jmask = jmask_base;
            unsigned int kmask = kmask_base;

            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (jmask & 1u) termj += Lrow[m];
                if (kmask & 1u) termk += Rrow[m];
                jmask >>= 1;
                kmask >>= 1;
            }

            Pout[i] = termj * termk;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];

            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            double* pout = Pout;
            double maxv = fmax(fmax(pout[0], pout[1]),
                               fmax(pout[2], pout[3]));
            int expv;
            frexp(maxv, &expv);
            if (expv < SCALE_THRESHOLD_EXPONENT) {
                unsigned int shift = SCALE_THRESHOLD_EXPONENT - expv;
                add_scaler_shift(D, site_scaler_ptr, r, shift);

                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    scale_double_pow2(pout[s], shift);
            }
        }
    }
}

__device__ __forceinline__ void compute_tip_tip_site_4_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)4 * (size_t)D.rate_cats;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int j = (unsigned int)left_tip[site];
    const unsigned int k = (unsigned int)right_tip[site];

    const unsigned int jmask_base = D.d_tipmap[j];
    const unsigned int kmask_base = D.d_tipmap[k];

    const double* __restrict__ jmat_base =
        D.d_pmat + (size_t)op.left_id  * D.rate_cats * 4 * 4;
    const double* __restrict__ kmat_base =
        D.d_pmat + (size_t)op.right_id * D.rate_cats * 4 * 4;

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        double* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        for (int r = 0; r < D.rate_cats; ++r) {
            double* out = lclv + site_off + (size_t)r * 4;
            unsigned int m = jmask_base;
            out[0] = (m & 1u) ? 1.0 : 0.0;
            out[1] = (m & 2u) ? 1.0 : 0.0;
            out[2] = (m & 4u) ? 1.0 : 0.0;
            out[3] = (m & 8u) ? 1.0 : 0.0;
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        double* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        for (int r = 0; r < D.rate_cats; ++r) {
            double* out = rclv + site_off + (size_t)r * 4;
            unsigned int m = kmask_base;
            out[0] = (m & 1u) ? 1.0 : 0.0;
            out[1] = (m & 2u) ? 1.0 : 0.0;
            out[2] = (m & 4u) ? 1.0 : 0.0;
            out[3] = (m & 8u) ? 1.0 : 0.0;
        }
    }

    const size_t parent_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    double* parent_pool = clv_write_pool_base<double>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    double* __restrict__ dst = parent_pool + parent_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, (unsigned int)D.rate_cats);

    for (int r = 0; r < D.rate_cats; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, 0u);
        const double* __restrict__ jmat = jmat_base + (size_t)r * 4 * 4;
        const double* __restrict__ kmat = kmat_base + (size_t)r * 4 * 4;
        double* __restrict__ Pout = dst + (size_t)r * 4;

        double col_scale_max_val = 0.0;

        const double* Lrow = jmat;
        const double* Rrow = kmat;
        for (int i = 0; i < 4; ++i) {
            double termj = 0.0;
            double termk = 0.0;
            unsigned int jmask = jmask_base;
            unsigned int kmask = kmask_base;

            for (int m = 0; m < 4; ++m) {
                if (jmask & 1u) termj += Lrow[m];
                if (kmask & 1u) termk += Rrow[m];
                jmask >>= 1;
                kmask >>= 1;
            }

            Pout[i] = termj * termk;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];

            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            double maxv = fmax(fmax(Pout[0], Pout[1]),
                               fmax(Pout[2], Pout[3]));
            int expv;
            frexp(maxv, &expv);
            if (expv < SCALE_THRESHOLD_EXPONENT) {
                unsigned int shift = SCALE_THRESHOLD_EXPONENT - expv;
                site_scaler_ptr[r] += shift;
                for (int s = 0; s < 4; ++s) {
                    scale_double_pow2(Pout[s], shift);
                }
            }
        }
    }
}

__device__ __forceinline__ void compute_tip_tip_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span     = (size_t)states * rate_cats;
    const size_t per_node = per_node_span(D);

    const unsigned char* left_tip  = D.d_tipchars + (size_t)op.left_tip_index  * D.sites;
    const unsigned char* right_tip = D.d_tipchars + (size_t)op.right_tip_index * D.sites;

    const unsigned int lmask = D.d_tipmap[left_tip[site]];
    const unsigned int rmask = D.d_tipmap[right_tip[site]];

    // Ensure tip nodes have CLV-up initialized for downstream use.
    const size_t site_off = (size_t)site * span;
    if (op.left_id >= 0 && D.d_clv_up) {
        double* lclv = D.d_clv_up + (size_t)op.left_id * per_node;
        for (unsigned int r = 0; r < rate_cats; ++r) {
            double* out = lclv + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (lmask & (1u << s)) ? 1.0 : 0.0;
            }
        }
    }
    if (op.right_id >= 0 && D.d_clv_up) {
        double* rclv = D.d_clv_up + (size_t)op.right_id * per_node;
        for (unsigned int r = 0; r < rate_cats; ++r) {
            double* out = rclv + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (rmask & (1u << s)) ? 1.0 : 0.0;
            }
        }
    }

    const double* Lbase = D.d_pmat + (size_t)op.left_id  * rate_cats * states * states;
    const double* Rbase = D.d_pmat + (size_t)op.right_id * rate_cats * states * states;

    const size_t dst_off = (size_t)op.parent_id * per_node + (size_t)site * span;
    double* parent_pool = clv_write_pool_base<double>(D, op);
    if (!parent_pool) return; // placeholder until preorder input logic is defined
    double* Pout = parent_pool + dst_off;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const double* Lmat = Lbase + (size_t)r * states * states;
        const double* Rmat = Rbase + (size_t)r * states * states;
        double* out_r = Pout + (size_t)r * states;

        double maxv = 0.0;
        for (unsigned int j = 0; j < states; ++j) {
            double left_term = 0.0;
            double right_term = 0.0;
            for (unsigned int k = 0; k < states; ++k) {
                if (lmask & (1u << k)) left_term  += Lmat[j * states + k];
                if (rmask & (1u << k)) right_term += Rmat[j * states + k];
            }
            double v = left_term * right_term;
            out_r[j] = v;
            if (v > maxv) maxv = v;
        }

        if (site_scaler_ptr) {
            int expv;
            frexp(maxv, &expv);
            if (expv < SCALE_THRESHOLD_EXPONENT) {
                unsigned int shift = SCALE_THRESHOLD_EXPONENT - expv;
                site_scaler_ptr[r] += shift;
                for (unsigned int s = 0; s < states; ++s) {
                    scale_double_pow2(out_r[s], shift);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_tip_inner_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)4 * RATE_CATS;
    const size_t per_node = per_node_span(D);

    const bool tip_on_left = op.left_tip_index >= 0;
    const int  tip_index   = tip_on_left ? op.left_tip_index  : op.right_tip_index;
    const int  inner_id    = tip_on_left ? op.right_id : op.left_id;
    const int  tip_node_id = tip_on_left ? op.left_id  : op.right_id;

    const unsigned char* d_left_tip = D.d_tipchars + (size_t)tip_index * D.sites;
    const double* d_right_clv = clv_read_ptr_for_node<const double>(D, op, inner_id);
    double* parent_clv = clv_write_ptr_for_node<double>(D, op, op.parent_id);
    if (!d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined

    const double* d_Lmat = D.d_pmat + (size_t)tip_node_id * RATE_CATS * 4 * 4;
    const double* d_Rmat = D.d_pmat + (size_t)inner_id * RATE_CATS * 4 * 4;

    const size_t site_off = (size_t)site * span;
    const unsigned int tmask = D.d_tipmap[d_left_tip[site]];

    // Write tip CLV into UP pool for downstream use.
    if (D.d_clv_up && tip_node_id >= 0) {
        double* tip_up = D.d_clv_up + (size_t)tip_node_id * per_node;
        #pragma unroll
        for (int r = 0; r < RATE_CATS; ++r) {
            double* out = tip_up + site_off + (size_t)r * 4;
            out[0] = (tmask & 1u) ? 1.0 : 0.0;
            out[1] = (tmask & 2u) ? 1.0 : 0.0;
            out[2] = (tmask & 4u) ? 1.0 : 0.0;
            out[3] = (tmask & 8u) ? 1.0 : 0.0;
        }
    }

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);
    unsigned int* inner_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, inner_id, site);

    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, read_scaler_shift(D, inner_scaler, r));
        const double* Lmat = d_Lmat + (size_t)r * 4 * 4;
        const double* Rmat = d_Rmat + (size_t)r * 4 * 4;
        const double* Rclv = d_right_clv + site_off + (size_t)r * 4;
        double* Pout = parent_clv + site_off + (size_t)r * 4;
        double col_scale_max_val = 0.0;

        const double r0 = Rclv[0];
        const double r1 = Rclv[1];
        const double r2 = Rclv[2];
        const double r3 = Rclv[3];

        const double* Lrow = Lmat;
        const double* Rrow = Rmat;
        for (int i = 0; i < 4; ++i) {
            double lefterm = 0.0;
            unsigned int lstate = tmask;
            if (lstate & 1u) lefterm += Lrow[0];
            if (lstate & 2u) lefterm += Lrow[1];
            if (lstate & 4u) lefterm += Lrow[2];
            if (lstate & 8u) lefterm += Lrow[3];

            double righterm = Rrow[0] * r0 + Rrow[1] * r1 + Rrow[2] * r2 + Rrow[3] * r3;
            Pout[i] = lefterm * righterm;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
            Lrow += 4;
            Rrow += 4;
        }

        if (site_scaler_ptr) {
            int scaling_exponent;
            frexp(col_scale_max_val, &scaling_exponent);
            if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
                add_scaler_shift(D, site_scaler_ptr, r, SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    scale_double_pow2(Pout[i], SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                }
            }
        }
    }
}

__device__ __forceinline__ void compute_tip_inner_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span     = (size_t)states * rate_cats;
    const size_t per_node = per_node_span(D);

    const bool tip_on_left = op.left_tip_index >= 0;
    const int  tip_index   = tip_on_left ? op.left_tip_index  : op.right_tip_index;
    const int  inner_id    = tip_on_left ? op.right_id : op.left_id;
    const int  tip_node_id = tip_on_left ? op.left_id  : op.right_id;

    const unsigned char* d_left_tip = D.d_tipchars + (size_t)tip_index * D.sites;
    const double* d_right_clv = clv_read_ptr_for_node<const double>(D, op, inner_id);
    double* parent_clv = clv_write_ptr_for_node<double>(D, op, op.parent_id);
    if (!d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined

    const double* d_Lmat = D.d_pmat + (size_t)tip_node_id * D.rate_cats * states * states;
    const double* d_Rmat = D.d_pmat + (size_t)inner_id * D.rate_cats * states * states;

    const size_t site_off = (size_t)site * span;
    const unsigned int tmask = D.d_tipmap[d_left_tip[site]];

    // Write tip CLV into UP pool for downstream use.
    if (D.d_clv_up && tip_node_id >= 0) {
        double* tip_up = D.d_clv_up + (size_t)tip_node_id * per_node;
        for (unsigned int r = 0; r < rate_cats; ++r) {
            double* out = tip_up + site_off + (size_t)r * states;
            for (unsigned int s = 0; s < states; ++s) {
                out[s] = (tmask & (1u << s)) ? 1.0 : 0.0;
            }
        }
    }

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);
    unsigned int* inner_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, inner_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        write_scaler_shift(D, site_scaler_ptr, r, read_scaler_shift(D, inner_scaler, r));
        double col_scale_max_val = 0.0;
        int scaling_exponent;
        const double* Lmat = d_Lmat + (size_t)r * states * states;
        const double* Rmat = d_Rmat + (size_t)r * states * states;
        const double* Rclv = d_right_clv + site_off + (size_t)r * states;
        double* Pout = parent_clv + site_off + (size_t)r * states;

        const double* Lrow = Lmat;
        const double* Rrow = Rmat;
        for (unsigned int i = 0; i < states; ++i) {
            double lefterm = 0.0, righterm = 0.0;
            unsigned int lstate = tmask;
            for (unsigned int j = 0; j < states; ++j) {
                if (lstate & 1u) lefterm += Lrow[j];
                righterm += Rrow[j] * Rclv[j];
                lstate >>= 1;
            }
            Pout[i] = lefterm * righterm;
            if (Pout[i] > col_scale_max_val) col_scale_max_val = Pout[i];
            Lrow += states;
            Rrow += states;
        }
        if (site_scaler_ptr) {
            frexp(col_scale_max_val, &scaling_exponent);
            if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
                site_scaler_ptr[r] += SCALE_THRESHOLD_EXPONENT - scaling_exponent;
                for (unsigned int i = 0; i < states; ++i) {
                    scale_double_pow2(Pout[i], SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                }
            }
        }
    }
}

template<int RATE_CATS>
__device__ __forceinline__ void compute_inner_inner_site_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const size_t span     = (size_t)RATE_CATS * 4;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * span;

    const double* d_left_clv  = clv_read_ptr_for_node<const double>(D, op, op.left_id);
    const double* d_right_clv = clv_read_ptr_for_node<const double>(D, op, op.right_id);

    double* parent_clv = clv_write_ptr_for_node<double>(D, op, op.parent_id);
    if (!d_left_clv || !d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined
    const double* d_left_mat  = D.d_pmat + (size_t)op.left_id  * RATE_CATS * 4 * 4;
    const double* d_right_mat = D.d_pmat + (size_t)op.right_id * RATE_CATS * 4 * 4;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, RATE_CATS);
    unsigned int* left_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.left_id, site);
    unsigned int* right_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.right_id, site);

    for (int r = 0; r < RATE_CATS; ++r) {
        write_scaler_shift(
            D,
            site_scaler_ptr,
            r,
            read_scaler_shift(D, left_scaler, r) +
            read_scaler_shift(D, right_scaler, r));
        const double* Lclv = d_left_clv  + site_off + (size_t)r * 4;
        const double* Rclv = d_right_clv + site_off + (size_t)r * 4;

        

        const double* Lmat = d_left_mat  + (size_t)r * 4 * 4;
        const double* Rmat = d_right_mat + (size_t)r * 4 * 4;

        double* Pout = parent_clv + site_off + (size_t)r * 4;
        double col_scale_max_val = 0.0;

        const double l0 = Lclv[0];
        const double l1 = Lclv[1];
        const double l2 = Lclv[2];
        const double l3 = Lclv[3];
        const double r0 = Rclv[0];
        const double r1 = Rclv[1];
        const double r2 = Rclv[2];
        const double r3 = Rclv[3];

        const double lt0 = Lmat[0]*l0 + Lmat[1]*l1 + Lmat[2]*l2 + Lmat[3]*l3;
        const double lt1 = Lmat[4]*l0 + Lmat[5]*l1 + Lmat[6]*l2 + Lmat[7]*l3;
        const double lt2 = Lmat[8]*l0 + Lmat[9]*l1 + Lmat[10]*l2 + Lmat[11]*l3;
        const double lt3 = Lmat[12]*l0 + Lmat[13]*l1 + Lmat[14]*l2 + Lmat[15]*l3;

        const double rt0 = Rmat[0]*r0 + Rmat[1]*r1 + Rmat[2]*r2 + Rmat[3]*r3;
        const double rt1 = Rmat[4]*r0 + Rmat[5]*r1 + Rmat[6]*r2 + Rmat[7]*r3;
        const double rt2 = Rmat[8]*r0 + Rmat[9]*r1 + Rmat[10]*r2 + Rmat[11]*r3;
        const double rt3 = Rmat[12]*r0 + Rmat[13]*r1 + Rmat[14]*r2 + Rmat[15]*r3;

        Pout[0] = lt0 * rt0;
        Pout[1] = lt1 * rt1;
        Pout[2] = lt2 * rt2;
        Pout[3] = lt3 * rt3;

        col_scale_max_val = fmax(fmax(Pout[0], Pout[1]), fmax(Pout[2], Pout[3]));

        if (site_scaler_ptr) {
            int scaling_exponent;
            frexp(col_scale_max_val, &scaling_exponent);
            if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
                add_scaler_shift(D, site_scaler_ptr, r, SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scale_double_pow2(Pout[j], SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                }
            }
        }
        
    }
    
}

// Midpoint helper for down pass (states=4): parent.down + sibling.up -> mid CLV.
template<int RATE_CATS>
__device__ __forceinline__ void compute_midpoint_inner_inner_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    bool proximal_mode,
    int op_idx)
{
    if (!D.d_clv_mid) return;
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    const int sibling_id = target_is_left ? op.right_id : op.left_id;
    if (op.parent_id < 0 || target_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;

    double*       target_mid = D.d_clv_mid + (size_t)target_id * per_node + site_off;
    const double* parent_down = D.d_clv_down
        ? D.d_clv_down + (size_t)op.parent_id * per_node + site_off
        : nullptr;
    const double* sibling_up  = D.d_clv_up
        ? D.d_clv_up + (size_t)sibling_id * per_node + site_off
        : nullptr;
    const double* mid_base    = D.d_clv_mid_base
        ? D.d_clv_mid_base + (size_t)target_id * per_node + site_off
        : nullptr;
    // proximal_mode uses query CLV as the "upper" side; pendant uses target_up.
    const double* target_up  = proximal_mode
        ? (D.d_query_clv ? (D.d_query_clv + site_off) : nullptr)
        : (D.d_clv_up    ? (D.d_clv_up   + (size_t)target_id * per_node + site_off) : nullptr);
    if (!target_up || !parent_down || !sibling_up) return;
    const double* target_mat = nullptr;
    if (proximal_mode && D.d_query_pmat) {
        target_mat = D.d_query_pmat + (size_t)op_idx * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid_prox) {
        target_mat = D.d_pmat_mid_prox + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid) {
        target_mat = D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else {
        // Fall back to half-branch pmats at minimum; avoid full-length pmats here.
        target_mat = D.d_pmat_mid ? (D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16) : nullptr;
    }
    const double* parent_mat = nullptr;
    if (D.d_pmat_mid_dist) {
        parent_mat = D.d_pmat_mid_dist + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else if (D.d_pmat_mid) {
        parent_mat = D.d_pmat_mid + (size_t)target_id * (size_t)RATE_CATS * 16;
    } else {
        // Avoid using full-length pmats on parent side in proximal mode.
        parent_mat = nullptr;
    }
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);
    unsigned int* down_scaler = down_scaler_ptr(D, target_id, site);
    unsigned int* target_up_scaler = proximal_mode ? nullptr : up_scaler_ptr(D, target_id, site);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        unsigned int inherited_shift = read_scaler_shift(D, down_scaler, r);
        if (target_up_scaler) {
            inherited_shift += read_scaler_shift(D, target_up_scaler, r);
        }
        write_scaler_shift(D, mid_scaler, r, inherited_shift);

        const double* Mtarget = target_mat + (size_t)r * 16;
        const double* Mparent = parent_mat + (size_t)r * 16;
        const double4 Pup   = reinterpret_cast<const double4*>(target_up   + (size_t)r * 4)[0];
        double*       Pmid  = target_mid + (size_t)r * 4;
        const double* Pbase = mid_base ? (mid_base + (size_t)r * 4) : nullptr;

        const double src0 = Pbase[0];
        const double src1 = Pbase[1];
        const double src2 = Pbase[2];
        const double src3 = Pbase[3];

        // propagate parent_down half-branch
        const double par0 = Mparent[0]*src0 + Mparent[1]*src1 + Mparent[2]*src2 + Mparent[3]*src3;
        const double par1 = Mparent[4]*src0 + Mparent[5]*src1 + Mparent[6]*src2 + Mparent[7]*src3;
        const double par2 = Mparent[8]*src0 + Mparent[9]*src1 + Mparent[10]*src2+ Mparent[11]*src3;
        const double par3 = Mparent[12]*src0+ Mparent[13]*src1+ Mparent[14]*src2+ Mparent[15]*src3;

        // propagate target up branch
        const double tgt0 = Mtarget[0]*Pup.x + Mtarget[1]*Pup.y + Mtarget[2]*Pup.z + Mtarget[3]*Pup.w;
        const double tgt1 = Mtarget[4]*Pup.x + Mtarget[5]*Pup.y + Mtarget[6]*Pup.z + Mtarget[7]*Pup.w;
        const double tgt2 = Mtarget[8]*Pup.x + Mtarget[9]*Pup.y + Mtarget[10]*Pup.z+ Mtarget[11]*Pup.w;
        const double tgt3 = Mtarget[12]*Pup.x+ Mtarget[13]*Pup.y+ Mtarget[14]*Pup.z+ Mtarget[15]*Pup.w;

        Pmid[0] = par0 * tgt0;
        Pmid[1] = par1 * tgt1;
        Pmid[2] = par2 * tgt2;
        Pmid[3] = par3 * tgt3;
        
        double col_scale_max_val = fmax(fmax(Pmid[0], Pmid[1]), fmax(Pmid[2], Pmid[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, mid_scaler, r, shift);
            Pmid[0] = ldexp(Pmid[0], shift);
            Pmid[1] = ldexp(Pmid[1], shift);
            Pmid[2] = ldexp(Pmid[2], shift);
            Pmid[3] = ldexp(Pmid[3], shift);
        }

    }
}

// Explicit instantiations for placement usage.
template __device__ void compute_midpoint_inner_inner_ratecat<1>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int);
template __device__ void compute_midpoint_inner_inner_ratecat<4>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int);
template __device__ void compute_midpoint_inner_inner_ratecat<8>(
    const DeviceTree&,
    const NodeOpInfo&,
    unsigned int,
    bool,
    int);

// Combine query CLV (after applying pendant PMAT) with midpoint CLV for placement.
template<int RATE_CATS>
__device__ __forceinline__ void combine_query_midpoint_ratecat(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    int op_idx)
{
    if (!D.d_query_clv || !D.d_query_pmat || !D.d_clv_mid) return;
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    if (target_id < 0) return;

    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)RATE_CATS * 4;
    const size_t per_query = (size_t)RATE_CATS * 16;

    double*       target_mid = D.d_clv_mid + (size_t)target_id * per_node + site_off;
    const double* query_clv  = D.d_query_clv + site_off;
    const double* query_mat  = D.d_query_pmat + (size_t)op_idx * per_query;
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    #pragma unroll
    for (int r = 0; r < RATE_CATS; ++r) {
        const double* M = query_mat + (size_t)r * 16;
        const double4 Q = reinterpret_cast<const double4*>(query_clv)[r];
        double*       Pmid = target_mid + (size_t)r * 4;

        const double q0 = Q.x, q1 = Q.y, q2 = Q.z, q3 = Q.w;
        const double t0 = M[0]*q0 + M[1]*q1 + M[2]*q2 + M[3]*q3;
        const double t1 = M[4]*q0 + M[5]*q1 + M[6]*q2 + M[7]*q3;
        const double t2 = M[8]*q0 + M[9]*q1 + M[10]*q2 + M[11]*q3;
        const double t3 = M[12]*q0+ M[13]*q1+ M[14]*q2+ M[15]*q3;

        Pmid[0] *= t0;
        Pmid[1] *= t1;
        Pmid[2] *= t2;
        Pmid[3] *= t3;

        double col_scale_max_val = fmax(fmax(Pmid[0], Pmid[1]), fmax(Pmid[2], Pmid[3]));
        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, mid_scaler, r, shift);
            Pmid[0] = ldexp(Pmid[0], shift);
            Pmid[1] = ldexp(Pmid[1], shift);
            Pmid[2] = ldexp(Pmid[2], shift);
            Pmid[3] = ldexp(Pmid[3], shift);
        }
    }
}

// Generic (any states/rate count) query-midpoint combination.
__device__ __forceinline__ void combine_query_midpoint_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    int op_idx)
{
    if (!D.d_query_clv || !D.d_query_pmat || !D.d_clv_mid) return;
    if (op.clv_pool != static_cast<uint8_t>(CLV_POOL_DOWN)) return;

    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id  = target_is_left ? op.left_id  : op.right_id;
    if (target_id < 0) return;

    const unsigned int states    = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * (size_t)states * (size_t)rate_cats;
    const size_t per_query = (size_t)rate_cats * (size_t)states * (size_t)states;

    double*       target_mid = D.d_clv_mid + (size_t)target_id * per_node + site_off;
    const double* query_clv  = D.d_query_clv + site_off;
    const double* query_mat  = D.d_query_pmat + (size_t)op_idx * per_query;
    unsigned int* mid_scaler = mid_scaler_ptr(D, target_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        const double* M = query_mat + (size_t)r * states * states;
        const double* Q = query_clv  + (size_t)r * states;
        double*       Pmid = target_mid + (size_t)r * states;

        double col_scale_max_val = 0.0;
        for (unsigned int i = 0; i < states; ++i) {
            double acc = 0.0;
            for (unsigned int j = 0; j < states; ++j) {
                acc += M[i * states + j] * Q[j];
            }
            Pmid[i] *= acc;
            if (Pmid[i] > col_scale_max_val) col_scale_max_val = Pmid[i];
        }

        int scaling_exponent;
        frexp(col_scale_max_val, &scaling_exponent);
        if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
            unsigned int shift = SCALE_THRESHOLD_EXPONENT - scaling_exponent;
            add_scaler_shift(D, mid_scaler, r, shift);
            for (unsigned int j = 0; j < states; ++j) {
                Pmid[j] = ldexp(Pmid[j], shift);
            }
        }
    }
}

// Explicit instantiations for the common rate category counts.
template __device__ void combine_query_midpoint_ratecat<1>(const DeviceTree&, const NodeOpInfo&, unsigned int, int);
template __device__ void combine_query_midpoint_ratecat<4>(const DeviceTree&, const NodeOpInfo&, unsigned int, int);
template __device__ void combine_query_midpoint_ratecat<8>(const DeviceTree&, const NodeOpInfo&, unsigned int, int);

// Site-parallel kernel: merge query CLV (after pendant PMAT) into midpoint CLV.
__global__ void UpdateMidpointWithQueryKernel(
    DeviceTree D,
    const NodeOpInfo* d_ops,
    int op_idx)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (!d_ops || op_idx < 0 || tid >= D.sites) return;
    const NodeOpInfo op = d_ops[op_idx];

    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1: combine_query_midpoint_ratecat<1>(D, op, tid, op_idx); break;
            case 4: combine_query_midpoint_ratecat<4>(D, op, tid, op_idx); break;
            case 8: combine_query_midpoint_ratecat<8>(D, op, tid, op_idx); break;
            default: combine_query_midpoint_generic(D, op, tid, op_idx); break;
        }
    } else {
        combine_query_midpoint_generic(D, op, tid, op_idx);
    }
}

__device__ __forceinline__ void compute_inner_inner_site_generic(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site)
{
    const unsigned int states = (unsigned int)D.states;
    const unsigned int rate_cats = (unsigned int)D.rate_cats;
    const size_t span = (size_t)states * (size_t)rate_cats;
    const size_t per_node = per_node_span(D);
    const size_t site_off = (size_t)site * span;

    const double* d_left_clv  = clv_read_ptr_for_node<const double>(D, op, op.left_id);
    const double* d_right_clv = clv_read_ptr_for_node<const double>(D, op, op.right_id);
    double* parent_clv = clv_write_ptr_for_node<double>(D, op, op.parent_id);
    if (!d_left_clv || !d_right_clv || !parent_clv) return; // placeholder until preorder input logic is defined
    const double* d_left_mat  = D.d_pmat + (size_t)op.left_id  * D.rate_cats * states * states;
    const double* d_right_mat = D.d_pmat + (size_t)op.right_id * D.rate_cats * states * states;

    unsigned int* site_scaler_ptr =
        site_scaler_ptr_base(D, op, site, rate_cats);
    unsigned int* left_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.left_id, site);
    unsigned int* right_scaler =
        scaler_ptr_for_pool(D, op.clv_pool, op.right_id, site);

    for (unsigned int r = 0; r < rate_cats; ++r) {
        write_scaler_shift(
            D,
            site_scaler_ptr,
            r,
            read_scaler_shift(D, left_scaler, r) +
            read_scaler_shift(D, right_scaler, r));
        const double* Lclv = d_left_clv  + site_off + (size_t)r * states;
        const double* Rclv = d_right_clv + site_off + (size_t)r * states;

        const double* Lmat = d_left_mat  + (size_t)r * states * states;
        const double* Rmat = d_right_mat + (size_t)r * states * states;

        double* Pout = parent_clv + site_off + (size_t)r * states;
        double col_scale_max_val = 0.0;

        const double* Lrow = Lmat;
        const double* Rrow = Rmat;
        for (unsigned int j = 0; j < states; ++j) {
            double lt = 0.0, rt = 0.0;
            #pragma unroll
            for (unsigned int k = 0; k < states; ++k) {
                lt += Lrow[k] * Lclv[k];
                rt += Rrow[k] * Rclv[k];
            }
            Pout[j] = lt * rt;
            if (Pout[j] > col_scale_max_val) col_scale_max_val = Pout[j];
            Lrow += states;
            Rrow += states;
        }

        if (site_scaler_ptr) {
            int scaling_exponent;
            frexp(col_scale_max_val, &scaling_exponent);
            if (scaling_exponent < SCALE_THRESHOLD_EXPONENT) {
                site_scaler_ptr[r] += SCALE_THRESHOLD_EXPONENT - scaling_exponent;
                for (unsigned int j = 0; j < states; ++j) {
                    scale_double_pow2(Pout[j], SCALE_THRESHOLD_EXPONENT - scaling_exponent);
                }
            }
        }
    }
}

// ===== Kernels =====
__global__ void UpdatePartialTipTipKernel(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_tip_tip_site_generic(D, ops[i], site);
        }
    }
}

template<int RATE_CATS>
__global__ void UpdatePartialTipTipKernel_states_4_ratecat(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_tip_tip_site_ratecat_nolookup<RATE_CATS>(D, ops[i], site);
        }
    }
}

__global__ void UpdatePartialTipTipKernel_states_4_generic(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_tip_tip_site_4_generic(D, ops[i], site);
        }
    }
}

__global__ void UpdatePartialTipInnerKernel(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_tip_inner_site_generic(D, ops[i], site);
        }
    }
}

template<int RATE_CATS>
__global__ void UpdatePartialTipInnerKernel_states_4_ratecat(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_tip_inner_site_ratecat<RATE_CATS>(D, ops[i], site);
        }
    }
}

__global__ void UpdatePartialInnerInnerKernel(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_inner_inner_site_generic(D, ops[i], site);
        }
    }
}

template<int RATE_CATS>
__global__ void  UpdatePartialInnerInnerKernel_states_4_ratecat(
    const DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops)
{
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            compute_inner_inner_site_ratecat<RATE_CATS>(D, ops[i], site);
        }
    }
}

// ===== Host launchers =====
static NodeOpInfo make_node_op(
    int parent_id,
    int left_id,
    int right_id,
    int left_tip_index,
    int right_tip_index,
    NodeOpType type,
    ClvPool clv_pool = CLV_POOL_UP,
    ClvDir dir_tag = CLV_DIR_UP)
{
    NodeOpInfo op{};
    op.parent_id = parent_id;
    op.left_id = left_id;
    op.right_id = right_id;
    op.left_tip_index = left_tip_index;
    op.right_tip_index = right_tip_index;
    op.op_type = static_cast<int>(type);
    op.clv_pool = static_cast<uint8_t>(clv_pool);
    op.dir_tag = static_cast<uint8_t>(dir_tag);
    return op;
}

static NodeOpInfo* upload_single_op(const NodeOpInfo& op, cudaStream_t stream) {
    NodeOpInfo* d_op = nullptr;
    CUDA_CHECK(cudaMalloc(&d_op, sizeof(NodeOpInfo)));
    CUDA_CHECK(cudaMemcpyAsync(
        d_op,
        &op,
        sizeof(NodeOpInfo),
        cudaMemcpyHostToDevice,
        stream));
    return d_op;
}

void Launch_Update_Partial_InnerInner(
    const DeviceTree& D,
    int parent_id,
    int left_id,
    int right_id,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    validate_states_rate(D.states, D.rate_cats, 64, 8);

    NodeOpInfo op = make_node_op(
        parent_id,
        left_id,
        right_id,
        -1,
        -1,
        OP_INNER_INNER,
        use_preorder ? CLV_POOL_DOWN : CLV_POOL_UP,
        static_cast<ClvDir>(dir_tag));
    NodeOpInfo* d_op = upload_single_op(op, stream);

    int block = 256;
    int grid = (D.sites + block - 1) / block;

    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                UpdatePartialInnerInnerKernel_states_4_ratecat<1><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            case 4:
                UpdatePartialInnerInnerKernel_states_4_ratecat<4><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            case 8:
                UpdatePartialInnerInnerKernel_states_4_ratecat<8><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            default:
                UpdatePartialInnerInnerKernel<<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
        }
    } else {
        UpdatePartialInnerInnerKernel<<<grid, block, 0, stream>>>(
            D,
            d_op,
            1);
    }
    CUDA_CHECK(cudaFree(d_op));
}

void partial_likelihood::compute_inner_inner(
    const DeviceTree& D,
    int parent_id,
    int left_id,
    int right_id,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    Launch_Update_Partial_InnerInner(
        D,
        parent_id,
        left_id,
        right_id,
        stream,
        use_preorder,
        dir_tag);
}

void Launch_Update_Partial_TipTip(
    const DeviceTree& D,
    int parent_id,
    int left_node_id,
    int right_node_id,
    int left_tip_index,
    int right_tip_index,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    validate_states_rate(D.states, D.rate_cats, 64, 8);

    NodeOpInfo op = make_node_op(
        parent_id,
        left_node_id,
        right_node_id,
        left_tip_index,
        right_tip_index,
        OP_TIP_TIP,
        use_preorder ? CLV_POOL_DOWN : CLV_POOL_UP,
        static_cast<ClvDir>(dir_tag));
    NodeOpInfo* d_op = upload_single_op(op, stream);

    int block = 256;
    int grid  = (D.sites + block - 1) / block;
    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                UpdatePartialTipTipKernel_states_4_ratecat<1><<<grid, block, 0, stream>>>(
                    D, d_op, 1);
                break;
            case 4:
                UpdatePartialTipTipKernel_states_4_ratecat<4><<<grid, block, 0, stream>>>(
                    D, d_op, 1);
                break;
            case 8:
                UpdatePartialTipTipKernel_states_4_ratecat<8><<<grid, block, 0, stream>>>(
                    D, d_op, 1);
                break;
            default:
                UpdatePartialTipTipKernel_states_4_generic<<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
        }
    } else {
        UpdatePartialTipTipKernel<<<grid, block, 0, stream>>>(
            D,
            d_op,
            1);
    }

    CUDA_CHECK(cudaFree(d_op));
}

void partial_likelihood::compute_tip_tip(
    const DeviceTree& D,
    int parent_id,
    int left_node_id,
    int right_node_id,
    int left_tip_index,
    int right_tip_index,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    Launch_Update_Partial_TipTip(
        D,
        parent_id,
        left_node_id,
        right_node_id,
        left_tip_index,
        right_tip_index,
        stream,
        use_preorder,
        dir_tag
    );
}

void Launch_Update_Partial_TipInner(
    const DeviceTree& D,
    int parent_id,
    int tip_node_id,
    int inner_node_id,
    int tip_index,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    validate_states_rate(D.states, D.rate_cats, 64, 8);

    const int block = 256;
    const int grid  = (D.sites + block - 1) / block;

    NodeOpInfo op = make_node_op(
        parent_id,
        tip_node_id,
        inner_node_id,
        tip_index,
        -1,
        OP_TIP_INNER,
        use_preorder ? CLV_POOL_DOWN : CLV_POOL_UP,
        static_cast<ClvDir>(dir_tag));
    NodeOpInfo* d_op = upload_single_op(op, stream);

    if (D.states == 4) {
        switch (D.rate_cats) {
            case 1:
                UpdatePartialTipInnerKernel_states_4_ratecat<1><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            case 4:
                UpdatePartialTipInnerKernel_states_4_ratecat<4><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            case 8:
                UpdatePartialTipInnerKernel_states_4_ratecat<8><<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
            default:
                UpdatePartialTipInnerKernel<<<grid, block, 0, stream>>>(
                    D,
                    d_op,
                    1);
                break;
        }
    } else {
        UpdatePartialTipInnerKernel<<<grid, block, 0, stream>>>(
            D,
            d_op,
            1);
    }
    CUDA_CHECK(cudaFree(d_op));
}

void partial_likelihood::compute_tip_inner(
    const DeviceTree& D,
    int parent_id,
    int tip_node_id,
    int inner_node_id,
    int tip_index,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    Launch_Update_Partial_TipInner(
        D,
        parent_id,
        tip_node_id,
        inner_node_id,
        tip_index,
        stream,
        use_preorder,
        dir_tag
    );
}

void partial_likelihood::compute_tip_inner_swap(
    const DeviceTree& D,
    int parent_id,
    int tip_node_id,
    int inner_node_id,
    int tip_index,
    cudaStream_t stream,
    bool use_preorder,
    uint8_t dir_tag)
{
    Launch_Update_Partial_TipInner(
        D,
        parent_id,
        tip_node_id,
        inner_node_id,
        tip_index,
        stream,
        use_preorder,
        dir_tag
    );
}

__global__ void Rtree_Likelihood_Site_Parallel_Upward_Kernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int num_ops
) {
    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            const NodeOpInfo& op = ops[i];
            switch (op.op_type) {
                case OP_TIP_TIP:
                    if (D.states == 4 && !D.force_generic_upward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_tip_tip_site_ratecat_nolookup<1>(D, op, site);
                                break;
                            case 4:
                                compute_tip_tip_site_ratecat_nolookup<4>(D, op, site);
                                break;
                            case 8:
                                compute_tip_tip_site_ratecat_nolookup<8>(D, op, site);
                                break;
                            default:
                                compute_tip_tip_site_4_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_tip_tip_site_generic(D, op, site);
                    }
                    break;
                case OP_TIP_INNER:
                    if (D.states == 4 && !D.force_generic_upward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_tip_inner_site_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_tip_inner_site_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_tip_inner_site_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_tip_inner_site_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_tip_inner_site_generic(D, op, site);
                    }
                    break;
                case OP_INNER_INNER:
                    if (D.states == 4 && !D.force_generic_upward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_inner_inner_site_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_inner_inner_site_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_inner_inner_site_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_inner_inner_site_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_inner_inner_site_generic(D, op, site);
                    }
                    break;
            default:
                break;
            
            
        }
    }
}
}
// Downward child kernel: compute target child clv_down from parent.down + sibling.up.
__global__ void Rtree_Likelihood_Site_Parallel_Downward_Kernel(
    const DeviceTree D,
    const NodeOpInfo* ops,
    int num_ops)
{

    unsigned int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = blockDim.x * gridDim.x;

    for (unsigned int site = tid; site < D.sites; site += step) {
        for (int i = 0; i < num_ops; ++i) {
            const NodeOpInfo& op = ops[i];
            switch (op.op_type) {
                case OP_DOWN_INNER_INNER:
                    if (D.states == 4 && !D.force_generic_downward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_inner_inner_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_inner_inner_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_inner_inner_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_inner_inner_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_inner_inner_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_INNER_TIP:
                    if (D.states == 4 && !D.force_generic_downward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_inner_tip_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_inner_tip_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_inner_tip_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_inner_tip_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_inner_tip_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_TIP_INNER:
                    if (D.states == 4 && !D.force_generic_downward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_tip_inner_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_tip_inner_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_tip_inner_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_tip_inner_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_tip_inner_generic(D, op, site);
                    }
                    break;
                case OP_DOWN_TIP_TIP:
                    if (D.states == 4 && !D.force_generic_downward) {
                        switch (D.rate_cats) {
                            case 1:
                                compute_downward_tip_tip_ratecat<1>(D, op, site);
                                break;
                            case 4:
                                compute_downward_tip_tip_ratecat<4>(D, op, site);
                                break;
                            case 8:
                                compute_downward_tip_tip_ratecat<8>(D, op, site);
                                break;
                            default:
                                compute_downward_tip_tip_generic(D, op, site);
                                break;
                        }
                    } else {
                        compute_downward_tip_tip_generic(D, op, site);
                    }
                    break;
                default:
                    break;
            }
        }
    }
}
