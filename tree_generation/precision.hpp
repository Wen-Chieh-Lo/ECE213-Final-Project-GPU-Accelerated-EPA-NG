#pragma once

#include <cuda_runtime.h>
#include <cmath>

#if defined(MLIPPER_USE_DOUBLE)
using fp_t = double;
using fp2_t = double2;
using fp4_t = double4;
#else
using fp_t = float;
using fp2_t = float2;
using fp4_t = float4;
#endif

#if defined(MLIPPER_USE_DOUBLE)
static constexpr fp_t FP_EPS = fp_t(1e-300);
#else
static constexpr fp_t FP_EPS = fp_t(1e-30f);
#endif

__host__ __device__ __forceinline__ fp_t fp_exp(fp_t x) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::exp(x);
#else
    return ::expf(x);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_expm1(fp_t x) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::expm1(x);
#else
    return ::expm1f(x);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_log(fp_t x) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::log(x);
#else
    return ::logf(x);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_ldexp(fp_t x, int exp) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::ldexp(x, exp);
#else
    return ::ldexpf(x, exp);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_fabs(fp_t x) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::fabs(x);
#else
    return ::fabsf(x);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_fma(fp_t a, fp_t b, fp_t c) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::fma(a, b, c);
#else
    return ::fmaf(a, b, c);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_fmax(fp_t a, fp_t b) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::fmax(a, b);
#else
    return ::fmaxf(a, b);
#endif
}

__host__ __device__ __forceinline__ fp_t fp_fmin(fp_t a, fp_t b) {
#if defined(MLIPPER_USE_DOUBLE)
    return ::fmin(a, b);
#else
    return ::fminf(a, b);
#endif
}
