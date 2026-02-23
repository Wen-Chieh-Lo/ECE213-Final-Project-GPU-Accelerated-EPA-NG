#pragma once
#include <cstdlib>   // abort
#include <csignal>   // raise, SIGTRAP
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>
#include <cstddef>
#include "tree_generation/tree.hpp"

// Branch length defaults to match epa-ng constants.
__host__ __device__ constexpr double DEFAULT_BRANCH_LENGTH = 0.10536051565782628; // -log(0.9)
__host__ __device__ constexpr double OPT_BRANCH_LEN_MIN = 1.0e-4; // PLLMOD_OPT_MIN_BRANCH_LEN
__host__ __device__ constexpr double OPT_BRANCH_LEN_MAX = 100.0;  // PLLMOD_OPT_MAX_BRANCH_LEN
__host__ __device__ constexpr double OPT_BRANCH_EPSILON = 1.0e-1;
__host__ __device__ constexpr double OPT_BRANCH_XTOL = OPT_BRANCH_LEN_MIN / 10.0;

// Basic tags describing the operation type and CLV buffer selection.
enum NodeOpType : int {
    OP_TIP_TIP = 0,
    OP_TIP_INNER = 1,
    OP_INNER_INNER = 2,
    OP_DOWN_INNER_INNER = 3,
    OP_DOWN_INNER_TIP = 4,
    OP_DOWN_TIP_INNER = 5,
    OP_DOWN_TIP_TIP   = 6
};

enum ClvPool : uint8_t {
    CLV_POOL_UP = 0,
    CLV_POOL_DOWN = 1
};

// Direction tags used by preorder/downward passes.
enum ClvDir : uint8_t {
    CLV_DIR_UNSET      = 0,
    CLV_DIR_UP         = 1, // child -> parent
    CLV_DIR_DOWN_LEFT  = 2, // parent -> left child
    CLV_DIR_DOWN_RIGHT = 3  // parent -> right child
};

struct NodeOpInfo {
    int parent_id = -1;
    int left_id = -1;
    int right_id = -1;
    int left_tip_index = -1;
    int right_tip_index = -1;
    int op_type = OP_TIP_TIP;
    uint8_t clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
    uint8_t dir_tag  = static_cast<uint8_t>(CLV_DIR_UP);
};

// Common CUDA error checking macro used across modules.
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr,                                                       \
                "CUDA ERROR: %s (%d)\n"                                            \
                "  at %s:%d\n"                                                    \
                "  call: %s\n",                                                   \
                cudaGetErrorString(err__), (int)err__,                            \
                __FILE__, __LINE__, #call);                                       \
            raise(SIGTRAP);   /* <<< 讓 gdb 停在這裡 */                            \
            abort();                                                             \
        }                                                                         \
    } while (0)
#endif

#define CHECK_CUDA_LAST()                                                        \
    do {                                                                         \
        cudaError_t err__ = cudaGetLastError();                                  \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr,                                                      \
                "CUDA KERNEL LAUNCH ERROR: %s (%d)\n"                            \
                "  at %s:%d\n",                                                  \
                cudaGetErrorString(err__), (int)err__,                           \
                __FILE__, __LINE__);                                             \
            raise(SIGTRAP);                                                      \
            abort();                                                            \
        }                                                                        \
    } while (0)

// Fast integer log2 ceil for small unsigned values.
inline unsigned int ceil_log2_u32(unsigned int x) {
    if (x <= 1u) return 0u;
    unsigned int v = x - 1u;
    unsigned int r = 0u;
    while (v) { v >>= 1u; ++r; }
    return r;
}

// RAII wrapper for device buffers allocated with cudaMalloc.
struct DeviceBuffer {
    double* ptr{nullptr};

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { CHECK_CUDA(cudaMalloc(&ptr, sizeof(double) * count)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    double* get() const { return ptr; }
};

// Bounds check helper for states / rate categories.
inline void validate_states_rate(int states, int rate_cats, int max_states, int max_rate_cats) {
    if (states <= 0 || rate_cats <= 0) throw std::runtime_error("Invalid states or rate categories.");
    if (states > max_states) throw std::runtime_error("states exceeds MAX_STATES.");
    if (rate_cats > max_rate_cats) throw std::runtime_error("rate_cats exceeds MAX_RATECATS.");
}

// ===== Device-side CLV helpers (inlined for reuse across CUDA units) =====
__device__ __forceinline__ size_t per_node_span(const DeviceTree& D) {
    return (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
}

template <typename T>
__device__ __forceinline__ T* clv_write_pool_base(const DeviceTree& D, const NodeOpInfo& op) {
    return (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN))
        ? reinterpret_cast<T*>(D.d_clv_down)
        : reinterpret_cast<T*>(D.d_clv_up);
}

template <typename T>
__device__ __forceinline__ T* clv_read_pool_base(const DeviceTree& D, const NodeOpInfo& op) {
    return (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN))
        ? reinterpret_cast<T*>(D.d_clv_down)
        : reinterpret_cast<T*>(D.d_clv_up);
}

template <typename T>
__device__ __forceinline__ T* clv_write_ptr_for_node(const DeviceTree& D, const NodeOpInfo& op, int node_id) {
    T* base = clv_write_pool_base<T>(D, op);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

template <typename T>
__device__ __forceinline__ T* clv_read_ptr_for_node(const DeviceTree& D, const NodeOpInfo& op, int node_id) {
    T* base = clv_read_pool_base<T>(D, op);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

// Variant when the pool is implicitly the "up" pool (used in derivative helpers).
template <typename T>
__device__ __forceinline__ T* clv_read_ptr_for_node(const DeviceTree& D, int node_id) {
    T* base = reinterpret_cast<T*>(D.d_clv_up);
    return base ? base + (size_t)node_id * per_node_span(D) : nullptr;
}

__device__ __forceinline__ unsigned int* site_scaler_ptr_base(
    const DeviceTree& D,
    const NodeOpInfo& op,
    unsigned int site,
    unsigned int rate_cats)
{
    return D.d_site_scaler
        ? D.d_site_scaler + (size_t)site * (size_t)rate_cats
        : nullptr;
}
