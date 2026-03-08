#pragma once
#include <cstdlib>   // abort
#include <csignal>   // raise, SIGTRAP
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream>
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

struct JplacePlacementRecord {
    std::string query_name;
    int edge_num = -1;
    double likelihood = 0.0;
    double like_weight_ratio = 1.0;
    double distal_length = 0.0;
    double pendant_length = 0.0;
};

inline std::string json_escape_string(const std::string& input) {
    std::ostringstream out;
    for (unsigned char ch : input) {
        switch (ch) {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

struct JplaceTreeExport {
    std::string tree;
    std::vector<int> edge_num_by_node;
};

inline JplaceTreeExport build_jplace_tree_export(const TreeBuildResult& tree) {
    if (tree.root_id < 0 || tree.root_id >= static_cast<int>(tree.nodes.size())) {
        throw std::runtime_error("build_jplace_tree: invalid root_id.");
    }

    JplaceTreeExport result;
    result.edge_num_by_node.assign(tree.nodes.size(), -1);
    int next_edge_num = 0;

    std::function<std::string(int, bool)> emit_node = [&](int node_id, bool suppress_parent_edge) -> std::string {
        if (node_id < 0 || node_id >= static_cast<int>(tree.nodes.size())) {
            throw std::runtime_error("build_jplace_tree: invalid node id.");
        }

        const TreeNode& node = tree.nodes[node_id];
        std::ostringstream out;
        if (node.is_tip) {
            out << node.name;
        } else {
            out << "(" << emit_node(node.left, false) << "," << emit_node(node.right, false) << ")";
        }

        if (node.parent >= 0 && !suppress_parent_edge) {
            const int edge_num = next_edge_num++;
            result.edge_num_by_node[node.id] = edge_num;
            out << ":" << std::setprecision(17) << node.branch_length_to_parent
                << "{" << edge_num << "}";
        }
        return out.str();
    };

    const TreeNode& root = tree.nodes[tree.root_id];
    const auto should_flatten_root_child = [&](int child_id) -> bool {
        if (child_id < 0 || child_id >= static_cast<int>(tree.nodes.size())) return false;
        const TreeNode& child = tree.nodes[child_id];
        return !child.is_tip && child.parent == tree.root_id && std::abs(child.branch_length_to_parent) <= 1e-15;
    };

    std::ostringstream out;
    if (!root.is_tip && should_flatten_root_child(root.left) != should_flatten_root_child(root.right)) {
        const int flat_child = should_flatten_root_child(root.left) ? root.left : root.right;
        const int other_child = (flat_child == root.left) ? root.right : root.left;
        out << "(" << emit_node(flat_child, true) << "," << emit_node(other_child, false) << ")";
    } else {
        out << emit_node(tree.root_id, false);
    }

    result.tree = out.str() + ";";
    return result;
}

inline std::string build_jplace_tree(const TreeBuildResult& tree) {
    return build_jplace_tree_export(tree).tree;
}

inline void write_jplace(
    const std::string& out_path,
    const std::string& tree_string,
    const std::vector<JplacePlacementRecord>& placements,
    const std::string& invocation)
{
    std::filesystem::path output_path(out_path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("Cannot open jplace output: " + out_path);
    }

    out << "{\n";
    out << "  \"tree\": \"" << json_escape_string(tree_string) << "\",\n";
    out << "  \"placements\": [\n";
    for (size_t i = 0; i < placements.size(); ++i) {
        const JplacePlacementRecord& rec = placements[i];
        out << "    {\n";
        out << "      \"p\": [["
            << rec.edge_num << ", "
            << std::setprecision(17) << rec.likelihood << ", "
            << std::setprecision(17) << rec.like_weight_ratio << ", "
            << std::setprecision(17) << rec.distal_length << ", "
            << std::setprecision(17) << rec.pendant_length << "]],\n";
        out << "      \"n\": [\"" << json_escape_string(rec.query_name) << "\"]\n";
        out << "    }";
        if (i + 1 < placements.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"metadata\": {\n";
    out << "    \"invocation\": \"" << json_escape_string(invocation) << "\"\n";
    out << "  },\n";
    out << "  \"version\": 3,\n";
    out << "  \"fields\": [\"edge_num\", \"likelihood\", \"like_weight_ratio\", \"distal_length\", \"pendant_length\"]\n";
    out << "}\n";
}

// ===== Device-side CLV helpers (inlined for reuse across CUDA units) =====
__device__ __forceinline__ size_t per_node_span(const DeviceTree& D) {
    return (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
}

__device__ __forceinline__ size_t scaler_span(const DeviceTree& D) {
    if (D.per_rate_scaling) {
        return (size_t)D.sites * (size_t)D.rate_cats;
    }
    return (size_t)D.sites;
}

__device__ __forceinline__ size_t scaler_site_offset(
    const DeviceTree& D,
    unsigned int site)
{
    if (D.per_rate_scaling) {
        return (size_t)site * (size_t)D.rate_cats;
    }
    return (size_t)site;
}

__device__ __forceinline__ unsigned int* scaler_ptr_for_node(
    unsigned int* base,
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    if (!base) return nullptr;
    if (node_id < 0 || node_id >= D.capacity_N) return nullptr;
    return base + (size_t)node_id * scaler_span(D) + scaler_site_offset(D, site);
}

__device__ __forceinline__ unsigned int* up_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_up, D, node_id, site);
}

__device__ __forceinline__ unsigned int* down_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_down, D, node_id, site);
}

__device__ __forceinline__ unsigned int* mid_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_mid, D, node_id, site);
}

__device__ __forceinline__ unsigned int* mid_base_scaler_ptr(
    const DeviceTree& D,
    int node_id,
    unsigned int site)
{
    return scaler_ptr_for_node(D.d_site_scaler_mid_base, D, node_id, site);
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
    (void)rate_cats;
    if (op.clv_pool == static_cast<uint8_t>(CLV_POOL_DOWN)) {
        return down_scaler_ptr(D, op.parent_id, site);
    }
    return up_scaler_ptr(D, op.parent_id, site);
}
