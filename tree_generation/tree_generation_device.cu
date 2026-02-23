#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "../mlipper_util.h"
#include "tree.hpp" 
#include "pmat.h"
#include "root_likelihood.cuh"
#include "tree_placement.cuh"
#include "partial_likelihood.cuh"
#include <tbb/parallel_for.h>


// Debug helper for device pointers used across placement path.
static void check_device_ptr(const void* ptr, const char* name) {
    cudaPointerAttributes attr{};
    cudaError_t st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        fprintf(stderr, "[dptr-check] %s invalid (ptr=%p, err=%s, type=%d)\n",
                name, ptr, cudaGetErrorString(st), (st == cudaSuccess) ? attr.type : -1);
        throw std::runtime_error(std::string("Invalid device pointer: ") + name);
    }
}

static void rebuild_traversals(TreeBuildResult& T) {
    T.preorder.clear();
    T.postorder.clear();
    if (T.root_id < 0 || T.root_id >= (int)T.nodes.size()) return;

    // Preorder: parent -> left -> right
    std::vector<int> stack;
    stack.push_back(T.root_id);
    while (!stack.empty()) {
        int id = stack.back();
        stack.pop_back();
        T.preorder.push_back(id);
        const TreeNode& n = T.nodes[id];
        if (!n.is_tip) {
            if (n.right >= 0) stack.push_back(n.right);
            if (n.left >= 0) stack.push_back(n.left);
        }
    }

    // Postorder: left -> right -> parent
    std::vector<std::pair<int, bool>> st;
    st.emplace_back(T.root_id, false);
    while (!st.empty()) {
        const std::pair<int, bool> cur = st.back();
        st.pop_back();
        const int id = cur.first;
        const bool visited = cur.second;
        if (id < 0) continue;
        const TreeNode& n = T.nodes[id];
        if (visited) {
            T.postorder.push_back(id);
        } else {
            st.emplace_back(id, true);
            if (!n.is_tip) {
                if (n.right >= 0) st.emplace_back(n.right, false);
                if (n.left >= 0) st.emplace_back(n.left, false);
            }
        }
    }
}

// TEMP: host-side insertion of a query tip with a new intermediate node.
// This updates TreeBuildResult only; device buffers are NOT updated.
struct InsertResult {
    int internal_id = -1;
    int tip_id = -1;
};

static InsertResult insert_query_with_intermediate(
    TreeBuildResult& T,
    const std::string& raw_name,
    int target_id,
    double pendant,
    double proximal)
{
    InsertResult out{};
    if (target_id < 0 || target_id >= (int)T.nodes.size()) {
        fprintf(stderr, "[insert] invalid target_id=%d\n", target_id);
        return out;
    }
    int parent_id = T.nodes[target_id].parent;
    const double total = T.nodes[target_id].branch_length_to_parent;
    double proximal_len = proximal;
    double distal_len = total - proximal_len;
    double pendant_len = pendant;

    int new_internal_id = (int)T.nodes.size();
    int new_tip_id = new_internal_id + 1;

    std::string name = raw_name.empty()
        ? ("query_" + std::to_string(new_tip_id))
        : raw_name;
    if (T.tip_node_by_name.count(name)) {
        name += "_" + std::to_string(new_tip_id);
    }

    TreeNode internal{};
    internal.id = new_internal_id;
    internal.is_tip = false;
    internal.parent = parent_id;
    internal.left = target_id;
    internal.right = new_tip_id;
    internal.branch_length_to_parent = proximal_len;

    TreeNode tip{};
    tip.id = new_tip_id;
    tip.is_tip = true;
    tip.parent = new_internal_id;
    tip.left = -1;
    tip.right = -1;
    tip.branch_length_to_parent = pendant_len;
    tip.name = name;

    if (parent_id >= 0) {
        if (T.nodes[parent_id].left == target_id) {
            T.nodes[parent_id].left = new_internal_id;
        } else if (T.nodes[parent_id].right == target_id) {
            T.nodes[parent_id].right = new_internal_id;
        } else {
            fprintf(stderr, "[insert] parent %d does not reference target %d\n", parent_id, target_id);
        }
    } else {
        // Target was root; new internal becomes root.
        T.root_id = new_internal_id;
    }

    T.nodes[target_id].parent = new_internal_id;
    T.nodes[target_id].branch_length_to_parent = distal_len;

    T.nodes.push_back(internal);
    T.nodes.push_back(tip);
    T.tip_node_by_name[name] = new_tip_id;

    rebuild_traversals(T);
    out.internal_id = new_internal_id;
    out.tip_id = new_tip_id;
    return out;
}

static void build_upward_ops_to_root(
    const TreeBuildResult& T,
    const std::vector<int>& node2tip,
    int start_id,
    std::vector<NodeOpInfo>& ops_out)
{
    ops_out.clear();
    int cur = start_id;
    while (cur >= 0) {
        if (cur >= (int)T.nodes.size()) break;
        const TreeNode& nd = T.nodes[cur];
        if (!nd.is_tip) {
            const int L = nd.left;
            const int R = nd.right;
            const bool tipL = (L >= 0 && T.nodes[L].is_tip);
            const bool tipR = (R >= 0 && T.nodes[R].is_tip);
            const int left_tip_idx  = tipL ? node2tip[L] : -1;
            const int right_tip_idx = tipR ? node2tip[R] : -1;
            NodeOpType type;
            if (tipL && tipR) {
                type = OP_TIP_TIP;
            } else if (tipL || tipR) {
                type = OP_TIP_INNER;
            } else {
                type = OP_INNER_INNER;
            }
            NodeOpInfo op{};
            op.parent_id = cur;
            op.left_id = L;
            op.right_id = R;
            op.left_tip_index = left_tip_idx;
            op.right_tip_index = right_tip_idx;
            op.op_type = static_cast<int>(type);
            op.clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
            op.dir_tag  = static_cast<uint8_t>(CLV_DIR_UP);
            ops_out.push_back(op);
        }
        cur = nd.parent;
    }
}

static void build_downward_ops_subtree(
    const TreeBuildResult& T,
    const std::vector<int>& node2tip,
    int parent_id,
    int child_id,
    std::vector<NodeOpInfo>& ops_out)
{
    if (parent_id < 0 || child_id < 0) return;
    const TreeNode& parent = T.nodes[parent_id];
    const bool child_is_left = (parent.left == child_id);
    const int Lp = parent.left;
    const int Rp = parent.right;
    const bool tipLp = (Lp >= 0 && T.nodes[Lp].is_tip);
    const bool tipRp = (Rp >= 0 && T.nodes[Rp].is_tip);
    // Target only the subtree root (child_id) for this parent.
    if (child_is_left) {
        NodeOpInfo op{};
        op.parent_id = parent_id;
        op.left_id = Lp;
        op.right_id = Rp;
        op.left_tip_index = tipLp ? node2tip[Lp] : -1;
        op.right_tip_index = tipRp ? node2tip[Rp] : -1;
        op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
        op.dir_tag  = static_cast<uint8_t>(CLV_DIR_DOWN_LEFT);
        if (tipLp && tipRp) {
            op.op_type = static_cast<int>(OP_DOWN_TIP_TIP);
        } else if (tipLp) {
            op.op_type = static_cast<int>(OP_DOWN_TIP_INNER);
        } else if (tipRp) {
            op.op_type = static_cast<int>(OP_DOWN_INNER_TIP);
        } else {
            op.op_type = static_cast<int>(OP_DOWN_INNER_INNER);
        }
        ops_out.push_back(op);
    } else {
        NodeOpInfo op{};
        op.parent_id = parent_id;
        op.left_id = Lp;
        op.right_id = Rp;
        op.left_tip_index = tipLp ? node2tip[Lp] : -1;
        op.right_tip_index = tipRp ? node2tip[Rp] : -1;
        op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
        op.dir_tag  = static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT);
        if (tipLp && tipRp) {
            op.op_type = static_cast<int>(OP_DOWN_TIP_TIP);
        } else if (tipRp) {
            op.op_type = static_cast<int>(OP_DOWN_TIP_INNER);
        } else if (tipLp) {
            op.op_type = static_cast<int>(OP_DOWN_INNER_TIP);
        } else {
            op.op_type = static_cast<int>(OP_DOWN_INNER_INNER);
        }
        ops_out.push_back(op);
    }

    // Preorder traversal of subtree rooted at child_id.
    std::vector<int> stack;
    stack.push_back(child_id);
    while (!stack.empty()) {
        const int nid = stack.back();
        stack.pop_back();
        if (nid < 0) continue;
        const TreeNode& nd = T.nodes[nid];
        if (nd.is_tip) continue;
        const int L = nd.left;
        const int R = nd.right;
        if (L < 0 || R < 0) continue;
        const bool tipL = (L >= 0 && T.nodes[L].is_tip);
        const bool tipR = (R >= 0 && T.nodes[R].is_tip);
        {
            // Target left child
            NodeOpInfo op{};
            op.parent_id = nid;
            op.left_id = L;
            op.right_id = R;
            op.left_tip_index = tipL ? node2tip[L] : -1;
            op.right_tip_index = tipR ? node2tip[R] : -1;
            op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
            op.dir_tag  = static_cast<uint8_t>(CLV_DIR_DOWN_LEFT);
            if (tipL && tipR) {
                op.op_type = static_cast<int>(OP_DOWN_TIP_TIP);
            } else if (tipL) {
                op.op_type = static_cast<int>(OP_DOWN_TIP_INNER);
            } else if (tipR) {
                op.op_type = static_cast<int>(OP_DOWN_INNER_TIP);
            } else {
                op.op_type = static_cast<int>(OP_DOWN_INNER_INNER);
            }
            ops_out.push_back(op);
        }
        {
            // Target right child
            NodeOpInfo op{};
            op.parent_id = nid;
            op.left_id = L;
            op.right_id = R;
            op.left_tip_index = tipL ? node2tip[L] : -1;
            op.right_tip_index = tipR ? node2tip[R] : -1;
            op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
            op.dir_tag  = static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT);
            if (tipL && tipR) {
                op.op_type = static_cast<int>(OP_DOWN_TIP_TIP);
            } else if (tipR) {
                op.op_type = static_cast<int>(OP_DOWN_TIP_INNER);
            } else if (tipL) {
                op.op_type = static_cast<int>(OP_DOWN_INNER_TIP);
            } else {
                op.op_type = static_cast<int>(OP_DOWN_INNER_INNER);
            }
            ops_out.push_back(op);
        }
        // Preorder: push right then left
        stack.push_back(R);
        stack.push_back(L);
    }
}

static void rebuild_host_topology_from_tree(const TreeBuildResult& T, HostPacking& H)
{
    const int N = (int)T.nodes.size();
    H.postorder = T.postorder;
    H.preorder  = T.preorder;
    H.parent.assign(N, -1);
    H.left.assign(N, -1);
    H.right.assign(N, -1);
    H.is_tip.assign(N, 0);
    H.blen.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const TreeNode& nd = T.nodes[i];
        H.parent[i] = nd.parent;
        H.left[i]   = nd.left;
        H.right[i]  = nd.right;
        H.is_tip[i] = nd.is_tip;
        H.blen[i]   = nd.branch_length_to_parent;
    }
}

static void append_query_tip_to_host_packing(
    HostPacking& H,
    const PlacementQueryBatch& Q,
    int query_idx,
    int new_tip_node_id,
    size_t sites)
{
    if (query_idx < 0 || (size_t)query_idx >= Q.count) {
        throw std::runtime_error("append_query_tip_to_host_packing: query_idx out of range.");
    }
    const size_t needed = ((size_t)query_idx + 1) * sites;
    if (Q.query_chars.size() < needed) {
        throw std::runtime_error("append_query_tip_to_host_packing: query_chars buffer too small.");
    }

    H.tip_node_ids.push_back(new_tip_node_id);
    const size_t old = H.tipchars.size();
    H.tipchars.resize(old + sites);
    std::memcpy(
        H.tipchars.data() + old,
        Q.query_chars.data() + (size_t)query_idx * sites,
        sites * sizeof(uint8_t));
}

// Try to incrementally sync DeviceTree buffers after a single committed insertion.
// Returns true if sync was applied; false if invariants/capacities don't hold and a full rebuild is required.
static void update_insertion_device(
    DeviceTree& D,
    const TreeBuildResult& T,
    const HostPacking& H,
    size_t sites,
    int states,
    int rate_cats,
    int target_id,
    int internal_id,
    int tip_id,
    cudaStream_t stream)
{
    const int old_tips = D.tips;
    const int new_tips = (int)H.tip_node_ids.size();
    const int newN = (int)T.nodes.size();

    D.N = newN;
    D.tips = new_tips;
    D.inners = D.N - D.tips;
    D.root_id = T.root_id;

    CUDA_CHECK(cudaMemcpyAsync(
        D.d_tipchars + (size_t)old_tips * sites,
        H.tipchars.data() + (size_t)old_tips * sites,
        sizeof(uint8_t) * sites,
        cudaMemcpyHostToDevice,
        stream));

    const size_t stride = (size_t)rate_cats * (size_t)states * (size_t)states;
    const double* mid_prox_src = H.pmats_mid_prox.empty() ? H.pmats_mid.data() : H.pmats_mid_prox.data();
    const double* mid_dist_src = H.pmats_mid_dist.empty() ? H.pmats_mid.data() : H.pmats_mid_dist.data();

    auto sync_node = [&](int nid) {
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_blen + (size_t)nid,
            &H.blen[(size_t)nid],
            sizeof(double),
            cudaMemcpyHostToDevice,
            stream));

        const size_t off = (size_t)nid * stride;
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_pmat + off,
            H.pmats.data() + off,
            sizeof(double) * stride,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_pmat_mid + off,
            H.pmats_mid.data() + off,
            sizeof(double) * stride,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_pmat_mid_prox + off,
            mid_prox_src + off,
            sizeof(double) * stride,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaMemcpyAsync(
            D.d_pmat_mid_dist + off,
            mid_dist_src + off,
            sizeof(double) * stride,
            cudaMemcpyHostToDevice,
            stream));
    };

    sync_node(target_id);
    sync_node(internal_id);
    sync_node(tip_id);
    return;
}

static inline int down_op_type_for_target(bool target_is_tip, bool sibling_is_tip) {
    if (target_is_tip && sibling_is_tip) return static_cast<int>(OP_DOWN_TIP_TIP);
    if (target_is_tip) return static_cast<int>(OP_DOWN_TIP_INNER);
    if (sibling_is_tip) return static_cast<int>(OP_DOWN_INNER_TIP);
    return static_cast<int>(OP_DOWN_INNER_INNER);
}

static inline void push_down_op(
    std::vector<NodeOpInfo>& ops,
    int parent_id,
    int left_id,
    int right_id,
    bool left_is_tip,
    bool right_is_tip,
    const std::vector<int>& node2tip,
    uint8_t dir_tag)
{
    NodeOpInfo op{};
    op.parent_id = parent_id;
    op.left_id   = left_id;
    op.right_id  = right_id;
    op.left_tip_index  = left_is_tip  ? node2tip[left_id]  : -1;
    op.right_tip_index = right_is_tip ? node2tip[right_id] : -1;
    op.clv_pool = static_cast<uint8_t>(CLV_POOL_DOWN);
    op.dir_tag  = dir_tag;

    const bool target_is_left = (dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const bool target_is_tip  = target_is_left ? left_is_tip : right_is_tip;
    const bool sibling_is_tip = target_is_left ? right_is_tip : left_is_tip;
    op.op_type = down_op_type_for_target(target_is_tip, sibling_is_tip);
    ops.push_back(op);
}

// Compute max-relative change between newly computed down-CLVs (in scratch) and current D.d_clv_down,
// commit scratch into D.d_clv_down for the op's target node, and record the max relative delta per op.
__global__ void DownwardMaxRelCommitKernel(
    DeviceTree D,
    const NodeOpInfo* __restrict__ ops,
    int num_ops,
    const double* __restrict__ scratch_down,
    double denom_floor,
    double* __restrict__ out_max_rel)
{
    const int op_idx = (int)blockIdx.x;
    if (op_idx < 0 || op_idx >= num_ops) return;
    if (!ops || !scratch_down || !D.d_clv_down || !out_max_rel) return;

    const NodeOpInfo op = ops[op_idx];
    const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
    const int target_id = target_is_left ? op.left_id : op.right_id;
    if (target_id < 0 || target_id >= D.N) {
        out_max_rel[op_idx] = 0.0;
        return;
    }

    const size_t per_node = per_node_span(D);
    const size_t base = (size_t)target_id * per_node;
    double* dst = D.d_clv_down + base;
    const double* src = scratch_down + base;

    double local_max = 0.0;
    for (size_t i = (size_t)threadIdx.x; i < per_node; i += (size_t)blockDim.x) {
        const double oldv = dst[i];
        const double newv = src[i];
        const double rel = fabs(newv - oldv);
        local_max += log(rel + 1e-100);
        dst[i] = newv;
    }

    __shared__ double smax[256];
    const int tid = (int)threadIdx.x;
    smax[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smax[tid] += smax[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) out_max_rel[op_idx] = smax[0] ;
}

__global__ void CopyParentDownToScratchKernel(
    const double* __restrict__ src_down,
    double* __restrict__ dst_scratch,
    const int* __restrict__ parent_ids,
    int num_parents,
    size_t per_node,
    int N)
{
    const size_t total = (size_t)num_parents * per_node;
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = tid; i < total; i += stride) {
        const int pidx = (int)(i / per_node);
        const size_t off = i - (size_t)pidx * per_node;
        const int nid = parent_ids[pidx];
        if (nid < 0 || nid >= N) continue;
        const size_t base = (size_t)nid * per_node + off;
        dst_scratch[base] = src_down[base];
    }
}

static void update_downward_until_converged(
    DeviceTree& D,
    const TreeBuildResult& T,
    const std::vector<int>& node2tip,
    double eps_rel,
    cudaStream_t stream)
{
    if (!D.d_clv_down) return;
    if (T.root_id < 0 || T.root_id >= (int)T.nodes.size()) return;
    const bool debug_downward_levels = false;

    const double denom_floor = 1e-12;
    const int max_levels = 64;

    std::vector<int> frontier;
    std::vector<int> next_frontier;
    frontier.push_back(T.root_id);

    std::vector<NodeOpInfo> ops_level;
    NodeOpInfo* d_ops_level = nullptr;
    double* d_max_rel = nullptr;
    int* d_frontier_parents = nullptr;
    std::vector<double> max_rel_host;

    const size_t per_node = D.per_node_elems();
    if (per_node == 0 || D.N <= 0) return;
    if (!D.d_downward_scratch) return;
    double* scratch = D.d_downward_scratch;
    CUDA_CHECK(cudaMemsetAsync(scratch, 0, sizeof(double) * per_node * (size_t)D.N, stream));

    const int max_ops_cap = std::max(1, D.N * 2);
    const int max_frontier_cap = std::max(1, D.N);
    CUDA_CHECK(cudaMalloc(&d_ops_level, sizeof(NodeOpInfo) * (size_t)max_ops_cap));
    CUDA_CHECK(cudaMalloc(&d_max_rel, sizeof(double) * (size_t)max_ops_cap));
    CUDA_CHECK(cudaMalloc(&d_frontier_parents, sizeof(int) * (size_t)max_frontier_cap));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int num_sms = prop.multiProcessorCount;

    cudaFuncAttributes down_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&down_attr, Rtree_Likelihood_Site_Parallel_Downward_Kernel));
    int down_block = 256;
    if (down_attr.maxThreadsPerBlock > 0 && down_block > down_attr.maxThreadsPerBlock) {
        down_block = down_attr.maxThreadsPerBlock;
    }
    int down_max_blocks_per_sm = 4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &down_max_blocks_per_sm,
        Rtree_Likelihood_Site_Parallel_Downward_Kernel,
        down_block,
        0));
    const int down_max_blocks = num_sms * down_max_blocks_per_sm;
    int down_grid = (int)((D.sites + (size_t)down_block - 1) / (size_t)down_block);
    if (down_max_blocks > 0 && down_grid > down_max_blocks) down_grid = down_max_blocks;

    cudaFuncAttributes copy_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&copy_attr, CopyParentDownToScratchKernel));
    int copy_block = 256;
    if (copy_attr.maxThreadsPerBlock > 0 && copy_block > copy_attr.maxThreadsPerBlock) {
        copy_block = copy_attr.maxThreadsPerBlock;
    }

    for (int level = 0; level < max_levels && !frontier.empty(); ++level) {
        ops_level.clear();
        ops_level.reserve((size_t)frontier.size() * 2);

        // Build only the ops for the current frontier parents.
        for (int parent_id : frontier) {
            if (parent_id < 0 || parent_id >= (int)T.nodes.size()) continue;
            const TreeNode& nd = T.nodes[parent_id];
            if (nd.is_tip) continue;

            const int L = nd.left;
            const int R = nd.right;
            const bool tipL = (L >= 0 && T.nodes[L].is_tip);
            const bool tipR = (R >= 0 && T.nodes[R].is_tip);

            push_down_op(
                ops_level,
                parent_id,
                L,
                R,
                tipL,
                tipR,
                node2tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
            push_down_op(
                ops_level,
                parent_id,
                L,
                R,
                tipL,
                tipR,
                node2tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        }

        const int num_ops = (int)ops_level.size();
        if (num_ops <= 0) break;
        if (num_ops > max_ops_cap) {
            throw std::runtime_error("update_downward_until_converged: op count exceeds preallocated capacity.");
        }

        const int num_frontier = (int)frontier.size();
        if (num_frontier > max_frontier_cap) {
            throw std::runtime_error("update_downward_until_converged: frontier exceeds preallocated capacity.");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            d_frontier_parents,
            frontier.data(),
            sizeof(int) * (size_t)num_frontier,
            cudaMemcpyHostToDevice,
            stream));
        {
            const size_t total_copy = (size_t)num_frontier * per_node;
            int copy_grid = (int)((total_copy + (size_t)copy_block - 1) / (size_t)copy_block);
            if (down_max_blocks > 0 && copy_grid > down_max_blocks) copy_grid = down_max_blocks;
            CopyParentDownToScratchKernel<<<copy_grid, copy_block, 0, stream>>>(
                D.d_clv_down,
                scratch,
                d_frontier_parents,
                num_frontier,
                per_node,
                D.N);
            CHECK_CUDA_LAST();
        }

        CUDA_CHECK(cudaMemcpyAsync(
            d_ops_level,
            ops_level.data(),
            sizeof(NodeOpInfo) * (size_t)num_ops,
            cudaMemcpyHostToDevice,
            stream));

        // Run downward kernel writing into scratch.
        // Keep midpoint buffers enabled so BFS and full-downward paths both update mid/mid_base.
        DeviceTree Dbfs = D;
        Dbfs.d_clv_down = scratch;

        Rtree_Likelihood_Site_Parallel_Downward_Kernel<<<down_grid, down_block, 0, stream>>>(
            Dbfs,
            d_ops_level,
            num_ops);
        CHECK_CUDA_LAST();

        // Compute per-op max-relative change and commit scratch -> D.d_clv_down for each target.
        CUDA_CHECK(cudaMemsetAsync(d_max_rel, 0, sizeof(double) * (size_t)num_ops, stream));
        {
            const int block = 256; // matches shared array above
            DownwardMaxRelCommitKernel<<<(unsigned)num_ops, block, 0, stream>>>(
                D,
                d_ops_level,
                num_ops,
                scratch,
                denom_floor,
                d_max_rel);
            CHECK_CUDA_LAST();
        }

        max_rel_host.resize((size_t)num_ops);
        CUDA_CHECK(cudaMemcpyAsync(
            max_rel_host.data(),
            d_max_rel,
            sizeof(double) * (size_t)num_ops,
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        next_frontier.clear();
        next_frontier.reserve(frontier.size() * 2);
        eps_rel = -100000000000;
        for (int i = 0; i < num_ops; ++i) {
            if (!(max_rel_host[(size_t)i] > eps_rel) && level > 0){
                continue;
            }
            const NodeOpInfo& op = ops_level[(size_t)i];
            const bool target_is_left = (op.dir_tag == static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));
            const int target_id = target_is_left ? op.left_id : op.right_id;
            if (target_id < 0 || target_id >= (int)T.nodes.size()) continue;
            if (T.nodes[target_id].is_tip) continue;
            next_frontier.push_back(target_id);
        }
        frontier.swap(next_frontier);
    }

    if (d_ops_level) CUDA_CHECK(cudaFree(d_ops_level));
    if (d_max_rel) CUDA_CHECK(cudaFree(d_max_rel));
    if (d_frontier_parents) CUDA_CHECK(cudaFree(d_frontier_parents));
}


// forward decl for device likelihood wrapper
double EvaluateTreeLogLikelihood_device(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream = 0);
    
void UpdateTreeLogLikelihood_device(
    DeviceTree&            D,
    TreeBuildResult&       T,
    HostPacking&           H,
    const EigResult& er,
    PlacementQueryBatch& Q,
    const std::vector<double>& rate_multipliers,
    const std::vector<NewPlacementQuery>* queries,
    const std::vector<double>* pi_debug,
    const std::vector<double>* rate_weights_debug,
    double baseline_loglik,
    bool   debug_mid,
    int smoothing,
    cudaStream_t           stream);

// Build transition probability matrices for each node/rate category on host.
void fill_pmats_in_host_packing(
    const TreeBuildResult&       T,
    HostPacking&                 H,
    const EigResult&             er,
    const std::vector<double>&   pi,                 // len = states
    const std::vector<double>&   rate_multipliers,   // len = rate_cats (per-category rate multipliers)
    int states,
    int rate_cats,
    const int* changed_nodes,
    int num_changed_nodes)
{
    const int N = (int)T.nodes.size();
    const size_t per_node = (size_t)rate_cats * states * states;

    const size_t required = (size_t)N * per_node;
    const bool want_incremental = (changed_nodes && num_changed_nodes > 0);

    auto reset_all = [&]() {
        H.pmats.assign(required, 0.0);
        H.pmats_mid.assign(required, 0.0);
        H.pmats_mid_prox.assign(required, 0.0);
        H.pmats_mid_dist.assign(required, 0.0);
    };

    auto buffers_look_compatible = [&]() -> bool {
        if (per_node == 0) return false;
        if (H.pmats.empty()) return false;
        if (H.pmats.size() % per_node != 0) return false;
        if (!H.pmats_mid.empty() && H.pmats_mid.size() % per_node != 0) return false;
        if (!H.pmats_mid_prox.empty() && H.pmats_mid_prox.size() % per_node != 0) return false;
        if (!H.pmats_mid_dist.empty() && H.pmats_mid_dist.size() % per_node != 0) return false;
        return true;
    };

    if (!want_incremental) {
        reset_all();
    } else if (!buffers_look_compatible()) {
        // No existing buffers to update incrementally; fall back to full rebuild.
        reset_all();
    } else {
        // Preserve existing pmats for unchanged nodes; extend buffers for newly added nodes.
        H.pmats.resize(required, 0.0);
        H.pmats_mid.resize(required, 0.0);
        if (!H.pmats_mid_prox.empty()) H.pmats_mid_prox.resize(required, 0.0);
        if (!H.pmats_mid_dist.empty()) H.pmats_mid_dist.resize(required, 0.0);
    }

    auto compute_node_pmats = [&](int nid) {
        if (nid < 0 || nid >= N) return;
        const TreeNode& nd = T.nodes[nid];
        if (nd.parent < 0) return;
        double* base = H.pmats.data() + (size_t)nid * per_node;
        double* base_mid = H.pmats_mid.data() + (size_t)nid * per_node;
        double* base_mid_prox = H.pmats_mid_prox.empty() ? nullptr : (H.pmats_mid_prox.data() + (size_t)nid * per_node);
        double* base_mid_dist = H.pmats_mid_dist.empty() ? nullptr : (H.pmats_mid_dist.data() + (size_t)nid * per_node);
        const double blen  = nd.branch_length_to_parent;

        for (int rc = 0; rc < rate_cats; ++rc) {
            double r = rate_multipliers[rc];  // rate category multiplier
            double t = blen;                  // branch length
            double p = 0;

            double* P = base + (size_t)rc * states * states;
            double* Pmid = base_mid + (size_t)rc * states * states;
            double* Pprox = base_mid_prox ? (base_mid_prox + (size_t)rc * states * states) : nullptr;
            double* Pdist = base_mid_dist ? (base_mid_dist + (size_t)rc * states * states) : nullptr;
            pmatrix_from_triple(
                er.Vinv.data(), er.V.data(), er.lambdas.data(),
                            r, t, p, P, states);
            // half-branch PMAT for midpoint
            pmatrix_from_triple(
                er.Vinv.data(), er.V.data(), er.lambdas.data(),
                            r, t * 0.5, p, Pmid, states);
            if (Pprox) std::memcpy(Pprox, Pmid, sizeof(double) * (size_t)states * (size_t)states);
            if (Pdist) std::memcpy(Pdist, Pmid, sizeof(double) * (size_t)states * (size_t)states);
        }
    };

    if (!want_incremental || !changed_nodes || num_changed_nodes <= 0) {
        tbb::parallel_for(0, N, [&](int nid) { compute_node_pmats(nid);});
    } else {
        for (int i = 0; i < num_changed_nodes; ++i) compute_node_pmats(changed_nodes[i]);
    }
}

void fill_query_pmats(
    PlacementQueryBatch&         Q,
    const EigResult&             er,
    const std::vector<double>&   rate_multipliers,
    int states,
    int rate_cats)
{
    const size_t per_query = (size_t)rate_cats * states * states;
    const size_t qcount = Q.count;
    Q.query_pmats.assign(per_query * qcount, 0.0);
    if (Q.branch_lengths.size() != qcount) {
        Q.branch_lengths.assign(qcount, 0.5);
    }
    for (size_t qi = 0; qi < qcount; ++qi) {
        double* base = Q.query_pmats.data() + qi * per_query;
        double blen  = Q.branch_lengths[qi];
        for (int rc = 0; rc < rate_cats; ++rc) {
            double r = rate_multipliers[rc];  // rate category multiplier
            double t = blen;                  // branch length
            double p = 0;

            double* P = base + (size_t)rc * states * states;

            pmatrix_from_triple(
                er.Vinv.data(), er.V.data(), er.lambdas.data(),
                            r, t, p, P, states);
        }
    }
}

// Select view for a specific query's PMAT chunk.
DeviceTree make_query_view(const DeviceTree& D, int query_idx) {
    DeviceTree view = D;
    if (query_idx < 0 || query_idx >= D.placement_queries) return view;
    const size_t per_query = (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
    const size_t clv_span = (size_t)D.sites * (size_t)D.rate_cats * (size_t)D.states;
    if (D.d_query_pmat) {
        // Shared query PMAT buffer across queries; rebuilt per-query.
        view.d_query_pmat = D.d_query_pmat;
    }
    if (D.d_query_clv) {
        view.d_query_clv = D.d_query_clv + (size_t)query_idx * clv_span;
    }
    return view;
}

__global__ void BuildQueryClvKernel(
    DeviceTree D,
    const uint8_t* query_chars,
    int query_idx)
{
    const unsigned site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= D.sites) return;
    const size_t base_char = (size_t)query_idx * D.sites + site;
    const uint8_t enc = query_chars ? query_chars[base_char] : 4;

    const size_t per_site = (size_t)D.rate_cats * (size_t)D.states;
    const size_t clv_span = (size_t)D.sites * per_site;
    double* out = D.d_query_clv + (size_t)query_idx * clv_span + (size_t)site * per_site;
    for (int rc = 0; rc < D.rate_cats; ++rc) {
        double* row = out + (size_t)rc * D.states;
        for (int s = 0; s < D.states; ++s) {
            row[s] = (enc < D.states) ? (s == enc ? 1.0 : 0.0) : 1.0;
        }
    }
}

void build_query_clv_on_device(
    const DeviceTree& D,
    int query_idx,
    cudaStream_t stream)
{
    if (!D.d_query_clv || !D.d_query_chars) return;
    if (query_idx < 0 || query_idx >= D.placement_queries) return;
    dim3 block(256);
    dim3 grid((unsigned)((D.sites + block.x - 1) / block.x));
    BuildQueryClvKernel<<<grid, block, 0, stream>>>(D, D.d_query_chars, query_idx);
    CUDA_CHECK(cudaGetLastError());
}

// Pack topology and tip encodings from tree/MSA into HostPacking.
HostPacking pack_host_arrays_from_tree_and_msa(
        const TreeBuildResult& T,
        const std::vector<std::string>& msa_tip_names,  // len = tips
        const std::vector<std::string>& msa_rows,       // len = tips, each row length = sites
        size_t sites,
        int states)
{
    if (msa_rows.size() != msa_tip_names.size())
        throw std::runtime_error("MSA rows/names size mismatch.");
    if (msa_rows.empty()) throw std::runtime_error("Empty MSA.");
    if (msa_rows[0].size() != sites)
        throw std::runtime_error("Sites mismatch.");

    const int N = (int)T.nodes.size();

    // Topology
    HostPacking H;
    H.postorder = T.postorder;
    H.preorder  = T.preorder;
    H.parent.resize(N, -1);
    H.left.resize(N, -1);
    H.right.resize(N, -1);
    H.is_tip.resize(N, 0);
    H.blen.resize(N, 0.0);

    for (int i = 0; i < N; ++i) {
        const auto& nd = T.nodes[i];
        H.parent[i] = nd.parent;
        H.left[i]   = nd.left;
        H.right[i]  = nd.right;
        H.is_tip[i] = nd.is_tip ? 1 : 0;
        H.blen[i]   = nd.branch_length_to_parent;
    }
    
    // Build tip name -> MSA row lookup
    std::unordered_map<std::string,int> name2row;
    name2row.reserve(msa_tip_names.size()*2);
    for (int r = 0; r < (int)msa_tip_names.size(); ++r) name2row[msa_tip_names[r]] = r;

    // Collect tip order in postorder and the corresponding node ids
    std::vector<int> tip_node_ids_host;
    tip_node_ids_host.reserve(msa_tip_names.size());
    for (int id : T.postorder) {
        if (T.nodes[id].is_tip) tip_node_ids_host.push_back(id);
    }
    const int tips = (int)tip_node_ids_host.size();

    if (tips != (int)msa_tip_names.size())
        throw std::runtime_error("Tip count in tree != MSA names.");

    H.tip_node_ids = tip_node_ids_host;

    // Encode MSA rows into tipchars following the fixed tipIndex order (0..tips-1)
    H.tipchars.resize((size_t)tips * sites);

    if (states == 4 || states == 5) {
        for (int t = 0; t < tips; ++t) {
            const int node_id = tip_node_ids_host[t];
            const auto& name  = T.nodes[node_id].name;
            auto it = name2row.find(name);
            if (it == name2row.end())
                throw std::runtime_error("Tip not found in MSA: " + name);
            const std::string& row = msa_rows[it->second];
            for (size_t s = 0; s < sites; ++s)
                H.tipchars[(size_t)t * sites + s] = encode_state_DNA5(row[s]);
        }
    } 
    // else if (states == 20) {
    //     auto encAA = [](char c)->uint8_t {
    //         // TODO: fill in your protein mapping; temporarily treat unknown as 19
    //         switch (c) {
    //             case 'A': case 'a': return 0;
    //             // ... remaining 18
    //             default: return 19;
    //         }
    //     };
    //     for (int t = 0; t < tips; ++t) {
    //         const int node_id = tip_node_ids_host[t];
    //         const auto& name  = T.nodes[node_id].name;
    //         auto it = name2row.find(name);
    //         if (it == name2row.end())
    //             throw std::runtime_error("Tip not found in MSA: " + name);
    //         const std::string& row = msa_rows[it->second];
    //         for (size_t s = 0; s < sites; ++s)
    //             H.tipchars[(size_t)t * sites + s] = encAA(row[s]);
    //     }
    // } else {
    //     throw std::runtime_error("Unsupported states; expect 4/5/20.");
    // }

    return H;
}

static inline std::vector<double> transpose_sq(const std::vector<double>& M, int n) {
    std::vector<double> T((size_t)n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T[(size_t)i * n + j] = M[(size_t)j * n + i];
        }
    }
    return T;
}
// ===== Copy host packing to GPU and build DeviceTree =====
// Upload host packing and model parameters to GPU, constructing DeviceTree.
DeviceTree upload_to_gpu(
    const TreeBuildResult& T,
    const HostPacking& H,
    const EigResult& er,
    const std::vector<double>& rate_weights,
    const std::vector<double>& rate_multipliers,
    const std::vector<double>& pi,
    size_t sites, int states, int rate_cats, bool per_rate_scaling,
    const PlacementQueryBatch* queries)
{
    DeviceTree D;
    D.N = (int)T.nodes.size();
    D.tips = (int)H.tip_node_ids.size();
    D.inners = D.N - D.tips;
    D.placement_queries = (queries ? (int)queries->size() : 0);
    D.root_id = T.root_id;
    // Reserve space for future insertions: each committed placement adds 2 nodes (1 internal + 1 tip).
    // This preallocates large CLV/PMAT/topology buffers so the tree can grow without reupload_to_gpu().
    const int reserve_inserts = D.placement_queries;
    D.capacity_N = D.N + 2 * reserve_inserts;
    D.capacity_tips = D.tips + reserve_inserts;
    D.query_capacity = D.placement_queries;
    D.sites = sites;
    D.states = states;
    D.rate_cats = rate_cats;
    D.log2_stride = ceil_log2_u32((unsigned int)(D.states + 1));
    D.per_rate_scaling = per_rate_scaling;
    
    // --- alloc topology ---
    CUDA_CHECK(cudaMalloc(&D.d_lambdas, sizeof(double) * (size_t)D.rate_cats * D.states));
    CUDA_CHECK(cudaMalloc(&D.d_V,       sizeof(double) * D.states * D.states));
    CUDA_CHECK(cudaMalloc(&D.d_Vinv,    sizeof(double) * D.states * D.states));

    CUDA_CHECK(cudaMalloc(&D.d_U,       sizeof(double) * D.states * D.states));
    CUDA_CHECK(cudaMalloc(&D.d_rate_weights, sizeof(double) * D.rate_cats));
    CUDA_CHECK(cudaMalloc(&D.d_frequencies, sizeof(double) * D.states));

    // expand lambdas per rate category, scaling by the per-category rate multiplier.
    {
        std::vector<double> lambdas_scaled((size_t)D.rate_cats * D.states, 0.0);
        for (int rc = 0; rc < D.rate_cats; ++rc) {
            double r = rate_multipliers[rc];
            for (int s = 0; s < D.states; ++s) {
                lambdas_scaled[(size_t)rc * D.states + s] = er.lambdas[s] * r;
            }
        }
        CUDA_CHECK(cudaMemcpy(D.d_lambdas,
                            lambdas_scaled.data(),
                            sizeof(double) * lambdas_scaled.size(),
                            cudaMemcpyHostToDevice));
    }

    //Temporary Fixed : Transpose of Eigen vectors for coalescent model
    auto V_T    = transpose_sq(er.V,    D.states);
    auto Vinv_T = transpose_sq(er.Vinv, D.states);
    auto U_T    = transpose_sq(er.U,    D.states);

    CUDA_CHECK(cudaMemcpy(D.d_V,    V_T.data(),    sizeof(double) * D.states * D.states, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_Vinv, Vinv_T.data(), sizeof(double) * D.states * D.states, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_U,    U_T.data(),    sizeof(double) * D.states * D.states, cudaMemcpyHostToDevice));
    if ((int)rate_weights.size() != rate_cats) {
        throw std::runtime_error("rate_weights size mismatch.");
    }
    CUDA_CHECK(cudaMemcpy(D.d_rate_weights, rate_weights.data(), sizeof(double) * D.rate_cats, cudaMemcpyHostToDevice));
    if ((int)pi.size() != states) {
        throw std::runtime_error("pi size mismatch.");
    }
    CUDA_CHECK(cudaMemcpy(D.d_frequencies, pi.data(), sizeof(double) * D.states, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&D.d_blen,      sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMalloc(&D.d_new_pendant_length,  sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMalloc(&D.d_new_proximal_length, sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMalloc(&D.d_prev_pendant_length,  sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMalloc(&D.d_prev_proximal_length, sizeof(double) * D.capacity_N));
    

    CUDA_CHECK(cudaMemcpy(D.d_blen,      H.blen.data(),      sizeof(double) * D.N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(D.d_new_pendant_length,  0, sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMemset(D.d_new_proximal_length, 0, sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMemset(D.d_prev_pendant_length,  0, sizeof(double) * D.capacity_N));
    CUDA_CHECK(cudaMemset(D.d_prev_proximal_length, 0, sizeof(double) * D.capacity_N));

    // --- tips ---

    CUDA_CHECK(cudaMalloc(&D.d_tipchars, sizeof(uint8_t) * (size_t)D.capacity_tips * sites));
    CUDA_CHECK(cudaMemcpy(D.d_tipchars, H.tipchars.data(), sizeof(uint8_t) * (size_t)D.tips * sites, cudaMemcpyHostToDevice));

    const unsigned int tipmap_size = D.states + 1;


    // --- CLV & Offsets ---
    const size_t per_node = (size_t)sites * (size_t)rate_cats * (size_t)states;
    const size_t clv_elems = (size_t)D.N * per_node;
    const size_t clv_capacity_elems = (size_t)D.capacity_N * per_node;
    const size_t clv_total = clv_capacity_elems * 2; // up + down
    CUDA_CHECK(cudaMalloc(&D.d_clv_up,  sizeof(double) * clv_total));
    D.d_clv_down = D.d_clv_up + clv_capacity_elems;
    D.clv_down_offset_elems = clv_capacity_elems;
    // Midpoint buffer: one per node
    CUDA_CHECK(cudaMalloc(&D.d_clv_mid, sizeof(double) * clv_capacity_elems));
    // Cached parent_down * sibling_up products for midpoint reuse
    CUDA_CHECK(cudaMalloc(&D.d_clv_mid_base, sizeof(double) * clv_capacity_elems));
    // Persistent workspace for downward convergence update.
    CUDA_CHECK(cudaMalloc(&D.d_downward_scratch, sizeof(double) * clv_capacity_elems));
    CUDA_CHECK(cudaMalloc(&D.d_placement_clv, sizeof(double) * D.sites));
    CUDA_CHECK(cudaMemset(D.d_placement_clv, 0, sizeof(double) * D.sites));
    // Reusable derivative workspaces
    const size_t sumtable_stride = (size_t)sites * (size_t)rate_cats * (size_t)states;
    const size_t max_ops = (size_t)D.capacity_N * 2;
    D.sumtable_capacity_ops = max_ops;
    D.likelihood_capacity_ops = max_ops;
    if (sumtable_stride > 0 && max_ops > 0) {
        CUDA_CHECK(cudaMalloc(&D.d_sumtable, sizeof(double) * sumtable_stride * max_ops));
    }
    if (max_ops > 0) {
        CUDA_CHECK(cudaMalloc(&D.d_likelihoods, sizeof(double) * max_ops));
    }

    // Optionally zero out the CLV pool (or let kernels overwrite)
    CUDA_CHECK(cudaMemset(D.d_clv_up, 0, sizeof(double) * clv_total));
    CUDA_CHECK(cudaMemset(D.d_clv_mid, 0, sizeof(double) * clv_capacity_elems));
    CUDA_CHECK(cudaMemset(D.d_clv_mid_base, 0, sizeof(double) * clv_capacity_elems));
    CUDA_CHECK(cudaMemset(D.d_downward_scratch, 0, sizeof(double) * clv_capacity_elems));
    // Initialize only the root slice of down pool to exact 1.0; others remain 0 until overwritten.
    if (per_node > 0) {
        std::vector<double> ones(per_node, 1.0);
        CUDA_CHECK(cudaMemcpy(D.d_clv_down + (size_t)T.root_id * per_node,
                              ones.data(),
                              per_node * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    const size_t pmat_elems_cur = (size_t)D.N * (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
    const size_t pmat_elems_cap = (size_t)D.capacity_N * (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
    const size_t pmat_bytes_cur = sizeof(double) * pmat_elems_cur;
    const size_t pmat_bytes_cap = sizeof(double) * pmat_elems_cap;
    CUDA_CHECK(cudaMalloc(&D.d_pmat, pmat_bytes_cap));
    CUDA_CHECK(cudaMemset(D.d_pmat, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat, H.pmats.data(), pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid, pmat_bytes_cap));
    CUDA_CHECK(cudaMemset(D.d_pmat_mid, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid, H.pmats_mid.data(), pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid_prox, pmat_bytes_cap));
    const double* pmat_mid_prox_src = !H.pmats_mid_prox.empty()
        ? H.pmats_mid_prox.data()
        : H.pmats_mid.data();
    if (!pmat_mid_prox_src) {
        throw std::runtime_error("pmats_mid_prox source is empty.");
    }
    printf("pmat_elems: %zu, pmat_bytes: %zu\n", pmat_elems_cur, pmat_bytes_cur);
    printf("H.pmats_mid_prox.size(): %zu\n", H.pmats_mid_prox.size());
    if (!H.pmats_mid_prox.empty() && H.pmats_mid_prox.size() != pmat_elems_cur) {
        throw std::runtime_error("pmats_mid_prox size mismatch.");
    }
    CUDA_CHECK(cudaMemset(D.d_pmat_mid_prox, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid_prox, pmat_mid_prox_src, pmat_bytes_cur, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&D.d_pmat_mid_dist, pmat_bytes_cap));
    const double* pmat_mid_dist_src = !H.pmats_mid_dist.empty()
        ? H.pmats_mid_dist.data()
        : H.pmats_mid.data();
    if (!pmat_mid_dist_src) {
        throw std::runtime_error("pmats_mid_dist source is empty.");
    }
    if (!H.pmats_mid_dist.empty() && H.pmats_mid_dist.size() != pmat_elems_cur) {
        throw std::runtime_error("pmats_mid_dist size mismatch.");
    }
    CUDA_CHECK(cudaMemset(D.d_pmat_mid_dist, 0, pmat_bytes_cap));
    CUDA_CHECK(cudaMemcpy(D.d_pmat_mid_dist, pmat_mid_dist_src, pmat_bytes_cur, cudaMemcpyHostToDevice));
    // Allocate query PMAT buffer sized for up to ~2*N placement ops (edges).
    {
        const size_t per_query = (size_t)D.rate_cats * (size_t)D.states * (size_t)D.states;
        const size_t query_slots = (size_t)D.capacity_N * 2;
        const size_t query_pmat_bytes = sizeof(double) * per_query * query_slots;
        CUDA_CHECK(cudaMalloc(&D.d_query_pmat, query_pmat_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_pmat, 0, query_pmat_bytes));
    }

    if (queries && !queries->empty()) {
        const size_t qcount = queries->size();
        const size_t qcap = (size_t)D.query_capacity;
        const size_t chars_bytes = sizeof(uint8_t) * qcap * sites;
        CUDA_CHECK(cudaMalloc(&D.d_query_chars, chars_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_chars, 0, chars_bytes));
        CUDA_CHECK(cudaMemcpy(D.d_query_chars, queries->query_chars.data(), sizeof(uint8_t) * qcount * sites, cudaMemcpyHostToDevice));

        const size_t query_clv_elems = qcap * (size_t)sites * (size_t)rate_cats * (size_t)states;
        const size_t query_clv_bytes = sizeof(double) * query_clv_elems;
        CUDA_CHECK(cudaMalloc(&D.d_query_clv, query_clv_bytes));
        CUDA_CHECK(cudaMemset(D.d_query_clv, 0, query_clv_bytes));

    }
    // --- scaler ---
    const size_t scaler_len = per_rate_scaling ? (sites * (size_t)rate_cats) : sites;
    // CUDA_CHECK(cudaMalloc(&D.d_site_scaler, sizeof(unsigned) * scaler_len));
    // CUDA_CHECK(cudaMemset(D.d_site_scaler, 0, sizeof(unsigned) * scaler_len));

    std::vector<unsigned int> tipmap(tipmap_size);
    for (unsigned int j = 0; j < tipmap_size; ++j) {
        if(j == D.states){
            tipmap[j] = 15;  // 0 -> 0001, 1 -> 0010, 2 -> 0100, ...
        }
        else{
            tipmap[j] = 1u << j;  // 0 -> 0001, 1 -> 0010, 2 -> 0100, ...
        }
    }
    CUDA_CHECK(cudaMalloc(&D.d_tipmap, sizeof(unsigned) * tipmap_size));
    CUDA_CHECK(cudaMemcpy(D.d_tipmap, tipmap.data(),  tipmap_size * sizeof(unsigned int), cudaMemcpyHostToDevice));

    return D;
}

// ===== Release GPU resources =====
// Release all device buffers held by DeviceTree.
void free_device_tree(DeviceTree& D) {
    auto F = [](void* p){ if(p) cudaFree(p); };
    F(D.d_blen);
    F(D.d_new_pendant_length); F(D.d_new_proximal_length);
    F(D.d_prev_pendant_length); F(D.d_prev_proximal_length);
    F(D.d_tipchars);
    // d_clv_down is an offset into d_clv_up; free only the base allocation.
    F(D.d_clv_up);
    F(D.d_clv_mid);
    F(D.d_clv_mid_base);
    F(D.d_downward_scratch);
    F(D.d_site_scaler);
    F(D.d_lambdas); F(D.d_V); F(D.d_Vinv); F(D.d_U); F(D.d_rate_weights); F(D.d_frequencies);
    F(D.d_pmat); F(D.d_pmat_mid); F(D.d_pmat_mid_prox); F(D.d_pmat_mid_dist);
    F(D.d_placement_clv);
    F(D.d_sumtable);
    F(D.d_likelihoods);
    F(D.d_query_clv); F(D.d_query_chars);
    F(D.d_query_pmat);
    F(D.d_tipmap);
    D = DeviceTree{};
}


// Assemble a NodeOpInfo record for device execution.
static inline NodeOpInfo make_node_op_device(
    int parent_id,
    int left_id,
    int right_id,
    int left_tip_index,
    int right_tip_index,
    NodeOpType type)
{
    NodeOpInfo op{};
    op.parent_id = parent_id;
    op.left_id = left_id;
    op.right_id = right_id;
    op.left_tip_index = left_tip_index;
    op.right_tip_index = right_tip_index;
    op.op_type = static_cast<int>(type);
    op.clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
    op.dir_tag  = static_cast<uint8_t>(CLV_DIR_UP);
    return op;
}

// Evaluate full tree log-likelihood on device using prepared DeviceTree.
double EvaluateTreeLogLikelihood_device(
    const DeviceTree&      D,
    const TreeBuildResult& T,
    const HostPacking&     H,
    const std::vector<double>& pi,
    const std::vector<double>& rate_weights,
    cudaStream_t stream)
{
    const int N = (int)T.nodes.size();
    std::vector<int> node2tip(N, -1);

    for (int tip_idx = 0; tip_idx < (int)H.tip_node_ids.size(); ++tip_idx) {
        int nid = H.tip_node_ids[tip_idx];
        node2tip[nid] = tip_idx;
    }

    std::vector<NodeOpInfo> ops_host;
    ops_host.reserve(N);

    for (int nid : T.postorder) {
        const TreeNode& nd = T.nodes[nid];
        if (nd.is_tip) continue;

        const int L = nd.left;
        const int R = nd.right;
        const bool tipL = T.nodes[L].is_tip;
        const bool tipR = T.nodes[R].is_tip;

        if (tipL && tipR) {
            int left_tip_idx  = node2tip[L];
            int right_tip_idx = node2tip[R];
            NodeOpInfo op = make_node_op_device(
                nid,
                L,
                R,
                left_tip_idx,
                right_tip_idx,
                OP_TIP_TIP);
            ops_host.push_back(op);
        } else if (tipL && !tipR) {
            int left_tip_idx = node2tip[L];
            NodeOpInfo op = make_node_op_device(
                nid,
                L,
                R,
                left_tip_idx,
                -1,
                OP_TIP_INNER);
            ops_host.push_back(op);
        } else if (!tipL && tipR) {
            int right_tip_idx = node2tip[R];
            NodeOpInfo op = make_node_op_device(
                nid,
                L,
                R,
                -1,
                right_tip_idx,
                OP_TIP_INNER);
            ops_host.push_back(op);
        } else {
            NodeOpInfo op = make_node_op_device(
                nid,
                L,
                R,
                -1,
                -1,
                OP_INNER_INNER);
            ops_host.push_back(op);
        }
    }

    NodeOpInfo* d_ops = nullptr;
    const int num_ops = (int)ops_host.size();
    if (num_ops > 0) {
        CUDA_CHECK(cudaMalloc(&d_ops, sizeof(NodeOpInfo) * (size_t)num_ops));
        CUDA_CHECK(cudaMemcpyAsync(
            d_ops,
            ops_host.data(),
            sizeof(NodeOpInfo) * (size_t)num_ops,
            cudaMemcpyHostToDevice,
            stream));
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;

    // Clamp block size to what the kernel can actually support on this GPU to avoid
    // "too many resources requested for launch" when register pressure is high.
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, Rtree_Likelihood_Site_Parallel_Upward_Kernel));
    int block = 256;
    if (attr.maxThreadsPerBlock > 0 && block > attr.maxThreadsPerBlock) {
        block = attr.maxThreadsPerBlock;
    }

    int max_blocks_per_sm = 4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        Rtree_Likelihood_Site_Parallel_Upward_Kernel,
        block,
        0));
    int max_blocks = num_sms * max_blocks_per_sm;
    int grid = (D.sites + block - 1) / block;
    if (max_blocks > 0 && grid > max_blocks) {
        grid = max_blocks;
    }

    if (num_ops > 0) {
        Rtree_Likelihood_Site_Parallel_Upward_Kernel<<<grid, block, 0, stream>>>(
            D,
            d_ops,
            num_ops);
        CUDA_CHECK(cudaGetLastError());
    }
    if (d_ops) CUDA_CHECK(cudaFree(d_ops));
    double total = root_likelihood::compute_root_loglikelihood_total(
        D,
        T.root_id,
        nullptr,
        nullptr,
        0.0,
        stream);
    printf("Root ID: %d\n", T.root_id);
    printf("Tree loglikelihood = %.10f\n", total);
    return total;
}

void UpdateTreeLogLikelihood_device(
    DeviceTree&                 D,
    TreeBuildResult&            T,
    HostPacking&                H,
    const EigResult&            er,
    PlacementQueryBatch&        Q,
    const std::vector<double>&  rate_multipliers,
    const std::vector<NewPlacementQuery>* queries,
    const std::vector<double>*  pi_debug,
    const std::vector<double>*  rate_weights_debug,
    double                      baseline_loglik,
    bool                        debug_mid,
    int                         smoothing,
    cudaStream_t                stream)
{
    if (!pi_debug || !rate_weights_debug) {
        throw std::runtime_error("UpdateTreeLogLikelihood_device: pi_debug and rate_weights_debug are required.");
    }
    const std::vector<double>& pi = *pi_debug;
    const std::vector<double>& rate_weights = *rate_weights_debug;

    const size_t sites = D.sites;
    const int states = D.states;
    const int rate_cats = D.rate_cats;
    const bool per_rate_scaling = D.per_rate_scaling;
    const double downward_bfs_eps_rel = 1e-6;

    int N = (int)T.nodes.size();
    std::vector<NodeOpInfo> ops_host;
    std::vector<NodeOpInfo> ops_up_host;
    std::vector<int> node2tip(N, -1);

    NodeOpInfo* d_ops = nullptr;
    int num_ops = 0;

    auto free_ops = [&]() {
        if (d_ops) {
            CUDA_CHECK(cudaStreamSynchronize(stream)); // debug safety
            CUDA_CHECK(cudaFree(d_ops));
            d_ops = nullptr;
        }
        num_ops = 0;
    };

    // Build node2tip map (host)
    auto rebuild_node2tip = [&]() {
        N = (int)T.nodes.size();
        node2tip.assign((size_t)N, -1);
        for (int tip_idx = 0; tip_idx < (int)H.tip_node_ids.size(); ++tip_idx) {
            const int nid = H.tip_node_ids[tip_idx];
            if (nid >= 0 && nid < N) node2tip[nid] = tip_idx;
        }
    };


    // Build UPWARD ops into ops_up_host (host)
    auto build_upward_ops_host_full = [&]() {
        ops_up_host.clear();
        ops_up_host.reserve((size_t)N);

        for (int nid : T.postorder) {
            if (nid < 0 || nid >= (int)T.nodes.size()) continue;
            const TreeNode& nd = T.nodes[nid];
            if (nd.is_tip) continue;

            const int L = nd.left;
            const int R = nd.right;
            if (L < 0 || R < 0) continue;

            const bool tipL = (L >= 0 && T.nodes[L].is_tip);
            const bool tipR = (R >= 0 && T.nodes[R].is_tip);
            const int left_tip_idx  = tipL ? node2tip[L] : -1;
            const int right_tip_idx = tipR ? node2tip[R] : -1;

            NodeOpType type;
            if (tipL && tipR) {
                type = OP_TIP_TIP;
            } else if (tipL || tipR) {
                type = OP_TIP_INNER;
            } else {
                type = OP_INNER_INNER;
            }

            NodeOpInfo op{};
            op.parent_id = nid;
            op.left_id = L;
            op.right_id = R;
            op.left_tip_index = left_tip_idx;
            op.right_tip_index = right_tip_idx;
            op.op_type = static_cast<int>(type);
            op.clv_pool = static_cast<uint8_t>(CLV_POOL_UP);
            op.dir_tag  = static_cast<uint8_t>(CLV_DIR_UP);
            ops_up_host.push_back(op);
        }
    };

    // Build DOWNWARD ops into ops_host (host)
    auto build_downward_ops_host = [&]() {
        ops_host.clear();
        ops_host.reserve((size_t)N * 2);

        for (int parent_id : T.preorder) {
            const TreeNode& nd = T.nodes[parent_id];
            if (nd.is_tip) continue;

            const int L = nd.left;
            const int R = nd.right;
            const bool tipL = (L >= 0 && T.nodes[L].is_tip);
            const bool tipR = (R >= 0 && T.nodes[R].is_tip);

            push_down_op(
                ops_host,
                parent_id, L, R,
                tipL, tipR,
                node2tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_LEFT));

            push_down_op(
                ops_host,
                parent_id, L, R,
                tipL, tipR,
                node2tip,
                static_cast<uint8_t>(CLV_DIR_DOWN_RIGHT));
        }

        free_ops();
        num_ops = (int)ops_host.size();
        if (num_ops <= 0) return;

        CUDA_CHECK(cudaMalloc(&d_ops, sizeof(NodeOpInfo) * (size_t)num_ops));
        CUDA_CHECK(cudaMemcpyAsync(
            d_ops,
            ops_host.data(),
            sizeof(NodeOpInfo) * (size_t)num_ops,
            cudaMemcpyHostToDevice,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        check_device_ptr(d_ops, "d_ops");
    };

    // Kernel launch config (computed once; if D changes devices/attrs, recompute)
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int num_sms = prop.multiProcessorCount;

    auto update_upward = [&](const std::vector<NodeOpInfo>& ops) {
        const int num_ops_up = (int)ops.size();
        if (num_ops_up <= 0)  throw std::runtime_error("No upward ops to update");;
        const bool debug_upward_clv_delta = true; // set false to disable CLV before/after checks
        const double changed_eps = 1e-15;

        const size_t per_node = D.per_node_elems();

        NodeOpInfo* d_ops_up = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ops_up, sizeof(NodeOpInfo) * (size_t)num_ops_up));
        CUDA_CHECK(cudaMemcpyAsync(
            d_ops_up,
            ops.data(),
            sizeof(NodeOpInfo) * (size_t)num_ops_up,
            cudaMemcpyHostToDevice,
            stream));

        cudaFuncAttributes attr{};
        CUDA_CHECK(cudaFuncGetAttributes(&attr, Rtree_Likelihood_Site_Parallel_Upward_Kernel));
        int block = 256;
        if (attr.maxThreadsPerBlock > 0 && block > attr.maxThreadsPerBlock) block = attr.maxThreadsPerBlock;

        int max_blocks_per_sm = 4;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            Rtree_Likelihood_Site_Parallel_Upward_Kernel,
            block,
            0));
        int max_blocks = num_sms * max_blocks_per_sm;
        int grid = (int)((D.sites + (size_t)block - 1) / (size_t)block);
        if (max_blocks > 0 && grid > max_blocks) grid = max_blocks;

        Rtree_Likelihood_Site_Parallel_Upward_Kernel<<<grid, block, 0, stream>>>(D, d_ops_up, num_ops_up);
        CHECK_CUDA_LAST();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_ops_up));
    };

    auto update_downward = [&]() {
        if (num_ops <= 0 || !d_ops) return;

        
        cudaFuncAttributes attr{};
        CUDA_CHECK(cudaFuncGetAttributes(&attr, Rtree_Likelihood_Site_Parallel_Downward_Kernel));
        int block = 256;
        if (attr.maxThreadsPerBlock > 0 && block > attr.maxThreadsPerBlock) block = attr.maxThreadsPerBlock;

        int max_blocks_per_sm = 4;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            Rtree_Likelihood_Site_Parallel_Downward_Kernel,
            block,
            0));
        int max_blocks = num_sms * max_blocks_per_sm;
        int grid = (int)((D.sites + (size_t)block - 1) / (size_t)block);
        if (max_blocks > 0 && grid > max_blocks) grid = max_blocks;

        Rtree_Likelihood_Site_Parallel_Downward_Kernel<<<grid, block, 0, stream>>>(D, d_ops, num_ops);
        CHECK_CUDA_LAST();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // --------------------------
    // Initial: build downward CLV once for current tree
    // --------------------------
    rebuild_node2tip();
    build_upward_ops_host_full();
    update_upward(ops_up_host);
    build_downward_ops_host();
    update_downward();
    // keep d_ops for query 0 placement (do NOT free here)

    PlacementPruneConfig prune_cfg{};
    prune_cfg.enable_pruning = true;
    prune_cfg.max_consecutive_drops = 2;
    prune_cfg.drop_threshold = 10;
    prune_cfg.enable_small_frontier_fallback = true;
    prune_cfg.small_frontier_threshold = 16;

    for (int qi = 0; qi < D.placement_queries; ++qi) {
        printf("Placing query %d \n", qi + 1);

        // At the start of each query, ensure d_ops corresponds to CURRENT tree's downward ops.
        // For qi==0 we already built it above; for qi>0, we rebuilt after previous insertion (see below).
        if (num_ops > 0) check_device_ptr(d_ops, "d_ops");

        CUDA_CHECK(cudaMemset(D.d_new_pendant_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_new_proximal_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_prev_pendant_length, 0, sizeof(double) * D.N));
        CUDA_CHECK(cudaMemset(D.d_prev_proximal_length, 0, sizeof(double) * D.N));

        build_query_clv_on_device(D, qi, stream);
        CHECK_CUDA_LAST();

        DeviceTree Dq = make_query_view(D, qi);

	        // d_ops/num_ops must be valid here
	        // PlacementResult pres = PlacementEvaluationKernel(Dq, er, rate_multipliers, d_ops, num_ops, smoothing, stream);
	        PlacementResult pres = PlacementEvaluationKernelPreorderPruned(
	            Dq,
	            T,
	            er,
	            rate_multipliers,
	            d_ops,
	            num_ops,
	            smoothing,
	            prune_cfg,
	            stream,
	            T.root_id);

        printf("Query %d -> insert in: %d (loglik=%.6f) pendant=%.6f proximal=%.6f\n",
               qi, pres.target_id, pres.loglikelihood, pres.pendant_length, pres.proximal_length);

        // const std::string qname = (queries && qi < (int)queries->size())
        //     ? (*queries)[qi].msa_name
        //     : ("query_" + std::to_string(qi));

        // InsertResult ins = insert_query_with_intermediate(T, qname, pres.target_id, pres.pendant_length, pres.proximal_length);

        // rebuild_host_topology_from_tree(T, H);
        // append_query_tip_to_host_packing(H, Q, qi, ins.tip_id, sites);

        // {
        //     const int changed[3] = { pres.target_id, ins.internal_id, ins.tip_id };
        //     fill_pmats_in_host_packing(
        //         T, H, er, pi, rate_multipliers,
        //         states, rate_cats,
        //         changed, 3);
        // }

        // update_insertion_device(D, T, H, sites, states, rate_cats,
        //                                    pres.target_id, ins.internal_id, ins.tip_id, stream);
        // // ---- Update upward CLV for the NEW tree ----
        // // Needed before any downward CLV update / next placement step.
        // rebuild_node2tip();
        // build_upward_ops_to_root(T, node2tip, ins.internal_id, ops_up_host);
        // update_upward(ops_up_host);

        // // ---- Rebuild downward ops + update downward CLV for the NEW tree ----
        // // This prepares d_ops for the NEXT query iteration.
        // build_downward_ops_host();
        // update_downward();
       
    }

    free_ops();
}
