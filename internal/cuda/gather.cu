// gather.cu - CUDA kernels for gathering/scattering KV cache blocks
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Gather kernel for FP32 - collect multiple blocks into contiguous buffer
__global__ void gatherKVKernel_fp32(
    float* __restrict__ dst,
    const float** __restrict__ block_ptrs,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    int block_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size;
    if (tid >= total_elements) return;

    int block_idx = tid / block_size;
    int elem_idx = tid % block_size;

    if (block_idx < num_blocks && block_ptrs[block_idx] != nullptr) {
        dst[tid] = block_ptrs[block_idx][elem_idx];
    }
}

// Gather kernel for FP16
__global__ void gatherKVKernel_fp16(
    half* __restrict__ dst,
    const half** __restrict__ block_ptrs,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    int block_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size;
    if (tid >= total_elements) return;

    int block_idx = tid / block_size;
    int elem_idx = tid % block_size;

    if (block_idx < num_blocks && block_ptrs[block_idx] != nullptr) {
        dst[tid] = block_ptrs[block_idx][elem_idx];
    }
}

// Optimized coalesced gather for aligned memory
__global__ void gatherKVCoalesced_fp32(
    float4* __restrict__ dst,
    const float4** __restrict__ block_ptrs,
    int num_blocks,
    int block_size_float4) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_float4s = num_blocks * block_size_float4;
    if (tid >= total_float4s) return;

    int block_idx = tid / block_size_float4;
    int elem_idx = tid % block_size_float4;

    if (block_idx < num_blocks && block_ptrs[block_idx] != nullptr) {
        dst[tid] = block_ptrs[block_idx][elem_idx];
    }
}

// Scatter kernel - inverse of gather, distribute contiguous buffer to blocks
__global__ void scatterKVKernel_fp32(
    float** __restrict__ block_ptrs,
    const float* __restrict__ src,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    int block_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size;
    if (tid >= total_elements) return;

    int block_idx = tid / block_size;
    int elem_idx = tid % block_size;

    if (block_idx < num_blocks && block_ptrs[block_idx] != nullptr) {
        block_ptrs[block_idx][elem_idx] = src[tid];
    }
}

// Batched gather for multiple sequences
__global__ void gatherKVBatched_fp32(
    float* __restrict__ dst,
    const float** __restrict__ block_ptrs,
    const int* __restrict__ block_indices,
    const int* __restrict__ seq_lengths,
    int batch_size,
    int max_blocks,
    int heads,
    int d_head,
    int block_tokens) {

    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size) return;

    int seq_len = seq_lengths[batch_idx];
    int num_blocks = (seq_len + block_tokens - 1) / block_tokens;
    int block_size = heads * d_head * block_tokens * 2; // K+V

    if (tid >= num_blocks * block_size) return;

    int block_idx = tid / block_size;
    int elem_idx = tid % block_size;

    int global_block_idx = block_indices[batch_idx * max_blocks + block_idx];

    if (block_idx < num_blocks && global_block_idx >= 0) {
        int dst_offset = batch_idx * max_blocks * block_size + tid;
        dst[dst_offset] = block_ptrs[global_block_idx][elem_idx];
    }
}

// Prefetch blocks to L2 cache
__global__ void prefetchBlocks(
    const void** __restrict__ block_ptrs,
    int num_blocks,
    int block_size_bytes) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_blocks) return;

    if (block_ptrs[tid] != nullptr) {
        // Prefetch to L2 cache
        __builtin_prefetch(block_ptrs[tid], 0, 3);

        // For larger blocks, prefetch multiple cache lines
        const char* ptr = (const char*)block_ptrs[tid];
        for (int offset = 0; offset < block_size_bytes; offset += 128) {
            __builtin_prefetch(ptr + offset, 0, 3);
        }
    }
}

extern "C" {

// C interface for gather operations
int cudaGatherKV_fp32(
    void* dst,
    const void** block_ptrs,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int block_size = heads * d_head * block_tokens * 2; // K+V
    int total_elements = num_blocks * block_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Use coalesced version if aligned
    if ((uintptr_t)dst % 16 == 0 && block_size % 4 == 0) {
        int block_size_float4 = block_size / 4;
        int total_float4s = num_blocks * block_size_float4;
        blocks = (total_float4s + threads - 1) / threads;

        gatherKVCoalesced_fp32<<<blocks, threads, 0, stream>>>(
            (float4*)dst, (const float4**)block_ptrs,
            num_blocks, block_size_float4);
    } else {
        gatherKVKernel_fp32<<<blocks, threads, 0, stream>>>(
            (float*)dst, (const float**)block_ptrs,
            num_blocks, heads, d_head, block_tokens, block_size);
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaGatherKV_fp16(
    void* dst,
    const void** block_ptrs,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int block_size = heads * d_head * block_tokens * 2; // K+V
    int total_elements = num_blocks * block_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    gatherKVKernel_fp16<<<blocks, threads, 0, stream>>>(
        (half*)dst, (const half**)block_ptrs,
        num_blocks, heads, d_head, block_tokens, block_size);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaScatterKV_fp32(
    void** block_ptrs,
    const void* src,
    int num_blocks,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int block_size = heads * d_head * block_tokens * 2; // K+V
    int total_elements = num_blocks * block_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    scatterKVKernel_fp32<<<blocks, threads, 0, stream>>>(
        (float**)block_ptrs, (const float*)src,
        num_blocks, heads, d_head, block_tokens, block_size);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaGatherKVBatched(
    void* dst,
    const void** block_ptrs,
    const int* block_indices,
    const int* seq_lengths,
    int batch_size,
    int max_blocks,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int block_size = heads * d_head * block_tokens * 2; // K+V
    int threads = 256;

    dim3 grid_dim;
    grid_dim.x = (max_blocks * block_size + threads - 1) / threads;
    grid_dim.y = batch_size;

    gatherKVBatched_fp32<<<grid_dim, threads, 0, stream>>>(
        (float*)dst, (const float**)block_ptrs,
        block_indices, seq_lengths,
        batch_size, max_blocks, heads, d_head, block_tokens);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaPrefetchBlocks(
    const void** block_ptrs,
    int num_blocks,
    int block_size_bytes,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;

    prefetchBlocks<<<blocks, threads, 0, stream>>>(
        block_ptrs, num_blocks, block_size_bytes);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"
