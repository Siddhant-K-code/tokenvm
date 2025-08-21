// pack.cu - CUDA kernels for packing KV cache blocks
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pack kernel for FP32
__global__ void packKVKernel_fp32(
    float* __restrict__ dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    int heads,
    int d_head,
    int block_tokens,
    int total_elements) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Layout: [K_block, V_block] where each block is [heads, block_tokens, d_head]
    int k_offset = tid;
    int v_offset = tid + (heads * block_tokens * d_head);

    if (tid < heads * block_tokens * d_head) {
        dst[k_offset] = k_src[tid];
        dst[v_offset] = v_src[tid];
    }
}

// Pack kernel for FP16
__global__ void packKVKernel_fp16(
    half* __restrict__ dst,
    const half* __restrict__ k_src,
    const half* __restrict__ v_src,
    int heads,
    int d_head,
    int block_tokens,
    int total_elements) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int k_offset = tid;
    int v_offset = tid + (heads * block_tokens * d_head);

    if (tid < heads * block_tokens * d_head) {
        dst[k_offset] = k_src[tid];
        dst[v_offset] = v_src[tid];
    }
}

// Optimized coalesced pack kernel for FP32
__global__ void packKVCoalesced_fp32(
    float4* __restrict__ dst,
    const float4* __restrict__ k_src,
    const float4* __restrict__ v_src,
    int heads,
    int d_head,
    int block_tokens,
    int total_float4s) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_float4s) return;

    int k_offset = tid;
    int v_offset = tid + (heads * block_tokens * d_head / 4);

    if (tid < heads * block_tokens * d_head / 4) {
        dst[k_offset] = k_src[tid];
        dst[v_offset] = v_src[tid];
    }
}

// INT8 quantized pack kernel
__global__ void packKVQuant_int8(
    int8_t* __restrict__ dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    int heads,
    int d_head,
    int block_tokens,
    int total_elements) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    // Simple per-tensor quantization for POC
    // Production would use per-channel or per-token quantization
    const float k_max = 127.0f;
    const float v_max = 127.0f;

    int k_offset = tid;
    int v_offset = tid + (heads * block_tokens * d_head);

    if (tid < heads * block_tokens * d_head) {
        // Quantize K
        float k_val = k_src[tid];
        dst[k_offset] = (int8_t)fminf(fmaxf(k_val * k_max, -128.0f), 127.0f);

        // Quantize V
        float v_val = v_src[tid];
        dst[v_offset] = (int8_t)fminf(fmaxf(v_val * v_max, -128.0f), 127.0f);
    }
}

extern "C" {

// C interface for pack operations
int cudaPackKV_fp32(
    void* dst,
    const void* k_src,
    const void* v_src,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int total_elements = heads * block_tokens * d_head;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Use coalesced version if aligned
    if ((uintptr_t)dst % 16 == 0 &&
        (uintptr_t)k_src % 16 == 0 &&
        (uintptr_t)v_src % 16 == 0 &&
        total_elements % 4 == 0) {

        int total_float4s = total_elements / 4;
        blocks = (total_float4s + threads - 1) / threads;

        packKVCoalesced_fp32<<<blocks, threads, 0, stream>>>(
            (float4*)dst, (const float4*)k_src, (const float4*)v_src,
            heads, d_head, block_tokens, total_float4s);
    } else {
        packKVKernel_fp32<<<blocks, threads, 0, stream>>>(
            (float*)dst, (const float*)k_src, (const float*)v_src,
            heads, d_head, block_tokens, total_elements);
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaPackKV_fp16(
    void* dst,
    const void* k_src,
    const void* v_src,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int total_elements = heads * block_tokens * d_head;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    packKVKernel_fp16<<<blocks, threads, 0, stream>>>(
        (half*)dst, (const half*)k_src, (const half*)v_src,
        heads, d_head, block_tokens, total_elements);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cudaPackKV_int8(
    void* dst,
    const void* k_src,
    const void* v_src,
    void* k_scale,
    void* v_scale,
    int heads,
    int d_head,
    int block_tokens,
    cudaStream_t stream) {

    int total_elements = heads * block_tokens * d_head;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    packKVQuant_int8<<<blocks, threads, 0, stream>>>(
        (int8_t*)dst, (const float*)k_src, (const float*)v_src,
        (float*)k_scale, (float*)v_scale,
        heads, d_head, block_tokens, total_elements);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"
