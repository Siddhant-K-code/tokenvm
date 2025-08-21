// tokenvm.h - Flat C ABI for TokenVM CUDA operations
#ifndef TOKENVM_H
#define TOKENVM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef void* tvm_ctx;
typedef void* tvm_stream;
typedef void* tvm_event;
typedef void* tvm_devptr;

// Context management
tvm_ctx tvmCreate(int device_id);
void    tvmDestroy(tvm_ctx ctx);

// Stream management
tvm_stream tvmStreamCreate(tvm_ctx ctx);
void       tvmStreamDestroy(tvm_stream stream);

// Event management
tvm_event tvmEventCreate(tvm_ctx ctx);
void      tvmEventRecord(tvm_event event, tvm_stream stream);
void      tvmStreamWaitEvent(tvm_stream stream, tvm_event event);
void      tvmEventDestroy(tvm_event event);

// Arena memory management
tvm_devptr tvmArenaAlloc(tvm_ctx ctx, size_t bytes);
void       tvmArenaFree(tvm_ctx ctx, tvm_devptr ptr);

// Async memory operations
int tvmMemcpyH2DAsync(tvm_ctx ctx, tvm_devptr dst, const void* src, size_t bytes, tvm_stream stream);
int tvmMemcpyD2HAsync(tvm_ctx ctx, void* dst, tvm_devptr src, size_t bytes, tvm_stream stream);

// KV cache operations (kernels - stubs first, optimize later)
int tvmPackKV(tvm_ctx ctx, tvm_devptr dst, const void* k_src, const void* v_src,
              int heads, int d_head, int block_tokens, int dtype);

int tvmGatherKV(tvm_ctx ctx, tvm_devptr dst, const tvm_devptr* block_ptrs, int num_blocks,
                int heads, int d_head, int block_tokens, int dtype);

// Error codes
#define TVM_SUCCESS 0
#define TVM_ERROR_INVALID_DEVICE -1
#define TVM_ERROR_OUT_OF_MEMORY -2
#define TVM_ERROR_CUDA_FAILURE -3
#define TVM_ERROR_INVALID_ARGUMENT -4

// Data types for KV cache
#define TVM_DTYPE_FP32 0
#define TVM_DTYPE_FP16 1
#define TVM_DTYPE_BF16 2
#define TVM_DTYPE_INT8 3

#ifdef __cplusplus
}
#endif

#endif // TOKENVM_H
