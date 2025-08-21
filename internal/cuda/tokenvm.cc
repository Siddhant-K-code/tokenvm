// tokenvm.cc - CUDA context, streams, events, and memory operations
#include "tokenvm.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cstring>

// Internal structures
struct TVMContext {
    int device_id;
    cudaStream_t copy_stream;
    cudaStream_t compute_stream;

    // Arena management
    void* arena_base;
    size_t arena_size;
    size_t arena_offset;
    std::mutex arena_mutex;
    std::unordered_map<void*, size_t> allocations;

    TVMContext(int dev_id) : device_id(dev_id), arena_base(nullptr),
                              arena_size(0), arena_offset(0) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&copy_stream);
        cudaStreamCreate(&compute_stream);

        // Allocate arena (8GB default, configurable via env)
        const char* arena_env = getenv("TOKENVM_ARENA_SIZE");
        arena_size = arena_env ? std::stoull(arena_env) : (8ULL << 30);

        if (cudaMalloc(&arena_base, arena_size) != cudaSuccess) {
            arena_base = nullptr;
            arena_size = 0;
        }
    }

    ~TVMContext() {
        if (arena_base) {
            cudaFree(arena_base);
        }
        cudaStreamDestroy(copy_stream);
        cudaStreamDestroy(compute_stream);
    }
};

struct TVMStream {
    cudaStream_t stream;
    TVMContext* ctx;

    TVMStream(TVMContext* context, cudaStream_t s) : ctx(context), stream(s) {}
};

struct TVMEvent {
    cudaEvent_t event;

    TVMEvent() {
        cudaEventCreate(&event);
    }

    ~TVMEvent() {
        cudaEventDestroy(event);
    }
};

// Helper macro for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return TVM_ERROR_CUDA_FAILURE; \
        } \
    } while(0)

// Context management
extern "C" tvm_ctx tvmCreate(int device_id) {
    if (device_id < 0) {
        return nullptr;
    }

    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_id >= device_count) {
        return nullptr;
    }

    try {
        return new TVMContext(device_id);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void tvmDestroy(tvm_ctx ctx) {
    if (ctx) {
        delete static_cast<TVMContext*>(ctx);
    }
}

// Stream management
extern "C" tvm_stream tvmStreamCreate(tvm_ctx ctx) {
    if (!ctx) return nullptr;

    TVMContext* context = static_cast<TVMContext*>(ctx);
    cudaStream_t stream;

    cudaSetDevice(context->device_id);
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return nullptr;
    }

    try {
        return new TVMStream(context, stream);
    } catch (...) {
        cudaStreamDestroy(stream);
        return nullptr;
    }
}

extern "C" void tvmStreamDestroy(tvm_stream stream) {
    if (stream) {
        TVMStream* s = static_cast<TVMStream*>(stream);
        cudaStreamDestroy(s->stream);
        delete s;
    }
}

// Event management
extern "C" tvm_event tvmEventCreate(tvm_ctx ctx) {
    if (!ctx) return nullptr;

    TVMContext* context = static_cast<TVMContext*>(ctx);
    cudaSetDevice(context->device_id);

    try {
        return new TVMEvent();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void tvmEventRecord(tvm_event event, tvm_stream stream) {
    if (!event || !stream) return;

    TVMEvent* e = static_cast<TVMEvent*>(event);
    TVMStream* s = static_cast<TVMStream*>(stream);

    cudaEventRecord(e->event, s->stream);
}

extern "C" void tvmStreamWaitEvent(tvm_stream stream, tvm_event event) {
    if (!stream || !event) return;

    TVMStream* s = static_cast<TVMStream*>(stream);
    TVMEvent* e = static_cast<TVMEvent*>(event);

    cudaStreamWaitEvent(s->stream, e->event, 0);
}

extern "C" void tvmEventDestroy(tvm_event event) {
    if (event) {
        delete static_cast<TVMEvent*>(event);
    }
}

// Arena memory management (simple bump allocator for POC)
extern "C" tvm_devptr tvmArenaAlloc(tvm_ctx ctx, size_t bytes) {
    if (!ctx || bytes == 0) return nullptr;

    TVMContext* context = static_cast<TVMContext*>(ctx);

    // Align to 256 bytes for GPU efficiency
    bytes = (bytes + 255) & ~255;

    std::lock_guard<std::mutex> lock(context->arena_mutex);

    if (context->arena_offset + bytes > context->arena_size) {
        return nullptr; // Out of arena space
    }

    void* ptr = static_cast<char*>(context->arena_base) + context->arena_offset;
    context->arena_offset += bytes;
    context->allocations[ptr] = bytes;

    return ptr;
}

extern "C" void tvmArenaFree(tvm_ctx ctx, tvm_devptr ptr) {
    if (!ctx || !ptr) return;

    TVMContext* context = static_cast<TVMContext*>(ctx);
    std::lock_guard<std::mutex> lock(context->arena_mutex);

    // For POC, just remove from tracking - no actual free
    // A production version would use a proper allocator
    context->allocations.erase(ptr);
}

// Async memory operations
extern "C" int tvmMemcpyH2DAsync(tvm_ctx ctx, tvm_devptr dst, const void* src,
                                 size_t bytes, tvm_stream stream) {
    if (!ctx || !dst || !src || bytes == 0) {
        return TVM_ERROR_INVALID_ARGUMENT;
    }

    TVMContext* context = static_cast<TVMContext*>(ctx);
    TVMStream* s = stream ? static_cast<TVMStream*>(stream) : nullptr;
    cudaStream_t cuda_stream = s ? s->stream : context->copy_stream;

    cudaSetDevice(context->device_id);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, cuda_stream));

    return TVM_SUCCESS;
}

extern "C" int tvmMemcpyD2HAsync(tvm_ctx ctx, void* dst, tvm_devptr src,
                                 size_t bytes, tvm_stream stream) {
    if (!ctx || !dst || !src || bytes == 0) {
        return TVM_ERROR_INVALID_ARGUMENT;
    }

    TVMContext* context = static_cast<TVMContext*>(ctx);
    TVMStream* s = stream ? static_cast<TVMStream*>(stream) : nullptr;
    cudaStream_t cuda_stream = s ? s->stream : context->copy_stream;

    cudaSetDevice(context->device_id);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cuda_stream));

    return TVM_SUCCESS;
}

// Kernel stubs (to be implemented in .cu files)
extern "C" int tvmPackKV(tvm_ctx ctx, tvm_devptr dst, const void* k_src, const void* v_src,
                        int heads, int d_head, int block_tokens, int dtype) {
    if (!ctx || !dst || !k_src || !v_src) {
        return TVM_ERROR_INVALID_ARGUMENT;
    }

    // Stub implementation - just copy for now
    TVMContext* context = static_cast<TVMContext*>(ctx);
    cudaSetDevice(context->device_id);

    size_t k_size = heads * d_head * block_tokens * sizeof(float);
    size_t v_size = k_size;

    CUDA_CHECK(cudaMemcpyAsync(dst, k_src, k_size, cudaMemcpyHostToDevice, context->compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + k_size, v_src, v_size,
                               cudaMemcpyHostToDevice, context->compute_stream));

    return TVM_SUCCESS;
}

extern "C" int tvmGatherKV(tvm_ctx ctx, tvm_devptr dst, const tvm_devptr* block_ptrs,
                          int num_blocks, int heads, int d_head, int block_tokens, int dtype) {
    if (!ctx || !dst || !block_ptrs || num_blocks <= 0) {
        return TVM_ERROR_INVALID_ARGUMENT;
    }

    // Stub implementation - will be replaced with optimized kernel
    TVMContext* context = static_cast<TVMContext*>(ctx);
    cudaSetDevice(context->device_id);

    size_t block_size = heads * d_head * block_tokens * sizeof(float) * 2; // K+V

    for (int i = 0; i < num_blocks; i++) {
        if (!block_ptrs[i]) continue;

        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + i * block_size,
                                  block_ptrs[i], block_size,
                                  cudaMemcpyDeviceToDevice, context->compute_stream));
    }

    return TVM_SUCCESS;
}
