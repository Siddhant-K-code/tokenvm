// tokenvm_stub.cc - Stub implementation for non-CUDA environments
#include "tokenvm.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

// Stub context structure
struct StubContext {
    int device_id;
    std::map<void*, size_t> allocations;
    std::mutex mutex;
    size_t total_allocated;

    StubContext(int id) : device_id(id), total_allocated(0) {}
};

// Global stub data
static std::map<void*, StubContext*> g_contexts;
static std::mutex g_mutex;

// Context management
extern "C" tvm_ctx tvmCreate(int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    StubContext* ctx = new StubContext(device_id);
    g_contexts[ctx] = ctx;
    return ctx;
}

extern "C" void tvmDestroy(tvm_ctx ctx) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    StubContext* stub = static_cast<StubContext*>(ctx);

    // Free all allocations
    for (auto& pair : stub->allocations) {
        free(pair.first);
    }

    g_contexts.erase(ctx);
    delete stub;
}

// Stream management (no-op in stub)
extern "C" tvm_stream tvmStreamCreate(tvm_ctx ctx) {
    static int stream_id = 1;
    return reinterpret_cast<tvm_stream>(stream_id++);
}

extern "C" void tvmStreamDestroy(tvm_stream stream) {
    // No-op
}

// Event management (no-op in stub)
extern "C" tvm_event tvmEventCreate(tvm_ctx ctx) {
    static int event_id = 1;
    return reinterpret_cast<tvm_event>(event_id++);
}

extern "C" void tvmEventRecord(tvm_event event, tvm_stream stream) {
    // No-op
}

extern "C" void tvmStreamWaitEvent(tvm_stream stream, tvm_event event) {
    // No-op
}

extern "C" void tvmEventDestroy(tvm_event event) {
    // No-op
}

// Memory management (CPU memory in stub)
extern "C" tvm_devptr tvmArenaAlloc(tvm_ctx ctx, size_t bytes) {
    if (!ctx || bytes == 0) return nullptr;

    StubContext* stub = static_cast<StubContext*>(ctx);
    std::lock_guard<std::mutex> lock(stub->mutex);

    void* ptr = malloc(bytes);
    if (ptr) {
        stub->allocations[ptr] = bytes;
        stub->total_allocated += bytes;
    }

    return ptr;
}

extern "C" void tvmArenaFree(tvm_ctx ctx, tvm_devptr ptr) {
    if (!ctx || !ptr) return;

    StubContext* stub = static_cast<StubContext*>(ctx);
    std::lock_guard<std::mutex> lock(stub->mutex);

    auto it = stub->allocations.find(ptr);
    if (it != stub->allocations.end()) {
        stub->total_allocated -= it->second;
        stub->allocations.erase(it);
        free(ptr);
    }
}

// Memory operations (CPU memcpy in stub)
extern "C" int tvmMemcpyH2DAsync(tvm_ctx ctx, tvm_devptr dst, const void* src, size_t bytes, tvm_stream stream) {
    if (!dst || !src || bytes == 0) return TVM_ERROR_INVALID_ARGUMENT;
    memcpy(dst, src, bytes);
    return TVM_SUCCESS;
}

extern "C" int tvmMemcpyD2HAsync(tvm_ctx ctx, void* dst, tvm_devptr src, size_t bytes, tvm_stream stream) {
    if (!dst || !src || bytes == 0) return TVM_ERROR_INVALID_ARGUMENT;
    memcpy(dst, src, bytes);
    return TVM_SUCCESS;
}

// Kernel operations (CPU implementation in stub)
extern "C" int tvmPackKV(tvm_ctx ctx, tvm_devptr dst, const void* k_src, const void* v_src,
                        int heads, int d_head, int block_tokens, int dtype) {
    if (!dst || !k_src || !v_src) return TVM_ERROR_INVALID_ARGUMENT;

    // Simple CPU implementation - just concatenate K and V
    size_t element_size = (dtype == TVM_DTYPE_FP32) ? 4 : 2;
    size_t k_size = heads * d_head * block_tokens * element_size;
    size_t v_size = k_size;

    memcpy(dst, k_src, k_size);
    memcpy(static_cast<char*>(dst) + k_size, v_src, v_size);

    return TVM_SUCCESS;
}

extern "C" int tvmGatherKV(tvm_ctx ctx, tvm_devptr dst, const tvm_devptr* block_ptrs, int num_blocks,
                          int heads, int d_head, int block_tokens, int dtype) {
    if (!dst || !block_ptrs || num_blocks <= 0) return TVM_ERROR_INVALID_ARGUMENT;

    // Simple CPU implementation - concatenate blocks
    size_t element_size = (dtype == TVM_DTYPE_FP32) ? 4 : 2;
    size_t block_size = heads * d_head * block_tokens * element_size * 2; // K+V

    for (int i = 0; i < num_blocks; i++) {
        if (block_ptrs[i]) {
            memcpy(static_cast<char*>(dst) + i * block_size, block_ptrs[i], block_size);
        }
    }

    return TVM_SUCCESS;
}
