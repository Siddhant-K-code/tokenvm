// hostpool.go - Pinned host memory pool management
package storage

// #include <stdlib.h>
// #include <string.h>
// #include <cuda_runtime.h>
//
// void* allocPinned(size_t size) {
//     void* ptr = NULL;
//     cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
//     return ptr;
// }
//
// void freePinned(void* ptr) {
//     cudaFreeHost(ptr);
// }
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// HostPool manages pinned host memory
type HostPool struct {
	mu         sync.Mutex
	totalSize  int64
	usedSize   int64
	allocations map[unsafe.Pointer]int64
}

// NewHostPool creates a new pinned host memory pool
func NewHostPool(size int64) (*HostPool, error) {
	return &HostPool{
		totalSize:   size,
		usedSize:    0,
		allocations: make(map[unsafe.Pointer]int64),
	}, nil
}

// Alloc allocates pinned host memory
func (p *HostPool) Alloc(size int64) (unsafe.Pointer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.usedSize+size > p.totalSize {
		return nil, fmt.Errorf("insufficient host memory: need %d, available %d",
			size, p.totalSize-p.usedSize)
	}

	ptr := C.allocPinned(C.size_t(size))
	if ptr == nil {
		return nil, fmt.Errorf("pinned host allocation failed")
	}

	p.allocations[ptr] = size
	p.usedSize += size
	return ptr, nil
}

// Free frees pinned host memory
func (p *HostPool) Free(ptr unsafe.Pointer) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if size, ok := p.allocations[ptr]; ok {
		C.freePinned(ptr)
		delete(p.allocations, ptr)
		p.usedSize -= size
	}
}

// Usage returns the current memory usage ratio
func (p *HostPool) Usage() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.totalSize == 0 {
		return 0
	}
	return float64(p.usedSize) / float64(p.totalSize)
}

// Close releases all host memory
func (p *HostPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for ptr := range p.allocations {
		C.freePinned(ptr)
	}
	p.allocations = make(map[unsafe.Pointer]int64)
	p.usedSize = 0
	return nil
}
