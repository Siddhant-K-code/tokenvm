// gpuarena.go - GPU VRAM arena management
package storage

// #cgo CFLAGS: -I../cuda
// #cgo LDFLAGS: -L../../build -ltokenvm -lcudart
// #include "tokenvm.h"
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// GPUArena manages GPU memory allocation
type GPUArena struct {
	mu       sync.Mutex
	ctx      unsafe.Pointer
	deviceID int
	totalSize int64
	usedSize  int64
}

// NewGPUArena creates a new GPU arena
func NewGPUArena(deviceID int, size int64) (*GPUArena, error) {
	ctx := C.tvmCreate(C.int(deviceID))
	if ctx == nil {
		return nil, fmt.Errorf("failed to create GPU context for device %d", deviceID)
	}

	return &GPUArena{
		ctx:       ctx,
		deviceID:  deviceID,
		totalSize: size,
		usedSize:  0,
	}, nil
}

// Alloc allocates GPU memory
func (a *GPUArena) Alloc(size int64) (uintptr, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.usedSize+size > a.totalSize {
		return 0, fmt.Errorf("insufficient GPU memory: need %d, available %d",
			size, a.totalSize-a.usedSize)
	}

	ptr := C.tvmArenaAlloc(a.ctx, C.size_t(size))
	if ptr == nil {
		return 0, fmt.Errorf("GPU allocation failed")
	}

	a.usedSize += size
	return uintptr(ptr), nil
}

// Free frees GPU memory
func (a *GPUArena) Free(ptr uintptr) {
	a.mu.Lock()
	defer a.mu.Unlock()

	C.tvmArenaFree(a.ctx, unsafe.Pointer(ptr))
	// Note: In production, track allocation sizes for accurate usedSize update
}

// CopyH2DAsync copies from host to device asynchronously
func (a *GPUArena) CopyH2DAsync(dst uintptr, src unsafe.Pointer, size int64) error {
	result := C.tvmMemcpyH2DAsync(a.ctx, unsafe.Pointer(dst), src, C.size_t(size), nil)
	if result != 0 {
		return fmt.Errorf("H2D copy failed with error %d", result)
	}
	return nil
}

// CopyD2HAsync copies from device to host asynchronously
func (a *GPUArena) CopyD2HAsync(dst unsafe.Pointer, src uintptr, size int64) error {
	result := C.tvmMemcpyD2HAsync(a.ctx, dst, unsafe.Pointer(src), C.size_t(size), nil)
	if result != 0 {
		return fmt.Errorf("D2H copy failed with error %d", result)
	}
	return nil
}

// Usage returns the current memory usage ratio
func (a *GPUArena) Usage() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.totalSize == 0 {
		return 0
	}
	return float64(a.usedSize) / float64(a.totalSize)
}

// Close releases GPU resources
func (a *GPUArena) Close() error {
	if a.ctx != nil {
		C.tvmDestroy(a.ctx)
		a.ctx = nil
	}
	return nil
}
