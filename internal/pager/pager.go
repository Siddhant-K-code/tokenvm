// pager.go - Core pager implementation with tiered storage management
package pager

import (
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"tokenvm/internal/metrics"
	"tokenvm/internal/storage"
)

// Tier represents storage tier levels
type Tier int

const (
	GPU Tier = iota
	HOST
	NVME
)

// BlockID uniquely identifies a KV cache block
type BlockID struct {
	Layer    int
	Head     int
	SeqBlock int
}

// BlockMetadata contains block metadata
type BlockMetadata struct {
	ID         BlockID
	Tier       Tier
	DevPtr     uintptr
	HostPtr    unsafe.Pointer
	LastAccess time.Time
	AccessCount int64
	Size       int64
	Pinned     bool
}

// Residency manages block location tracking
type Residency interface {
	Locate(BlockID) (Tier, bool)
	Set(BlockID, Tier)
	Delete(BlockID)
	GetMetadata(BlockID) (*BlockMetadata, bool)
	SetMetadata(BlockID, *BlockMetadata)
}

// Policy defines eviction policy interface
type Policy interface {
	Touch(BlockID)
	Victim() (BlockID, bool)
	Reset()
}

// Event represents an async operation completion event
type Event interface {
	WaitCompute()
	Done() bool
}

// Pager manages multi-tier KV cache storage
type Pager interface {
	EnsureGPU(ctx context.Context, blk BlockID) (devPtr uintptr, ready Event, err error)
	DemoteAsync(ctx context.Context, blk BlockID) error
	Prefetch(ctx context.Context, blocks []BlockID) error
	ProcessPrefetchQueue(ctx context.Context)
	ProcessEvictions(ctx context.Context)
	GetStats() *Stats
}

// Stats contains pager statistics
type Stats struct {
	Hits       int64
	Misses     int64
	GPUBlocks  int64
	HostBlocks int64
	NVMeBlocks int64
	Evictions  int64
	Promotions int64
}

// pagerImpl is the main pager implementation
type pagerImpl struct {
	gpu       *storage.GPUArena
	host      *storage.HostPool
	nvme      *storage.NVMeStore
	residency Residency
	policy    Policy
	metrics   *metrics.Registry

	prefetchQueue chan BlockID
	prefetchDepth int

	mu    sync.RWMutex
	stats Stats

	// Transfer tracking
	transfers sync.Map // BlockID -> *Transfer
}

// Transfer tracks ongoing block transfers
type Transfer struct {
	BlockID   BlockID
	FromTier  Tier
	ToTier    Tier
	StartTime time.Time
	Event     Event
	Error     error
}

// NewPager creates a new pager instance
func NewPager(
	gpu *storage.GPUArena,
	host *storage.HostPool,
	nvme *storage.NVMeStore,
	policy Policy,
	metrics *metrics.Registry,
	prefetchDepth int,
) Pager {
	return &pagerImpl{
		gpu:           gpu,
		host:          host,
		nvme:          nvme,
		residency:     NewResidencyMap(),
		policy:        policy,
		metrics:       metrics,
		prefetchQueue: make(chan BlockID, prefetchDepth*10),
		prefetchDepth: prefetchDepth,
	}
}

// EnsureGPU ensures a block is resident in GPU memory
func (p *pagerImpl) EnsureGPU(ctx context.Context, blk BlockID) (uintptr, Event, error) {
	// Check if already in GPU
	p.mu.RLock()
	if meta, ok := p.residency.GetMetadata(blk); ok && meta.Tier == GPU {
		p.stats.Hits++
		p.mu.RUnlock()
		p.policy.Touch(blk)
		p.metrics.RecordHit(GPU)
		return meta.DevPtr, &immediateEvent{}, nil
	}
	p.mu.RUnlock()

	// Miss - need to promote
	p.mu.Lock()
	defer p.mu.Unlock()

	p.stats.Misses++
	p.metrics.RecordMiss()

	// Check again under write lock
	if meta, ok := p.residency.GetMetadata(blk); ok && meta.Tier == GPU {
		p.stats.Hits++
		p.policy.Touch(blk)
		return meta.DevPtr, &immediateEvent{}, nil
	}

	// Find source tier
	meta, ok := p.residency.GetMetadata(blk)
	if !ok {
		// Block not in any tier - load from NVMe
		return p.loadFromNVMe(ctx, blk)
	}

	// Promote from current tier
	switch meta.Tier {
	case HOST:
		return p.promoteFromHost(ctx, blk, meta)
	case NVME:
		return p.promoteFromNVMe(ctx, blk, meta)
	default:
		return 0, nil, fmt.Errorf("block already in GPU")
	}
}

// loadFromNVMe loads a block from NVMe to GPU
func (p *pagerImpl) loadFromNVMe(ctx context.Context, blk BlockID) (uintptr, Event, error) {
	// Allocate GPU memory
	size := p.calculateBlockSize(blk)
	devPtr, err := p.gpu.Alloc(size)
	if err != nil {
		// Try eviction
		if err := p.evictToMakeSpace(ctx, size); err != nil {
			return 0, nil, fmt.Errorf("GPU allocation failed: %w", err)
		}
		devPtr, err = p.gpu.Alloc(size)
		if err != nil {
			return 0, nil, err
		}
	}

	// Allocate staging buffer in host memory
	hostPtr, err := p.host.Alloc(size)
	if err != nil {
		p.gpu.Free(devPtr)
		return 0, nil, fmt.Errorf("host allocation failed: %w", err)
	}

	// Start async NVMe read
	event := &asyncEvent{done: make(chan struct{})}
	go func() {
		defer close(event.done)

		// Read from NVMe to host
		if err := p.nvme.ReadBlock(blk, hostPtr); err != nil {
			p.gpu.Free(devPtr)
			p.host.Free(hostPtr)
			event.err = err
			return
		}

		// Copy from host to GPU
		if err := p.gpu.CopyH2DAsync(devPtr, hostPtr, size); err != nil {
			p.gpu.Free(devPtr)
			p.host.Free(hostPtr)
			event.err = err
			return
		}

		// Free staging buffer
		p.host.Free(hostPtr)

		// Update metadata
		p.mu.Lock()
		p.residency.SetMetadata(blk, &BlockMetadata{
			ID:          blk,
			Tier:        GPU,
			DevPtr:      devPtr,
			LastAccess:  time.Now(),
			AccessCount: 1,
			Size:        size,
		})
		p.stats.GPUBlocks++
		p.stats.Promotions++
		p.mu.Unlock()

		p.policy.Touch(blk)
		p.metrics.RecordPromotion(NVME, GPU)
	}()

	return devPtr, event, nil
}

// promoteFromHost promotes a block from host to GPU
func (p *pagerImpl) promoteFromHost(ctx context.Context, blk BlockID, meta *BlockMetadata) (uintptr, Event, error) {
	// Allocate GPU memory
	devPtr, err := p.gpu.Alloc(meta.Size)
	if err != nil {
		// Try eviction
		if err := p.evictToMakeSpace(ctx, meta.Size); err != nil {
			return 0, nil, fmt.Errorf("GPU allocation failed: %w", err)
		}
		devPtr, err = p.gpu.Alloc(meta.Size)
		if err != nil {
			return 0, nil, err
		}
	}

	// Start async H2D copy
	event := &asyncEvent{done: make(chan struct{})}
	go func() {
		defer close(event.done)

		if err := p.gpu.CopyH2DAsync(devPtr, meta.HostPtr, meta.Size); err != nil {
			p.gpu.Free(devPtr)
			event.err = err
			return
		}

		// Update metadata
		p.mu.Lock()
		meta.Tier = GPU
		meta.DevPtr = devPtr
		meta.LastAccess = time.Now()
		meta.AccessCount++
		p.stats.HostBlocks--
		p.stats.GPUBlocks++
		p.stats.Promotions++
		p.mu.Unlock()

		p.policy.Touch(blk)
		p.metrics.RecordPromotion(HOST, GPU)
	}()

	return devPtr, event, nil
}

// promoteFromNVMe promotes a block from NVMe to GPU
func (p *pagerImpl) promoteFromNVMe(ctx context.Context, blk BlockID, meta *BlockMetadata) (uintptr, Event, error) {
	return p.loadFromNVMe(ctx, blk)
}

// DemoteAsync asynchronously demotes a block to a lower tier
func (p *pagerImpl) DemoteAsync(ctx context.Context, blk BlockID) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	meta, ok := p.residency.GetMetadata(blk)
	if !ok {
		return fmt.Errorf("block not found")
	}

	if meta.Pinned {
		return fmt.Errorf("cannot demote pinned block")
	}

	switch meta.Tier {
	case GPU:
		// Demote to host
		return p.demoteToHost(ctx, blk, meta)
	case HOST:
		// Demote to NVMe
		return p.demoteToNVMe(ctx, blk, meta)
	default:
		return nil // Already at lowest tier
	}
}

// demoteToHost demotes a block from GPU to host
func (p *pagerImpl) demoteToHost(ctx context.Context, blk BlockID, meta *BlockMetadata) error {
	// Allocate host memory
	hostPtr, err := p.host.Alloc(meta.Size)
	if err != nil {
		// Try to demote something to NVMe
		if err := p.evictHostToNVMe(ctx, meta.Size); err != nil {
			return fmt.Errorf("host allocation failed: %w", err)
		}
		hostPtr, err = p.host.Alloc(meta.Size)
		if err != nil {
			return err
		}
	}

	// Start async D2H copy
	go func() {
		if err := p.gpu.CopyD2HAsync(hostPtr, meta.DevPtr, meta.Size); err != nil {
			p.host.Free(hostPtr)
			return
		}

		p.mu.Lock()
		p.gpu.Free(meta.DevPtr)
		meta.Tier = HOST
		meta.HostPtr = hostPtr
		meta.DevPtr = 0
		p.stats.GPUBlocks--
		p.stats.HostBlocks++
		p.stats.Evictions++
		p.mu.Unlock()

		p.metrics.RecordDemotion(GPU, HOST)
	}()

	return nil
}

// demoteToNVMe demotes a block from host to NVMe
func (p *pagerImpl) demoteToNVMe(ctx context.Context, blk BlockID, meta *BlockMetadata) error {
	go func() {
		if err := p.nvme.WriteBlock(blk, meta.HostPtr); err != nil {
			return
		}

		p.mu.Lock()
		p.host.Free(meta.HostPtr)
		meta.Tier = NVME
		meta.HostPtr = nil
		p.stats.HostBlocks--
		p.stats.NVMeBlocks++
		p.stats.Evictions++
		p.mu.Unlock()

		p.metrics.RecordDemotion(HOST, NVME)
	}()

	return nil
}

// Prefetch queues blocks for prefetching
func (p *pagerImpl) Prefetch(ctx context.Context, blocks []BlockID) error {
	for _, blk := range blocks {
		select {
		case p.prefetchQueue <- blk:
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Queue full, skip
		}
	}
	return nil
}

// ProcessPrefetchQueue processes queued prefetch requests
func (p *pagerImpl) ProcessPrefetchQueue(ctx context.Context) {
	for i := 0; i < p.prefetchDepth; i++ {
		select {
		case blk := <-p.prefetchQueue:
			go p.EnsureGPU(ctx, blk)
		case <-ctx.Done():
			return
		default:
			return
		}
	}
}

// ProcessEvictions processes evictions based on policy
func (p *pagerImpl) ProcessEvictions(ctx context.Context) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check GPU memory pressure
	if p.gpu.Usage() < 0.9 {
		return
	}

	// Evict based on policy
	for i := 0; i < 10; i++ {
		victim, ok := p.policy.Victim()
		if !ok {
			break
		}

		if err := p.DemoteAsync(ctx, victim); err == nil {
			break
		}
	}
}

// evictToMakeSpace evicts blocks to make space in GPU
func (p *pagerImpl) evictToMakeSpace(ctx context.Context, size int64) error {
	evicted := int64(0)
	attempts := 0
	maxAttempts := 100

	for evicted < size && attempts < maxAttempts {
		victim, ok := p.policy.Victim()
		if !ok {
			return fmt.Errorf("no eviction candidates")
		}

		meta, ok := p.residency.GetMetadata(victim)
		if !ok || meta.Tier != GPU || meta.Pinned {
			attempts++
			continue
		}

		if err := p.demoteToHost(ctx, victim, meta); err == nil {
			evicted += meta.Size
		}
		attempts++
	}

	if evicted < size {
		return fmt.Errorf("insufficient space after eviction")
	}

	return nil
}

// evictHostToNVMe evicts host blocks to NVMe
func (p *pagerImpl) evictHostToNVMe(ctx context.Context, size int64) error {
	// Simple implementation - evict LRU blocks from host
	// Production would use more sophisticated selection
	return nil
}

// calculateBlockSize calculates the size of a block
func (p *pagerImpl) calculateBlockSize(blk BlockID) int64 {
	// Simplified calculation
	// Real implementation would consider model dimensions
	blockTokens := 128
	heads := 32
	dHead := 128
	return int64(blockTokens * heads * dHead * 4 * 2) // FP32 K+V
}

// GetStats returns current statistics
func (p *pagerImpl) GetStats() *Stats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := p.stats
	return &stats
}

// immediateEvent is an event that's already complete
type immediateEvent struct{}

func (e *immediateEvent) WaitCompute() {}
func (e *immediateEvent) Done() bool   { return true }

// asyncEvent represents an async operation
type asyncEvent struct {
	done chan struct{}
	err  error
}

func (e *asyncEvent) WaitCompute() {
	<-e.done
}

func (e *asyncEvent) Done() bool {
	select {
	case <-e.done:
		return true
	default:
		return false
	}
}
