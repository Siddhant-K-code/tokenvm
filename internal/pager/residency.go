// residency.go - Block residency tracking
package pager

import (
	"sync"
)

// ResidencyMap tracks block locations across tiers
type ResidencyMap struct {
	mu       sync.RWMutex
	blocks   map[BlockID]*BlockMetadata
	byTier   map[Tier]map[BlockID]struct{}
}

// NewResidencyMap creates a new residency map
func NewResidencyMap() Residency {
	return &ResidencyMap{
		blocks: make(map[BlockID]*BlockMetadata),
		byTier: map[Tier]map[BlockID]struct{}{
			GPU:  make(map[BlockID]struct{}),
			HOST: make(map[BlockID]struct{}),
			NVME: make(map[BlockID]struct{}),
		},
	}
}

// Locate returns the tier where a block resides
func (r *ResidencyMap) Locate(id BlockID) (Tier, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if meta, ok := r.blocks[id]; ok {
		return meta.Tier, true
	}
	return 0, false
}

// Set updates the tier for a block
func (r *ResidencyMap) Set(id BlockID, tier Tier) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Remove from old tier
	if meta, ok := r.blocks[id]; ok {
		delete(r.byTier[meta.Tier], id)
	}

	// Add to new tier
	r.byTier[tier][id] = struct{}{}

	// Update or create metadata
	if meta, ok := r.blocks[id]; ok {
		meta.Tier = tier
	} else {
		r.blocks[id] = &BlockMetadata{
			ID:   id,
			Tier: tier,
		}
	}
}

// Delete removes a block from tracking
func (r *ResidencyMap) Delete(id BlockID) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if meta, ok := r.blocks[id]; ok {
		delete(r.byTier[meta.Tier], id)
		delete(r.blocks, id)
	}
}

// GetMetadata returns full metadata for a block
func (r *ResidencyMap) GetMetadata(id BlockID) (*BlockMetadata, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	meta, ok := r.blocks[id]
	return meta, ok
}

// SetMetadata updates full metadata for a block
func (r *ResidencyMap) SetMetadata(id BlockID, meta *BlockMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Remove from old tier if exists
	if oldMeta, ok := r.blocks[id]; ok {
		delete(r.byTier[oldMeta.Tier], id)
	}

	// Add to new tier
	r.byTier[meta.Tier][id] = struct{}{}
	r.blocks[id] = meta
}

// GetBlocksByTier returns all blocks in a specific tier
func (r *ResidencyMap) GetBlocksByTier(tier Tier) []BlockID {
	r.mu.RLock()
	defer r.mu.RUnlock()

	blocks := make([]BlockID, 0, len(r.byTier[tier]))
	for id := range r.byTier[tier] {
		blocks = append(blocks, id)
	}
	return blocks
}

// Count returns the number of blocks in each tier
func (r *ResidencyMap) Count() map[Tier]int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	counts := make(map[Tier]int)
	for tier, blocks := range r.byTier {
		counts[tier] = len(blocks)
	}
	return counts
}
