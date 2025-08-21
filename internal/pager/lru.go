// lru.go - LRU eviction policy implementation
package pager

import (
	"container/list"
	"sync"
)

// LRU implements a Least Recently Used eviction policy
type LRU struct {
	mu       sync.Mutex
	capacity int
	list     *list.List
	items    map[BlockID]*list.Element
}

// lruEntry represents an entry in the LRU list
type lruEntry struct {
	block BlockID
}

// NewLRU creates a new LRU policy
func NewLRU(capacity int) Policy {
	return &LRU{
		capacity: capacity,
		list:     list.New(),
		items:    make(map[BlockID]*list.Element),
	}
}

// Touch marks a block as recently used
func (l *LRU) Touch(block BlockID) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if elem, ok := l.items[block]; ok {
		// Move to front
		l.list.MoveToFront(elem)
		return
	}

	// Add new entry
	entry := &lruEntry{block: block}
	elem := l.list.PushFront(entry)
	l.items[block] = elem

	// Evict if over capacity
	if l.list.Len() > l.capacity {
		oldest := l.list.Back()
		if oldest != nil {
			l.list.Remove(oldest)
			entry := oldest.Value.(*lruEntry)
			delete(l.items, entry.block)
		}
	}
}

// Victim returns the least recently used block for eviction
func (l *LRU) Victim() (BlockID, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.list.Len() == 0 {
		return BlockID{}, false
	}

	// Get LRU element
	oldest := l.list.Back()
	if oldest == nil {
		return BlockID{}, false
	}

	entry := oldest.Value.(*lruEntry)

	// Remove from tracking
	l.list.Remove(oldest)
	delete(l.items, entry.block)

	return entry.block, true
}

// Reset clears the policy state
func (l *LRU) Reset() {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.list.Init()
	l.items = make(map[BlockID]*list.Element)
}
