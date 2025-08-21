// twoq.go - 2Q eviction policy implementation
package pager

import (
	"container/list"
	"sync"
)

// TwoQueue implements the 2Q eviction policy
type TwoQueue struct {
	mu       sync.Mutex
	capacity int
	inRatio  float64

	// A1in: FIFO for recent entries
	a1in     *list.List
	a1inMap  map[BlockID]*list.Element

	// A1out: Ghost entries (recently evicted from A1in)
	a1out    *list.List
	a1outMap map[BlockID]*list.Element

	// Am: LRU for frequently accessed entries
	am       *list.List
	amMap    map[BlockID]*list.Element
}

// twoQEntry represents an entry in the 2Q lists
type twoQEntry struct {
	block BlockID
}

// NewTwoQueue creates a new 2Q policy
func NewTwoQueue(capacity int, inRatio float64) Policy {
	if inRatio <= 0 || inRatio >= 1 {
		inRatio = 0.25 // Default to 25% for A1in
	}

	return &TwoQueue{
		capacity: capacity,
		inRatio:  inRatio,
		a1in:     list.New(),
		a1inMap:  make(map[BlockID]*list.Element),
		a1out:    list.New(),
		a1outMap: make(map[BlockID]*list.Element),
		am:       list.New(),
		amMap:    make(map[BlockID]*list.Element),
	}
}

// Touch marks a block as accessed
func (q *TwoQueue) Touch(block BlockID) {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Check if in Am (frequent list)
	if elem, ok := q.amMap[block]; ok {
		q.am.MoveToFront(elem)
		return
	}

	// Check if in A1out (ghost list)
	if elem, ok := q.a1outMap[block]; ok {
		// Promote to Am
		q.a1out.Remove(elem)
		delete(q.a1outMap, block)

		entry := &twoQEntry{block: block}
		elem = q.am.PushFront(entry)
		q.amMap[block] = elem

		// Evict from Am if needed
		q.evictFromAm()
		return
	}

	// Check if in A1in
	if _, ok := q.a1inMap[block]; ok {
		// Already in A1in, do nothing (FIFO)
		return
	}

	// New entry - add to A1in
	entry := &twoQEntry{block: block}
	elem := q.a1in.PushFront(entry)
	q.a1inMap[block] = elem

	// Evict from A1in if needed
	q.evictFromA1in()
}

// evictFromA1in evicts from the A1in queue if needed
func (q *TwoQueue) evictFromA1in() {
	a1inMax := int(float64(q.capacity) * q.inRatio)

	for q.a1in.Len() > a1inMax {
		oldest := q.a1in.Back()
		if oldest == nil {
			break
		}

		entry := oldest.Value.(*twoQEntry)
		q.a1in.Remove(oldest)
		delete(q.a1inMap, entry.block)

		// Move to A1out (ghost list)
		elem := q.a1out.PushFront(entry)
		q.a1outMap[entry.block] = elem

		// Limit A1out size
		if q.a1out.Len() > q.capacity/2 {
			ghost := q.a1out.Back()
			if ghost != nil {
				ghostEntry := ghost.Value.(*twoQEntry)
				q.a1out.Remove(ghost)
				delete(q.a1outMap, ghostEntry.block)
			}
		}
	}
}

// evictFromAm evicts from the Am queue if needed
func (q *TwoQueue) evictFromAm() {
	amMax := q.capacity - int(float64(q.capacity)*q.inRatio)

	for q.am.Len() > amMax {
		oldest := q.am.Back()
		if oldest == nil {
			break
		}

		entry := oldest.Value.(*twoQEntry)
		q.am.Remove(oldest)
		delete(q.amMap, entry.block)
	}
}

// Victim returns a block to evict
func (q *TwoQueue) Victim() (BlockID, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	// First try to evict from A1in (FIFO)
	if q.a1in.Len() > 0 {
		oldest := q.a1in.Back()
		if oldest != nil {
			entry := oldest.Value.(*twoQEntry)
			q.a1in.Remove(oldest)
			delete(q.a1inMap, entry.block)
			return entry.block, true
		}
	}

	// Then try Am (LRU)
	if q.am.Len() > 0 {
		oldest := q.am.Back()
		if oldest != nil {
			entry := oldest.Value.(*twoQEntry)
			q.am.Remove(oldest)
			delete(q.amMap, entry.block)
			return entry.block, true
		}
	}

	return BlockID{}, false
}

// Reset clears the policy state
func (q *TwoQueue) Reset() {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.a1in.Init()
	q.a1inMap = make(map[BlockID]*list.Element)
	q.a1out.Init()
	q.a1outMap = make(map[BlockID]*list.Element)
	q.am.Init()
	q.amMap = make(map[BlockID]*list.Element)
}
