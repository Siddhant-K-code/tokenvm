// predictor.go - Predictive prefetching policies
package pager

import (
	"sync"
)

// PredictorType defines the type of predictor
type PredictorType int

const (
	NextBlockPredictor PredictorType = iota
	MultiHeadCoupledPredictor
)

// Predictor implements predictive prefetching policy
type Predictor struct {
	mu            sync.Mutex
	predictorType PredictorType
	prefetchDepth int
	history       []BlockID
	maxHistory    int
}

// NewPredictor creates a new predictor policy
func NewPredictor(predictorType PredictorType, prefetchDepth int) Policy {
	return &Predictor{
		predictorType: predictorType,
		prefetchDepth: prefetchDepth,
		history:       make([]BlockID, 0, 1000),
		maxHistory:    1000,
	}
}

// Touch marks a block as accessed and predicts future accesses
func (p *Predictor) Touch(block BlockID) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Add to history
	p.history = append(p.history, block)
	if len(p.history) > p.maxHistory {
		p.history = p.history[1:]
	}
}

// Victim returns a block to evict (delegates to LRU for simplicity)
func (p *Predictor) Victim() (BlockID, bool) {
	// For simplicity, predictor doesn't handle eviction
	// In production, would maintain an LRU alongside prediction
	return BlockID{}, false
}

// Reset clears the predictor state
func (p *Predictor) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.history = p.history[:0]
}

// PredictNext returns blocks likely to be accessed next
func (p *Predictor) PredictNext(current BlockID) []BlockID {
	p.mu.Lock()
	defer p.mu.Unlock()

	predictions := make([]BlockID, 0, p.prefetchDepth)

	switch p.predictorType {
	case NextBlockPredictor:
		// Predict sequential access within same layer/head
		for i := 1; i <= p.prefetchDepth; i++ {
			next := BlockID{
				Layer:    current.Layer,
				Head:     current.Head,
				SeqBlock: current.SeqBlock + i,
			}
			predictions = append(predictions, next)
		}

	case MultiHeadCoupledPredictor:
		// Predict access to same sequence block across heads
		for h := 0; h < 32; h++ { // Assuming 32 heads
			if h != current.Head {
				coupled := BlockID{
					Layer:    current.Layer,
					Head:     h,
					SeqBlock: current.SeqBlock,
				}
				predictions = append(predictions, coupled)
				if len(predictions) >= p.prefetchDepth {
					break
				}
			}
		}
	}

	return predictions
}
