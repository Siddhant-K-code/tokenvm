// server.go - API server for TokenVM daemon
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"

	"tokenvm/internal/metrics"
	"tokenvm/internal/pager"
)

// Server provides the API interface for TokenVM
type Server struct {
	pager   pager.Pager
	metrics *metrics.Registry
	mu      sync.RWMutex
}

// NewServer creates a new API server
func NewServer(p pager.Pager, m *metrics.Registry) *Server {
	return &Server{
		pager:   p,
		metrics: m,
	}
}

// Request represents an API request
type Request struct {
	Method string          `json:"method"`
	Block  *BlockIDJSON    `json:"block,omitempty"`
	Blocks []BlockIDJSON   `json:"blocks,omitempty"`
}

// Response represents an API response
type Response struct {
	Success bool   `json:"success"`
	DevPtr  uint64 `json:"devptr,omitempty"`
	Error   string `json:"error,omitempty"`
	Stats   *Stats `json:"stats,omitempty"`
}

// BlockIDJSON is the JSON representation of BlockID
type BlockIDJSON struct {
	Layer    int `json:"layer"`
	Head     int `json:"head"`
	SeqBlock int `json:"seq_block"`
}

// Stats represents system statistics
type Stats struct {
	Hits       int64 `json:"hits"`
	Misses     int64 `json:"misses"`
	GPUBlocks  int64 `json:"gpu_blocks"`
	HostBlocks int64 `json:"host_blocks"`
	NVMeBlocks int64 `json:"nvme_blocks"`
}

// Serve starts serving on the given listener
func (s *Server) Serve(listener net.Listener) error {
	for {
		conn, err := listener.Accept()
		if err != nil {
			return err
		}
		go s.handleConnection(conn)
	}
}

// handleConnection handles a client connection
func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	decoder := json.NewDecoder(conn)
	encoder := json.NewEncoder(conn)

	for {
		var req Request
		if err := decoder.Decode(&req); err != nil {
			if err == io.EOF {
				return
			}
			encoder.Encode(Response{
				Success: false,
				Error:   fmt.Sprintf("decode error: %v", err),
			})
			continue
		}

		resp := s.handleRequest(&req)
		if err := encoder.Encode(resp); err != nil {
			return
		}
	}
}

// handleRequest processes a single request
func (s *Server) handleRequest(req *Request) *Response {
	ctx := context.Background()

	switch req.Method {
	case "EnsureGPU":
		if req.Block == nil {
			return &Response{
				Success: false,
				Error:   "missing block parameter",
			}
		}

		blockID := pager.BlockID{
			Layer:    req.Block.Layer,
			Head:     req.Block.Head,
			SeqBlock: req.Block.SeqBlock,
		}

		devPtr, event, err := s.pager.EnsureGPU(ctx, blockID)
		if err != nil {
			return &Response{
				Success: false,
				Error:   err.Error(),
			}
		}

		// Wait for completion
		event.WaitCompute()

		return &Response{
			Success: true,
			DevPtr:  uint64(devPtr),
		}

	case "Prefetch":
		if len(req.Blocks) == 0 {
			return &Response{
				Success: false,
				Error:   "missing blocks parameter",
			}
		}

		blocks := make([]pager.BlockID, len(req.Blocks))
		for i, b := range req.Blocks {
			blocks[i] = pager.BlockID{
				Layer:    b.Layer,
				Head:     b.Head,
				SeqBlock: b.SeqBlock,
			}
		}

		if err := s.pager.Prefetch(ctx, blocks); err != nil {
			return &Response{
				Success: false,
				Error:   err.Error(),
			}
		}

		return &Response{
			Success: true,
		}

	case "GetStats":
		stats := s.pager.GetStats()
		return &Response{
			Success: true,
			Stats: &Stats{
				Hits:       stats.Hits,
				Misses:     stats.Misses,
				GPUBlocks:  stats.GPUBlocks,
				HostBlocks: stats.HostBlocks,
				NVMeBlocks: stats.NVMeBlocks,
			},
		}

	default:
		return &Response{
			Success: false,
			Error:   fmt.Sprintf("unknown method: %s", req.Method),
		}
	}
}
