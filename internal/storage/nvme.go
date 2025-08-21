// nvme.go - NVMe storage management with io_uring (simplified for POC)
package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"tokenvm/internal/pager"
)

// NVMeStore manages NVMe-backed block storage
type NVMeStore struct {
	mu         sync.RWMutex
	basePath   string
	blockSize  int64
	blockFiles map[pager.BlockID]*os.File
}

// NewNVMeStore creates a new NVMe storage backend
func NewNVMeStore(basePath string, blockTokens int) (*NVMeStore, error) {
	// Create base directory
	if err := os.MkdirAll(basePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create NVMe storage path: %w", err)
	}

	// Calculate block size (simplified)
	// blockTokens * heads * d_head * sizeof(float32) * 2 (K+V)
	blockSize := int64(blockTokens * 32 * 128 * 4 * 2)

	return &NVMeStore{
		basePath:   basePath,
		blockSize:  blockSize,
		blockFiles: make(map[pager.BlockID]*os.File),
	}, nil
}

// getBlockPath returns the file path for a block
func (s *NVMeStore) getBlockPath(blk pager.BlockID) string {
	// Organize by layer/head for better locality
	dir := filepath.Join(s.basePath, fmt.Sprintf("L%d", blk.Layer))
	filename := fmt.Sprintf("H%d_B%d.bin", blk.Head, blk.SeqBlock)
	return filepath.Join(dir, filename)
}

// ensureBlockFile ensures the block file exists and is opened
func (s *NVMeStore) ensureBlockFile(blk pager.BlockID) (*os.File, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if already open
	if f, ok := s.blockFiles[blk]; ok {
		return f, nil
	}

	// Create directory if needed
	path := s.getBlockPath(blk)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	// Open or create file with O_DIRECT for NVMe optimization
	// Note: O_DIRECT requires aligned buffers in production
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	// Pre-allocate space
	if err := f.Truncate(s.blockSize); err != nil {
		f.Close()
		return nil, err
	}

	s.blockFiles[blk] = f
	return f, nil
}

// WriteBlock writes a block to NVMe storage
func (s *NVMeStore) WriteBlock(blk pager.BlockID, data unsafe.Pointer) error {
	f, err := s.ensureBlockFile(blk)
	if err != nil {
		return fmt.Errorf("failed to open block file: %w", err)
	}

	// Convert unsafe.Pointer to byte slice
	dataSlice := (*[1 << 30]byte)(data)[:s.blockSize:s.blockSize]

	// Write data (in production, use io_uring for async I/O)
	n, err := f.WriteAt(dataSlice, 0)
	if err != nil {
		return fmt.Errorf("failed to write block: %w", err)
	}
	if int64(n) != s.blockSize {
		return fmt.Errorf("incomplete write: %d/%d bytes", n, s.blockSize)
	}

	// Sync to ensure durability
	if err := f.Sync(); err != nil {
		return fmt.Errorf("failed to sync block: %w", err)
	}

	return nil
}

// ReadBlock reads a block from NVMe storage
func (s *NVMeStore) ReadBlock(blk pager.BlockID, data unsafe.Pointer) error {
	f, err := s.ensureBlockFile(blk)
	if err != nil {
		return fmt.Errorf("failed to open block file: %w", err)
	}

	// Convert unsafe.Pointer to byte slice
	dataSlice := (*[1 << 30]byte)(data)[:s.blockSize:s.blockSize]

	// Read data (in production, use io_uring for async I/O)
	n, err := f.ReadAt(dataSlice, 0)
	if err != nil {
		return fmt.Errorf("failed to read block: %w", err)
	}
	if int64(n) != s.blockSize {
		return fmt.Errorf("incomplete read: %d/%d bytes", n, s.blockSize)
	}

	return nil
}

// DeleteBlock removes a block from storage
func (s *NVMeStore) DeleteBlock(blk pager.BlockID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Close file if open
	if f, ok := s.blockFiles[blk]; ok {
		f.Close()
		delete(s.blockFiles, blk)
	}

	// Remove file
	path := s.getBlockPath(blk)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}

// Close closes all open files
func (s *NVMeStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, f := range s.blockFiles {
		f.Close()
	}
	s.blockFiles = make(map[pager.BlockID]*os.File)

	return nil
}

// Stats returns storage statistics
func (s *NVMeStore) Stats() (totalBlocks int, totalBytes int64) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Walk the storage directory
	filepath.Walk(s.basePath, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && filepath.Ext(path) == ".bin" {
			totalBlocks++
			totalBytes += info.Size()
		}
		return nil
	})

	return totalBlocks, totalBytes
}
