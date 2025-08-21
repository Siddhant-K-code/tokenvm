#!/usr/bin/env python3
"""
hf_bridge.py - HuggingFace Transformers attention hook for TokenVM
"""

import os
import ctypes
import socket
import struct
import json
import time
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from threading import Lock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load TokenVM C library
TOKENVM_LIB_PATH = os.environ.get("TOKENVM_LIB_PATH", "./build/libtokenvm.so")
try:
    tokenvm_lib = ctypes.CDLL(TOKENVM_LIB_PATH)
except OSError as e:
    logger.warning(f"Failed to load TokenVM library: {e}")
    tokenvm_lib = None

# Define C function signatures
if tokenvm_lib:
    # Context management
    tokenvm_lib.tvmCreate.argtypes = [ctypes.c_int]
    tokenvm_lib.tvmCreate.restype = ctypes.c_void_p

    tokenvm_lib.tvmDestroy.argtypes = [ctypes.c_void_p]
    tokenvm_lib.tvmDestroy.restype = None

    # Memory operations
    tokenvm_lib.tvmArenaAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    tokenvm_lib.tvmArenaAlloc.restype = ctypes.c_void_p

    tokenvm_lib.tvmArenaFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    tokenvm_lib.tvmArenaFree.restype = None

    # Pack/Gather operations
    tokenvm_lib.tvmPackKV.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_void_p,  # dst
        ctypes.c_void_p,  # k_src
        ctypes.c_void_p,  # v_src
        ctypes.c_int,     # heads
        ctypes.c_int,     # d_head
        ctypes.c_int,     # block_tokens
        ctypes.c_int,     # dtype
    ]
    tokenvm_lib.tvmPackKV.restype = ctypes.c_int

    tokenvm_lib.tvmGatherKV.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_void_p,  # dst
        ctypes.POINTER(ctypes.c_void_p),  # block_ptrs
        ctypes.c_int,     # num_blocks
        ctypes.c_int,     # heads
        ctypes.c_int,     # d_head
        ctypes.c_int,     # block_tokens
        ctypes.c_int,     # dtype
    ]
    tokenvm_lib.tvmGatherKV.restype = ctypes.c_int


@dataclass
class BlockID:
    """Block identifier matching Go BlockID struct"""
    layer: int
    head: int
    seq_block: int

    def to_dict(self):
        return {"layer": self.layer, "head": self.head, "seq_block": self.seq_block}


class TokenVMClient:
    """Client for communicating with TokenVM daemon"""

    def __init__(self, socket_path="/tmp/tokenvm.sock"):
        self.socket_path = socket_path
        self.sock = None
        self.lock = Lock()
        self.connect()

    def connect(self):
        """Connect to TokenVM daemon"""
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.socket_path)
            logger.info(f"Connected to TokenVM daemon at {self.socket_path}")
        except Exception as e:
            logger.error(f"Failed to connect to TokenVM daemon: {e}")
            self.sock = None

    def ensure_gpu(self, block_id: BlockID) -> Optional[int]:
        """Request block to be loaded to GPU"""
        if not self.sock:
            return None

        with self.lock:
            try:
                request = {
                    "method": "EnsureGPU",
                    "block": block_id.to_dict()
                }

                # Send request
                data = json.dumps(request).encode()
                self.sock.send(struct.pack("!I", len(data)) + data)

                # Receive response
                size_data = self.sock.recv(4)
                if not size_data:
                    return None

                size = struct.unpack("!I", size_data)[0]
                response_data = self.sock.recv(size)
                response = json.loads(response_data.decode())

                if response.get("error"):
                    logger.error(f"EnsureGPU error: {response['error']}")
                    return None

                return response.get("devptr")

            except Exception as e:
                logger.error(f"EnsureGPU failed: {e}")
                return None

    def prefetch(self, block_ids: List[BlockID]):
        """Prefetch multiple blocks"""
        if not self.sock:
            return

        with self.lock:
            try:
                request = {
                    "method": "Prefetch",
                    "blocks": [b.to_dict() for b in block_ids]
                }

                data = json.dumps(request).encode()
                self.sock.send(struct.pack("!I", len(data)) + data)

            except Exception as e:
                logger.error(f"Prefetch failed: {e}")


class TokenVMKVCache:
    """
    TokenVM-backed KV cache for HuggingFace models
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_tokens: int = 128,
        device_id: int = 0,
        disable: bool = False
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_tokens = block_tokens
        self.device_id = device_id
        self.disable = disable or os.environ.get("TOKENVM_DISABLE", "0") == "1"

        if self.disable:
            logger.info("TokenVM disabled, using default KV cache")
            self.ctx = None
            self.client = None
            self.cache = {}
        else:
            # Initialize TokenVM context
            if tokenvm_lib:
                self.ctx = tokenvm_lib.tvmCreate(device_id)
            else:
                self.ctx = None
                self.disable = True
                logger.warning("TokenVM library not loaded, falling back to default cache")

            # Connect to daemon
            self.client = TokenVMClient()

            # Block tracking
            self.block_ptrs = {}  # BlockID -> device pointer
            self.current_seq_len = 0

            logger.info(f"TokenVM KV cache initialized: layers={num_layers}, heads={num_heads}, "
                       f"head_dim={head_dim}, block_tokens={block_tokens}")

    def __del__(self):
        if self.ctx and tokenvm_lib:
            tokenvm_lib.tvmDestroy(self.ctx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states

        Args:
            key_states: [batch, heads, seq_len, head_dim]
            value_states: [batch, heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional cache arguments

        Returns:
            Updated key and value tensors
        """
        if self.disable:
            # Fallback to simple dict cache
            if layer_idx not in self.cache:
                self.cache[layer_idx] = (key_states, value_states)
            else:
                prev_k, prev_v = self.cache[layer_idx]
                key_states = torch.cat([prev_k, key_states], dim=2)
                value_states = torch.cat([prev_v, value_states], dim=2)
                self.cache[layer_idx] = (key_states, value_states)
            return key_states, value_states

        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Calculate block indices
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len
        start_block = start_pos // self.block_tokens
        end_block = (end_pos - 1) // self.block_tokens + 1

        # Pack new blocks
        for block_idx in range(start_block, end_block):
            block_start = block_idx * self.block_tokens
            block_end = min((block_idx + 1) * self.block_tokens, end_pos)
            block_len = block_end - block_start

            if block_len > 0:
                # Extract block data
                k_block = key_states[:, :, block_start - start_pos:block_end - start_pos, :]
                v_block = value_states[:, :, block_start - start_pos:block_end - start_pos, :]

                # Pad if necessary
                if block_len < self.block_tokens:
                    pad_len = self.block_tokens - block_len
                    k_block = torch.nn.functional.pad(k_block, (0, 0, 0, pad_len))
                    v_block = torch.nn.functional.pad(v_block, (0, 0, 0, pad_len))

                # Pack to GPU via TokenVM
                for head_idx in range(num_heads):
                    block_id = BlockID(layer_idx, head_idx, block_idx)

                    # Allocate GPU memory
                    block_size = self.block_tokens * head_dim * 4 * 2  # FP32 K+V
                    dev_ptr = tokenvm_lib.tvmArenaAlloc(self.ctx, block_size)

                    if dev_ptr:
                        # Pack K and V
                        k_data = k_block[0, head_idx].contiguous().cpu().numpy()
                        v_data = v_block[0, head_idx].contiguous().cpu().numpy()

                        result = tokenvm_lib.tvmPackKV(
                            self.ctx,
                            dev_ptr,
                            k_data.ctypes.data,
                            v_data.ctypes.data,
                            1,  # Single head
                            head_dim,
                            self.block_tokens,
                            0  # FP32
                        )

                        if result == 0:
                            self.block_ptrs[block_id] = dev_ptr
                        else:
                            logger.error(f"Failed to pack block {block_id}")

        # Gather all blocks for this layer
        all_blocks = []
        for block_idx in range(end_block):
            for head_idx in range(num_heads):
                block_id = BlockID(layer_idx, head_idx, block_idx)

                # Ensure block is in GPU
                if self.client:
                    dev_ptr = self.client.ensure_gpu(block_id)
                else:
                    dev_ptr = self.block_ptrs.get(block_id)

                if dev_ptr:
                    all_blocks.append((block_id, dev_ptr))

        # Prefetch next layer's blocks
        if self.client and layer_idx < self.num_layers - 1:
            next_blocks = [
                BlockID(layer_idx + 1, h, b)
                for b in range(end_block)
                for h in range(num_heads)
            ]
            self.client.prefetch(next_blocks)

        self.current_seq_len = end_pos

        # For now, return concatenated tensors (simplified)
        # Production would use gathered blocks directly
        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get current sequence length"""
        return self.current_seq_len

    def get_max_length(self) -> Optional[int]:
        """Get maximum sequence length"""
        return None  # No fixed maximum with paging

    def reset(self):
        """Reset cache state"""
        if self.disable:
            self.cache.clear()
        else:
            self.block_ptrs.clear()
            self.current_seq_len = 0


def patch_model_attention(model, block_tokens=128):
    """
    Patch a HuggingFace model to use TokenVM for KV cache

    Args:
        model: HuggingFace model instance
        block_tokens: Tokens per block
    """
    config = model.config

    # Create TokenVM cache
    kv_cache = TokenVMKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        block_tokens=block_tokens
    )

    # Replace cache in model
    model._tokenvm_cache = kv_cache

    # Patch forward methods
    original_forward = model.forward

    def patched_forward(*args, **kwargs):
        # Inject our cache
        kwargs["past_key_values"] = kv_cache
        return original_forward(*args, **kwargs)

    model.forward = patched_forward

    logger.info(f"Patched model {model.__class__.__name__} to use TokenVM")

    return model
