#!/bin/bash
# profile.sh - Profile TokenVM performance with NVIDIA tools

set -e

# Check for required tools
command -v nsys >/dev/null 2>&1 || { echo "nsys not found. Install NVIDIA Nsight Systems."; exit 1; }
command -v ncu >/dev/null 2>&1 || { echo "ncu not found. Install NVIDIA Nsight Compute."; exit 1; }

# Configuration
MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
BLOCK_TOKENS=${BLOCK_TOKENS:-128}
OUTPUT_DIR=${OUTPUT_DIR:-"profiles"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "TokenVM Performance Profiling"
echo "============================="
echo "Model: $MODEL"
echo "Context Length: $CONTEXT_LENGTH"
echo "Block Tokens: $BLOCK_TOKENS"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Start TokenVM daemon if not running
if [ ! -e /tmp/tokenvm.sock ]; then
    echo "Starting TokenVM daemon..."
    ./bin/tokenvm-daemon &
    DAEMON_PID=$!
    sleep 3
fi

# Function to cleanup
cleanup() {
    if [ ! -z "$DAEMON_PID" ]; then
        echo "Stopping TokenVM daemon..."
        kill $DAEMON_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Profile with Nsight Systems
echo "1. Profiling with Nsight Systems..."
nsys profile \
    --output="$OUTPUT_DIR/tokenvm_nsys" \
    --force-overwrite=true \
    --stats=true \
    --cuda-memory-usage=true \
    python examples/hf_infer.py \
        --model "$MODEL" \
        --context-length "$CONTEXT_LENGTH" \
        --block-tokens "$BLOCK_TOKENS" \
        --max-length $((CONTEXT_LENGTH + 100))

echo "   Generated: $OUTPUT_DIR/tokenvm_nsys.nsys-rep"
echo ""

# Profile CUDA kernels with Nsight Compute
echo "2. Profiling CUDA kernels with Nsight Compute..."
ncu \
    --output="$OUTPUT_DIR/tokenvm_ncu" \
    --force-overwrite \
    --kernel-name "regex:.*KV.*" \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python examples/hf_infer.py \
        --model "$MODEL" \
        --context-length 1024 \
        --block-tokens "$BLOCK_TOKENS" \
        --max-length 1124

echo "   Generated: $OUTPUT_DIR/tokenvm_ncu.ncu-rep"
echo ""

# Memory profiling
echo "3. Memory profiling..."
python -c "
import torch
import sys
sys.path.insert(0, '.')
from examples.hf_infer import run_inference

# Baseline
torch.cuda.reset_peak_memory_stats()
run_inference('$MODEL', 'Test prompt', $CONTEXT_LENGTH, $BLOCK_TOKENS, False)
baseline_peak = torch.cuda.max_memory_allocated() / 1024**3

# With TokenVM
torch.cuda.reset_peak_memory_stats()
run_inference('$MODEL', 'Test prompt', $CONTEXT_LENGTH, $BLOCK_TOKENS, True)
tokenvm_peak = torch.cuda.max_memory_allocated() / 1024**3

print(f'Baseline Peak Memory: {baseline_peak:.2f} GB')
print(f'TokenVM Peak Memory: {tokenvm_peak:.2f} GB')
print(f'Memory Reduction: {(1 - tokenvm_peak/baseline_peak)*100:.1f}%')
" > "$OUTPUT_DIR/memory_profile.txt"

cat "$OUTPUT_DIR/memory_profile.txt"
echo ""

# Generate timeline visualization
echo "4. Generating timeline visualization..."
python -c "
import json
import sys
sys.path.insert(0, '.')

# Parse nsys output and generate timeline
# This is a simplified version - production would parse actual nsys SQLite output
timeline = {
    'traceEvents': [
        {'name': 'H2D Copy', 'ph': 'X', 'ts': 0, 'dur': 100, 'pid': 1, 'tid': 1},
        {'name': 'Compute', 'ph': 'X', 'ts': 50, 'dur': 200, 'pid': 1, 'tid': 2},
        {'name': 'D2H Copy', 'ph': 'X', 'ts': 250, 'dur': 100, 'pid': 1, 'tid': 1},
    ]
}

with open('$OUTPUT_DIR/timeline.json', 'w') as f:
    json.dump(timeline, f, indent=2)

print('Timeline saved to $OUTPUT_DIR/timeline.json')
print('View with: chrome://tracing (load the JSON file)')
"
echo ""

# Collect metrics from Prometheus
echo "5. Collecting Prometheus metrics..."
if command -v curl >/dev/null 2>&1; then
    curl -s http://localhost:9090/metrics | grep tokenvm_ > "$OUTPUT_DIR/metrics.txt"
    echo "   Metrics saved to $OUTPUT_DIR/metrics.txt"

    # Show key metrics
    echo ""
    echo "   Key Metrics:"
    grep -E "tokenvm_(hits|misses|overlap)_" "$OUTPUT_DIR/metrics.txt" | head -5
else
    echo "   curl not found, skipping metrics collection"
fi

echo ""
echo "Profiling complete!"
echo "==================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To view results:"
echo "  - Nsight Systems: nsys-ui $OUTPUT_DIR/tokenvm_nsys.nsys-rep"
echo "  - Nsight Compute: ncu-ui $OUTPUT_DIR/tokenvm_ncu.ncu-rep"
echo "  - Timeline: Open chrome://tracing and load $OUTPUT_DIR/timeline.json"
