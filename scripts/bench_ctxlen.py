#!/usr/bin/env python3
"""
bench_ctxlen.py - Benchmark TokenVM across different context lengths
"""

import os
import sys
import csv
import time
import argparse
import json
import subprocess
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def start_tokenvm_daemon():
    """Start the TokenVM daemon if not running"""
    # Check if daemon is already running
    sock_path = "/tmp/tokenvm.sock"
    if os.path.exists(sock_path):
        print("TokenVM daemon already running")
        return None

    # Start daemon
    print("Starting TokenVM daemon...")
    daemon_process = subprocess.Popen(
        ["./bin/tokenvm-daemon"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for daemon to start
    time.sleep(3)

    if not os.path.exists(sock_path):
        print("Failed to start TokenVM daemon")
        daemon_process.terminate()
        return None

    print("TokenVM daemon started successfully")
    return daemon_process


def run_benchmark(
    model_name: str,
    context_lengths: list,
    block_tokens: int,
    num_runs: int,
    output_file: str,
):
    """
    Run comprehensive benchmark

    Args:
        model_name: HuggingFace model name
        context_lengths: List of context lengths to test
        block_tokens: Tokens per block
        num_runs: Number of runs per configuration
        output_file: Output CSV file
    """

    results = []
    timestamp = datetime.now().isoformat()

    # System info
    gpu_name = "Unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_memory = 0

    print(f"="*80)
    print(f"TokenVM Benchmark")
    print(f"="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"Block tokens: {block_tokens}")
    print(f"Context lengths: {context_lengths}")
    print(f"Runs per config: {num_runs}")
    print(f"Output file: {output_file}")
    print(f"="*80)

    for ctx_len in context_lengths:
        print(f"\nTesting context length: {ctx_len}")

        # Test configurations
        configs = [
            {"name": "baseline", "use_tokenvm": False, "policy": None},
            {"name": "tokenvm_lru", "use_tokenvm": True, "policy": "lru"},
            {"name": "tokenvm_2q", "use_tokenvm": True, "policy": "2q"},
        ]

        for config in configs:
            print(f"  Config: {config['name']}")

            # Set environment
            if config["use_tokenvm"]:
                os.environ["TOKENVM_DISABLE"] = "0"
                if config["policy"]:
                    os.environ["TOKENVM_POLICY"] = config["policy"]
            else:
                os.environ["TOKENVM_DISABLE"] = "1"

            # Run multiple times
            run_results = []
            for run in range(num_runs):
                print(f"    Run {run + 1}/{num_runs}...", end="", flush=True)

                # Run inference
                cmd = [
                    "python", "examples/hf_infer.py",
                    "--model", model_name,
                    "--context-length", str(ctx_len),
                    "--block-tokens", str(block_tokens),
                    "--max-length", str(ctx_len + 100),
                ]

                if not config["use_tokenvm"]:
                    cmd.append("--no-tokenvm")

                try:
                    start_time = time.time()
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    elapsed = time.time() - start_time

                    # Parse output for metrics
                    output = result.stdout
                    tokens_per_sec = 0
                    memory_gb = 0

                    for line in output.split("\n"):
                        if "Tokens/second:" in line:
                            tokens_per_sec = float(line.split(":")[1].strip())
                        elif "Memory used:" in line:
                            memory_gb = float(line.split(":")[1].strip().replace("GB", ""))

                    run_results.append({
                        "tokens_per_sec": tokens_per_sec,
                        "memory_gb": memory_gb,
                        "time": elapsed,
                    })

                    print(f" {tokens_per_sec:.1f} tok/s, {memory_gb:.2f} GB")

                except subprocess.TimeoutExpired:
                    print(" TIMEOUT")
                    run_results.append({
                        "tokens_per_sec": 0,
                        "memory_gb": 0,
                        "time": 300,
                    })
                except Exception as e:
                    print(f" ERROR: {e}")
                    run_results.append({
                        "tokens_per_sec": 0,
                        "memory_gb": 0,
                        "time": 0,
                    })

                # Clear GPU cache
                torch.cuda.empty_cache()
                time.sleep(2)

            # Calculate statistics
            if run_results:
                avg_tokens = np.mean([r["tokens_per_sec"] for r in run_results])
                std_tokens = np.std([r["tokens_per_sec"] for r in run_results])
                avg_memory = np.mean([r["memory_gb"] for r in run_results])
                std_memory = np.std([r["memory_gb"] for r in run_results])
                avg_time = np.mean([r["time"] for r in run_results])

                results.append({
                    "timestamp": timestamp,
                    "model": model_name,
                    "gpu": gpu_name,
                    "context_length": ctx_len,
                    "block_tokens": block_tokens,
                    "config": config["name"],
                    "policy": config["policy"],
                    "num_runs": num_runs,
                    "avg_tokens_per_sec": avg_tokens,
                    "std_tokens_per_sec": std_tokens,
                    "avg_memory_gb": avg_memory,
                    "std_memory_gb": std_memory,
                    "avg_time_sec": avg_time,
                })

    # Write results to CSV
    if results:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"{'Context':<10} {'Config':<20} {'Tokens/s':<15} {'Memory (GB)':<15} {'vs Baseline':<15}")
    print(f"{'-'*100}")

    baseline_results = {}
    for r in results:
        if r["config"] == "baseline":
            baseline_results[r["context_length"]] = r["avg_tokens_per_sec"]

    for r in results:
        speedup = ""
        if r["config"] != "baseline" and r["context_length"] in baseline_results:
            speedup = f"{r['avg_tokens_per_sec'] / baseline_results[r['context_length']]:.2f}x"

        print(f"{r['context_length']:<10} "
              f"{r['config']:<20} "
              f"{r['avg_tokens_per_sec']:<15.2f} "
              f"{r['avg_memory_gb']:<15.2f} "
              f"{speedup:<15}")

    return results


def plot_results(csv_file: str, output_dir: str = "plots"):
    """Generate plots from benchmark results"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Matplotlib/pandas not installed, skipping plots")
        return

    # Read results
    df = pd.read_csv(csv_file)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Plot 1: Throughput vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))
    for config in df["config"].unique():
        data = df[df["config"] == config]
        ax.plot(data["context_length"], data["avg_tokens_per_sec"],
                marker="o", label=config)
        ax.fill_between(data["context_length"],
                        data["avg_tokens_per_sec"] - data["std_tokens_per_sec"],
                        data["avg_tokens_per_sec"] + data["std_tokens_per_sec"],
                        alpha=0.2)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("TokenVM Throughput vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/throughput.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Memory Usage vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))
    for config in df["config"].unique():
        data = df[df["config"] == config]
        ax.plot(data["context_length"], data["avg_memory_gb"],
                marker="o", label=config)
        ax.fill_between(data["context_length"],
                        data["avg_memory_gb"] - data["std_memory_gb"],
                        data["avg_memory_gb"] + data["std_memory_gb"],
                        alpha=0.2)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Memory Usage (GB)")
    ax.set_title("TokenVM Memory Usage vs Context Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/memory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Speedup vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline = df[df["config"] == "baseline"]

    for config in df["config"].unique():
        if config == "baseline":
            continue

        data = df[df["config"] == config]
        speedups = []
        ctx_lens = []

        for _, row in data.iterrows():
            ctx_len = row["context_length"]
            baseline_row = baseline[baseline["context_length"] == ctx_len]
            if not baseline_row.empty:
                speedup = row["avg_tokens_per_sec"] / baseline_row["avg_tokens_per_sec"].values[0]
                speedups.append(speedup)
                ctx_lens.append(ctx_len)

        if speedups:
            ax.plot(ctx_lens, speedups, marker="o", label=config)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Speedup vs Baseline")
    ax.set_title("TokenVM Speedup vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/speedup.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="TokenVM Benchmark Script")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default="1024,2048,4096,8192,16384,32768",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=128,
        help="Tokens per block",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots from results",
    )
    parser.add_argument(
        "--start-daemon",
        action="store_true",
        help="Start TokenVM daemon before benchmarking",
    )

    args = parser.parse_args()

    # Parse context lengths
    context_lengths = [int(x) for x in args.contexts.split(",")]

    # Start daemon if requested
    daemon_process = None
    if args.start_daemon:
        daemon_process = start_tokenvm_daemon()

    try:
        # Run benchmark
        results = run_benchmark(
            model_name=args.model,
            context_lengths=context_lengths,
            block_tokens=args.block_tokens,
            num_runs=args.num_runs,
            output_file=args.output,
        )

        # Generate plots if requested
        if args.plot and results:
            plot_results(args.output)

    finally:
        # Stop daemon if we started it
        if daemon_process:
            print("Stopping TokenVM daemon...")
            daemon_process.terminate()
            daemon_process.wait()


if __name__ == "__main__":
    main()
