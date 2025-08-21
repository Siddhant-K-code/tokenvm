#!/usr/bin/env python3
"""
hf_infer.py - Example HuggingFace inference runner with TokenVM
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.hooks.hf_bridge import patch_model_attention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_memory():
    """Measure current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def run_inference(
    model_name: str,
    prompt: str,
    max_length: int,
    block_tokens: int,
    use_tokenvm: bool,
    temperature: float = 0.7,
    top_p: float = 0.9,
    token: str = None,
):
    """
    Run inference with optional TokenVM

    Args:
        model_name: HuggingFace model name
        prompt: Input prompt
        max_length: Maximum generation length
        block_tokens: Tokens per block for TokenVM
        use_tokenvm: Whether to use TokenVM
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        token: HuggingFace API token for gated models
    """

    logger.info(f"Loading model: {model_name}")
    logger.info(f"TokenVM: {'ENABLED' if use_tokenvm else 'DISABLED'}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )

    # Patch model if using TokenVM
    if use_tokenvm:
        model = patch_model_attention(model, block_tokens=block_tokens)
        logger.info(f"Model patched with TokenVM (block_tokens={block_tokens})")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Measure initial memory
    initial_memory = measure_memory()
    logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")

    # Generate
    logger.info(f"Generating up to {max_length} tokens...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    end_time = time.time()

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Calculate metrics
    num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    elapsed_time = end_time - start_time
    tokens_per_second = num_tokens / elapsed_time
    final_memory = measure_memory()
    memory_used = final_memory - initial_memory

    # Print results
    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("\n" + "="*80)
    print("METRICS:")
    print("="*80)
    print(f"Tokens generated: {num_tokens}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Tokens/second: {tokens_per_second:.2f}")
    print(f"Initial GPU memory: {initial_memory:.2f} GB")
    print(f"Final GPU memory: {final_memory:.2f} GB")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"TokenVM enabled: {use_tokenvm}")

    return {
        "text": generated_text,
        "num_tokens": num_tokens,
        "time": elapsed_time,
        "tokens_per_second": tokens_per_second,
        "memory_used": memory_used,
    }


def benchmark_context_lengths(
    model_name: str,
    context_lengths: list,
    block_tokens: int = 128,
    token: str = None,
):
    """
    Benchmark different context lengths with and without TokenVM

    Args:
        model_name: HuggingFace model name
        context_lengths: List of context lengths to test
        block_tokens: Tokens per block for TokenVM
        token: HuggingFace API token for gated models
    """

    results = []

    # Create a long prompt
    base_prompt = "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. "

    for ctx_len in context_lengths:
        # Create prompt of desired length
        prompt = base_prompt * (ctx_len // len(base_prompt.split()) + 1)
        prompt_words = prompt.split()[:ctx_len]
        prompt = " ".join(prompt_words)

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing context length: {ctx_len} tokens")
        logger.info(f"{'='*80}")

        # Test without TokenVM
        logger.info("Running baseline (no TokenVM)...")
        baseline_result = run_inference(
            model_name=model_name,
            prompt=prompt,
            max_length=ctx_len + 100,
            block_tokens=block_tokens,
            use_tokenvm=False,
            token=token,
        )

        # Clear cache
        torch.cuda.empty_cache()
        time.sleep(2)

        # Test with TokenVM
        logger.info("Running with TokenVM...")
        tokenvm_result = run_inference(
            model_name=model_name,
            prompt=prompt,
            max_length=ctx_len + 100,
            block_tokens=block_tokens,
            use_tokenvm=True,
            token=token,
        )

        # Record results
        results.append({
            "context_length": ctx_len,
            "baseline_tokens_per_sec": baseline_result["tokens_per_second"],
            "baseline_memory_gb": baseline_result["memory_used"],
            "tokenvm_tokens_per_sec": tokenvm_result["tokens_per_second"],
            "tokenvm_memory_gb": tokenvm_result["memory_used"],
            "speedup": tokenvm_result["tokens_per_second"] / baseline_result["tokens_per_second"],
            "memory_reduction": 1 - (tokenvm_result["memory_used"] / baseline_result["memory_used"]),
        })

        # Clear cache
        torch.cuda.empty_cache()
        time.sleep(2)

    # Print summary
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Context':<10} {'Baseline':<15} {'TokenVM':<15} {'Speedup':<10} {'Memory':<15} {'Memory':<15}")
    print(f"{'Length':<10} {'Tokens/s':<15} {'Tokens/s':<15} {'':<10} {'Baseline (GB)':<15} {'Saved (%)':<15}")
    print("-"*100)

    for r in results:
        print(f"{r['context_length']:<10} "
              f"{r['baseline_tokens_per_sec']:<15.2f} "
              f"{r['tokenvm_tokens_per_sec']:<15.2f} "
              f"{r['speedup']:<10.2f}x "
              f"{r['baseline_memory_gb']:<15.2f} "
              f"{r['memory_reduction']*100:<15.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TokenVM HuggingFace Inference Example",
        epilog="""
Examples:
  # Use open model (no token required):
  python hf_infer.py --model microsoft/phi-2

  # Use gated model with token:
  export HF_TOKEN=your_token_here
  python hf_infer.py --model meta-llama/Llama-2-7b-hf

  # Or pass token directly:
  python hf_infer.py --model meta-llama/Llama-2-7b-hf --hf-token your_token_here
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",  # Changed to open model
        help="HuggingFace model name (default: microsoft/phi-2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time in a distant galaxy,",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length for benchmarking",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=128,
        help="Tokens per block for TokenVM",
    )
    parser.add_argument(
        "--no-tokenvm",
        action="store_true",
        help="Disable TokenVM",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token for gated models (or set HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Check for token if using gated models
    if "meta-llama" in args.model.lower() or "gated" in args.model.lower():
        if not args.hf_token:
            logger.warning(
                "\n" + "="*80 + "\n"
                "WARNING: You're trying to use a gated model without authentication.\n"
                "To use gated models like Llama-2, you need:\n"
                "1. Accept the license at https://huggingface.co/" + args.model + "\n"
                "2. Get your token from https://huggingface.co/settings/tokens\n"
                "3. Set it via: export HF_TOKEN=your_token_here\n"
                "   Or pass it: --hf-token your_token_here\n"
                "\nUsing default open model instead: microsoft/phi-2\n" +
                "="*80
            )
            args.model = "microsoft/phi-2"

    # Set environment variable if disabling TokenVM
    if args.no_tokenvm:
        os.environ["TOKENVM_DISABLE"] = "1"

    if args.benchmark:
        # Run benchmark
        context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
        if args.context_length:
            context_lengths = [args.context_length]

        benchmark_context_lengths(
            model_name=args.model,
            context_lengths=context_lengths,
            block_tokens=args.block_tokens,
            token=args.hf_token,
        )
    else:
        # Run single inference
        run_inference(
            model_name=args.model,
            prompt=args.prompt,
            max_length=args.max_length,
            block_tokens=args.block_tokens,
            use_tokenvm=not args.no_tokenvm,
            temperature=args.temperature,
            top_p=args.top_p,
            token=args.hf_token,
        )


if __name__ == "__main__":
    main()
