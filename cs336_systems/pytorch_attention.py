"""Benchmark script for scaled dot-product attention.

Sweeps d_head × seq_len and reports forward/backward latency and peak memory.
Used for Assignment 2, Section 4: Problem `pytorch_attention`.

Usage
-----
  python -m cs336_systems.pytorch_attention
  python -m cs336_systems.pytorch_attention --dtype bfloat16
  python -m cs336_systems.pytorch_attention --num-warmup 10 --num-steps 100
"""

import argparse
import timeit

import torch
from cs336_basics.model import scaled_dot_product_attention

BATCH = 8
D_HEADS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def time_fn(fn, num_steps: int) -> float:
    """Run fn() num_steps times and return mean wall-clock time in ms."""
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(num_steps):
        fn()
    torch.cuda.synchronize()
    return (timeit.default_timer() - start) / num_steps * 1000


def make_inputs(seq_len: int, d_head: int, dtype: torch.dtype, device: str = "cuda"):
    q = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def benchmark_one(d_head: int, seq_len: int, dtype: torch.dtype,
                  num_warmup: int, num_steps: int,
                  attn_fn=scaled_dot_product_attention) -> dict:
    device = "cuda"

    # ---------- warmup ----------
    for _ in range(num_warmup):
        q, k, v = make_inputs(seq_len, d_head, dtype, device)
        attn_fn(Q=q, K=k, V=v)

    # ---------- time forward ----------
    torch.cuda.reset_peak_memory_stats(device)

    def fwd():
        q, k, v = make_inputs(seq_len, d_head, dtype, device)
        attn_fn(Q=q, K=k, V=v)

    fwd_ms = time_fn(fwd, num_steps)
    fwd_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    # ---------- backward warmup ----------
    for _ in range(num_warmup):
        q, k, v = make_inputs(seq_len, d_head, dtype, device)
        attn_fn(Q=q, K=k, V=v).sum().backward()

    # ---------- time forward + backward ----------
    torch.cuda.reset_peak_memory_stats(device)

    def fwd_bwd():
        q, k, v = make_inputs(seq_len, d_head, dtype, device)
        attn_fn(Q=q, K=k, V=v).sum().backward()

    fwd_bwd_ms = time_fn(fwd_bwd, num_steps)
    full_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    bwd_ms = fwd_bwd_ms - fwd_ms

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "fwd_mem_gb": fwd_mem_gb,
        "full_mem_gb": full_mem_gb,
    }


def run_sweep(dtype: torch.dtype, num_warmup: int, num_steps: int,
              attn_fn=scaled_dot_product_attention) -> dict:
    """Sweep all (d_head, seq_len) combos; return results dict keyed by (d_head, seq_len)."""
    print(f"{'d_head':>8} {'seq_len':>8} {'fwd (ms)':>10} {'bwd (ms)':>10} "
          f"{'fwd+bwd (ms)':>14} {'fwd mem (GB)':>13} {'full mem (GB)':>14}")
    print("-" * 85)

    results = {}
    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            try:
                r = benchmark_one(d_head, seq_len, dtype, num_warmup, num_steps, attn_fn)
                print(f"{d_head:>8} {seq_len:>8} {r['fwd_ms']:>10.2f} {r['bwd_ms']:>10.2f} "
                      f"{r['fwd_bwd_ms']:>14.2f} {r['fwd_mem_gb']:>13.3f} {r['full_mem_gb']:>14.3f}")
                results[(d_head, seq_len)] = r
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{d_head:>8} {seq_len:>8} {'OOM':>10} {'OOM':>10} {'OOM':>14} {'OOM':>13} {'OOM':>14}")
                results[(d_head, seq_len)] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="Attention benchmarking script")
    parser.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP))
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    dtype = DTYPE_MAP[args.dtype]

    print(f"\nAttention benchmark — dtype={args.dtype}, batch={BATCH}, "
          f"warmup={args.num_warmup}, steps={args.num_steps}")
    run_sweep(dtype, args.num_warmup, args.num_steps)
    print()


if __name__ == "__main__":
    main()
