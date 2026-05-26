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


def time_fn(fn, num_steps: int) -> float:
    """Run fn() num_steps times and return mean wall-clock time in ms."""
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(num_steps):
        fn()
    torch.cuda.synchronize()
    return (timeit.default_timer() - start) / num_steps * 1000


def benchmark_one(d_head: int, seq_len: int, dtype: torch.dtype,
                  num_warmup: int, num_steps: int) -> dict:
    device = "cuda"

    def make_inputs():
        q = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        return q, k, v

    # ---------- forward warmup ----------
    for _ in range(num_warmup):
        q, k, v = make_inputs()
        scaled_dot_product_attention(Q=q, K=k, V=v)

    # ---------- time forward ----------
    torch.cuda.reset_peak_memory_stats(device)

    def fwd():
        q, k, v = make_inputs()
        scaled_dot_product_attention(Q=q, K=k, V=v)

    fwd_ms = time_fn(fwd, num_steps)
    fwd_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    # ---------- backward warmup ----------
    for _ in range(num_warmup):
        q, k, v = make_inputs()
        out = scaled_dot_product_attention(Q=q, K=k, V=v)
        out.sum().backward()

    # ---------- time backward (forward + backward together, subtract forward) ----------
    torch.cuda.reset_peak_memory_stats(device)

    def fwd_bwd():
        q, k, v = make_inputs()
        out = scaled_dot_product_attention(Q=q, K=k, V=v)
        out.sum().backward()

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


def main():
    parser = argparse.ArgumentParser(description="Attention benchmarking script")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    print(f"\nAttention benchmark — dtype={args.dtype}, batch={BATCH}, "
          f"warmup={args.num_warmup}, steps={args.num_steps}")
    print(f"{'d_head':>8} {'seq_len':>8} {'fwd (ms)':>10} {'bwd (ms)':>10} "
          f"{'fwd+bwd (ms)':>14} {'fwd mem (GB)':>13} {'full mem (GB)':>14}")
    print("-" * 85)

    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            try:
                r = benchmark_one(d_head, seq_len, dtype, args.num_warmup, args.num_steps)
                print(f"{d_head:>8} {seq_len:>8} {r['fwd_ms']:>10.2f} {r['bwd_ms']:>10.2f} "
                      f"{r['fwd_bwd_ms']:>14.2f} {r['fwd_mem_gb']:>13.3f} {r['full_mem_gb']:>14.3f}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{d_head:>8} {seq_len:>8} {'OOM':>10} {'OOM':>10} {'OOM':>14} {'OOM':>13} {'OOM':>14}")

    print()


if __name__ == "__main__":
    main()
