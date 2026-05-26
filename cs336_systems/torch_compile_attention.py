"""Benchmark compiled vs uncompiled scaled dot-product attention.

Compares torch.compile(attention) against the vanilla implementation using
the same sweep as pytorch_attention.py (batch=8, d_head × seq_len grid).
Used for Assignment 2, Section 4: Problem `torch_compile` part (a).

Usage
-----
  python -m cs336_systems.torch_compile_attention
  python -m cs336_systems.torch_compile_attention --dtype bfloat16
"""

import argparse
import timeit

import torch
from cs336_basics.model import scaled_dot_product_attention

BATCH = 8
D_HEADS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def time_fn(fn, num_steps: int) -> float:
    torch.cuda.synchronize()
    start = timeit.default_timer()
    for _ in range(num_steps):
        fn()
    torch.cuda.synchronize()
    return (timeit.default_timer() - start) / num_steps * 1000


def benchmark_one(attn_fn, d_head: int, seq_len: int, dtype: torch.dtype,
                  num_warmup: int, num_steps: int) -> dict:
    device = "cuda"

    def make_inputs():
        q = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(BATCH, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
        return q, k, v

    # warmup (also triggers torch.compile compilation on first call)
    for _ in range(num_warmup):
        q, k, v = make_inputs()
        out = attn_fn(Q=q, K=k, V=v)
        out.sum().backward()

    # time forward
    torch.cuda.reset_peak_memory_stats(device)

    def fwd():
        q, k, v = make_inputs()
        attn_fn(Q=q, K=k, V=v)

    fwd_ms = time_fn(fwd, num_steps)

    # time forward+backward
    torch.cuda.reset_peak_memory_stats(device)

    def fwd_bwd():
        q, k, v = make_inputs()
        attn_fn(Q=q, K=k, V=v).sum().backward()

    fwd_bwd_ms = time_fn(fwd_bwd, num_steps)
    bwd_ms = fwd_bwd_ms - fwd_ms

    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "fwd_bwd_ms": fwd_bwd_ms}


def run_sweep(attn_fn, label: str, dtype: torch.dtype, num_warmup: int, num_steps: int):
    print(f"\n--- {label} ---")
    print(f"{'d_head':>8} {'seq_len':>8} {'fwd (ms)':>10} {'bwd (ms)':>10} {'fwd+bwd (ms)':>14}")
    print("-" * 55)
    results = {}
    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            try:
                r = benchmark_one(attn_fn, d_head, seq_len, dtype, num_warmup, num_steps)
                print(f"{d_head:>8} {seq_len:>8} {r['fwd_ms']:>10.2f} {r['bwd_ms']:>10.2f} {r['fwd_bwd_ms']:>14.2f}")
                results[(d_head, seq_len)] = r
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{d_head:>8} {seq_len:>8} {'OOM':>10} {'OOM':>10} {'OOM':>14}")
                results[(d_head, seq_len)] = None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    compiled_attn = torch.compile(scaled_dot_product_attention)

    print(f"\ntorch.compile attention benchmark — dtype={args.dtype}, batch={BATCH}, "
          f"warmup={args.num_warmup}, steps={args.num_steps}")

    vanilla = run_sweep(scaled_dot_product_attention, "Vanilla (uncompiled)", dtype, args.num_warmup, args.num_steps)
    compiled = run_sweep(compiled_attn, "torch.compile", dtype, args.num_warmup, args.num_steps)

    # Speedup summary
    print("\n--- Speedup (vanilla / compiled) ---")
    print(f"{'d_head':>8} {'seq_len':>8} {'fwd speedup':>12} {'fwd+bwd speedup':>16}")
    print("-" * 50)
    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            v = vanilla.get((d_head, seq_len))
            c = compiled.get((d_head, seq_len))
            if v and c:
                fwd_sp = v['fwd_ms'] / c['fwd_ms']
                tot_sp = v['fwd_bwd_ms'] / c['fwd_bwd_ms']
                print(f"{d_head:>8} {seq_len:>8} {fwd_sp:>12.2f}x {tot_sp:>16.2f}x")
            else:
                print(f"{d_head:>8} {seq_len:>8} {'OOM':>12} {'OOM':>16}")

    print()


if __name__ == "__main__":
    main()
