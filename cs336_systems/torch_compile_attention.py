"""Benchmark compiled vs uncompiled scaled dot-product attention.

Reuses pytorch_attention.run_sweep; adds a second sweep with torch.compile
and prints a speedup comparison table.
Used for Assignment 2, Section 4: Problem `torch_compile` part (a).

Usage
-----
  python -m cs336_systems.torch_compile_attention
  python -m cs336_systems.torch_compile_attention --dtype bfloat16
"""

import argparse

import torch
from cs336_basics.model import scaled_dot_product_attention

from cs336_systems.pytorch_attention import BATCH, D_HEADS, SEQ_LENS, DTYPE_MAP, run_sweep


def main():
    parser = argparse.ArgumentParser(description="torch.compile attention benchmarking script")
    parser.add_argument("--dtype", default="float32", choices=list(DTYPE_MAP))
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    dtype = DTYPE_MAP[args.dtype]
    compiled_attn = torch.compile(scaled_dot_product_attention)

    print(f"\ntorch.compile attention benchmark — dtype={args.dtype}, batch={BATCH}, "
          f"warmup={args.num_warmup}, steps={args.num_steps}")

    print("\n--- Vanilla (uncompiled) ---")
    vanilla = run_sweep(dtype, args.num_warmup, args.num_steps, attn_fn=scaled_dot_product_attention)

    print("\n--- torch.compile ---")
    compiled = run_sweep(dtype, args.num_warmup, args.num_steps, attn_fn=compiled_attn)

    # ---------- speedup table ----------
    print("\n--- Speedup (vanilla / compiled) ---")
    print(f"{'d_head':>8} {'seq_len':>8} {'fwd speedup':>12} {'fwd+bwd speedup':>16}")
    print("-" * 50)
    for d_head in D_HEADS:
        for seq_len in SEQ_LENS:
            v = vanilla.get((d_head, seq_len))
            c = compiled.get((d_head, seq_len))
            if v and c:
                fwd_sp = v["fwd_ms"] / c["fwd_ms"]
                tot_sp = v["fwd_bwd_ms"] / c["fwd_bwd_ms"]
                print(f"{d_head:>8} {seq_len:>8} {fwd_sp:>12.2f}x {tot_sp:>16.2f}x")
            else:
                print(f"{d_head:>8} {seq_len:>8} {'OOM':>12} {'OOM':>16}")
    print()


if __name__ == "__main__":
    main()
