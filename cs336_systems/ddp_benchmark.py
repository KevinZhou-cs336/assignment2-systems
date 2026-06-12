"""Benchmark comparing per-parameter DDP vs. flat-gradient DDP.

Problems covered:
  naive_ddp_benchmarking          -- 1 node x 2 GPUs, xl model size
  minimal_ddp_flat_benchmarking   -- same setup, single batched all-reduce

Measures per training iteration:
  - Total step time  (forward + backward + sync + optimizer)
  - Gradient communication time only

Usage
-----
  # On a 2-GPU machine:
  python -m cs336_systems.ddp_benchmark --world-size 2

  # Larger model (xl) or different batch/seq:
  python -m cs336_systems.ddp_benchmark --world-size 2 --model-size xl

  # Save a summary table to stdout and optionally a plot:
  python -m cs336_systems.ddp_benchmark --world-size 2 --plot
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

from cs336_systems.ddp import DDP, FlatDDP

# ---------------------------------------------------------------------------
# Model size table (Table 1 from the PDF)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE    = 10_000
BATCH_SIZE    = 4
CONTEXT_LEN   = 512
WARMUP_ITERS  = 5
BENCH_ITERS   = 10


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, ddp_cls: type, model_cfg: dict,
            port: str, results_queue):
    # Set up process group and pin to a GPU.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Build model + DDP wrapper.
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LEN,
        **model_cfg,
    ).to(device)
    ddp_model = ddp_cls(model)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    # Fixed random input (same every iteration; we're measuring throughput).
    torch.manual_seed(rank)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN), device=device)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN), device=device)

    def step():
        optimizer.zero_grad()
        logits = ddp_model(x)
        loss = cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()

    # Warmup.
    for _ in range(WARMUP_ITERS):
        step()

    # Timed iterations.
    total_times: list[float] = []
    comm_times: list[float] = []

    for _ in range(BENCH_ITERS):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits = ddp_model(x)
        loss = cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_times.append((t1 - t0) * 1_000)
        comm_times.append(ddp_model.last_comm_time * 1_000)

    # Gather across ranks and report from rank 0.
    all_total = [None] * world_size
    all_comm  = [None] * world_size
    dist.all_gather_object(all_total, total_times)
    dist.all_gather_object(all_comm,  comm_times)

    if rank == 0:
        avg_total = sum(sum(r) for r in all_total) / (world_size * BENCH_ITERS)
        avg_comm  = sum(sum(r) for r in all_comm)  / (world_size * BENCH_ITERS)
        results_queue.put({
            "cls":        ddp_cls.__name__,
            "total_ms":   avg_total,
            "comm_ms":    avg_comm,
            "comm_pct":   100.0 * avg_comm / avg_total,
        })

    dist.destroy_process_group()


def run_benchmark(world_size: int, ddp_cls: type, model_cfg: dict,
                  port: str = "29501") -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(
        _worker,
        args=(world_size, ddp_cls, model_cfg, port, q),
        nprocs=world_size,
        join=True,
    )
    return q.get()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict]):
    print()
    print(f"{'Impl':<12}  {'Total (ms)':>12}  {'Comm (ms)':>12}  {'Comm %':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['cls']:<12}  {r['total_ms']:>12.1f}  {r['comm_ms']:>12.1f}  {r['comm_pct']:>7.1f}%")
    print()


def save_plot(results: list[dict], path: str = "ddp_benchmark.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    names      = [r["cls"]      for r in results]
    total_vals = [r["total_ms"] for r in results]
    comm_vals  = [r["comm_ms"]  for r in results]

    x = range(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, total_vals, color=["steelblue", "darkorange"])
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("ms"); axes[0].set_title("Total step time (ms)")

    axes[1].bar(x, comm_vals, color=["steelblue", "darkorange"])
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("ms"); axes[1].set_title("Gradient communication time (ms)")

    fig.suptitle("DDP: per-parameter vs. flat-gradient all-reduce")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-parameter DDP vs. flat-gradient DDP"
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS), default="xl")
    parser.add_argument("--port",  type=str, default="29501")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA GPUs.")
    n_gpus = torch.cuda.device_count()
    if args.world_size > n_gpus:
        raise SystemExit(f"Requested {args.world_size} GPUs but only {n_gpus} available.")

    model_cfg = MODEL_CONFIGS[args.model_size]
    print(f"Model: {args.model_size}  ({model_cfg})")
    print(f"world_size={args.world_size}, batch={BATCH_SIZE}, ctx={CONTEXT_LEN}")

    results = []
    for cls, port_offset in [(DDP, 0), (FlatDDP, 1)]:
        print(f"\nRunning {cls.__name__} ...")
        port = str(int(args.port) + port_offset)
        res = run_benchmark(args.world_size, cls, model_cfg, port)
        results.append(res)
        print(f"  total={res['total_ms']:.1f} ms  comm={res['comm_ms']:.1f} ms  ({res['comm_pct']:.1f}%)")

    print_table(results)
    if args.plot:
        save_plot(results)


if __name__ == "__main__":
    main()
