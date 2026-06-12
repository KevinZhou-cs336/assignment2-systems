"""Benchmark script for all-reduce communication overhead in single-node multi-process setup.

Problem (distributed_communication_single_node): Distributed Communication (Single Node)

Measures the wall-clock time of dist.all_reduce across:
  - Data sizes: 1MB, 10MB, 100MB, 1GB (float32 tensors)
  - Number of processes/GPUs: 2, 4, or 6

Usage
-----
  # CPU benchmark (Gloo backend, 2 processes):
  python -m cs336_systems.distributed_benchmark --world-size 2 --backend gloo

  # GPU benchmark (NCCL backend, 4 GPUs):
  python -m cs336_systems.distributed_benchmark --world-size 4 --backend nccl

  # Run all world sizes (2, 4, 6) on CPU and print a summary table:
  python -m cs336_systems.distributed_benchmark --backend gloo --all-world-sizes

  # Save a plot:
  python -m cs336_systems.distributed_benchmark --backend gloo --all-world-sizes --plot
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# Float32 element counts corresponding to 1MB / 10MB / 100MB / 1GB
_MB = 1024 * 1024
_DATA_SIZES = {
    "1MB":   1 * _MB // 4,
    "10MB":  10 * _MB // 4,
    "100MB": 100 * _MB // 4,
    "1GB":   1024 * _MB // 4,
}

_WARMUP_ITERS = 5
_BENCH_ITERS  = 20


def setup(rank: int, world_size: int, backend: str, port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        # Each rank pins a distinct GPU so transfers happen between real devices.
        torch.cuda.set_device(rank)


def teardown():
    dist.destroy_process_group()


def _make_tensor(n_elements: int, backend: str, rank: int) -> torch.Tensor:
    if backend == "nccl":
        return torch.randn(n_elements, device=f"cuda:{rank}")
    return torch.randn(n_elements)


def benchmark_allreduce(rank: int, world_size: int, backend: str, port: str,
                        results_queue):
    """Worker function: benchmarks all-reduce for every data size and sends
    results back through a multiprocessing queue."""
    setup(rank, world_size, backend, port)

    timings = {}  # label -> avg seconds (only rank-0 reports)

    for label, n_elements in _DATA_SIZES.items():
        data = _make_tensor(n_elements, backend, rank)

        # Warmup — ensures NCCL buffers are allocated and JIT effects are gone.
        for _ in range(_WARMUP_ITERS):
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()

        # Timed iterations.
        t_start = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                # async_op=False only waits for the op to be *queued* on the GPU;
                # synchronize() blocks until the GPU has actually finished.
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        avg_ms = elapsed / _BENCH_ITERS * 1_000
        timings[label] = avg_ms

    # Gather timing from every rank so we can report the cross-rank average
    # (timings may differ slightly due to measurement noise).
    all_timings = [None] * world_size
    dist.all_gather_object(all_timings, timings)

    if rank == 0:
        # Average over ranks.
        avg_timings = {}
        for label in _DATA_SIZES:
            avg_timings[label] = sum(t[label] for t in all_timings) / world_size
        results_queue.put((world_size, avg_timings))

    teardown()


def run_benchmark(world_size: int, backend: str, port: str = "29500"):
    """Spawns `world_size` processes and returns {data_size_label: avg_ms}."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(
        benchmark_allreduce,
        args=(world_size, backend, port, q),
        nprocs=world_size,
        join=True,
    )
    _, timings = q.get()
    return timings


def print_table(all_results: dict):
    """Pretty-print results as a Markdown-style table.

    all_results: {world_size: {label: avg_ms}}
    """
    sizes  = list(_DATA_SIZES.keys())
    worlds = sorted(all_results.keys())

    col_w = 12
    header = f"{'Data size':<12}" + "".join(f"{'w='+str(w):>{col_w}}" for w in worlds)
    sep    = "-" * len(header)
    print()
    print("all-reduce latency (ms) — averaged over ranks and iterations")
    print(sep)
    print(header)
    print(sep)
    for sz in sizes:
        row = f"{sz:<12}"
        for w in worlds:
            row += f"{all_results[w][sz]:>{col_w}.2f}"
        print(row)
    print(sep)
    print()


def save_plot(all_results: dict, path: str = "allreduce_benchmark.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    sizes  = list(_DATA_SIZES.keys())
    worlds = sorted(all_results.keys())

    x = range(len(sizes))
    fig, ax = plt.subplots(figsize=(8, 5))
    for w in worlds:
        ys = [all_results[w][sz] for sz in sizes]
        ax.plot(x, ys, marker="o", label=f"{w} processes")

    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Data size (float32)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("all-reduce latency vs. data size and process count")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all-reduce on single node")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo",
                        help="Communication backend. Use gloo for CPU, nccl for GPU.")
    parser.add_argument("--world-size", type=int, default=2,
                        help="Number of processes/GPUs (ignored when --all-world-sizes).")
    parser.add_argument("--all-world-sizes", action="store_true",
                        help="Run benchmarks for world sizes 2, 4, and 6.")
    parser.add_argument("--plot", action="store_true",
                        help="Save a line plot comparing results.")
    parser.add_argument("--port", type=str, default="29500",
                        help="Master port for the process group rendezvous.")
    args = parser.parse_args()

    world_sizes = [2, 4, 6] if args.all_world_sizes else [args.world_size]

    if args.backend == "nccl":
        n_gpus = torch.cuda.device_count()
        world_sizes = [w for w in world_sizes if w <= n_gpus]
        if not world_sizes:
            raise RuntimeError(
                f"No eligible world sizes: need at least {min([2,4,6])} GPUs, "
                f"but only {n_gpus} available."
            )

    all_results = {}
    for ws in world_sizes:
        print(f"Running world_size={ws}, backend={args.backend} ...")
        all_results[ws] = run_benchmark(ws, args.backend, args.port)

    print_table(all_results)

    if args.plot:
        save_plot(all_results)


if __name__ == "__main__":
    main()
