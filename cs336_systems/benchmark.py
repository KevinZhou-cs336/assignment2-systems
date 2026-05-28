"""Benchmark script for the Transformer Language Model.

This script measures the runtime and memory cost of a Transformer LM's
forward pass, backward pass, and optimizer step.  It is the single entry
point for four assignment problems:

  §2.1.3  benchmarking_script        — basic end-to-end timing
  §2.1.4  nsys_profile               — NVTX annotations for nsys profiler
  §2.1.5  benchmarking_mixed_precision — BF16 autocast benchmarking
  §2.1.6  memory_profiling           — PyTorch memory snapshot
  §4      torch_compile (b)          — compiled vs vanilla Transformer

Usage examples
--------------
  # Default: forward + backward, small model (Table 1 "small"):
  python -m cs336_systems.benchmark --d-model 768 --num-layers 12 --num-heads 12

  # Time only the forward pass:
  python -m cs336_systems.benchmark --mode forward

  # Time the full training step (forward + backward + optimizer):
  python -m cs336_systems.benchmark --mode full

  # BF16 mixed-precision run (§2.1.5):
  python -m cs336_systems.benchmark --mixed-precision

  # Compiled model (§4 torch_compile b):
  python -m cs336_systems.benchmark --compile

  # Annotate for nsys profiler (§2.1.4):
  uv run nsys profile -- python -m cs336_systems.benchmark --nvtx

  # Dump a memory snapshot (§2.1.6):
  python -m cs336_systems.benchmark --memory-profile
"""

import argparse
import contextlib
import math
import timeit

import torch
import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
from einops import einsum


# =============================================================================
# NVTX-annotated scaled dot-product attention  (§2.1.4)
# =============================================================================
#
# Background — why annotate?
# --------------------------
# CUDA kernels execute *asynchronously*: when Python calls torch.matmul,
# the call returns immediately while the GPU runs the kernel in the
# background.  The NVIDIA Nsight Systems profiler (nsys) can capture a
# timeline of every CUDA kernel, but by default it can't tell *which part
# of your Python code* triggered each kernel.
#
# NVTX (NVIDIA Tools Extension) lets us insert named "ranges" into the
# timeline.  nsys records the start and end of each range alongside the
# kernel timeline, so we can see exactly which kernels belong to
# "computing attention scores", "computing softmax", etc.
#
# How it works here
# -----------------
# We define a drop-in replacement for cs336_basics' scaled_dot_product_attention
# that wraps each of the three sub-operations in an nvtx.range context.
# When --nvtx is passed, main() monkey-patches the module-level function
# so that every call inside the model uses the annotated version.
#
# The top-level @nvtx.range decorator marks the boundary of the entire
# attention computation, and the inner with-blocks subdivide it further.

@torch.cuda.nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,          # Query matrix,  shape (..., num_queries, head_dim)
    K: torch.Tensor,          # Key matrix,    shape (..., num_keys,    head_dim)
    V: torch.Tensor,          # Value matrix,  shape (..., num_keys,    head_dim)
    mask: torch.Tensor | None = None,  # Boolean causal mask, True = keep
) -> torch.Tensor:            # Output,        shape (..., num_queries, head_dim)
    """Scaled dot-product attention with NVTX labels for nsys profiling.

    Implements: Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
    This is a drop-in replacement for cs336_basics.model.scaled_dot_product_attention.
    """
    head_dim = K.shape[-1]  # d_k: dimensionality of each attention head

    with torch.cuda.nvtx.range("computing attention scores"):
        # Compute Q·Kᵀ / √d_k.
        # The 1/√d_k scaling prevents dot products from growing so large
        # that softmax saturates and gradients vanish.
        attention_scores = (
            einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
            / math.sqrt(head_dim)
        )
        # Apply causal mask: positions where mask=False get score -∞,
        # which becomes 0 after softmax, preventing tokens from attending
        # to future positions.
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with torch.cuda.nvtx.range("computing softmax"):
        # Normalize scores into a probability distribution over keys.
        # Each row of attention_weights sums to 1.
        attention_weights = softmax(attention_scores, dim=-1)

    with torch.cuda.nvtx.range("final matmul"):
        # Compute the weighted sum of value vectors.
        # Each query position gets a blend of all value vectors,
        # weighted by how much it attends to each key.
        return einsum(
            attention_weights, V,
            "... query key, ... key d_v -> ... query d_v",
        )


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark a Transformer language model (CS336 Assignment 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model architecture ────────────────────────────────────────────────────
    # These match Table 1 in §2.1.2.  The defaults correspond to the "small"
    # configuration.  Override them from the command line to benchmark other
    # sizes (e.g. --d-model 2560 --num-layers 32 --num-heads 32 for "xl").
    arch = parser.add_argument_group("model architecture")
    arch.add_argument(
        "--vocab-size", type=int, default=10_000,
        help="Number of token types.  Must match the tokenizer.",
    )
    arch.add_argument(
        "--context-length", type=int, default=512,
        help="Maximum sequence length the model can process.  "
             "§2.1.2 says to use 512 unless otherwise specified.",
    )
    arch.add_argument(
        "--d-model", type=int, default=768,
        help="Hidden / embedding dimension (D).  "
             "Controls the width of every layer in the model.",
    )
    arch.add_argument(
        "--num-heads", type=int, default=12,
        help="Number of attention heads.  Must evenly divide d-model.",
    )
    arch.add_argument(
        "--num-layers", type=int, default=12,
        help="Number of stacked Transformer blocks.",
    )
    arch.add_argument(
        "--d-ff", type=int, default=None,
        help="Feed-forward hidden dimension.  "
             "Defaults to the nearest multiple of 64 to (8/3)·d-model.",
    )
    arch.add_argument(
        "--rope-theta", type=float, default=10_000.0,
        help="RoPE base frequency θ for rotary position embeddings.",
    )

    # ── Benchmark settings ────────────────────────────────────────────────────
    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of sequences per step (B).  "
             "§2.1.2 says to use batch size 4.",
    )
    bench.add_argument(
        "--num-warmup-steps", type=int, default=5,
        help="Number of untimed warm-up steps run before measurement begins.  "
             "These let CUDA compile kernels and reach a steady thermal state "
             "so timings are representative.",
    )
    bench.add_argument(
        "--num-steps", type=int, default=10,
        help="Number of timed measurement steps.  "
             "§2.1.3(b) asks for 10 steps to compute mean ± std.",
    )
    bench.add_argument(
        "--device", type=str, default="cuda",
        help='PyTorch device string: "cuda", "cuda:0", "cpu", etc.',
    )
    bench.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "full"],
        default="forward-backward",
        help=(
            "What to time.  "
            "'forward': forward pass only (inference).  "
            "'forward-backward': forward + backward pass (no optimizer).  "
            "'full': forward + backward + AdamW optimizer step (one training iteration)."
        ),
    )

    # ── §2.1.5  Mixed-precision ───────────────────────────────────────────────
    # Modern NVIDIA GPUs have Tensor Cores that execute matrix multiplications
    # significantly faster in lower-precision formats (FP16, BF16) than in
    # FP32.  torch.autocast automatically casts eligible operations (mainly
    # matmuls) to the specified dtype while keeping others (e.g. reductions,
    # loss) in FP32 to preserve numerical stability.
    bench.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Run the forward (and backward) pass under "
             "torch.autocast(dtype=bfloat16).  "
             "BF16 has the same dynamic range as FP32 but lower precision, "
             "making it safer than FP16 for training.",
    )

    # ── §2.1.4  NVTX annotations ──────────────────────────────────────────────
    # NVTX ranges appear in the nsys timeline as colored blocks, letting us
    # map GPU kernels back to specific Python code regions.  Two things happen:
    #   1. The entire timed loop is wrapped in nvtx.range("benchmark") so we
    #      can use nsys' --nvtx-capture flag to exclude the warm-up phase.
    #   2. scaled_dot_product_attention is replaced with the annotated version
    #      above, subdividing attention into scores / softmax / final matmul.
    bench.add_argument(
        "--nvtx",
        action="store_true",
        help="Insert NVTX range labels for Nsight Systems profiling.  "
             "Also monkey-patches scaled_dot_product_attention so nsys can "
             "show time spent on each attention sub-operation.",
    )

    # ── §2.1.6  Memory profiling ──────────────────────────────────────────────
    # PyTorch's memory profiler records every GPU allocation and free event.
    # The resulting pickle file can be loaded into pytorch.org/memory_viz to
    # see the full "Active Memory Timeline" — useful for understanding when
    # activations are allocated and freed during a training step.
    bench.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record a full GPU memory history during the timed steps and "
             "dump it to memory_snapshot.pickle.  "
             "Open the file at pytorch.org/memory_viz to inspect it.",
    )

    # ── §4  torch.compile ─────────────────────────────────────────────────────
    # torch.compile() traces the model and emits optimized GPU kernels via the
    # Inductor backend (Triton under the hood).  Key benefits:
    #   - Fuses consecutive elementwise ops (softmax, residual, RMSNorm) into
    #     single kernels, reducing memory round-trips.
    #   - Can exploit Tensor Cores more aggressively in BF16.
    # The first forward pass triggers compilation and is slower; warmup steps
    # absorb this cost so the timed region only sees steady-state performance.
    bench.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile() before benchmarking (§4 torch_compile b).",
    )

    args = parser.parse_args()

    # Compute d_ff default: nearest multiple of 64 to (8/3)·d_model.
    # This formula comes from the SwiGLU paper and is a common heuristic
    # for setting the FFN hidden dimension relative to the model width.
    if args.d_ff is None:
        args.d_ff = int((8 * args.d_model / 3) / 64 + 0.5) * 64

    return args


# =============================================================================
# Helper functions
# =============================================================================

def make_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random batch of token IDs for benchmarking.

    Since we only care about speed and memory — not model accuracy — we
    generate random integer token IDs instead of loading real data.  This
    avoids the need for a dataset file and keeps the benchmark self-contained.

    Returns:
        input_tokens:  shape (batch_size, context_length), dtype int64.
                       The tokens fed to the model.
        target_tokens: shape (batch_size, context_length), dtype int64.
                       The next-token targets used to compute cross-entropy loss.
    """
    input_tokens  = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    target_tokens = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return input_tokens, target_tokens


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    """Return the population mean and standard deviation of a list of floats."""
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


# =============================================================================
# Main benchmark logic
# =============================================================================

def main() -> None:
    args = parse_args()
    use_cuda = "cuda" in args.device

    # ── §2.1.4  Monkey-patch attention with the NVTX-annotated version ────────
    # We replace the module-level function in cs336_basics.model so that every
    # call to scaled_dot_product_attention (from CausalMultiHeadSelfAttention)
    # uses the annotated version.  Because Python looks up names at call time,
    # replacing the function object in the module namespace is sufficient —
    # no changes to cs336_basics are needed.
    if args.nvtx:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # ── Build the model ───────────────────────────────────────────────────────
    # BasicsTransformerLM is a standard decoder-only Transformer (like GPT-2):
    # token embedding → N × TransformerBlock → linear head → logits.
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    # Move all parameters and buffers to the target device (GPU or CPU).
    # Must be done before creating the optimizer so that optimizer state
    # tensors are also created on the correct device.
    model.to(args.device)
    # train() enables any dropout layers and makes BatchNorm track statistics.
    # We call it so the benchmark reflects realistic training conditions.
    model.train()

    # ── Optimizer (only for --mode full) ─────────────────────────────────────
    # AdamW is the standard optimizer for Transformer training.  It maintains
    # two extra tensors per parameter (the first and second moment estimates),
    # roughly tripling the memory footprint of the parameters.
    # We only instantiate it when the optimizer step is actually being timed.
    optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "full" else None

    # ── §4  torch.compile ────────────────────────────────────────────────────
    if args.compile:
        model = torch.compile(model)

    # ── §2.1.5  Precision context ─────────────────────────────────────────────
    # torch.autocast automatically casts matmuls and convolutions to bfloat16
    # while leaving reductions (softmax, layer norm, loss) in float32.
    # When --mixed-precision is not set we use contextlib.nullcontext(), a
    # no-op context manager, so the rest of the timing loop is identical.
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.mixed_precision and use_cuda
        else contextlib.nullcontext()
    )

    # Total number of tokens processed in one step: used to compute throughput.
    tokens_per_step = args.batch_size * args.context_length

    # =========================================================================
    # Warm-up phase  (not timed)
    # =========================================================================
    # Why warm up?
    # ------------
    # The first few CUDA kernel calls are artificially slow because:
    #   - The CUDA JIT compiler compiles and caches kernels on first use.
    #   - GPU clocks may ramp up from an idle state.
    #   - Memory allocators set up internal pools.
    # Skipping warm-up would make the first timed step look much slower than
    # steady-state, inflating the mean and standard deviation.
    # The assignment specifies w=5 warm-up steps before timing begins.
    for _ in range(args.num_warmup_steps):
        input_tokens, target_tokens = make_random_batch(
            args.batch_size, args.context_length, args.vocab_size, args.device
        )
        with precision_context:
            # Forward pass: input_tokens (B, T) → logits (B, T, vocab_size).
            logits = model(input_tokens)
            # Cross-entropy loss: scalar, averaged over all B×T positions.
            loss = cross_entropy(logits, target_tokens)
        if args.mode != "forward":
            # Backward pass: compute ∂loss/∂θ for all model parameters θ.
            loss.backward()
            if args.mode == "full":
                # Optimizer step: update parameters using AdamW rule.
                optimizer.step()
        # Clear accumulated gradients so the next step starts fresh.
        # set_to_none=True frees the gradient tensors entirely (more memory-
        # efficient than filling them with zeros).
        model.zero_grad(set_to_none=True)

    if use_cuda:
        # Wait for all queued GPU kernels to finish before we reset stats.
        # Without this, some warm-up work could still be in flight.
        torch.cuda.synchronize()
        # Zero out the peak-memory counter so it only reflects the timed
        # region below, not the warm-up phase.
        torch.cuda.reset_peak_memory_stats(args.device)

    # ── §2.1.6  Start recording memory history ────────────────────────────────
    # We begin recording after warm-up so the snapshot only covers the
    # timed region.  max_entries limits how many allocation events are kept
    # in the ring buffer (1 M entries is usually more than enough).
    if args.memory_profile and use_cuda:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # =========================================================================
    # Timed measurement phase
    # =========================================================================
    # Collect per-step latencies (in milliseconds) for each phase.
    forward_times_ms:   list[float] = []
    backward_times_ms:  list[float] = []
    optimizer_times_ms: list[float] = []

    # Wrap the timed loop in an NVTX range named "benchmark".
    # In nsys, you can then pass --nvtx-capture benchmark to restrict the
    # captured trace to this region, cleanly excluding the warm-up steps.
    # When --nvtx is not set we fall back to a no-op nullcontext.
    nvtx_timed_region = (
        torch.cuda.nvtx.range("benchmark")
        if args.nvtx and use_cuda
        else contextlib.nullcontext()
    )

    with nvtx_timed_region:
        for _ in range(args.num_steps):
            input_tokens, target_tokens = make_random_batch(
                args.batch_size, args.context_length, args.vocab_size, args.device
            )

            # ── Forward pass timing ───────────────────────────────────────────
            # Why synchronize before starting the timer?
            # CUDA calls are asynchronous: Python dispatches work to the GPU
            # and continues immediately without waiting.  If we start the
            # timer before the GPU has finished the previous step's work,
            # we would undercount the forward time.  Calling synchronize()
            # forces the CPU to wait until all previously queued GPU kernels
            # have finished, giving us a clean starting point.
            if use_cuda:
                torch.cuda.synchronize()
            fwd_start = timeit.default_timer()

            with precision_context:
                # Run the full Transformer: embedding → N blocks → logits.
                # logits shape: (batch_size, context_length, vocab_size)
                logits = model(input_tokens)
                # Compute cross-entropy loss between logits and targets.
                # loss is a scalar tensor attached to the computation graph.
                loss = cross_entropy(logits, target_tokens)

            # synchronize() again so the timer captures the time until the
            # GPU actually finishes the forward kernels, not just until the
            # CPU finishes dispatching them.
            if use_cuda:
                torch.cuda.synchronize()
            fwd_end = timeit.default_timer()
            # Convert seconds → milliseconds and store.
            forward_times_ms.append((fwd_end - fwd_start) * 1e3)

            # ── Backward pass timing ──────────────────────────────────────────
            # loss.backward() traverses the computation graph in reverse,
            # computing ∂loss/∂θ for every parameter θ via the chain rule.
            # Each parameter's .grad tensor is filled with its gradient.
            if args.mode in ("forward-backward", "full"):
                bwd_start = timeit.default_timer()
                loss.backward()
                if use_cuda:
                    torch.cuda.synchronize()
                bwd_end = timeit.default_timer()
                backward_times_ms.append((bwd_end - bwd_start) * 1e3)

            # ── Optimizer step timing ─────────────────────────────────────────
            # AdamW reads each parameter's .grad and updates the parameter
            # using running estimates of the gradient mean (m) and variance (v).
            # This completes one full training iteration.
            if args.mode == "full":
                opt_start = timeit.default_timer()
                optimizer.step()
                if use_cuda:
                    torch.cuda.synchronize()
                opt_end = timeit.default_timer()
                optimizer_times_ms.append((opt_end - opt_start) * 1e3)

            # Free gradient tensors so they don't accumulate across steps.
            model.zero_grad(set_to_none=True)

    # ── §2.1.6  Save the memory snapshot and stop recording ───────────────────
    # _dump_snapshot writes a pickle file with every allocation event recorded
    # since _record_memory_history was called.  Load it at pytorch.org/memory_viz.
    if args.memory_profile and use_cuda:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Memory snapshot saved → memory_snapshot.pickle")

    # =========================================================================
    # Print results
    # =========================================================================
    fwd_mean, fwd_std = _mean_and_std(forward_times_ms)

    compile_tag = "  [torch.compile]" if args.compile else ""
    precision_tag = "  [BF16 mixed precision]" if args.mixed_precision else ""
    print(f"\nMode:            {args.mode}{compile_tag}{precision_tag}")
    print(f"Forward:         {fwd_mean:.2f} ± {fwd_std:.2f} ms")

    total_mean_ms = fwd_mean
    if backward_times_ms:
        bwd_mean, bwd_std = _mean_and_std(backward_times_ms)
        print(f"Backward:        {bwd_mean:.2f} ± {bwd_std:.2f} ms")
        total_mean_ms += bwd_mean

    if optimizer_times_ms:
        opt_mean, opt_std = _mean_and_std(optimizer_times_ms)
        print(f"Optimizer step:  {opt_mean:.2f} ± {opt_std:.2f} ms")
        total_mean_ms += opt_mean

    # Throughput: how many tokens the model can process per second.
    # Useful for comparing configurations and estimating training costs.
    throughput = tokens_per_step / (total_mean_ms / 1e3)
    print(f"Throughput:      {throughput:,.0f} tokens/s")

    if use_cuda:
        # Peak memory allocated on the GPU during the timed region.
        # Divide by 1024³ to convert bytes → gigabytes.
        peak_memory_gb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
        print(f"Peak GPU memory: {peak_memory_gb:.3f} GB")


if __name__ == "__main__":
    main()
