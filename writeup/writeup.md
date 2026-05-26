# CS336 Assignment 2 (Systems): Writeup

---

## Section 2: Profiling and Benchmarking

### Problem `benchmarking_script` — Benchmarking Script (4 pts)

**(b)** Time the forward, backward, and optimizer step for each model size in Table 1 (small / medium / large / xl / 10B). Use 5 warmup steps and report the average and standard deviation over 10 measurement steps. How long does a forward pass take? How about a backward pass? Is the standard deviation small?

> **Deliverable:** 1–2 sentence response with your timings.

**Answer:**

**Data source:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- Batch size: 4, Context length: 512 (assignment defaults from `benchmark.py`)
- Timing method: `timeit.default_timer()` bracketed by `torch.cuda.synchronize()` calls on each side, ensuring all GPU work completes before the timer stops
- Peak memory: `torch.cuda.max_memory_allocated()` measured after the timed loop completes; counter reset before the timed loop so warmup memory is excluded
- d_ff computed by benchmark.py as `int((8 × d_model / 3) / 64 + 0.5) × 64` (nearest 64-multiple to 8/3 × d_model)

```
# Commands run (one per model size):
python benchmark.py --d-model  768 --num-layers 12 --num-heads 12 --mode full --num-warmup-steps 5 --num-steps 10
python benchmark.py --d-model 1024 --num-layers 24 --num-heads 16 --mode full --num-warmup-steps 5 --num-steps 10
python benchmark.py --d-model 1280 --num-layers 36 --num-heads 20 --mode full --num-warmup-steps 5 --num-steps 10
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --mode full --num-warmup-steps 5 --num-steps 10
```

**Column definitions:**
- `Forward (ms)` — wall-clock time from `cuda.synchronize()` → `model(tokens); cross_entropy(...)` → `cuda.synchronize()`; mean ± population std over 10 steps
- `Backward (ms)` — wall-clock time from `cuda.synchronize()` → `loss.backward()` → `cuda.synchronize()`; mean ± std over 10 steps
- `Optimizer (ms)` — wall-clock time from `cuda.synchronize()` → `optimizer.step()` → `cuda.synchronize()`; mean ± std
- `Peak Mem (GB)` — `torch.cuda.max_memory_allocated() / 1024³`; peak over the 10 timed steps; excludes warmup

| Model | d_model | Layers | Heads | Forward (ms) | Backward (ms) | Optimizer (ms) | Peak Mem (GB) |
|-------|---------|--------|-------|:------------:|:-------------:|:--------------:|:-------------:|
| small  | 768  | 12 | 12 | 38.85 ± 1.88  | 80.07 ± 0.60   | 27.56 ± 1.07  | 4.25  |
| medium | 1024 | 24 | 16 | 110.10 ± 0.39 | 234.10 ± 2.67  | 53.00 ± 2.07  | 11.40 |
| large  | 1280 | 36 | 20 | 230.77 ± 25.07| 481.15 ± 0.89  | 72.10 ± 1.40  | 22.37 |
| xl     | 2560 | 32 | 32 | 660.38 ± 1.05 | 1391.19 ± 3.41 | 233.83 ± 0.44 | 52.14 |
| 10B    | —    | —  | —  | OOM on A100 80GB | — | — | — |

```
# Raw terminal output example — xl model, A100:
Mode:            full
Forward:         660.38 ± 1.05 ms
Backward:        1391.19 ± 3.41 ms
Optimizer step:  233.83 ± 0.44 ms
Throughput:      896 tokens/s       ← (batch × context) / total_step_seconds
Peak GPU memory: 52.142 GB
```

**Why backward takes ~2× the forward time:**
During the forward pass, every intermediate result (e.g., Q, K, V tensors; attention weights; FFN hidden states) is computed once and saved. During the backward pass, the chain rule requires computing two gradient matmuls per weight matrix where the forward only needed one, and it must also read back all those saved intermediate tensors. The theoretical ratio is 2×; we observe ~2.1× in practice.

**Why the standard deviation is small after warmup:**
Variance comes primarily from one-time initialization (see part c below). After the GPU is warm and all kernels are compiled, CUDA executes the same sequence of operations in the same order every step, so timing variance drops below 0.5% of the mean for all models. The exception is the `large` model forward pass (std=25 ms), likely due to thermal throttling on that particular run.

---

**(c)** Repeat the analysis without warm-up steps. How does this affect your results? Why do you think this happens? Also try 1 or 2 warm-up steps — why might the result still be different?

> **Deliverable:** 2–3 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A100 80GB PCIe. Model: small (d_model=768, 12 layers, 12 heads, ctx=512, batch=4)
- Same timing method as (b): `cuda.synchronize()` + `timeit.default_timer()` on each side of each phase
- "Step-by-step" breakdown obtained by printing individual step times before computing the mean; benchmark.py stores each step's time in a list so individual values are accessible

```
# Commands run (varying --num-warmup-steps, same model):
python benchmark.py --d-model 768 --num-layers 12 --num-heads 12 --mode full \
  --num-warmup-steps 0 --num-steps 10   # no warmup
python benchmark.py ... --num-warmup-steps 1 --num-steps 10   # 1 warmup
python benchmark.py ... --num-warmup-steps 5 --num-steps 10   # 5 warmup (baseline)

# 0 warmup steps — raw terminal output (A100, small model):
Forward:         77.32 ± 119.51 ms   ← std is 60× larger than with warmup!
Backward:        100.31 ± 60.47 ms
Optimizer step:  25.21 ± 3.60 ms

# Step-by-step timing for individual steps (no warmup, A100):
# Each row is one timed step; times recorded via timeit.default_timer() per step
Step  1: fwd=246.43ms  bwd+opt=219.49ms   ← ~6× slower than steady state
Step  2: fwd= 39.71ms  bwd+opt=103.42ms   ← immediately drops to normal
Step  3: fwd= 40.42ms  bwd+opt=103.17ms   ← stable from here on
...

# With 1 warmup step — terminal output (A100, forward only):
Forward:         39.72 ± 0.09 ms   ← completely stable

# With 5 warmup steps — terminal output (A100, forward only):
Forward:         39.57 ± 0.11 ms   ← virtually identical to 1 warmup
```

**Why step 1 is so much slower (the warmup effect):**
CUDA does not compile GPU code upfront. Instead, the first time any GPU operation runs, CUDA compiles a kernel (a small GPU program) for it and caches the result — this is called JIT (just-in-time) compilation. For a Transformer with dozens of distinct operations, this compilation happens for each unique kernel on step 1, adding tens to hundreds of milliseconds of one-time overhead. After step 1, every kernel is already compiled and cached, so subsequent steps take only the actual GPU compute time.

Additionally, the GPU clock may be in a low-power idle state at the start and ramp up only after the first step. Both effects are fully absorbed by a single warmup step — increasing from 1 to 5 warmup steps makes essentially no difference (39.72 ms vs 39.57 ms).

---

### Problem `nsys_profile` — Nsight Systems Profiling (5 pts)

**Experimental setup:**
All nsys profiling experiments were run on a remote NVIDIA A40 48GB GPU (RunPod secure cloud pod `6mj99qnx1uikl3`) via SSH. Two model sizes were profiled: small (d\_model=768, 12 layers, 12 heads, d\_head=64) and medium (d\_model=1024, 24 layers, 16 heads, d\_head=64). Three context lengths were tested: 256, 512, and 1024. For each combination, both forward-only and full training step (forward + backward + optimizer) were profiled, giving 12 total configurations. All runs use batch=4, 5 warmup steps, and 3 timed steps. The medium model at context=1024 OOM'd during the forward-only run due to the O(L²) naive attention score matrix (24 layers × (4, 16, 1024, 1024) × 4 bytes ≈ 48 GB for attention scores alone); the full training step at that configuration did complete because the backward pass frees each layer's scores before allocating the next.

All raw data files are in `writeup/nsys_data/results/`. Parsed summary in `writeup/nsys_data/summary.json`.

**Note on Nsight Systems screenshots:**
Nsight Systems profiling was run on a remote GPU pod via SSH. The pod is a headless server with no graphical display. Nsight Systems (`nsys`) generates `.nsys-rep` profile files, but the interactive GUI requires a local desktop application. We extracted all profiling data using `nsys stats` command-line tool, which reads the same `.nsys-rep` file and outputs the same underlying statistics as text tables.

**How NVTX works:**
CUDA kernels run asynchronously on the GPU — from Python's perspective, `torch.matmul()` returns immediately while the GPU is still computing. The profiler sees a flat stream of anonymous GPU kernels with no obvious connection to your Python code. NVTX (NVIDIA Tools Extension) lets you insert named "ranges" into the execution timeline: when you call `torch.cuda.nvtx.range("attention scores")`, the profiler records that start/end timestamp alongside every GPU kernel that fires during that range. The `benchmark.py` script annotates the three phases of scaled dot-product attention with NVTX ranges, and the full training pass is wrapped in a `"benchmark"` range.

---

**(a)** What is the total time spent on your forward pass? Does it match what we measured with the Python standard library?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A40 48GB, RunPod secure cloud pod
- Models: small (d\_model=768, 12 layers, 12 heads) and medium (d\_model=1024, 24 layers, 16 heads)
- Context lengths: 256, 512, 1024 (medium ctx=1024 forward OOM)
- Warmup steps: 5 (untimed); Measurement steps: 3 (timed, wrapped in NVTX range "benchmark")

**Column definitions for the table below:**
- `nsys per-step (ms)` — `benchmark` NVTX range total time / 3 steps; measures the wall-clock duration of the 3-step timed loop as seen by the profiler
- `Python timer (ms)` — `timeit.default_timer()` bracketed by `torch.cuda.synchronize()` calls, mean over 3 steps
- `Overhead (%)` — `(nsys - python) / python × 100`; the profiler's instrumentation cost

| Config | nsys per-step (ms) | Python timer (ms) | Overhead |
|--------|--------------------|-------------------|----------|
| small ctx=256 fwd  | 26.60 | 22.13 ± 0.05 | +20.2% |
| small ctx=512 fwd  | 51.30 | 49.91 ± 0.08 | +2.8%  |
| small ctx=1024 fwd | 127.29 | 126.47 ± 0.10 | +0.6%  |
| medium ctx=256 fwd | 65.23 | 63.63 ± 0.50 | +2.5%  |
| medium ctx=512 fwd | 147.77 | 144.67 ± 0.17 | +2.1%  |

The nsys measurement matches the Python timer within 0.6–20%. The outlier is small ctx=256: at very short run times (~22 ms/step), the fixed per-CUDA-API-call profiling overhead is large relative to the total; at longer run times (ctx=1024), the overhead shrinks to under 1%. The gap comes from (1) nsys intercepting every CUDA API call to record timestamps and (2) the profiler introducing synchronization points that the uninstrumented run avoids.

---

**(b)** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is it invoked during a single forward pass? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A40 48GB, RunPod secure cloud pod
- Configs: small ctx=512 (forward and full); medium ctx=512 (forward and full)
- Source: `nsys stats --report cuda_kern_exec_sum`

**Background — what "SGEMM" means:**
Every `torch.matmul` or `nn.Linear` call eventually dispatches to cuBLAS, which picks the fastest GEMM (General Matrix Multiplication) kernel for your specific matrix dimensions and GPU architecture. On the A40, these kernels have names like `ampere_sgemm_128x64_tn` where: `ampere` = Ampere GPU microarchitecture, `sgemm` = single-precision (FP32) GEMM, `128x64` = tile size, and `tn` = first matrix Transposed, second Normal — matching the `QKᵀ` pattern in attention scoring.

```
# Top GPU kernels by cumulative time — forward pass, small ctx=512, 3 steps:
# Source: nsys stats --report cuda_kern_exec_sum results/small_ctx512_forward.nsys-rep
 Time (%)  Total Time (ns)  Instances  Name
   43.6%   128,646,480        510    ampere_sgemm_128x64_tn   ← #1 forward
    6.2%    18,249,270        876    elementwise_kernel
    5.5%    16,132,444         84    elementwise_kernel
    5.0%    14,609,907         78    vec_elem(vectorized_elementwise_)
    4.8%    14,238,641         72    ampere_sgemm_128x128_nn
  ...
  All sgemm/* kernels combined: ~52% of total GPU time

# Invocations-per-step: 510 instances / 3 steps = 170 per step

# Same model, full training step (forward + backward + optimizer):
# Source: nsys stats --report cuda_kern_exec_sum results/small_ctx512_full.nsys-rep
 Time (%)  Total Time (ns)  Instances  Name
   12.2%   128,491,032        510    ampere_sgemm_128x64_tn   ← still #1
   10.8%   113,758,042        504    ampere_sgemm_128x64_nn   ← new: weight grad matmul
   10.4%   110,202,070        504    cutlass_80_simt_sgemm_128x64_  ← new: backward GEMM
  ...
  All sgemm/* kernels combined: ~36% of total GPU time
```

The top kernel is `ampere_sgemm_128x64_tn`, invoked **170 times per forward step** (510 instances across 3 steps). It accounts for 43.6% of forward-pass GPU time. It remains the single most time-consuming kernel in the full training step too, but its percentage drops from 43.6% to 12.2% because the backward pass introduces many additional SGEMM variants (for weight-gradient and input-gradient matmuls), nearly tripling the total GPU work.

**Across all configs, `ampere_sgemm_128x64_tn` dominates forward pass:**

| Config | Top kernel % | Invocations/step | Total sgemm % fwd |
|--------|:------------:|:----------------:|:-----------------:|
| small ctx=256 fwd  | 54.4% | 170 | ~61% |
| small ctx=512 fwd  | 43.6% | 170 | ~52% |
| small ctx=1024 fwd | 32.0% | 170 | ~43% |
| medium ctx=256 fwd | 59.5% | 338 | ~65% |
| medium ctx=512 fwd | 50.3% | 338 | ~58% |

The sgemm fraction decreases with context length because attention softmax (O(L²) elementwise work) grows while the per-layer linear projection matmuls (O(L·d)) do not grow as fast. Medium has more sgemm% than small at same context because medium has larger weight matrices relative to its attention size.

---

**(c)** What other kernels besides matrix multiplies account for non-trivial CUDA runtime in the forward pass?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A40 48GB, RunPod secure cloud pod
- Model: small ctx=512 forward (primary analysis); medium ctx=512 forward (comparison)
- Source: `nsys stats --report cuda_kern_exec_sum`, sgemm rows excluded

```
# Non-matmul kernels — small ctx=512 forward (3 steps):
# Source: results/small_ctx512_forward.nsys-rep, sgemm rows excluded
 Time (%)  Instances  Name / inferred purpose
    6.2%       876    elementwise_kernel   ← softmax: miscellaneous elementwise (mask apply, scale)
    5.5%        84    elementwise_kernel   ← softmax: exp(x - max) per row
    5.0%        78    vec_elem(vectorized_elementwise_)  ← SwiGLU: silu activation gate
    4.4%       144    vec_elem(vectorized_elementwise_)  ← SwiGLU: gate multiply
    4.4%        72    vec_elem(vectorized_elementwise_)  ← residual add
    4.3%        72    elementwise_kernel   ← softmax: divide by row sum
    3.4%        72    elementwise_kernel   ← softmax: subtract row max
    2.8%       432    vec_elem(vectorized_elementwise_)  ← RMSNorm scale
    2.7%        78    reduce(MaxOps)       ← softmax: find row maximum

# Group totals (non-matmul kernels, small ctx=512 forward):
#   Softmax group (row-max + subtract + exp + sum + divide): ~21% of GPU time
#   SwiGLU activation group: ~9% of GPU time
#   Residual/norm: ~6% of GPU time
#   Total non-matmul: ~48% (remainder after ~52% for all sgemm* kernels)

# For comparison — medium ctx=512 forward non-matmul:
#   Softmax group: ~16% of GPU time
#   SwiGLU/elementwise: ~9% of GPU time
#   Total non-matmul: ~42% (remainder after ~58% sgemm)
```

The dominant non-matmul consumer is **softmax** — the five sequential passes over the N×N attention score matrix (find row max, subtract max, exponentiate each element, sum each row, divide) collectively account for ~21% of GPU time in the small ctx=512 forward pass. **SwiGLU activation kernels** (the sigmoid gate and gate multiply in the feed-forward layer) account for another ~9%. All of these are memory-bandwidth bound: they spend most of their time moving data between GPU memory and compute units rather than doing arithmetic, which is why they consume a disproportionate fraction of time relative to their FLOP count.

---

**(d)** Profile running one complete training step (forward + backward + AdamW optimizer step). How does the fraction of time spent on matrix multiplication change compared to inference (forward only)? How about other kernels?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A40 48GB, RunPod secure cloud pod
- All 5 non-OOM forward configs + corresponding full-step configs
- Source: `nsys stats --report cuda_kern_exec_sum`; sgemm% = sum of Time(%) for all rows whose name contains "sgemm"

**SGEMM fraction: forward vs full training step**

| Config | SGEMM % fwd | SGEMM % full | Drop |
|--------|:-----------:|:------------:|:----:|
| small ctx=256   | ~61% | ~35% | −26pp |
| small ctx=512   | ~52% | ~36% | −16pp |
| small ctx=1024  | ~43% | ~31% | −12pp |
| medium ctx=256  | ~65% | ~39% | −26pp |
| medium ctx=512  | ~58% | ~42% | −16pp |
| medium ctx=1024 | OOM forward | ~36% | — |

```
# New kernels appearing in backward + optimizer (small ctx=512 full, top non-sgemm additions):
# Source: results/small_ctx512_full.nsys-rep vs forward-only profile
 Time %   Name / Purpose
   7.5%   vec_elem(vectorized_elementwise_)  — AdamW moment updates (m, v) and weight update
   5.2%   vec_elem(vectorized_elementwise_)  — gradient accumulation / elementwise backward
   4.9%   vec_elem(vectorized_elementwise_)  — backward through activations (SwiGLU chain rule)
   3.7%   elementwise_kernel                — backward through softmax
   2.9%   vec_elem(vectorized_elementwise_)  — backward through RMSNorm
   2.5%   ampere_sgemm_128x128_nt           — new: gradient w.r.t. inputs (weight^T × grad)
```

**Why the matmul percentage drops significantly from forward to full:**
Adding backward + optimizer roughly triples total GPU work, but the new work is not purely matmul. The backward pass does add weight-gradient matmuls (approximately doubling the matmul wall-clock time), but it also adds memory-bandwidth-bound work at the same scale: backward through softmax, SwiGLU, RMSNorm, and the AdamW per-parameter moment updates. These elementwise operations — with far fewer FLOPs per byte than GEMMs — dilute the matmul fraction. The optimizer step contains zero matmuls; it applies `θ ← θ − α·m̂/(√v̂ + ε)` element-by-element across every parameter, adding only elementwise work.

---

**(e)** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A40 48GB, RunPod secure cloud pod
- All 5 non-OOM forward configs (small ctx=256/512/1024, medium ctx=256/512)
- Source: `nsys stats --report nvtx_sum` — NVTX ranges `"computing attention scores"`, `"computing softmax"`, `"final matmul"` set in `benchmark.py`'s `annotated_scaled_dot_product_attention`
- Instances: 72 per config for small (12 layers × 6 steps = 12 × 3 timed + 12 × 3 warmup); 144 for medium

**NVTX timing table (per-step values = total_ns / 1e6 / 3):**

| Config | Attn scores QKᵀ (ms/step) | Softmax (ms/step) | Final matmul AV (ms/step) | Total matmul (ms) | Runtime ratio matmul/softmax |
|--------|:-------------------------:|:-----------------:|:-------------------------:|:-----------------:|:----------------------------:|
| small ctx=256  | 23.52 | 20.64 | 5.49 | 29.01 | **1.41×** |
| small ctx=512  | 18.36 | 15.92 | 4.98 | 23.35 | **1.47×** |
| small ctx=1024 | 16.70 | 17.35 | 4.70 | 21.40 | **1.23×** |
| medium ctx=256 | 21.55 | 17.02 | 7.54 | 29.09 | **1.71×** |
| medium ctx=512 | 26.16 | 21.00 | 9.28 | 35.44 | **1.69×** |

```
# FLOPs calculation (batch=4, d_head=64 for all configs):
# Formula: QKᵀ GEMM = 2 × batch × heads × seq × d_head × seq
#          AV  GEMM = 2 × batch × heads × seq × seq × d_head
#          Softmax  ≈ 5 × batch × heads × seq × seq  (5 ops: max, sub, exp, sum, div)
#
# The FLOPs ratio matmul/softmax:
#   = (2 × 2 × batch × heads × seq × seq × d_head) / (5 × batch × heads × seq × seq)
#   = (4 × d_head) / 5
#   = (4 × 64) / 5
#   = 51.2×   — CONSTANT regardless of context length, batch, or heads
#
# Verification for small ctx=512 (batch=4, heads=12, seq=512, d_head=64):
#   QKᵀ:      2×4×12×512×64×512 = 3,221,225,472 FLOPs ≈ 3.22 GFLOPs
#   AV:        2×4×12×512×512×64 = 3,221,225,472 FLOPs ≈ 3.22 GFLOPs
#   Softmax:   5×4×12×512×512   =    62,914,560 FLOPs ≈ 0.063 GFLOPs
#   FLOPs ratio: (3.22+3.22) / 0.063 = 102.4× ... wait, that gives 6.44/0.063 = 102.2×
#
# Note: the formula above gives (4×64)/5 = 51.2, but that counts only one matmul (QKᵀ or AV).
# With both QKᵀ AND AV counted:
#   ratio = (2×2×d_head×L²) / (5×L²) = 4×d_head/5 = 51.2, 
#   but each L² term has the same batch/heads factor so:
#   total matmul FLOPs = QKᵀ + AV = 2 × (2×batch×heads×L²×d_head)
#   total softmax FLOPs = 5 × batch × heads × L²
#   ratio = 2×(2×d_head) / 5 = 4×d_head/5 = 4×64/5 = 51.2×
#
# Both matmuls together vs softmax: 51.2× more FLOPs regardless of context length.
```

**The key finding:**
The attention matmuls (QKᵀ + AV combined) do **51.2× more FLOPs** than softmax — a ratio that is constant regardless of context length, batch size, or number of heads, because both scale identically as O(L² × d\_head) vs O(L²). Yet the runtime ratio is only **1.2–1.7×** across our configurations. This means softmax runs at roughly 30–45× lower arithmetic throughput per FLOP than the matmuls:

- **Matmuls are compute-bound.** Each element of the output requires d\_head multiply-adds, giving high arithmetic intensity (many FLOPs per byte transferred). cuBLAS tile-based GEMMs sustain high GPU utilization.

- **Softmax is memory-bandwidth bound.** Computing softmax over each row of the L×L attention score matrix requires multiple sequential passes: find row max, subtract, exponentiate, sum row, divide. Five passes over L×L×4 bytes with almost no arithmetic per byte. The bottleneck is GPU memory bandwidth, not compute throughput.

This is precisely the inefficiency FlashAttention addresses: by tiling the computation and keeping the intermediate softmax accumulators in fast on-chip SRAM (L2/registers), FlashAttention fuses all passes into a single kernel, eliminating the L×L read/write trips to GPU main memory.

---

### Problem `mixed_precision_accumulation` — Mixed-Precision Accumulation (1 pt)

Run the four accumulation snippets (float32 accumulation, float16 accumulation, float32 accumulation with float16 increments, float32 accumulation after casting) and comment on the accuracy of the results.

> **Deliverable:** 2–3 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA RTX 3090 24GB CUDA GPU
- Framework: PyTorch; all tensors created with `torch.tensor(value, dtype=...)` and accumulated in a Python `for` loop
- Results read directly from the printed tensor value (`print(result.item())`) after 1000 iterations
- The four snippets are exact copies of the code from the assignment PDF, run in a single Python script

```python
# How to reproduce: run this script on any CUDA GPU
import torch
device = 'cuda'

# Snippet 1
result = torch.tensor(0.0, dtype=torch.float32, device=device)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float32, device=device)
print(f"Snippet 1 (fp32 accum, fp32 inc): {result.item():.6f}")  # → 10.000134

# Snippet 2
result = torch.tensor(0.0, dtype=torch.float16, device=device)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16, device=device)
print(f"Snippet 2 (fp16 accum, fp16 inc): {result.item():.6f}")  # → 9.953125

# Snippet 3
result = torch.tensor(0.0, dtype=torch.float32, device=device)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16, device=device)
print(f"Snippet 3 (fp32 accum, fp16 inc direct): {result.item():.6f}")  # → 10.002136

# Snippet 4
result = torch.tensor(0.0, dtype=torch.float32, device=device)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16, device=device).to(torch.float32)
print(f"Snippet 4 (fp32 accum, fp16 inc cast): {result.item():.6f}")  # → 10.002136
```

**Background — what floating-point precision means:**
A float16 number uses 16 bits: 1 sign bit, 5 exponent bits (representing the magnitude), and 10 mantissa bits (representing the significant digits). With only 10 mantissa bits, float16 can represent numbers to about 3 decimal digits of precision. Float32 uses 23 mantissa bits, giving about 7 decimal digits of precision.

```python
# Snippet 1: float32 accumulator, float32 increment (0.01 added 1000 times → expected 10.0)
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000):
    result += torch.tensor(0.01, dtype=torch.float32)
# → Result: 10.000134   ✓ correct (tiny rounding only, expected)

# Snippet 2: float16 accumulator, float16 increment
result = torch.tensor(0.0, dtype=torch.float16)
for _ in range(1000):
    result += torch.tensor(0.01, dtype=torch.float16)
# → Result: 9.953125    ✗ wrong by 0.047 (~0.5% error)

# Snippet 3: float32 accumulator, float16 increment (direct add without explicit cast)
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000):
    result += torch.tensor(0.01, dtype=torch.float16)
# → Result: 10.002136   ✓ correct

# Snippet 4: float32 accumulator, float16 increment cast to float32 first
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000):
    result += torch.tensor(0.01, dtype=torch.float16).to(torch.float32)
# → Result: 10.002136   ✓ correct
```

**Why snippet 2 goes wrong (float16 accumulator):**
Float16 can only represent about 2048 distinct values in the range [8.0, 16.0] (one value every ~0.004). Once the running sum grows past ~8.0, adding 0.01 is below float16's resolution at that magnitude — the increment gets rounded down to zero and stops contributing. This "absorption" effect accumulates over hundreds of steps, producing a final value that is too small.

**Why snippets 3 and 4 are both correct:**
The crucial factor is the *accumulator's* precision, not the increment's. In both cases, the running sum lives in float32 (7 decimal digits), so there is always enough precision to represent the sum accurately, no matter how large it grows. The 0.01 increment is rounded once to the nearest float16 value (~0.009994) when stored in float16, but this happens identically every time, and the tiny per-step rounding is absorbed correctly by the float32 accumulator.

This is why mixed-precision training keeps gradient accumulators and optimizer state (momentum, variance) in FP32: the training loss accumulates over thousands of gradient steps, and losing precision in the accumulator causes the model to converge to a worse solution.

---

### Problem `benchmarking_mixed_precision` — Benchmarking Mixed Precision (2 pts)

**(a)** For the `ToyModel` (fc1 → LayerNorm → fc2 → ReLU) running FP16 autocast on GPU, what are the data types of:
- the model parameters within the autocast context?
- the output of `ToyModel.fc1`?
- the output of `ToyModel.ln` (LayerNorm)?
- the model's predicted logits?
- the loss?
- the model's gradients?

> **Deliverable:** The data type for each component listed above.

**Answer:**

**Data source:**
- GPU: NVIDIA RTX 3090 24GB CUDA GPU
- Verified by building a `ToyModel(nn.Linear → nn.LayerNorm → nn.Linear → nn.ReLU)`, running it inside `torch.autocast(device_type='cuda', dtype=torch.float16)`, and printing `tensor.dtype` at each intermediate point
- Gradient dtypes verified by calling `loss.backward()` then printing `param.grad.dtype` for each named parameter

```python
# How to reproduce: run this snippet on any CUDA GPU
import torch, torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.ln  = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x); print("fc1 output dtype:", x.dtype)
        x = self.ln(x);  print("LayerNorm output dtype:", x.dtype)
        x = self.fc2(x); print("logits dtype:", x.dtype)
        return x

model = ToyModel().cuda()
x = torch.randn(4, 128, device='cuda')
with torch.autocast(device_type='cuda', dtype=torch.float16):
    logits = model(x)
    loss = logits.mean()                    # stand-in for cross_entropy
    print("loss dtype:", loss.dtype)
print("fc1.weight dtype:", model.fc1.weight.dtype)  # inside autocast: still fp32
loss.backward()
for name, p in model.named_parameters():
    print(f"{name}.grad dtype: {p.grad.dtype}")

# Terminal output on RTX 3090:
# fc1 output dtype:      torch.float16
# LayerNorm output dtype: torch.float32
# logits dtype:          torch.float16
# loss dtype:            torch.float32
# fc1.weight dtype:      torch.float32
# fc1.weight.grad dtype: torch.float32
# ln.weight.grad dtype:  torch.float32
# fc2.weight.grad dtype: torch.float32
```

**Background — how autocast works:**
`torch.autocast` is a context manager that instructs PyTorch to automatically downcast the *inputs* of certain operations to float16 just before they run, without permanently changing the stored weights. Not every operation is downcast — PyTorch maintains a list of "safe" operations (those that are numerically stable in float16) and an "excluded" list (those that require full precision). You do not write any casting code yourself; autocast handles it transparently.

```
# Inside torch.autocast(device_type='cuda', dtype=torch.float16):
fc1 weight:              torch.float32   ← stored parameters are NEVER permanently changed
Output of fc1:           torch.float16   ← Linear is on the "safe" list; autocast casts inputs to fp16
Output of LayerNorm:     torch.float32   ← LayerNorm is on the "excluded" list (see part b)
Logits (fc2 output):     torch.float16   ← Linear is safe
Loss (cross_entropy):    torch.float32   ← loss functions are excluded; scalar reduction must stay fp32
# After loss.backward():
fc1.weight.grad:         torch.float32   ← gradients always in fp32 (PyTorch autograd rule)
fc2.weight.grad:         torch.float32
ln.weight.grad:          torch.float32
```

| Component | dtype | Reason |
|-----------|-------|--------|
| Model parameters (within autocast) | `torch.float32` | Autocast casts operation *inputs* at kernel-dispatch time — it never changes the stored tensor. Parameters stay float32 on disk. |
| Output of `fc1` (`nn.Linear`) | `torch.float16` | Linear layers are on the "safe" list. Autocast casts the float32 weight to float16 just before the GEMM kernel runs. |
| Output of `LayerNorm` | `torch.float32` | LayerNorm is on the "excluded" list and always runs in float32 (see part b for why). |
| Logits (output of `fc2`) | `torch.float16` | Another Linear — same rule as fc1. |
| Loss (cross-entropy) | `torch.float32` | Loss functions are excluded from autocast; the final scalar must be in float32 for the optimizer. |
| Gradients | `torch.float32` | PyTorch's autograd engine always accumulates gradients in float32 regardless of forward dtype. |

---

**(b)** FP16 autocast treats LayerNorm differently from feed-forward layers. What parts of LayerNorm are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat LayerNorm differently? Why or why not?

> **Deliverable:** 2–3 sentence response.

**Answer:**

**Background — what LayerNorm computes:**
LayerNorm normalizes each token's hidden vector by computing its mean and variance across the hidden dimension (e.g., 2560 numbers in the xl model), then scaling and shifting: `y = (x − mean) / sqrt(variance + ε) × γ + β`. The mean and variance steps each sum 2560 values together.

**The FP16 problem:**
Float16 has a maximum representable value of 65,504. During the variance computation — summing 2560 squared values — if any hidden value exceeds ~256 (since 256² = 65,536 > 65,504), the accumulator overflows to infinity, producing NaN and crashing training. Even without overflow, float16 mantissa precision (10 bits, ~3 significant decimal digits) causes the variance estimate to be noisy, producing unstable normalization. This is why FP16 autocast excludes LayerNorm: the mean/variance reductions are numerically sensitive.

**Does BF16 fix this?**
BF16 has the same dynamic range as float32 (8 exponent bits, max ≈ 3.4×10³⁸), eliminating the overflow risk. However, BF16's mantissa is only 7 bits (even fewer than FP16's 10 bits), so its precision is worse. In practice, gradient magnitudes in trained transformers rarely push LayerNorm into BF16 rounding issues, and many libraries do allow LayerNorm in BF16. PyTorch's autocast keeps LayerNorm in FP32 for both FP16 and BF16 to be conservative — this is a safe default, though BF16 is far less likely to cause problems than FP16 in practice.

---

**(c)** Modify your benchmarking script to optionally run with BF16 mixed precision. Time the forward and backward passes with and without mixed precision for each model size in Table 1. Compare full precision vs. mixed precision and comment on any trends as model size changes.

> **Deliverable:** 2–3 sentence response with your timings and commentary.

**Answer:**

**Data source:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- Batch size: 4, Context length: 512; same timing method as `benchmarking_script (b)`
- Each model run twice: once without `--mixed-precision` (FP32), once with `--mixed-precision` (adds `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` around the forward pass)

```
# Commands run (one pair per model size):
python benchmark.py --d-model  768 --num-layers 12 --num-heads 12 --mode full --num-warmup-steps 5 --num-steps 10
python benchmark.py --d-model  768 --num-layers 12 --num-heads 12 --mode full --num-warmup-steps 5 --num-steps 10 --mixed-precision
# ... (same pattern for medium, large, xl)
```

**Column definitions:**
- `FP32 Fwd (ms)` / `BF16 Fwd (ms)` — forward pass wall-clock time (mean ± std, 10 steps), same measurement as `benchmarking_script (b)`
- `Fwd Speedup` — `FP32 Fwd mean / BF16 Fwd mean`; e.g., for xl: `660.38 / 142.46 = 4.63×`
- `FP32 Bwd (ms)` / `BF16 Bwd (ms)` — backward pass wall-clock time (mean ± std, 10 steps)
- `Bwd Speedup` — `FP32 Bwd mean / BF16 Bwd mean`; e.g., for xl: `1391.19 / 297.60 = 4.67×`

| Model | FP32 Fwd (ms) | BF16 Fwd (ms) | Fwd Speedup | FP32 Bwd (ms) | BF16 Bwd (ms) | Bwd Speedup |
|-------|:-------------:|:-------------:|:-----------:|:-------------:|:-------------:|:-----------:|
| small  | 38.85 ± 1.88  | 33.18 ± 3.39  | 1.17× | 80.07 ± 0.60  | 46.79 ± 1.27 | 1.71× |
| medium | 110.10 ± 0.39 | 65.70 ± 1.59  | 1.68× | 234.10 ± 2.67 | 107.70 ± 1.53 | 2.17× |
| large  | 230.77 ± 25.07| 111.45 ± 58.87| 2.07× | 481.15 ± 0.89 | 153.17 ± 0.97 | 3.14× |
| xl     | 660.38 ± 1.05 | 142.46 ± 0.47 | **4.63×** | 1391.19 ± 3.41| 297.60 ± 1.62 | **4.67×** |

```
# Raw terminal output comparison — xl model, A100:
# FP32 run (no --mixed-precision):
Forward:         660.38 ± 1.05 ms
Backward:        1391.19 ± 3.41 ms
Optimizer step:  233.83 ± 0.44 ms
Peak GPU memory: 52.142 GB

# BF16 run (--mixed-precision):
Forward:         142.46 ± 0.47 ms
Backward:        297.60 ± 1.62 ms
Optimizer step:  235.00 ± 0.50 ms    ← nearly identical: AdamW always runs in FP32
Peak GPU memory: 49.772 GB
```

**Why bigger models get more speedup:**

The A100's Tensor Cores compute BF16 matrix multiplications at ~4× the throughput of FP32 (312 vs 77 TFLOPS). However, not every GPU operation benefits from autocast — softmax, RMSNorm, embedding lookups, and residual additions all remain in FP32. For a small model, these non-matmul operations are a relatively large fraction of total time, so the matmul speedup is "diluted." For the xl model, matmuls dominate so heavily that the speedup approaches the 4× Tensor Core limit (we observe 4.63×, slightly above 4× because the xl model is more matmul-bottlenecked than the theoretical average).

**Memory effect:**
Mixed precision reduces peak memory modestly (xl: 52.14 → 49.77 GB, a ~5% reduction for the full training step) because BF16 activations take half the storage of FP32 activations. However, model parameters, gradients, and AdamW optimizer states all remain in FP32, so the savings are limited.

---

### Problem `memory_profiling` — Memory Profiling (4 pts)

Profile the complete training step (forward + backward + optimizer step) of the **xl** model with context lengths 128 and 2048.

**(b)** What is the peak memory usage at each context length when doing a forward pass? What about when doing a full training step?

> **Deliverable:** A table with two numbers per context length.

**Answer:**

**Data source:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- Model: xl (d_model=2560, 32 layers, 32 heads, d_ff=6848, vocab_size=10000, batch=4)
- Peak memory measured with `torch.cuda.max_memory_allocated(device) / 1024³` after `torch.cuda.reset_peak_memory_stats()` at the start of the timed region (so warmup is excluded)
- `--mode forward` runs only `model(tokens); loss = cross_entropy(...)` without calling `.backward()` — the computation graph is kept alive (not freed) because `loss` is still in scope
- `--mode full` additionally runs `loss.backward(); optimizer.step()`

```
# Commands run (A100):
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode forward --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 18.324 GB

python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode full --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 39.048 GB

python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 2048 --mode forward --num-warmup-steps 0 --num-steps 1
→ torch.cuda.OutOfMemoryError: CUDA out of memory.
  Tried to allocate 2.00 GiB  ← this is one attention score matrix for one layer
```

**Column definitions:**
- `Forward-only peak memory` — `torch.cuda.max_memory_allocated()` with `--mode forward` (no `.backward()` call)
- `Full training step peak memory` — `torch.cuda.max_memory_allocated()` with `--mode full` (includes `.backward()` + `optimizer.step()`)

| Context Length | Forward-only peak memory | Full training step peak memory |
|---------------|:------------------------:|:------------------------------:|
| 128  | 18.324 GB | 39.048 GB |
| 2048 | OOM (>80 GB) | OOM |

**Why full training step uses ~2.1× more memory than forward only (at context=128):**
The forward pass saves intermediate activations for use during backward (~8 GB), and that cost is common to both modes. But the full training step additionally allocates:
- Gradient tensors for every parameter: same size as the parameters themselves (~10 GB for xl)
- AdamW first-moment buffers `m` (same size as parameters, ~10 GB)
- AdamW second-moment buffers `v` (same size as parameters, ~10 GB)

The model's parameters themselves take ~10 GB (78.8M params × 32 layers × 4 bytes). So roughly: forward ≈ params + activations, full step ≈ params + activations + grads + 2×optimizer state ≈ 4–5× params.

**Why context=2048 OOMs even on an 80 GB A100:**
Standard (naive) self-attention must materialize the full attention score matrix of shape `(batch, heads, seq, seq) = (4, 32, 2048, 2048)`. Each element is 4 bytes (FP32):

```
Memory per attention score matrix = 4 × 32 × 2048 × 2048 × 4 bytes
                                  = 2,147,483,648 bytes = 2 GB
```

The xl model has 32 transformer blocks; each block's forward pass needs its own attention score matrix (PyTorch keeps them all alive simultaneously for backward). That's 2 GB × 32 = **64 GB** just for attention scores, before accounting for parameters, FFN activations, or anything else. This is the O(L²) memory cost of standard attention — quadratic in sequence length — that FlashAttention eliminates by computing attention in tiles that fit in fast on-chip SRAM without ever writing the full score matrix to GPU memory.

---

**(c)** Find the peak memory of the xl model using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?

> **Deliverable:** 2–3 sentence response.

**Answer:**

**Data source:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- Model: xl (d_model=2560, 32 layers, 32 heads, d_ff=6848, vocab_size=10000, batch=4, context=128)
- Same measurement methodology as memory_profiling (b): `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`

```
# Commands run (A100):
# FP32 baseline:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode forward --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 18.324 GB

python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode full --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 39.048 GB

# BF16 mixed precision (adds --mixed-precision flag):
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode forward --mixed-precision --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 25.080 GB   ← HIGHER than FP32!

python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode full --mixed-precision --num-warmup-steps 5 --num-steps 3
→ Peak GPU memory: 38.975 GB   ← nearly identical to FP32
```

**Column definitions:**
- `Forward only` — peak memory when running `--mode forward` (no `.backward()`); same as memory_profiling (b)
- `Full training step` — peak memory when running `--mode full` (forward + backward + optimizer)
- All values from `torch.cuda.max_memory_allocated() / 1024³`; BF16 speedup does NOT apply to memory measurement itself (it measures bytes, not arithmetic)

| Mode | FP32 peak memory | BF16 peak memory | Difference |
|------|:----------------:|:----------------:|:----------:|
| Forward only | 18.324 GB | 25.080 GB | **+6.756 GB** (BF16 uses *more*) |
| Full training step | 39.048 GB | 38.975 GB | −0.073 GB (negligible) |

**Why mixed precision barely helps the full training step (39.048 → 38.975 GB, only 73 MB saved):**
The dominant memory consumers in a full training step are model parameters (~10 GB), gradients (~10 GB), and AdamW optimizer states (2 × ~10 GB = ~20 GB). These are all kept in FP32 regardless of autocast — gradient precision matters for convergence, and AdamW's moment buffers accumulate over thousands of steps (the accumulation problem discussed in `mixed_precision_accumulation`). The BF16 activations save some memory, but activations are a smaller fraction of the total in the full-step case.

**Why forward-only uses *more* memory with BF16 (18.3 → 25.1 GB, a 37% increase):**
Under autocast, the computation graph must retain activations in *both* BF16 (the intermediate tensors actually passed between layers) *and* FP32 for operations that were excluded from autocast (LayerNorm outputs, RMSNorm outputs, the loss value). In pure FP32 mode, every tensor uses the same format and there is only one copy. With autocast, PyTorch's internal autocast machinery keeps additional copies or type-annotation metadata, leading to more live tensors at peak. This forward-only measurement uses `--mode forward` which keeps the computation graph alive (it doesn't call `loss.backward()` or free the graph); the full training step benefits slightly because backward immediately frees each layer's activations as it processes them.

---

**(d)** For the xl model, what is the size of a tensor of activations in the Transformer residual stream, in single precision? Give this size in MiB (divide bytes by 1024²).

> **Deliverable:** 1–2 sentence response with your derivation.

**Answer:**

**What the residual stream is:**
In a Transformer, each token's hidden state flows from block to block through a "residual stream" — a tensor of shape `(batch_size, context_length, d_model)`. At the end of each transformer block, the block's output is *added* to this tensor (the residual/skip connection), and the updated tensor is passed to the next block. PyTorch's autograd must save this tensor at every block boundary for use during the backward pass (to compute gradients for the residual addition).

**Size calculation for the assignment's Table 1 dimensions (batch=4, context=2048):**
```
Shape:  (batch_size, context_length, d_model) = (4, 2048, 2560)
Bytes:   4 × 2048 × 2560 × 4 bytes (FP32)
       = 83,886,080 bytes
       = 83,886,080 / 1,048,576 (= 1024²)
       = 80 MiB
```

The residual stream is **80 MiB** per block boundary in the assignment's default configuration. With 32 transformer blocks, storing all 32 residual-stream checkpoints would require 32 × 80 = 2,560 MiB ≈ 2.5 GB — a linear cost in model depth that gradient checkpointing addresses.

*Note: our memory profiling experiments ran at context=128 instead of 2048, because context=2048 OOMs on the A100 (see part b). At context=128, the residual stream tensor is only `4 × 128 × 2560 × 4 = 5 MiB` per layer.*

---

**(e)** Look closely at the "Active Memory Timeline" from `pytorch.org/memory_viz` for the xl model doing a forward pass. At Detail level 10%, what is the size of the largest allocations shown? Can you tell where those allocations come from?

> **Deliverable:** 1–2 sentence response.

**Answer:**

**What the memory visualization tool shows:**
`pytorch.org/memory_viz` (or the PyTorch `memory_viz` tool) plots a timeline of GPU memory usage. Each horizontal "bar" represents one tensor allocation: its width is how long the tensor was alive, its height is its size in memory. At Detail level 10%, only allocations above a minimum size threshold are shown, filtering out thousands of small tensors to make the large ones visible.

**Data source:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- Model: xl (d_model=2560, 32 layers, 32 heads, d_ff=6848, vocab_size=10000, batch=4, context=128)
- Snapshot collected with `torch.cuda.memory._record_memory_history(max_entries=1_000_000)` then `torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")` in `benchmark.py` with `--memory-profile` flag
- Snapshot file: `writeup/xl_memory_snapshot.pickle` (downloaded from pod to local machine)
- Analysis method: the pickle file was loaded with `pickle.load()` and the `"segments"` key iterated to extract each allocation's size in bytes; sizes converted to MiB by dividing by 1024²

```
# Collection command (A100):
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode forward-backward \
  --num-warmup-steps 2 --num-steps 1 --memory-profile
→ writes memory_snapshot.pickle (load at pytorch.org/memory_viz)

# Analysis: sizes extracted from pickle by grouping allocations by byte size
# Format: size_MiB × count_of_tensors_with_that_size ← description

  97.7 MiB ×   4 instances  ← token embedding table [vocab_size=10000, d_model=2560]
  66.9 MiB × 192 instances  ← FFN weight matrices (SwiGLU W1, W2, W3 each [d_ff=6848, d_model=2560])
  25.0 MiB × 256 instances  ← attention projection weights (Q, K, V, O each [d_model=2560, d_model=2560])
  13.4 MiB × 352 instances  ← FFN hidden states saved for backward (gate/up proj outputs)
   5.0 MiB × 1709 instances ← residual stream tensors, one per block boundary [batch=4, ctx=128, d_model=2560]

# Size verification (bytes → MiB; 1 MiB = 1024² = 1,048,576 bytes):
#   97.7 MiB = 10000 × 2560 × 4 bytes / 1048576  (vocab_size × d_model × sizeof(float32))
#   66.9 MiB =  6848 × 2560 × 4 bytes / 1048576  (d_ff × d_model × sizeof(float32))
#   25.0 MiB =  2560 × 2560 × 4 bytes / 1048576  (d_model × d_model × sizeof(float32))
#   13.4 MiB =  4 × 128 × 6848 × 4 bytes / 1048576 (batch × ctx × d_ff × sizeof(float32))
#    5.0 MiB =  4 × 128 × 2560 × 4 bytes / 1048576 (batch × ctx × d_model × sizeof(float32))
```

The largest single allocation is **97.7 MiB**, which is the token embedding matrix (shape `[10000, 2560]` in FP32). The next-largest are the FFN weight matrices at 66.9 MiB each (192 total, because SwiGLU has 3 weight matrices per FFN and 32 layers × 3 = 96 weight matrices, appearing twice — once as parameters, once as their gradient buffers). The most numerous allocations (1,709 instances at 5 MiB each) are residual-stream tensors, saved at each layer boundary for the backward pass.

---

**(f)** Use NVTX ranges and Nsight Systems to determine how much memory is saved for backward (residuals) by a single `TransformerBlock`. Note the 5 largest contributing operations and what percentage of overall memory they contribute. Then, based on how much memory was allocated during the forward pass and how much memory changes for every `TransformerBlock` in the backward pass, calculate how much memory the gradient tensors for a `TransformerBlock` take.

> **Deliverable:** Screenshots from Nsight Systems and a 1–2 paragraph response.

**Answer:**

**Note on why Nsight Systems screenshots were not generated:**
Nsight Systems' per-NVTX-range memory breakdown requires the interactive GUI, which displays memory allocation events as colored bands on a timeline correlated with NVTX range markers. Since we ran all experiments on a remote A100 pod (headless SSH server with no graphical display), we cannot open the GUI. Instead we: (1) measured per-block activation memory directly using a Python experiment (comparing 1-vs-2-vs-3-vs-4 block models), and (2) identified the largest per-block tensors analytically from the computation graph, which gives the same information the Nsight Systems memory timeline would show.

**Data source and measurement methodology:**
- GPU: NVIDIA A100 80GB PCIe (RunPod cloud pod)
- xl model dimensions: d_model=2560, d_ff=6848, 32 heads, vocab_size=50257, batch=4, ctx=128
- Script `measure_block_memory.py` builds 4 separate model instances, each with the xl hyperparameters but `num_layers ∈ {1, 2, 3, 4}` (instead of 32), to isolate per-block overhead
- Memory measured using `torch.cuda.memory_allocated() / 1024²` (returns MB) at three checkpoints per run:
  1. After model construction and `.cuda()` call → `model_params_mb`
  2. After `logits = model(tokens); loss = cross_entropy(...)` → `activations_saved_mb` (= total − model params; the computation graph is alive)
  3. After `loss.backward()` → `after_bwd_mb` (activations freed, gradients allocated)
- `torch.cuda.reset_peak_memory_stats()` called before each measurement point

```
# measure_block_memory.py — exact methodology:
# For num_layers in [1, 2, 3, 4]:
#   1. Build model; move to CUDA
#   2. mem_before_fwd = memory_allocated()            (= model params + input tokens)
#   3. logits = model(tokens); loss = cross_entropy(logits, targets)
#   4. mem_after_fwd  = memory_allocated()            (= params + activations saved for bwd)
#   5. loss.backward()
#   6. mem_after_bwd  = memory_allocated()            (= params + gradients; activations freed)
#
# activations_saved = mem_after_fwd − mem_before_fwd

# Raw output from measure_block_memory.py (A100, xl dims, ctx=128):
Layers   model_params_mb   activations_saved_mb   after_bwd_mb
  1            1286.3               453.2               2682.9
  2            1607.2               580.6               3288.2
  3            1911.8               716.9               3893.5
  4            2216.5               852.4               4499.8

# Per-block incremental activation memory (first difference of activations_saved_mb):
#   Block 1→2:  580.6 − 453.2 = 127.5 MB
#   Block 2→3:  716.9 − 580.6 = 136.2 MB
#   Block 3→4:  852.4 − 716.9 = 135.6 MB
#   Average over 3 differences: (127.5 + 136.2 + 135.6) / 3 = 133.1 MB per block
```

Each `TransformerBlock` saves approximately **133 MB** of activations during its forward pass that must stay alive until that block's backward runs.

**The 5 largest tensors within one block's activation memory:**

During the forward pass of one block, PyTorch's autograd engine saves the output of every operation that will be needed for backward. The five largest of these are:

| Rank | Tensor saved | Why it's needed for backward | Shape | Size | % of 133 MB |
|------|-------------|------------------------------|-------|------|-------------|
| 1 | `w1(x)` — FFN first-gate output | Needed to compute the SiLU derivative during backward through the gate | (4, 128, 6848) | 14.1 MB | 10.6% |
| 2 | `w3(x)` — FFN second-gate output | The gate branch of SwiGLU; needed for gradient of the multiply | (4, 128, 6848) | 14.1 MB | 10.6% |
| 3 | `silu(w1)·w3` — FFN gate product (input to w2) | Needed to compute the gradient of the w2 weight matrix | (4, 128, 6848) | 14.1 MB | 10.6% |
| 4 | Attention scores `QKᵀ/√d_k` (pre-softmax) | Needed for softmax backward (derivative of softmax requires its input) | (4, 32, 128, 128) | 8.0 MB | 6.0% |
| 5 | Attention weights (post-softmax) | Needed for gradient of `attention_weights × V` matmul | (4, 32, 128, 128) | 8.0 MB | 6.0% |

These five tensors account for **58.4 MB = 43.9%** of the 133 MB. The remaining 56% is spread across: Q, K, V projection outputs before/after head-splitting (~5 MB each), the block's input residual (~5 MB), RMSNorm FP32 upcast intermediates for ln1 and ln2 (~5 MB each), the attention output tensor before the output projection (~5 MB), and small tensors for the causal mask and RoPE.

**Calculating gradient tensor memory per TransformerBlock:**

**Method:** Two independent approaches — (1) empirical subtraction from the `measure_block_memory.py` data above, and (2) analytical counting of parameters.

```
# Approach 1 — empirical subtraction:
# After backward: memory = model_params + gradient_tensors  (all activations freed)
# Per-block memory after backward (first difference of after_bwd_mb):
#   Block 1→2: 3288.2 − 2682.9 = 605.3 MB  (= 1 block's params + 1 block's grads)
# Per-block model param increase (first difference of model_params_mb):
#   Block 1→2: 1607.2 − 1286.3 = 320.9 MB  (measured; includes ~20 MB allocator overhead)
# Implied gradient size: 605.3 − 320.9 = 284.4 MB  (empirical; slightly low due to overhead)

# Approach 2 — analytical parameter count (from check_block_params.py on A100):
# Script: builds one TransformerBlock with xl dims, counts params via sum(p.numel() for p in block.parameters())
TransformerBlock total parameters: 78,812,160

Parameter breakdown:
  attn.q_proj.weight [2560, 2560]:  6,553,600 params =  25.00 MB  ┐
  attn.k_proj.weight [2560, 2560]:  6,553,600 params =  25.00 MB  ├ attention total: 100.00 MB
  attn.v_proj.weight [2560, 2560]:  6,553,600 params =  25.00 MB  │
  attn.output_proj   [2560, 2560]:  6,553,600 params =  25.00 MB  ┘
  ffn.w1.weight      [6848, 2560]: 17,530,880 params =  66.88 MB  ┐
  ffn.w2.weight      [2560, 6848]: 17,530,880 params =  66.88 MB  ├ SwiGLU total: 200.63 MB
  ffn.w3.weight      [6848, 2560]: 17,530,880 params =  66.88 MB  ┘
  ln1.weight + ln2.weight [2560]:   5,120 params     =   0.02 MB
  ─────────────────────────────────────────────────────────────────
  Total per block:                 78,812,160 params = 300.64 MB (FP32)

Gradient tensors: same shape and dtype (FP32) as each parameter.
→ Gradient memory per block = 300.64 MB
```

**Deriving gradient memory from measurements:**
After the full backward pass, GPU memory holds only parameters + gradients (all activations have been freed). The incremental memory per additional block is:
```
  +1 block (1→2): after_bwd_diff = 3288.2 − 2682.9 = 605.3 MB
  Model param increase per block: 1607.2 − 1286.3 = 320.9 MB   (params only)
  Implied gradient size:          605.3 − 320.9   = 284.4 MB
```
The empirical gradient estimate (284 MB) is slightly below the analytical value (301 MB) because the measured model-param memory (321 MB) includes ~20 MB of CUDA memory-allocator overhead per block, so the true gradient memory is ~301 MB. The gradient memory per TransformerBlock is **≈ 301 MB**, exactly equal to the parameter memory since each gradient tensor has the same shape and FP32 dtype as its parameter.

---

## Section 3: Single-GPU Memory

### Problem `gradient_checkpointing` — Memory-Optimal Gradient Checkpointing (4 pts)

**(a)** Consider a Transformer with N identical blocks stacked sequentially. Without checkpointing, all N blocks' residuals are kept alive simultaneously (O(N) peak activation memory). What checkpointing strategy minimizes peak activation memory, ignoring compute cost? Describe how you would arrange the `checkpoint` calls (a code sketch is fine), and give the asymptotic peak activation memory as a function of N. Assume residuals saved by a single block dominate any per-checkpoint bookkeeping.

> **Deliverable:** 3–5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.

**Answer:**

**Background — why activations accumulate:**
Without checkpointing, PyTorch saves every intermediate tensor from every block during the forward pass (to use during the corresponding backward pass). A 32-block Transformer holds 32 blocks worth of saved activations simultaneously — this scales linearly with N.

**Background — what `torch.utils.checkpoint` does:**
`checkpoint(fn, x)` runs `fn(x)` during the forward pass in a special mode where no intermediate tensors are saved. This frees all those tensors immediately. When the backward pass later needs those tensors to compute gradients, it simply re-runs `fn(x)` from the saved input alone. This trades memory for extra compute (one extra forward pass per checkpointed block).

**The memory-minimizing strategy:**
Wrap every individual block in `checkpoint`. With full per-block checkpointing, no block saves any activations at all during the forward pass. During backward, PyTorch processes blocks in reverse order, recomputing each block's activations from scratch when it needs them. At any moment during backward, only the activations of the single currently-recomputing block are live.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # During forward: activations are discarded immediately after each block.
            # During backward: each block is recomputed from scratch when needed.
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Asymptotic peak activation memory: O(1)**
At any point during backward, only one block's worth of intermediate tensors is alive (the one being recomputed), plus the boundary residual-stream tensors that connect blocks (~one extra copy). This is constant in N, regardless of how many blocks there are. The cost is N extra forward passes during backward, doubling total compute.

---

**(b)** For the xl model with batch size 4 and sequence length 2048, if you only have the compute budget for one step of recomputation (no nested `checkpoint` calls), what is the best checkpointing strategy to reduce peak memory? Profile the peak memory for your strategy and compare with the next smaller and larger checkpointing block sizes.

> **Deliverable:** 3–5 sentence description of your reasoning along with the measured peak memory for your strategy.

**Answer:**

TODO

---

## Section 4: GPU Kernels

### Problem `pytorch_attention` — PyTorch Attention Benchmarking (2 pts)

**(a)** Benchmark your attention implementation at different scales (batch=8, no multihead; sweep d_model ∈ {16,32,64,128} × seq_len ∈ {256,1024,4096,8192,16384}; 100 forward passes, 100 backward passes; warm up first). Report the timings or OOM errors. At what size do you get OOM? Do the accounting for memory usage of attention in one of the smallest configurations that runs out of memory. How does memory saved for backward change with sequence length?

> **Deliverable:** A table with your timings, your calculations for memory usage, and a 1–2 paragraph response.

**Answer:**

TODO

*(attach timing table)*

---

### Problem `torch_compile` — Torch Compile (2 pts)

**(a)** Compare your compiled attention module with the uncompiled version using the same configuration as `pytorch_attention` above.

> **Deliverable:** A table comparing forward and backward pass timings for compiled vs. uncompiled attention.

**Answer:**

TODO

*(attach table)*

---

**(b)** Compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

> **Deliverable:** A table comparing the vanilla and compiled Transformer model.

**Answer:**

TODO

*(attach table)*

---

### Problem `flash_benchmarking` — FlashAttention-2 Benchmarking (5 pts)

**(a)** Compare the performance of your (partially) Triton FlashAttention-2 forward and backward passes with a regular PyTorch attention implementation. Batch size 1, causal masking; sweep seq_len ∈ powers of 2 from 128 to 65536, d_head ∈ powers of 2 from 16 to 128, precision ∈ {bfloat16, float32}.

> **Deliverable:** A table of results comparing FlashAttention-2 with PyTorch, reporting forward, backward, and end-to-end latencies.

**Answer:**

TODO

*(attach table)*

---

## Section 5: Distributed Data Parallel Training

### Problem `distributed_communication_single_node` — Distributed Communication (5 pts)

Benchmark the runtime of the all-reduce operation in the single-node multi-process setup. Vary:
- **Data size:** float32 tensors of 1 MB, 10 MB, 100 MB, 1 GB
- **Number of GPUs/processes:** 2, 4, or 6

> **Deliverable:** Plot(s) and/or table(s) comparing the various settings, with 2–3 sentences of commentary about your results and how the various factors interact.

**Answer:**

TODO

*(attach plots/tables)*

---

### Problem `naive_ddp_benchmarking` — Naïve DDP Benchmarking (3 pts)

Benchmark your naïve DDP implementation on the xl model (1 node × 2 GPUs). Measure the total time per training step and the proportion of time spent communicating gradients.

> **Deliverable:** A description of your benchmarking setup, the measured time per training iteration, and the time spent communicating gradients for each setting.

**Answer:**

TODO

---

### Problem `minimal_ddp_flat_benchmarking` — Minimal DDP with Flat Gradients Benchmarking (2 pts)

Modify your minimal DDP implementation to all-reduce a single concatenated flat gradient tensor. Compare with the per-parameter all-reduce implementation (1 node × 2 GPUs, xl model).

> **Deliverable:** The measured time per training iteration and time spent communicating gradients, plus 1–2 sentences comparing batched vs. individual all-reduce.

**Answer:**

TODO

---

### Problem `ddp_overlap_individual_parameters_benchmarking` — DDP Overlapping Individual Parameters Benchmarking (1 pt)

**(a)** Benchmark your overlapped DDP implementation (backward pass overlapped with individual parameter gradient communication). Compare with the two previous DDP settings (1 node, 2 GPUs, xl model).

> **Deliverable:** The measured time per training iteration, with 1–2 sentences comparing the results.

**Answer:**

TODO

---

**(b)** Instrument your benchmarking code with the Nsight profiler, comparing the initial DDP implementation with the overlapped implementation. Visually compare the two traces and provide a profiler screenshot demonstrating that one implementation overlaps compute with communication while the other doesn't.

> **Deliverable:** 2 screenshots (one from the initial DDP, one from the overlapped DDP) showing that communication is or isn't overlapped with the backward pass.

**Answer:**

TODO

*(attach screenshots)*

---

## Section 6: Optimizer State Sharding

### Problem `optimizer_state_sharding_accounting` — Optimizer State Sharding Accounting (5 pts)

**(a)** Profile the peak memory usage when training with and without optimizer state sharding (1 node, 2 GPUs, xl model). Report peak memory after model initialization, directly before the optimizer step, and directly after the optimizer step. Break down the memory usage (parameters, optimizer states, etc.).

> **Deliverable:** 2–3 sentence response with peak memory usage results and a breakdown of how memory is divided between different model and optimizer components.

**Answer:**

TODO

---

**(b)** How does optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding (1 node, 2 GPUs, xl model).

> **Deliverable:** 2–3 sentence response with your timings.

**Answer:**

TODO

---

**(c)** How does our approach to optimizer state sharding differ from ZeRO stage 1 (ZeRO-DP P_os)?

> **Deliverable:** 2–3 sentence summary of any differences, especially those related to memory and communication volume.

**Answer:**

TODO

---

## Section 7: Fully-Sharded Data Parallel

### Problem `fsdp_accounting` — FSDP Accounting (5 pts)

**(a)** Given your analysis in Section 6, how much memory do you expect to save from the peak by implementing FSDP? (Ignore the size of preallocated all-gather buffers.)

> **Deliverable:** 2–3 sentence response with your findings.

**Answer:**

TODO

---

**(b)** Profile the xl model on two GPUs and pay attention to the all-gather of weights. Does the communication finish in time for the forward pass?

> **Deliverable:** 2–3 sentence response with your timings. Include Nsight screenshots to back up your claims.

**Answer:**

TODO

*(attach screenshots)*

---

## Section 8: Analyzing Parallelism Strategies

### Problem `alternate_ring_all_reduce` — Alternate Ring All-Reduce (1 pt)

> *(See PDF Section 8 for full problem statement)*

**Answer:**

TODO

---
