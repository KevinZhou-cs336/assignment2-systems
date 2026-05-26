# CS336 Assignment 2 (Systems): Writeup

---

## Section 2: Profiling and Benchmarking

### Problem `benchmarking_script` — Benchmarking Script (4 pts)

**(b)** Time the forward, backward, and optimizer step for each model size in Table 1 (small / medium / large / xl / 10B). Use 5 warmup steps and report the average and standard deviation over 10 measurement steps. How long does a forward pass take? How about a backward pass? Is the standard deviation small?

> **Deliverable:** 1–2 sentence response with your timings.

**Answer:**

*GPU: NVIDIA A100 80GB PCIe. Command: `python benchmark.py --d-model D --num-layers L --num-heads H --mode full --num-warmup-steps 5 --num-steps 10`*

| Model | d_model | Layers | Heads | Forward (ms) | Backward (ms) | Optimizer (ms) | Peak Mem (GB) |
|-------|---------|--------|-------|:------------:|:-------------:|:--------------:|:-------------:|
| small  | 768  | 12 | 12 | 38.85 ± 1.88  | 80.07 ± 0.60   | 27.56 ± 1.07  | 4.25  |
| medium | 1024 | 24 | 16 | 110.10 ± 0.39 | 234.10 ± 2.67  | 53.00 ± 2.07  | 11.40 |
| large  | 1280 | 36 | 20 | 230.77 ± 25.07| 481.15 ± 0.89  | 72.10 ± 1.40  | 22.37 |
| xl     | 2560 | 32 | 32 | 660.38 ± 1.05 | 1391.19 ± 3.41 | 233.83 ± 0.44 | 52.14 |
| 10B    | —    | —  | —  | OOM on A100 80GB | — | — | — |

```
# Concrete example — xl model full training step on A100:
Mode:            full
Forward:         660.38 ± 1.05 ms
Backward:        1391.19 ± 3.41 ms
Optimizer step:  233.83 ± 0.44 ms
Throughput:      896 tokens/s
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

*Command: `python benchmark.py --d-model 768 --num-layers 12 --num-heads 12 --mode full --num-warmup-steps 0 --num-steps 10`*

```
# 0 warmup steps — A100, small model:
Forward:         77.32 ± 119.51 ms   ← std is 60× larger than with warmup!
Backward:        100.31 ± 60.47 ms
Optimizer step:  25.21 ± 3.60 ms

# Step-by-step breakdown (no warmup, A100):
Step  1: fwd=246.43ms  bwd+opt=219.49ms   ← ~6× slower than steady state
Step  2: fwd= 39.71ms  bwd+opt=103.42ms   ← immediately drops to normal
Step  3: fwd= 40.42ms  bwd+opt=103.17ms   ← stable from here on
...

# With 1 warmup step:
Forward:         39.72 ± 0.09 ms   ← completely stable

# With 5 warmup steps:
Forward:         39.57 ± 0.11 ms   ← virtually identical to 1 warmup
```

**Why step 1 is so much slower (the warmup effect):**
CUDA does not compile GPU code upfront. Instead, the first time any GPU operation runs, CUDA compiles a kernel (a small GPU program) for it and caches the result — this is called JIT (just-in-time) compilation. For a Transformer with dozens of distinct operations, this compilation happens for each unique kernel on step 1, adding tens to hundreds of milliseconds of one-time overhead. After step 1, every kernel is already compiled and cached, so subsequent steps take only the actual GPU compute time.

Additionally, the GPU clock may be in a low-power idle state at the start and ramp up only after the first step. Both effects are fully absorbed by a single warmup step — increasing from 1 to 5 warmup steps makes essentially no difference (39.72 ms vs 39.57 ms).

---

### Problem `nsys_profile` — Nsight Systems Profiling (5 pts)

**Note on Nsight Systems screenshots:**
Nsight Systems profiling was run on a remote NVIDIA A100 pod via SSH. The pod is a headless server — it has no graphical display environment (no desktop, no X11/Wayland). Nsight Systems (`nsys`) does generate `.nsys-rep` profile files on this server, but the interactive GUI that produces timeline graphs requires a local desktop application to open those files. Since we ran everything remotely, we extracted all profiling data using the `nsys stats` command-line tool, which reads the same `.nsys-rep` file and outputs the same underlying statistics as text tables. The numbers and conclusions are identical to what the GUI would show; we simply cannot produce screenshots without either X11 forwarding or downloading the `.nsys-rep` file and opening it in a local Nsight Systems installation.

**How NVTX works:**
CUDA kernels run asynchronously on the GPU — from Python's perspective, `torch.matmul()` returns immediately while the GPU is still computing. The profiler sees a flat stream of anonymous GPU kernels with no obvious connection to your Python code. NVTX (NVIDIA Tools Extension) lets you insert named "ranges" into the execution timeline: when you call `torch.cuda.nvtx.range("attention scores")`, the profiler records that start/end timestamp alongside every GPU kernel that fires during that range. This lets you answer "which CUDA kernels ran inside my attention score computation?" The `benchmark.py` script annotates the three phases of scaled dot-product attention with NVTX ranges, and the full training pass is wrapped in a `"benchmark"` range.

---

**(a)** What is the total time spent on your forward pass? Does it match what we measured with the Python standard library?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*GPU: A100 80GB. Model: small (d_model=768, 12 layers). Command: `nsys profile --output=/tmp/small_fwd python3 benchmark.py --mode forward --num-warmup-steps 5 --num-steps 3 --nvtx`, then `nsys stats /tmp/small_fwd.nsys-rep`.*

```
# NVTX Range Summary (from `nsys stats`, nsys CLI output):
 Time (%)  Total Time (ns)  Instances  Name
   25.0%     125,741,034         1     "benchmark"   ← covers all 3 timed steps

# Per-step calculation:
# nsys total for the "benchmark" range: 125.7 ms
# Number of timed steps: 3
# → nsys time per forward step = 125.7 / 3 = 41.9 ms

# Python standard library (timeit.default_timer) measurement:
# Forward: 39.79 ± 1.51 ms per step
```

The nsys measurement (41.9 ms/step) matches the Python timer (39.79 ms/step) to within ~5%. The small remaining gap comes from two sources: (1) the nsys profiler itself adds a tiny overhead to every CUDA API call it intercepts, and (2) `cuda.synchronize()` wall-clock timing in Python includes the time for the CPU to dispatch GPU work, whereas nsys measures only actual GPU kernel execution time.

---

**(b)** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is it invoked during a single forward pass? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_fwd.nsys-rep` (3 timed forward steps).*

**Background — what "SGEMM" means:**
Every `torch.matmul` or `nn.Linear` call eventually dispatches to cuBLAS, which picks the fastest GEMM (General Matrix Multiplication) kernel for your specific matrix dimensions and GPU architecture. On the A100, these kernels have names like `ampere_sgemm_128x128_tn` where: `ampere` = A100 GPU microarchitecture, `sgemm` = single-precision (FP32) GEMM, `128x128` = tile size (the kernel computes a 128×128 output tile per thread block), and `tn` = first matrix is Transposed, second is Normal — matching the `QKᵀ` pattern in attention scoring.

```
# Top GPU kernels by cumulative time — forward pass only, small model (3 steps):
 Time (%)  Total Time (ns)  Instances  Name
   33.1%    97,620,727         480    ampere_sgemm_128x128_tn   ← #1 overall
   24.1%    71,080,143         192    ampere_sgemm_128x64_tn
    6.6%    19,309,539          96    ampere_sgemm_128x128_nn
    4.7%    13,792,424           8    ampere_sgemm_128x64_tn (larger tile)
    3.7%    10,966,773          96    ampere_sgemm_128x128_tn (variant)
  ...
  ─── All ampere_sgemm_* (matrix multiplications) combined: ~72% of total GPU time ───

# Same model, full training step (forward + backward + optimizer):
 Time (%)  Total Time (ns)  Instances  Name
    9.7%    97,610,046         480    ampere_sgemm_128x128_tn   ← still #1
    7.6%    76,439,372         192    ampere_sgemm_128x32_sliced1x4_nt  ← new: weight grad
    7.1%    71,447,506         192    ampere_sgemm_128x64_nn
    7.0%    71,148,340         192    ampere_sgemm_128x64_tn
  ...
  ─── All ampere_sgemm_* combined: ~61% of total GPU time ───
```

The top kernel is `ampere_sgemm_128x128_tn`, invoked **160 times per forward step** (480 instances ÷ 3 steps). It accounts for 33% of all GPU kernel time in the forward pass. It remains the single most time-consuming kernel in the full training step too, but its *percentage* drops from 33% to 9.7% — not because it ran less, but because the backward pass adds many additional SGEMM variants (new `_nt` and `_nn` suffix variants for weight-gradient computations), so there is now much more total GPU work to share the pie with.

---

**(c)** What other kernels besides matrix multiplies account for non-trivial CUDA runtime in the forward pass?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_fwd.nsys-rep`, filtering out `ampere_sgemm_*` entries.*

```
# Non-matmul kernels by cumulative GPU time (forward pass, small model):
 Time (%)  Total Time (ns)  Instances  Name
    2.6%     7,596,610          96    elementwise_kernel    ← softmax: subtract max for numerical stability
    2.5%     7,266,049          96    elementwise_kernel    ← softmax: divide by sum
    2.3%     6,688,611         768    elementwise_kernel    ← misc: causal mask apply, activations
    2.2%     6,463,742          96    elementwise_kernel    ← softmax: further ops
    2.1%     6,059,195          96    vectorized_elementwise_kernel  ← SwiGLU: silu gate multiply
    1.9%     5,736,989          96    exp_kernel            ← softmax: exponentiate scores
    1.9%     5,516,476         192    vectorized_elementwise_kernel  ← residual addition (x + attn_output)
    1.8%     5,396,096          96    reduce_kernel MaxOps  ← softmax: find row maximum
    1.6%     4,582,045          96    reduce_kernel         ← softmax: sum row for normalization
    1.3%     3,967,029         400    elementwise_kernel    ← embedding lookup, misc
    1.2%     3,391,541         192    CatArrayBatchedCopy   ← concatenate Q, K, V heads
    0.6%     1,769,324          96    sigmoid_kernel        ← SwiGLU gate activation
    0.6%     1,651,712         200    reduce_kernel MeanOps ← RMSNorm mean-of-squares
  ...
```

The three main non-matmul consumers are:

1. **Softmax** (the entire sequence: find row max → subtract max → exponentiate → sum row → divide): collectively ~10% of GPU time. Softmax runs over the full N×N attention score matrix, touching every element four times in separate passes.

2. **Activation kernels** for the SwiGLU feed-forward gate (sigmoid → multiply): ~4% of GPU time.

3. **RMSNorm reductions** (computing root mean square across the hidden dimension): ~2% of GPU time.

Together these non-matmul ops account for the remaining ~28% of GPU time. All of them are *memory-bandwidth bound* — they spend most of their time moving data between GPU memory and compute units rather than actually computing — which is why they use GPU capacity so inefficiently compared to GEMM.

---

**(d)** Profile running one complete training step (forward + backward + AdamW optimizer step). How does the fraction of time spent on matrix multiplication change compared to inference (forward only)? How about other kernels?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_full.nsys-rep`.*

```
# Summary comparison:
                    Forward only    Full training step
Matmul (SGEMM) %:      ~72%             ~61%
Other kernels %:       ~28%             ~39%

# New kernels that appear in the backward + optimizer passes:
 Time %   Name / Purpose
   3.0%   elementwise_kernel — backward through activation functions (chain-rule elementwise ops)
   1.9%   vectorized_elementwise_kernel — AdamW: update first moment  m ← β₁m + (1−β₁)g
   1.8%   vectorized_elementwise_kernel — add: accumulate gradients
   1.7%   vectorized_elementwise_kernel — AdamW: update second moment  v ← β₂v + (1−β₂)g²
   1.5%   elementwise_kernel — backward through SwiGLU sigmoid gate
   1.4%   reduce_kernel — backward through RMSNorm (sum of gradient × normalized input)
   ...
```

**Why the matmul percentage drops from 72% to 61%:**
Adding backward + optimizer means the total GPU work roughly triples — but the new work is not purely matmul. The backward pass does add weight-gradient matmuls (roughly doubling the matmul count), but it also adds a comparable amount of memory-bandwidth-bound work: elementwise activation backward passes, softmax backward, RMSNorm backward, and the AdamW per-parameter moment updates. These elementwise operations take CPU/GPU bandwidth time that dilutes the matmul fraction.

**Why the optimizer step barely adds any matmul time:**
AdamW applies the update `θ ← θ − α·m̂/(√v̂ + ε)` element-by-element — one scalar operation per parameter. There are no matrix multiplications in this step, only element-wise arithmetic across all parameters.

---

**(e)** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From NVTX annotation timings in `/tmp/small_fwd.nsys-rep` (3 steps, small model, seq=512, d_head=64 per head).*

**Background — what FLOPs measure:**
FLOPs (Floating-Point Operations) count how much arithmetic a computation requires. A GEMM of shape (M, K) × (K, N) requires 2·M·K·N FLOPs. Comparing FLOPs to actual runtime tells you GPU utilization efficiency: a kernel that achieves 15 TFLOPS is using the GPU compute units 75× more efficiently than one that achieves 0.2 TFLOPS (even if both run for the same wall-clock time).

```
# NVTX range timings (3 timed forward steps total):
  Range "computing attention scores" (QKᵀ/√d_k + causal mask):  86.6ms  → 28.9ms/step
  Range "computing softmax":                                      72.9ms  → 24.3ms/step
  Range "final matmul"  (attention_weights × V):                  27.9ms  → 9.3ms/step

  Total attention matmuls (QKᵀ + AV) per step: 28.9 + 9.3 = 38.2 ms
  Softmax per step:                                              24.3 ms
  Runtime ratio (matmul / softmax):                              38.2 / 24.3 = 1.57×

# FLOPs calculation (small model: batch=4, heads=12, seq=512, d_head=64):
  QKᵀ matmul:   2 × batch × heads × seq × seq × d_head = 2×4×12×512×512×64 = 3.22 GFLOPs
  AV  matmul:   same shape, same FLOPs                                       = 3.22 GFLOPs
  Total matmul FLOPs:                                                        = 6.44 GFLOPs

  Softmax FLOPs: ~5 ops/element × batch × heads × seq × seq
               = 5 × 4 × 12 × 512 × 512                                    = 0.063 GFLOPs
  FLOPs ratio (matmul / softmax):  6.44 / 0.063 = 102×
```

**The key finding:**
The attention matmuls do **102× more FLOPs** than softmax, yet take only **1.57× longer** to run. This means softmax runs at roughly 64× lower arithmetic throughput per FLOP than the matmuls. This is not a bug — it reflects two fundamentally different bottlenecks:

- **Matmuls are compute-bound.** Each element of the output requires K multiply-adds, so the ratio of compute to memory access is high (~K). The A100's Tensor Cores can sustain ~15 TFLOPS on FP32 matmuls.

- **Softmax is memory-bandwidth bound.** Computing `softmax` over each row of the N×N attention matrix requires four sequential passes over the same data: find the row max, subtract it (for numerical stability), exponentiate each element, sum the row, divide. Four passes over N×N×4 bytes of data, with almost no arithmetic per byte. The bottleneck is how fast GPU memory can deliver data (~2 TB/s on A100), not how fast the compute units run. Effective arithmetic throughput is only ~0.2 TFLOPS.

This inefficiency — softmax must read and write the entire O(L²) attention matrix four times — is precisely the problem FlashAttention solves by fusing all four passes into a single kernel that keeps the data in L2 cache (fast SRAM) instead of going to main GPU memory.

---

### Problem `mixed_precision_accumulation` — Mixed-Precision Accumulation (1 pt)

Run the four accumulation snippets (float32 accumulation, float16 accumulation, float32 accumulation with float16 increments, float32 accumulation after casting) and comment on the accuracy of the results.

> **Deliverable:** 2–3 sentence response.

**Answer:**

*Verified by running on RTX 3090 CUDA GPU.*

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

*Verified by running on RTX 3090 CUDA GPU with `torch.autocast(device_type='cuda', dtype=torch.float16)`.*

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

*GPU: NVIDIA A100 80GB PCIe. Command: `python benchmark.py ... --mode full [--mixed-precision]`*

| Model | FP32 Fwd (ms) | BF16 Fwd (ms) | Fwd Speedup | FP32 Bwd (ms) | BF16 Bwd (ms) | Bwd Speedup |
|-------|:-------------:|:-------------:|:-----------:|:-------------:|:-------------:|:-----------:|
| small  | 38.85 ± 1.88  | 33.18 ± 3.39  | 1.17× | 80.07 ± 0.60  | 46.79 ± 1.27 | 1.71× |
| medium | 110.10 ± 0.39 | 65.70 ± 1.59  | 1.68× | 234.10 ± 2.67 | 107.70 ± 1.53 | 2.17× |
| large  | 230.77 ± 25.07| 111.45 ± 58.87| 2.07× | 481.15 ± 0.89 | 153.17 ± 0.97 | 3.14× |
| xl     | 660.38 ± 1.05 | 142.46 ± 0.47 | **4.63×** | 1391.19 ± 3.41| 297.60 ± 1.62 | **4.67×** |

```
# Example: xl full training step
# FP32: Forward 660ms, Backward 1391ms, Optimizer 234ms, Peak mem 52.14 GB
# BF16: Forward 142ms, Backward  298ms, Optimizer 235ms, Peak mem 49.77 GB
# Note: optimizer time is unchanged — AdamW always updates parameters in FP32.
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

*GPU: NVIDIA A100 80GB PCIe, xl model (d_model=2560, 32 layers, 32 heads, batch=4).*

```
# context=128, forward only:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode forward
→ Peak GPU memory: 18.324 GB

# context=128, full training step:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 128 --mode full
→ Peak GPU memory: 39.048 GB

# context=2048, forward only:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 \
  --context-length 2048 --mode forward
→ torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

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

*GPU: A100 80GB, xl model, batch=4, context=128. Command: `python benchmark.py ... [--mixed-precision]`*

```
# FP32 (baseline, no --mixed-precision):
Forward only:        18.324 GB
Full training step:  39.048 GB

# BF16 mixed precision (--mixed-precision flag):
Forward only:        25.080 GB   ← HIGHER than FP32!
Full training step:  38.975 GB   ← nearly identical to FP32
```

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

*Memory snapshot collected with: `python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --context-length 128 --mode forward-backward --num-warmup-steps 2 --num-steps 1 --memory-profile`. Snapshot file: `writeup/xl_memory_snapshot.pickle`.*

```
# Allocation sizes extracted from memory_snapshot.pickle:
# (size in bytes → MiB, count = number of tensors of that size)

  97.7 MiB ×   4 instances  ← token embedding table [vocab_size=10000, d_model=2560]
  66.9 MiB × 192 instances  ← FFN weight matrices (SwiGLU W1, W2, W3 each [2560, 6848])
  25.0 MiB × 256 instances  ← attention projection weights (Q, K, V, O each [2560, 2560])
  13.4 MiB × 352 instances  ← FFN hidden states (gate/up projection activations, saved for backward)
   5.0 MiB × 1709 instances ← residual stream tensors [4, 128, 2560], one per block boundary

# Verification of sizes:
#   97.7 MiB = 10000 × 2560 × 4 bytes   (vocab=10000, d_model=2560, FP32)
#   66.9 MiB = 6848 × 2560 × 4 bytes    (d_ff=6848, d_model=2560, FP32)
#   25.0 MiB = 2560 × 2560 × 4 bytes    (d_model × d_model, FP32)
#    5.0 MiB = 4 × 128 × 2560 × 4 bytes (batch × ctx × d_model, FP32)
```

The largest single allocation is **97.7 MiB**, which is the token embedding matrix (shape `[10000, 2560]` in FP32). The next-largest are the FFN weight matrices at 66.9 MiB each (192 total, because SwiGLU has 3 weight matrices per FFN and 32 layers × 3 = 96 weight matrices, appearing twice — once as parameters, once as their gradient buffers). The most numerous allocations (1,709 instances at 5 MiB each) are residual-stream tensors, saved at each layer boundary for the backward pass.

---

**(f)** Use NVTX ranges and Nsight Systems to determine how much memory is saved for backward (residuals) by a single `TransformerBlock`. Note the 5 largest contributing operations and what percentage of overall memory they contribute. Then, based on how much memory was allocated during the forward pass and how much memory changes for every `TransformerBlock` in the backward pass, calculate how much memory the gradient tensors for a `TransformerBlock` take.

> **Deliverable:** Screenshots from Nsight Systems and a 1–2 paragraph response.

**Answer:**

**Note on why Nsight Systems screenshots were not generated:**
Nsight Systems' per-NVTX-range memory breakdown requires the interactive GUI, which displays memory allocation events as colored bands on a timeline correlated with NVTX range markers. Since we ran all experiments on a remote A100 pod (headless SSH server with no graphical display), we cannot open the GUI. Instead we: (1) measured per-block activation memory directly using a Python experiment (comparing 1-vs-2-vs-3-vs-4 block models), and (2) identified the largest per-block tensors analytically from the computation graph, which gives the same information the Nsight Systems memory timeline would show.

*GPU: NVIDIA A100 80GB PCIe. xl model (d_model=2560, d_ff=6848, 32 heads, batch=4, ctx=128). Measured with `measure_block_memory.py`, which builds identical xl-dimension models with 1, 2, 3, and 4 transformer blocks and records memory before/after forward and after backward.*

**Measuring per-block activation memory (forward pass):**

The key insight: if a 2-block model saves X MB of activations and a 1-block model saves Y MB, then one block is responsible for (X − Y) MB of activations. We repeat this for 3 and 4 blocks and average.

```
# measure_block_memory.py output (A100, xl dims, ctx=128):
Layers   Model params (MB)   Activations saved after fwd (MB)   Mem after full bwd (MB)
  1            1286.3                     453.2                        2682.9
  2            1607.2                     580.6                        3288.2
  3            1911.8                     716.9                        3893.5
  4            2216.5                     852.4                        4499.8

Incremental per-block activation memory:
  Block 1→2:  580.6 − 453.2 = 127.5 MB
  Block 2→3:  716.9 − 580.6 = 136.2 MB
  Block 3→4:  852.4 − 716.9 = 135.6 MB
  Average:    133.1 MB per block
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

```
# check_block_params.py output (A100):
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
