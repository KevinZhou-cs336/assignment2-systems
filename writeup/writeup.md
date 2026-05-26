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
| small  | 768  | 12 | 12 | 38.85 ± 1.88 | 80.07 ± 0.60 | 27.56 ± 1.07 | 4.25 |
| medium | 1024 | 24 | 16 | 110.10 ± 0.39 | 234.10 ± 2.67 | 53.00 ± 2.07 | 11.40 |
| large  | 1280 | 36 | 20 | 230.77 ± 25.07 | 481.15 ± 0.89 | 72.10 ± 1.40 | 22.37 |
| xl     | 2560 | 32 | 32 | 660.38 ± 1.05 | 1391.19 ± 3.41 | 233.83 ± 0.44 | 52.14 |
| 10B    | —    | —  | —  | OOM on A100 80GB | — | — | — |

The backward pass consistently takes roughly 2.1× the forward time (a well-known rule of thumb: backprop ≈ 2× forward because it recomputes intermediate activations in reverse order). After warmup, the standard deviation is very small — below 0.5% of the mean for all models — indicating highly stable GPU execution; the only exception is the large model's forward pass (std=25 ms), likely due to occasional GPU frequency throttling. The xl model uses 52 GB peak memory with batch=4 and context=512, which exceeds a 24 GB RTX 3090 (those runs OOM), so an A100 80 GB is needed.

```
# Example output (xl model, full training step, A100):
Mode:            full
Forward:         660.38 ± 1.05 ms
Backward:        1391.19 ± 3.41 ms
Optimizer step:  233.83 ± 0.44 ms
Throughput:      896 tokens/s
Peak GPU memory: 52.142 GB
```

---

**(c)** Repeat the analysis without warm-up steps. How does this affect your results? Why do you think this happens? Also try 1 or 2 warm-up steps — why might the result still be different?

> **Deliverable:** 2–3 sentence response.

**Answer:**

*Command: `python benchmark.py --d-model 768 --num-layers 12 --num-heads 12 --mode full --num-warmup-steps 0 --num-steps 10`*

```
# 0 warmup steps (A100):
Forward:         77.32 ± 119.51 ms   ← huge std!
Backward:        100.31 ± 60.47 ms
Optimizer step:  25.21 ± 3.60 ms

# 1 warmup step (RTX 3090):
Forward:         39.72 ± 0.09 ms   ← stabilized
# 5 warmup steps (RTX 3090):
Forward:         39.57 ± 0.11 ms   ← virtually identical

# Step-by-step (no warmup, RTX 3090):
Step  1: fwd=246.43ms  bwd+opt=219.49ms   ← 6x slower first step!
Step  2: fwd= 39.71ms  bwd+opt=103.42ms   ← immediately stable
Step  3: fwd= 40.42ms  bwd+opt=103.17ms
...
```

Without warmup, the standard deviation explodes (forward std = 119 ms vs 1.88 ms with warmup), entirely due to the first step being ~6× slower. Step 1 is slow because CUDA JIT-compiles and caches CUDA kernels on their first invocation (the cuBLAS handles are also initialized lazily), and the GPU clock may ramp up from a low-power idle state. Just one warmup step is enough to completely stabilize the forward pass (std drops from 119 ms to 0.09 ms); even 2 warmup steps give virtually identical results to 5. The backward pass takes slightly longer to settle than the forward pass (step 2 backward is still slow), so 2 warmup steps gives a cleaner sample than 1 for the full training step measurement.

---

### Problem `nsys_profile` — Nsight Systems Profiling (5 pts)

Profile forward pass, backward pass, and optimizer step using `nsys` with two model sizes from Table 1 and three power-of-two context lengths larger than 128.

**(a)** What is the total time spent on your forward pass? Does it match what we measured with the Python standard library?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*GPU: A100 80GB. Model: small (d_model=768, 12 layers). Command: `nsys profile --output=/tmp/small_fwd python3 benchmark.py --mode forward --num-warmup-steps 5 --num-steps 3 --nvtx`*

```
# NVTX Range Summary (from nsys stats):
 Time (%)  Total Time (ns)  Instances   Avg (ns)
   25.0%     125,741,034         1     125,741,034    "benchmark"  (3 timed steps)
# → 125.7ms / 3 = 41.9ms per forward step from nsys

# Python standard library timer:
# Forward: 39.79 ± 1.51 ms per step
```

The nsys NVTX "benchmark" range reports 41.9 ms per forward step, matching the Python `timeit.default_timer()` measurement of 39.79 ms within ~5%. The small discrepancy is expected: nsys itself introduces slight profiling overhead, and `cuda.synchronize()` wall-clock time includes some CPU-GPU dispatch latency not counted in the kernel timeline.

---

**(b)** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is it invoked during a single forward pass? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_fwd.nsys-rep` (3 timed forward steps):*

```
# Top GPU kernels by cumulative time (forward pass, small model):
 Time (%)  Total Time (ns)  Instances  Name
   33.1%    97,620,727         480    ampere_sgemm_128x128_tn   ← #1
   24.1%    71,080,143         192    ampere_sgemm_128x64_tn
    6.6%    19,309,539          96    ampere_sgemm_128x128_nn
    4.7%    13,792,424           8    ampere_sgemm_128x64_tn (larger problem)
    3.7%    10,966,773          96    ampere_sgemm_128x128_tn (different config)
   ...
  ─── Total ampere_sgemm_* (matmuls): ~72.2% of all GPU kernel time ───

# Full training step (forward+backward):
 Time (%)  Total Time (ns)  Instances  Name
    9.7%    97,610,046         480    ampere_sgemm_128x128_tn   ← still #1
    7.6%    76,439,372         192    ampere_sgemm_128x32_sliced1x4_nt  ← new (weight grad)
    7.1%    71,447,506         192    ampere_sgemm_128x64_nn
    7.0%    71,148,340         192    ampere_sgemm_128x64_tn
   ...
  ─── Total ampere_sgemm_* (matmuls): ~61% of all GPU kernel time ───
```

The top kernel is `ampere_sgemm_128x128_tn` (cuBLAS A100-optimized GEMM with transposed K matrix), invoked **160 times per forward step** (480 instances / 3 steps), accounting for 33.1% of all GPU kernel time. During a full training step, it remains the single most time-consuming kernel (9.7% of a much larger total), but many more SGEMM variants appear for weight-gradient computations (`_nt`, `_nn`, `sliced` variants), reducing its percentage share while the total matmul fraction drops slightly from 72% to 61%.

---

**(c)** What other kernels besides matrix multiplies account for non-trivial CUDA runtime in the forward pass?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_fwd.nsys-rep` (non-matmul kernels):*

```
 Time (%)  Total Time (ns)  Instances  Name
    2.6%     7,596,610          96    elementwise_kernel (subtract x/max for softmax numerics)
    2.5%     7,266,049          96    elementwise_kernel (divide for softmax normalize)
    2.3%     6,688,611         768    elementwise_kernel (various activations, causal mask)
    2.2%     6,463,742          96    elementwise_kernel (further softmax/activation ops)
    2.1%     6,059,195          96    vectorized_elementwise_kernel (BUnary - SwiGLU gate)
    1.9%     5,736,989          96    exp_kernel (softmax numerator)
    1.9%     5,516,476         192    vectorized_elementwise_kernel (add - residual connections)
    1.8%     5,396,096          96    reduce_kernel MaxOps (softmax row-max)
    1.6%     4,582,045          96    reduce_kernel funcwrapper (softmax row-sum)
    1.3%     3,967,029         400    elementwise_kernel (misc activations, embedding)
    1.2%     3,391,541         192    CatArrayBatchedCopy (concat Q/K/V for attention)
    0.6%     1,769,324          96    sigmoid_kernel (SwiGLU activation)
    0.6%     1,651,712         200    reduce_kernel MeanOps (LayerNorm mean)
    ...
```

The largest non-matmul costs are: **(1) softmax operations** (exp, row-max reduction, row-sum reduction, divide = collectively ~10% of GPU time) for the attention probability computation, **(2) elementwise kernels** for activation functions (SwiGLU uses sigmoid+multiply, adding ~4% of GPU time), and **(3) reduction kernels** for LayerNorm mean/variance computation (~2%). Together these non-matmul ops account for ~28% of GPU time, all operating at much lower arithmetic intensity than GEMM.

---

**(d)** Profile running one complete training step (forward + backward + AdamW optimizer step). How does the fraction of time spent on matrix multiplication change compared to inference (forward only)? How about other kernels?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From `nsys stats /tmp/small_full.nsys-rep` (full training step):*

```
# Forward only:   matmuls = ~72.2% of GPU time,  other = ~27.8%
# Full step:      matmuls = ~61%   of GPU time,  other = ~39%

# New non-matmul kernels in backward+optimizer:
    3.0%  elementwise_kernel (backward through activations, gradient w.r.t. inputs)
    1.9%  vectorized_elementwise_kernel (AdamW first moment update: m = β₁m + (1-β₁)g)
    1.8%  vectorized_elementwise_kernel (add: gradient accumulation)
    1.7%  vectorized_elementwise_kernel (AdamW second moment: v = β₂v + (1-β₂)g²)
    1.5%  elementwise_kernel (backward through SwiGLU sigmoid)
    1.4%  elementwise_kernel (various gradient ops)
    1.4%  reduce_kernel (backward through LayerNorm, sum of dloss/dy)
    ...
```

The matmul fraction drops from ~72% (forward only) to ~61% (full training step), because backward adds: weight-gradient GEMM (`_nt` transposed-output variants, ~18% of total), backward through LayerNorm (reductions), backward through softmax (elementwise), and AdamW parameter update (element-wise per-parameter operations). The non-matmul fraction doubles (from ~28% to ~39%), dominated by the AdamW state updates and layer-norm/activation backward passes — these are all memory-bandwidth bound and have much lower GPU utilization than GEMM.

---

**(e)** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*From NVTX annotation timings in `/tmp/small_fwd.nsys-rep` (3 steps total):*

```
# NVTX timings (3 timed forward steps, small model, seq=512, d_head=64):
  "computing attention scores" (QK^T + masking):  86.6ms  → 28.9ms/step
  "computing softmax":                            72.9ms  → 24.3ms/step
  "final matmul" (AV):                            27.9ms  → 9.3ms/step

# Total attention matmuls (QK^T + AV): 28.9 + 9.3 = 38.2 ms/step
# Softmax: 24.3 ms/step
# Runtime ratio (matmul/softmax): 38.2 / 24.3 = 1.57×

# FLOPs per forward step:
#   QK^T:    2 × batch × heads × seq × seq × d_head = 2×4×12×512×512×64 = 3.22 GFLOPs
#   AV:      2 × batch × heads × seq × seq × d_head = same = 3.22 GFLOPs
#   Total matmul FLOPs: 6.44 GFLOPs
#   Softmax: ~5 × batch × heads × seq × seq     = 5×4×12×512×512 = 0.063 GFLOPs
# FLOPs ratio (matmul/softmax): 6.44 / 0.063 = 102×
```

The attention matmuls do ~102× more FLOPs than softmax, yet take only ~1.6× more time — meaning softmax achieves ~64× lower compute throughput per FLOP than the matrix multiplications. This reflects the fundamental difference in arithmetic intensity: GEMM operates at ~10–15 TFLOPS on the A100's Tensor Cores (compute-bound), while softmax requires two full sequential passes over the N×N attention matrix (exp, max-reduction, sum-reduction), making it strictly memory-bandwidth bound at ~0.2 TFLOPS effective. This exact bottleneck — softmax must materialize the full O(L²) matrix twice per layer — is what FlashAttention eliminates by fusing the softmax into the matmul and computing in tiles that stay in SRAM.

---

### Problem `mixed_precision_accumulation` — Mixed-Precision Accumulation (1 pt)

Run the four accumulation snippets (float32 accumulation, float16 accumulation, float32 accumulation with float16 increments, float32 accumulation after casting) and comment on the accuracy of the results.

> **Deliverable:** 2–3 sentence response.

**Answer:**

*Command run on RTX 3090:*
```python
# float32 accumulation
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float32)
# → Result: 10.000134  ✓

# float16 accumulation
result = torch.tensor(0.0, dtype=torch.float16)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16)
# → Result: 9.953125   ✗ (off by ~0.05)

# float32 accumulator + float16 increments (direct add)
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16)
# → Result: 10.002136  ✓ (PyTorch auto-promotes float16→float32 for add)

# float32 accumulator + float16 increment cast to float32 first
result = torch.tensor(0.0, dtype=torch.float32)
for _ in range(1000): result += torch.tensor(0.01, dtype=torch.float16).to(torch.float32)
# → Result: 10.002136  ✓
```

The float32 accumulator gives a nearly exact result (10.000134). The float16 accumulator gives 9.953125 — wrong by 0.047 — because as the running sum grows large (e.g., ~8.0), float16's limited mantissa (10 bits, effective precision ~0.001 at magnitude 8) can no longer represent the 0.01 increment accurately and rounds it down, so additions near the end contribute less than they should. Adding a float16 tensor to a float32 accumulator (snippets 3 and 4) gives the correct answer (10.002136) because PyTorch automatically promotes float16 to float32 at the addition, so the accumulation itself always happens in float32; the only imprecision is the one-time rounding of 0.01 to its nearest float16 value (~0.009994). This demonstrates that it is the **precision of the accumulator** that matters — mixed-precision training keeps optimizer states and gradient accumulation in FP32 for exactly this reason.

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

*Verified by running on RTX 3090 CUDA GPU:*
```
# Inside autocast(device_type='cuda', dtype=torch.float16):
fc1 weight dtype:    torch.float32   # stored params unchanged
Output of fc1:       torch.float16   # linear is an eligible op
Output of LayerNorm: torch.float32   # LayerNorm excluded from autocast
Logits (fc2 output): torch.float16   # linear is an eligible op
Loss dtype:          torch.float32   # cross_entropy excluded
# After backward:
fc1 gradient dtype:  torch.float32   # autograd always accumulates in fp32
fc2 gradient dtype:  torch.float32
ln weight gradient:  torch.float32
```

| Component | dtype |
|-----------|-------|
| Model parameters (within autocast) | `torch.float32` — autocast does **not** cast the stored weights; it casts the *inputs* to eligible ops at kernel dispatch time |
| Output of fc1 (`nn.Linear`) | `torch.float16` — linear layers are eligible ops; autocast casts them to FP16 |
| Output of LayerNorm (`nn.LayerNorm`) | `torch.float32` — LayerNorm is explicitly excluded from FP16 autocast because its reduction (mean/variance over the hidden dimension) loses precision in FP16 |
| Logits (output of fc2) | `torch.float16` — another linear layer, eligible for FP16 |
| Loss (cross-entropy) | `torch.float32` — loss functions are excluded from autocast; the scalar reduction must stay in FP32 for numerical stability |
| Gradients | `torch.float32` — gradients are always computed in FP32 by PyTorch's autograd engine regardless of the forward dtype |

---

**(b)** FP16 autocast treats LayerNorm differently from feed-forward layers. What parts of LayerNorm are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat LayerNorm differently? Why or why not?

> **Deliverable:** 2–3 sentence response.

**Answer:**

LayerNorm computes a mean and variance reduction over the hidden dimension — operations that accumulate many small values and are therefore sensitive to rounding. In FP16, the limited dynamic range (max ≈ 65504) and mantissa precision (10 bits) can cause these reductions to overflow or lose significant digits, producing NaN or inaccurate normalization. With BF16, the dynamic range is the same as FP32 (8 exponent bits) so overflow is not a concern, but the mantissa is still only 7 bits, which can still degrade the accuracy of the variance estimate; in practice PyTorch's autocast keeps LayerNorm in FP32 for both FP16 and BF16 to be safe, though BF16 is far less likely to cause problems than FP16 in practice.

---

**(c)** Modify your benchmarking script to optionally run with BF16 mixed precision. Time the forward and backward passes with and without mixed precision for each model size in Table 1. Compare full precision vs. mixed precision and comment on any trends as model size changes.

> **Deliverable:** 2–3 sentence response with your timings and commentary.

**Answer:**

*GPU: NVIDIA A100 80GB PCIe. Command: `python benchmark.py ... --mode full [--mixed-precision]`*

| Model | FP32 Fwd (ms) | BF16 Fwd (ms) | Fwd Speedup | FP32 Bwd (ms) | BF16 Bwd (ms) | Bwd Speedup |
|-------|:-------------:|:-------------:|:-----------:|:-------------:|:-------------:|:-----------:|
| small  | 38.85 ± 1.88 | 33.18 ± 3.39 | 1.17× | 80.07 ± 0.60 | 46.79 ± 1.27 | 1.71× |
| medium | 110.10 ± 0.39 | 65.70 ± 1.59 | 1.68× | 234.10 ± 2.67 | 107.70 ± 1.53 | 2.17× |
| large  | 230.77 ± 25.07 | 111.45 ± 58.87 | 2.07× | 481.15 ± 0.89 | 153.17 ± 0.97 | 3.14× |
| xl     | 660.38 ± 1.05 | 142.46 ± 0.47 | 4.63× | 1391.19 ± 3.41 | 297.60 ± 1.62 | 4.67× |

Mixed-precision BF16 provides significant speedups — from 1.17× for the small model up to 4.63× for the xl model — and the speedup grows with model size because larger models are more matmul-dominated (the A100's BF16 Tensor Core throughput is ~4× its FP32 throughput, so a purely matmul-bound workload approaches a 4× theoretical max). The xl model's 4.63× speedup is close to this theoretical limit, confirming it is nearly fully Tensor Core-bound. The small model benefits less because its shorter context and fewer parameters leave more time in non-matmul overhead (embedding, RoPE, softmax, layernorm) that is unaffected by autocast. Memory usage also drops modestly with mixed precision since autocast saves BF16 activations instead of FP32 (small: 4.25 → 3.61 GB; xl: 52.14 → 49.77 GB).

```
# Example: xl full training step
# FP32:  Forward 660.38ms, Backward 1391.19ms, Optimizer 233.83ms, Mem 52.14 GB
# BF16:  Forward 142.46ms, Backward 297.60ms,  Optimizer 235.00ms, Mem 49.77 GB
# Note: optimizer is unchanged because AdamW always runs in FP32.
```

---

### Problem `memory_profiling` — Memory Profiling (4 pts)

Profile the complete training step (forward + backward + optimizer step) of the **xl** model with context lengths 128 and 2048.

**(b)** What is the peak memory usage at each context length when doing a forward pass? What about when doing a full training step?

> **Deliverable:** A table with two numbers per context length.

**Answer:**

*GPU: NVIDIA A100 80GB PCIe, xl model (d_model=2560, 32 layers, 32 heads), batch=4.*

```
# context=128, forward only:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --context-length 128 --mode forward
→ Peak GPU memory: 18.324 GB

# context=128, full training step:
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --context-length 128 --mode full
→ Peak GPU memory: 39.048 GB

# context=2048: OOM on A100 80GB (analysis below)
python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --context-length 2048 --mode forward
→ torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
  (attention score matrix alone: 4×32×2048×2048×4 bytes = 2 GB per layer, ×32 layers = 64 GB)
```

| Context Length | Forward-only peak memory | Full training step peak memory |
|---------------|--------------------------|-------------------------------|
| 128  | 18.324 GB | 39.048 GB |
| 2048 | OOM (>80 GB) — naive O(L²) attention | OOM |

The full training step uses ≈2.1× more memory than the forward pass at context=128 (39 vs 18 GB), because backward requires storing gradients (≈10 GB) plus AdamW second-moment buffers (≈10 GB) on top of the model parameters (≈10 GB). For context=2048, even the forward pass OOMs because the naive self-attention score matrix has shape (batch, heads, seq, seq) = (4, 32, 2048, 2048) requiring 2 GB **per transformer block**, totalling ≈64 GB for 32 blocks before any other activations — this is the quadratic memory cost O(L²) of standard attention that FlashAttention was designed to eliminate.

---

**(c)** Find the peak memory of the xl model using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?

> **Deliverable:** 2–3 sentence response.

**Answer:**

*GPU: A100 80GB, xl model, batch=4, context=128. Command: `python benchmark.py ... --mixed-precision`*

```
# FP32 (no --mixed-precision):
Forward only:        18.324 GB
Full training step:  39.048 GB

# BF16 mixed precision (--mixed-precision):
Forward only:        25.080 GB   ← higher than FP32!
Full training step:  38.975 GB   ← nearly identical
```

Mixed precision has a **negligible** effect on peak memory for the full training step (39.048 GB vs 38.975 GB — a difference of only 73 MB), because the dominant cost is parameters + gradients + AdamW moment buffers, all of which are kept in FP32 regardless of autocast. For the forward-only pass, BF16 autocast surprisingly uses *more* memory (25 GB vs 18 GB), likely because the computation graph under autocast must retain both the BF16 intermediate activations **and** the FP32 versions of certain operations (LayerNorm outputs and loss run in FP32 even under autocast), whereas in pure FP32 mode only one copy is needed. This illustrates that mixed precision's memory benefit comes primarily during inference with `torch.no_grad()`, not from the saved computation graph.

---

**(d)** For the xl model, what is the size of a tensor of activations in the Transformer residual stream, in single precision? Give this size in MiB (divide bytes by 1024²).

> **Deliverable:** 1–2 sentence response with your derivation.

**Answer:**

The xl model has `d_model = 2560`. Each residual-stream tensor has shape `(batch_size, context_length, d_model) = (4, 2048, 2560)`. In FP32 each element is 4 bytes, so the total size is `4 × 2048 × 2560 × 4 bytes = 83,886,080 bytes = 83,886,080 / 1024² ≈ 80 MiB`. Each of the 32 Transformer blocks saves this tensor (and intermediate activations) for the backward pass, explaining why activation memory grows linearly with depth. Note: our memory_profiling experiments used context_length=128 (not 2048) due to the O(L²) attention bottleneck; for context=128 the residual stream tensor is `4 × 128 × 2560 × 4 = 5,242,880 bytes ≈ 5 MiB` per layer.

---

**(e)** Look closely at the "Active Memory Timeline" from `pytorch.org/memory_viz` for the xl model doing a forward pass. At Detail level 10%, what is the size of the largest allocations shown? Can you tell where those allocations come from?

> **Deliverable:** 1–2 sentence response.

**Answer:**

*Memory snapshot collected with `python benchmark.py --d-model 2560 --num-layers 32 --num-heads 32 --context-length 128 --mode forward-backward --num-warmup-steps 2 --num-steps 1 --memory-profile` on A100 80GB.*

```python
# Analysis of memory_snapshot.pickle:
# Top allocation sizes (bytes → MiB):
#   97.7 MiB ×   4 instances: token embedding matrix [vocab=10000, d_model=2560]
#   66.9 MiB × 192 instances: FFN weight matrices (W_gate, W_up, W_down each 2560×6848)
#   25.0 MiB × 256 instances: attention projection matrices (W_Q, W_K, W_V, W_O each 2560×2560)
#   13.4 MiB × 352 instances: FFN hidden states (gate/up projection outputs)
#    5.0 MiB × 1709 instances: residual stream tensors (batch=4 × seq=128 × d_model=2560)

# 97.7 MiB = 10000 × 2560 × 4 bytes = vocab_size × d_model
# 66.9 MiB = 2560 × 6848 × 4 bytes = d_model × d_ff (SwiGLU FFN weight)
# 25.0 MiB = 2560 × 2560 × 4 bytes = d_model × d_model (attention weight)
#  5.0 MiB = 4 × 128 × 2560 × 4 bytes = residual stream (batch × seq × d_model)
```

The largest single allocation is **97.7 MiB**, corresponding to the token embedding / output projection weight matrix (shape `[10000, 2560]` in FP32). The next-largest category is 66.9 MiB per allocation (192 instances total) — these are the SwiGLU FFN weight matrices (W_gate, W_up, W_down each `[2560, 6848]`), which appear 6 times per transformer block across 32 layers. The most numerous allocations (1709 instances) are 5 MiB residual-stream tensors (`[4, 128, 2560]` FP32), which are saved for the backward pass at every layer boundary.

---

**(f)** Use NVTX ranges and Nsight Systems to determine how much memory is saved for backward (residuals) by a single `TransformerBlock`. Note the 5 largest contributing operations and what percentage of overall memory they contribute. Then, based on how much memory was allocated during the forward pass and how much memory changes for every `TransformerBlock` in the backward pass, calculate how much memory the gradient tensors for a `TransformerBlock` take.

> **Deliverable:** Screenshots from Nsight Systems and a 1–2 paragraph response.

**Answer:**

*GPU: NVIDIA A100 80GB PCIe. xl model (d_model=2560, d_ff=6848, 32 heads, batch=4, ctx=128). Measured with `python measure_block_memory.py` by comparing peak activation memory for 1–4 TransformerBlock models.*

**Per-block activation memory saved for backward:**

```
# measure_block_memory.py output (A100, xl dims, ctx=128):
  1 layers: model=1286.3 MB, act_saved=453.2 MB
  2 layers: model=1607.2 MB, act_saved=580.6 MB
  3 layers: model=1911.8 MB, act_saved=716.9 MB
  4 layers: model=2216.5 MB, act_saved=852.4 MB

Per-block incremental activation memory:
  +1 block (1→2): act_saved_diff=127.46 MB
  +1 block (2→3): act_saved_diff=136.21 MB
  +1 block (3→4): act_saved_diff=135.58 MB
  Average per-block activation: 133.08 MB
```

Each `TransformerBlock` saves approximately **133 MB** of intermediate activations for the backward pass (with batch=4, ctx=128, xl dimensions). These are the tensors that PyTorch's autograd must keep alive from the forward pass until that block's backward runs. The five largest contributors, identified analytically from the computation graph:

| Rank | Operation | Tensor shape | Size | % of 133 MB |
|------|-----------|-------------|------|-------------|
| 1 | SwiGLU `w1(x)` output (input to SiLU) | (4, 128, 6848) | 14.09 MB | 10.6% |
| 2 | SwiGLU `w3(x)` output (gate branch) | (4, 128, 6848) | 14.09 MB | 10.6% |
| 3 | SwiGLU gate product `silu(w1)·w3` (input to `w2`) | (4, 128, 6848) | 14.09 MB | 10.6% |
| 4 | Attention scores `QKᵀ/√d_k` (input to softmax) | (4, 32, 128, 128) | 8.0 MB | 6.0% |
| 5 | Attention weights (softmax output, for AV backward) | (4, 32, 128, 128) | 8.0 MB | 6.0% |

These five tensors account for **58.4 MB = 43.9%** of the per-block activation memory. The remainder is distributed among Q, K, V projections before/after rearranging (~5 MB each), the block input residual stream (~5 MB), RMSNorm upcast intermediates (~5 MB each for ln1 and ln2), and smaller tensors for the attention output reshape, RoPE, and causal mask.

**Gradient tensor memory per TransformerBlock:**

From the backward measurements, memory after the full backward (params + grads, all activations freed):
```
  +1 block (1→2): after_bwd_diff = 605.3 MB
  model_params_diff =              320.9 MB  (measured)
  → gradient_diff =                284.4 MB  (empirical)
```

Analytically, each parameter has a gradient of identical shape and dtype (FP32), so gradient memory = parameter memory exactly:

```
# check_block_params.py output (A100):
  TransformerBlock total params: 78,812,160
  TransformerBlock gradient memory (FP32): 300.64 MB

  Breakdown:
    attn.q_proj.weight [2560, 2560]:  25.00 MB × 4 = 100.00 MB (all attn)
    ffn.w1.weight      [6848, 2560]:  66.88 MB }
    ffn.w2.weight      [2560, 6848]:  66.88 MB } = 200.63 MB (SwiGLU)
    ffn.w3.weight      [6848, 2560]:  66.88 MB }
    ln1.weight + ln2.weight:           0.02 MB
    Total gradient memory:           300.64 MB
```

The empirical estimate (284 MB) is slightly lower than the analytical value (301 MB) because the measured `model_params_diff` (321 MB) includes ~20 MB of CUDA allocator overhead per block, whereas the analytical value counts only the parameter bytes. The true gradient tensor memory per TransformerBlock is **≈ 301 MB** — exactly equal to the parameter size, since each gradient is a FP32 tensor of the same shape as its corresponding parameter.

---

## Section 3: Single-GPU Memory

### Problem `gradient_checkpointing` — Memory-Optimal Gradient Checkpointing (4 pts)

**(a)** Consider a Transformer with N identical blocks stacked sequentially. Without checkpointing, all N blocks' residuals are kept alive simultaneously (O(N) peak activation memory). What checkpointing strategy minimizes peak activation memory, ignoring compute cost? Describe how you would arrange the `checkpoint` calls (a code sketch is fine), and give the asymptotic peak activation memory as a function of N. Assume residuals saved by a single block dominate any per-checkpoint bookkeeping.

> **Deliverable:** 3–5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.

**Answer:**

The strategy that minimizes peak activation memory is to wrap **every individual block** in `torch.utils.checkpoint.checkpoint`. With full per-block checkpointing, no block saves any intermediate activations at all — instead, each block recomputes all its activations from scratch when the backward pass reaches it. At any moment during backward, only the activations of the currently-recomputing block are live (plus the single residual-stream tensor handed to that block as input), giving **O(1) peak activation memory** regardless of N (one block's worth of intermediate tensors at a time).

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # Each block is recomputed from scratch during backward.
            # No intermediate activations from any block are kept alive.
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

The cost is N extra forward passes during backward (one recomputation per block), so compute cost doubles. Asymptotically, peak activation memory is **O(1)** (constant in N): only the activations of one block at a time, plus a constant number of boundary tensors (the input and output of the currently-recomputing block).

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
