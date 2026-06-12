"""
FlashAttention-2 forward and backward pass implementations.

Background:
    The core Attention operation in Transformers is:
        O = softmax(Q K^T / sqrt(d)) V
    where Q/K/V are the query, key, and value matrices, and d is the head dimension.

    Problem with the naive implementation:
        The intermediate matrix S = Q K^T has shape seq_len x seq_len.
        For long sequences this requires enormous GPU memory (e.g. ~256 MB at seq_len=8192).

    FlashAttention-2's key idea:
        Split Q, K, V into small tiles and process one tile at a time.
        Use the "online softmax" trick to incrementally update the output,
        so the full S matrix never has to be materialized in HBM.
        This also cuts HBM read/write traffic, making the kernel faster overall.
"""

import math
import torch

# Try to import Triton (GPU kernel framework); fall back to pure PyTorch if unavailable.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ===========================================================================
# Shared backward computation (pure PyTorch + torch.compile)
#
# Why is this defined at module level outside the classes?
#   FlashAttentionPyTorch and FlashAttentionTriton use identical backward logic.
#   Defining it once and compiling it once lets both classes share the same
#   optimized function without recompiling on every call.
# ===========================================================================

def _flash_backward_impl(Q, K, V, O, dO, L, is_causal, scale):
    """
    FlashAttention-2 backward pass (PDF equations 13-19).

    Core idea — Recomputation:
        The forward pass saves only L (the logsumexp) to reduce memory, not P
        (the full seq_len x seq_len attention weight matrix).
        During the backward pass we recompute S and P from the saved Q and K,
        so P only lives transiently in memory rather than across the full
        forward-to-backward interval.

    Why D = rowsum(O o dO) works:
        The standard softmax backward gives dS = P o (dP - rowsum(P o dP)).
        Because O = PV, one can show rowsum(P o dP) = rowsum(O o dO),
        so D can be computed from the already-saved O and the incoming dO
        without recomputing P a second time.

    Args:
        Q, K, V  : forward-pass inputs, shapes (batch, n_q/n_k, d)
        O        : forward-pass output, shape (batch, n_q, d)
        dO       : gradient of the loss w.r.t. O, shape (batch, n_q, d)
        L        : logsumexp saved during the forward pass, shape (batch, n_q)
        is_causal: whether causal masking was applied in the forward pass
        scale    : scaling factor 1/sqrt(d)

    Returns:
        dQ : shape (batch, n_q, d)
        dK : shape (batch, n_k, d)
        dV : shape (batch, n_k, d)
    """
    n_queries = Q.shape[1]
    n_keys    = K.shape[1]

    # ------------------------------------------------------------------
    # Pre-compute auxiliary vector D (the correction term in equation 17).
    #   D[b, i] = sum_k O[b, i, k] * dO[b, i, k]   shape: (batch, n_q)
    #   Interpretation: the row-wise dot product of the output and its gradient.
    # ------------------------------------------------------------------
    D = (O.float() * dO.float()).sum(dim=-1)  # (batch, n_q)

    # ------------------------------------------------------------------
    # Recompute attention scores S (equation 13: S = Q K^T / sqrt(d)).
    #   We use the saved Q and K instead of storing S during the forward pass.
    # ------------------------------------------------------------------
    S = torch.bmm(Q.float(), K.float().transpose(-1, -2)) * scale  # (batch, n_q, n_k)

    # ------------------------------------------------------------------
    # Apply causal mask, identical to the forward pass, so that the
    # recomputed P matches what was used to produce O.
    #   Mask out positions where query index < key index (future tokens).
    # ------------------------------------------------------------------
    if is_causal:
        q_idx = torch.arange(n_queries, device=Q.device).unsqueeze(1)  # (n_q, 1)
        k_idx = torch.arange(n_keys,    device=Q.device).unsqueeze(0)  # (1, n_k)
        S = S.masked_fill((q_idx < k_idx).unsqueeze(0), -1e6)

    # ------------------------------------------------------------------
    # Recompute normalized attention weights P (equation 14).
    #   P_ij = exp(S_ij - L_i)  which equals softmax(S)_ij.
    #   L has shape (batch, n_q); unsqueeze(-1) broadcasts over n_k.
    # ------------------------------------------------------------------
    P = torch.exp(S - L.float().unsqueeze(-1))  # (batch, n_q, n_k)

    # ------------------------------------------------------------------
    # Compute dV (equation 15: dV = P^T dO).
    #   P^T : (batch, n_k, n_q)
    #   dO  : (batch, n_q, d)
    #   result : (batch, n_k, d)
    # ------------------------------------------------------------------
    dV = torch.bmm(P.transpose(-1, -2), dO.float())

    # ------------------------------------------------------------------
    # Compute dP (equation 16: dP = dO V^T).
    #   dO : (batch, n_q, d)
    #   V^T: (batch, d, n_k)
    #   result: (batch, n_q, n_k)
    # ------------------------------------------------------------------
    dP = torch.bmm(dO.float(), V.float().transpose(-1, -2))

    # ------------------------------------------------------------------
    # Compute dS (equation 17: dS_ij = P_ij * (dP_ij - D_i)).
    #   D has shape (batch, n_q); unsqueeze(-1) broadcasts over n_k.
    #   This is the simplified form of the softmax Jacobian applied to dP.
    # ------------------------------------------------------------------
    dS = P * (dP - D.unsqueeze(-1))  # (batch, n_q, n_k)

    # ------------------------------------------------------------------
    # Compute dQ (equation 18: dQ = dS K / sqrt(d)).
    #   dS : (batch, n_q, n_k)
    #   K  : (batch, n_k, d)
    #   result: (batch, n_q, d)
    # ------------------------------------------------------------------
    dQ = torch.bmm(dS, K.float()) * scale

    # ------------------------------------------------------------------
    # Compute dK (equation 19: dK = dS^T Q / sqrt(d)).
    #   dS^T: (batch, n_k, n_q)
    #   Q   : (batch, n_q, d)
    #   result: (batch, n_k, d)
    # ------------------------------------------------------------------
    dK = torch.bmm(dS.transpose(-1, -2), Q.float()) * scale

    return dQ, dK, dV


# JIT-compile the backward function with torch.compile.
#   Benefits:
#     - Adjacent ops are automatically fused, reducing HBM read/write traffic.
#     - torch.compile creates separate specializations for is_causal=True/False.
#   Falls back to the eager function if torch.compile is unsupported
#   (e.g. older PyTorch builds or certain hardware configurations).
try:
    _compiled_flash_backward = torch.compile(_flash_backward_impl)
except Exception:
    _compiled_flash_backward = _flash_backward_impl


# ===========================================================================
# Part 1: Pure-PyTorch tiled implementation (Algorithm 1 from the paper)
# ===========================================================================

class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 forward pass implemented with plain PyTorch operators.

    This is a torch.autograd.Function subclass, requiring static forward and
    backward methods.  The forward method manually saves the tensors needed
    for the backward pass via ctx.save_for_backward.

    Note: This implementation is slower than writing softmax directly in PyTorch
    because the tiling loop runs in Python.  Its purpose is to serve as a
    readable, debuggable reference that matches the Triton kernel exactly.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Tiled forward pass: compute Attention output O and logsumexp L.

        Args:
            ctx       : autograd context used to stash tensors for backward
            Q         : query matrix, shape (batch_size, n_queries, d)
            K         : key   matrix, shape (batch_size, n_keys,   d)
            V         : value matrix, shape (batch_size, n_keys,   d)
            is_causal : if True, each position may only attend to earlier positions

        Returns:
            O : attention output, shape (batch_size, n_queries, d)

        Also saves (L, Q, K, V, O) via ctx.save_for_backward.
        L has shape (batch_size, n_queries) — the log of the softmax denominator.
        """
        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[1]

        # Scaling factor 1/sqrt(d) prevents the dot products from growing too
        # large, which would push softmax into a near-zero gradient region.
        scale = 1.0 / math.sqrt(d)

        # ------------------------------------------------------------------
        # Tile sizes: the paper requires at least 16x16; 32 is a good default
        # that balances register pressure and arithmetic intensity.
        # B_q: number of query rows per tile; B_k: number of key rows per tile.
        # ------------------------------------------------------------------
        B_q = 32
        B_k = 32

        # Number of tiles in each dimension (ceiling division handles remainders).
        T_q = math.ceil(n_queries / B_q)
        T_k = math.ceil(n_keys / B_k)

        # Output tensors kept in float32 for numerical stability.
        O = torch.zeros(batch_size, n_queries, d, device=Q.device, dtype=torch.float32)
        # L[b, i] = log(sum_j exp(S[b, i, j])), the logsumexp over all keys.
        L = torch.zeros(batch_size, n_queries, device=Q.device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # Outer loop over query tiles (Algorithm 1, line 4: for i = 1..T_q).
        # ------------------------------------------------------------------
        for i in range(T_q):
            q_start = i * B_q
            q_end = min((i + 1) * B_q, n_queries)  # guard against out-of-bounds
            B_q_actual = q_end - q_start            # last tile may be smaller

            # Load current query tile, cast to float32 for on-chip computation.
            Q_i = Q[:, q_start:q_end, :].float()   # (batch, B_q_actual, d)

            # ----------------------------------------------------------------
            # Initialize the three running accumulators for online softmax
            # (Algorithm 1, line 6):
            #   O_i : unnormalized output accumulator
            #   l_i : running estimate of the softmax denominator
            #   m_i : running row-wise maximum for numerical stability, starts at -inf
            # ----------------------------------------------------------------
            O_i = torch.zeros(batch_size, B_q_actual, d, device=Q.device, dtype=torch.float32)
            l_i = torch.zeros(batch_size, B_q_actual, device=Q.device, dtype=torch.float32)
            m_i = torch.full((batch_size, B_q_actual), float('-inf'), device=Q.device, dtype=torch.float32)

            # ----------------------------------------------------------------
            # Inner loop over key tiles (Algorithm 1, line 7: for j = 1..T_k).
            # ----------------------------------------------------------------
            for j in range(T_k):
                k_start = j * B_k
                k_end = min((j + 1) * B_k, n_keys)

                K_j = K[:, k_start:k_end, :].float()  # (batch, B_k_actual, d)
                V_j = V[:, k_start:k_end, :].float()  # (batch, B_k_actual, d)

                # ------------------------------------------------------------
                # Algorithm 1, line 9: S = Q_i K_j^T / sqrt(d)
                # Shape: (batch, B_q_actual, B_k_actual)
                # ------------------------------------------------------------
                S = torch.bmm(Q_i, K_j.transpose(-1, -2)) * scale

                # ------------------------------------------------------------
                # Causal mask: prevent position q from attending to position k > q.
                # Masked entries are set to -1e6 so exp(-1e6) ≈ 0 after softmax.
                # ------------------------------------------------------------
                if is_causal:
                    # q_idx shape: (B_q_actual, 1), k_idx shape: (1, B_k_actual)
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                    # mask[q, k] = True means position q cannot attend to position k
                    S = S.masked_fill((q_idx < k_idx).unsqueeze(0), -1e6)

                # ------------------------------------------------------------
                # Algorithm 1, line 10: update the running row-wise maximum.
                #   m_i_new[b, q] = max(m_i[b, q], max_k S[b, q, k])
                # This is the core trick for numerically stable online softmax.
                # ------------------------------------------------------------
                m_i_new = torch.maximum(m_i, S.amax(dim=-1))  # (batch, B_q_actual)

                # ------------------------------------------------------------
                # Algorithm 1, line 11: unnormalized softmax values.
                #   P_tilde[b, q, k] = exp(S[b, q, k] - m_i_new[b, q])
                # Subtracting the row maximum keeps all values in (0, 1],
                # avoiding floating-point overflow.
                # ------------------------------------------------------------
                P_tilde = torch.exp(S - m_i_new.unsqueeze(-1))  # (batch, B_q_actual, B_k_actual)

                # ------------------------------------------------------------
                # Algorithm 1, line 12: update the running denominator.
                #   The old l_i was computed relative to the old maximum m_i,
                #   so we rescale it by exp(m_i - m_i_new) before adding the
                #   new tile's contribution.
                # ------------------------------------------------------------
                correction = torch.exp(m_i - m_i_new)           # (batch, B_q_actual)
                l_i_new = correction * l_i + P_tilde.sum(dim=-1)

                # ------------------------------------------------------------
                # Algorithm 1, line 13: update the output accumulator.
                #   Apply the same rescaling to O_i, then add P_tilde @ V_j.
                # ------------------------------------------------------------
                O_i = correction.unsqueeze(-1) * O_i + torch.bmm(P_tilde, V_j)

                # Advance the running state for the next key tile.
                m_i = m_i_new
                l_i = l_i_new

            # ----------------------------------------------------------------
            # Algorithm 1, lines 15-16: after processing all key tiles,
            # normalize O_i by the final denominator and compute logsumexp L_i.
            #   O_i_final = O_i / l_i
            #   L_i = m_i + log(l_i)    (used by the backward pass)
            # ----------------------------------------------------------------
            O_i = O_i / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i)

            # Write this query tile's results into the global output tensors.
            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i

        # Cast back to the input dtype to avoid silent precision mismatches.
        O = O.to(Q.dtype)
        L = L.to(Q.dtype)

        # ------------------------------------------------------------------
        # Save tensors for the backward pass.
        # The test inspects saved_tensors looking for the one tensor with
        # shape (batch, n_queries) — that is L.  Q, K, V, O are all 3-D.
        # ------------------------------------------------------------------
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass: retrieve the saved tensors and call the compiled
        backward function.

        autograd convention:
            backward must return one gradient per forward input, in order.
            forward inputs are (ctx, Q, K, V, is_causal), so we return
            (dQ, dK, dV, None) — None because is_causal is a Python bool,
            not a tensor, and therefore has no gradient.
        """
        # Unpack saved tensors in the same order they were passed to save_for_backward.
        L, Q, K, V, O = ctx.saved_tensors
        scale = 1.0 / math.sqrt(Q.shape[-1])

        dQ, dK, dV = _compiled_flash_backward(
            Q, K, V, O, dO, L, ctx.is_causal, scale
        )

        # Cast gradients back to the original input dtype; return None for is_causal.
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None


# ===========================================================================
# Part 2: Triton GPU kernel implementation (Algorithm 1 executed on GPU)
# ===========================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _flash_fwd_kernel(
        # --- Input / output pointers ---
        Q_ptr, K_ptr, V_ptr,   # base pointers for Q, K, V
        O_ptr, L_ptr,          # base pointers for output O and logsumexp L
        # --- Per-tensor memory strides ---
        # A stride tells Triton how many elements to skip to advance one step
        # along a given dimension.  E.g. stride_qb = n_queries * d (batch stride).
        stride_qb, stride_qq, stride_qd,   # Q: batch / query / dim strides
        stride_kb, stride_kk, stride_kd,   # K: batch / key   / dim strides
        stride_vb, stride_vk, stride_vd,   # V: batch / key   / dim strides
        stride_ob, stride_oq, stride_od,   # O: batch / query / dim strides
        stride_lb, stride_lq,              # L: batch / query strides
        # --- Size parameters ---
        N_QUERIES, N_KEYS,   # total number of query and key positions
        scale,               # scaling factor 1/sqrt(d)
        # --- Compile-time constants (constexpr) ---
        # Triton requires these to be known at compile time to emit optimal code.
        D: tl.constexpr,              # head dimension
        Q_TILE_SIZE: tl.constexpr,    # query tile size B_q
        K_TILE_SIZE: tl.constexpr,    # key tile size B_k
        IS_CAUSAL: tl.constexpr,      # causal masking flag; generates separate code paths
    ):
        """
        Triton GPU kernel for the FlashAttention-2 forward pass.

        Triton's parallelism model:
            Each program instance (a group of threads running the same program)
            is responsible for one query tile of one batch element.
            The 2-D launch grid is (T_q, batch_size), so:
              program_id(0) selects the query tile index
              program_id(1) selects the batch index
        """
        query_tile_index = tl.program_id(0)
        batch_index      = tl.program_id(1)

        # ------------------------------------------------------------------
        # Build block pointers.
        # A block pointer abstracts a 2-D region of HBM: you specify the base
        # address, the overall tensor shape, per-dimension strides, the current
        # tile offset, and the tile shape.  Calling .advance() slides the
        # pointer to the next tile without recalculating all the arithmetic.
        # ------------------------------------------------------------------

        # Q block pointer: positioned at the current batch and query tile.
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,          # skip preceding batch elements
            shape=(N_QUERIES, D),                     # 2-D shape within this batch
            strides=(stride_qq, stride_qd),           # row and column strides
            offsets=(query_tile_index * Q_TILE_SIZE, 0),  # start of this query tile
            block_shape=(Q_TILE_SIZE, D),             # size of the block to load
            order=(1, 0),                             # row-major (fastest dim last)
        )

        # K block pointer: starts at key tile 0 and is advanced in the inner loop.
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # V block pointer: moves in lockstep with K.
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # O block pointer: output destination aligned with the Q tile.
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        # L block pointer: L is 1-D per batch (one scalar per query position).
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # ------------------------------------------------------------------
        # Load the Q tile into on-chip SRAM (Algorithm 1, line 5).
        # Q_i is reused across all key tiles, so it is loaded only once.
        # boundary_check pads out-of-bounds positions with zero.
        # ------------------------------------------------------------------
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # ------------------------------------------------------------------
        # Initialize on-chip accumulators in float32 (Algorithm 1, line 6).
        # ------------------------------------------------------------------
        O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)            # output accumulator
        l_i = tl.zeros((Q_TILE_SIZE,),   dtype=tl.float32)            # denominator estimate
        m_i = tl.full( (Q_TILE_SIZE,),   float('-inf'), dtype=tl.float32)  # running row-max

        # Absolute query indices for this tile (used in the causal mask comparison).
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)  # total number of key tiles

        # ------------------------------------------------------------------
        # Inner loop over key tiles (Algorithm 1, lines 7-14).
        # ------------------------------------------------------------------
        for j in range(T_k):
            # Load the current K and V tiles from HBM into SRAM.
            K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # --------------------------------------------------------------
            # Algorithm 1, line 9: S = Q_i K_j^T * scale
            # tl.dot performs matrix multiplication; tl.trans transposes.
            # Everything is cast to float32 to maintain numerical precision.
            # Result shape: (Q_TILE_SIZE, K_TILE_SIZE)
            # --------------------------------------------------------------
            S = tl.dot(Q_i.to(tl.float32), tl.trans(K_j).to(tl.float32)) * scale

            # --------------------------------------------------------------
            # Causal mask: zero out attention to future key positions.
            # IS_CAUSAL is a constexpr, so the compiler emits either the
            # masked or the unmasked code path — no runtime overhead.
            # --------------------------------------------------------------
            if IS_CAUSAL:
                # Absolute key indices for this tile.
                k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                # Broadcasting: (Q_TILE_SIZE, 1) < (1, K_TILE_SIZE) → (Q_TILE_SIZE, K_TILE_SIZE)
                causal_mask = q_offsets[:, None] < k_offsets[None, :]
                S = tl.where(causal_mask, -1e6, S)

            # --------------------------------------------------------------
            # Algorithm 1, line 10: update the running row-wise maximum.
            # tl.max(S, axis=1) reduces along the key dimension → (Q_TILE_SIZE,)
            # --------------------------------------------------------------
            m_i_new = tl.maximum(m_i, tl.max(S, axis=1))

            # --------------------------------------------------------------
            # Algorithm 1, line 11: unnormalized softmax values.
            # P_tilde = exp(S - m_new), shape (Q_TILE_SIZE, K_TILE_SIZE)
            # --------------------------------------------------------------
            P_tilde = tl.exp(S - m_i_new[:, None])

            # --------------------------------------------------------------
            # Algorithm 1, lines 12-13: update denominator and output.
            # correction = exp(m_old - m_new) rescales the old estimates to
            # the new maximum baseline before adding the current tile.
            # --------------------------------------------------------------
            correction = tl.exp(m_i - m_i_new)
            l_i = correction * l_i + tl.sum(P_tilde, axis=1)
            # Cast P_tilde to V's dtype before the matmul to exploit hardware
            # accelerators (e.g. bf16 Tensor Cores); accumulate in float32.
            O_i = correction[:, None] * O_i + tl.dot(
                P_tilde.to(V_j.dtype), V_j, out_dtype=tl.float32
            )

            m_i = m_i_new

            # Advance the K and V block pointers to the next key tile.
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # ------------------------------------------------------------------
        # Algorithm 1, lines 15-16: final normalization and logsumexp.
        # ------------------------------------------------------------------
        O_i = O_i / l_i[:, None]
        L_i = m_i + tl.log(l_i)

        # ------------------------------------------------------------------
        # Write results back to HBM, casting to the output tensor's dtype
        # (e.g. bfloat16 or float32).
        # *_block_ptr.type.element_ty returns the element type of the pointer.
        # ------------------------------------------------------------------
        tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty), boundary_check=(0,))


    class FlashAttentionTriton(torch.autograd.Function):
        """
        FlashAttention-2 using a Triton GPU kernel for the forward pass.

        The interface is identical to FlashAttentionPyTorch, but the forward
        pass runs as a fused GPU kernel, giving significantly better memory
        and compute efficiency for long sequences.

        The backward pass reuses the same compiled PyTorch implementation as
        FlashAttentionPyTorch — Triton is only used to accelerate the forward.
        """

        # Tile sizes; adjust here to tune for a specific GPU architecture.
        _Q_TILE = 32
        _K_TILE = 32

        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            """
            Launch the Triton kernel to compute O and L.

            Requirements: Q, K, V must be contiguous CUDA tensors.
            Returns O and stashes (L, Q, K, V, O) for the backward pass.
            """
            assert Q.is_cuda, "Triton kernels require CUDA tensors"
            assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
                "Input tensors must be contiguous; block pointer arithmetic assumes this"

            batch_size, n_queries, d = Q.shape
            n_keys = K.shape[1]
            scale = 1.0 / math.sqrt(d)

            Q_TILE = FlashAttentionTriton._Q_TILE
            K_TILE = FlashAttentionTriton._K_TILE
            # Number of query tiles determines the first dimension of the launch grid.
            T_q = triton.cdiv(n_queries, Q_TILE)

            # Pre-allocate output tensors with the same dtype as Q.
            O = torch.empty_like(Q)
            L = torch.empty(batch_size, n_queries, device=Q.device, dtype=Q.dtype)

            # ------------------------------------------------------------------
            # Launch the Triton kernel on a 2-D grid (T_q, batch_size):
            #   program_id(0) = query tile index  (0 .. T_q - 1)
            #   program_id(1) = batch index       (0 .. batch_size - 1)
            # Strides must be passed explicitly; Triton cannot infer them.
            # ------------------------------------------------------------------
            _flash_fwd_kernel[(T_q, batch_size)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                n_queries, n_keys,
                scale,
                D=d,
                Q_TILE_SIZE=Q_TILE,
                K_TILE_SIZE=K_TILE,
                IS_CAUSAL=is_causal,
            )

            ctx.save_for_backward(L, Q, K, V, O)
            ctx.is_causal = is_causal
            return O

        @staticmethod
        def backward(ctx, dO):
            """
            Backward pass: shared with FlashAttentionPyTorch.
            The Triton kernel only accelerates the forward; gradients are
            computed with the torch.compile-d PyTorch function.
            """
            L, Q, K, V, O = ctx.saved_tensors
            scale = 1.0 / math.sqrt(Q.shape[-1])

            dQ, dK, dV = _compiled_flash_backward(
                Q, K, V, O, dO, L, ctx.is_causal, scale
            )

            return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None

else:
    # Stub when Triton is not installed.
    # Tests that require CUDA are automatically skipped on non-CUDA hosts,
    # so this branch is only reached if someone calls the class directly.
    class FlashAttentionTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError("Triton is not installed; cannot use FlashAttentionTriton")

        @staticmethod
        def backward(ctx, dO):
            raise RuntimeError("Triton is not installed")
