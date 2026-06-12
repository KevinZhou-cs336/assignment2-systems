"""Distributed Data Parallel (DDP) wrappers for single-node multi-GPU training.

Two implementations are provided:

  DDP     -- overlapping: async all-reduce per-parameter fired from a
             post-accumulate-grad hook, overlapping communication with the
             tail of the backward pass.

  FlatDDP -- batched: after the entire backward pass, all gradients are
             flattened into a single tensor and communicated with ONE
             all-reduce call.  Simpler, but all communication happens
             sequentially after the backward pass.

Usage (both classes have the same external interface)::

    ddp_model = DDP(model)          # or FlatDDP(model)
    ...
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# DDP — per-parameter async all-reduce, overlapped with backward
# ---------------------------------------------------------------------------

class DDP(nn.Module):
    """Wraps a module for data-parallel training.

    On construction, parameters are broadcast from rank 0 so all ranks begin
    with identical weights.

    A post-accumulate-grad hook is registered on every unique parameter.
    PyTorch's AccumulateGrad node fires the hook ONCE per backward call after
    .grad has been fully accumulated (including tied-weight contributions),
    so no usage-count bookkeeping is needed.  The hook immediately launches
    an async all-reduce, letting communication overlap with the remainder of
    the backward pass.

    Call finish_gradient_synchronization() (via ddp_on_after_backward) after
    backward() and before optimizer.step() to wait for outstanding handles
    and divide by world_size.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # Broadcast rank-0 parameters to all other ranks.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Pending async handles: (Work, param) pairs collected by the hook.
        self._handles: list[tuple] = []

        # Timing accumulators (seconds); reset each step in
        # finish_gradient_synchronization.
        self.last_comm_time: float = 0.0
        self._comm_start: float | None = None

        # Register one hook per unique parameter.
        seen: set[int] = set()
        for param in self.module.parameters():
            pid = id(param)
            if pid not in seen and param.requires_grad:
                seen.add(pid)
                param.register_post_accumulate_grad_hook(self._allreduce_hook)

    def _allreduce_hook(self, param: torch.Tensor) -> None:
        if self._comm_start is None:
            self._comm_start = time.perf_counter()
        handle = dist.all_reduce(param.grad, async_op=True)
        self._handles.append((handle, param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait for all async all-reduces and average the gradients."""
        world_size = dist.get_world_size()
        for handle, param in self._handles:
            handle.wait()
            param.grad.div_(world_size)
        t_end = time.perf_counter()
        self.last_comm_time = (t_end - self._comm_start) if self._comm_start is not None else 0.0
        self._handles.clear()
        self._comm_start = None


# ---------------------------------------------------------------------------
# FlatDDP — single batched all-reduce after the full backward pass
# ---------------------------------------------------------------------------

class FlatDDP(nn.Module):
    """DDP variant that concatenates all gradients before communicating.

    Instead of N individual all-reduce calls (one per parameter), this
    class flattens every parameter gradient into a single contiguous tensor,
    issues ONE all-reduce, then copies the averaged gradients back.

    All communication happens synchronously in finish_gradient_synchronization,
    so it does not overlap with computation.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # Broadcast rank-0 parameters.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self.last_comm_time: float = 0.0

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Flatten all gradients, all-reduce once, copy back."""
        world_size = dist.get_world_size()

        # Collect unique parameters that have a gradient.
        seen: set[int] = set()
        params_with_grad: list[torch.Tensor] = []
        for param in self.module.parameters():
            pid = id(param)
            if pid not in seen and param.requires_grad and param.grad is not None:
                seen.add(pid)
                params_with_grad.append(param)

        if not params_with_grad:
            return

        # Flatten all gradients into one contiguous float32 buffer.
        grads = [p.grad.float() for p in params_with_grad]
        flat = torch._utils._flatten_dense_tensors(grads)

        t0 = time.perf_counter()
        dist.all_reduce(flat, async_op=False)
        flat.div_(world_size)
        self.last_comm_time = time.perf_counter() - t0

        # Scatter averaged values back to the original gradient tensors.
        for param, avg_grad in zip(
            params_with_grad,
            torch._utils._unflatten_dense_tensors(flat, grads),
        ):
            param.grad.copy_(avg_grad.to(param.grad.dtype))
