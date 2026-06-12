"""Optimizer state sharding across distributed ranks.

Problem (optimizer_state_sharding): Optimizer State Sharding (15 points)

Each rank's optimizer instance is responsible for updating only a subset of the
model's parameters (~1/world_size of them).  After every optimizer step, each
rank broadcasts its freshly-updated parameters to all other ranks so that every
rank ends the step with a fully-synchronized model.

This reduces optimizer-state memory by ~world_size because the first-moment
(m) and second-moment (v) buffers in Adam-style optimizers only need to exist
on the rank that owns the parameter.

Design
------
- Parameters are assigned to ranks in round-robin order: parameter i goes to
  rank (i % world_size).  This gives an even distribution for any number of
  parameters.
- ShardedOptimizer inherits from torch.optim.Optimizer so that zero_grad(),
  param_groups, and state_dict() work transparently.  The super-class
  constructor calls add_param_group(), so all custom attributes must be set
  BEFORE super().__init__().
- An _inner_optimizer (the wrapped optimizer_cls) is created after super().__init__
  with only the owned-parameter groups.  The inner optimizer is the only one
  that maintains per-parameter Adam state.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.optim


class ShardedOptimizer(torch.optim.Optimizer):
    """Wraps any torch.optim.Optimizer and shards its state across ranks.

    Usage::

        sharded_opt = ShardedOptimizer(
            model.parameters(), torch.optim.AdamW,
            lr=1e-4, weight_decay=0.1,
        )
        ...
        loss.backward()
        sharded_opt.step()   # updates own shard, then broadcasts all params
    """

    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ):
        # ----------------------------------------------------------------
        # Custom attributes MUST be set before super().__init__ because
        # Optimizer.__init__ immediately calls self.add_param_group().
        # ----------------------------------------------------------------
        self._rank        = dist.get_rank()
        self._world_size  = dist.get_world_size()
        self._optimizer_cls    = optimizer_cls
        self._optimizer_kwargs = kwargs

        # Global list of all unique parameters, in assignment order.
        self._all_params: list[torch.Tensor] = []
        # Maps id(param) -> owner rank.
        self._param_to_rank: dict[int, int] = {}
        # Running index for round-robin assignment.
        self._param_counter: int = 0

        # Inner-optimizer param groups collected during add_param_group
        # before the inner optimizer itself is created.
        self._inner_param_groups: list[dict] = []
        # Set to None until __init__ finishes building param groups.
        self._inner_optimizer: torch.optim.Optimizer | None = None

        # Optimizer base-class __init__; calls add_param_group for each group.
        defaults = dict(**kwargs)
        super().__init__(params, defaults)

        # Build the inner optimizer from owned-parameter groups.
        owned_groups = [g for g in self._inner_param_groups if g["params"]]
        if owned_groups:
            self._inner_optimizer = optimizer_cls(owned_groups, **kwargs)

    # ------------------------------------------------------------------
    # add_param_group — called by super().__init__ and externally
    # ------------------------------------------------------------------

    def add_param_group(self, param_group: dict) -> None:
        # Let the base class validate tensors, fill missing hyperparams from
        # self.defaults, and append the processed group to self.param_groups.
        super().add_param_group(param_group)
        processed = self.param_groups[-1]   # the just-added group

        owned_params: list[torch.Tensor] = []
        for param in processed["params"]:
            pid = id(param)
            if pid not in self._param_to_rank:
                # First time we see this parameter: assign to a rank.
                owner = self._param_counter % self._world_size
                self._param_to_rank[pid] = owner
                self._all_params.append(param)
                self._param_counter += 1

            if self._param_to_rank[pid] == self._rank:
                owned_params.append(param)

        # Mirror the group but with only owned parameters.
        inner_group = {k: v for k, v in processed.items() if k != "params"}
        inner_group["params"] = owned_params

        if self._inner_optimizer is not None:
            # Called after construction (e.g., gradually un-freezing layers).
            if owned_params:
                self._inner_optimizer.add_param_group(inner_group)
        else:
            # Called during construction; save for later inner-optimizer creation.
            self._inner_param_groups.append(inner_group)

    # ------------------------------------------------------------------
    # step — update owned params, then broadcast everything
    # ------------------------------------------------------------------

    def step(self, closure=None, **kwargs):
        # 1. Update only the parameters owned by this rank.
        loss = None
        if self._inner_optimizer is not None:
            loss = self._inner_optimizer.step(closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 2. Each rank broadcasts its freshly-updated parameters so all ranks
        #    stay in sync.  After this loop every rank holds the same weights.
        for param in self._all_params:
            owner = self._param_to_rank[id(param)]
            dist.broadcast(param.data, src=owner)

        return loss
