"""Replay a frozen Transformer front end with the sampling KV operations.

Numerical equivalence additionally requires the same environment order, batch
size, dtype and chunk size as sampling. This helper does not make shuffled
minibatches equivalent, nor does it substitute recorded routes for new routes.
Gradients remain enabled for trainable modules downstream of the frozen KV.
"""

from __future__ import annotations

import torch

_CACHE_STATE = (
    "_k_cache",
    "_v_cache",
    "_kv_cache_len",
    "_kv_cache_write_idx",
    "_kv_cache_abs_pos",
    "_prev_last_moe_router_p",
    "_prev_last_moe_router_valid",
    "_last_moe_router_js_sum",
    "_last_moe_router_js_count",
    "_last_moe_router_top1_switch_sum",
    "_last_moe_router_top1_switch_count",
)
_MISSING = object()


def _replay_mask(x, attn_mask, max_ctx_len):
    """Accept complete episode masks or their exact bounded-context form."""
    batch, length, _ = x.shape
    time = torch.arange(length, device=x.device)
    causal = time[:, None] >= time[None, :]
    if attn_mask is None:
        valid = torch.ones(batch, length, device=x.device, dtype=torch.bool)
        reset = torch.zeros_like(valid)
        reset[:, 0] = True
        return valid, reset
    if (
        attn_mask.dtype != torch.bool
        or attn_mask.device != x.device
        or tuple(attn_mask.shape) != (batch, length, length)
    ):
        raise ValueError("attn_mask must be boolean [B,T,T] on input device")
    valid = attn_mask.diagonal(dim1=1, dim2=2)
    reset = valid.clone()
    reset[:, 1:] &= ~(valid[:, :-1] & attn_mask[:, time[1:], time[:-1]])
    starts = torch.where(reset, time[None, :], 0).cummax(dim=1).values
    complete = (
        causal[None, :, :]
        & valid[:, :, None]
        & valid[:, None, :]
        & (time[None, None, :] >= starts[:, :, None])
    )
    bounded = (
        complete & (time[:, None] - time[None, :] < max_ctx_len)[None, :, :]
    )
    if not (
        torch.equal(attn_mask, complete) or torch.equal(attn_mask, bounded)
    ):
        raise ValueError(
            "Cached replay only accepts complete causal episode masks or "
            "their exact max_ctx_len window; sparse/future links are invalid"
        )
    return valid, reset


def _merge_step_records(records, batch):
    if (
        not records
        or any(not isinstance(record, dict) for record in records)
        or not records[0]
    ):
        raise ValueError("Cached replay requires a record from every MoE step")
    keys = records[0].keys()
    if "valid_mask" in keys:
        raise ValueError("valid_mask is reserved for replay padding metadata")
    if any(record.keys() != keys for record in records):
        raise ValueError("Cached replay record keys changed between chunks")
    merged = {}
    for key in keys:
        values = [record[key] for record in records]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"Replay record {key} must contain only tensors")
        if any(
            value.ndim < 2
            or value.shape[1] != 1
            or value.shape[2:] != values[0].shape[2:]
            for value in values
        ):
            raise ValueError(
                f"Replay record {key} must have shape [chunk_B,1,...]"
            )
        if sum(value.shape[0] for value in values) != batch:
            raise ValueError(f"Replay record {key} does not cover the batch")
        merged[key] = torch.cat(values, dim=0)
    return merged


def cached_sequence_replay(
    policy, x, attn_mask=None, chunk_size=0, *, return_pre_moe_hidden=False
):
    """Recompute [B,T,A] through temporary native single-step caches.

    Complete causal episode masks and their exact max-context windows are
    supported, including padding with an all-false row/column. Padded outputs
    are zero and their cache is cleared before the next valid observation.
    Trainable operations must follow the frozen KV-producing front end; this
    is the single-MoE incremental Actor's structural training contract.

    The native ``return_pre_moe_hidden`` name refers to layer zero's output,
    after its FFN, in the supported single-layer policy. Capture ``norm_f``'s
    input to retain that exact auxiliary-head convention and its gradients.
    """
    if x.ndim != 3 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("Cached replay requires nonempty [B,T,D] input")
    chunk_size = int(chunk_size)
    if chunk_size < 0 or int(policy.max_ctx_len) <= 0:
        raise ValueError("chunk_size must be nonnegative and context positive")
    valid, reset = _replay_mask(x, attn_mask, int(policy.max_ctx_len))
    saved = {name: getattr(policy, name, _MISSING) for name in _CACHE_STATE}
    layer = policy.layers[0]
    collect_records = hasattr(layer, "_replay_records") and callable(
        getattr(layer, "finalize_cached_replay", None)
    )
    saved_records = getattr(layer, "_replay_records", _MISSING)
    collect_flags = [
        (item, item.collect_routing_stats)
        for item in policy.layers
        if hasattr(item, "collect_routing_stats")
    ]
    outputs, step_records = [], []
    hidden_chunks, hidden_steps = [], []
    hidden_hook = None
    try:
        if return_pre_moe_hidden:
            if len(policy.layers) != 1:
                raise ValueError(
                    "Cached auxiliary hidden replay requires one native layer"
                )

            def capture_hidden(_module, inputs):
                hidden_chunks.append({"hidden": inputs[0]})

            hidden_hook = policy.norm_f.register_forward_pre_hook(
                capture_hidden
            )
        # Rebinding, rather than clearing, leaves the live rollout tensors
        # untouched even when native cache methods modify their inputs.
        policy.reset_kv_cache(x.shape[0], x.device)
        for item, _ in collect_flags:
            item.collect_routing_stats = False
        for time in range(x.shape[1]):
            hidden_chunks.clear()
            clear = reset[:, time] | ~valid[:, time]
            if clear.any():
                policy.clear_env_cache(clear.nonzero().flatten())
            step_x = torch.where(
                valid[:, time, None], x[:, time], torch.zeros_like(x[:, time])
            )
            if collect_records:
                layer._replay_records = []
            output = (
                policy.single_step_mu_chunked(step_x, chunk_size)
                if chunk_size > 0
                else policy.single_step_mu(step_x)
            )
            outputs.append(output.masked_fill(~valid[:, time, None], 0))
            if return_pre_moe_hidden:
                hidden = _merge_step_records(hidden_chunks, x.shape[0])[
                    "hidden"
                ]
                hidden_steps.append(
                    hidden.masked_fill(~valid[:, time, None, None], 0)
                )
            if collect_records:
                step_records.append(
                    _merge_step_records(layer._replay_records, x.shape[0])
                )
        if collect_records:
            if any(
                record.keys() != step_records[0].keys()
                for record in step_records
            ):
                raise ValueError("Replay record keys changed between steps")
            merged = {
                key: torch.cat([item[key] for item in step_records], dim=1)
                for key in step_records[0]
            }
            merged["valid_mask"] = valid
            layer.finalize_cached_replay(merged)
        mu = torch.stack(outputs, dim=1)
        if return_pre_moe_hidden:
            return mu, torch.cat(hidden_steps, dim=1)
        return mu
    finally:
        if hidden_hook is not None:
            hidden_hook.remove()
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(policy, name):
                    delattr(policy, name)
            else:
                setattr(policy, name, value)
        for item, collect in collect_flags:
            item.collect_routing_stats = collect
        if saved_records is not _MISSING:
            layer._replay_records = saved_records
