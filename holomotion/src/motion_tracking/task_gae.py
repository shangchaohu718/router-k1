"""Task-scoped GAE with separate bootstrap/trace boundaries and DDP moments."""

from __future__ import annotations

import torch

from holomotion.src.motion_tracking.task_contract import check_episode_tasks


@torch.no_grad()
def task_gae(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    tasks,
    *,
    discounts,
    final_values=None,
    finite_horizon=True,
):
    shape = rewards.shape
    if (
        rewards.ndim != 3
        or shape[-1] != 1
        or any(
            x.shape != shape
            for x in (values, next_values, terminated, truncated, tasks)
        )
    ):
        raise ValueError("GAE requires matching [T,E,1] tensors")
    terminated, truncated = terminated.bool(), truncated.bool()
    dones = terminated | truncated
    check_episode_tasks(tasks, dones)
    bootstrap = ~terminated
    if finite_horizon:
        bootstrap &= ~truncated
    elif torch.any(truncated & ~terminated):
        if final_values is None or final_values.shape != shape:
            raise ValueError(
                "Truncation requires pre-reset final observation values"
            )
        next_values = torch.where(truncated, final_values, next_values)
    carry = torch.zeros_like(values[0])
    advantages = torch.empty_like(values)
    for step in reversed(range(shape[0])):
        gamma, lam = torch.zeros_like(carry), torch.zeros_like(carry)
        for task in tasks.unique().tolist():
            if int(task) not in discounts:
                raise ValueError("Missing task discount contract")
            discount, trace_decay = discounts[int(task)]
            if not 0 <= discount <= 1 or not 0 <= trace_decay <= 1:
                raise ValueError("Invalid task discount")
            gamma = torch.where(tasks[step] == task, float(discount), gamma)
            lam = torch.where(tasks[step] == task, float(trace_decay), lam)
        delta = (
            rewards[step]
            + gamma * bootstrap[step] * next_values[step]
            - values[step]
        )
        carry = delta + gamma * lam * (~dones[step]) * carry
        advantages[step] = carry
    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite task GAE")
    return advantages, advantages + values


@torch.no_grad()
def normalize_by_task(advantages, tasks, valid=None, *, accelerator=None):
    if tasks.shape != advantages.shape or tasks.dtype != torch.int64:
        raise ValueError("Advantage/task shape or dtype mismatch")
    valid = (
        torch.ones_like(tasks, dtype=torch.bool)
        if valid is None
        else valid.bool()
    )
    if valid.shape != tasks.shape:
        raise ValueError("Invalid normalization mask")
    result = torch.zeros_like(advantages)
    moments = {}
    # Fixed registry order: all ranks enter the same collectives, even with no
    # local samples for a task. Local unique() is incorrect for DDP here.
    for task in (0, 1):
        mask = (tasks == task) & valid
        data = advantages[mask].double()
        stats = torch.stack(
            (data.sum(), data.square().sum(), mask.sum().double())
        )
        if accelerator is not None:
            stats = accelerator.reduce(stats, reduction="sum")
        total, squared, count = stats
        if count == 0:
            continue
        mean = total / count
        variance = (squared / count - mean.square()).clamp_min(0)
        std = variance.sqrt().clamp_min(1e-8)
        result[mask] = ((advantages[mask] - mean) / std).to(result.dtype)
        moments[task] = {
            "count": int(count),
            "mean": float(mean),
            "std": float(std),
        }
    if torch.any(valid & ~((tasks == 0) | (tasks == 1))):
        raise ValueError("Unknown normalization task")
    return result, moments
