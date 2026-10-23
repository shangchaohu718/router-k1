"""CPU-only Stage B calibration with ordered old/new teacher histories.

This is offline behavior/route preservation, not a replacement for closed-loop
task evaluation. Inputs must come from evaluate_internal_moe.py. Each loss
token includes its real contiguous context; a frame shuffled out of history is
never used as an independent observation.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.nn import functional as F

from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor
from tools.booster_k1_retarget.preflight_sliding_router import (
    actor_config,
    sample_actor_obs,
)
from tools.booster_k1_retarget.verify_sliding_checkpoint import actor_state


def read_ordered_recording(
    path, role, context, stride=1, new_frame_intervals=None
):
    if role not in {"old", "new", "transition"}:
        raise ValueError("Recording role must be old/new/transition")
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != 1:
            raise ValueError(
                "Need schema 1 full ordered internal-MoE observations"
            )
        names = (
            "observations",
            "motion_keys",
            "batch_ids",
            "record_steps",
            "reference_frames",
            "episode_positions",
        )
        arrays = {name: data[name].copy() for name in names}
    obs = arrays["observations"]
    count = len(obs)
    if (
        obs.ndim != 2
        or count < 1
        or not np.isfinite(obs).all()
        or any(len(a) != count for a in arrays.values())
    ):
        raise ValueError("Malformed ordered observation arrays")
    if np.any(np.diff(arrays["record_steps"]) != 1):
        raise ValueError("Record steps must be contiguous, not shuffled")
    windows = []
    start = 0
    for end in range(count):
        pos = int(arrays["episode_positions"][end])
        reset = (
            end == 0
            or pos == 0
            or arrays["batch_ids"][end] != arrays["batch_ids"][end - 1]
            or arrays["motion_keys"][end] != arrays["motion_keys"][end - 1]
        )
        if reset:
            if pos != 0:
                raise ValueError(
                    "Recording segment starts without its episode prefix"
                )
            start = end
        elif pos != int(arrays["episode_positions"][end - 1]) + 1:
            raise ValueError("Episode history is missing or reordered")
        if (end - start) % stride:
            continue
        label = 0 if role == "old" else 1
        if role == "transition":
            if not new_frame_intervals:
                raise ValueError(
                    "Transitions require explicit new_frame_intervals; "
                    "no guessed task labels"
                )
            frame = int(arrays["reference_frames"][end])
            label = int(
                any(int(a) <= frame < int(b) for a, b in new_frame_intervals)
            )
        prefix = max(start, end - context + 1)
        windows.append(
            (torch.from_numpy(obs[prefix : end + 1].astype(np.float32)), label)
        )
    return windows


def pack_windows(windows, dimension, context):
    if not windows:
        raise ValueError("Need nonempty ordered replay windows")
    obs = torch.zeros(len(windows), context, dimension)
    mask = torch.zeros(len(windows), context, context, dtype=torch.bool)
    labels = torch.empty(len(windows), dtype=torch.long)
    for index, (sequence, label) in enumerate(windows):
        if (
            sequence.ndim != 2
            or sequence.shape[1] != dimension
            or not 1 <= len(sequence) <= context
        ):
            raise ValueError("Replay observation shape/context mismatch")
        start = context - len(sequence)
        obs[index, start:] = sequence
        mask[index, start:, start:] = torch.ones(
            len(sequence), len(sequence), dtype=torch.bool
        ).tril()
        labels[index] = label
    return obs, mask, labels


def route_distribution(layer, logits, total):
    if (
        logits.ndim < 1
        or logits.shape[-1] > total
        or torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(dim=-1).all()
    ):
        raise ValueError("Route logits must contain finite eligible experts")
    scores = (
        logits.softmax(-1)
        if layer.routing_score_fn == "softmax"
        else logits.sigmoid()
    )
    scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-20)
    return torch.nn.functional.pad(scores, (0, total - scores.shape[-1]))


def _validate_labels(labels, batch, *, require_both=False):
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.int64
        or labels.shape != (batch,)
        or not ((labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("Calibration labels must be int64 [B] with IDs 0/1")
    if require_both and not ((labels == 0).any() and (labels == 1).any()):
        raise ValueError("Gate supervision requires both old/new classes")


def admission_supervision_loss(student, labels, expected_shape):
    """Balance negative and positive BCE without differentiating hard gates.

    Labels supervise cached gate logits after the student's reference-feature
    forward pass. They never enter the policy or alter expert selections.
    Multiple internal admission gates receive equally weighted supervision.
    """
    _validate_labels(labels, expected_shape[0], require_both=True)
    layers = [
        layer
        for layer in student.actor_module.layers
        if getattr(layer, "added_admission", None) is not None
    ]
    if not layers:
        raise ValueError("Gate supervision requires explicit admission gates")
    class_losses = {0: [], 1: []}
    class_rates = {0: [], 1: []}
    for layer in layers:
        if layer.added_admission.mode != "learned":
            raise ValueError(
                "Gate supervision requires admission mode=learned"
            )
        logits = layer.last_admission_logits
        admitted = layer.last_admission_mask
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != expected_shape
            or logits.ndim != 2
            or not torch.isfinite(logits).all()
            or not isinstance(admitted, torch.Tensor)
            or admitted.shape != expected_shape
            or admitted.dtype != torch.bool
        ):
            raise ValueError(
                "Admission replay must provide finite [B,T] logits"
            )
        if not logits.requires_grad:
            raise ValueError(
                "Admission logits must retain calibration gradients"
            )
        token_loss = F.binary_cross_entropy_with_logits(
            logits[:, -1], labels.to(logits), reduction="none"
        )
        for label in (0, 1):
            selected = labels == label
            class_losses[label].append(token_loss[selected].mean())
            class_rates[label].append(admitted[:, -1][selected].float().mean())
    negative, positive = (
        torch.stack(class_losses[label]).mean() for label in (0, 1)
    )
    balanced = 0.5 * (negative + positive)
    metrics = {
        "gate_balanced_bce": float(balanced.detach()),
        "old_gate_bce": float(negative.detach()),
        "new_gate_bce": float(positive.detach()),
        "old_admission_rate": float(torch.stack(class_rates[0]).mean()),
        "new_admission_rate": float(torch.stack(class_rates[1]).mean()),
        "gate_layers": len(layers),
    }
    return balanced, metrics


def targets(old, new, obs, mask):
    with torch.no_grad():
        old_mu = old.actor_module.sequence_mu(obs, attn_mask=mask)[:, -1]
        old_layer = old.actor_module.layers[0]
        old_route = route_distribution(
            old_layer,
            old_layer.last_expanded_logits[:, -1],
            old_layer.total_experts,
        )
        new_mu = new.actor_module.sequence_mu(obs, attn_mask=mask)[:, -1]
        new_layer = new.actor_module.layers[0]
        new_route = route_distribution(
            new_layer,
            new_layer.last_expanded_logits[:, -1],
            new_layer.total_experts,
        )
    return old_mu, new_mu, old_route, new_route


def preservation_loss(
    student,
    obs,
    mask,
    labels,
    teacher_targets,
    route_weight=1.0,
    gate_weight=0.0,
):
    if (
        not math.isfinite(route_weight)
        or route_weight < 0
        or not math.isfinite(gate_weight)
        or gate_weight < 0
    ):
        raise ValueError("Calibration loss weights must be finite/nonnegative")
    _validate_labels(labels, len(obs), require_both=gate_weight > 0)
    old_mu, new_mu, old_route, new_route = teacher_targets
    mu = student.actor_module.sequence_mu(obs, attn_mask=mask)[:, -1]
    choose_new = labels.bool()
    teacher_mu = torch.where(choose_new[:, None], new_mu, old_mu).detach()
    teacher_route = torch.where(
        choose_new[:, None], new_route, old_route
    ).detach()
    layer = student.actor_module.layers[0]
    probability = route_distribution(
        layer, layer.last_expanded_logits[:, -1], layer.total_experts
    )
    # Fixed equal sigma: this is the exact mean term of diagonal Gaussian KL.
    sigma = student._sigma_like(mu)
    behavior = (0.5 * ((mu - teacher_mu) / sigma).square()).sum(-1)
    route = (
        teacher_route
        * (
            teacher_route.clamp_min(1e-20).log()
            - probability.clamp_min(1e-20).log()
        )
    ).sum(-1)
    per_class = []
    metrics = {}
    for label, name in ((0, "old"), (1, "new")):
        select = labels == label
        if select.any():
            mean_behavior, mean_route = (
                behavior[select].mean(),
                route[select].mean(),
            )
            per_class.append(mean_behavior + route_weight * mean_route)
            metrics[name + "_behavior_kl"] = float(mean_behavior.detach())
            metrics[name + "_route_kl"] = float(mean_route.detach())
    loss = torch.stack(per_class).mean()
    if gate_weight > 0:
        gate_loss, gate_metrics = admission_supervision_loss(
            student, labels, obs.shape[:2]
        )
        loss = loss + gate_weight * gate_loss
        metrics.update(gate_metrics)
    return loss, metrics


def load_source(checkpoint, *, require_admission=False):
    config = OmegaConf.load(checkpoint.parent / "config.yaml")
    if str(config.modules.actor.type) != "InternalExpandedMoEPolicy":
        raise ValueError(
            "Stage B requires an internally extended Stage A checkpoint"
        )
    module = actor_config(config)
    admission = module["internal_expansion"].get("admission")
    source_state = actor_state(checkpoint)
    if require_admission and admission is None:
        raise ValueError(
            "Gate calibration needs an explicitly registered admission "
            "checkpoint; gate-free checkpoints are not migrated implicitly"
        )
    if admission is not None and not any(
        ".added_admission." in key for key in source_state
    ):
        raise ValueError(
            "Admission config/checkpoint mismatch: no saved gate state; "
            "explicit migration is required before calibration"
        )
    module["internal_expansion"].update(
        stage="eval",
        mode="cooperative",
        warmup=False,
        bootstrap_from_legacy=False,
    )
    actions = int(config.robot.actions_dim)
    model = PPOTFInternalMoEActor(
        module["obs_schema"],
        module,
        actions,
        float(config.algo.config.init_noise_std),
        obs_example=sample_actor_obs(module["obs_schema"], actions),
    )
    model.load_state_dict(source_state, strict=True)
    model.eval()
    return model, config


def calibration_parameters(student, *, train_gate):
    """Select new router/gate identities before an optimizer can be created."""
    router_ids = {
        id(parameter)
        for layer in student.actor_module.layers
        if hasattr(layer, "added_router")
        for parameter in layer.added_router.parameters()
    }
    gate_ids = {
        id(parameter)
        for layer in student.actor_module.layers
        if getattr(layer, "added_admission", None) is not None
        for parameter in layer.added_admission.parameters()
    }
    if not router_ids or (train_gate and not gate_ids):
        raise ValueError(
            "Calibration requires explicit added router/gate state"
        )
    names = dict(student.named_parameters())
    if any(
        parameter.requires_grad and id(parameter) not in router_ids | gate_ids
        for parameter in names.values()
    ):
        raise ValueError(
            "Calibration found trainable original/expert parameters"
        )
    allowed = router_ids | (gate_ids if train_gate else set())
    selected = {
        name: parameter
        for name, parameter in names.items()
        if id(parameter) in allowed
    }
    if any(not parameter.requires_grad for parameter in selected.values()):
        raise ValueError(
            "Calibration router/gate parameters must be trainable"
        )
    return selected


def fit(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    gate_weight = getattr(args, "gate_weight", 0.0)
    if (
        not 1 <= args.steps <= 2000
        or not 1 <= args.batch_size <= 64
        or not 0 < args.learning_rate <= 0.01
        or not math.isfinite(args.route_weight)
        or args.route_weight < 0
        or not math.isfinite(gate_weight)
        or gate_weight < 0
    ):
        raise ValueError("CPU calibration budget or loss setting is invalid")
    if gate_weight > 0:
        source, config = load_source(
            args.checkpoint.resolve(), require_admission=True
        )
        source.actor_module.set_expansion(
            "cooperative", stage="eval", admission_mode="open"
        )
    else:
        source, config = load_source(args.checkpoint.resolve())
    baseline = actor_state(args.baseline.resolve())
    for key, value in baseline.items():
        if key not in source.state_dict() or not torch.equal(
            source.state_dict()[key], value
        ):
            raise ValueError(
                f"Stage A changed original checkpoint tensor: {key}"
            )
    old = copy.deepcopy(source)
    old.actor_module.set_expansion("legacy", stage="eval")
    student = copy.deepcopy(source)
    if gate_weight > 0:
        student.actor_module.set_expansion(
            "cooperative", stage="B", admission_mode="learned"
        )
    else:
        student.actor_module.set_expansion("cooperative", stage="B")
    before = {
        key: value.clone() for key, value in student.state_dict().items()
    }
    manifest = json.loads(args.manifest.read_text())
    context, dimension = (
        int(source.actor_module.max_ctx_len),
        int(source.flat_obs_dim),
    )
    if not 1 <= args.stride <= context:
        raise ValueError(
            "Replay stride must be positive and no greater than context"
        )
    roles = {entry["role"] for entry in manifest["recordings"]}
    if not {"old", "new"}.issubset(roles):
        raise ValueError(
            "Need explicit ordinary-action replay and new-action recordings"
        )
    banks = {0: [], 1: []}
    lineage = []
    for entry in manifest["recordings"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        windows = read_ordered_recording(
            path,
            entry["role"],
            context,
            args.stride,
            entry.get("new_frame_intervals"),
        )
        for window in windows:
            banks[window[1]].append(window)
        lineage.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": entry["role"],
                "windows": len(windows),
                "new_frame_intervals": entry.get("new_frame_intervals"),
            }
        )
    if any(not bank for bank in banks.values()):
        raise ValueError(
            "Stage B requires ordinary-action negatives "
            "and new-action observations"
        )
    selected_parameters = calibration_parameters(
        student, train_gate=gate_weight > 0
    )
    optimizer = torch.optim.Adam(
        selected_parameters.values(), lr=args.learning_rate
    )
    generator = torch.Generator().manual_seed(args.seed)
    metrics = []
    for step in range(args.steps):
        windows = []
        for bank in banks.values():
            indices = torch.randint(
                len(bank), (max(1, args.batch_size // 2),), generator=generator
            )
            windows.extend(bank[int(index)] for index in indices)
        raw, mask, labels = pack_windows(windows, dimension, context)
        obs = student._normalize_actor_obs(raw, update=False)
        teacher = targets(old, source, obs, mask)
        loss, record = preservation_loss(
            student,
            obs,
            mask,
            labels,
            teacher,
            args.route_weight,
            gate_weight,
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Stage B loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selected_parameters.values(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            metrics.append(
                {"step": step + 1, "loss": float(loss.detach()), **record}
            )
    state = student.state_dict()
    changed = [
        key for key in before if not torch.equal(before[key], state[key])
    ]
    if any(key not in selected_parameters for key in changed):
        raise AssertionError("Stage B changed a frozen tensor")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "model_route.pt"
    shutil.copy2(args.checkpoint, checkpoint)
    actor_dir = checkpoint.with_suffix("") / "actor"
    actor_dir.mkdir(parents=True)
    save_file(
        {key: value.detach().contiguous() for key, value in state.items()},
        str(actor_dir / "model.safetensors"),
    )
    shutil.copytree(
        args.checkpoint.with_suffix("") / "critic",
        checkpoint.with_suffix("") / "critic",
    )
    config.modules.actor.internal_expansion.stage = "B"
    config.modules.actor.internal_expansion.mode = "cooperative"
    config.modules.actor.internal_expansion.warmup = False
    config.modules.actor.internal_expansion.bootstrap_from_legacy = False
    if gate_weight > 0:
        config.modules.actor.internal_expansion.admission.mode = "learned"
    config.algo.config.checkpoint_load_mode = "finetune"
    OmegaConf.save(config, args.output_dir / "config.yaml")
    report = {
        "stage": "B",
        "device": "cpu",
        "source_M_new": str(args.checkpoint.resolve()),
        "baseline_M_old": str(args.baseline.resolve()),
        "steps": args.steps,
        "gate_weight": gate_weight,
        "changed_tensors": changed,
        "all_experts_and_old_state_preserved": True,
        "windows_per_class": {key: len(bank) for key, bank in banks.items()},
        "recording_lineage": lineage,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "accepted": False,
        "scope": (
            "Ordered offline two-sided behavior and route calibration only. "
            "Metadata is evaluation/finetune-only; source PPO optimizer state "
            "is not a Stage B optimizer resume. "
            "Closed-loop old/new/transition "
            "evaluation and independent holdouts remain mandatory."
        ),
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--gate-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261023)
    print(json.dumps(fit(parser.parse_args()), indent=2))
