"""Fit only the upper gate on CPU using recorded normalized reference features.

Ordinary-motion holdouts are grouped by motion identity. Sliding positives
currently come from one synthetic clip: their temporal holdout does not prove
generalization to unseen sliding motions.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from tools.booster_k1_retarget.verify_sliding_checkpoint import actor_state


def goal_only_columns(config):
    from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
    from tools.booster_k1_retarget.preflight_sliding_router import (
        actor_config,
        sample_actor_obs,
    )

    module = actor_config(config)
    schema = module["obs_schema"]
    with torch.random.fork_rng(devices=[]):
        example = sample_actor_obs(schema, int(config.robot.actions_dim))
    indices = PPOTFRefRouterActor.infer_router_feature_indices(
        schema,
        example,
        router_obs_terms=module["router_obs_terms"],
    )
    excluded = set(
        PPOTFRefRouterActor.infer_router_feature_indices(
            schema,
            example,
            router_obs_terms=[
                "unified/actor_ref_future_root_ori_robot_frame_6d"
            ],
        )
    )
    return [
        column for column, index in enumerate(indices) if index in excluded
    ]


def group_name(name):
    return str(name).split("__start_", 1)[0].split("__frames_", 1)[0]


def load_samples(paths):
    rows = []
    for source_id, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            if "schema_version" not in data or int(data["schema_version"]) < 2:
                raise ValueError(
                    "Gate fitting requires schema-v2 recordings with actual "
                    "per-environment clip assignments; legacy group labels "
                    "can become stale after shorter clips finish"
                )
            x = data["features"].astype(np.float32)
            y = data["labels"].astype(np.float32)
            keys = np.array([group_name(k) for k in data["motion_keys"]])
        if x.ndim != 2 or y.shape != (len(x),) or keys.shape != y.shape:
            raise ValueError(f"Invalid sample shapes: {path}")
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(
                "Gate data must have finite inputs and binary intent labels"
            )
        rows.append((x, y, keys, np.full(len(x), source_id)))
    return tuple(np.concatenate([row[i] for row in rows]) for i in range(4))


def split_samples(y, keys, sources):
    ordinary = sorted(
        set(keys[y == 0]), key=lambda k: hashlib.sha256(k.encode()).hexdigest()
    )
    if len(ordinary) < 5 or int((y == 1).sum()) < 30:
        raise ValueError(
            "Need at least five ordinary-motion groups and 30 sliding samples"
        )
    held = max(1, int(np.ceil(len(ordinary) * 0.2)))
    sets = {
        "validation": set(ordinary[:held]),
        "test": set(ordinary[held : held * 2]),
        "train": set(ordinary[held * 2 :]),
    }
    masks = {
        part: (y == 0) & np.isin(keys, list(names))
        for part, names in sets.items()
    }
    # Preserve temporal blocks per recording and motion; never random-split
    # neighboring positive frames into all three partitions.
    for source_id in np.unique(sources):
        for key in np.unique(keys[y == 1]):
            indices = np.flatnonzero(
                (sources == source_id) & (keys == key) & (y == 1)
            )
            a, b = int(len(indices) * 0.6), int(len(indices) * 0.8)
            # Purge 32 frames around boundaries, beyond the 10-frame future
            # reference window. The same underlying clip is still shared.
            for part, selected in (
                ("train", indices[: max(0, a - 32)]),
                ("validation", indices[a + 32 : max(a + 32, b - 32)]),
                ("test", indices[b + 32 :]),
            ):
                masks[part][selected] = True
    for part, mask in masks.items():
        if len(np.unique(y[mask])) != 2:
            raise ValueError(f"Partition {part} does not contain both classes")
    return masks, sets


def scores(probability, labels, threshold):
    positive, negative = labels == 1, labels == 0
    return {
        "positive_samples": int(positive.sum()),
        "negative_samples": int(negative.sum()),
        "sliding_recall": float((probability[positive] >= threshold).mean()),
        "ordinary_false_positive_rate": float(
            (probability[negative] >= threshold).mean()
        ),
    }


def fold_gate_standardization(gate, mean, scale):
    """Fold a training-only affine transform into the first gate layer."""
    with torch.no_grad():
        weight = gate[0].weight.clone()
        gate[0].weight.copy_(weight / scale)
        gate[0].bias.sub_((weight * (mean / scale)).sum(-1))


def fit(args):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if not 1 <= args.steps <= 1000:
        raise ValueError("This CPU gate experiment is limited to 1000 updates")
    x, y, keys, sources = load_samples(args.samples)
    masks, groups = split_samples(y, keys, sources)
    state = actor_state(args.checkpoint)
    config = OmegaConf.load(args.checkpoint.parent / "config.yaml")
    zero_columns = goal_only_columns(config) if args.goal_only else []
    if zero_columns:
        x[:, zero_columns] = 0
    gate_state = {
        k.removeprefix("actor_module.gate."): v
        for k, v in state.items()
        if k.startswith("actor_module.gate.")
    }
    hidden, features = gate_state["0.weight"].shape
    if x.shape[1] != features:
        raise ValueError(
            "Recorded feature contract does not match the saved gate"
        )
    gate = nn.Sequential(
        nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, 1)
    )
    gate.load_state_dict(gate_state, strict=True)
    feature_mean = np.zeros(features, dtype=np.float32)
    feature_scale = np.ones(features, dtype=np.float32)
    if args.standardize:
        positive = x[masks["train"] & (y == 1)]
        negative = x[masks["train"] & (y == 0)]
        feature_mean = (positive.mean(0) + negative.mean(0)) * 0.5
        second = ((positive**2).mean(0) + (negative**2).mean(0)) * 0.5
        feature_scale = np.sqrt(np.maximum(second - feature_mean**2, 0)).clip(
            0.1
        )
    raw_x = x.copy()
    x = (x - feature_mean) / feature_scale
    data, targets = torch.from_numpy(x), torch.from_numpy(y)
    pos = torch.from_numpy(np.flatnonzero(masks["train"] & (y == 1)))
    neg = torch.from_numpy(np.flatnonzero(masks["train"] & (y == 0)))
    validation = torch.from_numpy(np.flatnonzero(masks["validation"]))
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    best_loss, best_state = float("inf"), None
    for step in range(args.steps):
        batch = torch.cat(
            [
                pos[torch.randint(len(pos), (64,))],
                neg[torch.randint(len(neg), (64,))],
            ]
        )
        loss = F.binary_cross_entropy_with_logits(
            gate(data[batch]).squeeze(-1), targets[batch]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                losses = F.binary_cross_entropy_with_logits(
                    gate(data[validation]).squeeze(-1),
                    targets[validation],
                    reduction="none",
                )
                val_labels = targets[validation]
                value = float(
                    0.5
                    * (
                        losses[val_labels == 0].mean()
                        + losses[val_labels == 1].mean()
                    )
                )
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(gate.state_dict())
    gate.load_state_dict(best_state)
    with torch.no_grad():
        probability = gate(data).sigmoid().squeeze(-1).numpy()
    fold_gate_standardization(
        gate, torch.from_numpy(feature_mean), torch.from_numpy(feature_scale)
    )
    with torch.no_grad():
        folded_probability = (
            gate(torch.from_numpy(raw_x)).sigmoid().squeeze(-1).numpy()
        )
    folding_error = float(np.max(np.abs(probability - folded_probability)))
    if folding_error > 2e-5:
        raise ValueError(
            "Folded gate preprocessing is not numerically equivalent"
        )
    best_state = copy.deepcopy(gate.state_dict())
    # Choose on validation only. Test false positives remain a separate result.
    threshold = min(
        0.999999,
        max(
            0.5,
            float(probability[masks["validation"] & (y == 0)].max()) + 1e-4,
        ),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "device": "cpu",
        "zero_feature_columns": zero_columns,
        "standardization": {
            "enabled": args.standardize,
            "statistics": "Class-balanced training partition only",
            "scale_floor": 0.1,
            "runtime": (
                "Folded into gate first-layer weights; "
                "old normalizer unchanged"
            ),
            "maximum_probability_folding_error": folding_error,
        },
        "threshold": threshold,
        "validation_balanced_bce": best_loss,
        "ordinary_groups": {
            part: sorted(names) for part, names in groups.items()
        },
        "partitions": {
            part: scores(probability[mask], y[mask], threshold)
            for part, mask in masks.items()
        },
        "positive_holdout": (
            "Temporal blocks with 32-frame boundary purges from the same "
            "synthetic sliding clip; not an unseen-motion test"
        ),
        "split_status": (
            "Development diagnostics. Previously inspected motion groups "
            "must not be treated as a fresh final test set."
        ),
        "sample_files_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.samples
        },
        "controller_tensors_changed": [],
    }
    for key, value in best_state.items():
        state["actor_module.gate." + key] = value
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_checkpoint = args.output_dir / "model_gate.pt"
    actor_dir = output_checkpoint.with_suffix("") / "actor"
    actor_dir.mkdir(parents=True)
    save_file(
        state, str(actor_dir / "model.safetensors"), metadata={"format": "pt"}
    )
    (actor_dir.parent / "critic").symlink_to(
        args.checkpoint.resolve().with_suffix("") / "critic",
        target_is_directory=True,
    )
    shutil.copy2(args.checkpoint, output_checkpoint)
    config.modules.actor.sliding_router.zero_feature_columns = zero_columns
    config.modules.actor.sliding_router.mode = "hard"
    config.modules.actor.sliding_router.threshold = threshold
    config.modules.actor.sliding_router.bootstrap_from_general = False
    config.modules.actor.sliding_router.train_gate = False
    config.modules.actor.sliding_router.train_specialist = False
    OmegaConf.save(config, args.output_dir / "config.yaml")
    original = actor_state(args.checkpoint)
    changed = [
        key
        for key in original
        if not key.startswith("actor_module.gate.")
        and not torch.equal(original[key], state[key])
    ]
    report["controller_tensors_changed"] = changed
    assert not changed, changed
    report["output_checkpoint"] = str(output_checkpoint)
    args.output_dir.joinpath("fit_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--goal-only", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
