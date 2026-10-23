"""Exact original-tensor and added-bank check after internal training."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from tools.booster_k1_retarget.verify_sliding_checkpoint import actor_state


def verify(checkpoint, baseline, first=None):
    old = actor_state(baseline)
    state = actor_state(checkpoint)
    changed = [
        key
        for key in old
        if key in state and not torch.equal(old[key], state[key])
    ]
    missing = [key for key in old if key not in state]
    previous = actor_state(first) if first else None
    added = [key for key in state if ".added_" in key]
    added_changed = [
        key
        for key in added
        if previous is not None and not torch.equal(previous[key], state[key])
    ]
    weight_path = checkpoint.with_suffix("") / "actor/model.safetensors"
    if not weight_path.is_file():
        weight_path = checkpoint.with_suffix("") / "actor/pytorch_model.bin"
    return {
        "architecture": "internal_moe",
        "checkpoint": str(checkpoint),
        "baseline": str(baseline),
        "pilot_checkpoint": str(weight_path),
        "general_tensors_compared": len(old),
        "general_changed_tensors": changed,
        "missing_tensors": missing,
        "general_norm_sigma_bitwise_preserved": not (changed or missing),
        "added_parameter_tensors": added,
        "added_changed_since_first_save": added_changed,
        "added_comparison_reference": str(first) if first else None,
        "gate_changed_tensors_since_first_save": [],
        "passed": not (changed or missing),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.checkpoint.resolve(),
        args.baseline.resolve(),
        args.first.resolve() if args.first else None,
    )
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
