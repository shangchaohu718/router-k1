"""Compare saved frozen branch tensors with the original general checkpoint."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from accelerate.utils import load_state_dict


def actor_state(checkpoint):
    folder = checkpoint.with_suffix("") / "actor"
    candidates = [folder / "model.safetensors", folder / "pytorch_model.bin"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(folder)
    state = {
        k.removeprefix("_orig_mod."): v
        for k, v in load_state_dict(str(path)).items()
    }
    return state


def verify(checkpoint, original, gate_reference=None):
    baseline = actor_state(original)
    candidate = actor_state(checkpoint)
    changed, missing, expert_changed = [], [], []
    for key, value in baseline.items():
        dest = (
            key.replace("actor_module.", "actor_module.general.", 1)
            if key.startswith("actor_module.")
            else key
        )
        if dest not in candidate:
            missing.append(dest)
        elif not torch.equal(value, candidate[dest]):
            changed.append(key)
        if key.startswith("actor_module."):
            expert_key = key.replace(
                "actor_module.", "actor_module.specialist.", 1
            )
            if expert_key in candidate and not torch.equal(
                value, candidate[expert_key]
            ):
                expert_changed.append(key)
    gate_changed = []
    if gate_reference:
        previous = actor_state(gate_reference)
        for key in previous:
            if key.startswith("actor_module.gate."):
                if key not in candidate or not torch.equal(
                    candidate[key], previous[key]
                ):
                    gate_changed.append(key)
    return {
        "checkpoint": str(checkpoint),
        "baseline": str(original),
        "general_tensors_compared": len(baseline),
        "general_changed_tensors": changed,
        "missing_tensors": missing,
        "general_norm_sigma_bitwise_preserved": not (changed or missing),
        "specialist_changed_tensor_count": len(expert_changed),
        "gate_changed_tensors": gate_changed,
        "gate_comparison_performed": gate_reference is not None,
        "passed": not (changed or missing or gate_changed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate-reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.checkpoint, args.baseline, args.gate_reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
