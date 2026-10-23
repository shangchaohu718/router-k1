"""Compare actual GPU cached rollouts with ordered CPU context recomputation."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.booster_k1_retarget.fit_internal_moe_router import (
    load_source,
    pack_windows,
    read_ordered_recording,
)


def verify(checkpoint, recording, route):
    torch.set_num_threads(2)
    model, _ = load_source(checkpoint)
    model.actor_module.set_expansion(route, stage="eval")
    context, dimension = model.actor_module.max_ctx_len, model.flat_obs_dim
    windows = read_ordered_recording(recording, "new", context, stride=1)
    means, sigmas, selections = [], [], []
    with torch.no_grad():
        for start in range(0, len(windows), 16):
            raw, mask, _ = pack_windows(
                windows[start : start + 16], dimension, context
            )
            obs = model._normalize_actor_obs(raw, update=False)
            mean = model.actor_module.sequence_mu(obs, attn_mask=mask)[:, -1]
            means.append(mean.numpy())
            sigmas.append(model._sigma_like(mean).numpy())
            selections.append(
                model.actor_module.layers[0].last_topk_indices[:, -1].numpy()
            )
    with np.load(recording, allow_pickle=False) as trace:
        mu_error = np.abs(np.concatenate(means) - trace["mu"])
        sigma_error = np.abs(np.concatenate(sigmas) - trace["sigma"])
        mismatch = int(
            np.any(
                np.sort(np.concatenate(selections), axis=1)
                != np.sort(trace["selected_experts"], axis=1),
                axis=1,
            ).sum()
        )
    maximum, mean = float(mu_error.max()), float(mu_error.mean())
    sigma_max = float(sigma_error.max())
    return {
        "checkpoint": str(checkpoint),
        "recording": str(recording),
        "mode": route,
        "frames": len(windows),
        "context": context,
        "mean_max_abs_error": maximum,
        "mean_mean_abs_error": mean,
        "sigma_max_abs_error": sigma_max,
        "top2_set_mismatch_frames": mismatch,
        "passed": bool(
            np.isfinite(mu_error).all()
            and maximum < 5e-5
            and sigma_max < 1e-6
            and mismatch == 0
        ),
        "scope": "Actual raw observations and frozen normalization; GPU cached online means vs CPU fixed-context ordered replay. Numerical cross-backend tolerance, not bitwise parity or task success.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument(
        "--route", choices=["legacy", "cooperative"], required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.recording, args.route)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
