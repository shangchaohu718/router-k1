"""CPU-only gate/replay audit using real weights and existing observations."""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from holomotion.src.motion_tracking.incremental_task import tensor_state_hash
from tools.booster_k1_retarget.audit_internal_moe_g0_g2 import (
    HISTORICAL,
    digest,
    run,
)

CONFIG = (
    "training/motion_tracking/motion_tracking_booster_k1_moonwalk_admission"
)


def auxiliary_probe(source):
    """Differentiate real auxiliary heads without any optimizer update."""
    model = copy.deepcopy(source)
    model.actor_module.set_expansion(
        "cooperative", stage="A", warmup=True, admission_mode="open"
    )
    model.train()
    assert model.aux_state_pred_enabled
    before = tensor_state_hash(model.state_dict())
    generator = torch.Generator().manual_seed(20261023)
    raw = torch.randn(3, 9, model.flat_obs_dim, generator=generator)
    x = model._normalize_actor_obs(raw, update=False)
    actions = torch.randn(
        3, 9, model.actor_module.output_dim, generator=generator
    )
    with torch.no_grad():
        model.actor_module.reset_kv_cache(3, "cpu")
        sampled = torch.stack(
            [model.actor_module.single_step_mu(x[:, t]) for t in range(9)],
            dim=1,
        )
    # This is the actual wrapper method PPO calls for log-probability and
    # frozen auxiliary heads; the heads may pass gradients to new experts.
    mu, sigma, logp, _, predictions = model.sequence_forward_logp(
        x, actions, None
    )
    torch.testing.assert_close(mu, sampled, atol=0, rtol=0)
    assert predictions and all(
        torch.isfinite(v).all() for v in predictions.values()
    )
    loss = mu.square().mean() + sum(
        v.square().mean() for v in predictions.values()
    )
    loss.backward()
    layer = model.actor_module.layers[0]
    gradients = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in layer.named_parameters()
        if parameter.requires_grad
    }
    assert gradients and all(value > 0 for value in gradients.values())
    assert all(
        p.grad is None for p in model.parameters() if not p.requires_grad
    )
    assert tensor_state_hash(model.state_dict()) == before
    return {
        "native_auxiliary_predictions": sorted(predictions),
        "mu_max_abs_error_vs_native_sampling": float(
            (mu.detach() - sampled).abs().max()
        ),
        "sigma_and_logp_finite": bool(
            torch.isfinite(sigma).all() and torch.isfinite(logp).all()
        ),
        "trainable_gradient_l1": gradients,
        "frozen_parameters_have_no_gradients": True,
        "all_state_tensors_unchanged": True,
        "optimizer_steps": 0,
        "scope": (
            "Actual 11999 architecture and heads; "
            "synthetic CPU differentiation only"
        ),
    }


@torch.no_grad()
def admission_replay(models, recording):
    legacy = models["legacy"]
    guarded = copy.deepcopy(models["fresh_expanded"])
    layer = guarded.actor_module.layers[0]
    historical = models["historical_single_critic_stageA"]
    # Deliberate audit-only transplantation of the four trained v1 added-bank
    # tensors into an untrained v2 wrapper. This is not checkpoint migration,
    # a resumable trained bundle, or new isolated-task skill evidence.
    transplanted = []
    for name in (
        "added_gate_up_proj",
        "added_down_proj",
        "added_router.weight",
        "added_router.bias",
    ):
        source = historical.actor_module.layers[0].state_dict()[name]
        layer.state_dict()[name].copy_(source)
        transplanted.append(name)
    guarded.actor_module.set_expansion(
        "cooperative", stage="eval", warmup=False, admission_mode="closed"
    )
    # Closed admission must exclude a bank even if its score dominates.
    layer.added_router.bias.fill_(1e6)
    frozen_before = {
        k: v.clone()
        for k, v in guarded.state_dict().items()
        if ".added_" not in k and not k.endswith("expansion_version")
    }
    online, replayed, online_routes, replay_routes = [], [], [], []
    with np.load(recording, allow_pickle=False) as data:
        raw = torch.from_numpy(data["observations"].copy())
        boundaries = np.flatnonzero(data["episode_positions"] == 0).tolist()
        assert boundaries and boundaries[0] == 0
        boundaries.append(len(raw))
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            legacy.reset_kv_cache(1, "cpu")
            means, routes = [], []
            for row in raw[begin:end]:
                x = legacy._normalize_actor_obs(row[None], update=False)
                means.append(legacy.actor_module.single_step_mu(x))
                routes.append(
                    legacy.actor_module.layers[0]
                    .last_topk_indices[0, 0]
                    .clone()
                )
            online.append(torch.cat(means).numpy())
            online_routes.append(torch.stack(routes).numpy())
            # E=1 matches the recorded sampling layout. Absolute RoPE/cache
            # wraps are preserved for the complete episode, not re-windowed.
            x = guarded._normalize_actor_obs(
                raw[None, begin:end], update=False
            )
            result = guarded.actor_module.sequence_mu(x)
            replayed.append(result[0].numpy())
            replay_routes.append(layer.last_topk_indices[0].numpy())
            assert not layer.last_admission_mask.any()
            print(
                f"CPU same-layout closed-gate replay {end}/{len(raw)}",
                flush=True,
            )
        online, replayed = np.concatenate(online), np.concatenate(replayed)
        online_routes, replay_routes = (
            np.concatenate(online_routes),
            np.concatenate(replay_routes),
        )
        gpu_error = float(np.abs(online - data["mu"]).max())
        replay_error = float(np.abs(replayed - online).max())
        gpu_route_mismatch = int(
            np.any(
                np.sort(online_routes, axis=-1)
                != np.sort(data["selected_experts"], axis=-1),
                axis=-1,
            ).sum()
        )
        closed_route_mismatch = int(
            np.any(online_routes != replay_routes, axis=-1).sum()
        )
        new_frames = int((replay_routes >= 8).any(-1).sum())
        sigma_error = float(
            np.abs(
                guarded._sigma_like(torch.from_numpy(replayed)).numpy()
                - data["sigma"]
            ).max()
        )
    frozen_after = {key: guarded.state_dict()[key] for key in frozen_before}
    assert tensor_state_hash(frozen_before) == tensor_state_hash(frozen_after)
    assert replay_error == 0 and closed_route_mismatch == 0 and new_frames == 0
    assert gpu_error < 5e-5 and gpu_route_mismatch == 0 and sigma_error < 1e-6
    with torch.enable_grad():
        auxiliary = auxiliary_probe(models["fresh_expanded"])
    return {
        "frames": len(raw),
        "episodes": len(boundaries) - 1,
        "recording_sha256": digest(recording),
        "scope": (
            "Existing observations; E=1 CPU native sampling vs cached replay; "
            "no new simulation"
        ),
        "legacy_replay_passed": True,
        "legacy_cached_vs_recorded_gpu_max_abs_error": gpu_error,
        "legacy_cached_top2_mismatch_frames": gpu_route_mismatch,
        "legacy_batched_replay_passed": False,
        "unrestricted_batching": (
            "Not supported; canonical E/order/chunk required; "
            "previous near-tie failure remains valid"
        ),
        "same_layout_cached_recomputation_passed": True,
        "closed_gate_max_abs_error_vs_native": replay_error,
        "closed_gate_top2_mismatch_frames": closed_route_mismatch,
        "closed_gate_frames_selecting_added": new_frames,
        "sigma_max_abs_error_vs_recorded": sigma_error,
        "old_actor_state_tensors_preserved": len(frozen_before),
        "adversarial_added_score_bias": 1e6,
        "audit_only_bank_transplant": {
            "source": str(HISTORICAL),
            "source_sha256": digest(HISTORICAL),
            "tensors": transplanted,
            "resumable_training_checkpoint": False,
            "saved_checkpoint_uses_this_transplant": False,
        },
        "automatic_gate_accuracy_validated": False,
        "continuous_stand_slide_stand_validated": False,
        "auxiliary_wrapper_probe": auxiliary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    output = parser.parse_args().output_dir.resolve()
    run(output, config_name=CONFIG, replay_fn=admission_replay)
    report = json.loads((output / "audit.json").read_text())
    print(
        json.dumps(
            {
                "output": str(output),
                "same_layout_passed": report["recorded_old_history_replay"][
                    "same_layout_cached_recomputation_passed"
                ],
                "automatic_gate_accuracy_validated": False,
                "training_started": False,
            },
            indent=2,
        )
    )
