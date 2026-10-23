"""CPU G0-G2 audit and replay of existing recordings. Never launches jobs."""

from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.incremental_critic import FrozenLegacySlidingCritic
from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor
from holomotion.src.motion_tracking.incremental_evaluation import (
    evaluation_task_spec,
    stage_registered_evaluation,
)
from holomotion.src.motion_tracking.incremental_task import (
    implementation_contract,
    tensor_state_hash,
)
from holomotion.src.motion_tracking.task_contract import (
    build_task_spec,
    reference_manifest,
    registered_evaluation_specs,
)
from tools.booster_k1_retarget.fit_internal_moe_router import (
    load_source,
    pack_windows,
    read_ordered_recording,
)
from tools.booster_k1_retarget.preflight_sliding_router import (
    actor_config,
    sample_actor_obs,
)
from tools.booster_k1_retarget.preflight_task_isolation import (
    sample_critic_obs,
)
from tools.booster_k1_retarget.verify_sliding_checkpoint import actor_state

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    "training/motion_tracking/"
    "motion_tracking_booster_k1_moonwalk_task_isolated"
)
OLD_RECORDING = (
    ROOT
    / "test_outputs/internal_moe_eval_legacy_old_20261023_a"
    / "ordered_observations.npz"
)
NEW_RECORDING = (
    ROOT
    / "test_outputs/internal_moe_eval_stageA_slide_20261023_c"
    / "ordered_observations.npz"
)
HISTORICAL = (
    ROOT / "test_outputs/internal_moe_stage_A_20261023_b/run/model_199.pt"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_manifest(module, optimizer=None):
    grouped = (
        {
            id(p): i
            for i, group in enumerate(optimizer.param_groups)
            for p in group["params"]
        }
        if optimizer
        else {}
    )
    result = {}
    for kind, entries in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in entries:
            result[name] = {
                "kind": kind,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_state_hash({name: value}),
                "requires_grad": bool(value.requires_grad),
                "numel": value.numel(),
                "optimizer_group": grouped.get(id(value)),
            }
    return result


def load_zero_with_ppo(cfg, actor, critic, checkpoint, *, offline=False):
    """Exercise strict native load/resume; never construct a simulator."""
    cfg = copy.deepcopy(cfg)
    cfg.incremental_task.bootstrap_from_legacy = False
    cfg.modules.actor.internal_expansion.bootstrap_from_legacy = False
    cfg.algo.config.checkpoint_load_mode = "resume"
    algo = PPOTF.__new__(PPOTF)
    algo.actor, algo.critic = actor, critic
    algo.config, algo.env_config = cfg.algo.config, cfg.env
    algo.is_main_process, algo.is_offline_eval = False, offline
    algo.is_distributed, algo.log_dir, algo.device = False, None, "cpu"
    command = SimpleNamespace(
        set_runtime_distributed_context=lambda **_: None,
        setup_dumping_dir=lambda _: None,
    )
    algo.env = SimpleNamespace(
        config=cfg.env.config,
        _env=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _: command)
        ),
    )
    algo.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model,
        process_index=0,
        num_processes=1,
    )
    algo._setup_configs()
    algo.actor_optimizer = torch.optim.AdamW(
        actor.optimizer_parameter_groups(adamw=True),
        lr=algo.actor_learning_rate,
    )
    algo.critic_optimizer = torch.optim.AdamW(
        critic.optimizer_parameters(),
        lr=algo.critic_learning_rate,
    )
    algo._set_optimizer_learning_rates()
    algo.load(str(checkpoint))
    return algo


@torch.no_grad()
def replay(models, recording):
    context = models["legacy"].actor_module.max_ctx_len
    windows = read_ordered_recording(recording, "old", context)
    means = {name: [] for name in models}
    selections = {name: [] for name in models}
    for start in range(0, len(windows), 16):
        raw, mask, _ = pack_windows(windows[start : start + 16], 506, context)
        for name, model in models.items():
            x = model._normalize_actor_obs(raw, update=False)
            means[name].append(
                model.actor_module.sequence_mu(x, attn_mask=mask)[:, -1]
                .cpu()
                .numpy()
            )
            selections[name].append(
                model.actor_module.layers[0]
                .last_topk_indices[:, -1]
                .cpu()
                .numpy()
            )
        if start % 1024 == 0:
            print(f"CPU old-history replay {start}/{len(windows)}", flush=True)
    with np.load(recording, allow_pickle=False) as data:
        observed, sigma, keys = data["mu"], data["sigma"], data["motion_keys"]
        means = {name: np.concatenate(value) for name, value in means.items()}
        selections = {
            name: np.concatenate(value) for name, value in selections.items()
        }
        result = {
            "frames": len(windows),
            "scope": "Same recorded histories; no new closed-loop simulation",
        }
        initial_error = np.abs(means["legacy"] - observed).max(-1)
        initial_route_mismatch = np.any(
            np.sort(selections["legacy"], axis=-1)
            != np.sort(data["selected_experts"], axis=-1),
            axis=-1,
        )
        result["legacy_batched_max_abs_error"] = float(initial_error.max())
        result["legacy_batched_top2_mismatch_frames"] = int(
            initial_route_mismatch.sum()
        )
        result["legacy_batched_replay_passed"] = bool(
            (initial_error < 5e-5).all() and not initial_route_mismatch.any()
        )
        retry = np.flatnonzero((initial_error > 5e-5) | initial_route_mismatch)
        result["single_history_rechecked_frames"] = retry.tolist()
        # Diagnostic fallback only. Preserve the failed batch result above;
        # never use it to bypass PPO's pre-update recomputation guard.
        for index in retry:
            raw, mask, _ = pack_windows([windows[index]], 506, context)
            for name, model in models.items():
                mu = model.actor_module.sequence_mu(
                    model._normalize_actor_obs(raw, False), attn_mask=mask
                )
                means[name][index] = mu[0, -1].numpy()
                selections[name][index] = (
                    model.actor_module.layers[0]
                    .last_topk_indices[0, -1]
                    .numpy()
                )
        reference = means["legacy"]
        result["legacy_vs_recorded_gpu_max_abs_error"] = float(
            np.abs(reference - observed).max()
        )
        result["legacy_replay_passed"] = (
            result["legacy_vs_recorded_gpu_max_abs_error"] < 5e-5
        )
        legacy = models["legacy"]
        legacy.reset_kv_cache(1, "cpu")
        cached = []
        cached_routes = []
        for index, row in enumerate(data["observations"]):
            if int(data["episode_positions"][index]) == 0:
                legacy.clear_env_cache(None)
            x = legacy._normalize_actor_obs(torch.from_numpy(row[None]), False)
            cached.append(legacy.actor_module.single_step_mu(x).numpy())
            cached_routes.append(
                legacy.actor_module.layers[0].last_topk_indices[0, 0].numpy()
            )
        cached = np.concatenate(cached)
        result["legacy_cached_vs_recorded_gpu_max_abs_error"] = float(
            np.abs(cached - observed).max()
        )
        result["legacy_cached_top2_mismatch_frames"] = int(
            np.any(
                np.sort(np.asarray(cached_routes), axis=-1)
                != np.sort(data["selected_experts"], axis=-1),
                axis=-1,
            ).sum()
        )
        for name in models:
            delta = means[name] - reference
            new_slots = selections[name] >= 8
            kl = (0.5 * (delta / sigma) ** 2).sum(-1)
            result[name] = {
                "mean_abs_action_delta": float(np.abs(delta).mean()),
                "max_abs_action_delta": float(np.abs(delta).max()),
                "mean_KL_legacy_to_current_sum_actions": float(kl.mean()),
                "frames_selecting_new": int(new_slots.any(-1).sum()),
                "new_assignment_fraction": float(new_slots.mean()),
                "per_clip": {
                    str(key): {
                        "frames": int((keys == key).sum()),
                        "max_abs_action_delta": float(
                            np.abs(delta[keys == key]).max()
                        ),
                        "new_frames": int(
                            new_slots[keys == key].any(-1).sum()
                        ),
                    }
                    for key in np.unique(keys)
                },
            }
        return result


def run(output, *, config_name=CONFIG, replay_fn=replay):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "experiment_name=g0_g2_cpu_audit",
                "headless=true",
                "num_processes=1",
                "main_process=true",
                "process_id=0",
                "motion_h5_path=null",
                f"experiment_save_dir={output}",
            ],
        )
    actor_cfg = actor_config(cfg)
    schema, actions = actor_cfg["obs_schema"], int(cfg.robot.actions_dim)
    baseline = ROOT / cfg.checkpoint
    source = actor_state(baseline)
    actor = PPOTFInternalMoEActor(
        schema,
        actor_cfg,
        actions,
        float(cfg.algo.config.init_noise_std),
        obs_example=sample_actor_obs(schema, actions),
    )
    actor.load_state_dict(source, strict=True)
    critic_cfg = OmegaConf.to_container(cfg.modules.critic, resolve=True)
    critic = FrozenLegacySlidingCritic(
        critic_cfg["obs_schema"],
        critic_cfg,
        obs_example=sample_critic_obs(
            critic_cfg["obs_schema"], actions, int(cfg.robot.num_bodies)
        ),
    )
    from accelerate.utils import load_state_dict

    cstate = {
        k.removeprefix("_orig_mod."): v
        for k, v in load_state_dict(
            str(baseline.with_suffix("") / "critic/model.safetensors")
        ).items()
    }
    critic.load_state_dict(cstate, strict=True)
    actor_optimizer = torch.optim.AdamW(
        actor.optimizer_parameter_groups(adamw=True), lr=3e-6
    )
    for group in actor_optimizer.param_groups:
        group["lr"] = 3e-6 * group["lr_multiplier"]
    critic_optimizer = torch.optim.AdamW(
        critic.optimizer_parameters(), lr=1e-5
    )
    contract = build_task_spec(
        OmegaConf.to_container(cfg.modules.actor, resolve=True),
        critic_cfg,
        OmegaConf.to_container(cfg.env.config, resolve=True),
        {
            "gamma": float(cfg.algo.config.gamma),
            "lam": float(cfg.algo.config.lam),
        },
    )
    old_roots = cfg.algo.config.incremental_legacy_evaluation.reference_roots
    legacy_cfg = stage_registered_evaluation(
        cfg,
        task="legacy",
        profile="legacy_control",
        dataset_root=ROOT / old_roots[0],
        max_frame_length=int(
            cfg.algo.config.incremental_legacy_evaluation.max_frame_length
        ),
    )
    legacy_contract = evaluation_task_spec(legacy_cfg)
    registrations = registered_evaluation_specs(
        OmegaConf.to_container(cfg.modules.actor, resolve=True),
        critic_cfg,
        OmegaConf.to_container(cfg.env.config, resolve=True),
        {
            "gamma": float(cfg.algo.config.gamma),
            "lam": float(cfg.algo.config.lam),
        },
        OmegaConf.to_container(
            cfg.algo.config.incremental_legacy_evaluation, resolve=True
        ),
    )
    registrations["sliding"] = asdict(contract)
    manifests = {
        "actor": tensor_manifest(actor, actor_optimizer),
        "critic": tensor_manifest(critic, critic_optimizer),
    }
    storage_groups = []
    for model in (actor, critic.legacy, critic.sliding):
        storage_groups.append(
            {
                t.untyped_storage().data_ptr()
                for t in [*model.parameters(), *model.buffers()]
                if t.numel()
            }
        )
    assert not any(
        storage_groups[i] & storage_groups[j]
        for i in range(3)
        for j in range(i)
    )
    # A real zero-update checkpoint, fresh empty optimizer states. CPU export
    # is serialization, not a training run; keep the initialization provenance.
    weights = output / "model_zero"
    for name, model in (("actor", actor), ("critic", critic)):
        target = weights / name
        target.mkdir(parents=True)
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in model.state_dict().items()
            },
            str(target / "model.safetensors"),
        )
    from safetensors.torch import load_file

    for name, model in (("actor", actor), ("critic", critic)):
        reloaded = copy.deepcopy(model)
        reloaded.load_state_dict(
            load_file(str(weights / name / "model.safetensors")), strict=True
        )
        assert tensor_state_hash(model.state_dict()) == tensor_state_hash(
            reloaded.state_dict()
        )
    torch.save(
        {
            **PPOTF.checkpoint_training_state(
                last_completed_iteration=-1,
                actor_learning_rate=3e-6,
                critic_learning_rate=1e-5,
            ),
            "infos": {"untrained_zero_update": True, "source": str(baseline)},
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "task_spec": asdict(contract),
            "registered_task_specs": registrations,
            "incremental_policy_version": 0,
            "incremental_task_contract": {
                k: v
                for k, v in critic_cfg["task_isolation"].items()
                if k != "bootstrap_from_legacy"
            },
            "incremental_actor_contract": actor_cfg["internal_expansion"],
            "incremental_implementation_contract": implementation_contract(),
        },
        output / "model_zero.pt",
    )
    restored_actor, restored_critic = (
        copy.deepcopy(actor),
        copy.deepcopy(critic),
    )
    resumed = load_zero_with_ppo(
        cfg, restored_actor, restored_critic, output / "model_zero.pt"
    )
    assert resumed.current_learning_iteration == 0
    assert tensor_state_hash(actor.state_dict()) == tensor_state_hash(
        restored_actor.state_dict()
    )
    assert tensor_state_hash(critic.state_dict()) == tensor_state_hash(
        restored_critic.state_dict()
    )
    assert [g["lr"] for g in resumed.actor_optimizer.param_groups] == [
        g["lr"] for g in actor_optimizer.param_groups
    ]
    original_contract = resumed.incremental_contract
    resumed.incremental_contract = replace(original_contract, gamma=0.1)
    try:
        resumed.load(str(output / "model_zero.pt"))
    except ValueError as exc:
        assert "TaskSpec" in str(exc)
    else:
        raise AssertionError("Changed task discount was accepted")
    resumed.incremental_contract = original_contract
    restored_actor.actor_module.set_expansion(
        "legacy",
        stage="eval",
        warmup=False,
        admission_mode="closed"
        if restored_actor.actor_module.layers[0].added_admission is not None
        else None,
    )
    restored_critic.task = "legacy"
    evaluated = load_zero_with_ppo(
        legacy_cfg,
        restored_actor,
        restored_critic,
        output / "model_zero.pt",
        offline=True,
    )
    assert asdict(evaluated.incremental_contract) == asdict(legacy_contract)
    OmegaConf.save(cfg, str(output / "config.yaml"), resolve=True)
    OmegaConf.save(cfg, str(output / "resolved_config.yaml"), resolve=True)
    mapping = {
        k: {"target": k, "source_sha256": tensor_state_hash({k: v})}
        for k, v in source.items()
    }
    for key in source:
        assert torch.equal(source[key], actor.state_dict()[key]), key
    fresh = copy.deepcopy(actor).eval()
    fresh.actor_module.set_expansion("cooperative", stage="eval", warmup=False)
    legacy = copy.deepcopy(actor).eval()
    legacy.actor_module.set_expansion("legacy", stage="eval", warmup=False)
    historical, _ = load_source(HISTORICAL)
    historical.actor_module.set_expansion(
        "cooperative", stage="eval", warmup=False
    )
    assert all(
        torch.equal(v, historical.state_dict()[k]) for k, v in source.items()
    )
    replay_result = replay_fn(
        {
            "legacy": legacy,
            "fresh_expanded": fresh,
            "historical_single_critic_stageA": historical,
        },
        OLD_RECORDING,
    )
    # Check that reference routing receives only existing deployable reference
    # features. Distinct features are evidence of observable differences, not
    # a fitted classifier or a guarantee of task separation.
    feature_sets = []
    with torch.no_grad():
        for recording in (OLD_RECORDING, NEW_RECORDING):
            with np.load(recording, allow_pickle=False) as data:
                raw = torch.from_numpy(data["observations"])
            x = legacy._normalize_actor_obs(raw, update=False)
            features = legacy.actor_module._compute_router_hidden(x[:, None])[
                :, 0
            ]
            feature_sets.append(features)
        nearest = torch.cdist(feature_sets[1], feature_sets[0]).min(1).values
    params = {k: v for k, v in actor.named_parameters()}
    layer = actor.actor_module.layers[0]
    donors = [
        {
            "new_index": i,
            "old_index": i % 8,
            "old_ffn_hash": tensor_state_hash(
                {
                    "gate": layer.gate_up_proj[i % 8],
                    "down": layer.down_proj[i % 8],
                }
            ),
            "new_ffn_hash": tensor_state_hash(
                {
                    "gate": layer.added_gate_up_proj[i],
                    "down": layer.added_down_proj[i],
                }
            ),
        }
        for i in range(layer.num_added_experts)
    ]
    baseline_report = (
        ROOT / "test_outputs/internal_moe_legacy_old_retention_20261023.json"
    )
    baseline_metrics = json.loads(baseline_report.read_text())
    references = {
        task: reference_manifest(
            OmegaConf.to_container(
                source_cfg.env.config.commands, resolve=True
            )
        )
        for task, source_cfg in (("sliding", cfg), ("legacy", legacy_cfg))
    }
    result = {
        "training_started": False,
        "simulation_started": False,
        "automation": "PAUSED",
        "architecture": {
            "one_actor": True,
            "moe_layers": 1,
            "old_fine": 8,
            "new_fine": 2,
            "shared": 1,
            "top_k": layer.top_k,
            "routing_score_fn": layer.routing_score_fn,
            "routing_scale": layer.routing_scale,
            "router_input": "existing reference-only features",
            "normalization": "native selected Top2 renormalization",
            "grouped_parameter_banks": True,
        },
        "repository": {
            "root": str(ROOT),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "tensordict",
                "accelerate",
                "hydra-core",
                "numpy",
            )
        },
        "baseline_checkpoint_sha256": digest(baseline),
        "task_spec": asdict(contract),
        "actor_parameters": sum(v.numel() for v in params.values()),
        "actor_trainable_parameters": sum(
            p.numel() for p in params.values() if p.requires_grad
        ),
        "old_state_tensors": len(source),
        "state_dict_roundtrip_passed": True,
        "native_ppo_resume_with_optimizers_passed": True,
        "registered_legacy_evaluation_load_passed": True,
        "changed_task_discount_load_rejected": True,
        "actor_legacy_critic_sliding_critic_storage_disjoint": True,
        "donors": donors,
        "reference_files": references,
        "reference_identifiability": {
            "new_recorded_frames": len(nearest),
            "nearest_old_router_feature_distance_min": float(nearest.min()),
            "scope": (
                "Recorded reference features differ; "
                "no automatic routing accuracy claim"
            ),
        },
        "implementation_contract": implementation_contract(),
        "historical_simulation_evidence": {
            "path": str(baseline_report),
            "sha256": digest(baseline_report),
            "clips": baseline_metrics["clip_count"],
            "steps": baseline_metrics["valid_recorded_steps"],
            "scope": (
                "Historical legacy-only closed-loop comparison; "
                "not current expanded behavior"
            ),
        },
        "recorded_old_history_replay": replay_result,
        "structural_gate_passed": (
            replay_result["legacy_replay_passed"]
            and replay_result["legacy_cached_vs_recorded_gpu_max_abs_error"]
            < 5e-5
            and replay_result["legacy_cached_top2_mismatch_frames"] == 0
        ),
        "unrestricted_batched_recomputation_passed": replay_result[
            "legacy_batched_replay_passed"
        ],
        "automatic_old_retention_proved": False,
        "not_run": [
            "new simulation",
            "training",
            "continuous stand-slide-stand",
            "G3/G4",
        ],
    }
    for name, value in (
        ("audit.json", result),
        ("param_manifest.json", manifests),
        ("migration.json", mapping),
        ("baseline_metrics.json", baseline_metrics),
        (
            "task_specs.json",
            registrations,
        ),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )
    (output / "dirty.patch").write_text(
        subprocess.check_output(
            ["git", "diff", "--binary"], cwd=ROOT, text=True
        )
    )
    snapshot = output / "source_snapshot"
    for relative in implementation_contract():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    print(
        json.dumps(
            {
                "output": str(output),
                "structural_gate_passed": result["structural_gate_passed"],
                "old_replay": replay_result,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir.resolve())
