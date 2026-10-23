#!/usr/bin/env python3
"""CPU checkpoint preflight. Never starts Isaac or training."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from accelerate.utils import load_state_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.sliding_actor import PPOTFFrozenBaseSlidingActor

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = (
    "training/motion_tracking/"
    "motion_tracking_booster_k1_moonwalk_frozen_router"
)


def compose_config():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "holomotion/config"), version_base=None
    ):
        return compose(config_name=CONFIG_NAME)


def actor_config(config):
    actor = OmegaConf.to_container(config.modules.actor, resolve=True)
    for name in (
        "noise_std_type",
        "min_sigma",
        "max_sigma",
        "fix_sigma",
        "aux_state_pred",
        "aux_router_command_recon",
        "aux_router_switch_penalty",
        "aux_router_future_recon",
        "moe_load_balance",
        "inactive_expert_margin_to_topk",
        "selected_expert_margin_to_unselected",
    ):
        if name in config.algo.config:
            value = config.algo.config[name]
            actor[name] = (
                OmegaConf.to_container(value, resolve=True)
                if OmegaConf.is_config(value)
                else value
            )
    return actor


def sample_actor_obs(schema, num_actions, batch_size=2):
    """Synthetic inputs for the existing flat K1 observation contract only."""
    dims = {
        "actor_ref_gravity_projection_cur": 3,
        "actor_ref_base_linvel_cur": 3,
        "actor_ref_base_angvel_cur": 3,
        "actor_ref_dof_pos_cur": num_actions,
        "actor_ref_root_height_cur": 1,
        "actor_ref_robot_yaw_error_sin_cos": 2,
        "actor_projected_gravity": 3,
        "actor_rel_robot_root_ang_vel": 3,
        "actor_dof_pos": num_actions,
        "actor_dof_vel": num_actions,
        "actor_last_action": num_actions,
        "actor_ref_dof_pos_fut": num_actions,
        "actor_ref_root_height_fut": 1,
        "actor_ref_gravity_projection_fut": 3,
        "actor_ref_base_linvel_fut": 3,
        "actor_ref_base_angvel_fut": 3,
        "actor_ref_future_yaw_delta_sin_cos": 2,
        "actor_ref_future_root_ori_robot_frame_6d": 6,
    }
    result = TensorDict({}, batch_size=[batch_size])
    for group in schema.values():
        length = int(group.get("seq_len", 1))
        for term in group["terms"]:
            leaf = term.rsplit("/", 1)[-1]
            if leaf not in dims:
                raise ValueError(f"Preflight has no audited shape for {term}")
            shape = (
                (batch_size, length, dims[leaf])
                if length != 1
                else (batch_size, dims[leaf])
            )
            result.set(tuple(term.split("/")), torch.randn(shape))
    return result


def check():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    config = compose_config()
    module_config = actor_config(config)
    schema = module_config["obs_schema"]
    actions = int(config.robot.actions_dim)
    sample = sample_actor_obs(schema, actions)
    checkpoint = REPO_ROOT / str(config.checkpoint)
    actor_dir = checkpoint.with_suffix("") / "actor"
    source_weights = actor_dir / "model.safetensors"
    if not source_weights.is_file():
        source_weights = actor_dir / "pytorch_model.bin"
    if not checkpoint.is_file() or not source_weights.is_file():
        raise FileNotFoundError(f"Incomplete general checkpoint: {checkpoint}")
    if not (checkpoint.with_suffix("") / "critic").is_dir():
        raise FileNotFoundError(
            "General checkpoint critic directory is missing"
        )
    roots = []
    for entry in config.train_hdf5_roots:
        root = REPO_ROOT / (
            str(entry.root) if OmegaConf.is_dict(entry) else str(entry)
        )
        if not root.is_dir() or not any(root.rglob("*.h5")):
            raise FileNotFoundError(f"Missing training HDF5 root: {root}")
        roots.append(str(root))
    state = load_state_dict(str(source_weights))
    state = {
        key.removeprefix("_orig_mod."): value for key, value in state.items()
    }
    source_config_path = checkpoint.parent / "config.yaml"
    source_config = OmegaConf.load(source_config_path)
    original_config = actor_config(source_config)
    if original_config["obs_schema"] != schema:
        raise ValueError(
            "Prototype observation schema differs from the original run"
        )
    kwargs = dict(
        obs_schema=schema,
        num_actions=actions,
        init_noise_std=float(config.algo.config.init_noise_std),
        obs_example=sample,
    )
    original = PPOTFRefRouterActor(
        module_config_dict=original_config,
        **{
            **kwargs,
            "init_noise_std": float(source_config.algo.config.init_noise_std),
        },
    )
    original.load_state_dict(state, strict=True)
    candidate = PPOTFFrozenBaseSlidingActor(
        module_config_dict=module_config, **kwargs
    )
    candidate.load_state_dict(state, strict=True)
    original.eval()
    candidate.train()
    candidate.actor_module.set_routing_mode("base")
    original.reset_kv_cache(2, "cpu")
    candidate.reset_kv_cache(2, "cpu")
    baseline_buffers = {
        name: value.clone()
        for name, value in candidate.obs_normalizer.state_dict().items()
    }
    max_error = 0.0
    with torch.no_grad():
        for _ in range(5):
            sample = sample_actor_obs(schema, actions)
            expected = original(
                sample, mode="inference", update_obs_norm=False
            )
            actual = candidate(sample, mode="inference", update_obs_norm=True)
            torch.testing.assert_close(
                actual["mu"], expected["mu"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual["sigma"], expected["sigma"], rtol=0, atol=0
            )
            max_error = max(
                max_error, float((actual["mu"] - expected["mu"]).abs().max())
            )
    for name, before in baseline_buffers.items():
        torch.testing.assert_close(
            candidate.obs_normalizer.state_dict()[name], before, rtol=0, atol=0
        )
    general = candidate.actor_module.general
    assert not any(param.requires_grad for param in general.parameters())
    candidate.actor_module.set_routing_mode("specialist")
    sequence = torch.stack(
        [sample_actor_obs(schema, actions) for _ in range(3)], dim=1
    )
    with torch.no_grad():
        output = candidate(
            sequence,
            actions=torch.zeros(2, 3, actions),
            mode="sequence_logp",
            update_obs_norm=False,
        )
    assert output["mu"].shape == (2, 3, actions)
    assert torch.isfinite(output["actions_log_prob"]).all()
    if bool(module_config.get("aux_state_pred", {}).get("enabled", False)):
        assert "aux_base_lin_vel_loc" in output.keys()
    return {
        "passed": True,
        "training_started": False,
        "simulator_started": False,
        "validation": (
            "CPU synthetic observations; "
            "no robot rollout or task-quality claim"
        ),
        "config": CONFIG_NAME,
        "checkpoint": str(checkpoint),
        "actor_weights": str(source_weights),
        "actor_weights_sha256": hashlib.sha256(
            source_weights.read_bytes()
        ).hexdigest(),
        "original_run_config": str(source_config_path),
        "training_hdf5_roots": roots,
        "actor_obs_dim": candidate.flat_obs_dim,
        "actions": actions,
        "strict_general_checkpoint_load": True,
        "base_mu_sigma_bitwise_equal": True,
        "base_max_abs_error": max_error,
        "frozen_normalizer_unchanged": True,
        "native_aux_sequence_output_valid": True,
        "general_parameters": sum(p.numel() for p in general.parameters()),
        "trainable_specialist_parameters": sum(
            p.numel()
            for p in candidate.actor_module.specialist.parameters()
            if p.requires_grad
        ),
        "gate_parameters": sum(
            p.numel() for p in candidate.actor_module.gate.parameters()
        ),
        "stage": "specialist only; gate frozen",
        "onnx_supported": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, help="Optional JSON report destination"
    )
    args = parser.parse_args()
    report = check()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
