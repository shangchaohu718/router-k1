"""CPU-only actual-checkpoint audit for internal fine-expert expansion."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from accelerate.utils import load_state_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor
from tools.booster_k1_retarget.preflight_sliding_router import (
    actor_config,
    sample_actor_obs,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    "training/motion_tracking/motion_tracking_booster_k1_moonwalk_internal_moe"
)


def audit():
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        cfg = compose(config_name=CONFIG)
    checkpoint = ROOT / str(cfg.checkpoint)
    source_config = OmegaConf.load(checkpoint.parent / "config.yaml")
    for key in (
        "robot.dof_names",
        "robot.actions_dim",
        "robot.actuators",
        "robot.init_state.default_joint_angles",
    ):
        current, previous = (
            OmegaConf.select(cfg, key),
            OmegaConf.select(source_config, key),
        )
        if OmegaConf.is_config(current):
            current = OmegaConf.to_container(current, resolve=True)
            previous = OmegaConf.to_container(previous, resolve=True)
        assert current == previous, key
    weights = checkpoint.with_suffix("") / "actor/model.safetensors"
    if not weights.is_file():
        weights = checkpoint.with_suffix("") / "actor/pytorch_model.bin"
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in load_state_dict(str(weights)).items()
    }
    module_cfg = actor_config(cfg)
    original_cfg = actor_config(source_config)
    assert module_cfg["obs_schema"] == original_cfg["obs_schema"]
    schema, actions = module_cfg["obs_schema"], int(cfg.robot.actions_dim)
    kwargs = dict(
        obs_schema=schema,
        num_actions=actions,
        init_noise_std=float(cfg.algo.config.init_noise_std),
        obs_example=sample_actor_obs(schema, actions),
    )
    old = PPOTFRefRouterActor(module_config_dict=original_cfg, **kwargs).eval()
    new = PPOTFInternalMoEActor(module_config_dict=module_cfg, **kwargs)
    old.load_state_dict(state, strict=True)
    new.load_state_dict(state, strict=True)
    layer = new.actor_module.layers[0]
    assert torch.equal(layer.added_gate_up_proj, layer.gate_up_proj[:2])
    assert torch.equal(layer.added_down_proj, layer.down_proj[:2])
    original_buffers = {
        key: value.clone()
        for key, value in layer.named_buffers()
        if key != "added_assignment_counts"
    }
    trainable = {
        name: p.numel()
        for name, p in new.named_parameters()
        if p.requires_grad
    }
    assert all(".added_" in name for name in trainable)
    new.train()
    seq = torch.stack(
        [sample_actor_obs(schema, actions) for _ in range(5)], dim=1
    )
    output = new(
        seq,
        actions=torch.randn(2, 5, actions),
        mode="sequence_logp",
        update_obs_norm=True,
    )
    loss = -output["actions_log_prob"].mean()
    loss.backward()
    gradient_l1 = {
        name: float(p.grad.abs().sum()) if p.grad is not None else 0.0
        for name, p in new.named_parameters()
        if p.requires_grad
    }
    assert layer.added_down_proj.grad.abs().sum() > 0
    assert layer.added_router.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(
        [p for p in new.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.1,
    )
    optimizer.step()
    for name, value in state.items():
        assert torch.equal(value, new.state_dict()[name]), name
    for name, value in original_buffers.items():
        assert torch.equal(value, dict(layer.named_buffers())[name]), name
    new.actor_module.set_expansion("legacy")
    old.reset_kv_cache(2, "cpu")
    new.reset_kv_cache(2, "cpu")
    max_error = 0.0
    with torch.no_grad():
        for step in range(70):
            if step in (11, 39):
                old.clear_env_cache(torch.tensor([1]))
                new.clear_env_cache(torch.tensor([1]))
            obs = sample_actor_obs(schema, actions)
            expected = old(obs, mode="inference", update_obs_norm=False)
            actual = new(obs, mode="inference", update_obs_norm=True)
            torch.testing.assert_close(
                actual["mu"], expected["mu"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual["sigma"], expected["sigma"], atol=0, rtol=0
            )
            max_error = max(
                max_error, float((actual["mu"] - expected["mu"]).abs().max())
            )
    # Strict expanded checkpoint restoration with bootstrap disabled.
    module_cfg["internal_expansion"]["bootstrap_from_legacy"] = False
    restored = PPOTFInternalMoEActor(module_config_dict=module_cfg, **kwargs)
    restored.load_state_dict(new.state_dict(), strict=True)
    files = [
        "holomotion/src/modules/internal_moe.py",
        "holomotion/src/modules/network_modules.py",
        "holomotion/src/modules/agent_modules.py",
        "holomotion/src/algo/ppo_tf.py",
        "holomotion/config/training/motion_tracking/motion_tracking_booster_k1_moonwalk_internal_moe.yaml",
        "tests/test_internal_moe.py",
        "tools/booster_k1_retarget/preflight_internal_moe.py",
        "tools/booster_k1_retarget/sliding_router_job.py",
        "tools/booster_k1_retarget/evaluate_internal_moe.py",
        "tools/booster_k1_retarget/verify_internal_moe_checkpoint.py",
        "tools/booster_k1_retarget/fit_internal_moe_router.py",
        "tests/test_internal_moe_replay.py",
        "holomotion/scripts/training/train_motion_tracking.sh",
        "holomotion/src/algo/ppo.py",
    ]
    files.extend(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "holomotion/config").rglob("*.yaml"))
        if str(path.relative_to(ROOT)) not in files
    )
    return {
        "architecture": "internal_moe",
        "passed": True,
        "ready_for_pilot": False,
        "scope": (
            "CPU synthetic histories with actual model_11999; "
            "no simulation or skill claim"
        ),
        "config": CONFIG,
        "checkpoint": str(checkpoint),
        "actor_weights": str(weights),
        "actor_weights_sha256": hashlib.sha256(
            weights.read_bytes()
        ).hexdigest(),
        "original_state_tensors_preserved": len(state),
        "legacy_cached_steps": 70,
        "legacy_mu_max_abs_error": max_error,
        "one_actor_one_attention_one_cache": True,
        "old_fine_experts": layer.num_fine_experts,
        "added_fine_experts": layer.num_added_experts,
        "frozen_original_parameters": sum(p.numel() for p in old.parameters()),
        "trainable_added_parameters": trainable,
        "gradient_l1": gradient_l1,
        "file_sha256": {
            file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
            for file in files
        },
        "training_started": False,
        "simulator_started": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "file_sha256"
            },
            indent=2,
        )
    )
