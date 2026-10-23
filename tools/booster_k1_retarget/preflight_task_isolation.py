"""Actual model_11999 CPU checks; never launches a simulator or RL job."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate.utils import load_state_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import PPOCritic
from holomotion.src.modules.incremental_critic import FrozenLegacySlidingCritic
from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor
from holomotion.src.motion_tracking.incremental_task import (
    SLIDING_TERMS,
    TASK_KEY,
    config_hash,
    implementation_contract,
    select_reward_profile,
    tensor_state_hash,
)
from tools.booster_k1_retarget.preflight_internal_moe import (
    audit as actor_audit,
)
from tools.booster_k1_retarget.preflight_sliding_router import (
    actor_config,
    sample_actor_obs,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    "training/motion_tracking/"
    "motion_tracking_booster_k1_moonwalk_task_isolated"
)


def sample_critic_obs(schema, actions, bodies, batch=4):
    """Audited K1 term dimensions; fail on unknown terms."""
    dims = {
        "critic_ref_dof_pos_cur": actions,
        "critic_global_anchor_diff": 9,
        "critic_ref_motion_cur_heading_aligned_root_pos": 3,
        "critic_ref_motion_cur_heading_aligned_root_rot6d": 6,
        "critic_ref_motion_cur_heading_aligned_root_lin_vel": 3,
        "critic_ref_motion_cur_heading_aligned_root_ang_vel": 3,
        "critic_rel_robot_root_lin_vel": 3,
        "critic_rel_robot_root_ang_vel": 3,
        "critic_global_robot_bodylink_lin_vel_flat": bodies * 3,
        "critic_global_robot_bodylink_ang_vel_flat": bodies * 3,
        "critic_root_rel_robot_bodylink_pos_flat": bodies * 3,
        "critic_root_rel_robot_bodylink_rot_mat_flat": bodies * 6,
        "critic_dof_pos": actions,
        "critic_dof_vel": actions,
        "critic_last_action": actions,
        "critic_ref_dof_pos_fut": actions,
        "critic_ref_root_height_fut": 1,
        "critic_ref_motion_fut_heading_aligned_root_pos": 3,
        "critic_ref_motion_fut_heading_aligned_root_rot6d": 6,
        "critic_ref_motion_fut_heading_aligned_root_lin_vel": 3,
        "critic_ref_motion_fut_heading_aligned_root_ang_vel": 3,
    }
    td = TensorDict({}, batch_size=[batch])
    for group in schema.values():
        length = int(group["seq_len"])
        for name in group["terms"]:
            dim = dims[name.split("/")[-1]]
            shape = (batch, dim) if length == 1 else (batch, length, dim)
            td.set(tuple(name.split("/")), torch.randn(shape))
    td.set(TASK_KEY, torch.ones(batch, 1, dtype=torch.int64))
    return td


def check(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    with initialize_config_dir(
        config_dir=str(ROOT / "holomotion/config"), version_base=None
    ):
        cfg = compose(
            config_name=CONFIG,
            overrides=[
                "experiment_name=cpu_task_isolation",
                "headless=true",
                "num_processes=1",
                "main_process=true",
                "process_id=0",
                "motion_h5_path=null",
                f"experiment_save_dir={output}",
            ],
        )
    source = ROOT / cfg.checkpoint
    old_cfg = OmegaConf.load(source.parent / "config.yaml")
    reward_cfg = OmegaConf.to_container(cfg.rewards, resolve=True)
    spec = OmegaConf.to_container(cfg.incremental_task, resolve=True)
    new_reward = select_reward_profile(reward_cfg)
    reward_cfg["_config"]["incremental_task"]["task"] = "legacy"
    old_reward = select_reward_profile(reward_cfg)
    assert old_reward == OmegaConf.to_container(old_cfg.rewards, resolve=True)
    reward_cfg[SLIDING_TERMS[0]]["weight"] *= 100
    assert old_reward == select_reward_profile(reward_cfg)
    assert config_hash(new_reward) == spec["sliding_reward_hash"]

    contracts = (
        "robot.dof_names",
        "robot.actions_dim",
        "robot.actuators",
        "robot.init_state.default_joint_angles",
        "terminations",
        "domain_rand",
        "env.config.simulation",
        "modules.actor.obs_schema",
        "modules.critic.obs_schema",
    )
    for path in contracts:
        current, previous = (
            OmegaConf.select(cfg, path),
            OmegaConf.select(old_cfg, path),
        )
        if OmegaConf.is_config(current):
            current = OmegaConf.to_container(current, resolve=True)
            previous = OmegaConf.to_container(previous, resolve=True)
        assert current == previous, path
    critic_cfg = OmegaConf.to_container(cfg.modules.critic, resolve=True)
    schema = critic_cfg["obs_schema"]
    example = sample_critic_obs(
        schema, int(cfg.robot.actions_dim), int(cfg.robot.num_bodies)
    )
    critic = FrozenLegacySlidingCritic(schema, critic_cfg, obs_example=example)
    old_critic_cfg = OmegaConf.to_container(
        old_cfg.modules.critic, resolve=True
    )
    old = PPOCritic(schema, old_critic_cfg, obs_example=example)
    source_state = {
        k.removeprefix("_orig_mod."): v
        for k, v in load_state_dict(
            str(source.with_suffix("") / "critic/model.safetensors")
        ).items()
    }
    assert tensor_state_hash(source_state) == spec["legacy_critic_state_hash"]
    old.load_state_dict(source_state, strict=True)
    # Exercise the actual PPO loader and strict sidecar migration, without
    # constructing the algorithm's simulator-owning __init__.
    actor_cfg = actor_config(cfg)
    actor = PPOTFInternalMoEActor(
        actor_cfg["obs_schema"],
        actor_cfg,
        int(cfg.robot.actions_dim),
        float(cfg.algo.config.init_noise_std),
        obs_example=sample_actor_obs(
            actor_cfg["obs_schema"], int(cfg.robot.actions_dim)
        ),
    )
    algo = PPOTF.__new__(PPOTF)
    algo.actor, algo.critic = actor, critic
    algo.incremental_task_spec = spec
    algo.config = cfg.algo.config
    algo.is_main_process, algo.is_offline_eval, algo.device = (
        False,
        False,
        "cpu",
    )
    algo.env_config = cfg.env
    algo.is_distributed, algo.log_dir = False, None
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
    assert algo.incremental_task_spec == spec
    algo.load(str(source))
    assert critic.sliding.assembler.output_dim == 843
    optimizer = torch.optim.AdamW(
        [p for p in critic.parameters() if p.requires_grad], lr=1e-5
    )
    before = tensor_state_hash(critic.sliding.state_dict())
    for _ in range(10):
        critic.train()
        optimizer.zero_grad(set_to_none=True)
        pred = critic(example, update_obs_norm=True)["values"]
        (pred - torch.randn_like(pred)).square().mean().backward()
        optimizer.step()
        assert (
            tensor_state_hash(critic.legacy.state_dict())
            == spec["legacy_critic_state_hash"]
        )
    assert tensor_state_hash(critic.sliding.state_dict()) != before
    critic.task = "legacy"
    example[TASK_KEY].zero_()
    expected = old(example, update_obs_norm=False)["values"]
    actual = critic(example, update_obs_norm=True, forward_chunk_size=2)[
        "values"
    ]
    # Chunked GEMM may differ; compare the identical unchunked path exactly.
    unchunked = critic(example, update_obs_norm=True)["values"]
    torch.testing.assert_close(unchunked, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actor_result = actor_audit()
    result = {
        "status": "cpu_contracts_passed_no_training",
        "simulation_run": False,
        "training_run": False,
        "automation": "PAUSED",
        "config": CONFIG,
        "actual_ppo_checkpoint_loader_passed": True,
        "critic_input_dim": 843,
        "legacy_critic_tensors_unchanged": len(source_state),
        "synthetic_cpu_critic_optimizer_steps": 10,
        "legacy_critic_max_abs_error": float(
            (unchunked - expected).detach().abs().max()
        ),
        "legacy_critic_chunked_max_abs_error": float(
            (actual - expected).detach().abs().max()
        ),
        "old_reward_equals_11999": True,
        "old_reward_immune_to_new_coefficient": True,
        "unchanged_contracts": list(contracts),
        "task_contract": spec,
        "actor_regression": actor_result,
        "implementation_contract": implementation_contract(),
        "not_validated": [
            "IsaacLab execution",
            "mixed-task PPO",
            "continuing-task truncation",
            "automatic old-task retention",
            "stand-slide-stand closed loop",
        ],
    }
    OmegaConf.save(cfg, str(output / "resolved_config.yaml"), resolve=True)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    check(parser.parse_args().output_dir.resolve())
