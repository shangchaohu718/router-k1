# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from dataclasses import asdict
from typing import Generator

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from omegaconf import OmegaConf
from tabulate import tabulate
from tensordict import TensorDict

from holomotion.src.algo.algo_utils import PpoAuxTransition
from holomotion.src.algo.ppo import PPO
from holomotion.src.modules.agent_modules import (
    PPOCondTFActor,
    PPOCritic,
    PPOTFActor,
    PPOTFRefRouterActor,
    PPOTFRefRouterSeqActor,
    PPOTFRefRouterV3Actor,
    TensorDictAssembler,
)
from holomotion.src.modules.incremental_critic import FrozenLegacySlidingCritic
from holomotion.src.modules.internal_moe import PPOTFInternalMoEActor
from holomotion.src.modules.network_modules import GroupedMoEBlock
from holomotion.src.modules.sliding_actor import PPOTFFrozenBaseSlidingActor
from holomotion.src.motion_tracking.incremental_task import (
    TASK_IDS,
    TASK_KEY,
    config_hash,
    implementation_contract,
    select_reward_profile,
    tensor_state_hash,
    validate_task_tensor,
)
from holomotion.src.motion_tracking.task_contract import (
    build_task_spec,
    critic_cycle_stamp,
    evaluation_registry_key,
    policy_stamp,
    registered_evaluation_specs,
    tag_rollout,
    validate_episode_ledger,
    validate_rollout,
)
from holomotion.src.motion_tracking.task_gae import normalize_by_task, task_gae


class PPOTF(PPO):
    """Transformer-policy PPO with TensorDict rollout and sequence update."""

    def _set_optimizer_learning_rates(self) -> None:
        super()._set_optimizer_learning_rates()
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_learning_rate * group.get(
                "lr_multiplier", 1.0
            )

    @staticmethod
    def _select_actor_wrapper_cls(actor_cfg: dict):
        actor_type = str(actor_cfg.get("type", ""))
        use_future_cross_attn = bool(
            actor_cfg.get("use_future_cross_attn", False)
        )
        if actor_type == "InternalExpandedMoEPolicy":
            if use_future_cross_attn:
                raise ValueError(
                    "Internal expansion does not support future cross attention"
                )
            return PPOTFInternalMoEActor
        if actor_type == "FrozenBaseSlidingPolicy":
            if use_future_cross_attn:
                raise ValueError(
                    "FrozenBaseSlidingPolicy does not support "
                    "future cross attention"
                )
            return PPOTFFrozenBaseSlidingActor
        if actor_type == "ReferenceRoutedGroupedMoETransformerPolicy":
            if use_future_cross_attn:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicy does not "
                    "support use_future_cross_attn=True."
                )
            return PPOTFRefRouterActor
        if actor_type == "ReferenceRoutedGroupedMoETransformerPolicyV2":
            if use_future_cross_attn:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicyV2 does not "
                    "support use_future_cross_attn=True."
                )
            return PPOTFRefRouterSeqActor
        if actor_type == "ReferenceRoutedGroupedMoETransformerPolicyV3":
            if use_future_cross_attn:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicyV3 does not "
                    "support use_future_cross_attn=True."
                )
            return PPOTFRefRouterV3Actor
        if use_future_cross_attn:
            return PPOCondTFActor
        return PPOTFActor

    @staticmethod
    def _summarize_moe_layer_stats(moe_layers) -> dict[str, float | None]:
        if len(moe_layers) == 0 or all(
            hasattr(layer, "expansion_metrics") for layer in moe_layers
        ):
            return {
                "moe_ema_dead_expert_ratio": None,
                "moe_ema_max_expert_frac": None,
                "moe_selected_expert_margin_to_unselected": None,
            }

        def _mean_attr(attr_name: str) -> float:
            values = torch.stack(
                [
                    getattr(layer, attr_name).to(torch.float32)
                    for layer in moe_layers
                ]
            )
            return float(values.mean().item())

        return {
            "moe_ema_dead_expert_ratio": _mean_attr(
                "last_ema_dead_expert_ratio"
            ),
            "moe_ema_max_expert_frac": _mean_attr(
                "last_ema_max_expert_frac"
            ),
            "moe_selected_expert_margin_to_unselected": _mean_attr(
                "last_selected_expert_margin_to_unselected"
            ),
        }

    def _setup_configs(self):
        super()._setup_configs()
        actor_cfg = self.config.get("module_dict", {}).get("actor", {})
        critic_cfg = self.config.get("module_dict", {}).get("critic", {})
        self.incremental_task_spec = None
        self.incremental_contract = None
        if critic_cfg.get("type") == "FrozenLegacySlidingCritic":
            spec = OmegaConf.to_container(
                critic_cfg.task_isolation, resolve=True
            )
            rewards = OmegaConf.to_container(
                self.env_config.config.rewards, resolve=True
            )
            selected = select_reward_profile(rewards)
            reward_spec = rewards.get("_config", {}).get(
                "incremental_task", {}
            )
            task = spec["task"]
            if task not in TASK_IDS or reward_spec.get("task") != task:
                raise ValueError("Reward/Critic task mismatch")
            if config_hash(selected) != spec[f"{task}_reward_hash"]:
                raise ValueError("Reward/Critic recipe hash mismatch")
            if actor_cfg.get("type") != "InternalExpandedMoEPolicy":
                raise ValueError(
                    "Task isolation requires the single internal MoE Actor"
                )
            if spec.get("timeout_semantics") != "finite_horizon_terminal":
                raise ValueError(
                    "Continuing-task truncation requires final observations; unsupported"
                )
            if (
                not getattr(self, "is_offline_eval", False)
                and task != "sliding"
            ):
                raise ValueError(
                    "Isolated Stage A PPO only trains the sliding task"
                )
            if (
                spec.get("bootstrap_from_legacy")
                and self.config.get("checkpoint_load_mode") != "finetune"
            ):
                raise ValueError(
                    "Critic bootstrap requires finetune with fresh optimizers"
                )
            self.incremental_task_spec = spec
            evaluation_profile = self.config.get(
                "incremental_evaluation_profile"
            )
            if evaluation_profile is None and getattr(
                self, "is_offline_eval", False
            ):
                evaluation_profile = (
                    "legacy_control"
                    if actor_cfg.internal_expansion.mode == "legacy"
                    else "automatic"
                )
            self.incremental_contract = build_task_spec(
                OmegaConf.to_container(actor_cfg, resolve=True),
                OmegaConf.to_container(critic_cfg, resolve=True),
                OmegaConf.to_container(self.env_config.config, resolve=True),
                {
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "seed": self.config.get("seed"),
                },
                evaluation_profile=evaluation_profile,
                offline_eval=getattr(self, "is_offline_eval", False),
            )
            self._incremental_reward_terms = [
                k for k in selected if k != "_config"
            ]
            self._incremental_reward_weights = {
                k: float(selected[k]["weight"])
                for k in self._incremental_reward_terms
            }
            self._incremental_policy_version = 0
            key = evaluation_registry_key(
                task, self.incremental_contract.execution_profile
            )
            self._incremental_registered_specs = {
                key: asdict(self.incremental_contract)
            }
            if not getattr(self, "is_offline_eval", False):
                registration = self.config.get("incremental_legacy_evaluation")
                self._incremental_registered_specs.update(
                    registered_evaluation_specs(
                        OmegaConf.to_container(actor_cfg, resolve=True),
                        OmegaConf.to_container(critic_cfg, resolve=True),
                        OmegaConf.to_container(
                            self.env_config.config, resolve=True
                        ),
                        {
                            "gamma": self.gamma,
                            "lam": self.lam,
                            "seed": self.config.get("seed"),
                        },
                        OmegaConf.to_container(registration, resolve=True)
                        if registration is not None else None,
                    )
                )
        if str(actor_cfg.get("type", "")) == "InternalExpandedMoEPolicy":
            expansion = actor_cfg.get("internal_expansion", {})
            self._incremental_sequence_replay = str(
                expansion.get("sequence_replay", "vectorized")
            )
            self._incremental_replay_chunk_size = expansion.get(
                "replay_chunk_size", 0
            )
            if self._incremental_sequence_replay not in {
                "vectorized",
                "cached",
            }:
                raise ValueError("Unknown incremental sequence_replay mode")
            if (
                type(self._incremental_replay_chunk_size) is not int
                or self._incremental_replay_chunk_size < 0
            ):
                raise ValueError(
                    "replay_chunk_size must be a nonnegative integer"
                )
            self._cached_incremental_indices(self.num_envs)
            if bool(self.config.get("override_sigma", False)) or bool(
                self.config.get("export_policy", False)
            ):
                raise ValueError(
                    "Internal expansion requires override_sigma=false and export_policy=false"
                )
            if (
                expansion.get("bootstrap_from_legacy", False)
                and self.config.get("checkpoint_load_mode", "resume")
                != "finetune"
            ):
                raise ValueError(
                    "Internal legacy bootstrap requires finetune and fresh optimizer state"
                )
            if not getattr(self, "is_offline_eval", False):
                if (
                    expansion.get("stage", "A") != "A"
                    or expansion.get("mode", "cooperative") != "cooperative"
                ):
                    raise ValueError(
                        "Internal PPO currently supports Stage A cooperative new-task training only; Stage B needs ordered replay calibration"
                    )
                if (
                    "admission" in expansion
                    and expansion["admission"].get("mode", "closed") != "open"
                ):
                    raise ValueError(
                        "Internal Stage A PPO requires admission.mode='open'; "
                        "learned/closed admission can suppress "
                        "expert gradients"
                    )
        if str(actor_cfg.get("type", "")) == "FrozenBaseSlidingPolicy":
            router_cfg = actor_cfg.get("sliding_router", {})
            if bool(self.config.get("override_sigma", False)):
                raise ValueError(
                    "FrozenBaseSlidingPolicy requires override_sigma=false"
                )
            if bool(self.config.get("export_policy", False)):
                raise ValueError(
                    "FrozenBaseSlidingPolicy does not yet support ONNX export; "
                    "set export_policy=false"
                )
            if (
                bool(router_cfg.get("bootstrap_from_general", False))
                and str(self.config.get("checkpoint_load_mode", "resume"))
                != "finetune"
            ):
                raise ValueError(
                    "General-policy bootstrap requires "
                    "checkpoint_load_mode=finetune and fresh optimizer state"
                )
            mode = str(router_cfg.get("mode", "specialist"))
            if not getattr(self, "is_offline_eval", False):
                if mode not in {"specialist", "soft"}:
                    raise ValueError(
                        "PPO training supports specialist or soft routing; "
                        "base/hard/temporal_hard are evaluation modes"
                    )
                if mode == "specialist" and bool(
                    router_cfg.get("train_gate", False)
                ):
                    raise ValueError(
                        "Forced specialist routing cannot "
                        "train the gate with PPO"
                    )
                if not bool(
                    router_cfg.get("train_specialist", True)
                ) and not bool(router_cfg.get("train_gate", False)):
                    raise ValueError(
                        "Sliding PPO requires at least one trainable branch"
                    )
                if mode == "specialist" and not bool(
                    router_cfg.get("train_specialist", True)
                ):
                    raise ValueError(
                        "Forced specialist PPO requires train_specialist=true"
                    )
        aux_cfg = self.config.get("aux_state_pred", {})
        self.use_aux_state_pred: bool = bool(aux_cfg.get("enabled", False))
        self.aux_state_pred_w_base_lin_vel = float(
            aux_cfg.get("w_base_lin_vel", 0.0)
        )
        self.aux_state_pred_w_root_height = float(
            aux_cfg.get("w_root_height", 0.0)
        )
        self.aux_state_pred_w_keybody_contact = float(
            aux_cfg.get("w_keybody_contact", 0.0)
        )
        self.aux_state_pred_w_ref_keybody_rel_pos = float(
            aux_cfg.get("w_ref_keybody_rel_pos", 0.0)
        )
        self.aux_state_pred_w_robot_keybody_rel_pos = float(
            aux_cfg.get("w_robot_keybody_rel_pos", 0.0)
        )
        self.aux_state_pred_keybody_contact_names = [
            str(name) for name in aux_cfg.get("keybody_contact_names", [])
        ]
        self.aux_state_pred_keybody_rel_pos_names = [
            str(name) for name in aux_cfg.get("keybody_rel_pos_names", [])
        ]
        self.aux_state_pred_num_contact_bodies = int(
            len(self.aux_state_pred_keybody_contact_names)
        )
        self.aux_state_pred_num_keybody_bodies = int(
            len(self.aux_state_pred_keybody_rel_pos_names)
        )
        self.use_aux_root_height = bool(
            self.use_aux_state_pred and self.aux_state_pred_w_root_height > 0.0
        )
        self.aux_state_pred_min_std = float(aux_cfg.get("min_std", 1.0e-3))
        self.aux_state_pred_max_std = float(aux_cfg.get("max_std", 5.0))
        self.aux_state_pred_raycast_z_offset = float(
            aux_cfg.get("raycast_z_offset", 1.0)
        )
        self.aux_state_pred_raycast_max_dist = float(
            aux_cfg.get("raycast_max_dist", 20.0)
        )
        if self.aux_state_pred_min_std <= 0.0:
            raise ValueError("aux_state_pred.min_std must be > 0.")
        if self.aux_state_pred_max_std <= self.aux_state_pred_min_std:
            raise ValueError(
                "aux_state_pred.max_std must be > aux_state_pred.min_std."
            )
        if self.aux_state_pred_w_base_lin_vel < 0.0:
            raise ValueError("aux_state_pred.w_base_lin_vel must be >= 0.")
        if self.aux_state_pred_w_root_height < 0.0:
            raise ValueError("aux_state_pred.w_root_height must be >= 0.")
        if self.aux_state_pred_w_keybody_contact < 0.0:
            raise ValueError("aux_state_pred.w_keybody_contact must be >= 0.")
        if self.aux_state_pred_w_ref_keybody_rel_pos < 0.0:
            raise ValueError(
                "aux_state_pred.w_ref_keybody_rel_pos must be >= 0."
            )
        if self.aux_state_pred_w_robot_keybody_rel_pos < 0.0:
            raise ValueError(
                "aux_state_pred.w_robot_keybody_rel_pos must be >= 0."
            )
        if self.use_aux_root_height:
            if self.aux_state_pred_raycast_max_dist <= 0.0:
                raise ValueError(
                    "aux_state_pred.raycast_max_dist must be > 0."
                )
            if self.aux_state_pred_raycast_z_offset < 0.0:
                raise ValueError(
                    "aux_state_pred.raycast_z_offset must be >= 0."
                )
        if (
            self.aux_state_pred_w_keybody_contact > 0.0
            and self.aux_state_pred_num_contact_bodies == 0
        ):
            raise ValueError(
                "aux_state_pred.w_keybody_contact > 0 requires "
                "aux_state_pred.keybody_contact_names to be non-empty."
            )
        if (
            self.aux_state_pred_w_ref_keybody_rel_pos > 0.0
            or self.aux_state_pred_w_robot_keybody_rel_pos > 0.0
        ) and self.aux_state_pred_num_keybody_bodies == 0:
            raise ValueError(
                "aux_state_pred keybody position weights > 0 require "
                "aux_state_pred.keybody_rel_pos_names to be non-empty."
            )
        if self.use_aux_state_pred and self.command_name != "ref_motion":
            raise ValueError(
                "aux_state_pred is only supported for PPOTF motion tracking "
                "(command_name='ref_motion')."
            )
        PpoAuxTransition.SHAPE_TOKENS["C"] = (
            self.aux_state_pred_num_contact_bodies
        )
        PpoAuxTransition.SHAPE_TOKENS["K"] = (
            self.aux_state_pred_num_keybody_bodies
        )
        aux_cmd_cfg = self.config.get("aux_router_command_recon", {})
        self.use_aux_router_command_recon: bool = bool(
            aux_cmd_cfg.get("enabled", False)
        )
        self.aux_router_command_recon_weight = float(
            aux_cmd_cfg.get("weight", 0.0)
        )
        self.aux_router_command_recon_hidden_dim = int(
            aux_cmd_cfg.get("hidden_dim", 0)
        )
        self.aux_router_command_recon_term_prefix = str(
            aux_cmd_cfg.get("term_prefix", "actor_ref_")
        )
        aux_switch_cfg = self.config.get("aux_router_switch_penalty", {})
        self.use_aux_router_switch_penalty = bool(
            aux_switch_cfg.get("enabled", False)
        )
        self.aux_router_switch_penalty_weight = float(
            aux_switch_cfg.get("weight", 0.0)
        )
        self.aux_router_switch_penalty_metric = str(
            aux_switch_cfg.get("metric", "js")
        ).lower()
        self.aux_router_switch_penalty_beta = float(
            aux_switch_cfg.get("beta", 1.0)
        )
        aux_router_future_cfg = self.config.get("aux_router_future_recon", {})
        self.use_aux_router_future_recon = bool(
            aux_router_future_cfg.get("enabled", False)
        )
        self.aux_router_future_recon_weight = float(
            aux_router_future_cfg.get("weight", 0.0)
        )
        self.aux_router_future_recon_hidden_dim = int(
            aux_router_future_cfg.get("hidden_dim", 0)
        )
        self.aux_router_future_recon_huber_beta = float(
            aux_router_future_cfg.get("huber_beta", 1.0)
        )
        action_butterworth_cfg = self.config.get(
            "aux_action_mean_butterworth", {}
        )
        self.use_aux_action_mean_butterworth = bool(
            action_butterworth_cfg.get("enabled", False)
        )
        self.aux_action_mean_butterworth_weight = float(
            action_butterworth_cfg.get("weight", 0.0)
        )
        self.aux_action_mean_butterworth_sample_hz = float(
            action_butterworth_cfg.get("sample_hz", 50.0)
        )
        self.aux_action_mean_butterworth_cutoff_hz = float(
            action_butterworth_cfg.get("cutoff_hz", 3.0)
        )
        action_butterworth_order = action_butterworth_cfg.get("order", 4)
        if (
            isinstance(action_butterworth_order, bool)
            or int(action_butterworth_order) != action_butterworth_order
        ):
            raise ValueError(
                "aux_action_mean_butterworth.order must be a positive integer."
            )
        self.aux_action_mean_butterworth_order = int(
            action_butterworth_order
        )
        self.aux_action_mean_butterworth_sos = (
            self._design_butterworth_highpass_sos(
                sample_hz=self.aux_action_mean_butterworth_sample_hz,
                cutoff_hz=self.aux_action_mean_butterworth_cutoff_hz,
                order=self.aux_action_mean_butterworth_order,
            )
            if self.use_aux_action_mean_butterworth
            else ()
        )
        load_balance_cfg = self.config.get("moe_load_balance", {})
        self.use_moe_load_balance = bool(
            load_balance_cfg.get("enabled", False)
        )
        self.moe_load_balance_weight = float(
            load_balance_cfg.get("weight", 0.0)
        )
        inactive_margin_cfg = self.config.get(
            "inactive_expert_margin_to_topk", {}
        )
        self.use_inactive_expert_margin_to_topk = bool(
            inactive_margin_cfg.get("enabled", False)
        )
        self.inactive_expert_margin_to_topk_weight = float(
            inactive_margin_cfg.get("weight", 0.0)
        )
        self.inactive_expert_margin_to_topk_ratio_floor = float(
            inactive_margin_cfg.get("ratio_floor", 0.0)
        )
        orth_cfg = self.config.get("router_expert_orthogonal", {})
        self.use_router_expert_orthogonal = bool(
            orth_cfg.get("enabled", False)
        )
        self.router_expert_orthogonal_weight = float(
            orth_cfg.get("weight", 0.0)
        )
        self.router_expert_orthogonal_min_active_usage = float(
            orth_cfg.get("min_active_usage", 1.0e-3)
        )
        self.router_expert_orthogonal_eps = float(orth_cfg.get("eps", 1.0e-8))
        selected_margin_cfg = self.config.get(
            "selected_expert_margin_to_unselected", {}
        )
        self.use_selected_expert_margin_to_unselected = bool(
            selected_margin_cfg.get("enabled", False)
        )
        self.selected_expert_margin_to_unselected_weight = float(
            selected_margin_cfg.get("weight", 0.0)
        )
        self.selected_expert_margin_to_unselected_target = float(
            selected_margin_cfg.get("target", 0.0)
        )
        if self.aux_router_switch_penalty_metric not in {
            "js",
            "normed_smooth_l1",
        }:
            raise ValueError(
                "aux_router_switch_penalty.metric must be one of "
                "{'js', 'normed_smooth_l1'}, got "
                f"{self.aux_router_switch_penalty_metric!r}."
            )
        if self.aux_router_command_recon_weight < 0.0:
            raise ValueError("aux_router_command_recon.weight must be >= 0.")
        if self.aux_router_future_recon_weight < 0.0:
            raise ValueError("aux_router_future_recon.weight must be >= 0.")
        if self.aux_action_mean_butterworth_weight < 0.0:
            raise ValueError(
                "aux_action_mean_butterworth.weight must be >= 0."
            )
        if self.aux_router_switch_penalty_weight < 0.0:
            raise ValueError("aux_router_switch_penalty.weight must be >= 0.")
        if self.moe_load_balance_weight < 0.0:
            raise ValueError("moe_load_balance.weight must be >= 0.")
        if self.inactive_expert_margin_to_topk_weight < 0.0:
            raise ValueError(
                "inactive_expert_margin_to_topk.weight must be >= 0."
            )
        if not (0.0 <= self.inactive_expert_margin_to_topk_ratio_floor <= 1.0):
            raise ValueError(
                "inactive_expert_margin_to_topk.ratio_floor must be in [0, 1]."
            )
        if self.router_expert_orthogonal_weight < 0.0:
            raise ValueError("router_expert_orthogonal.weight must be >= 0.")
        if self.router_expert_orthogonal_min_active_usage < 0.0:
            raise ValueError(
                "router_expert_orthogonal.min_active_usage must be >= 0."
            )
        if self.selected_expert_margin_to_unselected_weight < 0.0:
            raise ValueError(
                "selected_expert_margin_to_unselected.weight must be >= 0."
            )
        if self.router_expert_orthogonal_eps <= 0.0:
            raise ValueError("router_expert_orthogonal.eps must be > 0.")
        if self.aux_router_switch_penalty_beta <= 0.0:
            raise ValueError("aux_router_switch_penalty.beta must be > 0.")
        if self.aux_router_future_recon_huber_beta <= 0.0:
            raise ValueError("aux_router_future_recon.huber_beta must be > 0.")
        if self.selected_expert_margin_to_unselected_target < 0.0:
            raise ValueError(
                "selected_expert_margin_to_unselected.target must be >= 0."
            )
        if (
            self.use_moe_load_balance
            and self.moe_load_balance_weight == 0.0
        ):
            logger.warning(
                "moe_load_balance.enabled=True but weight=0.0; "
                "MoE load-balance loss will have no effect."
            )
        if (
            self.use_inactive_expert_margin_to_topk
            and self.inactive_expert_margin_to_topk_weight == 0.0
        ):
            logger.warning(
                "inactive_expert_margin_to_topk.enabled=True but weight=0.0; "
                "inactive-expert margin loss will have no effect."
            )
        if (
            self.use_router_expert_orthogonal
            and not self.use_inactive_expert_margin_to_topk
        ):
            raise ValueError(
                "router_expert_orthogonal.enabled=True requires "
                "inactive_expert_margin_to_topk.enabled=True in sparse top-k MoE."
            )
        if (
            self.use_router_expert_orthogonal
            and self.router_expert_orthogonal_weight == 0.0
        ):
            logger.warning(
                "router_expert_orthogonal.enabled=True but weight=0.0; "
                "orthogonal regularization will have no effect."
            )
        if (
            self.use_selected_expert_margin_to_unselected
            and self.selected_expert_margin_to_unselected_weight == 0.0
        ):
            logger.warning(
                "selected_expert_margin_to_unselected.enabled=True but "
                "weight=0.0; selected-expert margin loss will have no effect."
            )
        if (
            self.use_aux_router_switch_penalty
            and self.aux_router_switch_penalty_weight == 0.0
        ):
            logger.warning(
                "aux_router_switch_penalty.enabled=True but weight=0.0; "
                "router switch penalty will have no effect."
            )
        if (
            self.use_aux_router_future_recon
            and self.aux_router_future_recon_weight == 0.0
        ):
            logger.warning(
                "aux_router_future_recon.enabled=True but weight=0.0; "
                "future reconstruction loss will have no effect."
            )
        if (
            self.use_aux_action_mean_butterworth
            and self.aux_action_mean_butterworth_weight == 0.0
        ):
            logger.warning(
                "aux_action_mean_butterworth.enabled=True but weight=0.0; "
                "action-mean smoothing loss will have no effect."
            )
        if (
            self.use_aux_router_command_recon
            or self.use_aux_router_switch_penalty
            or self.use_aux_router_future_recon
        ) and self.command_name != "ref_motion":
            raise ValueError(
                "aux_router_command_recon, aux_router_future_recon, and "
                "aux_router_switch_penalty are "
                "only supported for PPOTF motion tracking "
                "(command_name='ref_motion')."
            )
        self.aux_command_router_num_moe_layers = 0
        self.aux_command_router_num_fine_experts = 0
        self.aux_router_command_recon_assembler: TensorDictAssembler | None = (
            None
        )
        actor_cfg = self.config.get("module_dict", {}).get("actor", {})
        actor_type = str(actor_cfg.get("type", ""))
        if actor_type in {
            "ReferenceRoutedGroupedMoETransformerPolicyV2",
            "ReferenceRoutedGroupedMoETransformerPolicyV3",
        }:
            if self.use_aux_router_command_recon:
                raise ValueError(
                    f"{actor_type} does not support aux_router_command_recon."
                )
            unsupported_aux_weights = {
                "w_root_height": self.aux_state_pred_w_root_height,
            }
            enabled_unsupported = [
                name
                for name, value in unsupported_aux_weights.items()
                if float(value) > 0.0
            ]
            if enabled_unsupported:
                raise ValueError(
                    f"{actor_type} only supports "
                    "aux_state_pred weights for base_lin_vel, keybody_contact, "
                    "ref_keybody_rel_pos, and robot_keybody_rel_pos. Unsupported "
                    "weights: " + ", ".join(enabled_unsupported)
                )
        elif self.use_aux_router_future_recon:
            raise ValueError(
                "aux_router_future_recon requires "
                "ReferenceRoutedGroupedMoETransformerPolicyV2 or V3."
            )

    @staticmethod
    def _unwrap_obs_schema(schema: dict | None) -> dict | None:
        if schema is None:
            return None
        has_terms = any(
            isinstance(v, dict) and ("terms" in v) for v in schema.values()
        )
        if has_terms:
            return schema
        if len(schema) == 1:
            only_value = next(iter(schema.values()))
            if isinstance(only_value, dict):
                return only_value
        return schema

    @staticmethod
    def _schema_term_leaf_name(term: str) -> str:
        return str(term).split("/")[-1]

    @classmethod
    def _is_aux_command_term(cls, term: str, term_prefix: str) -> bool:
        return cls._schema_term_leaf_name(term).startswith(term_prefix)

    @classmethod
    def _build_aux_router_command_recon_schema(
        cls, actor_schema: dict, term_prefix: str
    ) -> dict:
        command_schema = {}
        for group_name, seq_cfg in actor_schema.items():
            terms = [
                str(term)
                for term in seq_cfg.get("terms", [])
                if cls._is_aux_command_term(str(term), term_prefix)
            ]
            if len(terms) == 0:
                continue
            next_seq_cfg = dict(seq_cfg)
            next_seq_cfg["terms"] = terms
            command_schema[group_name] = next_seq_cfg
        if len(command_schema) == 0:
            raise ValueError(
                "aux_router_command_recon could not find any actor command terms in "
                f"obs_schema with prefix '{term_prefix}'."
            )
        return command_schema

    @staticmethod
    def _masked_aux_keybody_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_tok: torch.Tensor,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have the same shape for keybody MSE, "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}."
            )
        if pred.ndim != 4:
            raise ValueError(
                "Keybody MSE expects [B, T, K, 3] tensors, "
                f"got pred with shape {tuple(pred.shape)}."
            )
        per_token_mse = torch.square(pred - target).mean(dim=(-1, -2))
        valid_tok = valid_tok.to(per_token_mse.dtype)
        if valid_tok.shape != per_token_mse.shape:
            raise ValueError(
                "valid_tok must match per-token keybody MSE shape, "
                f"got {tuple(valid_tok.shape)} and "
                f"{tuple(per_token_mse.shape)}."
            )
        valid_count = valid_tok.sum().clamp_min(1.0)
        return (per_token_mse * valid_tok).sum() / valid_count

    @staticmethod
    def _masked_aux_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_tok: torch.Tensor,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must share the same shape for auxiliary MSE, "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}."
            )
        if pred.ndim < 3:
            raise ValueError(
                "Auxiliary MSE expects tensors with shape [B, T, ...], "
                f"got {tuple(pred.shape)}."
            )
        reduce_dims = tuple(range(2, pred.ndim))
        per_token_mse = torch.square(pred - target).mean(dim=reduce_dims)
        valid_tok = valid_tok.to(per_token_mse.dtype)
        if valid_tok.shape != per_token_mse.shape:
            raise ValueError(
                "valid_tok must match per-token auxiliary MSE shape, got "
                f"{tuple(valid_tok.shape)} and {tuple(per_token_mse.shape)}."
            )
        valid_count = valid_tok.sum().clamp_min(1.0)
        return (per_token_mse * valid_tok).sum() / valid_count

    @staticmethod
    def _design_butterworth_highpass_sos(
        *, sample_hz: float, cutoff_hz: float, order: int
    ) -> tuple[tuple[float, ...], ...]:
        if sample_hz <= 0.0:
            raise ValueError(
                "aux_action_mean_butterworth.sample_hz must be positive."
            )
        if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * sample_hz:
            raise ValueError(
                "aux_action_mean_butterworth.cutoff_hz must be between zero "
                "and the Nyquist frequency."
            )
        if isinstance(order, bool) or order <= 0 or int(order) != order:
            raise ValueError(
                "aux_action_mean_butterworth.order must be a positive integer."
            )

        from scipy.signal import butter

        sos = butter(
            int(order),
            cutoff_hz / (0.5 * sample_hz),
            btype="highpass",
            output="sos",
        )
        sos[:, :3] /= sos[:, 3:4]
        sos[:, 4:] /= sos[:, 3:4]
        sos[:, 3] = 1.0
        return tuple(tuple(float(value) for value in row) for row in sos)

    @staticmethod
    def _action_mean_butterworth_loss(
        *,
        action_mean: torch.Tensor,
        dones: torch.Tensor,
        valid_tok: torch.Tensor,
        sos: tuple[tuple[float, ...], ...],
    ) -> torch.Tensor:
        """Filter continuous policy means and average high-frequency energy."""
        if action_mean.ndim != 3:
            raise ValueError(
                "action_mean must have shape [B, T, A], got "
                f"{tuple(action_mean.shape)}."
            )
        if dones.shape != (*action_mean.shape[:2], 1):
            raise ValueError(
                "dones must have shape [B, T, 1], got "
                f"{tuple(dones.shape)}."
            )
        if valid_tok.shape != action_mean.shape[:2]:
            raise ValueError(
                "valid_tok must have shape [B, T], got "
                f"{tuple(valid_tok.shape)}."
            )
        if len(sos) == 0:
            raise ValueError("Butterworth SOS coefficients must be non-empty.")

        signal = action_mean.float()
        batch_size, sequence_length, action_dim = signal.shape
        sos_tensor = signal.new_tensor(sos)
        num_sections = int(sos_tensor.shape[0])
        z1 = [
            signal.new_zeros(batch_size, action_dim)
            for _ in range(num_sections)
        ]
        z2 = [
            signal.new_zeros(batch_size, action_dim)
            for _ in range(num_sections)
        ]

        episode_start = torch.zeros(
            batch_size,
            sequence_length,
            device=signal.device,
            dtype=torch.bool,
        )
        episode_start[:, 0] = True
        if sequence_length > 1:
            episode_start[:, 1:] = dones[:, :-1, 0].to(torch.bool)

        penalties: list[torch.Tensor] = []
        for time_idx in range(sequence_length):
            reset = episode_start[:, time_idx, None]
            section_input = signal[:, time_idx]
            next_z1: list[torch.Tensor] = []
            next_z2: list[torch.Tensor] = []
            for section_idx in range(num_sections):
                b0, b1, b2, _, a1, a2 = sos_tensor[section_idx].unbind()
                dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2)
                steady_output = dc_gain * section_input
                steady_z1 = steady_output - b0 * section_input
                steady_z2 = b2 * section_input - a2 * steady_output
                current_z1 = torch.where(reset, steady_z1, z1[section_idx])
                current_z2 = torch.where(reset, steady_z2, z2[section_idx])
                section_output = b0 * section_input + current_z1
                next_z1.append(
                    b1 * section_input - a1 * section_output + current_z2
                )
                next_z2.append(
                    b2 * section_input - a2 * section_output
                )
                section_input = section_output
            z1 = next_z1
            z2 = next_z2
            penalties.append(section_input.square().mean(dim=-1))

        per_token_penalty = torch.stack(penalties, dim=1)
        loss_mask = valid_tok.to(per_token_penalty.dtype) * (
            ~episode_start
        ).to(per_token_penalty.dtype)
        valid_count = loss_mask.sum().clamp_min(1.0)
        return (per_token_penalty * loss_mask).sum() / valid_count

    @staticmethod
    def _masked_adjacent_router_js(
        *,
        router_features: torch.Tensor,
        valid_tok: torch.Tensor,
        num_moe_layers: int,
        num_fine_experts: int,
    ) -> torch.Tensor:
        if router_features.ndim != 3:
            raise ValueError(
                "router_features must have shape [B, T, L*E], got "
                f"{tuple(router_features.shape)}."
            )
        if valid_tok.ndim != 2:
            raise ValueError(
                "valid_tok must have shape [B, T], got "
                f"{tuple(valid_tok.shape)}."
            )
        if num_moe_layers <= 0 or num_fine_experts <= 0:
            raise ValueError(
                "num_moe_layers and num_fine_experts must be positive, got "
                f"{num_moe_layers} and {num_fine_experts}."
            )
        bsz, seq_len, feat_dim = router_features.shape
        expected_dim = num_moe_layers * num_fine_experts
        if feat_dim != expected_dim:
            raise ValueError(
                "router_features last dim must equal num_moe_layers * "
                "num_fine_experts, got "
                f"{feat_dim} vs {expected_dim}."
            )
        if valid_tok.shape != (bsz, seq_len):
            raise ValueError(
                "valid_tok shape mismatch for router temporal loss: expected "
                f"{(bsz, seq_len)}, got {tuple(valid_tok.shape)}."
            )
        if seq_len <= 1:
            return router_features.new_zeros(())

        router_p = router_features.reshape(
            bsz, seq_len, num_moe_layers, num_fine_experts
        ).to(torch.float32)
        prev_p = router_p[:, :-1]
        curr_p = router_p[:, 1:]
        mix_p = 0.5 * (prev_p + curr_p)
        eps = 1.0e-20
        prev_safe = prev_p.clamp_min(eps)
        curr_safe = curr_p.clamp_min(eps)
        mix_safe = mix_p.clamp_min(eps)
        kl_prev = (prev_p * (torch.log(prev_safe) - torch.log(mix_safe))).sum(
            dim=-1
        )
        kl_curr = (curr_p * (torch.log(curr_safe) - torch.log(mix_safe))).sum(
            dim=-1
        )
        js = 0.5 * (kl_prev + kl_curr)
        adjacent_valid = (valid_tok[:, :-1] * valid_tok[:, 1:]).to(js.dtype)
        valid_count = adjacent_valid.sum().clamp_min(1.0) * float(
            num_moe_layers
        )
        return (js * adjacent_valid.unsqueeze(-1)).sum() / valid_count

    @staticmethod
    def _masked_adjacent_router_normed_smooth_l1(
        *,
        router_temporal_features: torch.Tensor,
        valid_tok: torch.Tensor,
        num_moe_layers: int,
        num_fine_experts: int,
        beta: float = 1.0,
    ) -> torch.Tensor:
        if router_temporal_features.ndim != 3:
            raise ValueError(
                "router_temporal_features must have shape [B, T, L*E], got "
                f"{tuple(router_temporal_features.shape)}."
            )
        if valid_tok.ndim != 2:
            raise ValueError(
                "valid_tok must have shape [B, T], got "
                f"{tuple(valid_tok.shape)}."
            )
        if num_moe_layers <= 0 or num_fine_experts <= 0:
            raise ValueError(
                "num_moe_layers and num_fine_experts must be positive, got "
                f"{num_moe_layers} and {num_fine_experts}."
            )
        if beta <= 0.0:
            raise ValueError(
                f"beta must be positive for SmoothL1, got {beta}."
            )
        bsz, seq_len, feat_dim = router_temporal_features.shape
        expected_dim = num_moe_layers * num_fine_experts
        if feat_dim != expected_dim:
            raise ValueError(
                "router_temporal_features last dim must equal "
                "num_moe_layers * num_fine_experts, got "
                f"{feat_dim} vs {expected_dim}."
            )
        if valid_tok.shape != (bsz, seq_len):
            raise ValueError(
                "valid_tok shape mismatch for router temporal loss: expected "
                f"{(bsz, seq_len)}, got {tuple(valid_tok.shape)}."
            )
        if seq_len <= 1:
            return router_temporal_features.new_zeros(())

        router_logits = router_temporal_features.reshape(
            bsz, seq_len, num_moe_layers, num_fine_experts
        ).to(torch.float32)
        router_logits = router_logits - router_logits.mean(
            dim=-1, keepdim=True
        )
        router_logits = F.normalize(router_logits, p=2.0, dim=-1, eps=1.0e-5)
        prev_logits = router_logits[:, :-1]
        curr_logits = router_logits[:, 1:]
        smooth_l1 = F.smooth_l1_loss(
            curr_logits,
            prev_logits,
            reduction="none",
            beta=beta,
        ).mean(dim=(-1, -2))
        adjacent_valid = (valid_tok[:, :-1] * valid_tok[:, 1:]).to(
            smooth_l1.dtype
        )
        valid_count = adjacent_valid.sum().clamp_min(1.0)
        return (smooth_l1 * adjacent_valid).sum() / valid_count

    @staticmethod
    def _masked_aux_gaussian_nll(
        *,
        loc: torch.Tensor,
        log_std: torch.Tensor,
        target: torch.Tensor,
        valid_tok: torch.Tensor,
        min_std: float,
        max_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if loc.shape != log_std.shape or loc.shape != target.shape:
            raise ValueError(
                "loc, log_std, and target must share the same shape for "
                "Gaussian aux loss, got "
                f"{tuple(loc.shape)}, {tuple(log_std.shape)}, "
                f"{tuple(target.shape)}."
            )
        if loc.ndim < 3:
            raise ValueError(
                "Gaussian aux loss expects tensors with shape [B, T, ...], "
                f"got {tuple(loc.shape)}."
            )
        per_elem_std = torch.clamp(
            torch.exp(log_std),
            min=float(min_std),
            max=float(max_std),
        )
        reduce_dims = tuple(range(2, loc.ndim))
        per_token_nll = 0.5 * (
            torch.square((target - loc) / per_elem_std)
            + 2.0 * torch.log(per_elem_std + 1.0e-8)
        ).sum(dim=reduce_dims)
        valid_tok = valid_tok.to(per_token_nll.dtype)
        if valid_tok.shape != per_token_nll.shape:
            raise ValueError(
                "valid_tok must match per-token Gaussian loss shape, got "
                f"{tuple(valid_tok.shape)} and {tuple(per_token_nll.shape)}."
            )
        valid_count = valid_tok.sum().clamp_min(1.0)
        loss = (per_token_nll * valid_tok).sum() / valid_count
        per_token_std = per_elem_std.reshape(
            per_elem_std.shape[0], per_elem_std.shape[1], -1
        ).mean(dim=-1)
        mean_std = (per_token_std * valid_tok).sum() / valid_count
        return loss, mean_std

    @staticmethod
    def _masked_aux_huber(
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_tok: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must share the same shape for Huber aux loss, "
                f"got {tuple(pred.shape)} and {tuple(target.shape)}."
            )
        if pred.ndim < 3:
            raise ValueError(
                "Huber aux loss expects tensors with shape [B, T, ...], "
                f"got {tuple(pred.shape)}."
            )
        per_elem = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
        reduce_dims = tuple(range(2, pred.ndim))
        per_token = per_elem.mean(dim=reduce_dims)
        valid_tok = valid_tok.to(per_token.dtype)
        if valid_tok.shape != per_token.shape:
            raise ValueError(
                "valid_tok must match per-token Huber loss shape, got "
                f"{tuple(valid_tok.shape)} and {tuple(per_token.shape)}."
            )
        valid_count = valid_tok.sum().clamp_min(1.0)
        return (per_token * valid_tok).sum() / valid_count

    def _compute_aux_router_future_recon_loss(
        self,
        *,
        actor_wrapper: PPOTFActor,
        actor_out: TensorDict,
        obs_b: TensorDict,
        valid_tok: torch.Tensor,
    ) -> torch.Tensor:
        future_assembler = actor_wrapper.aux_router_future_recon_assembler
        if future_assembler is None:
            raise ValueError(
                "aux_router_future_recon is enabled but future assembler was "
                "not initialized on the actor wrapper."
            )
        aux_router_future_recon_pred = actor_out.get("aux_router_future_recon")
        bsz, seq_len = int(obs_b.batch_size[0]), int(obs_b.batch_size[1])
        future_target = future_assembler(obs_b.flatten(0, 1)).reshape(
            bsz, seq_len, -1
        )
        normalized_future_target = actor_wrapper.actor_module.normalize_aux_router_future_recon_target(
            future_target
        ).to(aux_router_future_recon_pred.dtype)
        return self._masked_aux_huber(
            pred=aux_router_future_recon_pred,
            target=normalized_future_target,
            valid_tok=valid_tok,
            beta=self.aux_router_future_recon_huber_beta,
        )

    def _compute_routed_expert_orthogonal_loss(
        self,
        moe_layer: GroupedMoEBlock,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        usage = moe_layer.last_routed_expert_usage.to(
            device=device, dtype=torch.float32
        )
        active_mask = usage > float(
            self.router_expert_orthogonal_min_active_usage
        )
        active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        active_count = torch.tensor(
            float(active_idx.numel()), device=device, dtype=torch.float32
        )
        if active_idx.numel() < 2:
            zero = torch.zeros((), device=device, dtype=dtype)
            zero_f = torch.zeros((), device=device, dtype=torch.float32)
            return zero, active_count, zero_f

        expert_vecs = moe_layer.down_proj.index_select(0, active_idx)
        expert_vecs = expert_vecs.reshape(active_idx.numel(), -1).to(
            device=device, dtype=torch.float32
        )
        expert_vecs = F.normalize(
            expert_vecs,
            p=2.0,
            dim=-1,
            eps=float(self.router_expert_orthogonal_eps),
        )
        gram = expert_vecs @ expert_vecs.transpose(0, 1)
        offdiag_mask = ~torch.eye(
            gram.shape[0], dtype=torch.bool, device=gram.device
        )
        offdiag = gram.masked_select(offdiag_mask)
        if offdiag.numel() == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            zero_f = torch.zeros((), device=device, dtype=torch.float32)
            return zero, active_count, zero_f

        orth_loss = offdiag.square().sum().to(dtype)
        mean_offdiag_similarity = offdiag.abs().mean()
        return orth_loss, active_count, mean_offdiag_similarity

    @staticmethod
    def _root_relative_body_pos_from_mixed_position_frames(
        *,
        body_pos_w: torch.Tensor,
        root_pos_env: torch.Tensor,
        root_quat_w: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> torch.Tensor:
        """Convert world-frame body positions using an env-frame root pose.

        In IsaacLab, `isaaclab_mdp.root_pos_w(env)` is already in the
        environment frame (simulator world minus `env.scene.env_origins`),
        while `robot.data.body_pos_w` stays in simulator-world coordinates.
        """
        if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
            raise ValueError(
                "body_pos_w must have shape [B, N, 3], "
                f"got {tuple(body_pos_w.shape)}."
            )
        if root_pos_env.ndim != 2 or root_pos_env.shape[-1] != 3:
            raise ValueError(
                "root_pos_env must have shape [B, 3], "
                f"got {tuple(root_pos_env.shape)}."
            )
        if root_quat_w.ndim != 2 or root_quat_w.shape[-1] != 4:
            raise ValueError(
                "root_quat_w must have shape [B, 4], "
                f"got {tuple(root_quat_w.shape)}."
            )
        if env_origins.ndim != 2 or env_origins.shape[-1] != 3:
            raise ValueError(
                "env_origins must have shape [B, 3], "
                f"got {tuple(env_origins.shape)}."
            )
        if body_pos_w.shape[0] != root_pos_env.shape[0]:
            raise ValueError(
                "Batch size mismatch between body_pos_w and root_pos_env: "
                f"{body_pos_w.shape[0]} vs {root_pos_env.shape[0]}."
            )
        if body_pos_w.shape[0] != root_quat_w.shape[0]:
            raise ValueError(
                "Batch size mismatch between body_pos_w and root_quat_w: "
                f"{body_pos_w.shape[0]} vs {root_quat_w.shape[0]}."
            )
        if body_pos_w.shape[0] != env_origins.shape[0]:
            raise ValueError(
                "Batch size mismatch between body_pos_w and env_origins: "
                f"{body_pos_w.shape[0]} vs {env_origins.shape[0]}."
            )
        body_pos_env = body_pos_w - env_origins[:, None, :]
        rel_pos_env = body_pos_env - root_pos_env[:, None, :]
        quat_vec = root_quat_w[:, None, 1:].expand_as(rel_pos_env)
        quat_real = root_quat_w[:, None, :1].expand(
            -1, rel_pos_env.shape[1], -1
        )
        t = 2.0 * torch.cross(quat_vec, rel_pos_env, dim=-1)
        return rel_pos_env - quat_real * t + torch.cross(quat_vec, t, dim=-1)

    def _setup_models_and_optimizer(self):
        sample_obs_dict = self.env.reset_all()[0]
        sample_td = self._wrap_obs_dict(sample_obs_dict)

        actor_cfg = OmegaConf.to_container(
            self.config.module_dict.actor, resolve=True
        )
        critic_cfg = OmegaConf.to_container(
            self.config.module_dict.critic, resolve=True
        )
        actor_cfg["noise_std_type"] = getattr(
            self.config, "noise_std_type", "log"
        )
        actor_cfg["min_sigma"] = getattr(self.config, "min_sigma", 0.1)
        actor_cfg["max_sigma"] = getattr(self.config, "max_sigma", 1.5)
        actor_cfg["fix_sigma"] = getattr(self.config, "fix_sigma", False)
        self._future_mask_prob = float(actor_cfg.get("future_mask_prob", 0.0))
        self._future_mask_mode = str(
            actor_cfg.get("future_mask_mode", "random_suffix")
        ).lower()
        aux_cfg = self.config.get("aux_state_pred", {})
        if isinstance(aux_cfg, dict):
            actor_cfg["aux_state_pred"] = dict(aux_cfg)
        else:
            actor_cfg["aux_state_pred"] = OmegaConf.to_container(
                aux_cfg, resolve=True
            )
        aux_cmd_cfg = self.config.get("aux_router_command_recon", {})
        if isinstance(aux_cmd_cfg, dict):
            actor_aux_cmd_cfg = dict(aux_cmd_cfg)
        else:
            actor_aux_cmd_cfg = OmegaConf.to_container(
                aux_cmd_cfg, resolve=True
            )
        aux_switch_cfg = self.config.get("aux_router_switch_penalty", {})
        if isinstance(aux_switch_cfg, dict):
            actor_aux_switch_cfg = dict(aux_switch_cfg)
        else:
            actor_aux_switch_cfg = OmegaConf.to_container(
                aux_switch_cfg, resolve=True
            )
        aux_router_future_cfg = self.config.get("aux_router_future_recon", {})
        if isinstance(aux_router_future_cfg, dict):
            actor_aux_router_future_cfg = dict(aux_router_future_cfg)
        else:
            actor_aux_router_future_cfg = OmegaConf.to_container(
                aux_router_future_cfg, resolve=True
            )
        load_balance_cfg = self.config.get("moe_load_balance", {})
        if isinstance(load_balance_cfg, dict):
            actor_load_balance_cfg = dict(load_balance_cfg)
        else:
            actor_load_balance_cfg = OmegaConf.to_container(
                load_balance_cfg, resolve=True
            )
        inactive_margin_cfg = self.config.get(
            "inactive_expert_margin_to_topk", {}
        )
        if isinstance(inactive_margin_cfg, dict):
            actor_inactive_margin_cfg = dict(inactive_margin_cfg)
        else:
            actor_inactive_margin_cfg = OmegaConf.to_container(
                inactive_margin_cfg, resolve=True
            )
        selected_margin_cfg = self.config.get(
            "selected_expert_margin_to_unselected", {}
        )
        if isinstance(selected_margin_cfg, dict):
            actor_selected_margin_cfg = dict(selected_margin_cfg)
        else:
            actor_selected_margin_cfg = OmegaConf.to_container(
                selected_margin_cfg, resolve=True
            )

        actor_schema = self._unwrap_obs_schema(
            actor_cfg.get("obs_schema", None)
        )
        critic_schema = self._unwrap_obs_schema(
            critic_cfg.get("obs_schema", None)
        )
        if actor_schema is None:
            raise ValueError(
                "PPOTF requires actor obs_schema to infer flattened obs dim."
            )
        if self.use_aux_router_command_recon:
            aux_command_schema = self._build_aux_router_command_recon_schema(
                actor_schema, self.aux_router_command_recon_term_prefix
            )
            self.aux_router_command_recon_assembler = TensorDictAssembler(
                aux_command_schema, output_mode="flat"
            )
            actor_aux_cmd_cfg["output_dim"] = int(
                self.aux_router_command_recon_assembler.infer_output_dim(
                    sample_td
                )
            )
            if self.aux_router_command_recon_hidden_dim > 0:
                actor_aux_cmd_cfg["hidden_dim"] = (
                    self.aux_router_command_recon_hidden_dim
                )
        actor_cfg["aux_router_command_recon"] = actor_aux_cmd_cfg
        actor_cfg["aux_router_future_recon"] = actor_aux_router_future_cfg
        actor_cfg["aux_router_switch_penalty"] = actor_aux_switch_cfg
        actor_cfg["moe_load_balance"] = actor_load_balance_cfg
        actor_cfg["inactive_expert_margin_to_topk"] = actor_inactive_margin_cfg
        actor_cfg["selected_expert_margin_to_unselected"] = (
            actor_selected_margin_cfg
        )
        actor_obs_dim = int(
            TensorDictAssembler(
                actor_schema, output_mode="flat"
            ).infer_output_dim(sample_td)
        )
        use_future_cross_attn = bool(
            actor_cfg.get("use_future_cross_attn", False)
        )
        actor_cls = self._select_actor_wrapper_cls(actor_cfg)
        if use_future_cross_attn:
            if "flattened_obs" not in actor_schema:
                raise ValueError(
                    "use_future_cross_attn=True requires "
                    "actor obs_schema.flattened_obs."
                )
            if "flattened_obs_fut" not in actor_schema:
                raise ValueError(
                    "use_future_cross_attn=True requires "
                    "actor obs_schema.flattened_obs_fut."
                )
            state_schema = {"flattened_obs": actor_schema["flattened_obs"]}
            future_schema = {
                "flattened_obs_fut": actor_schema["flattened_obs_fut"]
            }
            state_obs_dim = int(
                TensorDictAssembler(
                    state_schema, output_mode="flat"
                ).infer_output_dim(sample_td)
            )
            future_asm = TensorDictAssembler(future_schema, output_mode="seq")
            future_token_dim = int(future_asm.infer_output_dim(sample_td))
            future_seq_len = int(future_asm.seq_len)
            actor_cfg["state_obs_dim"] = state_obs_dim
            actor_cfg["future_token_dim"] = future_token_dim
            actor_cfg["future_seq_len"] = future_seq_len
            actor_cfg["input_dim_override"] = state_obs_dim
        else:
            actor_cfg["input_dim_override"] = actor_obs_dim

        self.actor = actor_cls(
            obs_schema=actor_schema,
            module_config_dict=actor_cfg,
            num_actions=self.num_actions,
            init_noise_std=self.config.init_noise_std,
            obs_example=sample_td,
        ).to(self.device)
        actor_module_unwrapped = self.actor.actor_module
        self._cached_incremental_indices(int(sample_td.batch_size[0]))
        self.aux_command_router_num_moe_layers = int(
            getattr(actor_module_unwrapped, "_num_moe_layers", 0)
        )
        self.aux_command_router_num_fine_experts = int(
            getattr(actor_module_unwrapped, "num_fine_experts", 0)
        )
        if (
            self.use_aux_router_switch_penalty
            and self.aux_command_router_num_moe_layers <= 0
        ):
            raise ValueError(
                "aux_router_switch_penalty requires at least one "
                "GroupedMoEBlock."
            )
        critic_cls = (
            FrozenLegacySlidingCritic
            if critic_cfg.get("type") == "FrozenLegacySlidingCritic"
            else PPOCritic
        )
        self.critic = critic_cls(
            obs_schema=critic_schema,
            module_config_dict=critic_cfg,
            obs_example=sample_td,
        ).to(self.device)

        if self.is_main_process:
            actor = self.accelerator.unwrap_model(self.actor)
            critic = self.accelerator.unwrap_model(self.critic)

            logger.info("Actor (TensorDict module):\n{!r}", actor)
            logger.info(
                "Actor keys: in_keys={} out_keys={}",
                list(actor.in_keys),
                list(actor.out_keys),
            )
            logger.info("Actor core nn module:\n{!r}", actor.actor_module)

            logger.info("Critic (TensorDict module):\n{!r}", critic)
            logger.info(
                "Critic keys: in_keys={} out_keys={}",
                list(critic.in_keys),
                list(critic.out_keys),
            )
            logger.info("Critic core nn module:\n{!r}", critic.critic_module)

            actor_params = sum(p.numel() for p in self.actor.parameters())
            critic_params = sum(p.numel() for p in self.critic.parameters())
            params_table = [
                ["Actor(Transformer)", f"{actor_params / 1.0e6:.3f}"],
                ["Critic", f"{critic_params / 1.0e6:.3f}"],
                ["Total", f"{(actor_params + critic_params) / 1.0e6:.3f}"],
            ]
            logger.info(
                "Model Summary:\n"
                + tabulate(
                    params_table,
                    headers=["Model", "Params (M)"],
                    tablefmt="simple_outline",
                )
            )

        optimizer_class = getattr(optim, self.optimizer_type)
        optimizer_kwargs = self._build_optimizer_kwargs(optimizer_class)
        if hasattr(self.actor, "optimizer_parameter_groups"):
            groups = self.actor.optimizer_parameter_groups(adamw=self.optimizer_type == "AdamW")
            for group in groups:
                group["lr"] = self.actor_learning_rate * group["lr_multiplier"]
            self.actor_optimizer = optimizer_class(groups, lr=self.actor_learning_rate, betas=(self.actor_beta1, self.actor_beta2), **optimizer_kwargs)
        elif self.optimizer_type == "AdamW":
            decay_params = []
            non_decay_params = []
            for name, p in self.actor.named_parameters():
                if not p.requires_grad:
                    continue
                if (
                    p.ndim < 2
                    or ("log_std" in name)
                    or ("bias" in name)
                    or ("norm" in name)
                ):
                    non_decay_params.append(p)
                else:
                    decay_params.append(p)
            self.actor_optimizer = optimizer_class(
                [
                    {"params": decay_params, "weight_decay": 0.01},
                    {"params": non_decay_params, "weight_decay": 0.0},
                ],
                lr=self.actor_learning_rate,
                betas=(self.actor_beta1, self.actor_beta2),
                **optimizer_kwargs,
            )
        else:
            self.actor_optimizer = optimizer_class(
                [p for p in self.actor.parameters() if p.requires_grad],
                lr=self.actor_learning_rate,
                betas=(self.actor_beta1, self.actor_beta2),
                **optimizer_kwargs,
            )
        self.critic_optimizer = optimizer_class(
            self.critic.optimizer_parameters()
            if isinstance(self.critic, FrozenLegacySlidingCritic)
            else [p for p in self.critic.parameters() if p.requires_grad],
            lr=self.critic_learning_rate,
            betas=(self.critic_beta1, self.critic_beta2),
            **optimizer_kwargs,
        )

        (
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.accelerator.prepare(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
        )

        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        if hasattr(actor_for_kv, "reset_kv_cache"):
            actor_for_kv.reset_kv_cache(self.env.num_envs, self.device)
        self._kv_reset_pending = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )
        self._rollout_future_masks = None
        self._rollout_step_idx = 0

    def _setup_data_buffers(self):
        super()._setup_data_buffers()
        self._aux_height_scanner = None
        self._aux_contact_sensor = None
        self._aux_contact_body_ids = None
        self._aux_keybody_body_ids = None
        if not self.use_aux_state_pred:
            return
        if self.use_velocity_transition:
            raise ValueError(
                "aux_state_pred is not supported with velocity "
                "tracking in PPOTF."
            )
        self.transition_cls = PpoAuxTransition
        if self.use_aux_root_height:
            if "height_scanner" not in self.env._env.scene.sensors:
                raise ValueError(
                    "aux_state_pred requires a RayCaster sensor "
                    "named 'height_scanner' "
                    "in env.scene.sensors."
                )
            height_scanner = self.env._env.scene.sensors["height_scanner"]
            height_scanner.cfg.max_distance = (
                self.aux_state_pred_raycast_max_dist
            )
            height_scanner.cfg.ray_alignment = "world"
            height_scanner.cfg.offset.pos = (
                0.0,
                0.0,
                self.aux_state_pred_raycast_z_offset,
            )
            if height_scanner.is_initialized:
                height_scanner.ray_starts[..., 2] = (
                    self.aux_state_pred_raycast_z_offset
                )
            self._aux_height_scanner = height_scanner
        if self.aux_state_pred_num_contact_bodies > 0:
            if "contact_forces" not in self.env._env.scene.sensors:
                raise ValueError(
                    "aux_state_pred.keybody_contact_names requires "
                    "a ContactSensor "
                    "named 'contact_forces' in env.scene.sensors."
                )
            contact_sensor = self.env._env.scene.sensors["contact_forces"]
            sensor_body_names = list(contact_sensor.body_names)
            body_ids = []
            for body_name in self.aux_state_pred_keybody_contact_names:
                if body_name not in sensor_body_names:
                    raise ValueError(
                        f"Body '{body_name}' not found in contact "
                        "sensor body_names."
                    )
                body_ids.append(sensor_body_names.index(body_name))
            self._aux_contact_sensor = contact_sensor
            self._aux_contact_body_ids = torch.tensor(
                body_ids, dtype=torch.long, device=self.device
            )
        if self.aux_state_pred_num_keybody_bodies > 0:
            robot_body_names = list(self.env._env.scene["robot"].body_names)
            body_ids = []
            for body_name in self.aux_state_pred_keybody_rel_pos_names:
                if body_name not in robot_body_names:
                    raise ValueError(
                        f"Body '{body_name}' not found in robot body_names."
                    )
                body_ids.append(robot_body_names.index(body_name))
            self._aux_keybody_body_ids = torch.tensor(
                body_ids, dtype=torch.long, device=self.device
            )

    def _build_transition(
        self,
        obs_td: TensorDict,
        actor_out: TensorDict,
        critic_out: TensorDict,
    ):
        if not self.use_aux_state_pred:
            return super()._build_transition(obs_td, actor_out, critic_out)

        import isaaclab.envs.mdp as isaaclab_mdp

        actions = actor_out.get("actions")
        actions_log_prob = actor_out.get("actions_log_prob")
        mu = actor_out.get("mu")
        sigma = actor_out.get("sigma")
        values = critic_out.get("values")
        zero_scalar = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
            dtype=torch.float32,
        )
        zero_scalar_bool = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        gt_base_lin_vel_b = isaaclab_mdp.base_lin_vel(self.env._env)
        if self.use_aux_root_height:
            root_pos_w = isaaclab_mdp.root_pos_w(self.env._env)
            if self._aux_height_scanner is None:
                raise RuntimeError(
                    "Aux state prediction expected "
                    "_aux_height_scanner to be initialized."
                )
            terrain_z = self._aux_height_scanner.data.ray_hits_w[:, 0, 2:3]
            env_origin_z = self.env._env.scene.env_origins[:, 2:3]
            terrain_z = torch.where(
                torch.isfinite(terrain_z), terrain_z, env_origin_z
            )
            gt_root_height_rel_terrain = root_pos_w[:, 2:3] - terrain_z
        else:
            gt_root_height_rel_terrain = torch.zeros(
                self.num_envs, 1, device=self.device, dtype=torch.float32
            )
        if self.aux_state_pred_num_contact_bodies > 0:
            if (
                self._aux_contact_sensor is None
                or self._aux_contact_body_ids is None
            ):
                raise RuntimeError(
                    "Aux keybody contact prediction expects contact sensor "
                    "and body ids to be initialized."
                )
            contact_time = self._aux_contact_sensor.data.current_contact_time[
                :, self._aux_contact_body_ids
            ]
            gt_keybody_contacts = (contact_time > 0.0).to(torch.float32)
        else:
            gt_keybody_contacts = torch.zeros(
                self.num_envs, 0, device=self.device, dtype=torch.float32
            )
        command = self.env._env.command_manager.get_term(self.command_name)
        if self.aux_state_pred_num_keybody_bodies > 0:
            if self._aux_keybody_body_ids is None:
                raise RuntimeError(
                    "Aux keybody position prediction expects body "
                    "ids to be initialized."
                )
            # Both the ref-motion command and robot asset expose bodies in
            # simulator order, so the cached robot body indices align here.
            gt_ref_keybody_rel_pos = (
                command.get_ref_motion_bodylink_rel_pos_cur()[
                    :, self._aux_keybody_body_ids, :
                ]
            )
            robot = self.env._env.scene["robot"]
            robot_keybody_global_pos = robot.data.body_pos_w[
                :, self._aux_keybody_body_ids, :
            ]
            env_origins = self.env._env.scene.env_origins
            root_pos_w = isaaclab_mdp.root_pos_w(self.env._env)
            root_quat_w = isaaclab_mdp.root_quat_w(self.env._env)
            gt_robot_keybody_rel_pos = (
                self._root_relative_body_pos_from_mixed_position_frames(
                    body_pos_w=robot_keybody_global_pos,
                    root_pos_env=root_pos_w,
                    root_quat_w=root_quat_w,
                    env_origins=env_origins,
                )
            )
        else:
            gt_ref_keybody_rel_pos = torch.zeros(
                self.num_envs, 0, 3, device=self.device, dtype=torch.float32
            )
            gt_robot_keybody_rel_pos = torch.zeros(
                self.num_envs, 0, 3, device=self.device, dtype=torch.float32
            )
        return self.transition_cls(
            obs=obs_td,
            actions=actions.detach(),
            teacher_actions=torch.zeros_like(actions),
            mu=mu.detach(),
            sigma=sigma.detach(),
            actions_log_prob=actions_log_prob[..., None].detach(),
            values=values.detach(),
            rewards=zero_scalar.clone(),
            dones=zero_scalar_bool,
            returns=zero_scalar.clone(),
            advantages=zero_scalar.clone(),
            gt_base_lin_vel_b=gt_base_lin_vel_b.detach(),
            gt_root_height_rel_terrain=gt_root_height_rel_terrain.detach(),
            gt_keybody_contacts=gt_keybody_contacts.detach(),
            gt_ref_keybody_rel_pos=gt_ref_keybody_rel_pos.detach(),
            gt_robot_keybody_rel_pos=gt_robot_keybody_rel_pos.detach(),
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _build_storage(self, obs_td: TensorDict):
        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        actor_policy = actor_for_kv.actor_module
        if bool(getattr(actor_policy, "use_future_cross_attn", False)):
            n_fut = int(getattr(actor_policy, "future_seq_len", 0))
            if n_fut <= 0:
                raise ValueError(
                    "future_seq_len must be positive when "
                    "use_future_cross_attn=True"
                )
            obs_td = obs_td.clone(recurse=False)
            obs_td.set(
                "future_mask",
                torch.ones(
                    self.env.num_envs,
                    n_fut,
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        return super()._build_storage(obs_td)

    def _sample_iteration_future_masks(self) -> torch.Tensor | None:
        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        actor_policy = actor_for_kv.actor_module
        if not bool(getattr(actor_policy, "use_future_cross_attn", False)):
            return None

        n_fut = int(getattr(actor_policy, "future_seq_len", 0))
        if n_fut <= 0:
            raise ValueError(
                "future_seq_len must be positive when "
                "use_future_cross_attn=True"
            )
        if self._future_mask_mode != "random_suffix":
            raise ValueError(
                "Unsupported future_mask_mode: "
                f"{self._future_mask_mode}. "
                "Expected 'random_suffix'."
            )
        num_steps = int(self.num_steps_per_env)
        num_envs = int(self.env.num_envs)

        keep = torch.ones(
            num_steps,
            num_envs,
            n_fut,
            dtype=torch.bool,
            device=self.device,
        )
        if bool(getattr(self, "_offline_evaluating", False)):
            return keep
        if self._future_mask_prob <= 0.0:
            return keep
        apply_mask = (
            torch.rand(num_steps, num_envs, device=self.device)
            < self._future_mask_prob
        )
        keep_len = torch.randint(
            1,
            n_fut + 1,
            (num_steps, num_envs),
            device=self.device,
        )
        full_len = torch.full(
            (num_steps, num_envs),
            n_fut,
            dtype=torch.long,
            device=self.device,
        )
        keep_len = torch.where(apply_mask, keep_len, full_len)
        token_idx = torch.arange(n_fut, device=self.device, dtype=torch.long)[
            None, None, :
        ]
        return token_idx < keep_len[:, :, None]

    def _reset_rollout_forward_state(self) -> None:
        if (
            self._cached_incremental_indices(getattr(self, "num_envs", None))
            is not None
        ):
            self._incremental_rollout_layout = (
                self.num_envs,
                self.rollout_forward_chunk_size,
            )
        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        actor_for_kv.clear_env_cache(None)
        actor_policy = actor_for_kv.actor_module
        actor_policy.reset_routing_stats()
        actor_policy.set_collect_routing_stats(True)
        self._kv_reset_pending.zero_()
        self._rollout_future_masks = self._sample_iteration_future_masks()
        self._rollout_step_idx = 0
        if getattr(self, "incremental_contract", None) is not None:
            self._incremental_rollout_stamp = policy_stamp(actor_for_kv)
            self._incremental_critic_stamp = critic_cycle_stamp(
                self.accelerator.unwrap_model(self.critic)
            )
            self.accelerator.unwrap_model(
                self.critic
            ).freeze_cycle_statistics = True

    def _rollout_forward(
        self,
        obs_td: TensorDict,
        *,
        actor_mode: str = "sampling",
        collect_transition: bool = True,
        track_episode_stats: bool = True,
    ) -> TensorDict:
        self._cached_incremental_indices(int(obs_td.batch_size[0]))
        if (
            collect_transition
            and getattr(self, "incremental_contract", None) is not None
        ):
            self._check_incremental_policy_lease()
            obs_td = self._tag_incremental_observation(
                obs_td.clone(recurse=False)
            )
        if collect_transition and self._rollout_future_masks is not None:
            if self._rollout_step_idx >= int(
                self._rollout_future_masks.shape[0]
            ):
                raise RuntimeError(
                    "Rollout future-mask step index exceeded "
                    "pre-sampled mask length."
                )
            obs_td = obs_td.clone(recurse=False)
            obs_td.set(
                "future_mask",
                self._rollout_future_masks[self._rollout_step_idx],
            )

        actor_for_kv = self.accelerator.unwrap_model(self.actor)
        if torch.any(self._kv_reset_pending):
            env_ids = torch.nonzero(self._kv_reset_pending).squeeze(-1)
            if env_ids.numel() > 0:
                actor_for_kv.clear_env_cache(env_ids)
                self._kv_reset_pending[env_ids] = False
        next_obs_td = super()._rollout_forward(
            obs_td,
            actor_mode=actor_mode,
            collect_transition=collect_transition,
            track_episode_stats=track_episode_stats,
        )
        if collect_transition and self._rollout_future_masks is not None:
            self._rollout_step_idx += 1
        if not collect_transition:
            dones = self._last_rollout_dones
            if dones is not None:
                self._kv_reset_pending |= (
                    dones.view(-1).to(torch.bool).to(self.device)
                )
        return next_obs_td

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        infos: dict,
    ) -> None:
        if getattr(self, "incremental_task_spec", None) is not None:
            # This new reward task explicitly ends at the time limit. Do not
            # use the upstream approximation gamma * V(current_obs) on timeout.
            validate_task_tensor(
                self.transition_td.obs, self.incremental_task_spec["task"]
            )
            if getattr(self, "incremental_contract", None) is not None:
                self._record_incremental_transition(
                    rewards, dones, time_outs, infos
                )
            if self.incremental_task_spec["task"] == "sliding" and torch.any(
                time_outs.view(-1).bool() & ~dones.view(-1).bool()
            ):
                raise ValueError(
                    "Timeout must also terminate the finite-horizon episode"
                )
            time_outs_for_gae = (
                torch.zeros_like(time_outs)
                if self.incremental_task_spec["task"] == "sliding"
                else time_outs
            )
            super().process_env_step(rewards, dones, time_outs_for_gae, infos)
        else:
            super().process_env_step(rewards, dones, time_outs, infos)
        if getattr(self, "_kv_reset_pending", None) is not None:
            self._kv_reset_pending |= (
                dones.view(-1).to(torch.bool).to(self.device)
            )

    def _wrap_obs_dict(self, obs_dict):
        td = super()._wrap_obs_dict(obs_dict)
        spec = getattr(self, "incremental_task_spec", None)
        if spec is not None:
            if TASK_KEY in td.keys():
                validate_task_tensor(td, spec["task"])
            else:
                td.set(
                    TASK_KEY,
                    torch.full(
                        (*td.batch_size, 1),
                        TASK_IDS[spec["task"]],
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )
        return self._tag_incremental_observation(td)

    def _tag_incremental_observation(self, td):
        contract = getattr(self, "incremental_contract", None)
        if contract is None:
            return td
        actor = (
            self.accelerator.unwrap_model(self.actor)
            if hasattr(self, "actor")
            else None
        )
        critic = (
            self.accelerator.unwrap_model(self.critic)
            if hasattr(self, "critic")
            else None
        )
        count = (
            self._incremental_normalizer_version() if critic is not None else 0
        )
        tag_rollout(
            td,
            contract,
            policy_version=getattr(self, "_incremental_policy_version", 0),
            route_version=getattr(actor.actor_module, "route_revision", 0)
            if actor
            else 0,
            normalizer_version=count,
            route_profile=self._incremental_route_profile()
            if actor is not None
            else 1,
            episode_id=getattr(self, "_incremental_episode_ids", None),
            timestep=getattr(self, "_incremental_timesteps", None),
        )
        # Allocate these keys before storage infers its observation template.
        if "incremental_reward_components_valid" not in td.keys():
            td.set(
                "incremental_reward_components_valid",
                torch.zeros(
                    (*td.batch_size, len(self._incremental_reward_terms)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        for name in ("terminated", "truncated"):
            if "incremental_" + name not in td.keys():
                td.set(
                    "incremental_" + name,
                    torch.zeros(
                        (*td.batch_size, 1),
                        device=self.device,
                        dtype=torch.bool,
                    ),
                )
        for name, tail in (
            ("executed_action", (self.num_actions,)),
            ("joint_target", (self.num_actions,)),
            ("reward_components", (len(self._incremental_reward_terms), 3)),
            ("next_bootstrap_value", (1,)),
        ):
            if "incremental_" + name not in td.keys():
                td.set(
                    "incremental_" + name,
                    torch.zeros((*td.batch_size, *tail), device=self.device),
                )
        return td

    def _check_incremental_policy_lease(self):
        self._cached_incremental_indices(getattr(self, "num_envs", None))
        actor = self.accelerator.unwrap_model(self.actor)
        if getattr(self, "_incremental_rollout_stamp", None) != policy_stamp(
            actor
        ):
            raise ValueError(
                "Policy/route changed between rollout sampling and replay"
            )
        critic = self.accelerator.unwrap_model(self.critic)
        if getattr(
            self, "_incremental_critic_stamp", None
        ) != critic_cycle_stamp(critic):
            raise ValueError(
                "Critic state changed between rollout sampling and replay"
            )

    def _incremental_normalizer_version(self):
        critic = self.accelerator.unwrap_model(self.critic)
        model = getattr(critic, self.incremental_contract.task_id)
        return int(model.obs_normalizer.count) if model.obs_norm_enabled else 0

    def _incremental_route_profile(self):
        actor = self.accelerator.unwrap_model(self.actor)
        return {"legacy": 0, "cooperative": 1, "new_only": 2}[
            actor.actor_module.layers[0].route_mode
        ]

    def _cached_incremental_indices(self, num_envs, env_indices=None):
        """Enforce the sampling GEMM layout for native KV PPO replay."""
        configured = getattr(self, "_incremental_sequence_replay", None)
        policy = None
        if hasattr(self, "actor"):
            actor = self.accelerator.unwrap_model(self.actor)
            policy = getattr(actor, "actor_module", None)
        mode = getattr(policy, "sequence_replay", configured or "vectorized")
        if configured is not None and mode != configured:
            raise ValueError("Incremental sequence_replay changed after setup")
        if mode != "cached":
            return None
        accelerator = getattr(self, "accelerator", None)
        if (
            getattr(self, "is_distributed", False)
            or int(getattr(accelerator, "num_processes", 1)) != 1
            or int(getattr(self, "gpu_world_size", 1)) != 1
        ):
            raise ValueError(
                "Cached incremental replay currently requires a single rank; "
                "DDP buffer broadcast is not supported by its state lease"
            )
        precisions = (
            getattr(self, "config", {}).get("mixed_precision"),
            getattr(accelerator, "mixed_precision", None),
        )
        if any(value not in {None, "no", "fp32"} for value in precisions):
            raise ValueError(
                "Cached incremental replay requires full precision; "
                "AMP/mixed_precision cached writes are not validated"
            )
        if num_envs is None or int(num_envs) <= 0:
            raise ValueError(
                "Cached incremental replay requires a positive environment "
                "count registered before sampling"
            )
        if any(
            int(getattr(self, key, self.num_mini_batches)) != 1
            for key in (
                "configured_num_mini_batches",
                "requested_num_mini_batches",
                "num_mini_batches",
            )
        ):
            raise ValueError(
                "Cached incremental replay requires num_mini_batches=1 "
                "on every rank; environment batches cannot be partitioned"
            )
        configured_chunk = getattr(self, "_incremental_replay_chunk_size", 0)
        replay_chunk = getattr(policy, "replay_chunk_size", configured_chunk)
        if (
            type(replay_chunk) is not int
            or replay_chunk < 0
            or self.rollout_forward_chunk_size != replay_chunk
            or (
                hasattr(self, "_incremental_replay_chunk_size")
                and configured_chunk != replay_chunk
            )
        ):
            raise ValueError(
                "Cached incremental replay requires unchanged "
                "rollout_forward_chunk_size == replay_chunk_size"
            )
        expected_envs = int(getattr(self, "num_envs", num_envs))
        lease = getattr(self, "_incremental_rollout_layout", None)
        if int(num_envs) != expected_envs or (
            lease is not None and lease != (expected_envs, replay_chunk)
        ):
            raise ValueError("Cached replay environment/chunk layout changed")
        canonical = torch.arange(num_envs, device=self.device)
        if env_indices is not None and (
            env_indices.dtype != canonical.dtype
            or env_indices.device != canonical.device
            or not torch.equal(env_indices, canonical)
        ):
            raise ValueError(
                "Cached incremental replay requires the original environment "
                "order; permutations or subsets are unsupported"
            )
        return canonical

    @torch.no_grad()
    def _verify_incremental_recomputation(self, env_indices=None):
        """Fail before any optimizer step if frozen-policy PPO targets drift.

        Keep routing mathematically unchanged. Near-tied Top-K scores can
        still flip under different GEMM batching; that is a failed gate, not
        permission to silently reuse sampled routes or increase tolerance.
        """
        data = self.storage.data
        obs = data["obs"].transpose(0, 1)
        actor = self.accelerator.unwrap_model(self.actor)
        validate_rollout(
            obs,
            self.incremental_contract,
            policy_version=self._incremental_policy_version,
            route_version=getattr(actor.actor_module, "route_revision", 0),
            normalizer_version=self._incremental_normalizer_version(),
            route_profile=self._incremental_route_profile(),
        )
        canonical = self._cached_incremental_indices(
            int(obs.batch_size[0]), env_indices
        )
        # Cached replay uses exactly the sampling layout. The older vectorized
        # mode still checks every actual update permutation and partition.
        layouts = [torch.arange(obs.batch_size[0], device=self.device)]
        if canonical is None and env_indices is not None:
            count, width = self._resolve_sequence_batch_partition(
                obs.batch_size[0], self.num_mini_batches
            )
            layouts.extend(
                env_indices[i * width : (i + 1) * width]
                for i in range(count)
                if i * width < len(env_indices)
            )
        limits = (
            ("mu", 5e-5, 1e-5),
            ("sigma", 1e-6, 1e-5),
            ("actions_log_prob", 1e-4, 2e-5),
        )
        metrics = {key: 0.0 for key, _, _ in limits}
        for idx in layouts:
            mask = self._build_episode_causal_mask(
                data["dones"].transpose(0, 1)[idx]
            )
            with self.accelerator.autocast():
                result = self.actor(
                    obs[idx],
                    actions=data["actions"].transpose(0, 1)[idx],
                    mode="sequence_logp",
                    attn_mask=mask,
                    update_obs_norm=False,
                )
            for key, atol, rtol in limits:
                expected = data[key].transpose(0, 1)[idx]
                value = result[key]
                metrics[key] = max(
                    metrics[key], float((value - expected).abs().max())
                )
                failed = torch.tensor(
                    int(
                        not torch.allclose(
                            value, expected, atol=atol, rtol=rtol
                        )
                    ),
                    device=self.device,
                )
                if self.is_distributed:
                    failed = self.accelerator.reduce(failed, reduction="sum")
                if failed.item():
                    raise ValueError(
                        f"Pre-update {key} replay mismatch "
                        f"({metrics[key]:.8g}); no optimizer update allowed"
                    )
        metrics["layouts_checked"] = len(layouts)
        self._incremental_recompute_metrics = metrics

    def _record_incremental_transition(self, rewards, dones, time_outs, infos):
        obs = self.transition_td.obs
        actor = self.accelerator.unwrap_model(self.actor)
        validate_rollout(
            obs,
            self.incremental_contract,
            policy_version=self._incremental_policy_version,
            route_version=getattr(actor.actor_module, "route_revision", 0),
            normalizer_version=self._incremental_normalizer_version(),
            route_profile=self._incremental_route_profile(),
        )
        record = infos.get("incremental_transition")
        if record is None or any(
            record.get(k) is None for k in ("executed_action", "joint_target")
        ):
            raise ValueError(
                "Missing pre-reset incremental execution/reward record"
            )
        if not torch.equal(
            record["terminated"].bool() | record["truncated"].bool(),
            dones.reshape_as(record["terminated"]).bool(),
        ):
            raise ValueError("Termination/truncation record mismatch")
        for name in ("terminated", "truncated"):
            obs.set("incremental_" + name, record[name].reshape(-1, 1))
        if not torch.equal(
            record["truncated"].reshape(-1).bool(),
            time_outs.reshape(-1).bool(),
        ):
            raise ValueError(
                "Truncation record differs from environment timeout flags"
            )
        for name in ("executed_action", "joint_target"):
            obs.set("incremental_" + name, record[name])
        if not torch.equal(
            record["executed_action"], self.transition_td.actions
        ):
            raise ValueError(
                "Environment raw action differs from sampled PPO action"
            )
        components = obs["incremental_reward_components"].zero_()
        component_valid = obs["incremental_reward_components_valid"].zero_()
        for i, name in enumerate(self._incremental_reward_terms):
            value = record["reward_components"].get(name)
            if value is not None:
                components[:, i] = value
                component_valid[:, i] = True
            elif (
                getattr(self, "_incremental_reward_weights", {}).get(name, 1.0)
                != 0
            ):
                raise ValueError(f"Missing enabled reward component: {name}")
        if not torch.allclose(
            components[:, :, 2].sum(-1),
            rewards.reshape(-1),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "Native reward total differs from task-scoped components"
            )
        if not hasattr(self, "_incremental_episode_ids"):
            self._incremental_episode_ids = torch.zeros(
                self.num_envs, 1, dtype=torch.int64, device=self.device
            )
            self._incremental_timesteps = torch.zeros_like(
                self._incremental_episode_ids
            )
        self._incremental_episode_ids += dones.reshape(-1, 1).long()
        self._incremental_timesteps = torch.where(
            dones.reshape(-1, 1).bool(), 0, self._incremental_timesteps + 1
        )

    def _compute_returns(self, obs_td):
        spec = getattr(self, "incremental_task_spec", None)
        if spec is not None:
            validate_task_tensor(self.storage.data["obs"], spec["task"])
        contract = getattr(self, "incremental_contract", None)
        if contract is not None and contract.task_id == "sliding":
            self._check_incremental_policy_lease()
            data = self.storage.data
            actor = self.accelerator.unwrap_model(self.actor)
            validate_rollout(
                data["obs"],
                contract,
                policy_version=self._incremental_policy_version,
                route_version=getattr(actor.actor_module, "route_revision", 0),
                normalizer_version=self._incremental_normalizer_version(),
                route_profile=self._incremental_route_profile(),
            )
            validate_episode_ledger(data["obs"], data["dones"])
            with torch.no_grad(), self.accelerator.autocast():
                last = self.critic(obs_td, update_obs_norm=False)["values"]
            next_values = torch.cat((data["values"][1:], last[None]), dim=0)
            advantages, returns = task_gae(
                data["rewards"],
                data["values"],
                next_values,
                data["obs"]["incremental_terminated"],
                data["obs"]["incremental_truncated"],
                data["obs"][TASK_KEY],
                discounts={1: (contract.gamma, contract.gae_lambda)},
            )
            data["obs"]["incremental_next_bootstrap_value"].copy_(
                torch.where(data["dones"], 0, next_values)
            )
            data["returns"].copy_(returns)
            if self.global_advantage_norm:
                advantages, self._incremental_advantage_moments = (
                    normalize_by_task(
                        advantages,
                        data["obs"][TASK_KEY],
                        valid=data["obs"]["incremental_valid_mask"],
                        accelerator=self.accelerator
                        if self.is_distributed
                        else None,
                    )
                )
            data["advantages"].copy_(advantages)
            return
        return super()._compute_returns(obs_td)

    def _post_update_hook(self, loss_dict):
        if getattr(self, "incremental_contract", None) is not None:
            critic = self.accelerator.unwrap_model(self.critic)
            critic.finish_statistics_cycle(
                self.storage.data["obs"].flatten(0, 1),
                self.accelerator if self.is_distributed else None,
            )
            self._incremental_policy_version += 1
            self._incremental_rollout_stamp = None
            self._incremental_critic_stamp = None
            self.accelerator.unwrap_model(self.actor).clear_env_cache(None)
        super()._post_update_hook(loss_dict)

    def _extra_checkpoint_state(self):
        state = super()._extra_checkpoint_state()
        spec = getattr(self, "incremental_task_spec", None)
        if spec is not None:
            state["incremental_task_contract"] = {
                k: v for k, v in spec.items() if k != "bootstrap_from_legacy"
            }
            state["incremental_actor_contract"] = OmegaConf.to_container(
                self.config.module_dict.actor.internal_expansion,
                resolve=True,
            )
            state["incremental_implementation_contract"] = (
                implementation_contract()
            )
            if getattr(self, "incremental_contract", None) is not None:
                state["task_spec"] = asdict(self.incremental_contract)
                state["registered_task_specs"] = (
                    self._incremental_registered_specs
                )
                state["incremental_policy_version"] = (
                    self._incremental_policy_version
                )
        return state

    def load(self, ckpt_path):
        spec = getattr(self, "incremental_task_spec", None)
        if spec is not None and ckpt_path is not None:
            state = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            contract = state.get("incremental_task_contract")
            expected = {
                k: v for k, v in spec.items() if k != "bootstrap_from_legacy"
            }
            if contract is None:
                if not spec["bootstrap_from_legacy"]:
                    raise ValueError(
                        "Missing incremental task checkpoint contract"
                    )
                import hashlib
                from pathlib import Path

                digest = hashlib.sha256(
                    Path(ckpt_path).read_bytes()
                ).hexdigest()
                if digest != spec["legacy_checkpoint_sha256"]:
                    raise ValueError(
                        "Bootstrap checkpoint is not the registered "
                        "model_11999"
                    )
            else:
                if getattr(self, "incremental_contract", None) is not None:
                    expected_task = asdict(self.incremental_contract)
                    registered = state.get("registered_task_specs", {})
                    # Cross-task evaluation requires a complete pre-registered
                    # TaskSpec, including its reference contents. Training
                    # resume still requires the original active task exactly.
                    saved_task = state.get("task_spec")
                    if getattr(self, "is_offline_eval", False):
                        key = evaluation_registry_key(
                            spec["task"],
                            self.incremental_contract.execution_profile,
                        )
                        saved_task = registered.get(key)
                    if saved_task != expected_task:
                        raise ValueError(
                            "Immutable TaskSpec checkpoint mismatch"
                        )
                    if (
                        not getattr(self, "is_offline_eval", False)
                        and registered != self._incremental_registered_specs
                    ):
                        raise ValueError(
                            "Registered evaluation TaskSpec "
                            "checkpoint mismatch"
                        )
                if (
                    state.get("incremental_implementation_contract")
                    != implementation_contract()
                ):
                    raise ValueError(
                        "Incremental implementation checkpoint "
                        "contract mismatch"
                    )
                # Evaluation can select the other preserved value function.
                if getattr(self, "is_offline_eval", False):
                    contract = {
                        k: v for k, v in contract.items() if k != "task"
                    }
                    expected = {
                        k: v for k, v in expected.items() if k != "task"
                    }
                if contract != expected:
                    raise ValueError(
                        "Incremental task checkpoint contract mismatch"
                    )
                route = state.get("incremental_actor_contract")
                current = OmegaConf.to_container(
                    self.config.module_dict.actor.internal_expansion,
                    resolve=True,
                )
                ignore = {"bootstrap_from_legacy"}
                if getattr(self, "is_offline_eval", False):
                    ignore |= {"mode", "stage", "warmup"}
                    # Admission mode is an evaluation profile. Its threshold,
                    # initialization and version remain checkpoint contracts.
                    route = None if route is None else dict(route)
                    for profile in (route, current):
                        if profile is not None and isinstance(
                            profile.get("admission"), dict
                        ):
                            profile["admission"] = {
                                k: v
                                for k, v in profile["admission"].items()
                                if k != "mode"
                            }
                if route is None or {
                    k: v for k, v in route.items() if k not in ignore
                } != {k: v for k, v in current.items() if k not in ignore}:
                    raise ValueError(
                        "Incremental routing checkpoint contract mismatch"
                    )
        result = super().load(ckpt_path)
        if spec is not None and ckpt_path is not None:
            actor = self.accelerator.unwrap_model(self.actor)
            original = {
                k: v
                for k, v in actor.state_dict().items()
                if ".added_" not in k and not k.endswith("expansion_version")
            }
            if tensor_state_hash(original) != spec["legacy_actor_state_hash"]:
                raise ValueError(
                    "Frozen Actor differs from registered model_11999"
                )
            self._incremental_policy_version = int(
                state.get("incremental_policy_version", 0)
            )
            if state.get("registered_task_specs") is not None:
                self._incremental_registered_specs = state[
                    "registered_task_specs"
                ]
        return result

    def _restore_optimizer_state(
        self, optimizer, loaded_state_dict, *, optimizer_name
    ):
        if getattr(self, "incremental_task_spec", None) is not None:
            compatible, reason = self._optimizer_state_is_compatible(
                optimizer, loaded_state_dict
            )
            if not compatible:
                raise ValueError(
                    f"Strict {optimizer_name} optimizer restore: {reason}"
                )
            for group, saved in zip(
                optimizer.param_groups,
                loaded_state_dict["param_groups"],
                strict=True,
            ):
                for parameter, parameter_id in zip(
                    group["params"], saved["params"], strict=True
                ):
                    for name, value in (
                        loaded_state_dict["state"]
                        .get(parameter_id, {})
                        .items()
                    ):
                        if (
                            torch.is_tensor(value)
                            and value.ndim > 0
                            and value.shape != parameter.shape
                        ):
                            raise ValueError(
                                f"Strict {optimizer_name} optimizer "
                                f"state shape mismatch: {name}"
                            )
            optimizer.load_state_dict(loaded_state_dict)
            return True
        return super()._restore_optimizer_state(
            optimizer,
            loaded_state_dict,
            optimizer_name=optimizer_name,
        )

    @staticmethod
    def _build_episode_causal_mask(dones_seq: torch.Tensor) -> torch.Tensor:
        """Build [N, T, T] mask: causal and within the same episode segment."""
        n, t, _ = dones_seq.shape
        device = dones_seq.device
        dones = dones_seq.squeeze(-1).to(torch.long)
        seg = torch.cumsum(dones, dim=1) - dones
        same = seg[:, :, None] == seg[:, None, :]
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
        return same & causal

    @staticmethod
    def _resolve_sequence_batch_partition(
        num_envs: int,
        num_mini_batches: int,
    ) -> tuple[int, int]:
        if num_envs <= 0:
            raise RuntimeError(
                "PPOTF sequence batching requires at least one "
                "environment on each rank."
            )
        effective_num_mini_batches = max(
            1, min(int(num_mini_batches), int(num_envs))
        )
        mini_batch_envs = max(
            1,
            (num_envs + effective_num_mini_batches - 1)
            // effective_num_mini_batches,
        )
        return effective_num_mini_batches, mini_batch_envs

    def _sequence_batches(
        self, num_mini_batches: int, num_epochs: int, env_indices=None
    ) -> Generator[tuple, None, None]:
        data = self.storage.data
        obs_seq = data["obs"].transpose(0, 1)
        actions_seq = data["actions"].transpose(0, 1)
        values_seq = data["values"].transpose(0, 1)
        rewards_seq = data["rewards"].transpose(0, 1)
        returns_seq = data["returns"].transpose(0, 1)
        adv_seq = data["advantages"].transpose(0, 1)
        old_logp_seq = data["actions_log_prob"].transpose(0, 1)
        old_mu_seq = data["mu"].transpose(0, 1)
        old_sigma_seq = data["sigma"].transpose(0, 1)
        dones_seq = data["dones"].transpose(0, 1)
        gt_base_lin_vel_seq = None
        gt_root_height_seq = None
        gt_keybody_contact_seq = None
        gt_ref_keybody_rel_pos_seq = None
        gt_robot_keybody_rel_pos_seq = None
        if self.use_aux_state_pred:
            gt_base_lin_vel_seq = data["gt_base_lin_vel_b"].transpose(0, 1)
            gt_root_height_seq = data["gt_root_height_rel_terrain"].transpose(
                0, 1
            )
            gt_keybody_contact_seq = data["gt_keybody_contacts"].transpose(
                0, 1
            )
            gt_ref_keybody_rel_pos_seq = data[
                "gt_ref_keybody_rel_pos"
            ].transpose(0, 1)
            gt_robot_keybody_rel_pos_seq = data[
                "gt_robot_keybody_rel_pos"
            ].transpose(0, 1)

        num_envs = int(actions_seq.shape[0])
        if num_envs <= 0:
            raise RuntimeError(
                "PPOTF sequence batching requires at least one "
                "environment on each rank, "
                f"got num_envs={num_envs}."
            )
        canonical = self._cached_incremental_indices(num_envs, env_indices)
        if canonical is not None:
            if int(num_mini_batches) != 1:
                raise ValueError(
                    "Cached incremental replay requires num_mini_batches=1"
                )
            env_indices = canonical
        num_mini_batches, mb_env = self._resolve_sequence_batch_partition(
            num_envs, num_mini_batches
        )
        if env_indices is None:
            env_indices = torch.randperm(num_envs, device=self.device)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mb_env
                if start >= num_envs:
                    break
                end = min(num_envs, (i + 1) * mb_env)
                idx = env_indices[start:end]
                obs_b = obs_seq[idx]
                actions_b = actions_seq[idx]
                values_b = values_seq[idx]
                rewards_b = rewards_seq[idx]
                returns_b = returns_seq[idx]
                adv_b = adv_seq[idx]
                old_logp_b = old_logp_seq[idx]
                old_mu_b = old_mu_seq[idx]
                old_sigma_b = old_sigma_seq[idx]
                dones_b = dones_seq[idx]
                gt_base_lin_vel_b = (
                    gt_base_lin_vel_seq[idx]
                    if gt_base_lin_vel_seq is not None
                    else None
                )
                gt_root_height_b = (
                    gt_root_height_seq[idx]
                    if gt_root_height_seq is not None
                    else None
                )
                gt_keybody_contact_b = (
                    gt_keybody_contact_seq[idx]
                    if gt_keybody_contact_seq is not None
                    else None
                )
                gt_ref_keybody_rel_pos_b = (
                    gt_ref_keybody_rel_pos_seq[idx]
                    if gt_ref_keybody_rel_pos_seq is not None
                    else None
                )
                gt_robot_keybody_rel_pos_b = (
                    gt_robot_keybody_rel_pos_seq[idx]
                    if gt_robot_keybody_rel_pos_seq is not None
                    else None
                )
                attn_mask = self._build_episode_causal_mask(dones_b)
                yield (
                    obs_b,
                    actions_b,
                    values_b,
                    adv_b,
                    returns_b,
                    rewards_b,
                    old_logp_b,
                    old_mu_b,
                    old_sigma_b,
                    dones_b,
                    attn_mask,
                    gt_base_lin_vel_b,
                    gt_root_height_b,
                    gt_keybody_contact_b,
                    gt_ref_keybody_rel_pos_b,
                    gt_robot_keybody_rel_pos_b,
                )

    def update(self):
        incremental_indices = None
        if getattr(self, "incremental_contract", None) is not None:
            self._check_incremental_policy_lease()
        actor_unwrapped = self.accelerator.unwrap_model(self.actor)
        actor_policy = actor_unwrapped.actor_module
        actor_policy.set_collect_routing_stats(False)
        if getattr(self, "incremental_contract", None) is not None:
            incremental_indices = self._cached_incremental_indices(
                self.storage.num_envs
            )
            if incremental_indices is None:
                incremental_indices = torch.randperm(
                    self.storage.num_envs, device=self.device
                )
            self._verify_incremental_recomputation(incremental_indices)
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_token = 0.0
        mean_kl_loss = 0.0
        mean_kl_analytic = 0.0
        critic_explained_variance = self._compute_explained_variance(
            target=self.storage.data["returns"],
            prediction=self.storage.data["values"],
        )
        mean_aux_base_lin_vel_nll = 0.0
        mean_aux_root_height_nll = 0.0
        mean_aux_base_lin_vel_std = 0.0
        mean_aux_root_height_std = 0.0
        mean_aux_keybody_contact_bce = 0.0
        mean_aux_keybody_contact_acc = 0.0
        mean_aux_ref_keybody_rel_pos_mse = 0.0
        mean_aux_robot_keybody_rel_pos_mse = 0.0
        mean_aux_router_command_recon_mse = 0.0
        mean_aux_router_future_recon_huber = 0.0
        mean_aux_router_switch_penalty_js = 0.0
        mean_aux_action_mean_butterworth = 0.0
        mean_moe_load_balance_loss = 0.0
        mean_inactive_expert_margin_to_topk_loss = 0.0
        mean_router_expert_orthogonal_loss = 0.0
        mean_selected_expert_margin_to_unselected_loss = 0.0
        moe_layers = [
            layer
            for layer in actor_policy.layers
            if isinstance(layer, GroupedMoEBlock)
        ]

        (
            effective_num_mini_batches,
            mini_batch_envs,
        ) = self._resolve_sequence_batch_partition(
            self.storage.num_envs, self.num_mini_batches
        )
        self._last_update_metrics = {
            "0-Train/configured_num_mini_batches": float(
                self.configured_num_mini_batches
            ),
            "0-Train/requested_num_mini_batches": float(
                self.requested_num_mini_batches
            ),
            "0-Train/effective_num_mini_batches": float(
                effective_num_mini_batches
            ),
            "0-Train/mini_batch_size_per_rank": float(
                mini_batch_envs * self.num_steps_per_env
            ),
            "0-Train/mini_batch_num_envs_per_rank": float(mini_batch_envs),
            "0-Train/num_updates_executed": 0.0,
            "0-Train/lr_scale_factor": float(self.distributed_lr_scale_factor),
            "0-Train/scalable_distributed_update": float(
                self.distributed_update_mode == "scalable"
            ),
            "0-Train/kl_windowed": 0.0,
            "0-Train/kl_stop_triggered": 0.0,
            "0-Train/kl_stop_analytic": 0.0,
            "0-Train/kl_analytic_batch_last": 0.0,
            "0-Train/kl_analytic_batch_max": 0.0,
            "0-Train/clip_fraction_batch_mean": 0.0,
            "0-Train/clip_fraction_batch_last": 0.0,
        }
        entropy_coef = self._get_effective_entropy_coef()
        batch_kwargs = (
            {} if incremental_indices is None
            else {"env_indices": incremental_indices}
        )
        generator = self._sequence_batches(
            effective_num_mini_batches,
            self.num_learning_epochs,
            **batch_kwargs
        )
        measure_analytic_kl = self.desired_kl is not None
        normalize_per_mb = bool(self.normalize_advantage_per_mini_batch)
        num_updates = 0
        num_kl_measurements = 0
        kl_stop_triggered = False
        kl_stop_analytic = 0.0
        kl_windowed = None
        recent_analytic_kls: list[float] = []
        kl_analytic_batch_last = 0.0
        kl_analytic_batch_max = 0.0
        clip_fraction_batch_mean = 0.0
        clip_fraction_batch_last = 0.0

        for (
            obs_b,
            actions_b,
            target_values_b,
            advantages_b,
            returns_b,
            _rewards_b,
            old_logp_b,
            old_mu_b,
            old_sigma_b,
            dones_b,
            attn_mask_b,
            gt_base_lin_vel_b,
            gt_root_height_b,
            gt_keybody_contact_b,
            gt_ref_keybody_rel_pos_b,
            gt_robot_keybody_rel_pos_b,
        ) in generator:
            valid_tok = attn_mask_b.diagonal(dim1=1, dim2=2).to(torch.float32)
            valid_count = valid_tok.sum().clamp_min(1.0)

            if normalize_per_mb:
                with torch.no_grad():
                    flat = advantages_b.view(-1).float()
                    if self.global_advantage_norm and self.is_distributed:
                        count = torch.tensor(
                            [flat.numel()],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        sum_g = self.accelerator.reduce(
                            flat.sum(), reduction="sum"
                        )
                        sqsum_g = self.accelerator.reduce(
                            (flat * flat).sum(), reduction="sum"
                        )
                        count_g = self.accelerator.reduce(
                            count, reduction="sum"
                        )
                        mean = sum_g / count_g
                        var = (sqsum_g / count_g) - mean * mean
                        std = torch.sqrt(var.clamp_min(1.0e-8))
                    else:
                        mean = flat.mean()
                        std = flat.std().clamp_min(1.0e-8)
                    advantages_b = (advantages_b - mean) / std

            b, t = int(obs_b.batch_size[0]), int(obs_b.batch_size[1])
            critic_obs_flat = obs_b.flatten(0, 1)
            with self.accelerator.autocast():
                actor_out = self.actor(
                    obs_b,
                    actions=actions_b,
                    mode="sequence_logp",
                    attn_mask=attn_mask_b,
                    update_obs_norm=False,
                )
                critic_out = self.critic(
                    critic_obs_flat, update_obs_norm=False
                )
            logp_new_b = actor_out.get("actions_log_prob")
            mu_b = actor_out.get("mu")
            sigma_b = actor_out.get("sigma")
            entropy_b = actor_out.get("entropy")
            v_pred_flat = critic_out.get("values")
            value_batch = v_pred_flat.reshape(b, t, -1)
            returns_batch_norm = returns_b
            target_values_batch_norm = target_values_b

            analytic_kl = None
            if measure_analytic_kl:
                analytic_kl = self._compute_analytic_kl(
                    old_mu=old_mu_b.float(),
                    old_sigma=old_sigma_b.float(),
                    new_mu=mu_b.float(),
                    new_sigma=sigma_b.float(),
                    weight=valid_tok,
                )
                mean_kl_analytic += analytic_kl
                num_kl_measurements += 1
                kl_analytic_batch_last = analytic_kl
                kl_analytic_batch_max = max(kl_analytic_batch_max, analytic_kl)
                recent_analytic_kls.append(analytic_kl)
                if len(recent_analytic_kls) > self.kl_early_stop_window_size:
                    recent_analytic_kls.pop(0)
                kl_windowed = self._compute_windowed_kl_signal(
                    recent_analytic_kls
                )
                if self._should_early_stop_for_kl(
                    kl_windowed, num_kl_measurements
                ):
                    kl_stop_triggered = True
                    kl_stop_analytic = analytic_kl
                    break

            logp_new = logp_new_b.squeeze(-1).float()
            logp_old = old_logp_b.squeeze(-1).float()
            ratio = torch.exp(logp_new - logp_old)
            clip_fraction = self._compute_clip_fraction(
                ratio, weight=valid_tok
            )
            clip_fraction_batch_mean += clip_fraction
            clip_fraction_batch_last = clip_fraction
            adv = advantages_b.squeeze(-1)
            s1 = ratio * adv
            s2 = (
                torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                * adv
            )
            surrogate_loss = (
                -torch.min(s1, s2) * valid_tok
            ).sum() / valid_count

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch_norm + (
                    value_batch - target_values_batch_norm
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch_norm).pow(2)
                value_losses_clipped = (
                    value_clipped - returns_batch_norm
                ).pow(2)
                v_max = torch.max(value_losses, value_losses_clipped).squeeze(
                    -1
                )
                value_loss = (v_max * valid_tok).sum() / valid_count
            else:
                v_err = (returns_batch_norm - value_batch).pow(2).squeeze(-1)
                value_loss = (v_err * valid_tok).sum() / valid_count

            actor_loss = surrogate_loss
            critic_loss = self.value_loss_coef * value_loss
            aux_base_lin_vel_loss = None
            aux_root_height_loss = None
            aux_base_lin_vel_std = None
            aux_root_height_std = None
            aux_keybody_contact_loss = None
            aux_keybody_contact_acc = None
            aux_ref_keybody_rel_pos_loss = None
            aux_robot_keybody_rel_pos_loss = None
            aux_router_command_recon_loss = None
            aux_router_future_recon_loss = None
            aux_router_switch_penalty_loss = None
            aux_action_mean_butterworth_loss = None
            moe_load_balance_loss = None
            inactive_expert_margin_to_topk_loss = None
            router_expert_orthogonal_loss = None
            selected_expert_margin_to_unselected_loss = None
            if self.use_aux_state_pred:
                aux_base_lin_vel_loc = actor_out.get("aux_base_lin_vel_loc")
                aux_base_lin_vel_log_std = actor_out.get(
                    "aux_base_lin_vel_log_std"
                )
                aux_base_lin_vel_std = torch.clamp(
                    torch.exp(aux_base_lin_vel_log_std),
                    min=self.aux_state_pred_min_std,
                    max=self.aux_state_pred_max_std,
                )
                aux_base_lin_vel_nll = 0.5 * (
                    torch.square(
                        (gt_base_lin_vel_b - aux_base_lin_vel_loc)
                        / aux_base_lin_vel_std
                    )
                    + 2.0 * torch.log(aux_base_lin_vel_std + 1.0e-8)
                ).sum(dim=-1)
                aux_base_lin_vel_loss = (
                    aux_base_lin_vel_nll * valid_tok
                ).sum() / valid_count
                actor_loss = (
                    actor_loss
                    + self.aux_state_pred_w_base_lin_vel
                    * aux_base_lin_vel_loss
                )
                aux_root_height_loc = actor_out.get("aux_root_height_loc")
                aux_root_height_log_std = actor_out.get(
                    "aux_root_height_log_std"
                )
                if self.use_aux_root_height and gt_root_height_b is not None:
                    aux_root_height_std = torch.clamp(
                        torch.exp(aux_root_height_log_std),
                        min=self.aux_state_pred_min_std,
                        max=self.aux_state_pred_max_std,
                    )
                    aux_root_height_nll = 0.5 * (
                        torch.square(
                            (gt_root_height_b - aux_root_height_loc)
                            / aux_root_height_std
                        )
                        + 2.0 * torch.log(aux_root_height_std + 1.0e-8)
                    ).sum(dim=-1)
                    aux_root_height_loss = (
                        aux_root_height_nll * valid_tok
                    ).sum() / valid_count
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_root_height
                        * aux_root_height_loss
                    )
                else:
                    actor_loss = actor_loss + 0.0 * (
                        aux_root_height_loc.sum()
                        + aux_root_height_log_std.sum()
                    )
                if (
                    self.aux_state_pred_num_contact_bodies > 0
                    and gt_keybody_contact_b is not None
                ):
                    aux_keybody_contact_logits = actor_out.get(
                        "aux_keybody_contact_logits"
                    )
                    contact_bce = F.binary_cross_entropy_with_logits(
                        aux_keybody_contact_logits,
                        gt_keybody_contact_b,
                        reduction="none",
                    ).mean(dim=-1)
                    aux_keybody_contact_loss = (
                        contact_bce * valid_tok
                    ).sum() / valid_count
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_keybody_contact
                        * aux_keybody_contact_loss
                    )
                    contact_pred = (aux_keybody_contact_logits > 0.0).to(
                        gt_keybody_contact_b.dtype
                    )
                    contact_acc_tok = (
                        (contact_pred == gt_keybody_contact_b)
                        .to(torch.float32)
                        .mean(dim=-1)
                    )
                    aux_keybody_contact_acc = (
                        contact_acc_tok * valid_tok
                    ).sum() / valid_count
                aux_ref_keybody_rel_pos = actor_out.get(
                    "aux_ref_keybody_rel_pos"
                )
                aux_robot_keybody_rel_pos = actor_out.get(
                    "aux_robot_keybody_rel_pos"
                )
                if (
                    self.aux_state_pred_num_keybody_bodies > 0
                    and gt_ref_keybody_rel_pos_b is not None
                ):
                    aux_ref_keybody_rel_pos_loss = (
                        self._masked_aux_keybody_mse(
                            aux_ref_keybody_rel_pos,
                            gt_ref_keybody_rel_pos_b,
                            valid_tok,
                        )
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_ref_keybody_rel_pos
                        * aux_ref_keybody_rel_pos_loss
                    )
                elif aux_ref_keybody_rel_pos.numel() > 0:
                    actor_loss = (
                        actor_loss + 0.0 * aux_ref_keybody_rel_pos.sum()
                    )
                if (
                    self.aux_state_pred_num_keybody_bodies > 0
                    and gt_robot_keybody_rel_pos_b is not None
                ):
                    aux_robot_keybody_rel_pos_loss = (
                        self._masked_aux_keybody_mse(
                            aux_robot_keybody_rel_pos,
                            gt_robot_keybody_rel_pos_b,
                            valid_tok,
                        )
                    )
                    actor_loss = (
                        actor_loss
                        + self.aux_state_pred_w_robot_keybody_rel_pos
                        * aux_robot_keybody_rel_pos_loss
                    )
                elif aux_robot_keybody_rel_pos.numel() > 0:
                    actor_loss = (
                        actor_loss + 0.0 * aux_robot_keybody_rel_pos.sum()
                    )
            if self.use_aux_router_command_recon:
                if self.aux_router_command_recon_assembler is None:
                    raise ValueError(
                        "aux_router_command_recon is enabled but command "
                        "assembler was not initialized."
                    )
                aux_router_command_recon_pred = actor_out.get(
                    "aux_router_command_recon"
                )
                gt_aux_router_command_recon_b = (
                    self.aux_router_command_recon_assembler(
                        obs_b.flatten(0, 1)
                    ).reshape(b, t, -1)
                )
                aux_router_command_recon_loss = self._masked_aux_mse(
                    aux_router_command_recon_pred,
                    gt_aux_router_command_recon_b,
                    valid_tok,
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_command_recon_weight
                    * aux_router_command_recon_loss
                )
            if self.use_aux_router_future_recon:
                aux_router_future_recon_loss = (
                    self._compute_aux_router_future_recon_loss(
                        actor_wrapper=actor_unwrapped,
                        actor_out=actor_out,
                        obs_b=obs_b,
                        valid_tok=valid_tok,
                    )
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_future_recon_weight
                    * aux_router_future_recon_loss
                )
            if self.use_aux_router_switch_penalty:
                if self.aux_router_switch_penalty_metric == "js":
                    aux_router_features = actor_out.get("router_features")
                    aux_router_switch_penalty_loss = self._masked_adjacent_router_js(
                        router_features=aux_router_features,
                        valid_tok=valid_tok,
                        num_moe_layers=self.aux_command_router_num_moe_layers,
                        num_fine_experts=self.aux_command_router_num_fine_experts,
                    )
                else:
                    aux_router_temporal_features = actor_out.get(
                        "router_temporal_features"
                    )
                    aux_router_switch_penalty_loss = self._masked_adjacent_router_normed_smooth_l1(
                        router_temporal_features=aux_router_temporal_features,
                        valid_tok=valid_tok,
                        num_moe_layers=self.aux_command_router_num_moe_layers,
                        num_fine_experts=self.aux_command_router_num_fine_experts,
                        beta=self.aux_router_switch_penalty_beta,
                    )
                aux_router_switch_penalty_loss = (
                    aux_router_switch_penalty_loss.to(actor_loss.dtype)
                )
                actor_loss = (
                    actor_loss
                    + self.aux_router_switch_penalty_weight
                    * aux_router_switch_penalty_loss
                )
            if self.use_aux_action_mean_butterworth:
                aux_action_mean_butterworth_loss = (
                    self._action_mean_butterworth_loss(
                        action_mean=mu_b,
                        dones=dones_b,
                        valid_tok=valid_tok,
                        sos=self.aux_action_mean_butterworth_sos,
                    )
                )
                actor_loss = (
                    actor_loss
                    + self.aux_action_mean_butterworth_weight
                    * aux_action_mean_butterworth_loss
                )
            if self.use_moe_load_balance and len(moe_layers) > 0:
                load_balance_losses = [
                    layer.last_moe_load_balance_loss
                    for layer in moe_layers
                    if layer.last_moe_load_balance_loss is not None
                ]
                if len(load_balance_losses) > 0:
                    moe_load_balance_loss = torch.stack(
                        [
                            loss.to(actor_loss.device, dtype=actor_loss.dtype)
                            for loss in load_balance_losses
                        ]
                    ).mean()
                    actor_loss = (
                        actor_loss
                        + self.moe_load_balance_weight
                        * moe_load_balance_loss
                    )
            if self.use_inactive_expert_margin_to_topk and len(moe_layers) > 0:
                margin_losses = [
                    layer.last_inactive_expert_margin_to_topk_loss
                    for layer in moe_layers
                    if layer.last_inactive_expert_margin_to_topk_loss
                    is not None
                ]
                if len(margin_losses) > 0:
                    inactive_expert_margin_to_topk_loss = torch.stack(
                        [
                            loss.to(actor_loss.device, dtype=actor_loss.dtype)
                            for loss in margin_losses
                        ]
                    ).mean()
                    actor_loss = (
                        actor_loss
                        + self.inactive_expert_margin_to_topk_weight
                        * inactive_expert_margin_to_topk_loss
                    )
            if self.use_router_expert_orthogonal and len(moe_layers) > 0:
                orth_losses = []
                for layer in moe_layers:
                    layer_orth_loss, _, _ = (
                        self._compute_routed_expert_orthogonal_loss(
                            layer,
                            dtype=actor_loss.dtype,
                            device=actor_loss.device,
                        )
                    )
                    orth_losses.append(layer_orth_loss)
                if len(orth_losses) > 0:
                    router_expert_orthogonal_loss = torch.stack(
                        orth_losses
                    ).mean()
                    actor_loss = (
                        actor_loss
                        + self.router_expert_orthogonal_weight
                        * router_expert_orthogonal_loss
                    )
            if (
                self.use_selected_expert_margin_to_unselected
                and len(moe_layers) > 0
            ):
                selected_margin_losses = [
                    layer.last_selected_expert_margin_to_unselected_loss
                    for layer in moe_layers
                    if layer.last_selected_expert_margin_to_unselected_loss
                    is not None
                ]
                if len(selected_margin_losses) > 0:
                    selected_expert_margin_to_unselected_loss = torch.stack(
                        [
                            loss.to(actor_loss.device, dtype=actor_loss.dtype)
                            for loss in selected_margin_losses
                        ]
                    ).mean()
                    actor_loss = (
                        actor_loss
                        + self.selected_expert_margin_to_unselected_weight
                        * selected_expert_margin_to_unselected_loss
                    )

            kl_coef = float(
                getattr(self.config, "kl_coef", self.desired_kl or 0.0) or 0.0
            )
            if kl_coef > 0.0:
                delta_logp = logp_new - logp_old
                kl_token = (
                    ratio.detach() * delta_logp * valid_tok
                ).sum() / valid_count
                kl_loss = kl_coef * kl_token
                actor_loss = actor_loss + kl_loss
                mean_kl_token += float(kl_token.item())
                mean_kl_loss += float(kl_loss.item())

            if entropy_coef > 0.0:
                ent_tok = entropy_b.squeeze(-1)
                entropy_loss = (ent_tok * valid_tok).sum() / valid_count
                actor_loss = actor_loss - entropy_coef * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.accelerator.backward(actor_loss)
            self.accelerator.backward(critic_loss)

            if self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.accelerator.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            num_updates += 1
            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy_b.mean().item())
            if aux_base_lin_vel_loss is not None:
                mean_aux_base_lin_vel_nll += float(
                    aux_base_lin_vel_loss.item()
                )
            if aux_root_height_loss is not None:
                mean_aux_root_height_nll += float(aux_root_height_loss.item())
            if aux_base_lin_vel_std is not None:
                mean_aux_base_lin_vel_std += float(
                    aux_base_lin_vel_std.mean().item()
                )
            if aux_root_height_std is not None:
                mean_aux_root_height_std += float(
                    aux_root_height_std.mean().item()
                )
            if aux_keybody_contact_loss is not None:
                mean_aux_keybody_contact_bce += float(
                    aux_keybody_contact_loss.item()
                )
            if aux_keybody_contact_acc is not None:
                mean_aux_keybody_contact_acc += float(
                    aux_keybody_contact_acc.item()
                )
            if aux_ref_keybody_rel_pos_loss is not None:
                mean_aux_ref_keybody_rel_pos_mse += float(
                    aux_ref_keybody_rel_pos_loss.item()
                )
            if aux_robot_keybody_rel_pos_loss is not None:
                mean_aux_robot_keybody_rel_pos_mse += float(
                    aux_robot_keybody_rel_pos_loss.item()
                )
            if aux_router_command_recon_loss is not None:
                mean_aux_router_command_recon_mse += float(
                    aux_router_command_recon_loss.item()
                )
            if aux_router_future_recon_loss is not None:
                mean_aux_router_future_recon_huber += float(
                    aux_router_future_recon_loss.item()
                )
            if aux_router_switch_penalty_loss is not None:
                mean_aux_router_switch_penalty_js += float(
                    aux_router_switch_penalty_loss.item()
                )
            if aux_action_mean_butterworth_loss is not None:
                mean_aux_action_mean_butterworth += float(
                    aux_action_mean_butterworth_loss.item()
                )
            if moe_load_balance_loss is not None:
                mean_moe_load_balance_loss += float(
                    moe_load_balance_loss.item()
                )
            if inactive_expert_margin_to_topk_loss is not None:
                mean_inactive_expert_margin_to_topk_loss += float(
                    inactive_expert_margin_to_topk_loss.item()
                )
            if router_expert_orthogonal_loss is not None:
                mean_router_expert_orthogonal_loss += float(
                    router_expert_orthogonal_loss.item()
                )
            if selected_expert_margin_to_unselected_loss is not None:
                mean_selected_expert_margin_to_unselected_loss += float(
                    selected_expert_margin_to_unselected_loss.item()
                )

        actor_policy.apply_dynamic_bias_update_from_stats()
        denom = max(1, num_updates)
        mean_value_loss /= denom
        mean_surrogate_loss /= denom
        mean_entropy /= denom
        mean_kl_token /= denom
        mean_kl_loss /= denom
        mean_kl_analytic /= max(1, num_kl_measurements)
        clip_fraction_batch_mean /= denom
        if self.schedule == "adaptive":
            self._apply_adaptive_lr(kl_windowed)
        mean_aux_base_lin_vel_nll /= denom
        mean_aux_root_height_nll /= denom
        mean_aux_base_lin_vel_std /= denom
        mean_aux_root_height_std /= denom
        mean_aux_keybody_contact_bce /= denom
        mean_aux_keybody_contact_acc /= denom
        mean_aux_ref_keybody_rel_pos_mse /= denom
        mean_aux_robot_keybody_rel_pos_mse /= denom
        mean_aux_router_command_recon_mse /= denom
        mean_aux_router_future_recon_huber /= denom
        mean_aux_router_switch_penalty_js /= denom
        mean_aux_action_mean_butterworth /= denom
        mean_moe_load_balance_loss /= denom
        mean_inactive_expert_margin_to_topk_loss /= denom
        mean_router_expert_orthogonal_loss /= denom
        mean_selected_expert_margin_to_unselected_loss /= denom
        self._last_update_metrics["0-Train/num_updates_executed"] = float(
            num_updates
        )
        self._last_update_metrics["0-Train/kl_windowed"] = float(
            kl_windowed or 0.0
        )
        self._last_update_metrics["0-Train/kl_stop_triggered"] = float(
            kl_stop_triggered
        )
        self._last_update_metrics["0-Train/kl_stop_analytic"] = float(
            kl_stop_analytic
        )
        self._last_update_metrics["0-Train/kl_analytic_batch_last"] = float(
            kl_analytic_batch_last
        )
        self._last_update_metrics["0-Train/kl_analytic_batch_max"] = float(
            kl_analytic_batch_max
        )
        self._last_update_metrics["0-Train/clip_fraction_batch_mean"] = float(
            clip_fraction_batch_mean
        )
        self._last_update_metrics["0-Train/clip_fraction_batch_last"] = float(
            clip_fraction_batch_last
        )
        moe_layers = [
            layer
            for layer in actor_unwrapped.actor_module.layers
            if isinstance(layer, GroupedMoEBlock)
        ]
        moe_ema_dead_expert_ratio = None
        moe_ema_max_expert_frac = None
        moe_selected_expert_margin_to_unselected = None
        if len(moe_layers) > 0:
            moe_metrics = self._summarize_moe_layer_stats(moe_layers)
            moe_ema_dead_expert_ratio = moe_metrics[
                "moe_ema_dead_expert_ratio"
            ]
            moe_ema_max_expert_frac = moe_metrics["moe_ema_max_expert_frac"]
            moe_selected_expert_margin_to_unselected = moe_metrics[
                "moe_selected_expert_margin_to_unselected"
            ]

        self.storage.clear()
        loss_out = {
            "value_function": mean_value_loss,
            "critic_explained_variance": critic_explained_variance,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_token": mean_kl_token,
            "kl_loss": mean_kl_loss,
            "kl_analytic": mean_kl_analytic,
            "aux_base_lin_vel_nll": mean_aux_base_lin_vel_nll,
            "aux_root_height_nll": mean_aux_root_height_nll,
            "aux_base_lin_vel_std": mean_aux_base_lin_vel_std,
            "aux_root_height_std": mean_aux_root_height_std,
            "aux_keybody_contact_bce": mean_aux_keybody_contact_bce,
            "aux_keybody_contact_acc": mean_aux_keybody_contact_acc,
            "aux_ref_keybody_rel_pos_mse": mean_aux_ref_keybody_rel_pos_mse,
            "aux_robot_keybody_rel_pos_mse": (
                mean_aux_robot_keybody_rel_pos_mse
            ),
            "aux_router_command_recon_mse": mean_aux_router_command_recon_mse,
            "aux_router_future_recon_huber": (
                mean_aux_router_future_recon_huber
            ),
            "aux_router_switch_penalty_js": (
                mean_aux_router_switch_penalty_js
            ),
            "aux_action_mean_butterworth_raw": (
                mean_aux_action_mean_butterworth
            ),
            "aux_action_mean_butterworth_weighted": (
                self.aux_action_mean_butterworth_weight
                * mean_aux_action_mean_butterworth
            ),
            "moe_load_balance": mean_moe_load_balance_loss,
            "inactive_expert_margin_to_topk": (
                mean_inactive_expert_margin_to_topk_loss
            ),
            "router_expert_orthogonal": mean_router_expert_orthogonal_loss,
            "selected_expert_margin_to_unselected": (
                mean_selected_expert_margin_to_unselected_loss
            ),
            "moe_ema_dead_expert_ratio": moe_ema_dead_expert_ratio,
            "moe_ema_max_expert_frac": moe_ema_max_expert_frac,
            "moe_selected_expert_margin_to_unselected": (
                moe_selected_expert_margin_to_unselected
            ),
        }
        for layer in moe_layers:
            if hasattr(layer, "expansion_metrics"):
                loss_out.update(layer.expansion_metrics())
        if self.is_distributed:
            reduced_out = {}
            for k, v in loss_out.items():
                if v is None:
                    reduced_out[k] = None
                    continue
                t = torch.tensor(v, device=self.device, dtype=torch.float32)
                reduced_t = self.accelerator.reduce(t, reduction="mean")
                reduced_out[k] = float(reduced_t.item())
            loss_out = reduced_out

        self._post_update_hook(loss_out)
        return loss_out
