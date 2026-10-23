"""Frozen old value function and independent trainable sliding value net."""

from __future__ import annotations
import copy

import torch
from tensordict.nn import TensorDictModuleBase

from holomotion.src.modules.agent_modules import PPOCritic
from holomotion.src.motion_tracking.incremental_task import (
    TASK_KEY,
    config_hash,
    tensor_state_hash,
    validate_task_tensor,
)


class FrozenLegacySlidingCritic(TensorDictModuleBase):
    def __init__(self, obs_schema, module_config_dict, *, obs_example):
        super().__init__()
        cfg = copy.deepcopy(module_config_dict)
        cfg["type"] = "MLP"
        spec = cfg.pop("task_isolation")
        if config_hash(cfg) != spec["legacy_critic_config_hash"]:
            raise ValueError("Legacy Critic configuration mismatch")
        self.task = spec["task"]
        if self.task not in {"legacy", "sliding"}:
            raise ValueError("Unknown Critic task")
        self.allow_legacy_bootstrap = bool(spec["bootstrap_from_legacy"])
        self.legacy_state_hash = spec["legacy_critic_state_hash"]
        self._baseline_verified = False
        self.freeze_cycle_statistics = False
        # Both instances have separate parameters, assemblers and normalizers.
        self.legacy = PPOCritic(obs_schema, cfg, obs_example=obs_example)
        self.sliding = PPOCritic(obs_schema, cfg, obs_example=obs_example)
        self.in_keys = [*self.legacy.in_keys, TASK_KEY]
        self.out_keys = ["values"]
        self.register_buffer("isolation_version", torch.tensor(1))
        # task selection is an evaluation choice; both tasks share this bundle.
        identity = {
            k: v
            for k, v in spec.items()
            if k not in {"task", "bootstrap_from_legacy"}
        }
        identity["critic_config"] = cfg
        identity["critic_schema"] = obs_schema
        self.register_buffer(
            "contract_digest",
            torch.tensor(
                list(bytes.fromhex(config_hash(identity))), dtype=torch.uint8
            ),
        )
        self._freeze_legacy()
        self.register_load_state_dict_post_hook(self._after_load)

    @property
    def critic_module(self):
        return self.sliding.critic_module

    def _freeze_legacy(self):
        self.legacy.requires_grad_(False)
        self.legacy.eval()
        for parameter in self.legacy.parameters():
            parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze_legacy()
        return self

    def forward(self, obs_td, update_obs_norm=True, **kwargs):
        if not self._baseline_verified:
            raise ValueError("Critic baseline has not been strictly loaded")
        validate_task_tensor(obs_td, self.task)
        if self.task == "legacy":
            with torch.no_grad():
                return self.legacy(obs_td, update_obs_norm=False, **kwargs)
        return self.sliding(
            obs_td,
            update_obs_norm=update_obs_norm
            and not self.freeze_cycle_statistics,
            **kwargs,
        )

    @property
    def registry(self):
        from types import MappingProxyType

        return MappingProxyType(
            {
                "legacy_critic_v1": self.legacy,
                "sliding_critic_v1": self.sliding,
            }
        )

    def optimizer_parameters(self):
        allowed = tuple(self.sliding.parameters())
        if {id(p) for p in self.parameters() if p.requires_grad} != {
            id(p) for p in allowed
        }:
            raise ValueError("Critic trainable identity whitelist mismatch")
        return allowed

    def evaluate_registered_tasks(self, obs_td, update_obs_norm=False):
        """Task-indexed value diagnostics; mixed-task PPO remains gated off."""
        if not self._baseline_verified:
            raise ValueError("Critic baseline has not been strictly loaded")
        ids = obs_td.get(TASK_KEY, None)
        if (
            ids is None
            or ids.dtype != torch.int64
            or ids.shape != (*obs_td.batch_size, 1)
        ):
            raise ValueError("Malformed Critic registry task IDs")
        if (
            not torch.all((ids == 0) | (ids == 1))
            or len(obs_td.batch_size) != 1
        ):
            raise ValueError("Unknown or unflattened Critic registry task")
        result = obs_td.clone(recurse=False)
        values = torch.zeros_like(ids, dtype=torch.float32)
        for task, model in ((0, self.legacy), (1, self.sliding)):
            rows = torch.nonzero(ids[:, 0] == task, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            if task == 0:
                with torch.no_grad():
                    selected = model(obs_td[rows], update_obs_norm=False)[
                        "values"
                    ]
            else:
                selected = model(
                    obs_td[rows],
                    update_obs_norm=update_obs_norm
                    and not self.freeze_cycle_statistics,
                )["values"]
            values = values.index_copy(0, rows, selected)
        result.set("values", values)
        return result

    @torch.no_grad()
    def finish_statistics_cycle(self, obs_td, accelerator=None):
        """Update task-local statistics after PPO, using global DDP moments."""
        validate_task_tensor(obs_td, "sliding")
        if not self.sliding.obs_norm_enabled:
            return
        norm = self.sliding.obs_normalizer
        if norm.update_method != "ema":
            raise ValueError(
                "Isolated cycle statistics currently require registered EMA"
            )
        raw = self.sliding.assembler(obs_td).double()
        count = torch.tensor(
            float(len(raw)), device=raw.device, dtype=torch.float64
        )
        sums, squares = (
            raw.sum(0, keepdim=True),
            raw.square().sum(0, keepdim=True),
        )
        if accelerator is not None:
            count = accelerator.reduce(count, reduction="sum")
            sums = accelerator.reduce(sums, reduction="sum")
            squares = accelerator.reduce(squares, reduction="sum")
        if count <= 0:
            return
        momentum = float(norm.ema_momentum)
        norm._mean.mul_(1 - momentum).add_(
            (sums / count).to(norm._mean.dtype), alpha=momentum
        )
        norm._ex2.mul_(1 - momentum).add_(
            (squares / count).to(norm._ex2.dtype), alpha=momentum
        )
        norm._var.copy_((norm._ex2 - norm._mean.square()).clamp_min(0))
        norm._std.copy_(norm._var.sqrt())
        norm.count.add_(count.long())

    def _after_load(self, module, incompatible_keys):
        self._freeze_legacy()
        if (
            tensor_state_hash(self.legacy.state_dict())
            != self.legacy_state_hash
        ):
            raise ValueError(
                "Preserved legacy Critic differs from registered baseline"
            )
        self._baseline_verified = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        expanded = any(
            key.startswith(prefix + branch + ".")
            for key in state_dict
            for branch in ("legacy", "sliding")
        ) or any(
            prefix + k in state_dict
            for k in ("isolation_version", "contract_digest")
        )
        if expanded:
            for name in ("isolation_version", "contract_digest"):
                key = prefix + name
                if key in state_dict and not torch.equal(
                    state_dict[key].cpu(), getattr(self, name).cpu()
                ):
                    error_msgs.append(f"Incremental Critic {name} mismatch")
        elif self.allow_legacy_bootstrap:
            source = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if tensor_state_hash(source) != self.legacy_state_hash:
                error_msgs.append(
                    "Legacy Critic bootstrap source hash mismatch"
                )
            else:
                for key in source:
                    del state_dict[prefix + key]
                for branch in ("legacy", "sliding"):
                    for key, value in source.items():
                        state_dict[prefix + branch + "." + key] = value.clone()
                for name in ("isolation_version", "contract_digest"):
                    state_dict[prefix + name] = getattr(self, name).clone()
        else:
            error_msgs.append(
                "Legacy Critic bootstrap requires explicit permission"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
