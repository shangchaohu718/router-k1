"""Frozen general policy, independent sliding policy, and reference gate.

The specialist is a full policy copy, not an extra FFN competing in the old
Top-K pool. Both histories advance on every step, including while unselected.
"""

from __future__ import annotations
import copy

import torch
from torch import nn
from torch.nn import functional as F

from holomotion.src.modules.agent_modules import PPOTFRefRouterActor


class FrozenBaseSlidingPolicy(nn.Module):
    MODES = ("base", "specialist", "soft", "hard", "temporal_hard")

    def __init__(self, general: nn.Module, config: dict):
        super().__init__()
        self.general = general
        self.specialist = copy.deepcopy(general)
        self.routing_mode = str(config.get("mode", "specialist"))
        if self.routing_mode not in self.MODES:
            raise ValueError(
                f"sliding_router.mode must be one of {self.MODES}"
            )
        self.train_specialist = bool(config.get("train_specialist", True))
        self.train_gate = bool(config.get("train_gate", False))
        self.threshold = float(config.get("threshold", 0.5))
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("sliding_router.threshold must be in (0, 1)")
        self.exit_threshold = float(config.get("exit_threshold", 0.9))
        self.enter_steps = int(config.get("enter_steps", 20))
        self.exit_steps = int(config.get("exit_steps", 10))
        if self.routing_mode == "temporal_hard" and not (
            0 < self.exit_threshold < self.threshold < 1
        ):
            raise ValueError("Temporal gate requires 0 < exit < enter < 1")
        if self.enter_steps < 1 or self.exit_steps < 1:
            raise ValueError("Temporal gate dwell steps must be positive")
        for name in ("_slide_active", "_enter_count", "_exit_count"):
            self.register_buffer(
                name, torch.empty(0, dtype=torch.long), persistent=False
            )
        hidden = int(config.get("hidden_dim", 64))
        if hidden < 1:
            raise ValueError("sliding_router.hidden_dim must be positive")
        self.gate = nn.Sequential(
            nn.Linear(general.router_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        mask = torch.ones(general.router_input_dim)
        columns = [int(i) for i in config.get("zero_feature_columns", [])]
        if any(i < 0 or i >= len(mask) for i in columns):
            raise ValueError("Gate feature mask column is out of range")
        mask[columns] = 0
        self.register_buffer("_gate_feature_mask", mask, persistent=False)
        # Conservative classifier; forced specialist mode bypasses it.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(
            self.gate[-1].bias, float(config.get("initial_bias", -4.0))
        )
        self.last_gate_probability = None
        self.last_gate_weight = None
        self._apply_freeze()

    def __getattr__(self, name):
        # PPO consumes metadata, auxiliary heads, and MoE diagnostics.
        # Those belong exclusively to the specialist.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("specialist"), name)

    def _apply_freeze(self):
        self.general.requires_grad_(False)
        self.general.eval()
        self.general.set_collect_routing_stats(False)
        self.specialist.requires_grad_(self.train_specialist)
        # Respect the specialist's own internal router-freeze setting.
        if self.train_specialist:
            self.specialist._apply_freeze_router_state()
        self.specialist.train(self.training and self.train_specialist)
        self.gate.requires_grad_(self.train_gate)
        self.gate.train(self.training and self.train_gate)

    def train(self, mode=True):
        super().train(mode)
        self._apply_freeze()
        return self

    def set_routing_mode(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"routing mode must be one of {self.MODES}")
        if mode == "temporal_hard" and not (
            0 < self.exit_threshold < self.threshold < 1
        ):
            raise ValueError("Temporal gate requires 0 < exit < enter < 1")
        if mode != self.routing_mode:
            self._clear_gate_state(None)
        self.routing_mode = mode

    def _clear_gate_state(self, env_ids):
        for name in ("_slide_active", "_enter_count", "_exit_count"):
            state = getattr(self, name)
            if env_ids is None:
                state.zero_()
            else:
                state[env_ids] = 0

    def _gate_step(self, probability, active, entering, exiting):
        # Dwell counters count consecutive evidence, with separate entry and
        # exit thresholds. The two controllers still consume every input.
        entering = torch.where(
            (~active.bool()) & (probability >= self.threshold),
            (entering + 1).clamp_max(self.enter_steps),
            0,
        )
        exiting = torch.where(
            active.bool() & (probability < self.exit_threshold),
            (exiting + 1).clamp_max(self.exit_steps),
            0,
        )
        active = torch.where(entering >= self.enter_steps, 1, active)
        active = torch.where(exiting >= self.exit_steps, 0, active)
        return active, entering, exiting

    def _temporal_weights(self, probability, attn_mask=None):
        p = probability.detach().squeeze(-1)
        if p.ndim == 1:
            if self._slide_active.shape != p.shape:
                for name in ("_slide_active", "_enter_count", "_exit_count"):
                    setattr(self, name, torch.zeros_like(p, dtype=torch.long))
            states = self._gate_step(
                p, self._slide_active, self._enter_count, self._exit_count
            )
            self._slide_active, self._enter_count, self._exit_count = states
            return states[0].unsqueeze(-1).to(probability.dtype)
        if p.ndim != 2:
            raise ValueError("Temporal routing expects [B, D] or [B, T, D]")
        # Sequence evaluation replays from a fresh prefix, without mutating
        # live rollout state. Episode boundaries use the attention mask.
        active = torch.zeros_like(p[:, 0], dtype=torch.long)
        entering, exiting = active.clone(), active.clone()
        weights = []
        for t in range(p.shape[1]):
            if t and attn_mask is not None:
                keep = attn_mask[:, t, t - 1].to(torch.long)
                active, entering, exiting = (
                    active * keep,
                    entering * keep,
                    exiting * keep,
                )
            active, entering, exiting = self._gate_step(
                p[:, t], active, entering, exiting
            )
            weights.append(active)
        return torch.stack(weights, dim=1).unsqueeze(-1).to(probability.dtype)

    def gate_logits(self, normalized_obs):
        features = normalized_obs.index_select(
            -1, self.general._router_feature_indices
        )
        features = features * self._gate_feature_mask.to(features.dtype)
        return self.gate(features)

    def gate_supervision_loss(self, normalized_obs, slide_labels):
        """Labels include old-motion negatives and sliding positives.

        Inputs have already passed the frozen actor normalizer. This objective
        does not evaluate either controller or mutate their history caches.
        """
        logits = self.gate_logits(normalized_obs)
        labels = slide_labels.to(device=logits.device, dtype=logits.dtype)
        if labels.shape == logits.shape[:-1]:
            labels = labels.unsqueeze(-1)
        if labels.shape != logits.shape:
            raise ValueError("slide_labels must match gate logits shape")
        if (
            not torch.isfinite(labels).all()
            or ((labels < 0) | (labels > 1)).any()
        ):
            raise ValueError("slide_labels must be finite values in [0, 1]")
        return F.binary_cross_entropy_with_logits(logits, labels)

    def _combine(self, x, general_mu, specialist_mu, attn_mask=None):
        probability = self.gate_logits(x).sigmoid()
        self.last_gate_probability = probability.detach()
        if self.routing_mode == "base":
            weight = torch.zeros_like(probability)
            mu = general_mu
        elif self.routing_mode == "specialist":
            weight = torch.ones_like(probability)
            mu = specialist_mu
        elif self.routing_mode in {"hard", "temporal_hard"}:
            weight = (
                self._temporal_weights(probability, attn_mask)
                if self.routing_mode == "temporal_hard"
                else (probability >= self.threshold).to(specialist_mu.dtype)
            )
            # Exact base output when unselected; no interpolation rounding.
            mu = torch.where(weight.bool(), specialist_mu, general_mu)
        else:
            weight = probability
            mu = general_mu + weight * (specialist_mu - general_mu)
        self.last_gate_weight = weight.detach()
        return mu

    def sequence_mu(self, x, *, attn_mask=None, **kwargs):
        with torch.no_grad():
            general_mu = self.general.sequence_mu(x, attn_mask=attn_mask)
        result = self.specialist.sequence_mu(x, attn_mask=attn_mask, **kwargs)
        specialist_mu = result[0] if isinstance(result, tuple) else result
        mu = self._combine(x, general_mu, specialist_mu, attn_mask)
        return (mu, *result[1:]) if isinstance(result, tuple) else mu

    def single_step_mu(self, x):
        with torch.no_grad():
            general_mu = self.general.single_step_mu(x)
        specialist_mu = self.specialist.single_step_mu(x)
        return self._combine(x, general_mu, specialist_mu)

    def single_step_mu_chunked(self, x, chunk_size):
        with torch.no_grad():
            general_mu = self.general.single_step_mu_chunked(x, chunk_size)
        specialist_mu = self.specialist.single_step_mu_chunked(x, chunk_size)
        return self._combine(x, general_mu, specialist_mu)

    def reset_kv_cache(self, num_envs, device):
        self.general.reset_kv_cache(num_envs, device)
        self.specialist.reset_kv_cache(num_envs, device)
        for name in ("_slide_active", "_enter_count", "_exit_count"):
            setattr(
                self,
                name,
                torch.zeros(num_envs, device=device, dtype=torch.long),
            )

    def clear_env_cache(self, env_ids):
        self.general.clear_env_cache(env_ids)
        self.specialist.clear_env_cache(env_ids)
        self._clear_gate_state(env_ids)

    def set_collect_routing_stats(self, collect):
        self.general.set_collect_routing_stats(False)
        self.specialist.set_collect_routing_stats(
            collect and self.train_specialist
        )

    def reset_routing_stats(self):
        self.specialist.reset_routing_stats()

    def apply_dynamic_bias_update_from_stats(self):
        if self.train_specialist:
            self.specialist.apply_dynamic_bias_update_from_stats()

    def forward(self, x, past_key_values=None, current_pos=None):
        if past_key_values is not None:
            raise NotImplementedError(
                "Sliding prototype has two KV caches; "
                "ONNX export is not supported yet"
            )
        return self.sequence_mu(x[:, None, :])[:, 0, :]


class PPOTFFrozenBaseSlidingActor(PPOTFRefRouterActor):
    """Native PPO actor with frozen preprocessing and noise scale."""

    def __init__(
        self,
        obs_schema,
        module_config_dict,
        num_actions,
        init_noise_std,
        *,
        obs_example=None,
    ):
        config = copy.deepcopy(module_config_dict)
        router_config = config.get("sliding_router", {})
        self.allow_general_bootstrap = bool(
            router_config.get("bootstrap_from_general", False)
        )
        config["type"] = "ReferenceRoutedGroupedMoETransformerPolicy"
        config["fix_sigma"] = True
        if bool(config.get("use_dynamic_bias", False)):
            raise ValueError(
                "Sliding prototype requires use_dynamic_bias=false "
                "to preserve the frozen base"
            )
        super().__init__(
            obs_schema,
            config,
            num_actions,
            init_noise_std,
            obs_example=obs_example,
        )
        self.actor_net_type = "FrozenBaseSlidingPolicy"
        self.actor_module = FrozenBaseSlidingPolicy(
            self.actor_module, router_config
        )
        self.register_load_state_dict_post_hook(self._after_load)
        self._freeze_preprocessing()

    def _freeze_preprocessing(self):
        self.obs_normalizer.requires_grad_(False)
        self.obs_normalizer.eval()
        if hasattr(self, "log_std"):
            self.log_std.requires_grad_(False)
        else:
            self.std.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_preprocessing()
        return self

    def _normalize_actor_obs(self, obs, update):
        return super()._normalize_actor_obs(obs, update=False)

    def override_sigma(self, sigma_override):
        raise ValueError(
            "Sliding prototype preserves the general checkpoint sigma; "
            "disable override_sigma"
        )

    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError(
            "Sliding prototype ONNX export requires "
            "a two-cache deployment adapter"
        )

    def gate_supervision_loss(self, obs_td, slide_labels):
        flat_td = obs_td.flatten(0, 1) if obs_td.batch_dims == 2 else obs_td
        obs = self._normalize_actor_obs(self.assembler(flat_td), update=False)
        if obs_td.batch_dims == 2:
            obs = obs.reshape(*obs_td.batch_size, -1)
        return self.actor_module.gate_supervision_loss(obs, slide_labels)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result.set(
            "slide_gate_probability", self.actor_module.last_gate_probability
        )
        result.set("slide_gate_weight", self.actor_module.last_gate_weight)
        return result

    def _after_load(self, module, incompatible_keys):
        self._freeze_preprocessing()
        self.actor_module._apply_freeze()
        # KV tensors are runtime-only and depend on the previous weights.
        # A warm actor must never carry those keys into a new checkpoint.
        self.actor_module.clear_env_cache(None)
        self.actor_module.last_gate_probability = None
        self.actor_module.last_gate_weight = None

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
        old_prefix = prefix + "actor_module."
        # Only the explicitly enabled, legacy -> two-branch bootstrap may add
        # keys. Composite checkpoints always load strictly, including the gate.
        legacy = old_prefix + "obs_embed.0.weight" in state_dict
        if legacy:
            if not self.allow_general_bootstrap:
                error_msgs.append(
                    "Legacy general checkpoint requires "
                    "sliding_router.bootstrap_from_general=true "
                    "and finetune loading"
                )
            else:
                old_keys = [
                    key for key in state_dict if key.startswith(old_prefix)
                ]
                for key in old_keys:
                    value = state_dict.pop(key)
                    suffix = key[len(old_prefix) :]
                    state_dict[old_prefix + "general." + suffix] = value
                    state_dict[old_prefix + "specialist." + suffix] = (
                        value.clone()
                    )
                for key, value in self.actor_module.gate.state_dict().items():
                    state_dict[old_prefix + "gate." + key] = value.clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
