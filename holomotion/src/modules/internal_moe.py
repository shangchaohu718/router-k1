"""Fine-expert expansion inside one frozen reference-routed Transformer.

The original state keys and parameter storage stay intact. Only ``added_*``
parameters can train. This is independent of the earlier dual-Actor control.
"""

from __future__ import annotations
import copy
import inspect

import torch
from torch import nn
from torch.nn import functional as F

from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.cached_replay import cached_sequence_replay
from holomotion.src.modules.expert_admission import ExpertAdmissionGate
from holomotion.src.modules.network_modules import (
    GroupedMoEBlock,
    ReferenceRoutedGroupedMoETransformerPolicy,
    _grouped_linear,
)


class InternalExpertMoEBlock(GroupedMoEBlock):
    def __init__(self, native, block_kwargs, expansion):
        super().__init__(**block_kwargs)
        self.load_state_dict(native.state_dict(), strict=True)
        self.num_added_experts = int(expansion.get("num_added_experts", 2))
        if self.num_added_experts < 2 or self.top_k != 2:
            raise ValueError(
                "Internal expansion requires >=2 added experts and Top-2"
            )
        self.total_experts = self.num_fine_experts + self.num_added_experts
        self.route_mode = str(expansion.get("mode", "cooperative"))
        self.stage = str(expansion.get("stage", "A"))
        self.warmup = bool(expansion.get("warmup", False))
        self.warmup_margin = float(expansion.get("warmup_margin", 0.5))
        if self.warmup_margin <= 0 or not torch.isfinite(
            torch.tensor(self.warmup_margin)
        ):
            raise ValueError("warmup_margin must be finite and positive")
        self.added_router = nn.Linear(
            self.d_model, self.num_added_experts, bias=True
        )
        self.added_admission = (
            ExpertAdmissionGate(self.d_model, **expansion["admission"])
            if "admission" in expansion
            else None
        )
        self._native_expert_kernel = False
        self._replay_records = None
        self.added_gate_up_proj = nn.Parameter(
            self.gate_up_proj.new_empty(
                self.num_added_experts, self.d_model, 2 * self.intermediate_dim
            )
        )
        self.added_down_proj = nn.Parameter(
            self.down_proj.new_empty(
                self.num_added_experts, self.intermediate_dim, self.d_model
            )
        )
        self.initial_score_bias = float(
            expansion.get("initial_score_bias", -2.0)
        )
        if not torch.isfinite(torch.tensor(self.initial_score_bias)):
            raise ValueError("initial_score_bias must be finite")
        self.register_buffer(
            "expansion_version",
            torch.tensor(
                2 if self.added_admission is not None else 1, dtype=torch.long
            ),
        )
        self.register_buffer(
            "added_assignment_counts",
            torch.zeros(self.total_experts, dtype=torch.long),
            persistent=False,
        )
        self.last_topk_indices = None
        self.last_topk_weights = None
        self.last_expanded_logits = None
        self.initialize_added_from_old()
        self.set_expansion(self.route_mode, self.stage, self.warmup)

    @torch.no_grad()
    def initialize_added_from_old(self):
        indices = (
            torch.arange(
                self.num_added_experts, device=self.gate_up_proj.device
            )
            % self.num_fine_experts
        )
        self.added_gate_up_proj.copy_(self.gate_up_proj[indices])
        self.added_down_proj.copy_(self.down_proj[indices])
        self.added_router.weight.copy_(self.router.weight[indices])
        self.added_router.bias.fill_(self.initial_score_bias)

    def set_expansion(self, mode, stage, warmup=False):
        if mode not in {"legacy", "cooperative", "new_only"}:
            raise ValueError(f"Unknown internal routing mode: {mode}")
        if stage not in {"A", "B", "eval"}:
            raise ValueError(f"Unknown internal expansion stage: {stage}")
        if warmup and (stage != "A" or mode != "cooperative"):
            raise ValueError(
                "Warmup is restricted to Stage A cooperative routing"
            )
        if warmup and self.use_dynamic_bias:
            raise ValueError(
                "Pilot warmup requires native use_dynamic_bias=false"
            )
        self.route_mode, self.stage, self.warmup = mode, stage, bool(warmup)
        self.freeze_router = True
        # Freeze original routing buffers as well as old parameters.
        self.apply_freeze()

    def apply_freeze(self):
        allowed = {id(p) for p in self.trainable_parameters()}
        for parameter in self.parameters():
            trainable = id(parameter) in allowed
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None

    def trainable_parameters(self):
        router = tuple(self.added_router.parameters())
        if self.stage == "A":
            return (self.added_gate_up_proj, self.added_down_proj, *router)
        if self.stage == "B":
            gate = (
                ()
                if self.added_admission is None
                else tuple(self.added_admission.parameters())
            )
            return (*router, *gate)
        return ()

    def train(self, mode=True):
        # Native dropout/attention stay in the original evaluation mode.
        super().train(False)
        if hasattr(self, "added_router"):
            self.added_router.train(mode)
            self.apply_freeze()
        return self

    def accumulate_routing_stats(self, topk_idx):
        # Never mutate the original routing_counts_accum / EMA / bias.
        return None

    def apply_bias_update_from_counts(self):
        return None

    def _update_routed_expert_stats_and_floor_loss(self, **kwargs):
        # Legacy computation must not update old usage buffers or losses.
        zero = kwargs["dense_distribution"].new_zeros(())
        self.last_moe_load_balance_loss = zero
        self.last_inactive_expert_margin_to_topk_loss = zero
        self.last_selected_expert_margin_to_unselected_loss = zero
        return zero

    def reset_routing_stats(self):
        self.added_assignment_counts.zero_()
        self.last_router_distribution = None
        self.last_router_logits = None
        self.last_topk_indices = None
        self.last_topk_weights = None
        self.last_expanded_logits = None
        self.last_admission_logits = None
        self.last_admission_mask = None
        self.last_moe_load_balance_loss = None
        self.last_inactive_expert_margin_to_topk_loss = None
        self.last_selected_expert_margin_to_unselected_loss = None

    def _compute_with_grouped_mm(self, x_sorted, sorted_expert_ids):
        if self.route_mode == "legacy" or self._native_expert_kernel:
            # Exact native expert kernel, without concatenating or reordering.
            return super()._compute_with_grouped_mm(
                x_sorted, sorted_expert_ids
            )
        counts = torch.bincount(
            sorted_expert_ids.long(), minlength=self.total_experts
        )
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        gate_up = torch.cat(
            (self.gate_up_proj, self.added_gate_up_proj), dim=0
        )
        down = torch.cat((self.down_proj, self.added_down_proj), dim=0)
        projected = _grouped_linear(x_sorted, gate_up, offs=offsets)
        gate, up = projected.chunk(2, dim=-1)
        return _grouped_linear(F.silu(gate) * up, down, offs=offsets)

    def _record_expansion(self, indices, weights, logits):
        self.last_topk_indices = indices.detach()
        self.last_topk_weights = (
            weights.detach() if weights is not None else None
        )
        self.last_expanded_logits = logits
        if self.collect_routing_stats:
            with torch.no_grad():
                self.added_assignment_counts.add_(
                    torch.bincount(
                        indices.reshape(-1), minlength=self.total_experts
                    )
                )

    def expansion_metrics(self):
        counts = self.added_assignment_counts.float()
        total = counts.sum().clamp_min(1)
        added = counts[self.num_fine_experts :]
        return {
            "internal_added_assignment_fraction": float(
                (added.sum() / total).item()
            ),
            "internal_added_active_experts": float((added > 0).sum().item()),
            "internal_added_min_assignments": float(added.min().item()),
        }

    def _native_ffn(self, x, router_x):
        # Full original batch, original bank and grouped kernel. A subset
        # dispatch would change GEMM layouts even for closed-gate tokens.
        self._native_expert_kernel = True
        try:
            return super().compute_moe_ffn(
                x, router_x, return_routing_debug=True
            )
        finally:
            self._native_expert_kernel = False

    def _save_replay_record(self, record):
        if self._replay_records is not None:
            self._replay_records.append(record)

    def _added_losses(self, old, new, raw_new, indices, choice, winner, valid):
        zero = new.new_zeros(())
        self.last_moe_load_balance_loss = zero
        self.last_inactive_expert_margin_to_topk_loss = zero
        self.last_selected_expert_margin_to_unselected_loss = zero
        if (
            self.stage != "A"
            or self.route_mode != "cooperative"
            or not valid.any()
        ):
            return
        old, new, raw_new, indices, choice, winner = (
            value[valid]
            for value in (old, new, raw_new, indices, choice, winner)
        )
        conditional = new.softmax(-1).mean(0)
        counts = torch.bincount(
            indices.reshape(-1), minlength=self.total_experts
        )[self.num_fine_experts :].float()
        hard = counts / counts.sum().clamp_min(1)
        if self.moe_load_balance_enabled:
            self.last_moe_load_balance_loss = (
                self.num_added_experts * (hard.detach() * conditional).sum()
            )
        if self.inactive_expert_margin_to_topk_enabled:
            inactive = (
                counts
                < counts.max().clamp_min(1)
                * self.inactive_expert_margin_to_topk_ratio_floor
            )
            threshold = choice.gather(-1, indices)[..., -1:].detach()
            gaps = F.relu(threshold - choice[..., self.num_fine_experts :])
            self.last_inactive_expert_margin_to_topk_loss = (
                gaps * inactive
            ).sum() / (inactive.sum().clamp_min(1) * len(new))
            if self.warmup:
                old_threshold = old.topk(self.top_k, dim=-1).values[..., -1:]
                gap = old_threshold.detach() + 0.1 - raw_new.gather(-1, winner)
                self.last_inactive_expert_margin_to_topk_loss = (
                    self.last_inactive_expert_margin_to_topk_loss
                    + F.softplus(gap).mean()
                )

    def finalize_cached_replay(self, record):
        valid = record["valid_mask"] & record["admitted"].squeeze(-1).bool()
        self._added_losses(
            record["old_logits"],
            record["new_logits"],
            record["raw_new_logits"],
            record["indices"],
            record["choice"],
            record["winner"],
            valid,
        )
        self.last_topk_indices = record["indices"].detach()
        self.last_topk_weights = record["weights"].detach()
        self.last_expanded_logits = record["logits"]
        self.last_router_logits = (
            record["logits"] if self.capture_router_logits else None
        )
        self.last_admission_logits = record["gate_logits"].squeeze(-1)
        self.last_admission_mask = record["admitted"].squeeze(-1)
        self.last_router_distribution = (
            record["logits"].softmax(-1)
            if self.routing_score_fn == "softmax"
            else record["logits"].sigmoid()
        )
        if self.routing_score_fn != "softmax":
            self.last_router_distribution = (
                self.last_router_distribution
                / self.last_router_distribution.sum(
                    -1, keepdim=True
                ).clamp_min(1e-20)
            )

    def compute_moe_ffn(self, x, router_x=None, *, return_routing_debug=False):
        if torch.onnx.is_in_onnx_export():
            raise NotImplementedError(
                "Internal expansion export has not been validated"
            )
        gate = self.added_admission
        admitted = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        gate_logits = x.new_zeros(x.shape[:-1])
        if self.route_mode == "legacy":
            admitted.zero_()
        if gate is not None and self.route_mode != "legacy":
            gate_logits = gate.logits(router_x)
            admitted = gate.hard_mask(router_x)
        self.last_admission_logits, self.last_admission_mask = (
            gate_logits,
            admitted,
        )
        if self.route_mode == "legacy" or not admitted.all():
            # Exclude added experts before softmax AND top-k by running exactly
            # the old operation. Arbitrarily large added scores cannot act.
            output, indices, logits = self._native_ffn(x, router_x)
            native = output
        if self.route_mode == "legacy" or not admitted.any():
            if self.last_router_distribution is not None:
                self.last_router_distribution = F.pad(
                    self.last_router_distribution, (0, self.num_added_experts)
                )
            self._record_expansion(indices, None, logits)
            new = x.new_zeros((*x.shape[:-1], self.num_added_experts))
            selected = logits.gather(-1, indices)
            weights = (
                selected.softmax(-1)
                if self.routing_score_fn == "softmax"
                else selected.sigmoid()
            )
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
            self._save_replay_record(
                dict(
                    old_logits=logits,
                    new_logits=new,
                    raw_new_logits=new,
                    logits=F.pad(
                        logits, (0, self.num_added_experts), value=-torch.inf
                    ),
                    indices=indices,
                    weights=weights,
                    choice=F.pad(
                        logits, (0, self.num_added_experts), value=-torch.inf
                    ),
                    gate_logits=gate_logits.unsqueeze(-1),
                    admitted=torch.zeros_like(admitted).unsqueeze(-1),
                    winner=indices[..., :1] % self.num_added_experts,
                )
            )
            return (
                (output, indices, logits) if return_routing_debug else output
            )
        output, indices, logits = self._compute_expanded_ffn(
            x, router_x, admitted, gate_logits
        )
        if not admitted.all():
            output = torch.where(admitted.unsqueeze(-1), output, native)
        return (output, indices, logits) if return_routing_debug else output

    def _compute_expanded_ffn(self, x, router_x, admitted, gate_logits):
        if router_x is None or router_x.shape != x.shape:
            raise ValueError(
                "Internal expansion requires the existing reference "
                "router hidden features"
            )
        old_logits = self.router(router_x).float()
        new_logits = self.added_router(router_x).float()
        raw_new_logits = new_logits
        winner = old_logits.argmax(-1, keepdim=True) % self.num_added_experts
        if self.warmup:
            # Stateless per-reference warmup: exactly the same rule in rollout
            # and PPO replay. Guarantee at least one new slot, retaining Top-2.
            winner = (
                old_logits.argmax(-1, keepdim=True) % self.num_added_experts
            )
            chosen = new_logits.gather(-1, winner)
            target = (
                torch.cat((old_logits, new_logits), -1)
                .max(-1, keepdim=True)
                .values
            )
            boost = (target - chosen + self.warmup_margin).detach()
            new_logits = new_logits.scatter_add(-1, winner, boost)
        logits = torch.cat((old_logits, new_logits), -1)
        logits = logits.clone()
        logits[..., self.num_fine_experts :] = torch.where(
            admitted.unsqueeze(-1), new_logits, -torch.inf
        )
        if self.route_mode == "new_only":
            logits = logits.clone()
            logits[..., : self.num_fine_experts] = torch.where(
                admitted.unsqueeze(-1), -torch.inf, old_logits
            )
        bias = (
            F.pad(self.expert_bias.float(), (0, self.num_added_experts))
            if self.use_dynamic_bias
            else 0.0
        )
        if self.routing_score_fn == "softmax":
            choice = logits + bias
            indices = choice.topk(self.top_k, dim=-1).indices
            dense = logits.softmax(-1)
            selected = torch.exp(
                logits.gather(-1, indices) - logits.logsumexp(-1, keepdim=True)
            )
        else:
            scores = logits.sigmoid()
            choice = scores + bias
            if self.route_mode == "new_only":
                choice = choice.clone()
                choice[..., : self.num_fine_experts] = torch.where(
                    admitted.unsqueeze(-1),
                    -torch.inf,
                    choice[..., : self.num_fine_experts],
                )
            choice[..., self.num_fine_experts :] = torch.where(
                admitted.unsqueeze(-1),
                choice[..., self.num_fine_experts :],
                -torch.inf,
            )
            indices = choice.topk(self.top_k, dim=-1).indices
            dense = scores / scores.sum(-1, keepdim=True).clamp_min(1e-20)
            selected = scores.gather(-1, indices)
        weights = selected / selected.sum(-1, keepdim=True).clamp_min(1e-20)
        self.last_router_distribution = (
            dense
            if (
                self.capture_router_distribution
                or self.collect_router_distribution
            )
            else None
        )
        self.last_router_logits = (
            logits if self.capture_router_logits else None
        )
        self._record_expansion(indices, weights, logits)
        self._added_losses(
            old_logits,
            new_logits,
            raw_new_logits,
            indices,
            choice,
            winner,
            admitted,
        )
        self._save_replay_record(
            dict(
                old_logits=old_logits,
                new_logits=new_logits,
                raw_new_logits=raw_new_logits,
                logits=logits,
                indices=indices,
                weights=weights,
                choice=choice,
                gate_logits=gate_logits.unsqueeze(-1),
                admitted=admitted.unsqueeze(-1),
                winner=winner,
            )
        )
        sparse = self._compute_sparse_experts(
            x, indices, (weights * self.routing_scale).to(x.dtype)
        )
        output = self.mlp_dropout(self.shared_experts(x) + sparse)
        return output, indices, logits


class InternalExpandedMoEPolicy(ReferenceRoutedGroupedMoETransformerPolicy):
    def __init__(self, input_dim, output_dim, module_config_dict):
        config = copy.deepcopy(module_config_dict)
        expansion = config.get("internal_expansion", {})
        self.sequence_replay = expansion.get("sequence_replay", "vectorized")
        if self.sequence_replay not in {"vectorized", "cached"}:
            raise ValueError("sequence_replay must be vectorized or cached")
        self.replay_chunk_size = expansion.get("replay_chunk_size", 0)
        if (
            type(self.replay_chunk_size) is not int
            or self.replay_chunk_size < 0
        ):
            raise ValueError("replay_chunk_size must be a nonnegative integer")
        if self.sequence_replay == "cached":
            for key in ("aux_router_future_recon",):
                if config.get(key, {}).get("enabled", False):
                    raise ValueError(f"Cached replay does not support {key}")
        # For this first version, pre-MoE KV tensors are independent of expert
        # routing. Multi-layer route switches need a different history audit.
        if (
            int(config.get("n_layers", 0)) != 1
            or config.get("dense_layer_at_first", False)
            or config.get("dense_layer_at_last", False)
        ):
            raise ValueError(
                "Internal prototype requires one native MoE layer"
            )
        for key in ("aux_router_command_recon", "aux_router_switch_penalty"):
            if config.get(key, {}).get("enabled", False):
                raise ValueError(f"Internal prototype does not support {key}")
        super().__init__(input_dim, output_dim, config)
        if not isinstance(self.layers[0], GroupedMoEBlock):
            raise ValueError(
                "Internal prototype requires one native MoE layer, "
                "not a dense layer"
            )
        # Reconstruct the block with its native constructor settings.
        # Keep state keys, old expert count and original grouped kernel.
        kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(GroupedMoEBlock).parameters
            if hasattr(self, name)
        }
        kwargs["use_cross_attn"] = self.use_future_cross_attn
        self.layers[0] = InternalExpertMoEBlock(
            self.layers[0], kwargs, expansion
        )
        self.num_fine_experts = self.layers[0].total_experts
        self.apply_freeze()

    def apply_freeze(self):
        self.requires_grad_(False)
        self.layers[0].apply_freeze()

    def train(self, mode=True):
        super().train(False)
        self.layers[0].train(mode)
        self.apply_freeze()
        return self

    def set_expansion(
        self, mode, *, stage="eval", warmup=False, admission_mode=None
    ):
        gate = self.layers[0].added_admission
        if admission_mode is not None and (
            gate is None or admission_mode not in gate.MODES
        ):
            raise ValueError(
                "admission_mode requires a configured gate and valid mode"
            )
        previous = (
            self.layers[0].route_mode,
            self.layers[0].stage,
            self.layers[0].warmup,
            gate.mode if gate is not None else None,
        )
        self.layers[0].set_expansion(mode, stage, warmup)
        if admission_mode is not None:
            gate.set_mode(admission_mode)
        # Profiles are explicit inference contracts. Clear on changes even
        # though this single-layer KV happens to precede all expert updates.
        if previous != (
            mode,
            stage,
            bool(warmup),
            gate.mode if gate is not None else None,
        ):
            self.route_revision = getattr(self, "route_revision", 0) + 1
            self.clear_env_cache(None)
        if self._prev_last_moe_router_valid is not None:
            self._prev_last_moe_router_valid.zero_()

    def _validate_replay_precision(self, x):
        if self.sequence_replay == "cached" and torch.is_autocast_enabled(
            x.device.type
        ):
            raise ValueError(
                "Cached incremental replay requires autocast disabled; "
                "mixed-precision native chunk cache writes are not validated"
            )

    def single_step_mu(self, x):
        self._validate_replay_precision(x)
        return super().single_step_mu(x)

    def single_step_mu_chunked(self, x, chunk_size):
        self._validate_replay_precision(x)
        return super().single_step_mu_chunked(x, chunk_size)

    def sequence_mu(
        self,
        x,
        *,
        attn_mask=None,
        return_hidden=False,
        return_pre_moe_hidden=False,
        return_router_features=False,
        return_router_temporal_features=False,
    ):
        if self.sequence_replay == "cached":
            self._validate_replay_precision(x)
            if any(
                (
                    return_hidden,
                    return_router_features,
                    return_router_temporal_features,
                )
            ):
                raise ValueError(
                    "Cached replay supports action means and "
                    "native auxiliary hidden only"
                )
            return cached_sequence_replay(
                self,
                x,
                attn_mask,
                chunk_size=self.replay_chunk_size,
                return_pre_moe_hidden=return_pre_moe_hidden,
            )
        if self.layers[0].route_mode == "legacy":
            if attn_mask is not None and x.shape[1] > self.max_ctx_len:
                # Native sequence RoPE treats the first visible token as an
                # episode start. A sliding left edge inside an episode is
                # not a reset, so reject that unsupported combination rather
                # than silently changing the frozen legacy computation.
                first_visible = attn_mask.to(torch.int64).argmax(dim=-1)
                positions = torch.arange(x.shape[1], device=x.device)
                continuing = attn_mask[:, positions[1:], positions[:-1]]
                moving_left_edge = first_visible[:, 1:] > first_visible[:, :-1]
                if torch.any(continuing & moving_left_edge):
                    raise ValueError(
                        "Legacy long rolling-mask sequence is unsupported; "
                        "use cached single_step or pack each history window "
                        "within max_ctx_len"
                    )
            # Preserve original sequence semantics for exact legacy diagnosis.
            return super().sequence_mu(
                x,
                attn_mask=attn_mask,
                return_hidden=return_hidden,
                return_pre_moe_hidden=return_pre_moe_hidden,
                return_router_features=return_router_features,
                return_router_temporal_features=return_router_temporal_features,
            )
        if return_hidden and return_pre_moe_hidden:
            raise ValueError(
                "return_hidden and return_pre_moe_hidden are exclusive"
            )
        batch, length, _ = x.shape
        h = self.obs_embed(x)
        router_h = self._compute_router_hidden(x)
        t = torch.arange(length, device=x.device)
        mask = (t[:, None] >= t[None, :]).expand(batch, -1, -1)
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool or tuple(attn_mask.shape) != (
                batch,
                length,
                length,
            ):
                raise ValueError("attn_mask must be boolean [B,T,T]")
            mask = mask & attn_mask
        # Episode boundaries come from adjacency, not the first attendable
        # token: a rolling context window is not an episode reset.
        starts = torch.zeros(batch, length, dtype=torch.long, device=x.device)
        if length > 1:
            boundaries = ~mask[:, t[1:], t[:-1]]
            starts[:, 1:] = torch.where(boundaries, t[None, 1:], 0)
        episode_starts = starts.cummax(dim=1).values
        pos = t[None, :] - episode_starts
        mask = mask & (t[:, None] - t[None, :] < self.max_ctx_len)
        cos, sin = self.get_cos_sin(h, pos)
        result = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=mask[:, None],
            router_h=router_h,
            return_pre_moe_hidden=return_pre_moe_hidden,
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )
        extra = []
        if isinstance(result, tuple):
            h, extra = result[0], list(result[1:])
        else:
            h = result
        h = self.norm_f(h)
        mu = self.action_mu_head(h)
        if return_hidden:
            return mu, h
        return (mu, *extra) if extra else mu

    def _forward_inference_onnx(self, *args, **kwargs):
        raise NotImplementedError(
            "Internal expansion explicit-cache/export path is not validated"
        )


class PPOTFInternalMoEActor(PPOTFRefRouterActor):
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
        self.allow_legacy_bootstrap = bool(
            config.get("internal_expansion", {}).get(
                "bootstrap_from_legacy", False
            )
        )
        self.router_lr_multiplier = float(
            config.get("internal_expansion", {}).get(
                "router_lr_multiplier", 1.0
            )
        )
        if not 0 < self.router_lr_multiplier <= 1000:
            raise ValueError("router_lr_multiplier must be in (0,1000]")
        config["type"] = "ReferenceRoutedGroupedMoETransformerPolicy"
        config["fix_sigma"] = True
        super().__init__(
            obs_schema,
            config,
            num_actions,
            init_noise_std,
            obs_example=obs_example,
        )
        native = self.actor_module
        expanded_config = copy.deepcopy(native.module_config_dict)
        self.actor_module = InternalExpandedMoEPolicy(
            native.input_dim, native.output_dim, expanded_config
        )
        self.actor_net_type = "InternalExpandedMoEPolicy"
        self.register_load_state_dict_post_hook(self._after_load)
        self._freeze_preprocessing()

    def optimizer_parameter_groups(self, *, adamw):
        groups = {}
        layer = self.actor_module.layers[0]
        allowed = {id(p) for p in layer.trainable_parameters()}
        router_ids = {id(p) for p in layer.added_router.parameters()}
        if layer.added_admission is not None:
            router_ids.update(
                id(p) for p in layer.added_admission.parameters()
            )
        if {id(p) for p in self.parameters() if p.requires_grad} != allowed:
            raise ValueError(
                "Internal Actor trainable identity whitelist mismatch"
            )
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            scale = (
                self.router_lr_multiplier
                if id(parameter) in router_ids
                else 1.0
            )
            decay = 0.01 if adamw and parameter.ndim >= 2 else 0.0
            groups.setdefault((scale, decay), []).append(parameter)
        result = [
            {"params": params, "lr_multiplier": scale, "weight_decay": decay}
            for (scale, decay), params in groups.items()
        ]
        # The native evaluator constructs an optimizer for frozen models too.
        # Empty groups are legal, with no dummy or old parameters included.
        return result or [
            {"params": [], "lr_multiplier": 1.0, "weight_decay": 0.0}
        ]

    def _freeze_preprocessing(self):
        self.obs_normalizer.requires_grad_(False)
        self.obs_normalizer.eval()
        for name in ("log_std", "std"):
            if hasattr(self, name):
                getattr(self, name).requires_grad_(False)
                getattr(self, name).grad = None

    def train(self, mode=True):
        super().train(mode)
        self._freeze_preprocessing()
        return self

    def _normalize_actor_obs(self, obs, update):
        return super()._normalize_actor_obs(obs, update=False)

    def override_sigma(self, *args, **kwargs):
        raise ValueError(
            "Internal expansion preserves checkpoint sigma; "
            "disable override_sigma"
        )

    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError(
            "Internal expansion ONNX export has not been validated"
        )

    def _after_load(self, module, incompatible_keys):
        self._freeze_preprocessing()
        self.actor_module.apply_freeze()
        self.actor_module.clear_env_cache(None)
        self.actor_module.reset_routing_stats()

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
        expansion_prefix = prefix + "actor_module.layers.0."
        version_key = expansion_prefix + "expansion_version"
        added_keys = [
            expansion_prefix + name
            for name in self.actor_module.layers[0].state_dict()
            if name.startswith("added_")
        ]
        # Never mistake a partial expanded checkpoint for a legacy one.
        expanded = version_key in state_dict or any(
            key in state_dict for key in added_keys
        )
        if not expanded:
            if not self.allow_legacy_bootstrap:
                error_msgs.append(
                    "Legacy bootstrap requires "
                    "internal_expansion.bootstrap_from_legacy=true "
                    "and finetune"
                )
            else:
                # Fill only new keys. All old keys and shapes are still strict.
                layer = self.actor_module.layers[0]
                gate_key, down_key = (
                    expansion_prefix + "gate_up_proj",
                    expansion_prefix + "down_proj",
                )
                router_key = expansion_prefix + "router.weight"
                if all(
                    key in state_dict
                    for key in (gate_key, down_key, router_key)
                ):
                    index = (
                        torch.arange(layer.num_added_experts)
                        % layer.num_fine_experts
                    )
                    gate, down = state_dict[gate_key], state_dict[down_key]
                    old_layout = tuple(down.shape) == (
                        layer.num_fine_experts,
                        layer.d_model,
                        layer.intermediate_dim,
                    ) and tuple(down.shape) != tuple(layer.down_proj.shape)
                    if old_layout or (
                        tuple(gate.shape)
                        == (
                            layer.num_fine_experts,
                            2 * layer.intermediate_dim,
                            layer.d_model,
                        )
                        and tuple(gate.shape)
                        != tuple(layer.gate_up_proj.shape)
                    ):
                        gate = gate.transpose(-2, -1).contiguous()
                    if old_layout:
                        down = down.transpose(-2, -1).contiguous()
                    state_dict[expansion_prefix + "added_gate_up_proj"] = gate[
                        index
                    ].clone()
                    state_dict[expansion_prefix + "added_down_proj"] = down[
                        index
                    ].clone()
                    state_dict[expansion_prefix + "added_router.weight"] = (
                        state_dict[router_key][index].clone()
                    )
                    state_dict[expansion_prefix + "added_router.bias"] = (
                        torch.full_like(
                            layer.added_router.bias, layer.initial_score_bias
                        )
                    )
                    state_dict[version_key] = (
                        layer.expansion_version.detach().clone()
                    )
                    if layer.added_admission is not None:
                        fresh_gate = ExpertAdmissionGate(
                            layer.d_model,
                            threshold=layer.added_admission.threshold,
                            initial_bias=layer.added_admission.initial_bias,
                        )
                        for name, value in fresh_gate.state_dict().items():
                            state_dict[
                                expansion_prefix + "added_admission." + name
                            ] = value.detach().clone()
        elif version_key in state_dict and int(
            state_dict[version_key].item()
        ) != int(self.actor_module.layers[0].expansion_version):
            error_msgs.append(
                "Unsupported internal expansion checkpoint version"
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
