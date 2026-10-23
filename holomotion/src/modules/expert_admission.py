"""Hard admission of added experts from observable reference features only."""

from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdmissionConfig:
    """Immutable non-mode contract, persisted as a checkpoint digest."""

    input_dim: int
    threshold: float = 0.99
    initial_bias: float = -8.0

    def __post_init__(self):
        if isinstance(self.input_dim, bool) or not isinstance(
            self.input_dim, int
        ):
            raise ValueError("admission input_dim must be a positive integer")
        if self.input_dim <= 0:
            raise ValueError("admission input_dim must be a positive integer")
        for name in ("threshold", "initial_bias"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"admission {name} must be finite numeric")
            if not math.isfinite(value):
                raise ValueError(f"admission {name} must be finite numeric")
            object.__setattr__(self, name, float(value))
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("admission threshold must lie strictly in (0, 1)")
        if self.initial_bias >= 0.0:
            raise ValueError("admission initial_bias must be negative")
        if self.initial_bias >= self.logit_threshold:
            raise ValueError("admission initialization must start closed")

    @property
    def logit_threshold(self):
        return math.log(self.threshold) - math.log1p(-self.threshold)


class ExpertAdmissionGate(nn.Module):
    """Allow added experts without blending actions or changing old scores.

    ``forward`` returns a boolean mask with the reference feature's leading
    dimensions. ``logits`` returns differentiable scalar logits of that same
    shape for supervised calibration. Hard decisions carry no gradient and
    have no straight-through estimator. The caller applies this mask before
    expert score normalization and Top-K selection.

    Only observable reference hidden features belong here. Task IDs, labels,
    action heads, and reward tensors are deliberately absent from the API.
    Mode is controlled by the enclosing versioned route profile; the remaining
    configuration and component version are validated on checkpoint loading.
    """

    VERSION = 1
    MODES = frozenset({"closed", "open", "learned"})

    def __init__(
        self,
        input_dim,
        *,
        mode="closed",
        threshold=0.99,
        initial_bias=-8.0,
    ):
        super().__init__()
        self._config = AdmissionConfig(input_dim, threshold, initial_bias)
        self.set_mode(mode)
        # Direct Parameters avoid random initialization consuming the legacy
        # policy's RNG stream. This is one linear projection, without a norm.
        self.weight = nn.Parameter(torch.zeros(1, self.input_dim))
        self.bias = nn.Parameter(torch.full((1,), self.initial_bias))
        self.register_buffer(
            "metadata_version", torch.tensor(self.VERSION, dtype=torch.int64)
        )
        self.register_buffer("config_digest", self._expected_digest())

    @property
    def input_dim(self):
        return self._config.input_dim

    @property
    def threshold(self):
        return self._config.threshold

    @property
    def initial_bias(self):
        return self._config.initial_bias

    def config_dict(self):
        return {
            "version": self.VERSION,
            "feature_source": "reference_hidden",
            "decision": "logit_greater_equal",
            **asdict(self._config),
        }

    def _expected_digest(self):
        serialized = json.dumps(
            self.config_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return torch.tensor(
            list(hashlib.sha256(serialized).digest()), dtype=torch.uint8
        )

    def set_mode(self, mode):
        if not isinstance(mode, str) or mode not in self.MODES:
            raise ValueError(f"Unknown expert admission mode: {mode!r}")
        self.mode = mode

    def _check_features(self, reference_hidden):
        if not isinstance(reference_hidden, torch.Tensor):
            raise TypeError(
                "admission input must be a reference feature tensor"
            )
        if (
            reference_hidden.ndim < 1
            or reference_hidden.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "admission reference feature width must equal "
                f"input_dim={self.input_dim}"
            )
        if not reference_hidden.is_floating_point():
            raise TypeError(
                "admission reference features must be floating point"
            )

    def logits(self, reference_hidden):
        self._check_features(reference_hidden)
        return F.linear(reference_hidden, self.weight, self.bias).squeeze(-1)

    @torch.no_grad()
    def hard_mask(self, reference_hidden):
        self._check_features(reference_hidden)
        if self.mode in {"closed", "open"}:
            return torch.full(
                reference_hidden.shape[:-1],
                self.mode == "open",
                device=reference_hidden.device,
                dtype=torch.bool,
            )
        if self.mode != "learned":
            raise ValueError(f"Unknown expert admission mode: {self.mode!r}")
        return self.logits(reference_hidden) >= self._config.logit_threshold

    def forward(self, reference_hidden):
        return self.hard_mask(reference_hidden)

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
        expected = {
            "metadata_version": torch.tensor(self.VERSION, dtype=torch.int64),
            "config_digest": self._expected_digest(),
        }
        for name, tensor in expected.items():
            value = state_dict.get(prefix + name)
            if not isinstance(value, torch.Tensor):
                error_msgs.append(
                    f"Missing expert admission metadata: {prefix}{name}"
                )
                return
            if (
                value.dtype != tensor.dtype
                or value.shape != tensor.shape
                or not torch.equal(value.detach().cpu(), tensor)
            ):
                error_msgs.append(
                    f"Incompatible expert admission metadata: {prefix}{name}"
                )
                return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
