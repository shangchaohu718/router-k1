"""Offline admission supervision keeps labels outside the policy inputs."""

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_internal_moe import make_pair
from torch import nn
from torch.nn import functional as F

from holomotion.src.modules.expert_admission import ExpertAdmissionGate
from tools.booster_k1_retarget import fit_internal_moe_router as fitting


class GateLayer(nn.Module):
    def __init__(self, *, mode="learned"):
        super().__init__()
        self.added_admission = ExpertAdmissionGate(2, mode=mode)
        self.routing_score_fn = "softmax"
        self.total_experts = 4
        self.last_admission_logits = None
        self.last_admission_mask = None


class GateActor(nn.Module):
    def __init__(self, *, mode="learned", layers=1):
        super().__init__()
        self.layers = nn.ModuleList(
            GateLayer(mode=mode) for _ in range(layers)
        )
        self.last_obs = None

    def sequence_mu(self, obs, *, attn_mask):
        self.last_obs = obs
        for layer in self.layers:
            layer.last_admission_logits = layer.added_admission.logits(obs)
            layer.last_admission_mask = layer.added_admission(obs)
            layer.last_expanded_logits = obs.new_zeros((*obs.shape[:2], 4))
            layer.last_expanded_logits[..., 2:] = -torch.inf
        return obs.new_zeros((*obs.shape[:2], 1))


class GateStudent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.actor_module = GateActor(**kwargs)

    def _sigma_like(self, mu):
        return torch.ones_like(mu)


def fixture(labels, *, layers=1):
    labels = torch.tensor(labels, dtype=torch.long)
    model = GateStudent(layers=layers)
    obs = torch.stack((labels.float() * 2 - 1, torch.ones_like(labels)), -1)
    obs = obs[:, None, :].repeat(1, 3, 1)
    mask = torch.ones(len(labels), 3, 3, dtype=torch.bool).tril()
    mu = torch.zeros(len(labels), 1)
    routes = torch.tensor([0.5, 0.5, 0.0, 0.0]).repeat(len(labels), 1)
    return model, obs, mask, labels, (mu, mu, routes, routes)


def test_gate_bce_is_class_balanced_and_drives_only_gate_parameters():
    model, obs, mask, labels, teacher = fixture([0, 0, 0, 1])
    loss, metrics = fitting.preservation_loss(
        model, obs, mask, labels, teacher, gate_weight=2.0
    )
    logits = model.actor_module.layers[0].last_admission_logits[:, -1]
    token_loss = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    expected = 0.5 * (token_loss[:3].mean() + token_loss[3:].mean())
    torch.testing.assert_close(loss, 2.0 * expected)
    assert metrics["gate_balanced_bce"] == pytest.approx(expected.item())
    assert metrics["old_admission_rate"] == 0
    assert metrics["new_admission_rate"] == 0
    assert torch.equal(model.actor_module.last_obs, obs)
    loss.backward()
    gate = model.actor_module.layers[0].added_admission
    assert gate.weight.grad[0, 0] < 0
    assert not model.actor_module.layers[0].last_admission_mask.requires_grad


def test_duplicate_negatives_do_not_change_balanced_gate_loss_or_gradient():
    grads = []
    losses = []
    for labels in ([0, 1], [0, 0, 0, 0, 1]):
        model, obs, mask, ids, teacher = fixture(labels)
        loss, _ = fitting.preservation_loss(
            model, obs, mask, ids, teacher, gate_weight=1.0
        )
        loss.backward()
        losses.append(loss.detach())
        grads.append(model.actor_module.layers[0].added_admission.weight.grad)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(grads[0], grads[1])


def test_each_layer_receives_equal_gate_supervision():
    model, obs, mask, labels, teacher = fixture([0, 1], layers=2)
    loss, metrics = fitting.preservation_loss(
        model, obs, mask, labels, teacher, gate_weight=1.0
    )
    assert metrics["gate_layers"] == 2
    loss.backward()
    grads = [
        layer.added_admission.weight.grad
        for layer in model.actor_module.layers
    ]
    assert all(grad.abs().sum() > 0 for grad in grads)
    torch.testing.assert_close(grads[0], grads[1])


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_gate_loss_requires_both_classes(labels):
    args = fixture(labels)
    with pytest.raises(ValueError, match="both old/new"):
        fitting.preservation_loss(*args, gate_weight=1.0)


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0.0, 1.0]), torch.tensor([0, 2]), torch.tensor([[0], [1]])],
)
def test_invalid_labels_rejected(labels):
    model, obs, mask, _, teacher = fixture([0, 1])
    with pytest.raises(ValueError, match="int64"):
        fitting.preservation_loss(
            model, obs, mask, labels, teacher, gate_weight=1.0
        )


def test_zero_gate_weight_preserves_legacy_loss_and_needs_no_gate_gradient():
    model, obs, mask, labels, teacher = fixture([0, 0])
    loss, metrics = fitting.preservation_loss(
        model, obs, mask, labels, teacher, gate_weight=0.0
    )
    assert loss.item() == 0
    assert "gate_balanced_bce" not in metrics


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_gate_supervision_rejects_constant_admission_modes(mode):
    model, obs, mask, labels, teacher = fixture([0, 1])
    model.actor_module.layers[0].added_admission.set_mode(mode)
    with pytest.raises(ValueError, match="mode=learned"):
        fitting.preservation_loss(
            model, obs, mask, labels, teacher, gate_weight=1.0
        )


def test_gate_supervision_rejects_missing_or_stale_replay_records():
    model, obs, mask, labels, _ = fixture([0, 1])
    model.actor_module.sequence_mu(obs, attn_mask=mask)
    layer = model.actor_module.layers[0]
    layer.last_admission_logits = layer.last_admission_logits.detach()
    with pytest.raises(ValueError, match="retain.*gradients"):
        fitting.admission_supervision_loss(model, labels, obs.shape[:2])
    layer.last_admission_logits = torch.zeros(2, 2, requires_grad=True)
    with pytest.raises(ValueError, match=r"\[B,T\]"):
        fitting.admission_supervision_loss(model, labels, obs.shape[:2])
    layer.added_admission = None
    with pytest.raises(ValueError, match="explicit admission"):
        fitting.admission_supervision_loss(model, labels, obs.shape[:2])


@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
def test_excluded_new_logits_remain_zero_probability_and_finite(score):
    layer = SimpleNamespace(routing_score_fn=score)
    logits = torch.tensor(
        [[0.4, -0.2, -torch.inf, -torch.inf]], requires_grad=True
    )
    prob = fitting.route_distribution(layer, logits, 4)
    assert torch.isfinite(prob).all()
    assert torch.equal(prob[:, 2:], torch.zeros_like(prob[:, 2:]))
    (-prob[0, 0].log()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 2:].eq(0).all()
    short = fitting.route_distribution(layer, logits.detach()[:, :2], 4)
    torch.testing.assert_close(prob, short)


def test_all_experts_excluded_or_nonfinite_logits_fail_explicitly():
    layer = SimpleNamespace(routing_score_fn="softmax")
    for values in ([-torch.inf] * 4, [0.0, 1.0, float("nan"), 0.0]):
        with pytest.raises(ValueError, match="finite eligible"):
            fitting.route_distribution(layer, torch.tensor([values]), 4)


def test_load_source_rejects_implicit_gate_migration(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {"modules": {"actor": {"type": "InternalExpandedMoEPolicy"}}}
    )
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    module = {"internal_expansion": {}}
    monkeypatch.setattr(fitting, "actor_config", lambda _: module)
    monkeypatch.setattr(fitting, "actor_state", lambda _: {})
    with pytest.raises(ValueError, match="not migrated implicitly"):
        fitting.load_source(tmp_path / "old.pt", require_admission=True)
    module["internal_expansion"]["admission"] = {"mode": "learned"}
    with pytest.raises(ValueError, match="no saved gate state"):
        fitting.load_source(tmp_path / "old.pt")


def test_optimizer_identity_filter_rejects_original_or_expert_gradients():
    student = GateStudent()
    layer = student.actor_module.layers[0]
    layer.added_router = nn.Linear(2, 2)
    layer.old_weight = nn.Parameter(torch.zeros(2), requires_grad=False)
    router = fitting.calibration_parameters(student, train_gate=False)
    assert {id(p) for p in router.values()} == {
        id(p) for p in layer.added_router.parameters()
    }
    both = fitting.calibration_parameters(student, train_gate=True)
    assert {id(p) for p in both.values()} == {
        id(p)
        for module in (layer.added_router, layer.added_admission)
        for p in module.parameters()
    }
    layer.old_weight.requires_grad_(True)
    with pytest.raises(ValueError, match="original/expert"):
        fitting.calibration_parameters(student, train_gate=True)


def test_real_internal_actor_gate_loss_keeps_old_state_frozen():
    legacy, source = make_pair(admission={"mode": "open"})
    source.actor_module.set_expansion(
        "cooperative", stage="eval", admission_mode="open"
    )
    old = copy.deepcopy(source)
    old.actor_module.set_expansion("legacy", stage="eval")
    student = copy.deepcopy(source)
    student.actor_module.set_expansion(
        "cooperative", stage="B", admission_mode="learned"
    )
    obs, mask, labels = fitting.pack_windows(
        [(torch.randn(4, 8), 0), (torch.randn(2, 8), 1)], 8, 4
    )
    teacher = fitting.targets(old, source, obs, mask)
    loss, metrics = fitting.preservation_loss(
        student, obs, mask, labels, teacher, gate_weight=1.0
    )
    assert torch.isfinite(loss)
    assert metrics["old_admission_rate"] == 0
    assert metrics["new_admission_rate"] == 0
    loss.backward()
    gate = student.actor_module.layers[0].added_admission
    assert gate.weight.grad.abs().sum() > 0
    assert gate.bias.grad.abs().sum() > 0
    for key, value in legacy.state_dict().items():
        assert torch.equal(student.state_dict()[key], value), key
    old_parameter_names = set(dict(legacy.named_parameters()))
    assert all(
        parameter.grad is None
        for name, parameter in student.named_parameters()
        if name in old_parameter_names
    )
