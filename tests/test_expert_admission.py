"""Observable admission is independent per sample and closed at start."""

import copy
import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from holomotion.src.modules.expert_admission import ExpertAdmissionGate


def test_initialization_is_closed_and_preserves_rng():
    before = torch.random.get_rng_state().clone()
    gate = ExpertAdmissionGate(4, mode="learned")
    assert torch.equal(before, torch.random.get_rng_state())
    assert not gate(torch.randn(2, 7, 4)).any()
    assert torch.equal(gate.weight, torch.zeros_like(gate.weight))
    assert gate.bias.item() == -8.0
    with pytest.raises(FrozenInstanceError):
        gate._config.threshold = 0.5
    copied = gate.config_dict()
    copied["threshold"] = 0.5
    assert gate.threshold == 0.99


def test_closed_open_and_learned_use_hard_scalar_decisions():
    gate = ExpertAdmissionGate(4)
    features = torch.randn(2, 3, 4, requires_grad=True)
    assert gate(features).shape == (2, 3)
    assert not gate(features).any()
    gate.set_mode("open")
    assert gate(features).all()
    gate.set_mode("learned")
    threshold = math.log(gate.threshold / (1.0 - gate.threshold))
    with torch.no_grad():
        gate.bias.fill_(threshold - 1e-4)
    assert not gate(features).any()
    with torch.no_grad():
        gate.bias.fill_(threshold)
    assert gate(features).all()
    assert gate(features).dtype == torch.bool
    assert not gate(features).requires_grad
    assert features.grad is None
    assert all(parameter.grad is None for parameter in gate.parameters())


def test_samples_are_independent_and_no_task_label_is_an_input():
    gate = ExpertAdmissionGate(4, mode="learned", threshold=0.5)
    with torch.no_grad():
        gate.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        gate.bias.fill_(-0.5)
    inputs = torch.tensor([[1.0, 9.0, 3.0, 2.0], [-1.0, 0.0, 0.0, 0.0]])
    assert gate(inputs).tolist() == [True, False]
    for index in range(2):
        assert gate(inputs[index]).item() == gate(inputs)[index].item()
    changed = inputs.clone()
    changed[1] *= 1e6
    assert gate(changed)[0] == gate(inputs)[0]
    with pytest.raises(TypeError):
        gate(inputs, task_id=torch.ones(2, dtype=torch.long))


def test_supervised_logits_have_gradients_without_hard_mask_surrogate():
    gate = ExpertAdmissionGate(2, mode="learned")
    inputs = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1.0, 0.0])
    logits = gate.logits(inputs)
    assert logits.shape == labels.shape
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert gate.weight.grad[0, 0] < 0
    assert gate.weight.grad[0, 1] > 0
    assert gate.bias.grad.abs().sum() > 0
    assert not gate(inputs).requires_grad


def test_explicit_modes_ignore_learned_weights():
    gate = ExpertAdmissionGate(2)
    with torch.no_grad():
        gate.weight.fill_(float("nan"))
        gate.bias.fill_(float("nan"))
    features = torch.ones(3, 2)
    assert not gate(features).any()
    gate.set_mode("open")
    assert gate(features).all()


@pytest.mark.parametrize("input_dim", [True, 0, -1, 3.5, "4"])
def test_invalid_width_rejected(input_dim):
    with pytest.raises(ValueError, match="input_dim"):
        ExpertAdmissionGate(input_dim)


@pytest.mark.parametrize("threshold", [True, 0.0, 1.0, -1.0, float("nan")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        ExpertAdmissionGate(4, threshold=threshold)


@pytest.mark.parametrize("initial_bias", [True, 0.0, 1.0, float("inf")])
def test_invalid_initial_bias_rejected(initial_bias):
    with pytest.raises(ValueError, match="initial_bias"):
        ExpertAdmissionGate(4, initial_bias=initial_bias)


def test_initialization_must_be_closed_for_actual_threshold():
    with pytest.raises(ValueError, match="start closed"):
        ExpertAdmissionGate(4, threshold=1e-5, initial_bias=-8.0)


def test_invalid_mode_and_input_rejected():
    gate = ExpertAdmissionGate(4)
    for mode in ("soft", True, None, []):
        with pytest.raises(ValueError, match="mode"):
            gate.set_mode(mode)
    for features in (torch.zeros(2, 5), torch.tensor(1.0)):
        with pytest.raises(ValueError, match="width"):
            gate(features)
    for features in (torch.zeros(2, 4, dtype=torch.long), {"task_id": 1}):
        with pytest.raises(TypeError):
            gate(features)


def test_checkpoint_roundtrip_preserves_learned_scores_and_declared_mode():
    gate = ExpertAdmissionGate(4, mode="learned")
    with torch.no_grad():
        gate.weight.copy_(torch.tensor([[1.0, 2.0, -3.0, 4.0]]))
        gate.bias.fill_(0.2)
    restored = ExpertAdmissionGate(4, mode="closed")
    restored.load_state_dict(copy.deepcopy(gate.state_dict()), strict=True)
    assert restored.mode == "closed"
    features = torch.randn(3, 4)
    torch.testing.assert_close(
        restored.logits(features), gate.logits(features)
    )
    assert not restored(features).any()
    restored.set_mode("learned")
    assert torch.equal(restored(features), gate(features))


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("key", ["metadata_version", "config_digest"])
def test_missing_checkpoint_metadata_is_rejected_even_when_non_strict(
    strict, key
):
    gate = ExpertAdmissionGate(4)
    state = copy.deepcopy(gate.state_dict())
    del state[key]
    with pytest.raises(
        RuntimeError, match="Missing expert admission metadata"
    ):
        gate.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("key", ["metadata_version", "config_digest"])
def test_bad_checkpoint_metadata_rejected_without_parameter_mutation(key):
    gate = ExpertAdmissionGate(4)
    original = copy.deepcopy(gate.state_dict())
    state = copy.deepcopy(original)
    state["weight"].fill_(123.0)
    state[key].add_(1)
    with pytest.raises(RuntimeError, match="Incompatible.*metadata"):
        gate.load_state_dict(state, strict=False)
    for name, value in original.items():
        assert torch.equal(gate.state_dict()[name], value)


@pytest.mark.parametrize(
    "changes", [{"threshold": 0.98}, {"initial_bias": -9.0}, {"input_dim": 3}]
)
def test_checkpoint_does_not_silently_change_non_mode_contract(changes):
    state = ExpertAdmissionGate(4).state_dict()
    config = {"input_dim": 4, **changes}
    with pytest.raises(RuntimeError, match="Incompatible.*config_digest"):
        ExpertAdmissionGate(**config).load_state_dict(state, strict=False)


def test_nested_checkpoint_load_also_validates_component_metadata():
    owner = nn.ModuleDict({"admission": ExpertAdmissionGate(4)})
    state = copy.deepcopy(owner.state_dict())
    state["admission.metadata_version"].add_(1)
    with pytest.raises(RuntimeError, match="admission.metadata_version"):
        owner.load_state_dict(state, strict=True)


def test_checkpoint_rejects_metadata_type_shape_and_missing_weight():
    gate = ExpertAdmissionGate(4)
    for replacement in (torch.tensor([1]), torch.tensor(1.0), "1"):
        state = copy.deepcopy(gate.state_dict())
        state["metadata_version"] = replacement
        with pytest.raises(RuntimeError, match="admission metadata"):
            gate.load_state_dict(state, strict=False)
    state = copy.deepcopy(gate.state_dict())
    del state["weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        gate.load_state_dict(state, strict=True)
