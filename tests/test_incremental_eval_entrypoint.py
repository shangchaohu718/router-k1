"""The evaluation entry point must not rewrite source training artifacts."""

import copy

import pytest
from omegaconf import OmegaConf

from holomotion.src.evaluation import eval_motion_tracking_single as evaluator


@pytest.mark.parametrize("source_exists", [False, True])
def test_eval_entrypoint_never_writes_source_training_directory(
    tmp_path, monkeypatch, source_exists
):
    source = tmp_path / "original_training_run"
    original = b"preserve: original training configuration\n"
    if source_exists:
        source.mkdir()
        (source / "config.yaml").write_bytes(original)
    staged = tmp_path / "evaluation_staging"
    staged.mkdir()
    checkpoint = staged / "model_zero.pt"
    checkpoint.write_bytes(b"not opened by this configuration-only test")
    resolved_training = OmegaConf.create(
        {
            "checkpoint": str(checkpoint),
            "experiment_dir": str(source),
            "experiment_save_dir": str(source),
            "headless": True,
            "device": "cpu",
            "num_processes": 1,
            "process_id": 0,
            "main_process": True,
            "env": {"config": {"simulation": {}, "num_envs": 1}},
            "algo": {
                "_target_": "must.not.construct.Simulator",
                "config": {"checkpoint": str(checkpoint)},
            },
        }
    )
    supplied = OmegaConf.create({"checkpoint": str(checkpoint)})
    monkeypatch.setattr(
        evaluator,
        "load_training_config",
        lambda path, config: copy.deepcopy(resolved_training),
    )
    # Keep the real compiler, including its directory handling. A regression
    # to the default eval=False would create/overwrite the source config.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    compiled = []
    real_compile = evaluator.compile_config

    def compile_and_capture(*args, **kwargs):
        result = real_compile(*args, **kwargs)
        compiled.append(result)
        return result

    monkeypatch.setattr(evaluator, "compile_config", compile_and_capture)

    class StopBeforeAlgorithmConstructionError(Exception):
        pass

    def stop_before_algorithm(target):
        assert target == "must.not.construct.Simulator"
        raise StopBeforeAlgorithmConstructionError

    monkeypatch.setattr(evaluator, "get_class", stop_before_algorithm)
    with pytest.raises(StopBeforeAlgorithmConstructionError):
        evaluator.main.__wrapped__(supplied)

    assert len(compiled) == 1
    assert compiled[0].env.config.simulation.device == "cpu"
    assert compiled[0].experiment_dir == str(source)
    if source_exists:
        assert (source / "config.yaml").read_bytes() == original
        assert sorted(path.name for path in source.iterdir()) == [
            "config.yaml"
        ]
    else:
        assert not source.exists()
    assert sorted(path.name for path in staged.iterdir()) == ["model_zero.pt"]
