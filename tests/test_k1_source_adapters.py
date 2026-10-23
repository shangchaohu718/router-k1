import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from holosmpl.converters.bvh_family.cmu_asf import load_cmu_fk, metadata_for
from holosmpl.converters.bvh_family.style100 import (
    canonical_from_global,
    trim_range,
)


def test_canonical_identity_tpose_and_yup_units(monkeypatch):
    monkeypatch.setenv("HOLOSMPL_RESAMPLE_BACKEND", "scipy")
    rotations = [Rotation.identity(2) for _ in range(22)]
    result = canonical_from_global(
        rotations, np.array([[0, 1, 0], [1, 1, 2]]), 50, 50, {}
    )
    np.testing.assert_allclose(
        result["trans"], [[0, 0, 1], [1, -2, 1]], atol=1e-6
    )
    np.testing.assert_allclose(result["pose_body"], 0, atol=1e-6)
    assert result["pose_body"].shape == (2, 69)


def test_extra_spine_rotation_is_preserved_in_global_to_local(monkeypatch):
    monkeypatch.setenv("HOLOSMPL_RESAMPLE_BACKEND", "scipy")
    rotations = [Rotation.identity(2) for _ in range(22)]
    rotations[9] = Rotation.from_euler("x", [0.3, 0.4])
    result = canonical_from_global(rotations, np.zeros((2, 3)), 50, 50, {})
    np.testing.assert_allclose(
        result["pose_body"].reshape(2, 23, 3)[:, 8, 0], [0.3, 0.4], atol=1e-6
    )


def test_official_style_trim(tmp_path):
    (tmp_path / "Frame_Cuts.csv").write_text(
        "STYLE_NAME,FW_START,FW_STOP\nNeutral,12,99\n"
    )
    assert trim_range(tmp_path / "Neutral_FW.bvh") == (12, 99, "Neutral", "FW")


def test_cmu_fk_uses_meter_scale_and_parent_rotation(tmp_path):
    subject = tmp_path / "07"
    subject.mkdir()
    (subject / "07.asf").write_text("""
:units
mass 1.0
length 0.45
angle deg
:root
order TX TY TZ RX RY RZ
axis XYZ
position 0 0 0
orientation 0 0 0
:bonedata
begin
id 1
name bone
direction 1 0 0
length 1
axis 0 0 0 XYZ
dof rx
end
:hierarchy
begin
root bone
end
""")
    path = subject / "07_01.amc"
    path.write_text(
        ":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 90\nbone 0\n"
    )
    rotations, starts, ends, bones = load_cmu_fk(path)
    np.testing.assert_allclose(
        ends["bone"], [[0, 0.0254 / 0.45, 0]], atol=1e-8
    )


def test_cmu_missing_fps_is_not_guessed(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="FPS"):
        metadata_for(tmp_path / "07_01.amc")


def test_cmu_missing_fps_is_reported_without_crashing_batch(tmp_path):
    from holosmpl.workflows.convert_dataset import _process_source

    source = tmp_path / "07_01.amc"
    source.write_text(":DEGREES\n")
    result = _process_source(
        (
            0,
            str(source),
            str(tmp_path),
            str(tmp_path / "out"),
            str(tmp_path / "out/clips"),
            50.0,
            "cmu_asf",
        )
    )
    assert result["status"] == "rejected"
    assert result["sample"]["reject_status"] == "unsupported_missing_fps"
    assert "FPS" in result["sample"]["reason"]
    assert result["sample"]["source_fields"] == []


def test_difficulty_does_not_mark_stairs_as_bad_data():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/booster_k1_retarget/organize_motion_curriculum.py"
    )
    spec = importlib.util.spec_from_file_location("curriculum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.difficulty("walk upstairs")[0] == "L5"
    assert module.difficulty("moonwalk")[0] == "L4"
    assert module.difficulty("slow walk")[0] == "L1"
    assert module.difficulty("neutral Male walk, exact footfalls")[0] == "L1"
    assert module.difficulty("falling to ground")[0] == "L5"
    assert module.difficulty("transition")[0] == "L2"
    assert module.difficulty("unknown capture")[0] == "pending"
    assert module.repair_partition_name(
        Path("/data/batch_arm/cmu_asf/113/robot_filtered"), "CMU", "train"
    ) != module.repair_partition_name(
        Path("/data/batch_root/cmu_asf/113/robot_filtered"), "CMU", "train"
    )
