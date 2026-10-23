"""Pin local robot/FK XML, URDF and mesh bytes without simulator imports."""

import hashlib
from pathlib import Path

import pytest

from holomotion.src.motion_tracking import asset_contract
from holomotion.src.motion_tracking.asset_contract import (
    asset_content_manifest,
)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_contract, "_REPO_ROOT", tmp_path)
    folder = tmp_path / "robot"
    folder.mkdir()
    (folder / "meshes").mkdir()
    (folder / "meshes" / "body.STL").write_bytes(b"mesh bytes")
    (folder / "robot.xml").write_text(
        '<mujoco><compiler meshdir="meshes/"/><asset>'
        '<mesh name="body" file="body.STL"/></asset></mujoco>'
    )
    (folder / "robot.urdf").write_text(
        '<robot name="K1"><link name="body"><visual><geometry>'
        '<mesh filename="meshes/body.STL"/>'
        "</geometry></visual></link></robot>"
    )
    robot = {
        "asset_root": "./",
        "urdf_file": "robot/robot.urdf",
        "assetFileName": "robot/robot.xml",
    }
    motion = {
        "asset": {"assetRoot": "./", "assetFileName": "robot/robot.xml"},
        "fk_robot_file_path": "robot/robot.urdf",
    }
    return folder, robot, motion


def test_local_sources_meshdir_urdf_dedup_and_cwd_independence(
    assets, tmp_path, monkeypatch
):
    folder, robot, motion = assets
    monkeypatch.chdir(tmp_path / "robot" / "meshes")
    result = asset_content_manifest(robot, motion)
    assert len(result) == 3
    assert result == sorted(result, key=lambda item: item["path"])
    for entry in result:
        path = Path(entry["path"])
        assert path.is_absolute() and path.is_relative_to(folder)
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "filename", ["robot.xml", "robot.urdf", "meshes/body.STL"]
)
def test_each_asset_content_change_changes_manifest(assets, filename):
    folder, robot, motion = assets
    before = asset_content_manifest(robot, motion)
    path = folder / filename
    path.write_bytes(path.read_bytes() + b"\n")
    assert asset_content_manifest(robot, motion) != before


@pytest.mark.parametrize(
    "filename", ["robot.xml", "robot.urdf", "meshes/body.STL"]
)
def test_missing_source_or_mesh_is_rejected(assets, filename):
    folder, robot, motion = assets
    (folder / filename).unlink()
    with pytest.raises(ValueError, match="Missing registered asset"):
        asset_content_manifest(robot, motion)


@pytest.mark.parametrize(
    "key,value",
    [
        ("urdf_variants", ["some.urdf"]),
        ("urdf_variant_glob", "*.urdf"),
        ("urdf_variant_manifest", "manifest.json"),
    ],
)
def test_variant_sources_are_rejected(assets, key, value):
    _, robot, motion = assets
    robot[key] = value
    with pytest.raises(ValueError, match="variants/glob/manifest"):
        asset_content_manifest(robot, motion)


@pytest.mark.parametrize(
    "uri",
    [
        "https://host/model.urdf",
        "package://k1/model.urdf",
        "file:///tmp/a.urdf",
        "robot/*.urdf",
    ],
)
def test_external_or_glob_source_is_rejected(assets, uri):
    _, robot, motion = assets
    robot["urdf_file"] = uri
    with pytest.raises(ValueError, match="explicit local path"):
        asset_content_manifest(robot, motion)


@pytest.mark.parametrize(
    "uri",
    [
        "package://robot/body.STL",
        "https://host/body.STL",
        "file:///tmp/body.STL",
    ],
)
def test_external_mesh_reference_is_rejected(assets, uri):
    folder, robot, motion = assets
    (folder / "robot.urdf").write_text(
        f'<robot><link><visual><geometry><mesh filename="{uri}"/>'
        "</geometry></visual></link></robot>"
    )
    with pytest.raises(ValueError, match="explicit local path"):
        asset_content_manifest(robot, motion)


def test_unhashed_include_is_rejected(assets):
    folder, robot, motion = assets
    (folder / "robot.xml").write_text(
        '<mujoco><include file="other.xml"/></mujoco>'
    )
    with pytest.raises(ValueError, match="include or dependency"):
        asset_content_manifest(robot, motion)


def test_actual_k1_assets_have_complete_local_mesh_contents():
    robot = {
        "asset_root": "./",
        "urdf_file": "assets/robots/booster_k1/K1_22dof.urdf",
        "assetFileName": "assets/robots/booster_k1/K1_22dof.xml",
    }
    motion = {
        "asset": {"assetFileName": robot["assetFileName"]},
        "fk_robot_file_path": robot["urdf_file"],
    }
    result = asset_content_manifest(robot, motion)
    assert len(result) == 26  # Two definitions, 24 distinct local STL files.
    assert sum(item["bytes"] for item in result) > 10 * 1024 * 1024
