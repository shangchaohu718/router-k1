import json
from pathlib import Path

from tools.booster_k1_retarget.index_reference_repairs import build_index


def test_candidates_keep_curriculum_and_rejected_intervals_separate(tmp_path):
    batch = tmp_path / "batch_smpl_0000"
    review = batch / "attempt_000/review"
    review.mkdir(parents=True)
    (review / "report.json").write_text("{}")
    (batch / "status.json").write_text(
        json.dumps(
            {"status": "complete", "report": str(review / "report.json")}
        )
    )
    for category in (
        "01_kinematic_candidates",
        "80_short_or_source_review",
        "90_rejected_intervals",
    ):
        root = review / category
        root.mkdir()
        clips = {
            k: {
                "start": 0,
                "length": 100,
                "metadata": {
                    "geometry_review_parent_root": f"/frozen/{stage}/train/partition_01",
                    "parent_motion_key": "original_" + k,
                },
            }
            for k, stage in (
                ("walk", "02_locomotion"),
                ("dance", "04_coordination"),
            )
        }
        (root / "manifest.json").write_text(
            json.dumps({"clips": clips, "hdf5_shards": [{"file": "shard.h5"}]})
        )
    first = build_index(tmp_path)
    assert first == build_index(tmp_path)
    assert len(first["groups"]) == 6
    assert not first["training_approved"]
    for index in (
        tmp_path / "organized_candidates/candidate_dataset_roots"
    ).glob("01_kinematic_candidates*.json"):
        for path in json.loads(index.read_text()):
            assert "/01_kinematic_candidates/" in path
            manifest = json.loads((Path(path) / "manifest.json").read_text())
            assert not manifest["training_approved"]


def test_recovered_scene_and_quarantine_stay_origin_review(tmp_path):
    batch = tmp_path / "batch_native_0000"
    review = batch / "attempt_000/review"
    review.mkdir(parents=True)
    (review / "report.json").write_text("{}")
    (batch / "status.json").write_text(
        json.dumps(
            {"status": "complete", "report": str(review / "report.json")}
        )
    )
    for category in (
        "01_kinematic_candidates",
        "80_short_or_source_review",
        "90_rejected_intervals",
    ):
        root = review / category
        root.mkdir()
        clips = (
            {
                "stairs": {
                    "length": 100,
                    "metadata": {
                        "geometry_review_parent_root": "/raw/source/robot_raw",
                        "geometry_review_original_destination": "05_scene_and_semantic_review",
                        "active_split": "train",
                        "parent_motion_key": "stairs_original",
                    },
                }
            }
            if category == "01_kinematic_candidates"
            else {}
        )
        (root / "manifest.json").write_text(
            json.dumps({"clips": clips, "hdf5_shards": [{"file": "shard.h5"}]})
        )
    summary = build_index(tmp_path)
    assert "80_origin_review/05_scene_and_semantic_review" in summary["groups"]
    assert not list(
        (tmp_path / "organized_candidates/candidate_dataset_roots").glob(
            "01_kinematic_candidates*.json"
        )
    )
