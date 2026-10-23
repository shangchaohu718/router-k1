"""Index completed immutable repair batches by original curriculum stage."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def build_index(run_root):
    run_root = Path(run_root).resolve()
    output = run_root / "organized_candidates"
    roots = defaultdict(list)
    totals = defaultdict(
        lambda: {"segments": 0, "frames": 0, "original_motion_keys": set()}
    )
    failures = []
    stages = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}
    for path in sorted(run_root.glob("batch_*/status.json")):
        status = json.loads(path.read_text())
        if status["status"] == "source_or_solver_failure":
            failures.append(status)
        if status["status"] != "complete":
            continue
        review = Path(status["report"]).parent
        for category in (
            "01_kinematic_candidates",
            "80_short_or_source_review",
            "90_rejected_intervals",
        ):
            root = review / category
            manifest = json.loads((root / "manifest.json").read_text())
            groups = defaultdict(dict)
            for key, clip in manifest["clips"].items():
                original_root = Path(
                    clip["metadata"]["geometry_review_parent_root"]
                )
                stage = clip["metadata"].get(
                    "geometry_review_original_destination",
                    original_root.parts[-3],
                )
                split = clip["metadata"].get(
                    "active_split", original_root.parts[-2]
                )
                if split != "train":
                    raise ValueError(
                        "Recovered candidate has unknown/frozen curriculum provenance"
                    )
                groups[stage][key] = clip
            for stage, clips in groups.items():
                destination = (
                    "80_origin_review"
                    if stage not in stages
                    and category == "01_kinematic_candidates"
                    else category
                )
                target = output / destination / stage / path.parent.name
                view = {
                    **manifest,
                    "dataset_id": f"{run_root.name}_{category}_{stage}_{path.parent.name}",
                    "root": str(target),
                    "clips": clips,
                    "motion_keys": list(clips),
                    "hdf5_shards": [
                        {
                            **shard,
                            "file": str((root / shard["file"]).resolve()),
                        }
                        for shard in manifest["hdf5_shards"]
                    ],
                    "original_curriculum_stage": stage,
                    "training_approved": False,
                    "quality_status": "source_aware_kinematic_only_visual_physics_pending",
                }
                if (target / "manifest.json").exists():
                    if (
                        json.loads((target / "manifest.json").read_text())
                        != view
                    ):
                        raise ValueError(
                            "An already indexed immutable batch changed"
                        )
                else:
                    dump(target / "manifest.json", view)
                    dump(target / "nan_npz_paths.json", [])
                roots[f"{destination}_{stage}_train"].append(str(target))
                stat = totals[f"{destination}/{stage}"]
                stat["segments"] += len(clips)
                stat["frames"] += sum(int(c["length"]) for c in clips.values())
                stat["original_motion_keys"].update(
                    c["metadata"]["parent_motion_key"] for c in clips.values()
                )
    for name, paths in roots.items():
        dump(output / "candidate_dataset_roots" / (name + ".json"), paths)
    summary = {
        "run_root": str(run_root),
        "training_approved": False,
        "stage_order": [
            "01_basic",
            "02_locomotion",
            "03_dynamic",
            "04_coordination",
        ],
        "limitations": "Only completed batches indexed. Kinematic candidates require visual review and are not dynamics-approved. Short/source-review or rejected intervals never enter candidate roots. No training launched.",
        "groups": {
            k: {
                "segments": v["segments"],
                "frames": v["frames"],
                "hours": v["frames"] / 50 / 3600,
                "original_clips": len(v["original_motion_keys"]),
            }
            for k, v in totals.items()
        },
        "individual_source_or_solver_failures": len(failures),
    }
    dump(output / "summary.json", summary)
    dump(output / "99_source_or_solver_failures.json", failures)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.run_root), indent=2))
