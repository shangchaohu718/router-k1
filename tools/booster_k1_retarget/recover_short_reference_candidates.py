"""Recover already checked 50-99-frame runs without relaxing geometry gates.

Supplemental immutable views only. Original review pools remain preserved;
these views reclassify existing slices, not additional nonoverlapping data.
"""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import time


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def eligible(clip, report):
    if report.get("source_error") is not None:
        return False
    s, e = (
        int(clip["metadata"][k])
        for k in ("parent_frame_start", "parent_frame_end")
    )
    if e - s != int(clip["length"]) or not 50 <= e - s < 100:
        return False
    if s < 0 or e > report["frames"]:
        return False
    for spans in report["reason_intervals"].values():
        for a, b in spans:
            if s < min(report["frames"], b + 5) and e > max(0, a - 5):
                return False
    return True


def recover(run_root, output):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    roots, stats = (
        defaultdict(list),
        defaultdict(lambda: {"segments": 0, "frames": 0}),
    )
    ordinary = {"01_basic", "02_locomotion", "03_dynamic", "04_coordination"}
    for status_file in sorted(run_root.glob("batch_*/status.json")):
        status = json.loads(status_file.read_text())
        if status["status"] != "complete":
            continue
        report_path = Path(status["report"])
        reports = {
            r["motion_key"]: r
            for r in json.loads(report_path.read_text())["clips"]
        }
        root = report_path.parent / "80_short_or_source_review"
        manifest = json.loads((root / "manifest.json").read_text())
        groups = defaultdict(dict)
        for key, entry in manifest["clips"].items():
            meta = entry["metadata"]
            report = reports[meta["parent_motion_key"]]
            if not eligible(entry, report):
                continue
            stage = meta.get(
                "geometry_review_original_destination",
                Path(meta["geometry_review_parent_root"]).parts[-3],
            )
            if meta.get("active_split", "train") != "train":
                raise ValueError("Frozen split cannot be recovered here")
            category = (
                "01_short_candidates"
                if stage in ordinary
                else "80_origin_review"
            )
            clip = copy.deepcopy(entry)
            clip["metadata"].update(
                quality_status="source_checked_short_candidate_visual_physics_pending",
                training_approved=False,
                supplemental_short_recovery=True,
                short_recovery_source_view=str(root),
            )
            groups[(category, stage)][key] = clip
        for (category, stage), clips in groups.items():
            target = output / category / stage / status_file.parent.name
            view = {
                **manifest,
                "root": str(target),
                "dataset_id": f"{run_root.name}_short_{stage}_{status_file.parent.name}",
                "clips": clips,
                "motion_keys": list(clips),
                "training_approved": False,
                "hdf5_shards": [
                    {**s, "file": str((root / s["file"]).resolve())}
                    for s in manifest["hdf5_shards"]
                ],
            }
            path = target / "manifest.json"
            if path.exists():
                if json.loads(path.read_text()) != view:
                    raise ValueError(
                        "Previously recovered immutable view changed"
                    )
            else:
                dump(path, view)
                dump(target / "nan_npz_paths.json", [])
            roots[f"{category}_{stage}_train"].append(str(target))
            stats[f"{category}/{stage}"]["segments"] += len(clips)
            stats[f"{category}/{stage}"]["frames"] += sum(
                c["length"] for c in clips.values()
            )
    for name, paths in roots.items():
        dump(output / "candidate_dataset_roots" / (name + ".json"), paths)
    summary = {
        "input_run": str(run_root),
        "training_started": False,
        "training_approved": False,
        "groups": dict(stats),
        "segments": sum(s["segments"] for s in stats.values()),
        "frames": sum(s["frames"] for s in stats.values()),
        "minutes": sum(s["frames"] for s in stats.values()) / 3000,
        "policy": "50-99 frames at 50 Hz; same source/geometry gates and 5-frame bad-boundary padding; no joining, stretching, clipping or ground pinning. Supplemental reclassification of existing short-review slices, not new source frames. Scene/semantic origins stay separately held.",
        "updated_unix_seconds": time.time(),
    }
    dump(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-roots", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    last_change = time.monotonic()
    previous = None
    while True:
        summaries = [recover(r, a.output / r.name) for r in a.run_roots]
        states = [
            json.loads((r / "progress.json").read_text()) for r in a.run_roots
        ]
        totals = [(s.get("completed_clips"), s.get("status")) for s in states]
        if totals != previous:
            last_change = time.monotonic()
            previous = totals
        done = all(
            s.get("status") != "data_repair_in_progress" for s in states
        )
        summary = {
            "status": "following_data_repair"
            if not done
            else "parent_data_jobs_finished_review_pending",
            "training_started": False,
            "training_approved": False,
            "segments": sum(s["segments"] for s in summaries),
            "frames": sum(s["frames"] for s in summaries),
            "parent_progress": totals,
            "summaries": [
                str((a.output / r.name / "summary.json").resolve())
                for r in a.run_roots
            ],
        }
        dump(a.output / "progress.json", summary)
        if not a.follow or done:
            return
        if time.monotonic() - last_change > 7200:
            summary["status"] = "parent_progress_stale_review_required"
            dump(a.output / "progress.json", summary)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
