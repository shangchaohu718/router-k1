"""Repair missing source intervals, preserving the checked baseline unchanged.

Per-source pilot qualification precedes bulk work. New patches are independent
clips, never stitched to old trajectories. Original shards remain read-only.
"""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.booster_k1_retarget.consolidate_review_candidates import subtract
from tools.booster_k1_retarget.recover_short_reference_candidates import (
    dump,
    eligible,
)

PROFILES = ["old_root_weight", "lafan_arm_direction_rootpos200"]
SMPL = (
    "/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
)


def accepted(report):
    data = json.loads(report.read_text())
    row = data["clips"][0]
    items = []
    for category in ["01_kinematic_candidates", "80_short_or_source_review"]:
        root = report.parent / category
        m = json.loads((root / "manifest.json").read_text())
        for key, c in m["clips"].items():
            if category.startswith("80") and not eligible(c, row):
                continue
            items.append(
                (
                    key,
                    c,
                    {
                        **m["hdf5_shards"][c["shard"]],
                        "file": str(
                            (
                                root / m["hdf5_shards"][c["shard"]]["file"]
                            ).resolve()
                        ),
                    },
                )
            )
    return row, items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((a.baseline / "manifest.json").read_text())
    occupied = defaultdict(list)
    for c in baseline["clips"].values():
        meta = c["metadata"]
        s = meta["canonical_frame_start"]
        occupied[meta["source_path"]].append((s, s + c["length"]))
    cases = []
    input_frames = 0
    input_hashes = {}
    for file in sorted((a.snapshot / "01_candidates").glob("*/manifest.json")):
        m = json.loads(file.read_text())
        input_hashes[str(file.resolve())] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        input_frames += sum(c["length"] for c in m["clips"].values())
        for key, c in m["clips"].items():
            meta = c["metadata"]
            s = meta["canonical_frame_start"]
            spans = subtract(s, s + c["length"], occupied[meta["source_path"]])
            gaps = [(x, y) for x, y in spans if y - x >= 50]
            if gaps:
                cases.append(
                    {
                        "key": key,
                        "clip": c,
                        "manifest": m,
                        "root": file.parent,
                        "source": meta.get(
                            "geometry_review_source", meta["dataset"]
                        ),
                        "gaps": gaps,
                        "stage": file.parent.name,
                    }
                )
    # Freeze every code/config file used by this long-running job.
    files = [
        Path(__file__),
        *[
            Path("tools/booster_k1_retarget") / n
            for n in [
                "compare_lafan_retarget_pipelines.py",
                "probe_native_position_retarget.py",
                "retry_guarded_reference.py",
                "salvage_repaired_reference.py",
                "salvage_source_root_checked.py",
                "review_lafan_root_fidelity.py",
                "holosmpl_h5_to_k1_h5.py",
            ]
        ],
        *Path("tools/booster_k1_retarget/configs").glob(
            "*official_k1*v2.json"
        ),
    ]
    hashes = {
        str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }
    settings = {
        "snapshot": str(a.snapshot.resolve()),
        "baseline": str(a.baseline.resolve()),
        "baseline_manifest_sha256": hashlib.sha256(
            (a.baseline / "manifest.json").read_bytes()
        ).hexdigest(),
        "worker_sha256": hashes,
        "profiles": PROFILES,
        "training_started": False,
        "training_approved": False,
        "policy": "Protect baseline, pilot-qualify per source, remap original candidate spans and add only nonoverlapping checked patches. Never stitch.",
    }
    settings["input_manifest_sha256"] = input_hashes
    cfg = a.output / "run_config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != settings:
        raise ValueError("New version required for changed settings")
    dump(cfg, settings)

    def check():
        if any(
            hashlib.sha256(Path(f).read_bytes()).hexdigest() != h
            for f, h in hashes.items()
        ):
            raise RuntimeError("Worker changed during repair; use new version")

    pilot = {}
    counts = Counter()
    added = 0

    def progress(phase, current=None):
        dump(
            a.output / "progress.json",
            {
                "status": phase,
                "current": current,
                "training_started": False,
                "training_approved": False,
                "bulk_repair_is_non_overwriting": True,
                "input_candidate_frames": input_frames,
                "target_reference_segments": len(cases),
                "protected_baseline_hours": sum(
                    c["length"] for c in baseline["clips"].values()
                )
                / 180000,
                "counts": dict(counts),
                "added_repaired_candidate_frames": added,
                "added_repaired_candidate_hours": added / 180000,
                "source_pilots": pilot,
                "updated_unix_seconds": time.time(),
            },
        )

    progress("source_pilots_running")

    def prepare(case, root, pilot_window=False):
        c = copy.deepcopy(case["clip"])
        meta = c["metadata"]
        if pilot_window:
            gap = max(case["gaps"], key=lambda x: x[1] - x[0])
            offset = min(
                max(0, gap[0] - meta["canonical_frame_start"] - 25),
                max(0, c["length"] - 500),
            )
            n = min(500, c["length"] - offset)
            c["start"] += offset
            c["length"] = n
            meta["canonical_frame_start"] += offset
            meta["parent_frame_start"] += offset
            meta["parent_frame_end"] = meta["parent_frame_start"] + n
            meta["num_frames"] = n
            meta["canonical_frame_count"] = n
        m = case["manifest"]
        view = {
            **m,
            "array_names": ["ref_root_pos", "ref_root_rot", "ref_dof_pos"],
            "clips": {case["key"]: c},
            "motion_keys": [case["key"]],
            "hdf5_shards": [
                {**s, "file": str((case["root"] / s["file"]).resolve())}
                for s in m["hdf5_shards"]
            ],
        }
        dump(root / "input/manifest.json", view)

    def review(inp, out, log):
        if not (out / "report.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.booster_k1_retarget.salvage_source_root_checked",
                    "--input",
                    str(inp),
                    "--output",
                    str(out),
                    "--smpl-model",
                    SMPL,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return accepted(out / "report.json")

    def trial(root, profile, log):
        raw = root / profile / "raw"
        if not (raw / "manifest.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.booster_k1_retarget.compare_lafan_retarget_pipelines",
                    "--output",
                    str(root),
                    "--worker",
                    profile,
                    "--max-frames",
                    "10000000",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return review(raw, root / profile / "review", log)

    for source in sorted({c["source"] for c in cases}):
        chosen = []
        seen = set()
        for case in sorted(
            [c for c in cases if c["source"] == source],
            key=lambda c: -sum(e - s for s, e in c["gaps"]),
        ):
            path = case["clip"]["metadata"]["source_path"]
            if path in seen:
                continue
            chosen.append(case)
            seen.add(path)
            if len(chosen) == 2:
                break
        results = []
        for i, case in enumerate(chosen):
            if (a.output / "STOP_AFTER_CASE").exists():
                progress("stopped_between_cases")
                return
            check()
            root = a.output / "pilots" / source / str(i)
            prepare(case, root, True)
            with (root / "run.log").open("a") as log:
                before, before_items = review(
                    root / "input", root / "before_review", log
                )
                before_frames = sum(c["length"] for _, c, _ in before_items)
                for profile in PROFILES:
                    try:
                        after, items = trial(root, profile, log)
                        frames = sum(c["length"] for _, c, _ in items)
                        ok = (
                            frames >= before_frames + 50
                            and after["source_error"] is None
                            and after["native_leg_direction_p95_deg"]
                            <= before["native_leg_direction_p95_deg"] + 5
                            and after["source_upper_arm_direction_p95_deg"]
                            <= before["source_upper_arm_direction_p95_deg"] + 5
                        )
                        results.append(
                            {
                                "pilot": i,
                                "motion": case["key"],
                                "profile": profile,
                                "before_frames": before_frames,
                                "after_frames": frames,
                                "qualifies": bool(ok),
                                "review": str(
                                    (
                                        root / profile / "review/report.json"
                                    ).resolve()
                                ),
                            }
                        )
                    except subprocess.CalledProcessError as exc:
                        results.append(
                            {
                                "pilot": i,
                                "profile": profile,
                                "qualifies": False,
                                "error": str(exc),
                            }
                        )
            pilot[source] = {"results": results, "qualified_profiles": []}
            progress("source_pilots_running", source)
        qualified = [
            profile
            for profile in PROFILES
            if len(chosen) == 2
            and all(
                any(
                    r["pilot"] == i
                    and r["profile"] == profile
                    and r["qualifies"]
                    for r in results
                )
                for i in range(2)
            )
        ]
        pilot[source] = {"results": results, "qualified_profiles": qualified}
        dump(a.output / "source_pilots.json", pilot)
        progress("source_pilots_running", source)
    for case in cases:
        if (a.output / "STOP_AFTER_CASE").exists():
            progress("stopped_between_cases")
            return
        check()
        identity = hashlib.sha256(case["key"].encode()).hexdigest()[:20]
        root = a.output / "cases" / identity
        status = root / "status.json"
        if status.exists():
            old = json.loads(status.read_text())
            counts[old["status"]] += 1
            added += old.get("added_frames", 0)
            continue
        profiles = pilot[case["source"]]["qualified_profiles"]
        if not profiles:
            dump(
                status,
                {
                    "status": "source_profile_validation_pending",
                    "source": case["source"],
                    "motion": case["key"],
                    "added_frames": 0,
                },
            )
            counts["source_profile_validation_pending"] += 1
            progress("repairing", case["key"])
            continue
        prepare(case, root)
        best = []
        best_profile = None
        with (root / "run.log").open("a") as log:
            for profile in profiles:
                try:
                    _, items = trial(root, profile, log)
                    patches = []
                    for key, c, shard in items:
                        meta = c["metadata"]
                        s = meta["canonical_frame_start"]
                        e = s + c["length"]
                        for x, y in subtract(
                            s, e, occupied[meta["source_path"]]
                        ):
                            if y - x < 50:
                                continue
                            patch = copy.deepcopy(c)
                            patch["start"] += x - s
                            patch["length"] = y - x
                            pm = patch["metadata"]
                            pm["canonical_frame_start"] = x
                            pm["canonical_frame_count"] = y - x
                            pm["num_frames"] = y - x
                            pm["parent_frame_start"] += x - s
                            pm["parent_frame_end"] = (
                                pm["parent_frame_start"] + y - x
                            )
                            pm.update(
                                repair_profile=profile,
                                independent_patch_not_stitched=True,
                                training_approved=False,
                            )
                            patches.append(
                                (key + f"_patch_{x}_{y}", patch, shard)
                            )
                    if sum(c["length"] for _, c, _ in patches) > sum(
                        c["length"] for _, c, _ in best
                    ):
                        best = patches
                        best_profile = profile
                except subprocess.CalledProcessError:
                    continue
        clips = {}
        shards = []
        for key, c, shard in best:
            c["shard"] = len(shards)
            clips[key] = c
            shards.append(shard)
        frames = sum(c["length"] for c in clips.values())
        added += frames
        state = (
            "repaired_patches_retained"
            if frames
            else "no_checked_patch_recovered"
        )
        if frames:
            dump(
                root / "patches/manifest.json",
                {
                    "version": 2,
                    "dataset_id": "large_repair_v2_" + identity,
                    "dof_names": baseline["dof_names"],
                    "clips": clips,
                    "motion_keys": list(clips),
                    "hdf5_shards": shards,
                    "training_approved": False,
                },
            )
            dump(root / "patches/nan_npz_paths.json", [])
        dump(
            status,
            {
                "status": state,
                "source": case["source"],
                "motion": case["key"],
                "added_frames": frames,
                "selected_profile": best_profile,
                "original_preserved": True,
                "training_approved": False,
            },
        )
        counts[state] += 1
        progress("repairing", case["key"])
    progress("current_pass_complete_visual_physics_pending")


if __name__ == "__main__":
    main()
