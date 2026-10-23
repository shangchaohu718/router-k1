"""Retry unrecovered references in separate variants, never mixing trajectories.

Original batches/checkpoints/thresholds remain unchanged. Reuses guarded seed
repairs for the two known failures. All recovery is unapproved reference data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.booster_k1_retarget.index_reference_repairs import build_index
from tools.booster_k1_retarget.recover_short_reference_candidates import (
    dump,
    eligible,
    recover,
)

REPO = Path(__file__).resolve().parents[2]


def score_review(path):
    report = json.loads(path.read_text())
    if len(report["clips"]) != 1:
        raise ValueError("One clip required per retry variant")
    row = report["clips"][0]
    manifest = json.loads(
        (path.parent / "80_short_or_source_review/manifest.json").read_text()
    )
    short = sum(
        c["length"] for c in manifest["clips"].values() if eligible(c, row)
    )
    error = row.get("native_leg_direction_p95_deg")
    return (
        row["candidate_frames"] + short,
        -float(error if error is not None else 1e9),
    ), short


def collect_targets(run):
    targets = []
    for file in sorted(run.glob("batch_*/status.json")):
        status = json.loads(file.read_text())
        if status["status"] == "source_or_solver_failure":
            inp = Path(status["attempt"]) / "input"
            for key in status["motion_keys"]:
                targets.append((key, inp, True))
        if status["status"] != "complete":
            continue
        report_path = Path(status["report"])
        rows = json.loads(report_path.read_text())["clips"]
        short_manifest = json.loads(
            (
                report_path.parent / "80_short_or_source_review/manifest.json"
            ).read_text()
        )
        for row in rows:
            if row["candidate_frames"] or row.get("source_error") is not None:
                continue
            key = row["motion_key"]
            if any(
                c["metadata"]["parent_motion_key"] == key and eligible(c, row)
                for c in short_manifest["clips"].values()
            ):
                continue
            targets.append((key, Path(status["attempt"]) / "input", False))
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-roots", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--seed-failed-root", type=Path)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "parents": [str(r.resolve()) for r in a.run_roots],
        "smpl_model": str(a.smpl_model.resolve()),
        "foot_orientation_costs": [0.5, 0.1, 2.0],
        "training_approved": False,
        "worker_sha256": {
            name: hashlib.sha256(
                (REPO / "tools/booster_k1_retarget" / name).read_bytes()
            ).hexdigest()
            for name in [
                "retry_guarded_reference.py",
                "probe_native_position_retarget.py",
                "salvage_repaired_reference.py",
            ]
        },
    }
    config = a.output / "run_config.json"
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError("Retry settings changed; use a new versioned output")
    dump(config, settings)
    discovered_count = 0

    def progress():
        statuses = [
            json.loads(f.read_text())
            for f in a.output.glob("batch_*/status.json")
        ]
        complete = [s for s in statuses if s["status"] == "complete"]
        build_index(a.output)
        short = recover(a.output, a.output / "supplemental_short_candidates")
        result = {
            "status": "data_repair_in_progress",
            "input_clips": len(statuses),
            "discovered_retry_targets": discovered_count,
            "completed_clips": len(complete),
            "failed_individual_clips": sum(
                s["status"] == "source_or_solver_failure" for s in statuses
            ),
            "candidate_frames": sum(
                s.get("candidate_frames", 0) for s in complete
            ),
            "short_candidate_frames": short["frames"],
            "training_started": False,
            "training_approved": False,
            "selection_policy": "Select one whole variant per original reference by retained frames, then source leg fidelity. No stitching between variants; all original failed/held records preserved.",
            "updated_unix_seconds": time.time(),
        }
        dump(a.output / "progress.json", result)
        return result

    def process(run, key, source_input, was_failure):
        identity = hashlib.sha256((run.name + ":" + key).encode()).hexdigest()[
            :16
        ]
        folder = a.output / ("batch_retry_" + identity)
        status_path = folder / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())[
            "status"
        ] in {"complete", "source_or_solver_failure"}:
            return
        folder.mkdir(parents=True, exist_ok=True)
        source = json.loads((source_input / "manifest.json").read_text())
        entry = source["clips"][key]
        if a.seed_failed_root and was_failure:
            seed = (
                a.seed_failed_root
                / Path(entry["metadata"]["source_path"]).stem
                / "review/report.json"
            )
            if (
                seed.exists()
                and json.loads(seed.read_text())["clips"][0]["motion_key"]
                == key
            ):
                report = json.loads(seed.read_text())
                dump(
                    status_path,
                    {
                        "status": "complete",
                        "input_clips": 1,
                        "motion_keys": [key],
                        "report": str(seed.resolve()),
                        "original_pool": run.name,
                        "original_failure": True,
                        "seed_guarded_repair": True,
                        **{
                            k: report[k]
                            for k in [
                                "candidate_frames",
                                "rejected_frames",
                                "short_or_source_review_frames",
                            ]
                        },
                    },
                )
                progress()
                return
        inp = folder / "input"
        manifest = {
            **source,
            "clips": {key: entry},
            "motion_keys": [key],
            "hdf5_shards": [
                {**s, "file": str((source_input / s["file"]).resolve())}
                for s in source["hdf5_shards"]
            ],
        }
        dump(inp / "manifest.json", manifest)
        dump(
            status_path,
            {
                "status": "running",
                "motion_keys": [key],
                "original_pool": run.name,
                "original_failure": was_failure,
            },
        )
        options = []
        for cost in settings["foot_orientation_costs"]:
            base = folder / f"orientation_{cost}"
            ready = sorted(base.glob("attempt_*/review/report.json"))
            if ready:
                report = ready[-1]
            else:
                attempt = (
                    base / f"attempt_{len(list(base.glob('attempt_*'))):03d}"
                )
                attempt.mkdir(parents=True)
                commands = [
                    [
                        sys.executable,
                        "-m",
                        "tools.booster_k1_retarget.retry_guarded_reference",
                        "--input",
                        str(inp),
                        "--output",
                        str(attempt / "raw"),
                        "--smpl-model",
                        str(a.smpl_model),
                        "--foot-orientation-cost",
                        str(cost),
                    ],
                    [
                        sys.executable,
                        "-m",
                        "tools.booster_k1_retarget.salvage_repaired_reference",
                        "--input",
                        str(attempt / "raw"),
                        "--output",
                        str(attempt / "review"),
                        "--smpl-model",
                        str(a.smpl_model),
                    ],
                ]
                try:
                    for i, cmd in enumerate(commands):
                        with (attempt / f"step_{i}.log").open("w") as log:
                            subprocess.run(
                                cmd,
                                cwd=REPO,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=3600,
                            )
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    dump(
                        attempt / "failure.json",
                        {"error": str(exc), "source_preserved": True},
                    )
                    continue
                report = attempt / "review/report.json"
            value, short = score_review(report)
            r = json.loads(report.read_text())
            if r["source_frames"] != entry["length"]:
                raise ValueError("Retry source span changed")
            options.append((value, str(report.resolve()), cost, short))
            if value[0] == entry["length"]:
                break
        if not options:
            dump(
                status_path,
                {
                    "status": "source_or_solver_failure",
                    "motion_keys": [key],
                    "original_pool": run.name,
                    "source_preserved": True,
                },
            )
        else:
            value, report, cost, short = max(options)
            r = json.loads(Path(report).read_text())
            dump(
                status_path,
                {
                    "status": "complete",
                    "input_clips": 1,
                    "motion_keys": [key],
                    "report": report,
                    "original_pool": run.name,
                    "original_failure": was_failure,
                    "selected_foot_orientation_cost": cost,
                    "candidate_plus_short_frames": value[0],
                    "short_candidate_frames": short,
                    "variants": [
                        {"report": o[1], "cost": o[2], "kept_frames": o[0][0]}
                        for o in options
                    ],
                    **{
                        k: r[k]
                        for k in [
                            "candidate_frames",
                            "rejected_frames",
                            "short_or_source_review_frames",
                        ]
                    },
                },
            )
        progress()

    stale_since = time.monotonic()
    last_parent = None
    while True:
        before = [
            (s["completed_clips"], s["status"])
            for s in [
                json.loads((r / "progress.json").read_text())
                for r in a.run_roots
            ]
        ]
        pending = []
        for run in a.run_roots:
            pending.extend((run, *t) for t in collect_targets(run))
        pending.sort(key=lambda t: (not t[3], t[0].name, t[1]))
        discovered_count = len(pending)
        for run, key, inp, was_failure in pending:
            if (a.output / "STOP_AFTER_CASE").exists():
                return
            for name, digest in settings["worker_sha256"].items():
                if (
                    hashlib.sha256(
                        (
                            REPO / "tools/booster_k1_retarget" / name
                        ).read_bytes()
                    ).hexdigest()
                    != digest
                ):
                    raise RuntimeError(
                        "Retry worker changed; use a new versioned output"
                    )
            process(run, key, inp, was_failure)
        result = progress()
        parent = [
            json.loads((r / "progress.json").read_text()) for r in a.run_roots
        ]
        signature = [(s["completed_clips"], s["status"]) for s in parent]
        if a.follow and signature != before:
            continue
        if signature != last_parent:
            stale_since = time.monotonic()
            last_parent = signature
        done = all(s["status"] != "data_repair_in_progress" for s in parent)
        if not a.follow or done:
            result["status"] = (
                "data_repair_complete_visual_physics_review_pending"
                if done
                else "current_retry_targets_complete_parent_still_running"
            )
            dump(a.output / "progress.json", result)
            return
        if time.monotonic() - stale_since > 7200:
            result["status"] = "parent_progress_stale_review_required"
            dump(a.output / "progress.json", result)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
