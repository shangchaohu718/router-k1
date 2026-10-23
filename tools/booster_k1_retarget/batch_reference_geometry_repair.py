"""Resumable data-only repair; failed batches split down to individual clips.

Never changes a training config, source shard, active snapshot, or checkpoint.
All recovered pieces remain unapproved candidates in separate batch folders.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--proportional-skeleton", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    a.output.mkdir(parents=True, exist_ok=True)
    source_path = a.input.resolve() / "manifest.json"
    raw = source_path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    settings = {
        "input": str(a.input.resolve()),
        "input_manifest_sha256": fingerprint,
        "smpl_model": str(a.smpl_model.resolve()),
        "batch_size": a.batch_size,
        "proportional_skeleton": a.proportional_skeleton,
        "training_approved": False,
        "worker_code_sha256": {
            name: hashlib.sha256(
                (REPO / "tools/booster_k1_retarget" / name).read_bytes()
            ).hexdigest()
            for name in (
                "probe_native_position_retarget.py",
                "salvage_repaired_reference.py",
                "holosmpl_h5_to_k1_h5.py",
            )
        },
    }
    settings_path = a.output / "run_config.json"
    if (
        settings_path.exists()
        and json.loads(settings_path.read_text()) != settings
    ):
        raise ValueError(
            "Resume settings/input differ; use a new output directory"
        )
    dump(settings_path, settings)
    source = json.loads(raw)
    groups = {}
    for key, clip in source["clips"].items():
        native = Path(clip["metadata"]["source_path"]).suffix.lower() in {
            ".amc",
            ".bvh",
        }
        groups.setdefault(native, []).append(key)
    summary_path = a.output / "progress.json"

    def progress():
        from tools.booster_k1_retarget.index_reference_repairs import (
            build_index,
        )

        build_index(a.output)
        statuses = [
            json.loads(f.read_text())
            for f in a.output.glob("batch_*/status.json")
        ]
        complete = [s for s in statuses if s["status"] == "complete"]
        summary = {
            "status": "data_repair_in_progress",
            "training_started": False,
            "training_approved": False,
            "organized_candidate_summary": str(
                (a.output / "organized_candidates/summary.json").resolve()
            ),
            "input_clips": len(source["clips"]),
            "completed_clips": sum(s["input_clips"] for s in complete),
            "failed_individual_clips": sum(
                s["status"] == "source_or_solver_failure" for s in statuses
            ),
            "candidate_frames": sum(
                s.get("candidate_frames", 0) for s in complete
            ),
            "rejected_frames": sum(
                s.get("rejected_frames", 0) for s in complete
            ),
            "short_or_source_review_frames": sum(
                s.get("short_or_source_review_frames", 0) for s in complete
            ),
            "batch_status_counts": dict(
                Counter(s["status"] for s in statuses)
            ),
            "updated_unix_seconds": time.time(),
        }
        dump(summary_path, summary)
        return summary

    def process(keys, native, label):
        base = a.output / ("batch_" + label)
        status_path = base / "status.json"
        if status_path.exists():
            prior = json.loads(status_path.read_text())
            if prior["status"] in {"complete", "source_or_solver_failure"}:
                return
            if prior["status"] == "split_retry":
                mid = len(keys) // 2
                process(keys[:mid], native, label + "a")
                process(keys[mid:], native, label + "b")
                return
        base.mkdir(parents=True, exist_ok=True)
        # Each attempt is immutable; a killed worker resumes into a new attempt.
        attempt = base / f"attempt_{len(list(base.glob('attempt_*'))):03d}"
        inp = attempt / "input"
        inp.mkdir(parents=True)
        manifest = {
            **source,
            "clips": {k: source["clips"][k] for k in keys},
            "motion_keys": keys,
        }
        manifest["hdf5_shards"] = [
            {**s, "file": str((a.input / s["file"]).resolve())}
            for s in source["hdf5_shards"]
        ]
        dump(inp / "manifest.json", manifest)
        dump(
            status_path,
            {
                "status": "running",
                "motion_keys": keys,
                "input_clips": len(keys),
                "attempt": str(attempt.resolve()),
            },
        )
        progress()
        commands = [
            [
                sys.executable,
                str(
                    REPO
                    / "tools/booster_k1_retarget/probe_native_position_retarget.py"
                ),
                "--input",
                str(inp),
                "--output",
                str(attempt / "raw"),
                "--smpl-model",
                str(a.smpl_model),
                "--keys",
                "",
                "--max-frames",
                "10000000",
                "--collision",
                "--line-search",
                "--anatomical-scale",
            ]
            + (["--native"] if native else [])
            + (["--proportional-skeleton"] if a.proportional_skeleton else []),
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
            for i, command in enumerate(commands):
                with (attempt / f"step_{i}.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=3600,
                    )
            report = json.loads((attempt / "review/report.json").read_text())
            dump(
                status_path,
                {
                    "status": "complete",
                    "motion_keys": keys,
                    "input_clips": len(keys),
                    "attempt": str(attempt.resolve()),
                    "report": str((attempt / "review/report.json").resolve()),
                    **{
                        k: report[k]
                        for k in (
                            "candidate_frames",
                            "rejected_frames",
                            "short_or_source_review_frames",
                        )
                    },
                },
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            if len(keys) > 1:
                dump(
                    status_path,
                    {
                        "status": "split_retry",
                        "motion_keys": keys,
                        "input_clips": len(keys),
                        "error": str(exc),
                    },
                )
                mid = len(keys) // 2
                process(keys[:mid], native, label + "a")
                process(keys[mid:], native, label + "b")
            else:
                dump(
                    status_path,
                    {
                        "status": "source_or_solver_failure",
                        "motion_keys": keys,
                        "input_clips": 1,
                        "error": str(exc),
                        "source_preserved": True,
                        "attempt": str(attempt.resolve()),
                    },
                )
        progress()

    count = 0
    for native in sorted(groups):
        keys = groups[native]

        def priority(key):
            meta = source["clips"][key]["metadata"]
            description = (
                key + " " + meta.get("geometry_review_description", "")
            ).lower()
            dance = any(
                word in description
                for word in (
                    "dance",
                    "dancing",
                    "ballet",
                    "salsa",
                    "swing",
                    "charleston",
                )
            )
            return (not dance, meta["source_path"], key)

        keys.sort(key=priority)
        for i in range(0, len(keys), a.batch_size):
            if (a.output / "STOP_AFTER_BATCH").exists():
                summary = progress()
                summary["status"] = "stopped_after_batch_no_training"
                dump(summary_path, summary)
                return
            for name, digest in settings["worker_code_sha256"].items():
                if (
                    hashlib.sha256(
                        (
                            REPO / "tools/booster_k1_retarget" / name
                        ).read_bytes()
                    ).hexdigest()
                    != digest
                ):
                    raise RuntimeError(
                        "Repair worker changed during run; use a new versioned output"
                    )
            if a.max_batches and count >= a.max_batches:
                progress()
                return
            process(
                keys[i : i + a.batch_size],
                native,
                f"{'native' if native else 'smpl'}_{i // a.batch_size:04d}",
            )
            count += 1
    summary = progress()
    summary["status"] = "data_repair_complete_visual_physics_review_pending"
    dump(summary_path, summary)


if __name__ == "__main__":
    main()
