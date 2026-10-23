"""Resumable CPU-only source-group conversion, K1 retargeting and quarantine.

No training is launched. Each group owns separate output folders and logs.
Completed groups are immutable and skipped on resume; incomplete folders are
never overwritten. Failures are recorded for explicit repair, not hidden.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys
import time

from organize_motion_curriculum import partition

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run_group(source, group, path, args):
    destination = args.output / source / group
    status = destination / "status.json"
    if status.exists():
        prior = json.loads(status.read_text())
        if prior["status"] == "complete":
            return prior
        raise RuntimeError(
            f"Incomplete group needs inspection before retry: {destination}"
        )
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError("Less than 25 GiB disk space remains")
    destination.mkdir(parents=True, exist_ok=True)
    value = {
        "source": source,
        "group": group,
        "status": "running",
        "stage": "human",
        "started_at": time.time(),
    }
    save(status, value)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HOLOSMPL_RESAMPLE_BACKEND": "scipy",
        "PYTHONPATH": str(REPO),
    }

    def command(stage, arguments):
        value["stage"] = stage
        save(status, value)
        with (destination / (stage + ".log")).open("w") as log:
            subprocess.run(
                [sys.executable, *arguments],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        human = destination / "human"
        command(
            "human",
            [
                "-m",
                "holosmpl",
                "convert",
                "--source",
                source,
                "--input-root",
                str(path),
                "--output-root",
                str(human),
            ],
        )
        raw = destination / "robot_raw"
        filtered = destination / "robot_filtered"
        command(
            "retarget",
            [
                "tools/booster_k1_retarget/holosmpl_h5_to_k1_h5.py",
                "--input-root",
                str(human / "formal_h5"),
                "--smpl-model",
                str(args.smpl_model),
                "--all-clips",
                "--output-root",
                str(raw),
            ],
        )
        command(
            "quality",
            [
                "tools/booster_k1_retarget/ground_release_candidates.py",
                "--input",
                str(raw),
                "--output",
                str(filtered),
            ],
        )
        rows = json.loads((filtered / "preparation_report.json").read_text())
        rejected = {r["motion"] for r in rows if r.get("reasons")}
        if rejected:
            partition(raw, destination / "quarantine", rejected)
        split_number = (
            int(
                hashlib.sha256(f"{source}/{group}".encode()).hexdigest()[:8],
                16,
            )
            % 10
        )
        split = (
            "train"
            if split_number < 8
            else ("val" if split_number == 8 else "test")
        )
        if (source, group) in [("style100", "Neutral"), ("cmu_asf", "07")]:
            split = "train"
        for folder in [raw, filtered]:
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest.update(
                dataset_id=f"{source}_{group}",
                group_split=split,
                quality_status="numerical_only_requires_visual_and_physics_validation",
            )
            save(folder / "manifest.json", manifest)
        value.update(
            status="complete",
            stage="done",
            finished_at=time.time(),
            split=split,
            converted=len(rows),
            accepted=len(rows) - len(rejected),
            quarantined=len(rejected),
            filtered_root=str(filtered),
            raw_root=str(raw),
        )
    except Exception as exc:
        value.update(status="failed", error=str(exc), finished_at=time.time())
    save(status, value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--source", choices=["all", "style100", "cmu_asf"], default="all"
    )
    p.add_argument("--groups", nargs="*")
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.smpl_model.is_file():
        raise FileNotFoundError(args.smpl_model)
    jobs = []
    for source, raw in [
        ("style100", ROOT / "raw/100style/100STYLE"),
        ("cmu_asf", ROOT / "raw/cmu/all_asfamc/subjects"),
    ]:
        if args.source not in ("all", source):
            continue
        for path in sorted(raw.iterdir()):
            if path.is_dir() and (not args.groups or path.name in args.groups):
                jobs.append((source, path.name, path))
    # Validate baseline and common locomotion groups before exotic styles.
    jobs.sort(
        key=lambda item: (
            item[1] not in ("Neutral", "07", "09", "16"),
            item[0],
            item[1],
        )
    )
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_group, *job, args): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "source": job[0],
                    "group": job[1],
                    "status": "failed",
                    "error": str(exc),
                }
            outcomes.append(result)
            save(
                args.output / "progress.json",
                {
                    "groups_total": len(jobs),
                    "groups_finished": len(outcomes),
                    "outcomes": outcomes,
                },
            )
            print(f"{len(outcomes)}/{len(jobs)} {result}", flush=True)
    save(
        args.output / "completion.json",
        {
            "groups": len(jobs),
            "outcomes": outcomes,
            "formal_training_started": False,
            "physics_approval": False,
        },
    )
    # Build a new immutable classification snapshot when this batch finishes.
    snapshot = ROOT / ("organized_" + time.strftime("%Y%m%d_%H%M%S"))
    subprocess.run(
        [
            sys.executable,
            "tools/booster_k1_retarget/organize_motion_curriculum.py",
            "--output",
            str(snapshot),
        ],
        cwd=REPO,
        check=True,
    )
    save(args.output / "classification_snapshot.json", {"path": str(snapshot)})


if __name__ == "__main__":
    main()
