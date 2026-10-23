"""Rebuild selected IK failures in a new immutable batch; preserve source splits."""

import argparse
import contextlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
import time

from run_source_adaptation import save

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"
CONTROLS = {
    ("100STYLE", "Neutral_FW"),
    ("100STYLE", "Neutral_ID"),
    ("CMU", "02_01"),
    ("CMU", "113_21"),
    ("100STYLE", "Aeroplane_FW"),
}
PILOT = CONTROLS | {
    ("100STYLE", "Neutral_FR"),
    ("100STYLE", "Akimbo_FR"),
    ("CMU", "02_03"),
}


def work(job):
    old_status, names, output, smpl_model = job
    status = json.loads(Path(old_status).read_text())
    dest = Path(output) / status["source"] / status["group"]
    if (dest / "status.json").exists():
        old = json.loads((dest / "status.json").read_text())
        if old["status"] == "complete":
            return old
        raise RuntimeError(f"Incomplete repair group: {dest}")
    dest.mkdir(parents=True, exist_ok=False)
    value = {k: status[k] for k in ["source", "group", "split"]}
    value.update(
        status="running",
        started_at=time.time(),
        repair="first_reference_root_seed",
        previous_status=str(old_status),
        selected_motions=names,
    )
    save(dest / "status.json", value)
    try:
        from holosmpl_h5_to_k1_h5 import (
            create_k1_retargeter,
            HoloSmplToGmrConverter,
            load_all_holosmpl_clips,
            write_k1_robot_h5_dataset,
        )

        human = Path(old_status).parent / "human/formal_h5"
        with (
            (dest / "retarget.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            converter = HoloSmplToGmrConverter(Path(smpl_model))
            r = create_k1_retargeter(actual_human_height=1.8)
            clips = [
                c
                for c in load_all_holosmpl_clips(input_root=human)
                if Path(c.metadata["source_path"]).stem in names
            ]
            assert len(clips) == len(names)
            write_k1_robot_h5_dataset(
                output_root=dest / "robot_raw",
                input_root=human,
                clips=clips,
                retargeter=r,
                human_converter=converter,
                max_frames=0,
                overwrite=False,
                shard_target_frames=250_000,
            )
        with (dest / "quality.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "tools/booster_k1_retarget/ground_release_candidates.py",
                    "--input",
                    str(dest / "robot_raw"),
                    "--output",
                    str(dest / "robot_filtered"),
                ],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        reports = json.loads(
            (dest / "robot_filtered/preparation_report.json").read_text()
        )
        from organize_motion_curriculum import partition

        rejected = {x["motion"] for x in reports if x["reasons"]}
        if rejected:
            partition(dest / "robot_raw", dest / "quarantine", rejected)
        value.update(
            status="complete",
            converted=len(reports),
            accepted=len(reports) - len(rejected),
            quarantined=len(rejected),
            raw_root=str(dest / "robot_raw"),
            filtered_root=str(dest / "robot_filtered"),
        )
    except Exception as exc:
        value.update(status="failed", error=repr(exc))
    value["finished_at"] = time.time()
    save(dest / "status.json", value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smpl-model", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--root-only-supplement", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--selection-json",
        type=Path,
        help="Explicit source/motion/robot_root records",
    )
    a = p.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    audit = json.loads(
        (ROOT / "review_20261020/limb_direction_audit.json").read_text()
    )["clips"]
    selected = (
        PILOT
        if a.pilot
        else CONTROLS
        | {(x["source"], x["motion"]) for x in audit if x["flagged_limbs"]}
    )
    if a.root_only_supplement:
        audit = json.loads(
            (ROOT / "arm_repair_v1/baseline_limb_audit_v3.json").read_text()
        )["clips"]
        selected = {
            (x["source"], x["motion"])
            for x in audit
            if not x["flagged_limbs"]
            and x["root_orientation_error_deg"]["p95"] > 45
        } - CONTROLS
    jobs = {}
    if a.selection_json:
        audit = json.loads(a.selection_json.read_text())
        selected = {(x["source"], x["motion"]) for x in audit}
    for row in audit:
        if (row["source"], row["motion"]) in selected:
            status = Path(row["robot_root"]).parent / "status.json"
            jobs.setdefault(str(status), []).append(row["motion"])
    assert sum(map(len, jobs.values())) == len(selected)
    output = a.output.resolve()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs = [
            pool.submit(
                work, (s, names, str(output), str(a.smpl_model.resolve()))
            )
            for s, names in jobs.items()
        ]
        for future in as_completed(fs):
            results.append(future.result())
            save(
                output / "progress.json",
                {
                    "groups_total": len(jobs),
                    "groups_finished": len(results),
                    "outcomes": results,
                },
            )
            print(results[-1], flush=True)
    save(
        output / "completion.json",
        {"outcomes": results, "training_started": False},
    )


if __name__ == "__main__":
    main()
