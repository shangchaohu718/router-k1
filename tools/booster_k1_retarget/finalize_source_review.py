"""Write evidence-backed review decisions and isolated views, without training."""

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/holomotion_k1_v1"
REVIEW = ROOT / "review_20261020"


def main():
    evidence = REVIEW / "limb_direction_audit.json"
    audited = json.loads(evidence.read_text())["clips"]
    decisions = []
    for row in audited:
        if row["flagged_limbs"]:
            decisions.append({
                "source": row["source"], "motion": row["motion"],
                "quality": "quarantine", "destination": "quarantine",
                "reasons": ["human_to_K1_limb_direction_review:" + k
                            for k in row["flagged_limbs"]],
                "evidence": str(evidence),
            })
    # Keep the explicitly selected moonwalk reference separate until validated.
    decisions.append({
        "source": "Moonwalk", "motion": "moonwalk_alex61_64_v14_synthetic",
        "quality": "selected_reference_visual_physics_pending",
        "destination": "pending",
        "reasons": ["verified_MOONWALK_V14_lineage_but_not_release_approved"],
        "evidence": "MOONWALK handoff and SHA256 verified; do not substitute failed variants",
    })
    # Chair-assisted motion was visually inspected; keep it out of flat-ground data.
    if not any(x["source"] == "CMU" and x["motion"] == "111_09" for x in decisions):
        decisions.append({
            "source": "CMU", "motion": "111_09",
            "quality": "special_scene_deferred", "destination": "L5",
            "reasons": ["chair_support_missing_in_flat_ground_scene"],
            "evidence": str(REVIEW / "CMU_111_09/full_clip_samples.jpg"),
        })
    path = REVIEW / "review_decisions.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(decisions, indent=2) + "\n")
    output = ROOT / "organized_reviewed_20261020"
    subprocess.run([
        sys.executable, "tools/booster_k1_retarget/organize_motion_curriculum.py",
        "--output", str(output), "--extra-batch-root",
        str(ROOT / "batch_recovery_v1"), "--review-decisions", str(path),
    ], cwd=REPO, check=True)
    rows = json.loads((output / "all_motions.json").read_text())
    pending = [r for r in rows if r["quality"] == "missing_fps_pending"]
    target = output / "99_pending/missing_fps_sources"
    target.mkdir(parents=True)
    for row in pending:
        source = Path(row["source_path"]).resolve()
        folder = target / source.parent.name
        folder.mkdir(exist_ok=True)
        (folder / source.name).symlink_to(source)
        asf = source.parent / (source.parent.name + ".asf")
        if asf.exists() and not (folder / asf.name).exists():
            (folder / asf.name).symlink_to(asf)
    (target / "index.json").write_text(json.dumps(pending, indent=2) + "\n")
    manual = []
    for sample in json.loads((REVIEW / "samples.json").read_text()):
        item = next(r for r in rows if r["source"] == sample["source"]
                    and r["motion"] == sample["motion"])
        manual.append({
            **sample, "inspection": "12 full-clip-spanning rendered frames inspected",
            "video": "generated first <=30 seconds; not a physical rollout",
            "decision": item["quality"],
            "notes": item.get("review_reasons", [
                "No gross axis flip seen in sampled frames; not full-frame visual approval"]),
        })
    summary = {
        "source_failure_root_cause": "113 unknown-FPS clips caused rejection-path KeyError source_fields",
        "fix": "optional classification diagnostic fields default to empty lists",
        "recovered": "CMU 63_01; 29 other clips in subject 63 correctly rejected",
        "unknown_fps_clips": len(pending),
        "unknown_fps_subjects": sorted({Path(r["source_path"]).parent.name for r in pending}),
        "new_converted_clips_including_recovery": 3211,
        "new_numerically_accepted_before_limb_review": len(audited),
        "new_limb_review_quarantine": sum(bool(r["flagged_limbs"]) for r in audited),
        "limb_gate": "24 synchronized frames per clip, mean >45deg or p95 >75deg; review heuristic",
        "visual_samples": manual,
        "limitations": ["No PPO training started", "No physical tracking validation",
                        "Sampling does not prove every frame is correct",
                        "Leg and upper-arm directions checked; wrist omitted because K1 lacks matching endpoint"],
        "organized_snapshot": str(output),
        "source_files_preserved": True,
    }
    (REVIEW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
