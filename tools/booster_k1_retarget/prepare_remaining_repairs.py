"""Resolve unique original clips for newly exposed segment mapping failures."""

import json
from pathlib import Path

from localize_and_salvage import dump

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def main():
    current = json.loads(
        (ROOT / "organized_frame_salvage_v2/all_motions.json").read_text()
    )
    original = {
        (r["source"], r["motion"]): r
        for r in json.loads(
            (
                ROOT / "organized_arm_repaired_20261020_v2/all_motions.json"
            ).read_text()
        )
    }
    parents = {
        (r["source"], r["parent_motion"])
        for r in current
        if r["quality"] == "recovered_segment_direction_review_failed"
    }
    selected = [original[key] for key in sorted(parents)]
    for row in selected:
        status = Path(row["robot_root"]).parent / "status.json"
        if not status.is_file():
            raise FileNotFoundError(status)
        if not (status.parent / "human/formal_h5/manifest.json").is_file():
            raise FileNotFoundError(
                status.parent / "human/formal_h5/manifest.json"
            )
    output = ROOT / "remaining_repair_v1"
    output.mkdir(exist_ok=False)
    dump(output / "selection.json", selected)
    dump(
        output / "unresolved_descriptions.json",
        [
            r
            for r in current
            if r.get("semantic_hold") == "unresolved_action_description"
        ],
    )
    print(
        f"Selected {len(selected)} original clips; all human input manifests exist"
    )


if __name__ == "__main__":
    main()
