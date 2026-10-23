"""Inspect downloaded BVH headers and official trim ranges, without retargeting."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"


def main():
    with (ROOT / "downloads/Frame_Cuts.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        cuts = {row["STYLE_NAME"]: row for row in csv.DictReader(handle)}
    categories = {
        "FW": "forward_walk",
        "BW": "backward_walk",
        "ID": "idle",
        "FR": "forward_run",
        "BR": "backward_run",
        "SW": "side_walk",
        "SR": "side_run",
        "TR1": "transitions",
        "TR2": "transitions",
        "TR3": "transitions",
    }
    rows = []
    for path in sorted((ROOT / "raw/100style").rglob("*.bvh")):
        style, movement = path.stem.rsplit("_", 1)
        count = fps = None
        bones = []
        with path.open() as handle:
            for line in handle:
                tokens = line.split()
                if tokens and tokens[0] in ("ROOT", "JOINT"):
                    bones.append(tokens[1])
                if line.strip().startswith("Frames:"):
                    count = int(line.split(":", 1)[1])
                if line.strip().startswith("Frame Time:"):
                    fps = 1 / float(line.split(":", 1)[1])
                    break
        source_row = cuts.get(style, {})
        start = source_row.get(movement + "_START", "N/A")
        stop = source_row.get(movement + "_STOP", "N/A")
        problems = []
        if start == "N/A" or stop == "N/A":
            start = stop = None
            problems.append("official_trim_missing")
        else:
            start, stop = int(start), int(stop)
            if count is None or not 0 <= start < stop <= count:
                problems.append("trim_out_of_bounds")
        if fps is None or abs(fps - 60) > 0.1:
            problems.append("unexpected_fps")
        rows.append(
            {
                "path": str(path),
                "style": style,
                "movement": movement,
                "category": categories.get(movement, "unknown"),
                "source_frames": count,
                "source_fps": fps,
                "official_trim_start": start,
                "official_trim_stop": stop,
                "bones": bones,
                "problems": problems,
                "status": "RAW_ONLY_K1_ADAPTER_AND_SEMANTIC_VALIDATION_PENDING",
            }
        )
    if not rows:
        raise ValueError("No extracted 100STYLE BVH files found")
    report = {
        "source": "https://www.ianxmason.com/100style/",
        "license_from_author_page": "CC BY 4.0; credit The 100STYLE Dataset - Ian Mason",
        "clips": len(rows),
        "styles": len({r["style"] for r in rows}),
        "categories": dict(Counter(r["category"] for r in rows)),
        "raw_frames": sum(r["source_frames"] or 0 for r in rows),
        "header_or_trim_problem_clips": sum(bool(r["problems"]) for r in rows),
        "split_note": "One actor across all styles; do not claim subject-independent evaluation. Split complete recordings/style groups before windowing.",
        "trim_note": "Official indices preserved verbatim; confirm indexing convention on pilot before production trimming.",
        "files": rows,
    }
    destination = ROOT / "catalog/100style_motion_catalog.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2)
    )


if __name__ == "__main__":
    main()
