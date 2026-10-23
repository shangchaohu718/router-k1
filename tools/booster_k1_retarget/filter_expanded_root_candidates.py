"""Filter existing deduplicated candidates by intended source-root fidelity.

No retargeting, training or automatic approval. Canonical SMPL root target is
the intended GMR target, not independent certification of the source adapter.
"""

import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.localize_and_salvage import intervals, padded
from tools.booster_k1_retarget.probe_native_position_retarget import (
    canonical_arrays,
)
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


@lru_cache(maxsize=16)
def roots(canonical, source, dataset):
    if canonical:
        with np.load(canonical, allow_pickle=False) as f:
            return np.asarray(f["root_orient"])
    r, _, _, _ = canonical_arrays({"source_path": source, "dataset": dataset})
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.output / "summary.json").exists():
        raise FileExistsError("Use a new output version")
    cfg = json.loads(
        Path(
            "tools/booster_k1_retarget/configs/gmr_smplx_to_official_k1_v2.json"
        ).read_text()
    )
    offset = Rotation.from_quat(
        cfg["ik_match_table1"]["trunk"][4], scalar_first=True
    )
    stats = defaultdict(Counter)
    held = []
    processed = 0
    for file in sorted((a.snapshot / "01_candidates").glob("*/manifest.json")):
        m = json.loads(file.read_text())
        stage = file.parent.name
        target = a.output / stage
        view = copy.deepcopy(m)
        view["clips"] = {}
        shards = []
        for key, c in sorted(
            m["clips"].items(),
            key=lambda item: item[1]["metadata"]["source_path"],
        ):
            meta = c["metadata"]
            source = meta.get(
                "geometry_review_source", meta.get("dataset", "unknown")
            )
            stat = stats[source]
            stat["input_frames"] += c["length"]
            processed += 1
            try:
                root = roots(
                    meta.get("canonical_path"),
                    meta["source_path"],
                    meta["dataset"],
                )
                start = meta["canonical_frame_start"]
                end = start + c["length"]
                if start < 0 or end > len(root):
                    raise ValueError("Source frame mismatch")
                shard = m["hdf5_shards"][c["shard"]]
                with h5py.File(file.parent / shard["file"], "r") as f:
                    actual = np.asarray(
                        f["ref_root_rot"][
                            c["start"] : c["start"] + c["length"]
                        ]
                    )
                desired = Rotation.from_rotvec(root[start:end]) * offset
                errors = np.degrees(
                    (desired.inv() * Rotation.from_quat(actual)).magnitude()
                )
                if not np.isfinite(errors).all():
                    raise ValueError("Nonfinite error")
                spans = [
                    (s, e)
                    for s, e in intervals(~padded(errors > 10, 5))
                    if e - s >= 50
                ]
                stat["held_frames"] += c["length"] - sum(
                    e - s for s, e in spans
                )
                for s, e in spans:
                    clip = copy.deepcopy(c)
                    clip["start"] += s
                    clip["length"] = e - s
                    clip["shard"] = len(shards)
                    cm = clip["metadata"]
                    cm.update(
                        canonical_frame_start=start + s,
                        canonical_frame_count=e - s,
                        num_frames=e - s,
                        training_approved=False,
                        root_check="intended canonical GMR root target; 10deg gate + 5frame padding",
                        root_error_max_deg=float(errors[s:e].max()),
                    )
                    cm["parent_frame_start"] += s
                    cm["parent_frame_end"] = cm["parent_frame_start"] + e - s
                    newkey = f"{key}_rootchecked_{s}_{e}"
                    view["clips"][newkey] = clip
                    shards.append(
                        {
                            **shard,
                            "file": str(
                                (file.parent / shard["file"]).resolve()
                            ),
                        }
                    )
                    stat["segments"] += 1
                    stat["frames"] += e - s
            except (ValueError, KeyError, FileNotFoundError) as exc:
                stat["held_frames"] += c["length"]
                held.append(
                    {
                        "motion": key,
                        "source": source,
                        "reason": str(exc),
                        "frames": c["length"],
                    }
                )
            if processed % 100 == 0:
                dump(
                    a.output / "progress.json",
                    {
                        "status": "filtering",
                        "processed_segments": processed,
                        "training_started": False,
                        "training_approved": False,
                    },
                )
        view.update(
            root=str(target.resolve()),
            dataset_id=a.output.name + "_" + stage,
            motion_keys=list(view["clips"]),
            hdf5_shards=shards,
            training_approved=False,
            quality_status="expanded_experimental_candidates_visual_physics_pending",
        )
        dump(target / "manifest.json", view)
        dump(target / "nan_npz_paths.json", [])
    result = {
        "status": "filtered_experimental_candidates_not_formal_release",
        "training_started": False,
        "training_approved": False,
        "source_snapshot": str(a.snapshot.resolve()),
        "sources": {
            k: {**v, "hours": v["frames"] / 180000} for k, v in stats.items()
        },
        "segments": sum(v["segments"] for v in stats.values()),
        "frames": sum(v["frames"] for v in stats.values()),
        "hours": sum(v["frames"] for v in stats.values()) / 180000,
        "limitations": [
            "Inherited source/geometry gates and deduplication remain in force",
            "Canonical target check does not independently validate raw-source adapters",
            "Existing direction gate is primarily upper-arm, not full forearm or hand tracking",
            "No dynamic validation or automatic training approval",
            "No held scene/semantic origins promoted; original data unchanged",
        ],
    }
    dump(a.output / "held_source_errors.json", held)
    dump(a.output / "summary.json", result)
    dump(
        a.output / "progress.json",
        {
            "status": "complete_review_pending",
            "training_started": False,
            "training_approved": False,
            "processed_segments": processed,
        },
    )


if __name__ == "__main__":
    main()
