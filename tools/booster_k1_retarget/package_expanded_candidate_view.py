"""Package checked candidate views into one manifest; no training approval."""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("Do not overwrite existing package")
    clips = {}
    shards = []
    occupied = defaultdict(list)
    frames = 0
    stages = {}
    names = None
    for file in sorted(root.glob("*/manifest.json")):
        m = json.loads(file.read_text())
        stage = file.parent.name
        stage_frames = 0
        if names is not None and names != m["dof_names"]:
            raise ValueError("Different DOF order")
        names = m["dof_names"]
        assert len(names) == 22
        for key, c in m["clips"].items():
            meta = c["metadata"]
            s = meta["canonical_frame_start"]
            e = s + c["length"]
            assert c["length"] >= 50
            assert (
                meta["parent_frame_end"] - meta["parent_frame_start"]
                == c["length"]
            )
            assert meta["root_error_max_deg"] <= 10
            occupied[str(Path(meta["source_path"]).resolve())].append((s, e))
            shard = m["hdf5_shards"][c["shard"]]
            path = (file.parent / shard["file"]).resolve()
            with h5py.File(path, "r") as f:
                start, end = c["start"], c["start"] + c["length"]
                q = f["ref_dof_pos"][start:end]
                pos = f["ref_root_pos"][start:end]
                rot = f["ref_root_rot"][start:end]
                assert (
                    q.shape == (c["length"], 22)
                    and pos.shape == (c["length"], 3)
                    and rot.shape == (c["length"], 4)
                )
                assert all(np.isfinite(v).all() for v in [q, pos, rot])
                assert np.max(np.abs(np.linalg.norm(rot, axis=1) - 1)) < 0.001
            out = copy.deepcopy(c)
            out["shard"] = len(shards)
            out["metadata"]["provisional_difficulty"] = stage
            clips[stage + "__" + key] = out
            shards.append({**shard, "file": str(path)})
            frames += c["length"]
            stage_frames += c["length"]
        stages[stage] = {
            "segments": len(m["clips"]),
            "hours": stage_frames / 180000,
        }
    for source, spans in occupied.items():
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Source interval overlap: " + source)
    summary = json.loads((root / "summary.json").read_text())
    assert frames == summary["frames"] and len(clips) == summary["segments"]
    dump(
        root / "manifest.json",
        {
            "version": 2,
            "dataset_id": root.name,
            "root": str(root),
            "dof_names": names,
            "clips": clips,
            "motion_keys": list(clips),
            "hdf5_shards": shards,
            "array_names": ["ref_root_pos", "ref_root_rot", "ref_dof_pos"],
            "training_approved": False,
            "quality_status": "experimental_candidates_inherited_gates_plus_intended_root_fidelity_visual_dynamics_pending",
        },
    )
    dump(root / "nan_npz_paths.json", [])
    dump(
        root / "delivery_status.json",
        {
            "status": "experimental_candidate_package_available_not_formal_release",
            "training_started": False,
            "training_approved": False,
            "segments": len(clips),
            "frames": frames,
            "hours": frames / 180000,
            "stages": stages,
            "integrity_checks": [
                "All referenced arrays read and finite",
                "Correct 22 DOF dimensions/order and unit root quaternions",
                "Manifest frames match summary",
                "No canonical source-frame overlaps",
                "Retained intended-root errors <=10deg",
                "Minimum segment length 50 frames",
            ],
            "limitations": summary["limitations"]
            + [
                "Do not add old LaFAN baseline duration to this package; source overlap may exist",
                "No source metadata semantics or full-body visual approval implied",
            ],
        },
    )


if __name__ == "__main__":
    main()
