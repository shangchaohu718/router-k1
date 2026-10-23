"""Build traceable candidate data for HoloMotion-K1 v1, never overwrite originals.

These outputs are candidates until visual and simulation validation pass.
"""

from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
OUT = REPO / "data/holomotion_k1_v1"
DANCE = WORKSPACE / "dance-evaluator and tracker"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    raw = DANCE / "dance_expert/data/source/aistpp"
    cmu = Path("/home/tcl/Downloads/dataset/allasfamc.zip")
    with zipfile.ZipFile(cmu) as z:
        files = [
            {"member": f.filename, "bytes": f.file_size, "crc32": f.CRC}
            for f in z.infolist()
            if f.filename.endswith((".amc", ".asf"))
        ]
    save(
        OUT / "catalog/cmu_archive.json",
        {
            "source": str(cmu),
            "archive_bytes": cmu.stat().st_size,
            "files": files,
            "status": "catalogued_not_retargeted",
            "crc_verified": False,
        },
    )
    aist_hash = sha(raw / "motions.zip")
    expected = (
        "ceff08766802705888701ae27771ddf28d9c0141043b8a12fc342a0cddd2ed1e"
    )
    if aist_hash != expected:
        raise ValueError("Local AIST archive differs from official release")
    with zipfile.ZipFile(raw / "motions.zip") as z:
        if z.testzip() is not None:
            raise ValueError("AIST ZIP CRC failed")
    report = {
        "aistpp": {
            "path": str(raw),
            "sha256": aist_hash,
            "official_asset_url": "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip",
            "official_current_asset_bytes": 78684689,
            "available_clips": 411,
            "historical_webpage_clips": 1408,
            "new_download_required": False,
            "status": "official_current_asset_verified_subset_of_historical_description",
        },
        "cmu": {
            "path": str(cmu),
            "amc_clips": sum(f["member"].endswith(".amc") for f in files),
        },
        "style100": {
            "expected_bytes": 1468515354,
            "md5": "3cf627852fd8192024c04a8d0ef49583",
            "source": "https://zenodo.org/records/8127870",
        },
        "release_status": "PREPARATION_NOT_TRAINING_READY",
    }
    save(OUT / "catalog/sources.json", report)
    print(json.dumps(report, indent=2), flush=True)


def aist():
    source = DANCE / "dance_expert/scripts/build_aistpp_dataset.py"
    spec = importlib.util.spec_from_file_location("aist_builder", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    raw = DANCE / "dance_expert/data/source/aistpp/raw"
    target = OUT / "human/aistpp"
    if target.exists():
        raise FileExistsError(target)
    ignored = module.read_name_set(raw / "ignore_list.txt")
    splits = module.split_lookup(raw / "splits", "pose")
    cross = module.split_lookup(raw / "splits", "crossmodal")
    records = [
        module.inspect_motion(
            p, ignored=ignored, pose_splits=splits, crossmodal_splits=cross
        )
        for p in sorted((raw / "motions").glob("*.pkl"))
    ]
    target.mkdir(parents=True)
    module.write_records_csv(target / "audit_all.csv", records)
    accepted = [r for r in records if r.finite and not r.official_ignored]
    counts = {}
    for split in ("train", "val", "test"):
        selected = [r for r in accepted if r.split_pose == split]
        module.write_subset(
            split, selected, input_root=raw, output_root=target, target_fps=50
        )
        counts[split] = {
            "clips": len(selected),
            "genres": dict(Counter(r.genre_code for r in selected)),
        }
    save(
        target / "summary.json",
        {
            "counts": counts,
            "builder": str(source),
            "builder_sha256": sha(source),
            "split_policy": "official pose subject split; no random slicing",
            "status": "human_converted_robot_qa_pending",
        },
    )
    print(json.dumps(counts, indent=2), flush=True)


def moon():
    from types import SimpleNamespace
    from holosmpl_h5_to_k1_h5 import K1_DOF_NAMES, write_k1_robot_h5

    source = (
        DANCE
        / "k1_moonwalk/motions/k1_61s_64s_repaired_balanced_reference_v14_raw.npz"
    )
    with np.load(source, allow_pickle=False) as z:
        names = list(z["joint_names"].astype(str))
        if len(names) != 22 or set(names) != set(K1_DOF_NAMES):
            raise ValueError("Moonwalk joint-name contract mismatch")
        order = [names.index(name) for name in K1_DOF_NAMES]
        arrays = {
            "ref_root_pos": z["root_pos"].astype(np.float32),
            "ref_root_rot": z["root_quat_wxyz"][:, [1, 2, 3, 0]].astype(
                np.float32
            ),
            "ref_dof_pos": z["joint_pos"][:, order].astype(np.float32),
        }
        original_meta = json.loads(str(z["metadata"]))
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Nonfinite moonwalk data")
    if not np.allclose(
        np.linalg.norm(arrays["ref_root_rot"], axis=1), 1, atol=1e-4
    ):
        raise ValueError("Invalid moonwalk quaternion")
    meta = {
        "dataset": "K1Moonwalk",
        "motion_fps": 50,
        "target_fps": 50,
        "source_path": str(source),
        "source_sha256": sha(source),
        "source_metadata": original_meta,
        "synthetic_loop": True,
        "source_family": "alex_hasanov_61s_64s",
        "split": "train_candidate",
        "quality_status": "candidate_known_contact_tradeoffs_not_release_approved",
        "note": "500 frames are repaired repeats; not 10 seconds of independent capture.",
    }
    clip = SimpleNamespace(
        metadata=meta,
        motion_key="moonwalk_alex61_64_v14_synthetic",
        shard_path=source,
        clip_index=0,
    )
    target = OUT / "candidate_robot/moonwalk_v14"
    write_k1_robot_h5(output_root=target, clip=clip, arrays=arrays)
    correct_moon_metadata(target)
    print(f"Moonwalk candidate imported: {target}", flush=True)


def correct_moon_metadata(target):
    """Remove HoloSMPL/GMR provenance injected by the reusable H5 writer."""
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["source_schema"] = "k1_named_joint_npz_wxyz"
    manifest["retarget_backend"] = "existing_K1_reference_import"
    manifest.pop("source_holosmpl_h5", None)
    for clip in manifest["clips"].values():
        metadata = clip["metadata"]
        metadata["retarget_backend"] = "existing_K1_reference_import"
        metadata["schema_version"] = "imported_k1_reference_h5_clip_v1"
        metadata.pop("source_holosmpl_shard", None)
        metadata.pop("source_holosmpl_clip_index", None)
        shard = target / manifest["hdf5_shards"][clip["shard"]]["file"]
        with h5py.File(shard, "r+") as handle:
            handle["clips/metadata_json"][clip["clip_idx"]] = json.dumps(
                metadata
            )
    save(target / "manifest.json", manifest)


def extract_style():
    source = OUT / "downloads/100STYLE.zip"
    target = OUT / "raw/100style"
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError("100STYLE CRC failed")
        rows = []
        for f in z.infolist():
            if (
                not f.filename.lower().endswith((".bvh", ".csv", ".txt"))
                or "__MACOSX" in f.filename
            ):
                continue
            resolved = (target / f.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
            z.extract(f, target)
            rows.append({"path": str(resolved), "bytes": f.file_size})
    save(OUT / "catalog/100style_files.json", rows)
    print(f"100STYLE extracted {len(rows)} entries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["inventory", "aist", "moon", "extract-style"]
    )
    args = parser.parse_args()
    {
        "inventory": inventory,
        "aist": aist,
        "moon": moon,
        "extract-style": extract_style,
    }[args.stage]()
