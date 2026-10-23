"""CMU ASF/AMC forward kinematics and approximate SMPL rotation transfer.

Column-vector implementation of the Acclaim Cinv*M*C row-vector convention.
FPS is required from a cached per-trial official CMU metadata entry.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from holosmpl.converters.bvh_family.style100 import canonical_from_global

NODES = [
    "root",
    "lfemur",
    "rfemur",
    "lowerback",
    "ltibia",
    "rtibia",
    "upperback",
    "lfoot",
    "rfoot",
    "thorax",
    "ltoes",
    "rtoes",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "head",
    "lhumerus",
    "rhumerus",
    "lradius",
    "rradius",
    "lwrist",
    "rwrist",
]


def read_asf(path):
    sections = {}
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            current = line.split()[0][1:]
            sections[current] = []
        elif current:
            sections[current].append(line)
    units = dict(line.split(maxsplit=1) for line in sections["units"])
    if units.get("angle") != "deg":
        raise ValueError("ASF must use degrees")
    scale = 0.0254 / float(units["length"])
    root = dict(line.split(maxsplit=1) for line in sections["root"])
    if root["axis"] != "XYZ" or root["order"].upper() != "TX TY TZ RX RY RZ":
        raise ValueError("Unsupported root channel order")
    if np.any(np.asarray(root["orientation"].split(), float)):
        raise ValueError(
            "Nonzero root calibration requires separate validation"
        )
    bones = {}
    block = None
    for line in sections["bonedata"]:
        if line == "begin":
            block = {}
        elif line == "end":
            name = block["name"]
            axis = block["axis"].split()
            if axis[3] != "XYZ":
                raise ValueError("Unsupported bone axis order")
            dof = block.get("dof", "").split()
            if any(d not in ("rx", "ry", "rz") for d in dof):
                raise ValueError("Unsupported length/translation bone channel")
            bones[name] = {
                "direction": np.asarray(block["direction"].split(), float),
                "length": float(block["length"]) * scale,
                "basis": Rotation.from_euler(
                    "xyz", list(map(float, axis[:3])), degrees=True
                ),
                "dof": dof,
            }
            block = None
        elif block is not None and not line.startswith("("):
            key, _, value = line.partition(" ")
            block[key] = value.strip()
    parents = {}
    for line in sections["hierarchy"]:
        if line in ("begin", "end"):
            continue
        parent, *children = line.split()
        parents.update({child: parent for child in children})
    return (
        bones,
        parents,
        scale,
        np.asarray(root["position"].split(), float) * scale,
    )


def read_amc(path, bones):
    frames = []
    current = None
    text = Path(path).read_text()
    if ":DEGREES" not in text or ":RADIANS" in text:
        raise ValueError("AMC angle units must explicitly be degrees")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ":")):
            continue
        if line.isdigit():
            if int(line) != len(frames) + 1:
                raise ValueError("Non-contiguous AMC frame numbering")
            current = {}
            frames.append(current)
        else:
            name, *values = line.split()
            if current is None:
                raise ValueError("AMC channels before frame")
            current[name] = np.asarray(values, float)
    required = {
        "root": 6,
        **{n: len(b["dof"]) for n, b in bones.items() if b["dof"]},
    }
    result = {}
    for name, n in required.items():
        if any(name not in f or len(f[name]) != n for f in frames):
            raise ValueError(f"Missing/inconsistent AMC channel: {name}")
        result[name] = np.asarray([f[name] for f in frames])
    return result


def load_cmu_fk(path):
    path = Path(path)
    asf = path.with_name(path.parent.name + ".asf")
    bones, parents, scale, origin = read_asf(asf)
    values = read_amc(path, bones)
    count = len(values["root"])
    rotations = {
        "root": Rotation.from_euler("xyz", values["root"][:, 3:], degrees=True)
    }
    starts = {"root": values["root"][:, :3] * scale + origin}
    ends = {"root": starts["root"]}
    pending = set(bones)
    while pending:
        ready = [n for n in pending if parents[n] in rotations]
        if not ready:
            raise ValueError("Cyclic or missing ASF hierarchy")
        for name in ready:
            b = bones[name]
            angles = np.zeros((count, 3))
            for col, dof in enumerate(b["dof"]):
                angles[:, "xyz".index(dof[1])] = values[name][:, col]
            # CMU channels are ordered x,y,z (possibly a subset).
            if b["dof"] != sorted(b["dof"], key=lambda d: "xyz".index(d[1])):
                raise ValueError("Non-XYZ bone channels require validation")
            local = (
                b["basis"]
                * Rotation.from_euler("xyz", angles, degrees=True)
                * b["basis"].inv()
            )
            rotations[name] = rotations[parents[name]] * local
            starts[name] = ends[parents[name]]
            ends[name] = starts[name] + rotations[name].apply(
                b["direction"] * b["length"]
            )
            pending.remove(name)
    return rotations, starts, ends, bones


def metadata_for(path):
    path = Path(path)
    table = next(
        (
            p / "cmu_metadata.json"
            for p in path.parents
            if (p / "cmu_metadata.json").exists()
        ),
        None,
    )
    if table is None:
        raise ValueError("Missing official CMU FPS/description sidecar")
    row = json.loads(table.read_text())["clips"].get(path.stem)
    if row is None or row.get("fps") not in (30, 60, 120, 240):
        raise ValueError("Unknown per-trial FPS")
    return row


def classify_cmu_source(source_path):
    try:
        metadata_for(source_path)
        return {"status": "convertible_amc", "source_path": str(source_path)}
    except ValueError as exc:
        return {"status": "unsupported_missing_fps", "reason": str(exc)}


def convert_cmu_sample(source_path, *, input_root, target_fps):
    path = Path(source_path)
    meta = metadata_for(path)
    rotations, starts, ends, bones = load_cmu_fk(path)
    mapped = []
    for i, name in enumerate(NODES):
        rot = rotations[name]
        # Align SMPL rest bone directions to ASF's spread-leg/rest-shoulder pose.
        direction = None
        if i in (1, 2, 4, 5):
            direction = [0, -1, 0]
        elif i in (7, 8, 10, 11):
            direction = [0, 0, 1]
        elif i in (13, 16, 18, 20):
            direction = [1, 0, 0]
        elif i in (14, 17, 19, 21):
            direction = [-1, 0, 0]
        if direction is not None:
            alignment, _ = Rotation.align_vectors(
                [bones[name]["direction"]], [direction]
            )
            rot = rot * alignment
        mapped.append(rot)
    return canonical_from_global(
        mapped,
        starts["root"],
        meta["fps"],
        target_fps,
        {
            "dataset": "CMU",
            "source_format": "asf_amc",
            "source_path": str(path),
            "source_relative_path": path.relative_to(input_root).as_posix(),
            "subject": path.parent.name,
            "description": meta["description"],
            "fps_provenance": meta["source_url"],
            "unit_scale_policy": "0.0254/asf_length",
            "body_pose_policy": "ASF_FK_rest_bone_alignment_to_SMPL_parents",
        },
    )
