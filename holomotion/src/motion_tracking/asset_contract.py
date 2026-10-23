"""Content identities for the validation task's local K1 XML/URDF/STLs."""

from __future__ import annotations
import glob
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VARIANT_KEYS = (
    "urdf_variants",
    "urdf_variant_glob",
    "urdf_variant_manifest",
)


def _local_path(value, base, *, directory=False):
    if (
        not isinstance(value, str)
        or not value.strip()
        or urlsplit(value).scheme
        or value.startswith(("//", "~"))
        or glob.has_magic(value)
        or "${" in value
        or "\x00" in value
    ):
        raise ValueError(f"Asset requires an explicit local path: {value!r}")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing registered asset path: {path}")
    return path


def _mesh_paths(path):
    try:
        document = ElementTree.fromstring(path.read_bytes())
    except ElementTree.ParseError as error:
        raise ValueError(f"Invalid robot asset XML: {path}") from error
    if document.tag not in {"mujoco", "robot"}:
        raise ValueError(f"Unsupported robot asset root: {document.tag}")
    meshes = document.findall(".//mesh")
    for element in document.iter():
        if (
            element.tag in {"include", "attach"}
            or "}" in element.tag
            or (
                element.tag != "mesh"
                and any(k in element.attrib for k in ("file", "filename"))
            )
        ):
            raise ValueError("Unsupported robot asset include or dependency")
    base = path.parent
    attribute = "filename"
    if document.tag == "mujoco":
        attribute = "file"
        compilers = document.findall("compiler")
        if len(compilers) > 1:
            raise ValueError("Multiple MuJoCo compiler elements unsupported")
        if compilers:
            compiler = compilers[0]
            if (
                compiler.get("assetdir")
                or compiler.get("strippath", "false") != "false"
            ):
                raise ValueError("Unsupported MuJoCo asset directory rules")
            meshdir = compiler.get("meshdir")
            if meshdir:
                base = _local_path(meshdir, path.parent, directory=True)
    result = set()
    for mesh in meshes:
        resolved = _local_path(mesh.get(attribute), base)
        if resolved.suffix.lower() != ".stl":
            raise ValueError(
                "K1 asset contract currently supports STL meshes only"
            )
        result.add(resolved)
    return result


def asset_content_manifest(robot_asset, motion):
    """Hash explicit robot, reference and FK sources plus their local meshes.

    Top-level relative file paths resolve from the repository, matching the
    current scene/FK loaders. ``asset_root`` and ``assetRoot`` are validated
    directories, not additional filename prefixes. Mesh references resolve
    from their containing XML/URDF (plus MuJoCo's compiler meshdir).
    Includes, variants, external URIs and non-STL dependencies fail closed.
    """
    if not isinstance(robot_asset, dict) or not isinstance(motion, dict):
        raise ValueError("Asset contract requires resolved dictionaries")
    motion_asset = motion.get("asset")
    if not isinstance(motion_asset, dict):
        raise ValueError("Missing resolved motion asset configuration")
    for config in (robot_asset, motion_asset):
        if any(config.get(key) for key in _VARIANT_KEYS):
            raise ValueError("URDF variants/glob/manifest are unsupported")
    _local_path(
        robot_asset.get("asset_root", "./"), _REPO_ROOT, directory=True
    )
    _local_path(
        motion_asset.get("assetRoot", "./"), _REPO_ROOT, directory=True
    )
    sources = {
        _local_path(value, _REPO_ROOT)
        for value in (
            robot_asset.get("urdf_file"),
            robot_asset.get("assetFileName"),
            motion_asset.get("assetFileName"),
            motion.get("fk_robot_file_path"),
        )
    }
    paths = set(sources)
    for path in sources:
        if path.suffix.lower() not in {".xml", ".urdf"}:
            raise ValueError("K1 asset sources must be XML or URDF")
        paths.update(_mesh_paths(path))
    manifest = []
    for path in sorted(paths):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        manifest.append(
            {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}
        )
    return manifest
