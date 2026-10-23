"""Measure absolute sole heights without per-clip floor alignment; not dynamics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--repaired', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.repaired / 'manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(REPO / 'assets/robots/booster_k1/K1_22dof.xml'))
    data = mujoco.MjData(model)
    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_contype[g] and model.body(int(model.geom_bodyid[g])).name in {'left_ankle_roll_link', 'right_ankle_roll_link'}]
    assert len(geoms) == 2
    rows = []
    for key in manifest['clips']:
        metrics = {}
        original = load(a.original, key)
        repaired = load(a.repaired, key)
        assert original['ref_root_pos'].shape == repaired['ref_root_pos'].shape
        for label, arrays in [('original', original), ('repaired', repaired)]:
            values = []
            for pos, rot, q in zip(arrays['ref_root_pos'], arrays['ref_root_rot'], arrays['ref_dof_pos']):
                data.qpos[:] = np.r_[pos, rot[[3, 0, 1, 2]], q]
                mujoco.mj_forward(model, data)
                values.append([data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g] for g in geoms])
            heights = np.asarray(values)
            metrics[label] = {'min_sole_z_m': float(heights.min()), 'p01_sole_z_m': float(np.percentile(heights, 1)), 'frames_any_sole_below_minus2cm': int((heights.min(axis=1) < -.02).sum())}
        root_error = np.linalg.norm(repaired['ref_root_pos'] - original['ref_root_pos'], axis=1)
        rows.append({'motion': key, 'metrics': metrics, 'root_position_change_p95_m': float(np.percentile(root_error, 95)), 'root_position_change_max_m': float(root_error.max())})
    dump(a.output, {'motions': rows, 'ground_plane_z_m': 0, 'no_height_shift_applied': True, 'physics_validation': 'not performed', 'training_approved': False})


if __name__ == '__main__':
    main()
