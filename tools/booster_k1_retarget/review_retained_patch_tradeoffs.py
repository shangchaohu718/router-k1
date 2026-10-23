"""Compare native limb fidelity on identical retained source frames only."""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import (
    HoloSmplToGmrConverter, REPO, angular_error, source_positions,
)


def load(folder, clip, manifest, start, n):
    offset = start - clip['metadata']['canonical_frame_start']
    assert 0 <= offset and offset + n <= clip['length']
    shard = manifest['hdf5_shards'][clip['shard']]
    with h5py.File(folder / shard['file'], 'r') as handle:
        s = clip['start'] + offset
        arrays = [handle[name][s:s+n] for name in ['ref_root_pos', 'ref_root_rot', 'ref_dof_pos']]
    return np.c_[arrays[0], arrays[1][:, [3, 0, 1, 2]], arrays[2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flags', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    flags = json.loads(args.flags.read_text())['flags']
    converter = HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    model = mujoco.MjModel.from_xml_path(str(REPO / 'assets/robots/booster_k1/K1_22dof.xml'))
    data = mujoco.MjData(model)
    names = [side + '_' + joint for side in ['left', 'right'] for joint in ['hip_roll', 'knee_pitch', 'ankle_pitch', 'shoulder_roll', 'elbow_yaw']]
    ids = {name: model.body(name + '_link').id for name in names}
    results = []
    for index, flag in enumerate(flags):
        root = Path(flag['case'])
        patches = json.loads((root / 'patches/manifest.json').read_text())
        for patch_key, patch in patches['clips'].items():
            start, n = patch['metadata']['canonical_frame_start'], patch['length']
            human, _, origin = source_positions(patch['metadata'], n, converter)
            metrics = {}
            for profile in [flag['selected'], flag['alternative']]:
                folder = root / profile / 'raw'
                manifest = json.loads((folder / 'manifest.json').read_text())
                clip = manifest['clips'][flag['motion']]
                poses = load(folder, clip, manifest, start, n)
                robot = {name: np.empty((n, 3)) for name in names}
                for frame, pose in enumerate(poses):
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    for name, body_id in ids.items():
                        robot[name][frame] = data.xpos[body_id]
                legs = np.stack([angular_error(robot[side+'_'+b]-robot[side+'_'+a], human[side+'_'+hb]-human[side+'_'+ha]) for side in ['left','right'] for a,b,ha,hb in [('hip_roll','knee_pitch','hip','knee'),('knee_pitch','ankle_pitch','knee','ankle')]], axis=1)
                arms = np.stack([angular_error(robot[side+'_elbow_yaw']-robot[side+'_shoulder_roll'], human[side+'_elbow']-human[side+'_shoulder']) for side in ['left','right']], axis=1)
                metrics[profile] = {'leg_p95_deg': float(np.percentile(legs,95)), 'arm_p95_deg': float(np.percentile(arms,95))}
            selected, alternative = metrics[flag['selected']], metrics[flag['alternative']]
            regressions = {key:selected[key]-alternative[key] for key in selected if selected[key]-alternative[key]>5}
            row = {'case': str(root), 'patch': patch_key, 'source_frame_start': start, 'frames': n, 'origin': origin, 'selected': flag['selected'], 'alternative': flag['alternative'], 'metrics': metrics, 'regressions_over_5deg': regressions, 'requires_further_repair': bool(regressions), 'training_approved': False}
            results.append(row)
        dump(args.output / 'progress.json', {'processed_cases': index+1, 'target_cases': len(flags), 'patches': len(results), 'training_started': False})
        dump(args.output / 'results.json', {'patches': results, 'training_approved': False, 'no_original_or_active_outputs_modified': True, 'limitation': 'Alternative has not necessarily passed contact/collision gates on these exact frames; do not automatically replace.'})
    dump(args.output / 'summary.json', {'cases': len(flags), 'patches':len(results), 'flagged_patches': sum(row['requires_further_repair'] for row in results), 'flagged_frames': sum(row['frames'] for row in results if row['requires_further_repair']), 'training_approved':False})


if __name__ == '__main__':
    main()
