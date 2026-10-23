"""Experimental bounded arm-only repair, preserving root and legs bitwise."""

import argparse
import copy
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import REPO, HoloSmplToGmrConverter, source_positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--collision-weight', type=float, default=0)
    p.add_argument('--multi-start', action='store_true')
    p.add_argument('--native-forearm', action='store_true')
    p.add_argument('--forearm-weight', type=float, default=.5)
    p.add_argument('--reverse-time-solve', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = json.loads((a.input / 'manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(REPO / 'assets/robots/booster_k1/K1_22dof.xml'))
    data = mujoco.MjData(model)
    converter = HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    writer = RobotH5ShardWriter(a.output / 'shards/k1_000000.h5', dof_pos_dim=22)
    clips, reports = {}, []
    for key, clip in m['clips'].items():
        with h5py.File(a.input / m['hdf5_shards'][clip['shard']]['file'], 'r') as f:
            arrays = {name: f[name][clip['start']:clip['start']+clip['length']] for name in ['ref_root_pos','ref_root_rot','ref_dof_pos']}
        q = arrays['ref_dof_pos'].copy()
        human, _, origin = source_positions(clip['metadata'], clip['length'], converter)
        source_forearms = None
        if a.native_forearm:
            from tools.booster_k1_retarget.audit_native_forearm_fidelity import targets
            source_forearms, _ = targets(clip['metadata'], clip['length'], converter)
        for side_index, (side, sign) in enumerate([('left',1),('right',-1)]):
            names = ['aa'+side+'_shoulder_pitch_joint', side+'_shoulder_roll_joint', side+'_elbow_pitch_joint', side+'_elbow_yaw_joint']
            ids = [m['dof_names'].index(name) for name in names]
            limits = np.asarray([model.jnt_range[model.joint(name).id] for name in names])
            shoulder, elbow = [model.body(side+'_'+name+'_link').id for name in ['shoulder_roll','elbow_yaw']]
            previous = None
            order = range(clip['length']-1,-1,-1) if a.reverse_time_solve else range(clip['length'])
            for frame in order:
                data.qpos[:] = np.r_[arrays['ref_root_pos'][frame], arrays['ref_root_rot'][frame,[3,0,1,2]], q[frame]]
                mujoco.mj_forward(model,data)
                original = q[frame, ids].copy()
                forearm = data.xmat[elbow].reshape(3,3) @ np.array([0,sign,0])
                if source_forearms is not None:
                    forearm = source_forearms[side_index][frame].copy()
                    if np.linalg.norm(forearm)<1e-6:
                        raise ValueError('Degenerate source forearm')
                    forearm /= np.linalg.norm(forearm)
                target = human[side+'_elbow'][frame]-human[side+'_shoulder'][frame]
                if np.linalg.norm(target)<1e-6:
                    raise ValueError('Degenerate source upper arm')
                target /= np.linalg.norm(target)
                lo, hi = limits[:,0]+1e-6, limits[:,1]-1e-6
                if previous is not None:
                    lo, hi = np.maximum(lo,previous-.6), np.minimum(hi,previous+.6)
                def residual(angles):
                    data.qpos[7+np.asarray(ids)] = angles
                    mujoco.mj_forward(model,data)
                    upper = data.xpos[elbow]-data.xpos[shoulder]
                    upper /= np.linalg.norm(upper)
                    lower = data.xmat[elbow].reshape(3,3) @ np.array([0,sign,0])
                    penetration = np.zeros(3)
                    if a.collision_weight:
                        for contact in data.contact[:data.ncon]:
                            x,y = [model.body(int(model.geom_bodyid[g])).name for g in [contact.geom1,contact.geom2]]
                            def own_arm(name):
                                return name.startswith(side+'_') and any(part in name for part in ['shoulder','elbow'])
                            if not (own_arm(x) or own_arm(y)) or (own_arm(x) and own_arm(y)):
                                continue
                            other = y if own_arm(x) else x
                            category = 0 if other=='trunk' else (1 if any(part in other for part in ['hip','knee','ankle']) else 2)
                            penetration[category] = max(penetration[category], -float(contact.dist)-.002)
                    return np.r_[upper-target, a.forearm_weight*(lower-forearm), .01*(angles-original), a.collision_weight*penetration]
                seeds = [np.clip(original,lo,hi)]
                if a.multi_start:
                    seeds.extend([(lo+hi)/2, np.clip(np.zeros(4),lo,hi)])
                fits = [least_squares(residual,seed,bounds=(lo,hi),max_nfev=120 if a.multi_start else 60) for seed in seeds]
                fit = min(fits,key=lambda candidate: float(np.sum(candidate.fun**2)))
                q[frame,ids] = fit.x
                previous = fit.x
        assert np.array_equal(q[:,10:],arrays['ref_dof_pos'][:,10:])
        arrays['ref_dof_pos'] = q
        meta = {**clip['metadata'],'arm_repair':'bounded_4dof_fixed_root_and_legs_pilot','source_fidelity_origin':origin,'training_approved':False}
        start,n = writer.append_motion(motion_id=len(clips),arrays=arrays,metadata_json=json.dumps(meta))
        out = copy.deepcopy(clip);out.update(start=start,length=n,shard=0,clip_idx=len(clips),metadata=meta);clips[key]=out
        reports.append({'motion':key,'frames':n,'root_and_legs_unchanged':True,'forearm_target':'source elbow-to-wrist direction' if a.native_forearm else 'previous mapped forearm direction','forearm_weight':a.forearm_weight,'collision_weight':a.collision_weight,'multi_start':a.multi_start,'reverse_optimization_order_only':a.reverse_time_solve,'output_frame_order':'unchanged chronological','collision_and_contact_validation':'pending'})
    writer.finalize()
    dump(a.output/'manifest.json',{**m,'clips':clips,'motion_keys':list(clips),'hdf5_shards':[{'file':'shards/k1_000000.h5','num_clips':len(clips),'num_frames':sum(c['length'] for c in clips.values())}],'training_approved':False})
    dump(a.output/'pilot_report.json',{'motions':reports,'training_approved':False})


if __name__ == '__main__':
    main()
