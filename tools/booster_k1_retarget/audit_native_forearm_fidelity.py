"""Compare actual human forearm directions with K1 forearm axes."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.probe_native_position_retarget import canonical_arrays, load_bvh_motion, _local_rotations_and_positions, _forward_kinematics, YUP_TO_ZUP, load_cmu_fk
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import source_positions, HoloSmplToGmrConverter, angular_error


def targets(meta,n,converter):
    path=Path(meta['source_path']);start=meta['canonical_frame_start']
    if path.suffix.lower()=='.bvh':
        motion=load_bvh_motion(path)
        lr,lp=_local_rotations_and_positions(motion)
        _,points=_forward_kinematics(motion,lr,lp)
        _,_,_,cm=canonical_arrays(meta)
        t=(start+np.arange(n))*float(meta['source_fps'])/50+cm.get('trim_start',0)
        pairs=[('LeftForeArm','LeftHand'),('RightForeArm','RightHand')] if 'LeftForeArm' in points else [('LeftElbow','LeftWrist'),('RightElbow','RightWrist')]
        vectors=[YUP_TO_ZUP.apply(points[b]-points[a]) for a,b in pairs]
    elif path.suffix.lower()=='.amc':
        _,points,_,_=load_cmu_fk(path)
        t=(start+np.arange(n))*float(meta['source_fps'])/50
        vectors=[YUP_TO_ZUP.apply(points[side+'hand']-points[side+'radius']) for side in ['l','r']]
    else:
        human,_,origin=source_positions(meta,n,converter)
        return [human[side+'_wrist']-human[side+'_elbow'] for side in ['left','right']],origin
    if min(t)<0 or max(t)>len(vectors[0])-1+max(1,float(meta['source_fps'])/50):raise ValueError('Source correspondence outside bounds')
    return [np.stack([np.interp(t,np.arange(len(v)),v[:,axis]) for axis in range(3)],axis=1) for v in vectors],'native_forearm_FK'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--repaired',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.repaired/'manifest.json').read_text())
    converter=HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model);rows=[]
    for key,clip in manifest['clips'].items():
        human,origin=targets(clip['metadata'],clip['length'],converter)
        metrics={}
        for label,root in [('original',a.original),('repaired',a.repaired)]:
            arrays=load(root,key);robot=[]
            for pos,rot,q in zip(arrays['ref_root_pos'],arrays['ref_root_rot'],arrays['ref_dof_pos']):
                data.qpos[:]=np.r_[pos,rot[[3,0,1,2]],q];mujoco.mj_forward(model,data)
                robot.append([data.xmat[model.body(side+'_elbow_yaw_link').id].reshape(3,3)@np.array([0,sign,0]) for side,sign in [('left',1),('right',-1)]])
            robot=np.asarray(robot)
            errors=np.stack([angular_error(robot[:,side],human[side]) for side in range(2)],axis=1)
            metrics[label]={'p95_deg':float(np.percentile(errors,95)),'mean_deg':float(errors.mean()),'max_deg':float(errors.max())}
        rows.append({'motion':key,'frames':clip['length'],'source_origin':origin,'metrics':metrics,'p95_regression_deg':metrics['repaired']['p95_deg']-metrics['original']['p95_deg']})
    report={'motions':rows,'training_approved':False,'measurement':'human elbow-to-wrist direction versus K1 forearm local signed Y axis, not wrist/hand rotation'}
    dump(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
