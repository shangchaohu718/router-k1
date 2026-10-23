"""Source foot-ray and approximate grounded-foot clearance audit, not dynamics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import source_positions, HoloSmplToGmrConverter, angular_error


def foot_directions(data, bodies):
    """Own the sampled values; MuJoCo updates xmat in-place next frame."""
    return np.stack([data.xmat[body].reshape(3,3)[:,0] for body in bodies])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--repaired',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.repaired/'manifest.json').read_text())
    converter=HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model)
    bodies=[model.body(side+'_ankle_roll_link').id for side in ['left','right']]
    geoms=[next(g for g in range(model.ngeom) if model.geom_bodyid[g]==body and model.geom_contype[g] and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX) for body in bodies]
    rows=[]
    for key,clip in manifest['clips'].items():
        n=clip['length'];human,ground,origin=source_positions(clip['metadata'],n,converter)
        scale=clip['metadata'].get('anatomical_leg_scale',.410702/float(np.median([np.linalg.norm(human[s+'_knee']-human[s+'_hip'],axis=1)+np.linalg.norm(human[s+'_ankle']-human[s+'_knee'],axis=1) for s in ['left','right']])))
        targets=[human[s+'_foot']-human[s+'_ankle'] for s in ['left','right']]
        grounded=np.stack([(np.minimum(human[s+'_foot'][:,2],human[s+'_ankle'][:,2])-ground)*scale<.02 for s in ['left','right']],axis=1)
        metrics={}
        for label,root in [('original',a.original),('repaired',a.repaired)]:
            arrays=load(root,key);directions=[];heights=[]
            for pos,rot,q in zip(arrays['ref_root_pos'],arrays['ref_root_rot'],arrays['ref_dof_pos']):
                data.qpos[:]=np.r_[pos,rot[[3,0,1,2]],q];mujoco.mj_forward(model,data)
                directions.append(foot_directions(data,bodies))
                heights.append([data.geom_xpos[g,2]-np.abs(data.geom_xmat[g].reshape(3,3)[2])@model.geom_size[g] for g in geoms])
            directions=np.asarray(directions);heights=np.asarray(heights);shift=float(np.percentile(heights.min(axis=1),1));heights-=shift
            errors=np.stack([angular_error(directions[:,side],targets[side]) for side in range(2)],axis=1)
            contact_errors=heights[grounded]
            metrics[label]={'foot_ray_p95_deg':float(np.percentile(errors,95)),'foot_ray_grounded_p95_deg':float(np.percentile(errors[grounded],95)) if grounded.any() else None,'source_grounded_robot_clearance_p95_m':float(np.percentile(contact_errors,95)) if len(contact_errors) else None,'source_grounded_robot_over_5cm_samples':int(np.sum(contact_errors>.05)),'source_grounded_robot_over_8cm_samples':int(np.sum(contact_errors>.08)),'constant_floor_shift_m':shift}
        rows.append({'motion':key,'frames':n,'source_origin':origin,'source_grounded_foot_samples':int(grounded.sum()),'metrics':metrics})
    report={'motions':rows,'training_approved':False,'limitations':['Human ankle-to-toe ray vs robot foot +X; bone ray is not complete foot orientation','Source ground estimated from FK, not measured contacts','Constant whole-clip floor shift only','Grounded samples are per foot, not unique frames','No physical feasibility result']}
    dump(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
