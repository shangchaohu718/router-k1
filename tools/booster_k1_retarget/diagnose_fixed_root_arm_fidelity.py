"""Multistart fixed-root arm-direction reachability diagnostic; no pose export."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from tools.booster_k1_retarget.audit_native_forearm_fidelity import targets
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import source_positions, HoloSmplToGmrConverter, angular_error


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.input/'manifest.json').read_text());key,clip=next(iter(manifest['clips'].items()));arrays=load(a.input,key)
    converter=HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'));human,_,_=source_positions(clip['metadata'],clip['length'],converter);lower,_=targets(clip['metadata'],clip['length'],converter)
    model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model);rng=np.random.default_rng(42);rows=[]
    for frame in a.frames:
        for side_index,(side,sign) in enumerate([('left',1),('right',-1)]):
            data.qpos[:]=np.r_[arrays['ref_root_pos'][frame],arrays['ref_root_rot'][frame,[3,0,1,2]],arrays['ref_dof_pos'][frame]]
            names=['aa'+side+'_shoulder_pitch_joint',side+'_shoulder_roll_joint',side+'_elbow_pitch_joint',side+'_elbow_yaw_joint'];ids=np.asarray([manifest['dof_names'].index(name) for name in names]);bounds=np.asarray([model.jnt_range[model.joint(name).id] for name in names]);lo,hi=bounds[:,0]+1e-6,bounds[:,1]-1e-6
            shoulder,elbow=[model.body(side+'_'+joint+'_link').id for joint in ['shoulder_roll','elbow_yaw']]
            upper=human[side+'_elbow'][frame]-human[side+'_shoulder'][frame];upper/=np.linalg.norm(upper);fore=lower[side_index][frame].copy();fore/=np.linalg.norm(fore)
            def directions(angles):
                data.qpos[7+ids]=angles;mujoco.mj_forward(model,data)
                u=data.xpos[elbow]-data.xpos[shoulder];u=u/np.linalg.norm(u);f=data.xmat[elbow].reshape(3,3)@np.array([0,sign,0]);return u,f
            def residual(angles):
                u,f=directions(angles);return np.r_[u-upper,f-fore]
            def errors(angles):
                u,f=directions(angles);return float(angular_error(u[None],upper[None])[0]),float(angular_error(f[None],fore[None])[0])
            current=arrays['ref_dof_pos'][frame,ids];seeds=[np.clip(current,lo,hi),(lo+hi)/2]+[rng.uniform(lo,hi) for _ in range(10)]
            fits=[least_squares(residual,seed,bounds=(lo,hi),max_nfev=100) for seed in seeds];best=min(fits,key=lambda fit:max(errors(fit.x)));u,f=errors(best.x)
            rows.append({'local_frame':frame,'side':side,'actual_upper_forearm_deg':errors(current),'best_found_upper_forearm_deg':[u,f],'local_direction_jacobian_rank':int(np.linalg.matrix_rank(best.jac,tol=1e-5)),'fourth_joint_direction_sensitivity_norm':float(np.linalg.norm(best.jac[:,3])),'seeds':len(seeds)})
    report={'rows':rows,'pose_trajectories_exported':False,'training_approved':False,'limitations':['No collision or temporal-speed feasibility in diagnostic','Multistart failure is not a proof of global unreachability','Root and non-arm joints fixed','This diagnostic must not be used as repaired motion data']}
    dump(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
