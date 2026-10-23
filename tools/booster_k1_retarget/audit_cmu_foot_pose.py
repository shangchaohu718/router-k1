"""Audit full native CMU foot rotation with explicit rest-axis conversion."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget.cmu_native_foot_pose import foot_targets
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--repaired',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.repaired/'manifest.json').read_text());model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model);rows=[]
    for key,clip in manifest['clips'].items():
        desired=foot_targets(clip['metadata'],clip['length']);metrics={}
        for label,root in [('original',a.original),('repaired',a.repaired)]:
            arrays=load(root,key);actual={s:[] for s in ['left','right']}
            for pos,rot,q in zip(arrays['ref_root_pos'],arrays['ref_root_rot'],arrays['ref_dof_pos']):
                data.qpos[:]=np.r_[pos,rot[[3,0,1,2]],q];mujoco.mj_forward(model,data)
                for s in actual:actual[s].append(data.xmat[model.body(s+'_ankle_roll_link').id].reshape(3,3).copy())
            errors=np.stack([np.degrees((desired[s].inv()*Rotation.from_matrix(np.asarray(actual[s]))).magnitude()) for s in actual],axis=1)
            metrics[label]={'full_foot_rotation_p95_deg':float(np.percentile(errors,95)),'mean_deg':float(errors.mean()),'max_deg':float(errors.max())}
        rows.append({'motion':key,'metrics':metrics})
    report={'motions':rows,'training_approved':False,'frame_assumption':'CMU neutral X left/Y up/Z forward to K1 X forward/Y left/Z up, proper rotation no reflection','limitations':['ASF neutral basis inferred from documented skeletal convention; source mesh not available','Not native measured foot contacts','Requires visual source validation before authoritative absolute threshold']}
    dump(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
