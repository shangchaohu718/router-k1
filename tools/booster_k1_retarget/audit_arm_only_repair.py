"""Compare active arm contacts and forearm drift without physics rollout."""

import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--repaired',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.repaired/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'))
    data=mujoco.MjData(model)
    names=[model.body(int(body)).name for body in model.geom_bodyid]
    def arm(name):return any(term in name for term in ['shoulder','elbow'])
    def check(arrays):
        n=len(arrays['ref_dof_pos']);depth=np.zeros(n);pairs=Counter();forearm=[]
        for frame in range(n):
            data.qpos[:]=np.r_[arrays['ref_root_pos'][frame],arrays['ref_root_rot'][frame,[3,0,1,2]],arrays['ref_dof_pos'][frame]]
            mujoco.mj_forward(model,data)
            for c in data.contact[:data.ncon]:
                x,y=names[c.geom1],names[c.geom2]
                if not (arm(x) or arm(y)):continue
                if arm(x) and arm(y) and x.split('_')[0]==y.split('_')[0]:continue
                if c.dist<0:
                    depth[frame]=max(depth[frame],-float(c.dist));pairs[' / '.join(sorted([x,y]))]+=1
            forearm.append([data.xmat[model.body(side+'_elbow_yaw_link').id].reshape(3,3)@np.array([0,sign,0]) for side,sign in [('left',1),('right',-1)]])
        return depth,pairs,np.asarray(forearm)
    result=[]
    for key in manifest['clips']:
        old,new=load(a.original,key),load(a.repaired,key)
        assert np.array_equal(old['ref_root_pos'],new['ref_root_pos'])
        assert np.array_equal(old['ref_root_rot'],new['ref_root_rot'])
        assert np.array_equal(old['ref_dof_pos'][:,10:],new['ref_dof_pos'][:,10:])
        before,bpairs,bfore=check(old);after,apairs,afore=check(new)
        error=np.degrees(np.arccos(np.clip(np.sum(bfore*afore,axis=-1),-1,1)))
        result.append({'motion':key,'frames':len(after),'root_and_legs_bitwise_unchanged':True,'original_arm_contact_frames_over_5mm':int(np.sum(before>.005)),'repaired_arm_contact_frames_over_5mm':int(np.sum(after>.005)),'new_or_worsened_arm_contact_frames_over_5mm':np.flatnonzero(after>np.maximum(.005,before+.005)).tolist(),'original_max_arm_penetration_m':float(before.max()),'repaired_max_arm_penetration_m':float(after.max()),'original_pairs':dict(bpairs),'repaired_pairs':dict(apairs),'forearm_direction_change_p95_deg':float(np.percentile(error,95)),'forearm_direction_change_max_deg':float(error.max()),'max_joint_speed_rad_s':float(np.abs(np.diff(new['ref_dof_pos'],axis=0)*50).max())})
    report={'motions':result,'training_approved':False,'limitations':['Active model contact geometries only; no exact mesh-surface clearance certification','Same-arm adjacent/internal pairs ignored','Forearm measured relative to old mapped motion, not native human','No physical rollout']}
    dump(a.output,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
