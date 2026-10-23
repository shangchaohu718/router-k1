"""Diagnostic root-centered human/robot limb overlay, preserving world axes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from tools.booster_k1_retarget.audit_native_forearm_fidelity import targets
from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import source_positions, HoloSmplToGmrConverter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--repaired',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',nargs='+',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.repaired/'manifest.json').read_text());key,clip=next(iter(manifest['clips'].items()));meta=clip['metadata']
    converter=HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    human,_,origin=source_positions(meta,clip['length'],converter);forearms,_=targets(meta,clip['length'],converter)
    units=.01 if Path(meta['source_path']).suffix.lower()=='.bvh' else 1
    for side,index in [('left',0),('right',1)]:human[side+'_wrist']=human[side+'_elbow']+forearms[index]*units
    scale=meta.get('anatomical_leg_scale',.410702/float(np.median([np.linalg.norm(human[s+'_knee']-human[s+'_hip'],axis=1)+np.linalg.norm(human[s+'_ankle']-human[s+'_knee'],axis=1) for s in ['left','right']])))
    arrays=[load(a.original,key),load(a.repaired,key)];model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model)
    frames=[f for f in a.frames if 0<=f<clip['length']];fig=plt.figure(figsize=(4*len(frames),8))
    chains=[['pelvis',s+'_hip',s+'_knee',s+'_ankle'] for s in ['left','right']]+[[s+'_shoulder',s+'_elbow',s+'_wrist'] for s in ['left','right']]
    mapping={'hip':'hip_roll','knee':'knee_pitch','ankle':'ankle_pitch','shoulder':'shoulder_roll','elbow':'elbow_yaw'}
    for col,frame in enumerate(frames):
        hp={name:(value[frame]-human['pelvis'][frame])*scale for name,value in human.items()}
        for row,(label,values,color) in enumerate(zip(['ORIGINAL','REPAIRED'],arrays,['tab:red','tab:green'])):
            ax=fig.add_subplot(2,len(frames),row*len(frames)+col+1,projection='3d')
            data.qpos[:]=np.r_[values['ref_root_pos'][frame],values['ref_root_rot'][frame,[3,0,1,2]],values['ref_dof_pos'][frame]];mujoco.mj_forward(model,data)
            root=data.xpos[model.body('trunk').id].copy();rp={'pelvis':np.zeros(3)}
            for s,sign in [('left',1),('right',-1)]:
                for h,r in mapping.items():rp[s+'_'+h]=data.xpos[model.body(s+'_'+r+'_link').id].copy()-root
                elbow=model.body(s+'_elbow_yaw_link').id
                rp[s+'_wrist']=rp[s+'_elbow']+data.xmat[elbow].reshape(3,3)@np.array([0,sign*.12,0])
            for chain in chains:
                for points,style,c in [(hp,'--','gray'),(rp,'-',color)]:
                    v=np.stack([points[name] for name in chain]);ax.plot(v[:,0],v[:,1],v[:,2],style,color=c,marker='o',markersize=3)
            ax.set(xlim=(-.5,.5),ylim=(-.5,.5),zlim=(-.55,.4),title=f'{label} frame {frame}');ax.set_box_aspect((1,1,.95));ax.view_init(elev=15,azim=45)
    fig.suptitle('Gray: scaled native human; color: K1. Root-centered only; world axes unchanged. Wrist endpoint is an axis proxy; not contact/physics validation.')
    fig.tight_layout();fig.savefig(a.output/'native_limb_overlay.png',dpi=120);plt.close(fig)
    dump(a.output/'report.json',{'source_origin':origin,'motion':key,'local_frames':frames,'canonical_start':meta['canonical_frame_start'],'human_scale':scale,'no_yaw_alignment_or_mirroring':True,'root_centering':'independent translation only, no global displacement/contact inference','robot_wrist':'forearm-axis proxy 12cm, not actual hand landmark','human_vs_robot_proportions_differ':True,'training_approved':False})


if __name__=='__main__':main()
