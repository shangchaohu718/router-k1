"""Whole-candidate source-arm and model-contact audit; never edits motion data."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import h5py

from tools.booster_k1_retarget.compare_lafan_retarget_pipelines import REPO, load
from tools.booster_k1_retarget.probe_native_position_retarget import load_cmu_fk, load_bvh_motion, _local_rotations_and_positions, _forward_kinematics, YUP_TO_ZUP
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.salvage_repaired_reference import source_positions, source_bundle, HoloSmplToGmrConverter, angular_error
from tools.booster_k1_retarget.localize_and_salvage import intervals


@lru_cache(maxsize=4)
def native_forearms(path):
    if Path(path).suffix.lower()=='.amc':
        _,points,_,_=load_cmu_fk(path)
        return [YUP_TO_ZUP.apply(points[s+'hand']-points[s+'radius']) for s in ['l','r']]
    motion=load_bvh_motion(path);lr,lp=_local_rotations_and_positions(motion);_,points=_forward_kinematics(motion,lr,lp)
    pairs=[('LeftForeArm','LeftHand'),('RightForeArm','RightHand')] if 'LeftForeArm' in points else [('LeftElbow','LeftWrist'),('RightElbow','RightWrist')]
    return [YUP_TO_ZUP.apply(points[b]-points[a])*.01 for a,b in pairs]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifests=sorted(a.snapshot.glob('*/manifest.json'));hashes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in manifests}
    worker_files=[Path(__file__),REPO/'tools/booster_k1_retarget/salvage_repaired_reference.py',REPO/'tools/booster_k1_retarget/probe_native_position_retarget.py',REPO/'tools/booster_k1_retarget/compare_lafan_retarget_pipelines.py']
    workers={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in worker_files}
    settings={'snapshot':str(a.snapshot.resolve()),'input_hashes':hashes,'worker_hashes':workers,'training_started':False,'training_approved':False,'thresholds_provisional':{'upper_arm_deg':30,'forearm_deg':30,'arm_penetration_m':.005},'does_not_certify':'native foot pose, global scene interactions or physics'}
    if (a.output/'config.json').exists() and json.loads((a.output/'config.json').read_text())!=settings:raise ValueError('Use a new output version for changed inputs or workers')
    dump(a.output/'config.json',settings)
    entries=[];manifest_cache={}
    for f in manifests:
        manifest=json.loads(f.read_text())
        manifest_cache[f]=manifest
        entries.extend((c['metadata']['source_path'],f,key,c) for key,c in manifest['clips'].items())
    entries.sort(key=lambda item:(item[0],item[3]['metadata']['canonical_frame_start']))
    model=mujoco.MjModel.from_xml_path(str(REPO/'assets/robots/booster_k1/K1_22dof.xml'));data=mujoco.MjData(model)
    converter=HoloSmplToGmrConverter(Path('/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'))
    ids={s+'_'+j:model.body(s+'_'+j+'_link').id for s in ['left','right'] for j in ['shoulder_roll','elbow_yaw']}
    geom_bodies=[model.body(int(b)).name for b in model.geom_bodyid]
    counts=Counter();frames=flagged=0
    def progress(phase,current):
        dump(a.output/'progress.json',{'status':phase,'processed_segments':sum(counts.values()),'target_segments':len(entries),'counts':dict(counts),'frames_checked':frames,'flagged_frames':flagged,'flagged_frame_hours':flagged/180000,'current':current,'training_started':False,'training_approved':False,'input_data_unchanged':True})
    progress('running',None)
    for source,f,key,clip in entries:
        if (a.output/'STOP_AFTER_CASE').exists():progress('stopped_between_segments',key);return
        for file,expected in workers.items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=expected:raise RuntimeError('Worker changed during audit')
        identity=hashlib.sha256((str(f.resolve())+'::'+key).encode()).hexdigest()[:24];out=a.output/'segments'/(identity+'.json')
        if out.exists():row=json.loads(out.read_text())
        else:
            n=clip['length'];meta=clip['metadata']
            try:
                manifest=manifest_cache[f];shard=manifest['hdf5_shards'][clip['shard']]
                with h5py.File(f.parent/shard['file'],'r') as handle:
                    arrays={name:handle[name][clip['start']:clip['start']+n] for name in ['ref_root_pos','ref_root_rot','ref_dof_pos']}
                if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite motion arrays')
                human,_,origin=source_positions(meta,n,converter)
                if Path(source).suffix.lower() in {'.amc','.bvh'}:
                    _,_,_,_,offset,_=source_bundle(source,meta.get('canonical_path'),meta.get('dataset'),converter)
                    full=native_forearms(source);t=(meta['canonical_frame_start']+np.arange(n))*float(meta['source_fps'])/50+offset
                    if t.min()<0 or t.max()>len(full[0])-1+max(1,float(meta['source_fps'])/50):raise ValueError('Native forearm correspondence out of range')
                    lower=[np.stack([np.interp(t,np.arange(len(v)),v[:,axis]) for axis in range(3)],axis=1) for v in full]
                else:lower=[human[s+'_wrist']-human[s+'_elbow'] for s in ['left','right']]
                upper_actual=np.empty((n,2,3));fore_actual=np.empty((n,2,3));depth=np.zeros(n)
                for frame in range(n):
                    data.qpos[:]=np.r_[arrays['ref_root_pos'][frame],arrays['ref_root_rot'][frame,[3,0,1,2]],arrays['ref_dof_pos'][frame]];mujoco.mj_forward(model,data)
                    for side_index,(s,sign) in enumerate([('left',1),('right',-1)]):
                        elbow=ids[s+'_elbow_yaw'];upper_actual[frame,side_index]=data.xpos[elbow]-data.xpos[ids[s+'_shoulder_roll']];fore_actual[frame,side_index]=data.xmat[elbow].reshape(3,3)@np.array([0,sign,0])
                    for contact in data.contact[:data.ncon]:
                        x,y=geom_bodies[contact.geom1],geom_bodies[contact.geom2]
                        ax=any(j in x for j in ['shoulder','elbow']);ay=any(j in y for j in ['shoulder','elbow'])
                        if not (ax or ay) or (ax and ay and x.split('_')[0]==y.split('_')[0]):continue
                        depth[frame]=max(depth[frame],-float(contact.dist))
                upper=np.stack([angular_error(upper_actual[:,i],human[s+'_elbow']-human[s+'_shoulder']) for i,s in enumerate(['left','right'])],axis=1)
                fore=np.stack([angular_error(fore_actual[:,i],lower[i]) for i in range(2)],axis=1)
                reasons={'upper_arm_over_30deg':upper.max(axis=1)>30,'native_forearm_over_30deg':fore.max(axis=1)>30,'arm_model_penetration_over_5mm':depth>.005}
                bad=np.logical_or.reduce(list(reasons.values()))
                row={'status':'needs_repair_review' if bad.any() else 'numerical_arm_checks_passed_other_checks_pending','motion':key,'source_path':source,'canonical_frame_start':meta['canonical_frame_start'],'frames':n,'flagged_frames':int(bad.sum()),'reason_intervals':{reason:intervals(mask) for reason,mask in reasons.items() if mask.any()},'upper_arm_p95_deg':float(np.percentile(upper,95)),'native_forearm_p95_deg':float(np.percentile(fore,95)),'max_arm_model_penetration_m':float(depth.max()),'source_fidelity_origin':origin,'input_manifest':str(f.resolve()),'training_approved':False}
            except Exception as exc:
                row={'status':'source_or_geometry_check_failed','motion':key,'source_path':source,'canonical_frame_start':meta['canonical_frame_start'],'frames':n,'flagged_frames':n,'error':repr(exc),'input_manifest':str(f.resolve()),'training_approved':False}
            dump(out,row)
        counts[row['status']]+=1;frames+=row['frames'];flagged+=row['flagged_frames'];progress('running',key)
    progress('complete_visual_physics_other_body_checks_pending',None)


if __name__=='__main__':main()
