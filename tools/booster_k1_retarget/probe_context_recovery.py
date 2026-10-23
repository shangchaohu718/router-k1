"""Re-map with human-source context, then retain only the requested interval."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from tools.booster_k1_retarget.probe_native_position_retarget import canonical_arrays
from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.repair_large_candidate_pool import accepted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.case/'input/manifest.json').read_text())
    assert len(manifest['clips'])==1
    key,original=next(iter(manifest['clips'].items()))
    wanted=original['metadata']['canonical_frame_start'];end=wanted+original['length']
    root,_,_,_=canonical_arrays(original['metadata'])
    start=max(0,wanted-50);stop=min(len(root),end+50)
    view=copy.deepcopy(manifest);clip=view['clips'][key];meta=clip['metadata']
    clip['length']=stop-start
    meta.update(canonical_frame_start=start,canonical_frame_count=stop-start,num_frames=stop-start,parent_frame_start=0,parent_frame_end=stop-start)
    view['usage']='Source-only retarget request; inherited robot shard arrays do not represent the extended interval. Do not load as robot H5.'
    dump(a.output/'input/manifest.json',view)
    subprocess.run([sys.executable,'-m','tools.booster_k1_retarget.compare_lafan_retarget_pipelines','--output',str(a.output),'--worker','old_root_weight','--max-frames','10000000'],check=True)
    review=a.output/'old_root_weight/review'
    subprocess.run([sys.executable,'-m','tools.booster_k1_retarget.salvage_source_root_checked','--input',str(a.output/'old_root_weight/raw'),'--output',str(review),'--smpl-model','/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'],check=True)
    _,items=accepted(review/'report.json');clips={};shards=[]
    for name,c,shard in items:
        s=c['metadata']['canonical_frame_start'];x=max(s,wanted);y=min(s+c['length'],end)
        if y-x<50:continue
        out=copy.deepcopy(c);out.update(start=c['start']+x-s,length=y-x,shard=len(shards))
        m=out['metadata'];m.update(canonical_frame_start=x,canonical_frame_count=y-x,num_frames=y-x,parent_frame_start=m['parent_frame_start']+x-s)
        m['parent_frame_end']=m['parent_frame_start']+y-x
        m.update(context_recovered=True,training_approved=False)
        clips[name+'_target_only']=out;shards.append(shard)
    dump(a.output/'target_only/manifest.json',{'version':2,'dataset_id':a.output.name,'dof_names':manifest['dof_names'],'array_names':['ref_root_pos','ref_root_rot','ref_dof_pos'],'clips':clips,'motion_keys':list(clips),'hdf5_shards':shards,'training_approved':False})
    dump(a.output/'summary.json',{'requested_source_span':[wanted,end],'extended_source_span':[start,stop],'recovered_target_frames':sum(c['length'] for c in clips.values()),'previous_case':str(a.case.resolve()),'independent_alternative_not_yet_merged':True,'source_root_and_geometry_gates':'passed on retained patches','arm_contact_native_forearm_visual_physics':'pending','training_approved':False})


if __name__=='__main__':main()
