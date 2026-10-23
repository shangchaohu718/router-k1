"""Create an independent checked view of one source interval, without stitching."""

import argparse
import copy
from pathlib import Path

from tools.booster_k1_retarget.repair_large_candidate_pool import accepted
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    _,items=accepted(a.review/'report.json');clips={};shards=[];names=None
    import json
    names=json.loads((a.review/'01_kinematic_candidates/manifest.json').read_text())['dof_names']
    for key,c,shard in items:
        s=c['metadata']['canonical_frame_start'];x=max(a.start,s);y=min(a.end,s+c['length'])
        if y-x<50:continue
        out=copy.deepcopy(c);out.update(start=c['start']+x-s,length=y-x,shard=len(shards))
        meta=out['metadata'];meta.update(canonical_frame_start=x,canonical_frame_count=y-x,num_frames=y-x,parent_frame_start=meta['parent_frame_start']+x-s)
        meta['parent_frame_end']=meta['parent_frame_start']+y-x
        meta.update(training_approved=False,interval_view_not_stitched=True)
        clips[key+f'_source_{x}_{y}']=out;shards.append(shard)
    dump(a.output/'manifest.json',{'version':2,'dataset_id':a.output.name,'array_names':['ref_root_pos','ref_root_rot','ref_dof_pos'],'dof_names':names,'clips':clips,'motion_keys':list(clips),'hdf5_shards':shards,'training_approved':False})
    dump(a.output/'nan_npz_paths.json',[])
    dump(a.output/'summary.json',{'source_interval':[a.start,a.end],'frames':sum(c['length'] for c in clips.values()),'segments':len(clips),'global_dedup':'pending','visual_and_physics':'pending','training_approved':False})


if __name__=='__main__':main()
