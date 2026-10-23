"""Separate nonregressing numerical context-arm candidates from unresolved trials."""

import argparse
import copy
import json
from pathlib import Path

from tools.booster_k1_retarget.repair_large_candidate_pool import accepted
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    clips={};shards=[];decisions=[];names=None
    for case in sorted(a.root.iterdir()):
        if not case.is_dir():continue
        choices=[]
        for trial in sorted(case.glob('forearm*')):
            paths=[trial/'review/report.json',trial/'arm_contact_audit.json',trial/'native_forearm_audit.json']
            if not all(f.exists() for f in paths):continue
            row,items=accepted(paths[0]);contact=json.loads(paths[1].read_text())['motions'][0];forearm=json.loads(paths[2].read_text())['motions'][0]
            upper=row['source_upper_arm_direction_p95_deg'];lower=forearm['metrics']['repaired']['p95_deg'];old_lower=forearm['metrics']['original']['p95_deg']
            ok=bool(items) and contact['root_and_legs_bitwise_unchanged'] and contact['repaired_arm_contact_frames_over_5mm']==0 and contact['max_joint_speed_rad_s']<=35 and upper<=10 and lower<=5 and lower<=old_lower+.1
            choices.append((ok,upper+lower,trial,items,upper,lower))
        good=[choice for choice in choices if choice[0]]
        if not good:
            decisions.append({'case':case.name,'status':'separate_unresolved_review','trials':[str(c[2].resolve()) for c in choices]});continue
        _,_,trial,items,upper,lower=min(good,key=lambda choice:choice[1])
        names=json.loads((trial/'raw/manifest.json').read_text())['dof_names']
        for key,c,shard in items:
            out=copy.deepcopy(c);out['shard']=len(shards);out['metadata'].update(training_approved=False,additional_arm_gate='contacts<=5mm,upper_p95<=10deg,source_forearm_p95<=5deg_and_nonregression_provisional')
            clips[case.name+'__'+key]=out;shards.append(shard)
        decisions.append({'case':case.name,'selected':str(trial.resolve()),'upper_p95_deg':upper,'native_forearm_p95_deg':lower,'status':'numerical_candidate_visual_physics_pending'})
    dump(a.output/'manifest.json',{'version':2,'dataset_id':a.output.name,'array_names':['ref_root_pos','ref_root_rot','ref_dof_pos'],'dof_names':names,'clips':clips,'motion_keys':list(clips),'hdf5_shards':shards,'training_approved':False})
    dump(a.output/'nan_npz_paths.json',[])
    report={'segments':len(clips),'frames':sum(c['length'] for c in clips.values()),'decisions':decisions,'training_approved':False,'visual_and_physics_validation':'pending','global_dedup_against_main_required':True,'thresholds':'provisional, not a universal training-ready standard','originals_preserved':True}
    dump(a.output/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
