"""Independent CMU ankle pilots, preserving the original bulk outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.booster_k1_retarget.recover_short_reference_candidates import dump
from tools.booster_k1_retarget.repair_large_candidate_pool import accepted


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=5);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    candidates=[];seen=set()
    for f in a.root.glob('cmu_ankle_constraint_pilot_v2/*/experiment_settings.json'):
        old=Path(json.loads(f.read_text())['source_case']);m=json.loads((old/'input/manifest.json').read_text());seen.add(next(iter(m['clips'].values()))['metadata']['source_path'])
    for f in a.root.glob('cases/*/status.json'):
        s=json.loads(f.read_text());report=f.parent/'old_root_weight/review/report.json'
        if s['source']!='CMU' or s['status']!='no_checked_patch_recovered' or not report.exists():continue
        row=json.loads(report.read_text())['clips'][0]
        if row['source_error'] is None and row['frames']>=200 and 'ankle_limit_saturation_for_0p5s' in row['reason_intervals']:
            candidates.append((row['frames'],f.parent))
    cases=[]
    for _,case in sorted(candidates,reverse=True):
        m=json.loads((case/'input/manifest.json').read_text());source=next(iter(m['clips'].values()))['metadata']['source_path']
        if source in seen:continue
        seen.add(source);cases.append(case)
        if len(cases)>=a.limit:break
    dump(a.output/'selection.json',{'cases':[str(c.resolve()) for c in cases],'training_approved':False,'reason':'independent captures with prior ankle saturation, not just repeated windows'})
    results=[]
    for case in cases:
        out=a.output/'cases'/case.name
        commands=[
            ['probe_ankle_orientation_constraint','--case',str(case),'--output',str(out),'--foot-orientation-cost','0','--ankle-posture-cost','1'],
            ['salvage_source_root_checked','--input',str(out/'old_root_weight/raw'),'--output',str(out/'review'),'--smpl-model','/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl'],
            ['audit_foot_direction_contact','--original',str(case/'old_root_weight/raw'),'--repaired',str(out/'old_root_weight/raw'),'--output',str(out/'foot_audit.json')],
        ]
        code=0
        with (a.output/'run.log').open('a') as log:
            for module,*arguments in commands:
                code=subprocess.run([sys.executable,'-m','tools.booster_k1_retarget.'+module,*arguments],stdout=log,stderr=subprocess.STDOUT).returncode
                if code:break
        result={'case':str(case.resolve()),'exit_code':code}
        if not code:
            row,items=accepted(out/'review/report.json');before=json.loads((case/'old_root_weight/review/report.json').read_text())['clips'][0];foot=json.loads((out/'foot_audit.json').read_text())['motions'][0]['metrics']
            result.update(retained_frames_before_global_dedup=sum(c['length'] for _,c,_ in items),leg_p95_deg=row['native_leg_direction_p95_deg'],upper_arm_p95_deg=row['source_upper_arm_direction_p95_deg'],foot_metrics=foot,relative_fidelity_review_passed=row['native_leg_direction_p95_deg']<=before['native_leg_direction_p95_deg']+5 and row['source_upper_arm_direction_p95_deg']<=before['source_upper_arm_direction_p95_deg']+5 and foot['repaired']['foot_ray_p95_deg']<=foot['original']['foot_ray_p95_deg']+5 and foot['repaired']['source_grounded_robot_over_8cm_samples']<=foot['original']['source_grounded_robot_over_8cm_samples'])
        results.append(result)
        dump(a.output/'progress.json',{'processed':len(results),'target':len(cases),'results':results,'training_approved':False,'no_main_counts_changed':True})
    dump(a.output/'summary.json',{'cases':len(cases),'results':results,'training_approved':False,'visual_physics_arm_contacts_native_forearm_global_dedup':'pending'})


if __name__=='__main__':main()
