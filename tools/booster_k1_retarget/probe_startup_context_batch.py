"""Small independent context-recovery batch for startup-only failed intervals."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=20)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    pilots=json.loads((a.root/'source_pilots.json').read_text());cases=[]
    for f in sorted(a.root.glob('cases/*/status.json')):
        status=json.loads(f.read_text())
        if status['status']!='no_checked_patch_recovered' or 'old_root_weight' not in pilots[status['source']]['qualified_profiles']:continue
        report=f.parent/'old_root_weight/review/report.json'
        if not report.exists():continue
        row=json.loads(report.read_text())['clips'][0]
        spans=[span for values in row['reason_intervals'].values() for span in values]
        if row['frames']<=200 and row['source_error'] is None and spans and all(end<=7 for _,end in spans):cases.append(f.parent)
    cases=cases[:a.limit];results=[]
    dump(a.output/'selection.json',{'cases':[str(c.resolve()) for c in cases],'selection':'startup-only reasons within first 7 frames, <=200-frame input, source-qualified old_root_weight','training_approved':False})
    for case in cases:
        out=a.output/'cases'/case.name
        with (a.output/'run.log').open('a') as log:
            result=subprocess.run([sys.executable,'-m','tools.booster_k1_retarget.probe_context_recovery','--case',str(case),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
        row={'case':str(case.resolve()),'exit_code':result.returncode}
        if result.returncode==0:row.update(json.loads((out/'summary.json').read_text()))
        results.append(row)
        dump(a.output/'progress.json',{'processed':len(results),'target':len(cases),'results':results,'recovered_target_frames_before_global_dedup':sum(r.get('recovered_target_frames',0) for r in results),'not_added_to_main_counts':True,'training_started':False,'training_approved':False})
    dump(a.output/'summary.json',{'cases':len(results),'recovered_cases':sum(r.get('recovered_target_frames',0)>0 for r in results),'raw_target_frames_before_global_dedup':sum(r.get('recovered_target_frames',0) for r in results),'unresolved':[r['case'] for r in results if not r.get('recovered_target_frames',0)],'training_approved':False,'next':'Audit source intervals against protected baseline and completed bulk patches, check arms/forearms and visuals, then merge unique candidates only'})


if __name__=='__main__':main()
