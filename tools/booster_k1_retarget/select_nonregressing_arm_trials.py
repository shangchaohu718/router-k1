"""Keep independent arm candidates only when both source limb metrics do not regress."""

import argparse
import copy
import json
from pathlib import Path

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    selection = json.loads((a.batch / 'selection.json').read_text())['selected']
    decisions = []
    clips, shards = {}, []
    names = None
    for before in selection:
        case = a.batch / Path(before['audit_record']).stem
        trials = []
        for trial in sorted(case.glob('forearm*')):
            report = json.loads((trial / 'review/report.json').read_text())['clips'][0]
            fore = json.loads((trial / 'native_forearm_audit.json').read_text())['motions'][0]
            contact = json.loads((trial / 'arm_contact_audit.json').read_text())['motions'][0]
            frame = json.loads((trial / 'frame_arm_audit/progress.json').read_text())
            upper = report['source_upper_arm_direction_p95_deg']
            lower = fore['metrics']['repaired']['p95_deg']
            checks = {
                'upper_nonregression': upper <= before['upper_arm_p95_deg'] + 0.1,
                'forearm_nonregression': lower <= fore['metrics']['original']['p95_deg'] + 0.1,
                'frame_arm_checks_passed': frame['flagged_frames'] == 0,
                'all_geometry_frames_retained': report['candidate_frames'] == before['frames'],
                'root_legs_unchanged': contact['root_and_legs_bitwise_unchanged'],
                'no_arm_contact_over5mm': contact['repaired_arm_contact_frames_over_5mm'] == 0,
                'speed_limit': contact['max_joint_speed_rad_s'] <= 35,
            }
            trials.append({'trial': str(trial.resolve()), 'checks': checks, 'upper_p95_deg': upper, 'forearm_p95_deg': lower, 'passed': all(checks.values())})
        good = [t for t in trials if t['passed']]
        selected = min(good, key=lambda t: t['upper_p95_deg'] + t['forearm_p95_deg']) if good else None
        decisions.append({'case': case.name, 'dataset': before['dataset'], 'frames': before['frames'], 'original_upper_p95_deg': before['upper_arm_p95_deg'], 'selected': selected, 'trials': trials})
        if not selected:
            continue
        f = Path(selected['trial']) / 'raw/manifest.json'
        m = json.loads(f.read_text())
        names = m['dof_names']
        for key, clip in m['clips'].items():
            c = copy.deepcopy(clip)
            shard = copy.deepcopy(m['hdf5_shards'][c['shard']])
            shard['file'] = str((f.parent / shard['file']).resolve())
            c['shard'] = len(shards)
            shards.append(shard)
            clips[case.name + '__' + key] = c
    dump(a.output / 'manifest.json', {'version': 2, 'dataset_id': a.output.name, 'array_names': ['ref_root_pos', 'ref_root_rot', 'ref_dof_pos'], 'dof_names': names, 'clips': clips, 'motion_keys': list(clips), 'hdf5_shards': shards, 'training_approved': False})
    dump(a.output / 'decisions.json', {'cases': decisions, 'selected_segments': len(clips), 'selected_frames': sum(c['length'] for c in clips.values()), 'metric_tolerance_deg': 0.1, 'physical_and_continuous_visual_validation': 'pending', 'hours_added': 0, 'alternative_mapping_not_new_source_frames': True, 'training_approved': False})


if __name__ == '__main__':
    main()
