"""Flag limb regressions between completed alternative mappings; read-only inputs."""

import argparse
import json
from pathlib import Path
import time

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    compared = 0
    flagged = []
    missing = []
    metrics = ['native_leg_direction_p95_deg', 'source_upper_arm_direction_p95_deg']
    for file in sorted(args.root.glob('cases/*/status.json')):
        status = json.loads(file.read_text())
        if status['status'] != 'repaired_patches_retained':
            continue
        reports = {}
        for path in sorted(file.parent.glob('*/review/report.json')):
            reports[path.parent.parent.name] = json.loads(path.read_text())['clips'][0]
        selected = status['selected_profile']
        if len(reports) < 2:
            missing.append(status['motion'])
            continue
        compared += 1
        for alternative, report in reports.items():
            if alternative == selected:
                continue
            selected_report = reports[selected]
            regression = {metric: selected_report[metric] - report[metric] for metric in metrics if selected_report[metric] - report[metric] > 5}
            if regression:
                flagged.append({'motion': status['motion'], 'source': status['source'], 'selected': selected, 'alternative': alternative, 'regression_deg': regression, 'selected_metrics': {metric: selected_report[metric] for metric in metrics}, 'alternative_metrics': {metric: report[metric] for metric in metrics}, 'case': str(file.parent.resolve()), 'action': 'review source-matched retained intervals before release; do not blindly switch full clip', 'scope': 'full reviewed input segment, not isolated retained patch'})
    report = {'compared_cases': compared, 'flagged_comparisons': len(flagged), 'flags': flagged, 'single_profile_cases': missing, 'threshold_deg_provisional': 5, 'no_active_outputs_modified': True, 'training_approved': False, 'limitation': 'relative nonregression is not absolute correctness; metrics cover whole segment, so patch-specific confirmation is needed'}
    output = args.root / 'audits' / ('tradeoffs_' + str(time.time_ns()) + '.json')
    dump(output, report)
    print(json.dumps({'compared_cases': compared, 'flagged_comparisons': len(flagged), 'single_profile_cases': len(missing), 'report': str(output)}, indent=2))


if __name__ == '__main__':
    main()
