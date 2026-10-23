"""Index exact repair and protected spans without modifying or rejecting motion data."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.booster_k1_retarget.localize_and_salvage import intervals, padded
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audits', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = []
    total = flagged = protected = short = padded_count = 0
    identities = set()
    for audit in a.audits:
        state = json.loads((audit / 'progress.json').read_text())
        assert state['status'].startswith('complete')
        for f in sorted((audit / 'segments').glob('*.json')):
            r = json.loads(f.read_text())
            n = r['frames']
            identity = (r['source_path'], r['canonical_frame_start'], n)
            assert identity not in identities, identity
            identities.add(identity)
            bad = np.zeros(n, dtype=bool)
            for spans in r.get('reason_intervals', {}).values():
                for start, end in spans:
                    assert 0 <= start < end <= n
                    bad[start:end] = True
            if r.get('error'):
                bad[:] = True
            assert int(bad.sum()) == r['flagged_frames']
            expanded = padded(bad, 5)
            normal = intervals(~expanded)
            long = [[s, e] for s, e in normal if e - s >= 50]
            fragments = [[s, e] for s, e in normal if e - s < 50]
            base = r['canonical_frame_start']
            rows.append({
                'audit_record': str(f.resolve()), 'input_manifest': r['input_manifest'],
                'motion': r['motion'], 'source_path': r['source_path'],
                'canonical_start': base, 'frames': n,
                'repair_context_source_intervals': [[base+s, base+e] for s, e in intervals(expanded)],
                'protected_normal_source_intervals': [[base+s, base+e] for s, e in long],
                'short_normal_source_intervals_not_discarded': [[base+s, base+e] for s, e in fragments],
                'reasons_local_intervals': r.get('reason_intervals', {}),
                'error': r.get('error'),
            })
            total += n
            flagged += int(bad.sum())
            padded_count += int(expanded.sum())
            protected += sum(e-s for s, e in long)
            short += sum(e-s for s, e in fragments)
    assert total == padded_count + protected + short
    dump(a.output / 'source_frame_inventory.json', rows)
    dump(a.output / 'summary.json', {
        'segments': len(rows), 'total_frames': total, 'flagged_frames': flagged,
        'repair_context_frames_including_5frame_padding': padded_count,
        'protected_normal_continuous_frames_min50': protected,
        'short_normal_frames_preserved_in_inventory': short,
        'purpose': 'Repair localization and protection index only, not a filtered dataset or training approval',
        'short_normal_policy': 'Keep metadata; recover as part of coherent newly solved trajectories, never hard-stitch',
        'trajectory_arrays_modified': False, 'training_approved': False,
    })


if __name__ == '__main__':
    main()
