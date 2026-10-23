"""Verify canonical time identity before correcting cross-path source-frame counts."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from tools.booster_k1_retarget.probe_native_position_retarget import canonical_arrays
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def union_size(spans):
    end = -1
    total = 0
    for s, e in sorted(spans):
        total += max(0, e - max(s, end))
        end = max(end, e)
    return total


@lru_cache(maxsize=8)
def manifest(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    groups = json.loads((a.audit / 'duplicate_source_groups.json').read_text())
    results = []
    removed = 0
    aliases = {}
    for group in groups:
        signatures = []
        spans = []
        sources = group['sources']
        for source in sources:
            assert group['format'] == '.amc'  # AMC adapter has zero source-frame trim offset.
            signatures_for_path = set()
            for entry in source['clips']:
                clip = manifest(entry['manifest'])['clips'][entry['motion']]
                meta = clip['metadata']
                # Multiple canonical files on one source must all prove the same mapping.
                signatures_for_path.add((meta.get('canonical_path'), float(meta['source_fps'])))
                s = entry['canonical_start']
                spans.append((s, s + entry['frames']))
            path_signatures = []
            for canonical_path, fps in signatures_for_path:
                meta = {'source_path': source['source_path'], 'dataset': 'CMU', 'canonical_path': canonical_path}
                root, body, trans, cm = canonical_arrays(meta)
                h = hashlib.sha256()
                for array in [root, body, trans]:
                    h.update(str((array.shape, str(array.dtype))).encode())
                    h.update(array.tobytes())
                path_signatures.append((h.hexdigest(), fps, len(root)))
            signatures.extend(path_signatures)
        verified = len(set(signatures)) == 1
        overlap = sum(e-s for s, e in spans) - union_size(spans) if verified else 0
        if verified:
            assert all(0 <= s < e <= signatures[0][2] for s, e in spans)
            representative = min(s['source_path'] for s in sources)
            aliases.update({s['source_path']: representative for s in sources})
        removed += overlap
        results.append({'sources': [s['source_path'] for s in sources], 'canonical_arrays_and_timebase_identical': verified, 'duplicate_candidate_frames': overlap})
    total = sum(c['length'] for f in a.snapshot.glob('*/manifest.json') for c in manifest(str(f))['clips'].values())
    dump(a.output / 'verified_source_aliases.json', aliases)
    dump(a.output / 'groups.json', results)
    dump(a.output / 'summary.json', {'groups_checked': len(results), 'groups_verified': sum(r['canonical_arrays_and_timebase_identical'] for r in results), 'prior_path_unique_frames': total, 'duplicate_frames_across_verified_aliases': removed, 'source_content_unique_frames': total-removed, 'source_content_unique_hours': (total-removed)/180000, 'data_arrays_modified': False, 'candidate_views_not_yet_rewritten': True, 'scope': 'Exact source bytes plus canonical arrays/timebase identity; not semantic near-duplicate detection', 'training_approved': False})


if __name__ == '__main__':
    main()
