"""Report byte-identical motion sources across paths, without changing datasets."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = defaultdict(list)
    for f in sorted(a.snapshot.glob('*/manifest.json')):
        m = json.loads(f.read_text())
        for key, c in m['clips'].items():
            meta = c['metadata']
            sources[meta['source_path']].append({'manifest': str(f.resolve()), 'motion': key, 'canonical_start': meta['canonical_frame_start'], 'frames': c['length'], 'source_fps': meta['source_fps']})
    groups = defaultdict(list)
    failures = []
    for index, (source, clips) in enumerate(sorted(sources.items())):
        path = Path(source)
        try:
            motion_hash = digest(path)
            skeleton_hash = None
            if path.suffix.lower() == '.amc':
                skeletons = list(path.parent.glob('*.asf'))
                if len(skeletons) != 1:
                    raise ValueError('Ambiguous AMC skeleton')
                skeleton_hash = digest(skeletons[0])
            groups[(path.suffix.lower(), motion_hash, skeleton_hash)].append({'source_path': source, 'clips': clips})
        except Exception as exc:
            failures.append({'source_path': source, 'error': repr(exc)})
        dump(a.output / 'progress.json', {'processed_sources': index+1, 'total_sources': len(sources), 'status': 'running', 'training_started': False})
    duplicates = [{'format': k[0], 'source_sha256': k[1], 'skeleton_sha256': k[2], 'sources': v} for k, v in groups.items() if len(v) > 1]
    dump(a.output / 'duplicate_source_groups.json', duplicates)
    dump(a.output / 'summary.json', {'unique_source_paths_checked': len(sources), 'identical_content_groups': len(duplicates), 'paths_in_duplicate_groups': sum(len(g['sources']) for g in duplicates), 'failures': failures, 'dataset_modified': False, 'corrected_unique_duration': 'pending canonical time-correspondence and interval-union verification', 'training_approved': False})
    dump(a.output / 'progress.json', {'processed_sources': len(sources), 'status': 'complete_time_correspondence_check_pending', 'training_started': False})


if __name__ == '__main__':
    main()
