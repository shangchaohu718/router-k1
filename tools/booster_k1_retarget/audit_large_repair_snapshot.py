"""Audit completed repair outputs without touching active workers or originals."""

import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--package', action='store_true')
    args = parser.parse_args()
    config = json.loads((args.root / 'run_config.json').read_text())
    for file, expected in config['worker_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
    baseline_file = Path(config['baseline']) / 'manifest.json'
    assert hashlib.sha256(baseline_file.read_bytes()).hexdigest() == config['baseline_manifest_sha256']
    baseline = json.loads(baseline_file.read_text())
    package_inputs = [(baseline_file, baseline)]
    intervals = defaultdict(list)
    for clip in baseline['clips'].values():
        meta = clip['metadata']
        start = meta['canonical_frame_start']
        intervals[str(Path(meta['source_path']).resolve())].append((start, start + clip['length']))
    counts = Counter()
    sources = Counter()
    unresolved = []
    frames = segments = 0
    for status_file in sorted(args.root.glob('cases/*/status.json')):
        status = json.loads(status_file.read_text())
        counts[status['status']] += 1
        if status['status'] != 'repaired_patches_retained':
            diagnostics = []
            for report_file in sorted(status_file.parent.glob('*/review/report.json')):
                row = json.loads(report_file.read_text())['clips'][0]
                diagnostics.append({'profile': report_file.parent.parent.name,
                                    'frames': row['frames'],
                                    'source_error': row['source_error'],
                                    'reason_frames': {key: sum(end - start for start, end in spans) for key, spans in row['reason_intervals'].items()},
                                    'report': str(report_file.resolve())})
            unresolved.append({'motion': status['motion'], 'source': status['source'], 'diagnostics': diagnostics})
            continue
        file = status_file.parent / 'patches/manifest.json'
        manifest = json.loads(file.read_text())
        package_inputs.append((file, manifest))
        assert manifest['dof_names'] == baseline['dof_names']
        case_frames = 0
        for clip in manifest['clips'].values():
            n = clip['length']
            meta = clip['metadata']
            start = meta['canonical_frame_start']
            assert n >= 50 and meta['canonical_frame_count'] == n
            assert meta['parent_frame_end'] - meta['parent_frame_start'] == n
            assert meta['independent_patch_not_stitched']
            intervals[str(Path(meta['source_path']).resolve())].append((start, start + n))
            shard = manifest['hdf5_shards'][clip['shard']]
            with h5py.File(file.parent / shard['file'], 'r') as handle:
                arrays = [handle[name][clip['start']:clip['start'] + n] for name in ['ref_root_pos', 'ref_root_rot', 'ref_dof_pos']]
            assert [array.shape for array in arrays] == [(n, 3), (n, 4), (n, 22)]
            assert all(np.isfinite(array).all() for array in arrays)
            assert np.max(np.abs(np.linalg.norm(arrays[1], axis=1) - 1)) < .001
            case_frames += n
            segments += 1
        assert case_frames == status['added_frames']
        sources[status['source']] += case_frames
        frames += case_frames
    for source, spans in intervals.items():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), source
    report = {'timestamp_unix': time.time(), 'completed_case_counts': dict(counts), 'added_segments': segments, 'added_frames': frames, 'added_hours': frames / 180000, 'source_added_frames': dict(sources), 'array_integrity_passed': True, 'baseline_and_patch_source_interval_overlap_count': 0, 'frozen_workers_and_baseline_unchanged': True, 'visual_validation': 'pending', 'physics_validation': 'pending', 'training_approved': False, 'snapshot_only_worker_continues': True}
    output = args.root / 'audits' / ('snapshot_' + str(time.time_ns()) + '.json')
    dump(output, report)
    dump(output.with_name(output.stem + '_unresolved.json'), {'unresolved_cases': unresolved, 'not_declared_unfixable': True, 'next_review': 'Check startup transients with longer source context before giving up short intervals; do not relax geometry gates.'})
    if args.package:
        groups = defaultdict(list)
        for file, manifest in package_inputs:
            for key, clip in manifest['clips'].items():
                groups[clip['metadata'].get('provisional_difficulty', '99_unclassified')].append((file, manifest, key, clip))
        package_root = args.root / 'candidate_snapshots' / output.stem
        stage_summary = {}
        for stage, rows in groups.items():
            clips, shards = {}, []
            for index, (file, manifest, key, clip) in enumerate(rows):
                copied = copy.deepcopy(clip)
                shard = manifest['hdf5_shards'][clip['shard']]
                copied['shard'] = len(shards)
                copied['metadata']['difficulty_classification_status'] = 'inherited_provisional_not_recomputed_after_repair'
                copied['metadata']['training_approved'] = False
                clips[str(index) + '__' + key] = copied
                shards.append({**shard, 'file': str((file.parent / shard['file']).resolve())})
            dump(package_root / stage / 'manifest.json', {'version': 2, 'dataset_id': output.stem + '_' + stage, 'dof_names': baseline['dof_names'], 'array_names': ['ref_root_pos', 'ref_root_rot', 'ref_dof_pos'], 'clips': clips, 'motion_keys': list(clips), 'hdf5_shards': shards, 'training_approved': False})
            dump(package_root / stage / 'nan_npz_paths.json', [])
            stage_summary[stage] = {'segments': len(clips), 'hours': sum(c['length'] for c in clips.values()) / 180000}
        dump(package_root / 'summary.json', {'stages': stage_summary, 'training_approved': False, 'contains_protected_baseline_and_new_patches': True, 'source_interval_overlap_count': 0, 'difficulty_is_inherited_provisional': True, 'visual_and_physics_validation': 'pending', 'self_contained_copy': False, 'note': 'Absolute shard references; do not move alone or add overlapping old dataset durations.'})
        print('Candidate package:', package_root)
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == '__main__':
    main()
