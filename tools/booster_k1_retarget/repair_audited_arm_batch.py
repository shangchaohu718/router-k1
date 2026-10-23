"""Independent representative arm-repair batch; never promotes or replaces inputs."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=2)
    parser.add_argument('--exclude-selection', type=Path)
    parser.add_argument('--source-aliases', type=Path)
    args = parser.parse_args()
    if args.per_source < 1:
        raise ValueError('per-source must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    totals = {}
    manifests = {}
    aliases = json.loads(args.source_aliases.read_text()) if args.source_aliases else {}
    excluded_sources = set()
    if args.exclude_selection:
        excluded_sources = {aliases.get(r['source_path'], r['source_path']) for r in json.loads(args.exclude_selection.read_text())['selected']}
    for file in sorted((args.audit / 'segments').glob('*.json')):
        row = json.loads(file.read_text())
        name = row['input_manifest']
        if name not in manifests:
            manifests[name] = json.loads(Path(name).read_text())
        manifest = manifests[name]
        dataset = manifest['clips'][row['motion']]['metadata']['dataset']
        total = totals.setdefault(dataset, {'segments': 0, 'frames': 0, 'flagged_frames': 0})
        total['segments'] += 1
        total['frames'] += row['frames']
        total['flagged_frames'] += row['flagged_frames']
        if aliases.get(row['source_path'], row['source_path']) not in excluded_sources and row['status'] == 'needs_repair_review' and 100 <= row['frames'] <= 600 and row['flagged_frames'] >= 50:
            groups.setdefault(dataset, []).append((file, row))
    selected = []
    for dataset, entries in sorted(groups.items()):
        sources = set()
        for file, row in sorted(entries, key=lambda item: (-item[1]['flagged_frames'] / item[1]['frames'], item[0].name)):
            source_identity = aliases.get(row['source_path'], row['source_path'])
            if source_identity in sources:
                continue
            selected.append((dataset, file, row))
            sources.add(source_identity)
            if len(sources) == args.per_source:
                break
    dump(args.output / 'selection.json', {'source_totals': totals, 'selected': [{'dataset': d, 'audit_record': str(f.resolve()), **r} for d, f, r in selected], 'training_approved': False})
    completed = []
    def run(module, *extra):
        subprocess.run([sys.executable, '-m', 'tools.booster_k1_retarget.' + module, *map(str, extra)], check=True)
    for dataset, file, row in selected:
        case = args.output / file.stem
        case.mkdir()
        manifest_path = Path(row['input_manifest'])
        manifest = json.loads(manifest_path.read_text())
        clip = copy.deepcopy(manifest['clips'][row['motion']])
        shard = copy.deepcopy(manifest['hdf5_shards'][clip['shard']])
        shard['file'] = str((manifest_path.parent / shard['file']).resolve())
        clip['shard'] = 0
        single = {**manifest, 'clips': {row['motion']: clip}, 'motion_keys': [row['motion']], 'hdf5_shards': [shard], 'training_approved': False}
        dump(case / 'input/manifest.json', single)
        dump(args.output / 'progress.json', {'status': 'repairing', 'current': file.stem, 'completed': completed, 'target': len(selected), 'training_started': False})
        try:
            for weight in [1, 3]:
                trial = case / f'forearm{weight}'
                run('probe_fixed_lower_body_arm_repair', '--input', case / 'input', '--output', trial / 'raw', '--collision-weight', 1000, '--multi-start', '--native-forearm', '--forearm-weight', weight)
                run('audit_arm_only_repair', '--original', case / 'input', '--repaired', trial / 'raw', '--output', trial / 'arm_contact_audit.json')
                run('audit_native_forearm_fidelity', '--original', case / 'input', '--repaired', trial / 'raw', '--output', trial / 'native_forearm_audit.json')
                run('salvage_source_root_checked', '--input', trial / 'raw', '--output', trial / 'review', '--smpl-model', '/home/tcl/zsq/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl')
                run('audit_large_candidate_arms', '--snapshot', trial, '--output', trial / 'frame_arm_audit')
            completed.append({'case': file.stem, 'dataset': dataset, 'status': 'trials_complete_not_promoted'})
        except Exception as exc:
            completed.append({'case': file.stem, 'dataset': dataset, 'status': 'failed_isolated', 'error': repr(exc)})
        dump(case / 'status.json', completed[-1])
    dump(args.output / 'progress.json', {'status': 'complete_review_required', 'completed': completed, 'target': len(selected), 'training_started': False, 'training_approved': False, 'visual_and_physics': 'pending', 'originals_unchanged': True})


if __name__ == '__main__':
    main()
