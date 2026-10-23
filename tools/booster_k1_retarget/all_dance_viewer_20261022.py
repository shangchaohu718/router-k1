"""Build all labelled training-dance previews using frozen universal11999."""

from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np

from tools.booster_k1_retarget.expand_dance_viewer_20261022 import ROOT, SOURCE, CHECKPOINT, save

OUT = ROOT / 'test_outputs/universal11999_all_dance_20261022'
PATTERN = re.compile(r'danc|ballet|salsa|tango|waltz|charleston|jitterbug|polka|mambo|hip.?hop|locking|popping|moonwalk', re.I)


def pack(files, source, selected):
    target = OUT / 'viewer_build'
    refs = target / 'references'
    refs.mkdir(parents=True, exist_ok=False)
    clips, labels, chunks = {}, [], {k: [] for k in ['ref_root_pos', 'ref_root_rot', 'ref_dof_pos']}
    offset = 0
    seen = set()
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays['metadata'].item()))
        raw_key = meta.get('raw_motion_key', meta['motion_key'])
        if raw_key not in selected:
            continue
        if raw_key in seen:
            raise ValueError('Duplicate rollout: ' + raw_key)
        seen.add(raw_key)
        key = meta['motion_key']
        meta['original_raw_motion_key'] = raw_key
        meta['raw_motion_key'] = key
        arrays['metadata'] = np.array(json.dumps(meta))
        n = len(arrays['ref_dof_pos'])
        assert n > 0 and n == len(arrays['robot_dof_pos'])
        np.savez_compressed(target / (key + '.npz'), **arrays)
        for name, value in zip(chunks, [arrays['ref_global_translation'][:, 0], arrays['ref_global_rotation_quat'][:, 0], arrays['ref_dof_pos']]):
            assert len(value) == n and np.isfinite(value).all()
            chunks[name].append(value)
        clips[key] = {'shard': 0, 'clip_idx':len(clips), 'start': offset, 'length': n, 'metadata': {'motion_fps': 50.}}
        offset += n
        c = selected[raw_key]
        dataset = c['metadata'].get('dataset', 'unknown')
        label = ('旧舞蹈 LaFAN' if dataset == 'LaFAN1' else '舞蹈 ' + dataset)
        if dataset == 'AIST++':
            label += '：' + raw_key.split('_')[0]
        expected = max(1, int(meta.get('clip_length', n + 10)) - 10)
        description = c['metadata'].get('geometry_review_description', '')
        labels.append({'motion_key': key, 'category_label':label, 'description':f'{description}；通用11999；{n}帧；预览覆盖{min(1,n/expected):.0%}。提前终止也显示；最多20秒，不代表完整动作完成。参考是原训练映射，非新修复版。'})
    if seen != set(selected):
        raise ValueError('Missing rollouts: ' + str(sorted(set(selected) - seen)))
    with h5py.File(refs / 'references.h5', 'w') as handle:
        for name, arrays in chunks.items():
            handle.create_dataset(name, data=np.concatenate(arrays), compression='gzip')
    save(refs / 'manifest.json', {'version':2, 'dof_names':source['dof_names'], 'clips':clips, 'motion_keys':list(clips), 'hdf5_shards':[{'file':'references.h5'}]})
    save(target / 'comparison_selection.json', {'clips':labels})
    target.rename(OUT / 'viewer')
    return len(clips)


def main():
    OUT.mkdir(exist_ok=False)
    try:
        source = json.loads((SOURCE / 'manifest.json').read_text())
        selected = {k:c for k,c in source['clips'].items() if c['metadata'].get('dataset') == 'AIST++' or PATTERN.search(k + ' ' + c['metadata'].get('geometry_review_description', ''))}
        previous = ROOT / 'test_outputs/universal11999_more_dance_20261022/dance_reference_subset/manifest.json'
        previous_manifest = json.loads(previous.read_text())
        reuse_dir = CHECKPOINT.parent / 'isaaclab_eval_output_model_11999_dance_reference_subset'
        reused, reused_keys = [], set()
        for file in sorted(reuse_dir.glob('*.npz')):
            with np.load(file, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
            key = meta.get('raw_motion_key', meta['motion_key'])
            if key not in selected or key not in previous_manifest['clips']:
                continue
            old, new = previous_manifest['clips'][key], selected[key]
            assert old['start'] == new['start'] and old['length'] == min(1000, new['length'])
            reused.append(file)
            reused_keys.add(key)
        counts = Counter(c['metadata'].get('dataset') for c in selected.values())
        save(OUT / 'protocol.json', {'checkpoint': str(CHECKPOINT), 'source_manifest': str(SOURCE/'manifest.json'), 'selection_rule':PATTERN.pattern, 'selected_keys':list(selected), 'source_counts':dict(counts), 'reused_keys':sorted(reused_keys), 'max_preview_frames':1000, 'training_started':False, 'reference_status':'original training references, not newly repaired'})
        missing = {k:copy.deepcopy(c) for k,c in selected.items() if k not in reused_keys}
        data = OUT / 'all_dance_missing_20261022'
        if missing:
            subset = copy.deepcopy(source)
            subset['clips'] = missing
            subset['motion_keys'] = list(missing)
            subset['hdf5_shards'] = [{**s,'file':str((SOURCE/s['file']).resolve())} for s in source['hdf5_shards']]
            for c in missing.values():
                c['length'] = min(1000, c['length'])
            save(data / 'manifest.json', subset)
            save(OUT / 'status.json', {'status':'evaluating_missing_dances','selected':len(selected),'reused':len(reused_keys),'to_evaluate':len(missing),'source_counts':dict(counts),'training_started':False})
            env = dict(os.environ, CKPT_PATH=str(CHECKPOINT), TRAIN_H5_DATASET_PATH=str([str(data)]), MOTION_NAMES='null', EXCLUDED_MOTION_NAMES='null', NUM_ENVS='16', MAX_FRAME_LENGTH='1000', MAX_EVAL_FRAMES='1000', DATA_LOADER_NUM_WORKERS='0')
            with (OUT / 'evaluation.log').open('w') as log:
                subprocess.run(['bash','holomotion/scripts/evaluation/eval_booster_k1_training_set.sh'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw = CHECKPOINT.parent / ('isaaclab_eval_output_model_11999_' + data.name)
        count = pack(reused + sorted(raw.glob('*.npz')), source, selected)
        save(OUT / 'status.json', {'status':'viewer_ready','viewer_clips':count,'source_counts':dict(counts),'viewer':str(OUT/'viewer'),'training_started':False})
        print('Ready:', OUT / 'viewer', count, flush=True)
    except Exception as error:
        save(OUT / 'failure.json', {'error':repr(error),'training_started':False})
        raise


if __name__ == '__main__':
    main()
