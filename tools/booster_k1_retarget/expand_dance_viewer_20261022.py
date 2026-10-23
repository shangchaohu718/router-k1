"""Isolated extra dance replay for universal11999; no training/data repair edits."""

import copy
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test_outputs/universal11999_more_dance_20261022'
CHECKPOINT = ROOT / 'logs/HoloMotionK1SequentialCurriculum/overnight_20261020_153041_Bl3s1P/stage_3_complex/model_11999.pt'
SOURCE = ROOT / 'data/holomotion_k1_v1/reference_geometry_review_20261021/all_training_input'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    source = json.loads((SOURCE / 'manifest.json').read_text())
    chosen = []
    for name in ['dance1_subject1', 'dance1_subject2', 'dance1_subject3', 'dance2_subject1', 'dance2_subject2', 'dance2_subject3', 'dance2_subject4']:
        key = max((k for k in source['clips'] if k.split('__')[0] == name), key=lambda k: source['clips'][k]['length'])
        chosen.append((key, '旧舞蹈 LaFAN：' + name))
    aist = [(k,c) for k,c in source['clips'].items() if c['metadata'].get('dataset') == 'AIST++']
    for genre in sorted({k.split('_')[0] for k,c in aist}):
        captures = set()
        count = 0
        for key, clip in sorted(aist, key=lambda pair: -pair[1]['length']):
            if key.split('_')[0] != genre:
                continue
            capture = clip['metadata']['source_path']
            if capture in captures:
                continue
            chosen.append((key, '新舞蹈 AIST++：' + genre))
            captures.add(capture)
            count += 1
            if count == 2:
                break
    subset = copy.deepcopy(source)
    subset['clips'] = {k: copy.deepcopy(source['clips'][k]) for k,label in chosen}
    for c in subset['clips'].values():
        c['length'] = min(c['length'], 1000)
    subset['motion_keys'] = list(subset['clips'])
    subset['hdf5_shards'] = [{**s, 'file': str((SOURCE/s['file']).resolve())} for s in source['hdf5_shards']]
    data = OUT / 'dance_reference_subset'
    save(data/'manifest.json', subset)
    save(OUT/'protocol.json', {'checkpoint': str(CHECKPOINT), 'selection': chosen, 'max_frames':1000, 'reference_status':'existing trained reference mapping; not newly repaired or visually certified', 'training_started':False})
    env = dict(os.environ, CKPT_PATH=str(CHECKPOINT), TRAIN_H5_DATASET_PATH=str([str(data)]), MOTION_NAMES='null', EXCLUDED_MOTION_NAMES='null', NUM_ENVS='8', MAX_FRAME_LENGTH='1000', MAX_EVAL_FRAMES='1000', DATA_LOADER_NUM_WORKERS='0')
    with (OUT/'evaluation.log').open('w') as log:
        subprocess.run(['bash','holomotion/scripts/evaluation/eval_booster_k1_training_set.sh'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = CHECKPOINT.parent / ('isaaclab_eval_output_model_11999_' + data.name)
    target = OUT / 'viewer'
    references = target/'references'
    references.mkdir(parents=True)
    clips, selections = {}, []
    chunks = {name: [] for name in ['ref_root_pos','ref_root_rot','ref_dof_pos']}
    labels = dict(chosen)
    original = ROOT/'test_outputs/current3998_regression_20261021/viewer_universal11999'
    files = [(file, False) for file in sorted(original.glob('*.npz'))] + [(file, True) for file in sorted(raw.glob('*.npz'))]
    offset = 0
    for index, (file, added) in enumerate(files):
        with np.load(file, allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        meta = json.loads(str(arrays['metadata'].item()))
        raw_key = meta.get('raw_motion_key', meta['motion_key'])
        key = ('extra__' if added else 'existing__') + meta['motion_key']
        meta['original_raw_motion_key'] = raw_key
        meta['raw_motion_key'] = key
        meta['motion_key'] = key
        arrays['metadata'] = np.array(json.dumps(meta))
        n = len(arrays['ref_dof_pos'])
        assert n == len(arrays['robot_dof_pos']) and n > 0
        np.savez_compressed(target/(key+'.npz'), **arrays)
        values = [arrays['ref_global_translation'][:,0], arrays['ref_global_rotation_quat'][:,0], arrays['ref_dof_pos']]
        for name, value in zip(chunks, values):
            assert len(value) == n and np.isfinite(value).all()
            chunks[name].append(value)
        clips[key] = {'shard':0,'clip_idx':index,'start':offset,'length':n,'metadata':{'motion_fps':50.}}
        offset += n
        expected = max(1, int(meta.get('clip_length',n+10))-10)
        label = labels.get(raw_key, '原窗口动作') if added else '原窗口动作'
        selections.append({'motion_key':key,'category_label':label,'description':f'通用11999；有效播放{n}帧；覆盖{min(1,n/expected):.0%}。若提前终止，参考也仅显示相同时段，不代表完整完成。参考为原训练映射，非当前修复版。'})
    with h5py.File(references/'references.h5','w') as handle:
        for name, values in chunks.items():
            handle.create_dataset(name,data=np.concatenate(values),compression='gzip')
    save(references/'manifest.json',{'version':2,'dof_names':source['dof_names'],'clips':clips,'motion_keys':list(clips),'hdf5_shards':[{'file':'references.h5'}]})
    save(target/'comparison_selection.json',{'clips':selections})
    save(OUT/'status.json',{'status':'viewer_ready','new_selected_motions':len(chosen),'new_rollout_files':len(list(raw.glob('*.npz'))),'total_viewer_clips':len(clips),'viewer':str(target),'training_started':False})
    print(json.dumps({'viewer':str(target),'new_motions':len(chosen),'total_clips':len(clips)},indent=2),flush=True)


if __name__ == '__main__':
    main()
