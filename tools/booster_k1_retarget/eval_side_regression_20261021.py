"""Frozen, geometry-screened internal regression set; not a public benchmark."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'test_outputs/current3998_regression_20261021'
DATA = OUT / 'reference_subset'
SOURCE = ROOT / 'data/holomotion_k1_v1/reference_geometry_review_20261021/all_training_input'
KEYS = [
    'Neutral_ID__59a98e1989', 'Roadrunner_ID__5aee655158',
    'walk3_subject1__frames_5438_6613', 'walk3_subject1__frames_5090_5373',
    'Kick_ID__76d195959e', 'Punch_ID__e61564da3d',
    'dance1_subject1__frames_20_692', 'dance1_subject2__frames_1271_1544',
    'dance2_subject3__frames_8284_8645', 'TwoFootJump_ID__11f2073364',
]
MODELS = {
    'universal11999': 'logs/HoloMotionK1SequentialCurriculum/overnight_20261020_153041_Bl3s1P/stage_3_complex/model_11999.pt',
    'current3998': 'logs/HoloMotionK1MoonwalkV14Continue1000/20261021_134819-motion_tracking_booster_k1_moonwalk_v14_finetune/model_3998.pt',
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    manifest['clips'] = {k: manifest['clips'][k] for k in KEYS}
    manifest['hdf5_shards'] = [dict(p, file=str((SOURCE / p['file']).resolve())) for p in manifest['hdf5_shards']]
    manifest['root'] = str(DATA)
    (DATA / 'manifest.json').write_text(json.dumps(manifest))
    (OUT / 'protocol.json').write_text(json.dumps({
        'keys': KEYS, 'models': MODELS, 'max_frames': 1000,
        'note': 'Seen-training internal regression; geometry screen only, not visually certified. All selected clips reported, including failures.',
    }, indent=2))
    for label, checkpoint in MODELS.items():
        env = dict(os.environ, CKPT_PATH=checkpoint,
                   TRAIN_H5_DATASET_PATH=str([str(DATA)]), MOTION_NAMES='null',
                   EXCLUDED_MOTION_NAMES='null', NUM_ENVS='10',
                   MAX_FRAME_LENGTH='1000', MAX_EVAL_FRAMES='1000',
                   DATA_LOADER_NUM_WORKERS='0')
        with (OUT / f'{label}.log').open('w') as log:
            result = subprocess.run(['bash', 'holomotion/scripts/evaluation/eval_booster_k1_training_set.sh'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (OUT / f'{label}.status.json').write_text(json.dumps({'returncode': result.returncode}))
        if result.returncode:
            raise RuntimeError(f'{label} failed; see log')

if __name__ == '__main__':
    main()
