"""Build exact-window reference datasets; leave original viewer/data untouched."""
import json
from pathlib import Path
import h5py
import numpy as np
from eval_side_regression_20261021 import ROOT, OUT, DATA

def main():
    report = json.loads((OUT/'metrics.json').read_text())
    source_manifest = json.loads((DATA/'manifest.json').read_text())
    for label, results in report['models'].items():
        target = OUT/f'viewer_{label}'
        target.mkdir(exist_ok=True)
        refs = target/'references'
        refs.mkdir(exist_ok=True)
        rows = sorted(results['clips'], key=lambda r: (not r['completed'], r['root_frame_body_error_mm']))
        if label == 'current3998':
            moonwalk = next((ROOT/'logs/HoloMotionK1MoonwalkV14Continue1000/20261021_134819-motion_tracking_booster_k1_moonwalk_v14_finetune/isaaclab_eval_output_model_3998_moonwalk_v14').glob('*.npz'))
            rows.insert(0, {'npz':str(moonwalk), 'moonwalk':True})
        manifest = {'version':2, 'root':str(refs), 'clips':{}, 'hdf5_shards':[{'file':'references.h5'}], 'dof_names':source_manifest['dof_names']}
        selections = []
        chunks = {'ref_root_pos':[], 'ref_root_rot':[], 'ref_dof_pos':[]}
        offset = 0
        for idx, row in enumerate(rows):
            with np.load(row['npz'], allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            meta = json.loads(str(arrays['metadata'].item()))
            key = meta['motion_key']
            meta['original_raw_motion_key'] = meta['raw_motion_key']
            meta['raw_motion_key'] = key
            arrays['metadata'] = np.array(json.dumps(meta))
            np.savez_compressed(target/(key+'.npz'), **arrays)
            # Exported reference is exact valid prefix: avoids original viewer
            # falling back to frame zero when replaying a later HDF5 window.
            n = len(arrays['ref_dof_pos'])
            chunks['ref_root_pos'].append(arrays['ref_global_translation'][:,0])
            chunks['ref_root_rot'].append(arrays['ref_global_rotation_quat'][:,0])
            chunks['ref_dof_pos'].append(arrays['ref_dof_pos'])
            manifest['clips'][key] = dict(motion_key=key, shard=0, clip_idx=idx, start=offset, length=n, metadata={'motion_fps':50.})
            offset += n
            if row.get('moonwalk'):
                category = '00 太空步专项：当前3998'
                description = '太空步专项已训练动作；不纳入通用动作回归统计。完整490帧。'
            else:
                category = f"{idx+1:02d} {'完整' if row['completed'] else '提前终止'}"
                description = f"{label}；覆盖{row['coverage']:.0%}；局部身体误差{row['root_frame_body_error_mm']:.1f}mm；关节{row['joint_mae_deg']:.1f}度。画面仅含有效帧；失败片段的参考也截到同一时刻，不代表完成。"
            selections.append(dict(motion_key=key, category_label=category, description=description))
        with h5py.File(refs/'references.h5', 'w') as f:
            for key, values in chunks.items():
                f.create_dataset(key, data=np.concatenate(values), compression='gzip')
        (refs/'manifest.json').write_text(json.dumps(manifest))
        (target/'comparison_selection.json').write_text(json.dumps({'clips':selections}, ensure_ascii=False, indent=2))
        print(target)

if __name__ == '__main__':
    main()
