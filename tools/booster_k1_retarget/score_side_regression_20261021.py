"""Summarize every selected rollout; never treat early termination as success."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from eval_side_regression_20261021 import ROOT, OUT, DATA, MODELS

def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    report = {'protocol': json.loads((OUT / 'protocol.json').read_text()), 'models': {}}
    for label, checkpoint in MODELS.items():
        directory = ROOT / Path(checkpoint).parent / ('isaaclab_eval_output_' + Path(checkpoint).stem + '_reference_subset')
        rows = []
        files = list(directory.glob('*.npz'))
        if not files:
            raise RuntimeError(f'No rollouts: {directory}')
        for file in files:
            with np.load(file, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                key = meta.get('raw_motion_key', meta['motion_key'])
                ref, pred = z['ref_global_translation'], z['robot_global_translation']
                n = len(pred)
                refrot = Rotation.from_quat(z['ref_global_rotation_quat'][:, 0])
                predrot = Rotation.from_quat(z['robot_global_rotation_quat'][:, 0])
                reflocal = np.einsum('tij,tkj->tki', refrot.inv().as_matrix(), ref-ref[:, :1])
                predlocal = np.einsum('tij,tkj->tki', predrot.inv().as_matrix(), pred-pred[:, :1])
                expected = int(meta['clip_length']) - 10
                coverage = min(1., n / max(1, expected))
                row = dict(motion_key=key, window_key=meta['motion_key'], frames=n, expected_frames=expected,
                           coverage=coverage,
                           completed=bool(n >= expected-1),
                           root_height_within_25cm=bool(np.max(np.abs(pred[:, 0, 2]-ref[:, 0, 2])) < .25),
                           global_body_error_mm=float(np.linalg.norm(pred-ref, axis=-1).mean()*1000),
                           root_frame_body_error_mm=float(np.linalg.norm(predlocal-reflocal, axis=-1).mean()*1000),
                           joint_mae_rad=float(np.abs(z['robot_dof_pos']-z['ref_dof_pos']).mean()),
                           joint_mae_deg=float(np.rad2deg(np.abs(z['robot_dof_pos']-z['ref_dof_pos']).mean())),
                           npz=str(file))
                rows.append(row)
        missing = sorted(set(manifest['clips']) - {r['motion_key'] for r in rows})
        report['models'][label] = dict(directory=str(directory), clips=rows, missing=missing,
            completion_rate=sum(r['completed'] for r in rows)/len(rows),
            macro_global_body_error_mm=float(np.mean([r['global_body_error_mm'] for r in rows])),
            macro_root_frame_body_error_mm=float(np.mean([r['root_frame_body_error_mm'] for r in rows])),
            macro_joint_mae_rad=float(np.mean([r['joint_mae_rad'] for r in rows])))
    baseline_by_window = {r['window_key']:r for r in report['models']['universal11999']['clips']}
    paired = []
    for row in report['models']['current3998']['clips']:
        base = baseline_by_window[row['window_key']]
        with np.load(base['npz']) as a, np.load(row['npz']) as b:
            n = min(len(a['robot_dof_pos']), len(b['robot_dof_pos']))
            if not np.allclose(a['ref_dof_pos'][:n], b['ref_dof_pos'][:n], atol=1e-6):
                raise ValueError('Reference mismatch between checkpoints')
            scores = {}
            for label, z in [('universal11999',a), ('current3998',b)]:
                scores[label] = {
                    'global_body_error_mm':float(np.linalg.norm(z['robot_global_translation'][:n]-z['ref_global_translation'][:n],axis=-1).mean()*1000),
                    'joint_mae_rad':float(np.abs(z['robot_dof_pos'][:n]-z['ref_dof_pos'][:n]).mean()),
                }
            paired.append(dict(window_key=row['window_key'], common_frames=n, scores=scores))
    report['matched_prefix_comparison'] = paired
    report['caveat'] = 'Own-valid-prefix macro errors have different horizons when a model terminates early; not a full-horizon fair error comparison. Matched-prefix scores compare identical time ranges. Completion is benchmark-specific, not paper SR.'
    report['published_context_not_direct_ranking'] = [
        {'source':'https://arxiv.org/html/2605.15336v1','method':'HoloMotion-1','robot':'G1 29DoF','simulator':'MuJoCo','global_body_error_mm':124.57,'joint_mae_rad':0.0979,'success_rate_percent':97.55,'note':'Different unseen datasets and height-based success criterion; not comparable to internal K1 training regression.'},
        {'source':'https://arxiv.org/html/2608.13555v1','method':'Humanoid-GPT','benchmark':'HumanTracker Daily','success_rate_percent':94.4,'joint_mae_rad':0.046,'note':'Benchmark-specific result, not a universal SOTA ranking; different references and whole-body termination.'},
    ]
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2))
    current = report['models']['current3998']
    rows = sorted(current['clips'], key=lambda r: (not r['completed'], r['root_frame_body_error_mm']))
    selection = {'clips': [dict(motion_key=r['motion_key'], category_label=f"{i+1:02d} {'完整' if r['completed'] else '提前终止'}", description=f"覆盖{r['coverage']:.0%}；身体局部误差{r['root_frame_body_error_mm']:.1f}mm；关节{r['joint_mae_deg']:.1f}度。内部训练回归集，非SOTA基准。") for i, r in enumerate(rows)]}
    (Path(current['directory'])/'comparison_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    for label, results in report['models'].items():
        print(label, {k:v for k,v in results.items() if k not in ['clips','directory']})
        for r in results['clips']:
            print(r['motion_key'], r['frames'], r['expected_frames'], round(r['root_frame_body_error_mm'],1))

if __name__ == '__main__':
    main()
