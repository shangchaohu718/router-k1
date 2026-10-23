"""Isolated tradeoff pilot; never modifies the frozen bulk worker."""

import argparse
import copy
import json
from pathlib import Path

from tools.booster_k1_retarget import compare_lafan_retarget_pipelines as compare
from tools.booster_k1_retarget import retry_guarded_reference as guard
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm-cost', type=float, default=50)
    parser.add_argument('--foot-cost', type=float, default=200)
    parser.add_argument('--root-cost', type=float, default=200)
    parser.add_argument('--foot-orientation-cost', type=float, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.case / 'input/manifest.json').read_text())
    view = copy.deepcopy(manifest)
    for clip in view['clips'].values():
        n = min(1000, clip['length'])
        clip['length'] = n
        meta = clip['metadata']
        meta.update(num_frames=n, canonical_frame_count=n, parent_frame_end=meta['parent_frame_start']+n)
    dump(args.output / 'input/manifest.json', view)
    dump(args.output / 'pilot_settings.json', {'input_case': str(args.case.resolve()), 'arm_position_cost': args.arm_cost, 'foot_position_cost': args.foot_cost, 'foot_orientation_cost': args.foot_orientation_cost, 'root_position_cost': args.root_cost, 'training_approved': False, 'bulk_config_unchanged': True})
    create = guard.probe.create_k1_retargeter
    cfg = json.loads((compare.REPO / 'tools/booster_k1_retarget/configs/gmr_smplx_to_official_k1_v2.json').read_text())

    def make(**kwargs):
        robot = create(**kwargs)
        retarget = robot.retarget

        def balanced(*values, **options):
            for label, table in [('ik_match_table1', robot.human_body_to_task1), ('ik_match_table2', robot.human_body_to_task2)]:
                for name, entry in cfg[label].items():
                    if name == 'trunk':
                        table[entry[0]].set_position_cost(args.root_cost)
                    if 'elbow_yaw' in name:
                        table[entry[0]].set_position_cost(args.arm_cost)
                    if 'ankle_roll' in name:
                        table[entry[0]].set_position_cost(args.foot_cost)
                        table[entry[0]].set_orientation_cost(args.foot_orientation_cost)
            return retarget(*values, **options)

        robot.retarget = balanced
        return robot

    guard.probe.create_k1_retargeter = make
    compare.worker(argparse.Namespace(output=args.output, worker='lafan_arm_direction_rootpos200', max_frames=1000))


if __name__ == '__main__':
    main()
