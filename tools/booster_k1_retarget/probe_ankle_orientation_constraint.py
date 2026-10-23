"""Minimal isolated foot-orientation ablation; all real joint limits retained."""

import argparse
import json
from pathlib import Path

import mink
import mujoco
import numpy as np

from tools.booster_k1_retarget import compare_lafan_retarget_pipelines as compare
from tools.booster_k1_retarget import retry_guarded_reference as guard
from tools.booster_k1_retarget.recover_short_reference_candidates import dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--foot-orientation-cost',type=float,default=0)
    p.add_argument('--ankle-posture-cost',type=float,default=0)
    p.add_argument('--native-cmu-foot-pose',action='store_true')
    p.add_argument('--reference-root-position-cost',type=float,default=0,help='Soft position task targeting the original robot root, not a post-hoc height shift')
    p.add_argument('--source-grounded-foot-position-cost',type=float,default=0,help='Experimental inferred-contact IK objective; preserve original foot XY and solve sole Z=0, never edit solved poses afterwards')
    a=p.parse_args()
    if a.source_grounded_foot_position_cost and not (a.native_cmu_foot_pose and a.reference_root_position_cost):
        raise ValueError('Ground-contact trial requires native foot pose and original-root task')
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.case/'input/manifest.json').read_text())
    for c in manifest['clips'].values():c['metadata'].update(actual_foot_orientation_task_override=a.foot_orientation_cost,actual_ankle_posture_cost=a.ankle_posture_cost,native_cmu_foot_pose_override=a.native_cmu_foot_pose,training_approved=False)
    dump(a.output/'input/manifest.json',manifest)
    dump(a.output/'experiment_settings.json',{'base_profile':'old_root_weight','actual_foot_orientation_task_cost':a.foot_orientation_cost,'ankle_posture_cost':a.ankle_posture_cost,'native_cmu_foot_pose':a.native_cmu_foot_pose,'reference_root_position_task_cost':a.reference_root_position_cost,'source_grounded_foot_position_cost':a.source_grounded_foot_position_cost,'contact_source':'FK-inferred, not measured','root_task_target':'original robot reference, soft IK objective' if a.reference_root_position_cost else 'unchanged','joint_limits_unchanged':True,'source_case':str(a.case.resolve()),'training_approved':False})
    config=json.loads((compare.REPO/'tools/booster_k1_retarget/configs/gmr_smplx_to_official_k1_v2.json').read_text())
    create=guard.probe.create_k1_retargeter
    def make(**kwargs):
        robot=create(**kwargs);retarget=robot.retarget
        frame=0
        if a.native_cmu_foot_pose:
            from tools.booster_k1_retarget.cmu_native_foot_pose import foot_targets
            assert len(manifest['clips'])==1
            clip=next(iter(manifest['clips'].values()))
            desired=foot_targets(clip['metadata'],clip['length'])
            reference_root = compare.load(a.case/'input', next(iter(manifest['clips'])))['ref_root_pos'] if a.reference_root_position_cost else None
            contact_targets = {}
            if a.source_grounded_foot_position_cost:
                from tools.booster_k1_retarget.salvage_repaired_reference import source_positions
                human,ground,_=source_positions(clip['metadata'],clip['length'],None)
                scale=clip['metadata'].get('anatomical_leg_scale',.410702/float(np.median([np.linalg.norm(human[s+'_knee']-human[s+'_hip'],axis=1)+np.linalg.norm(human[s+'_ankle']-human[s+'_knee'],axis=1) for s in ['left','right']])))
                original=compare.load(a.case/'input',next(iter(manifest['clips'])))
                original_data=mujoco.MjData(robot.model)
                for side in ['left','right']:
                    body=robot.model.body(side+'_ankle_roll_link').id
                    geom=next(g for g in range(robot.model.ngeom) if robot.model.geom_bodyid[g]==body and robot.model.geom_contype[g] and robot.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX)
                    local_rotation=np.zeros(9)
                    mujoco.mju_quat2Mat(local_rotation,robot.model.geom_quat[geom])
                    grounded=(np.minimum(human[side+'_foot'][:,2],human[side+'_ankle'][:,2])-ground)*scale<.02
                    if not np.all(grounded):
                        raise ValueError('Contact-position prototype currently requires continuous inferred double support; swing/transition handling not validated')
                    positions=[]
                    for i in range(clip['length']):
                        original_data.qpos[:]=np.r_[original['ref_root_pos'][i],original['ref_root_rot'][i,[3,0,1,2]],original['ref_dof_pos'][i]]
                        mujoco.mj_forward(robot.model,original_data)
                        position=original_data.xpos[body].copy()
                        rotation=desired[side][i].as_matrix()
                        position[2]=float(np.abs((rotation@local_rotation.reshape(3,3))[2])@robot.model.geom_size[geom]-(rotation@robot.model.geom_pos[geom])[2])
                        positions.append(position)
                    contact_targets[side]=(grounded,np.asarray(positions))
            update=robot.update_targets
            def native_targets(*values,**options):
                update(*values,**options)
                for table in [robot.human_body_to_task1,robot.human_body_to_task2]:
                    for side in ['left','right']:
                        task=table[side+'_foot']
                        position=task.transform_target_to_world.translation()
                        if side in contact_targets and contact_targets[side][0][frame]:
                            task.set_position_cost(a.source_grounded_foot_position_cost)
                            position=contact_targets[side][1][frame]
                        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(desired[side][frame].as_matrix()),position))
                    if reference_root is not None:
                        root_task=table['pelvis']
                        root_task.set_position_cost(a.reference_root_position_cost)
                        root_task.set_target(mink.SE3.from_rotation_and_translation(root_task.transform_target_to_world.rotation(),reference_root[frame]))
            robot.update_targets=native_targets
        if a.ankle_posture_cost:
            costs=np.zeros(robot.model.nv)
            for j in range(robot.model.njnt):
                if 'ankle' in robot.model.joint(j).name:
                    costs[robot.model.jnt_dofadr[j]]=a.ankle_posture_cost
            for tasks in [robot.tasks1,robot.tasks2]:
                task=mink.PostureTask(robot.model,cost=costs)
                task.set_target(robot.model.qpos0.copy())
                tasks.append(task)
        def override(*values,**options):
            nonlocal frame
            for label,table in [('ik_match_table1',robot.human_body_to_task1),('ik_match_table2',robot.human_body_to_task2)]:
                for name,entry in config[label].items():
                    if 'ankle_roll' in name:table[entry[0]].set_orientation_cost(a.foot_orientation_cost)
            result=retarget(*values,**options)
            frame+=1
            return result
        robot.retarget=override
        return robot
    guard.probe.create_k1_retargeter=make
    compare.worker(argparse.Namespace(output=a.output,worker='old_root_weight',max_frames=10000000))


if __name__=='__main__':main()
