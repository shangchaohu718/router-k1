# Booster K1 retarget tools

This directory contains the local LaFAN1 BVH -> GMR -> Booster K1 pipeline.
It writes the HoloMotion robot HDF5 v2 schema directly; HoloSMPL and an SMPL
body-model file are not required for this route.

## Main files

- `bvh_to_robot_dataset.py`: batch converter from LaFAN1 BVH to K1 HDF5 v2.
- `configs/bvh_lafan1_to_official_k1.json`: LaFAN1 skeleton to official
  K1 22-DoF IK mapping.
- `render_k1_hdf5.py`: headless MuJoCo preview renderer for one output clip.
- `play_k1_hdf5_playlist.py`: interactive real-time MuJoCo playlist player.
- `vendor/general_motion_retargeting_k1/`: vendored GMR implementation and
  its MIT license.

The old SMPL-X/HoloSMPL reference scripts remain in this folder for study,
but they are not used by the direct BVH pipeline.

## Dataset split

The converter supports the following `--subject-split` values:

- `train`: subjects 1-4
- `val`: subject 5
- `all`: all subjects

Subject 5 is kept separate because the official LaFAN1 evaluation protocol
uses Subject 5 for evaluation windows.

## Generate the datasets

Run from the repository root with the HoloMotion training environment:

```bash
/home/tcl/miniconda3/envs/holomotion_train/bin/python \
  tools/booster_k1_retarget/bvh_to_robot_dataset.py \
  --src-folder data/lafan1_raw_bvh \
  --output-root data/lafan1_booster_k1_gmr_h5_train_subject1_4 \
  --subject-split train

/home/tcl/miniconda3/envs/holomotion_train/bin/python \
  tools/booster_k1_retarget/bvh_to_robot_dataset.py \
  --src-folder data/lafan1_raw_bvh \
  --output-root data/lafan1_booster_k1_gmr_h5_val_subject5 \
  --subject-split val
```

The converter deliberately refuses an output directory that already exists,
so an accidental rerun cannot overwrite a completed dataset.

## Current validated outputs

- Train: 64 clips, 677,979 frames at 50 Hz.
- Validation: 13 clips, 149,745 frames at 50 Hz.
- Each frame contains root position `(3)`, root quaternion `(4, xyzw)`, and
  K1 joint position `(22)` in the official XML joint order.
- All values are finite, quaternions are normalized, and joint values are
  within the official MuJoCo limits.

## Known limitations

- The current mapping drives the trunk, legs, and arms. K1's two head joints
  remain at zero because LaFAN1 does not provide a reliable head target for
  the initial mapping.
- Mapping quality has been checked with walk, dance, and fall/get-up previews,
  but it remains an initial IK mapping rather than a manufacturer-calibrated
  retarget profile.
- Some floor motions put the retargeted root slightly below zero. Decide on a
  consistent floor-alignment policy before a final production training run.
- The direct BVH mapping does not use HoloSMPL's learned human-motion cleanup,
  so noisy source motion remains the responsibility of this pipeline.

## Training entry

The K1/LaFAN training config is:

`holomotion/config/training/motion_tracking/motion_tracking_booster_k1_lafan.yaml`

Its HDF5 roots, 22-DoF robot config, dynamic observation dimensions, and CPU
reference FK path have been validated. To inspect the launch command without
starting Isaac Lab:

```bash
HOLOMOTION_TRAIN_DRY_RUN=1 NUM_ENVS=32 \
  CONFIG_NAME=motion_tracking_booster_k1_lafan \
  bash holomotion/scripts/training/train_motion_tracking.sh
```

For an initial GPU smoke test, use a small environment count and one learning
iteration before committing to a full run:

```bash
NUM_ENVS=32 CONFIG_NAME=motion_tracking_booster_k1_lafan \
  bash holomotion/scripts/training/train_motion_tracking.sh \
  algo.config.num_learning_iterations=1
```

Periodic MuJoCo sim-to-sim evaluation is disabled in this config until its
G1-specific wrapper is adapted for K1.

## Play every clip in MuJoCo

Play the 13 validation clips in real time, automatically advancing after each
clip:

```bash
/home/tcl/miniconda3/envs/holomotion_train/bin/python \
  tools/booster_k1_retarget/play_k1_hdf5_playlist.py \
  --dataset-root data/lafan1_booster_k1_gmr_h5_val_subject5
```

Play the complete 64-clip training set:

```bash
/home/tcl/miniconda3/envs/holomotion_train/bin/python \
  tools/booster_k1_retarget/play_k1_hdf5_playlist.py \
  --dataset-root data/lafan1_booster_k1_gmr_h5_train_subject1_4
```

Useful options:

- `--pattern 'walk*'`: play only matching motion names.
- `--speed 0.5`: half speed; use `--speed 2` for double speed.
- `--start-index 10`: start at the 11th selected clip.
- `--max-clips 5`: stop the playlist selection after five clips.
- `--loop`: restart automatically after the final clip.
- `--fixed-camera`: disable root-following camera movement.
- `--list-only`: show the selected names without opening MuJoCo.

The terminal prints the current motion name, frame count, FPS, and duration.
Close the viewer window or press `Ctrl+C` in the terminal to stop playback.
