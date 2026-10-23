# 100STYLE adapter

Use `--source style100` with the original BVH archive. `Frame_Cuts.csv`
must be in an ancestor of each BVH. The adapter preserves native YXZ channel
order, computes global rotations, composes the extra chest segment into SMPL
parents, trims using the official zero-based half-open range, and resamples to
50 Hz. Native centimeters/Y-up become meters/Z-up. Hands are identity-filled;
shape is neutral, not fitted. This is approximate skeleton transfer, not SMPL
ground truth. Validate human renders and K1 retargets before training.

Source and attribution: https://www.ianxmason.com/100style/ (CC BY 4.0).
Do not apply LaFAN bone-frame offsets to this skeleton.
