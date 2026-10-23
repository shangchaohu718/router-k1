"""CMU ASF foot rotations expressed in K1 forward/left/up convention."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.booster_k1_retarget.probe_native_position_retarget import load_cmu_fk, YUP_TO_ZUP


def foot_targets(metadata, length):
    rotations, _, _, _ = load_cmu_fk(metadata['source_path'])
    start=metadata['canonical_frame_start'];ratio=float(metadata['source_fps'])/50
    t=(start+np.arange(length))*ratio
    # ASF neutral frame: X left, Y up, Z forward; K1: X forward, Y left, Z up.
    neutral=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    result={}
    for side,short in [('left','l'),('right','r')]:
        source=rotations[short+'foot'];last=len(source)-1
        if t.min()<0 or t.max()>last+max(1,ratio):raise ValueError('CMU foot time correspondence out of bounds')
        r=Slerp(np.arange(len(source)),source)(np.clip(t,0,last))
        matrices=YUP_TO_ZUP.as_matrix()@r.as_matrix()@neutral
        result[side]=Rotation.from_matrix(matrices)
    return result
