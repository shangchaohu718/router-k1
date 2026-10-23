import numpy as np
from scipy.spatial.transform import Rotation

from tools.booster_k1_retarget import cmu_native_foot_pose as foot


def test_cmu_neutral_foot_axes_are_proper_not_mirrored(monkeypatch):
    rotations = {side+'foot': Rotation.identity(2) for side in ['l','r']}
    monkeypatch.setattr(foot,'load_cmu_fk',lambda _: (rotations,None,None,None))
    result = foot.foot_targets({'source_path':'test.amc','canonical_frame_start':0,'source_fps':50},2)
    expected = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    for side in ['left','right']:
        np.testing.assert_allclose(result[side].as_matrix(),np.stack([expected,expected]),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(result[side].as_matrix()),1,atol=1e-12)


def test_native_foot_rotation_preserves_time_correspondence(monkeypatch):
    source = Rotation.from_euler('y',[0,30,60],degrees=True)
    monkeypatch.setattr(foot,'load_cmu_fk',lambda _: ({'lfoot':source,'rfoot':source},None,None,None))
    result = foot.foot_targets({'source_path':'test.amc','canonical_frame_start':1,'source_fps':50},2)
    neutral = np.array([[0,1,0],[0,0,1],[1,0,0]])
    expected = foot.YUP_TO_ZUP.as_matrix()@source[1:].as_matrix()@neutral
    np.testing.assert_allclose(result['left'].as_matrix(),expected,atol=1e-12)
