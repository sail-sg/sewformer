from scipy.spatial.transform import Rotation as R
from math import degrees

def euler_angle_to_rot_6d(pose_angle):
    rot_mat = R.from_euler('xyz', pose_angle, degrees=True)
    rot_mat = rot_mat.as_matrix()
    return rot_mat[:, :2].reshape(-1)