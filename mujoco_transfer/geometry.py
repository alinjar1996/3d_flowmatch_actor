"""All transforms use metres, column vectors, and XYZW pose quaternions."""

import numpy as np
from scipy.spatial.transform import Rotation


def rigid_transform(value):
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Expected a finite 4x4 rigid transform")
    if not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("Transform rotation must be orthonormal (no scale)")
    if not np.isclose(np.linalg.det(t[:3, :3]), 1):
        raise ValueError("Transform must be a proper rotation (no reflection)")
    return t


def transform_points(points, transform):
    t = rigid_transform(transform)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Pose must be finite XYZ + quaternion XYZW")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    t[:3, 3] = pose[:3]
    return t


def matrix_pose(t):
    t = rigid_transform(t)
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat()]


def depth_to_world(depth, fovy, camera_position, camera_rotation):
    """Unproject mujoco.Renderer metric axial depth, in its top-down row order.

    MuJoCo cameras look down local -Z with +Y up. Pixel centres are half
    integers; no extra flip or OpenGL depth-buffer conversion is needed.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Depth must be a finite positive HxW array in metres")
    if not 0 < fovy < 180:
        raise ValueError("Invalid vertical field of view")
    h, w = depth.shape
    focal = h / (2 * np.tan(np.deg2rad(fovy) / 2))
    v, u = np.mgrid[:h, :w]
    camera_points = np.stack(
        ((u + 0.5 - w / 2) * depth / focal,
         -(v + 0.5 - h / 2) * depth / focal, -depth), axis=-1
    )
    return camera_points @ np.asarray(camera_rotation).reshape(3, 3).T + camera_position


def fit_rigid_transform(source, target):
    """Fit T_target_from_source from >=3 non-collinear corresponding points."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Correspondences must be matching Nx3 arrays")
    if len(source) < 3 or not np.isfinite([source, target]).all():
        raise ValueError("Need at least three finite correspondences")
    a, b = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Calibration points must not be collinear")
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    r = vt.T @ correction @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, target.mean(0) - r @ source.mean(0)
    rms = np.sqrt(np.mean(np.sum((transform_points(source, t) - target)**2, axis=1)))
    return t, float(rms)
