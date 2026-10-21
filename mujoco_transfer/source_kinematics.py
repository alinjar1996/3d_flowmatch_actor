"""Source robot kinematics for observation rendering, never action execution.

The branch's Panda geometry differs from the source robot.  Its current joint
angles seed an inverse-kinematics solve that makes the rendered source hand
agree with the actual MuJoCo policy TCP.  MuJoCo remains the physics authority.
"""

import io
import zipfile

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import pose_matrix


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0, 0], [sine, cosine, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def source_camera_calibration(archive, episode=3, task="bimanual_lift_ball"):
    """Read only fixed camera calibration, not rendered observations or actions."""
    from .reference import NumericReferenceUnpickler
    from .tasks import task_spec

    task_spec(task)
    member = (f"peract2_test/{task}/variation0/episodes/"
              f"episode{episode}/low_dim_obs.pkl")
    with zipfile.ZipFile(archive) as dataset:
        demo = NumericReferenceUnpickler(io.BytesIO(dataset.read(member))).load()
    observation = demo._observations[0]
    cameras = {}
    for name in ("front", "wrist_left", "wrist_right"):
        world = np.asarray(observation.misc[name + "_camera_extrinsics"], float)
        mount = world
        if name != "front":
            side = name.split("_")[1]
            mount = np.linalg.inv(pose_matrix(getattr(observation, side).gripper_pose)) @ world
        cameras[name] = {
            "matrix": mount.tolist(),
            "frame": "world" if name == "front" else "policy_tcp",
            "near": float(observation.misc[name + "_camera_near"]),
            "far": float(observation.misc[name + "_camera_far"]),
            "fovy": float(np.rad2deg(2 * np.arctan(
                128 / abs(observation.misc[name + "_camera_intrinsics"][1, 1])))),
        }
    return {"task": task, "episode": episode, "member": member, "cameras": cameras}


class SourceArmKinematics:
    def __init__(self, fixed_transforms, tip_transform, limits):
        self.fixed = np.asarray(fixed_transforms, float)
        self.tip = np.asarray(tip_transform, float)
        self.limits = np.asarray(limits, float)
        if self.fixed.shape != (7, 4, 4) or self.tip.shape != (4, 4):
            raise ValueError("Source arm requires seven fixed transforms and a TCP transform")
        if self.limits.shape != (7, 2) or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Source arm requires seven nonempty joint limits")

    def forward(self, joints):
        transform = np.eye(4)
        for fixed, angle in zip(self.fixed, joints):
            transform = transform @ fixed @ rotation_z(angle)
        return transform @ self.tip

    def solve(self, target, seed):
        target, seed = np.asarray(target, float), np.asarray(seed, float)
        if target.shape != (4, 4) or seed.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Invalid TCP target or joint seed")

        def residual(joints):
            current = self.forward(joints)
            rotation = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            return np.r_[current[:3, 3] - target[:3, 3], 0.2 * rotation]

        start = np.clip(seed, self.limits[:, 0] + 1e-9, self.limits[:, 1] - 1e-9)
        result = least_squares(residual, start, bounds=self.limits.T,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=200)
        final = residual(result.x)
        position_error, rotation_error = np.linalg.norm(final[:3]), np.linalg.norm(final[3:]) / 0.2
        return result.x, {
            "position_error_m": float(position_error),
            "rotation_error_rad": float(rotation_error),
            "aligned": bool(position_error <= 0.002 and rotation_error <= 0.02),
            "fallback": None if position_error <= 0.002 and rotation_error <= 0.02
            else "best_bounded_native_ik_solution",
            "joint_delta_l2_rad": float(np.linalg.norm(result.x - seed)),
            "nfev": int(result.nfev),
            "optimizer_success": bool(result.success),
        }
