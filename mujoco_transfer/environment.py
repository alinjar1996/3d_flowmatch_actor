"""MuJoCo-native observations and dynamics for the exported Panda task.

This module has no RLBench, PyRep, ROS, or MJX dependency. Its observation
adapter can also wrap the same MjModel/MjData used by manipulator_mujoco.
"""

import json
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .geometry import depth_to_world, matrix_pose, pose_matrix, rigid_transform, transform_points


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "panda_ball.json"


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["arms"] != ["left", "right"]:
        raise ValueError("PerAct2 requires policy arm order [left, right]")
    if config["cameras"] != ["front", "wrist_left", "wrist_right"]:
        raise ValueError("PerAct2 requires front, wrist_left, wrist_right cameras")
    if config["image_size"] != 256:
        raise ValueError("The released checkpoint uses 256x256 images")
    rigid_transform(config["T_policy_from_mujoco"])
    for side in config["arms"]:
        rigid_transform(config["T_site_from_policy_gripper"][side])
    return config


class ObservationAdapter:
    def __init__(self, model, data, config):
        self.model, self.data, self.config = model, data, config
        self.world_transform = rigid_transform(config["T_policy_from_mujoco"])
        self.tcp_transforms = {
            side: rigid_transform(config["T_site_from_policy_gripper"][side])
            for side in config["arms"]
        }
        self.renderer = None
        # Fail on missing names, instead of inadvertently selecting object -1.
        for camera in config["cameras"]:
            model.camera(camera)
        for side in config["arms"]:
            model.site(f"gripper_{side}")
            for finger in ("left", "right"):
                model.joint(f"{finger}_finger_{side}")

    def site_transform(self, side):
        site = self.data.site(f"gripper_{side}")
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = site.xmat.reshape(3, 3), site.xpos
        return t

    def state(self):
        states = []
        for side in self.config["arms"]:
            pose = matrix_pose(self.world_transform @ self.site_transform(side)
                               @ self.tcp_transforms[side])
            opening = np.mean([self.data.joint(f"{finger}_finger_{side}").qpos[0]
                               for finger in ("left", "right")]) / 0.04
            states.append(np.r_[pose, np.clip(opening, 0, 1)])
        return np.asarray(states, np.float32)

    def observe(self):
        if self.renderer is None:
            size = self.config["image_size"]
            # MSAA resolves the nearest depth sample, not the pixel-centre ray.
            # Disable it for metrically aligned RGB/point-cloud observations.
            self.model.vis.quality.offsamples = 0
            self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        rgbs, points = [], []
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0  # Hide collision proxies, retain visual meshes.
        options.sitegroup[:] = 0
        for name in self.config["cameras"]:
            cid = self.model.camera(name).id
            if self.model.cam_orthographic[cid] or np.any(self.model.cam_sensorsize[cid]):
                raise ValueError("This adapter expects perspective cameras specified by fovy")
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=name, scene_option=options)
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            world = depth_to_world(depth, self.model.cam_fovy[cid],
                                   self.data.cam_xpos[cid], self.data.cam_xmat[cid])
            rgbs.append(rgb)
            points.append(transform_points(world, self.world_transform))
        return np.stack(rgbs), np.asarray(points, np.float32), self.state()

    def action_to_sites(self, action):
        action = np.asarray(action, float)
        if action.shape != (2, 8) or not np.isfinite(action).all():
            raise ValueError("Action must be finite (left/right, XYZ/XYZW/open)")
        if np.any((action[:, 7] < 0) | (action[:, 7] > 1)):
            raise ValueError("Gripper probabilities must be in [0,1]")
        return [
            np.linalg.inv(self.world_transform) @ pose_matrix(action[i, :7])
            @ np.linalg.inv(self.tcp_transforms[side])
            for i, side in enumerate(self.config["arms"])
        ]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def bounded(vector, limit):
    return vector * min(1, limit / max(np.linalg.norm(vector), 1e-12))


class CartesianController:
    """Damped differential IK with joint limits and physical position servos.

    Each policy action is a keypose, not a joint target or velocity. Both arms
    advance together; grippers change after arrival, as in the source evaluator.
    This is a local controller, not a collision-aware path planner.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.model, self.data = adapter.model, adapter.data
        self.config = adapter.config["controller"]
        self.joints = {}
        for side in adapter.config["arms"]:
            ids = [self.model.joint(f"joint{i}_{side}").id for i in range(1, 8)]
            self.joints[side] = {
                "qpos": self.model.jnt_qposadr[ids], "dof": self.model.jnt_dofadr[ids],
                "limits": self.model.jnt_range[ids],
                "act": [self.model.actuator(f"drive_joint{i}_{side}").id for i in range(1, 8)],
            }
        self.substeps = round(self.config["period"] / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(
                self.substeps * self.model.opt.timestep, self.config["period"]):
            raise ValueError("Controller period must be an integer number of physics steps")
        self.backend = self.config.get("backend", "cartesian")
        if self.backend not in ("cartesian", "bounded_target_ik_initial"):
            raise ValueError(f"Unsupported controller backend: {self.backend}")
        self.reset()

    def reset(self):
        self._initial_target_pending = self.backend == "bounded_target_ik_initial"

    def errors(self, targets):
        errors = []
        for side, target in zip(self.adapter.config["arms"], targets):
            current = self.adapter.site_transform(side)
            pos = target[:3, 3] - current[:3, 3]
            rot = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            errors.append((pos, rot))
        return errors

    def tick(self, targets, callback=None):
        for side, (pos, rot) in zip(self.adapter.config["arms"], self.errors(targets)):
            j = self.joints[side]
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.model.site(f"gripper_{side}").id)
            jac = np.vstack((jp[:, j["dof"]], jr[:, j["dof"]]))
            velocity = np.r_[bounded(4 * pos, self.config["max_linear_speed"]),
                             bounded(4 * rot, self.config["max_angular_speed"])]
            reference = self.data.ctrl[j["act"]]
            speed = self.config["max_joint_speed"]
            lower = np.maximum(-speed, (j["limits"][:, 0] - reference) / self.config["period"])
            upper = np.minimum(speed, (j["limits"][:, 1] - reference) / self.config["period"])
            # Solve with limits in the optimization. Clipping an unconstrained
            # solution stalls at a wrist limit instead of using redundant joints.
            solution = lsq_linear(
                np.vstack((jac, self.config["damping"] * np.eye(7))),
                np.r_[velocity, np.zeros(7)], bounds=(lower, upper), method="bvls")
            dq = solution.x
            # Integrate the servo reference rather than repeatedly erasing its
            # tracking error. Bound its lead when contacts prevent motion.
            q = self.data.ctrl[j["act"]] + dq * self.config["period"]
            q = np.clip(q, self.data.qpos[j["qpos"]] - 0.1, self.data.qpos[j["qpos"]] + 0.1)
            self.data.ctrl[j["act"]] = np.clip(q, j["limits"][:, 0], j["limits"][:, 1])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("MuJoCo produced a non-finite state")
            if callback:
                callback()
        mujoco.mj_forward(self.model, self.data)

    def execute(self, action, callback=None):
        if self._initial_target_pending:
            from .target_ik import execute_target_ik
            self._initial_target_pending = False
            return execute_target_ik(self, action, callback)
        targets = self.adapter.action_to_sites(action)
        reached = False
        count = 0
        for count in range(int(self.config["max_seconds"] / self.config["period"])):
            errors = self.errors(targets)
            reached = all(np.linalg.norm(p) < self.config["position_tolerance"]
                          and np.linalg.norm(r) < self.config["rotation_tolerance"]
                          for p, r in errors)
            if reached:
                break
            self.tick(targets, callback)
        if reached:
            for i, side in enumerate(self.adapter.config["arms"]):
                for finger in ("left", "right"):
                    self.data.actuator(f"drive_{finger}_finger_{side}").ctrl[0] = (
                        0.04 if action[i, 7] >= 0.5 else 0.0)
            for _ in range(round(self.config["gripper_seconds"] / self.config["period"])):
                self.tick(targets, callback)
        errors = self.errors(targets)
        return {"reached": reached, "control_ticks": count,
                "position_errors_m": [float(np.linalg.norm(p)) for p, _ in errors],
                "rotation_errors_rad": [float(np.linalg.norm(r)) for _, r in errors]}


class PandaBallEnvironment:
    def __init__(self, scene, config, *, initial_state=None, reset_sampler=None):
        if initial_state is not None and reset_sampler is not None:
            raise ValueError("Choose a fixed initial state or a reset sampler")
        self.model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        self.data = mujoco.MjData(self.model)
        self.config = config
        self.object_name = config.get("object_name", "ball")
        self.dynamic_object_names = [self.object_name, *config.get("additional_object_names", [])]
        self.static_object_names = config.get("static_object_names", [])
        self.reset_sampler = reset_sampler
        self._fixed_initial_state = deepcopy(initial_state)
        self.adapter = ObservationAdapter(self.model, self.data, config)
        self.controller = CartesianController(self.adapter)
        self.reset()

    def reset(self):
        initial = (self.reset_sampler.sample() if self.reset_sampler is not None
                   else deepcopy(self._fixed_initial_state))
        if initial is not None:
            object_poses = initial.get("object_poses_xyzw")
            if self.object_name == "ball":
                # Keep the established public ball reset key authoritative.
                object_poses = {"ball": initial["ball_pose_xyzw"]}
            if object_poses is None:
                raise ValueError("Initial state has no task object poses")
            for name in (*self.dynamic_object_names, *self.static_object_names):
                pose_matrix(object_poses[name])
            for side in self.config["arms"]:
                joints = np.asarray(initial["joint_positions"][side], dtype=float)
                limits = self.controller.joints[side]["limits"]
                opening = initial["gripper_open"][side]
                if (joints.shape != (7,) or not np.isfinite(joints).all()
                        or np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1])):
                    raise ValueError(f"Source {side} starting joints exceed this robot's limits")
                if not np.isfinite(opening) or not 0 <= opening <= 1:
                    raise ValueError(f"Invalid {side} starting gripper opening")
        mujoco.mj_resetData(self.model, self.data)
        if initial is not None:
            for name in self.dynamic_object_names:
                pose = matrix_pose(pose_matrix(object_poses[name]))
                self.data.joint(name).qpos[:] = pose[[0, 1, 2, 6, 3, 4, 5]]
            for name in self.static_object_names:
                pose = matrix_pose(pose_matrix(object_poses[name]))
                self.model.body(name).pos[:] = pose[:3]
                self.model.body(name).quat[:] = pose[[6, 3, 4, 5]]
        for side in self.config["arms"]:
            home = initial["joint_positions"] if initial is not None else self.config["home_joints"]
            if isinstance(home, dict):
                home = home[side]
            for i, q in enumerate(home, 1):
                self.data.joint(f"joint{i}_{side}").qpos[0] = q
                self.data.actuator(f"drive_joint{i}_{side}").ctrl[0] = q
            for finger in ("left", "right"):
                opening = initial["gripper_open"][side] if initial is not None else 1.0
                self.data.joint(f"{finger}_finger_{side}").qpos[0] = 0.04 * opening
                self.data.actuator(f"drive_{finger}_finger_{side}").ctrl[0] = 0.04 * opening
        # Released episode starts are already settled observations. Do not
        # advance them before the first policy observation or alter their pose.
        settle_seconds = 0.0 if initial is not None else self.config.get("reset_settle_seconds", 0.5)
        for _ in range(round(settle_seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.initial_object_height = float(self.data.body(self.object_name).xpos[2])
        self.initial_ball_height = self.initial_object_height  # Existing ball API.
        self.held_seconds = 0
        self.success = False
        self.last_success_details = {}
        self.first_success = None
        self.initial_state = deepcopy(initial)
        self.controller.reset()

    def track_success(self):
        # mj_step's derived xpos precedes integration, so read the free joint.
        height = float(self.data.joint(self.object_name).qpos[2])
        condition = self.config.get("success_condition")
        if condition is not None:
            if condition["kind"] == "tray_lift":
                from .tray_success import evaluate_tray_success
                mujoco.mj_kinematics(self.model, self.data)
                self.last_success_details = evaluate_tray_success(self)
                if self.last_success_details["success"] and not self.success:
                    self.first_success = {
                        "simulation_time_s": float(self.data.time),
                        "conditions": deepcopy(self.last_success_details),
                        "qpos": self.data.qpos.tolist(),
                        "policy_state": self.adapter.state().tolist(),
                    }
                self.success |= self.last_success_details["success"]
                return
            if condition["kind"] != "world_height":
                raise ValueError(f"Unsupported task success condition: {condition}")
            self.success |= height >= condition["minimum_height_m"]
            return
        # Legacy uncalibrated baseline only.
        if height >= self.initial_ball_height + self.config["success_proxy"]["ball_rise_metres"]:
            self.held_seconds += self.model.opt.timestep
        else:
            self.held_seconds = 0
        self.success |= self.held_seconds >= self.config["success_proxy"]["hold_seconds"]

    def close(self):
        self.adapter.close()
