"""Bounded target IK and checked joint interpolation for policy keyposes.

This optional executor resolves wrist branches before moving. It uses only the
current physical state, the predicted Cartesian action, and model joint limits.
Candidate paths are checked in an isolated MjData; the real robot moves through
its existing position actuators. No demonstration joint targets are read.
"""

from itertools import product
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _seeds(current, limits, random):
    """Generic current/centre seeds plus both wrist branches and random restarts."""
    for origin in (current, limits.mean(axis=1)):
        for wrist_offset in (0.0, -np.pi, np.pi):
            seed = origin.copy()
            seed[-1] += wrist_offset
            yield np.clip(seed, limits[:, 0] + 1e-7, limits[:, 1] - 1e-7)
    for _ in range(8):
        yield random.uniform(limits[:, 0], limits[:, 1])


def _robot_geoms(model, controller):
    roots = {model.body(f"link0_{side}").id for side in controller.adapter.config["arms"]}
    result = set()
    for geom_id, body_id in enumerate(model.geom_bodyid):
        while body_id:
            if body_id in roots:
                result.add(geom_id)
                break
            body_id = model.body_parentid[body_id]
    return result


def plan_target_joints(controller, action):
    """Find a reachable joint target whose sampled straight path avoids overlap.

    This is a checked straight joint path, not a general obstacle-routing motion
    planner. Failure is explicit if no candidate passes. Contact checking uses
    the physical model's collision filters and allows at most 1 mm penetration.
    """
    started = time.monotonic()
    model, actual = controller.model, controller.data
    targets = controller.adapter.action_to_sites(action)
    probe = mujoco.MjData(model)
    probe.qpos[:] = actual.qpos
    probe.mocap_pos[:] = actual.mocap_pos
    probe.mocap_quat[:] = actual.mocap_quat
    current = actual.qpos.copy()
    candidates, solver_details = {}, {}
    random = np.random.default_rng(0)
    for side, target in zip(controller.adapter.config["arms"], targets):
        joint = controller.joints[side]
        site_id = model.site(f"gripper_{side}").id

        def residual(q):
            probe.qpos[joint["qpos"]] = q
            mujoco.mj_kinematics(model, probe)
            orientation = probe.site_xmat[site_id].reshape(3, 3)
            return np.r_[target[:3, 3] - probe.site_xpos[site_id],
                         .2 * Rotation.from_matrix(target[:3, :3] @ orientation.T).as_rotvec()]

        solutions, attempts = [], []
        for seed_index, seed in enumerate(_seeds(current[joint["qpos"]], joint["limits"], random)):
            fit = least_squares(residual, seed, bounds=joint["limits"].T,
                                max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            error = residual(fit.x)
            position_error = float(np.linalg.norm(error[:3]))
            rotation_error = float(np.linalg.norm(error[3:]) / .2)
            accepted = position_error < .001 and rotation_error < .01
            attempts.append({"seed_index": seed_index, "position_error_m": position_error,
                             "rotation_error_rad": rotation_error, "accepted": accepted,
                             "nfev": fit.nfev})
            if accepted and all(np.linalg.norm(fit.x - previous) > .05 for previous in solutions):
                solutions.append(fit.x.copy())
            # Six deterministic seeds normally cover both wrist branches.
            # Random restarts are reserved for harder targets.
            if seed_index >= 5 and len(solutions) >= 3:
                break
        candidates[side] = sorted(solutions, key=lambda q: np.linalg.norm(q - current[joint["qpos"]]))
        solver_details[side] = attempts
        probe.qpos[:] = current
    metadata = {
        "backend": "bounded_target_ik_joint_servos",
        "seed_source": "current_joints_joint_limit_midpoint_wrist_offsets_and_fixed_rng",
        "expert_actions_used": False,
        "ik_attempts": solver_details,
        "collision_check": {"maximum_joint_sample_spacing_rad": .03,
                            "maximum_allowed_penetration_m": .001,
                            "method": "sampled_straight_joint_path_in_physical_model"},
    }
    if any(not choices for choices in candidates.values()):
        metadata.update(planned=False, failure="no_bounded_ik_solution",
                        planning_seconds=time.monotonic() - started)
        return None, metadata
    sides = controller.adapter.config["arms"]
    combinations = list(product(*(candidates[side] for side in sides)))
    combinations.sort(key=lambda choice: sum(
        np.linalg.norm(q - current[controller.joints[side]["qpos"]])
        for side, q in zip(sides, choice)))
    robot_geoms = _robot_geoms(model, controller)
    rejected = []
    for combination in combinations:
        goal = current.copy()
        for side, q in zip(sides, combination):
            goal[controller.joints[side]["qpos"]] = q
        count = max(2, int(np.ceil(np.max(np.abs(goal - current)) / .03)) + 1)
        collision = None
        previous_sites = None
        maximum_position_derivative = 0.0
        maximum_rotation_derivative = 0.0
        for fraction in np.linspace(0, 1, count):
            probe.qpos[:] = current + fraction * (goal - current)
            mujoco.mj_forward(model, probe)
            sites = [(probe.site(f"gripper_{side}").xpos.copy(),
                      probe.site(f"gripper_{side}").xmat.reshape(3, 3).copy()) for side in sides]
            if previous_sites is not None:
                for (position, rotation), (old_position, old_rotation) in zip(sites, previous_sites):
                    maximum_position_derivative = max(maximum_position_derivative,
                        np.linalg.norm(position - old_position) * (count - 1))
                    maximum_rotation_derivative = max(maximum_rotation_derivative,
                        np.linalg.norm(Rotation.from_matrix(rotation @ old_rotation.T).as_rotvec()) * (count - 1))
            previous_sites = sites
            for contact in probe.contact:
                if (contact.dist < -.001 and
                        (contact.geom1 in robot_geoms or contact.geom2 in robot_geoms)):
                    collision = {"fraction": float(fraction), "distance_m": float(contact.dist),
                                 "geoms": [model.geom(contact.geom1).name, model.geom(contact.geom2).name]}
                    break
            if collision is not None:
                break
        if collision is not None:
            rejected.append(collision)
            continue
        result = {side: q.copy() for side, q in zip(sides, combination)}
        duration = max(float(np.max(np.abs(goal - current))) / controller.config["max_joint_speed"],
                       maximum_position_derivative / controller.config["max_linear_speed"],
                       maximum_rotation_derivative / controller.config["max_angular_speed"],
                       controller.config["period"])
        metadata.update(planned=True, joint_targets={side: q.tolist() for side, q in result.items()},
                        path_samples=count, rejected_paths=rejected,
                        minimum_reference_duration_s=float(duration),
                        planning_seconds=time.monotonic() - started)
        return result, metadata
    metadata.update(planned=False, failure="all_straight_joint_paths_collide",
                    rejected_paths=rejected, planning_seconds=time.monotonic() - started)
    return None, metadata


def execute_target_ik(controller, action, callback=None):
    """Execute a predicted target with bounded existing servos, then polish IK."""
    joint_targets, metadata = plan_target_joints(controller, action)
    targets = controller.adapter.action_to_sites(action)
    reached, ticks = False, 0
    if joint_targets is not None:
        period = controller.config["period"]
        maximum_ticks = int(controller.config["max_seconds"] / period)
        origins = {side: controller.data.qpos[controller.joints[side]["qpos"]].copy()
                   for side in joint_targets}
        deltas = {side: goal - origins[side] for side, goal in joint_targets.items()}
        progress = 0.0
        progress_increment = period / metadata["minimum_reference_duration_s"]
        polishing = False
        # The collision-checked path starts at actual joints, not the previous
        # controller references. Only controls change here; physical qpos never
        # changes outside mj_step.
        for side, origin in origins.items():
            controller.data.ctrl[controller.joints[side]["act"]] = origin
        for ticks in range(maximum_ticks):
            errors = controller.errors(targets)
            reached = all(np.linalg.norm(p) < controller.config["position_tolerance"]
                          and np.linalg.norm(r) < controller.config["rotation_tolerance"]
                          for p, r in errors)
            if reached:
                break
            polishing |= progress >= 1.0
            if polishing:
                controller.tick(targets, callback)
                continue
            # A common progress limit keeps references on the checked path and
            # bounds their lead over actual joints without per-joint clipping.
            upper_progress = 1.0
            for side, delta in deltas.items():
                joint = controller.joints[side]
                moving = np.abs(delta) > 1e-12
                lead_limit = (controller.data.qpos[joint["qpos"]][moving] +
                              .1 * np.sign(delta[moving]) - origins[side][moving]) / delta[moving]
                if len(lead_limit):
                    upper_progress = min(upper_progress, float(np.min(lead_limit)))
            progress = max(progress, min(progress + progress_increment, upper_progress))
            for side, delta in deltas.items():
                controller.data.ctrl[controller.joints[side]["act"]] = origins[side] + progress * delta
            for _ in range(controller.substeps):
                mujoco.mj_step(controller.model, controller.data)
                if not np.isfinite(controller.data.qpos).all() or not np.isfinite(controller.data.qvel).all():
                    raise RuntimeError("MuJoCo produced a non-finite state")
                if callback:
                    callback()
            mujoco.mj_forward(controller.model, controller.data)
        if reached:
            for index, side in enumerate(controller.adapter.config["arms"]):
                for finger in ("left", "right"):
                    controller.data.actuator(f"drive_{finger}_finger_{side}").ctrl[0] = (
                        .04 if action[index, 7] >= .5 else 0.0)
            for _ in range(round(controller.config["gripper_seconds"] / period)):
                controller.tick(targets, callback)
    errors = controller.errors(targets)
    return {"reached": reached, "control_ticks": ticks,
            "position_errors_m": [float(np.linalg.norm(p)) for p, _ in errors],
            "rotation_errors_rad": [float(np.linalg.norm(r)) for _, r in errors],
            "target_ik": metadata}
