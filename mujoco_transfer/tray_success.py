"""Evaluate the source tray task using object heights and hand sensor volumes.

The source does not require contact forces or a commanded closed gripper for
success. Its two rectangular proximity sensors must both detect the tray while
the tray and its independent item are at least 1.2 m high. Sensor transforms
are attached to the actual physical policy TCPs, never the desired actions.
"""

import numpy as np

from .geometry import pose_matrix, rigid_transform


def _half_size(value):
    half_size = np.asarray(value, dtype=float)
    if (half_size.shape != (3,) or not np.isfinite(half_size).all()
            or np.any(half_size <= 0)):
        raise ValueError("Box half sizes must be three finite positive metres")
    return half_size


def oriented_box_intersection(first_transform, first_half_size,
                              second_transform, second_half_size,
                              *, tolerance_m=1e-9):
    """Intersect two oriented boxes with all 15 separating-axis candidates.

    Transforms map box-centred coordinates into a shared world frame. Unlike
    centre-distance or world-AABB tests, this also checks the nine cross-product
    axes needed for skew box edges. ``separation_m`` is the largest projected
    gap, not a Euclidean object distance; positive means separated and negative
    means overlap along every tested axis. Only round-off tolerance is added.
    """
    first = rigid_transform(first_transform)
    second = rigid_transform(second_transform)
    first_size, second_size = _half_size(first_half_size), _half_size(second_half_size)
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Intersection tolerance must be finite and nonnegative")
    first_axes, second_axes = first[:3, :3], second[:3, :3]
    axes = [*first_axes.T, *second_axes.T]
    for first_axis in first_axes.T:
        for second_axis in second_axes.T:
            crossed = np.cross(first_axis, second_axis)
            norm = np.linalg.norm(crossed)
            if norm > 1e-12:
                axes.append(crossed / norm)
    axes = np.asarray(axes)
    offset = second[:3, 3] - first[:3, 3]
    first_radius = np.abs(axes @ first_axes) @ first_size
    second_radius = np.abs(axes @ second_axes) @ second_size
    gaps = np.abs(axes @ offset) - first_radius - second_radius
    index = int(np.argmax(gaps))
    gap = float(gaps[index])
    return {
        "intersects": bool(gap <= tolerance_m),
        "separation_m": gap,
        "limiting_axis_world": axes[index].tolist(),
        "center_distance_m": float(np.linalg.norm(offset)),
        "tested_axes": len(axes),
        "tolerance_m": float(tolerance_m),
    }


def _free_body_transform(env, joint_name):
    # Free-joint qpos is current after mj_step; body.xpos may still precede the
    # integration. Free joints are world-rooted, and MuJoCo uses WXYZ quaternions.
    qpos = np.asarray(env.data.joint(joint_name).qpos, dtype=float)
    if qpos.shape != (7,):
        raise ValueError(f"Expected free-joint state for {joint_name}")
    return rigid_transform(env.adapter.world_transform) @ pose_matrix(
        qpos[[0, 1, 2, 4, 5, 6, 3]])


def evaluate_tray_success(env):
    """Return instantaneous source-task conditions without changing physics.

    The caller must update current site kinematics before calling this after
    ``mj_step``. Geometry comes from the independently audited source fixture,
    copied into ``tray_collision_box`` and ``tray_proximity_sensors`` in the
    runtime config. Heights use source-world coordinates, including any world
    calibration transform; the item remains independent of the tray.
    """
    condition = env.config["success_condition"]
    if condition["kind"] != "tray_lift":
        raise ValueError("Tray success evaluator requires the tray_lift condition")
    minimum_height = float(condition["minimum_height_m"])
    if not np.isfinite(minimum_height):
        raise ValueError("Tray height threshold must be finite")
    tray = _free_body_transform(env, "tray")
    item = _free_body_transform(env, "item")
    geometry = env.config["tray_collision_box"]
    tray_box = tray @ rigid_transform(geometry["T_tray_from_box"])
    tray_size = _half_size(geometry["half_size_m"])
    state = np.asarray(env.adapter.state(), dtype=float)
    if state.shape != (2, 8) or not np.isfinite(state).all():
        raise ValueError("Expected finite left/right physical policy states (2, 8)")
    corners = np.array([[x, y, z] for x in (-1, 1)
                        for y in (-1, 1) for z in (-1, 1)]) * tray_size
    corners_world = corners @ tray_box[:3, :3].T + tray_box[:3, 3]
    sensors = {}
    for index, side in enumerate(("left", "right")):
        specification = env.config["tray_proximity_sensors"][side]
        sensor = pose_matrix(state[index, :7]) @ rigid_transform(
            specification["T_policy_tcp_from_sensor"])
        center = np.asarray(specification["center_local_m"], dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Sensor local centre must be three finite metres")
        size = _half_size(specification["half_size_m"])
        sensor_box = sensor.copy()
        sensor_box[:3, 3] += sensor[:3, :3] @ center
        result = oriented_box_intersection(tray_box, tray_size, sensor_box, size)
        corners_sensor = (corners_world - sensor[:3, 3]) @ sensor[:3, :3]
        result.update({
            "detected": result["intersects"],
            "volume_min_sensor_m": (center - size).tolist(),
            "volume_max_sensor_m": (center + size).tolist(),
            "tray_bounds_min_sensor_m": corners_sensor.min(axis=0).tolist(),
            "tray_bounds_max_sensor_m": corners_sensor.max(axis=0).tolist(),
            "T_policy_world_from_sensor": sensor.tolist(),
        })
        sensors[side] = result
    tray_height, item_height = float(tray[2, 3]), float(item[2, 3])
    tray_lifted, item_lifted = tray_height >= minimum_height, item_height >= minimum_height
    left, right = sensors["left"]["detected"], sensors["right"]["detected"]
    return {
        "success": bool(tray_lifted and item_lifted and left and right),
        "tray_height_m": tray_height,
        "item_height_m": item_height,
        "minimum_height_m": minimum_height,
        "tray_lifted": bool(tray_lifted),
        "item_lifted": bool(item_lifted),
        "left_detected": left,
        "right_detected": right,
        "sensors": sensors,
        "geometry_method": "15_axis_oriented_box_intersection",
        "state_source": "physical_free_joints_and_actual_policy_tcp_poses",
    }
