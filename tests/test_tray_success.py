"""Tray success checks against native sensor evidence and independent geometry."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from mujoco_transfer.geometry import matrix_pose, pose_matrix
from mujoco_transfer.tray_success import evaluate_tray_success, oriented_box_intersection


SOURCE = json.loads((Path(__file__).resolve().parents[1] / "mujoco_transfer" /
                     "configs" / "peract2_tray_scene.json").read_text())


def transform(position=(0, 0, 0), rotation=(0, 0, 0)):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    return result


@pytest.mark.parametrize("offset,expected", [(0, True), (.2, True), (.20000001, False)])
def test_oriented_boxes_include_touching_but_not_a_gap(offset, expected):
    result = oriented_box_intersection(np.eye(4), [.1] * 3,
                                       transform([offset, 0, 0]), [.1] * 3)
    assert result["intersects"] is expected


def test_skew_edges_require_cross_product_axes():
    # These slender skew boxes overlap along all six face-normal axes but
    # remain separated along an edge/edge axis. A six-axis check is insufficient.
    second = np.array([
        [-.08038473974946903, -.741070363662588, -.6665980870932725, .10879996512734091],
        [.7477963739586985, .3973381029952405, -.5319050808202218, .32159986357657844],
        [.659043911063317, -.5412366839093076, .522229810793317, -.13107378381456702],
        [0, 0, 0, 1],
    ])
    result = oriented_box_intersection(np.eye(4), [.4, .06, .06],
                                       second, [.4, .06, .06])
    assert not result["intersects"]
    assert result["tested_axes"] == 15
    assert result["separation_m"] > .14


def test_sat_agrees_with_independent_linear_feasibility():
    # Intersection also means there exist two bounded local points whose world
    # positions coincide. This linear-program oracle does not use SAT axes.
    random = np.random.default_rng(57)
    outcomes = []
    for _ in range(40):
        first = transform(random.uniform(-.2, .2, 3), random.normal(size=3))
        second = transform(random.uniform(-.2, .2, 3), random.normal(size=3))
        first_size, second_size = random.uniform(.01, .2, (2, 3))
        result = linprog(
            np.zeros(6), A_eq=np.c_[first[:3, :3], -second[:3, :3]],
            b_eq=second[:3, 3] - first[:3, 3],
            bounds=[(-value, value) for value in np.r_[first_size, second_size]],
            method="highs")
        assert result.status in (0, 2), result.message
        actual = oriented_box_intersection(first, first_size, second, second_size)
        assert actual["intersects"] == result.success
        outcomes.append(actual["intersects"])
    assert any(outcomes) and not all(outcomes)


@pytest.mark.parametrize("case", SOURCE["native_detection_checks"],
                         ids=lambda case: f"{case['sensor']}_{case['tray_pose_in_sensor_xyzw']}")
def test_source_native_detection_cases(case):
    specification = SOURCE["tray_proximity_sensors"][case["sensor"]]
    tray = pose_matrix(case["tray_pose_in_sensor_xyzw"])
    sensor = transform(specification["center_local_m"])
    result = oriented_box_intersection(
        tray, SOURCE["tray_collision_box"]["half_size_m"],
        sensor, specification["half_size_m"])
    assert result["intersects"] is case["native_detected"]


def make_environment():
    config = {name: deepcopy(SOURCE[name]) for name in (
        "success_condition", "tray_collision_box", "tray_proximity_sensors")}
    positions = {
        "tray": SimpleNamespace(qpos=np.array([0., 0., 1.2, 1., 0., 0., 0.])),
        "item": SimpleNamespace(qpos=np.array([0., 0., 1.25, 1., 0., 0., 0.])),
    }
    states = []
    for side, x in (("left", -.10), ("right", .10)):
        # Place sensor origins 2 cm below the tray, then recover each physical
        # TCP using the independently audited sensor mounting transform.
        sensor = transform([x, 0, 1.18])
        tcp = sensor @ np.linalg.inv(np.array(
            config["tray_proximity_sensors"][side]["T_policy_tcp_from_sensor"]))
        states.append(np.r_[matrix_pose(tcp), 1.0])
    states = np.array(states)
    return SimpleNamespace(
        config=config,
        data=SimpleNamespace(joint=lambda name: positions[name]),
        adapter=SimpleNamespace(world_transform=np.eye(4), state=lambda: states),
    )


def test_success_requires_both_heights_and_both_sensors_without_dwell():
    env = make_environment()
    state = env.adapter.state().copy()
    result = evaluate_tray_success(env)
    assert result["success"]
    assert result["tray_height_m"] == 1.2
    assert result["left_detected"] and result["right_detected"]
    # Opening commands and contact forces are not conditions in the source task.
    assert np.all(state[:, 7] == 1.0)
    np.testing.assert_array_equal(env.adapter.state(), state)
    assert result["sensors"]["left"]["volume_min_sensor_m"][2] == 0


@pytest.mark.parametrize("object_name", ["tray", "item"])
def test_either_object_below_threshold_prevents_success(object_name):
    env = make_environment()
    env.data.joint(object_name).qpos[2] = 1.2 - 1e-6
    result = evaluate_tray_success(env)
    assert not result["success"]
    assert not result[f"{object_name}_lifted"]
    assert result["left_detected"] and result["right_detected"]


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 1)])
def test_one_missing_hand_prevents_success(side, index):
    env = make_environment()
    env.adapter.state()[index, 0] += 1.0
    result = evaluate_tray_success(env)
    assert not result["success"]
    assert not result[f"{side}_detected"]
    assert result["tray_lifted"] and result["item_lifted"]


def test_tray_behind_positive_z_sensor_is_not_detected():
    env = make_environment()
    env.adapter.state()[0, 2] += .03
    result = evaluate_tray_success(env)
    assert not result["left_detected"]
    assert result["right_detected"]
    assert result["sensors"]["left"]["tray_bounds_max_sensor_m"][2] < 0


def test_world_transform_applies_to_both_free_objects_and_heights():
    env = make_environment()
    world = transform([.5, -.6, .8], [0, 0, .7])
    env.adapter.world_transform = world
    for name in ("tray", "item"):
        pose = env.data.joint(name).qpos
        source = pose_matrix(pose[[0, 1, 2, 4, 5, 6, 3]])
        physical = matrix_pose(np.linalg.inv(world) @ source)
        env.data.joint(name).qpos[:] = physical[[0, 1, 2, 6, 3, 4, 5]]
    result = evaluate_tray_success(env)
    assert result["success"]
    assert result["tray_height_m"] == pytest.approx(1.2)
    assert result["item_height_m"] == pytest.approx(1.25)


@pytest.mark.parametrize("half_size", [[0, 1, 1], [-1, 1, 1], [1, np.nan, 1], [1, 1]])
def test_invalid_geometry_fails_instead_of_reporting_success(half_size):
    with pytest.raises(ValueError, match="half sizes"):
        oriented_box_intersection(np.eye(4), half_size, np.eye(4), [1, 1, 1])
