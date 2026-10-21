"""Physical regression tests for the tray's first policy-target wrist branch."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_transfer.environment import PandaBallEnvironment
from mujoco_transfer.target_ik import plan_target_joints


ROOT = Path(__file__).resolve().parents[1]
TRAY = ROOT / "runtime/panda_tray_reference"
# Actual pretrained first prediction from tray episode 0, policy seed 0. These
# are Cartesian policy targets, not a demonstration trajectory or joint seed.
FIRST_ACTION = np.array([
    [.19289685785770416, .20492881536483765, .8565380573272705,
     .7085219621658325, -.023150598630309105, -.02263801358640194, .7049453854560852, 1.],
    [.18972350656986237, -.28092682361602783, .8559184074401855,
     .01993369869887829, .7076845765113831, .7060179710388184, .017992787063121796, 1.],
])


@pytest.fixture
def environment():
    if not (TRAY / "reference_config.json").is_file():
        pytest.skip("Prepare the source tray scene first")
    config = json.loads((TRAY / "reference_config.json").read_text())
    config["controller"]["backend"] = "bounded_target_ik_initial"
    env = PandaBallEnvironment(TRAY / "scene.xml", config)
    yield env
    env.close()


def test_bounded_plan_is_isolated_and_avoids_first_action_wrist_trap(environment):
    env = environment
    before = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    before_time = env.data.time
    targets, metadata = plan_target_joints(env.controller, FIRST_ACTION)
    assert metadata["planned"]
    assert metadata["expert_actions_used"] is False
    assert metadata["path_samples"] > 50
    for side, target in targets.items():
        limits = env.controller.joints[side]["limits"]
        assert np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1])
        assert target[-1] < 0  # The previous local solver stalls at the upper wrist limit.
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
    assert env.data.time == before_time


def test_first_prediction_reached_by_physics_with_bounded_actuators(environment):
    env = environment
    records = []
    # Model a previous servo reference that is ahead of actual joints. Planning
    # must still start from actual joints and execution must not teleport them.
    for joint in env.controller.joints.values():
        env.data.ctrl[joint["act"]] += .03

    def record():
        records.append((env.data.time, env.data.qpos.copy(), env.data.ctrl.copy()))

    initial_qpos = env.data.qpos.copy()
    result = env.controller.execute(FIRST_ACTION, record)
    assert result["reached"], result
    assert max(result["position_errors_m"]) < .01
    assert max(result["rotation_errors_rad"]) < .1
    assert result["target_ik"]["minimum_reference_duration_s"] > 3
    assert len(records) > 1000
    np.testing.assert_allclose(np.diff([row[0] for row in records]), env.model.opt.timestep)
    positions = np.array([initial_qpos, *(row[1] for row in records)])
    controls = np.array([row[2] for row in records])
    for side, joint in env.controller.joints.items():
        assert np.max(np.abs(np.diff(positions[:, joint["qpos"]], axis=0))) < .01
        values = controls[:, joint["act"]]
        assert np.all(values >= joint["limits"][:, 0] - 1e-9)
        assert np.all(values <= joint["limits"][:, 1] + 1e-9)
        assert np.max(np.abs(np.diff(values, axis=0))) <= (
            env.controller.config["period"] * env.controller.config["max_joint_speed"] + 1e-9)
    # The next policy target uses the existing Cartesian executor; a reset must
    # re-arm target IK for the following episode, not only the first process run.
    second = env.controller.execute(env.adapter.state().copy())
    assert second["reached"]
    assert "target_ik" not in second
    env.reset()
    assert env.controller._initial_target_pending
    third = env.controller.execute(env.adapter.state().copy())
    assert third["reached"]
    assert "target_ik" in third


def test_colliding_paths_are_rejected_without_moving_the_robot(environment):
    env = environment
    # Put the independent item inside an actual wrist collision shape. Every
    # path starts in collision and must fail instead of stepping through it.
    env.data.joint("item").qpos[:3] = env.data.geom("robot6_left").xpos.copy()
    mujoco.mj_forward(env.model, env.data)
    before = env.data.qpos.copy()
    before_time = env.data.time
    targets, metadata = plan_target_joints(env.controller, FIRST_ACTION)
    assert targets is None
    assert metadata["failure"] == "all_straight_joint_paths_collide"
    assert metadata["rejected_paths"]
    np.testing.assert_array_equal(env.data.qpos, before)
    assert env.data.time == before_time


def test_unreachable_action_never_changes_physical_state(environment):
    env = environment
    action = FIRST_ACTION.copy()
    action[:, :3] = 10
    before = {name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    result = env.controller.execute(action)
    assert not result["reached"]
    assert result["target_ik"]["failure"] == "no_bounded_ik_solution"
    assert env.data.time == 0
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data, name), value)
