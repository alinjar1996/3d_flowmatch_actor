"""Integration checks for paired source resets in the real Panda scene."""

from copy import deepcopy
import io
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.environment import PandaBallEnvironment, read_config
from mujoco_transfer.initial_states import SourceEpisodeSampler
from mujoco_transfer.reference import NumericReferenceUnpickler
from mujoco_transfer.verify_settings import verify_initial_state


@pytest.fixture(scope="module")
def source_assets():
    root = Path(__file__).resolve().parents[1]
    scene = root / "runtime/panda_ball_reference/scene.xml"
    config_path = scene.parent / "reference_config.json"
    archive = root / "runtime/peract2_test.zip"
    if not all(path.exists() for path in (scene, config_path, archive)):
        pytest.skip("Prepare the source Panda scene and released archive first")
    return scene, config_path, archive


@pytest.fixture(scope="module")
def recorded_start(source_assets):
    sampler = SourceEpisodeSampler(source_assets[2], seed=31)
    return sampler.sample(episode=42)


def assert_applied_recorded_start(env, state, archive):
    # Read the selected source independently of the sampler's parsed state.
    with zipfile.ZipFile(archive) as source:
        demo = NumericReferenceUnpickler(io.BytesIO(source.read(state["member"]))).load()
    first = demo._observations[0]
    expected_ball = np.asarray(first.task_low_dim_state).reshape(-1)[:7]
    actual_ball = env.data.joint("ball").qpos
    np.testing.assert_allclose(actual_ball[:3], expected_ball[:3], rtol=0, atol=1e-12)
    rotation_error = (
        Rotation.from_quat(actual_ball[[4, 5, 6, 3]])
        * Rotation.from_quat(expected_ball[3:]).inv()
    ).magnitude()
    assert rotation_error < 1e-12
    for side in ("left", "right"):
        expected = getattr(first, side)
        actual_joints = [
            env.data.joint(f"joint{i}_{side}").qpos[0] for i in range(1, 8)
        ]
        np.testing.assert_array_equal(actual_joints, expected.joint_positions)
        np.testing.assert_array_equal(state["joint_positions"][side],
                                      expected.joint_positions)
        assert state["gripper_open"][side] == expected.gripper_open
    report = verify_initial_state(env, state)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_explicit_episode_applies_full_pose_and_both_recorded_postures(
        source_assets, recorded_start):
    scene, config_path, archive = source_assets
    config = read_config(config_path)
    supplied = deepcopy(recorded_start)
    env = PandaBallEnvironment(scene, config, initial_state=supplied)
    try:
        assert_applied_recorded_start(env, recorded_start, archive)
        assert env.initial_state == recorded_start
        assert env.data.time == 0.0
        # Callers and returned metadata must not rewrite the fixed reset state.
        supplied["ball_pose_xyzw"][0] += .2
        env.initial_state["joint_positions"]["left"][0] += .2
        env.reset()
        assert_applied_recorded_start(env, recorded_start, archive)
    finally:
        env.close()


def test_seeded_environment_resets_are_reproducible_and_leave_baseline_untouched(
        source_assets):
    scene, config_path, archive = source_assets
    original_files = (scene.read_bytes(), config_path.read_bytes())
    config = read_config(config_path)
    original_config = deepcopy(config)
    original_rng = np.random.get_state()
    a = b = None
    try:
        a = PandaBallEnvironment(
            scene, config, reset_sampler=SourceEpisodeSampler(archive, seed=19))
        # Unrelated consumers of the global RNG must not change reset selection.
        np.random.seed(902)
        np.random.random(53)
        b = PandaBallEnvironment(
            scene, deepcopy(config), reset_sampler=SourceEpisodeSampler(archive, seed=19))
        episodes = []
        for draw in range(3):
            if draw:
                a.reset()
                np.random.seed(103 + draw)
                np.random.random(23)
                b.reset()
            assert a.initial_state == b.initial_state
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)
            assert_applied_recorded_start(a, a.initial_state, archive)
            episodes.append(a.initial_state["episode"])
        assert len(set(episodes)) > 1
        assert config == original_config
        assert scene.read_bytes() == original_files[0]
        assert config_path.read_bytes() == original_files[1]
    finally:
        if a is not None:
            a.close()
        if b is not None:
            b.close()
        np.random.set_state(original_rng)


@pytest.mark.parametrize(("fault", "failed_check"), [
    ("joint", "left_joint_positions_vs_selected_demo"),
    ("arm_servo", "right_servo_targets_vs_selected_demo"),
    ("gripper_servo", "left_gripper_servo_targets_vs_selected_demo"),
    ("base", "left_base_vs_source"),
    ("ball_position", "ball_reset_pose_vs_demo"),
    ("ball_rotation", "ball_reset_pose_vs_demo"),
    ("velocity", "initial_velocities"),
    ("time", "initial_simulation_time"),
])
def test_selected_start_verification_rejects_state_and_static_scene_drift(
        source_assets, recorded_start, fault, failed_check):
    scene, config_path, _ = source_assets
    env = PandaBallEnvironment(scene, read_config(config_path),
                               initial_state=recorded_start)
    try:
        if fault == "joint":
            env.data.joint("joint1_left").qpos[0] += .01
        elif fault == "arm_servo":
            env.data.actuator("drive_joint1_right").ctrl[0] += .01
        elif fault == "gripper_servo":
            env.data.actuator("drive_left_finger_left").ctrl[0] = 0.0
        elif fault == "base":
            env.model.body("link0_left").pos[0] += .01
        elif fault == "ball_position":
            env.data.joint("ball").qpos[0] += .01
        elif fault == "ball_rotation":
            env.data.joint("ball").qpos[3:] = [1, 0, 0, 0]
        elif fault == "velocity":
            env.data.qvel[0] = .1
        else:
            env.data.time = .01
        mujoco.mj_forward(env.model, env.data)
        report = verify_initial_state(env, recorded_start)
        checks = {check["name"]: check["passed"] for check in report["checks"]}
        assert not report["passed"]
        assert not checks[failed_check]
    finally:
        env.close()


def test_reset_after_physics_clears_motion_success_and_stale_commands(
        source_assets, recorded_start):
    scene, config_path, archive = source_assets
    env = PandaBallEnvironment(scene, read_config(config_path),
                               initial_state=recorded_start)
    try:
        env.data.actuator("drive_joint1_left").ctrl[0] += .1
        env.data.actuator("drive_left_finger_right").ctrl[0] = 0.0
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)
        assert env.data.time > 0
        assert np.any(np.abs(env.data.qvel) > 1e-6)
        env.data.joint("ball").qpos[2] = .96
        env.track_success()
        assert env.success
        env.held_seconds = 1.5
        env.reset()
        assert_applied_recorded_start(env, recorded_start, archive)
        assert env.data.time == 0.0
        np.testing.assert_array_equal(env.data.qvel, 0.0)
        assert not env.success
        assert env.held_seconds == 0
        assert env.initial_ball_height == pytest.approx(
            recorded_start["ball_pose_xyzw"][2], abs=1e-12)
    finally:
        env.close()
