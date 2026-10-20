"""Source-settings checks independent of inference and generated Panda assets."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.verify_settings import (
    SOURCE_SETTINGS, inspect_settings, verify_settings,
)


@pytest.fixture
def settings_scene():
    # A rotated parent ensures the verifier reads world poses, not XML locals.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body pos="0.2 -0.1 0.3" quat=".7071067811865476 0 0 .7071067811865476">
          <body name="link0_left" pos=".1 .2 .4"/>
          <body name="link0_right" pos="-.1 -.2 .4"/>
        </body>
        <geom name="table_top" type="box" pos="0 0 .73" size=".8 .5 .02"/>
        <body name="ball" pos=".3 -.1 .9" quat=".9238795325112867 0 0 .3826834323650898">
          <freejoint name="ball"/>
          <geom name="ball" type="sphere" size=".15" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_q = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
    expected = {
        "robot_bases": {
            "left": [0, 0, .7, *base_q],
            "right": [.4, -.2, .7, *base_q],
        },
        "ball_radius_m": .15,
        "ball_mass_kg": 1.0,
        "table_top_z_m": .75,
        "success_condition": {"kind": "world_height", "minimum_height_m": .95},
    }
    ball = [.3, -.1, .9, *Rotation.from_euler("z", 45, degrees=True).as_quat()]
    settings = deepcopy(expected)
    settings["ball_initial_pose_xyzw"] = ball
    config = {"scene_settings": settings,
              "success_condition": deepcopy(expected["success_condition"])}
    return model, data, config, expected, ball


def test_source_settings_accept_world_poses_and_equivalent_quaternion_sign(settings_scene):
    model, data, config, expected, ball = settings_scene
    config["scene_settings"]["ball_initial_pose_xyzw"][3:] = (
        -np.asarray(ball[3:])).tolist()
    report = inspect_settings(model, data, config, expected, ball)
    assert report["passed"], report["checks"]
    assert report["reference_demo_checked"]


@pytest.mark.parametrize("fault", ["ball_height", "base_rotation", "success_threshold"])
def test_source_settings_reject_pose_or_threshold_drift(settings_scene, fault):
    model, data, config, expected, ball = settings_scene
    if fault == "ball_height":
        data.joint("ball").qpos[2] -= .03
        mujoco.mj_forward(model, data)
        failed_name = "ball_reset_pose_vs_demo"
    elif fault == "base_rotation":
        model.body("link0_right").quat[:] = Rotation.from_euler(
            "z", .05).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        failed_name = "right_base_vs_source"
    else:
        config["success_condition"]["minimum_height_m"] = 1.0
        failed_name = "success_condition_vs_source"
    report = inspect_settings(model, data, config, expected, ball)
    assert not report["passed"]
    assert not next(c for c in report["checks"] if c["name"] == failed_name)["passed"]


def test_scene_and_config_cannot_agree_on_wrong_source_base(settings_scene):
    model, data, config, expected, ball = settings_scene
    model.body("link0_left").pos[0] += .1
    config["scene_settings"]["robot_bases"]["left"][1] += .1
    mujoco.mj_forward(model, data)
    report = inspect_settings(model, data, config, expected, ball)
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert checks["left_base_vs_config"]
    assert not checks["left_base_vs_source"]


def test_generated_scene_matches_source_and_demo_after_reset():
    folder = Path("runtime/panda_ball_reference")
    archive = Path("runtime/peract2_test.zip")
    required = [folder / "scene.xml", folder / "reference_config.json",
                archive, SOURCE_SETTINGS]
    if not all(path.exists() for path in required):
        pytest.skip("Prepare the source-matched Panda scene and reference archive first")
    report = verify_settings(folder / "scene.xml", folder / "reference_config.json",
                             archive=archive)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
