"""Verify the generated MuJoCo scene against source settings and a saved demo."""

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .environment import PandaBallEnvironment, read_config
from .reference import NumericReferenceUnpickler


SOURCE_SETTINGS = Path(__file__).parent / "configs" / "peract2_scene.json"
DEFAULT_SCENE = Path("runtime/panda_ball_reference/scene.xml")
DEFAULT_CONFIG = DEFAULT_SCENE.parent / "reference_config.json"


def _body_pose(data, name):
    body = data.body(name)
    return np.r_[body.xpos, Rotation.from_matrix(body.xmat.reshape(3, 3)).as_quat()]


def inspect_settings(model, data, config, expected_scene, reference_ball_pose=None,
                     position_tolerance_m=1e-6, rotation_tolerance_rad=1e-6):
    """Inspect already-reset physics state; report every discrepancy without edits.

    ``expected_scene`` is the independently extracted source fixture, while
    ``config['scene_settings']`` records the intended generated scene. Both must
    agree with the loaded physics model. Quaternions use XYZW and are compared
    as rotations, so opposite signs represent the same pose.
    """
    checks = []
    settings = config["scene_settings"]

    def equality(name, actual, expected):
        checks.append({
            "name": name, "passed": bool(actual == expected),
            "actual": actual, "expected": expected,
        })

    def scalar(name, actual, expected, tolerance=position_tolerance_m):
        actual, expected = float(actual), float(expected)
        error = abs(actual - expected)
        checks.append({
            "name": name, "passed": bool(np.isfinite(error) and error <= tolerance),
            "actual": actual, "expected": expected, "absolute_error": error,
            "tolerance": tolerance,
        })

    def pose(name, actual, expected):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        if actual.shape != (7,) or expected.shape != (7,):
            raise ValueError(f"{name}: expected XYZ plus XYZW pose")
        translation = float(np.linalg.norm(actual[:3] - expected[:3]))
        angle = float((Rotation.from_quat(actual[3:])
                       * Rotation.from_quat(expected[3:]).inv()).magnitude())
        checks.append({
            "name": name,
            "passed": bool(translation <= position_tolerance_m
                           and angle <= rotation_tolerance_rad),
            "actual_pose_xyzw": actual.tolist(),
            "expected_pose_xyzw": expected.tolist(),
            "position_error_m": translation, "rotation_error_rad": angle,
        })

    for side in ("left", "right"):
        actual = _body_pose(data, f"link0_{side}")
        pose(f"{side}_base_vs_source", actual, expected_scene["robot_bases"][side])
        pose(f"{side}_base_vs_config", actual, settings["robot_bases"][side])
    ball_pose = _body_pose(data, "ball")
    pose("ball_reset_pose_vs_config", ball_pose, settings["ball_initial_pose_xyzw"])
    if reference_ball_pose is not None:
        pose("ball_reset_pose_vs_demo", ball_pose, reference_ball_pose)
        pose("ball_config_pose_vs_demo", settings["ball_initial_pose_xyzw"],
             reference_ball_pose)

    equality("ball_is_sphere", int(model.geom("ball").type[0]),
             int(mujoco.mjtGeom.mjGEOM_SPHERE))
    actual_values = {
        "ball_radius_m": float(model.geom("ball").size[0]),
        "ball_mass_kg": float(model.body("ball").mass[0]),
    }
    table = model.geom("table_top")
    equality("table_top_is_box", int(table.type[0]), int(mujoco.mjtGeom.mjGEOM_BOX))
    # AABB top is evaluated in world coordinates, including table rotation.
    rotation = data.geom("table_top").xmat.reshape(3, 3)
    actual_values["table_top_z_m"] = float(
        data.geom("table_top").xpos[2] + np.abs(rotation[2]) @ table.size)
    for key, actual in actual_values.items():
        scalar(f"{key}_vs_source", actual, expected_scene[key])
        scalar(f"{key}_vs_config", actual, settings[key])

    source_condition = expected_scene["success_condition"]
    equality("source_uses_absolute_height", source_condition["kind"], "world_height")
    equality("success_condition_vs_source", config.get("success_condition"),
             source_condition)
    equality("recorded_success_condition_vs_source", settings["success_condition"],
             source_condition)
    equality("unrelated_target_0_absent",
             mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_0") == -1,
             True)
    return {
        "passed": all(check["passed"] for check in checks),
        "coordinate_convention": "metres; world frame; quaternion XYZW",
        "position_tolerance_m": position_tolerance_m,
        "rotation_tolerance_rad": rotation_tolerance_rad,
        "reference_demo_checked": reference_ball_pose is not None,
        "checks": checks,
        "source": deepcopy(expected_scene.get("source", {})),
        "note": "Checks scene settings, not equivalence of robot kinematics or dynamics.",
    }


def verify_settings(scene=DEFAULT_SCENE, config=DEFAULT_CONFIG, *, archive=None,
                    episode=3, expected_scene=None, position_tolerance_m=1e-6,
                    rotation_tolerance_rad=1e-6):
    """Reset an isolated environment, inspect settings, and probe task success."""
    if not isinstance(config, dict):
        config = read_config(config)
    if expected_scene is None:
        expected_scene = SOURCE_SETTINGS
    if not isinstance(expected_scene, dict):
        expected_scene = json.loads(Path(expected_scene).read_text())
    ball_pose, member = None, None
    if archive is not None:
        member = ("peract2_test/bimanual_lift_ball/variation0/episodes/"
                  f"episode{episode}/low_dim_obs.pkl")
        with zipfile.ZipFile(archive) as source:
            demo = NumericReferenceUnpickler(io.BytesIO(source.read(member))).load()
        ball_pose = np.asarray(demo._observations[0].task_low_dim_state).ravel()[:7]
    environment = PandaBallEnvironment(scene, config)
    try:
        report = inspect_settings(
            environment.model, environment.data, config, expected_scene, ball_pose,
            position_tolerance_m, rotation_tolerance_rad)
        minimum_height = expected_scene["success_condition"]["minimum_height_m"]
        # Probe the actual success implementation with no physics steps or hold
        # time. Reading config alone would miss an obsolete relative-height path.
        for name, height, expected in (
            ("success_below_threshold", minimum_height - 1e-5, False),
            ("success_at_threshold_immediate", minimum_height, True),
        ):
            environment.success = False
            environment.held_seconds = 0.0
            environment.data.joint("ball").qpos[2] = height
            mujoco.mj_forward(environment.model, environment.data)
            environment.track_success()
            actual = bool(environment.success)
            report["checks"].append({
                "name": name, "passed": actual == expected,
                "actual": actual, "expected": expected, "ball_z_m": height,
            })
        report["passed"] = all(check["passed"] for check in report["checks"])
        report["scene"] = str(scene)
        report["reference_archive"] = str(archive) if archive is not None else None
        report["reference_member"] = member
        return report
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, default=Path("runtime/peract2_test.zip"))
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--expected-scene", type=Path, default=SOURCE_SETTINGS)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_SCENE.parent / "settings_verification.json")
    args = parser.parse_args()
    report = verify_settings(args.scene, args.config, archive=args.archive,
                             episode=args.episode, expected_scene=args.expected_scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    failed = [check["name"] for check in report["checks"] if not check["passed"]]
    print(json.dumps({"passed": report["passed"], "failed_checks": failed,
                      "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
