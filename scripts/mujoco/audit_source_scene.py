#!/usr/bin/env python3
"""Extract original PerAct2 geometry with optional audit-only PyRep/CoppeliaSim.

Use the markusgrotz/PyRep fork and CoppeliaSim 4.1.0, configured as described
in the upstream 3DFA README. Neither is required for normal MuJoCo inference.

Example:
    python scripts/mujoco/audit_source_scene.py --rlbench-root /path/to/RLBench \
        --output /tmp/peract2_scene.json \
        --check-fixture mujoco_transfer/configs/peract2_scene.json

The authored task pose is extracted after moving its root to the workspace
center, as RLBench Scene.load does. Episode randomization and physics are not
run; a released demonstration supplies the episode-specific ball start.
"""

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


FILES = {
    "scene": "rlbench/task_design_bimanual.ttt",
    "task_model": "rlbench/task_ttms/bimanual_lift_ball.ttm",
    "task_code": "rlbench/bimanual_tasks/bimanual_lift_ball.py",
}
JOINT1_HEIGHT = 0.333


def extract_objects(root):
    # Keep this dependency isolated from the MuJoCo runtime.
    from pyrep import PyRep
    from pyrep.const import Verbosity
    from pyrep.objects.joint import Joint
    from pyrep.objects.shape import Shape

    simulator = PyRep()
    simulator.launch(str(root / FILES["scene"]), headless=True,
                     verbosity=Verbosity.ERRORS)
    try:
        task = simulator.import_model(str(root / FILES["task_model"]))
        task.set_position(Shape("workspace").get_position())
        objects = {}
        for name in ("ball", "diningTable", "Panda_leftArm_joint1",
                     "Panda_rightArm_joint1"):
            obj = Joint(name) if name.endswith("joint1") else Shape(name)
            record = {"pose_xyzw": obj.get_pose().tolist()}
            if isinstance(obj, Shape):
                record.update(bbox=obj.get_bounding_box(), mass=obj.get_mass(),
                              color=obj.get_color())
            objects[name] = record
        return {
            "objects": objects,
            "task_descendants": [
                obj.get_name() for obj in task.get_objects_in_tree(exclude_base=True)
            ],
        }
    finally:
        simulator.shutdown()


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def source_height(task_code):
    tree = ast.parse(task_code.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "LiftedCondition"
    ]
    if len(calls) != 1 or len(calls[0].args) != 2:
        raise ValueError("Expected exactly one LiftedCondition(item, height)")
    height = ast.literal_eval(calls[0].args[1])
    if not isinstance(height, (float, int)) or height <= 0:
        raise ValueError("Expected a positive literal success height")
    return float(height)


def settings_from_objects(root, capture, commit=None):
    objects = capture["objects"]
    bases, joint_poses = {}, {}
    for side in ("left", "right"):
        pose = objects[f"Panda_{side}Arm_joint1"]["pose_xyzw"]
        joint_poses[side] = pose
        transform = pose_matrix(pose)
        position = transform[:3, 3] - JOINT1_HEIGHT * transform[:3, 2]
        quaternion = np.asarray(pose[3:]) / np.linalg.norm(pose[3:])
        bases[side] = np.r_[position, quaternion].tolist()

    ball, table = objects["ball"], objects["diningTable"]
    ball_bounds = np.asarray(ball["bbox"]).reshape(3, 2)
    radii = (ball_bounds[:, 1] - ball_bounds[:, 0]) / 2
    if not np.allclose(radii, radii[0], atol=1e-6, rtol=0):
        raise ValueError("Ball bounding box is not spherical")
    corners = np.array(list(itertools.product(*np.asarray(table["bbox"]).reshape(3, 2))))
    transform = pose_matrix(table["pose_xyzw"])
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    minimum, maximum = world.min(axis=0), world.max(axis=0)
    manifest = {
        key: {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
        for key, path in FILES.items()
    }
    return {
        "schema_version": 1,
        "robot_bases": bases,
        "joint1_world_poses_xyzw": joint_poses,
        "base_frame_definition": (
            "Panda link0 datum: source joint1 fixed world frame translated by "
            "(0,0,-0.333) metres in its local axes. This is not the shared "
            "DualPanda object origin."
        ),
        "joint1_height_above_link0_m": JOINT1_HEIGHT,
        "ball_radius_m": float(radii[0]),
        "ball_mass_kg": ball["mass"],
        "ball_authored_pose_xyzw": ball["pose_xyzw"],
        "ball_color_rgb": ball["color"],
        "table_top_z_m": float(maximum[2]),
        "table_center_xy_m": ((maximum[:2] + minimum[:2]) / 2).tolist(),
        "table_half_size_xy_m": ((maximum[:2] - minimum[:2]) / 2).tolist(),
        "success_condition": {
            "kind": "world_height",
            "minimum_height_m": source_height(root / FILES["task_code"]),
        },
        "task_descendants": capture["task_descendants"],
        "source": {
            "repository": "https://github.com/markusgrotz/RLBench",
            "commit": commit,
            **manifest,
            "extraction": "Native CoppeliaSim via PyRep; no trajectory fitting used for robot bases",
            "scope": (
                "Authored source geometry, before episode placement or simulation. "
                "Checkpoint does not record its RLBench git commit."
            ),
        },
    }


def check_fixture(actual, expected, path=""):
    """Check derived geometry and source hashes; descriptive metadata can differ."""
    for key, value in expected.items():
        name = f"{path}.{key}" if path else key
        if name in ("source.extraction", "source.scope"):
            continue
        got = actual[key]
        if name == "source.commit" and got is None:
            continue
        if isinstance(value, dict):
            check_fixture(got, value, name)
        elif isinstance(value, (int, float)) or (
            isinstance(value, list) and value and isinstance(value[0], (int, float))
        ):
            if not np.allclose(got, value, atol=1e-6, rtol=0):
                raise ValueError(f"Fixture mismatch at {name}: {got!r} != {value!r}")
        elif got != value:
            raise ValueError(f"Fixture mismatch at {name}: {got!r} != {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rlbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", help="Revision for an exported source tree without .git")
    parser.add_argument("--check-fixture", type=Path)
    args = parser.parse_args()
    root = args.rlbench_root.resolve()
    for path in FILES.values():
        if not (root / path).is_file():
            parser.error(f"Missing source file: {root / path}")
    commit = args.source_commit
    if (root / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if commit is not None and commit != revision:
            parser.error("--source-commit differs from the checkout's HEAD")
        commit = revision
    settings = settings_from_objects(root, extract_objects(root), commit)
    if args.check_fixture:
        check_fixture(settings, json.loads(args.check_fixture.read_text()))
        print(f"Verified geometry and source hashes against {args.check_fixture}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
