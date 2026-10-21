"""Isolated CoppeliaSim rendering worker; invoked by native_renderer only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import pose_matrix
from .source_kinematics import SourceArmKinematics, rotation_z, source_camera_calibration


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from pyrep import PyRep
    from pyrep.const import RenderMode, Verbosity
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.joint import Joint
    from pyrep.objects.shape import Shape
    from pyrep.objects.vision_sensor import VisionSensor

    scene = args.source_root / "rlbench/task_design_bimanual.ttt"
    task_path = args.source_root / "rlbench/task_ttms/bimanual_lift_ball.ttm"
    calibration = source_camera_calibration(args.archive)
    simulation = PyRep()
    simulation.launch(str(scene), headless=True, verbosity=Verbosity.ERRORS)
    try:
        task = simulation.import_model(str(task_path))
        task.set_position(Shape("workspace").get_position())
        arms = {}
        for side in ("left", "right"):
            joints = [Joint(f"Panda_{side}Arm_joint{i}") for i in range(1, 8)]
            tip = Dummy(f"Panda_{side}Arm_tip")
            initial = [joint.get_joint_position() for joint in joints]
            limits = []
            for joint in joints:
                cyclic, interval = joint.get_joint_interval()
                limits.append([-np.pi, np.pi] if cyclic else
                              [interval[0], interval[0] + interval[1]])
            frames = [joint.get_matrix() for joint in joints]
            # CoppeliaSim object matrices exclude a joint's own intrinsic
            # rotation, but include every ancestor's rotation. Do not move to
            # q=0: zero lies outside Panda joint 4's valid interval.
            fixed = [frames[0]] + [rotation_z(-q) @ np.linalg.inv(a) @ b
                                   for q, a, b in zip(initial[:-1], frames[:-1], frames[1:])]
            tip_transform = rotation_z(-initial[-1]) @ np.linalg.inv(frames[-1]) @ tip.get_matrix()
            chain = SourceArmKinematics(fixed, tip_transform, limits)
            fk_error = float(np.max(np.abs(chain.forward(initial) - tip.get_matrix())))
            if fk_error > 1e-5:
                raise RuntimeError(f"Native {side} forward-kinematics validation failed: {fk_error}")
            arms[side] = {
                "joints": joints, "tip": tip, "chain": chain, "fk_validation_error": fk_error,
                "fingers": [Joint(f"Panda_{side}Arm_gripper_joint{i}") for i in (1, 2)],
            }
        cameras = {}
        for name, settings in calibration["cameras"].items():
            camera = VisionSensor("cam_" + name)
            camera.set_explicit_handling(1)
            camera.set_resolution([256, 256])
            camera.set_render_mode(RenderMode.OPENGL3)
            camera.set_near_clipping_plane(settings["near"])
            camera.set_far_clipping_plane(settings["far"])
            camera.set_perspective_angle(settings["fovy"])
            cameras[name] = camera
        provenance = {
            "renderer": "source_native", "physics": "MuJoCo only; source scene never stepped",
            "state_source": "current MuJoCo state on every request",
            "robot_alignment": "bounded native IK seeded with actual MuJoCo arm joints",
            "scene_path": str(scene), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "task_path": str(task_path), "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "calibration_archive": str(args.archive), "camera_calibration": calibration,
            "fk_validation_error": {side: arm["fk_validation_error"] for side, arm in arms.items()},
        }
        write_json(args.output / "ready.json", provenance)
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("quit"):
                break
            request = json.loads(Path(command["request"]).read_text())
            sequence = request["sequence"]
            response_path = args.output / f"response_{sequence:04d}.json"
            try:
                metadata = {
                    "renderer": "source_native", "sequence": sequence,
                    "simulation_time": request["simulation_time"], "arms": {}, "cameras": {},
                    "ball_pose_xyzw": request["ball_pose_xyzw"],
                    "policy_state": request["policy_state"],
                    "state_source": "current MuJoCo state; no demonstration replay",
                }
                for index, side in enumerate(("left", "right")):
                    target = pose_matrix(request["policy_state"][index][:7])
                    arm = arms[side]
                    values, alignment = arm["chain"].solve(target, request["joint_positions"][side])
                    for joint, value in zip(arm["joints"], values):
                        joint.set_joint_position(float(value))
                    for joint, value in zip(arm["fingers"], request["finger_positions"][side]):
                        joint.set_joint_position(float(value))
                    actual = arm["tip"].get_matrix()
                    # Report measured native object alignment in addition to
                    # optimizer residuals (CoppeliaSim stores float matrices).
                    alignment["native_position_error_m"] = float(np.linalg.norm(
                        actual[:3, 3] - target[:3, 3]))
                    alignment["native_rotation_error_rad"] = float(np.linalg.norm(
                        Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()))
                    alignment.update(native_joint_positions=values.tolist(),
                                     actual_mujoco_joint_positions=request["joint_positions"][side],
                                     actual_mujoco_finger_positions=request["finger_positions"][side],
                                     native_finger_positions=[joint.get_joint_position()
                                                              for joint in arm["fingers"]],
                                     rendered_tcp_matrix=actual.tolist(), target_tcp_matrix=target.tolist(),
                                     fk_runtime_error=float(np.max(np.abs(arm["chain"].forward(values) - actual))))
                    metadata["arms"][side] = alignment
                Shape("ball").set_pose(request["ball_pose_xyzw"])
                rgbs, clouds, depths = [], [], []
                for name, camera in cameras.items():
                    settings = calibration["cameras"][name]
                    matrix = np.asarray(settings["matrix"])
                    if name != "front":
                        index = 0 if name == "wrist_left" else 1
                        matrix = pose_matrix(request["policy_state"][index][:7]) @ matrix
                    camera.set_matrix(matrix)
                    camera.handle_explicitly()
                    rgb = (camera.capture_rgb() * 255).astype(np.uint8)
                    depth = camera.capture_depth(in_meters=True)
                    points = camera.pointcloud_from_depth(depth)
                    rgbs.append(rgb)
                    clouds.append(points)
                    depths.append(depth)
                    metadata["cameras"][name] = {
                        "world_matrix": camera.get_matrix().tolist(),
                        "target_world_matrix": matrix.tolist(),
                        "matrix_error_max": float(np.max(np.abs(camera.get_matrix() - matrix))),
                        "near": settings["near"], "far": settings["far"], "fovy": settings["fovy"],
                    }
                observation_path = args.output / f"observation_{sequence:04d}.npz"
                np.savez_compressed(observation_path, rgb=np.asarray(rgbs),
                                    points=np.asarray(clouds, np.float32), depth=np.asarray(depths))
                metadata["observation_file"] = observation_path.name
                metadata["all_arms_aligned"] = all(arm["aligned"] for arm in metadata["arms"].values())
                write_json(response_path, metadata)
            except Exception as error:
                traceback.print_exc()
                write_json(response_path, {"error": str(error), "sequence": sequence})
    finally:
        simulation.shutdown()


if __name__ == "__main__":
    main()
