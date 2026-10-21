#!/usr/bin/env python3
"""Independently audit the pinned native tray task and its 100 released resets.

This is an optional one-time CoppeliaSim/PyRep audit, not policy inference.
No simulation steps or demonstration actions are executed. Native geometry,
sensor checks and numeric demonstration metadata are captured independently of
the MuJoCo task builder. All poses are XYZW; primitive sizes are full sizes.
"""

import argparse
import ast
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mujoco_transfer.reference import NumericReferenceUnpickler


COMMIT = "8af748c51287989294e00c9c670e3330a0e35ed5"
FILES = {
    "scene": "rlbench/task_design_bimanual.ttt",
    "task_model": "rlbench/task_ttms/bimanual_lift_tray.ttm",
    "task_code": "rlbench/bimanual_tasks/bimanual_lift_tray.py",
}
HASHES = {
    "scene": "df2e87617b1078b7f6fe114bbc341df3263effe62a9c3ffdf3c64b511f46615f",
    "task_model": "795c1b7c18fc2d11b7f1432a47f7e7e63f88b03adf47e5184beb822b6728a240",
    "task_code": "df7538dd30e7e18a38b5566e3db14e4c8107a6cdb5e0ef65b3e7232f8762a732",
}
ARCHIVE_SHA256 = "d732886a9a9790a51d4ca71bcaa1727662c6dc0c490d402a1b0fce440c29d5a5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose(transform):
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()].tolist()


def xml_numbers(node, path):
    return np.fromstring(node.findtext(path), sep=" ").tolist()


def xml_pose(node):
    position = xml_numbers(node, "position")
    quaternion = xml_numbers(node, "quaternion")
    return position + quaternion[1:] + quaternion[:1]


def extract_native(source_root, native_output):
    from pyrep import PyRep
    from pyrep.backend import sim
    from pyrep.const import ObjectType, Verbosity
    from pyrep.objects.object import Object
    from pyrep.objects.proximity_sensor import ProximitySensor
    from pyrep.objects.shape import Shape
    from pyrep.objects.vision_sensor import VisionSensor

    simulator = PyRep()
    simulator.launch(str(source_root / FILES["scene"]), headless=True, verbosity=Verbosity.ERRORS)
    try:
        task = simulator.import_model(str(source_root / FILES["task_model"]))
        task.set_position(Shape("workspace").get_position())
        descendants = task.get_objects_in_tree(exclude_base=True)
        names = [obj.get_name() for obj in descendants]
        native_output.mkdir(parents=True, exist_ok=True)
        sim.simSaveScene(str(native_output / "source.simscene.xml"))
        xml_root = ET.parse(native_output / "source.simscene.xml").getroot()
        xml_objects = {node[0].findtext("common/name"): node[0]
                       for node in xml_root.findall("sceneObject")}
        records = {}
        for name in names + ["diningTable", "workspace", "Panda_leftArm_joint1", "Panda_rightArm_joint1"]:
            obj = Object.get_object(name)
            parent = obj.get_parent()
            record = {
                "type": obj.get_type().name,
                "parent": parent.get_name() if parent is not None else None,
                "pose_xyzw": obj.get_pose().tolist(),
                "local_pose_xyzw": obj.get_pose(relative_to=parent).tolist(),
            }
            if obj.get_type() == ObjectType.SHAPE:
                shape, node = Shape(name), xml_objects[name]
                record.update(mass_kg=shape.get_mass(), bounding_box_m=shape.get_bounding_box(),
                              color_rgb=shape.get_color(),
                              dynamic=node.findtext("dynamics/switches/static") == "false",
                              respondable=node.findtext("dynamics/switches/respondable") == "true",
                              renderable=node.findtext("common/localObjectSpecialProperty/renderable") == "true",
                              bullet_friction=float(node.findtext("dynamics/material/engines/bullet/friction")),
                              primitive_components=[])
                record.update(mass=record["mass_kg"], bbox=record["bounding_box_m"])
                for mesh in node.findall("meshProxy//mesh"):
                    primitive = mesh.find("primitive")
                    record["primitive_components"].append({
                        "kind": primitive.findtext("type"),
                        "full_size_m": xml_numbers(primitive, "sizes"),
                        "local_pose_xyzw": xml_pose(mesh.find("verticesLocalFrame")),
                    })
                inertia = node.find("meshProxy/mesh/common/dynamics/localInertiaFrame")
                if inertia is not None:
                    moments = xml_numbers(inertia, "principalMomentsOfInertia")
                    record["native_inertia"] = {
                        "local_pose_xyzw": xml_pose(inertia),
                        "principal_moments_per_mass_m2": moments,
                        "triangle_inequality_satisfied": 2 * max(moments) <= sum(moments) + 1e-12,
                        "note": "Captured native authored inertia, not inferred from geometry.",
                    }
            records[name] = record

        sensors, checks = {}, []
        tray = Shape("tray")
        original_tray = tray.get_pose().copy()
        for side in ("left", "right"):
            name = f"Panda_{side}Arm_gripper_attachProxSensor"
            sensor = ProximitySensor(name)
            tip = Object.get_object(f"Panda_{side}Arm_tip")
            node = xml_objects[name]
            volume = node.find("volume")
            close = xml_numbers(volume, "xySizeClose")
            far = xml_numbers(volume, "xySizeFar")
            if volume.findtext("type") != "pyramid" or close != far:
                raise ValueError("Expected the source's constant-width rectangular proximity volume")
            sensors[side] = {
                "name": name,
                "T_tcp_from_sensor": (np.linalg.inv(matrix(tip.get_pose())) @ matrix(sensor.get_pose())).tolist(),
                "volume_type": "rectangular_prism",
                "full_size_xy_m": close,
                "z_min_m": float(volume.findtext("offset")),
                "z_max_m": float(volume.findtext("offset")) + float(volume.findtext("range")),
                "normal_check": node.findtext("switches/normalCheck") == "true",
                "front_faces": node.findtext("switches/frontFaceDetection") == "true",
                "back_faces": node.findtext("switches/backFaceDetection") == "true",
            }
            sensors[side]["T_policy_tcp_from_sensor"] = sensors[side]["T_tcp_from_sensor"]
            sensors[side]["half_size_m"] = [close[0] / 2, close[1] / 2,
                                             (sensors[side]["z_max_m"] - sensors[side]["z_min_m"]) / 2]
            sensors[side]["center_local_m"] = [0, 0, (sensors[side]["z_max_m"] + sensors[side]["z_min_m"]) / 2]
            for center in ([0, 0, .02], [0, 0, .06], [0, 0, -.01], [.10, 0, .02],
                           [.12, 0, .02], [0, .22, .02], [0, .24, .02]):
                for degrees in (0, 35, 90):
                    relative = np.eye(4)
                    relative[:3, 3] = center
                    relative[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
                    tray.set_pose(pose(matrix(sensor.get_pose()) @ relative), reset_dynamics=False)
                    checks.append({"sensor": side, "tray_pose_in_sensor_xyzw": pose(relative),
                                   "native_detected": bool(sensor.is_detected(tray))})
        tray.set_pose(original_tray, reset_dynamics=False)

        cameras = {}
        for name in ("front", "wrist_left", "wrist_right"):
            camera = VisionSensor("cam_" + name)
            mount = matrix(camera.get_pose())
            if name != "front":
                side = name.split("_")[1]
                mount = np.linalg.inv(matrix(Object.get_object(f"Panda_{side}Arm_tip").get_pose())) @ mount
            cameras[name] = {"mount_matrix": mount.tolist(), "fovy_degrees": camera.get_perspective_angle(),
                             "near_m": camera.get_near_clipping_plane(), "far_m": camera.get_far_clipping_plane()}
        return {"task_descendants": names, "objects": records, "gripper_detection_sensors": sensors,
                "native_detection_checks": checks, "cameras": cameras}
    finally:
        simulator.shutdown()


def inspect_released_resets(archive, capture, output):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Released reference archive hash mismatch")
    pattern = re.compile(r"peract2_test/bimanual_lift_tray/variation0/episodes/episode([0-9]+)/low_dim_obs\.pkl")
    names = capture["task_descendants"]
    records = []
    camera_errors = {name: [] for name in capture["cameras"]}
    with zipfile.ZipFile(archive) as source:
        members = sorted((int(match[1]), member) for member in source.namelist()
                         if (match := pattern.fullmatch(member)))
        if [episode for episode, _ in members] != list(range(100)):
            raise ValueError("Expected the 100 released tray episodes 0..99")
        for episode, member in members:
            demo = NumericReferenceUnpickler(io.BytesIO(source.read(member))).load()
            first = demo._observations[0]
            states = np.asarray(first.task_low_dim_state, float).reshape(-1, 7)
            if states.shape != (len(names), 7) or not np.isfinite(states).all():
                raise ValueError(f"Unexpected tray task state shape in {member}")
            objects = {name: states[index].tolist() for index, name in enumerate(names)}
            for name, calibration in capture["cameras"].items():
                released = np.asarray(first.misc[name + "_camera_extrinsics"])
                if name != "front":
                    released = np.linalg.inv(matrix(getattr(first, name.split("_")[1]).gripper_pose)) @ released
                camera_errors[name].append(float(np.max(np.abs(released - calibration["mount_matrix"]))))
            records.append({"episode": episode, "member": member, "object_poses_xyzw": objects,
                            "joint_positions": {side: np.asarray(getattr(first, side).joint_positions).tolist()
                                                for side in ("left", "right")},
                            "gripper_open": {side: float(getattr(first, side).gripper_open)
                                             for side in ("left", "right")}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"archive_sha256": ARCHIVE_SHA256, "task_descendants": names,
                                 "episodes": records}, indent=2) + "\n")
    ranges = {}
    for name in ("tray", "item", "drawer_legs", "drawer_legs0"):
        positions = np.array([record["object_poses_xyzw"][name][:3] for record in records])
        ranges[name] = {"minimum_xyz_m": positions.min(axis=0).tolist(),
                        "maximum_xyz_m": positions.max(axis=0).tolist()}
    return {"episode_count": len(records), "episode_ids": [record["episode"] for record in records],
            "archive_sha256": ARCHIVE_SHA256, "task_state_size": 7 * len(names),
            "object_pose_indices": {name: index for index, name in enumerate(names)},
            "first_state_position_ranges": ranges,
            "maximum_authored_vs_released_camera_mount_element_error": {
                name: max(values) for name, values in camera_errors.items()},
            "sampling": "Uniform sampling with replacement of coupled released first states; no independent object/robot jitter."}


def make_fixture(capture, source_root, resets):
    objects = capture["objects"]
    bases = {}
    for side in ("left", "right"):
        joint = matrix(objects[f"Panda_{side}Arm_joint1"]["pose_xyzw"])
        joint[:3, 3] -= .333 * joint[:3, 2]
        bases[side] = pose(joint)
    table = objects["diningTable"]
    corners = np.array(list(itertools.product(*np.asarray(table["bounding_box_m"]).reshape(3, 2))))
    table_transform = matrix(table["pose_xyzw"])
    world = corners @ table_transform[:3, :3].T + table_transform[:3, 3]
    minimum, maximum = world.min(axis=0), world.max(axis=0)
    tree = ast.parse((source_root / FILES["task_code"]).read_text())
    heights = [float(ast.literal_eval(node.args[1])) for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LiftedCondition"]
    if heights != [1.2, 1.2]:
        raise ValueError("Unexpected native tray/item height conditions")
    return {
        "schema_version": 1, "task_name": "bimanual_lift_tray", "object_name": "tray",
        "instruction": "Lift the tray", "dynamic_object_names": ["tray", "item"],
        "static_task_object_names": ["drawer_legs", "drawer_legs0"],
        "robot_bases": bases,
        "joint1_world_poses_xyzw": {side: objects[f"Panda_{side}Arm_joint1"]["pose_xyzw"]
                                     for side in ("left", "right")},
        "joint1_height_above_link0_m": .333,
        "table_top_z_m": float(maximum[2]),
        "table_center_xy_m": ((maximum[:2] + minimum[:2]) / 2).tolist(),
        "table_half_size_xy_m": ((maximum[:2] - minimum[:2]) / 2).tolist(),
        "success_condition": {"kind": "tray_lift",
                              "minimum_height_m": 1.2, "height_objects": ["tray", "item"],
                              "detected_object": "tray", "detector_sides": ["left", "right"],
                              "combination": "all conditions simultaneously", "hold_seconds": 0},
        "tray_collision_box": {"half_size_m": [.105, .18, .003], "T_tray_from_box": np.eye(4).tolist()},
        "tray_proximity_sensors": capture["gripper_detection_sensors"],
        "reset_settings": {"variation": 0, "task_yaw_bounds_rad": [-np.pi / 8, np.pi / 8],
                           "item_position_relative_to_tray_before_settling_m": [0, 0, .001],
                           "graspable_objects": ["item"],
                           "note": "Tray is physically supported, not registered as graspable; released first states are already settled."},
        **capture, "released_resets": resets,
        "source": {"repository": "https://github.com/markusgrotz/RLBench", "commit": COMMIT,
                   **{key: {"path": path, "sha256": HASHES[key]} for key, path in FILES.items()},
                   "extraction": "Independent native CoppeliaSim4.1/PyRep capture and source XML serialization; no physics stepping or expert-action execution.",
                   "scope": "Pinned released task model; checkpoint itself does not record its RLBench source revision."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlbench-root", type=Path, default=Path("runtime/source-audit/RLBench"))
    parser.add_argument("--archive", type=Path, default=Path("runtime/peract2_test.zip"))
    parser.add_argument("--output", type=Path, default=Path("mujoco_transfer/configs/peract2_tray_scene.json"))
    parser.add_argument("--runtime-output", type=Path, default=Path("runtime/source-audit/tray"))
    args = parser.parse_args()
    root = args.rlbench_root.resolve()
    for key, path in FILES.items():
        if sha256(root / path) != HASHES[key]:
            raise ValueError(f"Pinned source hash mismatch: {path}")
    capture = extract_native(root, args.runtime_output.resolve())
    resets = inspect_released_resets(args.archive, capture, args.runtime_output / "released_initial_states.json")
    fixture = make_fixture(capture, root, resets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps({"fixture": str(args.output), "task_state_order": capture["task_descendants"],
                      "released_episode_count": resets["episode_count"],
                      "camera_mount_errors": resets["maximum_authored_vs_released_camera_mount_element_error"]}, indent=2))


if __name__ == "__main__":
    main()
