"""Match authored Panda bases and a released PerAct2 ball episode.

No RLBench installation or arbitrary pickle classes are executed. Only numeric
NumPy arrays and inert observation records are accepted from the reference ZIP.
Bases and ball/task settings come from the original scene. Only the TCP
correction is fitted, since the two robot models have different kinematics.
"""

import argparse
import io
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .environment import DEFAULT_CONFIG, read_config
from .geometry import matrix_pose, pose_matrix
from .policy import sha256


class Record:
    pass


class NumericReferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("rlbench.demo", "Demo"): Record,
            ("rlbench.backend.observation", "BimanualObservation"): Record,
            ("rlbench.backend.observation", "UnimanualObservationData"): Record,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected reference pickle global: {module}.{name}")
        return allowed[module, name]


def parameters_to_transform(parameters):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(parameters[3:]).as_matrix()
    t[:3, 3] = parameters[:3]
    return t


def transform_to_parameters(t):
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_rotvec()]


def body_transform(body):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = body.xmat.reshape(3, 3), body.xpos
    return t


def set_xml_pose(element, transform):
    pose = matrix_pose(transform)
    element.set("pos", " ".join(map(str, pose[:3])))
    element.set("quat", " ".join(map(str, pose[[6, 3, 4, 5]])))
    for attribute in ("xyaxes", "zaxis", "euler", "axisangle"):
        element.attrib.pop(attribute, None)


SOURCE_SCENE = Path(__file__).parent / "configs" / "peract2_scene.json"


def fit_arm(model, observations, side, source_base):
    data = mujoco.MjData(model)
    fks, targets = [], []
    for observation in observations:
        arm = getattr(observation, side)
        for i, value in enumerate(arm.joint_positions, 1):
            data.joint(f"joint{i}_{side}").qpos[0] = value
        mujoco.mj_forward(model, data)
        fks.append(body_transform(data.site(f"gripper_{side}")))
        targets.append(pose_matrix(arm.gripper_pose))
    # Pin the base to its authored joint1-derived datum. Never optimize base
    # placement to absorb the configuration-dependent joint4/5 discrepancy.
    current_base = body_transform(data.body(f"link0_{side}"))
    fks = source_base @ np.linalg.inv(current_base) @ np.array(fks)
    targets = np.array(targets)
    train = np.arange(0, len(fks), 3)
    validation = np.setdiff1d(np.arange(len(fks)), train)

    def residual(parameters, indices):
        predicted = fks[indices] @ parameters_to_transform(parameters)
        pos = predicted[:, :3, 3] - targets[indices, :3, 3]
        rot = Rotation.from_matrix(
            predicted[:, :3, :3] @ targets[indices, :3, :3].transpose(0, 2, 1)).as_rotvec()
        return np.c_[pos, 0.2 * rot].ravel()

    initial = transform_to_parameters(np.linalg.inv(fks[0]) @ targets[0])
    fit = least_squares(lambda x: residual(x, train), initial, max_nfev=200)
    errors = residual(fit.x, validation).reshape(-1, 6)
    rms_position = float(np.sqrt(np.mean(np.sum(errors[:, :3]**2, axis=1))))
    rms_rotation = float(np.sqrt(np.mean(np.sum((errors[:, 3:] / 0.2)**2, axis=1))))
    if not fit.success or rms_position > 0.05 or rms_rotation > 0.10:
        raise ValueError(f"{side}: reference fit too inaccurate: {rms_position}m, {rms_rotation}rad")
    tcp = parameters_to_transform(fit.x)
    return tcp, {
        "validation_position_rms_m": rms_position, "validation_rotation_rms_rad": rms_rotation,
        "fit_frames": len(train), "validation_frames": len(validation),
        "base_was_optimized": False,
    }


def calibrate_reference(archive, scene_path, output_config, episode=3):
    member = f"peract2_test/bimanual_lift_ball/variation0/episodes/episode{episode}/low_dim_obs.pkl"
    with zipfile.ZipFile(archive) as z:
        demo = NumericReferenceUnpickler(io.BytesIO(z.read(member))).load()
    observations = demo._observations
    source = json.loads(SOURCE_SCENE.read_text())
    if source["task_descendants"][0] != "ball":
        raise ValueError("Source task object order does not place the ball first")
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    config = read_config(DEFAULT_CONFIG)
    config["home_joints"] = {
        side: getattr(observations[0], side).joint_positions.tolist()
        for side in config["arms"]
    }
    config["controller"]["max_seconds"] = 8.0
    config["reset_settle_seconds"] = 0.0
    config.pop("success_proxy", None)
    config["success_condition"] = source["success_condition"].copy()
    robot = ET.parse(scene_path.parent / "panda_scene.xml")
    table = robot.find(".//body[@name='table']")
    # Preserve the source contact height and table footprint. The source table
    # has a tiny (~27 microradian) tilt; this box uses its upper world bbox.
    table_position = [*source["table_center_xy_m"], source["table_top_z_m"] - 0.02]
    table_transform = pose_matrix([*table_position, 0, 0, 0, 1])
    set_xml_pose(table, table_transform)
    table.find("./geom[@name='table_top']").set(
        "size", " ".join(map(str, [*source["table_half_size_xy_m"], 0.02])))
    metrics = {}
    optical_to_gl = np.diag([-1., 1., -1., 1.])
    for side in config["arms"]:
        base = pose_matrix(source["robot_bases"][side])
        tcp, metrics[side] = fit_arm(model, observations, side, base)
        config["T_site_from_policy_gripper"][side] = tcp.tolist()
        set_xml_pose(robot.find(f".//body[@name='link0_{side}']"),
                     np.linalg.inv(table_transform) @ base)
        gripper = pose_matrix(getattr(observations[0], side).gripper_pose)
        optical_camera = observations[0].misc[f"wrist_{side}_camera_extrinsics"]
        # The site is 10cm along the hand's local +Z in this Panda XML.
        hand_to_site = np.eye(4)
        hand_to_site[2, 3] = 0.1
        hand_to_camera = hand_to_site @ tcp @ np.linalg.inv(gripper) @ optical_camera @ optical_to_gl
        camera = robot.find(f".//camera[@name='wrist_{side}']")
        set_xml_pose(camera, hand_to_camera)
    robot.write(scene_path.parent / "panda_scene.xml")

    scene = ET.parse(scene_path)
    for name in config["cameras"]:
        k = observations[0].misc[f"{name}_camera_intrinsics"]
        fovy = np.rad2deg(2 * np.arctan(config["image_size"] / (2 * abs(k[1, 1]))))
        if name == "front":
            camera = scene.find(".//camera[@name='front']")
            set_xml_pose(camera, observations[0].misc["front_camera_extrinsics"] @ optical_to_gl)
        else:
            camera = robot.find(f".//camera[@name='{name}']")
        camera.set("fovy", str(fovy))
    robot.write(scene_path.parent / "panda_scene.xml")
    scene.write(scene_path)

    # Native source scene confirms the first seven task-state values are ball
    # XYZ + XYZW. Copy all seven, not only horizontal position.
    ball_pose = np.asarray(observations[0].task_low_dim_state).reshape(-1)[:7]
    objects = ET.parse(scene_path.parent / "objects.xml")
    ball = objects.find(".//body[@name='ball']")
    set_xml_pose(ball, pose_matrix(ball_pose))
    geom = ball.find("./geom[@name='ball']")
    geom.set("size", str(source["ball_radius_m"]))
    geom.set("mass", str(source["ball_mass_kg"]))
    geom.set("rgba", " ".join(map(str, [*source["ball_color_rgb"], 1])))
    # The original task has no target object, target XY, or target orientation.
    # Remove the unrelated branch's visual marker rather than relabeling it.
    target = objects.find("./body[@name='target_0']")
    if target is not None:
        objects.getroot().remove(target)
    objects.write(scene_path.parent / "objects.xml")
    config["calibration_status"] = (
        "Authored source bases and ball/task settings; full episode start pose; "
        "TCP-only fit remains approximate due to internal arm kinematic differences")
    config["scene_settings"] = {
        "robot_bases": source["robot_bases"],
        "ball_initial_pose_xyzw": ball_pose.tolist(),
        "ball_radius_m": source["ball_radius_m"],
        "ball_mass_kg": source["ball_mass_kg"],
        "table_top_z_m": source["table_top_z_m"],
        "success_condition": source["success_condition"],
        "source": source["source"],
    }
    config["reference"] = {
        "url": "https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip",
        "archive_sha256": sha256(archive), "member": member, "metrics": metrics,
        "note": "TCP-only fit, with authored bases fixed. Adjacent held-out frames from one demo.",
        "kinematic_limitation": (
            "The source joint4-to-joint5 frame has an approximately 3.9993-degree "
            "offset absent in the MuJoCo branch. Exact bases do not imply identical "
            "full-chain kinematics, contact dynamics, or task success."),
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, indent=2) + "\n")
    manifest_path = scene_path.parent / "source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["calibration_status"] = config["calibration_status"]
        manifest["reference_scene_source"] = source["source"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Export numeric replay targets for controller diagnostics without pickle.
    np.savez_compressed(output_config.parent / "reference_replay.npz", **{
        side + "_" + field: np.array([getattr(getattr(o, side), field) for o in observations])
        for side in config["arms"] for field in ("joint_positions", "gripper_pose", "gripper_open")
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path("runtime/panda_ball/scene.xml"))
    parser.add_argument("--output-config", type=Path, default=Path("runtime/panda_ball/reference_config.json"))
    parser.add_argument("--episode", type=int, default=3)
    args = parser.parse_args()
    config = calibrate_reference(args.archive, args.scene, args.output_config, args.episode)
    print(json.dumps(config["reference"], indent=2))


if __name__ == "__main__":
    main()
