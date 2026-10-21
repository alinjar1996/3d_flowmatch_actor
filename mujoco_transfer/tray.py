"""Prepare and verify the released tray-and-item task on the Panda embodiment."""

import argparse
import io
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .environment import PandaBallEnvironment, read_config
from .geometry import matrix_pose, pose_matrix
from .initial_states import SourceEpisodeSampler
from .policy import sha256
from .reference import NumericReferenceUnpickler, fit_arm, set_xml_pose
from .tasks import fixture_path


TASK = "bimanual_lift_tray"
OBJECTS = ("tray", "item", "drawer_legs", "drawer_legs0")


def numbers(values):
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).ravel())


def prepare_tray(base, assets, archive, output, episode=0):
    base, assets, output = Path(base), Path(assets).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new tray output directory: {output}")
    source = json.loads(fixture_path(TASK).read_text())
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest.get("task", {}).get("name") != TASK:
        raise ValueError("The visual export is not the tray task")
    for kind in ("scene", "task_model", "task_code"):
        if manifest["source"]["files"][kind]["sha256"] != source["source"][kind]["sha256"]:
            raise ValueError(f"Tray visual/source hash mismatch: {kind}")
    initial = SourceEpisodeSampler(archive, seed=0, task=TASK).sample(episode)
    with zipfile.ZipFile(archive) as data:
        demo = NumericReferenceUnpickler(io.BytesIO(data.read(initial["member"]))).load()
    config = read_config(base / "reference_config.json")
    shutil.copytree(base, output)
    model = mujoco.MjModel.from_xml_path(str((output / "scene.xml").resolve()))
    robot = ET.parse(output / "panda_scene.xml")
    scene = ET.parse(output / "scene.xml")
    metrics = {}
    for side in config["arms"]:
        tcp, metrics[side] = fit_arm(
            model, demo._observations, side, pose_matrix(source["robot_bases"][side]))
        config["T_site_from_policy_gripper"][side] = tcp.tolist()
        first = demo._observations[0]
        hand_to_site = np.eye(4)
        hand_to_site[2, 3] = .1
        optical_to_gl = np.diag([-1., 1., -1., 1.])
        camera_pose = (hand_to_site @ tcp @ np.linalg.inv(pose_matrix(getattr(first, side).gripper_pose))
                       @ first.misc[f"wrist_{side}_camera_extrinsics"] @ optical_to_gl)
        set_xml_pose(robot.find(f".//camera[@name='wrist_{side}']"), camera_pose)
    for name in config["cameras"]:
        first = demo._observations[0]
        camera = (scene.find(".//camera[@name='front']") if name == "front"
                  else robot.find(f".//camera[@name='{name}']"))
        k = first.misc[f"{name}_camera_intrinsics"]
        camera.set("fovy", str(np.rad2deg(2 * np.arctan(256 / (2 * abs(k[1, 1]))))))
        if name == "front":
            set_xml_pose(camera, first.misc["front_camera_extrinsics"] @ np.diag([-1., 1., -1., 1.]))
    robot.write(output / "panda_scene.xml")
    scene.write(output / "scene.xml")

    root = ET.Element("mujoco")
    asset = scene.getroot().find("asset")
    if asset is None:
        asset = ET.SubElement(scene.getroot(), "asset")
    records = manifest["objects"]
    if not isinstance(records, dict):
        records = {r["name"]: r for r in records}
    meshes = output / "tray_meshes"
    meshes.mkdir()
    for name in OBJECTS:
        record = source["objects"][name]
        body = ET.SubElement(root, "body", name=name)
        set_xml_pose(body, pose_matrix(initial["object_poses_xyzw"][name]))
        if name in ("tray", "item"):
            ET.SubElement(body, "freejoint", name=name)
        components = record["primitive_components"]
        for i, component in enumerate(components):
            full_size = np.asarray(component["full_size_m"])
            kind = {"cuboid": "box", "cylinder": "cylinder"}[component["kind"]]
            size = full_size / 2 if kind == "box" else full_size[[0, 2]] / 2
            geom = ET.SubElement(
                body, "geom", name=name if len(components) == 1 else f"{name}_{i}",
                type=kind, size=numbers(size), mass=str(record["mass_kg"] / len(components)),
                rgba=numbers([*record["color_rgb"], 1]),
                friction=f"{record['bullet_friction']} 0.005 0.0001",
                contype="1", conaffinity="2147483647", group="3" if name == "tray" else "0")
            set_xml_pose(geom, pose_matrix(component["local_pose_xyzw"]))
        if name == "tray":
            visual_record = records["tray_visual"]
            visual_body = ET.SubElement(body, "body", name="tray_visual")
            set_xml_pose(visual_body, np.asarray(visual_record["local_matrix"]))
            for i, component in enumerate(visual_record["meshes"]):
                filename = f"tray_visual_{i}.obj"
                shutil.copyfile(assets / component["obj_path"], meshes / filename)
                # Existing scene compiler has meshdir='assets'.
                ET.SubElement(asset, "mesh", name=f"tray_visual_{i}",
                              file=str((meshes / filename).resolve()))
                geom = ET.SubElement(visual_body, "geom", name=f"tray_visual_{i}",
                                     type="mesh", mesh=f"tray_visual_{i}", mass="0",
                                     rgba=numbers(component["material"]["rgba"]),
                                     contype="0", conaffinity="0", group="0")
                set_xml_pose(geom, np.asarray(component.get("local_matrix", np.eye(4))))
    ET.indent(root)
    ET.ElementTree(root).write(output / "objects.xml", encoding="unicode")
    scene.write(output / "scene.xml")
    config.update(
        source_task=TASK, object_name="tray", additional_object_names=["item"],
        static_object_names=["drawer_legs", "drawer_legs0"], instruction=source["instruction"],
        home_joints=initial["joint_positions"], reset_settle_seconds=0.,
        success_condition=source["success_condition"],
        tray_proximity_sensors=source["tray_proximity_sensors"],
        tray_collision_box=source["tray_collision_box"],
        calibration_status="Source tray/item/racks and Panda bases; TCP fit to tray episode; physical free bodies",
        dynamics_adaptations={
            "tray_inertia": "MuJoCo cuboid inertia: the native authored tensor violates the triangle inequality.",
            "contacts": "Source sliding friction; MuJoCo contact solver and Panda collision meshes.",
            "attachment": "No tray or item weld, parent attachment, pose slaving, or expert control.",
        },
    )
    config["controller"]["backend"] = "bounded_target_ik_initial"
    config["scene_settings"] = {
        "robot_bases": source["robot_bases"], "table_top_z_m": source["table_top_z_m"],
        "object_poses_xyzw": initial["object_poses_xyzw"], "source": source["source"],
        "success_condition": source["success_condition"],
    }
    config["reference"] = {
        "member": initial["member"], "archive_sha256": initial["archive_sha256"],
        "metrics": metrics, "note": "Tray episode TCP fit; robot base frames are independently pinned.",
    }
    (output / "reference_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Do not retain ball artifacts under a misleading tray directory.
    for filename in ("reference_replay.npz", "settings_verification.json"):
        (output / filename).unlink(missing_ok=True)
    provenance = json.loads((base / "source.json").read_text())
    provenance.update(source_task=TASK, reference_scene_source=source["source"],
                      calibration_status=config["calibration_status"],
                      source_visual_manifest_sha256=sha256(assets / "manifest.json"),
                      dynamics_adaptations=config["dynamics_adaptations"])
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return config


def inspect_tray_settings(model, data, config, source, poses=None, tolerance=2e-6):
    """Check physical geometry, reset poses, bases and all success parameters."""
    checks = []

    def compare(name, actual, expected, atol=tolerance):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        error = float(np.max(np.abs(a - b)))
        checks.append({"name": name, "passed": bool(np.isfinite(error) and error <= atol),
                       "actual": a.tolist(), "expected": b.tolist(), "max_absolute_error": error,
                       "tolerance": atol})

    def equal(name, actual, expected):
        if isinstance(actual, np.generic):
            actual = actual.item()
        checks.append({"name": name, "passed": actual == expected,
                       "actual": actual, "expected": expected})

    def body_pose(name, expected):
        body = data.body(name)
        compare(f"{name}_position", body.xpos, expected[:3])
        angle = (Rotation.from_matrix(body.xmat.reshape(3, 3))
                 * Rotation.from_quat(expected[3:]).inv()).magnitude()
        compare(f"{name}_rotation", angle, 0.)

    for side in ("left", "right"):
        body_pose(f"link0_{side}", source["robot_bases"][side])
    for name in OBJECTS:
        expected = source["objects"][name]
        body = model.body(name)
        body_pose(name, (poses or config["scene_settings"]["object_poses_xyzw"])[name])
        equal(f"{name}_world_rooted", int(body.parentid[0]), 0)
        equal(f"{name}_not_mocap", int(body.mocapid[0]), -1)
        equal(f"{name}_joint_count", int(body.jntnum[0]), int(name in ("tray", "item")))
        if name in ("tray", "item"):
            equal(f"{name}_joint_type", int(model.joint(name).type[0]), int(mujoco.mjtJoint.mjJNT_FREE))
            compare(f"{name}_mass", body.mass[0], expected["mass_kg"])
            dimensions = np.asarray(expected["primitive_components"][0]["full_size_m"])
            inertia = expected["mass_kg"] / 12 * (np.sum(dimensions ** 2) - dimensions ** 2)
            compare(f"{name}_cuboid_inertia", body.inertia, inertia, atol=1e-10)
            equal(f"{name}_physical_inertia", bool(np.all(body.inertia > 0)
                  and 2 * max(body.inertia) <= sum(body.inertia) + 1e-12), True)
            equal(f"{name}_unactuated", bool(np.any(
                model.actuator_trnid[:, 0] == model.joint(name).id)), False)
        for i, component in enumerate(expected["primitive_components"]):
            geom_name = name if len(expected["primitive_components"]) == 1 else f"{name}_{i}"
            geom = model.geom(geom_name)
            kind = mujoco.mjtGeom.mjGEOM_BOX if component["kind"] == "cuboid" else mujoco.mjtGeom.mjGEOM_CYLINDER
            equal(f"{geom_name}_type", int(geom.type[0]), int(kind))
            size = np.asarray(component["full_size_m"]) / 2
            actual_size = geom.size if kind == mujoco.mjtGeom.mjGEOM_BOX else geom.size[:2]
            compare(f"{geom_name}_size", actual_size, size if len(actual_size) == 3 else size[[0, 2]])
            compare(f"{geom_name}_position_local", geom.pos, component["local_pose_xyzw"][:3])
            angle = (Rotation.from_quat(geom.quat[[1, 2, 3, 0]])
                     * Rotation.from_quat(component["local_pose_xyzw"][3:]).inv()).magnitude()
            compare(f"{geom_name}_rotation_local", angle, 0.)
            equal(f"{geom_name}_collision_type", int(geom.contype[0]), 1)
            equal(f"{geom_name}_collision_affinity", int(geom.conaffinity[0]), 2147483647)
            compare(f"{geom_name}_sliding_friction", geom.friction[0], expected["bullet_friction"])
    table = model.geom("table_top")
    z = data.geom("table_top").xpos[2] + np.abs(data.geom("table_top").xmat.reshape(3, 3)[2]) @ table.size
    compare("table_top_z", z, source["table_top_z_m"])
    compare("table_center_xy", data.geom("table_top").xpos[:2], source["table_center_xy_m"])
    compare("table_half_size_xy", table.size[:2], source["table_half_size_xy_m"])
    equal("source_success_condition", config["success_condition"], source["success_condition"])
    equal("source_detector_volumes", config["tray_proximity_sensors"], source["tray_proximity_sensors"])
    equal("source_tray_detection_box", config["tray_collision_box"], source["tray_collision_box"])
    equal("no_weld_constraints", int(model.neq), 0)
    equal("no_ball_object", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"), -1)
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "source": source["source"], "reference_demo_checked": poses is not None,
            "dynamics_adaptations": config["dynamics_adaptations"]}


def verify_tray_settings(scene, config, *, archive=None, episode=0):
    source = json.loads(fixture_path(TASK).read_text())
    initial = SourceEpisodeSampler(archive, seed=0, task=TASK).sample(episode) if archive else None
    # The static prepared scene must match its configured reference episode.
    env = PandaBallEnvironment(scene, config)
    try:
        report = inspect_tray_settings(model=env.model, data=env.data, config=config, source=source,
                                       poses=initial["object_poses_xyzw"] if initial else None)
        env.track_success()
        report["checks"].append({"name": "initial_state_not_success", "passed": not env.success})
        report["passed"] = all(c["passed"] for c in report["checks"])
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runtime/panda_ball_reference"))
    parser.add_argument("--assets", type=Path, default=Path("runtime/source_visuals_tray"))
    parser.add_argument("--archive", type=Path, default=Path("runtime/peract2_test.zip"))
    parser.add_argument("--output", type=Path, default=Path("runtime/panda_tray_reference"))
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    config = prepare_tray(args.base, args.assets, args.archive, args.output, args.episode)
    report = verify_tray_settings(args.output / "scene.xml", config, archive=args.archive, episode=args.episode)
    (args.output / "settings_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"Tray verification failed: {[c['name'] for c in report['checks'] if not c['passed']]}")
    print(f"Prepared and verified tray task: {args.output}")


if __name__ == "__main__":
    main()
