"""Render the source scene's exported visual assets using MuJoCo alone.

This is a separate observation model. It never advances physics or changes the
physical task. Before each render, the current physical TCPs, finger openings
and task object poses are transferred to the source visual model. The different source
arm geometry is fitted to the actual TCPs using bounded inverse kinematics.

Run scripts/mujoco/export_source_visuals.py once to prepare the native meshes,
materials, textures and hierarchy. CoppeliaSim is not needed at inference time.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import depth_to_world, matrix_pose, pose_matrix
from .source_kinematics import SourceArmKinematics, rotation_z


CAMERAS = ("front", "wrist_left", "wrist_right")
OPTICAL_TO_GL = np.diag([-1., 1., -1., 1.])


def _numbers(values):
    return " ".join(f"{float(value):.12g}" for value in np.asarray(values).ravel())


def _pose_attributes(matrix):
    matrix = np.asarray(matrix, float)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {"pos": _numbers(matrix[:3, 3]), "quat": _numbers(quaternion[[3, 0, 1, 2]])}


def _objects(manifest):
    records = manifest["objects"]
    return records if isinstance(records, dict) else {record["name"]: record for record in records}


def task_object_names(manifest):
    """Old ball exports have no task metadata; all new exports name each body."""
    task = manifest.get("task", {})
    names = [task.get("object_name", "ball"), *task.get("additional_object_names", []),
             *task.get("static_object_names", [])]
    if len(set(names)) != len(names):
        raise ValueError(f"Repeated source task object names: {names}")
    return names


def _world(record):
    return np.asarray(record.get("world_matrix", record.get("matrix")), float)


def _joint_position(record):
    return float(record["joint"]["position"])


def _joint_limits(record):
    joint = record["joint"]
    if "limits" in joint:
        return np.asarray(joint["limits"], float)
    interval = joint["interval"]
    # PyRep returns (cyclic, [minimum, range]); exporters may separate cyclic.
    if len(interval) == 2 and isinstance(interval[1], (list, tuple)):
        interval = interval[1]
    minimum, extent = interval
    return np.array([minimum, minimum + extent], float)


def kinematics_from_manifest(manifest, side):
    """Recover zero-position link transforms from a captured native hierarchy."""
    objects = _objects(manifest)
    joints = [objects[f"Panda_{side}Arm_joint{i}"] for i in range(1, 8)]
    fixed = [_world(joints[0])]
    for previous, current in zip(joints, joints[1:]):
        fixed.append(rotation_z(-_joint_position(previous))
                     @ np.linalg.inv(_world(previous)) @ _world(current))
    tip = (rotation_z(-_joint_position(joints[-1])) @ np.linalg.inv(_world(joints[-1]))
           @ _world(objects[f"Panda_{side}Arm_tip"]))
    return SourceArmKinematics(fixed, tip, [_joint_limits(record) for record in joints])


def camera_calibration_from_manifest(manifest):
    objects = _objects(manifest)
    result = {}
    for name in CAMERAS:
        record = objects["cam_" + name]
        properties = record["camera"]
        mount = _world(record)
        if name != "front":
            side = name.split("_")[1]
            mount = np.linalg.inv(_world(objects[f"Panda_{side}Arm_tip"])) @ mount
        result[name] = {
            "matrix": mount,
            "fovy": properties.get("fovy", properties.get("perspective_angle_degrees")),
            "near": properties.get("near", properties.get("near_clipping_plane", 0.01)),
            "far": properties.get("far", properties.get("far_clipping_plane", 10.0)),
        }
    return result


def _visible(record):
    if "visible_for_camera" in record:
        return bool(record["visible_for_camera"])
    return bool(record.get("renderable", True)) and bool(record.get("visibility_layer", 1) & 255)


def build_source_visual_scene(manifest_path, output_path):
    """Compile a render-only MJCF from a deterministic native asset export."""
    manifest_path, output_path = Path(manifest_path).resolve(), Path(output_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    objects, root_dir = _objects(manifest), manifest_path.parent
    movable_objects = task_object_names(manifest)
    missing = set(movable_objects) - objects.keys()
    if missing:
        raise ValueError(f"Source manifest lacks task objects: {sorted(missing)}")
    root = ET.Element("mujoco", model="peract2_source_visuals")
    ET.SubElement(root, "compiler", angle="radian", fusestatic="false", autolimits="true", inertiafromgeom="false")
    ET.SubElement(root, "option", gravity="0 0 0")
    # Explicit extent makes near/far clipping distances metre-valued.
    ET.SubElement(root, "statistic", extent="1", center="0 0 1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="256", offheight="256")
    ET.SubElement(visual, "quality", offsamples="0", shadowsize="2048")
    scene = manifest.get("scene", manifest.get("environment", {}))
    ambient = scene.get("ambient_light", [0.2, 0.2, 0.2])
    ET.SubElement(visual, "headlight", active="1", ambient=_numbers(ambient), diffuse="0 0 0", specular="0 0 0")
    ET.SubElement(visual, "map", znear="0.01", zfar="10", shadowclip="2", shadowscale="1")
    assets, worldbody = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    # The source cameras look into an enclosed textured room. A MuJoCo skybox
    # can occlude its far wall in RGB while the depth pass still sees the wall.
    # Keep actual room geometry in both channels instead of adding a skybox.
    texture_names = {}

    def add_meshes(body, record):
        if not _visible(record):
            return
        for index, component in enumerate(record.get("meshes", [])):
            # The native workspace boundary and floor outline are wireframes,
            # not filled surfaces. Rendering their faces would cover the table.
            if component.get("wireframe", False):
                continue
            prefix = f"{record['name']}_{index}"
            mesh_path = root_dir / component["obj_path"]
            ET.SubElement(assets, "mesh", name=prefix, file=str(mesh_path), inertia="shell")
            material = component.get("material", {})
            kwargs = {"name": prefix, "rgba": _numbers(material.get("rgba", [0.7, 0.7, 0.7, 1.])),
                      "specular": str(float(np.mean(material.get("specular", [0.25])))),
                      "shininess": str(float(material.get("shininess", 32.0)) / 128.),
                      "emission": str(float(np.max(material.get("emission", [0.]))))}
            if material.get("texture_path"):
                path = material["texture_path"]
                if path not in texture_names:
                    texture_names[path] = f"texture_{len(texture_names)}"
                    ET.SubElement(assets, "texture", name=texture_names[path], type="2d", file=str(root_dir / path))
                kwargs["texture"] = texture_names[path]
                kwargs["texuniform"] = "false"
            ET.SubElement(assets, "material", **kwargs)
            ET.SubElement(body, "geom", name=prefix, type="mesh", mesh=prefix,
                          material=prefix, contype="0", conaffinity="0", mass="0",
                          **_pose_attributes(component.get("local_matrix", np.eye(4))))

    children = {}
    for name, record in objects.items():
        # Each synchronized object has an independent rendering-only free
        # joint. This includes static supports randomized by the task reset.
        parent = None if name in movable_objects else record.get("parent")
        if parent not in objects:
            parent = None
        children.setdefault(parent, []).append(name)

    def add_body(parent_element, name):
        record = objects[name]
        transform = (_world(record) if parent_element is worldbody
                     else np.asarray(record["local_matrix"], float))
        body = ET.SubElement(parent_element, "body", name=name, **_pose_attributes(transform))
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.000001 0.000001 0.000001")
        if name in movable_objects:
            ET.SubElement(body, "freejoint", name=f"source_{name}")
        elif "joint" in record:
            subtype = str(record["joint"].get("subtype", "revolute")).lower()
            slide = "prismatic" in subtype or subtype == "11"
            limits = _joint_limits(record)
            ET.SubElement(body, "joint", name=name, type="slide" if slide else "hinge",
                          axis="0 0 1", ref=str(_joint_position(record)), range=_numbers(limits))
        add_meshes(body, record)
        if name.endswith("Arm_tip"):
            ET.SubElement(body, "site", name=name, type="sphere", size="0.0001", rgba="0 0 0 0")
        for child in children.get(name, []):
            add_body(body, child)

    for name in children.get(None, []):
        add_body(worldbody, name)
    lights = [record for record in objects.values() if record.get("light")]
    for index, record in enumerate(lights):
        light, matrix = record["light"], _world(record)
        ET.SubElement(worldbody, "light", name=f"source_light_{index}", pos=_numbers(matrix[:3, 3]),
                      dir=_numbers(matrix[:3, 2]), directional=str(light.get("type", "directional") == "directional").lower(),
                      active=str(bool(light.get("active", light.get("state", 1)))).lower(),
                      ambient="0 0 0", diffuse=_numbers(light.get("diffuse", [0.8, 0.8, 0.8])),
                      specular=_numbers(light.get("specular", [0.3, 0.3, 0.3])), castshadow="true")
    if not lights:
        ET.SubElement(worldbody, "light", name="source_default_light", pos="0 0 3", dir="0 0 -1",
                      directional="true", ambient="0 0 0", diffuse="0.8 0.8 0.8", specular="0.3 0.3 0.3")
    calibration = camera_calibration_from_manifest(manifest)
    for name in CAMERAS:
        ET.SubElement(worldbody, "camera", name=name, fovy=str(calibration[name]["fovy"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output_path, encoding="unicode")
    return output_path


class SourceVisualObservationRenderer:
    def __init__(self, env, assets_dir=Path("runtime/source_visuals"), output_dir=None):
        self.env, self.assets_dir = env, Path(assets_dir).resolve()
        self.manifest_path = self.assets_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing source visual export: {self.manifest_path}. "
                "Run scripts/mujoco/export_source_visuals.py first.")
        self.manifest = json.loads(self.manifest_path.read_text())
        self.objects = _objects(self.manifest)
        self.object_names = task_object_names(self.manifest)
        self.object_name = env.config.get("object_name", "ball")
        self.static_object_names = env.config.get("static_object_names", [])
        physical_objects = [self.object_name, *env.config.get("additional_object_names", []),
                            *self.static_object_names]
        if self.object_names != physical_objects:
            raise ValueError(f"Source visual objects {self.object_names} do not match "
                             f"the physical task objects {physical_objects}; choose the task's visual assets")
        self.xml_path = (Path(output_dir).resolve() / "source_visual_scene.xml" if output_dir
                         else self.assets_dir / "source_visual_scene.xml")
        build_source_visual_scene(self.manifest_path, self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.kinematics = {side: kinematics_from_manifest(self.manifest, side) for side in env.config["arms"]}
        self.calibration = camera_calibration_from_manifest(self.manifest)
        self.renderer = None
        self.last_metadata = {}
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    def _sync(self):
        state = self.env.adapter.state()
        alignment = {}
        for index, side in enumerate(self.env.config["arms"]):
            actual_joints = [self.env.data.joint(f"joint{i}_{side}").qpos[0] for i in range(1, 8)]
            joints, alignment[side] = self.kinematics[side].solve(pose_matrix(state[index, :7]), actual_joints)
            for number, position in enumerate(joints, 1):
                self.data.joint(f"Panda_{side}Arm_joint{number}").qpos[0] = position
            for number, finger in enumerate(("left", "right"), 1):
                opening = self.env.data.joint(f"{finger}_finger_{side}").qpos[0]
                name = f"Panda_{side}Arm_gripper_joint{number}"
                limits = _joint_limits(self.objects[name])
                self.data.joint(name).qpos[0] = np.clip(opening, *limits)
        object_poses = {}
        for name in self.object_names:
            if name in self.static_object_names:
                body = self.env.data.body(name)
                physical_matrix = np.eye(4)
                physical_matrix[:3, 3] = body.xpos
                physical_matrix[:3, :3] = body.xmat.reshape(3, 3)
            else:
                physical_pose = self.env.data.joint(name).qpos
                physical_matrix = pose_matrix(physical_pose[[0, 1, 2, 4, 5, 6, 3]])
            world_pose = self.env.adapter.world_transform @ physical_matrix
            source_pose = matrix_pose(world_pose)
            self.data.joint(f"source_{name}").qpos[:] = source_pose[[0, 1, 2, 6, 3, 4, 5]]
            object_poses[name] = source_pose.tolist()
        for name, calibration in self.calibration.items():
            camera = np.asarray(calibration["matrix"], float)
            if name != "front":
                index = self.env.config["arms"].index(name.split("_")[1])
                camera = pose_matrix(state[index, :7]) @ camera
            camera = camera @ OPTICAL_TO_GL
            cid = self.model.camera(name).id
            self.model.cam_pos[cid] = camera[:3, 3]
            self.model.cam_quat[cid] = Rotation.from_matrix(camera[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.last_metadata = {
            "renderer": "source_mujoco", "physics": "unchanged_physical_mujoco_environment",
            "manifest": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
            "robot_alignment": alignment, "object_name": self.object_name,
            "object_pose_xyzw": object_poses[self.object_name], "object_poses_xyzw": object_poses,
            "state_source": "actual_physical_mujoco_tcp_and_finger_positions",
            "camera_source": "authored_source_mounts_attached_to_actual_physical_tcp",
        }
        if self.object_name == "ball":
            self.last_metadata["ball_pose_xyzw"] = object_poses["ball"]
        return state

    def observe(self):
        state = self._sync()
        if self.renderer is None:
            size = self.env.config["image_size"]
            self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        rgb, points = [], []
        for name in CAMERAS:
            cid = self.model.camera(name).id
            self.model.vis.map.znear = self.calibration[name]["near"]
            self.model.vis.map.zfar = self.calibration[name]["far"]
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=name, scene_option=options)
            rgb.append(self.renderer.render().copy())
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            points.append(depth_to_world(depth, self.model.cam_fovy[cid],
                                         self.data.cam_xpos[cid], self.data.cam_xmat[cid]))
        return np.stack(rgb), np.asarray(points, np.float32), state

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
