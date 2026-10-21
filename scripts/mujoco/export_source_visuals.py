#!/usr/bin/env python3
"""Export the authored PerAct2 scene's visual assets without running physics.

Requires the audit-only CoppeliaSim 4.1.0/PyRep installation. The resulting
manifest, OBJ meshes and PNG textures are consumed without CoppeliaSim by the
MuJoCo source-appearance renderer. No collision model is exported or modified.

Example (with the PyRep environment configured):
    python scripts/mujoco/export_source_visuals.py \
        --rlbench-root /tmp/3dfa_rlbench_source --output runtime/source_visuals

Mesh vertices/normals are in the named shape frame; OBJ UVs use the OpenGL
bottom-left convention. PNG files are saved top-down, with the native texture
rows flipped once. local_matrix is inv(parent.world_matrix) @ world_matrix;
for children of joints it includes the captured parent's joint displacement.
The joint intrinsic_matrix records that displacement explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


TASKS = {
    "bimanual_lift_ball": {"object_name": "ball", "additional_object_names": [],
                           "static_object_names": [], "fixture": "peract2_scene.json",
                           "assets": "source_visuals"},
    "bimanual_lift_tray": {"object_name": "tray", "additional_object_names": ["item"],
                           "static_object_names": ["drawer_legs", "drawer_legs0"],
                           "fixture": "peract2_tray_scene.json", "assets": "source_visuals_tray"},
}


def source_files(task_name):
    return {
        "scene": "rlbench/task_design_bimanual.ttt",
        "task_model": f"rlbench/task_ttms/{task_name}.ttm",
        "task_code": f"rlbench/bimanual_tasks/{task_name}.py",
    }


def matrix4(matrix):
    result = np.eye(4)
    result[:3] = np.asarray(matrix).reshape(3, 4)
    return result


def write_obj(path, vertices, faces, normals, uv):
    """Preserve distinct corner normals/UVs rather than merging mesh seams."""
    with path.open("w") as stream:
        stream.write("# Native CoppeliaSim mesh, metres, shape-local frame\n")
        for row in vertices:
            stream.write("v " + " ".join(f"{x:.9g}" for x in row) + "\n")
        for row in uv:
            stream.write("vt " + " ".join(f"{x:.9g}" for x in row) + "\n")
        for row in normals:
            stream.write("vn " + " ".join(f"{x:.9g}" for x in row) + "\n")
        for i, face in enumerate(faces):
            corners = []
            for j, vertex in enumerate(face):
                k = 3 * i + j + 1
                corners.append(f"{vertex + 1}/{k}/{k}" if len(uv)
                               else f"{vertex + 1}//{k}")
            stream.write("f " + " ".join(corners) + "\n")


def shape_components(sim, handle):
    """Read every compound component, including textures, freeing native buffers.

    PyRep's wrapper discards the return code and does not free these buffers;
    the direct C API distinguishes end-of-components from untextured meshes.
    """
    for index in range(10000):
        info = sim.ffi.new("struct SShapeVizInfo *")
        result = sim.lib.simGetShapeViz(handle, index, info)
        if result < 0:
            raise RuntimeError(f"simGetShapeViz failed for {handle}, {index}")
        if result == 0:
            return
        buffers = [info.vertices, info.indices, info.normals]
        if result > 1:
            buffers += [info.texture, info.textureCoords]
        try:
            vertices = np.array([info.vertices[i] for i in range(info.verticesSize)]).reshape(-1, 3)
            faces = np.array([info.indices[i] for i in range(info.indicesSize)], dtype=int).reshape(-1, 3)
            normals = np.array([info.normals[i] for i in range(info.indicesSize * 3)]).reshape(-1, 3)
            texture = np.empty((0, 0, 4), dtype=np.uint8)
            uv = np.empty((0, 2))
            if result > 1:
                width, height = list(info.textureRes)
                texture = np.frombuffer(sim.ffi.buffer(info.texture, width * height * 4),
                                        dtype=np.uint8).copy().reshape(height, width, 4)
                uv = np.array([info.textureCoords[i] for i in range(info.indicesSize * 2)]).reshape(-1, 2)
            record = {
                "component": index, "vertices": vertices, "faces": faces,
                "normals": normals, "uv": uv, "texture": texture,
                "colors": list(info.colors), "shading_angle": float(info.shadingAngle),
                "texture_id": int(info.textureId),
                "texture_apply_mode": int(info.textureApplyMode),
                "texture_options": int(info.textureOptions),
            }
        finally:
            for buffer in buffers:
                if buffer != sim.ffi.NULL:
                    sim.lib.simReleaseBuffer(sim.ffi.cast("char *", buffer))
        yield record
    raise RuntimeError("Unexpectedly many shape components")


def export(root, output, source_commit, task_name="bimanual_lift_ball"):
    from pyrep import PyRep
    from pyrep.backend import sim
    from pyrep.const import ObjectType, Verbosity
    from pyrep.objects.shape import Shape
    from pyrep.objects.vision_sensor import VisionSensor

    files = source_files(task_name)
    task_spec = TASKS[task_name]
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    simulator = PyRep()
    simulator.launch(str(root / files["scene"]), headless=True,
                     verbosity=Verbosity.ERRORS)
    try:
        task = simulator.import_model(str(root / files["task_model"]))
        task.set_position(Shape("workspace").get_position())
        # The XML serialization also retains visual switches not exposed by
        # simGetShapeViz (notably whether translucency is enabled).
        (output / "native").mkdir(exist_ok=True)
        sim.simSaveScene(str(output / "native" / "source.simscene.xml"))
        xml_scene = ET.parse(output / "native" / "source.simscene.xml").getroot()
        xml_objects = {node[0].findtext("common/name"): node[0]
                       for node in xml_scene.findall("sceneObject")}
        handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_handle_all, 0)
        names = {h: sim.simGetObjectName(h) for h in handles}
        for name in [task_spec["object_name"], *task_spec["additional_object_names"],
                     *task_spec["static_object_names"]]:
            if name not in names.values():
                raise ValueError(f"Task {task_name} has no movable object {name!r}")
        matrices = {h: matrix4(sim.simGetObjectMatrix(h, -1)) for h in handles}
        records = []
        for handle in handles:
            name = names[handle]
            kind = ObjectType(sim.simGetObjectType(handle)).name
            # -1 is the valid root sentinel; PyRep's generic wrapper rejects it.
            parent = int(sim.lib.simGetObjectParent(handle))
            world = matrices[handle]
            local = np.linalg.inv(matrices[parent]) @ world if parent in matrices else world
            properties = sim.simGetObjectSpecialProperty(handle)
            record = {
                "handle": handle, "name": name, "type": kind,
                "parent": names.get(parent), "local_matrix": local.tolist(),
                "world_matrix": world.tolist(),
                "special_properties": properties,
                "model_properties": sim.simGetModelProperty(handle),
                "renderable": bool(properties & sim.sim_objectspecialproperty_renderable),
                "visibility_layer": sim.simGetObjectInt32Parameter(handle, sim.sim_objintparam_visibility_layer),
                "meshes": [],
            }
            xml_object = xml_objects[name]
            record["authorized_viewable_objects"] = int(xml_object.findtext("common/authorizedViewableObjects", "-1"))
            if kind == "JOINT":
                cyclic, interval = sim.simGetJointInterval(handle)
                subtype = sim.simGetJointType(handle)
                record["joint"] = {
                    "position": sim.simGetJointPosition(handle), "subtype": subtype,
                    "kind": {sim.sim_joint_revolute_subtype: "revolute",
                             sim.sim_joint_prismatic_subtype: "prismatic",
                             sim.sim_joint_spherical_subtype: "spherical"}[subtype],
                    "axis": [0, 0, 1], "cyclic": cyclic, "interval": interval,
                    "intrinsic_matrix": matrix4(sim.simGetJointMatrix(handle)).tolist(),
                }
            if kind == "SHAPE":
                record["culling"] = bool(sim.simGetObjectInt32Parameter(handle, sim.sim_shapeintparam_culling))
                record["respondable"] = bool(sim.simGetObjectInt32Parameter(handle, sim.sim_shapeintparam_respondable))
                xml_components = list(xml_object.find("meshProxy").iter("mesh"))
                for component in shape_components(sim, handle):
                    xml_component = xml_components[component["component"]]
                    xml_color = xml_component.find("color")
                    xml_rgb = np.fromstring(xml_color.findtext("ambient"), sep=" ")
                    if not np.allclose(xml_rgb, component["colors"][:3], atol=1e-6, rtol=0):
                        raise ValueError(f"XML/component order disagrees for {name}/{component['component']}")
                    translucent = xml_color.findtext("switches/transparent") == "true"
                    opacity = float(xml_color.findtext("transparency")) if translucent else 1.0
                    slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)
                    mesh_path = Path("meshes") / f"{handle}_{slug}_{component['component']}.obj"
                    write_obj(output / mesh_path, component["vertices"], component["faces"],
                              component["normals"], component["uv"])
                    texture_path = None
                    if component["texture"].size:
                        texture_path = Path("textures") / f"texture_{component['texture_id']}.png"
                        Image.fromarray(component["texture"][::-1]).save(output / texture_path)
                    colors = component["colors"]
                    record["meshes"].append({
                        "component": component["component"], "obj_path": str(mesh_path),
                        "local_matrix": np.eye(4).tolist(),
                        "vertex_count": len(component["vertices"]),
                        "triangle_count": len(component["faces"]),
                        "shading_angle": component["shading_angle"],
                        "wireframe": xml_component.findtext("switches/wireframe") == "true",
                        "culling": xml_component.findtext("switches/culling") == "true",
                        "edges_visible": xml_component.findtext("switches/edgesVisible") == "true",
                        "material": {
                            "rgba": colors[:3] + [opacity], "specular": colors[3:6],
                            "emission": colors[6:9], "texture_path": str(texture_path) if texture_path else None,
                            "translucent": translucent, "shininess": int(xml_color.findtext("shininess")),
                            "texture_id": component["texture_id"],
                            "texture_apply_mode": component["texture_apply_mode"],
                            "texture_options": component["texture_options"],
                        },
                    })
            if kind == "VISION_SENSOR":
                camera = VisionSensor(handle)
                record["camera"] = {
                    "resolution": camera.get_resolution(), "perspective_angle_degrees": camera.get_perspective_angle(),
                    "near": camera.get_near_clipping_plane(), "far": camera.get_far_clipping_plane(),
                    "render_mode": camera.get_render_mode().value,
                    "disabled_light_components": sim.simGetObjectInt32Parameter(handle, sim.sim_visionintparam_disabled_light_components),
                    "rendering_attributes": sim.simGetObjectInt32Parameter(handle, sim.sim_visionintparam_rendering_attributes),
                    "entity_to_render": sim.simGetObjectInt32Parameter(handle, sim.sim_visionintparam_entity_to_render),
                    "optical_to_opengl": np.diag([-1., 1., -1., 1.]).tolist(),
                    "same_background_as_environment": xml_object.findtext("switches/sameBackgroundAsEnvironment") == "true",
                    "background_rgb": np.fromstring(xml_object.findtext("defaultBufferValues"), sep=" ").tolist(),
                }
            if kind == "LIGHT":
                state, diffuse, specular = sim.simGetLightParameters(handle)
                record["light"] = {"state": state, "diffuse": diffuse, "specular": specular,
                                   "type": xml_object.findtext("type"),
                                   "local": xml_object.findtext("switches/lightIsLocal") == "true",
                                   "extension_string": xml_object.findtext("common/extensionString")}
                for field, constant in (
                    ("spot_exponent", sim.sim_lightfloatparam_spot_exponent),
                    ("spot_cutoff", sim.sim_lightfloatparam_spot_cutoff),
                    ("constant_attenuation", sim.sim_lightfloatparam_const_attenuation),
                    ("linear_attenuation", sim.sim_lightfloatparam_lin_attenuation),
                    ("quadratic_attenuation", sim.sim_lightfloatparam_quad_attenuation),
                ):
                    record["light"][field] = sim.simGetObjectFloatParameter(handle, constant)
            records.append(record)
            print(f"Exported {name}: {kind}, {len(record['meshes'])} mesh components", flush=True)
        scene = {
            "background_color1": sim.simGetArrayParameter(sim.sim_arrayparam_background_color1),
            "background_color2": sim.simGetArrayParameter(sim.sim_arrayparam_background_color2),
            "ambient_light": sim.simGetArrayParameter(sim.sim_arrayparam_ambient_light),
            "fog_enabled": xml_scene.findtext("environment/fog/enabled") == "true",
        }
        validation = kinematic_checks(sim, records)
        manifest = {
            "schema_version": 1,
            "task": {"name": task_name, "object_name": task_spec["object_name"],
                     "additional_object_names": task_spec["additional_object_names"],
                     "static_object_names": task_spec["static_object_names"]},
            "source": {
                "repository": "https://github.com/markusgrotz/RLBench", "commit": source_commit,
                "files": {key: {"path": value, "sha256": hashlib.sha256((root / value).read_bytes()).hexdigest()}
                          for key, value in files.items()},
                "simulator": "CoppeliaSim 4.1.0", "physics_stepped": False,
            },
            "conventions": {
                "length": "metres", "angles": "radians", "matrix": "4x4 row-major homogeneous",
                "local_matrix": "inv(parent world_matrix) @ world_matrix at captured joint positions",
                "joint_frame": "world_matrix is joint base; intrinsic_matrix maps moving joint frame into base",
                "mesh_frame": "named shape local frame; component transform already baked into vertices",
                "texture_rows": "PNG top-down; native bottom-up RGBA rows flipped once",
                "uv": "OpenGL bottom-left; OBJ retains distinct UV/normal for every triangle corner",
                "transparency": "rgba alpha is per-component opacity with XML translucency-enable flag applied",
            },
            "scene": scene, "objects": records, "kinematic_validation": validation,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {output / 'manifest.json'} ({len(records)} objects)", flush=True)
        return manifest
    finally:
        simulator.shutdown()


def kinematic_checks(sim, records):
    """Capture native transforms at perturbed q to verify downstream FK.

    This changes only kinematic positions in the audit simulator, never advances
    time, and restores the captured configuration before returning.
    """
    from scipy.spatial.transform import Rotation

    objects = {record["name"]: record for record in records}
    joints = [record for record in records if "joint" in record]
    checks = []
    try:
        for sample in range(3):
            positions = {}
            for index, record in enumerate(joints):
                joint = record["joint"]
                delta = (0.12 if joint["kind"] == "revolute" else 0.004) * np.sin(index + 0.73 * sample)
                q = joint["position"] + (delta if sample else 0)
                if not joint["cyclic"]:
                    lo, size = joint["interval"]
                    q = np.clip(q, lo, lo + size)
                positions[record["name"]] = float(q)
                sim.simSetJointPosition(record["handle"], float(q))
            native = {record["name"]: matrix4(sim.simGetObjectMatrix(record["handle"], -1))
                      for record in records}
            calculated = {}

            def forward(name):
                if name in calculated:
                    return calculated[name]
                record = objects[name]
                local = np.array(record["local_matrix"])
                parent = objects.get(record["parent"])
                if parent is None:
                    result = local
                else:
                    parent_world = forward(parent["name"])
                    if "joint" in parent:
                        joint = parent["joint"]
                        intrinsic = np.eye(4)
                        if joint["kind"] == "revolute":
                            intrinsic[:3, :3] = Rotation.from_rotvec([0, 0, positions[parent["name"]]]).as_matrix()
                        elif joint["kind"] == "prismatic":
                            intrinsic[2, 3] = positions[parent["name"]]
                        else:
                            raise ValueError("Spherical joints require a separate validator")
                        local = intrinsic @ np.linalg.inv(joint["intrinsic_matrix"]) @ local
                    result = parent_world @ local
                calculated[name] = result
                return result

            position_errors, rotation_errors = [], []
            for name in native:
                got = forward(name)
                position_errors.append(float(np.linalg.norm(got[:3, 3] - native[name][:3, 3])))
                rotation_errors.append(float(Rotation.from_matrix(got[:3, :3] @ native[name][:3, :3].T).magnitude()))
            if max(position_errors) > 3e-6 or max(rotation_errors) > 3e-6:
                raise ValueError(f"Exported hierarchy FK disagrees with native simulator: {max(position_errors)}, {max(rotation_errors)}")
            checks.append({
                "sample": sample, "joint_positions": positions,
                "world_matrices": {name: matrix.tolist() for name, matrix in native.items()},
                "max_position_error_m": max(position_errors),
                "max_rotation_error_rad": max(rotation_errors),
            })
    finally:
        for record in joints:
            sim.simSetJointPosition(record["handle"], record["joint"]["position"])
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rlbench-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="bimanual_lift_ball")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    task_spec = TASKS[args.task]
    fixture_path = Path(__file__).resolve().parents[2] / "mujoco_transfer" / "configs" / task_spec["fixture"]
    fixture = json.loads(fixture_path.read_text())
    source_commit = args.source_commit or fixture["source"]["commit"]
    for key, path in source_files(args.task).items():
        if not (args.rlbench_root / path).is_file():
            parser.error(f"Missing source file: {args.rlbench_root / path}")
        if source_commit == fixture["source"]["commit"]:
            actual = hashlib.sha256((args.rlbench_root / path).read_bytes()).hexdigest()
            if actual != fixture["source"][key]["sha256"]:
                parser.error(f"Source {key} does not match the pinned {args.task} fixture")
    output = args.output or Path("runtime") / task_spec["assets"]
    export(args.rlbench_root.resolve(), output.resolve(), source_commit, args.task)


if __name__ == "__main__":
    main()
