"""Checks against independent native captures in the prepared visual export."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.source_visual_renderer import build_source_visual_scene, rotation_z


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


@pytest.fixture(scope="module", params=["source_visuals", "source_visuals_tray"])
def exported_scene(tmp_path_factory, request):
    manifest_path = RUNTIME / request.param / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Run export_source_visuals.py to prepare independent native captures")
    manifest = json.loads(manifest_path.read_text())
    path = build_source_visual_scene(manifest_path, tmp_path_factory.mktemp("visual_scene") / "scene.xml")
    return manifest, mujoco.MjModel.from_xml_path(str(path))


@pytest.mark.parametrize("sample_index", [0, 1, 2])
def test_rendered_hierarchy_matches_native_capture(exported_scene, sample_index):
    manifest, model = exported_scene
    objects = {record["name"]: record for record in manifest["objects"]}
    sample = manifest["kinematic_validation"][sample_index]
    data = mujoco.MjData(model)
    for name, position in sample["joint_positions"].items():
        data.joint(name).qpos[0] = position
    mujoco.mj_forward(model, data)
    for name, expected in sample["world_matrices"].items():
        actual = np.eye(4)
        actual[:3, 3] = data.body(name).xpos
        actual[:3, :3] = data.body(name).xmat.reshape(3, 3)
        record = objects[name]
        if "joint" in record:
            # Native joint objects expose the fixed joint frame; MuJoCo's
            # joint body includes its own displacement relative to qpos0.
            delta = sample["joint_positions"][name] - record["joint"]["position"]
            undo = np.eye(4)
            if record["joint"]["kind"] == "revolute":
                undo = rotation_z(-delta)
            else:
                undo[2, 3] = -delta
            actual = actual @ undo
        expected = np.asarray(expected)
        assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 2e-6, name
        rotation_error = Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude()
        assert rotation_error < 2e-6, name


def test_room_has_no_false_opaque_workspace_or_skybox(exported_scene):
    manifest, model = exported_scene
    # Native workspace/floor outlines are wireframes, so their triangle faces
    # must not occlude the table. A skybox can occlude the far wall only in RGB.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "workspace_0") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                           "ResizableFloor_5_25_visibleElement_2") == -1
    assert not np.any(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    object_geom = "tray_visual_0" if manifest.get("task", {}).get("object_name") == "tray" else "ball_0"
    for name in (object_geom, "Wall3_0", "diningTable_visible_1"):
        material = model.geom(name).matid[0]
        assert model.mat_rgba[material, 3] == 1.0


def test_tray_item_and_supports_remain_independent(exported_scene):
    manifest, model = exported_scene
    if manifest.get("task", {}).get("object_name") != "tray":
        pytest.skip("Tray-specific independent object synchronization")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ("tray", "item", "drawer_legs", "drawer_legs0")
    before = {name: data.body(name).xpos.copy() for name in (*names, "tray_visual")}
    # Although the source scene parents the item to the tray, physical contact
    # can release it. Rendering must not falsely keep that item attached.
    displacement = np.array([.1, -.2, .3])
    data.joint("source_tray").qpos[:3] += displacement
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("tray").xpos, before["tray"] + displacement)
    np.testing.assert_allclose(data.body("tray_visual").xpos, before["tray_visual"] + displacement)
    for name in names[1:]:
        np.testing.assert_array_equal(data.body(name).xpos, before[name])
    data.joint("source_item").qpos[:3] -= displacement
    data.joint("source_drawer_legs").qpos[0] += .2
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("item").xpos, before["item"] - displacement)
    np.testing.assert_allclose(data.body("tray").xpos, before["tray"] + displacement)
    np.testing.assert_array_equal(data.body("drawer_legs0").xpos, before["drawer_legs0"])
