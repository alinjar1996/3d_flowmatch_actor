"""Checks against independent native captures in the prepared visual export."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.source_visual_renderer import build_source_visual_scene, rotation_z


ASSETS = Path(__file__).resolve().parents[1] / "runtime" / "source_visuals"


@pytest.fixture(scope="module")
def exported_scene(tmp_path_factory):
    manifest_path = ASSETS / "manifest.json"
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
    _, model = exported_scene
    # Native workspace/floor outlines are wireframes, so their triangle faces
    # must not occlude the table. A skybox can occlude the far wall only in RGB.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "workspace_0") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                           "ResizableFloor_5_25_visibleElement_2") == -1
    assert not np.any(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    for name in ("ball_0", "Wall3_0", "diningTable_visible_1"):
        material = model.geom(name).matid[0]
        assert model.mat_rgba[material, 3] == 1.0
