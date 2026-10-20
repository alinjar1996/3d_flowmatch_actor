"""Geometry and physical-control regressions; no model/network needed by default."""

import os
import io
from pathlib import Path
import pickle

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from mujoco_transfer.environment import PandaBallEnvironment, read_config
from mujoco_transfer.geometry import (
    depth_to_world, fit_rigid_transform, matrix_pose, pose_matrix,
    rigid_transform, transform_points,
)
from mujoco_transfer.policy import PoseHistory, checkpoint_state


def test_depth_camera_convention_and_pixel_centres():
    # Identity MuJoCo camera looks down -Z. Top-left is negative X, positive Y.
    points = depth_to_world(np.full((2, 2), 2.0), 90, [1, 2, 3], np.eye(3))
    np.testing.assert_allclose(points[0, 0], [0, 3, 1], atol=1e-8)
    np.testing.assert_allclose(points[1, 1], [2, 1, 1], atol=1e-8)
    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    points = depth_to_world(np.array([[2.0]]), 60, [1, 2, 3], rotation)
    np.testing.assert_allclose(points[0, 0], [-1, 2, 3], atol=1e-8)


def test_rigid_fit_recovers_rotation_and_translation():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.7, 1.1]).as_matrix()
    transform[:3, 3] = [0.3, -0.2, 0.8]
    fitted, rms = fit_rigid_transform(source, transform_points(source, transform))
    np.testing.assert_allclose(fitted, transform, atol=1e-10)
    assert rms < 1e-10
    with pytest.raises(ValueError, match="collinear"):
        fit_rigid_transform(np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="reflection"):
        rigid_transform(np.diag([-1, 1, 1, 1]))


def test_history_is_oldest_first_replicated_and_reset():
    history = PoseHistory()
    a, b, c, d = [np.full((2, 8), i, dtype=np.float32) for i in range(4)]
    np.testing.assert_array_equal(history.append(a), [a, a, a])
    np.testing.assert_array_equal(history.append(b), [a, a, b])
    history.append(c)
    np.testing.assert_array_equal(history.append(d), [b, c, d])
    history.reset()
    np.testing.assert_array_equal(history.append(d), [d, d, d])
    # Mutating the caller's array must not corrupt retained history.
    d[:] = 10
    assert history.states[-1][0, 0] == 3


def test_checkpoint_prefixes_and_invalid_normalizer(tmp_path):
    path = tmp_path / "checkpoint.pth"
    bounds = torch.tensor([[0., 0, 0], [1, 1, 1]])
    torch.save({"weight": {"module._orig_mod.workspace_normalizer": bounds,
                           "module.traj_encoder.weight": torch.ones(2, 3)}}, path)
    state = checkpoint_state(path)
    assert set(state) == {"workspace_normalizer", "traj_encoder.weight"}
    torch.save({"workspace_normalizer": torch.zeros(2, 3)}, path)
    with pytest.raises(ValueError, match="bounds"):
        checkpoint_state(path)


@pytest.fixture
def panda():
    scene = Path(os.environ.get("PANDA_SCENE", "runtime/panda_ball/scene.xml"))
    if not scene.exists():
        pytest.skip("Run python -m mujoco_transfer.prepare first")
    env = PandaBallEnvironment(scene, read_config())
    yield env
    env.close()


def test_tcp_and_world_transform_roundtrip_for_both_arms(panda):
    adapter = panda.adapter
    adapter.world_transform = pose_matrix(
        np.r_[[0.3, -0.2, 0.1], Rotation.from_euler("z", 0.7).as_quat()])
    for i, side in enumerate(["left", "right"]):
        adapter.tcp_transforms[side] = pose_matrix(
            np.r_[[0, 0, 0.03 + i * 0.01], Rotation.from_euler("x", 0.3 + i).as_quat()])
    targets = adapter.action_to_sites(adapter.state())
    for side, target in zip(["left", "right"], targets):
        np.testing.assert_allclose(target, adapter.site_transform(side), atol=1e-6)


def test_physical_controller_reaches_small_translation_and_rotation(panda):
    action = panda.adapter.state()
    action[:, 2] += 0.02
    for i in range(2):
        action[i, 3:7] = (
            Rotation.from_euler("z", 0.12) * Rotation.from_quat(action[i, 3:7])).as_quat()
    original_qpos = panda.data.qpos.copy()
    result = panda.controller.execute(action)
    assert result["reached"], result
    assert max(result["position_errors_m"]) < 0.012
    assert np.linalg.norm(panda.data.qpos - original_qpos) > 0.01
    assert panda.data.time > 0.5


def test_gripper_command_polarity_and_episode_reset(panda):
    action = panda.adapter.state()
    action[:, 7] = 0
    result = panda.controller.execute(action)
    assert result["reached"]
    assert np.all(panda.adapter.state()[:, 7] < 0.15)
    action = panda.adapter.state()
    action[:, 7] = 1
    panda.controller.execute(action)
    assert np.all(panda.adapter.state()[:, 7] > 0.85)
    panda.reset()
    assert not panda.success
    assert np.all(panda.adapter.state()[:, 7] > 0.85)


def test_unreachable_keypose_reports_failure_without_closing_grippers(panda):
    panda.controller.config["max_seconds"] = 0.04
    action = panda.adapter.state()
    action[:, 2] += 5
    action[:, 7] = 0
    assert not panda.controller.execute(action)["reached"]
    assert np.all(panda.adapter.state()[:, 7] > 0.85)


def test_reference_unpickler_rejects_unexpected_classes():
    from mujoco_transfer.reference import NumericReferenceUnpickler
    payload = pickle.dumps(Path("/tmp/reference"))
    with pytest.raises(ValueError, match="Unexpected reference pickle global"):
        NumericReferenceUnpickler(io.BytesIO(payload)).load()


def test_reference_keypose_uses_redundancy_at_wrist_limit():
    folder = Path("runtime/panda_ball_reference")
    if not (folder / "reference_config.json").exists():
        pytest.skip("Prepare the reference-fitted scene first")
    env = PandaBallEnvironment(folder / "scene.xml", read_config(folder / "reference_config.json"))
    # Actual checkpoint prediction that stalled when unconstrained IK was
    # followed by clipping. The right wrist reaches its upper limit en route.
    action = np.array([
        [.26088321, .07530195, .85578644, .09854834, .96004695, .07440616, .25112101, 1],
        [.33473235, -.16281193, .90676761, -.54573673, .79923588, .09569226, .23288704, 1],
    ])
    try:
        result = env.controller.execute(action)
        assert result["reached"], result
        assert max(result["position_errors_m"]) < .01
        assert max(result["rotation_errors_rad"]) < .10
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("TEST_RENDER") != "1", reason="requires OpenGL context")
def test_rendered_depth_reprojects_to_plane():
    # This checks real Renderer depth units and row ordering, independently of
    # the adapter: every unprojected pixel must land on the tilted world plane.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <visual><quality offsamples="0"/></visual>
      <worldbody>
        <geom type="plane" size="20 20 .1"/>
        <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 .8660254 .5" fovy="45"/>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=32, width=40) as renderer:
        renderer.update_scene(data, camera="cam")
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    points = depth_to_world(depth, 45, data.cam_xpos[0], data.cam_xmat[0])
    # Allow 0.1mm for finite-precision depth-buffer quantization.
    np.testing.assert_allclose(points[..., 2], 0, atol=1e-4)


@pytest.mark.skipif(os.environ.get("TEST_CHECKPOINT") != "1", reason="requires downloaded checkpoint")
def test_full_checkpoint_loads_strictly_and_matches_upstream_behavior():
    from mujoco_transfer.policy import checkpoint_state, load_actor
    path = "checkpoints/3dfa_peract2.pth"
    actor = load_actor(path)
    assert len(actor.legacy_unused_checkpoint_keys) == 34
    original = checkpoint_state(path)
    for key, tensor in actor.state_dict().items():
        assert torch.equal(tensor, original[key]), key
    # Exactly the keys ignored by upstream's strict=False load, with no missing weights.
    result = actor.load_state_dict(original, strict=False)
    assert not result.missing_keys
    assert set(result.unexpected_keys) == set(actor.legacy_unused_checkpoint_keys)
