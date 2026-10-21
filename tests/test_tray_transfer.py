"""Physical tray transfer integration against independently audited source data.

These checks use prepared assets and numeric released states; no GPU, rendering
context, policy model, demonstration action or native simulator is required.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.environment import PandaBallEnvironment, read_config
from mujoco_transfer.geometry import pose_matrix
from mujoco_transfer.initial_states import SourceEpisodeSampler
from mujoco_transfer.reference import NumericReferenceUnpickler
from mujoco_transfer.tray import inspect_tray_settings


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "runtime/panda_tray_reference/scene.xml"
CONFIG = SCENE.parent / "reference_config.json"
ARCHIVE = ROOT / "runtime/peract2_test.zip"
SOURCE = ROOT / "mujoco_transfer/configs/peract2_tray_scene.json"
VISUALS = ROOT / "runtime/source_visuals_tray"
OBJECTS = ("tray", "item", "drawer_legs", "drawer_legs0")


@pytest.fixture(scope="module")
def inputs():
    if not all(path.is_file() for path in (SCENE, CONFIG, ARCHIVE, SOURCE)):
        pytest.skip("Prepare the released tray task before running asset integration tests")
    source = json.loads(SOURCE.read_text())
    config = read_config(CONFIG)
    sampler = SourceEpisodeSampler(ARCHIVE, seed=17, task="bimanual_lift_tray")
    return source, config, sampler


@pytest.fixture
def environment(inputs):
    _, config, sampler = inputs
    env = PandaBallEnvironment(SCENE, deepcopy(config), initial_state=sampler.sample(0))
    try:
        yield env
    finally:
        env.close()


def assert_body_pose(data, name, expected):
    actual = data.body(name)
    np.testing.assert_allclose(actual.xpos, expected[:3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual.xmat.reshape(3, 3),
                               Rotation.from_quat(expected[3:]).as_matrix(), atol=1e-10, rtol=0)


def test_all_100_resets_preserve_coupled_source_states_and_supports(inputs, environment):
    source, _, sampler = inputs
    env = environment
    assert sampler.episodes == tuple(range(100))
    indices = source["released_resets"]["object_pose_indices"]
    with zipfile.ZipFile(ARCHIVE) as archive:
        for episode in sampler.episodes:
            initial = sampler.sample(episode)
            # Decode the source state independently of the sampler's name-to-
            # index extraction; the index fixture comes from native task order.
            demo = NumericReferenceUnpickler(io.BytesIO(archive.read(initial["member"]))).load()
            first = demo._observations[0]
            task_state = np.asarray(first.task_low_dim_state).reshape(-1, 7)
            env._fixed_initial_state = initial
            env.reset()
            for name in OBJECTS:
                expected = task_state[indices[name]]
                np.testing.assert_array_equal(initial["object_poses_xyzw"][name], expected)
                assert_body_pose(env.data, name, expected)
            for side in ("left", "right"):
                expected = getattr(first, side)
                np.testing.assert_array_equal(
                    [env.data.joint(f"joint{i}_{side}").qpos[0] for i in range(1, 8)],
                    expected.joint_positions)
                assert expected.gripper_open == 1.
            assert env.data.time == 0.
            np.testing.assert_array_equal(env.data.qvel, 0.)
            assert not env.success
            # Rack-cylinder transforms and collision filtering must actually
            # support the authored tray at every independently sampled pose.
            support_contacts = [contact for contact in env.data.contact
                                if "tray" in (env.model.geom(contact.geom1).name,
                                              env.model.geom(contact.geom2).name)
                                and any(env.model.geom(index).name.startswith("drawer_legs0_")
                                        for index in (contact.geom1, contact.geom2))]
            assert support_contacts, f"Tray has no support contact at episode {episode}"
            assert min(contact.dist for contact in support_contacts) > -1e-4


def test_dynamic_objects_are_unattached_with_physical_inertias(inputs, environment):
    source, _, _ = inputs
    env = environment
    assert env.model.neq == 0
    assert env.model.nmocap == 0
    for name in ("tray", "item"):
        body = env.model.body(name)
        joint = env.model.joint(name)
        assert body.parentid[0] == 0
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
        assert not np.any(env.model.actuator_trnid[:, 0] == joint.id)
        record = source["objects"][name]
        size = np.asarray(record["primitive_components"][0]["full_size_m"])
        expected_inertia = record["mass_kg"] / 12 * (np.sum(size ** 2) - size ** 2)
        np.testing.assert_allclose(body.mass[0], record["mass_kg"], rtol=1e-12)
        np.testing.assert_allclose(body.inertia, expected_inertia, rtol=1e-12)
        assert np.all(body.inertia > 0)
        assert 2 * max(body.inertia) <= sum(body.inertia)

    item_before = env.data.joint("item").qpos.copy()
    rack_before = [env.data.body(name).xpos.copy() for name in ("drawer_legs", "drawer_legs0")]
    env.data.joint("tray").qpos[:3] += [.12, -.03, .2]
    env.data.joint("joint7_left").qpos[0] += .1
    mujoco.mj_forward(env.model, env.data)
    np.testing.assert_array_equal(env.data.joint("item").qpos, item_before)
    for name, expected in zip(("drawer_legs", "drawer_legs0"), rack_before):
        np.testing.assert_array_equal(env.data.body(name).xpos, expected)


def test_source_visual_sync_keeps_item_independent_and_physics_unchanged(inputs, environment, tmp_path):
    if not (VISUALS / "manifest.json").is_file():
        pytest.skip("Prepare source tray visuals")
    from mujoco_transfer.source_visual_renderer import SourceVisualObservationRenderer

    env = environment
    renderer = SourceVisualObservationRenderer(env, assets_dir=VISUALS, output_dir=tmp_path)
    try:
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
        actual_state = env.adapter.state().copy()
        np.testing.assert_array_equal(renderer._sync(), actual_state)
        for actual, expected in zip((env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time), before):
            np.testing.assert_array_equal(actual, expected)
        for name in OBJECTS:
            np.testing.assert_allclose(renderer.data.body(name).xpos, env.data.body(name).xpos, atol=1e-10)
            np.testing.assert_allclose(renderer.data.body(name).xmat, env.data.body(name).xmat, atol=1e-10)

        tray_before = renderer.data.joint("source_tray").qpos.copy()
        env.data.joint("item").qpos[:3] += [.09, .02, .04]
        env.data.joint("item").qpos[3:] = Rotation.from_euler("xyz", [.2, -.3, .5]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(env.model, env.data)
        renderer._sync()
        np.testing.assert_array_equal(renderer.data.joint("source_tray").qpos, tray_before)
        np.testing.assert_allclose(renderer.data.body("item").xpos, env.data.body("item").xpos, atol=1e-10)
        np.testing.assert_allclose(renderer.data.body("item").xmat, env.data.body("item").xmat, atol=1e-10)
        # The source visual child follows the tray, not the independent item.
        source_objects = {record["name"]: record for record in renderer.manifest["objects"]}
        expected_visual = pose_matrix(renderer.last_metadata["object_poses_xyzw"]["tray"]) @ np.asarray(
            source_objects["tray_visual"]["local_matrix"])
        np.testing.assert_allclose(renderer.data.body("tray_visual").xpos, expected_visual[:3, 3], atol=1e-10)
    finally:
        renderer.close()


@pytest.mark.parametrize("fault", ["rotated_support_cylinder", "disabled_tray_collision"])
def test_source_verifier_rejects_physical_geometry_regressions(inputs, environment, fault):
    source, _, _ = inputs
    env = environment
    if fault == "rotated_support_cylinder":
        env.model.geom("drawer_legs0_0").quat[:] = [1., 0., 0., 0.]
    else:
        env.model.geom("tray").contype[0] = 0
        env.model.geom("tray").conaffinity[0] = 0
    mujoco.mj_forward(env.model, env.data)
    report = inspect_tray_settings(env.model, env.data, env.config, source,
                                   poses=env.initial_state["object_poses_xyzw"])
    assert not report["passed"], f"Source parity verification accepted {fault}"
