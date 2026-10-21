"""Real-renderer synchronization checks, enabled when local assets/GL exist."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_transfer.environment import PandaBallEnvironment, read_config
from mujoco_transfer.initial_states import SourceEpisodeSampler


@pytest.mark.parametrize("backend", ["source_native", "source_mujoco"])
def test_live_rendering_preserves_physics_and_tracks_new_state(backend, tmp_path):
    flag = "TEST_NATIVE_RENDER" if backend == "source_native" else "TEST_SOURCE_RENDER"
    if os.environ.get(flag) != "1":
        pytest.skip(f"Requires {flag}=1 and local source assets/graphics")
    folder = Path("runtime/panda_ball_reference")
    initial = SourceEpisodeSampler("runtime/peract2_test.zip").sample(episode=94)
    env = PandaBallEnvironment(folder / "scene.xml", read_config(folder / "reference_config.json"),
                               initial_state=initial)
    renderer = None
    try:
        if backend == "source_native":
            from mujoco_transfer.native_renderer import NativeObservationRenderer
            renderer = NativeObservationRenderer(env, tmp_path / backend)
        else:
            from mujoco_transfer.source_visual_renderer import SourceVisualObservationRenderer
            renderer = SourceVisualObservationRenderer(env, output_dir=tmp_path / backend)

        def observe_and_check():
            before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time]
            expected_state = env.adapter.state().copy()
            rgb, points, state = renderer.observe()
            for actual, expected in zip((env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time), before):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(state, expected_state)
            assert rgb.shape == points.shape == (3, 256, 256, 3)
            assert rgb.dtype == np.uint8 and np.isfinite(points).all()
            metadata = renderer.last_metadata
            alignments = metadata.get("arms", metadata.get("robot_alignment"))
            for side in ("left", "right"):
                assert alignments[side]["position_error_m"] < .002
                assert alignments[side]["rotation_error_rad"] < .02
            np.testing.assert_allclose(metadata["ball_pose_xyzw"][:3],
                                       env.data.joint("ball").qpos[:3], atol=1e-6)
            # At this unoccluded reset, surface samples must follow the actual
            # ball centre. This checks the metric XYZ output as well as RGB.
            distance = np.linalg.norm(points - env.data.joint("ball").qpos[:3], axis=-1)
            assert np.count_nonzero(abs(distance - .15) < .008) > 100
            return rgb, points, state

        first = observe_and_check()
        env.data.joint("ball").qpos[0] += .08
        env.data.joint("joint7_left").qpos[0] += .04
        mujoco.mj_forward(env.model, env.data)
        second = observe_and_check()
        assert np.count_nonzero(first[0] != second[0]) > 1000
        assert np.max(abs(first[1] - second[1])) > .01
        assert np.max(abs(first[2] - second[2])) > .001
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
