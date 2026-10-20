"""Source reset sampling preserves coupled states, provenance, and RNG isolation."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import types
import zipfile

import numpy as np
import pytest

from mujoco_transfer.initial_states import SourceEpisodeSampler, SOURCE_ARCHIVE_SHA256


@pytest.fixture
def make_archive(tmp_path, monkeypatch):
    """Write only classes that the restricted source unpickler recognizes."""
    modules = {}
    for name in ("rlbench", "rlbench.demo", "rlbench.backend",
                 "rlbench.backend.observation"):
        modules[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    classes = {}
    for module, name in (("rlbench.demo", "Demo"),
                         ("rlbench.backend.observation", "BimanualObservation"),
                         ("rlbench.backend.observation", "UnimanualObservationData")):
        cls = type(name, (), {"__module__": module})
        setattr(modules[module], name, cls)
        classes[name] = cls
    count = 0

    def write(episodes=(12, 2, 7), mutate=None, payload=None):
        nonlocal count
        count += 1
        archive = tmp_path / f"reference-{count}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            # Neither other tasks nor other variations belong in the population.
            z.writestr("peract2_test/other_task/variation0/episodes/episode0/low_dim_obs.pkl",
                       b"not a pickle")
            z.writestr("peract2_test/bimanual_lift_ball/variation1/episodes/"
                       "episode0/low_dim_obs.pkl", b"not a pickle")
            for episode in episodes:
                demo = classes["Demo"]()
                obs = classes["BimanualObservation"]()
                task_state = np.zeros(63, dtype=np.float64)
                task_state[:7] = [episode / 100, -.1, .9, 0, 0, 0, 1]
                obs.task_low_dim_state = (task_state,)
                for side, offset in (("left", 0), ("right", .5)):
                    arm = classes["UnimanualObservationData"]()
                    arm.joint_positions = np.arange(7, dtype=float) + episode + offset
                    arm.gripper_open = 1. if side == "left" else .5
                    setattr(obs, side, arm)
                if mutate is not None:
                    mutate(obs)
                demo._observations = [obs]
                member = ("peract2_test/bimanual_lift_ball/variation0/episodes/"
                          f"episode{episode}/low_dim_obs.pkl")
                z.writestr(member, payload if payload is not None else pickle.dumps(demo))
        return archive

    return write


def test_sampling_is_seeded_uniform_with_replacement_and_couples_both_arms(make_archive):
    archive = make_archive()
    a = SourceEpisodeSampler(archive, seed=23, expected_sha256=None)
    b = SourceEpisodeSampler(archive, seed=23, expected_sha256=None)
    assert a.episodes == (2, 7, 12)
    expected = np.random.default_rng(23).choice(a.episodes, size=30)
    states = [a.sample() for _ in range(30)]
    assert states == [b.sample() for _ in range(30)]
    assert [s["episode"] for s in states] == expected.tolist()
    assert set(expected) == set(a.episodes)
    for state in states:
        episode = state["episode"]
        assert state["ball_pose_xyzw"][0] == episode / 100
        assert state["joint_positions"]["left"][0] == episode
        assert state["joint_positions"]["right"][0] == episode + .5
        assert state["selection"] == "uniform_stored_demo"
        assert state["seed"] == 23
        assert json.loads(json.dumps(state)) == state


def test_explicit_episode_does_not_consume_draw_and_results_cannot_mutate_cache(make_archive):
    archive = make_archive()
    sampler = SourceEpisodeSampler(archive, seed=19, expected_sha256=None)
    untouched = SourceEpisodeSampler(archive, seed=19, expected_sha256=None)
    state = sampler.sample(episode=7)
    pristine = deepcopy(state)
    assert state["selection"] == "specified_demo"
    state["ball_pose_xyzw"][0] = 99.
    state["joint_positions"]["right"][0] = 99.
    state["gripper_open"]["left"] = 0.
    assert sampler.sample(7) == pristine
    assert sampler.sample() == untouched.sample()
    with pytest.raises(ValueError, match="not present"):
        sampler.sample(episode=4)


def test_sampler_preserves_global_numpy_rng_and_reports_replayable_fresh_seed(make_archive):
    archive = make_archive()
    original = np.random.get_state()
    try:
        np.random.seed(813)
        expected = np.random.random(8)
        np.random.seed(813)
        sampler = SourceEpisodeSampler(archive, expected_sha256=None)
        assert 0 <= sampler.seed < 2**32
        states = [sampler.sample() for _ in range(6)]
        np.testing.assert_array_equal(np.random.random(8), expected)
        replay = SourceEpisodeSampler(archive, seed=sampler.seed, expected_sha256=None)
        assert states == [replay.sample() for _ in range(6)]
    finally:
        np.random.set_state(original)


def test_hash_pin_rejects_untrusted_archive_before_reading_records(make_archive):
    archive = make_archive(payload=b"not a pickle")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SourceEpisodeSampler(archive)


@pytest.mark.parametrize("fault", [
    "joint_shape", "joint_nan", "ball_nan", "zero_quaternion",
    "task_shape", "gripper_range", "gripper_shape",
])
def test_invalid_source_numeric_state_is_rejected(make_archive, fault):
    def corrupt(obs):
        if fault == "joint_shape":
            obs.left.joint_positions = np.zeros(6)
        elif fault == "joint_nan":
            obs.right.joint_positions[2] = np.nan
        elif fault == "ball_nan":
            obs.task_low_dim_state[0][0] = np.inf
        elif fault == "zero_quaternion":
            obs.task_low_dim_state[0][3:7] = 0
        elif fault == "task_shape":
            obs.task_low_dim_state = np.zeros(7)
        elif fault == "gripper_range":
            obs.left.gripper_open = 1.1
        elif fault == "gripper_shape":
            obs.right.gripper_open = [1.]
    sampler = SourceEpisodeSampler(make_archive(mutate=corrupt),
                                   seed=0, expected_sha256=None)
    with pytest.raises(ValueError, match="Invalid initial state"):
        sampler.sample()


def test_unsafe_pickle_global_is_rejected(make_archive):
    archive = make_archive(payload=pickle.dumps(Path("/tmp/untrusted")))
    sampler = SourceEpisodeSampler(archive, seed=0, expected_sha256=None)
    with pytest.raises(ValueError, match="Unexpected reference pickle global"):
        sampler.sample()


def test_empty_source_population_is_rejected(make_archive):
    with pytest.raises(ValueError, match="no bimanual_lift_ball"):
        SourceEpisodeSampler(make_archive(episodes=()), expected_sha256=None)


def test_released_archive_initial_states_are_finite_and_include_both_arms():
    archive = Path("runtime/peract2_test.zip")
    if not archive.exists():
        pytest.skip("Download the released PerAct2 archive first")
    sampler = SourceEpisodeSampler(archive, seed=42)
    assert sampler.archive_sha256 == SOURCE_ARCHIVE_SHA256
    assert sampler.episodes == tuple(range(100))
    balls, joints = [], {"left": [], "right": []}
    for episode in sampler.episodes:
        state = sampler.sample(episode)
        ball = np.asarray(state["ball_pose_xyzw"])
        assert np.isfinite(ball).all()
        assert np.linalg.norm(ball[3:]) == pytest.approx(1., abs=1e-6)
        assert ball[2] == pytest.approx(.9, abs=1e-4)
        balls.append(ball)
        for side in ("left", "right"):
            assert len(state["joint_positions"][side]) == 7
            assert np.isfinite(state["joint_positions"][side]).all()
            assert 0 <= state["gripper_open"][side] <= 1
            joints[side].append(state["joint_positions"][side])
    assert np.ptp(balls, axis=0)[0] > .2
    assert np.ptp(balls, axis=0)[1] > .2
    # Reset postures are essentially fixed in this task; preserve the recorded
    # numerical variation rather than adding arbitrary joint randomization.
    for side in ("left", "right"):
        assert np.max(np.ptp(joints[side], axis=0)) < .002
