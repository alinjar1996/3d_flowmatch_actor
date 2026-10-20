"""Reproduce released 3DFA ball-lifting reset states without inventing jitter.

The source evaluator iterates stored demonstrations and calls ``reset_to_demo``.
Each sample here keeps the ball and both arms from the same demonstration.
"""

from copy import deepcopy
import hashlib
import io
import operator
from pathlib import Path
import re
import secrets
import zipfile

import numpy as np


SOURCE_ARCHIVE_SHA256 = (
    "d732886a9a9790a51d4ca71bcaa1727662c6dc0c490d402a1b0fce440c29d5a5"
)
_EPISODE_MEMBER = re.compile(
    r"peract2_test/bimanual_lift_ball/variation0/episodes/"
    r"episode([0-9]+)/low_dim_obs\.pkl"
)


def _numeric(value, name, shape):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return array


class SourceEpisodeSampler:
    """Uniform, reproducible sampling of coupled source initial states.

    ``expected_sha256=None`` is reserved for synthetic test fixtures. Production
    callers should use the official archive hash, which is checked once when
    constructing the sampler. An instance keeps its own random generator and
    caches only the validated first state of each requested demonstration.
    """

    def __init__(self, archive, seed=None, expected_sha256=SOURCE_ARCHIVE_SHA256):
        self.archive = Path(archive)
        digest = hashlib.sha256()
        with self.archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.archive_sha256 = digest.hexdigest()
        if expected_sha256 is not None and self.archive_sha256 != expected_sha256:
            raise ValueError(
                "Source archive SHA-256 mismatch: expected "
                f"{expected_sha256}, got {self.archive_sha256}"
            )

        self._members = {}
        with zipfile.ZipFile(self.archive) as source:
            for member in source.namelist():
                match = _EPISODE_MEMBER.fullmatch(member)
                if match is None:
                    continue
                episode = int(match.group(1))
                if episode in self._members:
                    raise ValueError(f"Duplicate source episode {episode}")
                self._members[episode] = member
        if not self._members:
            raise ValueError("Archive contains no bimanual_lift_ball variation0 episodes")
        self.episodes = tuple(sorted(self._members))

        if seed is None:
            seed = secrets.randbits(32)
        try:
            self.seed = operator.index(seed)
        except TypeError as exc:
            raise ValueError("Seed must be a nonnegative integer") from exc
        if isinstance(seed, (bool, np.bool_)) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self._rng = np.random.default_rng(self.seed)
        self._cache = {}

    def _initial_state(self, episode):
        if episode in self._cache:
            return self._cache[episode]
        # Keep this import lazy: reference generation imports the environment,
        # which also uses this sampler to initialize a rollout.
        from .reference import NumericReferenceUnpickler

        member = self._members[episode]
        with zipfile.ZipFile(self.archive) as source:
            demo = NumericReferenceUnpickler(io.BytesIO(source.read(member))).load()
        try:
            first = demo._observations[0]
            # The audited task has nine objects, with the ball first. Source
            # observations wrap the 63-vector in a one-element tuple.
            task_state = np.asarray(first.task_low_dim_state, dtype=np.float64).reshape(-1)
            task_state = _numeric(task_state, "task_low_dim_state", (63,))
            ball = task_state[:7]
            if np.linalg.norm(ball[3:]) == 0:
                raise ValueError("Ball quaternion must have nonzero norm")
            joints, grippers = {}, {}
            for side in ("left", "right"):
                arm = getattr(first, side)
                joints[side] = _numeric(
                    arm.joint_positions, f"{side}.joint_positions", (7,)
                ).tolist()
                opened = float(_numeric(arm.gripper_open, f"{side}.gripper_open", ()))
                if not 0 <= opened <= 1:
                    raise ValueError(f"{side}.gripper_open must be in [0, 1]")
                grippers[side] = opened
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid initial state in {member}: {exc}") from exc
        state = {
            "episode": episode,
            "member": member,
            "archive_sha256": self.archive_sha256,
            "ball_pose_xyzw": ball.tolist(),
            "joint_positions": joints,
            "gripper_open": grippers,
        }
        self._cache[episode] = state
        return state

    def sample(self, episode=None):
        """Return a JSON-safe independent state, selected with replacement."""
        selection = "uniform_stored_demo"
        if episode is None:
            episode = int(self._rng.choice(self.episodes))
        else:
            selection = "specified_demo"
            try:
                episode = operator.index(episode)
            except TypeError as exc:
                raise ValueError("Episode must be an integer") from exc
            if episode not in self._members:
                raise ValueError(f"Source episode {episode} is not present in {self.archive}")
        state = deepcopy(self._initial_state(episode))
        state.update(selection=selection, seed=self.seed)
        return state
