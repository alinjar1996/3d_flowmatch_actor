"""Compare numerical source FK/IK with independent native simulator captures."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.source_kinematics import SourceArmKinematics


@pytest.fixture(params=["left", "right"])
def arm(request):
    captures = json.loads((Path(__file__).parent / "fixtures/source_arm_kinematics.json").read_text())
    capture = captures["arms"][request.param]
    return SourceArmKinematics(capture["fixed_transforms"], capture["tip_transform"],
                               capture["limits"]), capture


def test_source_forward_matches_independent_native_poses(arm):
    chain, capture = arm
    for sample in capture["samples"]:
        np.testing.assert_allclose(chain.forward(sample["joints"]), sample["tcp_matrix"], atol=2e-6)


def test_source_ik_matches_native_pose_from_different_configuration(arm):
    chain, capture = arm
    seed = np.asarray(capture["samples"][0]["joints"])
    for sample in capture["samples"][1:]:
        target = np.asarray(sample["tcp_matrix"])
        joints, metadata = chain.solve(target, seed)
        actual = chain.forward(joints)
        assert metadata["aligned"]
        assert metadata["fallback"] is None
        assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 2e-5
        assert np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()) < 2e-5
        assert np.all(joints >= chain.limits[:, 0])
        assert np.all(joints <= chain.limits[:, 1])


def test_source_ik_reports_unreachable_pose_without_hiding_residual(arm):
    chain, capture = arm
    target = np.asarray(capture["samples"][0]["tcp_matrix"])
    target[:3, 3] += 10
    joints, metadata = chain.solve(target, capture["samples"][0]["joints"])
    assert not metadata["aligned"]
    assert metadata["fallback"] == "best_bounded_native_ik_solution"
    assert metadata["position_error_m"] > 10
    assert np.isfinite(joints).all()
    assert np.all(joints >= chain.limits[:, 0])
    assert np.all(joints <= chain.limits[:, 1])
