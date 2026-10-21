"""Compare numerical source FK/IK with independent native simulator captures."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_transfer.source_kinematics import SourceArmKinematics, source_camera_calibration


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


@pytest.mark.parametrize("task,asset_directory", [
    ("bimanual_lift_ball", "source_visuals"),
    ("bimanual_lift_tray", "source_visuals_tray"),
])
def test_selected_task_camera_mounts_match_independent_native_export(task, asset_directory):
    from mujoco_transfer.source_visual_renderer import camera_calibration_from_manifest

    runtime = Path(__file__).resolve().parents[1] / "runtime"
    archive = runtime / "peract2_test.zip"
    manifest_path = runtime / asset_directory / "manifest.json"
    if not archive.is_file() or not manifest_path.is_file():
        pytest.skip("Requires released task archive and independent native visual export")
    dataset_calibration = source_camera_calibration(archive, task=task)
    assert dataset_calibration["member"] == (
        f"peract2_test/{task}/variation0/episodes/episode3/low_dim_obs.pkl")
    native_calibration = camera_calibration_from_manifest(json.loads(manifest_path.read_text()))
    for name, native in native_calibration.items():
        actual = dataset_calibration["cameras"][name]
        np.testing.assert_allclose(actual["matrix"], native["matrix"], atol=1e-6, rtol=0)
        np.testing.assert_allclose([actual[k] for k in ("near", "far", "fovy")],
                                   [native[k] for k in ("near", "far", "fovy")], atol=1e-5, rtol=0)
