"""Batch orchestration checks use fake children, never CUDA inference."""

import json
from pathlib import Path
import shlex

import pytest

from mujoco_transfer import sweep


def option(command, name):
    return command[command.index(name) + 1]


def test_default_range_is_inclusive_and_only_reset_seed_changes(tmp_path):
    args = sweep.build_parser().parse_args([])
    seeds = list(range(args.start_seed, args.end_seed + 1))
    assert seeds == list(range(101))
    for seed in (0, 1, 100):
        output = tmp_path / f"seed_{seed:03d}"
        command = sweep.command_for_seed(args, seed, output)
        assert "--randomize" in command
        assert option(command, "--reset-seed") == str(seed)
        assert option(command, "--seed") == "0"
        assert option(command, "--mode") == "rollout"
        assert option(command, "--device") == "cuda"
        assert option(command, "--steps") == "25"
        assert option(command, "--video-view") == "all"
        assert Path(option(command, "--output")) == output


@pytest.mark.parametrize("seed", [0, 42, 100])
def test_policy_seed_can_match_reset_seed(seed, tmp_path):
    args = sweep.build_parser().parse_args(["--policy-seed", "match"])
    command = sweep.command_for_seed(args, seed, tmp_path / "attempt")
    assert option(command, "--reset-seed") == str(seed)
    assert option(command, "--seed") == str(seed)


def test_explicit_fixed_policy_seed_is_forwarded(tmp_path):
    args = sweep.build_parser().parse_args(["--policy-seed", "19"])
    command = sweep.command_for_seed(args, 100, tmp_path / "attempt")
    assert option(command, "--reset-seed") == "100"
    assert option(command, "--seed") == "19"


def test_summary_counts_task_failures_but_excludes_execution_errors():
    def row(seed, state, success=None, episode=None):
        return {
            "reset_seed": seed, "state": state, "task_success": success,
            "source_episode": episode,
        }
    rows = [
        row(0, "completed", True, 8),
        row(1, "completed", False, 8),
        row(2, "execution_error"),
        row(3, "pending"),
    ]
    result = sweep.summarize(rows, 0, 3)
    assert result["seeds_total"] == 4
    assert result["completed"] == 2
    assert result["successes"] == 1
    assert result["execution_errors"] == 1
    assert result["pending"] == 1
    assert result["success_rate"] == .5
    assert result["source_episode_coverage"] == {"unique_count": 1, "episodes": [8]}


def test_summary_without_completed_runs_has_no_success_rate():
    result = sweep.summarize([], 0, 100)
    assert result["successes"] == 0
    assert result["completed"] == 0
    assert result["success_rate"] is None


def report_for_seed(command):
    """Only emulate rollout artifacts; no simulation or inference is imported."""
    seed = int(option(command, "--reset-seed"))
    output = Path(option(command, "--output"))
    return {
        "mode": "rollout",
        "seed": int(option(command, "--seed")),
        "task_success": seed == 0,
        "stop_reason": "task_success" if seed == 0 else "controller_timeout",
        "initial_state": {
            "seed": seed, "episode": seed % 2,
            "selection": "uniform_stored_demo",
        },
        "settings_verification": {"passed": True},
        "initial_state_verification": {"passed": True},
        "steps": [{
            "step": 0, "inference_seconds": .02, "ball_height_m": .9,
            "task_success": seed == 0,
            "controller": {"reached": seed == 0},
        }],
        "video": str(output / "rollout.mp4"),
    }


@pytest.fixture
def fake_children(monkeypatch):
    """Use process-shaped fakes so orchestration and disk resume remain real."""
    state = {
        "commands": [], "active": 0, "max_active": 0, "errors": set(),
        "interrupt_once": False, "terminated": 0, "children": [],
    }
    monkeypatch.setattr(
        sweep, "fingerprint", lambda args: {"fixture": "fixed", "steps": args.steps})
    monkeypatch.setattr(sweep, "preflight", lambda args: None)

    class Child:
        def __init__(self, command, **kwargs):
            assert state["active"] == 0, "The sweep must not overlap GPU children"
            self.command = command
            self.returncode = None
            self.pid = len(state["children"]) + 123456
            state["children"].append(self)
            self.seed = int(option(command, "--reset-seed"))
            self.output = Path(option(command, "--output"))
            self.output.mkdir(parents=True, exist_ok=False)
            self.report = report_for_seed(command)
            if self.seed in state["errors"]:
                self.report["error"] = "RuntimeError: fake child failure"
            (self.output / "report.json").write_text(json.dumps(self.report))
            (self.output / "rollout.mp4").write_bytes(b"fake-mp4-artifact")
            state["commands"].append(command)
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

        def wait(self, timeout=None):
            if state["interrupt_once"]:
                state["interrupt_once"] = False
                raise KeyboardInterrupt
            if self.returncode is None:
                self.returncode = 2 if self.seed in state["errors"] else 0
                state["active"] -= 1
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            state["terminated"] += 1
            if self.returncode is None:
                self.returncode = -15
                state["active"] -= 1

        def kill(self):
            self.terminate()

        def send_signal(self, signal):
            self.terminate()

    monkeypatch.setattr(sweep.subprocess, "Popen", Child)
    monkeypatch.setattr(
        sweep.os, "killpg",
        lambda pid, sig: next(child for child in state["children"] if child.pid == pid).terminate(),
    )
    return state


def batch_args(output, end_seed=2):
    return ["--output", str(output), "--end-seed", str(end_seed)]


def test_dry_run_lists_101_commands_without_creating_or_preflighting(
        tmp_path, monkeypatch, capsys):
    output = tmp_path / "dry-run"

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not initialize or execute the sweep")

    monkeypatch.setattr(sweep, "preflight", forbidden)
    monkeypatch.setattr(sweep, "fingerprint", forbidden)
    monkeypatch.setattr(sweep.subprocess, "Popen", forbidden)
    assert sweep.main(["--output", str(output), "--dry-run"]) == 0
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 101
    assert [int(option(cmd, "--reset-seed")) for cmd in commands] == list(range(101))
    assert all(option(cmd, "--seed") == "0" for cmd in commands)
    assert not output.exists()


def test_sequential_sweep_distinguishes_task_failure_and_execution_error(
        tmp_path, fake_children):
    output = tmp_path / "batch"
    fake_children["errors"].add(2)
    assert sweep.main(batch_args(output)) == 1
    assert fake_children["max_active"] == 1
    assert fake_children["active"] == 0
    assert [int(option(cmd, "--reset-seed")) for cmd in fake_children["commands"]] == [0, 1, 2]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] == 2
    assert summary["successes"] == 1
    assert summary["task_failures"] == 1
    assert summary["execution_errors"] == 1
    assert summary["success_rate"] == .5
    assert summary["rows"][1]["stop_reason"] == "controller_timeout"
    assert summary["rows"][1]["state"] == "completed"


def test_resume_skips_valid_task_failures_and_retry_preserves_failed_attempt(
        tmp_path, fake_children):
    output = tmp_path / "batch"
    fake_children["errors"].add(2)
    assert sweep.main(batch_args(output)) == 1
    old_report = output / "seed_002/attempt_001/report.json"
    preserved = old_report.read_bytes()
    assert sweep.main(batch_args(output) + ["--resume"]) == 1
    assert len(fake_children["commands"]) == 3
    fake_children["errors"].clear()
    assert sweep.main(batch_args(output) + ["--resume", "--retry-errors"]) == 0
    assert len(fake_children["commands"]) == 4
    retry = fake_children["commands"][-1]
    assert option(retry, "--reset-seed") == "2"
    assert Path(option(retry, "--output")).name == "attempt_002"
    assert old_report.read_bytes() == preserved
    assert (output / "seed_002/attempt_002/report.json").is_file()
    assert not (output / "seed_000/attempt_002").exists()
    assert not (output / "seed_001/attempt_002").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] == 3
    assert summary["execution_errors"] == 0
    assert summary["success_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("fault", [
    "missing_report", "malformed_report", "missing_video", "empty_video",
    "wrong_reset_seed", "wrong_policy_seed", "inconsistent_success",
    "nonfinite_metric", "missing_steps", "missing_settings_verification",
    "failed_settings_verification", "missing_initial_state_verification",
    "failed_initial_state_verification",
])
def test_incomplete_or_invalid_artifacts_are_not_resumable_as_complete(
        tmp_path, fake_children, fault):
    output = tmp_path / "batch"
    assert sweep.main(batch_args(output, end_seed=0)) == 0
    attempt = output / "seed_000/attempt_001"
    report_path = attempt / "report.json"
    video_path = attempt / "rollout.mp4"
    if fault == "missing_report":
        report_path.unlink()
    elif fault == "malformed_report":
        report_path.write_text("{not valid json")
    elif fault == "missing_video":
        video_path.unlink()
    elif fault == "empty_video":
        video_path.write_bytes(b"")
    else:
        report = json.loads(report_path.read_text())
        if fault == "wrong_reset_seed":
            report["initial_state"]["seed"] = 99
        elif fault == "wrong_policy_seed":
            report["seed"] = 99
        elif fault == "inconsistent_success":
            report["task_success"] = False
        elif fault == "nonfinite_metric":
            report["steps"][0]["inference_seconds"] = float("nan")
        elif fault == "missing_settings_verification":
            del report["settings_verification"]
        elif fault == "failed_settings_verification":
            report["settings_verification"]["passed"] = False
        elif fault == "missing_initial_state_verification":
            del report["initial_state_verification"]
        elif fault == "failed_initial_state_verification":
            report["initial_state_verification"]["passed"] = False
        else:
            report["steps"] = []
        report_path.write_text(json.dumps(report))
    row = sweep.read_result(0, 0, output / "seed_000/attempt_001.status.json")
    assert row["state"] == "execution_error"
    assert row["error"]
    assert row["task_success"] is None
    assert sweep.main(batch_args(output, end_seed=0) + ["--resume"]) == 1
    assert len(fake_children["commands"]) == 1


def test_resume_refuses_manifest_mismatch_before_launching_another_child(
        tmp_path, fake_children):
    output = tmp_path / "batch"
    assert sweep.main(batch_args(output, end_seed=0)) == 0
    original = (output / "manifest.json").read_bytes()
    assert sweep.main(batch_args(output, end_seed=0) + ["--resume", "--steps", "26"]) == 2
    assert len(fake_children["commands"]) == 1
    assert (output / "manifest.json").read_bytes() == original


def test_existing_batch_requires_resume_and_cannot_overwrite_results(
        tmp_path, fake_children):
    output = tmp_path / "batch"
    assert sweep.main(batch_args(output, end_seed=0)) == 0
    report = output / "seed_000/attempt_001/report.json"
    original = report.read_bytes()
    assert sweep.main(batch_args(output, end_seed=0)) == 2
    assert len(fake_children["commands"]) == 1
    assert report.read_bytes() == original


def test_interrupt_stops_child_and_resume_uses_a_new_attempt(tmp_path, fake_children):
    output = tmp_path / "batch"
    fake_children["interrupt_once"] = True
    assert sweep.main(batch_args(output, end_seed=1)) == 130
    assert fake_children["terminated"] == 1
    assert fake_children["active"] == 0
    assert len(fake_children["commands"]) == 1
    status_path = output / "seed_000/attempt_001.status.json"
    assert json.loads(status_path.read_text())["state"] == "interrupted"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] == 0
    assert summary["pending"] == 2
    assert sweep.main(batch_args(output, end_seed=1) + ["--resume"]) == 0
    assert len(fake_children["commands"]) == 3
    assert Path(option(fake_children["commands"][1], "--output")).name == "attempt_002"
    assert (output / "seed_000/attempt_001/report.json").exists()


@pytest.mark.parametrize("arguments", [
    ["--start-seed", "-1"],
    ["--start-seed", "10", "--end-seed", "9"],
    ["--policy-seed", "-1"],
    ["--policy-seed", "4294967296"],
    ["--policy-seed", "match", "--end-seed", "4294967296"],
    ["--policy-seed", "invalid"],
    ["--steps", "0"],
    ["--threads", "0"],
    ["--retry-errors"],
])
def test_invalid_cli_arguments_fail_before_execution(arguments, tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        sweep.main(arguments + ["--output", str(output), "--dry-run"])
    assert error.value.code == 2
    assert not output.exists()


def test_fingerprint_detects_nested_scene_asset_changes(tmp_path):
    scene = tmp_path / "scene/scene.xml"
    scene.parent.mkdir()
    scene.write_text("<mujoco/>")
    mesh = scene.parent / "meshes/panda/arm.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"first mesh")
    files = {"config": "config.json", "checkpoint": "checkpoint.pth",
             "reference-archive": "archive.zip"}
    arguments = ["--scene", str(scene)]
    for flag, filename in files.items():
        path = tmp_path / filename
        path.write_bytes(b"fixture")
        arguments.extend([f"--{flag}", str(path)])
    args = sweep.build_parser().parse_args(arguments)
    original = sweep.fingerprint(args)
    assert str(mesh.resolve()) in original["input_sha256"]
    mesh.write_bytes(b"modified mesh")
    changed = sweep.fingerprint(args)
    assert changed != original
    assert changed["input_sha256"][str(mesh.resolve())] != original["input_sha256"][str(mesh.resolve())]


def test_source_visual_texture_change_invalidates_sweep_fingerprint(tmp_path):
    arguments = ["--observation-renderer", "source_mujoco"]
    for flag in ("scene", "config", "checkpoint", "reference-archive"):
        path = tmp_path / "physical" / flag
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"unchanged physical input")
        arguments.extend([f"--{flag}", str(path)])
    assets = tmp_path / "visuals"
    assets.mkdir()
    (assets / "table.obj").write_text("source visual mesh")
    texture = assets / "wood.png"
    texture.write_bytes(b"original texture")
    (assets / "manifest.json").write_text(json.dumps({"objects": [{"meshes": [
        {"obj_path": "table.obj", "material": {"texture_path": "wood.png"}}
    ]}]}))
    arguments.extend(["--source-visual-assets", str(assets)])
    args = sweep.build_parser().parse_args(arguments)
    original = sweep.fingerprint(args)
    texture.write_bytes(b"different visual input")
    changed = sweep.fingerprint(args)
    assert changed != original
    assert changed["input_sha256"][str(texture)] != original["input_sha256"][str(texture)]
