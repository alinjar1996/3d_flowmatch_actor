"""Run an inclusive range of source-reset seeds, keeping videos and resumable results."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


FIELDS = (
    "reset_seed", "policy_seed", "state", "attempt", "source_episode",
    "task_success", "stop_reason", "steps", "mean_inference_seconds",
    "ball_height_m", "object_name", "object_height_m", "report_path", "video_path", "log_path", "error",
)
FINISHED_REASONS = {"task_success", "controller_timeout", "step_limit"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=100)
    parser.add_argument("--policy-seed", default="0",
                        help="Fixed inference RNG seed (default: 0), or 'match' the reset seed")
    parser.add_argument("--output", type=Path, default=Path("runtime/seed_sweeps/seeds_000_100"))
    parser.add_argument("--scene", type=Path, default=Path("runtime/panda_ball_reference/scene.xml"))
    parser.add_argument("--config", type=Path,
                        default=Path("runtime/panda_ball_reference/reference_config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/3dfa_peract2.pth"))
    parser.add_argument("--reference-archive", type=Path, default=Path("runtime/peract2_test.zip"))
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--video-view", choices=("all", "overview", "opposite", "side", "top", "orbit"),
                        default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--observation-renderer", choices=("mujoco", "source_native", "source_mujoco"),
                        default="mujoco")
    parser.add_argument("--source-visual-assets", type=Path, default=Path("runtime/source_visuals"))
    parser.add_argument("--source-scene-root", type=Path, default=Path("runtime/source-audit/RLBench"))
    parser.add_argument("--coppeliasim-root", type=Path,
                        default=Path("runtime/source-audit/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"))
    parser.add_argument("--resume", action="store_true", help="Skip existing valid results and errors")
    parser.add_argument("--retry-errors", action="store_true",
                        help="With --resume, retry execution errors in a new attempt directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running inference or creating files")
    return parser


def policy_seed_for(args, seed):
    return seed if args.policy_seed == "match" else int(args.policy_seed)


def command_for_seed(args, seed, output):
    return [
        sys.executable, "-u", "-m", "mujoco_transfer.run", "--mode", "rollout",
        "--scene", str(args.scene), "--config", str(args.config),
        "--checkpoint", str(args.checkpoint), "--reference-archive", str(args.reference_archive),
        "--device", args.device, "--threads", str(args.threads), "--steps", str(args.steps),
        "--video-view", args.video_view, "--randomize", "--reset-seed", str(seed),
        "--seed", str(policy_seed_for(args, seed)), "--output", str(output),
        "--observation-renderer", args.observation_renderer,
        "--source-visual-assets", str(args.source_visual_assets),
        "--source-scene-root", str(args.source_scene_root),
        "--coppeliasim-root", str(args.coppeliasim_root),
    ]


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(args):
    """Bind resumptions to inputs and implementation; no torch or MuJoCo imports."""
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {"output", "resume", "retry_errors", "dry_run"}}
    inputs = {getattr(args, key).resolve()
              for key in ("scene", "config", "checkpoint", "reference_archive")}
    # Include nested XML includes and mesh/texture assets, whose changes affect simulation.
    model_suffixes = {".xml", ".stl", ".obj", ".ply", ".msh", ".png", ".jpg", ".jpeg"}
    inputs.update(path.resolve() for path in args.scene.parent.rglob("*")
                  if path.is_file() and path.suffix.lower() in model_suffixes)
    if args.observation_renderer == "source_native":
        source_task = json.loads(args.config.read_text()).get("source_task", "bimanual_lift_ball")
        inputs.update((args.source_scene_root / path).resolve() for path in
                      ("rlbench/task_design_bimanual.ttt", f"rlbench/task_ttms/{source_task}.ttm"))
        inputs.add((args.coppeliasim_root / "libcoppeliaSim.so").resolve())
    elif args.observation_renderer == "source_mujoco":
        manifest_path = args.source_visual_assets / "manifest.json"
        inputs.add(manifest_path.resolve())
        manifest = json.loads(manifest_path.read_text())
        records = manifest["objects"]
        if isinstance(records, dict):
            records = records.values()
        for record in records:
            for component in record.get("meshes", []):
                inputs.add((args.source_visual_assets / component["obj_path"]).resolve())
                texture = component.get("material", {}).get("texture_path")
                if texture:
                    inputs.add((args.source_visual_assets / texture).resolve())
    package_root = Path(__file__).resolve().parent.parent
    for directory in ("mujoco_transfer", "modeling", "utils"):
        inputs.update(path.resolve() for path in (package_root / directory).rglob("*.py"))
    inputs.update(path.resolve() for path in (package_root / "mujoco_transfer/configs").glob("*.json"))
    hashes = {str(path): file_hash(path) for path in sorted(inputs)}
    return {
        "schema_version": 1, "settings": settings, "input_sha256": hashes,
        "python": str(Path(sys.executable).resolve()),
        "environment": {key: os.environ.get(key) for key in
                        ("MUJOCO_GL", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")},
    }


def preflight(args):
    for key in ("scene", "config", "checkpoint", "reference_archive"):
        path = getattr(args, key)
        if not path.is_file():
            raise ValueError(f"Missing {key.replace('_', ' ')}: {path}")
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is required to save the rollout videos")
    if args.observation_renderer == "source_mujoco" and not (args.source_visual_assets / "manifest.json").is_file():
        raise ValueError("Missing source visual export; run scripts/mujoco/export_source_visuals.py first")
    if args.observation_renderer == "source_native":
        source_task = json.loads(args.config.read_text()).get("source_task", "bimanual_lift_ball")
        for path in (args.source_scene_root / "rlbench/task_design_bimanual.ttt",
                     args.source_scene_root / f"rlbench/task_ttms/{source_task}.ttm",
                     args.coppeliasim_root / "libcoppeliaSim.so"):
            if not path.is_file():
                raise ValueError(f"Missing native renderer dependency: {path}")


def empty_row(seed, policy_seed):
    row = {field: None for field in FIELDS}
    row.update(reset_seed=seed, policy_seed=policy_seed, state="pending", steps=0)
    return row


def read_result(seed, policy_seed, status_path):
    """Validate a completed subprocess result instead of treating task failure as an error."""
    row = empty_row(seed, policy_seed)
    if status_path is None:
        return row
    try:
        status = json.loads(status_path.read_text())
        row.update(attempt=status_path.name.removesuffix(".status.json"),
                   report_path=status.get("report_path"), log_path=status.get("log_path"))
        if status.get("state") in ("running", "interrupted"):
            row["error"] = status.get("error", "Previous attempt did not finish")
            return row
        row["state"] = "execution_error"
        if status.get("exit_code") != 0:
            row["error"] = status.get("error") or f"Process exited with code {status.get('exit_code')}"
            return row
        report = json.loads(Path(status["report_path"]).read_text())
        if report.get("error"):
            raise ValueError(report["error"])
        if report.get("mode") != "rollout" or report.get("stop_reason") not in FINISHED_REASONS:
            raise ValueError("Report does not describe a completed rollout")
        for key in ("settings_verification", "initial_state_verification"):
            if report.get(key, {}).get("passed") is not True:
                raise ValueError(f"Report lacks successful {key}")
        success, steps, initial = (report.get("task_success"), report.get("steps"),
                                   report.get("initial_state", {}))
        if type(success) is not bool or not isinstance(steps, list) or not steps:
            raise ValueError("Report has missing or malformed success/steps fields")
        if success != (report["stop_reason"] == "task_success"):
            raise ValueError("Report success flag contradicts its stop reason")
        if report.get("seed") != policy_seed or initial.get("seed") != seed:
            raise ValueError("Report RNG seeds do not match this sweep entry")
        if type(initial.get("episode")) is not int:
            raise ValueError("Report has no source episode")
        times = [float(entry["inference_seconds"]) for entry in steps]
        object_name = report.get("object_name", "ball")
        height = float(steps[-1]["object_height_m"] if "object_height_m" in steps[-1]
                       else steps[-1]["ball_height_m"])
        if not all(math.isfinite(value) and value >= 0 for value in times) or not math.isfinite(height):
            raise ValueError("Report contains invalid numerical metrics")
        video = Path(report["video"])
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError("Rollout video is missing or empty")
        row.update(
            state="completed", source_episode=initial["episode"], task_success=success,
            stop_reason=report["stop_reason"], steps=len(steps),
            mean_inference_seconds=sum(times) / len(times),
            ball_height_m=height if object_name == "ball" else None,
            object_name=object_name, object_height_m=height,
            video_path=str(video), error=None,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        row.update(state="execution_error", error=f"{type(exc).__name__}: {exc}")
    return row


def summarize(rows, start_seed, end_seed):
    completed = [row for row in rows if row["state"] == "completed"]
    successes = sum(bool(row["task_success"]) for row in completed)
    errors = sum(row["state"] == "execution_error" for row in rows)
    episodes = sorted({row["source_episode"] for row in completed})
    return {
        "start_seed": start_seed, "end_seed": end_seed, "seeds_total": end_seed - start_seed + 1,
        "completed": len(completed), "successes": successes,
        "task_failures": len(completed) - successes, "execution_errors": errors,
        "pending": end_seed - start_seed + 1 - len(completed) - errors,
        "success_rate": successes / len(completed) if completed else None,
        "source_episode_coverage": {"unique_count": len(episodes), "episodes": episodes},
        "rows": rows,
    }


def latest_status(seed_dir):
    paths = sorted(seed_dir.glob("attempt_*.status.json"))
    return paths[-1] if paths else None


def next_attempt(seed_dir):
    numbers = []
    for path in seed_dir.glob("attempt_*"):
        suffix = path.name.removeprefix("attempt_").split(".")[0]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return seed_dir / f"attempt_{max(numbers, default=0) + 1:03d}"


def save_summary(args):
    rows = [read_result(seed, policy_seed_for(args, seed),
                        latest_status(args.output / f"seed_{seed:03d}"))
            for seed in range(args.start_seed, args.end_seed + 1)]
    summary = summarize(rows, args.start_seed, args.end_seed)
    atomic_json(args.output / "summary.json", summary)
    temporary = args.output / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output / "summary.csv")
    return summary


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_attempt(args, seed, output):
    status_path = output.with_name(output.name + ".status.json")
    log_path = output.with_name(output.name + ".log")
    command = command_for_seed(args, seed, output)
    status = {
        "state": "running", "reset_seed": seed, "policy_seed": policy_seed_for(args, seed),
        "command": command, "report_path": str(output / "report.json"),
        "log_path": str(log_path), "exit_code": None, "complete_valid": False,
        "started_unix_seconds": time.time(), "wall_seconds": None,
    }
    atomic_json(status_path, status)
    start, process = time.monotonic(), None
    try:
        with log_path.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status["exit_code"] = process.wait()
        status["state"] = "finished"
    except KeyboardInterrupt:
        if process is not None:
            stop_process(process)
            status["exit_code"] = process.returncode
        status.update(state="interrupted", error="Interrupted by user")
        raise
    except OSError as exc:
        status.update(state="execution_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        status["wall_seconds"] = time.monotonic() - start
        atomic_json(status_path, status)
        if status["state"] != "interrupted":
            row = read_result(seed, policy_seed_for(args, seed), status_path)
            status.update(state=row["state"], complete_valid=row["state"] == "completed")
            if row["error"]:
                status["error"] = row["error"]
            atomic_json(status_path, status)
    return read_result(seed, policy_seed_for(args, seed), status_path)


def run_batch(args):
    manifest_path = args.output / "manifest.json"
    print("Checking input hashes for resumable execution...", flush=True)
    manifest = fingerprint(args)
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(f"Output already has a manifest; use --resume: {args.output}")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume settings or input hashes differ; use a new --output directory")
    else:
        contents = [path for path in args.output.iterdir() if path.name != ".sweep.lock"]
        if contents:
            raise ValueError("Output contains artifacts without a manifest; use a new --output directory")
        atomic_json(manifest_path, manifest)
    try:
        for seed in range(args.start_seed, args.end_seed + 1):
            seed_dir = args.output / f"seed_{seed:03d}"
            previous = read_result(seed, policy_seed_for(args, seed), latest_status(seed_dir))
            if previous["state"] == "completed":
                print(f"Seed {seed}: skipping completed result", flush=True)
                continue
            if previous["state"] == "execution_error" and not args.retry_errors:
                print(f"Seed {seed}: keeping execution error (use --resume --retry-errors)", flush=True)
                continue
            seed_dir.mkdir(exist_ok=True)
            output = next_attempt(seed_dir)
            print(f"Seed {seed}/{args.end_seed}: {output.name}; log {output}.log", flush=True)
            row = run_attempt(args, seed, output)
            summary = save_summary(args)
            print(f"Seed {seed}: {row['state']} | {row['stop_reason'] or row['error']} | "
                  f"completed {summary['completed']}/{summary['seeds_total']}, "
                  f"successes {summary['successes']}, errors {summary['execution_errors']}", flush=True)
    except KeyboardInterrupt:
        save_summary(args)
        print("Interrupted; artifacts saved. Use --resume to continue.", flush=True)
        return 130
    summary = save_summary(args)
    print(f"Results: {args.output / 'summary.csv'}\nSummary: {args.output / 'summary.json'}", flush=True)
    return 1 if summary["execution_errors"] else 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_seed < 0 or args.end_seed < args.start_seed:
        parser.error("Seeds must satisfy 0 <= --start-seed <= --end-seed (inclusive)")
    if args.steps < 1 or args.threads < 1:
        parser.error("--steps and --threads must be positive")
    if args.policy_seed != "match":
        try:
            number = int(args.policy_seed)
            if not 0 <= number < 2**32:
                raise ValueError
            args.policy_seed = str(number)
        except ValueError:
            parser.error("--policy-seed must be 'match' or an integer in [0, 2**32)")
    if args.policy_seed == "match" and args.end_seed >= 2**32:
        parser.error("--policy-seed match requires reset seeds below 2**32")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.dry_run:
        for seed in range(args.start_seed, args.end_seed + 1):
            seed_dir = args.output / f"seed_{seed:03d}"
            print(shlex.join(command_for_seed(args, seed, next_attempt(seed_dir))))
        return 0
    # Absolute artifact paths make reports and summaries usable from another working directory.
    args.output = args.output.resolve()
    for name in ("scene", "config", "checkpoint", "reference_archive"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        preflight(args)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".sweep.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"Another sweep is using this output: {args.output}") from None
            return run_batch(args)
    except (OSError, ValueError) as exc:
        print(f"Sweep error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
