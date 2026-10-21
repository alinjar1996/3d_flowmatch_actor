"""Render, infer, and execute the pretrained PerAct2 policy in MuJoCo."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .policy import CHECKPOINT_SHA256, FlowMatchPolicy, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("runtime/panda_ball_reference/scene.xml"))
    parser.add_argument("--config", type=Path,
                        default=Path("runtime/panda_ball_reference/reference_config.json"))
    parser.add_argument("--reference-archive", type=Path,
                        default=Path("runtime/peract2_test.zip"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/3dfa_peract2.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["observe", "infer", "rollout"], default="infer")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    reset_options = parser.add_mutually_exclusive_group()
    reset_options.add_argument("--randomize", action="store_true",
                               help="Sample a complete initial state from released ball demos")
    reset_options.add_argument("--episode", type=int,
                               help="Use a specific released ball-demo initial state")
    parser.add_argument("--reset-seed", type=int,
                        help="Reproduce demo selection with --randomize; default is a fresh seed")
    parser.add_argument("--instruction")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable MP4 recording (rollouts record by default)")
    parser.add_argument("--video-view",
                        choices=("all", "overview", "opposite", "side", "top", "orbit"),
                        default="all",
                        help="Saved camera view: all gives a synchronized 2x2 grid; "
                             "fixed views also set the initial live-view angle")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--observation-renderer", choices=("mujoco", "source_native", "source_mujoco"),
                        default="mujoco", help="Camera renderer; physics always runs in MuJoCo")
    parser.add_argument("--source-visual-assets", type=Path, default=Path("runtime/source_visuals"))
    parser.add_argument("--source-scene-root", type=Path)
    parser.add_argument("--coppeliasim-root", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.reset_seed is not None and not args.randomize:
        parser.error("--reset-seed requires --randomize")
    if args.reset_seed is not None and args.reset_seed < 0:
        parser.error("--reset-seed must be nonnegative")
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer else "egl")
    os.environ.setdefault("HF_HOME", str(Path("runtime/huggingface").resolve()))
    from .environment import PandaBallEnvironment, read_config
    from PIL import Image

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = read_config(args.config)
    if (args.randomize or args.episode is not None) and "scene_settings" not in config:
        parser.error("Source-demo resets require the source-calibrated scene and config")
    instruction = args.instruction or config["instruction"]
    output = args.output or Path("runtime/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "mode": args.mode, "device": args.device, "seed": args.seed,
        "instruction": instruction, "config": config, "steps": [],
        "source_success_criterion_verified": False,
        "metric": (config.get("success_condition") or config.get("success_proxy")),
        "observation_renderer": args.observation_renderer,
        "physics_backend": "mujoco",
        "implementation_sha256": {
            name: sha256(Path(__file__).parent / name) for name in (
                "environment.py", "policy.py", "run.py", "source_kinematics.py",
                "native_renderer.py", "native_worker.py", "source_visual_renderer.py")
            if (Path(__file__).parent / name).is_file()},
    }
    manifest = args.scene.parent / "source.json"
    if manifest.exists():
        report["scene_source"] = json.loads(manifest.read_text())
    env, viewer, video, observation_renderer = None, None, None, None
    resources = ExitStack()
    try:
        if "scene_settings" in config:
            from .verify_settings import verify_settings
            episode = int(Path(config["reference"]["member"]).parent.name.removeprefix("episode"))
            verification = verify_settings(
                args.scene, config, archive=args.reference_archive, episode=episode)
            report["settings_verification"] = verification
            if not verification["passed"]:
                failed = [c["name"] for c in verification["checks"] if not c["passed"]]
                raise ValueError(f"Source scene settings mismatch: {failed}")
            report["source_success_criterion_verified"] = True
            print("Source scene settings verified against native scene and released demo", flush=True)
        initial_state, reset_sampler = None, None
        if args.randomize or args.episode is not None:
            from .initial_states import SourceEpisodeSampler
            sampler = SourceEpisodeSampler(args.reference_archive, seed=args.reset_seed)
            report["initial_state_population_size"] = len(sampler.episodes)
            if args.randomize:
                reset_sampler = sampler
            else:
                initial_state = sampler.sample(episode=args.episode)
        env = PandaBallEnvironment(
            args.scene, config, initial_state=initial_state, reset_sampler=reset_sampler)
        resources.callback(env.close)
        if env.initial_state is not None:
            from .verify_settings import verify_initial_state
            report["initial_state"] = env.initial_state
            reset_verification = verify_initial_state(env, env.initial_state)
            report["initial_state_verification"] = reset_verification
            if not reset_verification["passed"]:
                failed = [c["name"] for c in reset_verification["checks"] if not c["passed"]]
                raise ValueError(f"Selected source reset mismatch: {failed}")
            state = env.initial_state
            print(f"Source reset: episode {state['episode']} | selection={state['selection']} "
                  f"| reset seed={state['seed']} | ball XYZ={state['ball_pose_xyzw'][:3]}",
                  flush=True)
        if args.observation_renderer == "source_native":
            from .native_renderer import NativeObservationRenderer
            observation_renderer = NativeObservationRenderer(
                env, output / "source_renderer", source_root=args.source_scene_root,
                coppelia_root=args.coppeliasim_root, archive=args.reference_archive)
        elif args.observation_renderer == "source_mujoco":
            from .source_visual_renderer import SourceVisualObservationRenderer
            observation_renderer = SourceVisualObservationRenderer(
                env, assets_dir=args.source_visual_assets, output_dir=output / "source_renderer")
        else:
            observation_renderer = env.adapter
        if observation_renderer is not env.adapter:
            resources.callback(observation_renderer.close)
        from .video import make_ball_goal_marker
        goal_marker = make_ball_goal_marker(env)
        report["ball_goal_marker"] = goal_marker
        if args.mode == "rollout" and not args.no_video:
            from .video import RolloutVideo
            video = RolloutVideo(env, output / "rollout.mp4", view=args.video_view)
            resources.callback(video.close)
            label = f"Initial observation | pretrained 3DFA | {args.observation_renderer}"
            if env.initial_state is not None:
                label += f" | source episode {env.initial_state['episode']}"
            video.hold(label, seconds=1)
            report["video"] = str(output / "rollout.mp4")
            report["video_view"] = args.video_view
            report["video_settings"] = video.settings
        if args.viewer:
            import mujoco.viewer
            from .video import add_ball_goal_marker, make_camera
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            resources.callback(viewer.close)
            camera = make_camera(
                env, args.video_view if args.video_view not in ("all", "orbit") else "overview")
            with viewer.lock():
                viewer.cam.type = camera.type
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
                add_ball_goal_marker(viewer.user_scn, goal_marker)
            viewer.sync()
            print("Live camera: left-drag to rotate, right-drag to pan, scroll to zoom", flush=True)
        print(f"Calibration: {config['calibration_status']}", flush=True)
        policy = None
        if args.mode != "observe":
            digest = sha256(args.checkpoint)
            report["checkpoint_sha256"] = digest
            report["official_checkpoint"] = digest == CHECKPOINT_SHA256
            policy = FlowMatchPolicy(args.checkpoint, args.device)
            policy.reset(instruction)
            bounds = policy.actor.workspace_normalizer.detach().cpu().numpy()
            report["workspace_normalizer"] = bounds.tolist()
            report["legacy_unused_checkpoint_keys"] = policy.actor.legacy_unused_checkpoint_keys
            print(f"Loaded {len(policy.actor.state_dict())} tensors strictly; "
                  f"migrated {len(policy.actor.legacy_unused_checkpoint_keys)} unused legacy tensors",
                  flush=True)
        maximum_ball_height = float(env.data.joint("ball").qpos[2])
        longest_goal_duration = goal_duration = 0.0
        observation_reports = []
        report["observations"] = observation_reports
        for step in range(args.steps if args.mode == "rollout" else 1):
            if viewer and not viewer.is_running():
                report["stop_reason"] = "viewer_closed"
                break
            observation_start = time.perf_counter()
            rgb, points, state = observation_renderer.observe()
            observation_report = {
                "step": step, "renderer": args.observation_renderer,
                "seconds": time.perf_counter() - observation_start,
                "simulation_time": float(env.data.time),
                "alignment": getattr(observation_renderer, "last_metadata", {}),
            }
            observation_reports.append(observation_report)
            np.savez_compressed(output / f"observation_{step:03d}.npz",
                                rgb=rgb, points=points, state=state,
                                qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                                ctrl=env.data.ctrl.copy())
            Image.fromarray(np.concatenate(list(rgb), axis=1)).save(output / f"cameras_{step:03d}.png")
            if args.mode == "observe":
                report["stop_reason"] = "observation_saved"
                break
            start = time.perf_counter()
            action = policy.predict(rgb, points, state)
            entry = {"step": step, "inference_seconds": time.perf_counter() - start,
                     "action_policy_frame": action.tolist(),
                     "state_inside_checkpoint_bounds": np.all(
                         (state[:, :3] >= bounds[0]) & (state[:, :3] <= bounds[1]), axis=1).tolist()}
            report["steps"].append(entry)
            print(json.dumps(entry), flush=True)
            if args.mode == "infer":
                report["stop_reason"] = "inference_complete"
                break
            if video:
                video.set_action(action, step)
                video.label += f" | {args.observation_renderer}"
                video.hold(video.label, seconds=0.5)
            def callback():
                nonlocal maximum_ball_height, goal_duration, longest_goal_duration
                env.track_success()
                height = float(env.data.joint("ball").qpos[2])
                maximum_ball_height = max(maximum_ball_height, height)
                goal = config.get("success_condition", {}).get("minimum_height_m")
                if goal is not None:
                    goal_duration = goal_duration + env.model.opt.timestep if height >= goal else 0.0
                    longest_goal_duration = max(longest_goal_duration, goal_duration)
                if video:
                    video.capture()
                if viewer:
                    viewer.sync()
            entry["controller"] = env.controller.execute(action, callback)
            entry["state_after_action"] = env.adapter.state().tolist()
            entry["ball_height_m"] = float(env.data.body("ball").xpos[2])
            entry["task_success"] = bool(env.success)
            entry["maximum_ball_height_m"] = maximum_ball_height
            if env.success:
                report["stop_reason"] = "task_success"
                break
            if not entry["controller"]["reached"]:
                report["stop_reason"] = "controller_timeout"
                break
        report.setdefault("stop_reason", "step_limit")
        report["task_success"] = bool(env.success)
        report["maximum_ball_height_m"] = maximum_ball_height
        report["longest_goal_duration_s"] = longest_goal_duration
        report["goal_duration_note"] = "Measured during executed actions; no extra holding phase."
        if video:
            video.hold(f"Result: {report['stop_reason']} | task success: {env.success}", seconds=2)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # ExitStack releases every resource even if report writing or another
        # cleanup callback fails, including the isolated native renderer.
        with resources:
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
