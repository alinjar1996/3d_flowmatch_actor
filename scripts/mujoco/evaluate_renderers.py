"""Evaluate camera renderers on distinct source resets with the same policy seed.

Each child runs actual closed-loop checkpoint inference and writes its own
four-view physics video, camera observations, alignment metadata and report.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderers", nargs="+", choices=("mujoco", "source_native", "source_mujoco"),
                        default=["source_native", "source_mujoco"])
    parser.add_argument("--episodes", nargs="+", type=int,
                        default=[3, 8, 87, 94, 85, 72, 12, 26, 48, 63])
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-visual-assets", type=Path, default=Path("runtime/source_visuals"))
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if len(args.episodes) != len(set(args.episodes)) or any(e < 0 or e > 99 for e in args.episodes):
        parser.error("--episodes must be distinct released episode IDs from 0 to 99")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_inputs = [Path("runtime/panda_ball_reference/scene.xml"),
                       Path("runtime/panda_ball_reference/panda_scene.xml"),
                       Path("runtime/panda_ball_reference/objects.xml"),
                       Path("runtime/panda_ball_reference/reference_config.json"),
                       Path("mujoco_transfer/environment.py"), Path("mujoco_transfer/policy.py")]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_inputs}
    results = []
    environment = dict(os.environ)
    environment.update(PYTHONPATH="", MUJOCO_GL="egl", OMP_NUM_THREADS="4",
                       HF_HOME=str(Path("runtime/huggingface").resolve()))

    def save_summary():
        groups = {}
        for renderer in args.renderers:
            rows = [r for r in results if r["renderer"] == renderer]
            completed = [r for r in rows if r["exit_code"] == 0 and "error" not in r]
            groups[renderer] = {"planned": len(args.episodes), "completed": len(completed),
                                "successes": sum(r["success"] for r in completed),
                                "errors": len(rows) - len(completed)}
        summary = {"episodes": args.episodes, "policy_seed": args.policy_seed,
                   "physics_and_policy_sha256": provenance, "summary": groups, "results": results}
        temporary = args.output / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(args.output / "summary.json")

    save_summary()
    for renderer in args.renderers:
        for episode in args.episodes:
            folder = args.output / renderer / f"episode_{episode:03d}"
            folder.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "mujoco_transfer.run", "--mode", "rollout",
                       "--device", "cuda", "--episode", str(episode), "--seed", str(args.policy_seed),
                       "--steps", str(args.steps), "--observation-renderer", renderer,
                       "--source-visual-assets", str(args.source_visual_assets),
                       "--video-view", "all", "--output", str(folder)]
            if args.no_video:
                command.append("--no-video")
            log_path = folder.with_suffix(".log")
            print(f"Starting {renderer}: episode {episode}", flush=True)
            with log_path.open("w") as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            row = {"renderer": renderer, "episode": episode, "exit_code": child.returncode,
                   "report": str(folder / "report.json"), "log": str(log_path)}
            if (folder / "report.json").is_file():
                report = json.loads((folder / "report.json").read_text())
                row.update(success=report.get("task_success", False),
                           stop_reason=report.get("stop_reason"), steps=len(report.get("steps", [])),
                           maximum_ball_height_m=report.get("maximum_ball_height_m"),
                           longest_goal_duration_s=report.get("longest_goal_duration_s"),
                           video=report.get("video"))
                if "error" in report:
                    row["error"] = report["error"]
            else:
                row["error"] = "No rollout report was written"
            results.append(row)
            save_summary()
            print(json.dumps(row), flush=True)
    if any(row["exit_code"] != 0 or "error" in row for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
