"""Replay saved predictions with contact telemetry and isolated execution ablations.

This never calls the policy or modifies the prepared scene/sweep. Disabling robot
contacts is a diagnostic intervention, not a valid task evaluation.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mujoco_transfer.environment import PandaBallEnvironment


def replay(report, scene, variant, max_steps):
    config = deepcopy(report["config"])
    if variant == "long_timeout":
        config["controller"]["max_seconds"] = 24.0
    env = PandaBallEnvironment(scene, config, initial_state=report["initial_state"])
    model, data = env.model, env.data
    if variant == "no_robot_contacts":
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        # Preserve the table supporting the ball; remove all robot contacts.
        for name in ("ball", "table_top"):
            model.geom(name).contype[:] = 1
            model.geom(name).conaffinity[:] = 1
        mujoco.mj_forward(model, data)
    results = []
    for entry in report["steps"][:max_steps]:
        counts, peaks = Counter(), Counter()
        samples = []
        next_sample = float(data.time)
        max_height = float(data.joint("ball").qpos[2])

        def callback():
            nonlocal next_sample, max_height
            max_height = max(max_height, float(data.joint("ball").qpos[2]))
            env.track_success()
            if data.time + 1e-9 < next_sample:
                return
            next_sample = float(data.time) + .1
            contacts = []
            for i in range(data.ncon):
                contact = data.contact[i]
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                names = [model.geom(int(g)).name or f"geom{g}" for g in contact.geom]
                pair = " / ".join(sorted(names))
                counts[pair] += 1
                peaks[pair] = max(peaks[pair], float(abs(force[0])))
                contacts.append({"pair": pair, "normal_force_N": float(force[0]),
                                 "distance_m": float(contact.dist)})
            samples.append({"time": float(data.time), "ball_xyz": data.joint("ball").qpos[:3].tolist(),
                            "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                            "ctrl": data.ctrl.tolist(), "actuator_force": data.actuator_force.tolist(),
                            "contacts": contacts})

        action = np.asarray(entry["action_policy_frame"])
        execution = env.controller.execute(action, callback)
        joints = {}
        for side, j in env.controller.joints.items():
            actual = data.qpos[j["qpos"]]
            joints[side] = {"q": actual.tolist(),
                            "limit_margin_rad": np.minimum(actual - j["limits"][:, 0],
                                                           j["limits"][:, 1] - actual).tolist(),
                            "servo_error_rad": (data.ctrl[j["act"]] - actual).tolist(),
                            "actuator_force_Nm": data.actuator_force[j["act"]].tolist()}
        result = {"step": entry["step"], "controller": execution, "joints": joints,
                  "ball_xyz": data.joint("ball").qpos[:3].tolist(), "max_ball_height_m": max_height,
                  "success": bool(env.success), "contact_sample_counts": dict(counts),
                  "peak_sampled_normal_force_N": dict(peaks), "samples": samples}
        results.append(result)
        if not execution["reached"]:
            break
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=Path("runtime/seed_sweeps/seeds_000_100"))
    parser.add_argument("--scene", type=Path, default=Path("runtime/panda_ball_reference/scene.xml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42, 32])
    parser.add_argument("--variants", nargs="+", choices=["baseline", "no_robot_contacts", "long_timeout"],
                        default=["baseline", "no_robot_contacts", "long_timeout"])
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("runtime/diagnosis/execution"))
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in args.seeds:
        report_path = args.sweep / f"seed_{seed:03d}" / "attempt_001" / "report.json"
        report = json.loads(report_path.read_text())
        for variant in args.variants:
            results = replay(report, args.scene, variant, args.max_steps)
            path = args.output / f"seed_{seed:03d}_{variant}.json"
            path.write_text(json.dumps({"source_report": str(report_path), "variant": variant,
                                        "steps": results}, indent=2) + "\n")
            row = {"seed": seed, "source_episode": report["initial_state"]["episode"],
                   "variant": variant, "steps": len(results), "last_controller": results[-1]["controller"],
                   "max_ball_height_m": max(r["max_ball_height_m"] for r in results),
                   "evidence": str(path)}
            summary.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
