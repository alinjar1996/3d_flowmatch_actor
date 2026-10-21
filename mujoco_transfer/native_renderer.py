"""Live source-appearance observations of the actual MuJoCo state.

CoppeliaSim runs in a separate process because its Qt/OpenGL context conflicts
with MuJoCo EGL in one process.  It never advances physics or supplies actions.
Each request and response is retained for auditing the transferred state.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .geometry import matrix_pose, pose_matrix
from .tasks import task_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeObservationRenderer:
    def __init__(self, env, output_dir, *, source_root=None, coppelia_root=None,
                 archive=None, timeout=120):
        self.env = env
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.sequence = 0
        self.last_metadata = {}
        self.process = None
        self.task_name = env.config.get("source_task", "bimanual_lift_ball")
        spec = task_spec(self.task_name)
        self.object_name = env.config.get("object_name", "ball")
        self.static_object_names = env.config.get("static_object_names", [])
        self.object_names = [self.object_name, *env.config.get("additional_object_names", []),
                             *self.static_object_names]
        expected_objects = [spec["object_name"], *spec["additional_object_names"],
                            *spec["static_object_names"]]
        if self.object_names != expected_objects:
            raise ValueError(f"Source task {self.task_name} expects {expected_objects}, "
                             f"but physical environment declares {self.object_names}")
        source_root = Path(source_root or PROJECT_ROOT / "runtime/source-audit/RLBench").resolve()
        coppelia_root = Path(coppelia_root or PROJECT_ROOT /
                             "runtime/source-audit/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04").resolve()
        archive = Path(archive or PROJECT_ROOT / "runtime/peract2_test.zip").resolve()
        for path in (source_root / "rlbench/task_design_bimanual.ttt",
                     source_root / f"rlbench/task_ttms/{self.task_name}.ttm",
                     coppelia_root / "libcoppeliaSim.so", archive):
            if not path.is_file():
                raise FileNotFoundError(f"Source rendering dependency missing: {path}")
        environment = os.environ.copy()
        environment.update({
            "COPPELIASIM_ROOT": str(coppelia_root),
            "LD_LIBRARY_PATH": str(coppelia_root) + ":" + environment.get("LD_LIBRARY_PATH", ""),
            "QT_PLUGIN_PATH": str(coppelia_root),
            "QT_QPA_PLATFORM_PLUGIN_PATH": str(coppelia_root / "platforms"),
            "QT_QPA_PLATFORM": "xcb",
            # Use the installed PyRep wheel, which includes its compiled CFFI
            # extension. The source checkout does not contain that extension.
            "PYTHONPATH": str(PROJECT_ROOT),
        })
        self.log = (self.output_dir / "worker.log").open("w")
        ready = self.output_dir / "ready.json"
        ready.unlink(missing_ok=True)
        command = [sys.executable, "-m", "mujoco_transfer.native_worker",
                   "--source-root", str(source_root), "--archive", str(archive),
                   "--output", str(self.output_dir), "--task", self.task_name]
        try:
            self.process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                            stdin=subprocess.PIPE, stdout=self.log,
                                            stderr=subprocess.STDOUT, text=True)
            self.provenance = self._wait_json(ready)
        except BaseException:
            self.close()
            raise

    def _wait_json(self, path):
        deadline = time.monotonic() + self.timeout
        while not path.is_file():
            if self.process.poll() is not None:
                raise RuntimeError(f"Source renderer exited with {self.process.returncode}; "
                                   f"see {self.output_dir / 'worker.log'}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Source renderer timed out; see {self.output_dir / 'worker.log'}")
            time.sleep(0.02)
        result = json.loads(path.read_text())
        if "error" in result:
            raise RuntimeError(f"Source renderer: {result['error']}; see {self.output_dir / 'worker.log'}")
        return result

    def observe(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Source renderer is closed")
        state = self.env.adapter.state()
        object_poses = {}
        for name in self.object_names:
            if name in self.static_object_names:
                body = self.env.data.body(name)
                physical_matrix = np.eye(4)
                physical_matrix[:3, 3] = body.xpos
                physical_matrix[:3, :3] = body.xmat.reshape(3, 3)
            else:
                physical_pose = self.env.data.joint(name).qpos.copy()
                physical_matrix = pose_matrix(physical_pose[[0, 1, 2, 4, 5, 6, 3]])
            object_poses[name] = matrix_pose(self.env.adapter.world_transform @ physical_matrix).tolist()
        request = {
            "sequence": self.sequence,
            "simulation_time": float(self.env.data.time),
            "object_name": self.object_name,
            "object_pose_xyzw": object_poses[self.object_name],
            "object_poses_xyzw": object_poses,
            "policy_state": state.tolist(),
            "joint_positions": {
                side: [float(self.env.data.joint(f"joint{i}_{side}").qpos[0])
                       for i in range(1, 8)] for side in ("left", "right")},
            "finger_positions": {
                side: [float(self.env.data.joint(f"{finger}_finger_{side}").qpos[0])
                       for finger in ("left", "right")] for side in ("left", "right")},
            "world_from_policy": np.linalg.inv(self.env.adapter.world_transform).tolist(),
        }
        if self.object_name == "ball":
            request["ball_pose_xyzw"] = object_poses["ball"]
        request_path = self.output_dir / f"request_{self.sequence:04d}.json"
        response_path = self.output_dir / f"response_{self.sequence:04d}.json"
        response_path.unlink(missing_ok=True)
        request_path.write_text(json.dumps(request, indent=2))
        self.process.stdin.write(json.dumps({"request": str(request_path)}) + "\n")
        self.process.stdin.flush()
        self.last_metadata = self._wait_json(response_path)
        with np.load(self.output_dir / self.last_metadata["observation_file"]) as observation:
            rgb, points = observation["rgb"].copy(), observation["points"].copy()
        if rgb.shape != (3, 256, 256, 3) or points.shape != rgb.shape or not np.isfinite(points).all():
            raise RuntimeError("Invalid native renderer observation")
        self.sequence += 1
        return rgb, points, state

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"quit": true}\n')
                    process.stdin.flush()
                    process.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            if process.stdin is not None:
                process.stdin.close()
        if getattr(self, "log", None) is not None:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
