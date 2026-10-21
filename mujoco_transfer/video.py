"""Record simulation frames with observer-only goal and action markers."""

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


VIEW_PRESETS = {
    "overview": {"azimuth": 115.0, "elevation": -30.0},
    "opposite": {"azimuth": 295.0, "elevation": -30.0},
    "side": {"azimuth": 25.0, "elevation": -25.0},
    "top": {"azimuth": 115.0, "elevation": -89.9},
}
VIDEO_VIEWS = ("all", *VIEW_PRESETS, "orbit")
ORBIT_DEGREES_PER_SECOND = 15.0
TILE_WIDTH, TILE_HEIGHT = 640, 480


def make_ball_goal_marker(env):
    """A fixed lift-height reference, not a physical or policy-camera object.

    The source goal constrains height only. Initial ball XY anchors the visual
    reference without imposing a new XY target or a cube-containment condition.
    """
    condition = env.config.get("success_condition")
    if condition is not None:
        if condition["kind"] != "world_height":
            return None
        height = condition["minimum_height_m"]
    else:
        proxy = env.config.get("success_proxy")
        if proxy is None:
            return None
        height = env.initial_ball_height + proxy["ball_rise_metres"]
    center = env.data.joint("ball").qpos[:3].copy()
    center[2] = height
    return {
        "shape": "box",
        "center_world_m": center.tolist(),
        "half_size_m": [0.1, 0.1, 0.1],
        "rgba": [0.85, 0.85, 0.85, 0.25],
        "meaning": "Cube center marks goal height; XY is the initial ball position, "
                   "not a task constraint. Observer only; no physics or policy input.",
    }


def add_ball_goal_marker(scene, marker):
    """Append the same translucent cube to video scenes or viewer.user_scn."""
    if marker is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX, marker["half_size_m"],
        marker["center_world_m"], np.eye(3).ravel(), marker["rgba"])
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def make_camera(env, view="overview"):
    """Create an observer camera without changing the policy's input cameras."""
    if view not in VIDEO_VIEWS:
        raise ValueError(f"Unknown video view {view!r}; choose from {VIDEO_VIEWS}")
    preset = VIEW_PRESETS["overview" if view in ("all", "orbit") else view]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Include the raised home pose as well as tabletop contacts, leaving room
    # for the panel title above the robot.
    camera.lookat[:] = env.data.body("table").xpos + [0, 0, 0.45]
    camera.distance = 2.7
    camera.azimuth = preset["azimuth"]
    camera.elevation = preset["elevation"]
    return camera


class RolloutVideo:
    def __init__(self, env, path, fps=25, view="overview"):
        if view not in VIDEO_VIEWS:
            raise ValueError(f"Unknown video view {view!r}; choose from {VIDEO_VIEWS}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video fps must be positive and finite")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Video recording requires ffmpeg on PATH")
        path = Path(path)
        self.env, self.fps = env, fps
        self.view = view
        self.views = tuple(VIEW_PRESETS) if view == "all" else (view,)
        self.width = TILE_WIDTH * (2 if view == "all" else 1)
        self.height = TILE_HEIGHT * (2 if view == "all" else 1)
        self.camera = make_camera(env, view)
        self.goal_marker = make_ball_goal_marker(env)
        self.start_time = float(env.data.time)
        self.closed = False
        self.renderer = mujoco.Renderer(env.model, height=TILE_HEIGHT, width=TILE_WIDTH)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.options.sitegroup[:] = 0
        self.log = None
        try:
            self.log = open(path.with_suffix(".ffmpeg.log"), "w")
            self.process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.width}x{self.height}", "-r", str(fps), "-i", "-", "-an",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(path)],
                stdin=subprocess.PIPE, stderr=self.log,
            )
        except Exception:
            if self.log is not None:
                self.log.close()
            self.renderer.close()
            raise
        self.next_frame = env.data.time
        self.targets = []
        self.label = "Initial observation"

    @property
    def settings(self):
        """Observer configuration saved alongside the inference report."""
        return {
            "view": self.view,
            "panels": list(self.views),
            "resolution": [self.width, self.height],
            "fps": self.fps,
            "camera_distance_m": self.camera.distance,
            "camera_lookat": self.camera.lookat.tolist(),
            "ball_goal_marker": self.goal_marker,
            "fixed_views": {name: dict(preset) for name, preset in VIEW_PRESETS.items()
                            if name in self.views},
            "orbit_degrees_per_second": (
                ORBIT_DEGREES_PER_SECOND if self.view == "orbit" else None
            ),
        }

    def set_action(self, action, step):
        self.targets = self.env.adapter.action_to_sites(action)
        self.label = f"Policy keypose {step + 1} | left: pink, right: cyan"

    def _render_view(self, view):
        preset = VIEW_PRESETS["overview" if view == "orbit" else view]
        self.camera.azimuth = preset["azimuth"]
        self.camera.elevation = preset["elevation"]
        if view == "orbit":
            elapsed = float(self.env.data.time) - self.start_time
            self.camera.azimuth = (self.camera.azimuth
                                   + elapsed * ORBIT_DEGREES_PER_SECOND) % 360.0
        self.renderer.update_scene(self.env.data, camera=self.camera, scene_option=self.options)
        add_ball_goal_marker(self.renderer.scene, self.goal_marker)
        for target, color in zip(self.targets, [(1, .1, .6, .8), (0, .9, 1, .8)]):
            scene = self.renderer.scene
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                              [.018, .018, .018], target[:3, 3], np.eye(3).ravel(), color)
            scene.ngeom += 1
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, TILE_WIDTH, 60), fill=(15, 20, 30))
        draw.text((10, 7), f"{view.title()} view", fill=(170, 210, 255))
        draw.text((10, 24), self.label, fill="white")
        rise = self.env.data.body("ball").xpos[2] - self.env.initial_ball_height
        status = f"simulation t={self.env.data.time:.2f}s | ball rise={rise:.3f}m"
        if self.goal_marker is not None:
            status += f" | grey cube: goal z={self.goal_marker['center_world_m'][2]:.2f}m"
        draw.text((10, 41), status, fill="white")
        return frame

    def capture(self, force=False):
        if self.closed:
            raise RuntimeError("Cannot capture a frame after closing the video")
        if not force and self.env.data.time + 1e-9 < self.next_frame:
            return
        self.next_frame = self.env.data.time + 1 / self.fps
        # No physics steps occur between panels: each depicts the same state.
        if self.view == "all":
            frame = Image.new("RGB", (self.width, self.height))
            for index, view in enumerate(self.views):
                frame.paste(self._render_view(view),
                            ((index % 2) * TILE_WIDTH, (index // 2) * TILE_HEIGHT))
        else:
            frame = self._render_view(self.view)
        self.process.stdin.write(np.asarray(frame).tobytes())

    def hold(self, label, seconds=1):
        self.label = label
        for _ in range(round(self.fps * seconds)):
            self.capture(force=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("ffmpeg did not finish encoding within 30 seconds")
        finally:
            self.log.close()
            self.renderer.close()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
