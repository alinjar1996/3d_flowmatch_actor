"""Record physical simulation frames and predicted Cartesian target markers."""

import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class RolloutVideo:
    def __init__(self, env, path, fps=25):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Video recording requires ffmpeg on PATH")
        self.env, self.fps = env, fps
        self.renderer = mujoco.Renderer(env.model, height=480, width=640)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.options.sitegroup[:] = 0
        # Observer view is separate from the policy's calibrated input cameras.
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = env.data.body("table").xpos + [0, 0, 0.3]
        self.camera.distance = 2.2
        self.camera.azimuth = 115
        self.camera.elevation = -30
        self.log = open(path.with_suffix(".ffmpeg.log"), "w")
        self.process = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "640x480", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=self.log,
        )
        self.next_frame = env.data.time
        self.targets = []
        self.label = "Initial observation"

    def set_action(self, action, step):
        self.targets = self.env.adapter.action_to_sites(action)
        self.label = f"Policy keypose {step + 1} | left: pink, right: cyan"

    def capture(self, force=False):
        if not force and self.env.data.time + 1e-9 < self.next_frame:
            return
        self.next_frame = self.env.data.time + 1 / self.fps
        self.renderer.update_scene(self.env.data, camera=self.camera, scene_option=self.options)
        for target, color in zip(self.targets, [(1, .1, .6, .8), (0, .9, 1, .8)]):
            scene = self.renderer.scene
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                              [.018, .018, .018], target[:3, 3], np.eye(3).ravel(), color)
            scene.ngeom += 1
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 640, 42), fill=(15, 20, 30))
        draw.text((10, 7), self.label, fill="white")
        rise = self.env.data.body("ball").xpos[2] - self.env.initial_ball_height
        draw.text((10, 24), f"simulation t={self.env.data.time:.2f}s | ball rise={rise:.3f}m",
                  fill="white")
        self.process.stdin.write(np.asarray(frame).tobytes())

    def hold(self, label, seconds=1):
        self.label = label
        for _ in range(round(self.fps * seconds)):
            self.capture(force=True)

    def close(self):
        self.renderer.close()
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.log.close()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
