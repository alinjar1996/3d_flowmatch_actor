# Pretrained 3DFA → manipulator_mujoco

This bridge runs the released **PerAct2 dual-Panda checkpoint** on the ball
task exported from `manipulator_mujoco` branch `origin/issue_38_lift_ball_panda`.
It renders RGB-D, predicts absolute Cartesian keyposes, and executes them
through differential IK and MuJoCo position actuators. No ROS, RLBench,
PyRep, or CoppeliaSim installation is required.

**The integration is runnable; successful zero-shot ball lifting is not
established.** Loading weights and moving the arms are separate from solving
the task. Videos and JSON reports record the actual result, including controller
timeouts. Base poses, ball starting pose, radius, mass, and task success
threshold now match the source settings described below. Internal arm
kinematics, finger collision shapes, and contact dynamics still differ.

The latest recording using those settings is
[`runtime/video-source-settings/rollout.mp4`](../runtime/video-source-settings/rollout.mp4),
with [its report](../runtime/video-source-settings/report.json). Earlier
`video-reference*` recordings used an approximate scene and are retained only
as diagnostics; they do not demonstrate the corrected source settings.
The new run reached the first predicted keypose, timed out on the second, and
did not lift the ball. The 19.72-second video records the result. Inference
took 0.254s initially and 0.030s on the next call on an RTX 4090.

## Run the prepared setup

From the repository root:

```bash
bash scripts/mujoco/rollout.sh --steps 25
```

This uses the local `.venv`, CUDA, the downloaded checkpoint, and the source
scene settings. It verifies the loaded scene against the extracted source
fixture and released demonstration before inference. Every rollout writes:

- `runtime/runs/<timestamp>/rollout.mp4`: synchronized 1280×960 video with
  overview, opposite, side, and top views at 25 simulation fps.
  Pink/cyan spheres show left/right predicted **site** targets. Short pauses
  display each new target and the final result.
- `cameras_*.png`: front, left wrist, right wrist input images.
- `observation_*.npz`: uint8 RGB, policy-frame point clouds, gripper states,
  and simulation joint positions.
- `report.json`: checkpoint hash, source branch/commit, settings verification,
  calibration details, raw predictions, inference time, pose errors, and task outcome.

`ffmpeg` must be on PATH. Use `--no-video` to disable recording, `--device cpu`
for CPU inference, or `--output runtime/my-new-run` for a named output directory.
Output directories must not already exist. GPU/GL calls need access to the
GPU device; an agent sandbox can hide an otherwise working CUDA installation.

Select a single 640×480 view, or a camera that orbits during simulation:

```bash
bash scripts/mujoco/rollout.sh --video-view top
bash scripts/mujoco/rollout.sh --video-view orbit
```

Available values are `all` (default), `overview`, `opposite`, `side`, `top`,
and `orbit`. The four views in `all` render the same simulation state; they
are not separate policy runs. Observer angles do not change policy camera
inputs. `report.json` records the selected video view.

For a live viewer with an initial side view:

```bash
MUJOCO_GL=glfw bash scripts/mujoco/rollout.sh --viewer --video-view side
```

Left-drag to rotate, right-drag to pan, and scroll to zoom. The live camera is
independent of the saved video; moving it does not change policy observations.

For inference without executing a predicted action:

```bash
.venv/bin/python -m mujoco_transfer.run --mode infer --device cuda \
  --scene runtime/panda_ball_reference/scene.xml \
  --config runtime/panda_ball_reference/reference_config.json
```

## Recreate the environment and assets

Python 3.10 was tested with torch 2.5.1/cu124 and MuJoCo 3.3.1 on an RTX 4090.
The existing environment is isolated under this repo.

```bash
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements-mujoco.txt
bash scripts/mujoco/prepare.sh /path/to/manipulator_mujoco
```

The preparation script downloads and SHA256-verifies the public 422MB
checkpoint and 229MB test-state ZIP. It reads an existing git ref with
`git archive`; it does **not** switch branches, commit, push, or modify the
`manipulator_mujoco` working tree. The source commit is recorded in `source.json`.
Re-running preparation regenerates the scene and calibration, then verifies
the result independently after an environment reset.
All generated files, recordings, checkpoints, and the environment are ignored
by git.

The Panda branch originally has fixed-open fingers, no joint actuators, and no
wrist cameras. Preparation adds four finger sliders, 14 arm position actuators,
three named cameras, and a 2ms implicit integrator. The source assets and their
licensing remain in their original repository; they are exported locally.

## Model and observation contract

| Item | Required convention |
|---|---|
| Checkpoint | `katefgroup/3d_flowmatch_actor/3dfa_peract2.pth` |
| Architecture | 120 dimensions, 8 heads, 3 vision-language layers, 4 shared layers |
| Denoising | Rectified flow, 5 steps; upstream inference behavior retained |
| Arms | **left, right**, independently of XML joint storage order |
| Cameras | **front, wrist_left, wrist_right**, 256×256 |
| RGB | uint8 input, divided by 255 once, then CLIP normalization inside the model |
| Geometry | Dense aligned XYZ in metres, in the checkpoint's world frame |
| Proprioception | `(1,3,2,8)`, oldest → newest; replicate initial state for padding |
| State/action | `xyz, qx, qy, qz, qw, openness`; 1=open, 0=closed |
| Output | One absolute keypose per arm, `(2,8)`; not joint actions or deltas |
| Gripper execution | Threshold 0.5; change after the arm targets are reached |

The checkpoint's workspace normalizer is loaded unchanged:

```text
min = [-0.18802239, -0.67799366, 0.67463952]
max = [ 0.77680248,  0.49763644, 1.56688786]
```

Do not replace these values with MuJoCo workspace bounds. Transform observations
and actions consistently instead. The report identifies out-of-bounds
proprioception without silently clipping it.

The published checkpoint has 869 tensors. Its 34 legacy `norm_kv` tensors
belong to post-normalization attention layers where the released code sets
`norm_kv=None` and never uses them. The adapter audits and removes only those
known unused tensors, then strictly loads all **835 active tensors**. Missing,
unexpected, or shape-mismatched active weights are errors. Frozen CLIP weights
are already included, so separate backbone-weight downloads are unnecessary.
OpenAI CLIP's packaged BPE tokenizer supplies the same vocabulary, with padding
removed for the single instruction.

## Verified source scene settings

There is no single ball starting pose across the dataset: the task randomizes
ball placement. This setup matches **released test `variation0/episode3`**.
The ball's full XYZ + XYZW pose is copied from its first observation. Reset
does not advance physics before the initial policy observation.

Robot base poses come from native CoppeliaSim extraction of the upstream
RLBench scene, not a trajectory fit. The shared `DualPanda` root is not an
individual arm's base. We define each Panda `link0` frame from its first fixed
joint frame, translated by local `(0, 0, -0.333)` metres.

| Setting | Source value applied in MuJoCo (world frame, metres) |
|---|---|
| Left base XYZ | `[-0.242567182, 0.022205890, 0.750214760]` |
| Left base yaw | approximately `+5.000098°` |
| Right base XYZ | `[0.792041898, -0.243609285, 0.750214760]` |
| Right base yaw | approximately `-179.999926°` |
| Ball initial XYZ | `[0.373337835, -0.109500423, 0.900002301]` |
| Ball initial XYZW | `[-0.000007615, -0.000140926, 0.127347127, 0.991858244]` |
| Ball radius / mass | `0.150000006 m` / `1.0 kg` |
| Task goal | ball centre world `z >= 0.95 m` |
| Table top | `0.750021972 m` |

The exact full base quaternions, source asset hashes, and frame definition
are in [`configs/peract2_scene.json`](configs/peract2_scene.json).
The [original task code](https://github.com/markusgrotz/RLBench/blob/8af748c51287989294e00c9c670e3330a0e35ed5/rlbench/bimanual_tasks/bimanual_lift_ball.py)
has **no fixed target X/Y/orientation and no hold-time requirement**.
The demo's final ball pose is an achieved state, not a target constraint.
The unrelated `target_0` marker from the MuJoCo branch is removed.

The checkpoint does not record the RLBench git revision used in training.
The pinned upstream scene was checked against the released demo: its front
camera matrix agrees exactly and home joint configurations agree within
0.000661 rad. This establishes the reproduced source settings without claiming
an independently verified training revision.

The table is a flat MuJoCo box with the source's world bounding-box footprint
and upper height. It omits the native table's roughly 27-microradian tilt.

To reproduce the native extraction, use an optional audit environment with
the upstream `markusgrotz/PyRep` fork and CoppeliaSim 4.1.0 configured as in the
main repository README:

```bash
.venv/bin/python scripts/mujoco/audit_source_scene.py \
  --rlbench-root /path/to/RLBench \
  --output runtime/source-audit/reproduced_scene.json \
  --check-fixture mujoco_transfer/configs/peract2_scene.json
```

Use the source revision recorded in the fixture. The audit checks the extracted
geometry and all three source asset hashes. This extraction was run successfully
against the fixture; PyRep and CoppeliaSim remain optional for normal inference.

## TCP calibration and remaining differences

`reference.py` extracts numeric metadata from the released demo using a
restricted unpickler. With source bases held fixed, it fits only the
end-effector frame correction, separately for each arm:

```text
T_RLBench_gripper(q) ≈ T_MuJoCo_site_with_source_base(q) × T_site_gripper
```

It uses 47 frames for fitting and 94 other frames for checking the fit.
Measured RMS residuals on those frames were approximately:

| Arm | Position | Orientation |
|---|---:|---:|
| Left | 14.75 mm | 0.03619 rad |
| Right | 17.87 mm | 0.04368 rad |

These are adjacent frames from one demonstration, not independent task-success
validation. A rigid fit cannot remove every difference in the robot models.
In particular, the source joint4-to-joint5 frame has a roughly 3.9993-degree
offset and different translation from this MuJoCo Panda. With the same joint
angles and bases, the fifth joint pivot differs by about 27mm. A fixed TCP
correction cannot reproduce this configuration-dependent difference.
We retain the branch's internal kinematics and report this residual explicitly.

The generated scene adopts reference starting joints and front camera
intrinsics/extrinsics; wrist mounts use the approximate TCP fit. Source-matched
base and ball poses do not imply exact full-chain kinematics, wrist images,
contact dynamics, or successful policy transfer.

The standalone `configs/panda_ball.json` is an **uncalibrated identity-frame
baseline**. Use `runtime/panda_ball_reference/reference_config.json` with the
reference scene for transfer experiments. Mixing those scene/config pairs is
incorrect.

For another scene, `calibrate.py` fits a common world transform from measured
corresponding landmarks:

```bash
.venv/bin/python -m mujoco_transfer.calibrate \
  --landmarks measured_landmarks.npz --output runtime/my_config.json
```

The NPZ must contain `mujoco_points` and `policy_points` (Nx3 metres, at least
three non-collinear corresponding points). This fits a rigid transform, without
scaling. Per-arm TCP offsets, base placements, and camera calibration remain
separate requirements.

For each arm, the adapter uses:

```text
T_policy_gripper = T_policy_mujoco × T_mujoco_site × T_site_gripper
T_mujoco_site_target = inv(T_policy_mujoco) × T_policy_gripper_target × inv(T_site_gripper)
```

MuJoCo camera -Z/+Y conventions, XYZW ↔ rotation-matrix conversion, metric
depth, and image row order are handled explicitly. Multisample antialiasing
is disabled for RGB-D observations because resolved depth may otherwise
represent an off-centre sample. A rendered tilted-plane regression verifies
the projection to 0.1mm.

## Use inside an existing manipulator_mujoco loop

The exported task is a standalone integration harness. It is not automatically
registered as a ROS planner. To wrap an existing simulator instance, prepare
the matching cameras/fingers and use its current `MjModel` and `MjData`:

```python
from mujoco_transfer.environment import ObservationAdapter, read_config
from mujoco_transfer.policy import FlowMatchPolicy

config = read_config("runtime/panda_ball_reference/reference_config.json")
adapter = ObservationAdapter(model, data, config)
policy = FlowMatchPolicy("checkpoints/3dfa_peract2.pth", device="cuda")
policy.reset("Lift the ball")  # Also reset history on every episode reset.

rgb, xyz, state = adapter.observe()
action = policy.predict(rgb, xyz, state)
site_targets = adapter.action_to_sites(action)  # [left, right] 4×4 poses
# Submit these targets to the task's Cartesian/path planner. Apply the
# corresponding action[:, 7] gripper commands once the keyposes are reached.
adapter.close()
```

The included controller uses local differential IK and joint-limited servos.
Joint limits are part of the bounded least-squares solve so a saturated wrist
can redistribute motion to other joints.
It has no obstacle-aware path planning. Timeout reports distinguish inability
to reach a keypose from the policy predicting an unhelpful keypose. For cluttered
tasks, connect the targets to the existing collision-aware planner.

The reference setup implements the source's absolute `z >= 0.95 m` success
condition. The uncalibrated baseline alone retains its older relative-height
proxy. The unchanged checkpoint is task-trained on PerAct2; it is not a
general guarantee of zero-shot transfer to every MuJoCo task. If camera/frame
matching and controller checks pass but task success remains poor, collect
MuJoCo demonstrations and fine-tune from these weights in the same input/action
conventions. The supplied observation NPZs are diagnostics, not a complete
demonstration-training dataset.

## Verification

```bash
.venv/bin/python -m mujoco_transfer.verify_settings

PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HOME="$PWD/runtime/huggingface" MUJOCO_GL=egl \
TEST_RENDER=1 TEST_CHECKPOINT=1 OMP_NUM_THREADS=4 \
.venv/bin/python -m pytest -q tests/test_mujoco_transfer.py tests/test_scene_settings.py
```

The [settings report](../runtime/panda_ball_reference/settings_verification.json)
checks compiled world poses after reset against both source data and generated
configuration, with tolerances of 1 micrometre and 1 microradian. It checks ball
radius/mass, table height, removal of the unrelated target, and actual success
behavior immediately below and at the threshold. Tests deliberately perturb
poses and the threshold to verify mismatches are detected.

Other checks cover camera projection, nontrivial world/TCP round trips, arm order,
history/reset semantics, strict checkpoint parity, physical Cartesian motion,
gripper polarity, and unreachable-target reporting. Clearing `PYTHONPATH` and
disabling pytest plugin autoload prevent an already-sourced ROS installation
from injecting unrelated test plugins. All 18 tests passed locally, including
the actual renderer, checkpoint, and generated-scene integration checks.

Sources: [checkpoint and test metadata](https://huggingface.co/katefgroup/3d_flowmatch_actor/tree/main),
[actor evaluation configuration](../online_evaluation_rlbench/eval_peract2.sh),
[upstream observation/execution logic](../online_evaluation_rlbench/utils_with_bimanual_rlbench.py),
[Panda task branch](https://github.com/alinjar1996/manipulator_mujoco/tree/issue_38_lift_ball_panda),
and [MuJoCo Python API](https://mujoco.readthedocs.io/en/3.3.1/python.html).
