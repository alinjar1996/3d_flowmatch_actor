# Pretrained 3DFA tray lifting in MuJoCo

This pipeline runs the released `checkpoints/3dfa_peract2.pth` checkpoint on
the supported `bimanual_lift_tray` task with the instruction **“Lift the tray”**.
It uses the physical dual-Panda embodiment prepared from
`manipulator_mujoco`'s `issue_38_lift_ball_panda` branch, with the source tray,
item and support geometry replacing the ball task. The policy is unchanged.

The scene, all 100 source resets and both observation renderers have been
checked. The final setup succeeded on **1 of 3 smoke-test starts**; a reliable
success rate has not yet been established. The ball task's results do not
describe this task's performance.

| Source episode | Result | Maximum tray height |
|---|---|---|
| 0 | Grasp failed; later controller timeout | 0.902 m |
| 85 (reset seed 0) | All four source conditions passed | 1.243 m |
| 47 (reset seed 1) | Step limit; all four conditions never passed together | 1.387 m |

Episode 85 ended with both tray and item above the goal and both sensors
detecting the tray. Its first successful physical state was independently
rechecked from saved joint positions. See the
[four-angle successful video](../runtime/tray_transfer/smoke_seeds_0_1/seed_000/attempt_001/rollout.mp4)
and [validation with every tested result](../runtime/tray_transfer/validation.json).
The final regression suite passed **169 tests**, with one expected skip.

## Run the prepared task

```bash
cd /home/aks-lab/3dfa/3d_flowmatch_actor

PYTHONPATH= MUJOCO_GL=egl bash scripts/mujoco/rollout_tray.sh \
  --episode 85 --seed 0 --steps 25 --video-view all
```

The wrapper selects CUDA, `runtime/panda_tray_reference/scene.xml`, its
`reference_config.json`, and `source_mujoco` observations from
`runtime/source_visuals_tray`. Inference needs MuJoCo and the exported assets;
it does not launch CoppeliaSim. The default output is a fresh directory under
`runtime/runs/<UTC timestamp>/`. Add `--output runtime/my-tray-run` to choose
a new directory explicitly.

`rollout.mp4` shows synchronized overview, opposite, side and top views in a
1280×960 grid. The translucent grey 20 cm cube is centered at the initial
tray X/Y and **z = 1.2 m**. It marks the target height, has no collision, and
is absent from policy observations. Success also requires the item height and
both hand sensors described below; being inside the cube is not a condition.

For a live viewer with a movable camera, while still recording all four views:

```bash
PYTHONPATH= MUJOCO_GL=glfw bash scripts/mujoco/rollout_tray.sh \
  --episode 85 --seed 0 --steps 25 --viewer --video-view all
```

Use `--video-view orbit`, `overview`, `opposite`, `side`, or `top` for another
recording view. Observer camera changes do not change policy camera inputs.
Add `--observation-renderer source_native` to compare the live native rendering
bridge; that option requires the installed CoppeliaSim/PyRep environment and
a working display. Physics still runs entirely in MuJoCo.

## Prepare on a fresh setup

The task is already prepared on this machine. Preparation requires the existing
ball setup in `runtime/panda_ball_reference`, the repository `.venv`, the
checkpoint, `runtime/peract2_test.zip`, and the CoppeliaSim 4.1/PyRep source-audit
installation. For a fresh setup where `runtime/panda_tray_reference` does not
yet exist:

```bash
PYTHONPATH= bash scripts/mujoco/prepare_tray.sh
```

This downloads and verifies the pinned tray task sources, exports source visuals,
creates the physical scene, and writes `settings_verification.json`. Preparation
refuses to overwrite an existing physical scene. It reuses the Panda embodiment;
no git checkout, commit or push is needed.

The independent audit is recorded in
[peract2_tray_scene.json](configs/peract2_tray_scene.json), from RLBench commit
`8af748c51287989294e00c9c670e3330a0e35ed5`. The prepared task preserves:

| Setting | Source value or implementation |
|---|---|
| Robot bases | Both complete source base poses from the audited fixture |
| Tray collision shape | 0.21 × 0.36 × 0.006 m cuboid, approximately 0.4536 kg |
| Item | Independent 0.04 m cube, approximately 0.064 kg |
| Supports | Both source compound cylinder racks, at the selected reset's poses |
| Table | Source dimensions and surface height, approximately 0.750022 m |
| Cameras | Source front and wrist camera calibration |
| Sliding friction | Source value 0.5 on the task objects |

The tray and item are separate physical free bodies. There are no welds,
kinematic attachments, replayed expert actions, or source-simulator physics.
For the initial approach, [target_ik.py](target_ik.py) solves the predicted
hand poses using several generic joint seeds, checks the candidate joint path
against the physical collision geometry, and executes position-servo references
within the configured speed limits. This avoids the wrist/shoulder joint-limit
trap observed with local IK on the tray's sideways grasp. Later keyposes use
the existing Cartesian controller. Both stages move the robot through physics.
The path check samples a straight joint interpolation; it is not a general
obstacle-routing planner or a guarantee about subsequent contact dynamics.
The source tray inertia tensor violates the physical triangle inequality, so
MuJoCo uses the inertia of the tray's cuboid geometry with the source mass.
This difference is recorded under `dynamics_adaptations` in the configuration
and reports. Contact dynamics also use MuJoCo's solver and the physical Panda
collision meshes.

## Reset randomization

```bash
PYTHONPATH= bash scripts/mujoco/rollout_tray.sh \
  --randomize --reset-seed 42 --seed 0 --steps 25 --video-view all
```

`--reset-seed` selects uniformly, with replacement, from the **100 released
tray test initial states**. Each selection copies the paired tray, item and
both rack poses, both arm postures and gripper openings from the same episode.
These are actual simulation resets, not images or action trajectories replayed
from demonstrations. Robot bases stay fixed, and no independent joint or object
jitter is added. To select a particular start, use `--episode 0` through
`--episode 99` instead of `--randomize`.

`--seed` controls policy sampling separately. Reports save the selected source
episode, reset seed and verification of the actual physical reset state.

## Run reset seeds 0–100

```bash
PYTHONPATH= MUJOCO_GL=egl bash scripts/mujoco/sweep_tray.sh \
  --start-seed 0 --end-seed 100 --policy-seed 0 \
  --device cuda --steps 25 --video-view all \
  --output runtime/seed_sweeps/tray_000_100
```

This runs **101 trials**, with policy seed fixed at zero and a separately
sampled source reset for each reset seed. Sampling with replacement means these
are not 101 distinct starts or an exhaustive traversal of all 100 episodes.
The batch does not run until this command is invoked.

Append `--dry-run` to preview commands. If interrupted, repeat the same command
with `--resume`; add `--retry-errors` to retry execution errors. Resume checks
the scene, assets, checkpoint, implementation and settings fingerprints before
reusing prior results. Choose a fresh output directory for changed experiments.

Outputs under the sweep directory:

- `seed_000/attempt_001/rollout.mp4` through `seed_100/...`: four-angle videos.
- `seed_XXX/attempt_001/report.json`: settings/reset verification, predictions,
  controller outcome, object heights and source-task success details.
- `seed_XXX/attempt_001/observation_*.npz` and `cameras_*.png`: policy inputs and
  actual simulation state; `source_renderer/` contains renderer diagnostics.
- `summary.json` and `summary.csv`: completed trials, successes, failures,
  execution errors, selected source episodes and artifact paths.

## Success and inference code

The original source task requires **all four conditions simultaneously**:

1. Tray origin world z ≥ 1.2 m.
2. Independent item origin world z ≥ 1.2 m.
3. The left hand's source proximity sensor detects the tray.
4. The right hand's source proximity sensor detects the tray.

[tray_success.py](tray_success.py) evaluates the audited sensor volumes against
the tray collision box using the actual physical TCPs and current free-body
poses. Merely lifting the tray past the cube is insufficient. Success is latched
once all four conditions occur; the source task has no sustained-holding period,
and this controller completes its current action. Inspect the video and final
state separately when evaluating whether the object remained held.
`first_success` retains the first simultaneous success time, conditions and
physical state; `final_success_details` describes the state after the action.

The execution path is:

1. [tray.py](tray.py) prepares the physical task and verifies geometry and goals.
2. [initial_states.py](initial_states.py) loads only a selected demonstration's
   initial state and resets the physical environment.
3. [source_visual_renderer.py](source_visual_renderer.py) synchronizes both
   actual physical hands, tray, independent item and supports into the source
   visual scene, then renders current RGB and metric XYZ point clouds.
4. [run.py](run.py) calls `FlowMatchPolicy.predict(...)` from [policy.py](policy.py)
   and sends the predicted absolute hand keyposes and gripper commands to the
   Cartesian controller in [environment.py](environment.py).
5. Each physical simulation step checks the source success conditions and
   supplies frames to [video.py](video.py); the next prediction uses fresh state.

Source visual robot geometry is aligned to the physical TCPs through bounded
inverse kinematics. It does not move the physical robot or replace controller
commands. This uses the same observation-transfer approach as
[the ball setup](RENDERING_TRANSFER.md), with the tray task's own assets, resets,
instruction and success conditions.
