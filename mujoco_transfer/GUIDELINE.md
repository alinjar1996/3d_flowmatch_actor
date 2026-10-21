# Ball lifting: compare Panda CEM against pretrained 3DFA

Use **two independent methods on the same physical Panda scene**:

1. Your Panda CEM planner chooses joint commands from simulator state.
2. The released 3DFA policy chooses hand keyposes from rendered observations;
   the existing Cartesian controller executes them.

This is a comparison of complete control systems. CEM does not execute 3DFA's
predictions, and 3DFA predictions are not used to initialize CEM.

## What is ready, and what needs connecting

| Component | Current status |
|---|---|
| Panda CEM | Exists on `manipulator_mujoco` branch `issue_38_lift_ball_panda`. |
| 3DFA ball lifting | Runnable with the commands below, including four-view videos. |
| Shared scene, reset sampler, verification, video writer | Already implemented in this repository. |
| CEM running through that shared environment | Requires the integration described in section 4; not implemented by this document. |
| Paired CEM/3DFA sweep | Requires that integration before it can produce comparable results. |

There is currently no `--controller cem` or `--method cem` option in
`mujoco_transfer.run` or `mujoco_transfer.sweep`. The existing sweep runs 3DFA.
This guide documents the route with the fewest changes; it does not claim
that the combined comparison runner already exists.

## 1. Use the existing Panda branch and prepared assets

Working directories:

```text
3DFA:             /home/aks-lab/3dfa/3d_flowmatch_actor
manipulator:      /home/aks-lab/colcon_ws/src/manipulator_mujoco
CEM source ref:   issue_38_lift_ball_panda
CEM audited SHA:  7f7ff07ef76357f579a0b688a893ecc913a6d058
```

The Panda branch is already the source of the physical assets exported into
`runtime/panda_ball_reference/`. Its origin is recorded in
[`source.json`](../runtime/panda_ball_reference/source.json).
The currently checked-out `issue_60_ball_lift_comparison` is a different,
UR5e-based implementation. Read or copy the Panda ref without switching that
working tree:

```bash
git -C /home/aks-lab/colcon_ws/src/manipulator_mujoco \
  rev-parse issue_38_lift_ball_panda

git -C /home/aks-lab/colcon_ws/src/manipulator_mujoco \
  show issue_38_lift_ball_panda:real_demo/sampling_based_planner/mpc_planner.py
```

Use these existing files as the common experiment definition:

| File | Purpose |
|---|---|
| `runtime/panda_ball_reference/scene.xml` and its includes/assets | Physical model for both methods |
| `runtime/panda_ball_reference/reference_config.json` | Exact bases, TCP transforms, homes and success criterion |
| `runtime/peract2_test.zip` | Released paired ball/robot initial states |
| `runtime/source_visuals/manifest.json` and its assets | Source appearance for 3DFA observations |
| `checkpoints/3dfa_peract2.pth` | Unchanged pretrained policy |

Do not recreate the prepared scene if these assets are already present and
verification passes. Reproduction instructions are in [README.md](README.md).

The shared scene has a ball radius of approximately **0.15 m**, mass **1 kg**,
table surface **z = 0.750022 m**, and initial ball centre approximately
**z = 0.90 m**. Physics runs at **0.002 s**. Copy the exact base poses from the
configuration; do not approximate them from the visualization.

The original CEM branch has a different ball (0.12 m radius, 0.01 kg mass),
different homes/bases, and a target initially at z = 2 m. Launching its
unmodified scene alongside the prepared 3DFA scene is not the same experiment.

## 2. Run and inspect the existing 3DFA baseline

Run from a terminal with your working CUDA setup:

```bash
cd /home/aks-lab/3dfa/3d_flowmatch_actor
PYTHONPATH= .venv/bin/python -m mujoco_transfer.verify_settings
PYTHONPATH= .venv/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
command -v ffmpeg
```

Then run a known source start with the existing controller:

```bash
PYTHONPATH= bash scripts/mujoco/rollout.sh \
  --observation-renderer source_mujoco \
  --episode 94 --seed 0 --steps 25 --video-view all \
  --output runtime/ball_cem_vs_3dfa/3dfa_episode094
```

Choose a new output directory on a rerun. A single rollout does not overwrite
an existing directory. `--steps` counts policy keyposes, not physics steps.

Inspect:

```text
runtime/ball_cem_vs_3dfa/3dfa_episode094/
  rollout.mp4            actual physics: overview, opposite, side, top
  cameras_000.png        three camera images supplied to the policy
  observation_000.npz    RGB, XYZ, TCP state, qpos, qvel and ctrl
  report.json           reset, predictions, execution errors and outcome
```

The light grey cube marks the initial ball XY and height z = 0.95 m. It is an
observer overlay, with no collision geometry or policy-camera appearance.
Ball containment inside the cube is not a success condition.

For an interactive viewer, add `--viewer` and prefix the command with
`MUJOCO_GL=glfw`. Keep `--video-view all` to save all four angles regardless of
how you move the live camera.

Always select `source_mujoco` explicitly: the default `mujoco` option uses the
older branch-camera observation baseline. The source visual model is driven
by the actual simulated ball and robot state; later demonstration actions are
not replayed. Record this observation adaptation in any comparison report.

## 3. Inspect the existing Panda CEM baseline

The Panda branch's existing entry points are:

| File in `manipulator_mujoco` Panda ref | Role |
|---|---|
| `real_demo/launch/dual_arm_demo.launch.py` | ROS launch arguments |
| `real_demo/real_demo/dual_arm_demo.py` | Simulation loop and pick/move phase gates |
| `real_demo/sampling_based_planner/mpc_planner.py` | `run_cem_planner.compute_control(...)` |
| `real_demo/sampling_based_planner/mjx_planner.py` | CEM sampling, projection, costs and predicted dynamics |

In a terminal already configured with the **Panda branch's installed ROS
package and its existing CEM environment**, the native inspection command is:

```bash
ros2 pkg prefix real_demo
ros2 launch real_demo dual_arm_demo.launch.py \
  use_hardware:=false record_data:=false \
  num_batch:=500 num_steps:=15 maxiter_cem:=1 \
  maxiter_projection:=5 num_elite:=0.05 timestep:=0.1
```

Confirm the package prefix belongs to your Panda build. Changing a source
branch alone does not select a different installed ROS package. If necessary,
build the Panda branch in a separate workspace and source that workspace;
keep the working CEM dependencies separate from the 3DFA `.venv`.

This command opens the original CEM experiment, **not the matched comparison**.
It is verified against the launch source but was not executed for this guide.
The branch's `record_data` path has stale table-body references and missing
result-buffer keys; use the existing 3DFA video/report helpers for the new
comparison instead of depending on that recorder. Native CEM has no equivalent
four-view MP4 output in this branch.

## 4. Minimum integration for the same-scene comparison

Keep the pretrained weights, 3DFA inference, CEM sampler, CEM update,
projection and cost weights unchanged initially. Reuse the prepared physical
scene and existing evaluation helpers. The necessary changes concern the
CEM environment interface and how its predicted controls are applied.

Recommended change surface, **proposed files rather than existing commands**:

- One `cem_baseline.py` adapter that calls the Panda planner and handles joint
  order, task phase, goal, reset and actuation.
- A pinned copy of the Panda planner and its helper modules, with the
  constructor and rollout-interface changes below. Preserve provenance and
  the integration diff. `mpc_planner.py` unconditionally imports `quat_math`,
  `Simple_MLP.mlp_singledof` and `ik_based_planner.ik_solver`; copy their
  dependencies from the same Panda ref, or remove unused optional imports in
  the isolated copy when learned projection/IK are disabled. Do not resolve
  these modules accidentally from the current UR5e checkout.
- One `compare_ball.py` runner reusing `PandaBallEnvironment`,
  `SourceEpisodeSampler`, scene verification and `RolloutVideo`. Keep the
  existing single-method scripts working unchanged.

### 4.1 Reuse CEM's existing API

The existing call is:

```python
velocity, cost, cost_terms, velocity_horizon, joint_horizon, right_ee, left_ee = (
    planner.compute_control(cost_weights, current_pos, current_vel, task)
)
```

`current_pos` and `current_vel` contain **14 arm joints**, right seven followed
by left seven in the audited Panda model. Obtain their qpos/dof/actuator
addresses by joint names, excluding fingers and the ball free joint; do not
assume the first 14 state entries are all arm joints. Boolean masks preserve
model order, not the order of the list used to construct them.

Keep CEM's existing `pick` to `move` phase gates and its velocity output rule
(the wrapper averages part of the planned horizon). Changing that rule to
execute only the first planned sample would be another algorithm variant.
3DFA orders its hands **left, right** and uses XYZW quaternions; CEM's MuJoCo
body poses use WXYZ. The two methods do not share action tensors.

### 4.2 Supply the goal numerically

The prepared scene deliberately has no `target_0` body. Make that body lookup
optional in both planner constructors and pass a numeric object goal instead:

```text
target_0 = [initial_ball_x, initial_ball_y, 0.95, 1, 0, 0, 0]
target_2 = [current_ball_x, current_ball_y, current_ball_z, qw, qx, qy, qz]
```

Use the existing `update_targets(target_idx=0, ...)` after construction. The
orientation entry follows the CEM pose format; its current ball-goal cost
uses XYZ. Keep the goal fixed throughout an episode and update `target_2`
from actual state. Use the same grey visualization overlay for both methods.
Exclude missing optional geometry names such as `wall_0` and `ball_pick`
instead of allowing `mj_name2id(...) == -1` into collision indexing.

### 4.3 Preserve physical dynamics

The Panda CEM code changes `model.opt.timestep` to its planning interval and
the native ROS runner writes arm velocities directly, zeroing all other
velocities. Those behaviors cannot be copied into the shared experiment:
they would change the ball's dynamics and fight the prepared position servos.

Retain the **2 ms physical timestep** and use the prepared arm actuators for
CEM execution. Integrate its commanded joint velocity into bounded joint
position references, then step the physical model. Reuse the reference
controller's joint limits, maximum joint speed and 0.1 rad reference-lead
bound. Hold fingers open for the original fixed-open CEM baseline and record
that choice; 3DFA retains its predicted gripper commands.

Keep CEM's original **0.1 s planning interval** and 15-step horizon initially.
One CEM interval must therefore cover **50 physical substeps**. Its internal
MJX transition must predict the same actuator integration and substeps used
by the actual simulator. A private prediction model must not modify the live
model; any MJX backend limitations or necessary model differences must be
tested and recorded before calling the rollout matched.

This changes the CEM transition/execution interface, so label the method
**Panda CEM with shared-scene servo execution**. It is not a claim of unchanged
native execution. Keep the optimizer and objective unchanged for the first
comparison; leave the native branch intact as a reference.

### 4.4 Start predictions from the complete current state

The original CEM rollout copies cached initial MJX data and replaces arm
positions/velocities and ball XYZ only. Pass the current simulation state
explicitly into each compiled rollout instead: qpos, qvel, actuator state,
ctrl, mocap state and other integration state used by the model. This includes
ball orientation/velocity and finger states. Do not rely on mutating a Python
attribute captured by `jax.jit` to refresh state.

Use measured arm velocities, and initialize CEM's references from current
actuator references. Check one candidate trajectory against the same commands
executed in MuJoCo before starting the evaluation. Planning must not advance
or mutate the live environment.

### 4.5 Reuse reset, recording and evaluation code

Use [initial_states.py](initial_states.py) and
`PandaBallEnvironment(..., initial_state=case)` for both methods. Reset CEM's
sampling key, distribution, projection warm starts and task phase per episode;
call `policy.reset("Lift the ball")` for 3DFA.

Call the same success/video callback after every physical substep. Reuse
[video.py](video.py) for the four views and goal marker. Feed fresh observations
to 3DFA after its executed keyposes; CEM reads current simulator state at each
planning call. Run the methods in separate processes so their environments,
random generators and GPU allocations stay independent.

Keep original CEM grasp offsets, inward orientations and weights in the first
baseline. They were chosen for another ball size and base arrangement; poor
transfer may therefore be informative. If you tune them, use separate
development starts and report a second, explicitly tuned baseline.

## 5. Pair the initial states and define the score

For your requested seeds **0 through 100 inclusive**, use exactly the existing
selection rule: create `SourceEpisodeSampler(archive, seed=reset_seed)` and
take its first sample for each run. Apply that same returned initial state to
both methods. Sampling is with replacement: 101 seeds are not 101 distinct
source episodes. Save the selected episode and full initial state in each
report, and verify both resets with `verify_initial_state`.

For a small development check, use source episodes 3, 8 and 94 explicitly.
If those are used to tune CEM, disclose overlap when reporting the historical
0–100 sweep; use separate source episodes for a held-out comparison.

Use these shared evaluation rules:

| Item | Rule |
|---|---|
| Physics | Same scene, solver settings, contacts, ball and actuators |
| Randomization | Same recorded ball pose, arm joints and gripper state; no extra posture noise |
| Seeds | Fixed policy seed 0 and CEM seed 0, independent of reset selection |
| Primary success | Ball centre reaches world z >= 0.95 m at any physical substep |
| Additional measurements | Final height, maximum height, longest continuous interval above 0.95 m |
| Budget | Choose one simulation-time limit for both methods before the paired sweep; suggested starting budget: 30 s |
| Timing | Record planning/inference, rendering and execution wall time separately; separate JAX compilation/warm-up |
| Failure accounting | Distinguish task failure, numerical/planner error and incomplete run |

The common simulation-time cap needs to be enforced in the proposed paired
runner for both controllers, including during a long 3DFA keypose execution.
The existing 3DFA `--steps 25` option is not a 30-second limit. For the primary
comparison, stop both methods at the first height crossing or the common
time limit; keep historical 3DFA runs labelled with their original
action-completion termination behavior.

A threshold crossing does not establish stable holding. A separate hold test
requires continuing physics under a defined controller for both methods;
the existing `longest_goal_duration_s` field alone does not add that test.
Likewise, do not use CEM's original distance-to-target `< 0.04 m` check as the
success metric for one method and the height threshold for the other.

CEM uses privileged simulator state and a dynamics model; 3DFA uses rendered
observations plus TCP history, with simulator state used by its source-visual
bridge. Describe those information differences with the results.

## 6. Existing 3DFA sweep command and videos

This runs the **existing 3DFA side only**, with its existing stopping rules.
It can reproduce the earlier experiment while the CEM adapter is prepared:

```bash
cd /home/aks-lab/3dfa/3d_flowmatch_actor
PYTHONPATH= bash scripts/mujoco/sweep.sh \
  --start-seed 0 --end-seed 100 --policy-seed 0 \
  --steps 25 --video-view all \
  --observation-renderer source_mujoco \
  --output runtime/ball_cem_vs_3dfa/3dfa_seeds_000_100
```

Add `--dry-run` to preview without inference. To resume, repeat exactly the
same command with `--resume`. Changed arguments or implementation require a
new output directory. This avoids the existing-manifest error.

Results are under:

```text
runtime/ball_cem_vs_3dfa/3dfa_seeds_000_100/
  summary.json
  summary.csv
  seed_000/attempt_001/report.json
  seed_000/attempt_001/rollout.mp4
  ...
  seed_100/attempt_001/rollout.mp4
```

Read the running totals with:

```bash
PYTHONPATH= .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("runtime/ball_cem_vs_3dfa/3dfa_seeds_000_100/summary.json")
s = json.loads(p.read_text())
for key in ("completed", "successes", "task_failures", "execution_errors",
            "pending", "success_rate", "source_episode_coverage"):
    print(f"{key}: {s[key]}")
PY
```

For the eventual paired runner, save one report/video per method per reset,
plus a paired table with reset seed, source episode, method, success, heights,
duration, timing and error status. Verify matching initial states and scene
fingerprints before comparing rates. Report successes/valid runs and execution
errors separately; also show total attempted runs so failures are visible.

## 7. Implementation order with minimal changes

1. Reproduce one existing 3DFA rollout and inspect its four-view video.
2. Read the Panda CEM ref and reuse its planner; do not port the UR5e branch.
3. Add only the CEM goal/state/servo interface described above.
4. Validate one prediction against execution, then run both methods on the
   three development starts with matching resets and recording.
5. Freeze settings and run the paired seeds 0–100 evaluation.

Relevant existing code: [run.py](run.py) for policy execution,
[policy.py](policy.py) for the unchanged checkpoint,
[environment.py](environment.py) for scene/reset/actuation/success,
[source_visual_renderer.py](source_visual_renderer.py) for observations,
[verify_settings.py](verify_settings.py) for checks, and [video.py](video.py)
for recordings. [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md) traces the
network call in detail.

Documentation validation: the prepared scene verification passed, both CLI
help pages and the 101-command sweep dry run were checked, shell/Python
examples and local links were validated, and the Panda CEM launch/API were
inspected from the pinned branch. A matched CEM rollout has not been
implemented or run. CUDA was
not visible to this documentation session; use the user's CUDA-enabled
terminal for the inference commands above.
