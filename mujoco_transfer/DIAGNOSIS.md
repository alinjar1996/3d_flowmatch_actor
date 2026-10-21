# Why the pretrained policy did not lift in the MuJoCo sweep

The robot **can lift the ball in the current MuJoCo setup**. The strongest
identified problem is the policy's transferred visual input: the unchanged
checkpoint predicts a substantially different, unsuitable approach from the
MuJoCo images. The local Cartesian controller introduces additional failures
when executing those predictions. Matching robot bases, initial ball pose and
the success threshold was insufficient to establish a working policy transfer.

This investigation preserves the original sweep, checkpoint, prepared scene,
observation adapter and controller. The original learned-policy result remains
**0/101**. The successful experiments below are explicitly labeled expert
controls; native-camera experiments measure first predictions, not complete
learned-policy rollouts.

## Evidence that separates the causes

| Controlled experiment | Result | What it establishes |
|---|---|---|
| Original policy sweep, reset seeds 0–100 | 0/101; all stop with `controller_timeout`; 56 distinct source episodes | Current transfer fails; repeated reset samples are not independent source scenes |
| Released expert joint trajectories through current physical MuJoCo actuators | 6/6 lift; maximum heights 1.138–1.164 m | Current embodiment and contact dynamics permit lifting |
| Exact source expert Cartesian keyposes through the unchanged controller | 6/6 lift; all 24 keyposes reached | The existing Cartesian action interface can perform this task with suitable targets |
| Same checkpoint and identical random noise; native source RGB+XYZ instead of MuJoCo RGB+XYZ | Mean first-target error versus expert approach falls **20.91 → 1.87 cm**, orientation error **41.0° → 3.73°**, across four episodes | Observation differences are responsible for most of the tested first-action discrepancy |
| Execute those four native-input first predictions in MuJoCo | 4/4 reached within the original 8-second limit | Their approach targets are executable with the current controller |
| Increase timeout to 24 seconds in a full policy rollout of source episode 94 | First action reaches, second action fails and ball falls | Increasing the timeout alone does not repair the transfer |

Expert controls use source episodes 3, 85, 94, 87, 8 and 72. Native-input
comparisons use episodes 3, 8, 87 and 94. Expert-keypose episodes 3 and 8 later
lose the ball; success here follows the original condition, any ball-centre
height `z >= 0.95 m`, rather than sustained holding. Dense expert joint replay
ends above the threshold in all six cases. Native first-action execution only
tests approaching the ball and must not be counted as a lifting success.

[Expert control details and commands](../runtime/diagnosis/physics/AUDIT.md),
[input comparison scores](../runtime/diagnosis/policy_parity/prediction_vs_expert.json),
[native-input action execution](../runtime/diagnosis/execution/native_summary.json),
[longer-timeout policy report](../runtime/diagnosis/long_timeout_episode094/report.json).

## The main tested cause: visual observations

With source-rendered observations, the checkpoint predicts the demonstrated
side approach: hands about 0.29 m to either side of the ball along Y and at
world height 0.80 m, below the ball centre. With MuJoCo observations it predicts
hands much nearer the ball horizontally and at heights 0.846–0.924 m. Those
motions can press into the ball from above instead of positioning both hands
for the scoop.

Across the sweep's 56 distinct source episodes, ball X spans 30.72 cm, but the
first predicted right-hand X spans just 6.22 cm. Its correlation with ball X is
−0.098, versus 0.815 for the expert first keypose. Both source grippers remain
open in every frame of all 100 released ball demonstrations. The transferred
policy nevertheless requests 42 individual close actions across 292
predictions. This task uses two open hands to lift the large ball; lack of a
close command is not the cause.

The source comparisons reconstruct the native CoppeliaSim scene from the
pinned source assets and archived arm/finger joint coordinates, full ball pose
and camera metadata. No physics settling was run for these re-renders, so
small differences from recorded dynamic robot geometry may remain. Camera
world poses were set explicitly from metadata. The archive contains
low-dimensional observations, not original RGB
frames. These are native re-renders, not recovered training images. The
checkpoint revision does not identify the exact RLBench training revision.
The MuJoCo prediction was reproduced **bit for bit** before making the input
interventions, using the same checkpoint and restored random-number-generator
state for each prediction.

The more specific interventions are:

| First-action inputs | Mean position error against source expert approach | Episodes |
|---|---:|---:|
| MuJoCo RGB + XYZ + robot state | 20.91 cm | 4 |
| Native RGB + XYZ + robot state | 1.87 cm | 4 |
| Native RGB + XYZ, MuJoCo robot state | 1.87 cm | 4 |
| MuJoCo RGB + XYZ, native robot state | 21.65 cm | 4 |
| MuJoCo RGB + XYZ, matched two-episode subset | 22.70 cm | 2 |
| Native RGB, MuJoCo XYZ | 4.19 cm | 2 |
| MuJoCo RGB, native XYZ | 23.76 cm | 2 |
| MuJoCo rendering at exact source camera world poses | 22.03 cm | 2 |

The mixed RGB/XYZ pairs are diagnostic interventions: pixel geometry is not
perfectly registered between the two renderers, so they are not valid
deployment observations. Together with the fully matched native RGB+XYZ
result, they point to **RGB appearance as the dominant tested input factor**.
They do not isolate a single texture or lighting parameter.

The wrist images also have severe extra occlusion. Segmentation identifies
the MuJoCo palm meshes, not target markers or collision proxies, in much of
the lower image. In episode 3, those meshes cover approximately 58% and 97% of
the bottom 96 rows in the left and right views. Right-wrist visible ball pixels,
estimated using point-cloud membership on the ball surface, fall from 8,195 in
the native render to 693 in MuJoCo. Forcing exact source
camera poses still leaves the extra occlusion. **Hiding the palm meshes alone
does not restore the native predictions**: appearance differences elsewhere
remain important.

![Native source above; MuJoCo below. Front, left wrist, right wrist.](../runtime/diagnosis/policy_parity/input_comparison_ep094.png)

[Prediction error chart](../runtime/diagnosis/policy_parity/prediction_vs_expert.png),
[camera and segmentation measurements](../runtime/diagnosis/policy_parity/camera_audit.json),
[same-noise predictions](../runtime/diagnosis/policy_parity/predictions.json),
[input audit and reproduction commands](../runtime/diagnosis/policy_parity/AUDIT.md).

## Additional execution and embodiment problems

The left arm fails its final tolerance in 98/101 runs: 75 left-only timeouts,
23 with both arms outside tolerance, and 3 right-only. Eighty runs stop within
the first two predictions. The local controller attempts to move directly
toward Cartesian targets; it does not construct a planned approach or choose
another IK branch after becoming stuck.

Deterministic action replays reproduce the original controller results and
separate several mechanisms:

| Sweep seed / source episode | Baseline failure | Diagnostic intervention |
|---|---|---|
| 0 / 85, first prediction | Left position error 10.58 cm; hand/ball sampled normal force reaches about 150 N | Removing robot contacts gives 3.86 mm error; longer timeout reaches its first tolerance crossing at 9.34 s |
| 7 / 94, first prediction | Left error 12.49 cm; hand presses into ball | Removing robot contacts gives 4.35 mm error; first tolerance crossing with longer timeout is 10.88 s |
| 32 / 87, prediction 21 | Hands collide; sampled palm/palm normal force about 140 N | Removing robot contacts reaches both targets; even 24 seconds with contacts still times out |
| 42 / 8, prediction 5 | Left error 4.37 cm with joints 5 and 6 at their upper limits | Neither removing contacts nor 24 seconds resolves it; multistart IK finds alternative valid configurations for both arms |

Removing contacts is only a mechanism test and is **not a lifting solution**.
The multistart IK result establishes reachable endpoint configurations, not a
collision-free path from the current pose. At that late seed-42 state, the
ball has already fallen away.

There are further measured limitations:

- The branch's joint-4-to-joint-5 geometry differs from the source robot.
  Matching bases and fitting a fixed TCP cannot remove all configuration
  dependence. Initial policy TCP discrepancies are roughly 2.5–3.2 cm left
  and 2.77 cm right. Wrist camera errors at the audited reset are about
  20.8/25.1 mm and 4.36/4.44 degrees. Correcting these remains desirable,
  although the first-action ablations show they do not explain the dominant
  visual discrepancy by themselves.
- Both hands and fingers can collide with the ball. The branch's finger
  capsule proxies differ from the native source shapes. The positive controls
  prove lifting is possible, but do not establish equivalence of contacts.
- Ball/floor collisions are disabled by the inherited collision masks. A ball
  pushed off the table therefore continues falling through the floor. This
  explains the extreme negative heights in some reports; it is secondary to
  the failed lift.
- `CartesianController.execute()` retains `reached=True` from before its
  0.4-second gripper phase, without checking the final tolerances again. In the
  longer seed-0 replay it returns true with final position error 18.6 mm,
  exceeding the 10 mm tolerance. This reporting issue should be corrected
  alongside controller changes; it does not explain the original 0/101 result.

Arm order, absolute world XYZ, quaternion XYZW, RGB scaling/CLIP normalization,
three-step history padding, camera ordering, instruction and five flow steps
match the local upstream evaluator. The strict checkpoint load reproduces the
saved predictions. No evidence was found that swapping arms, changing
quaternion order or increasing the timeout alone would repair the transfer.

## Videos and reproduction

- [Source expert keyposes, source episode 94, four views](../runtime/diagnosis/physics/expert_keyposes_episode094.mp4): unchanged Cartesian controller; final ball height **1.20335 m**. Labeled **NOT POLICY**.
- [Expert joint replay, source episode 3, four views](../runtime/diagnosis/physics/expert_episode003.mp4): final ball height **1.15647 m**. Labeled **NOT POLICY**.
- [Actual checkpoint rollout with longer timeout, source episode 94](../runtime/diagnosis/long_timeout_episode094/rollout.mp4): failure persists after the first approach.
- [Original policy sweep seed 7](../runtime/seed_sweeps/seeds_000_100/seed_007/attempt_001/rollout.mp4): first-action timeout.

Run from the repository root:

```bash
# Replays saved actions, without running inference or editing the sweep.
PYTHONPATH= OMP_NUM_THREADS=2 .venv/bin/python \
  scripts/mujoco/diagnose_execution.py --seeds 0 7 42 32

# Six positive controls through the existing Cartesian controller.
PYTHONPATH= OMP_NUM_THREADS=1 .venv/bin/python \
  runtime/diagnosis/physics/expert_keypose.py 3,85,94,87,8,72

# Recompute complete sweep statistics from the original reports.
PYTHONPATH= .venv/bin/python runtime/diagnosis/sweep_statistics.py
```

Diagnostic scripts and data are under `runtime/diagnosis/`; they are local
artifacts excluded from Git. Native renderer reproduction also requires the
downloaded CoppeliaSim/PyRep/source assets; see the input audit linked above.
The repository execution diagnostic uses only the existing MuJoCo
environment and saved sweep reports. No commit or push was performed.

## What should be corrected before another large sweep

First establish source-like observations in MuJoCo and measure the checkpoint
against the same source first-keypose controls. Match visible geometry and
camera visibility, then appearance, and verify RGB and XYZ together. Camera
pose corrections or hiding a palm alone are not supported as sufficient fixes
by these experiments. If rendering parity cannot be achieved while retaining
the desired embodiment, fine-tuning on MuJoCo demonstrations is a separate
route; these tests do not measure how much training would be required.

Then improve keypose execution with planned approach paths or alternate IK
configurations, failure recovery and final-tolerance checks. Repair dropped
ball handling and floor contact so a failed grasp terminates cleanly. Validate
on a small fixed set with videos, keeping expert controls separate from policy
successes, before repeating the 101-seed evaluation.
