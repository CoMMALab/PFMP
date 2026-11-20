# FR3 Experiments

Two tasks share this directory: **push-T** (below) and **cube grasp-and-lift**
(see [Cube grasp](#cube-grasp)). They do not share kinematics: push-T uses the
precomputed `ik_cache.py` grid on a fixed-height plane, while the grasp task
uses go_fr3's analytic cylindrical IK, which it needs because grasping moves in
z.

## Files

| File | Purpose |
|------|---------|
| `push_tee_planner.py` | Main entry point. Sim mode (`ps`/`sir`/`mppi`/`cem`/`dial` subcommands) runs one trial in an interactive MuJoCo viewer; `realtime` closes the loop on hardware. |
| `push_tee_controller.py` | Runs near the robot; executes XY waypoint commands via `franky`. |
| `push_tee_optitrack.py` | Streams T-block/robot pose from Motive. |
| `pub_sub/` | ZeroMQ planner↔controller protocol (`protocol.py`, `planner.py`, `controller.py`). |
| `geom_tee.py` | T-block geometry, SDF, and Panda forward kinematics. |
| `ik_cache.py` | Build/validate the precomputed IK lookup grid. |
| `ik_inspector.py` | Viser tool for debugging IK cache solutions. |
| `tee_interface.py` | Interactive click-to-move tool for manual data collection. |
| `sysid.py` | Scripted EE sweep comparing real vs. MuJoCo T-block response. |
| `view_proposal.py` | Visualize proposal-distribution samples at a clicked EE position. |
| `view_start_region.py` | Live view of tee pose, workspace bounds, and sampling region. |
| `mbm.py` | MotionBenchMaker 7-DoF planning benchmark — unrelated to push-T, reuses the Panda geometry code. |
| `grasp.py` | Cube grasp-and-lift task definition (dynamics, cost, proposal). See below. |
| `sim_grasp.py` | Headless batched grasp benchmark, the analogue of `push_tee_simulation.py`. |
| `go_fr3/` | Analytic FR3 IK in cylindrical `(theta, r, z, q7)` coordinates, plus a standalone CLI for scripted moves. |

## Push-T

Real-robot push-T manipulation using PFMP on a Franka FR3. Perception is
OptiTrack/Motive motion capture; the planner and the robot run as separate
processes talking over ZeroMQ (`pub_sub/`).

### Setup

Build the IK cache once (lives at `fr3/ik_cache.pkl`):

```bash
python fr3/ik_cache.py
python fr3/ik_cache.py --validate   # check an existing cache
```

### Running

**Sim (no robot)** — opens an interactive MuJoCo viewer for one trial:

```bash
python fr3/push_tee_planner.py sir --trial 0
python fr3/push_tee_planner.py mppi --trial 0 --temperature 1.0
```

Subcommands: `ps`, `sir`, `mppi`, `cem`, `dial` (see `--help` on each for its
own flags, e.g. `cem --num-elites`, `dial --beta1 --beta2 --n-inner`).

**Real robot** — start the controller near the arm, then the planner:

```bash
python fr3/push_tee_controller.py --ip <robot-ip>
python fr3/push_tee_planner.py realtime \
    --server <motive-ip> --local <this-machine-ip> \
    --server-host <controller-host> --method sir
```

`--method` accepts `sir` (PFMP), `ps`, `mppi`, `cem`, `dial`. Goal is set via
`--goal-x --goal-y --goal-yaw`.

#### Motive setup (required, one-time per session)

In Motive's Data Streaming pane:
- **Local Interface**: the network adapter Motive is actually bound to — must
  be reachable from whichever machine runs the client scripts.
- **Transmission Type: Unicast**. The client code (`push_tee_optitrack.py`,
  `tee_problem_gen.py`, `view_start_region.py`) hardcodes `use_multicast=False`;
  leaving Motive on Multicast connects fine (command handshake succeeds) but
  silently drops all frame data.
- **Up Axis: Z-Axis**. Mismatched up-axis doesn't break the connection — poses
  just come out wrong (e.g. the tracked marker appears below the table).

### Notes

- Biased-MPPI / Biased-CEM (the custom-proposal + mean/elite-update baselines
  used in the Push-T sim sweep) are not yet wired into `push_tee_planner.py`.
- `wf.npz` is unused by the current pipeline (leftover from an earlier
  camera-calibration approach, superseded by OptiTrack).


## Cube grasp

Pick a 45 mm cube off the table and lift it 50 cm, by sampling control
sequences. `grasp.py` holds the task and `sim_grasp.py` runs it batched; the
split is the same one push-T uses between planner and simulator, so the two
cannot drift.

```bash
python fr3/sim_grasp.py --method sir --num 20            # full run
python fr3/sim_grasp.py --num 2 --replans 8 --particles 64   # smoke test
python fr3/sim_grasp.py --method mppi --proposal gaussian    # ablate the q axis
```

### Watching it

`sim_grasp.py` is headless. `--video` replays the run's own recorded states
through an offscreen renderer, so it costs no extra rollout and needs no viewer
window (it works over ssh):

```bash
python fr3/sim_grasp.py --method sir --num 4 --video           # -> out/grasp.mp4
python fr3/sim_grasp.py --method sir --video out/sir.mp4 --video-camera closeup
python fr3/sim_grasp.py --method mppi --video out/mppi.mp4 --video-trial 0
```

It renders the first successful trial by default, since a video of a failure is
usually not what you want; `--video-trial N` overrides that to inspect one.
The scene carries two fixed cameras, `overview` (table plus the full 0.5 m
lift) and `closeup` (tight on the grasp). Neither tracks the cube, so the cube
leaving frame means it actually left.

Needs `ffmpeg` on PATH. There is no live-viewer path for this task yet;
`pfmp.run_interactive` is what push-T uses and would be the place to add one.

**Control space.** A knot is `u = (theta, r, z, grip)`: go_fr3's cylindrical
chart plus the commanded finger opening. Box bounds on `(theta, r, z)` are
very nearly the arm's reachable set, so sampled particles land on feasible
poses instead of being thrown away. The wrist angle `q7` is not sampled, it is
solved to point the jaws along the cube's yaw.

**Kinematics.** `go_fr3/ik.py` dispatches on its inputs' array namespace, so
the same source runs under numpy for the CLI and traces under `jax.jit` and
`jax.vmap` inside a rollout. There is no second JAX copy to drift. Traced code
calls `solve_clamped`, which returns `(q, violation)` instead of raising, since
a sampler proposes unreachable knots constantly and an exception cannot cross a
trace; `running_cost` charges for `violation`.

**What makes it work.** The gripper is sampled like every other dimension, so
nothing scripts the open/close. The structure lives in the proposal:
`phase_anchor` reads whether the hand is approaching, descending, closing or
holding, and anchors that knot's draw on the matching cube-relative pose, at a
width matched to that phase's tolerance. This is the object-centric-proposal
axis, the same role push-T's orbital branch plays. The width is set by the
cube's geometry rather than by taste: the hand opens to 40 mm on a 22.5 mm
half-cube, leaving **17.5 mm of lateral clearance**, so a lateral sampling sigma
anywhere near the approach phase's width misses far more often than it hits.
Sizing `STDEV` by eye instead is what made the first version of this task fail
outright, with zero particles out of 128 ever grasping.

**Warm start.** `initial_mean` is deliberately neutral: a straight ramp from
the hand to a pose above the cube, gripper open at every knot. It carries the
same knowledge push-T's warm start does ("move toward the object") and no
more. Seeding it with a scripted grasp would hand the uninformed gaussian
baselines the answer and make the method comparison meaningless.

**Models.** `models/mjx_cube_grasp.xml` -> `mjx_grasp_scene.xml` ->
`mjx_panda_hand.xml`. The last is the push-T Panda with the fingers restored
(`mjx_panda.xml` has them commented out and carries a push rod instead) and a
tendon-coupled gripper actuator, so `nu = 8`.

**Compile time.** The nested scan/vmap over a contact-rich scene takes minutes
to compile. `sim_grasp.py` enables XLA's persistent cache at
`~/.cache/pfmp-jax` (`PFMP_CACHE_DIR` to relocate, `""` to disable), so re-runs
at the same shapes skip it.
