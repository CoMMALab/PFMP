# FR3 Push-T Experiments

Real-robot push-T manipulation using PFMP on a Franka FR3. Perception is
OptiTrack/Motive motion capture; the planner and the robot run as separate
processes talking over ZeroMQ (`pub_sub/`).

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

## Setup

Build the IK cache once (lives at `fr3/ik_cache.pkl`):

```bash
python fr3/ik_cache.py
python fr3/ik_cache.py --validate   # check an existing cache
```

## Running

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

### Motive setup (required, one-time per session)

In Motive's Data Streaming pane:
- **Local Interface**: the network adapter Motive is actually bound to — must
  be reachable from whichever machine runs the client scripts.
- **Transmission Type: Unicast**. The client code (`push_tee_optitrack.py`,
  `tee_problem_gen.py`, `view_start_region.py`) hardcodes `use_multicast=False`;
  leaving Motive on Multicast connects fine (command handshake succeeds) but
  silently drops all frame data.
- **Up Axis: Z-Axis**. Mismatched up-axis doesn't break the connection — poses
  just come out wrong (e.g. the tracked marker appears below the table).

## Notes

- Biased-MPPI / Biased-CEM (the custom-proposal + mean/elite-update baselines
  used in the Push-T sim sweep) are not yet wired into `push_tee_planner.py`.
- `wf.npz` is unused by the current pipeline (leftover from an earlier
  camera-calibration approach, superseded by OptiTrack).
