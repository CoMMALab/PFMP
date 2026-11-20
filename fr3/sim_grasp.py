"""sim_grasp.py — headless batched MuJoCo cube grasp-and-lift sim for the FR3.

Reuses fr3/grasp.py's dynamics / cost / proposal and go_fr3's analytic IK, so
the task has one definition and the sim cannot drift from it. Mirrors
fr3/push_tee_simulation.py: the same three SMC axes (proposal q, update U,
resampling rho), the same METHODS table, the same success-rate reporting.

    python fr3/sim_grasp.py --method sir --num 20            # full sir run
    python fr3/sim_grasp.py --num 2 --replans 8 --particles 64   # smoke test
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# Compiling the nested scan/vmap (replans -> trials -> particles -> intervals ->
# substeps) over a contact-rich scene takes minutes, and every re-run pays it
# again unless XLA can cache the result on disk. Shapes key the cache, so
# re-running at the same --num/--particles/--replans is near-instant while
# changing them recompiles. Set PFMP_CACHE_DIR to relocate it, or "" to disable.
_CACHE = os.environ.get("PFMP_CACHE_DIR", os.path.expanduser("~/.cache/pfmp-jax"))
if _CACHE:
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import pfmp
from grasp import (
    dynamics, running_cost, terminal_cost, init_control, proposal,
    gaussian_proposal, clip_control, initial_mean, load_scene, GraspParams,
    CUBE_QPOS_OFFSET, CUBE_HALF, TARGET_Z, LIFT_HEIGHT, SUCCESS_LIFT_TOL,
    HORIZON, NUM_KNOTS, PARTICLES, RUN_INTERACTIVE_FREQUENCY, STDEV,
)
from pfmp.algorithm import ps_update, mppi_update, cem_update

# method -> (proposal, resampling algorithm, update rule) — the three SMC axes.
# proposal: grasp-aware object-centric vs uninformed gaussian (which controller);
# algorithm: "sir" resamples (rho>0), "sis" does not;
# update:    ps (MAP) / mppi (softmax mean) / cem (elite mean) / dial (annealed mppi).
METHODS = {
    "sir":         ("grasp",    "sir", "ps"),   # SMC: custom proposal + resample
    "sis":         ("grasp",    "sis", "ps"),   # sir minus resampling (isolates rho)
    "ps":          ("gaussian", "sis", "ps"),
    "mppi":        ("gaussian", "sis", "mppi"),
    "biased_mppi": ("grasp",    "sis", "mppi"),
    "cem":         ("gaussian", "sis", "cem"),
    "dial":        ("gaussian", "sis", "dial"),
}


def make_update(name, temperature, num_elites, particles):
    """Single-particle-batch update matching push-T's dispatch. Kept pure-JAX
    (no float() on traced values) so it runs inside run_batch's jitted scan."""
    if name == "ps":
        return ps_update
    if name == "mppi":
        def _u(t, tr, c):
            finite = jnp.isfinite(jnp.sum(c, axis=1))
            c = jnp.where(finite[:, None], c, jnp.zeros_like(c))
            return mppi_update(t, tr, c, max(temperature, 1e-6))
        return _u
    if name == "cem":
        ef = num_elites / particles
        return lambda t, tr, c: cem_update(t, tr, c, elite_frac=ef)
    if name == "dial":
        def _u(t, tr, c):
            th = t
            for _ in range(5):                       # annealed MPPI passes
                temp = jnp.maximum(jnp.std(jnp.sum(c, axis=1)), 1e-6)
                th, _w = mppi_update(th, tr, c, temp)
            return th, jnp.zeros(particles)
        return _u
    raise ValueError(name)


def require_gpu(allow_cpu: bool):
    """Fail fast if JAX has fallen back to CPU.

    JAX falls back silently: an interrupted run can leave the CUDA context wedged
    (`cuInit` returns CUDA_ERROR_UNKNOWN while nvidia-smi still looks healthy),
    and the next run then grinds on CPU for half an hour instead of taking one
    minute. Better to stop and say so. `sudo rmmod nvidia_uvm && sudo modprobe
    nvidia_uvm` clears that state without a reboot.
    """
    kinds = {d.platform for d in jax.devices()}
    if kinds == {"cpu"} and not allow_cpu:
        raise SystemExit(
            "sim_grasp: JAX is on CPU, which will take orders of magnitude longer "
            "than the GPU path this script is sized for.\n"
            "  devices: " + repr(jax.devices()) + "\n"
            "  if a run was interrupted, the CUDA context may be wedged:\n"
            "      sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\n"
            "  check with: python -c 'import jax; print(jax.devices())'\n"
            "  pass --allow-cpu to run anyway.")


def make_model(dt):
    mj_model, mj_data = load_scene()
    mj_model.opt.timestep = dt
    return mj_model, mj_data


def build(proposal_kind, particles, ess_thresh, plan_dt):
    """Planning-side model + controller.

    Rollouts use plan_dt, so keep it coarse: the particles x horizon rollout
    dominates runtime. Execution physics (run_batch's stepping) can be finer and
    is decoupled, see --exec-dt.

    proposal_kind selects q: the grasp-aware object-centric mixture, or the
    uninformed gaussian the PS/MPPI/CEM/DIAL baselines use.
    """
    mj_model, mj_data = make_model(plan_dt)
    mjx_model = mjx.put_model(mj_model)
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    prop = proposal if proposal_kind == "grasp" else gaussian_proposal
    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init_control, prop,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=plan_dt, ess_resample_thresh=ess_thresh, interp="zoh",
    )
    return mj_model, mj_data, mjx_model, ctrl, gripper_site_id


def sample_cube_pose(seed: int, trial: int):
    """Deterministic (seed, trial) cube placement on the reachable table band.

    Radius and bearing are drawn inside the arm's comfortable shell, well short
    of the U_MAX corner, so a failure is the planner's and not the workspace's.
    Yaw spans a quarter turn, which is the cube's full symmetry group.
    """
    rng = np.random.default_rng((seed, trial))
    r = rng.uniform(0.38, 0.58)
    theta = rng.uniform(-0.45, 0.45)
    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    return float(r * np.cos(theta)), float(r * np.sin(theta)), float(yaw)


def make_theta(mjx_model, gripper_site_id, stdev):
    return GraspParams(
        mean=jnp.zeros((NUM_KNOTS, 4)), mjx_model=mjx_model,
        stdev=stdev, gripper_site_id=gripper_site_id,
    )


def write_video(mj_model, qpos, out, camera="overview", fps=30, width=960, height=720):
    """Replay one trial's qpos through an offscreen renderer into an mp4.

    Same approach as experiments/mjx_video.py: the batched driver already
    recorded every state, so this replays them rather than re-running anything,
    and it needs no viewer window (it works over ssh and cannot steal focus).
    """
    cam = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam < 0:
        raise SystemExit(f"no camera named {camera!r} in the scene; "
                         f"have {[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mj_model.ncam)]}")
    # Render every `stride`-th state, then play back at the rate that stride
    # actually represents. Writing the requested fps instead would run the video
    # fast or slow by the rounding error (at dt=0.02 and 30 fps, 1.2x fast).
    stride = max(1, int(round(1.0 / (mj_model.opt.timestep * fps))))
    out_fps = 1.0 / (mj_model.opt.timestep * stride)
    frames = qpos[::stride]

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{out_fps:.4f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
        stdin=subprocess.PIPE)
    d = mujoco.MjData(mj_model)
    for q in frames:
        if np.any(np.isnan(q)):      # a blown-up trial: stop at the last good frame
            break
        d.qpos[:] = q
        mujoco.mj_forward(mj_model, d)
        renderer.update_scene(d, camera=cam)
        p.stdin.write(renderer.render().tobytes())
    p.stdin.close()
    p.wait()
    renderer.close()
    print(f"  wrote {out}  ({len(frames)} frames @ {out_fps:.1f} fps = real time, "
          f"camera={camera})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="sir", choices=list(METHODS))
    ap.add_argument("--temperature", type=float, default=1.0, help="MPPI/DIAL temperature")
    ap.add_argument("--num-elites", type=int, default=None, help="CEM elite count (default particles//10)")
    ap.add_argument("--proposal", default=None, choices=["gaussian", "grasp"],
                    help="override the method's proposal (ablates the q axis)")
    ap.add_argument("--num", type=int, default=20, help="parallel trials (= problem count)")
    ap.add_argument("--replans", type=int, default=12, help="replan steps per trial")
    ap.add_argument("--particles", type=int, default=PARTICLES)
    ap.add_argument("--ess-thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-dt", type=float, default=0.02, help="planning rollout dt (coarse=fast, dominates runtime)")
    ap.add_argument("--exec-dt", type=float, default=0.005,
                    help="execution physics dt. Defaults to 0.005, NOT to plan-dt: "
                         "at 0.02 a held cube creeps out from between the closed "
                         "fingers under the arm's motion (~59 mm over 8 s) and "
                         "eventually falls, which silently corrupts the benchmark. "
                         "0.005 converges (0.002 gives the same answer) and only "
                         "costs execution steps, not the planning rollout.")
    ap.add_argument("--sim-dt", type=float, default=None, help="set BOTH plan and exec dt (equal)")
    ap.add_argument("--video", nargs="?", const="out/grasp.mp4", default=None,
                    help="render a trial to mp4 (default out/grasp.mp4); replays the "
                         "run's own states, so it costs no extra rollout")
    ap.add_argument("--video-trial", type=int, default=None,
                    help="which trial to render (default: the first success, else trial 0)")
    ap.add_argument("--video-camera", default="overview", choices=["overview", "closeup"])
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even if JAX has no GPU (very slow; see require_gpu)")
    ap.add_argument("--save-traj", default=None,
                    help="npz of the run's ctrl/cost/qpos/qvel, for diagnosing "
                         "what the planner actually commanded")
    args = ap.parse_args()

    plan_dt = args.sim_dt if args.sim_dt is not None else args.plan_dt
    exec_dt = args.sim_dt if args.sim_dt is not None else args.exec_dt

    require_gpu(args.allow_cpu)

    default_prop, algorithm, update_name = METHODS[args.method]
    proposal_kind = args.proposal if args.proposal is not None else default_prop
    num_elites = args.num_elites if args.num_elites is not None else max(1, args.particles // 10)
    update_fn = make_update(update_name, args.temperature, num_elites, args.particles)

    mj_model, mj_data, mjx_model, ctrl, gsid = build(
        proposal_kind, args.particles, args.ess_thresh, plan_dt)
    mj_exec, _ = make_model(exec_dt)
    print(f"nu={mj_model.nu}  method={args.method}  proposal={proposal_kind}  "
          f"resample={'yes' if algorithm=='sir' else 'no'}  update={update_name}  "
          f"plan_dt={plan_dt}  exec_dt={exec_dt}  particles={args.particles}")
    assert mj_model.nu == 8, f"expected 7 arm actuators + 1 gripper, got nu={mj_model.nu}"

    problems = [sample_cube_pose(args.seed, i) for i in range(args.num)]

    def sample_problem(i):
        cx, cy, yaw = problems[i]
        d = mujoco.MjData(mj_exec)
        mujoco.mj_resetDataKeyframe(mj_exec, d, 0)
        d.qpos[CUBE_QPOS_OFFSET]     = cx
        d.qpos[CUBE_QPOS_OFFSET + 1] = cy
        d.qpos[CUBE_QPOS_OFFSET + 2] = CUBE_HALF
        d.qpos[CUBE_QPOS_OFFSET + 3] = float(np.cos(yaw / 2))
        d.qpos[CUBE_QPOS_OFFSET + 4] = 0.0
        d.qpos[CUBE_QPOS_OFFSET + 5] = 0.0
        d.qpos[CUBE_QPOS_OFFSET + 6] = float(np.sin(yaw / 2))
        mujoco.mj_forward(mj_exec, d)

        th = make_theta(mjx_model, gsid, STDEV)
        # Warm start from this trial's own cube pose, as realtime would.
        th = th._replace(mean=initial_mean(mjx.put_data(mj_exec, d), th))
        return d, th

    base_d, base_th = sample_problem(0)
    sspr = max(1, int((1.0 / RUN_INTERACTIVE_FREQUENCY) / exec_dt))
    max_step = args.replans * sspr

    _wall0 = time.time()
    traj = pfmp.run_batch(
        ctrl, base_th, mj_exec, base_d,
        frequency=RUN_INTERACTIVE_FREQUENCY,
        algorithm=algorithm, max_step=max_step, num=args.num,
        # run_batch only tests this for None. A non-None ik_interp selects the
        # dynamics-driven executor, which ramps the joint reference toward the
        # knot's IK solution rather than writing the knot straight into ctrl,
        # matching both the planner's rollout and the velocity-limited robot.
        # The callable is not invoked at present; this is the knot-to-ctrl map
        # the executor would use if it were.
        ik_interp=clip_control,
        sample_problem=sample_problem,
        temperature=args.temperature, update_fn=update_fn,
    )
    wall = time.time() - _wall0
    print(f"\nwall-clock: {wall:.1f}s for {args.num} trials (incl. one-time JIT compile). "
          f"Re-runs at the same shapes skip compile.")

    if args.save_traj:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_traj)) or ".", exist_ok=True)
        traj.save(args.save_traj)
        print(f"  saved trajectory to {args.save_traj}")

    qpos = np.asarray(traj.qpos)   # (num, T, nq)
    dt = float(exec_dt)
    T = qpos.shape[1]
    n_succ = n_nan = n_fail = n_near = n_dropped = 0
    times, successes = [], []
    for i in range(args.num):
        cube_z = qpos[i, :, CUBE_QPOS_OFFSET + 2]
        valid = ~np.isnan(cube_z)
        lift = cube_z - CUBE_HALF                      # metres off the table
        reached = valid & (np.abs(cube_z - TARGET_Z) < SUCCESS_LIFT_TOL)
        # Touching the target once is not a grasp: the cube can creep out of the
        # jaws and fall while the trial still scores on `any(reached)`. Require it
        # to still be up over the last half second.
        tail = max(1, int(0.5 / dt))
        held = bool(np.all(reached[-tail:]))
        has_nan = bool(np.any(~valid))
        best = float(np.nanmax(lift)) if np.any(valid) else float("nan")
        if np.any(reached) and held:
            n_succ += 1
            successes.append(i)
            k = int(np.argmax(reached)); times.append(k * dt)
            tag = f"OK    t_lift={k*dt:5.1f}s"
        elif np.any(reached):
            n_dropped += 1
            k = int(np.argmax(reached))
            tag = f"DROP  lifted at {k*dt:4.1f}s, not held at the end"
        elif has_nan:
            # Contact instability (state -> NaN) before lifting: sim contamination,
            # not a planner failure. Excluded from the clean rate, counted separately.
            n_nan += 1
            tag = "NAN   (contact blew up — contamination)"
        else:
            n_fail += 1
            is_near = best > 0.5 * LIFT_HEIGHT
            n_near += int(is_near)
            tag = f"FAIL  {'(lifted, short)' if is_near else '(never grasped)'}"
        print(f"  t{i:2d} {tag}  best_lift={best:.3f}m")

    clean = n_succ + n_fail + n_dropped
    raw_rate = n_succ / args.num
    clean_rate = (n_succ / clean) if clean else 0.0
    print(f"\nsim {args.method}: clean {n_succ}/{clean}={clean_rate*100:.0f}%  "
          f"raw {n_succ}/{args.num}={raw_rate*100:.0f}%  nan={n_nan}  "
          f"proposal={proposal_kind} update={update_name} resample={'yes' if algorithm=='sir' else 'no'}")
    if times:
        print(f"  time-to-lift: mean={np.mean(times):.1f}s  median={np.median(times):.1f}s")
    print(f"  breakdown: {n_succ} success | {n_dropped} lifted-then-dropped | "
          f"{n_fail} failed ({n_near} lifted but short) | {n_nan} nan-unstable")

    if args.video:
        # Default to a trial that actually worked, since a video of a failure is
        # rarely what you want; --video-trial overrides to inspect a failure.
        trial = args.video_trial if args.video_trial is not None else (
            successes[0] if successes else 0)
        print(f"\nrendering trial {trial} ({'success' if trial in successes else 'failure'})…")
        write_video(mj_exec, qpos[trial], args.video, camera=args.video_camera,
                    fps=args.video_fps)


if __name__ == "__main__":
    main()
