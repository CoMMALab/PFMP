"""sim_push_tee_2d.py — headless batched MuJoCo push-T sim for calibration (and later, tuning).

Reuses push_tee_planner's exact dynamics / cost / proposal / IK so the sim and the
hardware pipeline share one definition and cannot drift. The first and only job of
this cut is calibration: does sim-SIR reproduce the real-world SIR success rate on
the same (seed, trial) problems? Everything else (the method matrix, tuning, the
run_batch refactor) waits until that number is trustworthy.

    python fr3/sim_push_tee_2d.py --num 20 --replans 60            # full sir calibration
    python fr3/sim_push_tee_2d.py --num 2 --replans 3 --particles 64   # smoke test
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import mjx

import pfmp
import push_tee_planner as _P   # module handle so --no-orbital can zero ORBITAL_MIX
from push_tee_planner import (
    dynamics, running_cost, terminal_cost, init_control, proposal, mppi_proposal,
    _project_reachable, PushParams, load_scene, SCENE_XML_PATH, T_QPOS_OFFSET,
    HORIZON, NUM_KNOTS, PARTICLES, RUN_INTERACTIVE_FREQUENCY,
    SUCCESS_POS_RADIUS, SUCCESS_ANGLE_TOL,
)
from pfmp.algorithm import ps_update, mppi_update, cem_update
from ik_cache import get_or_build_cache, ik_from_cache
from tee_problem_gen import sample_problem as gen_problem

# Arm EE start on the pushing plane, matching realtime's --init-xy default.
EE_START = np.array([0.4, 0.0])

# method -> (proposal, resampling algorithm, update rule) — the three SMC axes.
# proposal: custom object-centric vs gaussian (which controller);
# algorithm: "sir" resamples (rho>0), "sis" does not;
# update:    ps (MAP) / mppi (softmax mean) / cem (elite mean) / dial (annealed mppi).
# method -> (default proposal, resampling algorithm, update rule) — the three SMC
# axes. proposal is one of gaussian / object / object_orbital and can be overridden
# with --proposal (used to ablate the orbital branch of the object-centric proposal).
METHODS = {
    "sir":         ("object_orbital", "sir", "ps"),   # SMC: custom proposal + resample
    "sis":         ("object_orbital", "sis", "ps"),   # sir minus resampling (isolates rho)
    "ps":          ("gaussian",       "sis", "ps"),
    "mppi":        ("gaussian",       "sis", "mppi"),
    "biased_mppi": ("object_orbital", "sis", "mppi"),
    "cem":         ("gaussian",       "sis", "cem"),
    "biased_cem":  ("object_orbital", "sis", "cem"),
    "dial":        ("gaussian",       "sis", "dial"),
}


def make_update(name, temperature, num_elites, particles):
    """Single-particle-batch update matching the realtime dispatch. Kept pure-JAX
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


def make_model(dt):
    mj_model, mj_data = load_scene(SCENE_XML_PATH)
    mj_model.opt.timestep = dt
    return mj_model, mj_data


_ORBITAL_DEFAULT = _P.ORBITAL_MIX   # full object-centric proposal's orbital weight


def build(proposal_kind, particles, ess_thresh, plan_dt):
    # Planning-side model + controller. Rollouts use this dt, so keep it coarse:
    # the 2048-particle x horizon rollout dominates runtime. Execution physics
    # (run_batch's mjx.step) can be finer, decoupled — see --exec-dt.
    #
    # proposal_kind selects q: gaussian (mppi_proposal) vs the object-centric
    # proposal with (object_orbital) or without (object) its orbital branch. The
    # orbital weight is a module global read when the proposal is traced, so set it
    # before building the controller.
    if proposal_kind == "object":
        _P.ORBITAL_MIX = 0.0
    else:
        _P.ORBITAL_MIX = _ORBITAL_DEFAULT
    mj_model, mj_data = make_model(plan_dt)
    mjx_model = mjx.put_model(mj_model)
    ik_cache = get_or_build_cache()
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    prop = mppi_proposal if proposal_kind == "gaussian" else proposal
    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init_control, prop,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=plan_dt, ess_resample_thresh=ess_thresh, interp="zoh",
    )
    return mj_model, mj_data, mjx_model, ik_cache, ctrl, gripper_site_id


def make_theta(mjx_model, ik_cache, gripper_site_id, init_xy, goal_xy, goal_yaw):
    # Initial knots ramp from the EE start to the tee, as realtime seeds theta.
    alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)[:, None]
    mean = (1.0 - alphas) * jnp.asarray(EE_START) + alphas * jnp.asarray(init_xy)
    return PushParams(
        mean=mean, mjx_model=mjx_model, ik_cache=ik_cache,
        goal_xy=jnp.asarray(goal_xy, dtype=jnp.float32),
        goal_angle=float(goal_yaw),
        pos_weight=15.0, angle_weight=0.3, stdev=0.025,
        gripper_site_id=gripper_site_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="sir", choices=list(METHODS))
    ap.add_argument("--temperature", type=float, default=1.0, help="MPPI/DIAL temperature")
    ap.add_argument("--num-elites", type=int, default=None, help="CEM elite count (default particles//10)")
    ap.add_argument("--proposal", default=None, choices=["gaussian", "object", "object_orbital"],
                    help="override the method's proposal (object = custom w/o orbital branch)")
    ap.add_argument("--num", type=int, default=20, help="parallel trials (= problem count)")
    ap.add_argument("--replans", type=int, default=60, help="replan steps per trial")
    ap.add_argument("--particles", type=int, default=PARTICLES)
    ap.add_argument("--ess-thresh", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-dt", type=float, default=0.02, help="planning rollout dt (coarse=fast, dominates runtime)")
    ap.add_argument("--exec-dt", type=float, default=None, help="execution physics dt (fine=stable contact); default = plan-dt")
    ap.add_argument("--sim-dt", type=float, default=None, help="set BOTH plan and exec dt (equal)")
    args = ap.parse_args()

    plan_dt = args.sim_dt if args.sim_dt is not None else args.plan_dt
    exec_dt = args.sim_dt if args.sim_dt is not None else (args.exec_dt if args.exec_dt is not None else plan_dt)

    default_prop, algorithm, update_name = METHODS[args.method]
    proposal_kind = args.proposal if args.proposal is not None else default_prop
    num_elites = args.num_elites if args.num_elites is not None else max(1, args.particles // 10)
    update_fn = make_update(update_name, args.temperature, num_elites, args.particles)

    # Planning controller/theta at plan_dt; execution model at exec_dt (finer
    # contact). Same scene, only the timestep differs — decoupled so fine contact
    # tracking doesn't 4x the expensive planning rollout.
    mj_model, mj_data, mjx_model, ik_cache, ctrl, gsid = build(
        proposal_kind, args.particles, args.ess_thresh, plan_dt)
    mj_exec, _ = make_model(exec_dt)
    print(f"nu={mj_model.nu}  method={args.method}  proposal={proposal_kind}  "
          f"resample={'yes' if algorithm=='sir' else 'no'}  update={update_name}  "
          f"plan_dt={plan_dt}  exec_dt={exec_dt}  particles={args.particles}")
    assert mj_model.nu == 7, f"expected 7 arm actuators, got nu={mj_model.nu}"

    q0 = np.asarray(ik_from_cache(jnp.asarray(EE_START), ik_cache))

    # Deterministic (seed, trial) problems; recomputed here for success evaluation.
    problems = [gen_problem(args.seed, i, ik_cache) for i in range(args.num)]

    def sample_problem(i):
        init_xy, init_yaw, goal_xy, goal_yaw = problems[i]
        d = mujoco.MjData(mj_exec)
        d.qpos[:7] = q0
        d.ctrl[:7] = q0
        d.qpos[T_QPOS_OFFSET]     = float(init_xy[0])
        d.qpos[T_QPOS_OFFSET + 1] = float(init_xy[1])
        d.qpos[T_QPOS_OFFSET + 2] = float(init_yaw)
        mujoco.mj_forward(mj_exec, d)
        return d, make_theta(mjx_model, ik_cache, gsid, init_xy, goal_xy, goal_yaw)

    def ik_interp(u, theta):
        # Faithful execution mapping (matches run_realtime): snap the executed
        # waypoint into the reachable annulus, THEN IK. MAP-update methods already
        # emit reachable waypoints (no-op for them); mean-update methods emit
        # unreachable averages that would otherwise NaN the IK — this makes them
        # fail cleanly (snapped to reachable) exactly as they would on hardware.
        u = _project_reachable(u, theta)
        return ik_from_cache(u, theta.ik_cache)

    base_d, base_th = sample_problem(0)
    sspr = max(1, int((1.0 / RUN_INTERACTIVE_FREQUENCY) / exec_dt))
    max_step = args.replans * sspr

    _wall0 = time.time()
    traj = pfmp.run_batch(
        ctrl, base_th, mj_exec, base_d,
        frequency=RUN_INTERACTIVE_FREQUENCY,
        algorithm=algorithm, max_step=max_step, num=args.num,
        ik_interp=ik_interp, sample_problem=sample_problem,
        temperature=args.temperature, update_fn=update_fn,
    )
    wall = time.time() - _wall0
    print(f"\nwall-clock: {wall:.1f}s for {args.num} trials (incl. one-time JIT compile). "
          f"Re-runs at the same shapes skip compile — budget tuning by the 'Simulation completed in' time.")

    qpos = np.asarray(traj.qpos)   # (num, T, nq)
    dt = float(exec_dt)   # qpos timesteps are execution steps
    T = qpos.shape[1]
    n_succ = n_nan = n_fail = clipped = 0
    times = []
    for i in range(args.num):
        _, _, goal_xy, goal_yaw = problems[i]
        tee = qpos[i, :, T_QPOS_OFFSET:T_QPOS_OFFSET + 3]
        dist = np.linalg.norm(tee[:, :2] - np.asarray(goal_xy), axis=1)
        ang = np.abs(((tee[:, 2] - float(goal_yaw) + np.pi) % (2 * np.pi)) - np.pi)
        valid = ~np.isnan(dist) & ~np.isnan(ang)
        reached = valid & (dist < SUCCESS_POS_RADIUS) & (ang < SUCCESS_ANGLE_TOL)
        has_nan = bool(np.any(~valid))
        md = float(np.nanmin(dist)) if np.any(valid) else float("nan")
        if np.any(reached):
            # Reached at a finite timestep — counts even if the sim later blew up.
            n_succ += 1
            k = int(np.argmax(reached)); times.append(k * dt)
            tag = f"OK    t_goal={k*dt:5.1f}s"
        elif has_nan:
            # Contact instability (state -> NaN) before reaching: sim contamination,
            # not a planner failure. Excluded from the clean rate, counted separately.
            n_nan += 1
            tag = "NAN   (contact blew up — contamination)"
        else:
            n_fail += 1
            k = int(np.nanargmin(dist))
            frac = k / (T - 1)
            is_clip = frac > 0.75 and md < 0.08
            clipped += int(is_clip)
            tag = f"FAIL  {'(clipped?)' if is_clip else '(stuck)   '} best@{frac*100:3.0f}%"
        print(f"  t{i:2d} {tag}  min_dist={md:.3f}m")

    clean = n_succ + n_fail
    raw_rate = n_succ / args.num
    clean_rate = (n_succ / clean) if clean else 0.0
    # Self-contained summary (the driver greps '^sim '): clean rate first (NaNs
    # excluded as contamination), with raw rate and NaN count alongside.
    print(f"\nsim {args.method}: clean {n_succ}/{clean}={clean_rate*100:.0f}%  "
          f"raw {n_succ}/{args.num}={raw_rate*100:.0f}%  nan={n_nan}  "
          f"proposal={proposal_kind} update={update_name} resample={'yes' if algorithm=='sir' else 'no'}")
    if times:
        print(f"  time-to-goal: mean={np.mean(times):.1f}s  median={np.median(times):.1f}s")
    print(f"  breakdown: {n_succ} success | {n_fail} stuck ({clipped} clipped?) | {n_nan} nan-unstable")
    print("  real-SIR reference: ~17/20 clean (85%), duration mean ~59s")


if __name__ == "__main__":
    main()
