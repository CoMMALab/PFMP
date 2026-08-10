"""Batched runner for the NamedTuple-defined tasks (geometric, push-T, MBM).

Mirrors `mjx_runner.py` (yaml config -> per-algorithm results in an outdir) but
targets tasks that are plain JAX functions over a NamedTuple `theta`, rather
than MuJoCo. That avoids materialising an `mjx.Data` per particle, which is the
dominant memory cost in `run_batch` and buys nothing for these tasks.

One task is selected by a flag; everything else is shared.

Usage:
    python experiments/geom_runner.py experiments/configs/mbm.yaml out/mbm
"""

import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from pfmp import PFMP
from pfmp.obstacle_grid import sdf_aabb, hinge_loss, generate_map_batch
from pfmp.algorithm import ps_update, mppi_update, cem_update, sample_update
from pfmp.update import dial_step
from pfmp.interpolate import hermite_spline


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The shared perturbation proposal
#
# Every baseline (PS/MPPI/CEM/DIAL) samples the same way: Gaussian noise around
# the nominal plan `theta.mean`, at a width read per-knot. That is one function,
# not one per task, so it lives here. A task only supplies its action clip.
#
# The three `theta` types agree on exactly what this needs: `mean` (K, *action),
# `stdev` (scalar, or (K, *action) once CEM refits it), and `step_len`.
# ---------------------------------------------------------------------------

def sigma_at(theta, idx):
    """Sampling noise for knot `idx`.

    Handles both conventions: a scalar `stdev` (PS/MPPI/SIS/SIR, and DIAL, which
    overwrites it per annealing level) and a per-knot array (CEM, which refits it
    from the elites each replan). `jnp.ndim` is static, so this is free under jit.
    """
    return theta.stdev if jnp.ndim(theta.stdev) == 0 else theta.stdev[idx]


def knot_index(theta, t):
    """Knot whose interval contains time `t`."""
    return jnp.clip(jnp.round(t / theta.step_len).astype(jnp.int32) + 1,
                    0, theta.mean.shape[0] - 1)


def make_perturbation(clip=None):
    """Build the (init, proposal) pair for a task, given its action clip."""
    clip = clip or (lambda u, theta: u)

    def at(key, theta, idx):
        u = theta.mean[idx] + jax.random.normal(
            key, theta.mean.shape[1:]) * sigma_at(theta, idx)
        return clip(u, theta)

    return (lambda key, x, theta: at(key, theta, 0),
            lambda key, x, u, theta, t: at(key, theta, knot_index(theta, t)))

class GeomTask(NamedTuple):
    """What a task must provide to be runnable here.

    The first five are the PFMP controller's own arguments. `sample_problem`
    generates problem i as (x0, theta). `execute`/`shift` are the outer MPC loop:
    open-loop planning (MBM) leaves both as None, closed-loop control (push-T)
    supplies them. `measure` is the task's metric, applied to the final state.
    """
    dynamics: Callable            # (x, u, theta, t) -> (x_next, aux)
    running_cost: Callable        # (x, u, theta, t) -> float
    terminal_cost: Callable       # (x, theta) -> float
    sample_problem: Callable      # (i) -> (x0, theta)
    measure: Callable             # (x_final, theta) -> pytree of scalars
    horizon: float
    num_knots: int
    dt: float
    # The object-informed proposal, if the task has one. Optional: 2D geometric
    # has no second proposal to contrast against, so its SMC methods fall back
    # to the shared perturbation and differ from the baselines only in the
    # update and the resampling.
    custom_init: Callable = None      # (key, x, theta) -> u
    custom_proposal: Callable = None  # (key, x, u, theta, t) -> u
    clip_action: Callable = None      # (u, theta) -> u   bounds for the shared
                                      #   perturbation proposal
    n_replans: int = 1            # 1 = open-loop planning; >1 = closed-loop MPC
    execute: Callable = None      # (x, theta) -> x        None = world is static
    shift: Callable = None        # (theta) -> theta       None = no re-timing
    warm_start: Callable = None   # (x0, theta) -> theta   baselines' nominal plan
    hp_defaults: dict = {}        # method knobs the task module already defines.
                                  #   Sourced from the module, never retyped
                                  #   here: a literal default in the runner is
                                  #   how a tuned value silently stops applying.
    apply_hp: Callable = None     # (theta, hp) -> theta   tuned knobs that live
                                  #   on theta (proposal widths, noise scale)
                                  #   rather than in the update rule


# Which proposal / update / resampling each algorithm name selects. This IS the
# decomposition: everything else in this file is plumbing.
def parse_algorithm(name):
    """Split a method name into (update, rho).

    Resampling is an axis, not a method, and rho is a value on it, so it lives
    in the name: `mppi+r0.25` is MPPI resampling at 0.25 and several thresholds
    can appear in one sweep. A bare `+r` means "whatever ess_thresh resolves to"
    (task default or tuned), returned as None so the caller fills it in. No
    suffix is rho = 0, which is exactly `ess_resample_thresh=0` (never resample).

    `sir` is kept as the legacy alias for `sis+r`.
    """
    if name == "sir":
        return "sis", None
    if "+r" in name:
        base, _, val = name.partition("+r")
        return base, (float(val) if val else None)
    return name, 0.0


CUSTOM_PROPOSAL = ("sis", "sir", "sis_sample", "biased_mppi", "biased_cem")
ARGMIN_UPDATE   = ("sis", "sir", "ps")
ALGORITHMS = ("sis", "sir", "sis_sample", "ps", "mppi", "cem",
              "biased_mppi", "biased_cem", "dial")
# ...each of which may take a "+r[rho]" suffix: "mppi+r", "cem+r0.25".


def build_controller(task: GeomTask, algorithm: str, particles: int,
                     ess_thresh: float = 0.0) -> PFMP:
    base, rho = parse_algorithm(algorithm)
    perturb = make_perturbation(task.clip_action)
    init, prop = ((task.custom_init, task.custom_proposal)
                  if base in CUSTOM_PROPOSAL and task.custom_proposal is not None
                  else perturb)
    return PFMP(task.dynamics, task.running_cost, task.terminal_cost, init, prop,
                horizon=task.horizon, num_knots=task.num_knots, particles=particles,
                ess_resample_thresh=ess_thresh if rho is None else rho,
                dt=task.dt)


# Continuous knobs passed as traced arguments rather than baked in as Python
# constants. A tuner varies these every trial, and closing over them means a
# fresh compile per trial: on push-T that was 5.8s of compile against 0.6s of
# compute. The rest (elite count, ess_thresh, DIAL's schedule) are structural or
# discrete and stay static.
TRACED_HP = ("mppi_temperature", "cem_stdev_floor", "sample_temperature")


def traced_hp(hp: dict):
    return {k: jnp.asarray(float(hp.get(k, 0.0))) for k in TRACED_HP}


def make_replan(ctrl: PFMP, algorithm: str, hp: dict, particles: int):
    """One planning iteration: (key, x, theta) -> theta.

    DIAL is the one algorithm that owns its own sampling loop, so it replaces
    this whole function rather than plugging into the update slot.
    """
    algorithm, rho = parse_algorithm(algorithm)
    sample = ctrl.run_sis if rho == 0.0 else ctrl.run_sir

    if algorithm == "dial":
        n = int(hp.get("dial_n_inner", 2))
        return lambda key, x, theta, s: dial_step(
            key, ctrl, x, theta, n=n,
            beta1=hp.get("dial_beta1", 1.0), beta2=hp.get("dial_beta2", 1.0),
            temp_sample=hp.get("dial_temp_sample", 1.0))

    if algorithm in ARGMIN_UPDATE:
        # The MAP update. On a task with no custom proposal this makes `sis`
        # and `ps` the same algorithm; `sis_sample` is what separates them.
        update = lambda th, tr, co, s: ps_update(th, tr, co)
    elif algorithm == "sis_sample":
        # Posterior sampling (Piche et al.): draw one trajectory from the
        # weighted particle set rather than summarising it. Needs a key, so it
        # is the one update that consumes randomness of its own.
        #
        # Shares MPPI's temperature by default, and it is a fixed absolute
        # value, not one rescaled per iteration by the return spread. Both
        # updates softmax the same returns and differ only in how they reduce
        # them (weighted mean vs a draw), so a shared fixed temperature is what
        # makes them a controlled comparison on the update axis. The loop is
        # open here, so the cost scale does not move between iterations and
        # there is nothing a per-iteration rescale would be tracking.
        key_ = "sample_temperature" if "sample_temperature" in hp else "mppi_temperature"
        update = lambda th, tr, co, k, s: sample_update(
            th, tr, co, k, jnp.maximum(s[key_], 1e-6))
    elif algorithm in ("mppi", "biased_mppi"):
        update = lambda th, tr, co, s: mppi_update(
            th, tr, co, jnp.maximum(s["mppi_temperature"], 1e-6))
    elif algorithm in ("cem", "biased_cem"):
        # The elite count sets an array shape, so it must stay static.
        k = max(1, int(hp.get("cem_elite_frac", 0.1) * particles))
        update = lambda th, tr, co, s: cem_update(
            th, tr, co, k / particles, stdev_floor=s["cem_stdev_floor"])
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    needs_key = algorithm == "sis_sample"

    def replan(key, x, theta, scalars):
        # Only split when the update itself needs randomness. Splitting
        # unconditionally would shift the sampling stream for every other
        # method, silently changing results that are meant to be comparable.
        sub, rest = (jax.random.split(key) if needs_key else (key, None))
        trajs, costs = sample(sub, x, theta)
        args = (theta, trajs, costs) + ((rest,) if needs_key else ())
        return update(*args, scalars)[0]
    return replan


def make_simulator(task: GeomTask, replan: Callable):
    """jit(vmap(...)) of the outer MPC loop over a batch of problems.

        for _ in range(n_replans):
            theta = replan(key, x, theta)
            x     = execute(x, theta)     # world advances (closed-loop only)
            theta = shift(theta)          # plan re-times for the next replan
        return measure(x, theta)
    """
    execute = task.execute or (lambda x, theta: x)
    shift = task.shift or (lambda theta: theta)

    def one(key, x0, theta, scalars):
        # Warm-start unconditionally, including for the methods whose proposal
        # ignores `theta.mean`: it is what gives `mean` its shape, and the scan
        # carry's pytree structure has to survive the first update writing it.
        if task.warm_start is not None:
            theta = task.warm_start(x0, theta)

        def step(carry, _):
            key, x, theta = carry
            key, sub = jax.random.split(key)
            theta = replan(sub, x, theta, scalars)
            x = execute(x, theta)
            theta = shift(theta)
            return (key, x, theta), None

        (_, x, theta), _ = jax.lax.scan(
            step, (key, x0, theta), None, length=task.n_replans)
        return task.measure(x, theta)

    # scalars are shared across the batch, not per problem
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))


_SIM_CACHE = {}


def _simulator(task, algorithm, particles, hp):
    """Compiled simulator, reused across calls that differ only in the traced
    knobs. Keyed on everything that is baked in at trace time, so a tuner
    sweeping a temperature compiles once instead of once per trial."""
    static = tuple(sorted((k, v) for k, v in hp.items() if k not in TRACED_HP
                          and isinstance(v, (int, float, bool))))
    key = (id(task), algorithm, particles, static)
    if key not in _SIM_CACHE:
        ctrl = build_controller(task, algorithm, particles,
                                ess_thresh=float(hp.get("ess_thresh", 0.0)))
        _SIM_CACHE[key] = make_simulator(
            task, make_replan(ctrl, algorithm, hp, particles))
    return _SIM_CACHE[key]


def run_algorithm(task: GeomTask, algorithm: str, num: int, particles: int,
                  hp: dict, seed: int = 0, group_size: int = None):
    """Run `num` problems. Batched in groups because some tasks (MBM) have
    per-problem array shapes that only match within a dataset."""
    group_size = group_size or num
    sim = _simulator(task, algorithm, particles, hp)
    scalars = traced_hp(hp)

    out = []
    for lo in range(0, num, group_size):
        idx = range(lo, min(lo + group_size, num))
        probs = [task.sample_problem(i) for i in idx]
        x0 = jnp.stack([p[0] for p in probs])
        thetas = [p[1] for p in probs]
        if task.apply_hp is not None:
            thetas = [task.apply_hp(th, hp) for th in thetas]
        theta = jax.tree.map(lambda *xs: jnp.stack(xs), *thetas)
        keys = jnp.stack([jax.random.key(seed + i) for i in idx])
        out.append(jax.tree.map(np.asarray, sim(keys, x0, theta, scalars)))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *out) if len(out) > 1 else out[0]


# ---------------------------------------------------------------------------
# Task registry. One adapter per task; `load_task` is the flag swap.
# ---------------------------------------------------------------------------

def _mbm_task(config):
    """MBM: open-loop 7-DoF configuration-space planning. n_replans=1, the world
    never advances, and `measure` rolls the plan out to score it."""
    import mbm
    
    robot = mbm.load_robot(config.get("urdf", "models/panda_spherized.urdf"))
    datasets = config["datasets"]
    per_ds = config.get("problems_per_dataset", 100)
    off = int(config.get("problem_offset", 0))
    problems, groups = [], []
    for ds in datasets:
        ps = mbm.load_mbm_dataset(ds, max_problems=off + per_ds)[off:]
        groups.append(len(ps))
        problems.extend(ps)
    # every group is the same size, so a single group_size batches without padding
    assert len(set(groups)) == 1, f"datasets differ in size: {groups}"

    dt = 0.01                                  # PFMP's default; MBM never overrides
    tq = jnp.linspace(0, mbm.HORIZON, int(mbm.HORIZON / dt))
    knot_t = jnp.linspace(0, mbm.HORIZON, mbm.NUM_KNOTS)

    def sample_problem(i):
        prob = problems[i]
        return prob["q_start"], mbm.Params(
            goal=prob["q_goal"], env=prob["env"],
            q_min=mbm.Q_MIN, q_max=mbm.Q_MAX, robot=robot)

    def apply_hp(theta, hp):
        # The baselines' one knob is the perturbation width; SMC's are the
        # proposal mixture. Both live on theta, so the tuner and the runner set
        # them the same way. Absent keys keep the module default.
        return theta._replace(**{k: float(hp[k]) for k in
                                 ("goal_frac", "gauss_frac", "gauss_std") if k in hp},
                              **({"stdev": float(hp["stdev"])} if "stdev" in hp else {}))

    def warm_start(x0, theta):
        a = jnp.linspace(0.0, 1.0, mbm.NUM_KNOTS)[:, None]
        return theta._replace(mean=(1 - a) * x0 + a * theta.goal)

    def measure(x, theta):
        us = hermite_spline(tq, knot_t, theta.mean)
        _, path = jax.lax.scan(
            lambda q, u: (mbm.dynamics(q, u, theta, 0.0)[0], q), x, us)
        collided = jax.vmap(
            lambda q: mbm.robot_sdf(q, theta.env, theta.robot) < 0)(path).any()
        dist = jnp.linalg.norm(path[-1] - theta.goal)
        return {"collided": collided, "dist": dist,
                "success": (~collided) & (dist < 0.5)}

    task = GeomTask(
        dynamics=mbm.dynamics, running_cost=mbm.cost_fn,
        terminal_cost=mbm.terminal_cost,
        custom_init=mbm.init, custom_proposal=mbm.proposal,
        clip_action=lambda u, th: jnp.clip(u, th.q_min, th.q_max),
        sample_problem=sample_problem, measure=measure,
        horizon=mbm.HORIZON, num_knots=mbm.NUM_KNOTS, dt=dt,
        n_replans=1, execute=None, shift=None, warm_start=warm_start,
        apply_hp=apply_hp,
        # Inlined rather than read off mbm's module constants: these are this
        # runner's defaults, and importing them would let a value change under
        # the numbers reported here without the runner saying so. Tuned JSON
        # overrides all of them.
        hp_defaults={"stdev": 0.395, "mppi_temperature": 1.0,
                     "cem_elite_frac": 0.1, "ess_thresh": 0.552,
                     "goal_frac": 0.339, "gauss_frac": 0.333,
                     "gauss_std": 0.395},
    )
    return task, groups[0]      # batch per dataset: env shapes match only within

def _pushT_task(config):
    """Push-T: closed-loop planar pushing. The world advances between replans,
    so `execute` runs the physics and `shift` re-times the plan onto the new
    clock, which is exactly what MBM leaves as None."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    import importlib
    pt = importlib.import_module("pushT")
    
    off = int(config.get("problem_offset", 0))
    seeds = list(range(config.get("seed", 0) + off,
                       config.get("seed", 0) + off + config["num"]))
    distance = config.get("distance")
    n_obs = config.get("n_obs")
    corridor = config.get("corridor", False)
    built = [pt._build_scenario(sd, distance, n_obs, corridor) for sd in seeds]
    # Scenarios batch together only if their obstacle arrays agree in shape;
    # `_random_obstacles_arena` can place fewer than requested.
    shapes = {tuple(b[1].obstacles.shape) for b in built}
    assert len(shapes) == 1, f"scenarios differ in obstacle count: {shapes}"
    n_replans = built[0][4]

    tk0 = jnp.linspace(0.0, pt.HORIZON, pt.NUM_KNOTS)
    tq = jnp.arange(pt.STEPS_PER_REPLAN) * pt.DT

    def execute(x, theta):
        us = hermite_spline(tq, tk0, theta.mean)
        x, _ = jax.lax.scan(
            lambda x, a: (pt.dynamics(x, a[0], theta, a[1])[0],) * 2, x, (us, tq))
        return x

    def shift(theta):
        # Re-time the plan onto the post-execution clock: resample the same
        # spline at knots advanced by one replan interval.
        return theta._replace(
            mean=hermite_spline(tk0 + pt.STEPS_PER_REPLAN * pt.DT, tk0, theta.mean))

    def warm_start(x0, theta):
        a = jnp.linspace(0.0, 1.0, pt.NUM_KNOTS)[:, None]
        return theta._replace(mean=(1 - a) * x0[6:8] + a * x0[0:2])

    def measure(x, theta):
        pos_err = jnp.linalg.norm(x[0:2] - theta.goal_xy)
        angle_err = jnp.abs(pt._angle_wrap(x[2] - theta.goal_angle))
        return {"pos_err": pos_err, "angle_err": angle_err,
                "success": (pos_err < pt.SUCCESS_POS_THRESH)
                           & (angle_err < pt.SUCCESS_ANGLE_THRESH)}

    def apply_hp(theta, hp):
        return theta._replace(stdev=float(hp["stdev"])) if "stdev" in hp else theta

    task = GeomTask(
        dynamics=pt.dynamics, running_cost=pt.running_cost,
        terminal_cost=pt.terminal_cost,
        custom_init=pt.init_control, custom_proposal=pt.proposal,
        clip_action=None,          # push-T clips inside the dynamics, not the plan
        sample_problem=lambda i: (built[i][0], built[i][1]), measure=measure,
        horizon=pt.HORIZON, num_knots=pt.NUM_KNOTS, dt=pt.DT,
        n_replans=n_replans, execute=execute, shift=shift,
        warm_start=warm_start, apply_hp=apply_hp,
        hp_defaults={"stdev": 0.05, "mppi_temperature": 0.5,
                     "cem_elite_frac": 10 / 2048, "cem_stdev_floor": 0.01,
                     "ess_thresh": 0.25,
                     "dial_n_inner": 2, "dial_beta1": 1.0, "dial_beta2": 1.0,
                     "dial_temp_sample": 0.06},
    )
    return task, len(seeds)


DRIVE_SEED = 127
DRIVE_PARTICLES = 1024
DRIVE_ESS_THRESH = 0.5
DRIVE_DT = 0.01
DRIVE_ARENA_WIDTH = 6.0
DRIVE_OBSTACLE_DENSITY = 0.3
DRIVE_BOX_SIZE_RANGE = (0.4, 1.2)
DRIVE_MIN_MARGIN = 0.3
DRIVE_WHEELBASE = 0.3
DRIVE_SPEED = 2.0
DRIVE_U_STDEV = 0.2
DRIVE_MPPI_TEMPERATURE = 1.0
DRIVE_CEM_ELITE_FRAC = 0.1
DRIVE_STEER_LIMIT = 0.8            # steering angle clip (rad)

def _drive_task(config):
    """Open-loop like MBM (no `execute`/`shift`), but iterated: the plan is
    refined over `n_replans` passes against a static world. No object-informed
    proposal exists here, so `custom_*` stay None and the SMC methods run the
    shared perturbation, differing from the baselines only in update and
    resampling."""
    class DriveParams(NamedTuple):
        goal: jax.Array       # desired position
        obstacles: jax.Array  # (N, 4) AABB array
        L: float              # wheelbase (m)
        v: float              # forward speed (m/s)
        mean: jax.Array       # (num_knots,) proposal mean
        # Sampling noise. Scalar for the fixed-noise methods; CEM refits it per knot,
        # giving shape (num_knots,). Always read through `_sigma_at`.
        stdev: jax.Array
        step_len: float       # horizon / (num_knots - 1)


    def drive_cost(x: jax.Array, u: jax.Array, th: DriveParams, t: float) -> float:
        loc = x[:2]
        sdf_val = jnp.min(jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles))
        obstacle_loss = jnp.nan_to_num(100. * hinge_loss(sdf_val, 0.25),
                                       nan=0.0, posinf=1e6, neginf=0.0)
        goal_loss = jnp.linalg.norm(loc - th.goal) / 10.
        return obstacle_loss**2 + goal_loss**2


    def drive_terminal(x, th) -> float:
        return 10 * jnp.linalg.norm(x[:2] - th.goal) ** 2


    def drive_dynamics(x: jax.Array, u: jax.Array, theta: DriveParams, t):
        """Bicycle kinematics: x = [px, py, heading], u = steering angle."""
        px, py, th = x
        L, v = theta.L, theta.v
        px_next = px + v * jnp.cos(th) * 0.01
        py_next = py + v * jnp.sin(th) * 0.01
        th_next = jnp.mod(th + (v / L) * jnp.tan(u) * 0.01 + jnp.pi, 2 * jnp.pi) - jnp.pi
        x_next = jnp.array([px_next, py_next, th_next])
        return (x_next, x_next)
    num_knots = int(config.get("num_knots", 20))
    arena_length = float(config.get("arena_length", 1.0 * num_knots + 1.0))
    horizon = num_knots * 0.5
    n = config["num"]

    off = int(config.get("problem_offset", 0))
    obs, x0s, goals, rejected = generate_map_batch(
        config.get("seed", DRIVE_SEED) + num_knots * 10000, off + n, arena_length,
        DRIVE_ARENA_WIDTH, DRIVE_OBSTACLE_DENSITY, DRIVE_BOX_SIZE_RANGE, DRIVE_MIN_MARGIN)
    obs, x0s, goals = obs[off:], x0s[off:], goals[off:]
    print(f"  generated {n} maps (rejected {rejected})")

    tk0 = jnp.linspace(0.0, horizon, num_knots)
    tq = jnp.arange(int(horizon / DRIVE_DT)) * DRIVE_DT

    def sample_problem(i):
        return x0s[i], DriveParams(
            goal=goals[i], obstacles=obs[i], L=DRIVE_WHEELBASE, v=DRIVE_SPEED,
            mean=jnp.zeros(num_knots), stdev=DRIVE_U_STDEV,
            step_len=horizon / (num_knots - 1))

    def measure(x, theta):
        """Cost of the final plan, plus feasibility and proximity separately.

        The scalar cost blends them, so a method can buy a shorter goal distance
        by driving through obstacles (CEM does exactly that here). Reporting
        `collided` and `goal_dist` apart keeps that visible.
        """
        us = hermite_spline(tq, tk0, theta.mean)

        def step(x, inp):
            u, t = inp
            xn, _ = drive_dynamics(x, u, theta, t)
            return xn, (drive_cost(xn, u, theta, t), xn)

        x_final, (running, path) = jax.lax.scan(step, x, (us, tq))
        sdf = jax.vmap(lambda p: jnp.min(jax.vmap(
            lambda b: sdf_aabb(p[:2], b))(theta.obstacles)))(path)
        return {"cost": jnp.sum(running) + drive_terminal(x_final, theta),
                "goal_dist": jnp.linalg.norm(x_final[:2] - theta.goal),
                "collided": jnp.any(sdf < 0.0),
                "collision_frac": jnp.mean(sdf < 0.0)}

    def apply_hp(theta, hp):
        return theta._replace(stdev=float(hp["stdev"])) if "stdev" in hp else theta

    task = GeomTask(
        dynamics=drive_dynamics, running_cost=drive_cost, terminal_cost=drive_terminal,
        clip_action=lambda u, th: jnp.clip(u, -DRIVE_STEER_LIMIT, DRIVE_STEER_LIMIT),
        sample_problem=sample_problem, measure=measure,
        horizon=horizon, num_knots=num_knots, dt=DRIVE_DT,
        # The legacy runner gave the baselines `num_knots` refinement passes but
        # SIS/SIR only one. Here every method gets the same budget.
        n_replans=int(config.get("n_replans", num_knots)),
        execute=None, shift=None, warm_start=None, apply_hp=apply_hp,
        hp_defaults={"stdev": DRIVE_U_STDEV,
                     "mppi_temperature": DRIVE_MPPI_TEMPERATURE,
                     "cem_elite_frac": DRIVE_CEM_ELITE_FRAC,
                     "ess_thresh": DRIVE_ESS_THRESH},
    )
    return task, n


_TASKS = {"mbm": _mbm_task, "pushT": _pushT_task, "drive": _drive_task}


def load_task(name, config):
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; have {sorted(_TASKS)}")
    return _TASKS[name](config)


def load_tuned(spec, algorithms):
    """Resolve `tuned_params` to {algorithm: {param: value}}.

    Two layouts exist in figures/: one file per method, named by an
    `{algorithm}` placeholder in the path (push-T), and a single file keyed by
    method (MBM). A missing file warns rather than raising, so a partially
    tuned sweep still runs with module defaults for the rest, but it says so.
    """
    import json
    if not spec:
        return {}
    out = {}
    for alg in algorithms:
        path = Path(spec.format(algorithm=alg)) if "{algorithm}" in spec else Path(spec)
        if not path.exists():
            print(f"  ! no tuned params for {alg} ({path}); using defaults")
            continue
        blob = json.load(open(path))
        entry = blob if "{algorithm}" in spec else blob.get(alg)
        if entry is None:
            print(f"  ! no tuned params for {alg} in {path}; using defaults")
            continue
        out[alg] = entry.get("best_params", entry)
    return out


def sweep_axis(config, key=None, values=None):
    """(key, values) for the sweep: the config's `sweep` block, or a CLI
    override. Returns (None, None) when the config declares no axis."""
    blk = config.get("sweep") or {}
    if key is not None:
        return key, values
    return blk.get("key"), blk.get("values")


def run_sweep(config, key, values, algorithms, cache_path):
    """Run every (swept value, algorithm) pair, caching raw per-problem values.

    The sweep axis is just a yaml key overridden across a list, so nothing here
    knows what is being swept. Resumable per cell: a pair already in the cache
    is skipped, so a sweep that dies partway (or gains a method later) does not
    re-run what it has. Values are cached raw, never aggregated, because a
    success rate needs its numerator and denominator and a cost needs the whole
    sample; `geom_analyze` decides how to reduce them.
    """
    import json
    cache_path = Path(cache_path)
    # The cache is self-describing: it stores the axis and the method list
    # beside the results, so plotting needs the results file and nothing else.
    blob = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    results = blob.get("results", {})
    axis = dict(config.get("sweep") or {}, key=key, values=values)
    axis.setdefault("name", f"fig_{config['task']}_{key}")

    # A cell is only reusable if it carries every metric the config now asks
    # for. Without this, adding a metric to `measure` would leave old cells in
    # place and silently plot a figure that is missing series.
    wanted = {p["metric"] for p in (axis.get("panels") or [])}
    wanted |= {axis["metric"]} if axis.get("metric") else set()

    def fresh(cell):
        return wanted <= set(cell)

    # The resolved hyperparameters travel with the results: a label like
    # "PS (rho=0.5)" has to state the value actually used, not a constant
    # retyped in the plotting code.
    used_hp = {}

    def flush():
        cache_path.write_text(json.dumps(
            {"task": config["task"], "sweep": axis, "algorithms": algorithms,
             "hyperparams": used_hp, "results": results}))

    for value in values:
        vkey = str(value)
        results.setdefault(vkey, {})
        cfg = dict(config, **{key: value})
        task = group_size = tuned = None

        for alg in algorithms:
            if alg in results[vkey] and fresh(results[vkey][alg]):
                print(f"  {key}={vkey} {alg:12s} cached")
                continue
            if task is None:                     # build lazily, once per value
                task, group_size = load_task(cfg["task"], cfg)
                tuned = load_tuned(cfg.get("tuned_params"), algorithms)
            hp = dict(task.hp_defaults)
            hp.update(cfg.get("hyperparams", {}))
            hp.update(tuned.get(alg, {}))
            res = run_algorithm(task, alg, num=cfg["num"],
                                particles=cfg["num_samples"], hp=hp,
                                seed=cfg.get("seed", 0), group_size=group_size)
            used_hp[alg] = {k: v for k, v in hp.items()
                            if isinstance(v, (int, float, bool))}
            _, rho = parse_algorithm(alg)
            used_hp[alg]["ess_thresh"] = (hp.get("ess_thresh", 0.0)
                                          if rho is None else rho)
            results[vkey][alg] = {k: np.asarray(v).tolist() for k, v in res.items()}
            print(f"  {key}={vkey} {alg:12s} " + "  ".join(
                f"{k}={float(np.mean(v)):.4f}" for k, v in res.items()))
            flush()

    flush()
    return results


def run_point(config, algorithms, outdir):
    """One config, no sweep: `<algorithm>.npz` per method in `outdir`."""
    task, group_size = load_task(config["task"], config)
    tuned = load_tuned(config.get("tuned_params"), algorithms)

    for algorithm in algorithms:
        # task module defaults < yaml overrides < tuned params
        hp = dict(task.hp_defaults)
        hp.update(config.get("hyperparams", {}))
        hp.update(tuned.get(algorithm, {}))
        res = run_algorithm(task, algorithm, num=config["num"],
                            particles=config["num_samples"], hp=hp,
                            seed=config.get("seed", 0), group_size=group_size)
        np.savez(os.path.join(outdir, f"{algorithm}.npz"),
                 **{k: np.asarray(v) for k, v in res.items()})
        print(f"  {algorithm:12s} " + "  ".join(
            f"{k}={float(np.mean(v)):.4f}" for k, v in res.items()))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("outdir")
    ap.add_argument("--sweep", help="yaml key to sweep instead of the config's axis")
    ap.add_argument("--values", nargs="+", help="values for --sweep (yaml scalars)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config = yaml.safe_load(open(args.config))
    algorithms = config.pop("algorithms")

    values = [yaml.safe_load(v) for v in args.values] if args.values else None
    if args.sweep and values is None:
        ap.error("--sweep requires --values")
    if args.sweep:                       # an override is a different axis
        config["sweep"] = dict(config.get("sweep") or {}, xticks=None,
                               xlabel=args.sweep,
                               name=f"fig_{config['task']}_{args.sweep}")
    key, values = sweep_axis(config, args.sweep, values)

    # A config declaring a `sweep` block runs the whole axis; otherwise the
    # single point it describes.
    if key is None:
        run_point(config, algorithms, args.outdir)
    else:
        run_sweep(config, key, values, algorithms,
                  Path(args.outdir) / f"sweep_{key}.json")


if __name__ == "__main__":
    main()
