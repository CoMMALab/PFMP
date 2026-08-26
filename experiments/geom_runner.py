"""Batched runner for the NamedTuple-defined tasks (drive, push-T, MBM).

Mirrors `mjx_runner.py` (yaml config -> per-algorithm results in an outdir) for
tasks that are plain JAX functions over a NamedTuple `theta`, avoiding an
`mjx.Data` per particle. One task is selected by a flag; everything else is
shared, so the tuner, the sweep, and the reported table cannot drift apart.

    python experiments/geom_runner.py experiments/configs/mbm.yaml out/mbm
"""

import os
import sys
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from pfmp import (PFMP, METHOD_ALIASES, method_decomp,
                  make_perturbation, sigma_at, knot_index)
from pfmp.obstacle_grid import sdf_aabb, hinge_loss, generate_map_batch
from pfmp.algorithm import ps_update, mppi_update, cem_update, sample_update
from pfmp.update import dial_step, ess
from pfmp.interpolate import hermite_spline


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------

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
    # The optional object-informed proposal, if the task has one.
    # Otherwise fall back to shared perturbation.
    custom_init: Callable = None      # (key, x, theta) -> u
    custom_proposal: Callable = None  # (key, x, u, theta, t) -> u
    clip_action: Callable = None      # (u, theta) -> u   bounds for the shared
                                      #   perturbation proposal
    n_replans: int = 1            # 1 = open-loop planning; >1 = closed-loop MPC
    execute: Callable = None      # (x, theta) -> x        None = world is static
    shift: Callable = None        # (theta) -> theta       None = no re-timing
    warm_start: Callable = None   # (x0, theta) -> theta   baselines' nominal plan
    num_problems: int = None      # how many problems the task actually has.
                                  #   MBM's follows from its datasets, so a
                                  #   `num` in the config would only restate it
                                  #   and could disagree with it.


# Which proposal / update / resampling each algorithm name selects.
ALGORITHMS = tuple(METHOD_ALIASES)   # each may take a "+r[rho]" suffix

def apply_hp(theta, hp: dict):
    """Route the tuned values that name a theta field onto theta."""
    return theta._replace(**{k: float(v) for k, v in hp.items()
                             if k in theta._fields})


ESS_THRESH_DEFAULT = PFMP._field_defaults["ess_resample_thresh"]


def build_controller(task: GeomTask, algorithm: str, particles: int) -> PFMP:
    proposal, _, rho = method_decomp(algorithm)
    perturb = make_perturbation(task.clip_action)
    # A task with no q of its own collapses the proposal axis to gaussian, so on
    # drive every `informed` name resolves to its gaussian counterpart.
    init, prop = ((task.custom_init, task.custom_proposal)
                  if proposal == "informed" and task.custom_proposal is not None
                  else perturb)
    if rho is not None:
        return PFMP(task.dynamics, task.running_cost, task.terminal_cost, init, prop,
                    horizon=task.horizon, num_knots=task.num_knots, particles=particles,
                    ess_resample_thresh=rho,
                    dt=task.dt)
    else:  # use default ess_resample_thresh
        return PFMP(task.dynamics, task.running_cost, task.terminal_cost, init, prop,
                    horizon=task.horizon, num_knots=task.num_knots, particles=particles,
                    dt=task.dt)


def _elite_count(hp: dict, particles: int) -> int:
    """CEM's elite count. `cem_num_elites` is authoritative; `cem_elite_frac` is
    the legacy form, kept so older tuned JSON still loads."""
    if "cem_num_elites" in hp:
        return max(1, min(int(hp["cem_num_elites"]), particles))
    return max(1, int(hp.get("cem_elite_frac", 0.1) * particles))


def make_replan(ctrl: PFMP, algorithm: str, hp: dict, particles: int,
                return_trajs: bool = False):
    """One planning iteration: (key, x, theta) -> (theta, per-knot ESS).

    `return_trajs` additionally returns the sampled knots and their costs, so a
    figure can draw the particle cloud the update actually saw rather than
    re-deriving one. DIAL replaces this wholesale: it owns its own sampling
    loop, so it does not support the flag.
    """
    if return_trajs and method_decomp(algorithm)[1] == "diffuse":
        raise ValueError("return_trajs is unsupported for DIAL (owns its loop)")
    _, update_kind, rho = method_decomp(algorithm)
    sample = ctrl.run_sis if rho == 0.0 else ctrl.run_sir

    if update_kind == "diffuse":
        kw = {a: hp[k] for k, a in (("dial_n_inner", "n"), ("dial_beta1", "beta1"),
                                    ("dial_beta2", "beta2"),
                                    ("dial_temp_sample", "temp_sample")) if k in hp}
        kw["n"] = int(kw.get("n", 2))

        def dial_replan(key, x, theta):
            # Costs from DIAL's final annealing level, so its ESS is the same
            # per-knot quantity reported for the single-pass methods.
            theta, costs = dial_step(key, ctrl, x, theta,
                                     return_costs=True, **kw)
            return theta, jax.vmap(ess, in_axes=1)(jnp.cumsum(costs, axis=1))

        return dial_replan
    if update_kind == "argmin":
        # The MAP update. On a task with no custom proposal this makes `sis`
        # and `ps` the same algorithm; `sis_sample` is what separates them.
        update = lambda th, tr, co: ps_update(th, tr, co)
    elif update_kind == "sample":
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
        temp = hp.get("sample_temperature", hp.get("mppi_temperature"))
        kw = {} if temp is None else {"temperature": max(float(temp), 1e-6)}
        update = lambda th, tr, co, k: sample_update(th, tr, co, k, **kw)
    elif update_kind == "mean":
        kw = ({"temperature": max(float(hp["mppi_temperature"]), 1e-6)}
              if "mppi_temperature" in hp else {})
        update = lambda th, tr, co: mppi_update(th, tr, co, **kw)
    elif update_kind == "topk":
        # The elite count sets an array shape, so it must stay static.
        #
        # Parameterised as an absolute count, not a fraction: k = int(frac * N)
        # is a step function of the particle count, so a fraction tuned at one
        # budget silently changes the method at another. Measured on push-T at
        # N=512, k=1 -> 100% success and k=2 -> 12.5%; a fraction tuned at N=256
        # (where it floored to k=1) crossed to k=2 at N=512 and produced a
        # non-monotone particle-count curve.
        kw = {}
        if "cem_num_elites" in hp or "cem_elite_frac" in hp:
            kw["elite_frac"] = _elite_count(hp, particles) / particles
        if "cem_stdev_floor" in hp:
            kw["stdev_floor"] = float(hp["cem_stdev_floor"])
        update = lambda th, tr, co: cem_update(th, tr, co, **kw)
    else:
        raise ValueError(f"no update implemented for {update_kind!r} "
                         f"(method {algorithm!r})")

    needs_key = update_kind == "sample"

    # Effective sample size of the weights each update implies, reported per
    # replan and never averaged here: in open-loop tasks the replans are
    # iterations of one optimisation (concentration is convergence), while in
    # closed-loop they are separate inference problems (concentration is
    # degeneracy), so only the caller knows how to reduce it.
    def replan(key, x, theta):
        # Only split when the update itself needs randomness. Splitting
        # unconditionally would shift the sampling stream for every other
        # method, silently changing results that are meant to be comparable.
        sub, rest = (jax.random.split(key) if needs_key else (key, None))
        trajs, costs = sample(sub, x, theta)
        args = (theta, trajs, costs) + ((rest,) if needs_key else ())
        # ESS of the importance weights under the target exp(-cost), per knot:
        # entry k is the ESS of the return accumulated through knot k, i.e. the
        # sequential filtering ESS. The last entry is the total-return ESS and
        # includes the terminal cost, so comparing [-1] against [-2] shows how
        # much of the collapse the terminal term is responsible for.
        # One definition for every method: it measures the particle cloud the
        # proposal produced, not the update applied to it.
        total = jnp.cumsum(costs, axis=1)                    # (P, K)
        ess_k = jax.vmap(ess, in_axes=1)(total)              # (K,)
        theta = update(*args)[0]
        return (theta, ess_k, trajs, costs) if return_trajs else (theta, ess_k)
    return replan


def _make_simulator(task: GeomTask, replan: Callable, widen_stdev: bool = False):
    """jit(vmap(...)) of the outer loop: replan -> execute -> shift, then measure.

    `execute`/`shift` are None for open-loop tasks, where the world never moves.
    """
    execute = task.execute or (lambda x, theta: x)
    shift = task.shift or (lambda theta: theta)

    def one(key, x0, theta):
        # Warm-start unconditionally, including for the methods whose proposal
        # ignores `theta.mean`: it is what gives `mean` its shape.
        if task.warm_start is not None:
            theta = task.warm_start(x0, theta)
        # CEM's covariance refit only fires when `stdev` is already per-knot, and
        # theta is the scan carry, so the widening has to happen before the loop.
        # `topk` only: DIAL builds its schedule from a scalar `theta.stdev`.
        if widen_stdev and jnp.ndim(theta.stdev) == 0:
            theta = theta._replace(stdev=jnp.full_like(theta.mean, theta.stdev))

        def step(carry, _):
            key, x, theta = carry
            key, sub = jax.random.split(key)
            theta, ess = replan(sub, x, theta)
            x = execute(x, theta)
            theta = shift(theta)
            return (key, x, theta), ess

        (_, x, theta), ess = jax.lax.scan(
            step, (key, x0, theta), None, length=task.n_replans)
        # (n_replans, num_knots) per problem, dense and unreduced.
        return dict(task.measure(x, theta), ess=ess)

    return jax.jit(jax.vmap(one))


def run_algorithm(task: GeomTask, algorithm: str, num: int, particles: int,
                  hp: dict, seed: int = 0, group_size: int = None):
    """Run `num` problems of `task` under one method. Batched in groups because
    some tasks (MBM) have per-problem array shapes that match only within a
    dataset. Every hyperparameter is closed into the compiled program.
    """
    group_size = group_size or num
    _, update_kind, _ = method_decomp(algorithm)
    ctrl = build_controller(task, algorithm, particles)
    sim = _make_simulator(task, make_replan(ctrl, algorithm, hp, particles),
                          widen_stdev=update_kind == "topk")

    out = []
    for lo in range(0, num, group_size):
        idx = range(lo, min(lo + group_size, num))
        probs = [task.sample_problem(i) for i in idx]
        x0 = jnp.stack([p[0] for p in probs])
        thetas = [p[1] for p in probs]
        thetas = [apply_hp(th, hp) for th in thetas]
        theta = jax.tree.map(lambda *xs: jnp.stack(xs), *thetas)
        keys = jnp.stack([jax.random.key(seed + i) for i in idx])
        out.append(jax.tree.map(np.asarray, sim(keys, x0, theta)))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *out) if len(out) > 1 else out[0]


class Evaluator:
    """A loaded task over a fixed slice of its problems: (method, hp) -> metrics.

    Holds the loaded task because `load_task` is expensive -- MBM reads its
    datasets, drive generates its maps -- and both the tuner and the sweep call
    this many times against the same slice. `num`/`offset` carve out the slice,
    which is how tuning and validation stay disjoint.
    """

    def __init__(self, config: dict, num: int = None, offset: int = 0,
                 seed: int = None):
        self.cfg = dict(config, problem_offset=offset)
        if num is not None:
            self.cfg["num"] = num
        self.seed = self.cfg.get("seed", 0) if seed is None else seed
        self.task, self.group_size = load_task(self.cfg["task"], self.cfg)
        self.num = num or self.task.num_problems

    def __call__(self, method: str, hp: dict = None) -> dict:
        full = dict(self.cfg.get("hyperparams", {}))
        full.update(hp or {})
        return run_algorithm(self.task, method, num=self.num,
                             particles=self.cfg["num_samples"], hp=full,
                             seed=self.seed, group_size=self.group_size)


# ---------------------------------------------------------------------------
# Task registry. One adapter per task; `load_task` is the flag swap.
# ---------------------------------------------------------------------------

def _mbm_task(config):
    """MBM: open-loop 7-DoF configuration-space planning. n_replans=1, the world
    never advances, and `measure` rolls the plan out to score it."""
    import mbm
    
    robot = mbm.load_robot(config.get("urdf", "models/panda_spherized.urdf"))
    datasets = config["datasets"]
    # How many problems there are is a property of the datasets, not something
    # the config restates. `num` and `problem_offset` are totals across datasets
    # and convert to per-dataset counts here: a held-out split at offset 350
    # means problem 50 of each of 7 datasets, not problem 350 of one.
    n_ds = len(datasets)
    loaded = [mbm.load_mbm_dataset(ds) for ds in datasets]
    avail = min(len(ps) for ps in loaded)
    off_per = int(config.get("problem_offset", 0)) // n_ds
    n_per = (max(1, int(config["num"]) // n_ds) if config.get("num")
             else avail - off_per)
    if off_per + n_per > avail:
        raise ValueError(
            f"MBM: {n_per} problems/dataset at offset {off_per} exceeds the "
            f"{avail} available; lower `num`")
    problems, groups = [], []
    for ps in loaded:
        ps = ps[off_per:off_per + n_per]
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
        num_problems=len(problems),
        sample_problem=sample_problem, measure=measure,
        horizon=mbm.HORIZON, num_knots=mbm.NUM_KNOTS, dt=dt,
        n_replans=1, execute=None, shift=None, warm_start=warm_start,
)
    return task, groups[0]      # batch per dataset: env shapes match only within

def _push_tee_task(config):
    """Push-T: closed-loop planar pushing. The world advances between replans,
    so `execute` runs the physics and `shift` re-times the plan onto the new
    clock, which is exactly what MBM leaves as None."""
    import push_tee_2d as pt
    
    off = int(config.get("problem_offset", 0))
    seeds = list(range(config.get("seed", 0) + off,
                       config.get("seed", 0) + off + config["num"]))
    distance = config.get("distance")
    n_obs = config.get("n_obs")
    built = [pt._build_scenario(sd, distance, n_obs) for sd in seeds]
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
        return theta._replace(mean=pt.mean_from_state(x0))

    def measure(x, theta):
        pos_err = jnp.linalg.norm(x[0:2] - theta.goal_xy)
        angle_err = jnp.abs(pt._angle_wrap(x[2] - theta.goal_angle))
        return {"pos_err": pos_err, "angle_err": angle_err,
                "success": (pos_err < pt.SUCCESS_POS_THRESH)
                           & (angle_err < pt.SUCCESS_ANGLE_THRESH)}

    task = GeomTask(
        dynamics=pt.dynamics, running_cost=pt.running_cost,
        terminal_cost=pt.terminal_cost,
        custom_init=pt.init_control, custom_proposal=pt.proposal,
        clip_action=None,          # push-T clips inside the dynamics, not the plan
        sample_problem=lambda i: (built[i][0], built[i][1]), measure=measure,
        horizon=pt.HORIZON, num_knots=pt.NUM_KNOTS, dt=pt.DT,
        n_replans=n_replans, execute=execute, shift=shift,
        warm_start=warm_start, num_problems=len(seeds),
        # Mixture weights frozen at the FR3 planner's, so sim and robot share q.
)
    return task, len(seeds)



def _drive_task(config):
    """Open-loop like MBM (no `execute`/`shift`), but iterated: the plan is
    refined over `n_replans` passes against a static world. No object-informed
    proposal exists here, so `custom_*` stay None and the SMC methods run the
    shared perturbation, differing from the baselines only in update and
    resampling."""
    import drive_2d as d2

    num_knots = int(config.get("num_knots", 20))
    arena_length = float(config.get("arena_length", 1.0 * num_knots + 1.0))
    horizon = num_knots * 0.5
    n = config["num"]

    off = int(config.get("problem_offset", 0))
    obs, x0s, goals, rejected = generate_map_batch(
        config.get("seed", d2.SEED) + num_knots * 10000, off + n, arena_length,
        d2.ARENA_WIDTH, d2.OBSTACLE_DENSITY, d2.BOX_SIZE_RANGE, d2.MIN_MARGIN)
    obs, x0s, goals = obs[off:], x0s[off:], goals[off:]
    print(f"  generated {n} maps (rejected {rejected})")

    tk0 = jnp.linspace(0.0, horizon, num_knots)
    tq = jnp.arange(int(horizon / d2.DT)) * d2.DT

    def sample_problem(i):
        return x0s[i], d2.Params(
            goal=goals[i], obstacles=obs[i], L=d2.WHEELBASE, v=d2.SPEED,
            mean=jnp.zeros(num_knots), stdev=d2.U_STDEV,
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
            xn, _ = d2.dynamics(x, u, theta, t)
            return xn, (d2.cost_fn(xn, u, theta, t), xn)

        x_final, (running, path) = jax.lax.scan(step, x, (us, tq))
        sdf = jax.vmap(lambda p: jnp.min(jax.vmap(
            lambda b: sdf_aabb(p[:2], b))(theta.obstacles)))(path)
        return {"cost": jnp.sum(running) + d2.terminal_cost(x_final, theta),
                "goal_dist": jnp.linalg.norm(x_final[:2] - theta.goal),
                "collided": jnp.any(sdf < 0.0),
                "collision_frac": jnp.mean(sdf < 0.0)}

    task = GeomTask(
        dynamics=d2.dynamics, running_cost=d2.cost_fn, terminal_cost=d2.terminal_cost,
        clip_action=d2.clip_action,
        sample_problem=sample_problem, measure=measure,
        horizon=horizon, num_knots=num_knots, dt=d2.DT,
        # The legacy runner gave the baselines `num_knots` refinement passes but
        # SIS/SIR only one. Here every method gets the same budget.
        n_replans=int(config.get("n_replans", num_knots)),
        execute=None, shift=None, warm_start=None,
        num_problems=n,
)
    return task, n


_TASKS = {"mbm": _mbm_task, "push_tee": _push_tee_task, "drive": _drive_task}


def load_task(name, config):
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; have {sorted(_TASKS)}")
    return _TASKS[name](config)


def load_tuned(spec, algorithms):
    """Resolve `tuned_params` to {algorithm: {param: value}}.

    Accepts one file keyed by method, or a path with an `{algorithm}`
    placeholder. Missing entries warn and fall back to defaults.
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


def code_hash():
    """Hash of the sources that determine a result.

    A running sweep holds the code as of process start, so a mid-run edit leaves
    a stale cache with a fresh mtime. This makes that detectable.
    """
    import hashlib
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for rel in sorted(["pfmp/algorithm.py", "pfmp/update.py", "pfmp/interpolate.py",
                       "experiments/geom_runner.py", "examples/mbm.py",
                       "examples/push_tee_2d.py", "examples/drive_2d.py"]):
        p = root / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_sweep(config, key, values, algorithms, cache_path):
    """Run every (swept value, algorithm) pair into a resumable cache.

    The axis is a yaml key overridden across a list. Values are cached raw, not
    aggregated -- a rate needs its numerator and denominator, a cost needs the
    whole sample. The cache also stores the axis, the resolved hyperparameters,
    and `code_hash`, so plotting needs nothing but the file.
    """
    import json
    cache_path = Path(cache_path)
    blob = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    results = blob.get("results", {})
    axis = dict(config.get("sweep") or {}, key=key, values=values)
    axis.setdefault("name", f"{config['task']}_{key}_fig")

    wanted = {p["metric"] for p in (axis.get("panels") or [])}
    wanted |= {axis["metric"]} if axis.get("metric") else set()

    def fresh(cell):
        return wanted <= set(cell)

    used_hp = {}

    code = code_hash()
    if blob.get("code") not in (None, code):
        print(f"  ! cache was written by different code ({blob['code']} vs "
              f"{code}); cached cells will NOT be recomputed. Delete "
              f"{cache_path} if the change affects results.")

    def flush():
        cache_path.write_text(json.dumps(
            {"task": config["task"], "sweep": axis, "algorithms": algorithms,
             "code": code, "hyperparams": used_hp, "results": results}))

    for value in values:
        vkey = str(value)
        results.setdefault(vkey, {})
        cfg = dict(config, **{key: value})
        ev = tuned = None

        for alg in algorithms:
            cached = alg in results[vkey] and fresh(results[vkey][alg])
            if tuned is None:
                tuned = load_tuned(cfg.get("tuned_params"), algorithms)
            hp = dict(cfg.get("hyperparams", {}))
            hp.update(tuned.get(alg, {}))
            used_hp[alg] = {k: v for k, v in hp.items()
                            if isinstance(v, (int, float, bool))}
            *_, rho = method_decomp(alg)
            used_hp[alg]["ess_thresh"] = ESS_THRESH_DEFAULT if rho is None else rho
            if cached:
                print(f"  {key}={vkey} {alg:12s} cached")
                continue
            if ev is None:                       # build lazily, once per value
                ev = Evaluator(cfg)
            res = ev(alg, tuned.get(alg, {}))
            results[vkey][alg] = {k: np.asarray(v).tolist() for k, v in res.items()}
            print(f"  {key}={vkey} {alg:12s} " + _summarise(res))
            flush()

    flush()
    return results


def _summarise(res):
    """One line per method. Non-finite entries are skipped rather than allowed
    to poison the mean: ESS is legitimately undefined on a problem where every
    particle collided, and a single such problem would otherwise NaN the whole
    column. The count is reported so the omission is visible."""
    parts = []
    for k, v in sorted(res.items()):
        a = np.asarray(v, dtype=float)
        bad = int((~np.isfinite(a)).sum())
        parts.append(f"{k}={np.nanmean(np.where(np.isfinite(a), a, np.nan)):.4f}"
                     + (f"[{bad} n/a]" if bad else ""))
    return "  ".join(parts)


def run_point(config, algorithms, outdir):
    """One config, no sweep: `<algorithm>.npz` per method in `outdir`."""
    ev = Evaluator(config)
    tuned = load_tuned(config.get("tuned_params"), algorithms)

    for algorithm in algorithms:
        # yaml overrides < tuned params
        res = ev(algorithm, tuned.get(algorithm, {}))
        np.savez(os.path.join(outdir, f"{algorithm}.npz"),
                 **{k: np.asarray(v) for k, v in res.items()})
        print(f"  {algorithm:12s} " + _summarise(res))


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
                               name=f"{config['task']}_{args.sweep}_fig")
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
