# PFMP — Particle Filtering Motion Planning

Sampling-based MPC as sequential Monte Carlo inference, in JAX and MuJoCo MJX.

Predictive Sampling, MPPI, CEM, iCEM and DIAL-MPC are usually presented as
separate algorithms. They are all the same SMC kernel with three choices made
differently:

| axis | what it is | examples |
|------|------------|----------|
| **proposal** `q` | where candidate control sequences come from | Gaussian perturbation, a task-specific program, a learned policy |
| **update** `U` | how cost-weighted particles collapse to a nominal plan | `argmin`, softmax mean, elite set, annealed |
| **resampling** `ρ` | the ESS fraction below which particles are resampled | `0` (never, SIS) … `1` (every step, SIS/R) |

`PFMP` is one implementation of that kernel in about 280 lines in
[pfmp/algorithm.py](pfmp/algorithm.py).

```python
from pfmp import method_decomp

method_decomp("mppi")       # ('gaussian', 'mean',   0.0)   MPPI
method_decomp("cem")        # ('gaussian', 'topk',   0.0)   CEM
method_decomp("sis")        # ('informed', 'argmin', 0.0)   programmable proposal, no resampling
method_decomp("mppi+r0.5")  # ('gaussian', 'mean',   0.5)   MPPI, resampling below ESS 0.5
```

Any `+r` suffix adds resampling; `+r0.2` defines an ESS resampling threshold

## Install

```shell
uv venv && source .venv/bin/activate && uv pip install .
```

JAX needs `XLA_PYTHON_CLIENT_PREALLOCATE=false` whenever it shares the GPU with
anything else. The included `.envrc` sets that (plus `LD_LIBRARY_PATH` for the
pip-installed NVIDIA libs); without direnv, export it yourself.

## Try it

Non-prehensile push-T in a 2D rigid-body sim, with the programmable proposal:

```shell
python examples/push_tee_2d.py --method pfmp --distance 4  # try distance 8, 12
```

`--method` takes `pfmp`, `ps`, `mppi`, `cem`, `dial`, `biased_mppi` or
`biased_cem`, so you can select an MPC proposal (gaussian or informed) and update (ps/mppi/cem/dial).
`--particles`, `--distance` and `--seed` vary the problem.
Long-horizon navigation through an obstacle field, as an animation:

```shell
python examples/drive_2d.py --particles 1024
```

7-DoF motion planning on MotionBenchMaker, in a viser browser scene:

```shell
tar -xvf models/problems.tar.bz2 -C models
python examples/mbm.py viser models/problems/table_pick_panda
```

`mbm.py` takes `--sis`, `--sir` or `--both` (default) to run the two kernels
side by side. MuJoCo control tasks live in `cart_pole.py`, `double_cart_pole.py`,
`half_cheetah.py` and `walker.py`.

## Defining a new task

A task is five functions and a parameter record, no subclassing. `PFMP` is a
`NamedTuple`, so the whole planner is a valid `jit` static argument.

**1. The parameter record `theta`.** A class inheriting `NamedTuple`, which contains whatever the
dynamics and cost need as read-only arguments, plus two fields the library requires: `mean`,
the nominal plan of shape `(num_knots, ...)`, and `stdev`, the sampling noise.
`stdev` may be a scalar or per-knot — CEM refits it per knot — so always read it
through `pfmp.sigma_at(theta, idx)`.

```python
class Theta(NamedTuple):
    goal:      jax.Array   # whatever your cost needs
    obstacles: jax.Array
    mean:      jax.Array   # (num_knots, action_dim) — required only for library updates
    stdev:     jax.Array   # scalar or (num_knots,)  — required only for library updates
```

**2. The five callables.** Each takes `theta` and returns plain arrays:

| callable | signature | returns |
|---|---|---|
| `mpc_dynamics` | `(x, u, theta, t)` | `(x_next, aux)` |
| `mpc_running_cost` | `(x, u, theta, t)` | scalar |
| `mpc_terminal_cost` | `(x, theta)` | scalar, must be deterministic |
| `mpc_init` | `(key, x, theta)` | first knot `u` |
| `mpc_proposal` | `(key, x, u, theta, t)` | next knot `u`, given the previous |

`mpc_init` and `mpc_proposal` are random sampling programs that contains the arbitrary control flow of a policy. For a Gaussian
perturbation around `theta.mean` — what PS, MPPI, CEM and DIAL all use — a decent prior is provided through the `make_perturbation` function.

```python
from pfmp import PFMP, make_perturbation, mppi_update

init, proposal = make_perturbation(clip=clip_action)   # clip defines joint limits
```

Every rule has the same shape, `update(theta, trajs, costs, *hyperparams) ->
(theta, aux)`, and each writes the new nominal plan back into `theta.mean`:

| rule | extra arguments | recovers |
|---|---|---|
| `ps_update` | — | Predictive Sampling |
| `mppi_update` | `temperature` | MPPI |
| `cem_update` | `elite_frac`, `stdev_floor` | CEM |
| `dial_update` | `temp_sample` | DIAL-MPC |

Swapping `mppi_update` for `cem_update` lets you pick MPPI or CEM updates.

To write a programmable proposal, implement `mpc_init` and `mpc_proposal`. Everything else about the planner is in a typed tuple; see `sample_mixture` in
[examples/push_tee_2d.py](examples/push_tee_2d.py) for the three-branch mixture
used in the push-T experiments.

**3. Build the planner and plan.**

```python
planner = PFMP(
    dynamics, running_cost, terminal_cost, init, proposal,
    horizon=5.0, num_knots=10, particles=1024,
    ess_resample_thresh=0.5,   # 0 disables resampling (SIS)
    dt=0.02,
)
for _ in range(num_iterations):
    key, subkey = jax.random.split(key)
    trajs, costs = planner.run_sir(subkey, x0, theta)   # or run_sis
    theta, _ = mppi_update(theta, trajs, costs, temperature=0.1)

plan = theta.mean
```

`run_sir` returns trajectories of shape `(particles, num_knots, ...)` and costs
of shape `(particles, num_knots)`, which can be passed to a library update.

## Benchmarks

Paper numbers come from the batched runner:

```shell
python experiments/geom_runner.py experiments/configs/push_tee.yaml out/push_tee
python experiments/geom_analyze.py out/push_tee
```

The config names the sweep axis and the methods; the analyzer prints the table
and writes the figure. `experiments/configs/` holds one per experiment.

## Layout

| path | role |
|------|------|
| `pfmp/` | the library — `algorithm.py` is the whole kernel |
| `examples/` | one script per task, each runnable from the repo root |
| `experiments/` | batched runner, analysis, and Optuna tuning |
| `fr3/` | real-robot Franka FR3 push-T ([README](fr3/README.md)) |
| `models/` | MuJoCo XML/URDF/meshes and the MotionBenchMaker problem set |

## Tests

The test runner is not a runtime dependency; install it first:

```shell
uv pip install ".[dev]"          # or: uv pip install pytest
```

```shell
python -m pytest -m "not slow"   # fast subset, ~19s
python -m pytest                 # everything; ~11 tests JIT an MJX rollout
```