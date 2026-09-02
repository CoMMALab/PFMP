# Experiments

Two independent pipelines. Both take a YAML config and write a cache that an
analyzer turns into tables and figures.

| pipeline | scripts | used for |
|----------|---------|----------|
| **geom** | `geom_runner.py`, `geom_analyze.py`, `geom_figure.py`, `geom_optuna.py` | 2D navigation, MotionBenchMaker, push-T — the sweep figures |
| **mjx** | `mjx_runner.py`, `mjx_analyze.py` | MuJoCo non-linear control benchmarks — the cost/time table |

They do not share config formats or method vocabularies. Pick by task.

## geom: sweeps over one axis

```shell
python experiments/geom_runner.py experiments/configs/push_tee.yaml out/push_tee
python experiments/geom_analyze.py out/push_tee
```

The runner solves every (sweep value, algorithm) cell and caches results to
`<outdir>/sweep_<key>.json`. Cells are cached individually, so re-running skips
work already done.

The analyzer prints a table and writes `<name>.pdf`/`.png`. Pass several caches
with `--name` to draw them as panels of one figure:

```shell
python experiments/geom_analyze.py \
    out/push_tee/sweep_distance.json out/push_tee/sweep_num_samples.json \
    --name push_tee_2d_results_fig
```

`geom_figure.py <task>` draws the qualitative figures instead — particle clouds
and trajectory traces rather than aggregate curves.

### Methods

Method names are decomposition strings, parsed by `pfmp.method_decomp`:

| name | proposal | update | resampling |
|------|----------|--------|-----------|
| `ps` | Gaussian | `argmin` | 0 |
| `mppi` | Gaussian | `mean` | 0 |
| `cem` | Gaussian | `topk` | 0 |
| `dial` | Gaussian | `diffuse` | 0 |
| `sis` | programmable | `argmin` | 0 |
| `sir` | programmable | `argmin` | task default |
| `biased_mppi` | programmable | `mean` | 0 |
| `biased_cem` | programmable | `topk` | 0 |

Any name takes a `+r` suffix to add resampling — `mppi+r0.5` pins the ESS
threshold, bare `+r` uses the task's default. That is the only way to set the
resampling axis; it is not a separate config field.

### Configs

| config | task | sweeps |
|--------|------|--------|
| `drive.yaml` | 2D bicycle navigation | planning horizon (knots) |
| `mbm.yaml` | MotionBenchMaker 7-DoF | dataset (categorical) |
| `push_tee.yaml` | 2D push-T | arena size, i.e. obstacle count |
| `push_tee_particles.yaml` | 2D push-T | particle budget |

A config names the task, the methods, the fixed hyperparameters, and one sweep
axis. `tuned_params` points at a JSON in `checkpoints/` holding per-method
values found by Optuna.

### Tuning

```shell
python experiments/geom_optuna.py experiments/configs/push_tee.yaml --trials 200
```

The search space is set by the **update rule**, not the method name: every
method tunes `stdev`, `mean` adds `mppi_temperature`, `topk` adds
`cem_num_elites` and a covariance floor, `diffuse` adds DIAL's annealing
parameters. Proposal mixture weights are fixed, never tuned. Results land in
`checkpoints/<task>_tune.json`.

## mjx: MuJoCo control benchmarks

```shell
python experiments/mjx_runner.py experiments/configs/CartPole.yaml results/CartPole
python experiments/mjx_analyze.py results/CartPole
```

The runner saves per-algorithm trajectories under `<outdir>/<algorithm>`; the
analyzer prints a booktabs cost table and a scatter plot per method. Both use
`fire`, so arguments are positional.

Configs: `CartPole.yaml`, `DoubleCartPole.yaml`, `HalfCheetah.yaml`,
`Walker.yaml`, `Cube.yaml`.

Algorithms here are plain names, **not** decomposition strings — `ps`, `mppi`,
`cem` and `dial` come from [hydrax](https://github.com/vincekurtz/hydrax),
`icem` is ours (subclassing hydrax's CEM to override `sample_knots`), and `sis`,
`sir`, `learned` and `biased_mppi` are PFMP. The resampling threshold is
hardcoded at 0.5 for the PFMP entries rather than read from the config.

`learned` needs an actor trained by the scripts in `mjlab/`:

```shell
python experiments/mjlab/mjlab_cartpole.py --train
```

The runner skips `learned` with a warning when no checkpoint is found, so the
other methods still run. `learned_checkpoint: null` picks the newest run.

## Notes

`mjx_runner.py` keeps a `pfmp_native` branch for tasks that carry their own
dynamics, cost and proposal rather than wrapping a hydrax task. No config uses
it today; the push-T experiments go through the geom pipeline instead.
