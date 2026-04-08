# Experiments

Benchmarking scripts for comparing PFMP against baseline MPC algorithms.

## Running an experiment

```bash
python experiments/mjx_runner.py experiments/configs/CartPole.yaml results/CartPole
```

`runner.py` takes a YAML config and an output directory.
It runs each algorithm in `algorithms`, saves per-algorithm trajectories to `<outdir>/<algorithm>.npz`.

## Analyzing results

```bash
python experiments/mjx_analyze.py results/CartPole
```

Prints a `latex_booktabs` cost table and shows a scatter plot per method.

## Configs

| Config | Task |
|--------|------|
| `CartPole.yaml` | Inverted cart-pole |
| `DoubleCartPole.yaml` | Double inverted cart-pole |
| `Drone.yaml` | Quadrotor stabilization |
| `Walker.yaml` | Bipedal walker |

### Algorithms

| Key | Description |
|-----|-------------|
| `sis` | PFMP (SIS) |
| `sir` | PFMP (SIR) |
| `ps` | Predictive Sampling (Hydrax) |
| `mppi` | MPPI (Hydrax) |
| `cem` | CEM (Hydrax) |

PFMP-native tasks (e.g. `PushT`) only support `sis` / `sir`.

### Config fields

```yaml
task: "CartPole"
algorithms: ["sis", "sir", "ps", "mppi", "cem"]
num: 20              # rollouts per algorithm
max_step: 1000       # steps per rollout
plan_horizon: 1.0    # seconds
num_knots: 4
num_samples: 1024
noise_level: 0.3
spline_type: "linear"
mppi_temperature: 0.1
cem_num_elites: 3
cem_sigma_start: 0.5
cem_sigma_min: 0.1
```
