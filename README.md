# Particle Filtering Motion Planning

Particle Filtering Motion Planning is a planning-as-inference library for fast path generation in JAX and Mujoco XLA. Specifically, `PFMP` class is an implementation of the highly-parallel sequential importance sampling-resampling (SIS/R) kernel, also known as sequential Monte Carlo (SMC).

## Installation

```shell
uv venv && source .venv/bin/activate
uv pip install .
```

## Control

Run a controller example with FR3:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/push_tee_2d.py --method pfmp
```

which will show sequential importance resampling (SIR) solving a non-prehensile manipulation push-T task with Mujoco physics. `--method` also accepts `ps`, `mppi`, `biased_mppi`, `cem`, `biased_cem`, and `dial`. Try some other examples: `cart_pole.py`, `double_cart_pole.py`, `half_cheetah.py`, `walker.py`.

## Planning

Run a planner example:

```
python examples/drive_2d.py --smooth
```

Or FR3 on MotionBenchMaker:

```
tar -xvf models/problems.tar.bz2 -C models
python examples/mbm.py viser models/problems/table_pick_panda
```

`mbm.py` takes `--sis`, `--sir`, or `--both` (default) to compare the two kernels.

## Benchmarks

The numbers in the paper come from the batched runner, not these scripts:

```
python experiments/geom_runner.py experiments/configs/push_tee.yaml out/push_tee
python experiments/geom_analyze.py out/push_tee
```
