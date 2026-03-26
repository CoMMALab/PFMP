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
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/fr3/pushT.py [sir|ps]
```

which will show sequential importance resampling (SIR) solving a non-prehensile manipulation push-T task with Mujoco physics. Try some other examples: `cart_pole.py`, `double_cart_pole.py`, `walker.py`.

## Planning

Run a planner example:

```
python examples/2d_simple.py --smooth
```

Or FR3 on MotionBenchMaker:

```
tar -xvf models/problems.tar.bz2 -C models
python examples/fr3/mbm.py bench models/problems/table_pick_panda 50
```

Each planning example shows a side-by-side of the SIS and SIS/R algorithms.
