# Particle Flow Motion Planning

Particle Flow Motion Planning (PFMP) is a planning-as-inference library for fast path generation in JAX and Mujoco XLA. Specifically, PFMP provides an implementation of the highly-parallelizable sequential importance sampling-resampling (SIS/R) kernel, also known as sequential Monte Carlo (SMC).

## Installation

```shell
uv venv && source .venv/bin/activate
uv pip install .[test]
```

## Control

Run a control example:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/double_cart_pole.py sir
```

## Planning

Run a planner example:

```
python examples/car.py
```
