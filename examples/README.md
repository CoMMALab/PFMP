# PFMP Examples

Each script is runnable from the repo root.

## 2D

| Script | Description |
|--------|-------------|
| `2d_simple.py` | Bicycle model navigating a regular obstacle grid |
| `2d_complex.py` | Bicycle model through random obstacle fields; supports ablation vs SMC-MPC / MPPI / CEM baselines |
| `2d_pushT.py` | T-block pushing in 2D (no MuJoCo) |

```bash
python examples/2d_simple.py
python examples/2d_complex.py
python examples/2d_pushT.py
```

## MuJoCo

| Script | Description |
|--------|-------------|
| `cart_pole.py` | Inverted cart-pole balancing |
| `double_cart_pole.py` | Double inverted cart-pole |
| `walker.py` | Bipedal walker |

```bash
python examples/cart_pole.py
python examples/double_cart_pole.py
python examples/drone.py
python examples/walker.py
python examples/3bar.py
```

## FR3 Real Robot

Push-T on a Franka FR3 using PFMP + MuJoCo MJX.

See [`fr3/README.md`](fr3/README.md) for setup, calibration, and run instructions.

```bash
# Sim only (no robot)
python examples/fr3/push_t_real.py --dry-run

# Real robot with kinesthetic teach
python examples/fr3/push_t_real.py --ip 172.16.0.2 --teach
```
