# geofikinpy

**GeoFIK-in-Python** (`geofikinpy`) is a lightweight Python binding — with optional MuJoCo snippets — for the **analytic inverse kinematics** of the **Franka Emika Panda** (7-DoF), powered by the upstream **GeoFIK** library.

It calls exact closed-form solvers (no iteration), returns all valid joint solutions, and plays nicely with NumPy.

---

## ✨ Features

- **Analytic IK** for Panda: fast, multi-solution, no numerical optimization.
- **Pythonic API** with NumPy arrays.
- **FK & Jacobians** (including Jacobian-from-IK variants).
- **Swivel/q7 families** for redundancy control.
- **Drop-in visual snippets** for MuJoCo.
- Simple `pip install -e .` build via **pybind11 + Eigen**.

---

## Install

### Prereqs
- Python 3.9+
- A C++17 compiler
- **Eigen** headers (no linking)
  - Ubuntu: `sudo apt install libeigen3-dev`
  - macOS (Homebrew): `brew install eigen`
  - Or set `EIGEN3_INCLUDE_DIR=/path/to/eigen3`

### Clone (with submodule)
```bash
git clone https://github.com/yourname/geofikinpy
cd geofikinpy
git submodule update --init --recursive