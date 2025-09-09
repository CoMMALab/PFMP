import jax
import jax.numpy as jnp
import mujoco
import matplotlib.pyplot as plt
import time
import os, math, pickle, itertools, time, traceback
import numpy as np

from genjax import gen, mv_normal
from pfmp.algorithm_spline import make_transition_model, PFMP
from pfmp.interpolate import hermite_spline
from mujoco import mjx
from typing import NamedTuple, Type
from mujoco.viewer import launch_passive
from pathlib import Path
from typing import Sequence, Tuple, Any, Iterable
from collections import OrderedDict
from matplotlib.lines import Line2D

# plt.rcParams.update({'font.size': 32})
plt.rcParams.update({
    'axes.labelsize': 16,      # X and Y axis labels
    'xtick.labelsize': 12,     # X tick labels
    'ytick.labelsize': 12,     # Y tick labels
    'axes.titlesize': 24,      # Title
    'legend.fontsize': 12      # Legend
})

class PParams(NamedTuple):
    mjx_model: mjx.Model
    nominal_u: jax.Array
    sigma: float
    fine_angle_bounds: float
    fine_angle_sigma: float

class StateTracker(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array

def load_mj(name: str):
    with open(f"assets/{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    return (mj_model, mj_data)

def init_mjx(mj_model, mj_data):
    mjx_model = mjx.put_model(mj_model)

    x0 = mjx.put_data(mj_model, mj_data)
    return x0, PParams(
        mjx_model,
        jnp.zeros((H,1)),
        sigma=0.1,
        fine_angle_bounds=0.1,
        fine_angle_sigma=0.01
    )

@gen
def linear_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    u = jnp.array([ctrl[0]])
    return u

@gen
def cubic_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    dctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'dctrl'
    u = jnp.array([ctrl[0], dctrl[0]])
    return u

@gen
def quintic_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    dctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'dctrl'
    ddctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'ddctrl'
    u = jnp.array([ctrl[0], dctrl[0], ddctrl[0]])
    return u

@jax.jit
def cost_fn(data: mjx.Data, _u: jax.Array, _th: PParams, _t: float) -> float:
    p, theta1, theta2 = data.qpos
    dp, dtheta1, dtheta2 = data.qvel
    
    # Angular deviations from upright (more direct formulation)
    angle_cost1 = 1 + jnp.cos(theta1)
    angle_cost2 = 1 - jnp.cos(theta2)
    
    # Position and velocity costs with different weights
    pos_cost = 0.1 * p**2    # light penalty on cart position
    vel_cost = 0.02 * dp**2  # light penalty on cart velocity
    
    # Angular velocity costs (prevent excessive spinning)
    ang_vel_cost1 = 0.01 * dtheta1**2
    ang_vel_cost2 = 0.01 * dtheta2**2
    
    total_cost = (
        angle_cost1 + angle_cost2 +    # Primary objective: keep upright
        pos_cost + vel_cost +          # Secondary: control cart
        ang_vel_cost1 + ang_vel_cost2  # + Prevent excessive motion
    )
    
    return total_cost

@jax.jit
def dynamics_fn(x: mjx.Data, u: jax.Array, theta: PParams, t: float) -> mjx.Data:
    x = x.replace(ctrl=u[None,0])
    return jax.jit(mjx.step)(theta.mjx_model, x), StateTracker(x.qpos, x.qvel, u)

K = 100000
DEG = 3
H, L = 64, 16
SIGMA = 0.1
assert H*L == 1024

# Mock Test
# d_space = [1, 3]
# k_space = [16, 32]
# hl_space = [[8, 1], [2, 4], [1, 8]]
# s_space = [SIGMA]

# One
# d_space = [DEG]
# k_space = [K]
# hl_space = [[H,L]]
# s_space = [SIGMA]

# Full
d_space = [1, 3, 5]
k_space = [2**n for n in range(11,17)] + [100000]
# hl_space = [[2**n, 2**(10-n)] for n in range(11)]
hl_space = [[2**n, 2**(10-n)] for n in range(5,11)]
s_space = [2**n for n in range(-4,2)]

# def load_mjx_mpc(name: str):
#     parent = importlib.import_module(f"mjx_mod")
#     module = getattr(parent, name)
#     data = {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}
#     raise NotImplementedError

def init_mjx_mpc(deg: int = DEG, l: int = L):
    """
    Build an MPC transition model with a proposal matched to `deg` and `l`.

    deg: 1 (linear), 3 (cubic), or 5 (quintic)
    l:   segment length
    """
    if l is None:
        try:
            l = L
        except NameError as e:
            raise ValueError("`l` not provided and global `L` is undefined.") from e

    proposal_map = {
        1: linear_proposal_fn,
        3: cubic_proposal_fn,
        5: quintic_proposal_fn,
    }

    try:
        proposal_fn = proposal_map[deg]
    except KeyError:
        valid = ", ".join(map(str, sorted(proposal_map)))
        raise ValueError(
            f"Unsupported degree: deg={deg}. Valid options are {{{valid}}} "
            "(1=linear, 3=cubic, 5=quintic)."
        )

    # basic sanity
    for name, fn in [("proposal_fn", proposal_fn), ("cost_fn", cost_fn), ("dynamics_fn", dynamics_fn)]:
        if not callable(fn):
            raise TypeError(f"{name} must be callable; got {type(fn).__name__}")

    return make_transition_model(proposal_fn, cost_fn, dynamics_fn, l=l)

_ffmt = lambda x: f"{float(x):g}".replace(".", "p")


import itertools, json, os, tempfile, time, traceback
from pathlib import Path
from typing import Any, Sequence, Tuple

def _to_jsonable(x):
    """Recursively convert a pytree to JSON-safe types."""
    import numpy as _np
    import jax.numpy as _jnp
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, _np.generic):  # numpy scalar -> python scalar
        return x.item()
    if isinstance(x, (_np.ndarray, _jnp.ndarray)):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return str(x)  # fallback

def _atomic_write_json(obj, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, allow_nan=True)
            f.write("\n")
        os.replace(tmp, str(path))
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass

def save_trajs_npz_named(trajs: Any, path) -> None:
    """
    If `trajs` is a NamedTuple, save fields by name.
    """
    arrays = {}
    for f in trajs._fields:
        arrays[f] = np.asarray(getattr(trajs, f))
    np.savez_compressed(path, **arrays)

def load_trajs_npz_named(path, as_namedtuple: Type | None = None):
    """
    Load .npz to:
      - dict[str, np.ndarray] (default), or
      - a NamedTuple instance if `as_namedtuple` is given (fields must match keys).
    """
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}

    if as_namedtuple is not None:
        # Ensure fields exist (no silent mismatch)
        missing = [f for f in as_namedtuple._fields if f not in d]
        if missing:
            raise KeyError(f"Missing fields in NPZ for {as_namedtuple.__name__}: {missing}")
        # Extra keys are ignored
        return as_namedtuple(**{f: d[f] for f in as_namedtuple._fields})

    return d  # plain dict


def evaluate_one(name: str, k: int, deg: int, h: int, l: int, s: float, n_samples=10):
    results = {
        'name': name,
        'deg': deg,
        'k' : k,
        'h' : h,
        'l': l,
        'sigma': s
    }
    key = jax.random.key(42)
    (mj_model, mj_data) = load_mj(name)
    (x0, theta) = init_mjx(mj_model, mj_data)
    u0 = jnp.zeros((deg+1)//2)
    theta = theta._replace(sigma = s)
    pfmp = PFMP(init_mjx_mpc(deg, l))

    # run and profile algorithm
    t0 = time.time()
    (states, ancestors) = pfmp.run_sir(key, x0, u0, theta, h=h, k=k, rho=0.8)
    t1 = time.time()
    results["runtime"] = t1 - t0

    # reconstruct trajectories
    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(n_samples))
    results["sample_trajs"] = jax.tree.map(lambda x: x[:n_samples], trajs)

    # time grid as float
    t_seq = jnp.arange(h * l, dtype=jnp.float32) / jnp.float32(l)

    # sum over T for a single trajectory (leaves shaped (T, ...))
    traj_cost_fn = lambda tau: jnp.sum(
        jax.vmap(cost_fn, in_axes=(0, 0, None, 0))(tau, tau.ctrl, theta, t_seq)
    )

    # mean over K of per-trajectory totals (leaves shaped (K, T, ...))
    average_cost_fn = lambda trajs_flat: jnp.mean(
        jax.vmap(traj_cost_fn, in_axes=0)(trajs_flat)
    )

    # (K,H,L,...) -> (K,H*L,...)
    trajs_flat = jax.tree.map(
        lambda x: x.reshape((x.shape[0], h * l) + x.shape[3:]),
        trajs,
    )

    results["average_cost"] = average_cost_fn(trajs_flat)

    return results

def evaluate_mjx(
    results_dir: str | Path,
    name: str,
    k_space: Sequence[int],
    d_space: Sequence[int],
    hl_space: Sequence[Tuple[Any, Any]],
    s_space: Sequence[int],
    resume: bool = True,
    log_exceptions: bool = True,
) -> None:
    """
    Runs evaluate_one(name, k, deg, h, l, s) over the grid and saves:
      - Indented JSON profiling/results to:  {name}_k{…}_d{…}_h{…}_l{…}_s{…}.json
      - Sample trajectories (if present) to: {name}_trajs_n{n}_k{…}_d{…}_h{…}_l{…}_s{…}.npz
    Notes:
      - Results are converted to host NumPy before JSON conversion.
      - Writes are atomic; resume=True skips existing JSON files.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    total_exp = len(k_space) * len(d_space) * len(hl_space) * len(s_space)

    n = 0
    start_time = time.time()

    for k, deg, (h, l), s in itertools.product(k_space, d_space, hl_space, s_space):
        n += 1
        params_str = f"_k{k}_d{deg}_h{h}_l{l}_s{_ffmt(s)}"
        json_name  = f"{name}{params_str}.json"
        json_path  = results_dir / json_name

        print(f"[{n}/{total_exp}] k={k}; deg={deg}; h={h}; l={l}; sigma={s}")
        if resume and json_path.exists():
            print(f"SKIP (exists): {json_name}")
            continue

        t0 = time.time()
        try:
            results = evaluate_one(name, k, deg, h, l, s)

            # peel off potentially large trajectories; save to npz
            sample_trajs = results.pop("sample_trajs", None)
            n_samples = sample_trajs[0].shape[0]

            payload = {
                "meta": {
                    "name": name,
                    "k": k,
                    "deg": deg,
                    "h": h,
                    "l": l,
                    "sigma": s,
                    "n_samples": n_samples,
                    "timestamp": time.time(),
                    "elapsed_sec": None,  # filled below
                    "jax_version": getattr(jax, "__version__", "unknown"),
                    "mujoco_version": getattr(mujoco, "__version__", "unknown"),
                },
                # convert remaining results to NumPy → JSON-safe
                "results": _to_jsonable(results),
            }
            payload["meta"]["elapsed_sec"] = time.time() - t0

            trajs_fname = f"{name}_trajs_n{n_samples}{params_str}.npz"
            trajs_path  = results_dir / trajs_fname
            if sample_trajs is not None:
                save_trajs_npz_named(sample_trajs, trajs_path)
                payload["meta"]["trajs_file"] = trajs_fname
                # Add a tiny schema preview for convenience (optional)
                with np.load(trajs_path) as z:
                    payload["meta"]["trajs_fields"] = [
                        {"key": k, "shape": list(z[k].shape), "dtype": str(z[k].dtype)}
                        for k in sorted(z.files)
                    ]

            _atomic_write_json(payload, json_path, indent=2)
            print(f"  → saved {json_name} in {payload['meta']['elapsed_sec']:.2f}s")
            print(f"    cost = {results["average_cost"]}")

        except Exception as e:
            msg = f"ERROR [{n}/{total_exp}] {json_name}: {e}"
            print(msg)
            if log_exceptions:
                (results_dir / (json_name + ".error.txt")).write_text(
                    msg + "\n" + traceback.format_exc()
                )
            # raise  # uncomment to hard-fail on first error

    print(f"Done {n}/{total_exp} in {time.time() - start_time:.1f}s → {results_dir}")

# def evaluate_mjx(
#     results_dir: str | Path,
#     name: str,
#     d_space: Sequence[int],
#     k_space: Sequence[int],
#     hl_space: Sequence[Tuple[Any, Any]],
#     s_space: Sequence[int],
#     *,
#     resume: bool = True,
#     log_exceptions: bool = True,
# ) -> None:
#     """
#     Runs evaluate_one(name, k, deg, h, l, s) over the grid and saves pickled results.
#     - Constant, safe filenames: {name}_k{k}_d{deg}_h{h}_l{l}_s{s}.pkl
#     - Results are converted to host NumPy before pickling.
#     - Atomic writes; resume skips existing files.
#     """
#     results_dir = Path(results_dir)
#     results_dir.mkdir(parents=True, exist_ok=True)

#     # Precompute total experiments
#     total_exp = len(k_space) * len(d_space) * len(hl_space) * len(s_space)

#     n = 0
#     start_time = time.time()

#     for k, deg, (h, l), s in itertools.product(k_space, d_space, hl_space, s_space):
#         n += 1
#         fname = f"{name}_k{k}_d{deg}_h{h}_l{l}_s{_ffmt(s)}.pkl"
#         fpath = results_dir / fname

#         if resume and fpath.exists():
#             print(f"[{n}/{total_exp}] SKIP (exists): {fname}")
#             continue

        
#         print(f"[{n}/{total_exp}] k={k}, deg={deg}, h={h}, l={l}, σ={s}")
#         t0 = time.time()
#         try:
#             results = evaluate_one(name, k, deg, h, l, s)

#             payload = {
#                 "meta": {
#                     "name": name,
#                     "k": k,
#                     "deg": deg,
#                     "h": h,
#                     "l": l,
#                     "s": s,
#                     "timestamp": time.time(),
#                     "elapsed_sec": None,  # filled below
#                     "jax_version": getattr(jax, "__version__", "unknown"),
#                 },
#                 "results": _to_numpy_pytree(results),
#             }
#             payload["meta"]["elapsed_sec"] = time.time() - t0

#             _atomic_pickle(payload, fpath)
#             print(f"  → saved {fname} in {payload['meta']['elapsed_sec']:.2f}s")

#         except Exception as e:
#             msg = f"ERROR [{n}/{total_exp}] {fname}: {e}"
#             print(msg)
#             if log_exceptions:
#                 (results_dir / (fname + ".error.txt")).write_text(
#                     msg + "\n" + traceback.format_exc()
#                 )
#             # Optionally continue; re-raise if you prefer hard fail
#             # raise

#     print(f"Done {n}/{total_exp} in {time.time() - start_time:.1f}s → {results_dir}")



def test_mjx():
    N = 10

    key = jax.random.key(42)
    (mj_model, mj_data) = load_mj("inverted_double_pendulum")
    (x0, theta) = init_mjx(mj_model, mj_data)
    u0 = jnp.zeros((DEG+1)//2)
    pfmp = PFMP(init_mjx_mpc(DEG, L))
    t0 = time.time()
    (states, ancestors) = pfmp.run_sir(key, x0, u0, theta, h=H, k=K, rho=0.8)
    print(time.time() - t0)

    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(N))
    print(trajs.qpos.shape)
    fig, ax = plt.subplots(figsize=(8, 8))
    traj = pfmp.choose_path(states, ancestors, 0)

    # for k in range(N):
    #     u_segs = []
    #     (_, u0, _) = init_mjx(mj_model, mj_data)
    #     u_knots = jnp.concatenate((u0[None,0], trajs[k,:,0]), axis=0)

    #     for t in range(H):
    #         u1 = trajs[k, t]
    #         u_seg = hermite_spline(u0[...,None], u1[...,None], t, t+1, L+1)
    #         u_segs.append(u_seg[0,1:])
    #         u0 = u1

    #     ax.scatter(jnp.arange(H+1), u_knots, color='magenta', zorder=10)

    #     u_interpolated = jnp.concatenate(u_segs)
    #     ax.plot(jnp.arange(u_interpolated.shape[0])/L, u_interpolated, color='magenta', alpha=0.5, ls='dotted')

    # ax.set_title("Hermite-Interpolated Control (u₁ vs u₂)")
    # ax.set_xlabel("u₁")
    # ax.set_ylabel("u₂")
    # ax.grid(False)
    # plt.tight_layout()
    # plt.show()

    traj_flat = jax.tree.map(lambda x: x.reshape((-1,)+x.shape[2:]), traj)
    total_cost_fn = lambda tau: jnp.sum(jax.vmap(cost_fn, in_axes=[0,None,None,0])(tau, tau.ctrl, theta, jnp.arange(H*L)/L))
    total_cost = total_cost_fn(traj_flat)
    print(f"cost: {total_cost}")
    with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        for t in range(traj.qpos.shape[0]):
            for s in range(L):
                mj_data.qpos[:] = jax.device_get(traj.qpos[t,s])
                mj_data.qvel[:] = jax.device_get(traj.qvel[t,s])
                mujoco.mj_forward(mj_model, mj_data)
                time.sleep(0.01)
                viewer.sync()

    # t = 0
    x = x0

import os
def load_mj(name: str):
    os.chdir("assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    return (mj_model, mj_data)

if __name__ == "__main__":
    # test_mjx()
    name = "inverted_double_pendulum"
    # results = evaluate_one(name, k_space[0], d_space[0], *hl_space[0], s_space[0])
    # print(results)
    print(f"k_space: {k_space}")
    print(f"d_space: {d_space}")
    print(f"hl_space: {hl_space}")
    print(f"s_space: {s_space}")
    # evaluate_mjx("experiments/spline_grid", "inverted_double_pendulum", k_space, d_space, hl_space, s_space)
    # exit()


    # Replay
    (mj_model, mj_data) = load_mj("inverted_double_pendulum")
    (x0, theta) = init_mjx(mj_model, mj_data)
    # k = K; deg = DEG; h = H; l = L; s = SIGMA

    n_samples = 10

    # REALLY GOOD SWING UP!!
    k=2048; deg=1; h=128; l=8; sigma=0.0625
    # k=2048; deg=1; h=256; l=4; sigma=0.125  

    # cost = 1049.73 better cost, but not very smooth
    k=2048; deg=3; h=256; l=4; sigma=1

    # cost = 954.26 very good cost for low k
    k=2048; deg=5; h=64; l=16; sigma=0.5

    # cost = 688.6336059570312
    k=4096; deg=1; h=128; l=8; sigma=2

    # example of HFQ that works OK
    # k=2048; deg=1; h=1024; l=1; sigma=1

    # cubic (bad)
    # k=2048; deg=3; h=16; l=64; sigma=1

    # quintic (bad)
    # k=2048; deg=5; h=32; l=32; sigma=0.125
    # k=2048; deg=5; h=8; l=128; sigma=0.0625

    # It's sensitive to sigma!
    # k=100000; deg=3; h=64; l=16; sigma=2**(-4)
    # k=100000; deg=3; h=64; l=16; sigma=2**(-3)

    # Best so far (stable)
    # k=32768; deg=5; h=64; l=16; sigma=2.0
    # k=100000; deg=3; h=64; l=16; sigma=0.1
    # k=8192; deg=1; h=256; l=4; sigma=0.25
    # k=32768//2; deg=1; h=128; l=8; sigma=2
    # top_k = OrderedDict({(k, deg, h, l, sigma): 0.}).items()

    # evaluate_mjx(results_dir, name, [k], [deg], [[h,l]], [sigma])

    results_dir = "experiments/spline_grid"
    results_map = {}

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), 'r') as fp:
                results = json.load(fp)["results"]

            param_str = (
                filename.removeprefix(name + '_')
                        .removesuffix(".json")
            )
            params = [v[1:].replace('p', '.') for v in param_str.split('_')]
            (k, deg, h, l) = (int(v) for v in params[:-1])
            sigma = float(params[-1])

            cost = results["average_cost"]
            runtime = results["runtime"]
            results_map[(k, deg, h, l, sigma)] = (cost, runtime)

    # sort by cost into an OrderedDict
    sorted_results = OrderedDict(
        sorted(results_map.items(), key=lambda item: item[1])
    )

    thresh = 400
    n = sum([v[0] <= thresh for (k, v) in sorted_results.items()])
    print(f"{n} of {len(sorted_results.values())} have cost <= {thresh}")
    # exit()

    # get top-K entries
    K = 5
    top_k = list(sorted_results.items())[:K]

    for i, (params, (cost, runtime)) in enumerate(top_k[1:]):
        print(f"Rank #{i+1}")
        print(f"cost: {cost}")
        print(f"runtime: {runtime}")
        print(f"params: {params}")
        (k, deg, h, l, sigma) = params

        trajs = load_trajs_npz_named(f"{results_dir}/{name}_trajs_n{n_samples}_k{k}_d{deg}_h{h}_l{l}_s{_ffmt(sigma)}.npz", as_namedtuple=StateTracker)

        traj = jax.tree.map(lambda x: x[0], trajs)

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 10), sharex=True)
        
        axes[0].set_title(f"PFMP Control and Response")
        axes[0].set_ylabel("$u$")
        axes[1].set_ylabel("$x$")
        axes[2].set_ylabel("$\\theta_1$")
        axes[3].set_ylabel("$\\theta_2$")
        axes[3].set_xlabel("Time (seconds)")

        bbox = axes[0].get_position()          # figure coords
        x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        fig.add_artist(Line2D([x0 - 0.05, x0 + width + 0.05], [y0 + height - 0.185, y0 + height - 0.185],
                            transform=fig.transFigure,
                            color="0.", linewidth=2, clip_on=False))

        u0 = jnp.zeros((DEG+1)//2)
        ctrl = trajs.ctrl[:,:,:,0].reshape(10,-1)
        T = ctrl.shape[1]
        t_seq = jnp.arange(h * l, dtype=jnp.float32) * mj_model.opt.timestep
        for t in range(ctrl.shape[0]):
            axes[0].plot(t_seq, ctrl[t], c='magenta', ls='solid', alpha=0.5)

        for t in range(trajs.qpos.shape[0]):
            q = trajs.qpos[:,:,:,0].reshape(10,-1)
            axes[1].plot(t_seq, q[t], ls='solid', color="red", alpha=0.5)

        for t in range(trajs.qpos.shape[0]):
            q = trajs.qpos[:,:,:,1].reshape(10,-1)
            axes[2].plot(t_seq, q[t], ls='solid', color="green", alpha=0.5)

        for t in range(trajs.qpos.shape[0]):
            q = trajs.qpos[:,:,:,2].reshape(10,-1)
            axes[3].plot(t_seq, q[t], ls='solid', color="blue", alpha=0.5)

        plt.show()

        with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            time.sleep(2.0)
            for t in range(traj.qpos.shape[0]):
                for s in range(traj.qpos.shape[1]):
                    mj_data.qpos[:] = jax.device_get(traj.qpos[t,s])
                    mj_data.qvel[:] = jax.device_get(traj.qvel[t,s])
                    mujoco.mj_forward(mj_model, mj_data)
                    viewer.sync()
                    time.sleep(0.01)