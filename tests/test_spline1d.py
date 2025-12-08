from typing import NamedTuple
from pfmp import PFMP
from pfmp.interpolate import Jet3
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Qt5Agg')


class ParParams(NamedTuple):
    u_stdev: float
    du_stdev: float
    ddu_stdev: float

def mpc_dynamics(x, u: jax.Array, theta, t):
    return (u.u[0], u)

def mpc_cost(x, u, theta, t):
    return (jnp.sin(2*jnp.pi*t/4.0) - x) ** 2

terminal_cost = lambda _x, _th: 0.

def mpc_init(key, x, theta):
    keys = jax.random.split(key, 3)
    pred_u = jax.random.normal(keys[0]) * theta.u_stdev
    pred_du = jax.random.normal(keys[1]) * theta.du_stdev
    pred_ddu = jax.random.normal(keys[2]) * theta.ddu_stdev
    return Jet3(pred_u, pred_du, pred_ddu)

def mpc_proposal(key, x, u, theta, t):
    keys = jax.random.split(key, 3)
    pred_u = jax.random.normal(keys[0]) * theta.u_stdev
    pred_du = jax.random.normal(keys[1]) * theta.du_stdev
    pred_ddu = jax.random.normal(keys[2]) * theta.ddu_stdev
    return Jet3(pred_u, pred_du, pred_ddu)

# K, H, L = 10000, 4, 100
horizon = 4.0
particles = 100000
num_knots = 5

def test_sis():
    # model_t = make_transition_model(transition_fn, cost_fn, l = L)
    # pfmp = PFMP(model_t)
    # pfmp = Spline1DController()
    # pfmp = PFMP(
    #     mpc_dynamics, mpc_cost, mpc_proposal,
    #     horizon=H, particles=K, steps_per_knot=L)

    pfmp = PFMP(
        mpc_dynamics, mpc_cost, terminal_cost, mpc_init, mpc_proposal,
        horizon, num_knots, particles)

    x0 = 0.
    theta = ParParams(
        u_stdev = 1.0,
        du_stdev = 2.0,
        ddu_stdev = 0.5
    )

    key = jax.random.key(42)
    for i in range(5):
        t0 = time.time()
        (trajs, scores) = pfmp.run_sis(key, x0, theta)
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    # Unnormalized log-weights
    log_w = jnp.sum(scores, axis=-1) - jax.scipy.special.logsumexp(jnp.sum(scores, axis=-1))
    p = jnp.exp(log_w)

    # ------------------------
    # 1. Sample trajectories
    # ------------------------
    nsamples = 10
    idxs = jax.random.choice(key, p.shape[0], shape=(nsamples,), p=p)
    U = trajs.u[idxs, ...]        # (nsamples, nseg, seglen)

    ns, nseg = U.shape
    assert nseg == num_knots - 1, "Inconsistent num_knots vs trajectory shape"

    # ------------------------
    # 2. Cost background grid
    # ------------------------
    t_grid = jnp.linspace(0, horizon, 800)
    u_grid = jnp.linspace(-2, 2, 200)

    T, Umesh = jnp.meshgrid(t_grid, u_grid)
    cost_grid = mpc_cost(x=Umesh, u=None, theta=None, t=T)

    # ------------------------
    # 3. Plot cost heatmap
    # ------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        cost_grid,
        extent=[0, horizon, u_grid[0], u_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='Greys_r',
        alpha=1.0
    )

    # ------------------------
    # 4. Plot sampled spline controls
    # ------------------------
    # Each Hermite segment spans dt = horizon / (num_knots - 1)
    dt = pfmp.dt

    nseg = num_knots - 1
    dt = horizon / nseg               # physical segment duration
    sample_dt = dt / seglen           # time between discrete samples

    for i in range(ns):
        for j in range(nseg):
            # Rescaled time positions inside the horizon
            t_local = j * dt + jnp.arange(seglen) * sample_dt
            y_local = U[i, j]
            ax.plot(t_local, y_local, color="magenta",
                    alpha=0.5, ls="dotted", lw=2)
            ax.scatter(t_local[0], y_local[0], color="magenta", s=40)

    ax.set_title("Sampled Control Trajectories with MPC Cost Background")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control u")
    ax.grid(axis="x", visible=True)
    plt.tight_layout()
    plt.show()

def test_sir():
    # model_t = make_transition_model(transition_fn, cost_fn, l = L)
    # pfmp = Spline1DController()
    pfmp = PFMP(
        mpc_dynamics, mpc_cost, terminal_cost, mpc_proposal,
        horizon, num_knots, particles)
    (x0, u0, theta) = init()

    key = jax.random.key(42)
    for i in range(5):
        t0 = time.time()
        # (states, ancestors) = pfmp.run_sir(key, x0, u0, theta)
        trajs, scores = pfmp.run_sir(key, x0, u0, theta)
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    # first_10 = trajs.u[:10,:,:,0]  # shape (10, 4, 200)
    # first_10 = pfmp.choose_paths(states, ancestors, jnp.arange(10))[:,:,:,0]
    U = trajs.u[:10,:,:,0]
    ns, nseg, seglen = U.shape
    assert nseg == num_knots - 1, "Inconsistent num_knots vs trajectory shape"

    # ------------------------
    # 2. Cost background grid
    # ------------------------
    t_grid = jnp.linspace(0, horizon, 800)
    u_grid = jnp.linspace(-2, 2, 200)

    T, Umesh = jnp.meshgrid(t_grid, u_grid)
    cost_grid = mpc_cost(x=Umesh, u=None, theta=None, t=T)

    # ------------------------
    # 3. Plot cost heatmap
    # ------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        cost_grid,
        extent=[0, horizon, u_grid[0], u_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='Greys_r',
        alpha=1.0
    )

    # ------------------------
    # 4. Plot sampled spline controls
    # ------------------------
    # Each Hermite segment spans dt = horizon / (num_knots - 1)
    dt = pfmp.dt

    nseg = num_knots - 1
    dt = horizon / nseg               # physical segment duration
    sample_dt = dt / seglen           # time between discrete samples

    for i in range(ns):
        for j in range(nseg):
            # Rescaled time positions inside the horizon
            t_local = j * dt + jnp.arange(seglen) * sample_dt
            y_local = U[i, j]
            ax.plot(t_local, y_local, color="magenta",
                    alpha=0.5, ls="dotted", lw=2)
            ax.scatter(t_local[0], y_local[0], color="magenta", s=40)

    ax.set_title("Sampled Control Trajectories with MPC Cost Background")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control u")
    ax.grid(axis="x", visible=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_sis()
    test_sir()