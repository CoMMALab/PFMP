from typing import NamedTuple
from pfmp import PFMP
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

def mpc_dynamics(x, u, theta, t):
    return (u, u)

def mpc_cost(x, u, theta, t):
    return (jnp.sin(2*jnp.pi*t/0.8) - x[0]) ** 2

def mpc_proposal(key, x, u, theta, t):
    keys = jax.random.split(key, 3)
    pred_u = jax.random.normal(keys[0]) * theta.u_stdev
    pred_du = jax.random.normal(keys[1]) * theta.du_stdev
    pred_ddu = jax.random.normal(keys[2]) * theta.ddu_stdev
    return jnp.array([pred_u, pred_du, pred_ddu])  # quintic

def init():
    # x0 = jnp.array([0.])          # linear
    # x0 = jnp.array([0., 0.])      # cubic
    x0 = jnp.array([0., 0., 0.])  # quintic
    u0 = x0

    theta = ParParams(
        u_stdev = 0.5,
        du_stdev = 0.5,
        ddu_stdev = 0.5
    )

    return (x0, u0, theta)

K, H, L = 1000, 4, 100

def test_sis():
    # model_t = make_transition_model(transition_fn, cost_fn, l = L)
    # pfmp = PFMP(model_t)
    # pfmp = Spline1DController()
    # pfmp = PFMP(
    #     mpc_dynamics, mpc_cost, mpc_proposal,
    #     horizon=H, particles=K, steps_per_knot=L)
    horizon = 0.8
    particles = 1000
    num_knots = 5

    pfmp = PFMP(
        mpc_dynamics, mpc_cost, mpc_proposal,
        horizon=0.8, particles=K, num_knots=5)
    (x0, u0, theta) = init()

    key = jax.random.key(42)
    for i in range(5):
        t0 = time.time()
        (trajs, scores) = pfmp.run_sis(key, x0, u0, theta)
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    # Unnormalized log-weights
    log_w = jnp.sum(scores, axis=-1) - jax.scipy.special.logsumexp(jnp.sum(scores, axis=-1))
    p = jnp.exp(log_w)

    # Draw 10 weighted indices from categorical distribution
    key = jax.random.key(0)
    sampled_idxs = jax.random.choice(key, p.shape[0], shape=(10,), p=p)

    # Use those indices to grab trajectories
    # first_10 = trajs[:10,:,:,0]  # shape (10, 4, 200)
    first_10 = trajs[sampled_idxs, :, :, 0]  # shape (10, 4, 200)
    print(first_10.shape)

    fig, ax = plt.subplots(figsize=(14, 6))

    segment_length = first_10.shape[-1]  # 200
    num_segments = first_10.shape[1]     # 4
    num_samples = first_10.shape[0]      # 10
    total_length = segment_length * num_segments

    # === 1. Define a grid over time and control values ===
    t_grid = jnp.linspace(0, total_length - 1, 800)
    u_grid = jnp.linspace(-2, 2, 200)
    T, U = jnp.meshgrid(t_grid, u_grid)

    # === 2. Evaluate cost_fn: cost = -u / 20 ===
    cost_grid = (jnp.sin(2 * jnp.pi * T / (H*L)) - U) ** 2  # shape (200, 800)

    # === 3. Plot cost background as colormap ===
    ax.imshow(
        cost_grid,
        extent=[0, total_length, u_grid[0], u_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='Greys_r',
        alpha=1.0
    )

    # === 4. Plot interpolated control trajectories ===
    for i in range(num_samples):
        for j in range(num_segments):
            t = jnp.arange(segment_length) + j * segment_length
            y = first_10[i, j]
            ax.plot(t, y, color="magenta", alpha=0.5, ls="dotted", lw=2)
            ax.scatter(t[0], y[0], color="magenta", s=100)

    ax.set_title("Hermite-Interpolated Control with Cost Gradient (-u / 20)")
    ax.set_xticks([t*L for t in range(H+1)])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Control (u)")
    ax.grid(axis='x', visible=True)
    plt.tight_layout()
    plt.show()

def test_sir():
    # model_t = make_transition_model(transition_fn, cost_fn, l = L)
    # pfmp = Spline1DController()
    pfmp = PFMP(
        mpc_dynamics, mpc_cost, mpc_proposal,
        horizon=0.8, particles=K, num_knots=5)
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
    first_10 = trajs[:10,:,:,0]

    fig, ax = plt.subplots(figsize=(14, 6))

    segment_length = first_10.shape[-1]  # 200
    num_segments = first_10.shape[1]     # 4
    num_samples = first_10.shape[0]      # 10
    total_length = segment_length * num_segments

    # === 1. Define a grid over time and control values ===
    t_grid = jnp.linspace(0, total_length - 1, 800)
    u_grid = jnp.linspace(-2, 2, 200)
    T, U = jnp.meshgrid(t_grid, u_grid)

    # === 2. Evaluate cost_fn: cost = -u / 20 ===
    cost_grid = (jnp.sin(2 * jnp.pi * T / (H*L)) - U) ** 2  # shape (200, 800)

    # === 3. Plot cost background as colormap ===
    ax.imshow(
        cost_grid,
        extent=[0, total_length, u_grid[0], u_grid[-1]],
        origin='lower',
        aspect='auto',
        cmap='Greys_r',
        alpha=1.0
    )

    # === 4. Plot interpolated control trajectories ===
    for i in range(num_samples):
        for j in range(num_segments):
            t = jnp.arange(segment_length) + j * segment_length
            y = first_10[i, j]
            ax.plot(t, y, color="magenta", alpha=0.5, ls="dotted", lw=2)
            ax.scatter(t[0], y[0], color="magenta", s=100)

    ax.set_title("Hermite-Interpolated Control with Cost Gradient (-u / 20)")
    ax.set_xticks([t*L for t in range(H+1)])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Control (u)")
    ax.grid(axis='x', visible=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_sis()
    test_sir()