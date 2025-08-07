from typing import NamedTuple
from pfmp import hermite_spline
from pfmp.algorithm_spline import PFMP, make_transition_model
from genjax import gen, normal
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Qt5Agg')


class ParState(NamedTuple):
    ctrl: jax.Array

class ParParams(NamedTuple):
    u_stdev: float
    du_stdev: float
    ddu_stdev: float

@gen
def transition_fn(state: ParState, theta: ParParams, t: int) -> ParState:
    pred_u = normal(0., theta.u_stdev) @ "u"
    pred_du = normal(0., theta.du_stdev) @ "du"
    pred_ddu = normal(0., theta.ddu_stdev) @ "ddu"
    return state._replace(
        # ctrl = jnp.array([pred_u]),                   # linear
        # ctrl = jnp.array([pred_u, pred_du]),          # cubic
        ctrl = jnp.array([pred_u, pred_du, pred_ddu]),  # quintic
    )

@jax.jit
def cost_fn(state: ParState, theta: ParParams, t: float) -> float:
    return (jnp.sin(2*jnp.pi*t/H) - state.ctrl[0]) ** 2

def init():
    x0 = ParState(
        # ctrl = jnp.array([0.]),             # linear
        # ctrl = jnp.array([0., 0.]),       # cubic
        ctrl = jnp.array([0., 0., 0.]), # quintic
    )

    theta = ParParams(
        u_stdev = 0.5,
        du_stdev = 0.5,
        ddu_stdev = 0.5
    )

    return (x0, theta)

K, H, L = 1000, 4, 100

def test_sis():
    model_t = make_transition_model(transition_fn, cost_fn, l = L)
    pfmp = PFMP(model_t)
    (x0, theta) = init()

    key = jax.random.key(42)
    j_run_sis = jax.jit(pfmp.run_sis, static_argnames=['h', 'k'])
    for i in range(5):
        t0 = time.time()
        (trajs, scores) = jax.block_until_ready(j_run_sis(key, x0, theta, H, K))
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")


    # Unnormalized log-weights
    log_w = scores - jax.scipy.special.logsumexp(scores)
    p = jnp.exp(log_w)

    # Draw 10 weighted indices from categorical distribution
    key = jax.random.key(0)
    sampled_idxs = jax.random.choice(key, p.shape[0], shape=(10,), p=p)

    # Use those indices to grab trajectories
    first_10 = trajs.ctrl[:10,:,:,0]  # shape (10, 4, 200)
    # first_10 = trajs.ctrl[sampled_idxs, :, :, 0]  # shape (10, 4, 200)

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
    model_t = make_transition_model(transition_fn, cost_fn, l = L)
    pfmp = PFMP(model_t)
    (x0, theta) = init()

    key = jax.random.key(42)
    j_run_sir = jax.jit(pfmp.run_sir, static_argnames=['h', 'k', 'rho'])
    for i in range(5):
        t0 = time.time()
        (states, ancestors) = jax.block_until_ready(j_run_sir(key, x0, theta, H, K, 0.8))
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    # first_10 = trajs.ctrl[:10,:,:,0]  # shape (10, 4, 200)
    first_10 = pfmp.choose_paths(states, ancestors, jnp.arange(10)).ctrl[:,:,:,0]

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

def cl(t, x, freq=3.0, sharpness=20.0):
    y1 = 1.0 - jnp.exp(-sharpness * jnp.sin(jnp.pi * freq * (t - x))**2)
    y2 = 1.0 - jnp.exp(-sharpness * jnp.sin(jnp.pi * freq * (t + x))**2)
    return y1 * y2  # Peak at diamond centers

def test_soft_lattice():
    t = jnp.linspace(-3, 3, 400)
    x = jnp.linspace(-3, 3, 400)
    T, X = jnp.meshgrid(t, x)

    Z = cl(T, X, freq=0.33, sharpness=5.0)

    plt.imshow(Z, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
    plt.colorbar(label="fence(t, x)")
    plt.title("Smooth Chain-Link Fence Function")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_sis()
    test_sir()