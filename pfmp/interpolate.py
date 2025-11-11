import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


# ----- constant coefficient matrices -----
POS_LIN = jnp.array([[1., -1.],
                     [0.,  1.]])
VEL_LIN = jnp.array([[-1., 0.],
                     [ 1., 0.]])
ACC_LIN = jnp.array([[0., 0.],
                     [0., 0.]])

POS_CUB = jnp.array([[ 1.,  0., -3.,  2.],
                     [ 0.,  1., -2.,  1.],
                     [ 0.,  0.,  3., -2.],
                     [ 0.,  0., -1.,  1.]])
VEL_CUB = jnp.array([[ 0., -6.,  6.,  0.],
                     [ 1., -4.,  3.,  0.],
                     [ 0., -6.,  6.,  0.],
                     [ 0., -2.,  3.,  0.]])
ACC_CUB = jnp.array([[-6., 12., 0., 0.],
                     [-4.,  6., 0., 0.],
                     [-6., 12., 0., 0.],
                     [-2.,  6., 0., 0.]])

POS_QUI = jnp.array([[ 1.,  0.,  0., -10.,  15.,  -6.],
                     [ 0.,  1.,  0.,  -6.,   8.,  -3.],
                     [ 0.,  0.,  0.5, -1.5,  1.5, -0.5],
                     [ 0.,  0.,  0.,  10., -15.,   6.],
                     [ 0.,  0.,  0.,  -4.,   7.,  -3.],
                     [ 0.,  0.,  0.,   0.5, -1.,   0.5]])
VEL_QUI = jnp.array([[ 0.,  0., -30.,  60., -30.,  0.],
                     [ 1.,  0., -18.,  32., -15.,  0.],
                     [ 0.,  1.,  -4.5,  6.,  -2.5,  0.],
                     [ 0.,  0.,  30., -60.,  30.,  0.],
                     [ 0.,  0., -12.,  28., -15.,  0.],
                     [ 0.,  0.,   1.5, -4.,   2.5,  0.]])
ACC_QUI = jnp.array([[ 0., -60., 180., -120.,   0.,   0.],
                     [ 0., -36.,  96.,  -60.,   0.,   0.],
                     [ 1.,  -9.,  18.,  -10.,   0.,   0.],
                     [ 0.,  60., -180., 120.,   0.,   0.],
                     [ 0., -24.,  84.,  -60.,   0.,   0.],
                     [ 0.,   3., -12.,   10.,   0.,   0.]])

BASES = (
    (POS_LIN, VEL_LIN, ACC_LIN),  # m = 1
    (POS_CUB, VEL_CUB, ACC_CUB),  # m = 2
    (POS_QUI, VEL_QUI, ACC_QUI),  # m = 3
)

@partial(jax.jit, static_argnames=("nout"))
def hermite_eval(
    s: float, J0: jax.Array, J1: jax.Array, nout: int = 1
) -> jax.Array:
    # coerce jets to (m, D)
    a = jnp.asarray(J0)
    b = jnp.asarray(J1, dtype=a.dtype)
    if a.ndim == 0:
        A = a.reshape(1, 1); B = b.reshape(1, 1)
    elif a.ndim == 1:
        A = a[:, None];      B = b[:, None]
    elif a.ndim == 2:
        A = a;               B = b
    else:
        raise ValueError("J0/J1 must be scalar, 1D, or 2D (m,D).")
    if A.shape != B.shape:
        raise ValueError(f"J0 and J1 must match; got {A.shape} vs {B.shape}.")

    m = A.shape[0]                 # 1/2/3
    POS, _, _ = BASES[m - 1]       # (2m,2m), increasing powers
    N = POS.shape[1]               # 2/4/6

    # differentiation matrix for increasing powers (subdiagonal)
    one_to_Nm1 = jnp.arange(1, N, dtype=POS.dtype)
    D = jnp.zeros((N, N), dtype=POS.dtype).at[jnp.arange(1, N), jnp.arange(N - 1)].set(one_to_Nm1)

    s = jnp.asarray(s, dtype=A.dtype)          # scalar
    poly = lambda C: jnp.polyval(jnp.flip(C), s)

    J = jnp.concatenate([A, B], axis=0)        # (N, D)

    coeff = POS
    outs = []
    for _ in range(nout):                      # unrolled (nout static)
        Bvals = jax.vmap(poly, in_axes=0, out_axes=-1)(coeff)   # (N,)
        outs.append(Bvals @ J)                                  # (D,)
        coeff = coeff @ D

    # v1
    out = outs[0] if nout == 1 else jnp.stack(outs, axis=0)     # (D,) or (nout, D)
    return out[..., 0] if out.shape[-1] == 1 else out           # drop D==1 channel

    # v2
    # out = jnp.stack(outs, axis=0)     # (nout, D)
    # return out

    # v3
    # out = outs[0] if nout == 1 else jnp.stack(outs, axis=0)     # (D,) or (nout, D)
    # return out


@partial(jax.jit, static_argnames=("ord", "nout"))
def hermite_spline(
    tq: jax.Array, tk: jax.Array, knots: jax.Array, ord: int, nout: int = 1
) -> jax.Array:
    """Evaluate Hermite spline at query times.
    
    Args:
        tq: Query times, shape (Q,)
        tk: Knot times, shape (K,) - must be sorted
        knots: Knot values, shape (K, m, D) where m=(ord+1)//2
        ord: Order (1=linear, 3=cubic, 5=quintic)
        nout: Number of derivatives to output (1=value only, 2=value+derivative, etc.)
    
    Returns:
        If nout==1: shape (Q, D)
        If nout>1: shape (nout, Q, D)
    """
    tq = jnp.asarray(tq)
    tk = jnp.asarray(tk)
    knots = jnp.asarray(knots)
    
    # Validate inputs
    K = tk.shape[0]
    if knots.shape[0] != K:
        raise ValueError(f"knots must have {K} entries to match tk")
    
    m = (ord + 1) // 2
    if knots.shape[1] != m:
        raise ValueError(f"For ord={ord}, knots must have shape (K, {m}, D)")
    
    D = knots.shape[2]
    Q = tq.shape[0]
    
    # Find which segment each query point belongs to
    # searchsorted gives index i such that tk[i-1] <= tq < tk[i]
    seg_idx = jnp.searchsorted(tk, tq, side='right') - 1
    seg_idx = jnp.clip(seg_idx, 0, K - 2)  # Clamp to valid segments
    
    def eval_single(t, idx):
        """Evaluate spline at single time t in segment idx"""
        t0 = tk[idx]
        t1 = tk[idx + 1]
        delta = t1 - t0
        
        J0 = knots[idx]      # (m, D)
        J1 = knots[idx + 1]  # (m, D)
        
        # Scale derivatives by segment length
        scale = (delta ** jnp.arange(m, dtype=knots.dtype))[:, None]  # (m, 1)
        J0s = J0 * scale
        J1s = J1 * scale
        
        s = (t - t0) / delta
        return hermite_eval(s, J0s, J1s, nout)
    
    # Vectorize over query points
    result = jax.vmap(eval_single)(tq, seg_idx)
    
    return result


@partial(jax.jit, static_argnames=("ord", "num", "nout"))
def hermite_interp1d(
    t0: float, t1: float, J0: jax.Array, J1: jax.Array, ord: int, num: int, nout: int = 1
) -> jax.Array:
    """Sample on [t0,t1].
    Returns:
      t_grid : (num,)
      JJ     : if nout==1 → (num, D); else → (nout, num, D)
    """
    t0 = jnp.asarray(t0, dtype=J0.dtype)
    t1 = jnp.asarray(t1, dtype=J0.dtype)
    t_grid = jnp.linspace(t0, t1, num, dtype=J0.dtype)

    delta  = t1 - t0
    m = (ord + 1) // 2
    scale = (delta ** jnp.arange(m, dtype=J0.dtype))[:, None]  # (m,1) → [1, Δ, Δ²]
    J0s = J0[:m].reshape(m, -1) * scale
    J1s = J1[:m].reshape(m, -1) * scale

    return t_grid, hermite_eval((t_grid - t0) / delta, J0s, J1s, nout)


L = 512

def test_spline_1d():
    """Test the new spline interface with 1D data"""
    # Define 4 knots at t = [0, 0.3, 0.7, 1.0]
    tk = jnp.array([0.0, 0.3, 0.7, 1.0])
    
    # Knots with [value, velocity, acceleration] for quintic
    knots = jnp.array([
        [[-1.2], [22.9], [-279.8]],   # knot 0
        [[0.5],  [5.0],  [-50.0]],     # knot 1
        [[0.8],  [-3.0], [100.0]],     # knot 2
        [[0.2],  [2.9],  [35.5]],      # knot 3
    ])  # shape (4, 3, 1)
    
    # Query at many points
    tq = jnp.linspace(0, 1, L)
    
    # Evaluate with different orders
    y1 = hermite_spline(tq, tk, knots[:, :1, :], ord=1)  # linear (only position)
    y3 = hermite_spline(tq, tk, knots[:, :2, :], ord=3)  # cubic (pos + vel)
    y5 = hermite_spline(tq, tk, knots, ord=5)             # quintic (pos + vel + acc)
    
    plt.figure()
    plt.plot(tq, y1, label="Linear Hermite", alpha=0.7)
    plt.plot(tq, y3, label="Cubic Hermite", alpha=0.7)
    plt.plot(tq, y5, label="Quintic Hermite", alpha=0.7)
    plt.scatter(tk, knots[:, 0, 0], color='red', s=50, zorder=5, label='Knots')
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Hermite spline with 4 knots (1D)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_spline_2d():
    """Test the new spline interface with 2D trajectory"""
    # Define 5 knots
    tk = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # 2D trajectory knots with [pos, vel, acc]
    knots = jnp.array([
        [[-1.2, 0.0],   [25.9, -2.3],  [-279.8, 3.0]],    # knot 0
        [[0.0, -0.5],   [10.0, 5.0],   [-50.0, 20.0]],    # knot 1
        [[0.5, 1.0],    [-5.0, 8.0],   [100.0, -30.0]],   # knot 2
        [[0.3, 1.5],    [3.0, 2.0],    [-20.0, 10.0]],    # knot 3
        [[0.2, 2.0],    [2.9, 3.84],   [35.5, -2.22]],    # knot 4
    ])  # shape (5, 3, 2)
    
    tq = jnp.linspace(0, 1, L)
    
    xy1 = hermite_spline(tq, tk, knots[:, :1, :], ord=1)
    xy3 = hermite_spline(tq, tk, knots[:, :2, :], ord=3)
    xy5 = hermite_spline(tq, tk, knots, ord=5)
    
    plt.figure()
    plt.plot(xy1[:, 0], xy1[:, 1], label="linear", alpha=0.7)
    plt.plot(xy3[:, 0], xy3[:, 1], label="cubic", alpha=0.7)
    plt.plot(xy5[:, 0], xy5[:, 1], label="quintic", alpha=0.7)
    plt.scatter(knots[:, 0, 0], knots[:, 0, 1], color='red', s=50, zorder=5, label='Knots')
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Hermite spline curve in R^2 (5 knots)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_1d():
    # Test data 1D
    J0 = jnp.array([-1.2, 22.9, -279.8])  # [y0, v0, a0]
    J1 = jnp.array([ 0.2,  2.9,   35.5])  # [y0, v0, a0]

    print(hermite_eval(0.3, -1.0, 4.0, nout=1))
    print(hermite_eval(jnp.array([0.3, 0.4, 0.5]), -1.0, 4.0, nout=1))
    print(hermite_eval((0.5, 0.6), jnp.array([-1.0, 2.0]), jnp.array([1.0, 5.0]), nout=1))
    print(hermite_eval((0.5, 0.6), jnp.array([-1.0, 2.0]), jnp.array([1.0, 5.0]), nout=2))
    print(hermite_eval(0.5, jnp.array([-1.0]), jnp.array([5.0]), nout=3))

    # Evaluate
    t_grid, y1 = hermite_interp1d(0, 1, J0, J1, ord=1, num=L)
    plt.plot(t_grid, y1, label="Linear Hermite")

    _, y3 = hermite_interp1d(0, 1, J0, J1, ord=3, num=L)
    plt.plot(t_grid, y3, label="Cubic Hermite")

    _, y5 = hermite_interp1d(0, 1, J0, J1, ord=5, num=L)
    plt.plot(t_grid, y5, label="Quintic Hermite")

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Hermite interpolation on [0,1]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_2d():
    # Test data 2D
    J0 = jnp.array([
        [  -1.2 , 0.0],  # [px0, py0]
        [  25.9, -2.3],  # [vx0, vy0]
        [-279.8 , 3.0],  # [ax0, axy]
    ])
    J1 = jnp.array([
        [ 0.2,  2.0],    # [px0, py1]
        [ 2.9,  3.84],   # [vx0, vy1]
        [35.5, -2.22],   # [ax0, ay1]
    ])

    _, xy1 = hermite_interp1d(0, 1, J0, J1, ord=1, num=L)
    plt.plot(xy1[:,0], xy1[:,1], label="linear")

    _, xy3 = hermite_interp1d(0, 1, J0, J1, ord=3, num=L)
    plt.plot(xy3[:,0], xy3[:,1], label="cubic")

    _, xy5 = hermite_interp1d(0, 1, J0, J1, ord=5, num=L)
    plt.plot(xy5[:,0], xy5[:,1], label="quintic")

    plt.xlabel("y0(t)")
    plt.ylabel("y1(t)")
    plt.title("Hermite curve in R^2")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_7d():
    # --- Sample normally-distributed 7-DoF jets (value, first-derivative, second-derivative) ---
    key = jax.random.PRNGKey(0)  # change seed for different draws
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Means (you can tweak these). Here: start near 0, end near 1.
    mu_y0 = jnp.zeros(7)
    mu_y1 = jnp.ones(7)

    # Zero-mean velocities/accelerations; tune stds to taste (normalized t-domain)
    mu_v0 = jnp.zeros(7); mu_v1 = jnp.zeros(7)
    mu_a0 = jnp.zeros(7); mu_a1 = jnp.zeros(7)

    # Standard deviations
    sigma_y = 0.25
    sigma_v = 1.0
    sigma_a = 3.0

    # Draw normals per component
    y0  = mu_y0 + sigma_y * jax.random.normal(k1, (7,))
    y1  = mu_y1 + sigma_y * jax.random.normal(k2, (7,))
    v0  = mu_v0 + sigma_v * jax.random.normal(k3, (7,))
    v1  = mu_v1 + sigma_v * jax.random.normal(k4, (7,))
    a0  = mu_a0 + sigma_a * jax.random.normal(k5, (7,))
    a1  = mu_a1 + sigma_a * jax.random.normal(k6, (7,))

    # Build jets (shape (3, 7)): [value; first-derivative; second-derivative]
    J0 = jnp.stack([y0, v0, a0])
    J1 = jnp.stack([y1, v1, a1])

    # --- Evaluate like the other tests ---
    t_grid, y_lin = hermite_interp1d(0, 1, J0, J1, ord=1, num=L)  # (num, 7)
    _, y_cub = hermite_interp1d(0, 1, J0, J1, ord=3, num=L)  # (num, 7)
    _, y_qui = hermite_interp1d(0, 1, J0, J1, ord=5, num=L)  # (num, 7)

    # --- Plot: 7 stacked axes (one per DoF) ---
    fig, axes = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
    for d in range(7):
        ax = axes[d]
        ax.plot(t_grid, y_lin[:, d], label="linear")
        ax.plot(t_grid, y_cub[:, d], label="cubic")
        ax.plot(t_grid, y_qui[:, d], label="quintic")
        ax.set_ylabel(f"q{d+1}(t)")
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)

    axes[-1].set_xlabel("t")
    fig.suptitle("Hermite interpolation per DoF (7-DoF) — normal jets", y=0.995)
    fig.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Original tests
    print("Running original tests...")
    test_1d()
    test_2d()
    test_7d()
    
    # New spline tests
    print("\nRunning new spline tests...")
    test_spline_1d()
    test_spline_2d()