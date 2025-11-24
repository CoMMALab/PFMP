import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from typing import NamedTuple, Union


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


# --------- normalization helpers ---------


def normalize_single(x: jax.Array) -> jax.Array:
    """
    Normalize a *single* jet component into shape (D,).

    Allowed:
      scalar  → (1,)
      (D,)    → (D,)
    """
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x[None]      # scalar → (1,)
    elif x.ndim == 1:
        return x            # (D,)
    else:
        raise ValueError(f"Single Jet component must be scalar or 1D; got {x.shape}")


def normalize_knots(x: jax.Array) -> jax.Array:
    """
    Normalize a *batched* knot sequence into shape (K, D).

    Allowed:
      scalar    → (1,1)
      (K,)      → (K,1)
      (K,D)     → (K,D)
    """
    x = jnp.asarray(x)

    if x.ndim == 0:
        return x[None, None]      # scalar → (1,1)
    elif x.ndim == 1:
        return x[:, None]         # (K,) → (K,1)
    elif x.ndim == 2:
        return x                  # (K,D)
    else:
        raise ValueError(f"Knot sequence must be 0D, 1D, or 2D; got {x.shape}")


# --------- Jet classes ---------


class Jet1(NamedTuple):
    u: jax.Array

    def to_array(self) -> jax.Array:
        """Convert to shape (1, D)."""
        x = normalize_single(self.u)
        return x[None, :]   # (1, D)

    @classmethod
    def from_array(cls, arr: jax.Array) -> "Jet1":
        """arr: (1, D) → Jet1(u=(D,))"""
        return cls(arr[0])

    @classmethod
    def to_batch(cls, u) -> jax.Array:
        """Build (K,1,D) batch array from knot positions."""
        u = normalize_knots(u)      # (K,D)
        return u[:, None, :]        # (K,1,D)


class Jet2(NamedTuple):
    u: jax.Array
    du: jax.Array

    def to_array(self) -> jax.Array:
        """Convert to shape (2, D)."""
        u = normalize_single(self.u)
        du = normalize_single(self.du)
        return jnp.stack([u, du], axis=0)

    @classmethod
    def from_array(cls, arr: jax.Array) -> "Jet2":
        """arr: (2, D)"""
        return cls(u=arr[0], du=arr[1])

    @classmethod
    def to_batch(cls, u, du) -> jax.Array:
        """Build (K,2,D) batch array from knot pos/vel."""
        u = normalize_knots(u)
        du = normalize_knots(du)
        return jnp.stack([u, du], axis=1)  # (K,2,D)


class Jet3(NamedTuple):
    u: jax.Array
    du: jax.Array
    ddu: jax.Array

    def to_array(self) -> jax.Array:
        """Convert to shape (3, D)."""
        u = normalize_single(self.u)
        du = normalize_single(self.du)
        ddu = normalize_single(self.ddu)
        return jnp.stack([u, du, ddu], axis=0)

    @classmethod
    def from_array(cls, arr: jax.Array) -> "Jet3":
        """arr: (3, D)"""
        return cls(u=arr[0], du=arr[1], ddu=arr[2])

    @classmethod
    def to_batch(cls, u, du, ddu) -> jax.Array:
        """Build (K,3,D) batch array from knot pos/vel/acc."""
        u = normalize_knots(u)
        du = normalize_knots(du)
        ddu = normalize_knots(ddu)
        return jnp.stack([u, du, ddu], axis=1)  # (K,3,D)


JetType = Union[Jet1, Jet2, Jet3]


def as_jet(j: Union[JetType, jax.Array, float]) -> JetType:
    """Coerce raw scalar/array into Jet1 if needed."""
    if isinstance(j, (Jet1, Jet2, Jet3)):
        return j
    return Jet1(jnp.asarray(j))


def batched_jet_to_array(knots: Union[JetType, jax.Array, float]) -> jax.Array:
    """
    Convert Jet with knot dimension to canonical (K, m, D).

    If `knots` is a raw array / scalar, it's treated as Jet1 knot data.
    """
    if isinstance(knots, Jet3):
        return Jet3.to_batch(knots.u, knots.du, knots.ddu)   # (K,3,D)
    elif isinstance(knots, Jet2):
        return Jet2.to_batch(knots.u, knots.du)              # (K,2,D)
    elif isinstance(knots, Jet1):
        return Jet1.to_batch(knots.u)                        # (K,1,D)
    else:
        # Treat as Jet1 knot sequence (positions only)
        return Jet1.to_batch(knots)


# --------- Core array-only Hermite evaluation (private) ---------


@partial(jax.jit, static_argnames=("nout",))
def _hermite_eval_array(
    s: jax.Array, A: jax.Array, B: jax.Array, nout: int = 1
) -> jax.Array:
    """Core Hermite evaluation on arrays.

    Args:
        s: scalar parameter in [0,1]
        A: starting jet array, shape (m, D)
        B: ending  jet array, shape (m, D)
        nout: number of derivatives to output

    Returns:
        out: array of shape (nout, D)
    """
    if A.shape != B.shape:
        raise ValueError(f"A and B must match; got {A.shape} vs {B.shape}.")

    m, Ddim = A.shape
    POS, _, _ = BASES[m - 1]  # (2m,2m)
    N = POS.shape[1]          # 2/4/6

    # Differentiation matrix for increasing powers (subdiagonal)
    one_to_Nm1 = jnp.arange(1, N, dtype=POS.dtype)
    Dmat = jnp.zeros((N, N), dtype=POS.dtype).at[
        jnp.arange(1, N), jnp.arange(N - 1)
    ].set(one_to_Nm1)

    s = jnp.asarray(s, dtype=A.dtype)  # scalar or array
    poly = lambda C: jnp.polyval(jnp.flip(C), s)

    J = jnp.concatenate([A, B], axis=0)  # (N, D)

    coeff = POS
    outs = []
    for _ in range(nout):
        Bvals = jax.vmap(poly, in_axes=0, out_axes=-1)(coeff)  # (N,)
        outs.append(Bvals @ J)  # (D,)
        coeff = coeff @ Dmat

    return jnp.stack(outs, axis=0)  # (nout, D)


# --------- Public Jet-typed hermite_eval (single s) ---------


def hermite_eval(
    s: float,
    J0: Union[JetType, jax.Array, float],
    J1: Union[JetType, jax.Array, float],
    nout: int = 1,
):
    """Evaluate Hermite interpolation at parameter s ∈ [0,1] on Jet-like inputs.

    Returns:
        nout==1 → jax.Array (D,)
        nout==2 → Jet2
        nout==3 → Jet3
    """
    J0_t = as_jet(J0)
    J1_t = as_jet(J1)
    A = J0_t.to_array()
    B = J1_t.to_array()

    out = _hermite_eval_array(s, A, B, nout)  # (nout, D)

    if nout == 1:
        return out[0]
    elif nout == 2:
        return Jet2.from_array(out)
    elif nout == 3:
        return Jet3.from_array(out)
    else:
        raise ValueError("nout must be 1, 2, or 3")


# --------- Spline over multiple knots (Jet-typed interface) ---------


@partial(jax.jit, static_argnames=("nout",))
def hermite_spline(
    tq: jax.Array,
    tk: jax.Array,
    knots: Union[JetType, jax.Array, float],
    nout: int = 1,
):
    """Evaluate a Hermite spline over a sequence of knots.

    Args:
        tq: query times, shape (Q,)
        tk: knot times, shape (K,)
        knots: Jet1/Jet2/Jet3 *with knot dimension* K, or raw array
        nout: number of derivatives to output

    Returns:
        if nout == 1: array of shape (Q, D)
        if nout > 1: list of Q Jet2/Jet3 values
    """
    tq = jnp.asarray(tq)
    tk = jnp.asarray(tk)

    # Convert to (K, m, D)
    knot_arr = batched_jet_to_array(knots)  # (K, m, D)
    K, m, D = knot_arr.shape

    # segment index for each query
    seg_idx = jnp.searchsorted(tk, tq, side="right") - 1
    seg_idx = jnp.clip(seg_idx, 0, K - 2)

    def eval_single(t, idx):
        t0, t1 = tk[idx], tk[idx + 1]
        delta = t1 - t0

        J0_arr = knot_arr[idx]       # (m, D)
        J1_arr = knot_arr[idx + 1]   # (m, D)

        scale = (delta ** jnp.arange(m, dtype=J0_arr.dtype))[:, None]
        J0s = J0_arr * scale
        J1s = J1_arr * scale

        s = (t - t0) / delta
        return _hermite_eval_array(s, J0s, J1s, nout)  # (nout, D)

    result_arrays = jax.vmap(eval_single)(tq, seg_idx)  # (Q, nout, D)

    # if nout == 1:
    #     arr = result_arrays[:, 0, :]  # (Q, D)
    #     return arr[:, 0] if arr.shape[1] == 1 else arr
    # else:
    #     jets = []
    #     for i in range(result_arrays.shape[0]):
    #         slice_i = result_arrays[i]  # (nout, D)
    #         if nout == 2:
    #             jets.append(Jet2.from_array(slice_i))
    #         elif nout == 3:
    #             jets.append(Jet3.from_array(slice_i))
    #         else:
    #             raise ValueError("nout must be 1, 2, or 3")
    #     return jets


    if nout == 1:
        arr = result_arrays[:, 0, :]   # (num, D)
        jet = arr[:, 0] if arr.shape[1] == 1 else arr
    elif nout == 2:
        jet = Jet2(
            u  = result_arrays[:, 0],
            du = result_arrays[:, 1],
        )
    elif nout == 3:
        jet = Jet3(
            u   = result_arrays[:, 0],
            du  = result_arrays[:, 1],
            ddu = result_arrays[:, 2],
        )
    else:
        raise ValueError("nout must be 2 or 3")
    return jet


# --------- Single-segment Hermite (Jet-typed interface) ---------


@partial(jax.jit, static_argnames=("num", "nout"))
def hermite_interp1d(
    t0: float,
    t1: float,
    J0: Union[JetType, jax.Array, float],
    J1: Union[JetType, jax.Array, float],
    num: int,
    nout: int = 1,
):
    """Sample Hermite interpolation on [t0, t1] between two jets.

    Returns:
        t_grid: (num,)
        values:
          if nout==1: (num, D)
          if nout==2: list[Jet2] (length num)
          if nout==3: list[Jet3] (length num)
    """
    J0_t = as_jet(J0)
    J1_t = as_jet(J1)
    A0 = J0_t.to_array()  # (m,D)
    A1 = J1_t.to_array()

    t0 = jnp.asarray(t0, dtype=jnp.float32)
    t1 = jnp.asarray(t1, dtype=jnp.float32)
    t_grid = jnp.linspace(t0, t1, num, dtype=jnp.float32)

    m = A0.shape[0]
    delta = t1 - t0
    scale = (delta ** jnp.arange(m, dtype=jnp.float32))[:, None]

    J0s = A0 * scale
    J1s = A1 * scale

    s_vals = (t_grid - t0) / delta

    def eval_at_s(s):
        return _hermite_eval_array(s, J0s, J1s, nout)  # (nout,D)

    result_arrays = jax.vmap(eval_at_s)(s_vals)  # (num, nout, D)

    if nout == 1:
        arr = result_arrays[:, 0, :]   # (num, D)
        jet = (arr[:, 0] if arr.shape[1] == 1 else arr)
    elif nout == 2:
        jet = Jet2(
            u  = result_arrays[:, 0],
            du = result_arrays[:, 1],
        )
    elif nout == 3:
        jet = Jet3(
            u   = result_arrays[:, 0],
            du  = result_arrays[:, 1],
            ddu = result_arrays[:, 2],
        )
    else:
        raise ValueError("nout must be 2 or 3")

    return t_grid, jet


# --------- Tests / demos ---------

L = 512


def test_spline_1d():
    tk = jnp.array([0.0, 0.3, 0.7, 1.0])

    # Jet1 knots: positions only, scalar → (K,)
    knots_jet1 = jnp.array([-1.2, 0.5, 0.8, 0.2])

    # Jet2 knots: pos + vel
    knots_jet2 = Jet2(
        u=jnp.array([-1.2, 0.5, 0.8, 0.2]),
        du=jnp.array([22.9, 5.0, -3.0, 2.9]),
    )

    # Jet3 knots: pos + vel + acc
    knots_jet3 = Jet3(
        u=jnp.array([-1.2, 0.5, 0.8, 0.2]),
        du=jnp.array([22.9, 5.0, -3.0, 2.9]),
        ddu=jnp.array([-279.8, -50.0, 100.0, 35.5]),
    )

    tq = jnp.linspace(0, 1, L)

    y1 = hermite_spline(tq, tk, knots_jet1, nout=1)
    y3 = hermite_spline(tq, tk, knots_jet2, nout=1)
    y5 = hermite_spline(tq, tk, knots_jet3, nout=1)

    plt.figure()
    plt.plot(tq, y1, label="Linear Hermite", alpha=0.7)
    plt.plot(tq, y3, label="Cubic Hermite", alpha=0.7)
    plt.plot(tq, y5, label="Quintic Hermite", alpha=0.7)

    # knot positions (Jet3.u is (K,))
    knot_pos = knots_jet3.u
    plt.scatter(tk, knot_pos, s=50, color='red', label='Knots')

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Hermite spline with 4 knots (1D)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_spline_2d():
    tk = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    knots_jet3 = Jet3(
        u=jnp.array([
            [-1.2,  0.0],
            [ 0.0, -0.5],
            [ 0.5,  1.0],
            [ 0.3,  1.5],
            [ 0.2,  2.0],
        ]),
        du=jnp.array([
            [25.9, -2.3],
            [10.0,  5.0],
            [-5.0,  8.0],
            [ 3.0,  2.0],
            [ 2.9,  3.84],
        ]),
        ddu=jnp.array([
            [-279.8,   3.0],
            [ -50.0,  20.0],
            [ 100.0, -30.0],
            [ -20.0,  10.0],
            [  35.5,  -2.22],
        ]),
    )

    knots_jet2 = Jet2(
        u=knots_jet3.u,
        du=knots_jet3.du,
    )

    knots_jet1 = knots_jet3.u  # (K,2)

    tq = jnp.linspace(0, 1, L)

    xy1 = hermite_spline(tq, tk, knots_jet1, nout=1)
    xy3 = hermite_spline(tq, tk, knots_jet2, nout=1)
    xy5 = hermite_spline(tq, tk, knots_jet3, nout=1)

    plt.figure()
    plt.plot(xy1[:, 0], xy1[:, 1], label="linear", alpha=0.7)
    plt.plot(xy3[:, 0], xy3[:, 1], label="cubic", alpha=0.7)
    plt.plot(xy5[:, 0], xy5[:, 1], label="quintic", alpha=0.7)

    plt.scatter(knots_jet3.u[:, 0], knots_jet3.u[:, 1],
                color='red', s=50, label='Knots')

    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Hermite spline in R^2 (5 knots)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def test_1d():
    J0 = Jet3(u=-1.2, du=22.9, ddu=-279.8)
    J1 = Jet3(u=0.2, du=2.9, ddu=35.5)

    print("Single scalar evaluation (linear):",
          hermite_eval(0.3, -1.0, 4.0, nout=1))
    print("Jet2 evaluation (cubic):",
          hermite_eval(0.5, Jet2(u=-1.0, du=2.0),
                       Jet2(u=1.0, du=5.0), nout=2))

    t_grid, y1 = hermite_interp1d(0, 1, J0.u, J1.u, num=L, nout=1)
    plt.plot(t_grid, y1, label="Linear Hermite")

    _, y3 = hermite_interp1d(0, 1,
                             Jet2(u=J0.u, du=J0.du),
                             Jet2(u=J1.u, du=J1.du),
                             num=L, nout=1)
    plt.plot(t_grid, y3, label="Cubic Hermite")

    _, y5 = hermite_interp1d(0, 1, J0, J1, num=L, nout=1)
    plt.plot(t_grid, y5, label="Quintic Hermite")

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Hermite interpolation on [0,1]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_2d():
    J0 = Jet3(
        u=jnp.array([-1.2, 0.0]),
        du=jnp.array([25.9, -2.3]),
        ddu=jnp.array([-279.8, 3.0]),
    )
    J1 = Jet3(
        u=jnp.array([0.2, 2.0]),
        du=jnp.array([2.9, 3.84]),
        ddu=jnp.array([35.5, -2.22]),
    )

    _, xy1 = hermite_interp1d(0, 1, J0.u, J1.u, num=L, nout=1)
    plt.plot(xy1[:, 0], xy1[:, 1], label="linear")

    _, xy3 = hermite_interp1d(0, 1,
                             Jet2(u=J0.u, du=J0.du),
                             Jet2(u=J1.u, du=J1.du),
                             num=L, nout=1)
    plt.plot(xy3[:, 0], xy3[:, 1], label="cubic")

    _, xy5 = hermite_interp1d(0, 1, J0, J1, num=L, nout=1)
    plt.plot(xy5[:, 0], xy5[:, 1], label="quintic")

    plt.xlabel("y0(t)")
    plt.ylabel("y1(t)")
    plt.title("Hermite curve in R^2")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_7d():
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    mu_y0 = jnp.zeros(7)
    mu_y1 = jnp.ones(7)
    mu_v0 = jnp.zeros(7)
    mu_v1 = jnp.zeros(7)
    mu_a0 = jnp.zeros(7)
    mu_a1 = jnp.zeros(7)

    sigma_y = 0.25
    sigma_v = 1.0
    sigma_a = 3.0

    y0 = mu_y0 + sigma_y * jax.random.normal(k1, (7,))
    y1 = mu_y1 + sigma_y * jax.random.normal(k2, (7,))
    v0 = mu_v0 + sigma_v * jax.random.normal(k3, (7,))
    v1 = mu_v1 + sigma_v * jax.random.normal(k4, (7,))
    a0 = mu_a0 + sigma_a * jax.random.normal(k5, (7,))
    a1 = mu_a1 + sigma_a * jax.random.normal(k6, (7,))

    J0 = Jet3(u=y0, du=v0, ddu=a0)
    J1 = Jet3(u=y1, du=v1, ddu=a1)

    t_grid, y_lin = hermite_interp1d(0, 1, J0.u, J1.u, num=L, nout=1)
    _, y_cub = hermite_interp1d(0, 1,
                                Jet2(u=J0.u, du=J0.du),
                                Jet2(u=J1.u, du=J1.du),
                                num=L, nout=1)
    _, y_qui = hermite_interp1d(0, 1, J0, J1, num=L, nout=1)

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

def test_local_jacfwd_cubic_correct():
    s = 0.37
    t0, t1 = 0.0, 1.0
    delta = t1 - t0

    POS_C = BASES[1][0]
    powers = jnp.array([1., s, s**2, s**3])
    basis_vals = POS_C @ powers

    # define f: ℝ⁴ → ℝ (scalar hermite evaluation)
    def f(params):
        J0p = Jet2(params[0], params[1])
        J1p = Jet2(params[2], params[3])
        return hermite_eval(s, J0p, J1p, nout=1)

    params = jnp.array([1.3, -0.8, -0.4, 0.5])
    jac = jax.jacfwd(f)(params)

    expected = jnp.array([
        basis_vals[0],
        basis_vals[1] * delta,
        basis_vals[2],
        basis_vals[3] * delta,
    ])

    print("numerical:", jac)
    print("analytic :", expected)
    print("diff    R:", jac - expected)
import time
import jax
import jax.numpy as jnp

def perf_test_local_jacfwd():
    s = 0.37
    t0, t1 = 0.0, 1.0
    delta = t1 - t0

    POS_C = BASES[1][0]  # cubic position matrix
    powers = jnp.array([1., s, s**2, s**3])
    basis_vals = POS_C @ powers

    # ---- scalar Hermite eval: ℝ⁴ → ℝ ----
    def f(params):
        J0p = Jet2(params[0], params[1])
        J1p = Jet2(params[2], params[3])
        return hermite_eval(s, J0p, J1p, nout=1)

    params = jnp.array([1.3, -0.8, -0.4, 0.5])

    # warmup (compile)
    _ = jax.jacfwd(f)(params).block_until_ready()

    # ---- perf: AD Jacobian ----
    t0_ad = time.time()
    jac = jax.jacfwd(f)(params).block_until_ready()
    t1_ad = time.time()

    # ---- perf: analytic Jacobian ----
    t0_analytic = time.time()
    expected = jnp.array([
        basis_vals[0],
        basis_vals[1] * delta,
        basis_vals[2],
        basis_vals[3] * delta,
    ]).block_until_ready()
    t1_analytic = time.time()

    print("numerical:", jac)
    print("analytic :", expected)
    print("diff     :", jac - expected)

    print("\n=== timings (microseconds) ===")
    print("AD Jacobian :", (t1_ad - t0_ad) * 1e6)
    print("Analytic    :", (t1_analytic - t0_analytic) * 1e6)






if __name__ == "__main__":
    print("Running original tests...")
    test_1d()
    test_2d()
    test_7d()

    print("\nRunning new spline tests...")
    test_spline_1d()
    test_spline_2d()
    test_local_jacfwd_cubic_correct()
    perf_test_local_jacfwd()
