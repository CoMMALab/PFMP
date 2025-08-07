# === test_interpolate.py ===

import pytest
import numpy as np
import jax.numpy as jnp
from pfmp.interpolate import hermite_spline

@pytest.fixture
def t_fine():
    return jnp.linspace(0.0, 1.0, num=50)

@pytest.fixture
def sample_vectors():
    p0 = jnp.array([1.0, 2.0, 3.0])
    p1 = jnp.array([4.0, 5.0, 6.0])
    return p0, p1

def test_linear_endpoints_and_midpoint(t_fine, sample_vectors):
    p0, p1 = sample_vectors
    constraints_start = jnp.stack([p0])
    constraints_end = jnp.stack([p1])
    out = hermite_spline(constraints_start, constraints_end,
                         t_start=0.0, t_end=1.0,
                         num_points=t_fine.shape[0])
    pos = out[0]
    # endpoints
    np.testing.assert_allclose(pos[0], p0, atol=1e-6)
    np.testing.assert_allclose(pos[-1], p1, atol=1e-6)
    # midpoint
    mid_i = t_fine.shape[0] // 2
    s_mid = float(t_fine[mid_i])
    np.testing.assert_allclose(
        pos[mid_i],
        (1 - s_mid) * p0 + s_mid * p1,
        atol=1e-2
    )


def test_cubic_endpoints_and_derivatives(t_fine, sample_vectors):
    p0, p1 = sample_vectors
    v0 = jnp.array([1.0, -1.0, 0.5])
    v1 = jnp.array([-0.5, 2.0, -1.0])
    constraints_start = jnp.stack([p0, v0])
    constraints_end = jnp.stack([p1, v1])
    out = hermite_spline(constraints_start, constraints_end,
                         t_start=0.0, t_end=1.0,
                         num_points=t_fine.shape[0])
    pos, vel = out[0], out[1]
    # positions
    np.testing.assert_allclose(pos[0], p0, atol=1e-6)
    np.testing.assert_allclose(pos[-1], p1, atol=1e-6)
    # velocities
    np.testing.assert_allclose(vel[0], v0, atol=1e-6)
    np.testing.assert_allclose(vel[-1], v1, atol=1e-6)


def test_quintic_endpoints_and_derivatives(t_fine, sample_vectors):
    p0, p1 = sample_vectors
    v0 = jnp.array([1.0, -1.0, 0.5])
    v1 = jnp.array([-0.5, 2.0, -1.0])
    a0 = jnp.zeros_like(p0)
    a1 = jnp.zeros_like(p1)
    constraints_start = jnp.stack([p0, v0, a0])
    constraints_end = jnp.stack([p1, v1, a1])
    out = hermite_spline(constraints_start, constraints_end,
                         t_start=0.0, t_end=1.0,
                         num_points=t_fine.shape[0])
    pos, vel, acc = out[0], out[1], out[2]
    # positions
    np.testing.assert_allclose(pos[0], p0, atol=1e-6)
    np.testing.assert_allclose(pos[-1], p1, atol=1e-6)
    # velocities
    np.testing.assert_allclose(vel[0], v0, atol=1e-6)
    np.testing.assert_allclose(vel[-1], v1, atol=1e-6)
    # accelerations
    np.testing.assert_allclose(acc[0], a0, atol=1e-6)
    np.testing.assert_allclose(acc[-1], a1, atol=1e-6)


def test_wrong_signature_raises():
    with pytest.raises(TypeError):
        # missing all arguments
        hermite_spline()
    p0, p1 = jnp.array([0.0]), jnp.array([1.0])
    with pytest.raises(TypeError):
        # missing num_points
        hermite_spline(jnp.stack([p0]), jnp.stack([p1]), 0.0, 1.0)


# === plot_interpolation.py ===

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pfmp.interpolate import hermite_spline

def plot_methods():
    p0 = jnp.array([0.0])
    p1 = jnp.array([1.0])
    t = jnp.linspace(0.0, 1.0, 200)

    # sample derivatives
    v0 = jnp.array([10.0])
    v1 = jnp.array([0.0])
    a0 = jnp.array([10.0])
    a1 = jnp.array([-70.0])

    # ZOH (position only)
    zoh = hermite_spline(jnp.stack([p0]), jnp.stack([p1]), 0.0, 1.0, t.shape[0])[0]
    linear = hermite_spline(jnp.stack([p0]), jnp.stack([p1]), 0.0, 1.0, t.shape[0])[0]
    cubic = hermite_spline(jnp.stack([p0, v0]), jnp.stack([p1, v1]), 0.0, 1.0, t.shape[0])[0]
    quintic = hermite_spline(jnp.stack([p0, v0, a0]), jnp.stack([p1, v1, a1]), 0.0, 1.0, t.shape[0])[0]

    plt.figure(figsize=(8, 5))
    plt.plot(t, jnp.squeeze(zoh), label="ZOH")
    plt.plot(t, jnp.squeeze(linear), label="Linear")
    plt.plot(t, jnp.squeeze(cubic), label="Cubic")
    plt.plot(t, jnp.squeeze(quintic), label="Quintic")

    plt.scatter([0, 1], [0, 1], c="k", zorder=5)
    plt.text(0, 0, "  start", va="bottom")
    plt.text(1, 1, "  end", va="bottom")

    plt.title("ZOH • Linear • Cubic • Quintic Interpolation")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_methods()
    import pytest, sys
    sys.exit(pytest.main([__file__]))
