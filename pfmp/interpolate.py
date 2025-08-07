import jax
import jax.numpy as jnp


def normalized_time(t_start, t_end, num_points):
    """Create normalized time grid and return s, h."""
    t = jnp.linspace(t_start, t_end, num_points)
    h = t_end - t_start
    s = (t - t_start) / h
    return s, h
normalized_time = jax.jit(normalized_time, static_argnames=["num_points"])

@jax.jit
def hermite_basis_linear(s):
    """Linear Hermite basis functions."""
    return jnp.stack([1 - s, s])

@jax.jit
def hermite_basis_cubic(s):
    """Cubic Hermite basis functions and derivatives."""
    s2, s3 = s**2, s**3
    pos_basis = jnp.stack([
        2*s3 - 3*s2 + 1,    # H00
        s3 - 2*s2 + s,      # H10
        -2*s3 + 3*s2,       # H01
        s3 - s2             # H11
    ])
    vel_basis = jnp.stack([
        6*s2 - 6*s,         # H00'
        3*s2 - 4*s + 1,     # H10'
        -6*s2 + 6*s,        # H01'
        3*s2 - 2*s          # H11'
    ])
    return pos_basis, vel_basis

@jax.jit
def hermite_basis_quintic(s):
    """Quintic Hermite basis functions and derivatives."""
    s2, s3, s4, s5 = s**2, s**3, s**4, s**5
    
    pos_basis = jnp.stack([
        1 - 10*s3 + 15*s4 - 6*s5,        # H00
        s - 6*s3 + 8*s4 - 3*s5,          # H10
        0.5*s2 - 1.5*s3 + 1.5*s4 - 0.5*s5,  # H20
        10*s3 - 15*s4 + 6*s5,            # H01
        -4*s3 + 7*s4 - 3*s5,             # H11
        0.5*s3 - s4 + 0.5*s5             # H21
    ])
    
    vel_basis = jnp.stack([
        -30*s2 + 60*s3 - 30*s4,          # H00'
        1 - 18*s2 + 32*s3 - 15*s4,       # H10'
        s - 4.5*s2 + 6*s3 - 2.5*s4,      # H20'
        30*s2 - 60*s3 + 30*s4,           # H01'
        -12*s2 + 28*s3 - 15*s4,          # H11'
        1.5*s2 - 4*s3 + 2.5*s4           # H21'
    ])
    
    acc_basis = jnp.stack([
        -60*s + 180*s2 - 120*s3,         # H00''
        -36*s + 96*s2 - 60*s3,           # H10''
        1 - 9*s + 18*s2 - 10*s3,         # H20''
        60*s - 180*s2 + 120*s3,          # H01''
        -24*s + 84*s2 - 60*s3,           # H11''
        3*s - 12*s2 + 10*s3              # H21''
    ])
    
    return pos_basis, vel_basis, acc_basis

@jax.jit
def linear_interpolation(constraints_start, constraints_end, s, h):
    """Linear Hermite interpolation."""
    basis = hermite_basis_linear(s)  # [2, N]
    controls = jnp.stack([constraints_start[0], constraints_end[0]])  # [2, D]
    pos = jnp.einsum('in,id->nd', basis, controls)
    return pos[None, ...]  # [1, N, D]

@jax.jit
def cubic_interpolation(constraints_start, constraints_end, s, h):
    """Cubic Hermite interpolation."""
    pos_basis, vel_basis = hermite_basis_cubic(s)  # [4, N], [4, N]
    controls = jnp.stack([
        constraints_start[0], h * constraints_start[1],
        constraints_end[0], h * constraints_end[1]
    ])  # [4, D]
    
    pos = jnp.einsum('in,id->nd', pos_basis, controls)
    vel = jnp.einsum('in,id->nd', vel_basis, controls) / h
    return jnp.stack([pos, vel])  # [2, N, D]

@jax.jit
def quintic_interpolation(constraints_start, constraints_end, s, h):
    """Quintic Hermite interpolation."""
    pos_basis, vel_basis, acc_basis = hermite_basis_quintic(s)  # [6, N] each
    controls = jnp.stack([
        constraints_start[0], h * constraints_start[1], h**2 * constraints_start[2],
        constraints_end[0], h * constraints_end[1], h**2 * constraints_end[2]
    ])  # [6, D]
    
    pos = jnp.einsum('in,id->nd', pos_basis, controls)
    vel = jnp.einsum('in,id->nd', vel_basis, controls) / h
    acc = jnp.einsum('in,id->nd', acc_basis, controls) / h**2
    return jnp.stack([pos, vel, acc])  # [3, N, D]

def hermite_spline(constraints_start, constraints_end, t_start, t_end, num_points):
    """
    Hermite spline interpolation with automatic method selection.

    Always returns shape [3, N, D], with NaN for undefined higher-order derivatives.
    """
    s, h = normalized_time(t_start, t_end, num_points)
    n_constraints = constraints_start.shape[0]

    return jax.lax.switch(
        jnp.minimum(n_constraints - 1, 2),
        [
            # Linear branch
            lambda: jnp.concatenate([
                linear_interpolation(constraints_start, constraints_end, s, h),
                jnp.full((2, num_points, constraints_start.shape[1]), jnp.nan)
            ], axis=0),

            # Cubic branch
            lambda: jnp.concatenate([
                cubic_interpolation(constraints_start, constraints_end, s, h),
                jnp.full((1, num_points, constraints_start.shape[1]), jnp.nan)
            ], axis=0),

            # Quintic branch
            lambda: quintic_interpolation(constraints_start, constraints_end, s, h)
        ]
    )
hermite_spline = jax.jit(hermite_spline, static_argnames=['num_points'])

# def hermite_spline(constraints_start, constraints_end, t_start, t_end, num_points):
#     """
#     Hermite spline interpolation with automatic method selection.
#     - constraints_* must be shape (D, ...) where D ∈ {1, 2, 3}
#     - num_constraints must be passed as a static argument if used in shape logic
#     - Returns array of shape (3, num_points, ...)
#     """
#     s, h = normalized_time(t_start, t_end, num_points)

#     # Precompute fixed-shape padding tensors (regardless of which is used)
#     num_constraints = constraints_start.shape[0]
#     shape_tail = constraints_start.shape[1:]  # e.g. (), or (state_dim,)
#     nan_1 = jnp.full((2, num_points) + shape_tail, jnp.nan)
#     nan_2 = jnp.full((1, num_points) + shape_tail, jnp.nan)

#     spline_order = jnp.minimum(num_constraints - 1, 2)

#     lin = jnp.concatenate([
#         linear_interpolation(constraints_start, constraints_end, s),
#         nan_1
#     ], axis=0)

#     cub = jnp.concatenate([
#         cubic_interpolation(constraints_start, constraints_end, s, h),
#         nan_2
#     ], axis=0)

#     qui = quintic_interpolation(constraints_start, constraints_end, s, h)

#     return jax.lax.switch(spline_order, [lambda: lin, lambda: cub, lambda: qui])
# hermite_spline = jax.jit(hermite_spline, static_argnames=['num_points'])


# Example usage
if __name__ == "__main__":
    import time
    
    # Test data
    p0, p1 = jnp.array([0., 0.]), jnp.array([1., 1.])
    v0, v1 = jnp.array([1., 0.]), jnp.array([0., 1.])
    a0, a1 = jnp.array([0., 1.]), jnp.array([1., 0.])
    
    print("Testing Hermite spline with different constraint orders:")
    
    # Linear interpolation
    constraints_start = jnp.stack([p0])
    constraints_end = jnp.stack([p1])
    result = hermite_spline(constraints_start, constraints_end, 0.0, 1.0, 5)
    print(f"Linear - shape: {result.shape}")
    print(f"Positions: {result[0]}")
    
    # Cubic interpolation
    constraints_start = jnp.stack([p0, v0])
    constraints_end = jnp.stack([p1, v1])
    result = hermite_spline(constraints_start, constraints_end, 0.0, 1.0, 5)
    print(f"\nCubic - shape: {result.shape}")
    print(f"Positions: {result[0]}")
    print(f"Velocities: {result[1]}")
    
    # Quintic interpolation
    constraints_start = jnp.stack([p0, v0, a0])
    constraints_end = jnp.stack([p1, v1, a1])
    result = hermite_spline(constraints_start, constraints_end, 0.0, 1.0, 5)
    print(f"\nQuintic - shape: {result.shape}")
    print(f"Positions: {result[0]}")
    print(f"Velocities: {result[1]}")
    print(f"Accelerations: {result[2]}")
    
    # Benchmark
    print(f"\nBenchmarking cubic interpolation (1000 runs, 100 points each):")
    
    constraints_start = jnp.stack([p0, v0])
    constraints_end = jnp.stack([p1, v1])
    
    # Warmup
    for _ in range(10):
        hermite_spline(constraints_start, constraints_end, 0.0, 1.0, 100)
    
    start = time.time()
    for _ in range(1000):
        result = hermite_spline(constraints_start, constraints_end, 0.0, 1.0, 100)
    end = time.time()
    
    print(f"Total time: {(end-start)*1000:.1f}ms")
    print(f"Per interpolation: {(end-start)*1000:.2f}μs")