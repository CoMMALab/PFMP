
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def mat4_from_pose(t: jax.Array, R: jax.Array) -> jax.Array:
    return jnp.array([
        [R[0,0], R[0,1], R[0,2], t[0]],
        [R[1,0], R[1,1], R[1,2], t[1]],
        [R[2,0], R[2,1], R[2,2], t[2]],
        [0.    , 0.    , 0.    , 1.  ]
    ])

@jax.jit
def quat_wxyz_from_R(R: jax.Array):
    return Rotation.from_matrix(R).as_quat(scalar_first=True)

@jax.jit
def pose_from_mat4(mat4: jax.Array) -> Tuple[jax.Array, jax.Array]:
    if mat4.shape[-1] == 16:
        m = jnp.reshape(mat4, mat4.shape[:-1] + (4, 4))
    else:
        m = mat4
    R = m[..., :3, :3]
    t = m[..., :3, 3]
    return t, quat_wxyz_from_R(R)


@jax.jit
def quat_yaw_wxyz(q):
    # q = (w, x, y, z) per MuJoCo
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)

@jax.jit
def angle_wrap(a):
    # maps to (-pi, pi]
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
