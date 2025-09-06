import xml.etree.ElementTree as ET
from io import StringIO
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy

from pfmp.interpolateh import hermite_interp1d


# ===============================
# Robot setup
# ===============================

ROBOT_URDF_PATH = "/home/austin/src/panda_scripts/assets/panda_spherized.urdf"

# Load URDF
urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)

# Convert prismatic joints to fixed (like in IK-Beam benchmark)
xml_tree = urdf.write_xml()
for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
    print(joint)  # optional debug
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics"):
        child = joint.find(tag)
        if child is not None:
            joint.remove(child)

# Reload robot with modified XML
xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
urdf = yourdfpy.URDF.load(StringIO(xml_str))

# Build Pyroki robot object
PANDA = pk.Robot.from_urdf(urdf)
EE_ID = 12


# --- Helpers ---

import jax_dataclasses as jdc

@jax.jit
def mat4_from_pose(t: jax.Array, R: jax.Array) -> jax.Array:
    return jnp.array([
        [R[0,0], R[0,1], R[0,2], t[0]],
        [R[1,0], R[1,1], R[1,2], t[1]],
        [R[2,0], R[2,1], R[2,2], t[2]],
        [0.    , 0.    , 0.    , 1.  ]
    ])

def quat_wxyz_from_R(R: jax.Array):
    return Rotation.from_matrix(R).as_quat(scalar_first=True)

# def quat_wxyz_from_R(R: jax.Array) -> jax.Array:
#     return jaxlie.SO3.from_matrix(R).wxyz

@jax.jit
def pose_from_mat4(mat4: jax.Array) -> Tuple[jax.Array, jax.Array]:
    if mat4.shape[-1] == 16:
        m = jnp.reshape(mat4, mat4.shape[:-1] + (4, 4))
    else:
        m = mat4
    R = m[..., :3, :3]
    t = m[..., :3, 3]
    return t, quat_wxyz_from_R(R)

@jdc.jit
def make_ik_problem(
        target_mat4: jax.Array,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        weight: float = 100.0
    ):
    print("DEBUG target_mat4 type:", type(target_mat4))
    print("DEBUG target_mat4 aval:", getattr(target_mat4, "aval", None))
    target_p, target_q = pose_from_mat4(target_mat4)
    joint_var = PANDA.joint_var_cls(0)
    T_target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_q), target_p
    )
    factors = [
        pk.costs.pose_cost_analytic_jac(
            PANDA,
            joint_var,
            T_target,
            target_link_index=jnp.array(EE_ID),
            pos_weight=pos_weight,
            ori_weight=ori_weight
        ),
        pk.costs.limit_cost(
            PANDA,
            joint_var,
            weight=weight,
        ),
    ]
    return jaxls.LeastSquaresProblem(factors, [joint_var]).analyze(), joint_var


@jax.jit
def ik_7d_from_mat4(target_mat4: jax.Array) -> jax.Array:
    prob0, joint_var0 = make_ik_problem(target_mat4)
    sol0 = prob0.solve(
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
    )
    return sol0[joint_var0]

@jax.jit
def ik_step_7d_from_mat4(prev_q, target_mat4):
    prob, joint_var = make_ik_problem(target_mat4)
    sol = prob.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
    )
    q = sol[joint_var]
    return q, q

@jax.jit
def ik_7d_from_mat4s(target_mat4s: jax.Array) -> jax.Array:
    q0 = ik_7d_from_mat4(target_mat4s[0])
    _, qs = jax.lax.scan(ik_step_7d_from_mat4, q0, target_mat4s[1:])
    return jnp.concatenate([q0[None, ...], qs])

if __name__ == "__main__":
    # Test Parameters
    T = 100

    ERROR_ATOL = 0.1
    MAX_JUMP = 10.0 / T

    VISUALIZE = True
    DWELL_SEC = 0.025  # lower for faster viz

    ASSERT_STRICT = True  # crash if error is too big

    # Start Constraints (modify these)
    h = 0.2
    x0 = [0.3, -0.3,   h]  # [px0, py0, pz0]
    v0 = [1.0,  0.8, 0.0]  # [vx0, vy0, vz0]

    # End Constraints (modify these)
    x1 = [ 0.4,  0.3,   h]  # [px1, py1, pz1]
    v1 = [-1.0, -1.3, 0.0]  # [vx1, vy1, vz1]

    # Invert an interpolated cubic polynomial
    # with Cartesian start and end on a 2D plane.
    _, inp_vec3s = hermite_interp1d(
        t0 = 0., t1 = 1.,
        J0 = jnp.array([x0, v0]),
        J1 = jnp.array([x1, v1]),
        ord = 3,
        num = T
    )
    R_fixed = jnp.array([
        [1.,  0.,  0.],
        [0., -1.,  0.],
        [0.,  0., -1.],
    ])
    target_mat4s = jax.vmap(mat4_from_pose, in_axes=[0,None])(inp_vec3s, R_fixed).reshape(-1,4,4)

    print("First three target mat4s:")
    print(target_mat4s[:3])
    print(f"target.shape = {target_mat4s.shape}")
    assert target_mat4s.shape == (T,4,4)

    # out_vec7s = jax.vmap(ik_7d_from_mat4)(target_mat4s)
    out_vec7s = jnp.array([ik_7d_from_mat4(t) for t in target_mat4s])
    # out_vec7s = ik_7d_from_mat4s(target_mat4s)

    print(out_vec7s[10])
    print(out_vec7s[11])