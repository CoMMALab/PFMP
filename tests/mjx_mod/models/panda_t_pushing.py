import pyroki as pk
import jaxlie
import jaxls
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import yourdfpy
from mujoco import mjx
from typing import NamedTuple, Tuple

from genjax import gen, mv_normal

# for urdf
from xml.etree import ElementTree as ET
from io import StringIO

from jax.scipy.spatial.transform import Rotation


from .panda_pose_utils import mat4_from_pose, quat_wxyz_from_R, pose_from_mat4, quat_yaw_wxyz, angle_wrap

class PParams(NamedTuple):
    mjx_model: mjx.Model
    t_qpos_offset: int
    goal_xy: jax.Array
    goal_angle: jax.Array
    cost_factors: jax.Array

    # The following for theta
    u_stdev: float
    du_stdev: float
    ddu_stdev: float

class StateTracker(NamedTuple):
    qpos: jax.Array
    txyz: jax.Array



class TPusher:
    def __init__(self):
        # ===============================
        # Robot setup
        # ===============================

        ROBOT_URDF_PATH = "tests/mjx_mod/assets/panda_spherized.urdf"

        # Load URDF
        urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)

        # Convert prismatic joints to fixed (like in IK-Beam benchmark)
        xml_tree = urdf.write_xml()
        for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                child = joint.find(tag)
                if child is not None:
                    joint.remove(child)

        # Reload robot with modified XML
        xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
        urdf = yourdfpy.URDF.load(StringIO(xml_str))

        # Build Pyroki robot object
        self.PANDA = pk.Robot.from_urdf(urdf)
        self.EE_ID = 12
        self.T_QPOS_OFFSET = 9

        self.HEIGHT = 0.06

    def init_mjx(self, mj_model, mj_data):
        mjx_model = mjx.put_model(mj_model)
        x0 = mjx.put_data(mj_model, mj_data)

        return x0, PParams(
            mjx_model,
            t_qpos_offset=9,
            goal_xy=jnp.array([0.4, 0.]),
            goal_angle=3*jnp.pi/4,
            cost_factors=jnp.array([3.0, 3.0, 0.5]),
            u_stdev=0.0125,
            du_stdev=0.01,
            ddu_stdev=0.1
        )


    def initialize_problem(self):
        """
        Set up an initial state and initialize the mjx
        
        """


        u0 = jnp.array([
            [0.4, -0.3],  # ( x0,  y0)
            [0.0,  0.2]   # (vx0, vy0)
        ])
        
        txyz = jnp.array([u0[0,0], u0[0,1], HEIGHT])
        tR = jnp.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.],
        ])
        q0 = ik_7d_from_mat4(mat4_from_pose(txyz, tR))
        mj_data.qpos[:7] = q0
        (x0, theta) = init_mjx(mj_model, mj_data)
        theta = theta._replace(goal_xy = jnp.array([-0.4, 0.3]), goal_angle=-0.2)
        qpos = jnp.concat((q0,mj_data.qpos[7:]))
        x0 = x0.replace(qpos=qpos)


        return x0, u0, theta



    # ---- Utils needed for IK ----------------

    @jdc.jit
    def make_ik_problem(
            self,
            target_mat4: jax.Array,
            pos_weight: float = 50.0,
            ori_weight: float = 10.0,
            weight: float = 100.0
        ):
        target_p, target_q = pose_from_mat4(target_mat4)
        joint_var = self.PANDA.joint_var_cls(0)
        T_target = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_q), target_p
        )
        factors = [
            pk.costs.pose_cost_analytic_jac(
                self.PANDA,
                joint_var,
                T_target,
                target_link_index=jnp.array(self.EE_ID),
                pos_weight=pos_weight,
                ori_weight=ori_weight
            ),
            pk.costs.limit_cost(
                self.PANDA,
                joint_var,
                weight=weight,
            ),
        ]
        return jaxls.LeastSquaresProblem(factors, [joint_var]).analyze(), joint_var

    @jax.jit
    def ik_7d_from_mat4(self, target_mat4: jax.Array) -> jax.Array:
        prob0, joint_var0 = make_ik_problem(target_mat4)
        sol0 = prob0.solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
        return sol0[joint_var0]

    @jax.jit
    def step_ik_7d_from_mat4(self, prev_q: jax.Array, target_mat4: jax.Array) -> tuple[jax.Array,jax.Array]:
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
    def ik_7d_from_mat4s(self, target_mat4s: jax.Array) -> jax.Array:
        q0 = ik_7d_from_mat4(target_mat4s[0])
        _, qs = jax.lax.scan(step_ik_7d_from_mat4, q0, target_mat4s[1:])
        return jnp.concatenate([q0[None, ...], qs])


    @gen
    def proposal_fn(self, x: mjx.Data, u: jax.Array, th: PParams, t: int) -> jax.Array:
        nu = u.shape[1]  # eg. Cartesian 2D/3D 
        pred_x = mv_normal(x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+2], jnp.eye(nu) * theta.u_stdev) @ "u"
        pred_v = mv_normal(u[1], jnp.eye(nu) * th.du_stdev) @ "du"
        return jnp.array([pred_x, pred_v])  # cubic


    @jax.jit
    def dynamics_fn(self, x, u, th, t) -> Tuple[jax.Array,jax.Array]:
        target_xyz = jnp.array([u[0,0], u[0,1], HEIGHT])
        target_R = jnp.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.],
        ])
        target_eefk = mat4_from_pose(target_xyz, target_R)
        target_qpos = step_ik_7d_from_mat4(x.qpos[:7], target_eefk)[0]
        x = x.replace(ctrl = target_qpos)
        x_next = mjx.step(th.mjx_model, x)
        return x_next, StateTracker(qpos=x.qpos, txyz=target_xyz)


    """
    Utils for cost function
    """

    def error_fn(th: PParams, t_qpos: jax.Array) -> jax.Array:
        t_xy    = t_qpos[:2]
        t_quat  = t_qpos[3:7]
        t_yaw   = quat_yaw_wxyz(t_quat)
        x_err, y_err = t_xy - th.goal_xy
        ang_err = angle_wrap(t_yaw - th.goal_angle)
        return jnp.array([x_err, y_err, ang_err])


    @jax.jit
    def cost_fn(x: mjx.Data, _u, th: PParams, _t) -> float:
        # Free joint layout: [px, py, pz, qw, qx, qy, qz]
        error = error_fn(th, x.qpos[self.T_QPOS_OFFSET:self.T_QPOS_OFFSET+7])
        return jnp.dot(th.cost_factors, error ** 2)
