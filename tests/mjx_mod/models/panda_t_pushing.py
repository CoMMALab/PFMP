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
from .problem import ProblemDesc
from .mj_utils import load_mj
from functools import partial

# for viz
from mujoco.viewer import launch_passive


T_QPOS_OFFSET=9

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


@gen
def quintic_proposal_fn(x: mjx.Data, u: jax.Array, theta: PParams, t: int) -> jax.Array:
    nu = u.shape[1]  # eg. Cartesian 2D/3D 
    pred_x = mv_normal(x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+2], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_v = mv_normal(u[1], jnp.eye(nu) * th.du_stdev) @ "du"
    return jnp.array([pred_x, pred_v])  # cubic

@gen
def cubic_proposal_fn(x: mjx.Data, u: jax.Array, theta: PParams, t: int) -> jax.Array:
    nu = u.shape[1]  # eg. Cartesian 2D/3D 
    pred_x = mv_normal(x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+2], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_v = mv_normal(u[1], jnp.eye(nu) * theta.du_stdev) @ "du"
    return jnp.array([pred_x, pred_v])  # cubic

@gen
def linear_proposal_fn(x: mjx.Data, u: jax.Array, theta: PParams, t: int) -> jax.Array:
    nu = u.shape[1]  # eg. Cartesian 2D/3D 
    pred_x = mv_normal(x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+2], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_v = mv_normal(u[1], jnp.eye(nu) * theta.du_stdev) @ "du"
    return jnp.array([pred_x, pred_v])  # cubic



class TPusher(ProblemDesc):
    def __init__(self):
        # ===============================
        # Robot setup
        # ===============================

        self.name = "panda_t_push"
        self.xml_filename = "mjx_t_push"


        ROBOT_URDF_PATH = "tests/mjx_mod/assets/panda_assets/panda/panda_spherized.urdf"

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
        self.T_QPOS_OFFSET = T_QPOS_OFFSET

        self.HEIGHT = 0.06

        self.linear_proposal_fn = linear_proposal_fn
        self.quintic_proposal_fn = quintic_proposal_fn
        self.cubic_proposal_fn = cubic_proposal_fn

        super().__init__()


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


    def initialize_problem(self, config):
        """
        Set up an initial state and initialize the mjx

        """
        (self.mj_model, self.mj_data) = load_mj(self.xml_filename)

        u0 = jnp.array([
            [0.4, -0.3],  # ( x0,  y0)
            [0.0,  0.2]   # (vx0, vy0)
        ])
        
        txyz = jnp.array([u0[0,0], u0[0,1], self.HEIGHT])
        tR = jnp.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.],
        ])
        q0 = self.ik_7d_from_mat4(mat4_from_pose(txyz, tR))
        self.mj_data.qpos[:7] = q0
        (x0, theta) = self.init_mjx(self.mj_model, self.mj_data)
        theta = theta._replace(goal_xy = jnp.array([-0.4, 0.3]), goal_angle=-0.2)
        qpos = jnp.concat((q0,self.mj_data.qpos[7:]))
        x0 = x0.replace(qpos=qpos)

        return x0, u0, theta



    # ---- Utils needed for IK ----------------

    # @jdc.jit
    # @partial(jdc.jit)
    @partial(jax.jit, static_argnames=['self'])
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

    @partial(jax.jit, static_argnames=['self'])
    def ik_7d_from_mat4(self, target_mat4: jax.Array) -> jax.Array:
        prob0, joint_var0 = self.make_ik_problem(target_mat4)
        sol0 = prob0.solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
        return sol0[joint_var0]

    @partial(jax.jit, static_argnames=['self'])
    def step_ik_7d_from_mat4(self, prev_q: jax.Array, target_mat4: jax.Array) -> tuple[jax.Array,jax.Array]:
        prob, joint_var = self.make_ik_problem(target_mat4)
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
        q = sol[joint_var]
        return q, q

    @partial(jax.jit, static_argnames=['self'])
    def ik_7d_from_mat4s(self, target_mat4s: jax.Array) -> jax.Array:
        q0 = self.ik_7d_from_mat4(target_mat4s[0])
        _, qs = jax.lax.scan(self.step_ik_7d_from_mat4, q0, target_mat4s[1:])
        return jnp.concatenate([q0[None, ...], qs])




    @partial(jax.jit, static_argnames=['self'])
    def dynamics_fn(self, x: mjx.Data, u: jax.Array, th: PParams, t: int) -> Tuple[jax.Array,jax.Array]:
        target_xyz = jnp.array([u[0,0], u[0,1], self.HEIGHT])
        target_R = jnp.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.],
        ])
        target_eefk = mat4_from_pose(target_xyz, target_R)
        target_qpos = self.step_ik_7d_from_mat4(x.qpos[:7], target_eefk)[0]
        x = x.replace(ctrl = target_qpos)
        x_next = mjx.step(th.mjx_model, x)
        return x_next, StateTracker(qpos=x.qpos, txyz=target_xyz)


    """
    Utils for cost function
    """

    def error_fn(self, th: PParams, t_qpos: jax.Array) -> jax.Array:
        t_xy    = t_qpos[:2]
        t_quat  = t_qpos[3:7]
        t_yaw   = quat_yaw_wxyz(t_quat)
        x_err, y_err = t_xy - th.goal_xy
        ang_err = angle_wrap(t_yaw - th.goal_angle)
        return jnp.array([x_err, y_err, ang_err])


    @partial(jax.jit, static_argnames=['self'])
    def cost_fn(self, x: mjx.Data, _u, th: PParams, _t) -> float:
        # Free joint layout: [px, py, pz, qw, qx, qy, qz]
        error = self.error_fn(th, x.qpos[self.T_QPOS_OFFSET:self.T_QPOS_OFFSET+7])
        return jnp.dot(th.cost_factors, error ** 2)


    # some viz utils
    # Debug draw: red line + green target
    def add_line(self, user_scn, p0, p1, radius=0.004, color=(1,0,0,0.7)):
        p0 = np.asarray(p0); p1 = np.asarray(p1)
        mid = 0.5*(p0+p1)
        v = p1 - p0
        L = np.linalg.norm(v) + 1e-12
        z = v / L
        a = np.array([1.0,0.0,0.0]) if abs(z[0]) < 0.9 else np.array([0.0,1.0,0.0])
        x = a - np.dot(a, z)*z; x /= (np.linalg.norm(x)+1e-12)
        y = np.cross(z, x)
        R = np.stack([x,y,z], axis=1)
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[radius, 0.5*L, 0.0], pos=mid, mat=R.flatten(), rgba=color)
        user_scn.ngeom += 1

    def add_sphere(self, scene, pos, radius=0.0025, rgba=(0,1,0,0.9)):
        if scene.ngeom >= scene.maxgeom: return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[radius,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=rgba)
        scene.ngeom += 1

    def add_goal_T(self,
                user_scn,
                goal_xy,
                yaw_rad,
                rgba=(0., 0., 0., 0.25),
                z_shadow=1e-3,
                thickness=5e-4):
        """
        Draw a translucent T-shaped 'shadow' using the user scene (passive viewer).
        Geometry matches the model's t_stem/t_bar XY half-sizes; Z is made very thin.

        Args:
            user_scn: viewer.user_scn
            goal_xy: (x, y) world position for the T body origin (matching t_block frame)
            yaw_rad: rotation about +Z (radians)
            rgba: color/alpha of the shadow
            z_shadow: world Z to place the shadow (slightly above plane to avoid z-fighting)
            thickness: half-thickness along Z for the two box geoms (very small)
        """
        # Guard: need 2 geoms available
        if user_scn.ngeom + 2 > user_scn.maxgeom:
            return

        # ---- Match the model's T dimensions (half-sizes) ----
        # stem: centered at body origin
        stem_half = np.array([0.0225, 0.0675, thickness], dtype=float)
        # bar: centered at +y by 0.0675 in the T local frame
        bar_half  = np.array([0.0675, 0.0225, thickness], dtype=float)
        bar_off_y = 0.0675

        # ---- Pose ----
        x, y = np.asarray(goal_xy, dtype=float)
        pos_world = np.array([x, y, float(z_shadow)], dtype=float)

        c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
        R = np.array([[ c, -s, 0.0],
                    [ s,  c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)

        # ---- Stem (centered at body origin) ----
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            size=stem_half,
            pos=pos_world,
            mat=R.flatten(),
            rgba=rgba
        )
        user_scn.ngeom += 1

        # ---- Bar (offset +y in local frame, then rotated to world) ----
        bar_center_world = pos_world + R @ np.array([0.0, bar_off_y, 0.0], dtype=float)

        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            size=bar_half,
            pos=bar_center_world,
            mat=R.flatten(),
            rgba=rgba
        )
        user_scn.ngeom += 1



    def visualize(self, theta, config, traj):

        t = 0

        EE_SITE = "gripper"
        ee_sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

        with launch_passive(
                self.mj_model,
                data=mj_data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
            while True:
                # txyz = traj[t]
                # mj_data.ctrl[:7] = step_ik_7d_from_mat4(mj_data.qpos[:7], mat4_from_pose(txyz, R_des))[0]
                txyz = traj.txyz[t]
                self.mj_data.qpos = traj.qpos[t]
                mujoco.mj_step(mj_model, self.mj_data)

                print(f"errors = {self.error_fn(theta, self.mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+7])}")

                # Clear previous markers
                viewer.user_scn.ngeom = 0

                # Add goal marker
                self.add_goal_T(viewer.user_scn, theta.goal_xy, theta.goal_angle, rgba = (0, 0, 1, 0.9))

                # Add knot markers
                for n in range(0,config.H*config.L,config.L):
                    self.add_sphere(viewer.user_scn, traj.txyz[n], radius=0.01)

                # Add interpolated markers
                # for n in range(0,H*L):
                #     add_sphere(viewer.user_scn, traj.txyz[n], radius=0.0025)
                
                # Add continuous PBC marker
                self.add_sphere(viewer.user_scn, txyz, radius=0.02)

                # Add continuous PBC error line
                xpos = self.mj_data.site_xpos[ee_sid].copy()
                self.add_line(
                    viewer.user_scn,
                    xpos,
                    txyz,
                    radius=0.005,
                    color=[1, 0, 0, 0.5]
                )
                
                time.sleep(0.01)
                viewer.sync()
                t += 1        