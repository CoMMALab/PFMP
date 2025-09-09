import jax
import jax.numpy as jnp
from mujoco import mjx
from .inverted_pendulum import transition_fn, ControlParams, DataTracer, u, mjx_step
from genjax import gen, mv_normal


@jax.jit
def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
    p, theta1, theta2, theta3 = data.qpos
    dp, dtheta1, dtheta2, dtheta3 = data.qvel
    
    # Angular deviations from upright (more direct formulation)
    angle_cost1 =  1 + jnp.cos(theta1)
    angle_cost2 = (1 - jnp.cos(theta2)) / 2
    angle_cost3 = (1 - jnp.cos(theta3)) / 2
    
    # Position and velocity costs with different weights
    pos_cost = 0.1  *  p**2  # light penalty on cart position
    vel_cost = 0.02 * dp**2  # light penalty on cart velocity
    
    # Angular velocity costs (prevent excessive spinning)
    ang_vel_cost1 = 0.03 * dtheta1**2
    ang_vel_cost2 = 0.03 * dtheta2**2
    ang_vel_cost3 = 0.03 * dtheta3**2
    
    total_cost = (
        angle_cost1 + angle_cost2 + angle_cost3 +
        pos_cost + vel_cost +
        ang_vel_cost1 + ang_vel_cost2 + ang_vel_cost3
    )
    
    return total_cost

u = ControlParams(
    mjx_model = None,
    ctrl_nominal = None,
    coarse_control = 1.0,
    fine_control = 0.1,
    fine_angle_bounds = 0.3
)