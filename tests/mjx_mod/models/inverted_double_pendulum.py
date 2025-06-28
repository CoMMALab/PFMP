import jax
import jax.numpy as jnp
from mujoco import mjx
from .inverted_pendulum import transition_fn, ControlParams, u


@jax.jit
def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
    p, theta1, theta2 = data.qpos
    dp, dtheta1, dtheta2 = data.qvel
    
    # Angular deviations from upright (more direct formulation)
    angle_cost1 = (1 + jnp.cos(theta1)) / 2  # 0 when upright, 20 when downward
    angle_cost2 = (1 - jnp.cos(theta2)) / 2  # 0 when upright, 20 when downward
    
    # Position and velocity costs with different weights
    pos_cost = 0.1 * p**2     # light penalty on cart position
    vel_cost = 0.02 * dp**2   # light penalty on cart velocity
    
    # Angular velocity costs (prevent excessive spinning)
    ang_vel_cost1 = 0.01 * dtheta1**2
    ang_vel_cost2 = 0.01 * dtheta2**2
    
    # Control effort penalty (energy efficiency)
    control_cost = 0.001 * data.ctrl[0]**2
    
    total_cost = (
        angle_cost1 + angle_cost2 +      # Primary objective: keep upright
        pos_cost + vel_cost +            # Secondary: control cart
        ang_vel_cost1 + ang_vel_cost2 +  # Prevent excessive motion
        control_cost                     # Penalize large control efforts
    )
    
    return total_cost