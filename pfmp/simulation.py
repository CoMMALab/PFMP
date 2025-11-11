""" Visualization tools, aimed at parity with Hydrax """
import time
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx


def run_interactive(  # noqa: PLR0912, PLR0915
    controller,
    theta,  # (mean, mjx_model)
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
    """
    # Report the planning horizon in seconds for debugging
    interval_len = controller.horizon / (controller.num_knots - 1)
    steps_per_knot = round(interval_len / controller.dt)
    print(
        f"Planning with {steps_per_knot} steps "
        f"over a {controller.horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    key = jax.random.key(42)

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    # policy_params = controller.init_params(initial_knots=initial_knots)
    # jit_optimize = jax.jit(controller.optimize)
    # jit_interp_func = jax.jit(controller.interp_func)
    u0 = jnp.zeros((mj_model.nu,))

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    # policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    # policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    def ps_update(theta, trajs, scores):
        # Different from Hydrax (and more like Dynamic Expectation Maximization),
        # the model params `theta` are updated separately from the SMC kernel
        best_idx = jnp.argmax(jnp.sum(scores, axis=-1))  # Predictive Sampling
        new_mean = jax.tree.map(lambda field: field[best_idx], trajs[0])
        theta = theta._replace(mean = new_mean)
        return theta

    key, subkey = jax.random.split(key)
    (trajs, scores) = controller.run_sir(subkey, mjx_data, u0, theta)
    theta = ps_update(theta, trajs, scores)
    key, subkey = jax.random.split(key)
    (trajs, scores) = controller.run_sir(subkey, mjx_data, u0, theta)
    theta = ps_update(theta, trajs, scores)
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Do a replanning step
            plan_start = time.time()
            # policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            (key, subkey) = jax.random.split(key)
            (trajs, scores) = controller.run_sir(subkey, mjx_data, u0, theta)
            theta = ps_update(theta, trajs, scores)
            plan_time = time.time() - plan_start

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            # sim_dt = mj_model.opt.timestep
            # t_curr = mj_data.time

            # tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            # tk = policy_params.tk
            # knots = policy_params.mean[None, ...]
            # us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)
            us = theta.mean.reshape(
                controller.num_knots,
                mj_model.nu
            )

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

            # Update the mean based on elapsed offset
            tk = controller.tk()
            new_tk = controller.tk() + elapsed
            new_mean = hermite_spline(new_tk, tk, theta.mean)
            theta = theta._replace(mean=new_mean)
            u0 = new_mean[0]


    # Preserve the last printout
    print("")