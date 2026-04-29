""" Visualization tools, aimed at parity with Hydrax """
import time
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
from pfmp.interpolate import Jet2, hermite_spline
from pfmp.algorithm import ps_update, ess
from hydrax.utils.video import VideoRecorder
from typing import Optional, Any

from .run_batch import Trajectory


def run_interactive(  # noqa: PLR0912, PLR0915
    controller,
    theta,  # (mean, mjx_model)
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    ik_interp: Optional[callable] = None,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    method = "sir",
    record_video: bool = False,
    record_traj: bool = False,
    max_step: int = int(1e6),
    goal_marker_fn: Optional[callable] = None
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
    if method == "sis":
        replan = controller.run_sis
        interp = hermite_spline
        family = "pfmp"
    elif method == "sir":
        replan = controller.run_sir
        interp = hermite_spline
        family = "pfmp"
    elif method == "ps" or method == "cem" or method == "mppi":
        replan = jax.jit(controller.optimize)
        interp = jax.jit(controller.interp_func)
        family = "hydrax"
        assert theta is None

    # Report the planning horizon in seconds for debugging
    if family == "pfmp":
        ctrl_steps = round(controller.horizon / controller.dt)
        horizon = controller.horizon
    else:
        ctrl_steps = controller.ctrl_steps
        horizon = controller.plan_horizon
    print(
        f"Planning with {ctrl_steps} steps "
        f"over a {horizon} second horizon "
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

    key = jax.random.key(0)

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()

    if family == "pfmp":
        init_shape = theta.mean.shape
        print(f"knots shape: {init_shape}")
        key, subkey = jax.random.split(key)
        (trajs, costs) = replan(subkey, mjx_data, theta)
        # assert theta.mean.shape == init_shape
        theta, _ = ps_update(theta, trajs, costs)
        # assert theta.mean.shape == init_shape
        key, subkey = jax.random.split(key)
        (trajs, costs) = replan(subkey, mjx_data, theta)
        theta, _ = ps_update(theta, trajs, costs)
        # assert theta.mean.shape == init_shape
    else:
        policy_params = controller.init_params(initial_knots=initial_knots)
        policy_params, rollouts = replan(mjx_data, policy_params)
        policy_params, rollouts = replan(mjx_data, policy_params)

        tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
        tk = policy_params.tk
        knots = policy_params.mean[None, ...]
        _ = interp(tq, tk, knots)
        _ = interp(tq, tk, knots)

    print(f"Time to jit: {time.time() - st:.3f} seconds")

    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir="recordings",
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    if record_traj:
        traj = Trajectory()

    # Start the simulation
    step = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # t_init = mj_data.time
        while viewer.is_running() and step < max_step:
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

            # Replan
            if family == "pfmp":
                (key, subkey) = jax.random.split(key)
                (trajs, costs) = replan(subkey, mjx_data, theta)
                total_costs = jnp.sum(costs, axis=1)
                ess_val = ess(costs)
                print(f"cost min={float(total_costs.min()):.3f} max={float(total_costs.max()):.3f} std={float(total_costs.std()):.3f}  ESS/K={float(ess_val):.3f}")
                theta, _ = ps_update(theta, trajs, costs)
            else:
                policy_params, rollouts = replan(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep

            # theta.mean = jnp.concat([u0, theta.mean])
            if family == "pfmp":
                tk = controller.tk()
                tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
                us = hermite_spline(tq, tk, theta.mean)
            else:
                t_curr = mj_data.time
                tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
                tk = policy_params.tk
                knots = policy_params.mean[None, ...]
                us = np.asarray(interp(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            print(f"T qpos: {mj_data.qpos[7:10]}, mean[0]: {np.array(theta.mean[0])}")
            for i in range(sim_steps_per_replan):
                step_start = time.time()

                if ik_interp is not None:
                    target = us[i]
                    u = ik_interp(target, theta, mjx_data)
                else:
                    u = us[i]
                mj_data.ctrl[:] = np.array(u)
                mujoco.mj_step(mj_model, mj_data)
                
                # Draw goal marker
                if goal_marker_fn is not None:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        goal_marker_fn(viewer.user_scn, theta)
                
                viewer.sync()

                step_elapsed = time.time() - step_start
                if step_elapsed < sim_dt:
                    time.sleep(sim_dt - step_elapsed)

                if record_traj:
                    if family == "pfmp":
                        cost = controller.mpc_running_cost(mj_data, mj_data.ctrl, theta, mj_data.time)
                    else:
                        cost = controller.task.running_cost(mj_data, mj_data.ctrl) * mj_model.opt.timestep
                    traj.push(mj_data.ctrl, cost, mj_data.qpos, mj_data.qvel)
                step += 1

            # Capture frame if recording
            if record_video and recorder.is_recording:
                renderer.update_scene(mj_data, viewer.cam)
                frame = renderer.render()
                recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            # elapsed = time.time() - start_time
            # if elapsed < step_dt:
            #     time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

            if family == "pfmp":
                tk = controller.tk()
                new_tk = tk + sim_steps_per_replan * mj_model.opt.timestep
                if isinstance(theta.mean, Jet2):
                    nout = 2
                else:
                    nout = 1
                new_mean = hermite_spline(new_tk, tk, theta.mean, nout=nout)
                if isinstance(theta.mean, Jet2):
                    theta = theta._replace(mean=new_mean)
                else:
                    if new_mean.shape != init_shape:
                        # 1D array
                        theta = theta._replace(mean=new_mean[:,None])
                    else:
                        theta = theta._replace(mean=new_mean)
                    # assert theta.mean.shape == init_shape

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()

    if record_traj:
        traj.save()
