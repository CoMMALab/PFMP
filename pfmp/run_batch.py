import time
import numpy as np
import mujoco
import jax
import jax.numpy as jnp

from mujoco import mjx
from dataclasses import dataclass
from pfmp.interpolate import Jet2, hermite_spline
from pfmp.algorithm import ps_update


@dataclass
class Trajectory:
    ctrl: np.ndarray  # (B, T, nu)
    cost: np.ndarray  # (B, T)
    qpos: np.ndarray  # (B, T, nq)
    qvel: np.ndarray  # (B, T, nv)

    def save(self, path):
        np.savez(path, ctrl=self.ctrl, cost=self.cost, qpos=self.qpos, qvel=self.qvel)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Trajectory(
            ctrl=data["ctrl"],
            cost=data["cost"],
            qpos=data["qpos"],
            qvel=data["qvel"]
        )

def _batch_scenes(sample_problem, mj_model, mj_data, theta, num):
    """Create batched (mjx_data, theta) for the actual simulation run.

    If sample_problem is provided, each batch element gets a unique scene
    (different initial state and/or goal).  Otherwise all elements share
    the same scene from mj_data / theta.
    """
    if sample_problem is not None:
        mjx_datas, thetas = [], []
        for i in range(num):
            mj_data_i, theta_i = sample_problem(i)
            mjx_datas.append(mjx.put_data(mj_model, mj_data_i))
            thetas.append(theta_i)
        mjx_data_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *mjx_datas)
        theta_batched = jax.tree.map(
            lambda *xs: jnp.stack([jnp.array(x) for x in xs]), *thetas
        )
    else:
        mjx_data_batched = mjx.put_data(mj_model, mj_data)
        mjx_data_batched = jax.tree.map(
            lambda x: jnp.repeat(x[None], num, axis=0), mjx_data_batched
        )
        theta_batched = jax.tree.map(
            lambda x: jnp.repeat(jnp.array(x)[None], num, axis=0), theta
        )
    return mjx_data_batched, theta_batched


def run_batch(  # noqa: PLR0912, PLR0915
    controller,
    theta,  # (mean, mjx_model)
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    algorithm = "sir",
    max_step: int = 1000,
    num: int = 1,
    ik_interp = None,
    sample_problem = None,
) -> Trajectory:
    """Run batched simulations with the MPC controller using MJX.

    This runs multiple trajectories in parallel using JAX vectorization for
    efficient GPU/TPU execution.

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
        algorithm: Algorithm to use ("sir", "sis", "ps", "cem", "mppi")
        max_step: Maximum number of simulation steps
        num: Number of trajectories to run in parallel (batched)
        ik_interp: Optional callable (target, theta) -> ctrl that maps planner
                   output to joint-space controls (e.g. IK cache lookup).
        sample_problem: Optional callable (seed: int) -> (mj_data, theta) that
                        produces a randomized scene.  If None, all batch
                        elements share the same initial state and goal.

    Returns:
        Trajectory with batched arrays of shape (num, T, ...)
    """
    if algorithm == "sis":
        replan = controller.run_sis
        interp = hermite_spline
        family = "pfmp"
    elif algorithm == "sir":
        replan = controller.run_sir
        interp = hermite_spline
        family = "pfmp"
    elif algorithm == "ps" or algorithm == "cem" or algorithm == "mppi":
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
        f"simulating at {1.0 / mj_model.opt.timestep} Hz, "
        f"running {num} trajectories in parallel"
    )

    # Create batched initial state (uniform — used for JIT warmup only)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Batch the initial state (B, ...)
    mjx_data = jax.tree.map(lambda x: jnp.repeat(x[None], num, axis=0), mjx_data)

    # Initialize random keys for each batch
    keys = jax.random.split(jax.random.key(0), num)

    # Warm-up jit compilation
    print("Jitting batched controller...")
    st = time.time()

    if family == "pfmp":
        init_shape = theta.mean.shape
        print(f"knots shape: {init_shape}")

        # Batch theta (uniform for warmup)
        theta_batched = jax.tree.map(lambda x: jnp.repeat(jnp.array(x)[None], num, axis=0), theta)

        # Create the step function for pfmp
        def pfmp_step_fn(carry, _):
            keys, mjx_data, theta_batched = carry

            # Split keys
            split_keys = jax.vmap(jax.random.split)(keys)
            keys, subkeys = split_keys[:, 0], split_keys[:, 1]

            # Replan
            trajs, costs = jax.vmap(replan)(subkeys, mjx_data, theta_batched)
            theta_batched = jax.vmap(lambda t, tr, c: ps_update(t, tr, c)[0])(theta_batched, trajs, costs)

            # Get controls
            tk = controller.tk()
            tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
            us = jax.vmap(lambda mean: hermite_spline(tq, tk, mean))(theta_batched.mean)  # (B, ss, nu)

            # Simulate forward
            def sim_step_fn(mjx_data_inner, u):
                if ik_interp is not None:
                    ctrl = jax.vmap(ik_interp)(u, theta_batched)
                else:
                    ctrl = u
                mjx_data_inner = mjx_data_inner.replace(ctrl=ctrl)
                mjx_data_inner = jax.vmap(lambda m, d: mjx.step(m, d), in_axes=[None,0])(mjx_model, mjx_data_inner)
                cost = jax.vmap(lambda d, c, t: controller.mpc_running_cost(d, c, t, d.time))(
                    mjx_data_inner, u, theta_batched
                )
                return mjx_data_inner, (u, cost, mjx_data_inner.qpos, mjx_data_inner.qvel)

            if len(us.shape) == 2:
                us = us[..., None]
            mjx_data, trajectory_data = jax.lax.scan(sim_step_fn, mjx_data, us.transpose(1, 0, 2))

            # Update theta for next iteration
            tk = controller.tk()
            new_tk = tk + sim_steps_per_replan * mj_model.opt.timestep
            if isinstance(theta_batched.mean[0], Jet2):
                nout = 2
            else:
                nout = 1
            new_mean = jax.vmap(lambda mean: hermite_spline(new_tk, tk, mean, nout=nout))(theta_batched.mean)
            if new_mean.shape[1:] != init_shape:
                new_mean = new_mean[:, :, None]
            theta_batched = theta_batched._replace(mean=new_mean)

            return (keys, mjx_data, theta_batched), trajectory_data

        # Warmup
        (keys, mjx_data, theta_batched), _ = pfmp_step_fn((keys, mjx_data, theta_batched), None)
        (keys, mjx_data, theta_batched), _ = pfmp_step_fn((keys, mjx_data, theta_batched), None)

        # JIT the full loop
        pfmp_step_fn = jax.jit(pfmp_step_fn)
        (keys, mjx_data, theta_batched), _ = pfmp_step_fn((keys, mjx_data, theta_batched), None)

    else:
        # For hydrax family
        seeds = jnp.arange(num)

        # Initialize policy_params with different rngs for each batch element
        policy_params_batched = jax.vmap(
            lambda s: controller.init_params(initial_knots=initial_knots, seed=s)
        )(seeds)

        def hydrax_step_fn(carry, _):
            mjx_data, policy_params_batched = carry

            # Replan - each instance uses its own policy_params.rng
            policy_params_batched, _ = jax.vmap(controller.optimize)(mjx_data, policy_params_batched)

            # Get controls
            tk = policy_params_batched.tk[0]
            tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep + mjx_data.time[0]
            knots = policy_params_batched.mean[:, None, ...]
            us = jax.vmap(lambda k: interp(tq, tk, k)[0])(knots)  # (B, ss, nu)

            # Simulate forward
            def sim_step_fn(mjx_data_inner, u):
                mjx_data_inner = mjx_data_inner.replace(ctrl=u)
                mjx_data_inner = jax.vmap(mjx.step, in_axes=[None, 0])(mjx_model, mjx_data_inner)
                cost = jax.vmap(
                    lambda d, c: controller.task.running_cost(d, c) * mj_model.opt.timestep
                )(mjx_data_inner, u)
                return mjx_data_inner, (u, cost, mjx_data_inner.qpos, mjx_data_inner.qvel)

            mjx_data, trajectory_data = jax.lax.scan(sim_step_fn, mjx_data, us.transpose(1, 0, 2))

            return (mjx_data, policy_params_batched), trajectory_data

        # Warmup
        (mjx_data, policy_params_batched), _ = hydrax_step_fn((mjx_data, policy_params_batched), None)
        (mjx_data, policy_params_batched), _ = hydrax_step_fn((mjx_data, policy_params_batched), None)

        # JIT the full loop
        hydrax_step_fn = jax.jit(hydrax_step_fn)
        (mjx_data, policy_params_batched), _ = hydrax_step_fn((mjx_data, policy_params_batched), None)

    print(f"Time to jit batched: {time.time() - st:.3f} seconds")

    # Run the full simulation using scan
    num_replans = max_step // sim_steps_per_replan
    print(f"Running {num_replans} replanning steps...")

    st = time.time()
    if family == "pfmp":
        # Reset to initial state (optionally randomized per batch element)
        mjx_data, theta_batched = _batch_scenes(
            sample_problem, mj_model, mj_data, theta, num
        )
        keys = jax.random.split(jax.random.key(0), num)

        _, all_trajectory_data = jax.lax.scan(
            pfmp_step_fn,
            (keys, mjx_data, theta_batched),
            None,
            length=num_replans
        )
    else:
        # Reset to initial state
        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = jax.tree.map(lambda x: jnp.repeat(x[None], num, axis=0), mjx_data)

        # Initialize policy_params with different seeds for each batch element
        seeds = jnp.arange(num)
        policy_params_batched = jax.vmap(
            lambda s: controller.init_params(initial_knots=initial_knots, seed=s)
        )(seeds)

        _, all_trajectory_data = jax.lax.scan(
            hydrax_step_fn,
            (mjx_data, policy_params_batched),
            None,
            length=num_replans
        )

    print(f"Simulation completed in {time.time() - st:.3f} seconds")

    # Unpack trajectory data
    # all_trajectory_data is (num_replans, sim_steps_per_replan, B, ...)
    ctrl_data, cost_data, qpos_data, qvel_data = all_trajectory_data

    # Reshape to (B, T, ...)
    ctrl_array = np.array(ctrl_data.transpose(2, 0, 1, 3).reshape(num, -1, ctrl_data.shape[-1]))
    cost_array = np.array(cost_data.transpose(2, 0, 1).reshape(num, -1))
    qpos_array = np.array(qpos_data.transpose(2, 0, 1, 3).reshape(num, -1, qpos_data.shape[-1]))
    qvel_array = np.array(qvel_data.transpose(2, 0, 1, 3).reshape(num, -1, qvel_data.shape[-1]))

    return Trajectory(ctrl=ctrl_array, cost=cost_array, qpos=qpos_array, qvel=qvel_array)
