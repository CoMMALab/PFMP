"""
Train CartPole swingup with SB3 PPO on the actual hydrax task cost.
Extracts actor and wraps as a PFMP proposal.
"""

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import gymnasium as gym
import pfmp

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from hydrax.tasks.cart_pole import CartPole


# ── Gymnasium wrapper ─────────────────────────────────────────────────────────

task     = CartPole()
mj_model = task.mj_model


class CartPoleSwingupEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.mj_model = task.mj_model
        self.mj_data  = mujoco.MjData(self.mj_model)
        obs_dim = self.mj_model.nq + self.mj_model.nv
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (self.mj_model.nu,), np.float32)

    def _obs(self):
        return np.concatenate([self.mj_data.qpos, self.mj_data.qvel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # small random perturbation from default (pole-down) start
        self.mj_data.qpos[:] += self.np_random.uniform(-0.05, 0.05, self.mj_model.nq)
        self.mj_data.qvel[:] += self.np_random.uniform(-0.05, 0.05, self.mj_model.nv)
        return self._obs(), {}

    def step(self, action):
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        obs    = self._obs()
        cost   = task.running_cost(
            # task.running_cost accepts mj_data-like via qpos/qvel attributes
            self.mj_data, action
        )
        reward = -float(cost)
        return obs, reward, False, False, {}


# ── Train ─────────────────────────────────────────────────────────────────────

vec_env = make_vec_env(CartPoleSwingupEnv, n_envs=8)

model = PPO(
    "MlpPolicy", vec_env, device="cpu",
    n_steps=256,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    learning_rate=3e-4,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
)
model.learn(total_timesteps=500_000)
model.save("cartpole_ppo")
print("Saved cartpole_ppo.zip")


# ── Wrap as PFMP proposal ─────────────────────────────────────────────────────

import torch

# Extract weights from SB3's actor MLP into a simple JAX forward pass
sb3_actor = model.policy.mlp_extractor.policy_net
sb3_action_net = model.policy.action_net

def get_param(module, idx):
    params = list(module.parameters())
    return params[idx].detach().numpy()

# Two hidden layers (64, 64) + action head
W0, b0 = get_param(sb3_actor, 0), get_param(sb3_actor, 1)
W1, b1 = get_param(sb3_actor, 2), get_param(sb3_actor, 3)
Wa, ba = get_param(sb3_action_net, 0), get_param(sb3_action_net, 1)

# SB3 PPO outputs action mean; log_std is a separate parameter
log_std = model.policy.log_std.detach().numpy()

actor_params = {
    "W0": jnp.array(W0), "b0": jnp.array(b0),
    "W1": jnp.array(W1), "b1": jnp.array(b1),
    "Wa": jnp.array(Wa), "ba": jnp.array(ba),
    "log_std": jnp.array(log_std),
}


def actor_forward(params, obs):
    x = jnp.tanh(obs @ params["W0"].T + params["b0"])
    x = jnp.tanh(x   @ params["W1"].T + params["b1"])
    return x @ params["Wa"].T + params["ba"], params["log_std"]


def make_proposal(deterministic=False):
    def obs_fn(x):
        return jnp.concatenate([x.qpos, x.qvel])

    def init(key, x, theta):
        mean, log_std = actor_forward(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        return jnp.clip(mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape), -1, 1)

    def proposal(key, x, _u, theta, _t):
        mean, log_std = actor_forward(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        return jnp.clip(mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape), -1, 1)

    return init, proposal


HORIZON   = 1.0
NUM_KNOTS = 4
PARTICLES = 128

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
mjx_model = mjx.put_model(mj_model)

init_fn, proposal_fn = make_proposal()
ctrl = pfmp.PFMP(
    pfmp.mjx_physics,
    running_cost, terminal_cost,
    init_fn, proposal_fn,
    horizon=HORIZON,
    num_knots=NUM_KNOTS,
    particles=PARTICLES,
    dt=mj_model.opt.timestep,
    ess_resample_thresh=0.3,
)
theta = pfmp.GenericMJXTaskParams(
    mjx_model=mjx_model,
    mean=jnp.zeros((NUM_KNOTS, mj_model.nu)),
    noise_level=0.3,
    actor_params=actor_params,
)

eval_init, eval_proposal = make_proposal(deterministic=True)
eval_ctrl = ctrl._replace(particles=1, mpc_init=eval_init, mpc_proposal=eval_proposal)
pfmp.run_interactive(
    eval_ctrl, theta, mj_model, mujoco.MjData(mj_model),
    frequency=50, fixed_camera_id=0, method="sir",
    record_video=False, record_traj=False, max_step=1000,
)
