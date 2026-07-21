"""
PPO for CartPole (MuJoCo/MJX) — adapted line-for-line from PureJaxRL's ppo.py.

gymnax  → mjx env_reset / env_step (with auto-reset on termination)
distrax → inline Gaussian sample / log_prob / entropy
wrappers → LogWrapper replaced by inline done tracking

After training, actor params are extracted and wrapped as a PFMP proposal.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import mujoco
import mujoco.mjx as mjx
import pfmp

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple
from hydrax.tasks.cart_pole import CartPole


# ── Task / Environment ────────────────────────────────────────────────────────

task      = CartPole()
mj_model  = task.mj_model
mjx_model = mjx.put_model(mj_model)
mjx_data_0 = mjx.put_data(mj_model, mujoco.MjData(mj_model))

OBS_DIM    = mj_model.nq + mj_model.nv  # 4
ACTION_DIM = mj_model.nu                # 1

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


def obs_fn(x):
    return jnp.concatenate([x.qpos, x.qvel])


UPRIGHT_ANGLE = -jnp.pi   # qpos[1] = -pi means pole upright
FALL_THRESH   = jnp.pi / 2  # terminate if pole > 90 deg from upright


def env_reset(key):
    # Start near upright so there's a meaningful balance task
    noise = jax.random.uniform(key, (OBS_DIM,), minval=-0.05, maxval=0.05)
    x = mjx_data_0.replace(
        qpos=jnp.array([noise[0], UPRIGHT_ANGLE + noise[1]]),
        qvel=noise[2:] * 0.1,
    )
    return obs_fn(x), x


def env_step(key, env_state, action):
    x = mjx.step(mjx_model, env_state.replace(ctrl=action))
    obs = obs_fn(x)
    reward = -task.running_cost(x, action) * 0.01
    # Terminate when pole falls more than FALL_THRESH from upright
    done = jnp.abs(x.qpos[1] - UPRIGHT_ANGLE) > FALL_THRESH
    # Auto-reset: replace state with fresh reset on termination (gymnax LogWrapper)
    reset_obs, reset_state = env_reset(key)
    obs   = jnp.where(done, reset_obs, obs)
    x     = jax.tree.map(lambda a, b: jnp.where(done, a, b), reset_state, x)
    return obs, x, reward, done


# ── Network ───────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.tanh if self.activation == "tanh" else nn.relu

        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = act(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = act(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = act(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return actor_mean, actor_log_std, jnp.squeeze(critic, axis=-1)


def gaussian_sample(key, mean, log_std):
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def gaussian_log_prob(mean, log_std, action):
    std = jnp.exp(log_std)
    return jnp.sum(-0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi))


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


# ── Transition ────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    done:     jnp.ndarray
    action:   jnp.ndarray
    value:    jnp.ndarray
    reward:   jnp.ndarray
    log_prob: jnp.ndarray
    obs:      jnp.ndarray


# ── Training ──────────────────────────────────────────────────────────────────

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(ACTION_DIM, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, jnp.zeros(OBS_DIM))
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env_reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                actor_mean, actor_log_std, value = network.apply(train_state.params, last_obs)
                sample_keys = jax.random.split(_rng, config["NUM_ENVS"])
                action = jax.vmap(gaussian_sample, in_axes=(0, 0, None))(
                    sample_keys, actor_mean, actor_log_std
                )
                log_prob = jax.vmap(gaussian_log_prob, in_axes=(0, None, 0))(
                    actor_mean, actor_log_std, action
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done = jax.vmap(env_step)(rng_step, env_state, action)
                transition = Transition(done, action, value, reward, log_prob, last_obs)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE (GAE)
            train_state, env_state, last_obs, rng = runner_state
            _, _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        actor_mean, actor_log_std, value = network.apply(params, traj_batch.obs)
                        log_prob = jax.vmap(gaussian_log_prob, in_axes=(0, None, 0))(
                            actor_mean, actor_log_std, traj_batch.action
                        )

                        # VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()

                        # ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                        ).mean()

                        entropy = gaussian_entropy(actor_log_std)
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            if config.get("DEBUG"):
                def callback(info):
                    loss_info, reward_info = info
                    total_loss, (vl, al, ent) = loss_info
                    mean_r, ep_done = reward_info
                    print(
                        f"loss={float(total_loss.mean()):.4f}  "
                        f"vl={float(vl.mean()):.4f}  "
                        f"al={float(al.mean()):.4f}  "
                        f"ent={float(ent.mean()):.4f}  "
                        f"r={float(mean_r):.4f}  "
                        f"ep_done={float(ep_done):.2f}"
                    )
                mean_r   = traj_batch.reward.mean()
                ep_done  = traj_batch.done.mean()
                jax.debug.callback(callback, (loss_info, (mean_r, ep_done)))

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, loss_info

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ── Run ───────────────────────────────────────────────────────────────────────

config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "DEBUG": True,
}

rng = jax.random.PRNGKey(30)
print("Compiling...")
train_fn = jax.jit(make_train(config))
out = train_fn(rng)
actor_params = out["runner_state"][0].params
print("Training done.")


# ── PFMP Proposal ─────────────────────────────────────────────────────────────

HORIZON   = 1.0
NUM_KNOTS = 4
PARTICLES = 128

network = ActorCritic(ACTION_DIM, activation=config["ACTIVATION"])


def make_proposal(deterministic=False):
    def init(key, x, theta):
        mean, log_std, _ = network.apply(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        return jnp.clip(gaussian_sample(key, mean, log_std), -1, 1)

    def proposal(key, x, _u, theta, _t):
        mean, log_std, _ = network.apply(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        return jnp.clip(gaussian_sample(key, mean, log_std), -1, 1)

    return init, proposal


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
    mean=jnp.zeros((NUM_KNOTS, ACTION_DIM)),
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
