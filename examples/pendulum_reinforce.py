"""
Learned proposal for inverted pendulum via REINFORCE.

Train: unroll particles, compute per-knot log_prob weighted by cost-to-go.
Eval:  run_sis or run_sir with the learned proposal.
"""

import argparse
import jax
import jax.numpy as jnp
import flax.linen as nn
import mujoco
import optax
import pfmp

from hydrax.tasks.cart_pole import CartPole
from mujoco import mjx


# ── Task ──────────────────────────────────────────────────────────────────────

task      = CartPole()
mj_model  = task.mj_model
mj_data   = mujoco.MjData(mj_model)

HORIZON    = 1.0
NUM_KNOTS  = 4
PARTICLES  = 128
OBS_DIM    = mj_model.nq + mj_model.nv  # 4
ACTION_DIM = mj_model.nu                # 1

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


# ── Policy ────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -1
LOG_STD_MAX = 2

class Policy(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array):
        x = nn.relu(nn.Dense(256)(obs))
        x = nn.relu(nn.Dense(256)(x))
        mean    = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std


policy = Policy(action_dim=ACTION_DIM)


def obs_fn(x):
    return jnp.concatenate([x.qpos, x.qvel])


def make_proposal(policy: Policy, deterministic: bool = False):
    def init(key, x, theta):
        mean, log_std = policy.apply(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        std = jnp.exp(log_std)
        return jnp.clip(mean + std * jax.random.normal(key, mean.shape), -1, 1)

    def proposal(key, x, _u, theta, _t):
        mean, log_std = policy.apply(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        std = jnp.exp(log_std)
        return jnp.clip(mean + std * jax.random.normal(key, mean.shape), -1, 1)

    return init, proposal


def log_prob(params, x, u):
    mean, log_std = policy.apply(params, obs_fn(x))
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(((u - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))


# ── Controller ────────────────────────────────────────────────────────────────

init, proposal = make_proposal(policy)
ctrl = pfmp.PFMP(
    pfmp.mjx_physics,
    running_cost, terminal_cost,
    init, proposal,
    horizon=HORIZON,
    num_knots=NUM_KNOTS,
    particles=PARTICLES,
    dt=mj_model.opt.timestep,
    ess_resample_thresh=0.3,
)

key    = jax.random.key(0)
params = policy.init(key, jnp.zeros(OBS_DIM))
theta  = pfmp.GenericMJXTaskParams(
    mjx_model=mjx.put_model(mj_model),
    mean=jnp.zeros((NUM_KNOTS, ACTION_DIM)),
    noise_level=0.3,
    actor_params=params,
)


# ── Training ──────────────────────────────────────────────────────────────────

ALPHA = 0.2
EPS   = 0.2
GAMMA = 0.99

tx     = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
opt_st = tx.init(params)


@jax.jit
def step(params, opt_st, key, x0):
    th = theta._replace(actor_params=params)
    us, costs, xs = ctrl.unroll(key, x0, th)

    T = costs.shape[1]
    gammas = GAMMA ** jnp.arange(T)
    discounted = costs * gammas[None, :]
    returns = jnp.cumsum(discounted[:, ::-1], axis=1)[:, ::-1] / gammas[None, :]

    xs_sg  = jax.lax.stop_gradient(xs)
    us_sg  = jax.lax.stop_gradient(us[:, 1:])
    R_sg   = jax.lax.stop_gradient(returns[:, 1:])
    lp_old = jax.lax.stop_gradient(
        jax.vmap(jax.vmap(log_prob, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(params, xs_sg, us_sg)
    )

    def loss_fn(p):
        lp  = jax.vmap(jax.vmap(log_prob, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(p, xs_sg, us_sg)
        adv = -(R_sg - R_sg.mean()) / (R_sg.std() + 1e-8)
        ratio = jnp.exp(lp - lp_old)
        return -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - EPS, 1 + EPS) * adv))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_st = tx.update(grads, opt_st, params)
    return optax.apply_updates(params, updates), opt_st, costs, loss, grads


x0 = mjx.put_data(mj_model, mj_data)
print("Compiling...")
for i in range(10000):
    key, k = jax.random.split(key)
    params, opt_st, costs, loss, grads = step(params, opt_st, k, x0)
    if i % 20 == 0:
        returns = costs.sum(axis=1)
        gnorm = optax.global_norm(grads)
        print(f"[{i:4d}]  loss={float(loss):.4f}  best={float(returns.min()):.4f}  mean={float(returns.mean()):.4f}  |g|={float(gnorm):.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["sis", "sir"], default="sir")
args = parser.parse_args()

theta = theta._replace(actor_params=params)
eval_init, eval_proposal = make_proposal(policy, deterministic=True)
eval_ctrl = ctrl._replace(particles=1, mpc_init=eval_init, mpc_proposal=eval_proposal)
pfmp.run_interactive(
    eval_ctrl, theta, mj_model, mj_data,
    frequency=50,
    fixed_camera_id=0,
    method=args.method,
    record_video=False,
    record_traj=False,
    max_step=1000,
)
