"""
Train/eval Unitree Go1 flat-terrain velocity tracking via mjlab.

Train:
    WANDB_MODE=disabled python experiments/mjlab/mjlab_go1.py --train --num-envs 1024

Play (after training):
    python experiments/mjlab/mjlab_go1.py --checkpoint logs/rsl_rl/go1_velocity/<run>/model_*.pt
"""

import argparse
import torch
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from typing import NamedTuple
from pathlib import Path

import pfmp
import mjlab.tasks.velocity.config.go1  # registers the task

# ── Model ─────────────────────────────────────────────────────────────────────

from mjlab.asset_zoo.robots import get_go1_robot_cfg
_entity_cfg = get_go1_robot_cfg()
_spec       = _entity_cfg.spec_fn()
mj_model    = _spec.compile()
mjx_model   = mjx.put_model(mj_model)

ACTION_DIM = 12   # 3 joints × 4 legs
# obs: lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + joint_pos(12) + joint_vel(12) + last_action(12) + cmd(3)
OBS_DIM    = 48

# Default standing joint positions (thigh=0.9, calf=-1.8, R_hip=0.1, L_hip=-0.1)
# Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
Q_DEFAULT = jnp.array([
     0.1,  0.9, -1.8,   # FR
    -0.1,  0.9, -1.8,   # FL
     0.1,  0.9, -1.8,   # RR
    -0.1,  0.9, -1.8,   # RL
])

# Position action scale (matches JointPositionActionCfg scale in velocity env)
ACTION_SCALE = 0.5


# ── Task params ───────────────────────────────────────────────────────────────

class Go1TaskParams(NamedTuple):
    mjx_model:   mjx.Model
    mean:        jax.Array
    noise_level: float
    actor_params: any        = None
    vel_cmd:     jax.Array  = None   # [vx, vy, wz]


# ── JAX obs / cost ────────────────────────────────────────────────────────────

def _quat_rotate_inv(q, v):
    """Rotate vector v by q^{-1}, q = [w, x, y, z]."""
    w, xyz = q[0], q[1:]
    return v + 2*w*jnp.cross(xyz, v) + 2*jnp.cross(xyz, jnp.cross(xyz, v))


def obs_fn(x, u_prev, vel_cmd):
    """48-dim obs matching the mjlab velocity task actor observation."""
    q = x.qpos[3:7]  # root quaternion [w, x, y, z]
    return jnp.concatenate([
        _quat_rotate_inv(q, x.qvel[0:3]),               # base_lin_vel_b    (3)
        _quat_rotate_inv(q, x.qvel[3:6]),               # base_ang_vel_b    (3)
        _quat_rotate_inv(q, jnp.array([0., 0., -1.])),  # projected_gravity (3)
        x.qpos[7:19] - Q_DEFAULT,                       # joint_pos_rel     (12)
        x.qvel[6:18],                                   # joint_vel         (12)
        u_prev,                                         # last_action       (12)
        vel_cmd,                                        # velocity command  (3)
    ])


def go1_running_cost(x, u, th, t):
    q = x.qpos[3:7]
    lin_vel_b = _quat_rotate_inv(q, x.qvel[0:3])
    cmd_xy    = th.vel_cmd[:2]
    vel_err   = jnp.sum((lin_vel_b[:2] - cmd_xy) ** 2)
    ctrl_cost = 0.001 * jnp.sum(u ** 2)
    fall_cost = jnp.where(x.qpos[2] < 0.15, 5.0, 0.0)
    return (vel_err + ctrl_cost + fall_cost) * 0.01


def go1_terminal_cost(x, th):
    return jnp.where(x.qpos[2] < 0.15, 1.0, 0.0)


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_actor_params(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    def w(key): return jnp.array(sd[key].numpy())
    return {
        "W0": w("mlp.0.weight"), "b0": w("mlp.0.bias"),
        "W1": w("mlp.2.weight"), "b1": w("mlp.2.bias"),
        "W2": w("mlp.4.weight"), "b2": w("mlp.4.bias"),
        "Wa": w("mlp.6.weight"), "ba": w("mlp.6.bias"),
        "log_std": w("distribution.std_param"),
    }


def actor_forward(params, obs):
    elu = lambda x: jnp.where(x >= 0, x, jnp.exp(x) - 1)
    x = elu(obs @ params["W0"].T + params["b0"])
    x = elu(x   @ params["W1"].T + params["b1"])
    x = elu(x   @ params["W2"].T + params["b2"])
    return x @ params["Wa"].T + params["ba"], params["log_std"]


# ── PFMP Proposal ─────────────────────────────────────────────────────────────

def make_proposal(deterministic=False, blend=0.0):
    """Policy outputs joint position targets; converted to ctrl via Q_DEFAULT + scale * action."""
    def _sample(key, x, u_prev, theta):
        key1, key2, key3 = jax.random.split(key, 3)
        mean, _ = actor_forward(theta.actor_params, obs_fn(x, u_prev, theta.vel_cmd))
        # Convert network output to joint position ctrl: q_default + scale * mean
        mean_ctrl = Q_DEFAULT + ACTION_SCALE * mean
        if deterministic:
            return mean_ctrl
        learned = mean_ctrl + theta.noise_level * jax.random.normal(key1, mean_ctrl.shape)
        linear  = Q_DEFAULT + theta.noise_level * jax.random.normal(key2, mean_ctrl.shape)
        return jnp.where(jax.random.uniform(key3) < blend, linear, learned)

    def init(key, x, theta):
        u0 = Q_DEFAULT  # start from default standing pose
        return _sample(key, x, u0, theta)

    def proposal(key, x, u_prev, theta, _t):
        return _sample(key, x, u_prev, theta)

    return init, proposal


# ── Train ─────────────────────────────────────────────────────────────────────

def train(num_envs: int = 1024):
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["train", "Mjlab-Velocity-Flat-Unitree-Go1",
                f"--env.scene.num-envs={num_envs}"]
    from mjlab.scripts.train import main
    main()


# ── Interactive eval ──────────────────────────────────────────────────────────

def run_interactive(
    checkpoint_path: str,
    particles: int = 64,
    deterministic: bool = True,
    vel_cmd: tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    actor_params = load_actor_params(checkpoint_path)
    HORIZON, NUM_KNOTS = 0.5, 4
    init_fn, proposal_fn = make_proposal(deterministic=deterministic)
    ctrl = pfmp.PFMP(
        pfmp.mjx_physics, go1_running_cost, go1_terminal_cost,
        init_fn, proposal_fn,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.5,
    )
    theta = Go1TaskParams(
        mjx_model=mjx_model,
        mean=jnp.tile(Q_DEFAULT, (NUM_KNOTS, 1)),
        noise_level=0.05,
        actor_params=actor_params,
        vel_cmd=jnp.array(vel_cmd),
    )
    pfmp.run_interactive(
        ctrl, theta, mj_model, mujoco.MjData(mj_model),
        frequency=50, fixed_camera_id=0, method="sir",
        record_video=False, record_traj=False, max_step=2000,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--num-envs",   type=int,   default=1024)
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--particles",  type=int,   default=64)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--vel-cmd",    type=float, nargs=3,
                        default=[1.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    args = parser.parse_args()

    if args.train:
        train(args.num_envs)
    elif args.checkpoint:
        run_interactive(
            args.checkpoint, args.particles,
            deterministic=not args.stochastic,
            vel_cmd=tuple(args.vel_cmd),
        )
    else:
        parser.print_help()
