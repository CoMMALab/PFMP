"""
Train Double CartPole swingup via mjlab (MuJoCo-Warp + RSL-RL PPO).

Train:
    WANDB_MODE=disabled python experiments/mjlab/mjlab_double_cartpole.py --train \
        --num-envs 1024

Play (after training):
    python experiments/mjlab/mjlab_double_cartpole.py --checkpoint logs/rsl_rl/double_cartpole/<run>/model_500.pt
"""

import argparse
import torch
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from pathlib import Path
from hydrax.tasks.double_cart_pole import DoubleCartPole
import pfmp


# ── Task ──────────────────────────────────────────────────────────────────────

task      = DoubleCartPole()
mj_model  = task.mj_model
mjx_model = mjx.put_model(mj_model)

# obs: cart_pos, cos/sin(hinge1), cos/sin(hinge2), cart_vel, hinge1_vel, hinge2_vel
OBS_DIM    = 8
ACTION_DIM = mj_model.nu  # 1

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


def obs_fn(x):
    return jnp.array([
        x.qpos[0],
        jnp.cos(x.qpos[1]), jnp.sin(x.qpos[1]),
        jnp.cos(x.qpos[2]), jnp.sin(x.qpos[2]),
        x.qvel[0], x.qvel[1], x.qvel[2],
    ])


# ── mjlab task registration ───────────────────────────────────────────────────

def register_double_cartpole(max_iters=1000):
    import mujoco as _mj
    from mjlab.actuator.xml_actuator import XmlActuatorCfg
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, reset_joints_by_offset, time_out
    from mjlab.envs.mdp.actions import JointEffortActionCfg
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.viewer import ViewerConfig
    from mjlab.tasks.registry import register_mjlab_task

    XML_PATH = str(Path(__file__).parent.parent /
                   ".venv/lib/python3.12/site-packages/hydrax/models/double_cart_pole/scene.xml")

    _CART_CFG  = SceneEntityCfg("cartpole", joint_names=("slider",))
    _H1_CFG    = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
    _H2_CFG    = SceneEntityCfg("cartpole", joint_names=("hinge_2",))

    def _get_spec():
        return _mj.MjSpec.from_file(XML_PATH)

    articulation = EntityArticulationInfoCfg(
        actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
    )
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"slider": 0.0, "hinge_1": 0.0, "hinge_2": 0.0},
        joint_vel={".*": 0.0},
    )
    entity_cfg = EntityCfg(spec_fn=_get_spec, articulation=articulation, init_state=init_state)

    def pole_angle_cos_sin_1(env, asset_cfg=_H1_CFG):
        import torch as _t
        asset: Entity = env.scene[asset_cfg.name]
        angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return _t.cat([_t.cos(angle), _t.sin(angle)], dim=-1)

    def pole_angle_cos_sin_2(env, asset_cfg=_H2_CFG):
        import torch as _t
        asset: Entity = env.scene[asset_cfg.name]
        angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return _t.cat([_t.cos(angle), _t.sin(angle)], dim=-1)

    def pole_velocity_exceeded(env, asset_cfg=_H1_CFG):
        import torch as _t
        asset: Entity = env.scene[asset_cfg.name]
        vel = asset.data.joint_vel.abs()
        return (vel > 30.0).any(dim=-1)

    def double_cartpole_reward(env, cart_cfg=_CART_CFG, h1_cfg=_H1_CFG, h2_cfg=_H2_CFG):
        import torch as _t
        asset: Entity = env.scene[cart_cfg.name]
        h1 = asset.data.joint_pos[:, h1_cfg.joint_ids].squeeze(-1)
        h2 = asset.data.joint_pos[:, h2_cfg.joint_ids].squeeze(-1)
        # h1 up = π from cart, h2 straight = 0 relative to h1; additive so each pole gets gradient
        upright = ((1 - _t.cos(h1)) / 2 + (1 + _t.cos(h2)) / 2) / 2
        cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
        centered = (1 + _t.exp(-0.5 * (cart_pos / 2.0) ** 2)) / 2
        control = env.action_manager.action.squeeze(-1)
        small_control = (4 + _t.clamp(1 - (control / 1.0) ** 2, min=0.0)) / 5
        return upright * centered * small_control

    actor_terms = {
        "cart_pos":  ObservationTermCfg(func=joint_pos_rel,       params={"asset_cfg": _CART_CFG}),
        "pole1_cs":  ObservationTermCfg(func=pole_angle_cos_sin_1, params={"asset_cfg": _H1_CFG}),
        "pole2_cs":  ObservationTermCfg(func=pole_angle_cos_sin_2, params={"asset_cfg": _H2_CFG}),
        "cart_vel":  ObservationTermCfg(func=joint_vel_rel,        params={"asset_cfg": _CART_CFG}),
        "h1_vel":    ObservationTermCfg(func=joint_vel_rel,        params={"asset_cfg": _H1_CFG}),
        "h2_vel":    ObservationTermCfg(func=joint_vel_rel,        params={"asset_cfg": _H2_CFG}),
    }

    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"cartpole": entity_cfg},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations={"actor": ObservationGroupCfg(actor_terms), "critic": ObservationGroupCfg({**actor_terms})},
        actions={"effort": JointEffortActionCfg(entity_name="cartpole", actuator_names=("slider",), scale=1.0)},
        events={
            "reset_slider": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (0.0, 0.0), "velocity_range": (-0.01, 0.01),
                        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",))}),
            "reset_h1": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-3.14, 3.14), "velocity_range": (-0.5, 0.5),
                        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",))}),
            "reset_h2": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-3.14, 3.14), "velocity_range": (-0.5, 0.5),
                        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_2",))}),
        },
        rewards={"reward": RewardTermCfg(func=double_cartpole_reward, weight=1.0,
                    params={"cart_cfg": _CART_CFG, "h1_cfg": _H1_CFG, "h2_cfg": _H2_CFG})},
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "velocity_exceeded": TerminationTermCfg(
                func=pole_velocity_exceeded, params={"asset_cfg": _H1_CFG}, time_out=False,
            ),
        },
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",))),
        decimation=5,
        episode_length_s=50.0,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="cartpole", body_name="cart",
                            distance=6.0, elevation=-15.0, azimuth=0.0),
    )

    runner_cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128), activation="elu",
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="double_cartpole",
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=max_iters,
    )

    register_mjlab_task(
        task_id="Mjlab-DoubleCartpole-Swingup",
        env_cfg=env_cfg,
        play_env_cfg=env_cfg,
        rl_cfg=runner_cfg,
    )


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_actor_params(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    def w(key): return jnp.array(sd[key].numpy())
    return {
        "W0": w("mlp.0.weight"), "b0": w("mlp.0.bias"),
        "W1": w("mlp.2.weight"), "b1": w("mlp.2.bias"),
        "Wa": w("mlp.4.weight"), "ba": w("mlp.4.bias"),
        "log_std": w("distribution.std_param"),
    }


def actor_forward(params, obs):
    elu = lambda x: jnp.where(x >= 0, x, jnp.exp(x) - 1)
    x = elu(obs @ params["W0"].T + params["b0"])
    x = elu(x   @ params["W1"].T + params["b1"])
    return x @ params["Wa"].T + params["ba"], params["log_std"]


# ── PFMP Proposal ─────────────────────────────────────────────────────────────

def make_proposal(deterministic=False, blend=0.0):
    """blend: fraction of particles drawn from linear Gaussian (0=all learned, 1=all linear, 0.5=half-half)."""
    def _sample(key, x, theta):
        key1, key2, key3 = jax.random.split(key, 3)
        mean, _ = actor_forward(theta.actor_params, obs_fn(x))
        if deterministic:
            return jnp.clip(mean, -1, 1)
        learned = jnp.clip(mean + theta.noise_level * jax.random.normal(key1, mean.shape), -1, 1)
        linear  = jnp.clip(       theta.noise_level * jax.random.normal(key2, mean.shape), -1, 1)
        return jnp.where(jax.random.uniform(key3) < blend, linear, learned)

    def init(key, x, theta):
        return _sample(key, x, theta)

    def proposal(key, x, _u, theta, _t):
        return _sample(key, x, theta)

    return init, proposal


# ── Train ─────────────────────────────────────────────────────────────────────

def train(num_envs: int = 1024, iters: int = 1000):
    import os, sys
    register_double_cartpole(iters)
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["train", "Mjlab-DoubleCartpole-Swingup", f"--env.scene.num-envs={num_envs}"]
    from mjlab.scripts.train import main
    main()


# ── Interactive eval ──────────────────────────────────────────────────────────

def run_interactive(checkpoint_path: str, particles: int = 128, deterministic: bool = True):
    actor_params = load_actor_params(checkpoint_path)
    HORIZON, NUM_KNOTS = 1.0, 4
    init_fn, proposal_fn = make_proposal(deterministic=deterministic)
    ctrl = pfmp.PFMP(
        pfmp.mjx_physics, running_cost, terminal_cost,
        init_fn, proposal_fn,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.5,
    )
    theta = pfmp.GenericMJXTaskParams(
        mjx_model=mjx_model,
        mean=jnp.zeros((NUM_KNOTS, ACTION_DIM)),
        noise_level=0.025,
        actor_params=actor_params,
    )
    pfmp.run_interactive(
        ctrl, theta, mj_model, mujoco.MjData(mj_model),
        frequency=50, fixed_camera_id=0, method="sir",
        record_video=False, record_traj=False, max_step=1000,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--particles", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(args.num_envs, args.iters)
    elif args.checkpoint:
        run_interactive(args.checkpoint, args.particles, deterministic=not args.stochastic)
    else:
        parser.print_help()
