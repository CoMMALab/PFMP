"""
Train the planar Walker (hydrax XML) via mjlab (MuJoCo-Warp + RSL-RL PPO).

Unlike the cart-poles, the walker is contact-rich (feet vs. floor), so contacts
are left enabled and the reward is a DeepMind-control-style stand x walk product.

Train:
    WANDB_MODE=disabled python experiments/mjlab/mjlab_walker.py --train --num-envs 4096

Play (after training):
    python experiments/mjlab/mjlab_walker.py --checkpoint logs/rsl_rl/walker/<run>/model_*.pt
"""

import argparse
import torch
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from pathlib import Path
from hydrax.tasks.walker import Walker
import pfmp


# ── Task ──────────────────────────────────────────────────────────────────────

task      = Walker()
mj_model  = task.mj_model
mjx_model = mjx.put_model(mj_model)

# qpos/qvel order (XML): rootz, rootx, rooty, right_hip, right_knee, right_ankle,
#                        left_hip, left_knee, left_ankle  (9 each, all 1-DoF)
# obs (18): root_height, cos/sin(pitch), 6 leg pos, 9 vel (incl. forward vel)
OBS_DIM    = 18
ACTION_DIM = mj_model.nu  # 6

STAND_HEIGHT = 1.2
WALK_SPEED   = 1.5
FALL_HEIGHT  = 0.6

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


def obs_fn(x):
    return jnp.concatenate([
        x.qpos[0:1],                              # rootz (height proxy)
        jnp.array([jnp.cos(x.qpos[2]), jnp.sin(x.qpos[2])]),  # rooty (pitch)
        x.qpos[3:9],                              # 6 leg joints
        x.qvel,                                   # 9 velocities (qvel[1] = forward)
    ])


# ── mjlab task registration ───────────────────────────────────────────────────

def register_walker(max_iters=1500):
    import mujoco as _mj
    from mjlab.actuator.xml_actuator import XmlActuatorCfg
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, reset_joints_by_offset, time_out
    from mjlab.envs.mdp.actions import JointEffortActionCfg
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
                   ".venv/lib/python3.12/site-packages/hydrax/models/walker/scene.xml")

    LEGS = ("right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle")
    _PITCH_CFG = SceneEntityCfg("walker", joint_names=("rooty",))
    _LEGS_CFG  = SceneEntityCfg("walker", joint_names=LEGS)
    _ROOTV_CFG = SceneEntityCfg("walker", joint_names=("rootz", "rootx", "rooty"))

    def _get_spec():
        return _mj.MjSpec.from_file(XML_PATH)

    articulation = EntityArticulationInfoCfg(
        actuators=(XmlActuatorCfg(target_names_expr=LEGS),),
    )
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.3),  # torso standing height (XML body pos); mjlab zeroes it otherwise
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    entity_cfg = EntityCfg(spec_fn=_get_spec, articulation=articulation, init_state=init_state)

    def _root_height(env):
        asset: Entity = env.scene["walker"]
        return asset.data.root_link_pose_w[:, 2]

    def _forward_vel(env):
        asset: Entity = env.scene["walker"]
        return asset.data.root_link_vel_w[:, 0]

    def torso_pitch_cos_sin(env, asset_cfg=_PITCH_CFG):
        import torch as _t
        asset: Entity = env.scene[asset_cfg.name]
        angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return _t.cat([_t.cos(angle), _t.sin(angle)], dim=-1)

    def walker_reward(env, pitch_cfg=_PITCH_CFG):
        # exp(-J) with J mirroring the hydrax Walker cost so the policy is a
        # proposal aligned with the inference objective. The torso z-axis
        # z-component (sensor torso_zaxis[2]) equals cos(pitch) for this planar
        # torso, so its upright deviation is cos(pitch) - 1.
        import torch as _t
        height = _root_height(env)
        vel    = _forward_vel(env)
        asset: Entity = env.scene["walker"]
        pitch  = asset.data.joint_pos[:, pitch_cfg.joint_ids].squeeze(-1)
        J = (10.0 * (height - STAND_HEIGHT) ** 2
             + 3.0 * (_t.cos(pitch) - 1.0) ** 2
             + 1.0 * (vel - WALK_SPEED) ** 2)
        return _t.exp(-J)

    def fell_over(env):
        return _root_height(env) < FALL_HEIGHT

    actor_terms = {
        "root_height": ObservationTermCfg(func=joint_pos_rel,    params={"asset_cfg": SceneEntityCfg("walker", joint_names=("rootz",))}),
        "torso_pitch": ObservationTermCfg(func=torso_pitch_cos_sin, params={"asset_cfg": _PITCH_CFG}),
        "leg_pos":     ObservationTermCfg(func=joint_pos_rel,    params={"asset_cfg": _LEGS_CFG}),
        "root_vel":    ObservationTermCfg(func=joint_vel_rel,    params={"asset_cfg": _ROOTV_CFG}),
        "leg_vel":     ObservationTermCfg(func=joint_vel_rel,    params={"asset_cfg": _LEGS_CFG}),
    }

    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"walker": entity_cfg},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations={"actor": ObservationGroupCfg(actor_terms), "critic": ObservationGroupCfg({**actor_terms})},
        actions={"effort": JointEffortActionCfg(entity_name="walker", actuator_names=LEGS, scale=1.0)},
        events={
            "reset_legs": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1),
                        "asset_cfg": _LEGS_CFG}),
            "reset_pitch": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-0.05, 0.05), "velocity_range": (-0.05, 0.05),
                        "asset_cfg": _PITCH_CFG}),
        },
        rewards={"reward": RewardTermCfg(func=walker_reward, weight=1.0,
                    params={"pitch_cfg": _PITCH_CFG})},
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "fell_over": TerminationTermCfg(func=fell_over, time_out=False),
        },
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),  # contacts ENABLED
        decimation=4,
        episode_length_s=10.0,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="walker", body_name="torso",
                            distance=5.0, elevation=-15.0, azimuth=90.0),
    )

    runner_cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 256), activation="elu",
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(256, 256), activation="elu"),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
            entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
            learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95,
            desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="walker",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iters,
    )

    register_mjlab_task(
        task_id="Mjlab-HydraxWalker-Walk",
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
    """blend: fraction of particles drawn from linear Gaussian (0=all learned)."""
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

def train(num_envs: int = 4096, iters: int = 1500):
    import os, sys
    register_walker(iters)
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["train", "Mjlab-HydraxWalker-Walk", f"--env.scene.num-envs={num_envs}"]
    from mjlab.scripts.train import main
    main()


# ── Interactive eval ──────────────────────────────────────────────────────────

def run_interactive(checkpoint_path: str, particles: int = 128, deterministic: bool = True):
    actor_params = load_actor_params(checkpoint_path)
    HORIZON, NUM_KNOTS = 0.6, 5
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
        noise_level=0.3,
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
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(args.num_envs, args.iters)
    elif args.checkpoint:
        run_interactive(args.checkpoint, args.particles, deterministic=not args.stochastic)
    else:
        parser.print_help()
