"""
Train the planar HalfCheetah (local Brax XML) via mjlab (MuJoCo-Warp + RSL-RL PPO).

The PPO reward is exp(-J) where J mirrors the inference-time cost
(track forward velocity 2.0 m/s, keep rootz and pitch near 0), so the learned
policy is a proposal aligned with the benchmark objective rather than a
max-speed runner.

Train:
    WANDB_MODE=disabled python examples/mjlab_half_cheetah.py --train --num-envs 4096

Play (after training):
    python examples/mjlab_half_cheetah.py --checkpoint logs/rsl_rl/half_cheetah/<run>/model_*.pt
"""

import argparse
import os
import torch
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from pathlib import Path
import importlib.util


def _load_hc_task():
    path = os.path.join(os.path.dirname(__file__), "half_cheetah.py")
    spec = importlib.util.spec_from_file_location("half_cheetah", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HalfCheetah()


import pfmp


# ── Task ──────────────────────────────────────────────────────────────────────

task      = _load_hc_task()
mj_model  = task.mj_model
mjx_model = mjx.put_model(mj_model)

# qpos/qvel order (XML): rootx, rootz, rooty, bthigh, bshin, bfoot,
#                        fthigh, fshin, ffoot  (9 each, all 1-DoF)
# obs (17, gym convention): qpos[1:] (exclude global x), qvel (incl forward vel)
OBS_DIM    = 17
ACTION_DIM = mj_model.nu  # 6

TARGET_VEL   = 2.0
TARGET_ROOTZ = 0.0

running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


def obs_fn(x):
    return jnp.concatenate([x.qpos[1:], x.qvel])  # 8 + 9 = 17


# ── mjlab task registration ───────────────────────────────────────────────────

def register_half_cheetah(max_iters=1500):
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

    XML_PATH = str(Path(__file__).parent.parent / "models" / "half_cheetah.xml")

    LEGS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
    POS8 = ("rootz", "rooty") + LEGS          # qpos[1:9]
    ALL9 = ("rootx", "rootz", "rooty") + LEGS  # qvel[0:9]
    _POS8_CFG = SceneEntityCfg("cheetah", joint_names=POS8)
    _ALL9_CFG = SceneEntityCfg("cheetah", joint_names=ALL9)
    _ROOTX_CFG = SceneEntityCfg("cheetah", joint_names=("rootx",))
    _ROOTZ_CFG = SceneEntityCfg("cheetah", joint_names=("rootz",))
    _ROOTY_CFG = SceneEntityCfg("cheetah", joint_names=("rooty",))

    def _get_spec():
        return _mj.MjSpec.from_file(XML_PATH)

    articulation = EntityArticulationInfoCfg(
        actuators=(XmlActuatorCfg(target_names_expr=LEGS),),
    )
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),  # torso XML height; mjlab zeroes it otherwise
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    entity_cfg = EntityCfg(spec_fn=_get_spec, articulation=articulation, init_state=init_state)

    def cheetah_reward(env, rootx_cfg=_ROOTX_CFG, rootz_cfg=_ROOTZ_CFG, rooty_cfg=_ROOTY_CFG):
        import torch as _t
        asset: Entity = env.scene["cheetah"]
        v     = asset.data.joint_vel[:, rootx_cfg.joint_ids].squeeze(-1)
        rootz = asset.data.joint_pos[:, rootz_cfg.joint_ids].squeeze(-1)
        pitch = asset.data.joint_pos[:, rooty_cfg.joint_ids].squeeze(-1)
        J = (v - TARGET_VEL) ** 2 + 5.0 * (rootz - TARGET_ROOTZ) ** 2 + 0.5 * pitch ** 2
        return _t.exp(-J)

    actor_terms = {
        "pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _POS8_CFG}),
        "vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _ALL9_CFG}),
    }

    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"cheetah": entity_cfg},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations={"actor": ObservationGroupCfg(actor_terms), "critic": ObservationGroupCfg({**actor_terms})},
        actions={"effort": JointEffortActionCfg(entity_name="cheetah", actuator_names=LEGS, scale=1.0)},
        events={
            "reset_joints": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1),
                        "asset_cfg": _ALL9_CFG}),
        },
        rewards={"reward": RewardTermCfg(func=cheetah_reward, weight=1.0,
                    params={"rootx_cfg": _ROOTX_CFG, "rootz_cfg": _ROOTZ_CFG, "rooty_cfg": _ROOTY_CFG})},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),  # contacts ENABLED
        decimation=5,
        episode_length_s=10.0,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="cheetah", body_name="torso",
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
        experiment_name="half_cheetah",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iters,
    )

    register_mjlab_task(
        task_id="Mjlab-HalfCheetah-Run",
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
    register_half_cheetah(iters)
    os.environ["WANDB_MODE"] = "disabled"
    sys.argv = ["train", "Mjlab-HalfCheetah-Run", f"--env.scene.num-envs={num_envs}"]
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
    mj_data = mujoco.MjData(mj_model)
    if hasattr(task, "initial_qpos"):
        mj_data.qpos[:] = task.initial_qpos
        mujoco.mj_forward(mj_model, mj_data)
    theta = pfmp.GenericMJXTaskParams(
        mjx_model=mjx_model,
        mean=jnp.zeros((NUM_KNOTS, ACTION_DIM)),
        noise_level=0.3,
        actor_params=actor_params,
    )
    pfmp.run_interactive(
        ctrl, theta, mj_model, mj_data,
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
