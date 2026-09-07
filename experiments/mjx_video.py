#!/usr/bin/env python3
"""Render a MuJoCo control task to an mp4, headlessly.

`run_interactive` needs a viewer window; this uses the batched driver instead
and replays the resulting qpos through an offscreen renderer, so it runs over
ssh and cannot steal focus.

    python experiments/mjx_video.py experiments/configs/CartPole.yaml sis out/cartpole.mp4
"""
import os, sys, argparse, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, yaml, mujoco, jax.numpy as jnp
from mujoco import mjx
import pfmp
from mjx_runner import Experiment, _load_task, main as _unused  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config"); ap.add_argument("algorithm"); ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    cfg_d = yaml.safe_load(open(a.config)); cfg_d.pop("algorithms", None)
    cfg_d.pop("learned_checkpoint", None)
    cfg_d["max_step"] = int(a.seconds * cfg_d.get("frequency", 50))
    cfg_d["num"] = 1
    cfg = Experiment(**cfg_d, algorithm=a.algorithm)

    task = _load_task(cfg.task)
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    if hasattr(task, "initial_qpos"):
        mj_data.qpos[:] = task.initial_qpos; mujoco.mj_forward(mj_model, mj_data)

    running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
    init, proposal = pfmp.mjx_linear_gaussian_proposal(task, cfg.plan_horizon, cfg.num_knots)
    ctrl = pfmp.PFMP(pfmp.mjx_physics, running_cost, terminal_cost, init, proposal,
                     horizon=cfg.plan_horizon, num_knots=cfg.num_knots,
                     particles=cfg.num_samples, dt=mj_model.opt.timestep,
                     ess_resample_thresh=0.5)
    theta = pfmp.GenericMJXTaskParams(mjx_model=mjx.put_model(mj_model),
                                      mean=jnp.zeros((cfg.num_knots, mj_model.nu)),
                                      noise_level=cfg.noise_level)

    print(f"[{a.algorithm}] {cfg.task}: {cfg.max_step} steps…", flush=True)
    traj = pfmp.run_batch(ctrl, theta, mj_model, mj_data, frequency=cfg.frequency,
                          algorithm=a.algorithm, max_step=cfg.max_step, num=1,
                          temperature=cfg.mppi_temperature)
    qpos = np.asarray(traj.qpos)[0]                     # (T, nq)
    stride = max(1, int(round(1.0 / (mj_model.opt.timestep * a.fps))))
    print(f"  {qpos.shape[0]} states -> every {stride}th at {a.fps} fps")

    renderer = mujoco.Renderer(mj_model, height=a.height, width=a.width)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE)
    d = mujoco.MjData(mj_model)
    for q in qpos[::stride]:
        d.qpos[:] = q; mujoco.mj_forward(mj_model, d)
        renderer.update_scene(d, camera=0 if mj_model.ncam else -1)
        p.stdin.write(renderer.render().tobytes())
    p.stdin.close(); p.wait()
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
