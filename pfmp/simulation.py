""" Visualization tools, aimed at parity with Hydrax """
import time
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco import mjx
# from pfmp.interpolate import hermite_spline
from pfmp.interpolate import Jet2, hermite_spline

import os
import subprocess
from datetime import datetime

class VideoRecorder:
    """Class for recording videos using FFmpeg."""

    def __init__(
        self,
        output_dir: str,
        width: int = 720,
        height: int = 480,
        fps: float = 30.0,
    ):
        """Initialize the video recorder.

        Args:
            output_dir: Directory to save the video.
            width: Width of the video in pixels.
            height: Height of the video in pixels.
            fps: Frames per second.
        """
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps

        self.ffmpeg_process = None
        self.video_path = None
        self.is_recording = False

    def start(self) -> bool:
        """Start recording the video.

        Returns:
            True if recording started successfully, False otherwise.
        """
        if self.is_recording:
            print("Warning: Recording already in progress")
            return True

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Generate output path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            self.output_dir, f"simulation_{timestamp}.mp4"
        )

        # Check if FFmpeg is available
        try:
            # Test FFmpeg availability
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Set up FFmpeg process
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-f",
                "rawvideo",  # Input format
                "-vcodec",
                "rawvideo",  # Input codec
                "-s",
                f"{self.width}x{self.height}",  # Frame dimensions
                "-pix_fmt",
                "rgb24",  # Pixel format
                "-r",
                str(self.fps),  # FPS
                "-i",
                "-",  # Input from pipe
                "-an",  # No audio
                "-vcodec",
                "h264",  # H.264 codec
                "-crf",
                "1",  # Quality (1=highest)
                "-preset",
                "slow",  # Encoding speed/compression tradeoff
                "-movflags",
                "+faststart",  # Optimize for web playback
                "-pix_fmt",
                "yuv420p",  # Standard compatible format
                "-profile:v",
                "high",  # H.264 profile
                "-tune",
                "film",  # Encoding optimization
                "-loglevel",
                "error",  # Suppress output except errors
                self.video_path,
            ]

            self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.is_recording = True
            print(f"Recording video to {self.video_path}")
            return True

        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

    def add_frame(self, frame: bytes) -> bool:
        """Add a frame to the video.

        Args:
            frame: Raw RGB frame data.

        Returns:
            True if the frame was added successfully, False otherwise.
        """
        if (
            not self.is_recording
            or self.ffmpeg_process is None
            or self.ffmpeg_process.stdin is None
        ):
            return False

        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            print("Warning: Failed to write frame to video")
            self.is_recording = False
            return False

    def stop(self) -> bool:
        """Stop recording and finalize the video.

        Returns:
            True if the video was finalized successfully, False otherwise.
        """
        if not self.is_recording or self.ffmpeg_process is None:
            return False

        try:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            print(f"Video saved to {self.video_path}")
            self.is_recording = False
            return True
        except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as e:
            print(f"Warning: Error finalizing video: {e}")
            # Try to terminate the process if it's still running
            try:
                self.ffmpeg_process.terminate()
            except:
                pass
            self.is_recording = False
            return False


def run_interactive(  # noqa: PLR0912, PLR0915
    controller,
    u0,     # initial control
    theta,  # (mean, mjx_model)
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    method = "sir",
    record_video: bool = False
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
    elif method == "sir":
        replan = controller.run_sir

    # Report the planning horizon in seconds for debugging
    ctrl_steps = round(controller.horizon / controller.dt)
    print(
        f"Planning with {ctrl_steps} steps "
        f"over a {controller.horizon} second horizon "
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

    key = jax.random.key(42)

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()

    def ps_update(theta, trajs, scores):
        # Different from Hydrax (and more like Dynamic Expectation Maximization),
        # the model params `theta` are updated separately from the SMC kernel
        best_idx = jnp.argmax(jnp.sum(scores, axis=-1))  # Predictive Sampling
        new_mean = jax.tree.map(lambda field: field[best_idx,:,-1], trajs)
        theta = theta._replace(mean = jnp.concat([theta.mean[0,None], new_mean[:,None]]))
        return theta

    init_shape = theta.mean.shape
    print(f"init shape: {init_shape}")
    key, subkey = jax.random.split(key)
    (trajs, scores) = replan(subkey, mjx_data, u0, theta)
    assert theta.mean.shape == init_shape
    theta = ps_update(theta, trajs, scores)
    print(theta.mean.shape)
    assert theta.mean.shape == init_shape
    key, subkey = jax.random.split(key)
    (trajs, scores) = replan(subkey, mjx_data, u0, theta)
    theta = ps_update(theta, trajs, scores)
    assert theta.mean.shape == init_shape
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

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # t_init = mj_data.time
        while viewer.is_running():
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
            # policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            (key, subkey) = jax.random.split(key)
            (trajs, scores) = replan(subkey, mjx_data, u0, theta)
            theta = ps_update(theta, trajs, scores)
            plan_time = time.time() - plan_start

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep

            # knots = jnp.concat([u0, theta.mean])
            knots = theta.mean
            tk = controller.tk()
            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt
            us = hermite_spline(tq, tk, knots)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Capture frame if recording
            if record_video and recorder.is_recording:
                renderer.update_scene(mj_data, viewer.cam)
                frame = renderer.render()
                recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

            # Update the mean based on elapsed offset
            new_tk = controller.tk() + sim_steps_per_replan * controller.dt
            if isinstance(knots, Jet2):
                nout = 2
            else:
                nout = 1
            new_mean = hermite_spline(new_tk, tk, knots, nout=nout)
            theta = theta._replace(mean=new_mean[:,None])
            u0 = theta.mean[0]
            assert theta.mean.shape == init_shape

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()