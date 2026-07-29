"""
push_tee_controller.py — Minimal real-robot execution loop.

Connects to the Franka via franky and accepts simple XY waypoint commands.

Usage:
    python examples/fr3/push_tee_controller.py --ip 172.16.0.3
    python examples/fr3/push_tee_controller.py --dry-run
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import select
import sys
import time
from typing import Any, Optional

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(__file__))
from ik_cache import ik_solve, get_or_build_cache, HOME_Q


def _current_q(robot):
    """Robot's current joint config as an IK seed (falls back to HOME_Q)."""
    if robot is None:
        return np.asarray(HOME_Q, float)
    try:
        return np.asarray(robot.current_joint_state.position, float)
    except Exception:
        return np.asarray(HOME_Q, float)
from pub_sub.controller import ControllerEndpoint
from pub_sub.protocol import Command, decode_json, encode_json

import franky


ROBOT_IP = "172.16.0.3"
VELOCITY_SCALE = 0.2
CONTROL_ATTEMPTS = 50


def robot_current_xy(robot) -> np.ndarray:
    pose = robot.current_cartesian_state.pose.end_effector_pose
    return np.array([
        float(pose.translation[0]),
        float(pose.translation[1]),
    ])

# recover from errors
# time.sleep()

def go_pushing_plane(robot: franky.Robot, target_xy: np.ndarray, ik_cache=None) -> None:
    target_xy = np.asarray(target_xy, dtype=float)
    target_q = ik_solve(target_xy, _current_q(robot))
    if target_q is None:
        raise ValueError(f"target {target_xy.tolist()} is unreachable")
    robot.move(franky.JointMotion(target_q.tolist()))

def go_home(robot: franky.Robot) -> None:
    robot.move(franky.JointMotion(HOME_Q.tolist()))

def execute_xy(robot: franky.Robot, target_xy: np.ndarray, ik_cache=None) -> None:
    target_xy = np.asarray(target_xy, dtype=float)
    if ik_cache is not None:
        target_q = ik_solve(target_xy, _current_q(robot))
        if target_q is None:
            raise ValueError(f"target {target_xy.tolist()} is unreachable")
        robot.move(franky.JointMotion(np.asarray(target_q, float).tolist()))
    else:
        current_xy = robot_current_xy(robot)
        delta_xy = target_xy - current_xy
        robot.move(franky.CartesianMotion(
            franky.Affine([float(delta_xy[0]), float(-delta_xy[1]), 0.0]),
            franky.ReferenceType.Relative,
        ))


def main(args):
    # ── World frame & IK cache ────────────────────────────────────────────────
    cache_path = args.cache or os.path.join(
        os.path.dirname(__file__),
        "..",
        os.path.join(os.path.dirname(__file__), "ik_cache.pkl"),
    )
    ik_cache = get_or_build_cache(path=cache_path)

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot = None
    current_xy = None  # None means: not on the pushing plane yet

    if not args.dry_run:
        print(f"Connecting to {args.ip}…")
        robot = franky.Robot(args.ip)
        robot.relative_dynamics_factor = args.speed
        go_home(robot)
        print("At home. Not on pushing plane.")

    rng = np.random.default_rng()

    # ── ZMQ endpoint (server mode only) ───────────────────────────────────────
    endpoint: Optional[ControllerEndpoint] = None
    if args.mode == 'server':
        cmd_addr = f"tcp://*:{args.cmd_port}"
        endpoint = ControllerEndpoint(
            command_bind_addr=cmd_addr,
            recv_timeout_ms=0,  # non-blocking REP
        )

    # Stay at home; the planner triggers the descend onto the pushing plane when
    # planning starts (Command action="descend") and homes us when it finishes.
    current_xy = None  # not on the plane yet
    if robot is None:
        print(f"[dry-run] at home; will descend to init_xy {args.init_xy} on planner signal")
    else:
        print(f"At home. Awaiting descend signal (init_xy {args.init_xy}).")

    if args.mode == 'server':
        print(f"\nLoop running in SERVER mode.")
        print(f"  commands  REP ← {cmd_addr}")
    elif args.mode == 'manual':
        print("\nLoop running in MANUAL mode.")
        print("Commands: xy <x> <y>  |  home  |  q")
    elif args.mode == 'random':  # random
        print("\nLoop running in AUTO random-target mode.")
        print("Commands: q")
    else:
        raise Exception(f"Unknown control mode: {args.mode}") 


    # main loop
    try:
        seq = 0
        failures = 0
        while True:
            if args.mode == 'server':
                if endpoint is not None:
                    try:
                        cmd = endpoint.recv_command()
                    except zmq.Again:
                        cmd = None

                    if cmd is not None:
                        print(f"[server] received command: {cmd}")
                        action = getattr(cmd, "action", "") or ""
                        if action in ("home", "descend"):
                            # Lifecycle commands: home on finish, descend on start.
                            try:
                                if robot is None:
                                    print(f"[dry-run] would {action}")
                                elif action == "home":
                                    go_home(robot); current_xy = None
                                else:  # descend
                                    go_pushing_plane(robot, args.init_xy, ik_cache)
                                    current_xy = np.asarray(args.init_xy, dtype=float)
                                endpoint.reply_command(accepted=True)
                            except Exception as e:
                                print(f"[server] {action} failed: {e!r}")
                                if robot is not None:
                                    try:
                                        robot.recover_from_errors()
                                    except Exception as rec_e:
                                        print(f"[server] recover_from_errors failed: {rec_e!r}")
                                endpoint.reply_command(accepted=False, reason=f"{action} failed: {e}")
                            continue
                        try:
                            target_xy = np.asarray(cmd.xy, dtype=float)
                        except Exception as e:
                            endpoint.reply_command(accepted=False, reason=f"bad command: {e}")
                        else:
                            # Don't let a single failed motion kill the server.
                            # Log, try to recover, and keep accepting commands.
                            actual_xy = None
                            try:
                                if robot is None:
                                    print(f"[dry-run] would execute xy: {target_xy.tolist()}")
                                else:
                                    execute_xy(robot, target_xy, ik_cache)
                                    current_xy = robot_current_xy(robot)
                                    actual_xy = current_xy.tolist()
                                    print(f"[server] executed xy command: {target_xy.tolist()} actual: {actual_xy}")
                            except Exception as e:
                                print(f"[server] execute_xy failed: {e!r}")
                                if robot is not None:
                                    try:
                                        robot.recover_from_errors()
                                    except Exception as rec_e:
                                        print(f"[server] recover_from_errors failed: {rec_e!r}")
                            endpoint.reply_command(accepted=True,
                                extra={"actual_xy": actual_xy} if actual_xy else None)

            elif args.mode == 'manual':
                # Manual mode: consume stdin only when user typed a command.
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    line = sys.stdin.readline().strip()
                    parts = line.split()

                    if not parts:
                        continue

                    if parts[0] == "q":
                        break

                    if parts[0] == "home":
                        if robot is None:
                            print("[dry-run] would go home")
                        else:
                            go_home(robot)
                            current_xy = None
                        continue

                    if parts[0] == "xy" and len(parts) == 3:
                        if robot is not None:
                            current_xy = robot_current_xy(robot)
                        elif current_xy is None:
                            current_xy = np.asarray(args.init_xy, dtype=float)
                        target_xy = np.array([float(parts[1]), float(parts[2])])
                        if robot is None:
                            print(f"[dry-run] would execute xy: {target_xy.tolist()}")
                        else:
                            execute_xy(robot, target_xy, ik_cache)
                        current_xy = target_xy
                        continue

                    print(f"[WARN] unknown command: {line}")

            else:
                if robot is not None:
                    current_xy = robot_current_xy(robot)
                elif current_xy is None:
                    current_xy = np.asarray(args.init_xy, dtype=float)
                target_xy = rng.normal(args.init_xy, 0.05, size=2)
                for attempt in range(CONTROL_ATTEMPTS):
                    try:
                        seq += 1
                        execute_xy(robot, target_xy, ik_cache)
                        break
                    except Exception as e:
                        failures += 1
                        print(f"FAILED (success rate = {1 - failures/seq:.3f}): {e}")
                        robot.recover_from_errors()
                current_xy = robot_current_xy(robot)

            time.sleep(0.001)

    finally:
        if endpoint is not None:
            try:
                endpoint.close()
            except Exception as e:
                print(f"[WARN] could not close ZMQ endpoint: {e}")

        if robot is not None:
            robot.move(franky.JointMotion(HOME_Q.tolist()))
            current_xy = None
            print("Returned to home. Exited pushing plane.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--speed", type=float, default=VELOCITY_SCALE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--cmd-port", type=int, default=5555,
                        help="ZMQ REP port for commands (server mode)")

    # Default behavior: autonomous random target sampling.
    # Manual behavior: wait for stdin commands: xy <x> <y>, home, q.
    parser.add_argument("--mode", type=str, default='manual', help="options: manual, random, server")

    parser.add_argument("--init-xy", type=float, nargs=2, default=[0.4, 0.0])

    args = parser.parse_args()

    main(args)
