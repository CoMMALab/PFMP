"""Smallest reproducer: HOME → JointMotion(target) with no perception/ZMQ."""

import os
import sys
import time

import numpy as np

# JAX is imported transitively by ik_cache; pin it to CPU so this script
# doesn't fight with anything for the GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples", "fr3"))
from ik_cache import HOME_Q, EE_HEIGHT, ik_from_cache, get_or_build_cache

import franky


ROBOT_IP = "172.16.0.3"
TARGET_XY = np.array([0.5, -0.15])  # one of the sweep waypoints

print("Building / loading IK cache…")
ik_cache = get_or_build_cache()
target_q = np.asarray(ik_from_cache(jnp.asarray(TARGET_XY), ik_cache))

home_q = np.asarray(HOME_Q)
print(f"HOME_Q   = {home_q}")
print(f"target_q = {target_q}  (xy={TARGET_XY}, z={EE_HEIGHT})")

print(f"Connecting to {ROBOT_IP}…")
r = franky.Robot(ROBOT_IP)
r.relative_dynamics_factor = 0.05

print("Recovering from errors and moving HOME…")
r.recover_from_errors()
r.move(franky.JointMotion(home_q.tolist()))

time.sleep(0.5)

print("Moving to target…")
r.move(franky.JointMotion(target_q.tolist()))

time.sleep(0.5)

print("Returning HOME…")
r.move(franky.JointMotion(home_q.tolist()))

print("OK")
