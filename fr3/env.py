"""Shared Motive/robot network defaults for the fr3/ scripts.

Loads fr3/env.yaml if it exists and falls back to loopback otherwise,
so scripts still work with nothing configured.

env.yaml:
    mocap_ip: <Motive/OptiTrack server IP>
    local_ip: <this machine's IP>
    controller_ip: <robot controller host>
"""

import os

import yaml

_ENV_PATH = os.path.join(os.path.dirname(__file__), "env.yaml")

_DEFAULTS = {"mocap_ip": "127.0.0.1", "local_ip": "127.0.0.1", "controller_ip": "localhost"}


def load_env() -> dict:
    """--mocap-ip/--local-ip/--controller-ip defaults: env.yaml's values where
    set, the loopback fallback above for anything it omits or if it's absent."""
    env = dict(_DEFAULTS)
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        env.update({k: v for k, v in loaded.items() if k in _DEFAULTS})
    return env
