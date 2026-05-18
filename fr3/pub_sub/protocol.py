# msg_type.py

import json
from typing import NamedTuple, Literal, Any


class Command(NamedTuple):
    xy: list[float]

class TeeBelief(NamedTuple):
    mean: list[float]        # (3,)
    cov: list[list[float]]   # (3, 3)

class Response(NamedTuple):
    timestamp: float
    seq: int
    rgbd_encoding: Literal["jpeg+z16", "rgb8+d32f", "rgbd_npz"]
    rgbd_width: int
    rgbd_height: int

    # Prefer (T, 7), i.e. one row per timestep.
    joint_q: list[list[float]]  # (T, 7)
    joint_v: list[list[float]]  # (T, 7)

    beliefs: list[TeeBelief]    # (T,)

def to_jsonable(x: Any) -> Any:
    if isinstance(x, tuple) and hasattr(x, "_asdict"):
        out = {}
        for key, value in x._asdict().items():
            # Python cannot use `class` as a field name, but JSON can.
            json_key = "class" if key == "class_name" else key
            out[json_key] = to_jsonable(value)
        return out

    if isinstance(x, list):
        return [to_jsonable(v) for v in x]

    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    return x

def encode_json(x: Any) -> bytes:
    return json.dumps(
        to_jsonable(x),
        separators=(",", ":"),
    ).encode("utf-8")


def decode_json(b: bytes) -> dict[str, Any]:
    return json.loads(b.decode("utf-8"))