# msg_type.py

import json
from typing import NamedTuple, Any


class Command(NamedTuple):
    xy: list[float] = None
    # "" = move to xy (default); "descend" = go from home down onto the pushing
    # plane; "home" = return to the home configuration. Lets the planner drive the
    # arm's descend-on-start / home-on-finish lifecycle.
    action: str = ""

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