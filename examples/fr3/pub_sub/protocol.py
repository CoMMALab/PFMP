# pub_sub/protocol.py
#
# Synchronous request/reply protocol between planner (REQ) and controller (REP):
#
#   planner --Command(kind="query")------------> controller
#   planner <-----------Reply(belief, joint_q)-- controller   (immediately)
#
#   planner --Command(kind="move", xy=...)-----> controller
#                                                controller does execute_xy(xy)
#                                                controller refreshes perception
#   planner <-----------Reply(belief, joint_q)-- controller   (after motion completes)
#
# The planner only sends the next Command after receiving the previous Reply.

import json
from typing import Any, NamedTuple, Optional


class TeeBelief(NamedTuple):
    mean: list[float]              # (3,) [x, y, yaw]
    cov: list[list[float]]         # (3, 3)


class Command(NamedTuple):
    kind: str                      # "move" or "query"
    xy: Optional[list[float]] = None  # required for kind="move"


class Reply(NamedTuple):
    success: bool
    error: Optional[str]
    timestamp: float
    belief: TeeBelief
    joint_q: list[float]           # (7,)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, tuple) and hasattr(x, "_asdict"):
        return {k: to_jsonable(v) for k, v in x._asdict().items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    return x


def encode_json(x: Any) -> bytes:
    return json.dumps(to_jsonable(x), separators=(",", ":")).encode("utf-8")


def decode_json(b: bytes) -> dict[str, Any]:
    return json.loads(b.decode("utf-8"))


def parse_command(msg: dict[str, Any]) -> Command:
    return Command(
        kind=str(msg.get("kind", "move")),
        xy=msg.get("xy"),
    )


def parse_reply(msg: dict[str, Any]) -> Reply:
    b = msg.get("belief") or {}
    return Reply(
        success=bool(msg.get("success", False)),
        error=msg.get("error"),
        timestamp=float(msg.get("timestamp", 0.0)),
        belief=TeeBelief(
            mean=list(b.get("mean", [0.0, 0.0, 0.0])),
            cov=list(b.get("cov", [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])),
        ),
        joint_q=list(msg.get("joint_q", [0.0] * 7)),
    )
