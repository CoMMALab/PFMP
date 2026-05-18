# pub_sub/planner.py

from __future__ import annotations

from typing import Any
import time

import numpy as np
import zmq

from .protocol import Command, decode_json, encode_json


DEFAULT_COMMAND_ADDR = "tcp://localhost:5555"
DEFAULT_RESPONSE_ADDR = "tcp://localhost:5556"
DEFAULT_RESPONSE_TOPIC = b"controller.response"


class PlannerClient:
    """
    Planner-side ZMQ client.

    Responsibilities:
      - Send Command messages to the controller.
      - Receive controller Response metadata.
      - Reconstruct RGB/depth arrays from multipart payloads.

    Expected command socket:
      REQ -> controller REP

    Expected response socket:
      SUB <- controller PUB

    Response multipart layout:
      [topic][response_json][rgb_bytes][depth_bytes]
    """

    def __init__(
        self,
        *,
        command_addr: str = DEFAULT_COMMAND_ADDR,
        response_addr: str = DEFAULT_RESPONSE_ADDR,
        response_topic: bytes | str = DEFAULT_RESPONSE_TOPIC,
        command_timeout_ms: int = 1000,
        response_timeout_ms: int = 1000,
        rcvhwm: int = 1,
        conflate: bool = True,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.command_addr = command_addr
        self.response_addr = response_addr

        if isinstance(response_topic, str):
            response_topic = response_topic.encode("utf-8")
        self.response_topic = response_topic

        # Planner -> Controller command socket.
        self.cmd_sock = self.ctx.socket(zmq.REQ)
        self.cmd_sock.setsockopt(zmq.RCVTIMEO, command_timeout_ms)
        self.cmd_sock.setsockopt(zmq.SNDTIMEO, command_timeout_ms)
        self.cmd_sock.connect(command_addr)

        # Controller -> Planner response socket.
        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.setsockopt(zmq.RCVTIMEO, response_timeout_ms)
        self.sub_sock.setsockopt(zmq.RCVHWM, rcvhwm)

        if conflate:
            # Keep only the newest response if the planner falls behind.
            self.sub_sock.setsockopt(zmq.CONFLATE, 1)

        self.sub_sock.connect(response_addr)
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, self.response_topic)

    def close(self) -> None:
        self.cmd_sock.close(linger=0)
        self.sub_sock.close(linger=0)

    def send_command(self, command: Command) -> dict[str, Any]:
        """
        Send a planner Command to the controller.

        Controller is expected to reply with JSON, for example:
            {"accepted": true}
        or:
            {"accepted": false, "reason": "..."}
        """
        self.cmd_sock.send(encode_json(command))
        reply_bytes = self.cmd_sock.recv()
        return decode_json(reply_bytes)

    def send_xy(self, xy: list[float]) -> dict[str, Any]:
        """
        Convenience helper for your current Command type.
        """
        return self.send_command(Command(xy=xy))

    def recv_response(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """
        Receive one controller response.

        Returns:
            response_meta, rgb, depth

        response_meta is currently a dict because msg_type.decode_json returns dict.
        """
        parts = self.sub_sock.recv_multipart()

        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 ZMQ frames [topic][json][rgb][depth], got {len(parts)}"
            )

        topic, response_json, rgb_bytes, depth_bytes = parts

        if not topic.startswith(self.response_topic):
            raise ValueError(
                f"Unexpected topic {topic!r}; expected prefix {self.response_topic!r}"
            )

        response = decode_json(response_json)
        rgb, depth = decode_rgbd(response, rgb_bytes, depth_bytes)

        return response, rgb, depth

    def recv_response_latest(
        self,
        *,
        drain: bool = True,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """
        Receive the latest available response.

        If drain=True, consume all currently queued response messages and return
        the newest one. This is useful for planners that only care about current
        robot state.
        """
        response, rgb, depth = self.recv_response()

        if not drain:
            return response, rgb, depth

        while True:
            try:
                parts = self.sub_sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            if len(parts) != 4:
                continue

            topic, response_json, rgb_bytes, depth_bytes = parts

            if not topic.startswith(self.response_topic):
                continue

            response = decode_json(response_json)
            rgb, depth = decode_rgbd(response, rgb_bytes, depth_bytes)

        return response, rgb, depth


def decode_rgbd(
    response: dict[str, Any],
    rgb_bytes: bytes,
    depth_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct RGB and depth arrays from multipart ZMQ payload.

    Supported encoding:
        rgbd_encoding == "rgb8+d32f"

    Uses:
        rgbd_width
        rgbd_height

    Produces:
        rgb:   uint8 array, shape (H, W, 3)
        depth: float32 array, shape (H, W)
    """
    encoding = response["rgbd_encoding"]
    width = int(response["rgbd_width"])
    height = int(response["rgbd_height"])

    if encoding != "rgb8+d32f":
        raise ValueError(f"Unsupported rgbd_encoding: {encoding!r}")

    expected_rgb_nbytes = height * width * 3 * np.dtype(np.uint8).itemsize
    expected_depth_nbytes = height * width * np.dtype(np.float32).itemsize

    if len(rgb_bytes) != expected_rgb_nbytes:
        raise ValueError(
            f"RGB byte size mismatch: got {len(rgb_bytes)}, "
            f"expected {expected_rgb_nbytes}"
        )

    if len(depth_bytes) != expected_depth_nbytes:
        raise ValueError(
            f"Depth byte size mismatch: got {len(depth_bytes)}, "
            f"expected {expected_depth_nbytes}"
        )

    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(height, width)

    return rgb, depth