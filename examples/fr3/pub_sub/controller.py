# pub_sub/controller.py

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import zmq

from .protocol import Response, TeeBelief, Command, decode_json, encode_json


class ControllerEndpoint:
    """
    Controller-side ZMQ endpoint.

    Responsibilities:
      - Receive planner Command messages over REP.
      - Publish controller Response messages over PUB.

    Command socket:
      controller REP <- planner REQ

    Response socket:
      controller PUB -> planner SUB

    Response multipart layout:
      [topic][response_json][rgb_bytes][depth_bytes]
    """

    def __init__(
        self,
        *,
        command_bind_addr: str = "tcp://*:5555",
        response_bind_addr: str = "tcp://*:5556",
        response_topic: str | bytes = "controller.response",
        sndhwm: int = 1,
        conflate: bool = False,
        recv_timeout_ms: int | None = None,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.command_bind_addr = command_bind_addr
        self.response_bind_addr = response_bind_addr

        self.response_topic = (
            response_topic.encode("utf-8")
            if isinstance(response_topic, str)
            else response_topic
        )

        self.cmd_sock = self.ctx.socket(zmq.REP)
        if recv_timeout_ms is not None:
            self.cmd_sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.cmd_sock.bind(command_bind_addr)

        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.setsockopt(zmq.SNDHWM, sndhwm)

        if conflate:
            self.pub_sock.setsockopt(zmq.CONFLATE, 1)

        self.pub_sock.bind(response_bind_addr)

        self.seq = 0

    def close(self) -> None:
        self.cmd_sock.close(linger=0)
        self.pub_sock.close(linger=0)

    def recv_command(self) -> Command:
        """
        Receive a planner command.

        Assumes current Command is:
            Command(xy=list[float])
        """
        msg = decode_json(self.cmd_sock.recv())
        return Command(**msg)

    def reply_command(
        self,
        *,
        accepted: bool,
        reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        reply = {
            "accepted": accepted,
            "timestamp": time.time(),
        }

        if reason is not None:
            reply["reason"] = reason

        if extra is not None:
            reply.update(extra)

        self.cmd_sock.send(encode_json(reply))

    def publish_response(
        self,
        *,
        response: Response,
        rgb_bytes: bytes,
        depth_bytes: bytes,
        copy: bool = False,
    ) -> None:
        self.pub_sock.send_multipart(
            [
                self.response_topic,
                encode_json(response),
                rgb_bytes,
                depth_bytes,
            ],
            copy=copy,
        )

    def publish_rgbd_response(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_q: Sequence[Sequence[float]],
        joint_v: Sequence[Sequence[float]],
        beliefs: Sequence[TeeBelief],
        timestamp: float | None = None,
        copy: bool = False,
    ) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected rgb shape (H, W, 3), got {rgb.shape}")

        if depth.ndim != 2:
            raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")

        h, w, _ = rgb.shape

        if depth.shape != (h, w):
            raise ValueError(
                f"RGB/depth shape mismatch: rgb is {(h, w)}, depth is {depth.shape}"
            )

        rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth_c = np.ascontiguousarray(depth, dtype=np.float32)

        response = Response(
            timestamp=time.time() if timestamp is None else timestamp,
            seq=self.seq,
            rgbd_encoding="rgb8+d32f",
            rgbd_width=w,
            rgbd_height=h,
            joint_q=[list(row) for row in joint_q],
            joint_v=[list(row) for row in joint_v],
            beliefs=list(beliefs),
        )

        self.publish_response(
            response=response,
            rgb_bytes=rgb_c.tobytes(order="C"),
            depth_bytes=depth_c.tobytes(order="C"),
            copy=copy,
        )

        self.seq += 1