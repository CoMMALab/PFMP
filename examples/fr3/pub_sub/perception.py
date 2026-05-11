from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import zmq

from .planner import decode_rgbd
from .protocol import Response, TeeBelief, decode_json, encode_json


DEFAULT_PERCEPTION_ADDR = "tcp://localhost:5557"
DEFAULT_PERCEPTION_BIND_ADDR = "tcp://*:5557"
DEFAULT_PERCEPTION_TOPIC = b"perception.state"


class PerceptionPublisher:
    """
    Publishes perception RGBD frames and T-block beliefs.

    Multipart layout:
        [topic][response_json][rgb_bytes][depth_bytes]

    The JSON schema intentionally reuses protocol.Response so existing RGBD
    decoding utilities and belief consumers can be shared.
    """

    def __init__(
        self,
        *,
        bind_addr: str = DEFAULT_PERCEPTION_BIND_ADDR,
        topic: bytes | str = DEFAULT_PERCEPTION_TOPIC,
        sndhwm: int = 1,
        conflate: bool = False,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.bind_addr = bind_addr

        if isinstance(topic, str):
            topic = topic.encode("utf-8")
        self.topic = topic

        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.setsockopt(zmq.SNDHWM, sndhwm)
        if conflate:
            self.pub_sock.setsockopt(zmq.CONFLATE, 1)
        self.pub_sock.bind(bind_addr)
        self.seq = 0

    def close(self) -> None:
        self.pub_sock.close(linger=0)

    def publish(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
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
            joint_q=[],
            joint_v=[],
            beliefs=list(beliefs),
        )

        self.pub_sock.send_multipart(
            [
                self.topic,
                encode_json(response),
                rgb_c.tobytes(order="C"),
                depth_c.tobytes(order="C"),
            ],
            copy=copy,
        )
        self.seq += 1


class PerceptionClient:
    """Subscribes to perception RGBD frames and T-block beliefs."""

    def __init__(
        self,
        *,
        addr: str = DEFAULT_PERCEPTION_ADDR,
        topic: bytes | str = DEFAULT_PERCEPTION_TOPIC,
        timeout_ms: int = 1000,
        rcvhwm: int = 1,
        conflate: bool = True,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.addr = addr

        if isinstance(topic, str):
            topic = topic.encode("utf-8")
        self.topic = topic

        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sub_sock.setsockopt(zmq.RCVHWM, rcvhwm)
        if conflate:
            self.sub_sock.setsockopt(zmq.CONFLATE, 1)
        self.sub_sock.connect(addr)
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, self.topic)

    def close(self) -> None:
        self.sub_sock.close(linger=0)

    def recv(self) -> tuple[dict, np.ndarray, np.ndarray]:
        parts = self.sub_sock.recv_multipart()
        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 ZMQ frames [topic][json][rgb][depth], got {len(parts)}"
            )

        topic, response_json, rgb_bytes, depth_bytes = parts
        if not topic.startswith(self.topic):
            raise ValueError(f"Unexpected topic {topic!r}; expected prefix {self.topic!r}")

        response = decode_json(response_json)
        rgb, depth = decode_rgbd(response, rgb_bytes, depth_bytes)
        return response, rgb, depth

    def recv_latest(self, *, drain: bool = True) -> tuple[dict, np.ndarray, np.ndarray]:
        response, rgb, depth = self.recv()
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
            if not topic.startswith(self.topic):
                continue

            response = decode_json(response_json)
            rgb, depth = decode_rgbd(response, rgb_bytes, depth_bytes)

        return response, rgb, depth
