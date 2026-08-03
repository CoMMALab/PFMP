# pub_sub/controller.py

from __future__ import annotations

import time
from typing import Any

import zmq

from .protocol import Command, decode_json, encode_json


class ControllerEndpoint:
    """
    Controller-side ZMQ endpoint: receives planner Command messages and
    replies with JSON.

    Command socket:
      controller REP <- planner REQ
    """

    def __init__(
        self,
        *,
        command_bind_addr: str = "tcp://*:5555",
        recv_timeout_ms: int | None = None,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.command_bind_addr = command_bind_addr

        self.cmd_sock = self.ctx.socket(zmq.REP)
        if recv_timeout_ms is not None:
            self.cmd_sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.cmd_sock.bind(command_bind_addr)

    def close(self) -> None:
        self.cmd_sock.close(linger=0)

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
