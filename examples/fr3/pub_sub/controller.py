# pub_sub/controller.py
#
# Synchronous REP endpoint. The controller blocks on recv_command(), runs the
# motion / query, then calls reply(...). One round-trip per command.

from __future__ import annotations

import zmq

from .protocol import Command, Reply, decode_json, encode_json, parse_command


class ControllerEndpoint:
    """Controller-side ZMQ REP endpoint."""

    def __init__(
        self,
        *,
        command_bind_addr: str = "tcp://*:5555",
        recv_timeout_ms: int | None = None,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.command_bind_addr = command_bind_addr

        self.sock = self.ctx.socket(zmq.REP)
        if recv_timeout_ms is not None:
            self.sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.sock.bind(command_bind_addr)

    def close(self) -> None:
        self.sock.close(linger=0)

    def recv_command(self) -> Command:
        msg = decode_json(self.sock.recv())
        return parse_command(msg)

    def reply(self, reply: Reply) -> None:
        self.sock.send(encode_json(reply))
