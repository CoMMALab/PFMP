# pub_sub/planner.py
#
# Synchronous REQ client. Each send() blocks until the controller replies.
# A "move" command's reply only arrives after the motion completes, so the
# default RCV/SND timeout is generous (60 s).

from __future__ import annotations

import zmq

from .protocol import Command, Reply, decode_json, encode_json, parse_reply


DEFAULT_COMMAND_ADDR = "tcp://localhost:5555"


class PlannerClient:
    """Planner-side ZMQ REQ client."""

    def __init__(
        self,
        *,
        command_addr: str = DEFAULT_COMMAND_ADDR,
        timeout_ms: int = 60_000,
    ) -> None:
        self.ctx = zmq.Context.instance()
        self.command_addr = command_addr
        self._timeout_ms = timeout_ms
        self.sock = self._make_sock()

    def _make_sock(self) -> zmq.Socket:
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        s.connect(self.command_addr)
        return s

    def reconnect(self) -> None:
        """Recreate the REQ socket. Required after a recv timeout (zmq.Again),
        which leaves the REQ state machine stuck — it expects a recv before
        another send is allowed."""
        self.sock.close(linger=0)
        self.sock = self._make_sock()

    def close(self) -> None:
        self.sock.close(linger=0)

    def send(self, cmd: Command) -> Reply:
        """Send a command and block until the controller's reply arrives."""
        self.sock.send(encode_json(cmd))
        msg = decode_json(self.sock.recv())
        return parse_reply(msg)

    def query(self) -> Reply:
        """Get the controller's current belief + joint state without commanding motion."""
        return self.send(Command(kind="query"))

    def move(self, xy) -> Reply:
        """Send an xy waypoint and block until the motion completes (or fails)."""
        return self.send(Command(kind="move", xy=list(xy)))
