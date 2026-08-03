# pub_sub/planner.py

from __future__ import annotations

from typing import Any

import zmq

from .protocol import Command, decode_json, encode_json


DEFAULT_COMMAND_ADDR = "tcp://localhost:5555"


class PlannerClient:
    """
    Planner-side ZMQ client: sends Command messages to the controller and
    receives its JSON reply.

    Expected command socket:
      REQ -> controller REP
    """

    def __init__(
        self,
        *,
        command_addr: str = DEFAULT_COMMAND_ADDR,
        command_timeout_ms: int = 1000,
    ) -> None:
        self.ctx = zmq.Context.instance()

        self.command_addr = command_addr

        self.cmd_sock = self.ctx.socket(zmq.REQ)
        self.cmd_sock.setsockopt(zmq.RCVTIMEO, command_timeout_ms)
        self.cmd_sock.setsockopt(zmq.SNDTIMEO, command_timeout_ms)
        self.cmd_sock.connect(command_addr)

    def close(self) -> None:
        self.cmd_sock.close(linger=0)

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

    def lifecycle(self, action: str) -> None:
        """Send a lifecycle action (e.g. "descend" / "home") to the controller.

        Never raises: a failed lifecycle signal is logged, not fatal, so a
        caller mid-trial doesn't abort over it.
        """
        try:
            self.send_command(Command(action=action))
        except Exception as e:
            print(f"[lifecycle] {action} signal failed: {e!r}")
