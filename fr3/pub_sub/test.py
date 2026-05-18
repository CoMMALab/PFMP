# pub_sub/test.py

import time
import threading
import traceback

import numpy as np

from protocol import Command, TeeBelief
from planner import PlannerClient
from controller import ControllerEndpoint


def run_thread(name, fn):
    def wrapped():
        try:
            fn()
        except Exception:
            print(f"\n{name} crashed:")
            traceback.print_exc()

    thread = threading.Thread(target=wrapped, name=name, daemon=True)
    thread.start()
    return thread


def controller_command_loop(controller: ControllerEndpoint):
    while True:
        cmd = controller.recv_command()
        print("controller received command:", cmd)

        controller.reply_command(
            accepted=True,
            extra={"xy": cmd.xy},
        )


def controller_response_loop(controller: ControllerEndpoint):
    time.sleep(1.0)

    while True:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)

        joint_q = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
        joint_v = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        beliefs = [
            TeeBelief(
                mean=[0.1, 0.2, 0.3],
                cov=[
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        ]

        controller.publish_rgbd_response(
            rgb=rgb,
            depth=depth,
            joint_q=joint_q,
            joint_v=joint_v,
            beliefs=beliefs,
        )

        time.sleep(0.1)


def main():
    controller = ControllerEndpoint(
        command_bind_addr="tcp://*:5555",
        response_bind_addr="tcp://*:5556",
        response_topic="controller.response",
        conflate=False,
    )

    run_thread("controller_command_loop", lambda: controller_command_loop(controller))

    client = PlannerClient(
        command_addr="tcp://localhost:5555",
        response_addr="tcp://localhost:5556",
        response_topic="controller.response",
        command_timeout_ms=5000,
        response_timeout_ms=5000,
        conflate=False,
    )

    run_thread("controller_response_loop", lambda: controller_response_loop(controller))

    try:
        time.sleep(1.0)

        reply = client.send_command(Command(xy=[0.4, 0.2]))
        print("planner got command reply:", reply)

        response, rgb, depth = client.recv_response_latest()

        print("planner got response:")
        print("  seq:", response["seq"])
        print("  timestamp:", response["timestamp"])
        print("  rgb shape:", rgb.shape)
        print("  depth shape:", depth.shape)
        print("  joint_q:", response["joint_q"])
        print("  beliefs:", response["beliefs"])

    finally:
        client.close()
        controller.close()


if __name__ == "__main__":
    main()