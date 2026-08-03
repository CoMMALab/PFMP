# pub_sub/test.py

import time
import threading
import traceback

from protocol import Command
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

        if cmd.action:
            controller.reply_command(accepted=True, extra={"action": cmd.action})
        else:
            controller.reply_command(accepted=True, extra={"xy": cmd.xy})


def main():
    controller = ControllerEndpoint(
        command_bind_addr="tcp://*:5555",
    )

    run_thread("controller_command_loop", lambda: controller_command_loop(controller))

    client = PlannerClient(
        command_addr="tcp://localhost:5555",
        command_timeout_ms=5000,
    )

    try:
        time.sleep(1.0)

        reply = client.send_command(Command(xy=[0.4, 0.2]))
        print("planner got command reply:", reply)

        client.lifecycle("descend")
        client.lifecycle("home")

    finally:
        client.close()
        controller.close()


if __name__ == "__main__":
    main()
