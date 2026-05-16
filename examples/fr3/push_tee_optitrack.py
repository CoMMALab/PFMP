"""
push_tee_optitrack.py — Motive/OptiTrack perception for the push-T task.

Stage 1: Connect to Motive and print labeled markers each frame.

Usage:
    python examples/fr3/push_tee_optitrack.py --server 192.168.1.100
"""

import argparse

from new_natnet_client.Client import NatNetClient, NatNetParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1",
                        help="IP address of the Motive machine")
    parser.add_argument("--local", default="127.0.0.1",
                        help="IP address of this machine")
    args = parser.parse_args()

    params = NatNetParams(
        server_address=args.server,
        local_ip_address=args.local,
        use_multicast=False,
    )

    print(f"Connecting to Motive at {args.server}...")
    with NatNetClient(params) as client:
        if client is None:
            print("Failed to connect.")
            return

        print("Connected. Streaming rigid bodies (Ctrl-C to stop)...\n")
        for frame in client.MoCap(timeout=None):
            t = frame.suffix_data.timestamp
            for rb in frame.rigid_body_data.rigid_bodies:
                if not rb.tracking:
                    continue
                p = rb.pos
                q = rb.rot
                print(f"t={t:.3f}  id={rb.identifier}  "
                      f"pos=({p.x:7.4f}, {p.y:7.4f}, {p.z:7.4f})  "
                      f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})  "
                      f"err={rb.err:.4f}")


if __name__ == "__main__":
    main()
