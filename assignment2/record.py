"""
Record rigid body trajectories from Motive via NatNet.
Usage:
    python record.py                      # record all rigid bodies
    python record.py --name "myobject"    # record specific rigid body by name
    python record.py --out traj.csv       # custom output filename
"""

import argparse
import csv
import signal
import sys
import time
from natnet import NatNetClient, DataFrame, DataDescriptions

SERVER_IP = "192.168.1.92"
LOCAL_IP  = "192.168.1.94"

frames = []
target_name = None
rigid_body_names = {}  # id -> name, populated from data descriptions
recording = True
frame_count = 0  # counts all received frames, including untracked


def on_descriptions(desc: DataDescriptions):
    for rb in desc.rigid_bodies:
        rigid_body_names[rb.id_num] = rb.name
    if rigid_body_names:
        print(f"Rigid bodies found: {list(rigid_body_names.values())}")


def on_frame(data: DataFrame):
    global frame_count
    if not recording:
        return
    frame_count += 1
    if frame_count % 50 == 0:
        statuses = [(rigid_body_names.get(rb.id_num, rb.id_num), rb.tracking_valid) for rb in data.rigid_bodies]
        ts = data.suffix.timestamp
        print(f"\r  raw={frame_count} recorded={len(frames)} t={ts:.2f} rbs={statuses}    ", end="", flush=True)
    for rb in data.rigid_bodies:
        if rb.tracking_valid is False:
            continue
        name = rigid_body_names.get(rb.id_num, f"id_{rb.id_num}")
        if target_name and name != target_name:
            continue
        frames.append({
            "timestamp": data.suffix.timestamp,
            "name":      name,
            "x":         rb.pos[0],
            "y":         rb.pos[1],
            "z":         rb.pos[2],
            "qx":        rb.rot[0],
            "qy":        rb.rot[1],
            "qz":        rb.rot[2],
            "qw":        rb.rot[3],
        })
    if len(frames) % 100 == 0 and frames:
        f = frames[-1]
        print(f"\r  {len(frames)} frames | {f['name']}  x={f['x']:.3f}  y={f['y']:.3f}  z={f['z']:.3f}", end="", flush=True)


def save(path):
    if not frames:
        print("No frames recorded.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frames[0].keys())
        writer.writeheader()
        writer.writerows(frames)
    print(f"\nSaved {len(frames)} frames to {path}")


def main():
    global target_name, recording

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="Rigid body name to record")
    parser.add_argument("--out",  default="trajectory.csv", help="Output CSV file")
    args = parser.parse_args()
    target_name = args.name

    client = NatNetClient(
        server_ip_address=SERVER_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )
    client.on_data_description_received_event.handlers.append(on_descriptions)
    client.on_data_frame_received_event.handlers.append(on_frame)

    def stop(sig, frame):
        global recording
        recording = False
        print("\nStopping...")
        save(args.out)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)

    print(f"Connecting to Motive at {SERVER_IP}...")
    with client:
        client.request_modeldef()
        time.sleep(0.5)  # wait for descriptions to arrive
        if not rigid_body_names:
            print("WARNING: No rigid bodies received. Check Motive Assets panel.")
        else:
            print(f"Recording {'all rigid bodies' if not target_name else target_name}.")
            print("Press Ctrl+C to stop and save.")
        while True:
            try:
                client.update_sync()
            except OSError as e:
                if e.errno == 55:  # No buffer space available — transient on macOS, skip
                    continue
                raise


if __name__ == "__main__":
    main()
