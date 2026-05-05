# Assignment 2 — Mocap, UR5e Trajectory Execution & Digital Twin

## Overview

This assignment covers three tasks:
1. Recording a 3D trajectory with the Optitrack Mocap system
2. Preprocessing the trajectory and replaying it on a UR5e robot arm
3. Building a real-time digital twin of the UR5e in MuJoCo

---

## Setup

```bash
uv sync
```

The project requires Python ≥ 3.11. On macOS, `ur-rtde` is built from source and needs `cmake`, `ninja`, and `boost` via Homebrew.

**Network:** connect to the lab WiFi. The Motive PC is at `192.168.1.92`, the UR5e at `192.168.1.103`.

---

## Part 1 — Mocap Recording

### Record a trajectory

```bash
uv run record.py --name "RigidBody" --out data/trajectory.csv
```

Move the object, press `Ctrl+C` to stop. Saves `timestamp, name, x, y, z, qx, qy, qz, qw` per frame.

**Motive setup required:**
- Transmission Type: Unicast
- Stream Rigid Bodies: On
- Coordinate system: Y-up

---

## Part 2 — Trajectory Preprocessing

All preprocessing scripts read from `--file` and write to stdout, so they can be chained with redirects.

### Inspect

```bash
uv run preprocessing/analyze.py --file data/trajectory.csv
```

### Full pipeline

```bash
uv run preprocessing/normalize.py --file data/trajectory.csv --scale 0.3 \
    > data/trajectory_normalized.csv

uv run preprocessing/resample.py --file data/trajectory_normalized.csv \
    --points 500 --trim 30 --swap-xy \
    > data/trajectory_resampled.csv

uv run preprocessing/convert.py --file data/trajectory_resampled.csv \
    > data/trajectory_ready.csv
```

| Step | Script | What it does |
|---|---|---|
| Normalize | `normalize.py` | Center, flatten y=0, scale to robot workspace |
| Resample | `resample.py` | Cubic spline (positions) + SLERP (orientations), trim tails, optionally swap axes |
| Convert | `convert.py` | Quaternions → rotation vectors (axis-angle) for UR5e |

### Visualize

```bash
uv run preprocessing/visualize.py --file data/trajectory_ready.csv
```

Opens three Plotly tabs: 3D trajectory, position vs time, frame interval (dropped frame detector).

---

## Part 3 — Execute on UR5e

**Robot must be in Remote mode** (pendant → Settings → System → Remote Control).

```bash
uv run perform.py --file data/trajectory_ready.csv
```

The script:
1. Moves the robot to a known home joint configuration
2. Reads the home TCP pose and offsets the trajectory around it
3. Waits for Enter (verify waypoints before proceeding)
4. Streams waypoints via `servoL` at 5 cm/s

---

## Part 4 — Digital Twin (MuJoCo)

**Robot must be in Local mode** — the twin only reads joints, so the pendant works normally.

```bash
.venv/bin/mjpython twin.py
```

Mirrors the real robot's joint positions in MuJoCo at ~100 Hz. Prints live joint angles and network read latency. The UR5e model is in `ur5e/` (XML + mesh assets from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)).

> **macOS note:** MuJoCo's passive viewer requires `mjpython`. First recreate the venv with Homebrew Python:
> ```bash
> uv venv --python /opt/homebrew/opt/python@3.13/bin/python3.13 --clear && uv sync
> ```
