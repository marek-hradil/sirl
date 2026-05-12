# Assignment 3 — Spatial Algebra & Articulated Body Dynamics

A 3-link planar pendulum simulated with MuJoCo. Exercises build from visual inspection to full forward dynamics via Featherstone's Articulated Body Algorithm (ABA).

## Model

`threelinks.xml` — three identical aluminium-density box links (0.3 m, 2.025 kg each) connected by hinge joints rotating about the Z-axis. Links chain vertically upward at rest.

## Setup

```bash
uv sync
```

All exercises must be run from this directory (they load `threelinks.xml` relative to cwd).

---

## Exercise 1 — Visualise the model

Opens the MuJoCo interactive viewer.

```bash
uv run python exercise1.py
```

Press **Space** to start the simulation. The chain starts in the unstable upright equilibrium and falls under gravity.

---

## Exercise 2 — Kinematics: transforms and spatial velocities

Computes adjacent COM-frame transforms and spatial velocities for a set of test configurations, then prints them.

```bash
uv run python exercise2.py
```

**What it does:**

- `compute_adjacent_transforms(q)` — uses MuJoCo kinematics to build relative transforms `T_rel[i]` between consecutive link COM frames, plus the constant body→COM transforms `T_body_to_com`.
- `compute_spatial_velocities(qdot, T_rel, T_body_to_com)` — recursively propagates the chain using the joint screw axes (properly offset from the body origin via `model.jnt_pos`):

  ```
  V_i = T_rel[i] * V_{i-1} + S_i * qdot_i
  ```

Output is the 4×4 relative transform matrices followed by spatial velocities `[ω, v_CoM]` for each link.

---

## Exercise 3 — Forward dynamics: Featherstone's ABA

Implements the **Articulated Body Algorithm** and verifies it against MuJoCo's forward dynamics.

```bash
uv run python exercise3.py
```

**What it does:**

The `forward_dynamics_aba(q, qd, tau)` function runs three passes:

| Pass | Function | What happens |
|------|----------|-------------|
| Outward (kinematics) | `outward_kinematics` | Compute spatial velocities `V_i` and velocity-product bias `b_i = V_i × S_i q̇_i` |
| Inward | `inward_inertia_force` | Propagate articulated inertias `IA_i` and bias forces `pA_i` from leaves to root |
| Outward (accelerations) | `outward_acceleration` | Compute joint accelerations `q̈_i` and propagate body accelerations |

Verification note: `threelinks.xml` includes `damping="0.4"` per joint (joint friction). ABA models pure rigid-body dynamics, so `verify_with_mujoco` temporarily zeros joint damping before calling `mj_forward` to get a fair comparison.

Expected output — all diffs should be ~1e-13 (floating-point noise):

```
--- q=0, qd=0, tau=0 ---           max abs diff : 0.00e+00
--- q=0, qd=0, tau=[1,0,0] ---     max abs diff : 2.66e-14
--- q=[pi/2,0,0], qd=0, tau=0 ---  max abs diff : 1.92e-13
--- q=[pi/4,pi/4,0], ...        ---  max abs diff : 3.55e-15
--- q=[pi/3,-pi/4,pi/6], ...    ---  max abs diff : 1.35e-13
```

---

## Library files

| File | Purpose |
|------|---------|
| `spatial_algebra.py` | `Transform`, `Motion`, `Force`, `Inertia` — Featherstone-convention spatial vector types |
| `math_utils.py` | Quaternion arithmetic and rotation helpers (from Brax) |
| `threelinks.xml` | MuJoCo model |
