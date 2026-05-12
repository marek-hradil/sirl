"""Probe: compare exercise2's spatial velocities against MuJoCo, using
the velocity of each body at its own CoM (not subtree CoM)."""
import numpy as np
import mujoco
from exercise2 import (
    model, data, compute_adjacent_transforms, compute_spatial_velocities,
)


def mjc_body_velocity_at_own_com(body_idx, model, data):
    """Spatial velocity of body i at its own CoM, expressed in the body's COM frame.

    mj_objectVelocity with mjOBJ_BODY returns velocity expressed in the
    body's *inertial* frame (which here equals the body frame), measured
    at the body's CoM. Returned as [ang(3), lin(3)].
    """
    out = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_idx, out, 1)
    return out[:3].copy(), out[3:].copy()


def main():
    print("\n=== Joint and inertial info ===")
    for i in range(model.njnt):
        print(f"  joint {i}: axis={model.jnt_axis[i]}, pos={model.jnt_pos[i]}, body={model.jnt_bodyid[i]}")
    for i in range(1, model.nbody):
        print(f"  body {i}: ipos={model.body_ipos[i]}, iquat={model.body_iquat[i]}")

    test_cases = [
        ("q=0  qd=[1,0,0]", np.zeros(3), np.array([1.0, 0.0, 0.0])),
        ("q=[pi/2, 0, 0]  qd=[1,0,0]",
         np.array([np.pi/2, 0.0, 0.0]),  np.array([1.0, 0.0, 0.0])),
        ("q=[pi/4, pi/4, 0]  qd=[0, 1, 2]",
         np.array([np.pi/4, np.pi/4, 0.0]), np.array([0.0, 1.0, 2.0])),
    ]

    for label, q, qd in test_cases:
        print(f"\n--- {label} ---")
        data.qpos[:] = q
        data.qvel[:] = qd
        mujoco.mj_forward(model, data)

        T_rel, T_bc = compute_adjacent_transforms(q)
        Vs = compute_spatial_velocities(qd, T_rel, T_bc)

        for i, V_ours in enumerate(Vs):
            mj_ang, mj_lin = mjc_body_velocity_at_own_com(i + 1, model, data)
            print(f"  V_{i+1}  ours:   ang={np.round(V_ours.ang, 5)}  lin={np.round(V_ours.lin, 5)}")
            print(f"        mujoco: ang={np.round(mj_ang, 5)}  lin={np.round(mj_lin, 5)}")


if __name__ == "__main__":
    main()
