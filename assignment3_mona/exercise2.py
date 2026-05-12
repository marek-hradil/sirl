import numpy as np
import math_utils as math
import mujoco

from spatial_algebra import Transform, Motion

model = mujoco.MjModel.from_xml_path("threelinks.xml")
data  = mujoco.MjData(model)


def compute_adjacent_transforms(q):
    """
    Compute relative kinematic transforms between adjacent link COM(center-of-mass) frames.
    """
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)

    # World-frame poses of each link's frame (from <body pos=... quat=...>)
    T_world_body = [
        Transform(trans=data.xpos[i].copy(), 
                  rot=data.xquat[i].copy())
        for i in range(1, model.nbody)
    ]

    # Constant body-to-COM transforms (from <inertial pos=... quat=...>)
    T_body_to_com = [
        Transform(trans=model.body_ipos[i].copy(),
                  rot=model.body_iquat[i].copy())
        for i in range(1, model.nbody)
    ]

    # COM-frame poses in world: T_world_com = T_world_body * T_body_to_com
    T_world_com = [
        T_wb.apply_transform(T_bc)
        for T_wb, T_bc in zip(T_world_body, T_body_to_com)
    ]

    # Adjacent COM-to-COM relative transforms
    T_base = Transform()  # world frame (identity)
    T_rel = [
        T_child.create_local(T_world_com[i - 1] if i > 0 else T_base)
        for i, T_child in enumerate(T_world_com)
    ]

    return T_rel, T_body_to_com


def print_transforms(T_rel):
    def matrix(T):
        R = math.rotate(np.eye(3), T.rot)
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3,  3] = T.trans
        return mat

    labels = ["0_T_1", "1_T_2", "2_T_3"]
    for label, T in zip(labels, T_rel):
        print(f"\n    {label} =")
        for row in np.round(matrix(T), 4):
            print("     ", row)


def compute_spatial_velocities(qdot, T_rel, T_body_to_com):
    """
    Recursively compute the spatial velocity of each link, expressed and
    referenced at that link's COM frame.
    """

    S_com = []

    for i in range(model.njnt):
        body_id = model.jnt_bodyid[i]

        # In this model, MuJoCo body 0 is the world.
        # T_body_to_com[0] corresponds to body_id 1, etc.
        link_id = body_id - 1

        # Joint screw in the joint frame.
        S_joint = Motion(
            ang=model.jnt_axis[i].copy(),
            lin=np.zeros(3)
        )

        # Pose of joint frame relative to body frame
        T_body_to_joint = Transform(
            trans=model.jnt_pos[i].copy(),
            rot=np.array([1., 0., 0., 0.])
        )

        # Pose of COM frame relative to body frame
        T_body_to_com_i = T_body_to_com[link_id]

        # Pose of COM frame relative to joint frame
        T_joint_to_com = T_body_to_com_i.create_local(T_body_to_joint)

        # Transform joint screw into the COM frame
        S_com_i = S_joint.apply_transform(T_joint_to_com)
        S_com.append(S_com_i)

    V = [Motion()]  # base/world velocity = zero

    for i, T in enumerate(T_rel):
        V_parent_in_child = V[-1].apply_transform(T)
        V_i = V_parent_in_child + S_com[i] * qdot[i]
        V.append(V_i)

    return V[1:]


def print_velocities(V):
    for i, v in enumerate(V):
        print(f"    V_{i+1}:  w = {np.round(v.ang, 4)},   v = {np.round(v.lin, 4)}")


test_cases = [
    ("q=[0, 0, 0]            qdot=[1, 0, 0]",
        np.array([0.0, 0.0, 0.0]),             np.array([1.0, 0.0, 0.0])),
    ("q=[90deg, 0, 0]        qdot=[1, 0, 0]",
        np.array([np.pi/2, 0.0, 0.0]),         np.array([1.0, 0.0, 0.0])),
    ("q=[90deg, 90deg, 90deg]  qdot=[1, 1, 1]",
        np.array([np.pi/2, np.pi/2, np.pi/2]), np.array([1.0, 1.0, 1.0])),
    ("q=[45deg, 45deg, 0]      qdot=[0, 1, 2]",
        np.array([np.pi/4, np.pi/4, 0.0]),     np.array([0.0, 1.0, 2.0])),
]


if __name__ == "__main__":
    for label, q, qdot in test_cases:
        print(f"\n{'='*64}")
        print(f"  {label}")
        print('='*64)
        T_rel, T_bc = compute_adjacent_transforms(q)
        print_transforms(T_rel)
        print()
        V = compute_spatial_velocities(qdot, T_rel, T_bc)
        print_velocities(V)