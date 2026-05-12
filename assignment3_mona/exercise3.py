import numpy as np
import mujoco

from spatial_algebra import Motion, Inertia
from exercise2 import (
    model,
    data,
    compute_adjacent_transforms,
    compute_spatial_velocities,
)


def _link_inertias():
    """
    Spatial inertia of each link, referenced in its own COM frame.
    """
    return [
        Inertia.from_it_and_mass(
            it=np.diag(model.body_inertia[i].copy()),
            mass=float(model.body_mass[i]),
        )
        for i in range(1, model.nbody)
    ]


def _joint_screw_axes(T_body_to_com):
    """Each joint's screw axis, expressed in the corresponding link's COM frame.

    The three-link XML defines hinge joints about the body-frame z-axis at the
    body origin, so S in the body frame is [0,0,1, 0,0,0]. We transform it into
    the COM frame so it matches the convention used everywhere else.
    """
    S_body = Motion(ang=np.array([0.0, 0.0, 1.0]), lin=np.array([0.0, 0.0, 0.0]))
    return [S_body.apply_transform(T_bc) for T_bc in T_body_to_com]


class ABA:
    def __init__(self, q, n, include_gravity=True):
        T_rel, T_body_to_com = compute_adjacent_transforms(q)
        self.S = _joint_screw_axes(T_body_to_com)
        self.N = n
        self.M = _link_inertias()
        self.Vdot = ABA.compute_base_acceleration(include_gravity)
        self.T_rel = T_rel
        self.T_body_to_com = T_body_to_com
        self.include_gravity = include_gravity

    # this computes the link velocities and velocity bias
    def outward_kinematics(self, qd):
        V = compute_spatial_velocities(qd, self.T_rel, self.T_body_to_com)
        b = [V[i].cross_motion(self.S[i] * qd[i]) for i in range(self.N)]

        return V, b

    def inward_inertia_force(self, V, b, tau):
        IA = [
            self.M[i] for i in range(self.N)
        ]  # base inertias, but without the children!
        pA = [
            V[i].cross_force(self.M[i].mul(V[i])) for i in range(self.N)
        ]  # bias forces, V_i x M_i * V_i

        U = (
            [None] * self.N
        )  # U[i] -> intertia for the given joint (but with the children accounted for!)
        D = (
            [None] * self.N
        )  # scalar: D_i = S_i^T . U_i -> intertia for the given joint, but only in the directions it moves
        u = (
            [None] * self.N
        )  # scalar: u_i = tau_i - S_i^T · pA_i -> force we are pushing the joint with, but minus the bias resistance

        for i in range(self.N - 1, -1, -1):
            U[i] = IA[i].mul(self.S[i])
            D[i] = self.S[i].dot_force(U[i])
            u[i] = float(tau[i]) - self.S[i].dot_force(pA[i])

            if i > 0:  # if not base joint (which does not have to account for children)
                Ia_i = IA[i] - Inertia.from_dyad(U[i]) * (1.0 / D[i])
                pa_i = pA[i] + Ia_i.mul(b[i]) + U[i] * (u[i] / D[i])
                IA[i - 1] = IA[i - 1] + Ia_i.apply_transform(
                    self.T_rel[i].create_inverse()
                )
                pA[i - 1] = pA[i - 1] + pa_i.apply_inv_transform(self.T_rel[i])

        return IA, D, u

    @staticmethod
    def compute_base_acceleration(include_gravity: bool):
        if not include_gravity:
            return [Motion()]

        g = np.asarray(model.opt.gravity, dtype=float)
        # Base "fictitious" upward acceleration so gravity acts on every link
        Vdot_base = Motion(ang=np.zeros(3), lin=-g)

        return [Vdot_base]

    def outward_acceleration(self, b, IA, D, u):
        qdd = np.zeros(self.N)

        for i in range(self.N):
            Vdot_parent_in_i = self.Vdot[-1].apply_transform(self.T_rel[i])
            a_prime = Vdot_parent_in_i + b[i]
            qdd_i = (u[i] - self.S[i].dot_force(IA[i].mul(a_prime))) / D[i]
            qdd[i] = qdd_i
            self.Vdot.append(a_prime + self.S[i] * qdd_i)

        return qdd


def forward_dynamics_aba(q, qd, tau, include_gravity=True):
    """
    Featherstone's Articulated Body Algorithm.
    """
    aba = ABA(q, model.nbody - 1, include_gravity)

    V, b = aba.outward_kinematics(qd)
    IA, D, u = aba.inward_inertia_force(V, b, tau)

    return aba.outward_acceleration(b, IA, D, u)


def verify_with_mujoco(q, qd, tau, label=""):
    qdd_ours = forward_dynamics_aba(q, qd, tau, include_gravity=True)

    data.qpos[:] = q
    data.qvel[:] = qd
    data.qfrc_applied[:] = tau
    mujoco.mj_forward(model, data)
    qdd_mjc = data.qacc.copy()

    print(f"\n--- {label} ---")
    print(f"  qdd (ABA)    : {np.round(qdd_ours, 6)}")
    print(f"  qdd (MuJoCo) : {np.round(qdd_mjc, 6)}")
    print(f"  max abs diff : {np.max(np.abs(qdd_ours - qdd_mjc)):.2e}")


if __name__ == "__main__":
    test_cases = [
        (
            "q=0, qd=0, tau=0 (free fall under gravity only)",
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        ),
        ("q=0, qd=0, tau=[1,0,0]", np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0])),
        (
            "q=[pi/2,0,0], qd=0, tau=0",
            np.array([np.pi / 2, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
        ),
        (
            "q=[pi/4,pi/4,0], qd=[0,1,2], tau=[0.1,0.2,0.3]",
            np.array([np.pi / 4, np.pi / 4, 0.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.1, 0.2, 0.3]),
        ),
        (
            "q=[pi/3,-pi/4,pi/6], qd=[1,-1,0.5], tau=[0.5,-0.3,0.2]",
            np.array([np.pi / 3, -np.pi / 4, np.pi / 6]),
            np.array([1.0, -1.0, 0.5]),
            np.array([0.5, -0.3, 0.2]),
        ),
    ]

    for label, q, qd, tau in test_cases:
        verify_with_mujoco(q, qd, tau, label=label)
