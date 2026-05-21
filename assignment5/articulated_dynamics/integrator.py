from articulated_dynamics.math_utils import nplib
import articulated_dynamics.math_utils as math
from articulated_dynamics.articulated_system import Free, Revolute, Prismatic
from articulated_dynamics.dynamics import Dynamics

def _jintpos(dof, dt, q, qd):
    if isinstance(dof, Free):
        #for free joint, will receive qd as spatial velocity to integrate
        rot, ang = q[3:7], qd[3:6]
        ang_norm = nplib.linalg.norm(ang) + 1e-8
        axis = ang / ang_norm
        angle = dt * ang_norm
        qrot = math.quat_rot_axis(axis, angle)
        rot = math.quat_mul(rot, qrot)
        rot = rot / nplib.linalg.norm(rot)
        pos, vel = q[0:3], qd[0:3]
        pos += vel * dt

        q_next = nplib.concatenate([pos, rot]) 
    elif isinstance(dof, Revolute) or isinstance(dof, Prismatic):
        q_next = q + qd * dt
    else:
        return NotImplementedError
    return q_next

def _jintpos_sys(sys, dt, q, qd):
    q_next_lst = []
    index_ptr_q = 0
    index_ptr_qd = 0
    for i in range(sys.num_links()):
        link = sys.links[i]
        
        q_size, qd_size = link.joint_dofs.q_size, link.joint_dofs.qd_size
        q_next = _jintpos(link.joint_dofs, dt, q[index_ptr_q:index_ptr_q+q_size], qd[index_ptr_qd:index_ptr_qd+qd_size])

        index_ptr_q += q_size
        index_ptr_qd += qd_size

        q_next_lst.append(q_next)
    return nplib.concatenate(q_next_lst)

def integrate_euler(sys, dt, q, qd, qdd, symplectic=False):
    #integrate qd
    qd_next = qd + qdd * dt
    #integrate q
    if symplectic:
        q_next = _jintpos_sys(sys, dt, q, qd_next)
    else:
        q_next = _jintpos_sys(sys, dt, q, qd)
    
    return q_next, qd_next

from scipy.optimize import fsolve, root

def integrate_implicit(sys, dt, q, qd, tau):
    #lets only be implicit about the velocity part

    def dynequ_residual(qd_next):
        q_next = _jintpos_sys(sys, dt, q, qd_next)
        qdd = Dynamics.forward(sys, q_next, qd_next, tau)
        return qd+qdd*dt - qd_next

    #info contains jacobian info that could be useful for adjoint methods
    #qd_next, info, ier, mesg = fsolve(dynequ_residual, qd, fprime=None, full_output=True)    #we can use auto-diff to provide jacobian as well
    sol = root(dynequ_residual, qd, method='broyden1')
    qd_next = sol.x
    q_next = _jintpos_sys(sys, dt, q, qd_next)
    return q_next, qd_next