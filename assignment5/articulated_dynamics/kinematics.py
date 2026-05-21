import articulated_dynamics.math_utils as math
from articulated_dynamics.articulated_system import *

class Kinematics:

    @classmethod
    def _jcalc_pos(cls, dof, q):
        if isinstance(dof, Weld):
            return Transform()   #no displacement, zero vel and acc
        elif isinstance(dof, Free):
            return Transform(rot=q[..., 3:7], trans=q[..., :3])
        elif isinstance(dof, Revolute):
            rot = math.normalize(math.quat_rot_axis(dof.motion.ang, q[None, ...][..., 0][..., None]))[0].squeeze()
            return Transform(rot=rot, trans=nplib.zeros(q[None, ...][..., 0].shape[:-1]+(3,)))
        elif isinstance(dof, Prismatic):
            rot = nplib.broadcast_to(nplib.array([1, 0, 0, 0]), q[None, ...][..., 0].shape[1:-1]+(4,))
            return Transform(trans=(dof.motion.lin*(q[None, ...][..., 0][..., None])).squeeze(), rot=rot)
        else:
            return NotImplementedError
        
    @classmethod
    def _jcalc_vel(cls, dof, qd):
        if isinstance(dof, Weld):
            return Motion()   # zero vel and acc
        elif isinstance(dof, Free):
            return Motion(ang=qd[..., 3:6], lin=qd[..., :3])
        elif isinstance(dof, Revolute):
            return Motion(ang=(dof.motion.ang*(qd[None, ...][..., 0][..., None])).squeeze(), lin=nplib.zeros(qd[None, ...][..., 0].shape[1:-1]+(3,)))
        elif isinstance(dof, Prismatic):
            return Motion(lin=(dof.motion.lin*(qd[None, ...][..., 0][..., None])).squeeze(), ang=nplib.zeros(qd[None, ...][..., 0].shape[1:-1]+(3,)))
        else:
            return NotImplementedError

    @classmethod
    def _forward_pos(cls, sys, q):
        x_ws_lst, x_ps_lst = [], []
        index_ptr_q = 0
        
        for i in range(sys.num_links()):
            link = sys.links[i]
            q_size = link.joint_dofs.q_size
            x_jpjs = cls._jcalc_pos(link.joint_dofs, q[index_ptr_q:index_ptr_q+q_size])
            index_ptr_q += q_size
            
            #compute world representation of link pose based on the result of parent link
            if sys.parents[i] == -1:
                #parent is the world frame
                x_wp = Transform()
            else:
                #parent must be computed in x_res and xd_res
                x_wp = x_ws_lst[sys.parents[i]]

            #pose: chain of coordinate transform matrices
            #x_ps = x_pjp * x_jpjs * inv(x_sjs)
            x_ps = link.prev_transform.apply_transform(x_jpjs).apply_transform(link.joint_frame.create_inverse())
            x_ws = x_wp.apply_transform(x_ps)
        
            x_ws_lst.append(x_ws)
            x_ps_lst.append(x_ps)

        return x_ws_lst, x_ps_lst

    @classmethod
    def _forward_vel(cls, sys, x_ps_lst, qd):
        #x_ps_lst: the list of relative pose of successor links w.r.t. their parents
        xd_ss_lst, xd_srel_lst = [], []

        index_ptr_qd = 0
        for i in range(sys.num_links()):
            link = sys.links[i]
            #calculate joint motion induced by joint motion 
            #the joint motion is characterised by the difference between joint frame of current link (successor) and the same joint frame attached to the predecessor link
            qd_size = link.joint_dofs.qd_size
            xd_jsrel = cls._jcalc_vel(link.joint_dofs, qd[index_ptr_qd:index_ptr_qd+qd_size])
            index_ptr_qd += qd_size

            #compute world representation of link pose and velocity based on the result of parent link
            if sys.parents[i] == -1:
                #parent is the world frame
                xd_pp = Motion()
            else:
                #parent must be computed in x_res and xd_res
                xd_pp = xd_ss_lst[sys.parents[i]]

            #velocity: link frame origin as referene point, expressed in link frame of reference
            #feel relative velocity is measured in successor joint frame, so xd_ps actually means xd_jspsjs...            
            #Featherstone Eq. (2.16)
            xd_srel = xd_jsrel.apply_inv_transform(link.joint_frame)
            xd_ss = xd_pp.apply_transform(x_ps_lst[i]) + xd_srel

            xd_ss_lst.append(xd_ss)
            xd_srel_lst.append(xd_srel)

        return xd_ss_lst, xd_srel_lst

    @classmethod
    def _forward_acc(cls, sys, x_ps_lst, xd_ss_lst, xd_srel_lst, qdd, g_flag=False):
        xdd_ss_lst = []
        xdd_srel_bias_lst = []
        index_ptr_qd = 0

        for i in range(sys.num_links()):
            link = sys.links[i]
            #calculate joint frame pose and motion induced by joint motion 
            #the joint motion is characterised by the difference between joint frame of current link (successor) and the same joint frame attached to the predecessor link
            qd_size = link.joint_dofs.qd_size
            #so far vel and acc can share a same jcalc
            qdd_link = qdd[..., index_ptr_qd:index_ptr_qd+qd_size]
            xdd_jsrel = cls._jcalc_vel(link.joint_dofs, qdd_link)

            index_ptr_qd += qd_size

            #compute world representation of link acc based on the result of parent link
            if sys.parents[i] == -1:
                #parent is the world frame
                grav_lin = nplib.broadcast_to(-sys.gravity * g_flag, qdd_link.shape[:-1]+(3,))
                grav_ang = nplib.zeros(qdd_link.shape[:-1]+(3,))
                xdd_pp = Motion(lin=grav_lin, ang=grav_ang)
            else:
                #parent must be computed in x_res and xd_res
                xdd_pp = xdd_ss_lst[sys.parents[i]]
            
            #
            #acceleration: link frame origin as reference point, expressed in world frame of reference
            #calculating relative acc needs a rethink for best implementation, now it only allows for 1-dof angular or translational motion
            #Featherstone Eq. (2.55)
            #Note jcalc here only returns ScrewAxis * qdd, this is only part of the relative acceleration
            xdd_srel_bias = xd_ss_lst[i].cross_motion(xd_srel_lst[i])
            xdd_srel = xdd_jsrel.apply_inv_transform(link.joint_frame) + xdd_srel_bias
            xdd_ss = xdd_pp.apply_transform(x_ps_lst[i]) + xdd_srel

            xdd_ss_lst.append(xdd_ss)
            xdd_srel_bias_lst.append(xdd_srel_bias)

        return xdd_ss_lst, xdd_srel_bias_lst

    @classmethod
    def _forward(cls, sys, q, qd, qdd, g_flag=False):
        x_ws_lst, x_ps_lst = cls._forward_pos(sys, q)
        xd_ss_lst, xd_srel_lst = cls._forward_vel(sys, x_ps_lst, qd)
        xdd_ss_lst, xdd_srel_bias_lst = cls._forward_acc(sys, x_ps_lst, xd_ss_lst, xd_srel_lst, qdd, g_flag) 
        return x_ws_lst, x_ps_lst, xd_ss_lst, xdd_ss_lst, xdd_srel_bias_lst
    
    @classmethod
    def forward(cls, sys, q, qd, qdd):
        #forward kinematics but change the express-in frame of motion to world frame of reference
        x_ws_lst, x_ps_lst, xd_ss_lst, xdd_ss_lst, _ = cls._forward(sys, q, qd, qdd, g_flag=True)
        xd_ws_lst = []
        xdd_ws_lst = []
        for x_ws, xd_ss, xdd_ss in zip(x_ws_lst, xd_ss_lst, xdd_ss_lst):
            xd_ws = Motion(ang=math.rotate(xd_ss.ang, x_ws.rot), lin=math.rotate(xd_ss.lin, x_ws.rot))

            acc_ang = math.rotate(xdd_ss.ang, x_ws.rot)
            acc_lin = math.rotate(xdd_ss.lin, x_ws.rot)
            #note accleration here is spatial acceleration, shall we recover it for material point acceleration?
            #Featherstone Eq. (2.50)
            xdd_ws = Motion(ang=acc_ang, lin=acc_lin + nplib.cross(xd_ws.ang, xd_ws.lin))

            xd_ws_lst.append(xd_ws)
            xdd_ws_lst.append(xdd_ws)
        return x_ws_lst, xd_ws_lst, xdd_ws_lst

    @classmethod
    def inverse(cls, sys, x, xd):
        return NotImplementedError


    @classmethod
    def jacobian(cls, sys, q, body_com=False):
        #body jacobian by feeding forward velocity kinematics with one-hot qd to calculate columns
        batched_qd = nplib.eye(sys.qd_size())
        x_ws_lst, x_ps_lst = Kinematics._forward_pos(sys, q)

        #using list comprehension instead of map or jax.vmap for small dof and interoperability with numpy
        xd_ss_lst_lst = [cls._forward_vel(sys, x_ps_lst, qd)[0] for qd in batched_qd]
        #assembly motion components and convert to (n_links, 6, qd)
        jac_lst = nplib.array([ [ nplib.concatenate([xd_ss.ang, xd_ss.lin]) for xd_ss in xd_ss_lst] for xd_ss_lst in xd_ss_lst_lst]).transpose([1, 2, 0])

        return jac_lst
    
