from articulated_dynamics.kinematics import *

class Dynamics:
    
    @classmethod
    def _inverse_rne(cls, sys, x_ps_lst, xd_lst, xdd_lst):
        #compute inertial forces under motion xd and xdd
        f_lst = []
        for i in range(sys.num_links()):
            #Featherstone Eq. (5.9)
            link = sys.links[i]
            f_b =  link.inertia.mul(xdd_lst[i]) + xd_lst[i].cross_force(link.inertia.mul(xd_lst[i]))
            f_lst.append(f_b)
        
        #backward pass
        tau_lst = []
        for i in reversed(range(sys.num_links())):
            f_jb = f_lst[i].apply_transform(sys.links[i].joint_frame)   #transform force in body frame to joint frame

            #Featherstone Eq. (5.11)
            if isinstance(sys.links[i].joint_dofs, Free):
                #remember to flip as it will be reversed again
                tau_lst += [f_jb.ang[2], f_jb.ang[1], f_jb.ang[0], f_jb.lin[2], f_jb.lin[1], f_jb.lin[0]]
            else:
                tau = sys.links[i].joint_dofs.motion.dot_force(f_jb)
                tau_lst.append(tau)

            #Featherstone Eq. (5.10)
            if sys.parents[i] != -1:
                f_lst[sys.parents[i]] = f_lst[sys.parents[i]] + f_lst[i].apply_inv_transform(x_ps_lst[i])

        return nplib.flip(nplib.array(tau_lst), 0)

    @classmethod
    def inverse(cls, sys, q, qd, qdd, f_ext=None, g_flag=True):
        #calc body pose, body twist and body acc (spatial acc)
        x_lst, x_ps_lst, xd_lst, xdd_lst, _ = Kinematics._forward(sys, q, qd, qdd, g_flag)
        return cls._inverse_rne(sys, x_ps_lst, xd_lst, xdd_lst)
        
    
    @classmethod
    def forward(cls, sys, q, qd, tau, f_ext=None):
        #C(q, qd)qd + g(q)
        qdd_zero = nplib.zeros(sys.qd_size())        
        bias = tau - cls.inverse(sys, q, qd, qdd_zero, g_flag=True)

        #M(q)
        Mq = cls.mass(sys, q)
        #solve M(q)qdd + C(q, qd)qd + g(q) = tau
        qdd = nplib.linalg.solve(Mq, bias)
        return qdd
    
    @classmethod
    def _mass_from_inverse(cls, sys, q):
        #get mass matrix under current configuration q
        batched_qdd = nplib.eye(sys.qd_size())
        
        #reuse the result from inverse dynamics M(q)qdd + C(q, qd)qd + g(q) = tau
        #so each column of M is tau under one-hot qdd with 1 at the column index, and qd=0 & g(q)=0
        qd_zero = nplib.zeros(sys.qd_size())
        x_lst, x_ps_lst = Kinematics._forward_pos(sys, q)
        xd_ss_lst, xd_srel_lst = Kinematics._forward_vel(sys, x_ps_lst, qd_zero)
        #xdd_ss_lst_lst = [ Kinematics._forward_acc(sys, x_ps_lst, xd_ss_lst, xd_srel_lst, qdd, g_flag=False) for qdd in batched_qdd ]
        xdd_ss_lst, _ = Kinematics._forward_acc(sys, x_ps_lst, xd_ss_lst, xd_srel_lst, batched_qdd, g_flag=False)
        #M_col = lambda xdd_ss_lst: cls._inverse_rne(sys, x_ps_lst, xd_ss_lst, xdd_ss_lst)
        #M = jax.vmap(M_col, 0)
        #mass = nplib.stack([ cls._inverse_rne(sys, x_ps_lst, xd_ss_lst, xdd_ss_lst) for xdd_ss_lst in xdd_ss_lst_lst] )
        mass = cls._inverse_rne(sys, x_ps_lst, xd_ss_lst, xdd_ss_lst)

        #mass = M(batched_xdd)
        mass = (mass + mass.transpose()) / 2.
        return mass

    @classmethod
    def mass(cls, sys, q):
        return cls._mass_from_inverse(sys, q)