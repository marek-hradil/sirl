from articulated_dynamics.math_utils import nplib

from articulated_dynamics.spatial_algebra import Transform, Motion, Force, Inertia

class Link:
    def __init__(self, name) -> None:
        self.name = name                    #must be unique so we can quickly index the link
        self.prev_transform = Transform()   #predecessor joint frame relative to parent link frame
        self.joint_frame = Transform()      #joint frame w.r.t. current link frame
                                            #note for these transforms, it is more intuitive for users to define [R, t; 0, 1] with both R and t w.r.t. the reference frame, i.e. relative configuration
                                            #do not regard this as a displacement operator as the series of joint DOFs to populate coordinate transformation matrix
                                            #for instance, joint motions each expressed in the local frame, will lead to right multiplication of transformation matrices
                                            #i.e. [R, 0; 0, 1]*[I, t; 0, 1]. In this case, it will yield [R, Rt; 0, 1] which correspondes to coordinate transforms [R^T, -t; 0, 1]
        self.joint_dofs = Weld()            #by default the link is welded to its parent
        
        #default inertia with inertia frame same as the link frame, cubnoid with unit sizes and mass  
        self.inertia = Inertia.from_it_and_mass(it=nplib.eye(3)/6., mass=1.)  
        
        #shape for rendering, can be None if no visualization is needed.
        #dictionary key and params:
        #shape:     box/cylinder/sphere
        #size:      [x, y, z] - w.r.t. link frame. 
        #           box: x - width, y - height, z - depth; 
        #           cylinder: x - radius, y - height; 
        #           sphere: x - radius
        #pos/quat:  pose w.r.t. link frame   
        #color:     html color code, need extra process to convert rgb tuples
        self.viz_shape ={'shape':'box', 'size':[1, 1, 1], 'pos':[0, 0, 0], 'quat':[0, 0, 0, 1], 'color':'LightSlateGray'}       
                   
class DOF:
    def __init__(self) -> None:
        self.motion = Motion()  
        self.param = None       #used by DOF of some types
        self.stiffness = 0
        self.damping = 0
        self.q_size = 0
        self.qd_size = 0

class Weld(DOF):
    def __init__(self) -> None:
        super().__init__()

class Free(DOF):
    def __init__(self) -> None:
        super().__init__()
        self.q_size = 7
        self.qd_size = 6

class Revolute(DOF):
    def __init__(self) -> None:
        super().__init__()
        self.motion = Motion(ang=nplib.array([0, 0, 1]))   #always use z-axis as the DOF, see Featherstone Table 4.1
        self.q_size = 1
        self.qd_size = 1

class Prismatic(DOF):
    def __init__(self) -> None:
        super().__init__()
        self.motion = Motion(lin=nplib.array([0, 0, 1]))   #always use z-axis as the DOF, see Featherstone Table 4.1
        self.q_size = 1
        self.qd_size = 1

class State:
    """
    Dynamic state
    """
    def __init__(self, q=None, qd=None, x=None, xd=None) -> None:
        self.q =    q
        self.qd =   qd
        self.x =    x           #a list of link pose w.r.t. world frame
        self.xd =   xd          #a list of link velocity w.r.t. world frame

class ArticulatedSystem:
    def __init__(self, num_links=2) -> None:
        self.gravity = nplib.array([0, -9.81, 0])
        #links must be subject to a topological sort to ensure the index of parent link is before its successors.
        self.links = [Link('link_{0}'.format(i)) for i in range(num_links)]
        self.parents = [i-1 for i in range(num_links)]  # a single serial link chain

    def num_links(self):
        return len(self.links)
    
    def q_size(self):
        return sum([l.joint_dofs.q_size for l in self.links])
    
    def qd_size(self):
        return sum([l.joint_dofs.qd_size for l in self.links])



        

