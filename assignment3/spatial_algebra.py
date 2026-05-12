from typing import NamedTuple

#from articulated_dynamics.math_utils import nplib
#import articulated_dynamics.math_utils as math

from math_utils import nplib
import math_utils as math

class Transform(NamedTuple):
    """
    Transforms the position and rotation of a coordinate frame.
    When applied to the coordinate frame, 
    with the rotation and translation represented in the reference frame that the frame is represented in
    i.e. left multiply 
    Or rotation and translation represented in the frame itself i.e. right multiply

    Check http://royfeatherstone.org/spatial/v2/xforms.html for using transformation to represent "coordinate transform" and "displacement"

    May also used to represent the relative pose of a frame to the other one, e.g. T_IA for pose of A frame w.r.t. I frame

    Be careful about how this is used for plucker coordinate transform
    For homogenous coordinate transform, i.e. a point with coordinates w.r.t. A and B frames, T_IA * p_A = T_IB * p_B
    T_BA * p_A = p_B -> T_BA transform coordinate of P from A frame to B frame, with the pose of frame A relative to frame B
    This relative pose matrix also represents the inverse of displacement operator acting on frame A to move it aligning with frame B, with rotation and translation represented in frame A
    T_IA = T_IB * T_BA -> T_IA * T_AB = T_IB
    Assuming this displacement operator is T_AB = [R, t; 0, 1] -> T_BA = [R^T, -R^Tt; 0, 1] 
    from Featherstone (2.28) note that E = R^T, r = t and this yields the coordinate transform for Featherstone (2.24) and (2.25)
    
    This convention is different from the adjoint representation of SE(3) in Lynch&Park Definition 3.20
    There the adjoint representation is used in a similar semantic meaning as homogenous coordinate transformation
    "T_IA transforms homogenous/Plucker coordinates in A frame to I frame" while here when applied to Spatial vectors,
    we follow Featherstone's convention for operation of "using Plucker transform defined by T_IA to transform coordinate referenced and expressed in I frame to A frame" 
    """
    trans:  nplib.ndarray = nplib.array([0, 0, 0])     #translation represented by vector
    rot:    nplib.ndarray = nplib.array([1, 0, 0, 0])    #rotation represented by quaternion
    
    def create_local(self, t):
        #create a transform that represents a local pose of self w.r.t. t
        #T_IA = T_IB * T_BA  -> T_BA = inv(T_IB) * T_IA
        rot_t = math.quat_inv(t.rot)
        trans = math.rotate(self.trans - t.trans, rot_t) # t_BA = inv(R_IB) * (t_A - t_B)
        rot = math.quat_mul(rot_t, self.rot)             # R_BA = inv(R_IB) * R_IA
        return Transform(trans=trans, rot=rot)
    
    def apply_transform(self, t):
        #apply the transform self to a frame defined by transform t
        trans = self.trans + math.rotate(t.trans, self.rot)
        rot = math.quat_mul(self.rot, t.rot)
        return Transform(trans, rot)
    
    def create_inverse(self):
        #return inverse of the transform
        rot_t = math.quat_inv(self.rot)
        return Transform(trans=-math.rotate(self.trans, rot_t), rot=rot_t)
    
    def to_6dmat(self):
        #this is for coordination matrix of motion vector
        rot_mat = math.rotate(nplib.eye(3), self.rot)
        return nplib.concatenate(
            [
                nplib.concatenate((rot_mat, nplib.zeros((3,3))), axis=1),
                nplib.concatenate((nplib.cross(self.trans, rot_mat), rot_mat), axis=1)
            ],
            axis=0
        )
    
    def to_6dmat_adjoint(self):
        #Featherstone (2.13)
        return self.create_inverse().to_6dmat().T

    
class SpatialVector(NamedTuple):
    ang: nplib.ndarray = nplib.zeros(3)     #angular velocity of the rigid body expressed in frame whose origin is coincidentally the reference point
    lin: nplib.ndarray = nplib.zeros(3)     #linear velocity of the reference point on the body expressed in frame whose origin is coincidentally the reference point

    def __add__(self, v):
        return self.__class__(ang=self.ang+v.ang, lin=self.lin+v.lin)
    
    def __mul__(self, v):
        return self.__class__(ang=self.ang*v, lin=self.lin*v)

    @classmethod
    def from_6darray(cls, arr):
        #simple conversion, no validity check!
        return cls(ang=arr[..., :3], lin=arr[..., 3:6])

    def to_6darray(self):
        return nplib.concatenate([self.ang, self.lin], axis=-1)
        
class Motion(SpatialVector):
    #twist of a rigid body
    #angular velocity of the rigid body expressed in frame whose origin is coincidentally the reference point
    #linear velocity of the reference point on the body expressed in frame whose origin is coincidentally the reference point
    def cross_motion(self, m):
        #Featherstone Eq. (2.31) and (2.33)
        lin = nplib.cross(self.ang, m.lin) + nplib.cross(self.lin, m.ang) 
        ang = nplib.cross(self.ang, m.ang)
        return Motion(ang, lin)
    
    def cross_force(self, f):
        #Featherstone Eq. (2.32) and (2.34)
        force = nplib.cross(self.ang, f.lin)
        trq = nplib.cross(self.ang, f.ang) + nplib.cross(self.lin, f.lin)
        return Force(trq, force)

    def apply_transform(self, t):
        #apply Plucker coordinate transform induced by Transform t
        #Featherstone Eq. (2.24)
        #note the difference to original (2.24) where E is actually the transpose of R given R is the orientation of B relative to A (and hence coordinate transform from B to A)
        rot_t = math.quat_inv(t.rot)
        ang = math.rotate(self.ang, rot_t)
        lin = math.rotate(self.lin - nplib.cross(t.trans, self.ang), rot_t)
        return Motion(ang, lin)
    
    def dot_force(self, f):
        #dot product of motion vector and force vector
        #Featherstone Eq. (2.12)
        return f.lin@self.lin + f.ang@self.ang
    
    def apply_inv_transform(self, t):
        #apply the inverse of Plucker transform specified by Transform t
        #Featherstone (2.26)
        #same note as apply_transform. also refer to Modern Robotics (3.83) and (3.84)
        ang = math.rotate(self.ang, t.rot)
        lin = nplib.cross(t.trans, ang) + math.rotate(self.lin, t.rot)
        return Motion(ang, lin)
    


class Force(SpatialVector):
    def apply_transform(self, t):
        #Featherstone Eq. (2.25)
        rot_t = math.quat_inv(t.rot)
        trq = math.rotate(self.ang - nplib.cross(t.trans, self.lin), rot_t)
        force = math.rotate(self.lin, rot_t)
        return Force(trq, force)
    
    def apply_inv_transform(self, t):
        #Featherstone Eq. (2.27)
        #this is same to _transform_do of Brax, hmmm....
        force = math.rotate(self.lin, t.rot)
        trq = math.rotate(self.ang, t.rot) + nplib.cross(t.trans, force)
        return Force(trq, force)
    
class Inertia(NamedTuple):
    # inertia_mat:    nplib.ndarray = nplib.eye(3)    #3x3 inertia matrix about the origin of LINK frame
    # mass:           float = 1                       #mass scalar
    inertia_tensor: nplib.ndarray = nplib.eye(6)

    def __add__(self, v):
        return self.__class__(inertia_tensor=self.inertia_tensor+v.inertia_tensor)
    
    def __sub__(self, v):
        return self.__class__(inertia_tensor=self.inertia_tensor-v.inertia_tensor)
    
    def __mul__(self, v):
        return self.__class__(inertia_tensor=self.inertia_tensor*v)

    def __rmul__(self, v):
        return self.__mul__(v)

    @classmethod
    def from_it_and_mass(cls, it=nplib.eye(3), mass=1):
        return cls(inertia_tensor=nplib.concatenate(
            [
                nplib.concatenate((it, nplib.zeros((3,3))), axis=1),
                nplib.concatenate((nplib.zeros((3,3)), mass*nplib.eye(3)), axis=1)
            ],
            axis=0
        ))
    
    @classmethod
    def from_dyad(cls, force: Force):
        #from outer product of two spatial force vectors
        #Featherstone (2.64)
        force_array = force.to_6darray()
        return cls(inertia_tensor=nplib.outer(force_array, force_array))
    
    def mul(self, motion: Motion):
        force = motion.to_6darray() @ self.inertia_tensor #note motion can be in a batched format
        return Force.from_6darray(force)
    
    def apply_transform(self, t):
        transform = t.create_inverse().to_6dmat()
        return Inertia(inertia_tensor=transform.T@self.inertia_tensor@transform)
    
    




