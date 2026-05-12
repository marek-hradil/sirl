#math utility functions for manipulating vectors and SO(3)
#simplified for numpy interoperability from https://github.com/google/brax/blob/main/brax/math.py
import os
if 'SIRL_USE_JAX' in os.environ:
  import jax.numpy as nplib
else:
  import numpy as nplib


def rotate(vec, quat):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector or batched vector (b,3)
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat or batched result (b,3)
  """
  # if len(vec.shape) != 1:
  #   raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  #r = 2 * (nplib.dot(u, vec) * u) + (s * s - nplib.dot(u, u)) * vec
  r = 2 * (u*(vec@u)[None].T) + (s * s - nplib.dot(u, u)) * vec
  r = r + 2 * s * nplib.cross(u, vec)
  return r


def inv_rotate(vec, quat):
  """Rotates a vector vec by an inverted unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by the inverse of quat.
  """
  return rotate(vec, quat_inv(quat))


def quat_mul(u, v):
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return nplib.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])



def quat_inv(q):
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * nplib.array([1, -1, -1, -1])


def quat_rot_axis(axis, angle):
  """Provides a quaternion that describes rotating around axis v by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around v by angle
  """
  qx = axis[0] * nplib.sin(angle / 2)
  qy = axis[1] * nplib.sin(angle / 2)
  qz = axis[2] * nplib.sin(angle / 2)
  qw = nplib.cos(angle / 2)
  return nplib.array([qw, qx, qy, qz])



def safe_norm(x):
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.

  Avoids a poorly defined gradient for numpy.linal.norm(0) see
  https://github.com/google/jax/issues/3058 for details
  Args:
    x: A ndarray
    axis: The axis along which to compute the norm

  Returns:
    Norm of the array x.
  """

  is_zero = nplib.allclose(x, 0.0)
  # temporarily swap x with ones if is_zero, then swap back
  x = x + is_zero * 1.0
  n = nplib.linalg.norm(x) * (1.0 - is_zero)
  return n


def normalize(x):
  """Normalizes an array.

  Args:
    x: A ndarray.array
    axis: The axis along which to compute the norm

  Returns:
    A tuple of (normalized array x, the norm).
  """
  norm = safe_norm(x)
  n = x / (norm + 1e-6 * (norm == 0.0))
  return n, norm

def quat_from_mat(m):
  """return quaternion q from its rotation matrix representation mat. assume m is valid SO(3) with trace(m) != -1

  Args:
    m: (3,3) 3x3 rotation matrix

  Returns:
    quat: (4,) quaternion [w, x, y, z]
  """
  trace = nplib.linalg.trace(m)
  r = nplib.sqrt(1.+trace)
  return nplib.array([0.5*r, 0.5*(m[2,1]-m[1,2])/r, 0.5*(m[0,2]-m[2,0])/r, 0.5*(m[1,0]-m[0,1])/r])


