# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use

import functools
from typing import Optional

import torch
import torch.nn.functional as F


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    
    :param quaternions: quaternions as tensor of shape (..., 4), with real part first
    :type quaternions: torch.Tensor 

    :returns: Rotation matrices as tensor of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    :param a: source tensor.
    :type a: torch.Tensor
    :param b: tensor whose signs will be used, of the same shape as a.
    :type b: torch.Tensor

    :returns: Tensor of the same shape as a with the signs of b.
    :rtype: torch.Tensor
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.

    :param x: input tensor
    :type x: torch.Tensor

    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    :param matrix: Rotation matrices as tensor of shape (..., 3, 3).
    :type matrix: torch.Tensor

    :returns: quaternions with real part first, as tensor of shape (..., 4).
    :rtype: torch.Tensor
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    :param axis: Axis label "X" or "Y or "Z".
    :type axis: str
    :param angle: any shape tensor of Euler angles in radians
    :type angle: torch.Tensor

    :returns: Rotation matrices as tensor of shape (..., 3, 3).
    :rtype: torch.Tensor
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    :param euler_angles: Euler angles in radians as tensor of shape (..., 3).
    :type euler_angles: torch.Tensor
    :param convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
    :type convention: str

    :returns: Rotation matrices as tensor of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(axis: str, other_axis: str, data: torch.Tensor, horizontal: bool, tait_bryan: bool) ->  torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    :param axis: Axis label "X" or "Y or "Z" for the angle we are finding.
    :type axis: str
    :param other_axis: Axis label "X" or "Y or "Z" for the middle axis in the convention.
    :type other_axis: str
    :param data: Rotation matrices as tensor of shape (..., 3, 3).
    :type data: torch.Tensor
    :param horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
    :type horizontal: bool
    :param tait_bryan: Whether the first and third axes in the convention differ.
    :type tait_bryan: bool

    :returns: Euler Angles in radians for each matrix in data as a tensor of shape (...).
    :rtype: torch.Tensor
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    """ return the index of axis from string"""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    :param matrix: Rotation matrices as tensor of shape (..., 3, 3).
    :type matrix: torch.Tensor
    :param convention: Convention string of three uppercase letters.
    :type convention: str

    :returns: Euler angles in radians as tensor of shape (..., 3).
    :rtype: torch.Tensor
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    :param n: Number of quaternions in a batch to return.
    :type n: int
    :param dtype: Type to return.
    :type dtype: Optional[torch.dtype]
    :param device: Desired device of returned tensor. Default: uses the current device for the default tensor type.
    :type device: str
    :param requires_grad: Whether the resulting tensor should have the gradient flag set.
    :type requires_grad: bool

    :returns: Quaternions as tensor of shape (N, 4).
    :rtype: torch.Tensor
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    :param n: Number of rotation matrices in a batch to return.
    :type n: int
    :param dtype: Type to return.
    :type dtype: Optional[torch.dtype]
    :param device: Device of returned tensor. Default: if None, uses the current device for the default tensor type.
    :type device: str
    :param requires_grad: Whether the resulting tensor should have the gradient flag set.
    :type requires_grad: bool

    :returns: Rotation matrices as tensor of shape (n, 3, 3).
    :rtype: torch.Tensor
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    :param dtype: Type to return
    :type dtype: Optional[torch.dtype]
    :param device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
    :type device: str
    :param requires_grad: Whether the resulting tensor should have the gradient flag set
    :type requires_grad: bool

    :returns: Rotation matrix as tensor of shape (3, 3).
    :rtype: torch.Tensor
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    :param quaternions: Quaternions with real part first, as tensor of shape (..., 4).
    :type quaternions: torch.Tensor

    :returns: Standardized quaternions as tensor of shape (..., 4).
    :rtype: torch.Tensor
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    :param a: Quaternions as tensor of shape (..., 4), real part first.
    :type a: torch.Tensor
    :param b: Quaternions as tensor of shape (..., 4), real part first.
    :type b: torch.Tensor

    :returns: The product of a and b, a tensor of quaternions shape (..., 4).
    :rtype: torch.Tensor
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    :param a: Quaternions as tensor of shape (..., 4), real part first.
    :type a: torch.Tensor
    :param b: Quaternions as tensor of shape (..., 4), real part first.
    :type b: torch.Tensor

    :returns: The product of a and b, a tensor of quaternions shape (..., 4).
    :rtype: torch.Tensor
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    :param: quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    :type quaternion: torch.Tensor

    :returns: The inverse, a tensor of quaternions of shape (..., 4).
    :rtype: torch.Tensor
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    :param quaternion: Tensor of quaternions, real part first, of shape (..., 4).
    :type quaternion: torch.Tensor
    :param point: Tensor of 3D points of shape (..., 3).
    :type point: torch.Tensor

    :returns: Tensor of rotated points of shape (..., 3).
    :rtype: torch.Tensor
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    :param axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    :type axis_angle: torch.Tensor

    :returns: Rotation matrices as tensor of shape (..., 3, 3).
    :rtype: torch.Tensor
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    :param matrix: Rotation matrices as tensor of shape (..., 3, 3).
    :type matrix: torch.Tensor

    :returns: Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    :rtype: torch.Tensor
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    :param axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    :type axis_angle: torch.Tensor

    :returns: quaternions with real part first, as tensor of shape (..., 4).
    :rtype: torch.Tensor
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    :param quaternions: quaternions with real part first, as tensor of shape (..., 4).
    :type quaternions: torch.Tensor

    :returns: Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    :rtype: torch.Tensor
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].

    :param d6: 6D rotation representation, of size (x, 6)
    :type d6: torch.Tensor

    :returns: batch of rotation matrices of size (x, 3, 3)
    :rtype: torch.Tensor

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.

    :param matrix: batch of rotation matrices of size (x, 3, 3)
    :type matrix: torch.Tensor

    :returns: 6D rotation representation, of size (x, 6)
    :rtype: torch.Tensor

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)