import numpy as np
import torch
import torch.nn.functional as F


def batch_vector_dot_torch(vec1, vec2):
    """
    Batch vector dot product.
    """
    dot = torch.matmul(vec1[..., None, :], vec2[..., None])
    return dot.squeeze(-1)


def normalize_torch(tensor, dim=-1, eps=1e-5):
    """
    Normalize tensor along given dimension.

    Args:
        tensor (Tensor): Tensor.
        dim (int, optional): Dimension to normalize. Defaults to -1.
        eps (float, optional): Small value to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        Tensor: Normalized tensor.
    """
    return F.normalize(tensor, p=2, dim=dim, eps=eps)


def matrix9D_to_6D_torch(mat):
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Simply drop last column and flatten.

    Args:
        mat (Tensor): Input rotation matrix. Shape:(..., 3, 3)

    Returns:
        Tensor: Output matrix. Shape: (..., 6)
    """
    return torch.flatten(mat[..., :-1], start_dim=-2)


def matrix6D_to_9D_torch(mat):
    """
    Convert 6D rotation representation to 3x3 rotation matrix using
    Gram-Schmidt orthogonalization.

    Args:
        mat (Tensor): Input 6D rotation representation. Shape: (..., 6)

    Raises:
        ValueError: Last dimension of mat is not 6.

    Returns:
        Tensor: Output rotation matrix. Shape: (..., 3, 3)
    """
    if mat.shape[-1] != 6:
        raise ValueError(
            "Last two dimension should be 6, got {0}.".format(mat.shape[-1]))

    mat = mat.reshape(*mat.shape[:-1], 3, 2)

    # normalize column 0
    col0 = normalize_torch(mat[..., 0], dim=-1)

    # calculate row 1
    dot_prod = batch_vector_dot_torch(col0, mat[..., 1])
    col1 = normalize_torch(mat[..., 1] - dot_prod * col0, dim=-1)

    col0 = col0.unsqueeze(-1)
    col1 = col1.unsqueeze(-1)

    # calculate last column using cross product
    col2 = torch.cross(col0, col1, dim=-2)
    return torch.cat([col0, col1, col2], dim=-1)


def extract_foot_vel(gpos, foot_joint_idx=(3, 4, 7, 8)):
    # gpos: global position, (batch, seq, joint, 3)

    foot_vel = (
        gpos[..., 1:, foot_joint_idx, :] -
        gpos[..., :-1, foot_joint_idx, :]
    )

    # Pad zero on the first frame for shape consistency
    zeros_shape = list(foot_vel.shape)
    zeros_shape[-3] = 1
    zeros = torch.zeros(
        zeros_shape, device=foot_vel.device, dtype=foot_vel.dtype)
    foot_vel = torch.cat([zeros, foot_vel], dim=-3)

    return foot_vel


def fk_torch(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (Tensor): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (Tensor): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D int Tensor): Parent indices.

    Returns:
        Tensor, Tensor: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = torch.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
            torch.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2)


def euler_to_matrix9D_torch(euler, order="zyx", unit="degrees"):
    """
    Euler angle to 3x3 rotation matrix.

    Args:
        euler (tensor): Euler angle. Shape: (..., 3)
        order (str, optional):
            Euler rotation order AND order of parameter euler.
            E.g. "yxz" means parameter "euler" is (y, x, z) and
            rotation order is z applied first, then x, finally y.
            i.e. p' = YXZp, where p' and p are column vectors.
            Defaults to "zyx".
    """
    dtype = euler.dtype
    device = euler.device

    mat = torch.eye(3, dtype=dtype, device=device)

    if unit == "degrees":
        euler_radians = euler / 180.0 * np.pi
    elif unit == "radians":
        euler_radians = euler
    else:
        raise ValueError("Invalid unit value. Given: {},"
                         "supports: degrees or radians.".format(unit))

    for idx, axis in enumerate(order):
        angle_radians = euler_radians[..., idx:idx + 1]

        # shape: (..., 1)
        sin = torch.sin(angle_radians)
        cos = torch.cos(angle_radians)

        ones = torch.ones(sin.shape, dtype=dtype, device=device)
        zeros = torch.zeros(sin.shape, dtype=dtype, device=device)

        if axis == "x":
            # shape(..., 9)
            rot_mat = torch.cat([
                ones, zeros, zeros,
                zeros, cos, -sin,
                zeros, sin, cos], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        elif axis == "y":
            # shape(..., 9)
            rot_mat = torch.cat([
                cos, zeros, sin,
                zeros, ones, zeros,
                -sin, zeros, cos], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        else:
            # shape(..., 9)
            rot_mat = torch.cat([
                cos, -sin, zeros,
                sin, cos, zeros,
                zeros, zeros, ones], dim=-1)

            # shape(..., 3, 3)
            rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 3, 3)

        mat = torch.matmul(mat, rot_mat)

    return mat


def quat_to_matrix9D_torch(quat):
    """
    Convert quaternion to 3x3 matrix.

    Args:
        quat (tensor): Shape: (..., 4), in the form of (qw, qx, qy, qz)
    """
    quat = normalize_torch(quat)
    qw, qx, qy, qz = quat.unbind(-1)

    qxx = qx * qx
    qyy = qy * qy
    qzz = qz * qz

    qxy = qx * qy
    qxz = qx * qz
    qxw = qx * qw
    qyz = qy * qz
    qyw = qy * qw
    qzw = qz * qw

    mat = [
        1 - 2 * (qyy + qzz), 2 * (qxy - qzw), 2 * (qxz + qyw),
        2 * (qxy + qzw), 1 - 2 * (qxx + qzz), 2 * (qyz - qxw),
        2 * (qxz - qyw), 2 * (qyz + qxw), 1 - 2 * (qxx + qyy)
    ]
    mat = torch.stack(mat, dim=-1)
    mat = mat.reshape(*mat.shape[:-1], 3, 3)

    return mat


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix9D_to_quat_torch(mat):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        mat : Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if mat.size(-1) != 3 or mat.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{mat.shape}.")
    m00 = mat[..., 0, 0]
    m11 = mat[..., 1, 1]
    m22 = mat[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, mat[..., 2, 1] - mat[..., 1, 2])
    o2 = _copysign(y, mat[..., 0, 2] - mat[..., 2, 0])
    o3 = _copysign(z, mat[..., 1, 0] - mat[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def flip_quat_if_necessary(quat, ref_quat):
    """
    Convert quat into closest quaternion form to ref_quat.
    quat is either kept how it is or fliped(times -1).

    Args:
        quat (tensor): Shape: (..., 4), input quaterion.
        ref_quat (tensor): Shape: (..., 4), referece quaternion.
            Has the same shape as quat.
    """
    if quat.shape[-1] != 4:
        raise ValueError("quat expects shape (..., 4), got {}".format(
            tuple(quat.shape)))
    if ref_quat.shape[-1] != 4:
        raise ValueError("ref_quat expects shape (..., 4), got {}".format(
            tuple(ref_quat.shape)))

    delta = ref_quat - quat
    delta_inv = ref_quat + quat
    dist = torch.sum(delta * delta, dim=-1, keepdim=True)
    dist_inv = torch.sum(delta_inv * delta_inv, dim=-1, keepdim=True)

    should_reverse = dist_inv < dist
    signs = torch.ones(*dist.shape, device=dist.device, dtype=dist.dtype)
    signs = signs.masked_fill(should_reverse, -1)

    return quat * signs


def remove_quat_discontinuities(rotations):
    """
    Removing quat discontinuities on the time dimension (removing flips)
    Note: this function cannot be back propagated.

    Args:
        rotations (tensor): Shape: (..., seq, joint, 4)

    Returns:
        tensor: The processed tensor without quaternion inversion.
    """
    with torch.no_grad():
        rots_inv = -rotations

        for i in range(1, rotations.shape[-3]):
            # Compare dot products
            prev_rot = rotations[..., i - 1, :, :]
            curr_rot = rotations[..., i, :, :]
            curr_inv_rot = rots_inv[..., i, :, :]
            replace_mask = (
                torch.sum(prev_rot * curr_rot, dim=-1, keepdim=True) <
                torch.sum(prev_rot * curr_inv_rot, dim=-1, keepdim=True)
            )
            rotations[..., i, :, :] = (
                replace_mask * rots_inv[..., i, :, :] +
                replace_mask.logical_not() * rotations[..., i, :, :]
            )

    return rotations


def quat_slerp_torch(x, y, t):
    """
    Perform spherical linear interpolation (SLERP) between x and y,
    with proportion t.
    """
    dot = torch.sum(x * y, dim=-1)

    neg = dot < 0.0
    dot[neg] = -dot[neg]
    y[neg] = -y[neg]

    t = torch.zeros_like(x[..., 0]) + t
    amount0 = torch.zeros_like(t)
    amount1 = torch.zeros_like(t)

    linear = (1.0 - dot) < 0.01
    omegas = torch.acos(dot[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - t[linear]
    amount0[~linear] = torch.sin((1.0 - t[~linear]) * omegas) / sinoms

    amount1[linear] = t[linear]
    amount1[~linear] = torch.sin(t[~linear] * omegas) / sinoms
    res = amount0[..., None] * x + amount1[..., None] * y

    return res


def to_start_centered_data(positions, rotations, context_len,
                           forward_axis="x", root_idx=0, return_offset=False):
    """
    Center raw data at the start of transition.
    Last context frame is moved to origin (only x and z axis, y unchanged)
    and facing forward_axis.

    Args:
        positions (tensor): (..., seq, joint, 3), raw position data
        rotations (tensor): (..., seq, joint, 3, 3), raw rotation data
        context_len (int): length of context frames
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.
        return_offset (bool): If True, return root position and rotation
            offset as well.
    Returns:
        If return_offset == False:
        (tensor, tensor): (new position, new rotation), shape same as input
        If return_offset == True:
        (tensor, tensor, tensor, tensor):
            (new positions, new rotations, root pos offset, root rot offset)
    """
    pos = positions.clone().detach()
    rot = rotations.clone().detach()
    frame = context_len - 1

    with torch.no_grad():
        # root position on xz axis at last context frame as position offset
        root_pos_offset = pos[..., frame:frame + 1, root_idx, ::2]
        root_pos_offset = root_pos_offset.clone().detach()
        pos = _apply_root_pos_offset(pos, root_pos_offset, root_idx)

        # last context frame root rotation as rotation offset
        root_rot_offset = _get_root_rot_offset_at_frame(
            pos, rot, frame, forward_axis, root_idx)
        pos, rot = _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx)

    if return_offset:
        return pos.detach(), rot.detach(), root_pos_offset, root_rot_offset
    else:
        return pos.detach(), rot.detach()


# def get_model_input(positions, rotations):
#     # positions: (batch, seq, joint, 3)
#     # rotation: (batch, seq, joint, 3, 3)
#     # return (batch, seq, joint*6+3)
#     rot_6d = matrix9D_to_6D_torch(rotations)
#     rot = rot_6d.flatten(start_dim=-2)
#     x = torch.cat([rot, positions[..., 0, :]], dim=-1)
#     return x

def get_model_input(positions, rotations):
    # positions: (batch, seq, joint, 3)
    # rotation: (batch, seq, joint, 3, 3)
    # return (batch, seq, joint*6+3)
    
    # local joint rotation
    local_rot_6d = matrix9D_to_6D_torch(rotations)
    # root position
    root_pos = positions[..., 0:1, :]
    
    # local joint rotation shift
    diff_local_rot_6d = local_rot_6d[..., 1:, :, :] - local_rot_6d[..., :-1, :, :]  # B S-1 22 6
    pad_local_rot_6d = torch.zeros_like(diff_local_rot_6d[..., :1, :, :], device=diff_local_rot_6d.device)
    diff_local_rot_6d = torch.cat([pad_local_rot_6d, diff_local_rot_6d], dim=-3)  # B S 22 6
    
    # root position shift
    diff_root_pos = root_pos[..., 1:, :, :] - root_pos[..., :-1, :, :]  # B S-1 1 3
    pad_root_pos = torch.zeros_like(root_pos[..., :1, :, :], device=diff_root_pos.device)
    root_pos = torch.cat([pad_root_pos, diff_root_pos], dim=-3)  # B S 1 3
    return dict(
        rot_6d=local_rot_6d,
        root_pos=root_pos,
        diff_rot_6d=diff_local_rot_6d,
        diff_root_pos=diff_root_pos
    )


def get_shift_model_input(positions, rotations, return_offset=False):
    # positions: (batch, seq, joint, 3)
    # rotation: (batch, seq, joint, 3, 3)
    # return (batch, seq, joint*6+3)
    rot_6d = matrix9D_to_6D_torch(rotations)  # B S 22 6

    skeleton_offset = positions.clone()  # B S 22 3
    frame0_root_pos = skeleton_offset[..., 0:1, :].clone()  # B S 1 3
    # 将root的位置设置为0
    skeleton_offset[..., 0, :] = 0
    rot_6d_with_skeleton_offset = torch.cat([rot_6d, skeleton_offset], dim=-1)  # B S 22 9

    # 得到root的位置
    root_pos = positions[..., 0:1, :].clone()  # B S 1 3
    # 得到root在帧之间的差
    root_pos = root_pos[..., 1:, :, :] - root_pos[..., :-1, :, :]  # B S-1 1 3
    pad_root_pos = torch.zeros_like(root_pos[..., :1, :, :], device=root_pos.device)
    root_pos = torch.cat([pad_root_pos, root_pos], dim=-3)  # B S 1 3

    if return_offset:
        return rot_6d_with_skeleton_offset, root_pos, frame0_root_pos
    else:
        return rot_6d_with_skeleton_offset, root_pos


def reverse_shift_model_input(root_pos, frame0_root_pos_offset):
    # root_pos: (batch, seq, 1, 3)
    # rotation: (batch, seq, joint, 3, 3)
    # return (batch, seq, joint*6+3)
    root_pos = torch.cumsum(root_pos, dim=-3)  # B S 1 3
    root_pos = root_pos + frame0_root_pos_offset  # B S 1 3
    return root_pos


def reduce_frame_root_shift_and_rotation(
        positions, rotations, base_frame_id=0,
        forward_axis="x", root_idx=0, return_offset=False):
    """
    Center raw data at the start of transition.
    Last context frame is moved to origin (only x and z axis, y unchanged)
    and facing forward_axis.

    Args:
        positions (tensor): (..., seq, joint, 3), raw position data
        rotations (tensor): (..., seq, joint, 3, 3), raw rotation data
        context_len (int): length of context frames
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.
        return_offset (bool): If True, return root position and rotation
            offset as well.
    Returns:
        If return_offset == False:
        (tensor, tensor): (new position, new rotation), shape same as input
        If return_offset == True:
        (tensor, tensor, tensor, tensor):
            (new positions, new rotations, root pos offset, root rot offset)
    """
    pos = positions.clone().detach()
    rot = rotations.clone().detach()

    with torch.no_grad():
        # root position on xz axis at last context frame as position offset
        root_pos_offset = pos[..., base_frame_id:base_frame_id + 1, root_idx, ::2]
        root_pos_offset = root_pos_offset.clone().detach()
        pos = _apply_root_pos_offset(pos, root_pos_offset, root_idx)

        # last context frame root rotation as rotation offset
        root_rot_offset = _get_root_rot_offset_at_frame(
            pos, rot, base_frame_id, forward_axis, root_idx)
        pos, rot = _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx)

    if return_offset:
        return pos.detach(), rot.detach(), root_pos_offset, root_rot_offset
    else:
        return pos.detach(), rot.detach()


def to_mean_centered_data(positions, rotations, context_len,
                          forward_axis="x", root_idx=0, return_offset=False):
    """
    Center raw data at mean position (only x and z axis, y unchanged) and
    face forward_axis at last context frame.

    Robust Motion Inbetweening is using this data processing method.

    Args:
        positions (tensor): (..., seq, joint, 3), raw position data
        rotations (tensor): (..., seq, joint, 3, 3), raw rotation data
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.
        return_offset (bool): If True, return root position and rotation
            offset as well.

    Returns:
        If return_offset == False:
        (tensor, tensor): (new position, new rotation), shape same as input
        If return_offset == True:
        (tensor, tensor, tensor, tensor):
            (new positions, new rotations, root pos offset, root rot offset)
    """
    pos = positions.clone().detach()
    rot = rotations.clone().detach()
    frame = context_len - 1

    with torch.no_grad():
        # mean root position on xz axis as position offset
        root_pos_offset = torch.mean(pos[..., :, root_idx, ::2], -2,
                                     keepdim=True).detach()
        pos = _apply_root_pos_offset(pos, root_pos_offset, root_idx)

        # last context frame root rotation as rotation offset
        root_rot_offset = _get_root_rot_offset_at_frame(
            pos, rot, frame, forward_axis, root_idx)
        pos, rot = _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx)

    if return_offset:
        return pos.detach(), rot.detach(), root_pos_offset, root_rot_offset
    else:
        return pos.detach(), rot.detach()


def apply_root_pos_rot_offset(positions, rotations, root_pos_offset,
                              root_rot_offset, root_idx=0):
    pos = positions.clone().detach()
    rot = rotations.clone().detach()
    pos = _apply_root_pos_offset(pos, root_pos_offset, root_idx)
    pos, rot = _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx)
    return pos.detach(), rot.detach()


def reverse_root_pos_rot_offset(positions, rotations, root_pos_offset,
                                root_rot_offset, root_idx=0):
    pos = positions.clone().detach()
    rot = rotations.clone().detach()
    root_rot_offset = root_rot_offset.transpose(-1, -2).clone().detach()
    pos, rot = _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx)
    pos = _apply_root_pos_offset(pos, -root_pos_offset, root_idx)
    return pos.detach(), rot.detach()


def _get_root_rot_offset_at_frame(
        pos, rot, frame,
        forward_axis="x", root_idx=0):
    """
    Get the rotation offset that makes root joint faces forward_axis at
    given frame.

    Args:
        pos (tensor): (..., seq, joint, 3),
        rot (tensor): (..., seq, joint, 3, 3)
        frame (int): frame index
        forward_axis (str, optional): "x" or "z". Defaults to "x".
        root_idx (int, optional): root joint index. Defaults to 0.

    Raises:
        ValueError: if forward_axis is given an invalid value

    Returns:
        tensor: (..., seq, 3, 3), root rotation offset
    """
    dtype = pos.dtype
    device = pos.device

    root_rot = rot[..., frame:frame + 1, root_idx, :, :]

    # y axis is the local forward axis for root joint
    # We want to make root's local y axis after rotation,
    # align with world forward_axis
    y = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    y = y.repeat(*root_rot.shape[:-2], 1)
    y_rotated = torch.matmul(root_rot, y[..., None]).squeeze(-1)
    y_rotated[..., 1] = 0   # project to xz-plane
    y_rotated = normalize_torch(y_rotated)

    if forward_axis == "x":
        forward = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
    elif forward_axis == "z":
        forward = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    else:
        raise ValueError("forward_axis expect value 'x' or 'z', "
                         "got '{}'.".format(forward_axis))

    forward = forward.repeat(*root_rot.shape[:-2], 1)

    dot = batch_vector_dot_torch(y_rotated, forward)
    cross = torch.cross(y_rotated, forward)
    angle = torch.atan2(batch_vector_dot_torch(cross, y), dot)

    zeros = torch.zeros(angle.shape, dtype=dtype, device=device)
    euler_angle = torch.cat([zeros, angle, zeros], dim=-1)
    root_rot_offset = euler_to_matrix9D_torch(euler_angle, unit="radians")

    return root_rot_offset.detach()


def _apply_root_pos_offset(pos, root_pos_offset, root_idx=0):
    """
    Apply root joint position offset.

    Args:
        pos (tensor): (..., seq, joint, 3)
        root_pos_offset (tensor): (..., seq, 3)
        root_idx (int, optional): Root joint index. Defaults to 0.

    Returns:
        tensor: new pos, shape same as input pos
    """
    pos[..., :, root_idx, ::2] = pos[..., :, root_idx, ::2] - root_pos_offset
    return pos


def _apply_root_rot_offset(pos, rot, root_rot_offset, root_idx=0):
    """
    Apply root rotation offset

    Args:
        pos (tensor): (..., seq, joint, 3)
        rot ([type]): (..., seq, joint, 3, 3)
        root_rot_offset: (..., seq, 3, 3)
        root_idx (int, optional): [description]. Defaults to 0.

    Returns:
        (tensor, tensor): new pos, new rot. Shape same as input.
    """
    rot[..., :, root_idx, :, :] = torch.matmul(
        root_rot_offset, rot[..., :, root_idx, :, :])
    pos[..., :, root_idx, :] = torch.matmul(
        root_rot_offset, pos[..., :, root_idx, :, None]).squeeze(-1)

    return pos, rot
