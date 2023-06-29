import numpy as np
import json

def batch_vector_dot(vec1, vec2):
    """
    Batch vector dot product.
    """
    dot = np.matmul(vec1[..., None, :], vec2[..., None])
    return dot.squeeze(-1)


def normalize(array, axis=-1, eps=1e-5):
    """
    Normalize ndarray along given axis.

    Args:
        array (ndarray) N-dimensional array.
        axis (int, optional): Axis. Defaults to -1.
        eps (float, optional): Small value to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        ndarray: Normalized N-dimensional array.
    """
    magnitude = np.linalg.norm(array, axis=axis, keepdims=True)
    magnitude[magnitude < eps] = np.inf
    return array / magnitude


def matrix9D_to_6D(mat):
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Simply drop last column and flatten.

    Args:
        mat (ndarray): Input rotation matrix. Shape:(..., 3, 3)

    Returns:
        ndarray: Output matrix. Shape: (..., 6)
    """
    return mat[..., :-1].reshape(*mat.shape[:-2], -1)


def matrix6D_to_9D(mat):
    """
    Convert 6D rotation representation to 3x3 rotation matrix using
    Gram-Schmidt orthogonalization.

    Args:
        mat (ndarray): Input 6D rotation representation. Shape: (..., 6)

    Raises:
        ValueError: Last dimension of mat is not 6.

    Returns:
        ndarray: Output rotation matrix. Shape: (..., 3, 3)
    """
    if mat.shape[-1] != 6:
        raise ValueError(
            "Last two dimension should be 6, got {0}.".format(mat.shape[-1]))

    mat = mat.copy().reshape(*mat.shape[:-1], 3, -1)

    # normalize column 0
    mat[..., 0] = normalize(mat[..., 0], axis=-1)

    # calculate column 1
    dot_prod = batch_vector_dot(mat[..., 0], mat[..., 1])
    mat[..., 1] -= dot_prod * mat[..., 0]
    mat[..., 1] = normalize(mat[..., 1], axis=-1)

    # calculate last column using cross product
    last_col = np.cross(mat[..., 0:1], mat[..., 1:2],
                        axisa=-2, axisb=-2, axisc=-2)

    return np.concatenate([mat, last_col], axis=-1)


def euler_to_matrix9D(euler, order="zyx", unit="degrees"):
    """
    Euler angle to 3x3 rotation matrix.

    Args:
        euler (ndarray): Euler angle. Shape: (..., 3)
        order (str, optional):
            Euler rotation order AND order of parameter euler.
            E.g. "yxz" means parameter "euler" is (y, x, z) and
            rotation order is z applied first, then x, finally y.
            i.e. p' = YXZp, where p' and p are column vectors.
            Defaults to "zyx".
        unit (str, optional):
            Can be either degrees and radians.
            Defaults to degrees.
    """
    mat = np.identity(3)
    if unit == "degrees":
        euler_radians = euler / 180.0 * np.pi
    elif unit == "radians":
        euler_radians = euler
    else:
        raise RuntimeError("Invalid unit: {}".format(unit))

    for idx, axis in enumerate(order):
        angle_radians = euler_radians[..., idx:idx + 1]

        # shape: (..., 1)
        sin = np.sin(angle_radians)
        cos = np.cos(angle_radians)

        # shape(..., 4)
        rot_mat = np.concatenate([cos, sin, sin, cos], axis=-1)
        # shape(..., 2, 2)
        rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 2, 2)

        if axis == "x":
            rot_mat *= np.array([[1, -1], [1, 1]])
            rot_mat = np.insert(rot_mat, 0, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 0, [1, 0, 0], axis=-1)
        elif axis == "y":
            rot_mat *= np.array([[1, 1], [-1, 1]])
            rot_mat = np.insert(rot_mat, 1, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 1, [0, 1, 0], axis=-1)
        else:
            rot_mat *= np.array([[1, -1], [1, 1]])
            rot_mat = np.insert(rot_mat, 2, [0, 0], axis=-2)
            rot_mat = np.insert(rot_mat, 2, [0, 0, 1], axis=-1)

        mat = np.matmul(mat, rot_mat)

    return mat


def fk(lrot, lpos, parents):
    """
    Calculate forward kinematics.

    Args:
        lrot (ndarray): Local rotation of joints. Shape: (..., joints, 3, 3)
        lpos (ndarray): Local position of joints. Shape: (..., joints, 3)
        parents (list of int or 1D ndarray): Parent indices.

    Returns:
        ndarray, ndarray: (global rotation, global position).
            Shape: (..., joints, 3, 3), (..., joints, 3)
    """
    gr = [lrot[..., :1, :, :]]
    gp = [lpos[..., :1, :]]

    for i in range(1, len(parents)):
        gr_parent = gr[parents[i]]
        gp_parent = gp[parents[i]]

        gr_i = np.matmul(gr_parent, lrot[..., i:i + 1, :, :])
        gp_i = gp_parent + \
            np.matmul(gr_parent, lpos[..., i:i + 1, :, None]).squeeze(-1)

        gr.append(gr_i)
        gp.append(gp_i)

    return np.concatenate(gr, axis=-3), np.concatenate(gp, axis=-2)


def extract_feet_contacts(global_pos, lfoot_idx, rfoot_idx,
                          vel_threshold=0.2):
    """
    Extracts binary tensors of feet contacts.

    Args:
        global_pos (ndarray): Global positions of joints.
            Shape: (frames, joints, 3)
        lfoot_idx (int): Left foot joints indices.
        rfoot_idx (int): Right foot joints indices.
        vel_threshold (float, optional): Velocity threshold to consider a
            joint as stationary. Defaults to 0.2.

    Returns:
        ndarray: Binary ndarray indicating left and right foot's contact to
            the ground. Shape: (frames, len(lfoot_idx) + len(rfoot_idx))
    """
    lfoot_vel = np.abs(global_pos[1:, lfoot_idx, :] -
                       global_pos[:-1, lfoot_idx, :])
    rfoot_vel = np.abs(global_pos[1:, rfoot_idx, :] -
                       global_pos[:-1, rfoot_idx, :])

    contacts_l = (np.sum(lfoot_vel, axis=-1) < vel_threshold)
    contacts_r = (np.sum(rfoot_vel, axis=-1) < vel_threshold)

    # Duplicate the last frame for shape consistency
    contacts_l = np.concatenate([contacts_l, contacts_l[-1:]], axis=0)
    contacts_r = np.concatenate([contacts_r, contacts_r[-1:]], axis=0)

    return contacts_l, contacts_r


def extract_root_vel(pos, root_idx=0):
    """
    Extract root joint velocity.

    Args:
        pos (ndarray): Positions of joints. Can be either global or local
            joint positions. Shape: (frames, joints, 3)
        root_idx (int, optional): Root joint index. Defaults to 0.

    Returns:
        ndarray: Root velocity. Shape: (frames, 3)
    """
    root_vel = pos[1:, root_idx] - pos[:-1, root_idx]

    # Pad zero on the first frame for shape consistency
    root_vel = np.concatenate([np.zeros((1, 3)), root_vel], axis=0)

    return root_vel


def extract_root_rot_vel(rot, root_idx=0):
    """
    Extract root rotation velocity.

    Args:
        rot (ndarray): Rotation of joints. Can be either global or local
            joint rotations. Shape: (frame, joints, 3, 3).
        root_idx (int, optional): Root joint index. Defaults to 0.

    Returns:
        ndarray: Root rotation velocity. Shape: (frame, 3, 3)
    """
    # Since rotation matrices are orthogonal matrices, so we take advantage
    # of R^T = R^-1.
    root_rot_vel = np.matmul(
        rot[1:, root_idx],
        rot[:-1, root_idx].transpose(0, 2, 1)
    )

    # Pad identity matrix on the first frame for shape consistency.
    root_rot_vel = np.concatenate(
        [np.identity(3)[None, ...],
         root_rot_vel], axis=0)

    return root_rot_vel


def extract_root_rot_vel_simple(rot, root_idx=0):
    """
    Extract root rotation velocity. Simply substract two rotation matrices
    to obtain velocity.

    Args:
        rot (ndarray): Rotation of joints. Can be either global or local
            joint rotations. Shape: (frame, joints, 3, 3).
        root_idx (int, optional): Root joint index. Defaults to 0.

    Returns:
        ndarray: Root rotation velocity. Shape: (frame, 3, 3)
    """
    # Simple substract rotation matrix of previous frame to obtain velocity
    # rather than multiplying inverse matrix.
    root_rot_vel = rot[1:, root_idx] - rot[:-1, root_idx]

    # Pad zero on the first frame for shape consistency
    root_rot_vel = np.concatenate([np.zeros((1, 3, 3)), root_rot_vel], axis=0)

    return root_rot_vel


def save_data_to_json(json_path, positions, rotations, foot_contact,
                      parents, global_positions=None, global_rotations=None,
                      debug=False):
    """
    Save animation data to json.

    Args:
        json_path (str): JSON file path.
        positions (ndarray or torch.Tensor):
            Joint local positions. Shape: (frames, joints, 3)
        rotations (ndarray or torch.Tensor):
            Joint local rotations. Shape: (frames, joints, 3, 3)
        foot_contact (ndarray):
            Left foot and right foot contact. Shape: (frames, 4)
        parents (1D int ndarray or torch.Tensor):
            Joint parent indices.
        debug (bool, optional): Extra data will be included in the json
            file for debugging purposes. Defaults to false.
    """
    with open(json_path, "w") as fh:
        data = {
            "positions": positions.tolist(),
            "rotations": rotations.tolist(),
            "foot_contact": foot_contact.tolist(),
            "parents": parents.tolist(),
        }

        if debug:
            global_rot, global_pos = fk(rotations, positions, parents)
            if global_positions is None:
                data["global_positions"] = global_pos.tolist()
            else:
                data["global_positions"] = global_positions.tolist()

            if global_rotations is None:
                data["global_rotations"] = global_rot.tolist()
            else:
                data["global_rotations"] = global_rotations.tolist()

        json.dump(data, fh)
