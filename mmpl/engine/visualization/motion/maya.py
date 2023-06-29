import os
import json


import pymel.core as pm
import pymel.core.datatypes as dt


def visualize(json_path, joint_r=5, foot_joint_idx=(3, 4, 7, 8)):
    baseName = os.path.basename(json_path).rsplit(".", 1)[0]

    with open(json_path) as fh:
        d = json.load(fh)

    debug = "global_positions" in d

    # create joints
    joints = _create_joints(d["parents"], radius=joint_r, name_prefix="jnt")
    if debug:
        debug_joints = _create_joints(
            d["parents"], radius=joint_r, name_prefix="jnt_debug")

    # create animation
    for frame in range(len(d["positions"])):
        if frame % 10 == 0:
            print(frame)

        for jIdx in range(len(joints)):
            jnt = joints[jIdx]

            # joint local positions
            pos = d["positions"][frame][jIdx]

            # joint local rotation
            # Maya uses row vectors. Matrix multiply to the right.
            # Needs to transpose matrix before multiplication.
            rot = dt.Matrix(d["rotations"][frame][jIdx]).transpose()
            trans = dt.TransformationMatrix(rot)

            # Maya uses centimeters as internal length unit,
            # so whatever pos we set in here is in centimeters.
            trans.setTranslation(pos, space="object")

            jnt.setTransformation(trans)
            pm.setKeyframe(jnt, t=frame)

            # foot contact
            for foot_idx in range(len(foot_joint_idx)):
                jnt = joints[foot_joint_idx[foot_idx]]

                if d["foot_contact"][frame][foot_idx] > 0.5:
                    pm.setKeyframe(jnt, t=frame, at="radius", v=joint_r * 2)
                else:
                    pm.setKeyframe(jnt, t=frame, at="radius", v=joint_r)

            # debug joints
            if debug:
                debug_jnt = debug_joints[jIdx]

                gpos = d["global_positions"][frame][jIdx]
                grot = dt.Matrix(d["global_rotations"][frame][jIdx])
                grot = dt.TransformationMatrix(grot.transpose())
                gq = grot.getRotationQuaternion()

                # set global rotations first then global positions
                debug_jnt.setRotation(gq, space="world")
                debug_jnt.setTranslation(gpos, space="world")

                pm.setKeyframe(debug_jnt, t=frame)
    if debug:
        grp = pm.group(joints[0:1] + debug_joints[0:1], n=baseName)
    else:
        grp = pm.group(joints[0:1], n=baseName)

    pm.xform(grp, piv=(0, 0, 0))


def _create_joints(parents, radius=1, name_prefix="jnt"):
    """
    Create joints based on joint parent indices.

    Args:
        parents (list of int or 1D int ndarray):
            Joint parent indices.
        radius (float, optional): Joint display radius. Defaults to 1.
        name_prefix (str, optional): Joint name prefix. Defaults to "jnt".

    Returns:
        list of pymel.core.nodetypes.Joint: List of joints created in Maya.
    """

    joints = []
    for i in range(len(parents)):
        parentIdx = parents[i]

        if parentIdx == -1:
            parent_joint = None
        else:
            parent_joint = joints[parentIdx]

        pm.select(parent_joint)
        jnt = pm.joint(n="{0}_{1}".format(name_prefix, i), rad=radius)
        joints.append(jnt)

    return joints


if __name__ == '__main__':
    json_path = "/Users/kyanchen/codes/motion/motion_inbetweening/scripts/lafan1_context_model_benchmark_30_0_gt.json"
    visualize(json_path)
