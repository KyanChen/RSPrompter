import re
import numpy as np

from . import utils_np


class Animation(object):
    def __init__(self, rotations, positions, offsets, parents, names):
        self.rotations = rotations
        self.positions = positions
        self.offsets = offsets
        self.parents = parents
        self.names = names


CHANNEL_MAP = {
    "Xrotation": "x",
    "Yrotation": "y",
    "Zrotation": "z",
}


def load_bvh_file(bvh_path, start=None, end=None, order=None):
    """
    Load bvh file.

    Args:
        bvh_path (str): File path.
        start (int, optional): Start frame.
            If not specified, use start frame from bvh file.
            Defaults to None.
        end (int, optional): End frame. Defaults to None.
            If not specified, use end frame from bvh file.
        order (str, optional): Euler rotation order (e.g. "xyz", "zxy", etc).
            If not specified, use order from bvh file.
            (e.g. "Zrotation Yrotation Xrotation" = "zyx", which means X is
            applied first, then Y, then Z, i.e. p' = ZYXp, p' and p are
            column vectors.)
            Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        BvhAnimation
    """
    with open(bvh_path) as fh:
        i = 0
        active = -1
        end_site = False

        names = []
        orients = np.array([]).reshape((0, 4))
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in fh:
            if "HIERARCHY" in line:
                continue
            if "MOTION" in line:
                continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "{" in line:
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(
                r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)",
                line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array(
                        [list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if any([p not in CHANNEL_MAP for p in parts]):
                        continue
                    order = "".join([CHANNEL_MAP[p] for p in parts])
                continue

            jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
            if fmatch:
                if end is None:
                    end = int(fmatch.group(1)) - 1
                if start is None:
                    start = 0
                fnum = end - start + 1
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros((fnum, len(orients), 3))
                continue

            fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue

            if start is not None and i < start:
                i += 1
                continue

            if end is not None and i > end:
                i += 1
                continue

            dmatch = line.strip().split(' ')
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * \
                        data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        rotations = utils_np.euler_to_matrix9D(rotations, order=order)

        return Animation(rotations, positions, offsets, parents, names)
