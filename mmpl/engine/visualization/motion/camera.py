import h5py
import numpy as np
from pathlib import Path

def load_camera_params(file):
    cam_file = Path(file)
    cam_params = {}
    azimuth = {
        '54138969': 70, '55011271': -70, '58860488': 110, '60457274': -100
    }
    with h5py.File(cam_file) as f:
        subjects = [1, 5, 6, 7, 8, 9, 11]
        for s in subjects:
            cam_params[f'S{s}'] = {}
            for _, params in f[f'subject{s}'].items():
                name = params['Name']
                name = ''.join([chr(c) for c in name])
                val = {}
                val['R'] = np.array(params['R'])
                val['T'] = np.array(params['T'])
                val['c'] = np.array(params['c'])
                val['f'] = np.array(params['f'])
                val['k'] = np.array(params['k'])
                val['p'] = np.array(params['p'])
                val['azimuth'] = azimuth[name]
                cam_params[f'S{s}'][name] = val
    
    return cam_params


def world2camera(pose, R, T):
    """
    Args:
        pose: numpy array with shape (-1, 3)
        R: numpy array with shape (3, 3)
        T: numyp array with shape (3, 1)
    """
    assert pose.shape[-1] == 3
    original_shape = pose.shape 
    pose_world = pose.copy().reshape((-1, 3)).T
    pose_cam = np.matmul(R.T, pose_world - T)
    pose_cam = pose_cam.T.reshape(original_shape)
    return pose_cam


def camera2world(pose, R, T):
    """
    Args:
        pose: numpy array with shape (..., 3)
        R: numpy array with shape (3, 3)
        T: numyp array with shape (3, 1)
    """
    assert pose.shape[-1] == 3
    original_shape = pose.shape
    pose_cam = pose.copy().reshape((-1, 3)).T
    pose_world = np.matmul(R, pose_cam) + T
    pose_world = pose_world.T.reshape(original_shape)
    return pose_world
