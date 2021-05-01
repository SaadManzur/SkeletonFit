import os
import h5py
import numpy as np
import cv2 as opencv

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose import draw_skeleton
from utils.camera import load_cameras

SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
INDICES = [0, 1, 2, 3, 6, 7, 8, 13, 17, 18, 19, 25, 26, 27]

class H36MDataset(object):

    def __init__(self, base_path, cam_path):
        super().__init__()
        self._base_path = base_path
        self._cams = load_cameras(cam_path)

    def load_file(self, subj, action, cam_id):

        file_path = os.path.join(self._base_path, "processed", subj, action, "annot.h5")

        pose2d = None
        pose3d_world = None

        with h5py.File(file_path, 'r') as file_:

            pose2d = file_['pose/2d'][:, INDICES, :]
            pose3d_world = file_['pose/3d-world'][:, INDICES, :]

            file_.close()

        return pose3d_world, self._cams[(int(subj[1:]), cam_id)]

