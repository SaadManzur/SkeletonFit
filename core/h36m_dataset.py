import os
import h5py
import numpy as np
import cv2 as opencv
from tqdm import tqdm

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose import draw_skeleton
from utils.camera import load_cameras, project_point_radial
from utils.region import find_cube_from_vector

SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
INDICES = [0, 1, 2, 3, 6, 7, 8, 13, 17, 18, 19, 25, 26, 27]
EDGES = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
LIMBS = [[4, 5], [5, 6], [8, 9], [9, 10], [1, 2], [2, 3], [11, 12], [12, 13]]
LEFTS = [4, 5, 6, 8, 9, 10]
RIGHTS = [1, 2, 3, 11, 12, 13]

class H36MDataset(object):

    def __init__(self, base_path, cam_path):
        super().__init__()
        self._base_path = base_path
        self._cams = load_cameras(cam_path)

    def load_file(self, subj, action, cam_id):

        file_path = os.path.join(self._base_path, "processed", subj, action, "annot.h5")

        pose2d = None
        pose3d_world = None

        square_pts = []

        with h5py.File(file_path, 'r') as file_:

            pose3d_world = file_['pose/3d-world'][:, INDICES, :]

            R, T, f, c, k, p, id_ = self._cams[(int(subj[1:]), cam_id)]

            for i in tqdm(range(pose3d_world.shape[0]//4)):

                square_pts.append([])

                for u, v in LIMBS:
                    pts, pts_e = find_cube_from_vector(pose3d_world[i, u, :], pose3d_world[i, v, :], 100)
                    pts = np.array(pts)

                    projected, _, _, _, _ = project_point_radial(pts.reshape(-1, 3), R, T, f, c, k, p)

                    upper = projected[:4, 0]
                    u_min, u_max = np.argmin(upper, 0), np.argmax(upper, 0)
                    lower = projected[4:, 0]
                    l_min, l_max = np.argmin(lower, 0), np.argmax(lower, 0)

                    square_pts[i].append(np.array([
                        projected[u_min], projected[u_max],
                        projected[4+l_min], projected[4+l_max]
                    ]))

            file_.close()

        square_pts = np.array(square_pts)

        return pose3d_world, self._cams[(int(subj[1:]), cam_id)], square_pts

    def visualize(self, subj, action, cam_id, rect_pts=None):

        file_path = os.path.join(self._base_path, "processed", subj, action, "annot.h5")

        video_action = action.replace("-", " ")

        rect_pts_re = rect_pts.reshape((rect_pts.shape[0], -1, 2))

        with h5py.File(file_path, 'r') as file_:

            start_frame = 0

            subj_number = int(subj[1:])

            pose2d = file_['pose/2d'][:, INDICES, :]
            max_frame = np.max(file_['frame'][:])

            for i in range(0, pose2d.shape[0], max_frame):

                if cam_id == file_['camera'][i]:
                    start_frame = i
                    break

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            vid_path = os.path.join(self._base_path, "extracted", subj, "Videos", f"{video_action}.{cam_id}.mp4")

            capture = opencv.VideoCapture(vid_path)

            total_frames = int(capture.get(opencv.CAP_PROP_FRAME_COUNT))

            for i_frame in range(start_frame, start_frame+total_frames):

                ret, frame = capture.read()
                frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)

                frame_pose = pose2d[i_frame, :, :]

                if rect_pts is not None:

                    for limb in range(8):

                        ax.plot([rect_pts[i_frame, limb, 0, 0], rect_pts[i_frame, limb, 1, 0]],
                                [rect_pts[i_frame, limb, 0, 1], rect_pts[i_frame, limb, 1, 1]], color='y')

                        ax.plot([rect_pts[i_frame, limb, 1, 0], rect_pts[i_frame, limb, 3, 0]],
                                [rect_pts[i_frame, limb, 1, 1], rect_pts[i_frame, limb, 3, 1]], color='y')

                        ax.plot([rect_pts[i_frame, limb, 3, 0], rect_pts[i_frame, limb, 2, 0]],
                                [rect_pts[i_frame, limb, 3, 1], rect_pts[i_frame, limb, 2, 1]], color='y')

                        ax.plot([rect_pts[i_frame, limb, 0, 0], rect_pts[i_frame, limb, 2, 0]],
                                [rect_pts[i_frame, limb, 0, 1], rect_pts[i_frame, limb, 2, 1]], color='y')

                draw_skeleton(frame_pose, ax)

                ax.imshow(frame)
                plt.draw()
                plt.pause(1e-8)
                ax.clear()

            file_.close()

    def visualize_cube(self, subj, action):

        file_path = os.path.join(self._base_path, "processed", subj, action, "annot.h5")

        with h5py.File(file_path, 'r') as file_:

            pose3d_world = file_['pose/3d-world'][:, INDICES, :]

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            for i in tqdm(range(0, pose3d_world.shape[0]//4, 10)):

                hip_centered = (pose3d_world[i, :, :] - pose3d_world[i, 0, :])/1000

                draw_skeleton(hip_centered, ax, True, True)

                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
                ax.set_zlim((-1, 1))

                plt.draw()
                plt.pause(1e-8)
                ax.clear()

            file_.close()