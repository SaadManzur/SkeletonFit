import numpy as np

from matplotlib import pyplot as plt

from utils.pose import draw_skeleton
from utils.pose import get_angles_from_joints
from utils.pose import get_joints_from_angles

HIP    = [ 0., 0., 0. ]
RHIP   = [ 135., 0., 0. ]
RKNEE  = [ 135., -480., 0. ]
RFOOT  = [ 135., -844., 0. ]
LHIP   = [ -135., 0., 0. ]
LKNEE  = [ -135., -480., 0. ]
LFOOT  = [ -135., -844., 0. ]
THORAX = [ 0., 585., 0. ]
LSHLDR = [ -205., 585., 0. ]
LELBOW = [ -539., 585., 0. ]
LWRIST = [ -813., 585., 0. ]
RSHLDR = [ 205., 585., 0. ]
RELBOW = [ 539., 585., 0. ]
RWRIST = [ 813., 585., 0. ]

POSITIONS = [ HIP, RHIP, RKNEE, RFOOT, LHIP, LKNEE, LFOOT, THORAX, LSHLDR, LELBOW, LWRIST, RSHLDR, RELBOW, RWRIST ]

EDGES = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
LEFTS = [4, 5, 6, 8, 9, 10]
RIGHTS = [1, 2, 3, 11, 12, 13]

class Joint(object):

    def __init__(self, position, name):

        self.position = np.array(position)
        self.name = name
        self.parent = None
        self.children = []

    def set_parent(self, parent):

        assert isinstance(parent, Joint)

        self.parent = parent

    def add_child(self, child):

        assert isinstance(child, Joint)

        self.children.append(child)

    def __str__(self):
        
        return self.name

class Skeleton(object):

    def __init__(self):

        self._base = []
        self._names = ["hip", "rhip", "rknee", "rfoot", "lhip", "lknee", "lfoot", "thorax",
                       "lshldr", "lelbow", "lwrist", "rwrist", "relbow", "rshldr"]
        self._positions = POSITIONS
        self._bone_names = []
        self.make_skeleton()

    def make_skeleton(self):

        for i in range(len(self._names)):
            self._base.append(Joint(POSITIONS[i], self._names[i]))

        for u, v in EDGES:
            self._base[u].add_child(self._base[v])
            self._base[v].set_parent(self._base[u])
            self._bone_names.append(f"{self._names[u]}_{self._names[v]}")

    def update_pose(self, joints):

        for i in range(len(self._base)):

            self._base[i].position = joints[i]

    def get_joints_and_parents(self):

        joints = []
        parents = []
        adjacency = []

        for joint in self._base:

            joints.append(joint.position)
            parent = joint.parent
            parent_idx = self._names.index(parent.name) if parent is not None else -1
            parents.append(parent_idx)
            adjacency.append([ self._names.index(child.name) for child in joint.children ])

        return joints, parents, adjacency

    def get_bone_lengths(self):

        bone_lengths = {}

        for u, v in EDGES:
            name = f"{self._names[u]}_{self._names[v]}"
            length = np.linalg.norm(self._base[u].position - self._base[v].position)
            bone_lengths[name] = length

        return bone_lengths

    def set_pose(self, joints):

        if isinstance(joints, list):
            joints = np.array(joints)

        _, parents, adjacency = self.get_joints_and_parents()

        angles = get_angles_from_joints(joints, parents, self._names, self._bone_names)

        joints = get_joints_from_angles(angles, adjacency, self._names, self._bone_names, self.get_bone_lengths())

        self.update_pose(joints)

        self._positions = joints

    def get_pose(self):

        return self._positions

    def visualize(self):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        pose = np.array(self._positions)/1000
        pose -= pose[0, :]

        draw_skeleton(pose, ax, True)

        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        ax.set_zlim((-2, 2))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()
        #plt.draw()
        #plt.pause(0.0001)
        #ax.clear()