import argparse
import numpy as np

from core.skeleton import Skeleton
from core.h36m_dataset import H36MDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_2d", action="store_true", default=False)
    parser.add_argument("--visualize_3d", action="store_true", default=False)
    parser.add_argument("--dpath", type=str, default="/home/saad/Personal/Research/Dataset/H36M/h36m-fetch")
    parser.add_argument("--cpath", type=str, default="/home/saad/Personal/Research/Experiments/MartinezBaseline/data/h36m/cameras.h5")
    parser.add_argument("--subj", type=str, default="S1")
    parser.add_argument("--act", type=str, default="Directions-1")
    parser.add_argument("--cam", type=int, default=54138969)
    parser.add_argument("--load_cube", action="store_true", default=False)
    parser.add_argument("--skeleton", action="store_true", default=False)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    dataset = H36MDataset(args.dpath, args.cpath)

    if args.visualize_2d:
        pose3d, cam, pts = dataset.load_file(args.subj, args.act, args.cam)
        dataset.visualize(args.subj, args.act, args.cam, pts)

    elif args.visualize_3d:
        dataset.visualize_cube(args.subj, args.act)

    if args.skeleton:

        pose3d, cam, pts = dataset.load_file(args.subj, args.act, args.cam)
        idx = np.random.randint(low=0, high=pose3d.shape[0])

        skeleton = Skeleton()
        skeleton.set_pose(pose3d[idx, :, :])
        skeleton.visualize()