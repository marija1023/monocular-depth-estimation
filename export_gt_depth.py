import os

import numpy as np
import PIL.Image as pil

from utils import *
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    data_path = '/content/drive/My Drive/ML/Projekat/monocular-depth-estimation/data'

    split_folder = os.path.join(os.path.dirname(__file__), "splits")
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths")

    gt_depths = []

    for line in lines:
        folder = line[:line.find('/')]
        frame = line[line.rfind('/')+1 : line.rfind('\.')-3]
        frame_id = int(frame)

        calib_dir = os.path.join(data_path, "calib")
        velo_filename = os.path.join(data_path, "photos", folder, "velodyne_points/data", "{:010d}.bin".format(frame_id))
        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to split")

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()