

import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image
import pdb
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('--cityscapes_path', default='data/cityscapes', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    new_cityscapes_path_img = 'data/CS_50_s3/images'   # NOTE need to make these dirs before running
    new_cityscapes_path_lbl = 'data/CS_50_s3/labels'
    seed = 3
    n_labeled_samples = 50

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    # get all CS images
    files = []
    for f in mmcv.scandir(osp.join(cityscapes_path, 'leftImg8bit/train'), '.png', recursive=True):
        files.append(osp.join(cityscapes_path, 'leftImg8bit/train', f))
    assert len(files) == 2975

    # select random 100 images
    idxs = np.arange(len(files))
    idxs = np.random.permutation(idxs)
    labeled_files = (np.array(files)[idxs[:n_labeled_samples]]).tolist()
    
    # copy selected images and corresponding labels in new folder
    for f in labeled_files:
        shutil.copy(f, new_cityscapes_path_img)
        f_split = f.split(os.sep)
        label = osp.join(gt_dir, f_split[-3], f_split[-2], f_split[-1][:-15] + 'gtFine_labelTrainIds.png') # -3: train, -2: <city>, -1: <sample>
        shutil.copy(label, new_cityscapes_path_lbl)



if __name__ == '__main__':
    main()
