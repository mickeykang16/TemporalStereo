import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
import random
random.seed(1)


def getKITTIRAWMetas(root, num_view):
    assert num_view >=1 and num_view <= 11, num_view

    trainMetas = []
    scenes = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    _exclude = [] # ['2011_09_29_drive_0004_sync', '2011_09_29_drive_0071_sync', '2011_10_03_drive_0047_sync']

    for sc in scenes:
        seqIds = os.listdir(osp.join(root, 'rawdata', sc))
        seqIds = [name for name in seqIds if name.endswith('_sync') and name not in _exclude]
        for seqId in seqIds:
            Images = os.listdir(osp.join(root, 'rawdata', sc, seqId, 'image_02/data'))
            Images = [name for name in Images if name.endswith('.png')]
            Images.sort()
            for idx in range(num_view - 1, len(Images)):
                meta = {
                    'intrinsic_path': osp.join('rawdata', sc, 'calib_cam_to_cam.txt'),
                    'extrinsic_path': osp.join('oxt_extrinsic', sc, seqId, 'image_02', 'poses.txt')
                }
                for vid in range(-num_view + 1, 0 + 1):
                    fileNmae = Images[idx + vid]
                    meta[vid] = dict(
                        left_image_path = osp.join('rawdata', sc, seqId, 'image_02/data/', fileNmae),
                        right_image_path = osp.join('rawdata', sc, seqId, 'image_03/data/', fileNmae),
                        left_disp_path = osp.join('rawdata', sc, seqId, 'leastereo/data/', fileNmae),
                    )
                trainMetas.append(meta)

    return trainMetas


def build_annoFile(root, save_annotation_root, view_num, split):
    """
    Build annotation files for KITTI Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    os.makedirs(save_annotation_root, exist_ok=True)

    trainMetas = getKITTIRAWMetas(root, view_num)
    for meta in tqdm(trainMetas):
        for k, v in meta.items():
            if isinstance(v, str):
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
            elif isinstance(v, dict):
                for kk, vv in meta[k].items():
                    assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)

    info_str = 'KITTI Dataset contains:\n' \
               '    {:5d}   training samples \n'.format(len(trainMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, 'view_' + '{}'.format(view_num) + '_' + name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    leading = '_all'
    make_json(name='raw_pseudo_train'+leading, metas=trainMetas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KITTIRAW Data PreProcess.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="root of data",
        type=str,
    )
    parser.add_argument(
        "--save-annotation-root",
        default='./',
        help="save root of generated annotation file",
        type=str,
    )
    parser.add_argument(
        "--view-num",
        default=2,
        help="the number of frames in a view window",
        type=int,
    )

    parser.add_argument(
        "--split",
        default=False,
        help="whether split dataset into train/val, if >0 yes, <=0 no",
        type=int,
    )

    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.view_num, args.split>0)
