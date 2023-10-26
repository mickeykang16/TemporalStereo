import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
import random
random.seed(1)


def getMvsecMetas(root, data_type, num_view):
    assert num_view >=1 and num_view <= 11, num_view

    Metas = []
    sequence_name = 'indoor_flying'
    if 'train' in data_type:
        seqs = [1, 2]
    elif 'test' in data_type:
        seqs = [3]
    else:
        raise TypeError(data_type)

    for seqnum in seqs:
        sequence_dir_path = f'{sequence_name}_{seqnum}'
        assert osp.exists(osp.join(root, sequence_dir_path))
        maxSeqId = len(os.listdir(osp.join(root, sequence_dir_path, 'image0')))

        for iteration in range(0, maxSeqId):
            if iteration < num_view - 1:
                continue

            extrinsicPath = osp.join(sequence_dir_path, 'odometry.txt')
            intrinsicPath = osp.join('calib', 'camchain-imucam-indoor_flying.yaml')

            meta = {
                'extrinsic_path': extrinsicPath,
                'intrinsic_path': intrinsicPath,
            }
            for vid in range(-num_view + 1, 1):
                seqId = iteration + vid
                seqId6 = "{:06d}".format(seqId)

                imgName = f'{seqId6}.npy'
                dispName = f'{seqId6}.png'
                meta[vid] = dict(
                    left_image_path = osp.join(
                        sequence_dir_path, 'num5voxel0', imgName
                    ),
                    right_image_path=osp.join(
                        sequence_dir_path, 'num5voxel1', imgName
                    ),
                )

                meta[vid]['left_disp_path'] = osp.join(
                    sequence_dir_path, 'disparity_image', dispName
                )

            Metas.append(meta)

    return Metas


def build_annoFile(root, save_annotation_root, view_num, phase):
    """
    Build annotation files for MVSEC Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    os.makedirs(save_annotation_root, exist_ok=True)

    Metas = getMvsecMetas(root, phase, view_num)

    for meta in tqdm(Metas):
        for k, v in meta.items():
            if isinstance(v, str):
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
            elif isinstance(v, dict):
                for kk, vv in meta[k].items():
                    assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)


    info_str = f'MVSEC Dataset contains:\n {len(Metas)} {phase} samples'
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, 'view_' + '{}'.format(view_num) + '_' + name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    make_json(name=phase, metas=Metas)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mvsec Data PreProcess.")
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
        default=4,
        help="the number of frames in a view window",
        type=int,
    )

    parser.add_argument(
        "--phase",
        default='train',
        help="sequence",
        type=str,
    )

    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.view_num, args.phase)
