import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
import random
random.seed(1)

# For test we use same frames as
# "Realtime Time Synchronized Event-based Stereo"
# by Alex Zhu et al. for consistency of test results.
FRAMES_FILTER_FOR_TEST = {
    'indoor_flying': {
        1: list(range(140, 1201)),
        2: list(range(120, 1421)),
        3: list(range(73, 1616)),
        4: list(range(190, 290))
    }
}

# For the training we use different frames, since we found
# that frames recomended by "Realtime Time Synchronized
# Event-based Stereo" by Alex Zhu include some still frames.
FRAMES_FILTER_FOR_TRAINING = {
    'indoor_flying': {
        1: list(range(80, 1260)),
        2: list(range(160, 1580)),
        3: list(range(125, 1815)),
        4: list(range(190, 290))
    }
}
NUM_VALIDATION = 200

def getMvsecMetas(root, data_type, num_view, voxel_size, split):
    assert num_view >=1 and num_view <= 11, num_view

    Metas = []
    sequence_name = 'indoor_flying'
    

    if 'train' in data_type:
        if split == 1:
            seqs = [2, 3]
        elif split == 2:
            seqs = [1, 3]
        elif split == 3:
            seqs = [1, 2]
        else:
            return
        frame_filter = FRAMES_FILTER_FOR_TRAINING
    elif 'test' in data_type or 'val' in data_type:
        seqs = [split]
        frame_filter = FRAMES_FILTER_FOR_TEST
    else:
        raise TypeError(data_type)

    for seqnum in seqs:
        sequence_dir_path = f'{sequence_name}_{seqnum}'
        assert osp.exists(osp.join(root, sequence_dir_path))
        maxSeqId = len(os.listdir(osp.join(root, sequence_dir_path, 'image0')))

        seq_start = frame_filter['indoor_flying'][seqnum][0]
        seq_end = frame_filter['indoor_flying'][seqnum][-1]

        for iteration in range(0, maxSeqId):
            if iteration < seq_start + num_view -1:
                continue
            elif iteration > seq_end:
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
                        sequence_dir_path, f'num{voxel_size}voxel0', imgName
                    ),
                    right_image_path=osp.join(
                        sequence_dir_path, f'num{voxel_size}voxel1', imgName
                    ),
                )

                meta[vid]['left_disp_path'] = osp.join(
                    sequence_dir_path, 'disparity_image', dispName
                )

            Metas.append(meta)
    
    # if 'val' in data_type:
    #     Metas = Metas[:NUM_VALIDATION]
    # elif 'test' in data_type:
    #     Metas = Metas[NUM_VALIDATION:]
    return Metas


def build_annoFile(root, save_annotation_root, view_num, phase, voxel, split, shuffle):
    """
    Build annotation files for MVSEC Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    os.makedirs(save_annotation_root, exist_ok=True)

    Metas = getMvsecMetas(root, phase, view_num, voxel, split)

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
        filepath = osp.join(save_annotation_root, 'view_' + '{}'.format(view_num) + 
                '_' + name + '_v{}'.format(voxel) + '_split{}'.format(split) 
                + '{}'.format('_shuffle' if shuffle and 'test' in phase else '') +'.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)
    if 'train' in phase:
        make_json(name=phase, metas=Metas)
    elif 'test' in phase:
        if shuffle:
            random.shuffle(Metas)
            print("Shuffle Data before Validation-Test Split")
        make_json(name='val', metas=Metas[:NUM_VALIDATION])
        make_json(name='test', metas=Metas[NUM_VALIDATION:])
    
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
        choices=['test', 'train']
    )
    parser.add_argument(
        "--voxel",
        default=5,
        type=int,
        choices=[5, 7, 10]
    )
    parser.add_argument(
        "--split",
        default=1,
        type=int,
        choices=[1, 2, 3]
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        type=bool
    )
    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.view_num, args.phase, args.voxel, args.split, args.shuffle)
