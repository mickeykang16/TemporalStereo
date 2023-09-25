import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
import random
random.seed(1)


def getKITTI2012Metas(root, data_type, num_view):
    assert num_view >=1 and num_view <= 11, num_view

    # trainMetas = []
    # testMetas =[]
    # maxSeqId = 194

    # testIds = []

    # for seqId in range(maxSeqId):
    #     # if seqId in hard_cases and seqId not in testIds:
    #     if seqId not in testIds:
    #         continue
    #     seqId = "{:06d}".format(seqId)
    #     extrinsicPath = osp.join(data_type, 'sequences', seqId, 'orbslam3_pose.txt')
    #     intrinsicPath = osp.join(data_type, 'sequences', seqId, seqId+'_processed.txt')
    #     meta = {
    #         'extrinsic_path': extrinsicPath,
    #         'intrinsic_path': intrinsicPath,
    #     }

    #     centerImgId = 10
    #     for vid in range(-num_view+1, 0+1):
    #         imgNames = "{}_{:02d}.png".format(seqId, vid+centerImgId)
    #         meta[vid] = dict(
    #             left_image_path = osp.join(
    #                 data_type, 'sequences', seqId, 'image_2', imgNames
    #             ),
    #             right_image_path=osp.join(
    #                 data_type, 'sequences', seqId, 'image_3', imgNames
    #             ),
    #         )
    #         if vid == 0 and data_type.find('training') > -1:
    #             meta[vid]['left_disp_path'] = osp.join(
    #                 data_type, 'disp_occ', imgNames

    #             )

    #     if int(seqId) in testIds:
    #         testMetas.append(meta)
    #     else:
    #         trainMetas.append(meta)

    # return trainMetas, testMetas


def getKITTI2015Metas(root, data_type, num_view, split):
    assert num_view >=1 and num_view <= 11, num_view

    trainMetas = []
    testMetas =[]
    maxSeqId = 200

    if split:
        testIds = [i for i in range(165, maxSeqId)]
    else:
        testIds = []
    

    for seqId in range(maxSeqId):
        # if seqId in hard_cases and seqId not in testIds:
        # if seqId not in testIds:
        #     continue
        seqId = "{:06d}".format(seqId)
        extrinsicPath = osp.join(data_type, 'sequences', seqId, 'orbslam3_pose.txt')
        intrinsicPath = osp.join(data_type, 'sequences', seqId, seqId+'.txt')
        meta = {
            'extrinsic_path': extrinsicPath,
            'intrinsic_path': intrinsicPath,
        }
        centerImgId = 10
        for vid in range(-num_view+1, 0+1):
            imgNames = "{}_{:02d}.png".format(seqId, vid+centerImgId)
            meta[vid] = dict(
                left_image_path = osp.join(
                    data_type, 'sequences', seqId, 'image_2', imgNames
                ),
                right_image_path=osp.join(
                    data_type, 'sequences', seqId, 'image_3', imgNames
                ),
            )
            
            # if vid == 0 and data_type.find('training') > -1:
            #     meta[vid]['left_disp_path'] = osp.join(
            #         data_type, 'disp_occ_0', imgNames

            #     )


        if int(seqId) in testIds:
            testMetas.append(meta)
        else:
            trainMetas.append(meta)

    return trainMetas, testMetas


def build_annoFile(root, save_annotation_root, view_num, split):
    """
    Build annotation files for KITTI2012&2015 Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    os.makedirs(save_annotation_root, exist_ok=True)

    totalTrainMetas = []
    totalTestMetas = []

    trainMetas, testMetas = getKITTI2015Metas(root, 'KITTI-2015/training', view_num, split)
    
    for meta in tqdm(trainMetas):
        for k, v in meta.items():
            if isinstance(v, str):
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
            elif isinstance(v, dict):
                for kk, vv in meta[k].items():
                    assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)

    for meta in tqdm(testMetas):
        for k, v in meta.items():
            if isinstance(v, str):
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
            elif isinstance(v, dict):
                for kk, vv in meta[k].items():
                    assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)

    
    totalTrainMetas.extend(trainMetas)
    # totalTestMetas.extend(testMetas)
    # trainMetas, testMetas = getKITTI2012Metas(root, 'KITTI-2012/training', view_num)
    # for meta in tqdm(trainMetas):
    #     for k, v in meta.items():
    #         if isinstance(v, str):
    #             assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
    #         elif isinstance(v, dict):
    #             for kk, vv in meta[k].items():
    #                 assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)

    # for meta in tqdm(testMetas):
    #     for k, v in meta.items():
    #         if isinstance(v, str):
    #             assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
    #         elif isinstance(v, dict):
    #             for kk, vv in meta[k].items():
    #                 assert osp.exists(osp.join(root, vv)), 'trainMetas:{} not exists'.format(vv)

    # totalTrainMetas.extend(trainMetas)
    # totalTestMetas.extend(testMetas)

    info_str = 'KITTI2015&2012 Dataset contains:\n' \
               '    {:5d}   training samples \n' \
               '    {:5d}   testing samples'.format(len(totalTrainMetas), len(totalTestMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, 'view_' + '{}'.format(view_num) + '_' + name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    leading = '_split_12&15' if split else '_12&15'
    make_json(name='train'+leading, metas=totalTrainMetas)
    make_json(name='test'+leading, metas=totalTestMetas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KITTI2015&2012 Data PreProcess.")
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
        help="whether split dataset into train/val",
        type=int,
    )

    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.view_num, args.split>0)
