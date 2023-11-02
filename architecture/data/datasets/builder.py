import sys
import os

sys.path.insert(0, '/home/jaeyoung/ws/TemporalEventStereo')
from architecture.data.datasets import VKITTI2StereoDataset
from architecture.data.datasets import SceneFlowStereoDataset
from architecture.data.datasets import TARTANAIRStereoDataset
from architecture.data.datasets import KITTI2015StereoDataset
from architecture.data.datasets import KITTIRAWStereoDataset
from architecture.data.datasets import MVSECStereoDataset

def build_stereo_dataset(cfg, phase):

    data_root = cfg.DATA_ROOT
    data_type = cfg.TYPE
    annFile = cfg.ANNFILE
    height = cfg.HEIGHT
    width = cfg.WIDTH
    frame_idxs = cfg.FRAME_IDXS
    use_common_intrinsics = cfg.get('USE_COMMON_INTRINSICS', True)
    do_same_lr_transform = cfg.get('DO_SAME_LR_TRANSFORM', True)
    mean = cfg.get('MEAN', (0.485, 0.456, 0.406))
    std = cfg.get('STD', (0.229, 0.224, 0.225))


    is_train = True if phase == 'train' else False

    if 'VKITTI2' in data_type:
        dataset = VKITTI2StereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                       do_same_lr_transform, mean, std)

    elif 'SceneFlow' in data_type:
        dataset = SceneFlowStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'TartanAir' in data_type:
        dataset = TARTANAIRStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'KITTI2015' in data_type:
        dataset = KITTI2015StereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)

    elif 'KITTIRAW' in data_type:
        dataset = KITTIRAWStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)
    elif 'mvsec' in data_type.lower():
        dataset = MVSECStereoDataset(annFile, data_root, height, width, frame_idxs, is_train, use_common_intrinsics,
                                         do_same_lr_transform, mean, std)
    else:
        raise ValueError("invalid data type: {}".format(data_type))

    return dataset


if __name__ == '__main__':
    """
    Test the Stereo Dataset
    """
    import sys
    sys.path.insert(0, '/home/jaeyoung/ws/TemporalEventStereo')
    import matplotlib.pyplot as plt

    from projects.TemporalStereo.config import get_cfg, get_parser
    args = get_parser().parse_args()
    args.config_file = '/home/jaeyoung/ws/TemporalEventStereo/projects/TemporalStereo/configs/mvsec_4.yaml'
    cfg = get_cfg(args)

    dataset = build_stereo_dataset(cfg.DATA.TRAIN, 'train')
    # dataset = build_stereo_dataset(cfg.DATA.VAL, 'val')
    print(dataset)
    breakpoint()
    print("Dataset contains {} items".format(len(dataset)))
    idxs = [0, ] # 100, 1000]
    for i in range(len(dataset)):
        sample = dataset[i]
        
        _include_keys = ['color_aug', 'color', 'disp_gt', 'depth_gt']
        for key in _include_keys:
            if sample.get((key, 0, 'l'), None) is None:
                breakpoint()

    print('Done!')


