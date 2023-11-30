import numpy as np
import os
from architecture.data.datasets.base import StereoDatasetBase

class MVSECStereoDatasetBase(StereoDatasetBase):
    def __init__(self, annFile, root, height, width, frame_idxs,
                 is_train=False, use_common_intrinsics=False, do_same_lr_transform=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        super(MVSECStereoDatasetBase, self).__init__(annFile, root, height, width, frame_idxs,
                                                     is_train, use_common_intrinsics, do_same_lr_transform,
                                                     mean, std)

        # self.K=np.array([[721.5377/1242,        0,                  609.5593/1242,      0],
        #                  [0,                    721.5377/375,       172.854/375,        0],
        #                  [0,                    0,                  1,                  0],
        #                  [0,                    0,                  0,                  1]])
        # (h, w)
        self.full_resolution = (260, 346)

        self.baseline = 0.0882308457599
        # Below is right according to the data provider
        # refer '/home/user/jaeyoung/ws/event_stereo_ICCV2019/tools/convert_mvsec_with_odom.py'
        # self.baseline = 19.941771812941038

        self.with_depth_gt = False
        self.with_disp_gt = True
        self.with_flow_gt = False
        self.with_pose_gt = True
