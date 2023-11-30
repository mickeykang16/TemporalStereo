import numpy as np
import cv2
import yaml

def np_loader(image_path):
    voxel = np.load(image_path)
    assert voxel.ndim == 3
    # num_voxel_channel = voxel.shape[0]
    # vi = np.random.choice(num_voxel_channel, num_voxel_channel, replace=False)
    # return voxel[vi, : :].copy()
    # return voxel[::-1, :, :].copy()
    return voxel

def read_mvsec_intrinsic(intrinsic_fn):
    data = {}
    resolution = None
    with open(intrinsic_fn, 'r') as fp:
        yml = yaml.load(fp, Loader=yaml.FullLoader)
        # for line in fp.readlines():
        #     if line.find('P_rect_02') > -1:
                # line = line[11:]
                # values = line.rstrip().split(' ')
        values = yml['cam0']['intrinsics']
        camera = '02'
        key = 'K_cam{}'.format(camera)
        inv_key = 'inv_K_cam{}'.format(int(camera))
        matrix = np.array([float(values[i]) for i in range(len(values))])
        K = np.eye(4)
        K[0, 0] = matrix[0]
        K[1, 1] = matrix[1]
        K[0, 2] = matrix[2]
        K[1, 2] = matrix[3]
        item = {
            key: K,
            inv_key: np.linalg.pinv(K)
        }
        data['{}'.format(camera)] = item
            # S_rect_02: 1.242000e+03 3.750000e+02
            # if line.find('S_rect_02') > -1:
        # line = line[11:]
        # values = line.rstrip().split(' ')
        values = yml['cam0']['resolution']
        w, h = float(values[0]), float(values[1])
        resolution = (h, w)

    return data, resolution


def read_mvsec_extrinsic(extrinsic_fn):
    """
    We assume the extrinsic is obtained by ORBSLAM3,
    The pose of it is from camera to world, but we need world to camera, so we have to inverse it
    """
    lineid = 0
    data = {}
    with open(extrinsic_fn, 'r') as fp:
        for line in fp.readlines():
            values = line.rstrip().split(' ')
            frame = '{:06d}'.format(lineid)
            camera = '02'
            key = 'T_cam{}'.format(camera)
            inv_key = 'inv_T_cam{}'.format(camera)
            matrix = np.array([float(values[i]) for i in range(len(values))])
            matrix = matrix.reshape(3, 4)
            matrix = np.concatenate((matrix, np.array([[0, 0, 0, 1]])), axis=0)
            # matrix = np.eye(4)
            item = {
                key: matrix,
                inv_key: np.linalg.pinv(matrix),
                # inv_key: matrix,
                # key: np.linalg.pinv(matrix),
            }
            data['Frame{}:{}'.format(frame, camera)] = item
            lineid += 1

    return data


def read_mvsec_png_disparity(disp_fn, K=np.zeros((4, 4)), b=0.1):
    disp = cv2.imread(disp_fn, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR)
    disp = disp / 7.0
    # uint8
    valid_mask = disp > 0
    # disp = disp / 256.0

    f = K[0, 0]
    # b = 0.0882308457599 # meter

    depth = b * f / (disp + 1e-12)
    depth = depth * valid_mask

    return depth, disp

def read_kitti_png_depth(depth_fn, K=np.array([[721.5377,       0,              609.5593,    0],
                                               [0,              721.5377,       172.854,     0],
                                               [0,              0,              1,           0],
                                               [0,              0,              0,           1]])):

    raise NotImplementedError

def read_kitti_png_flow(flow_fn):
    raise NotImplementedError
