import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
import lib.utils.object3d as object3d
from PIL import Image
import glob


class KittiTrackingDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = self.split == 'test'
        # self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')
        self.imageset_dir = os.path.join(root_dir, 'KITTI_TRACKING', 'object', 'testing' if is_test else 'training')

        # split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        # self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        # self.num_sample = self.image_idx_list.__len__()
        self.video_idx_list, self.image_count_list, self.image_name_list = self.get_image_count_list(self.imageset_dir)
        # print(self.image_idx_list[0])
        self.image_idx_list = ['%06d' % idx for idx in range(np.sum(self.image_count_list))]
        self.num_sample = np.sum(self.image_count_list)

        self.image_dir = os.path.join(self.imageset_dir, 'image_02')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_02')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

    def convert_idx(self, idx):
        image_count_cumsum = np.cumsum(self.image_count_list)
        # print(image_count_cumsum)
        for i in range(len(image_count_cumsum)):
            if idx >= image_count_cumsum[i]:
                continue

            else:
                if i == 0:
                    frame_id_in_video = idx
                else:
                    frame_id_in_video = idx - image_count_cumsum[i-1]
                frame_id = self.image_name_list[i][frame_id_in_video] # string
                video_id = self.video_idx_list[i] # string
                break

        return video_id, frame_id

    def get_image_count_list(self, imageset_dir):
        video_idx_list = sorted(glob.glob(imageset_dir + '/velodyne/*/'))
        image_name_list = []
        # print(video_idx_list)
        folder_idx_list = []
        image_count_list = []
        for dir_name in video_idx_list:
            folder_idx_list.append(os.path.basename(os.path.normpath(dir_name))) # e.g. 0000
            image_idx_in_video = sorted(glob.glob(dir_name + '/*'))
            image_name_list.append([os.path.splitext(os.path.basename(image_name))[0] for image_name in image_idx_in_video]) # e.g. 000000
            image_count_list.append(len(image_idx_in_video)) # e.g. 154
        
        return folder_idx_list, image_count_list, image_name_list

    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file), img_file
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        video_id, frame_id = self.convert_idx(idx)
        img_file = os.path.join(self.image_dir, video_id, frame_id+'.png')
        # img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file), img_file
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        video_id, frame_id = self.convert_idx(idx)
        lidar_file = os.path.join(self.lidar_dir, video_id, frame_id+'.bin')
        # lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file), lidar_file
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        video_id, _ = self.convert_idx(idx)
        calib_file = os.path.join(self.calib_dir, video_id+'.txt')
        assert os.path.exists(calib_file), calib_file
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        video_id, frame_id = self.convert_idx(idx)
        label_file = os.path.join(self.label_dir, video_id+'.txt')
        assert os.path.exists(label_file), label_file
        # return kitti_utils.get_objects_from_label(label_file)
        return self.get_label_with_frame_id(label_file, frame_id)

    def get_label_with_frame_id(self, label_file, frame_id):
        if isinstance(frame_id, str):
            frame_id = int(frame_id)

        with open(label_file, 'r') as f:
            lines = f.readlines()

        objects = []
        
        for line in lines:
            label = line.strip().split(' ')
            line_frame_id = int(label[0])
            if line_frame_id == frame_id:
                objects.append(object3d.Object3d(" ".join(label[2:])))

        return objects

    def get_road_plane(self, idx):
        assert False, 'WE DONNOT HAVE ROAD PLANE DATA'
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
