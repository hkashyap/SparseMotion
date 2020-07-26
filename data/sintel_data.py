import torch.utils.data as data
import os
from glob import glob
from skimage import io
import numpy as np
from utils import rotation_lib, io_utils


class SintelDataset(data.Dataset):
    """ A sequence loader where the rgb image files are stored as:
        root/training/final/alley_1/frame_0001.png
        .
        ..
        And depthmaps are stored as:
        root/depth/training/depth/alley_1/frame_0001.dpt
        .
        ..
        And flow fields are stored as:
        root/training/flow/alley_1/frame_0001.flo
        .
        ..
        And camera pose data are stored as:
        .
        ..
        root/depth/training/camdata_left/alley_1/frame_0001.cam
        .
        ..
        The pose data is read out from the files and stored in an array.
    """

    def __init__(self, dataset_dir, drives, transform=None):
        self.dataset_dir = dataset_dir
        self.drives = drives
        self.transform = transform
        self.samples = self.collect_samples()
        self.num_samples = len(self.samples)

    def collect_samples(self):
        samples = []
        id = 0
        for drive in self.drives:
            im_frames = []
            depth_frames = []
            flow_frames = []
            pose_frames = []

            img_dir = os.path.join(self.dataset_dir, 'training', 'final', drive)
            # list all png images in this sequence directory
            imfiles = sorted(glob(img_dir + '/*.png'))
            # save path to all images in a list
            im_frames.extend([f for f in imfiles])

            if len(im_frames) < 2:
                print('Not enough frames in this drive {}, moving on to the next drive!'.format(drive))
                continue

            depth_dir = os.path.join(self.dataset_dir, 'depth', 'training', 'depth', drive)
            # list all png images in this sequence directory
            depthfiles = sorted(glob(depth_dir + '/*.dpt'))
            # save path to all images in a list
            depth_frames.extend([f for f in depthfiles])

            flow_dir = os.path.join(self.dataset_dir, 'training', 'flow', drive)
            # list all png images in this sequence directory
            flowfiles = sorted(glob(flow_dir + '/*.flo'))
            # save path to all images in a list
            flow_frames.extend([f for f in flowfiles])

            # collect the poses for this drive
            pose_dir = os.path.join(self.dataset_dir, 'depth', 'training', 'camdata_left', drive)
            posefiles = sorted(glob(pose_dir + '/*.cam'))
            # save path to all pose files in a list
            pose_frames.extend([f for f in posefiles])

            # drive_times = drive_poses[:, 0]
            # drive_poses = drive_poses[:, 1:].reshape(-1, 4, 4)
            #
            # first_pose_mat = drive_poses[0, :, :]
            # tx = first_pose_mat[0, 3]
            # ty = first_pose_mat[1, 3]
            # tz = first_pose_mat[2, 3]
            # rot = first_pose_mat[:3, :3]
            # rotz, roty, rotx = rotation_lib.mat2euler(rot)
            # self.first_pose_of_drive = np.array([tx, ty, tz, rotx, roty, rotz])
            #
            # derived_pose_file = os.path.join(self.dataset_dir, 'poses_derived', '%.2d.txt' % drive)

            for i in range(len(flow_frames)):
                sample = {'img': im_frames[i],
                          'depth': depth_frames[i],
                          'flow': flow_frames[i],
                          'img_ref': im_frames[i + 1],
                          'pose_t': None,
                          'pose_r': None,
                          'id': id,
                          'K': None}
                id += 1

                # K_target, target_pose_mat = io_utils.cam_read(pose_frames[i])
                # K_ref, ref_pose_mat = io_utils.cam_read(pose_frames[i + 1])
                K_target, target_pose_mat = io_utils.cam_read(pose_frames[i+1])
                K_ref, ref_pose_mat = io_utils.cam_read(pose_frames[i])

                # tx = target_pose_mat[0, 3]
                # ty = target_pose_mat[1, 3]
                # tz = target_pose_mat[2, 3]
                # rot = target_pose_mat[:3, :3]
                # qw, qx, qy, qz = rotation_lib.rot2quat(rot)

                # R = np.linalg.inv(target_pose_mat[:3, :3])
                # t = - target_pose_mat[:3, 3]
                # ref_R = R @ ref_pose_mat[:3, :3]
                # ref_t = R @ (ref_pose_mat[:3, 3] + t)
                # rotz, roty, rotx = rotation_lib.mat2euler(ref_R)

                ref_pose_mat = np.dot(target_pose_mat, np.linalg.inv(ref_pose_mat))
                tx = ref_pose_mat[0, 3]
                ty = ref_pose_mat[1, 3]
                tz = ref_pose_mat[2, 3]
                rot = ref_pose_mat[:3, :3]
                rotz, roty, rotx = rotation_lib.mat2euler(rot)

                sample['pose_t'] = ref_pose_mat[:3, 3]
                sample['pose_r'] = [rotx, roty, rotz]
                sample['K'] = K_target

                samples.append(sample)
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        img = io.imread(sample['img'])
        img_ref = io.imread(sample['img_ref'])
        id = sample['id']
        K = sample['K']
        depth = io_utils.depth_read(sample['depth'])
        flow = io_utils.flow_read_from_flo(sample['flow'])

        tr_images, depth, flow, pose_r, pose_t, K = self.transform([img] + [img_ref], depth, flow, sample['pose_r'],
                                                                   sample['pose_t'], K)
        img = tr_images[0]
        img_ref = tr_images[1]

        sample_read = {'img_t': img,
                       'img_tplus1': img_ref,
                       'depth': depth,
                       'flow': flow,
                       'pose_r': pose_r,
                       'pose_t': pose_t,
                       'id': id,
                       'K': K}
        return sample_read

    def __len__(self):
        return len(self.samples)