import torch
from skimage import transform
import cv2
import numpy as np

"""Set of transformations on the data"""


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, depth, flow, pose_r, pose_t, K):
        for t in self.transforms:
            images, depth, flow, pose_r, pose_t, K = t(images, depth, flow, pose_r, pose_t, K)
        return images, depth, flow, pose_r, pose_t, K


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        out_height : int (height required after scaling)
        out_width : int
    """

    def __init__(self, out_height, out_width):
        assert isinstance(out_height, int)
        assert isinstance(out_width, int)
        self.out_height = int(out_height)
        self.out_width = int(out_width)

    def __call__(self, images, depth, flow, pose_r, pose_t, K):
        assert K is not None
        output_K = np.copy(K)

        if self.out_height == depth.shape[0] and self.out_width == depth.shape[1]:
            pass
        else:
            in_h, in_w = depth.shape
            x_scaling = self.out_width / in_w
            y_scaling = self.out_height / in_h

            output_K[0] *= x_scaling
            output_K[1] *= y_scaling

            images = [transform.resize(image, (self.out_height, self.out_width), preserve_range=True, mode='constant')
                      for image in images]
            depth = cv2.resize(depth, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)
            flow = cv2.resize(flow, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)

            flow[:, :, 0] = flow[:, :, 0] * x_scaling
            flow[:, :, 1] = flow[:, :, 1] * y_scaling

        return images, depth, flow, pose_r, pose_t, output_K


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to
        a list of torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, images, depth, flow, pose_r, pose_t, K):
        tensored_images = []

        for image in images:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose(2, 0, 1)
            tensored_images.append(torch.from_numpy(image).float() / 255)

        # numpy flow: H x W x C
        # torch flow: C X H X W
        flow = flow.transpose(2, 0, 1)
        flow_tensored = torch.from_numpy(flow).float()

        # numpy depth: H x W
        # torch depth: 1 X H X W
        depth = depth.reshape(1, depth.shape[0], depth.shape[1])
        depth_tensored = torch.from_numpy(depth).float()

        pose_r_tensored = torch.FloatTensor(pose_r)
        pose_t_tensored = torch.FloatTensor(pose_t)
        return tensored_images, depth_tensored, flow_tensored, pose_r_tensored, pose_t_tensored, K


# Rescale flow function
def rescale_flow(flow, out_h, out_w):
    x_scaling = out_w / flow.size(2)
    y_scaling = out_h / flow.size(1)
    flow = flow.cpu().numpy().transpose(1, 2, 0)

    flow = cv2.resize(flow, (out_w, out_h), interpolation=cv2.INTER_AREA)

    flow[:, :, 0] = flow[:, :, 0] * x_scaling
    flow[:, :, 1] = flow[:, :, 1] * y_scaling
    return torch.tensor(flow.transpose(2, 0, 1))


# Rescale image tensor
def rescale_img(img, out_h, out_w):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = transform.resize(img, (out_h, out_w), preserve_range=True, mode='constant')
    return torch.tensor(img.transpose(2, 0, 1))