from utils import flowlib
import torch
import numpy as np

def cputensor2array(image_tensor):
    return image_tensor.numpy().transpose(1, 2, 0)

def cputensor2array3hw(image_tensor):
    return image_tensor.numpy()

def gputensor2array(flow_tensor):
    fl = flow_tensor.cpu().numpy()
    fl = np.swapaxes(np.swapaxes(fl, 0, 1), 1, 2)
    fl_u = fl[:, :, 0]
    fl_v = fl[:, :, 1]
    fl = np.dstack((fl_u, fl_v))
    return fl

def gpudepth2invarray(depth):
    depth_im_t = torch.squeeze(depth)
    depth_im_inv_t = 1 / depth_im_t
    depth_im_inv_t = depth_im_inv_t/torch.max(depth_im_inv_t)
    depth_im_inv_t = depth_im_inv_t.cpu().numpy()
    return depth_im_inv_t

def gpudepth2array(depth):
    depth_im_t = torch.squeeze(depth)
    depth_im_inv_t = depth_im_t
    depth_im_inv_t = depth_im_inv_t/torch.max(depth_im_inv_t)
    depth_im_inv_t = depth_im_inv_t.cpu().numpy()
    return depth_im_inv_t
