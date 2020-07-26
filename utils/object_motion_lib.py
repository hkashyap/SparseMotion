import torch


def find_object_mask_sintel(residue, invalid, th):
    mag_residue = torch.sqrt(residue[0, :, :] ** 2 + residue[1, :, :] ** 2)
    mask = mag_residue > th
    return mask


def find_object_mask_d1(residue, th, eps):
    mask = torch.zeros(residue.size(1), residue.size(2)).cuda()
    mag_residue = torch.sqrt(residue[0, :, :]**2 + residue[1, :, :]**2)
    return mask

