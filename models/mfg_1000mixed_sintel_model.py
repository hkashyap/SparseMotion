# Author: Hirak J. Kashyap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def conv_relu_2x(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


def conv_relu(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride),
        nn.ReLU(inplace=True)
    )


class MFG1000Mixed(nn.Module):

    def __init__(self, h, w):
        super(MFG1000Mixed, self).__init__()

        conv_channels = [32, 64, 128, 256, 512, 1000]
        self.mf_h = h
        self.mf_w = w

        self.conv0 = conv_relu_2x(2, conv_channels[0], kernel_size=5, stride=2)
        self.conv1 = conv_relu_2x(conv_channels[0], conv_channels[1], kernel_size=5, stride=2)
        self.conv2 = conv_relu(conv_channels[1], conv_channels[2], kernel_size=5, stride=2)
        self.conv3 = conv_relu(conv_channels[2], conv_channels[3], kernel_size=3, stride=2)
        self.conv4 = conv_relu(conv_channels[3], conv_channels[4], kernel_size=3, stride=2)
        self.conv5 = conv_relu(conv_channels[4], conv_channels[5], kernel_size=3, stride=1)

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        mf_gen_neurons = (self.mf_h // 2) * (self.mf_w // 2) * 2
        self.t_mf_gen = nn.Linear(conv_channels[5], mf_gen_neurons, bias=False)
        self.r_mf_gen = nn.Linear(conv_channels[5], mf_gen_neurons, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input):
        out_conv0 = self.conv0(input)
        out_conv1 = self.conv1(out_conv0)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5_f = self.conv5(out_conv4)

        out_conv5 = self.global_pool(out_conv5_f)

        t_mf_downscaled = self.t_mf_gen(torch.squeeze(out_conv5))
        r_mf_downscaled = self.r_mf_gen(torch.squeeze(out_conv5))

        t_mf_downscaled = t_mf_downscaled.reshape(-1, 2, self.mf_h // 2, self.mf_w // 2)
        r_mf_downscaled = r_mf_downscaled.reshape(-1, 2, self.mf_h // 2, self.mf_w // 2)

        t_mf = F.interpolate(t_mf_downscaled, scale_factor=2, mode='bilinear', align_corners=True)
        r_mf = F.interpolate(r_mf_downscaled, scale_factor=2, mode='bilinear', align_corners=True)

        return {'t_mf': t_mf, 'r_mf': r_mf, 'out_conv5': out_conv5}
