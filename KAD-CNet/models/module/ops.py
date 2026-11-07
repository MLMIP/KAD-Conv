"""Deformable Spatial Convolution (DSC) and its Low-Cost Spatial Attention (LCSA) mask.

In the code, LCSA is ``RCA`` and DSC is ``DCN_RCA``; the names match the released checkpoints.
"""
import math

import torch
import torchvision.ops
from torch import nn


class RCA(nn.Module):
    """Low-Cost Spatial Attention (LCSA).

    Row and column average pooling are added and passed through a separable
    band convolution (1 x k, then k x 1) and a 1x1 projection to ``oup`` channels.
    The output is the mask logit for every kernel location; DSC applies the sigmoid.
    """

    def __init__(self, inp, oup, ratio=1, band_kernel_size=11):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Conv2d(inp, oup, 1),
        )

    def sge(self, x):
        return self.excite(self.pool_h(x) + self.pool_w(x))

    def forward(self, x):
        return self.sge(x)


class DCN_RCA(nn.Module):
    """Deformable Spatial Convolution (DSC): a modulated deformable convolution (DCNv2)
    whose offsets come from a plain convolution and whose modulation mask comes from LCSA."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, mask=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.mask = mask

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=True)
        if mask:
            self.RCA_mask = RCA(in_channels, kernel_size * kernel_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels * self.kernel_size ** 2)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        nn.init.constant_(self.conv_offset.weight, 0.)
        nn.init.constant_(self.conv_offset.bias, 0.)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.RCA_mask(x)) if self.mask else None
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias,
                                             padding=self.padding, mask=mask, stride=self.stride)
