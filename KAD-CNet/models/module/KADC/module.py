"""KAD-Conv and the CSP-ELAN blocks built on it.

Class names and attribute names match the released checkpoints, which store
whole module objects; rename them only together with new checkpoints.
"""
import torch
from torch import nn

from ...common import Conv
from ..ops import DCN_RCA
from ..wtconv.wtconv2d import _ScaleModule


class KADC4(nn.Module):
    """Kernel Adaptive Deformable Convolution (KAD-Conv).

    The input is split into ``n`` branches at relative resolutions
    1, 1/2, ..., 1/2^(n-1). Each branch applies a 1x1 projection, a Deformable
    Spatial Convolution (DSC), BatchNorm and ReLU, followed by a learnable
    branch scale and channel attention. The branches are upsampled back,
    concatenated and added to the input, so the output shape equals the input.
    """

    se_min_hidden = 32

    def __init__(self, dim, n=3):
        super().__init__()
        self.n = n
        fe = [dim - (dim // n) * (n - 1)] + [dim // n] * (n - 1)  # channels per branch
        hidden = [max(self.se_min_hidden, c) for c in fe]

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fe[i], kernel_size=1),
                DCN_RCA(fe[i], fe[i]),
                nn.BatchNorm2d(fe[i]),
                nn.ReLU(inplace=True),
            ) for i in range(n)
        ])
        self.fc = nn.ModuleList([  # channel attention
            nn.Sequential(
                nn.Linear(fe[i], hidden[i]),
                nn.ReLU(inplace=False),
                nn.Linear(hidden[i], fe[i]),
                nn.Sigmoid(),
            ) for i in range(n)
        ])
        self.weight = nn.ModuleList([  # branch selection
            nn.Sequential(_ScaleModule([1, fe[i], 1, 1])) for i in range(n)
        ])

    def forward(self, x):
        features = []
        for i, (conv, fc, w) in enumerate(zip(self.convs, self.fc, self.weight)):
            xi = nn.functional.avg_pool2d(x, kernel_size=int(2 ** i)) if i > 0 else x
            feature = w(conv(xi))
            attention = fc(feature.mean(dim=(2, 3), keepdim=False)).view(x.size(0), -1, 1, 1)
            feature = feature * attention
            if i > 0:
                feature = nn.functional.interpolate(feature, size=x.size()[2:], mode='bilinear', align_corners=False)
            features.append(feature)
        return torch.cat(features, dim=1) + x


class KADC4_m(KADC4):
    """KAD-Conv with a wider channel attention (hidden width at least 64), used by KAD-CNet-S with MGL."""

    se_min_hidden = 64


KAD_CONV_VARIANTS = {'KADC4': KADC4, 'KADC4_m': KADC4_m}


class RepNCSP_KADC(nn.Module):
    """CSP bottleneck whose inner blocks are KAD-Conv layers."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, kadc_variant=None):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        block = KAD_CONV_VARIANTS[kadc_variant] if isinstance(kadc_variant, str) else (kadc_variant or KADC4)
        if block not in KAD_CONV_VARIANTS.values():
            raise ValueError(f'unsupported KAD-Conv variant: {kadc_variant}')
        self.m = nn.Sequential(*(block(c_) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4_KADC(nn.Module):
    """GELAN block (RepNCSPELAN4) with KAD-Conv in both computational branches."""

    def __init__(self, c1, c2, c3, c4, c5=1, kadc_variant=None):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP_KADC(c3 // 2, c4, c5, kadc_variant=kadc_variant), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP_KADC(c4, c4, c5, kadc_variant=kadc_variant), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
