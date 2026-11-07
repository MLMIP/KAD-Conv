import torch
from torch import nn


class _ScaleModule(nn.Module):
    """Learnable per-channel scale; used for the branch selection weights in KAD-Conv.

    The module path is kept because the released checkpoints reference it.
    """

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
