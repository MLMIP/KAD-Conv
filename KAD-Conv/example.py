"""Minimal example: use KAD-Conv inside your own network.

    python example.py
"""
import torch
import torch.nn as nn

from kadconv import KADConv


class Block(nn.Module):
    """A simple residual stage: 1x1 projection followed by KAD-Conv."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())
        self.kad = KADConv(c_out)   # keeps the shape: (B, c_out, H, W) -> (B, c_out, H, W)

    def forward(self, x):
        return self.kad(self.proj(x))


if __name__ == '__main__':
    net = nn.Sequential(Block(3, 32), nn.MaxPool2d(2), Block(32, 64))
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    print('output:', tuple(y.shape))
    print('KAD-Conv parameters:', sum(p.numel() for m in net.modules() if isinstance(m, KADConv) for p in m.parameters()))

    # For deployment (CUDA, batch size 1):
    # from kadconv import enable_fast_inference
    # net = enable_fast_inference(net.cuda().eval())
