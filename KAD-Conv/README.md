# KAD-Conv

A standalone implementation of **Kernel Adaptive Deformable Convolution**, for using the operator in your own networks without the detection code. It depends only on PyTorch and torchvision; the optional fast path also needs Triton.

```
kadconv/
├── kad_conv.py     KADConv, DSC (Deformable Spatial Convolution), LCSA (Low-Cost Spatial Attention)
├── fast.py         enable_fast_inference(): switches KAD-Conv layers to the accelerated implementation
└── acceleration/   the accelerated implementation used for the speed results in the paper
example.py          using KADConv in a small network
check_fast.py       accuracy and timing of the accelerated path on your GPU
```

## Usage

Copy the `kadconv/` folder into your project, or run from this directory.

```python
import torch
from kadconv import KADConv

layer = KADConv(64)                 # input and output have the same shape
x = torch.randn(2, 64, 80, 80)
y = layer(x)                        # (2, 64, 80, 80)
```

`KADConv(dim, n=3, se_min_hidden=32)`:

- `dim`: input/output channels.
- `n`: number of branches, with relative resolutions 1, 1/2, ..., 1/2^(n-1). The paper uses `n=3`.
- `se_min_hidden`: minimum hidden width of the channel attention. It is 32 in KAD-CNet-S; the MGL-trained KAD-CNet-S† uses 64.

Each branch is a 1x1 projection followed by DSC, BatchNorm and ReLU. The branches are weighted by learnable scales and channel attention, upsampled back, concatenated and added to the input. A typical way to use it is to replace the 3x3 convolution inside a residual or CSP block with a 1x1 projection plus `KADConv`. `example.py` shows a minimal network.

`DSC(in_channels, out_channels)` can also be used on its own as a drop-in for a 3x3 modulated deformable convolution (DCNv2) whose mask comes from LCSA.

BatchNorm layers use `eps=1e-3, momentum=0.03`, the YOLO setting KAD-CNet was trained with.

## Loading weights from KAD-CNet

Module and parameter names are the same as in the KAD-CNet code, so every KAD-Conv layer of the released checkpoints loads directly. This needs the KAD-CNet folder on the path to unpickle the checkpoint:

```python
import sys, torch
sys.path.insert(0, '../KAD-CNet')
from kadconv import KADConv

ckpt = torch.load('../KAD-CNet/weights/kad_cnet_s_dior_mgl.pt', map_location='cpu', weights_only=False)
layers = [m for m in ckpt['model'].modules() if type(m).__name__ == 'KADC4_m']   # 'KADC4' for the base model (ema entry)
src = layers[0]
layer = KADConv(src.convs[0][0].in_channels, se_min_hidden=64)
layer.load_state_dict(src.state_dict())
```

We checked that all 42 KAD-Conv layers of both checkpoints load this way and give exactly the same outputs as in KAD-CNet.

## Faster inference

```python
from kadconv import enable_fast_inference

model = enable_fast_inference(model.cuda().eval())
```

This switches every KAD-Conv layer to the implementation we used for the speed results in the paper. `kadconv/acceleration/` is the KAD-Conv part of `KAD-CNet/inference/acceleration`, unchanged except that the layer names of this folder are also recognized. It contains:

- Triton kernels for LCSA, branch pooling and upsampling, and the channel attention;
- a CUDA kernel and a Triton kernel for the deformable convolution, with BatchNorm and the branch scale fused into it;
- the scheduling of the three branches.

The kernel settings in `acceleration/a100.json` are those of KAD-CNet-S on A100.

A layer uses the fast path for CUDA float32 inputs of shape (1, C, H, H) in eval mode. Any other input, for example a batch of several images, automatically uses the plain PyTorch forward. The CUDA kernel is compiled on the first call, which needs the CUDA toolkit (`nvcc`), a C++ compiler and `ninja`. Compiled files go to `~/.cache/kadconv`. `disable_fast_inference(model)` switches back. The same call also works on a KAD-CNet model, but there `KAD-CNet/inference` additionally fuses the rest of the network.

The kernels were tuned and validated on A100; other GPUs need `allow_untested_gpu=True`. Run `python check_fast.py` once (add `--graph` to time with CUDA Graph, as in the paper). It compares the two paths on random layers (max abs error ≤ 1e-3, relative ≤ 2e-6) and prints the time of each. The CUDA kernel is specialized for the feature sizes that occur in KAD-CNet-S at 512 input; other sizes run on the Triton kernel.

## Citation

Please cite our paper if you use KAD-Conv; see the [main README](../README.md).
