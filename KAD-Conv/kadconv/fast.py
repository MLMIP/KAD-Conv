"""Fast inference for KAD-Conv, using the implementation behind the speed results in the paper.

The code in ``acceleration/`` is the KAD-Conv part of KAD-CNet/inference/acceleration: Triton
kernels for LCSA, the branch pooling/upsampling and channel attention, a CUDA kernel and a Triton
kernel for the deformable convolution (with BatchNorm and the branch scale fused into it), and
the scheduling of the three branches. The settings in ``acceleration/a100.json`` are those of
KAD-CNet-S on A100.

Layers run the fast path for CUDA float32 inputs of shape (1, C, H, H) in eval mode; any other
input automatically uses the plain PyTorch forward.
"""
import json
import os
import sys
import types
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent / 'acceleration'
_LAYER_TYPES = {'KADConv', 'KADC4', 'KADC4_m'}   # KADC4 / KADC4_m: the same layer inside KAD-CNet
_DSC_TYPES = {'DSC', 'DCN_RCA'}
_active = False   # set while a KAD-Conv layer runs its fast forward


def _configure(device, build_dir):
    for key, value in json.loads((_DIR / 'a100.json').read_text(encoding='utf-8')).items():
        os.environ.setdefault(key, str(value))
    major, minor = torch.cuda.get_device_capability(device)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{major}.{minor}')
    os.environ.setdefault('KADC_DCN16_SOURCE_DIR', str(_DIR))
    os.environ.setdefault('KADC_DCN16_BUILD_DIR', str(build_dir))
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))


def _fast_ok(x):
    return (x.is_cuda and x.dtype == torch.float32 and x.dim() == 4 and x.shape[0] == 1
            and x.shape[2] == x.shape[3] and x.is_contiguous())


def _wrap_layer(layer):
    fast = layer.forward
    plain = types.MethodType(type(layer).forward, layer)

    def forward(x):
        global _active
        if layer.training or not _fast_ok(x):
            return plain(x)
        _active = True
        try:
            return fast(x)
        finally:
            _active = False

    layer.forward = forward


def _wrap_dsc(dsc):
    # the accelerated DSC also applies the following BatchNorm and branch scale, so it may only
    # be used from inside the accelerated KAD-Conv forward
    fast = dsc.forward
    plain = types.MethodType(type(dsc).forward, dsc)
    dsc.forward = lambda *args, **kwargs: fast(*args, **kwargs) if _active else plain(*args, **kwargs)


def enable_fast_inference(model, build_dir=None, allow_untested_gpu=False):
    """Switch every KAD-Conv layer in ``model`` to the accelerated implementation (in place).

    ``model`` must already be on the GPU and in eval mode. The CUDA kernel is compiled on the
    first call (CUDA toolkit, a C++ compiler and ninja are needed) into ``build_dir``
    (default: ~/.cache/kadconv). Returns ``model``.
    """
    if model.training:
        raise ValueError('call model.eval() before enable_fast_inference()')
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('move the model to a CUDA device first')
    if torch.cuda.get_device_capability(device) != (8, 0) and not allow_untested_gpu:
        raise ValueError('the accelerated kernels were tuned and validated on A100 (SM80); '
                         'pass allow_untested_gpu=True to use them on this GPU, then run check_fast.py')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    layers = [m for m in model.modules() if type(m).__name__ in _LAYER_TYPES]
    if not layers:
        raise ValueError('no KAD-Conv layer found in the model')
    keep = set(map(id, layers))
    for layer in layers:
        keep.update(id(m) for m in layer.modules() if type(m).__name__ in _DSC_TYPES)
    before = {id(m) for m in model.modules() if 'forward' in vars(m)}

    _configure(device, build_dir or Path.home() / '.cache' / 'kadconv')
    import kadc_dcn16_backend
    import kadc_fused_parallel

    kadc_fused_parallel.patch(model)
    kadc_dcn16_backend.install(model)

    # the original patch also targets some YOLO layers by class name; undo anything outside KAD-Conv
    for m in model.modules():
        if id(m) not in keep and id(m) not in before and 'forward' in vars(m):
            del m.forward
    for layer in layers:
        for m in layer.modules():
            if type(m).__name__ in _DSC_TYPES:
                _wrap_dsc(m)
        _wrap_layer(layer)
    return model


def disable_fast_inference(model):
    """Go back to the plain PyTorch forward."""
    for m in model.modules():
        if type(m).__name__ in _LAYER_TYPES | _DSC_TYPES and 'forward' in vars(m):
            del m.forward
    return model
