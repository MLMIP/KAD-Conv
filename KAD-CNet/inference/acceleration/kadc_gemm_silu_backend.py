import os
import types
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _load():
    global _EXTENSION
    if _EXTENSION is None:
        source = Path(__file__).resolve().parent
        build = Path(os.environ.get(
            "KADC_GEMM_SILU_BUILD", "/dev/shm/kadc_gemm_silu_build"
        ))
        build.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        _EXTENSION = load(
            name="kadc_gemm_silu_sm75",
            sources=[str(source / "kadc_gemm_silu.cpp"),
                     str(source / "kadc_gemm_silu.cu")],
            build_directory=str(build),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=os.environ.get("KADC_GEMM_SILU_VERBOSE", "0") == "1",
        )
    return _EXTENSION


def _forward(self, value):
    return self._sm75_gemm_extension.forward(
        value, self.conv.weight, self.conv.bias
    )


def _native_conv_bias_silu(self, value):
    conv = self.conv
    result = torch.nn.functional.conv2d(
        value, conv.weight, None, conv.stride, conv.padding,
        conv.dilation, conv.groups,
    )
    return self._sm75_gemm_extension.bias_silu(result, conv.bias)


def install(model):
    extension = _load()
    if os.environ.get("KADC_NATIVE_ADD_BIAS_SILU", "0") == "1":
        import kadc_fused_nsmall
        native_add_threads = int(os.environ.get("KADC_NATIVE_ADD_THREADS", "256"))

        def native_add_bias_silu(lhs, rhs, bias):
            return extension.add_bias_silu(lhs, rhs, bias, native_add_threads)

        kadc_fused_nsmall.add_bias_silu = native_add_bias_silu
        print(
            f"kadc_native_add_bias_silu=installed threads={native_add_threads}",
            flush=True,
        )
    if os.environ.get("KADC_NATIVE_ADOWN_POOL", "0") == "1":
        import kadc_fused_nsmall

        kadc_fused_nsmall.avg_pool2x2_s1 = extension.avg_pool2x2_s1
        print("kadc_native_adown_pool=installed", flush=True)
    if os.environ.get("KADC_NATIVE_BIAS_SILU", "0") == "1":
        standard_only = os.environ.get("KADC_NATIVE_BIAS_STANDARD_ONLY", "0") == "1"
        if standard_only:
            from kadc_fused_sm75 import conv_fwd_bias_silu as standard_forward
        patched_epilogues = 0
        for module in model.modules():
            if (type(module).__name__ == "Conv"
                    and isinstance(getattr(module, "act", None), torch.nn.SiLU)
                    and getattr(getattr(module, "conv", None), "bias", None) is not None
                    and (not standard_only
                         or getattr(getattr(module, "forward", None), "__func__", None)
                         is standard_forward)):
                module._sm75_gemm_extension = extension
                module.forward = types.MethodType(_native_conv_bias_silu, module)
                patched_epilogues += 1
        print(
            f"kadc_native_bias_silu=installed modules={patched_epilogues} "
            f"standard_only={standard_only}",
            flush=True,
        )
    if os.environ.get("KADC_GEMM_SILU", "0") != "1":
        return model
    shapes = set()
    for item in os.environ.get(
        "KADC_GEMM_SILU_SHAPES", "256x128x32"
    ).split(","):
        if item:
            cin, cout, height = (int(value) for value in item.split("x"))
            shapes.add((cin, cout, height))
    patched = 0
    for module in model.modules():
        if type(module).__name__ != "Conv":
            continue
        conv = getattr(module, "conv", None)
        if (conv is None or conv.kernel_size != (1, 1)
                or conv.stride != (1, 1) or conv.padding != (0, 0)
                or conv.groups != 1 or conv.bias is None):
            continue
        # The spatial dimension is discovered on the first call by the input,
        # so patch only channel signatures listed by the experiment.
        signatures = {(cin, cout) for cin, cout, _ in shapes}
        if (conv.in_channels, conv.out_channels) not in signatures:
            continue

        allowed_heights = {
            height for cin, cout, height in shapes
            if cin == conv.in_channels and cout == conv.out_channels
        }

        def dispatch(self, value, allowed=allowed_heights):
            if value.shape[2] not in allowed:
                from kadc_fused_sm75 import conv_fwd_bias_silu
                return conv_fwd_bias_silu(self, value)
            return _forward(self, value)

        module._sm75_gemm_extension = extension
        module.forward = types.MethodType(dispatch, module)
        patched += 1
    print(f"kadc_gemm_silu=installed shapes={shapes} modules={patched}", flush=True)
    return model
