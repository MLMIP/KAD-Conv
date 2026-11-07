"""Low-overhead inference candidates layered on top of deform_serial.

Optional environment switches:
  KADC_CSP_PARALLEL=1
  KADC_HEAD_MODE=fast | scale3 | branch6
"""

import os
import types

import torch

# The deploy-safe default must be established before kadc_fused_parallel is
# imported because that module reads its scheduling mode at import time.
os.environ.setdefault("KADC_PARALLEL_MODE", "deform_serial")
from kadc_fused_parallel import patch as patch_deform_serial
from kadc_fused_sm75 import (
    add3_bias_silu,
    add4_bias_silu,
    add_bias_silu,
    adown_pool_split,
    avg_pool2x2_s1,
    bias_silu_inplace,
    cat2_contiguous,
    cat4_contiguous,
    conv1x1_two_input_bias_silu,
    max_pool3x3_s2p1,
)
from utils.tal.anchor_generator import dist2bbox, make_anchors


_STREAMS = {}
CSP_PARALLEL = os.environ.get("KADC_CSP_PARALLEL", "1") == "1"
CSP_MAIN_FIRST = os.environ.get("KADC_CSP_MAIN_FIRST", "0") == "1"
CSP_SPLIT_PROJ = os.environ.get("KADC_CSP_SPLIT_PROJ", "0") == "1"
CSP_SPLIT_MIN_H = int(os.environ.get("KADC_CSP_SPLIT_MIN_H", "0"))
CSP_SPLIT_MAX_H = int(os.environ.get("KADC_CSP_SPLIT_MAX_H", "1000000"))
CSP_POINTER_PROJ = os.environ.get("KADC_CSP_POINTER_PROJ", "0") == "1"
CSP_CUSTOM_CAT = os.environ.get("KADC_CSP_CUSTOM_CAT", "0") == "1"
CSP_BLAS_PROJECTION = os.environ.get("KADC_CSP_BLAS_PROJECTION", "0") == "1"
ELAN_CUSTOM_CAT = os.environ.get("KADC_ELAN_CUSTOM_CAT", "0") == "1"
ELAN_HALF_SPLIT = os.environ.get("KADC_ELAN_HALF_SPLIT", "0") == "1"
ELAN_HYBRID_SPLIT = os.environ.get("KADC_ELAN_HYBRID_SPLIT", "0") == "1"
ELAN_HALF_MIN_H = int(os.environ.get("KADC_ELAN_HALF_MIN_H", "0"))
ELAN_HALF_MAX_H = int(os.environ.get("KADC_ELAN_HALF_MAX_H", "1000000"))
ELAN_SPLIT_PROJ = os.environ.get("KADC_ELAN_SPLIT_PROJ", "0") == "1"
GLOBAL_CUSTOM_CAT = os.environ.get("KADC_GLOBAL_CUSTOM_CAT", "0") == "1"
DEFER_NECK_CAT = os.environ.get("KADC_DEFER_NECK_CAT", "0") == "1"
ADOWN_PARALLEL = os.environ.get("KADC_ADOWN_PARALLEL", "0") == "1"
ADOWN_CUSTOM_POOL = os.environ.get("KADC_ADOWN_CUSTOM_POOL", "0") == "1"
ADOWN_CUSTOM_POOL_CAT = os.environ.get("KADC_ADOWN_CUSTOM_POOL_CAT", "0") == "1"
ADOWN_CUSTOM_MAX_POOL = os.environ.get("KADC_ADOWN_CUSTOM_MAX_POOL", "0") == "1"
ADOWN_FUSED_POOL_SPLIT = os.environ.get("KADC_ADOWN_FUSED_POOL_SPLIT", "0") == "1"
ADOWN_FUSED_POOL_PARALLEL = os.environ.get("KADC_ADOWN_FUSED_POOL_PARALLEL", "0") == "1"
SPPELAN_SPLIT_PROJ = os.environ.get("KADC_SPPELAN_SPLIT_PROJ", "0") == "1"
HEAD_MODE = os.environ.get("KADC_HEAD_MODE", "branch6")
HEAD_COMPILE = os.environ.get("KADC_HEAD_COMPILE", "0") == "1"
HEAD_FUSED_DECODE = os.environ.get("KADC_HEAD_FUSED_DECODE", "0") == "1"
HEAD_FUSE_FINAL_BIAS = os.environ.get("KADC_HEAD_FUSE_FINAL_BIAS", "0") == "1"

if HEAD_FUSED_DECODE:
    from kadc_head_decode import fused_detect_decode


def _streams(device, family, count):
    index = device.index if device.index is not None else torch.cuda.current_device()
    priority = int(os.environ.get("KADC_AUX_STREAM_PRIORITY", "0"))
    key = (index, family, count, priority)
    value = _STREAMS.get(key)
    if value is None:
        value = tuple(
            torch.cuda.Stream(device=device, priority=priority) for _ in range(count)
        )
        _STREAMS[key] = value
    return value


def conv2d_without_bias(self, x):
    return torch.nn.functional.conv2d(
        x, self.weight, None, self.stride, self.padding,
        self.dilation, self.groups,
    )


def repncsp_parallel(self, x):
    """Overlap the independent 1x1 bypass with the KADC main path."""
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "csp", 1)[0]
    side.wait_stream(origin)
    if CSP_MAIN_FIRST:
        main = self.m(self.cv1(x))
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
    else:
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
        main = self.m(self.cv1(x))
    origin.wait_stream(side)
    bypass.record_stream(origin)
    return self.cv3(torch.cat((main, bypass), 1))


def repncsp_split_projection(self, x):
    """Distribute the post-concat 1x1 projection over the two CSP inputs."""
    if not CSP_SPLIT_MIN_H <= x.shape[2] <= CSP_SPLIT_MAX_H:
        return repncsp_parallel(self, x)
    if CSP_BLAS_PROJECTION:
        return repncsp_blas_projection(self, x)
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "csp_split_projection", 1)[0]
    side.wait_stream(origin)
    conv = self.cv3.conv

    if CSP_MAIN_FIRST:
        main = self.m(self.cv1(x))
        main_out = torch.nn.functional.conv2d(
            main, self._sm75_main_weight, None, conv.stride, conv.padding,
            conv.dilation, conv.groups,
        )
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
            bypass_out = torch.nn.functional.conv2d(
                bypass, self._sm75_bypass_weight, None, conv.stride,
                conv.padding, conv.dilation, conv.groups,
            )
    else:
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
            bypass_out = torch.nn.functional.conv2d(
                bypass, self._sm75_bypass_weight, None, conv.stride,
                conv.padding, conv.dilation, conv.groups,
            )
        main = self.m(self.cv1(x))
        main_out = torch.nn.functional.conv2d(
            main, self._sm75_main_weight, None, conv.stride, conv.padding,
            conv.dilation, conv.groups,
        )

    origin.wait_stream(side)
    bypass_out.record_stream(origin)
    return add_bias_silu(main_out, bypass_out, conv.bias)


def repncsp_blas_projection(self, x):
    """Accumulate the two FP32 1x1 projections into one GEMM output buffer."""
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "csp_blas_projection", 1)[0]
    side.wait_stream(origin)
    if CSP_MAIN_FIRST:
        main = self.m(self.cv1(x))
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
    else:
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
        main = self.m(self.cv1(x))

    conv = self.cv3.conv
    spatial = main.shape[2] * main.shape[3]
    main_matrix = main.view(main.shape[1], spatial)
    main_weight = self._sm75_main_weight.view(conv.weight.shape[0], main.shape[1])
    output = torch.mm(main_weight, main_matrix)

    origin.wait_stream(side)
    bypass.record_stream(origin)
    bypass_matrix = bypass.view(bypass.shape[1], spatial)
    bypass_weight = self._sm75_bypass_weight.view(conv.weight.shape[0], bypass.shape[1])
    output.addmm_(bypass_weight, bypass_matrix)
    output = output.view(1, conv.weight.shape[0], main.shape[2], main.shape[3])
    return bias_silu_inplace(output, conv.bias)


def repncsp_pointer_projection(self, x):
    """Project two CSP inputs directly without materializing their cat."""
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "csp_pointer_projection", 1)[0]
    side.wait_stream(origin)
    if CSP_MAIN_FIRST:
        main = self.m(self.cv1(x))
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
    else:
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
        main = self.m(self.cv1(x))
    origin.wait_stream(side)
    bypass.record_stream(origin)
    return conv1x1_two_input_bias_silu(
        main, bypass, self._sm75_main_weight,
        self._sm75_bypass_weight, self.cv3.conv.bias,
    )


def repncsp_custom_cat(self, x):
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "csp_custom_cat", 1)[0]
    side.wait_stream(origin)
    if CSP_MAIN_FIRST:
        main = self.m(self.cv1(x))
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
    else:
        with torch.cuda.stream(side):
            bypass = self.cv2(x)
        main = self.m(self.cv1(x))
    origin.wait_stream(side)
    bypass.record_stream(origin)
    return self.cv3(cat2_contiguous(main, bypass))


def _elan_cv1(self, x):
    """Apply a fused 1x1 projection directly to a deferred two-input concat."""
    if not (DEFER_NECK_CAT and isinstance(x, (list, tuple)) and len(x) == 2):
        return self.cv1(x)
    first, second = x
    conv = self.cv1.conv
    first_channels = first.shape[1]
    cache = getattr(self, "_sm75_neck_input_weights", None)
    if cache is None or cache[0] != first_channels:
        if (conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                or conv.padding != (0, 0) or conv.groups != 1
                or first_channels + second.shape[1] != conv.weight.shape[1]):
            raise ValueError("unsupported deferred neck projection")
        cache = (
            first_channels,
            conv.weight[:, :first_channels].contiguous(),
            conv.weight[:, first_channels:].contiguous(),
        )
        self._sm75_neck_input_weights = cache

    origin = torch.cuda.current_stream(first.device)
    side = _streams(first.device, "neck_input_projection", 1)[0]
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        second_out = torch.nn.functional.conv2d(
            second, cache[2], None, conv.stride, conv.padding,
            conv.dilation, conv.groups,
        )
    first_out = torch.nn.functional.conv2d(
        first, cache[1], None, conv.stride, conv.padding,
        conv.dilation, conv.groups,
    )
    origin.wait_stream(side)
    second_out.record_stream(origin)
    return add_bias_silu(first_out, second_out, conv.bias)


def repncspelan_custom_cat(self, x):
    values = list(_elan_cv1(self, x).chunk(2, 1))
    values.extend((module(values[-1])) for module in (self.cv2, self.cv3))
    return self.cv4(cat4_contiguous(tuple(values)))


def repncspelan_half_split(self, x):
    """Overlap the first half of cv4 with the dependent cv2/cv3 chain."""
    input_height = x[0].shape[2] if isinstance(x, (list, tuple)) else x.shape[2]
    if not ELAN_HALF_MIN_H <= input_height <= ELAN_HALF_MAX_H:
        values = list(_elan_cv1(self, x).chunk(2, 1))
        values.extend((module(values[-1])) for module in (self.cv2, self.cv3))
        return self.cv4(torch.cat(values, 1))
    input_device = x[0].device if isinstance(x, (list, tuple)) else x.device
    origin = torch.cuda.current_stream(input_device)
    side = _streams(input_device, "elan_half_split", 1)[0]
    conv = self.cv4.conv
    first_half = _elan_cv1(self, x)
    _, second = first_half.chunk(2, 1)
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        early = torch.nn.functional.conv2d(
            first_half, self._sm75_elan_early_weight, None,
            conv.stride, conv.padding, conv.dilation, conv.groups,
        )
    third = self.cv2(second)
    if ELAN_HYBRID_SPLIT:
        side.wait_stream(origin)
        with torch.cuda.stream(side):
            third_out = torch.nn.functional.conv2d(
                third, self._sm75_elan_third_weight, None,
                conv.stride, conv.padding, conv.dilation, conv.groups,
            )
        fourth = self.cv3(third)
        fourth_out = torch.nn.functional.conv2d(
            fourth, self._sm75_elan_fourth_weight, None,
            conv.stride, conv.padding, conv.dilation, conv.groups,
        )
        origin.wait_stream(side)
        early.record_stream(origin)
        third_out.record_stream(origin)
        return add3_bias_silu((early, third_out, fourth_out), conv.bias)
    fourth = self.cv3(third)
    late_input = (
        cat2_contiguous(third, fourth)
        if ELAN_CUSTOM_CAT else torch.cat((third, fourth), 1)
    )
    late = torch.nn.functional.conv2d(
        late_input, self._sm75_elan_late_weight, None,
        conv.stride, conv.padding, conv.dilation, conv.groups,
    )
    origin.wait_stream(side)
    early.record_stream(origin)
    return add_bias_silu(early, late, conv.bias)


def repncspelan_split_projection(self, x):
    """Pipeline the four post-concat 1x1 projection terms with cv2/cv3."""
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "elan_split_projection", 1)[0]
    conv = self.cv4.conv
    first, second = _elan_cv1(self, x).chunk(2, 1)
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        part0 = torch.nn.functional.conv2d(
            first, self._sm75_elan_weights[0], None, conv.stride,
            conv.padding, conv.dilation, conv.groups,
        )
        part1 = torch.nn.functional.conv2d(
            second, self._sm75_elan_weights[1], None, conv.stride,
            conv.padding, conv.dilation, conv.groups,
        )

    third = self.cv2(second)
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        part2 = torch.nn.functional.conv2d(
            third, self._sm75_elan_weights[2], None, conv.stride,
            conv.padding, conv.dilation, conv.groups,
        )

    fourth = self.cv3(third)
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        part3 = torch.nn.functional.conv2d(
            fourth, self._sm75_elan_weights[3], None, conv.stride,
            conv.padding, conv.dilation, conv.groups,
        )

    origin.wait_stream(side)
    for part in (part0, part1, part2, part3):
        part.record_stream(origin)
    return add4_bias_silu((part0, part1, part2, part3), conv.bias)


def concat_custom(self, values):
    """Equivalent channel concat for the two-input neck merge nodes."""
    if self.d == 1 and len(values) == 2:
        return cat2_contiguous(values[0], values[1])
    return torch.cat(values, self.d)


def concat_deferred(self, values):
    """Let the following 1x1 projection consume the two inputs directly."""
    if self.d == 1 and len(values) == 2:
        return values
    return torch.cat(values, self.d)


def adown_custom_cat(self, x):
    """Preserve ADown arithmetic while replacing its final channel concat."""
    x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    first, second = x.chunk(2, 1)
    first = self.cv1(first)
    second = torch.nn.functional.max_pool2d(second, 3, 2, 1)
    second = self.cv2(second)
    return cat2_contiguous(first, second)


def adown_custom_pool(self, x):
    """Replace ADown's fixed 2x2/stride-1 average pool with an exact FP32 kernel."""
    if ADOWN_FUSED_POOL_SPLIT:
        if ADOWN_FUSED_POOL_PARALLEL:
            origin = torch.cuda.current_stream(x.device)
            side = _streams(x.device, "adown_fused_pool", 1)[0]
            first, second = adown_pool_split(x, second_stream=side)
            with torch.cuda.stream(side):
                second = self.cv2(second)
            first = self.cv1(first)
            origin.wait_stream(side)
            second.record_stream(origin)
            if ADOWN_CUSTOM_POOL_CAT:
                return cat2_contiguous(first, second)
            return torch.cat((first, second), 1)
        first, second = adown_pool_split(x)
    else:
        x = avg_pool2x2_s1(x)
        first, second = x.chunk(2, 1)
    first = self.cv1(first)
    if not ADOWN_FUSED_POOL_SPLIT:
        if ADOWN_CUSTOM_MAX_POOL:
            second = max_pool3x3_s2p1(second)
        else:
            second = torch.nn.functional.max_pool2d(second, 3, 2, 1)
    second = self.cv2(second)
    if ADOWN_CUSTOM_POOL_CAT:
        return cat2_contiguous(first, second)
    return torch.cat((first, second), 1)


def adown_parallel(self, x):
    """Overlap the independent ADown convolution and max-pool branches."""
    x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    first, second = x.chunk(2, 1)
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "adown", 1)[0]
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        second = torch.nn.functional.max_pool2d(second, 3, 2, 1)
        second = self.cv2(second)
    first = self.cv1(first)
    origin.wait_stream(side)
    second.record_stream(origin)
    return torch.cat((first, second), 1)


def sppelin_custom_cat(self, x):
    """Preserve SPPELAN arithmetic while replacing its four-input concat."""
    values = [self.cv1(x)]
    values.extend(module(values[-1]) for module in (self.cv2, self.cv3, self.cv4))
    return self.cv5(cat4_contiguous(tuple(values)))


def sppelin_split_projection(self, x):
    """Overlap the first cv5 projection term with the serial SPP pool chain."""
    conv = self.cv5.conv
    first = self.cv1(x)
    origin = torch.cuda.current_stream(x.device)
    side = _streams(x.device, "sppelan_projection", 1)[0]
    side.wait_stream(origin)
    with torch.cuda.stream(side):
        early = torch.nn.functional.conv2d(
            first, self._sm75_spp_early_weight, None, conv.stride,
            conv.padding, conv.dilation, conv.groups,
        )
    second = self.cv2(first)
    third = self.cv3(second)
    fourth = self.cv4(third)
    late = torch.nn.functional.conv2d(
        torch.cat((second, third, fourth), 1),
        self._sm75_spp_late_weight, None, conv.stride,
        conv.padding, conv.dilation, conv.groups,
    )
    origin.wait_stream(side)
    early.record_stream(origin)
    return add_bias_silu(early, late, conv.bias)


def _detect_post(self, x, shape):
    if self.training:
        return x
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5)
        )
        self.shape = shape
    if HEAD_FUSED_DECODE:
        y = fused_detect_decode(
            x, self.anchors, self.strides, self.nc,
            getattr(self, "_sm75_raw_biases", None),
        )
        return y if self.export else (y, x)
    box, cls = torch.cat(
        [value.view(shape[0], self.no, -1) for value in x], 2
    ).split((self.reg_max * 4, self.nc), 1)
    # The repository's three clones only protect a CAM/autograd path. In eval
    # they do not change values, so omit the copies on the inference path.
    dbox = dist2bbox(
        self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1
    ) * self.strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)


def ddetect_fast(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return _detect_post(self, x, shape)


def ddetect_scale3(self, x):
    """Run the three independent detection scales on three streams."""
    shape = x[0].shape
    origin = torch.cuda.current_stream(x[0].device)
    streams = _streams(x[0].device, "head_scale", self.nl)
    outputs = [None] * self.nl
    for i, stream in enumerate(streams):
        stream.wait_stream(origin)
        with torch.cuda.stream(stream):
            outputs[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    for stream in streams:
        origin.wait_stream(stream)
    for output in outputs:
        output.record_stream(origin)
    x[:] = outputs
    return _detect_post(self, x, shape)


def ddetect_branch6(self, x):
    """Run box/class branches of all three scales on six streams."""
    shape = x[0].shape
    origin = torch.cuda.current_stream(x[0].device)
    streams = _streams(x[0].device, "head_branch", 2 * self.nl)
    box = [None] * self.nl
    cls = [None] * self.nl
    for i in range(self.nl):
        box_stream, cls_stream = streams[2 * i], streams[2 * i + 1]
        box_stream.wait_stream(origin)
        cls_stream.wait_stream(origin)
        with torch.cuda.stream(box_stream):
            box[i] = self.cv2[i](x[i])
        with torch.cuda.stream(cls_stream):
            cls[i] = self.cv3[i](x[i])
    for stream in streams:
        origin.wait_stream(stream)
    for i in range(self.nl):
        box[i].record_stream(origin)
        cls[i].record_stream(origin)
        x[i] = (
            cat2_contiguous(box[i], cls[i])
            if GLOBAL_CUSTOM_CAT else torch.cat((box[i], cls[i]), 1)
        )
    return _detect_post(self, x, shape)


def patch(model):
    model = patch_deform_serial(model)
    for module in model.modules():
        name = type(module).__name__
        if DEFER_NECK_CAT and name == "Concat":
            module.forward = types.MethodType(concat_deferred, module)
        elif GLOBAL_CUSTOM_CAT and name == "Concat":
            module.forward = types.MethodType(concat_custom, module)
        elif ADOWN_PARALLEL and name == "ADown":
            module.forward = types.MethodType(adown_parallel, module)
        elif ADOWN_CUSTOM_POOL and name == "ADown":
            module.forward = types.MethodType(adown_custom_pool, module)
        elif GLOBAL_CUSTOM_CAT and name == "ADown":
            module.forward = types.MethodType(adown_custom_cat, module)
        elif SPPELAN_SPLIT_PROJ and name == "SPPELAN":
            conv = module.cv5.conv
            edge = module.c
            if (conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                    or conv.padding != (0, 0) or conv.groups != 1
                    or conv.weight.shape[1] != 4 * edge):
                raise ValueError("unsupported SPPELAN projection")
            module._sm75_spp_early_weight = conv.weight[:, :edge].contiguous()
            module._sm75_spp_late_weight = conv.weight[:, edge:].contiguous()
            module.forward = types.MethodType(sppelin_split_projection, module)
        elif GLOBAL_CUSTOM_CAT and name == "SPPELAN":
            module.forward = types.MethodType(sppelin_custom_cat, module)
        elif CSP_PARALLEL and name == "RepNCSP_KADC":
            if CSP_POINTER_PROJ:
                conv = module.cv3.conv
                hidden = conv.weight.shape[1] // 2
                module._sm75_main_weight = conv.weight[:, :hidden].contiguous()
                module._sm75_bypass_weight = conv.weight[:, hidden:].contiguous()
                module.forward = types.MethodType(repncsp_pointer_projection, module)
            elif CSP_SPLIT_PROJ:
                conv = module.cv3.conv
                hidden = conv.weight.shape[1] // 2
                if (conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                        or conv.padding != (0, 0) or conv.groups != 1
                        or conv.weight.shape[1] != 2 * hidden):
                    raise ValueError("unsupported RepNCSP_KADC projection")
                module._sm75_main_weight = conv.weight[:, :hidden].contiguous()
                module._sm75_bypass_weight = conv.weight[:, hidden:].contiguous()
                module.forward = types.MethodType(repncsp_split_projection, module)
            else:
                forward = repncsp_custom_cat if CSP_CUSTOM_CAT else repncsp_parallel
                module.forward = types.MethodType(forward, module)
        elif ELAN_HALF_SPLIT and name == "RepNCSPELAN4_KADC":
            conv = module.cv4.conv
            edge = 2 * module.c
            module._sm75_elan_early_weight = conv.weight[:, :edge].contiguous()
            module._sm75_elan_late_weight = conv.weight[:, edge:].contiguous()
            if ELAN_HYBRID_SPLIT:
                middle = (conv.weight.shape[1] - edge) // 2
                module._sm75_elan_third_weight = (
                    conv.weight[:, edge:edge + middle].contiguous()
                )
                module._sm75_elan_fourth_weight = (
                    conv.weight[:, edge + middle:].contiguous()
                )
            module.forward = types.MethodType(repncspelan_half_split, module)
        elif ELAN_CUSTOM_CAT and name == "RepNCSPELAN4_KADC":
            module.forward = types.MethodType(repncspelan_custom_cat, module)
        elif ELAN_SPLIT_PROJ and name == "RepNCSPELAN4_KADC":
            conv = module.cv4.conv
            edge = module.c
            middle = (conv.weight.shape[1] - 2 * edge) // 2
            sizes = (edge, edge, middle, middle)
            if (conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                    or conv.padding != (0, 0) or conv.groups != 1
                    or sum(sizes) != conv.weight.shape[1]):
                raise ValueError("unsupported RepNCSPELAN4_KADC projection")
            weights = []
            offset = 0
            for size in sizes:
                weights.append(conv.weight[:, offset:offset + size].contiguous())
                offset += size
            module._sm75_elan_weights = tuple(weights)
            module.forward = types.MethodType(repncspelan_split_projection, module)
        elif name == "DDetect":
            if HEAD_FUSED_DECODE and HEAD_FUSE_FINAL_BIAS:
                biases = []
                for index in range(module.nl):
                    box_final = module.cv2[index][-1]
                    cls_final = module.cv3[index][-1]
                    biases.append(torch.cat((box_final.bias, cls_final.bias)).contiguous())
                    box_final.forward = types.MethodType(conv2d_without_bias, box_final)
                    cls_final.forward = types.MethodType(conv2d_without_bias, cls_final)
                module._sm75_raw_biases = biases
            if HEAD_COMPILE:
                import torch._inductor.config as inductor_config
                inductor_config.triton.cudagraphs = False
                for index in range(module.nl):
                    module.cv2[index] = torch.compile(
                        module.cv2[index], fullgraph=True, mode="default"
                    )
                    module.cv3[index] = torch.compile(
                        module.cv3[index], fullgraph=True, mode="default"
                    )
            if HEAD_MODE == "scale3":
                forward = ddetect_scale3
            elif HEAD_MODE == "branch6":
                forward = ddetect_branch6
            else:
                forward = ddetect_fast
            module.forward = types.MethodType(forward, module)
    return model
