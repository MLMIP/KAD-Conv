"""Experimental multi-stream KADC branch scheduler, originally tuned on RTX 2080 Ti.

The math kernels are imported unchanged from kadc_fused_sm75.  Only the three
independent scale branches are issued on separate CUDA streams and joined before
the fused concat/residual kernel.
"""

import os
import types
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision

from kadc_fused_sm75 import (
    avg_pool_pow2,
    bnrelu_scale_gap,
    cat_residual_att0,
    kadc4_fwd_tri2,
    patch as patch_sm75,
    pool2_pool4,
    pool_project_1x1,
    rca_triton,
    se_mlp,
    upsample_bilinear_att,
)


_STREAMS = {}
_EVENTS = {}
_BRANCH_EXECUTORS = tuple(ThreadPoolExecutor(max_workers=1) for _ in range(3))
PARALLEL_MODE = os.environ.get("KADC_PARALLEL_MODE", "two")
DCN_BACKEND = os.environ.get("KADC_DCN_BACKEND", "torchvision")
DCN_FOLD_POST_BN = os.environ.get("KADC_DCN_FOLD_POST_BN", "0") == "1"
DCN_FUSE_OFFSET_BIAS = os.environ.get("KADC_DCN_FUSE_OFFSET_BIAS", "0") == "1"
DCN_FUSE_INPUT_BIAS = os.environ.get("KADC_DCN_FUSE_INPUT_BIAS", "0") == "1"
DCN_FUSE_EPILOGUE = os.environ.get("KADC_DCN_FUSE_EPILOGUE", "0") == "1"
DCN_INPUT_BIAS_MIN_H = int(os.environ.get("KADC_DCN_INPUT_BIAS_MIN_H", "0"))
DCN_INPUT_BIAS_MAX_H = int(os.environ.get("KADC_DCN_INPUT_BIAS_MAX_H", "1000000"))
FUSED_POOL = os.environ.get("KADC_FUSED_POOL", "0") == "1"
FUSED_POOL_PROJ = os.environ.get("KADC_FUSED_POOL_PROJ", "0") == "1"
RCA_ALREADY_SIGMOID = (
    os.environ.get("KADC_RCA_FUSED_VPW", "0") == "1"
    and os.environ.get("KADC_RCA_FUSED_SIGMOID", "0") == "1"
)

if DCN_BACKEND == "direct":
    from kadc_dcn_direct import deform_conv2d_direct


def _branch_streams(device, count, family="branch"):
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    priority = int(os.environ.get("KADC_BRANCH_STREAM_PRIORITY", "0"))
    key = (device_index, count, family, priority)
    streams = _STREAMS.get(key)
    if streams is None:
        streams = tuple(
            torch.cuda.Stream(device=device, priority=priority) for _ in range(count)
        )
        _STREAMS[key] = streams
    return streams


def _branch_events(device, count, family):
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, count, family)
    events = _EVENTS.get(key)
    if events is None:
        events = tuple(torch.cuda.Event() for _ in range(count))
        _EVENTS[key] = events
    return events


def _run_branch(self, x, i):
    conv, fc, weight = self.convs[i], self.fc[i], self.weight[i]
    projection = conv[0]
    scaled_h = x.shape[2] // (2**i) if i else x.shape[2]
    fuse_input_bias = (
        DCN_FUSE_INPUT_BIAS and DCN_BACKEND == "direct"
        and DCN_INPUT_BIAS_MIN_H <= scaled_h <= DCN_INPUT_BIAS_MAX_H
    )
    if i and FUSED_POOL_PROJ:
        projected = pool_project_1x1(
            x, projection, 2**i, add_bias=not fuse_input_bias,
        )
    else:
        if i:
            x_scaled = avg_pool_pow2(x, 2**i) if FUSED_POOL else torch.nn.functional.avg_pool2d(x, kernel_size=2**i)
        else:
            x_scaled = x
        if fuse_input_bias:
            projected = torch.nn.functional.conv2d(
                x_scaled, projection.weight, None, projection.stride,
                projection.padding, projection.dilation, projection.groups,
            )
        else:
            projected = projection(x_scaled)
    value = conv[1](projected)
    feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
    attention = se_mlp(gap, fc)
    if i:
        feature = upsample_bilinear_att(feature, attention, x.shape[2:])
    return feature, attention


def _run_branch_scaled(self, x_scaled, i, full_shape):
    conv, fc, weight = self.convs[i], self.fc[i], self.weight[i]
    value = conv[1](conv[0](x_scaled))
    feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
    attention = se_mlp(gap, fc)
    feature = upsample_bilinear_att(feature, attention, full_shape)
    return feature


def kadc4_fwd_fused_scales(self, x):
    origin = torch.cuda.current_stream(x.device)
    prep, branch1, branch2 = _branch_streams(x.device, 3, "fused_scales")
    for stream in (prep, branch1, branch2):
        stream.wait_stream(origin)
    features = [None, None, None]
    features[0], att0 = _run_branch(self, x, 0)
    with torch.cuda.stream(prep):
        scaled2, scaled4 = pool2_pool4(x)
    branch1.wait_stream(prep)
    branch2.wait_stream(prep)
    with torch.cuda.stream(branch1):
        features[1] = _run_branch_scaled(self, scaled2, 1, x.shape[2:])
    with torch.cuda.stream(branch2):
        features[2] = _run_branch_scaled(self, scaled4, 2, x.shape[2:])
    for stream in (branch1, branch2):
        origin.wait_stream(stream)
    scaled2.record_stream(branch1)
    scaled4.record_stream(branch2)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def _run_branch_thread(self, x, i, stream, origin):
    """Enqueue one branch from a persistent CPU thread.

    PyTorch's cuBLAS handle pool is thread-local. Fixed workers therefore keep
    the concurrent deform-conv GEMM workspaces independent across branches.
    """
    torch.cuda.set_device(x.device)
    stream.wait_stream(origin)
    with torch.cuda.stream(stream), torch.no_grad():
        return _run_branch(self, x, i)


def kadc4_fwd_threaded_three(self, x):
    origin = torch.cuda.current_stream(x.device)
    streams = _branch_streams(x.device, 3)
    futures = [
        _BRANCH_EXECUTORS[i].submit(_run_branch_thread, self, x, i, streams[i], origin)
        for i in range(3)
    ]
    results = [future.result() for future in futures]
    features = [result[0] for result in results]
    attentions = [result[1] for result in results]
    for stream in streams:
        origin.wait_stream(stream)
    for feature in features:
        feature.record_stream(origin)
    attentions[0].record_stream(origin)
    return cat_residual_att0(features, attentions[0], x)


def kadc4_fwd_split_prep_three(self, x):
    """Overlap offset convolution and RCA mask generation per branch."""
    origin = torch.cuda.current_stream(x.device)
    side_streams = _branch_streams(x.device, 2, "split_branch")
    branch_streams = (origin, side_streams[0], side_streams[1])
    mask_streams = _branch_streams(x.device, 3, "split_mask")
    projected = [None, None, None]
    offsets = [None, None, None]
    masks = [None, None, None]

    for i, conv in enumerate(self.convs):
        branch = branch_streams[i]
        mask_stream = mask_streams[i]
        if branch is not origin:
            branch.wait_stream(origin)
        with torch.cuda.stream(branch):
            x_scaled = torch.nn.functional.avg_pool2d(x, kernel_size=2**i) if i else x
            projected[i] = conv[0](x_scaled)

        # wait_stream observes the projection already queued above. Offset is
        # queued afterwards, so RCA and offset can overlap.
        mask_stream.wait_stream(branch)
        with torch.cuda.stream(mask_stream):
            dcn = conv[1]
            masks[i] = rca_triton(projected[i], dcn.RCA_mask) if dcn.mask else None
            if masks[i] is not None and not RCA_ALREADY_SIGMOID:
                masks[i] = torch.sigmoid(masks[i])
        with torch.cuda.stream(branch):
            if DCN_BACKEND == "direct" and DCN_FUSE_OFFSET_BIAS:
                offset_conv = dcn.conv_offset
                offsets[i] = torch.nn.functional.conv2d(
                    projected[i], offset_conv.weight, None, offset_conv.stride,
                    offset_conv.padding, offset_conv.dilation, offset_conv.groups,
                )
            else:
                offsets[i] = dcn.conv_offset(projected[i])

    features = [None, None, None]
    attentions = [None, None, None]
    for i, (conv, fc, weight) in enumerate(zip(self.convs, self.fc, self.weight)):
        branch = branch_streams[i]
        mask_stream = mask_streams[i]
        branch.wait_stream(mask_stream)
        dcn = conv[1]
        with torch.cuda.stream(branch):
            if DCN_BACKEND == "direct":
                value = deform_conv2d_direct(
                    input=projected[i], offset=offsets[i], weight=dcn.weight,
                    bias=dcn.bias, padding=dcn.padding, mask=masks[i], stride=dcn.stride,
                    offset_bias=dcn.conv_offset.bias if DCN_FUSE_OFFSET_BIAS else None,
                )
            else:
                value = torchvision.ops.deform_conv2d(
                    input=projected[i], offset=offsets[i], weight=dcn.weight,
                    bias=dcn.bias, padding=dcn.padding, mask=masks[i], stride=dcn.stride,
                )
            feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
            attention = se_mlp(gap, fc)
            if i:
                feature = upsample_bilinear_att(feature, attention, x.shape[2:])
            features[i] = feature
            attentions[i] = attention

    for branch in side_streams:
        origin.wait_stream(branch)
    for feature in features:
        feature.record_stream(origin)
    attentions[0].record_stream(origin)
    return cat_residual_att0(features, attentions[0], x)


def kadc4_fwd_parallel(self, x):
    origin = torch.cuda.current_stream(x.device)
    features = [None, None, None]

    # Schedule the low-resolution branches first so they can start while the
    # full-resolution branch is subsequently queued on the origin stream.
    side_count = 2 if PARALLEL_MODE in ("three", "auto") else 1
    streams = _branch_streams(x.device, side_count)
    for i in (1, 2):
        stream = streams[i - 1] if side_count == 2 else streams[0]
        stream.wait_stream(origin)
        with torch.cuda.stream(stream):
            feature, _ = _run_branch(self, x, i)
            features[i] = feature

    features[0], att0 = _run_branch(self, x, 0)
    for stream in streams:
        origin.wait_stream(stream)

    # The branch tensors were allocated on side streams but are consumed and
    # released from the origin stream. Mark that use explicitly; without it the
    # eager caching allocator can recycle side-stream storage early, especially
    # for batch>1 and dynamic validation shapes.
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def kadc4_fwd_origin_first(self, x):
    """Queue the critical full-resolution branch before low-res side work."""
    origin = torch.cuda.current_stream(x.device)
    streams = _branch_streams(x.device, 2, "origin_first")
    features = [None, None, None]
    # Fork before branch-0 is queued, then intentionally enqueue branch-0's
    # kernels first on the host so they reach the device scheduler first.
    for stream in streams:
        stream.wait_stream(origin)
    features[0], att0 = _run_branch(self, x, 0)
    for i in (1, 2):
        stream = streams[i - 1]
        with torch.cuda.stream(stream):
            features[i], _ = _run_branch(self, x, i)
    for stream in streams:
        origin.wait_stream(stream)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def kadc4_fwd_stagger_projection(self, x):
    """Let branch-0 projection finish before low-resolution branches start."""
    origin = torch.cuda.current_stream(x.device)
    streams = _branch_streams(x.device, 2, "stagger_projection")
    projection_ready = _branch_events(x.device, 1, "stagger_projection")[0]
    features = [None, None, None]

    conv, fc, weight = self.convs[0], self.fc[0], self.weight[0]
    projected = conv[0](x)
    projection_ready.record(origin)
    value = conv[1](projected)
    feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
    features[0] = feature
    att0 = se_mlp(gap, fc)

    # These waits target the explicit projection event, not all branch-0 work
    # queued after it.
    for i in (1, 2):
        stream = streams[i - 1]
        stream.wait_event(projection_ready)
        with torch.cuda.stream(stream):
            features[i], _ = _run_branch(self, x, i)
    for stream in streams:
        origin.wait_stream(stream)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def kadc4_fwd_b0_split_origin_first(self, x):
    """Overlap branch-0 offset prediction with its fused Triton RCA mask."""
    origin = torch.cuda.current_stream(x.device)
    low_streams = _branch_streams(x.device, 2, "b0_split_low")
    mask_stream = _branch_streams(x.device, 1, "b0_split_mask")[0]
    projection_ready = _branch_events(x.device, 1, "b0_split_projection")[0]
    for stream in low_streams:
        stream.wait_stream(origin)

    features = [None, None, None]
    conv, fc, weight = self.convs[0], self.fc[0], self.weight[0]
    projected = conv[0](x)
    projection_ready.record(origin)
    dcn = conv[1]
    mask_stream.wait_event(projection_ready)
    with torch.cuda.stream(mask_stream):
        mask = rca_triton(projected, dcn.RCA_mask) if dcn.mask else None
        if mask is not None and not RCA_ALREADY_SIGMOID:
            mask = torch.sigmoid(mask)
    if DCN_BACKEND == "direct" and DCN_FUSE_OFFSET_BIAS:
        offset_conv = dcn.conv_offset
        offset = torch.nn.functional.conv2d(
            projected, offset_conv.weight, None, offset_conv.stride,
            offset_conv.padding, offset_conv.dilation, offset_conv.groups,
        )
    else:
        offset = dcn.conv_offset(projected)
    origin.wait_stream(mask_stream)
    mask.record_stream(origin)
    if DCN_BACKEND == "direct":
        value = deform_conv2d_direct(
            input=projected, offset=offset, weight=dcn.weight, bias=dcn.bias,
            padding=dcn.padding, mask=mask, stride=dcn.stride,
            offset_bias=dcn.conv_offset.bias if DCN_FUSE_OFFSET_BIAS else None,
        )
    else:
        value = torchvision.ops.deform_conv2d(
            input=projected, offset=offset, weight=dcn.weight, bias=dcn.bias,
            padding=dcn.padding, mask=mask, stride=dcn.stride,
        )
    feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
    features[0] = feature
    att0 = se_mlp(gap, fc)

    for i in (1, 2):
        with torch.cuda.stream(low_streams[i - 1]):
            features[i], _ = _run_branch(self, x, i)
    for stream in low_streams:
        origin.wait_stream(stream)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def _run_branch_native_split(self, x, i, mask_stream, projection_ready):
    """Overlap only the two independent preparation kernels of one branch.

    Unlike the historical b0_split path, this preserves every optimization
    required by the selective CUDA backend: projection/input-bias folding,
    offset correction, fused RCA tail, DCN epilogue, and fused GAP.  The
    candidate intentionally supports the final direct-backend configuration;
    any other configuration falls back to the already verified serial branch.
    """
    conv, fc, weight = self.convs[i], self.fc[i], self.weight[i]
    dcn = conv[1]
    scaled_h = x.shape[2] // (2**i) if i else x.shape[2]
    supported = (
        DCN_BACKEND == "direct"
        and DCN_FUSE_OFFSET_BIAS
        and DCN_FUSE_INPUT_BIAS
        and DCN_FUSE_EPILOGUE
        and getattr(dcn, "_sm75_input_bias", None) is not None
        and DCN_INPUT_BIAS_MIN_H <= scaled_h <= DCN_INPUT_BIAS_MAX_H
        and os.environ.get("KADC_DCN_FUSE_OFFSET_CONV", "0") != "1"
        and os.environ.get("KADC_RCA_FUSE_OFFSET_CORRECTION", "0") != "1"
    )
    if not supported:
        return _run_branch(self, x, i)

    projection = conv[0]
    if i and FUSED_POOL_PROJ:
        projected = pool_project_1x1(x, projection, 2**i, add_bias=False)
    else:
        x_scaled = (
            avg_pool_pow2(x, 2**i) if i and FUSED_POOL
            else torch.nn.functional.avg_pool2d(x, kernel_size=2**i) if i
            else x
        )
        projected = torch.nn.functional.conv2d(
            x_scaled, projection.weight, None, projection.stride,
            projection.padding, projection.dilation, projection.groups,
        )
    origin = torch.cuda.current_stream(x.device)
    projection_ready.record(origin)

    input_bias = dcn._sm75_input_bias
    key = (projected.shape[2], projected.shape[3])
    cache = getattr(dcn, "_sm75_offset_correction", None)
    if cache is None or cache[0] != key:
        bias_image = input_bias.view(1, -1, 1, 1).expand_as(projected)
        correction = torch.nn.functional.conv2d(
            bias_image, dcn.conv_offset.weight, None, dcn.conv_offset.stride,
            dcn.conv_offset.padding, dcn.conv_offset.dilation,
            dcn.conv_offset.groups,
        ).contiguous()
        cache = (key, correction, dcn.conv_offset.bias)
        dcn._sm75_offset_correction = cache
    offset_correction, effective_offset_bias = cache[1], cache[2]

    # RCA writes both its horizontal intermediate (returned as mask) and the
    # projection with its folded bias.  Offset prediction is independent and
    # stays on the origin stream.
    mask_stream.wait_event(projection_ready)
    with torch.cuda.stream(mask_stream):
        biased_x = torch.empty_like(projected)
        mask = rca_triton(
            projected, dcn.RCA_mask, input_bias=input_bias,
            biased_output=biased_x, offset=None, offset_correction=None,
        ) if dcn.mask else None
        if mask is not None and not RCA_ALREADY_SIGMOID:
            mask = torch.sigmoid(mask)

    offset_conv = dcn.conv_offset
    offset = torch.nn.functional.conv2d(
        projected, offset_conv.weight, None, offset_conv.stride,
        offset_conv.padding, offset_conv.dilation, offset_conv.groups,
    )
    origin.wait_stream(mask_stream)
    if mask is not None:
        mask.record_stream(origin)
    biased_x.record_stream(origin)

    epilogue = getattr(dcn, "_sm75_post_epilogue", None)
    if epilogue is None:
        bn = dcn._sm75_post_bn
        post_mul = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
        post_add = (bn.bias - bn.running_mean * post_mul).contiguous()
        post_scale = dcn._sm75_post_scale.reshape(-1).contiguous()
        epilogue = (post_mul, post_add, post_scale)
        dcn._sm75_post_epilogue = epilogue

    value = deform_conv2d_direct(
        input=biased_x, offset=offset, weight=dcn.weight, bias=dcn.bias,
        padding=dcn.padding, mask=mask, stride=dcn.stride,
        offset_bias=effective_offset_bias, input_bias=input_bias,
        offset_correction=offset_correction, input_already_biased=True,
        post_mul=epilogue[0], post_add=epilogue[1], post_scale=epilogue[2],
    )
    feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
    attention = se_mlp(gap, fc)
    if i:
        feature = upsample_bilinear_att(feature, attention, x.shape[2:])
    return feature, attention


def kadc4_fwd_b0_native_split(self, x):
    """Origin-first scheduling plus a lifecycle-safe branch-0 prep split."""
    origin = torch.cuda.current_stream(x.device)
    low_streams = _branch_streams(x.device, 2, "b0_native_split_low")
    mask_stream = _branch_streams(x.device, 1, "b0_native_split_mask")[0]
    projection_ready = _branch_events(x.device, 1, "b0_native_split_projection")[0]
    for stream in low_streams:
        stream.wait_stream(origin)

    features = [None, None, None]
    features[0], att0 = _run_branch_native_split(
        self, x, 0, mask_stream, projection_ready,
    )
    for i in (1, 2):
        with torch.cuda.stream(low_streams[i - 1]):
            features[i], _ = _run_branch(self, x, i)
    for stream in low_streams:
        origin.wait_stream(stream)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, att0, x)


def kadc4_fwd_all_native_split(self, x):
    """Apply the verified prep split independently to all three branches."""
    origin = torch.cuda.current_stream(x.device)
    branch_side = _branch_streams(x.device, 2, "all_native_split_branch")
    branch_streams = (origin, branch_side[0], branch_side[1])
    mask_streams = _branch_streams(x.device, 3, "all_native_split_mask")
    projection_ready = _branch_events(x.device, 3, "all_native_split_projection")
    for stream in branch_side:
        stream.wait_stream(origin)

    features = [None, None, None]
    attentions = [None, None, None]
    features[0], attentions[0] = _run_branch_native_split(
        self, x, 0, mask_streams[0], projection_ready[0],
    )
    for i in (1, 2):
        with torch.cuda.stream(branch_streams[i]):
            features[i], attentions[i] = _run_branch_native_split(
                self, x, i, mask_streams[i], projection_ready[i],
            )
    for stream in branch_side:
        origin.wait_stream(stream)
    for feature in features[1:]:
        feature.record_stream(origin)
    return cat_residual_att0(features, attentions[0], x)


def kadc4_fwd_staged(self, x):
    """Parallelize safe stages while serializing all DCN_RCA calls.

    torchvision.deform_conv2d is not reliable when several large-batch calls
    overlap on different streams in this environment.  The branch projections
    and post-DCN attention/resize stages remain concurrent.
    """
    origin = torch.cuda.current_stream(x.device)
    streams = _branch_streams(x.device, 3)
    projected = [None, None, None]

    # Stage 1: independent scale transforms and 1x1 projections.
    for i, conv in enumerate(self.convs):
        stream = streams[i]
        stream.wait_stream(origin)
        with torch.cuda.stream(stream):
            x_scaled = torch.nn.functional.avg_pool2d(x, kernel_size=2**i) if i else x
            projected[i] = conv[0](x_scaled)

    # Stage 2: DCN/RCA is intentionally serialized on origin.  As soon as one
    # branch finishes DCN, its safe epilogue is released to its branch stream,
    # where it can overlap the next branch's DCN.
    features = [None, None, None]
    attentions = [None, None, None]
    for i, (conv, fc, weight) in enumerate(zip(self.convs, self.fc, self.weight)):
        stream = streams[i]
        origin.wait_stream(stream)
        projected[i].record_stream(origin)
        value = conv[1](projected[i])
        stream.wait_stream(origin)
        value.record_stream(stream)
        with torch.cuda.stream(stream):
            feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
            attention = se_mlp(gap, fc)
            if i:
                feature = upsample_bilinear_att(feature, attention, x.shape[2:])
            features[i] = feature
            attentions[i] = attention

    for stream in streams:
        origin.wait_stream(stream)
    for feature in features:
        feature.record_stream(origin)
    attentions[0].record_stream(origin)
    return cat_residual_att0(features, attentions[0], x)


def kadc4_fwd_deform_serial(self, x):
    """Parallelize DCN preparation, serialize only deform sampling/GEMM."""
    origin = torch.cuda.current_stream(x.device)
    streams = _branch_streams(x.device, 3)
    projected = [None, None, None]
    offsets = [None, None, None]
    masks = [None, None, None]

    # Projection, offset prediction and RCA mask generation are independent.
    for i, conv in enumerate(self.convs):
        stream = streams[i]
        stream.wait_stream(origin)
        with torch.cuda.stream(stream):
            x_scaled = torch.nn.functional.avg_pool2d(x, kernel_size=2**i) if i else x
            projected[i] = conv[0](x_scaled)
            dcn = conv[1]
            offsets[i] = dcn.conv_offset(projected[i])
            masks[i] = torch.sigmoid(rca_triton(projected[i], dcn.RCA_mask)) if dcn.mask else None

    # Only torchvision deform sampling/GEMM is serialized. Each completed
    # branch immediately releases its safe epilogue to the branch stream.
    features = [None, None, None]
    attentions = [None, None, None]
    for i, (conv, fc, weight) in enumerate(zip(self.convs, self.fc, self.weight)):
        stream = streams[i]
        dcn = conv[1]
        origin.wait_stream(stream)
        projected[i].record_stream(origin)
        offsets[i].record_stream(origin)
        if masks[i] is not None:
            masks[i].record_stream(origin)
        if DCN_BACKEND == "direct":
            value = deform_conv2d_direct(
                input=projected[i], offset=offsets[i], weight=dcn.weight,
                bias=dcn.bias, padding=dcn.padding, mask=masks[i], stride=dcn.stride,
            )
        else:
            value = torchvision.ops.deform_conv2d(
                input=projected[i],
                offset=offsets[i],
                weight=dcn.weight,
                bias=dcn.bias,
                padding=dcn.padding,
                mask=masks[i],
                stride=dcn.stride,
            )
        stream.wait_stream(origin)
        value.record_stream(stream)
        with torch.cuda.stream(stream):
            feature, gap = bnrelu_scale_gap(value, conv[2], weight[0].weight)
            attention = se_mlp(gap, fc)
            if i:
                feature = upsample_bilinear_att(feature, attention, x.shape[2:])
            features[i] = feature
            attentions[i] = attention

    for stream in streams:
        origin.wait_stream(stream)
    for feature in features:
        feature.record_stream(origin)
    attentions[0].record_stream(origin)
    return cat_residual_att0(features, attentions[0], x)


def kadc4_fwd_auto(self, x):
    # Full three-stream DCN overlap is verified exact for the real-time bs1
    # path, but torchvision.deform_conv2d is not stream-safe for larger batches
    # in this environment. Preserve correctness by falling back automatically.
    if x.shape[0] == 1:
        return kadc4_fwd_parallel(self, x)
    return kadc4_fwd_tri2(self, x)


def patch(model):
    model = patch_sm75(model)
    for module in model.modules():
        if type(module).__name__ in {"KADC4", "KADC4_m", "KADConv"}:
            if DCN_FUSE_EPILOGUE and DCN_BACKEND == "direct":
                for index, branch in enumerate(module.convs):
                    dcn, bn = branch[1], branch[2]
                    dcn._sm75_post_bn = bn
                    dcn._sm75_post_scale = module.weight[index][0].weight
                    bn._sm75_dcn_epilogue_fused = True
            if DCN_FUSE_INPUT_BIAS and DCN_BACKEND == "direct":
                for branch in module.convs:
                    branch[1]._sm75_input_bias = branch[0].bias
            if DCN_FOLD_POST_BN and not getattr(module, "_sm75_dcn_bn_folded", False):
                with torch.no_grad():
                    for branch in module.convs:
                        dcn, bn = branch[1], branch[2]
                        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                        dcn.weight.mul_(scale[:, None, None, None])
                        dcn.bias.copy_((dcn.bias - bn.running_mean) * scale + bn.bias)
                        bn._sm75_dcn_bn_folded = True
                module._sm75_dcn_bn_folded = True
            if PARALLEL_MODE == "staged":
                forward = kadc4_fwd_staged
            elif PARALLEL_MODE == "origin_first":
                forward = kadc4_fwd_origin_first
            elif PARALLEL_MODE == "stagger_projection":
                forward = kadc4_fwd_stagger_projection
            elif PARALLEL_MODE == "b0_split_origin_first":
                forward = kadc4_fwd_b0_split_origin_first
            elif PARALLEL_MODE == "b0_native_split":
                forward = kadc4_fwd_b0_native_split
            elif PARALLEL_MODE == "all_native_split":
                forward = kadc4_fwd_all_native_split
            elif PARALLEL_MODE == "split_prep_three":
                forward = kadc4_fwd_split_prep_three
            elif PARALLEL_MODE == "threaded_three":
                forward = kadc4_fwd_threaded_three
            elif PARALLEL_MODE == "fused_scales":
                forward = kadc4_fwd_fused_scales
            elif PARALLEL_MODE == "deform_serial":
                forward = kadc4_fwd_deform_serial
            elif PARALLEL_MODE == "auto":
                forward = kadc4_fwd_auto
            else:
                forward = kadc4_fwd_parallel
            module.forward = types.MethodType(forward, module)
    return model
