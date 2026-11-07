"""Fused bs1 DDetect DFL decode for fixed-shape CUDA Graph inference."""

import os
import torch
import triton
import triton.language as tl


@triton.jit
def _dfl_distance(raw_ptr, bias_ptr, spatial, active,
                  HW: tl.constexpr, SIDE: tl.constexpr,
                  ADD_BIAS: tl.constexpr):
    bins = tl.arange(0, 16)
    logits = tl.load(
        raw_ptr + (SIDE * 16 + bins[:, None]) * HW + spatial[None, :],
        mask=active[None, :], other=-float("inf"),
    )
    if ADD_BIAS:
        logits += tl.load(bias_ptr + SIDE * 16 + bins)[:, None]
        tl.store(
            raw_ptr + (SIDE * 16 + bins[:, None]) * HW + spatial[None, :],
            logits, mask=active[None, :],
        )
    maximum = tl.max(logits, axis=0)
    numerator = tl.exp(logits - maximum[None, :])
    denominator = tl.sum(numerator, axis=0)
    return tl.sum(numerator * bins[:, None], axis=0) / denominator


@triton.jit
def _decode_scale(
    raw_ptr, bias_ptr, anchors_ptr, strides_ptr, out_ptr,
    HW: tl.constexpr, ANCHOR_OFFSET: tl.constexpr,
    TOTAL_ANCHORS: tl.constexpr, NC: tl.constexpr,
    BLOCK: tl.constexpr, ADD_BIAS: tl.constexpr,
):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    anchor_index = ANCHOR_OFFSET + spatial
    left = _dfl_distance(raw_ptr, bias_ptr, spatial, active, HW=HW, SIDE=0, ADD_BIAS=ADD_BIAS)
    top = _dfl_distance(raw_ptr, bias_ptr, spatial, active, HW=HW, SIDE=1, ADD_BIAS=ADD_BIAS)
    right = _dfl_distance(raw_ptr, bias_ptr, spatial, active, HW=HW, SIDE=2, ADD_BIAS=ADD_BIAS)
    bottom = _dfl_distance(raw_ptr, bias_ptr, spatial, active, HW=HW, SIDE=3, ADD_BIAS=ADD_BIAS)
    anchor_x = tl.load(anchors_ptr + anchor_index, mask=active)
    anchor_y = tl.load(anchors_ptr + TOTAL_ANCHORS + anchor_index, mask=active)
    stride = tl.load(strides_ptr + anchor_index, mask=active)
    x1 = anchor_x - left
    y1 = anchor_y - top
    x2 = anchor_x + right
    y2 = anchor_y + bottom
    tl.store(out_ptr + 0 * TOTAL_ANCHORS + anchor_index, ((x1 + x2) / 2.0) * stride, mask=active)
    tl.store(out_ptr + 1 * TOTAL_ANCHORS + anchor_index, ((y1 + y2) / 2.0) * stride, mask=active)
    tl.store(out_ptr + 2 * TOTAL_ANCHORS + anchor_index, (x2 - x1) * stride, mask=active)
    tl.store(out_ptr + 3 * TOTAL_ANCHORS + anchor_index, (y2 - y1) * stride, mask=active)

    for cls in range(NC):
        logit = tl.load(
            raw_ptr + (64 + cls) * HW + spatial, mask=active
        ).to(tl.float32)
        if ADD_BIAS:
            logit += tl.load(bias_ptr + 64 + cls)
            tl.store(raw_ptr + (64 + cls) * HW + spatial, logit, mask=active)
        probability = 1.0 / (1.0 + tl.exp(-logit))
        tl.store(
            out_ptr + (4 + cls) * TOTAL_ANCHORS + anchor_index,
            probability, mask=active,
        )


def fused_detect_decode(raw_outputs, anchors, strides, nc, raw_biases=None):
    if len(raw_outputs) != 3 or raw_outputs[0].shape[0] != 1 or nc != 20:
        raise ValueError("fused detect decode supports bs1, 3 scales, nc=20")
    sizes = [value.shape[2] * value.shape[3] for value in raw_outputs]
    total = sum(sizes)
    output = torch.empty((1, 4 + nc, total), device=raw_outputs[0].device, dtype=raw_outputs[0].dtype)
    offset = 0
    for index, (value, hw) in enumerate(zip(raw_outputs, sizes)):
        block = int(os.environ.get("KADC_HEAD_DECODE_BLOCK", "64"))
        warps = int(os.environ.get("KADC_HEAD_DECODE_WARPS", "4"))
        _decode_scale[(triton.cdiv(hw, block),)](
            value.contiguous(),
            raw_biases[index] if raw_biases is not None else value,
            anchors.contiguous(), strides.contiguous(), output,
            HW=hw, ANCHOR_OFFSET=offset, TOTAL_ANCHORS=total, NC=nc,
            BLOCK=block, ADD_BIAS=raw_biases is not None, num_warps=warps,
        )
        offset += hw
    return output
