import os
import torch, triton, triton.language as tl, torchvision, types, time

UPSAMPLE_TILE = int(os.environ.get("KADC_UP_TILE", "128"))
UPSAMPLE_WARPS = int(os.environ.get("KADC_UP_WARPS", "4"))
CAT_TILE = int(os.environ.get("KADC_CAT_TILE", "128"))
CAT_WARPS = int(os.environ.get("KADC_CAT_WARPS", "4"))
RCA_FUSED_VPW = os.environ.get("KADC_RCA_FUSED_VPW", "0") == "1"
RCA_VPW_BLOCK = int(os.environ.get("KADC_RCA_VPW_BLOCK", "32"))
RCA_VPW_WARPS = int(os.environ.get("KADC_RCA_VPW_WARPS", "4"))
RCA_VPW_STAGES = int(os.environ.get("KADC_RCA_VPW_STAGES", "2"))
RCA_VPW_SMALL_THRESHOLD = int(os.environ.get("KADC_RCA_VPW_SMALL_THRESHOLD", "0"))
RCA_VPW_SMALL_BLOCK = int(os.environ.get("KADC_RCA_VPW_SMALL_BLOCK", str(RCA_VPW_BLOCK)))
RCA_VPW_SMALL_WARPS = int(os.environ.get("KADC_RCA_VPW_SMALL_WARPS", str(RCA_VPW_WARPS)))
RCA_VPW_SMALL_STAGES = int(os.environ.get("KADC_RCA_VPW_SMALL_STAGES", str(RCA_VPW_STAGES)))
RCA_VPW_TINY_THRESHOLD = int(os.environ.get("KADC_RCA_VPW_TINY_THRESHOLD", "0"))
RCA_VPW_TINY_BLOCK = int(os.environ.get("KADC_RCA_VPW_TINY_BLOCK", str(RCA_VPW_SMALL_BLOCK)))
RCA_VPW_TINY_WARPS = int(os.environ.get("KADC_RCA_VPW_TINY_WARPS", str(RCA_VPW_SMALL_WARPS)))
RCA_VPW_TINY_STAGES = int(os.environ.get("KADC_RCA_VPW_TINY_STAGES", str(RCA_VPW_SMALL_STAGES)))
RCA_FUSED_SIGMOID = os.environ.get("KADC_RCA_FUSED_SIGMOID", "0") == "1"
RCA_REDUCE_WARPS = int(os.environ.get("KADC_RCA_REDUCE_WARPS", "4"))
RCA_HCONV_WARPS = int(os.environ.get("KADC_RCA_HCONV_WARPS", "4"))
RCA_FUSED_RH = os.environ.get("KADC_RCA_FUSED_RH", "0") == "1"
RCA_RH_WARPS = int(os.environ.get("KADC_RCA_RH_WARPS", "8"))
RCA_FOLD_CONSTANTS = os.environ.get("KADC_RCA_FOLD_CONSTANTS", "0") == "1"
RCA_PACK_PW = os.environ.get("KADC_RCA_PACK_PW", "0") == "1"
RCA_VECTOR_PW = os.environ.get("KADC_RCA_VECTOR_PW", "0") == "1"
RCA_VECTOR4_PW = os.environ.get("KADC_RCA_VECTOR4_PW", "0") == "1"
RCA_FUSE_OFFSET_CORRECTION = os.environ.get("KADC_RCA_FUSE_OFFSET_CORRECTION", "0") == "1"
FUSE_CONV_BIAS_SILU = os.environ.get("KADC_FUSE_CONV_BIAS_SILU", "0") == "1"
FUSE_RESIDUAL_ADD = os.environ.get("KADC_FUSE_RESIDUAL_ADD", "0") == "1"
OFFSET_TRITON = os.environ.get("KADC_OFFSET_TRITON", "0") == "1"
OFFSET_SPATIAL_BLOCK = int(os.environ.get("KADC_OFFSET_SPATIAL_BLOCK", "32"))
OFFSET_CHANNEL_BLOCK = int(os.environ.get("KADC_OFFSET_CHANNEL_BLOCK", "3"))
OFFSET_WARPS = int(os.environ.get("KADC_OFFSET_WARPS", "4"))
CONV1X1_TRITON = os.environ.get("KADC_CONV1X1_TRITON", "0") == "1"
CONV1X1_BM = int(os.environ.get("KADC_CONV1X1_BM", "32"))
CONV1X1_BN = int(os.environ.get("KADC_CONV1X1_BN", "16"))
CONV1X1_BK = int(os.environ.get("KADC_CONV1X1_BK", "16"))
CONV1X1_WARPS = int(os.environ.get("KADC_CONV1X1_WARPS", "4"))
CONV1X1_MIN_H = int(os.environ.get("KADC_CONV1X1_MIN_H", "0"))
CONV1X1_MAX_H = int(os.environ.get("KADC_CONV1X1_MAX_H", "1000000"))
POOL_PROJ_SPATIAL_BLOCK = int(os.environ.get("KADC_POOL_PROJ_SPATIAL_BLOCK", "32"))
POOL_PROJ_CHANNEL_BLOCK = int(os.environ.get("KADC_POOL_PROJ_CHANNEL_BLOCK", "8"))
POOL_PROJ_WARPS = int(os.environ.get("KADC_POOL_PROJ_WARPS", "4"))
CONV_BIAS_SILU_BLOCK = int(os.environ.get("KADC_CONV_BIAS_SILU_BLOCK", "256"))
CONV_BIAS_SILU_WARPS = int(os.environ.get("KADC_CONV_BIAS_SILU_WARPS", "4"))
SE_MLP_WARPS = int(os.environ.get("KADC_SE_MLP_WARPS", "4"))


@triton.jit
def k_bias_silu(x_ptr, bias_ptr, out_ptr, HW: tl.constexpr, NT: tl.constexpr,
                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    channel = pid // NT
    tile = pid - channel * NT
    spatial = tile * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    index = channel * HW + spatial
    value = tl.load(x_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + channel).to(tl.float32)
    result = value / (1.0 + tl.exp(-value))
    tl.store(out_ptr + index, result, mask=active)


@triton.jit
def k_bias_silu_residual(
    x_ptr, bias_ptr, residual_ptr, out_ptr,
    HW: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = pid // NT
    tile = pid - channel * NT
    spatial = tile * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    index = channel * HW + spatial
    value = tl.load(x_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + channel).to(tl.float32)
    result = value / (1.0 + tl.exp(-value))
    result += tl.load(residual_ptr + index, mask=active, other=0.0)
    tl.store(out_ptr + index, result, mask=active)


@triton.jit
def k_add_bias_silu(
    lhs_ptr, rhs_ptr, bias_ptr, out_ptr,
    HW: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = pid // NT
    tile = pid - channel * NT
    spatial = tile * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    index = channel * HW + spatial
    value = tl.load(lhs_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(rhs_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + channel).to(tl.float32)
    value = value / (1.0 + tl.exp(-value))
    tl.store(out_ptr + index, value, mask=active)


@triton.jit
def k_add3_bias_silu(
    x0_ptr, x1_ptr, x2_ptr, bias_ptr, out_ptr,
    HW: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = pid // NT
    tile = pid - channel * NT
    spatial = tile * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    index = channel * HW + spatial
    value = tl.load(x0_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(x1_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(x2_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + channel).to(tl.float32)
    value = value / (1.0 + tl.exp(-value))
    tl.store(out_ptr + index, value, mask=active)


@triton.jit
def k_add4_bias_silu(
    x0_ptr, x1_ptr, x2_ptr, x3_ptr, bias_ptr, out_ptr,
    HW: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = pid // NT
    tile = pid - channel * NT
    spatial = tile * BLOCK + tl.arange(0, BLOCK)
    active = spatial < HW
    index = channel * HW + spatial
    value = tl.load(x0_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(x1_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(x2_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(x3_ptr + index, mask=active, other=0.0).to(tl.float32)
    value += tl.load(bias_ptr + channel).to(tl.float32)
    value = value / (1.0 + tl.exp(-value))
    tl.store(out_ptr + index, value, mask=active)


def add_bias_silu(lhs, rhs, bias):
    output = torch.empty_like(lhs)
    n, c, h, w = lhs.shape
    assert n == 1 and rhs.shape == lhs.shape
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_add_bias_silu[(c * nt,)](
        lhs, rhs, bias, output, HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return output


def add3_bias_silu(values, bias):
    x0, x1, x2 = values
    output = torch.empty_like(x0)
    n, c, h, w = x0.shape
    assert n == 1 and all(value.shape == x0.shape for value in values[1:])
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_add3_bias_silu[(c * nt,)](
        x0, x1, x2, bias, output,
        HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return output


def add4_bias_silu(values, bias):
    x0, x1, x2, x3 = values
    output = torch.empty_like(x0)
    n, c, h, w = x0.shape
    assert n == 1 and all(value.shape == x0.shape for value in values[1:])
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_add4_bias_silu[(c * nt,)](
        x0, x1, x2, x3, bias, output,
        HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return output


def bias_silu_inplace(value, bias):
    """Apply the fused FP32 bias/SiLU epilogue without another activation buffer."""
    n, c, h, w = value.shape
    assert n == 1 and value.is_contiguous() and bias.numel() == c
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_bias_silu[(c * nt,)](
        value, bias, value, HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return value


@triton.jit
def k_avg_pool2x2_s1(
    x_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    OHW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    spatial = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = spatial < OHW
    row = spatial // (W - 1)
    col = spatial - row * (W - 1)
    source = channel * H * W + row * W + col
    value = tl.load(x_ptr + source, mask=active, other=0.0)
    value += tl.load(x_ptr + source + 1, mask=active, other=0.0)
    value += tl.load(x_ptr + source + W, mask=active, other=0.0)
    value += tl.load(x_ptr + source + W + 1, mask=active, other=0.0)
    tl.store(out_ptr + channel * OHW + spatial, value * 0.25, mask=active)


@triton.jit
def k_avg_pool2x2_s1_rows(
    x_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    OHW: tl.constexpr, BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    active = col < W - 1
    source = channel * H * W + row * W + col
    value = tl.load(x_ptr + source, mask=active, other=0.0)
    value += tl.load(x_ptr + source + 1, mask=active, other=0.0)
    value += tl.load(x_ptr + source + W, mask=active, other=0.0)
    value += tl.load(x_ptr + source + W + 1, mask=active, other=0.0)
    target = channel * OHW + row * (W - 1) + col
    tl.store(out_ptr + target, value * 0.25, mask=active)


def avg_pool2x2_s1(value):
    n, c, h, w = value.shape
    assert n == 1 and h == w and value.is_contiguous()
    output = torch.empty((n, c, h - 1, w - 1), device=value.device, dtype=value.dtype)
    block = int(os.environ.get("KADC_ADOWN_POOL_BLOCK", "256"))
    ohw = (h - 1) * (w - 1)
    warps = int(os.environ.get("KADC_ADOWN_POOL_WARPS", "4"))
    if os.environ.get("KADC_ADOWN_POOL_LAYOUT", "flat") == "row":
        k_avg_pool2x2_s1_rows[(c, h - 1, triton.cdiv(w - 1, block))](
            value, output, H=h, W=w, OHW=ohw,
            BLOCK=block, num_warps=warps,
        )
    else:
        k_avg_pool2x2_s1[(c, triton.cdiv(ohw, block))](
            value, output, H=h, W=w, OHW=ohw,
            BLOCK=block, num_warps=warps,
        )
    return output


@triton.jit
def k_avgmax_pool2x2_then_3x3_s2p1(
    x_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    C_HALF: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse the second ADown branch's average and max pooling."""
    channel = tl.program_id(0)
    out_row = tl.program_id(1)
    out_col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    active = out_col < OW

    base_row = out_row * 2 - 1
    base_col = out_col * 2 - 1
    channel_offset = (channel + C_HALF) * H * W
    r0 = base_row
    r1 = base_row + 1
    r2 = base_row + 2
    r3 = base_row + 3
    c0 = base_col
    c1 = base_col + 1
    c2 = base_col + 2
    c3 = base_col + 3
    vr0 = (r0 >= 0) & (r0 < H)
    vr1 = (r1 >= 0) & (r1 < H)
    vr2 = (r2 >= 0) & (r2 < H)
    vr3 = (r3 >= 0) & (r3 < H)
    vc0 = (c0 >= 0) & (c0 < W)
    vc1 = (c1 >= 0) & (c1 < W)
    vc2 = (c2 >= 0) & (c2 < W)
    vc3 = (c3 >= 0) & (c3 < W)

    x00 = tl.load(x_ptr + channel_offset + r0 * W + c0, mask=active & vr0 & vc0, other=0.0)
    x01 = tl.load(x_ptr + channel_offset + r0 * W + c1, mask=active & vr0 & vc1, other=0.0)
    x02 = tl.load(x_ptr + channel_offset + r0 * W + c2, mask=active & vr0 & vc2, other=0.0)
    x03 = tl.load(x_ptr + channel_offset + r0 * W + c3, mask=active & vr0 & vc3, other=0.0)
    x10 = tl.load(x_ptr + channel_offset + r1 * W + c0, mask=active & vr1 & vc0, other=0.0)
    x11 = tl.load(x_ptr + channel_offset + r1 * W + c1, mask=active & vr1 & vc1, other=0.0)
    x12 = tl.load(x_ptr + channel_offset + r1 * W + c2, mask=active & vr1 & vc2, other=0.0)
    x13 = tl.load(x_ptr + channel_offset + r1 * W + c3, mask=active & vr1 & vc3, other=0.0)
    x20 = tl.load(x_ptr + channel_offset + r2 * W + c0, mask=active & vr2 & vc0, other=0.0)
    x21 = tl.load(x_ptr + channel_offset + r2 * W + c1, mask=active & vr2 & vc1, other=0.0)
    x22 = tl.load(x_ptr + channel_offset + r2 * W + c2, mask=active & vr2 & vc2, other=0.0)
    x23 = tl.load(x_ptr + channel_offset + r2 * W + c3, mask=active & vr2 & vc3, other=0.0)
    x30 = tl.load(x_ptr + channel_offset + r3 * W + c0, mask=active & vr3 & vc0, other=0.0)
    x31 = tl.load(x_ptr + channel_offset + r3 * W + c1, mask=active & vr3 & vc1, other=0.0)
    x32 = tl.load(x_ptr + channel_offset + r3 * W + c2, mask=active & vr3 & vc2, other=0.0)
    x33 = tl.load(x_ptr + channel_offset + r3 * W + c3, mask=active & vr3 & vc3, other=0.0)

    a00 = (((x00 + x01) + x10) + x11) * 0.25
    a01 = (((x01 + x02) + x11) + x12) * 0.25
    a02 = (((x02 + x03) + x12) + x13) * 0.25
    a10 = (((x10 + x11) + x20) + x21) * 0.25
    a11 = (((x11 + x12) + x21) + x22) * 0.25
    a12 = (((x12 + x13) + x22) + x23) * 0.25
    a20 = (((x20 + x21) + x30) + x31) * 0.25
    a21 = (((x21 + x22) + x31) + x32) * 0.25
    a22 = (((x22 + x23) + x32) + x33) * 0.25

    avg_h = H - 1
    avg_w = W - 1
    valid_r0 = (base_row >= 0) & (base_row < avg_h)
    valid_r1 = (base_row + 1 >= 0) & (base_row + 1 < avg_h)
    valid_r2 = (base_row + 2 >= 0) & (base_row + 2 < avg_h)
    valid_c0 = (base_col >= 0) & (base_col < avg_w)
    valid_c1 = (base_col + 1 >= 0) & (base_col + 1 < avg_w)
    valid_c2 = (base_col + 2 >= 0) & (base_col + 2 < avg_w)
    neg_inf = -float("inf")
    a00 = tl.where(valid_r0 & valid_c0, a00, neg_inf)
    a01 = tl.where(valid_r0 & valid_c1, a01, neg_inf)
    a02 = tl.where(valid_r0 & valid_c2, a02, neg_inf)
    a10 = tl.where(valid_r1 & valid_c0, a10, neg_inf)
    a11 = tl.where(valid_r1 & valid_c1, a11, neg_inf)
    a12 = tl.where(valid_r1 & valid_c2, a12, neg_inf)
    a20 = tl.where(valid_r2 & valid_c0, a20, neg_inf)
    a21 = tl.where(valid_r2 & valid_c1, a21, neg_inf)
    a22 = tl.where(valid_r2 & valid_c2, a22, neg_inf)

    result = tl.maximum(tl.maximum(a00, a01), a02)
    result = tl.maximum(result, tl.maximum(tl.maximum(a10, a11), a12))
    result = tl.maximum(result, tl.maximum(tl.maximum(a20, a21), a22))
    target = channel * OH * OW + out_row * OW + out_col
    tl.store(out_ptr + target, result, mask=active)


def adown_pool_split(value, second_stream=None):
    """Return the two pooled ADown channel halves without a large second-half map."""
    n, c, h, w = value.shape
    assert n == 1 and h == w and c % 2 == 0 and value.is_contiguous()
    half = c // 2
    avg_h = h - 1
    avg_hw = avg_h * avg_h
    out_h = (avg_h + 1) // 2
    first = torch.empty((n, half, avg_h, avg_h), device=value.device, dtype=value.dtype)
    second = torch.empty((n, half, out_h, out_h), device=value.device, dtype=value.dtype)

    avg_block = int(os.environ.get("KADC_ADOWN_POOL_BLOCK", "64"))
    avg_warps = int(os.environ.get("KADC_ADOWN_POOL_WARPS", "2"))
    fused_block = int(os.environ.get("KADC_ADOWN_FUSED_POOL_BLOCK", "16"))
    fused_warps = int(os.environ.get("KADC_ADOWN_FUSED_POOL_WARPS", "1"))

    def launch_first():
        k_avg_pool2x2_s1[(half, triton.cdiv(avg_hw, avg_block))](
            value, first, H=h, W=w, OHW=avg_hw,
            BLOCK=avg_block, num_warps=avg_warps,
        )

    def launch_second():
        k_avgmax_pool2x2_then_3x3_s2p1[
            (half, out_h, triton.cdiv(out_h, fused_block))
        ](
            value, second, H=h, W=w, C_HALF=half, OH=out_h, OW=out_h,
            BLOCK=fused_block, num_warps=fused_warps,
        )

    if second_stream is None:
        launch_first()
        launch_second()
    else:
        origin = torch.cuda.current_stream(value.device)
        second_stream.wait_stream(origin)
        with torch.cuda.stream(second_stream):
            launch_second()
        value.record_stream(second_stream)
        second.record_stream(second_stream)
        launch_first()
    return first, second


@triton.jit
def k_max_pool3x3_s2p1(
    x_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    out_row = tl.program_id(1)
    out_col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    active = out_col < OW
    source_row0 = out_row * 2 - 1
    source_col0 = out_col * 2 - 1
    result = tl.full((BLOCK,), -float("inf"), tl.float32)
    for delta_row in range(3):
        source_row = source_row0 + delta_row
        valid_row = (source_row >= 0) & (source_row < H)
        for delta_col in range(3):
            source_col = source_col0 + delta_col
            valid = active & valid_row & (source_col >= 0) & (source_col < W)
            index = channel * H * W + source_row * W + source_col
            value = tl.load(x_ptr + index, mask=valid, other=-float("inf"))
            result = tl.maximum(result, value)
    target = channel * OH * OW + out_row * OW + out_col
    tl.store(out_ptr + target, result, mask=active)


def max_pool3x3_s2p1(value):
    n, c, h, w = value.shape
    assert n == 1 and h == w and value.is_contiguous()
    output_height = (h + 1) // 2
    output = torch.empty(
        (n, c, output_height, output_height),
        device=value.device,
        dtype=value.dtype,
    )
    block = int(os.environ.get("KADC_ADOWN_MAX_BLOCK", "32"))
    warps = int(os.environ.get("KADC_ADOWN_MAX_WARPS", "1"))
    k_max_pool3x3_s2p1[(c, output_height, triton.cdiv(output_height, block))](
        value, output,
        H=h, W=w, OH=output_height, OW=output_height,
        BLOCK=block, num_warps=warps,
    )
    return output


@triton.jit
def k_cat2_contiguous(
    x0_ptr, x1_ptr, out_ptr,
    N0: tl.constexpr, N1: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    from0 = index < N0
    from1 = (index >= N0) & (index < N0 + N1)
    value0 = tl.load(x0_ptr + index, mask=from0, other=0.0)
    value1 = tl.load(x1_ptr + index - N0, mask=from1, other=0.0)
    tl.store(out_ptr + index, value0 + value1, mask=index < N0 + N1)


@triton.jit
def k_cat4_contiguous(
    x0_ptr, x1_ptr, x2_ptr, x3_ptr, out_ptr,
    N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
    N3: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    edge1 = N0
    edge2 = N0 + N1
    edge3 = N0 + N1 + N2
    total = edge3 + N3
    value = tl.load(x0_ptr + index, mask=index < edge1, other=0.0)
    value += tl.load(
        x1_ptr + index - edge1,
        mask=(index >= edge1) & (index < edge2), other=0.0,
    )
    value += tl.load(
        x2_ptr + index - edge2,
        mask=(index >= edge2) & (index < edge3), other=0.0,
    )
    value += tl.load(
        x3_ptr + index - edge3,
        mask=(index >= edge3) & (index < total), other=0.0,
    )
    tl.store(out_ptr + index, value, mask=index < total)


def cat2_contiguous(x0, x1):
    assert x0.shape[0] == 1 and x1.shape[0] == 1 and x0.shape[2:] == x1.shape[2:]
    n0, n1 = x0.numel(), x1.numel()
    output = torch.empty(
        (1, x0.shape[1] + x1.shape[1], x0.shape[2], x0.shape[3]),
        device=x0.device, dtype=x0.dtype,
    )
    k_cat2_contiguous[(triton.cdiv(n0 + n1, CAT_TILE),)](
        x0, x1, output, N0=n0, N1=n1, BLOCK=CAT_TILE,
        num_warps=CAT_WARPS,
    )
    return output


def cat4_contiguous(values):
    x0, x1, x2, x3 = values
    assert all(value.shape[0] == 1 and value.shape[2:] == x0.shape[2:]
               for value in values)
    sizes = tuple(value.numel() for value in values)
    output = torch.empty(
        (1, sum(value.shape[1] for value in values), x0.shape[2], x0.shape[3]),
        device=x0.device, dtype=x0.dtype,
    )
    k_cat4_contiguous[(triton.cdiv(sum(sizes), CAT_TILE),)](
        x0, x1, x2, x3, output,
        N0=sizes[0], N1=sizes[1], N2=sizes[2], N3=sizes[3],
        BLOCK=CAT_TILE, num_warps=CAT_WARPS,
    )
    return output


@triton.jit
def k_conv1x1_bias_silu(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    HW: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    active_m = m < HW
    active_n = n < COUT
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, CIN, BK):
        k = k0 + tl.arange(0, BK)
        active_k = k < CIN
        a = tl.load(
            x_ptr + m[:, None] + k[None, :] * HW,
            mask=active_m[:, None] & active_k[None, :], other=0.0,
        )
        b = tl.load(
            weight_ptr + k[:, None] + n[None, :] * CIN,
            mask=active_k[:, None] & active_n[None, :], other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=False)
    acc += tl.load(bias_ptr + n, mask=active_n, other=0.0)[None, :]
    acc = acc / (1.0 + tl.exp(-acc))
    tl.store(
        out_ptr + n[None, :] * HW + m[:, None], acc,
        mask=active_m[:, None] & active_n[None, :],
    )


@triton.jit
def k_conv1x1_two_input_bias_silu(
    x0_ptr, x1_ptr, weight0_ptr, weight1_ptr, bias_ptr, out_ptr,
    HW: tl.constexpr, CIN0: tl.constexpr, CIN1: tl.constexpr,
    COUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    active_m = m < HW
    active_n = n < COUT
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, CIN0, BK):
        k = k0 + tl.arange(0, BK)
        active_k = k < CIN0
        a = tl.load(
            x0_ptr + m[:, None] + k[None, :] * HW,
            mask=active_m[:, None] & active_k[None, :], other=0.0,
        )
        b = tl.load(
            weight0_ptr + k[:, None] + n[None, :] * CIN0,
            mask=active_k[:, None] & active_n[None, :], other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=False)
    for k0 in range(0, CIN1, BK):
        k = k0 + tl.arange(0, BK)
        active_k = k < CIN1
        a = tl.load(
            x1_ptr + m[:, None] + k[None, :] * HW,
            mask=active_m[:, None] & active_k[None, :], other=0.0,
        )
        b = tl.load(
            weight1_ptr + k[:, None] + n[None, :] * CIN1,
            mask=active_k[:, None] & active_n[None, :], other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=False)
    acc += tl.load(bias_ptr + n, mask=active_n, other=0.0)[None, :]
    acc = acc / (1.0 + tl.exp(-acc))
    tl.store(
        out_ptr + n[None, :] * HW + m[:, None], acc,
        mask=active_m[:, None] & active_n[None, :],
    )


def conv1x1_fwd_bias_silu(self, x):
    if not (CONV1X1_MIN_H <= x.shape[2] <= CONV1X1_MAX_H):
        return conv_fwd_bias_silu(self, x)
    n, cin, h, w = x.shape
    conv = self.conv
    cout = conv.weight.shape[0]
    assert n == 1 and conv.groups == 1
    output = torch.empty((1, cout, h, w), device=x.device, dtype=x.dtype)
    hw = h * w
    bm, bn, bk = CONV1X1_BM, CONV1X1_BN, CONV1X1_BK
    k_conv1x1_bias_silu[(triton.cdiv(hw, bm), triton.cdiv(cout, bn))](
        x, conv.weight, conv.bias, output,
        HW=hw, CIN=cin, COUT=cout, BM=bm, BN=bn, BK=bk,
        num_warps=CONV1X1_WARPS,
    )
    return output


def conv1x1_two_input_bias_silu(x0, x1, weight0, weight1, bias):
    n, cin0, h, w = x0.shape
    cin1 = x1.shape[1]
    cout = weight0.shape[0]
    assert (n == 1 and x1.shape[0] == 1 and x1.shape[2:] == (h, w)
            and weight0.shape[1] == cin0 and weight1.shape == (cout, cin1, 1, 1))
    output = torch.empty((1, cout, h, w), device=x0.device, dtype=x0.dtype)
    hw = h * w
    bm, bn, bk = CONV1X1_BM, CONV1X1_BN, CONV1X1_BK
    k_conv1x1_two_input_bias_silu[
        (triton.cdiv(hw, bm), triton.cdiv(cout, bn))
    ](
        x0, x1, weight0, weight1, bias, output,
        HW=hw, CIN0=cin0, CIN1=cin1, COUT=cout,
        BM=bm, BN=bn, BK=bk, num_warps=CONV1X1_WARPS,
    )
    return output


def conv_fwd_bias_silu(self, x):
    conv = self.conv
    value = torch.nn.functional.conv2d(
        x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    output = torch.empty_like(value)
    n, c, h, w = value.shape
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_bias_silu[(n * c * nt,)](
        value, conv.bias, output, HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return output


def conv_fwd_bias_silu_residual(self, x, residual):
    conv = self.conv
    value = torch.nn.functional.conv2d(
        x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    output = torch.empty_like(value)
    n, c, h, w = value.shape
    block = CONV_BIAS_SILU_BLOCK
    nt = triton.cdiv(h * w, block)
    k_bias_silu_residual[(n * c * nt,)](
        value, conv.bias, residual, output,
        HW=h * w, NT=nt, BLOCK=block,
        num_warps=CONV_BIAS_SILU_WARPS,
    )
    return output


def repn_bottleneck_fused_residual(self, x):
    return conv_fwd_bias_silu_residual(self.cv2, self.cv1(x), x)


@triton.jit
def k_pool2_pool4(x_ptr, out2_ptr, out4_ptr,
                  H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    h4 = H // 4
    w4 = W // 4
    hw4 = h4 * w4
    pid = tl.program_id(0)
    channel = pid // tl.cdiv(hw4, BLOCK)
    tile = pid - channel * tl.cdiv(hw4, BLOCK)
    linear = tile * BLOCK + tl.arange(0, BLOCK)
    active = linear < hw4
    row4 = linear // w4
    col4 = linear - row4 * w4
    base = channel * H * W + row4 * 4 * W + col4 * 4
    a00 = tl.zeros((BLOCK,), tl.float32)
    a01 = tl.zeros((BLOCK,), tl.float32)
    a10 = tl.zeros((BLOCK,), tl.float32)
    a11 = tl.zeros((BLOCK,), tl.float32)
    for dy in range(4):
        for dx in range(4):
            value = tl.load(x_ptr + base + dy * W + dx, mask=active, other=0.0)
            if dy < 2 and dx < 2:
                a00 += value
            elif dy < 2:
                a01 += value
            elif dx < 2:
                a10 += value
            else:
                a11 += value
    out2_w = W // 2
    out2_hw = (H // 2) * out2_w
    row2 = row4 * 2
    col2 = col4 * 2
    out2_base = channel * out2_hw + row2 * out2_w + col2
    tl.store(out2_ptr + out2_base, a00 * 0.25, mask=active)
    tl.store(out2_ptr + out2_base + 1, a01 * 0.25, mask=active)
    tl.store(out2_ptr + out2_base + out2_w, a10 * 0.25, mask=active)
    tl.store(out2_ptr + out2_base + out2_w + 1, a11 * 0.25, mask=active)
    tl.store(out4_ptr + channel * hw4 + linear,
             (a00 + a01 + a10 + a11) * 0.0625, mask=active)


def pool2_pool4(x):
    n, c, h, w = x.shape
    if n != 1 or h % 4 or w % 4:
        raise ValueError("fused scale pooling requires bs1 and H/W divisible by four")
    out2 = torch.empty((1, c, h // 2, w // 2), device=x.device, dtype=x.dtype)
    out4 = torch.empty((1, c, h // 4, w // 4), device=x.device, dtype=x.dtype)
    block = 64
    grid = c * triton.cdiv((h // 4) * (w // 4), block)
    k_pool2_pool4[(grid,)](x.contiguous(), out2, out4, H=h, W=w, BLOCK=block, num_warps=4)
    return out2, out4


@triton.jit
def k_avg_pool_pow2(x_ptr, out_ptr,
                    H: tl.constexpr, W: tl.constexpr, FACTOR: tl.constexpr,
                    BLOCK: tl.constexpr):
    oh = H // FACTOR
    ow = W // FACTOR
    ohw = oh * ow
    pid = tl.program_id(0)
    channel = pid // tl.cdiv(ohw, BLOCK)
    tile = pid - channel * tl.cdiv(ohw, BLOCK)
    linear = tile * BLOCK + tl.arange(0, BLOCK)
    active = linear < ohw
    row = linear // ow
    col = linear - row * ow
    base = channel * H * W + row * FACTOR * W + col * FACTOR
    acc = tl.zeros((BLOCK,), tl.float32)
    for dy in range(FACTOR):
        for dx in range(FACTOR):
            acc += tl.load(x_ptr + base + dy * W + dx, mask=active, other=0.0)
    tl.store(out_ptr + channel * ohw + linear, acc / (FACTOR * FACTOR), mask=active)


def avg_pool_pow2(x, factor):
    n, c, h, w = x.shape
    if n != 1 or factor not in (2, 4) or h % factor or w % factor:
        raise ValueError("specialized avg pool requires bs1, factor 2/4, divisible H/W")
    out = torch.empty((1, c, h // factor, w // factor), device=x.device, dtype=x.dtype)
    block = int(os.environ.get(f"KADC_POOL{factor}_BLOCK", os.environ.get("KADC_POOL_BLOCK", "128")))
    warps = int(os.environ.get(f"KADC_POOL{factor}_WARPS", os.environ.get("KADC_POOL_WARPS", "4")))
    grid = c * triton.cdiv((h // factor) * (w // factor), block)
    k_avg_pool_pow2[(grid,)](x.contiguous(), out, H=h, W=w, FACTOR=factor,
                             BLOCK=block, num_warps=warps)
    return out


@triton.jit
def k_pool_project_1x1(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    FACTOR: tl.constexpr, SPATIAL_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr, ADD_BIAS: tl.constexpr,
):
    co = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    spatial = tl.program_id(1) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    oh = H // FACTOR
    ow = W // FACTOR
    out_hw = oh * ow
    active_c = co < COUT
    active_s = spatial < out_hw
    oy = spatial // ow
    ox = spatial - oy * ow
    acc = tl.zeros((CHANNEL_BLOCK, SPATIAL_BLOCK), dtype=tl.float32)
    scale = 1.0 / (FACTOR * FACTOR)
    for ci in range(CIN):
        pooled = tl.zeros((SPATIAL_BLOCK,), dtype=tl.float32)
        for dy in range(FACTOR):
            for dx in range(FACTOR):
                iy = oy * FACTOR + dy
                ix = ox * FACTOR + dx
                pooled += tl.load(
                    x_ptr + ci * H * W + iy * W + ix,
                    mask=active_s, other=0.0,
                )
        pooled *= scale
        weight = tl.load(
            weight_ptr + co * CIN + ci, mask=active_c, other=0.0,
        )
        acc += weight[:, None] * pooled[None, :]
    if ADD_BIAS:
        acc += tl.load(bias_ptr + co, mask=active_c, other=0.0)[:, None]
    tl.store(
        out_ptr + co[:, None] * out_hw + spatial[None, :], acc,
        mask=active_c[:, None] & active_s[None, :],
    )


def pool_project_1x1(x, projection, factor, add_bias=True):
    n, cin, h, w = x.shape
    cout = projection.weight.shape[0]
    assert n == 1 and projection.kernel_size == (1, 1) and projection.groups == 1
    oh, ow = h // factor, w // factor
    out = torch.empty((1, cout, oh, ow), device=x.device, dtype=x.dtype)
    sb, cb = POOL_PROJ_SPATIAL_BLOCK, POOL_PROJ_CHANNEL_BLOCK
    k_pool_project_1x1[(triton.cdiv(cout, cb), triton.cdiv(oh * ow, sb))](
        x, projection.weight,
        projection.bias if projection.bias is not None else projection.weight,
        out, H=h, W=w, CIN=cin, COUT=cout, FACTOR=factor,
        SPATIAL_BLOCK=sb, CHANNEL_BLOCK=cb, ADD_BIAS=add_bias,
        num_warps=POOL_PROJ_WARPS,
    )
    return out

# ---------- RCA: reduce (x_h,x_w) ----------
@triton.jit
def k_reduce_hw(x_ptr, xh_ptr, xw_ptr, H, W, BH: tl.constexpr, BW: tl.constexpr):
    pid = tl.program_id(0)
    base = pid*H*W
    rh=tl.arange(0,BH); rw=tl.arange(0,BW); mh=rh<H; mw=rw<W
    off=base+rh[:,None]*W+rw[None,:]; mask=mh[:,None]&mw[None,:]
    x=tl.load(x_ptr+off,mask=mask,other=0.0)
    xh=tl.sum(x,axis=1)/W; xw=tl.sum(x,axis=0)/H
    tl.store(xh_ptr+pid*H+rh, xh, mask=mh)
    tl.store(xw_ptr+pid*W+rw, xw, mask=mw)

# ---------- RCA: horizontal band conv (reads xh scalar + xw vec) + BN + ReLU ----------
@triton.jit
def k_hconv_bn_relu(xh_ptr, xw_ptr, w_ptr, b_ptr, mean_ptr, var_ptr, gam_ptr, bet_ptr, out_ptr,
                    C, H, W, eps, K: tl.constexpr, BW: tl.constexpr):
    pid=tl.program_id(0); nc=pid//H; h=pid%H; c=nc%C
    xh=tl.load(xh_ptr+nc*H+h)
    rw=tl.arange(0,BW); mw=rw<W
    acc=tl.zeros((BW,),dtype=tl.float32); pad=K//2
    for k in range(K):
        src=rw+k-pad; ms=(src>=0)&(src<W)
        xw=tl.load(xw_ptr+nc*W+src, mask=ms, other=0.0)
        g=tl.where(ms, xh+xw, 0.0)
        acc += g*tl.load(w_ptr+c*K+k)
    acc += tl.load(b_ptr+c)
    acc=(acc-tl.load(mean_ptr+c))/tl.sqrt(tl.load(var_ptr+c)+eps)*tl.load(gam_ptr+c)+tl.load(bet_ptr+c)
    acc=tl.maximum(acc,0.0)
    tl.store(out_ptr+nc*H*W+h*W+rw, acc, mask=mw)


@triton.jit
def k_reduce_hconv_bn_relu(
    x_ptr, input_bias_ptr, biased_out_ptr, xw_ptr, w_ptr, b_ptr,
    mean_ptr, var_ptr, gam_ptr, bet_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr,
    K: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr,
    BN_FOLDED: tl.constexpr,
    INPUT_BIAS_FUSED: tl.constexpr,
    STORE_BIASED_INPUT: tl.constexpr,
):
    nc = tl.program_id(0)
    rh = tl.arange(0, BH)
    rw = tl.arange(0, BW)
    valid_h = rh < H
    valid_w = rw < W
    matrix_mask = valid_h[:, None] & valid_w[None, :]
    base = nc * H * W
    value = tl.load(
        x_ptr + base + rh[:, None] * W + rw[None, :],
        mask=matrix_mask, other=0.0,
    )
    if INPUT_BIAS_FUSED:
        value = tl.where(matrix_mask, value + tl.load(input_bias_ptr + nc), 0.0)
    if STORE_BIASED_INPUT:
        tl.store(biased_out_ptr + base + rh[:, None] * W + rw[None, :],
                 value, mask=matrix_mask)
    row_mean = tl.sum(value, axis=1) / W
    col_mean = tl.sum(value, axis=0) / H
    tl.store(xw_ptr + nc * W + rw, col_mean, mask=valid_w)
    tl.debug_barrier()

    channel = nc
    # The horizontal band convolution is separable with respect to the
    # broadcast row_mean term.  Compute the W-vector convolution once instead
    # of repeating identical column work for every row.
    column_acc = tl.zeros((BW,), dtype=tl.float32)
    row_coefficient = tl.zeros((BW,), dtype=tl.float32)
    pad = K // 2
    for k in range(K):
        source = rw + k - pad
        valid_source = (source >= 0) & (source < W)
        shifted_col = tl.load(
            xw_ptr + nc * W + source,
            mask=valid_source, other=0.0,
        )
        kernel_weight = tl.load(w_ptr + channel * K + k)
        column_acc += shifted_col * kernel_weight
        row_coefficient += tl.where(valid_source, kernel_weight, 0.0)
    acc = column_acc[None, :] + row_mean[:, None] * row_coefficient[None, :]
    acc += tl.load(b_ptr + channel)
    if not BN_FOLDED:
        acc = (acc - tl.load(mean_ptr + channel)) / tl.sqrt(
            tl.load(var_ptr + channel) + EPS
        ) * tl.load(gam_ptr + channel) + tl.load(bet_ptr + channel)
    acc = tl.maximum(acc, 0.0)
    tl.store(
        out_ptr + base + rh[:, None] * W + rw[None, :],
        acc, mask=matrix_mask,
    )

# ---------- RCA: vertical band conv ----------
@triton.jit
def k_vconv(x_ptr, w_ptr, b_ptr, out_ptr, C, H, W, K: tl.constexpr, BH: tl.constexpr):
    pid=tl.program_id(0); w_idx=pid%W; nc=pid//W; c=nc%C
    base=nc*H*W+w_idx
    rh=tl.arange(0,BH); mh=rh<H
    acc=tl.zeros((BH,),dtype=tl.float32); pad=K//2
    for k in range(K):
        src=rh+k-pad; ms=(src>=0)&(src<H)
        v=tl.load(x_ptr+base+src*W, mask=ms, other=0.0)
        acc += v*tl.load(w_ptr+c*K+k)
    acc += tl.load(b_ptr+c)
    tl.store(out_ptr+base+rh*W, acc, mask=mh)


@triton.jit
def k_vconv_pointwise9(
    x_ptr, v_w_ptr, v_b_ptr, pw_w_ptr, pw_b_ptr, out_ptr,
    offset_ptr, offset_correction_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    K: tl.constexpr, BLOCK: tl.constexpr, APPLY_SIGMOID: tl.constexpr,
    VERTICAL_BIAS_FOLDED: tl.constexpr,
    PACKED_PW: tl.constexpr,
    APPLY_OFFSET_CORRECTION: tl.constexpr,
    MASK_PROGRAMS: tl.constexpr,
):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hw = H * W
    active = spatial < hw
    row = spatial // W
    col = spatial - row * W
    acc0 = tl.full((BLOCK,), tl.load(pw_b_ptr + 0), tl.float32)
    acc1 = tl.full((BLOCK,), tl.load(pw_b_ptr + 1), tl.float32)
    acc2 = tl.full((BLOCK,), tl.load(pw_b_ptr + 2), tl.float32)
    acc3 = tl.full((BLOCK,), tl.load(pw_b_ptr + 3), tl.float32)
    acc4 = tl.full((BLOCK,), tl.load(pw_b_ptr + 4), tl.float32)
    acc5 = tl.full((BLOCK,), tl.load(pw_b_ptr + 5), tl.float32)
    acc6 = tl.full((BLOCK,), tl.load(pw_b_ptr + 6), tl.float32)
    acc7 = tl.full((BLOCK,), tl.load(pw_b_ptr + 7), tl.float32)
    acc8 = tl.full((BLOCK,), tl.load(pw_b_ptr + 8), tl.float32)
    pad = K // 2
    for c in range(C):
        vertical = tl.zeros((BLOCK,), tl.float32) if VERTICAL_BIAS_FOLDED else tl.full((BLOCK,), tl.load(v_b_ptr + c), tl.float32)
        for k in range(K):
            source_row = row + k - pad
            valid = active & (source_row >= 0) & (source_row < H)
            value = tl.load(
                x_ptr + c * hw + source_row * W + col,
                mask=valid, other=0.0,
            )
            vertical += value * tl.load(v_w_ptr + c * K + k)
        stride = 1 if PACKED_PW else C
        base_pw = c * 9 if PACKED_PW else c
        acc0 += vertical * tl.load(pw_w_ptr + base_pw + 0 * stride)
        acc1 += vertical * tl.load(pw_w_ptr + base_pw + 1 * stride)
        acc2 += vertical * tl.load(pw_w_ptr + base_pw + 2 * stride)
        acc3 += vertical * tl.load(pw_w_ptr + base_pw + 3 * stride)
        acc4 += vertical * tl.load(pw_w_ptr + base_pw + 4 * stride)
        acc5 += vertical * tl.load(pw_w_ptr + base_pw + 5 * stride)
        acc6 += vertical * tl.load(pw_w_ptr + base_pw + 6 * stride)
        acc7 += vertical * tl.load(pw_w_ptr + base_pw + 7 * stride)
        acc8 += vertical * tl.load(pw_w_ptr + base_pw + 8 * stride)
    if APPLY_SIGMOID:
        acc0 = 1.0 / (1.0 + tl.exp(-acc0))
        acc1 = 1.0 / (1.0 + tl.exp(-acc1))
        acc2 = 1.0 / (1.0 + tl.exp(-acc2))
        acc3 = 1.0 / (1.0 + tl.exp(-acc3))
        acc4 = 1.0 / (1.0 + tl.exp(-acc4))
        acc5 = 1.0 / (1.0 + tl.exp(-acc5))
        acc6 = 1.0 / (1.0 + tl.exp(-acc6))
        acc7 = 1.0 / (1.0 + tl.exp(-acc7))
        acc8 = 1.0 / (1.0 + tl.exp(-acc8))
    tl.store(out_ptr + 0 * hw + spatial, acc0, mask=active)
    tl.store(out_ptr + 1 * hw + spatial, acc1, mask=active)
    tl.store(out_ptr + 2 * hw + spatial, acc2, mask=active)
    tl.store(out_ptr + 3 * hw + spatial, acc3, mask=active)
    tl.store(out_ptr + 4 * hw + spatial, acc4, mask=active)
    tl.store(out_ptr + 5 * hw + spatial, acc5, mask=active)
    tl.store(out_ptr + 6 * hw + spatial, acc6, mask=active)
    tl.store(out_ptr + 7 * hw + spatial, acc7, mask=active)
    tl.store(out_ptr + 8 * hw + spatial, acc8, mask=active)
    if APPLY_OFFSET_CORRECTION:
        correction_index = ((tl.program_id(0) - MASK_PROGRAMS) * BLOCK
                            + tl.arange(0, BLOCK))
        border_count = 2 * H + 2 * W - 4
        border = (tl.program_id(0) >= MASK_PROGRAMS) & (correction_index < border_count)
        is_top = correction_index < W
        is_bottom = (correction_index >= W) & (correction_index < 2 * W)
        is_left = ((correction_index >= 2 * W)
                   & (correction_index < 2 * W + H - 2))
        row_correction = tl.where(
            is_top, 0,
            tl.where(is_bottom, H - 1,
                     tl.where(is_left, 1 + correction_index - 2 * W,
                              1 + correction_index - (2 * W + H - 2))),
        )
        col_correction = tl.where(
            is_top, correction_index,
            tl.where(is_bottom, correction_index - W,
                     tl.where(is_left, 0, W - 1)),
        )
        correction_spatial = row_correction * W + col_correction
        for oc in range(18):
            address = oc * hw + correction_spatial
            value = tl.load(offset_ptr + address, mask=border, other=0.0)
            value += tl.load(offset_correction_ptr + address, mask=border, other=0.0)
            tl.store(offset_ptr + address, value, mask=border)


@triton.jit
def k_vconv_pointwise9_vector(
    x_ptr, v_w_ptr, v_b_ptr, pw_w_ptr, pw_b_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    K: tl.constexpr, BLOCK: tl.constexpr, APPLY_SIGMOID: tl.constexpr,
    VERTICAL_BIAS_FOLDED: tl.constexpr,
):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_lane = tl.arange(0, 8)
    hw = H * W
    active = spatial < hw
    row = spatial // W
    col = spatial - row * W
    acc = tl.load(pw_b_ptr + out_lane)[:, None] + tl.zeros(
        (8, BLOCK), dtype=tl.float32
    )
    acc8 = tl.full((BLOCK,), tl.load(pw_b_ptr + 8), tl.float32)
    pad = K // 2
    for c in range(C):
        vertical = (tl.zeros((BLOCK,), tl.float32) if VERTICAL_BIAS_FOLDED
                    else tl.full((BLOCK,), tl.load(v_b_ptr + c), tl.float32))
        for k in range(K):
            source_row = row + k - pad
            valid = active & (source_row >= 0) & (source_row < H)
            value = tl.load(
                x_ptr + c * hw + source_row * W + col,
                mask=valid, other=0.0,
            )
            vertical += value * tl.load(v_w_ptr + c * K + k)
        weights = tl.load(pw_w_ptr + c * 9 + out_lane)
        acc += weights[:, None] * vertical[None, :]
        acc8 += tl.load(pw_w_ptr + c * 9 + 8) * vertical
    if APPLY_SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))
        acc8 = 1.0 / (1.0 + tl.exp(-acc8))
    tl.store(
        out_ptr + out_lane[:, None] * hw + spatial[None, :], acc,
        mask=active[None, :],
    )
    tl.store(out_ptr + 8 * hw + spatial, acc8, mask=active)


@triton.jit
def k_vconv_pointwise9_vector4(
    x_ptr, v_w_ptr, v_b_ptr, pw_w_ptr, pw_b_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    K: tl.constexpr, BLOCK: tl.constexpr, APPLY_SIGMOID: tl.constexpr,
    VERTICAL_BIAS_FOLDED: tl.constexpr,
):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 4)
    hw = H * W
    active = spatial < hw
    row = spatial // W
    col = spatial - row * W
    acc0 = tl.load(pw_b_ptr + lane)[:, None] + tl.zeros((4, BLOCK), tl.float32)
    acc1 = tl.load(pw_b_ptr + 4 + lane)[:, None] + tl.zeros((4, BLOCK), tl.float32)
    acc8 = tl.full((BLOCK,), tl.load(pw_b_ptr + 8), tl.float32)
    pad = K // 2
    for c in range(C):
        vertical = (tl.zeros((BLOCK,), tl.float32) if VERTICAL_BIAS_FOLDED
                    else tl.full((BLOCK,), tl.load(v_b_ptr + c), tl.float32))
        for k in range(K):
            source_row = row + k - pad
            valid = active & (source_row >= 0) & (source_row < H)
            value = tl.load(
                x_ptr + c * hw + source_row * W + col,
                mask=valid, other=0.0,
            )
            vertical += value * tl.load(v_w_ptr + c * K + k)
        acc0 += tl.load(pw_w_ptr + c * 9 + lane)[:, None] * vertical[None, :]
        acc1 += tl.load(pw_w_ptr + c * 9 + 4 + lane)[:, None] * vertical[None, :]
        acc8 += tl.load(pw_w_ptr + c * 9 + 8) * vertical
    if APPLY_SIGMOID:
        acc0 = 1.0 / (1.0 + tl.exp(-acc0))
        acc1 = 1.0 / (1.0 + tl.exp(-acc1))
        acc8 = 1.0 / (1.0 + tl.exp(-acc8))
    tl.store(out_ptr + lane[:, None] * hw + spatial[None, :], acc0,
             mask=active[None, :])
    tl.store(out_ptr + (4 + lane[:, None]) * hw + spatial[None, :], acc1,
             mask=active[None, :])
    tl.store(out_ptr + 8 * hw + spatial, acc8, mask=active)


def rca_triton(x, rca, input_bias=None, biased_output=None,
               offset=None, offset_correction=None):
    N,C,H,W=x.shape; x=x.contiguous()
    e=rca.excite; conv1,bn,conv2,conv3=e[0],e[1],e[3],e[4]
    BH=triton.next_power_of_2(H); BW=triton.next_power_of_2(W)
    K=conv1.weight.shape[-1]
    if RCA_FOLD_CONSTANTS:
        cache = getattr(rca, "_sm75_folded_constants", None)
        if cache is None:
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w1 = (conv1.weight.reshape(C, K) * scale[:, None]).contiguous()
            b1 = ((conv1.bias - bn.running_mean) * scale + bn.bias).contiguous()
            w2_raw = conv2.weight.reshape(C, conv2.weight.shape[-2]).contiguous()
            pw_raw = conv3.weight.reshape(9, C).contiguous()
            pw_bias = (conv3.bias + torch.mv(pw_raw, conv2.bias)).contiguous()
            pw_weight = pw_raw.transpose(0, 1).contiguous() if RCA_PACK_PW else pw_raw
            cache = (w1, b1, w2_raw, pw_weight, pw_bias)
            rca._sm75_folded_constants = cache
        w1, b1, _, _, _ = cache
    else:
        w1=conv1.weight.reshape(C,K).contiguous(); b1=conv1.bias.contiguous()
    h1=torch.empty_like(x)
    xw=torch.empty(N*C*W,device=x.device,dtype=x.dtype)
    if RCA_FUSED_RH and N == 1:
        k_reduce_hconv_bn_relu[(N*C,)](
            x, input_bias if input_bias is not None else b1,
            biased_output if biased_output is not None else h1,
            xw, w1, b1, bn.running_mean, bn.running_var, bn.weight, bn.bias, h1,
            H=H, W=W, EPS=float(bn.eps), K=K, BH=BH, BW=BW,
            BN_FOLDED=RCA_FOLD_CONSTANTS,
            INPUT_BIAS_FUSED=input_bias is not None,
            STORE_BIASED_INPUT=biased_output is not None,
            num_warps=RCA_RH_WARPS,
        )
    else:
        xh=torch.empty(N*C*H,device=x.device,dtype=x.dtype)
        k_reduce_hw[(N*C,)](x,xh,xw,H,W,BH=BH,BW=BW,num_warps=RCA_REDUCE_WARPS)
        k_hconv_bn_relu[(N*C*H,)](xh,xw,w1,b1,bn.running_mean,bn.running_var,bn.weight,bn.bias,h1,
                                  C,H,W,float(bn.eps),K=K,BW=BW,num_warps=RCA_HCONV_WARPS)
    K2=conv2.weight.shape[-2]
    if RCA_FOLD_CONSTANTS:
        _, _, w2, pw_weight, pw_bias = rca._sm75_folded_constants
        b2 = conv2.bias.contiguous()
    else:
        w2=conv2.weight.reshape(C,K2).contiguous(); b2=conv2.bias.contiguous()
        pw_weight=conv3.weight.reshape(9,C).contiguous(); pw_bias=conv3.bias.contiguous()
    if RCA_FUSED_VPW and N == 1 and conv3.weight.shape[0] == 9:
        out = torch.empty((1, 9, H, W), device=x.device, dtype=x.dtype)
        if RCA_VPW_TINY_THRESHOLD and H <= RCA_VPW_TINY_THRESHOLD:
            block, vpw_warps, vpw_stages = (
                RCA_VPW_TINY_BLOCK, RCA_VPW_TINY_WARPS, RCA_VPW_TINY_STAGES
            )
        elif RCA_VPW_SMALL_THRESHOLD and H <= RCA_VPW_SMALL_THRESHOLD:
            block, vpw_warps, vpw_stages = (
                RCA_VPW_SMALL_BLOCK, RCA_VPW_SMALL_WARPS, RCA_VPW_SMALL_STAGES
            )
        else:
            block, vpw_warps, vpw_stages = RCA_VPW_BLOCK, RCA_VPW_WARPS, RCA_VPW_STAGES
        if RCA_PACK_PW and RCA_VECTOR4_PW:
            k_vconv_pointwise9_vector4[(triton.cdiv(H * W, block),)](
                h1, w2, b2, pw_weight, pw_bias, out,
                H=H, W=W, C=C, K=K2, BLOCK=block,
                APPLY_SIGMOID=RCA_FUSED_SIGMOID,
                VERTICAL_BIAS_FOLDED=RCA_FOLD_CONSTANTS,
                num_warps=vpw_warps, num_stages=vpw_stages,
            )
        elif RCA_PACK_PW and RCA_VECTOR_PW:
            k_vconv_pointwise9_vector[(triton.cdiv(H * W, block),)](
                h1, w2, b2, pw_weight, pw_bias, out,
                H=H, W=W, C=C, K=K2, BLOCK=block,
                APPLY_SIGMOID=RCA_FUSED_SIGMOID,
                VERTICAL_BIAS_FOLDED=RCA_FOLD_CONSTANTS,
                num_warps=vpw_warps, num_stages=vpw_stages,
            )
        else:
            mask_programs = triton.cdiv(H * W, block)
            correction_programs = (
                triton.cdiv(2 * H + 2 * W - 4, block)
                if offset is not None else 0
            )
            k_vconv_pointwise9[(mask_programs + correction_programs,)](
                h1, w2, b2, pw_weight, pw_bias, out,
                offset if offset is not None else out,
                offset_correction if offset_correction is not None else out,
                H=H, W=W, C=C, K=K2, BLOCK=block,
                APPLY_SIGMOID=RCA_FUSED_SIGMOID,
                VERTICAL_BIAS_FOLDED=RCA_FOLD_CONSTANTS, num_warps=vpw_warps,
                PACKED_PW=RCA_PACK_PW,
                APPLY_OFFSET_CORRECTION=offset is not None,
                MASK_PROGRAMS=mask_programs,
                num_stages=vpw_stages,
            )
        return out
    h2=torch.empty_like(x)
    k_vconv[(N*C*W,)](h1,w2,b2,h2,C,H,W,K=K2,BH=BH)
    return torch.nn.functional.conv2d(h2, conv3.weight, conv3.bias)


# ---------- DCN BN+ReLU + scale + gap ----------
@triton.jit
def k_bnrelu_scale_gap(x_ptr, mean_ptr,var_ptr,gam_ptr,bet_ptr, scale_ptr, out_ptr, gap_ptr,
                       C, HW, eps, BLOCK: tl.constexpr, BN_FOLDED: tl.constexpr):
    pid=tl.program_id(0); c=pid%C; base=pid*HW
    idx=tl.arange(0,BLOCK); m=idx<HW
    v=tl.load(x_ptr+base+idx, mask=m, other=0.0)
    if not BN_FOLDED:
        v=(v-tl.load(mean_ptr+c))/tl.sqrt(tl.load(var_ptr+c)+eps)*tl.load(gam_ptr+c)+tl.load(bet_ptr+c)
    v=tl.maximum(v,0.0)*tl.load(scale_ptr+c)
    tl.store(out_ptr+base+idx, v, mask=m)
    tl.store(gap_ptr+pid, tl.sum(tl.where(m,v,0.0))/HW)


@triton.jit
def k_gap_only(x_ptr, gap_ptr, HW: tl.constexpr, BLOCK: tl.constexpr):
    channel = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    active = index < HW
    value = tl.load(x_ptr + channel * HW + index, mask=active, other=0.0)
    tl.store(gap_ptr + channel, tl.sum(tl.where(active, value, 0.0)) / HW)

@triton.jit
def k_apply_tiled(feat_ptr, att_ptr, HW, NT: tl.constexpr, BLOCK: tl.constexpr):
    """Apply per-channel attention in SM75-friendly spatial tiles.

    This preserves the original element-wise operation exactly while avoiding a
    single very large Triton program for high-resolution feature maps.
    """
    pid = tl.program_id(0)
    nc = pid // NT
    tile = pid - nc * NT
    idx = tile * BLOCK + tl.arange(0, BLOCK)
    mask = idx < HW
    value = tl.load(feat_ptr + nc * HW + idx, mask=mask, other=0.0)
    att = tl.load(att_ptr + nc)
    tl.store(feat_ptr + nc * HW + idx, value * att, mask=mask)

def bnrelu_scale_gap(x, bn, scale_w):
    N,C,H,W=x.shape; HW=H*W; x=x.contiguous()
    gap=torch.empty(N*C,device=x.device,dtype=x.dtype)
    BLOCK=triton.next_power_of_2(HW)
    if getattr(bn, "_sm75_dcn_epilogue_fused", False):
        k_gap_only[(N*C,)](x, gap, HW=HW, BLOCK=BLOCK)
        return x, gap.view(N,C)
    out=torch.empty_like(x)
    k_bnrelu_scale_gap[(N*C,)](x,bn.running_mean,bn.running_var,bn.weight,bn.bias,
        scale_w.reshape(1,C).expand(N,C).reshape(-1).contiguous(), out,gap, C,HW,float(bn.eps),
        BLOCK=BLOCK, BN_FOLDED=bool(getattr(bn, "_sm75_dcn_bn_folded", False)))
    return out, gap.view(N,C)

def apply_att(feature, att):
    N,C,H,W=feature.shape; HW=H*W; feature=feature.contiguous()
    # 256 elements gives eight warps and good occupancy on Turing (SM75).
    # NT is constexpr because it participates in program-id decoding.
    BLOCK=256
    NT=triton.cdiv(HW, BLOCK)
    k_apply_tiled[(N*C*NT,)](
        feature, att.reshape(-1).contiguous(), HW, NT=NT, BLOCK=BLOCK,
        num_warps=8,
    )
    return feature


@triton.jit
def k_upsample_bilinear_att(
    x_ptr, att_ptr, out_ptr,
    IH: tl.constexpr, IW: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused align_corners=False bilinear upsample and channel attention."""
    pid = tl.program_id(0)
    nc = pid // tl.cdiv(OH * OW, BLOCK)
    tile = pid - nc * tl.cdiv(OH * OW, BLOCK)
    linear = tile * BLOCK + tl.arange(0, BLOCK)
    mask = linear < OH * OW
    oy = linear // OW
    ox = linear - oy * OW

    # PyTorch align_corners=False source coordinates for integer scale factors.
    fy = (oy.to(tl.float32) + 0.5) * (IH / OH) - 0.5
    fx = (ox.to(tl.float32) + 0.5) * (IW / OW) - 0.5
    # Float-to-int truncates toward zero. align_corners=False coordinates can
    # only be negative in (-0.5, 0), so subtracting one gives floor exactly.
    y0_trunc = fy.to(tl.int32)
    x0_trunc = fx.to(tl.int32)
    y0_raw = tl.where(fy < 0.0, y0_trunc - 1, y0_trunc)
    x0_raw = tl.where(fx < 0.0, x0_trunc - 1, x0_trunc)
    wy = fy - y0_raw.to(tl.float32)
    wx = fx - x0_raw.to(tl.float32)
    y0 = tl.maximum(0, y0_raw)
    x0 = tl.maximum(0, x0_raw)
    y1 = tl.minimum(y0_raw + 1, IH - 1)
    x1 = tl.minimum(x0_raw + 1, IW - 1)

    base = nc * IH * IW
    v00 = tl.load(x_ptr + base + y0 * IW + x0, mask=mask, other=0.0)
    v01 = tl.load(x_ptr + base + y0 * IW + x1, mask=mask, other=0.0)
    v10 = tl.load(x_ptr + base + y1 * IW + x0, mask=mask, other=0.0)
    v11 = tl.load(x_ptr + base + y1 * IW + x1, mask=mask, other=0.0)
    top = v00 + (v01 - v00) * wx
    bot = v10 + (v11 - v10) * wx
    value = top + (bot - top) * wy
    value *= tl.load(att_ptr + nc)
    tl.store(out_ptr + nc * OH * OW + linear, value, mask=mask)


def upsample_bilinear_att(feature, att, output_hw):
    n, c, ih, iw = feature.shape
    oh, ow = output_hw
    block = UPSAMPLE_TILE
    tiles = triton.cdiv(oh * ow, block)
    out = torch.empty((n, c, oh, ow), device=feature.device, dtype=feature.dtype)
    k_upsample_bilinear_att[(n * c * tiles,)](
        feature.contiguous(), att.reshape(-1).contiguous(), out,
        IH=ih, IW=iw, OH=oh, OW=ow, BLOCK=block,
        num_warps=UPSAMPLE_WARPS,
    )
    return out


@triton.jit
def k_cat_residual_att0(
    b0_ptr, b1_ptr, b2_ptr, att0_ptr, residual_ptr, out_ptr,
    N: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
    HW: tl.constexpr, BLOCK: tl.constexpr,
):
    """Fuse branch-0 attention, three-way channel concat, and residual add."""
    total_c: tl.constexpr = C0 + C1 + C2
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total: tl.constexpr = N * total_c * HW
    mask = linear < total
    spatial = linear % HW
    nc = linear // HW
    c = nc % total_c
    n = nc // total_c

    is0 = c < C0
    is1 = (c >= C0) & (c < C0 + C1)
    c0 = c
    c1 = c - C0
    c2 = c - C0 - C1
    v0 = tl.load(b0_ptr + (n * C0 + c0) * HW + spatial, mask=mask & is0, other=0.0)
    a0 = tl.load(att0_ptr + n * C0 + c0, mask=mask & is0, other=0.0)
    v1 = tl.load(b1_ptr + (n * C1 + c1) * HW + spatial, mask=mask & is1, other=0.0)
    v2 = tl.load(b2_ptr + (n * C2 + c2) * HW + spatial, mask=mask & ~(is0 | is1), other=0.0)
    branch = tl.where(is0, v0 * a0, tl.where(is1, v1, v2))
    residual = tl.load(residual_ptr + linear, mask=mask, other=0.0)
    tl.store(out_ptr + linear, branch + residual, mask=mask)


def cat_residual_att0(branches, att0, residual):
    b0, b1, b2 = branches
    n, total_c, h, w = residual.shape
    c0, c1, c2 = b0.shape[1], b1.shape[1], b2.shape[1]
    assert c0 + c1 + c2 == total_c
    hw = h * w
    block = CAT_TILE
    out = torch.empty_like(residual)
    k_cat_residual_att0[(triton.cdiv(n * total_c * hw, block),)](
        b0.contiguous(), b1.contiguous(), b2.contiguous(),
        att0.reshape(-1).contiguous(), residual.contiguous(), out,
        N=n, C0=c0, C1=c1, C2=c2, HW=hw, BLOCK=block, num_warps=CAT_WARPS,
    )
    return out


@triton.jit
def k_offset_conv3x3(
    x_ptr, packed_weight_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr,
):
    co = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    spatial = tl.program_id(1) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    hw = H * W
    active_c = co < COUT
    active_s = spatial < hw
    oy = spatial // W
    ox = spatial - oy * W
    acc = tl.zeros((CHANNEL_BLOCK, SPATIAL_BLOCK), tl.float32)
    for ci in range(CIN):
        for k in range(9):
            ky = k // 3
            kx = k - ky * 3
            iy = oy + ky - 1
            ix = ox + kx - 1
            valid = active_s & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            value = tl.load(x_ptr + ci * hw + iy * W + ix, mask=valid, other=0.0)
            weight = tl.load(
                packed_weight_ptr + (ci * 9 + k) * COUT + co,
                mask=active_c, other=0.0,
            )
            acc += weight[:, None] * value[None, :]
    tl.store(
        out_ptr + co[:, None] * hw + spatial[None, :], acc,
        mask=active_c[:, None] & active_s[None, :],
    )


def offset_conv3x3(x, conv):
    n, cin, h, w = x.shape
    assert n == 1 and conv.out_channels == 18 and conv.kernel_size == (3, 3)
    packed = getattr(conv, "_sm75_packed_weight", None)
    if packed is None:
        packed = conv.weight.reshape(18, cin * 9).transpose(0, 1).contiguous()
        conv._sm75_packed_weight = packed
    out = torch.empty((1, 18, h, w), device=x.device, dtype=x.dtype)
    sb = OFFSET_SPATIAL_BLOCK
    cb = OFFSET_CHANNEL_BLOCK
    k_offset_conv3x3[(triton.cdiv(18, cb), triton.cdiv(h * w, sb))](
        x, packed, out, H=h, W=w, CIN=cin, COUT=18,
        SPATIAL_BLOCK=sb, CHANNEL_BLOCK=cb, num_warps=OFFSET_WARPS,
    )
    return out

# ---------- patched forwards ----------
def dcn_fwd_tri(self, x):
    dcn_backend = os.environ.get("KADC_DCN_BACKEND", "torchvision")
    fuse_offset_bias = (
        dcn_backend == "direct"
        and os.environ.get("KADC_DCN_FUSE_OFFSET_BIAS", "0") == "1"
    )
    fuse_input_bias = (
        dcn_backend == "direct"
        and os.environ.get("KADC_DCN_FUSE_INPUT_BIAS", "0") == "1"
        and getattr(self, "_sm75_input_bias", None) is not None
        and int(os.environ.get("KADC_DCN_INPUT_BIAS_MIN_H", "0")) <= x.shape[2]
        and x.shape[2] <= int(os.environ.get("KADC_DCN_INPUT_BIAS_MAX_H", "1000000"))
    )
    input_bias = self._sm75_input_bias if fuse_input_bias else None
    if fuse_offset_bias:
        conv = self.conv_offset
        if OFFSET_TRITON:
            offset = offset_conv3x3(x, conv)
        else:
            offset = torch.nn.functional.conv2d(
                x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups
            )
    else:
        offset=self.conv_offset(x)
    offset_correction = None
    effective_offset_bias = self.conv_offset.bias
    rca_offset_correction = (
        fuse_input_bias and RCA_FUSE_OFFSET_CORRECTION
        and x.shape[2] >= int(os.environ.get("KADC_RCA_OFFSET_CORRECTION_MIN_H", "0"))
    )
    if fuse_input_bias:
        key = (x.shape[2], x.shape[3])
        cache = getattr(self, "_sm75_offset_correction", None)
        if cache is None or cache[0] != key:
            bias_image = input_bias.view(1, -1, 1, 1).expand_as(x)
            correction = torch.nn.functional.conv2d(
                bias_image, self.conv_offset.weight, None, self.conv_offset.stride,
                self.conv_offset.padding, self.conv_offset.dilation, self.conv_offset.groups,
            ).contiguous()
            if rca_offset_correction:
                interior = correction[0, :, x.shape[2] // 2, x.shape[3] // 2].contiguous()
                correction = (correction - interior.view(1, -1, 1, 1)).contiguous()
                effective = (self.conv_offset.bias + interior).contiguous()
                cache = (key, correction, effective)
            else:
                cache = (key, correction, self.conv_offset.bias)
            self._sm75_offset_correction = cache
        offset_correction = cache[1]
        effective_offset_bias = cache[2]
    biased_x = torch.empty_like(x) if fuse_input_bias else None
    mask=rca_triton(
        x, self.RCA_mask, input_bias=input_bias, biased_output=biased_x,
        offset=offset if rca_offset_correction else None,
        offset_correction=offset_correction if rca_offset_correction else None,
    ) if self.mask else None
    if mask is not None and not (RCA_FUSED_VPW and RCA_FUSED_SIGMOID):
        mask=torch.sigmoid(mask)
    if dcn_backend in ("direct", "im2col"):
        from kadc_dcn_direct import deform_conv2d_direct, deform_conv2d_im2col
        implementation = deform_conv2d_direct if dcn_backend == "direct" else deform_conv2d_im2col
        extra = {}
        if dcn_backend == "direct" and os.environ.get("KADC_DCN_PACK_WEIGHT", "0") == "1":
            packed = getattr(self, "_sm75_packed_dcn_weight", None)
            if packed is None:
                packed = self.weight.permute(1, 2, 3, 0).contiguous().reshape(-1, self.weight.shape[0])
                self._sm75_packed_dcn_weight = packed
            extra["packed_weight"] = packed
        if fuse_offset_bias:
            extra["offset_bias"] = effective_offset_bias
        if fuse_input_bias:
            if not rca_offset_correction:
                extra["input_bias"] = input_bias
                extra["offset_correction"] = offset_correction
                extra["input_already_biased"] = True
        if os.environ.get("KADC_DCN_FUSE_EPILOGUE", "0") == "1":
            cache = getattr(self, "_sm75_post_epilogue", None)
            if cache is None:
                bn = self._sm75_post_bn
                post_mul = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
                post_add = (bn.bias - bn.running_mean * post_mul).contiguous()
                post_scale = self._sm75_post_scale.reshape(-1).contiguous()
                cache = (post_mul, post_add, post_scale)
                self._sm75_post_epilogue = cache
            extra["post_mul"], extra["post_add"], extra["post_scale"] = cache
        return implementation(input=biased_x if biased_x is not None else x,
            offset=offset, weight=self.weight, bias=self.bias,
            padding=self.padding, mask=mask, stride=self.stride, **extra)
    return torchvision.ops.deform_conv2d(input=x,offset=offset,weight=self.weight,bias=self.bias,
        padding=self.padding,mask=mask,stride=self.stride)

def kadc4_fwd_tri(self, x):
    fl=[]
    for i,(conv,fc,w) in enumerate(zip(self.convs,self.fc,self.weight)):
        x_d=torch.nn.functional.avg_pool2d(x,kernel_size=int(2**i)) if i>0 else x
        t=conv[1](conv[0](x_d))                 # Conv1x1 -> DCN
        feature, gap = bnrelu_scale_gap(t, conv[2], w[0].weight)   # BN+ReLU+scale+gap
        feature = apply_att(feature, fc(gap))
        if i>0: feature=torch.nn.functional.interpolate(feature,size=x.size()[2:],mode='bilinear',align_corners=False)
        fl.append(feature)
    return torch.cat(fl,dim=1)+x

def patch(m):
    for _,mod in m.named_modules():
        t=type(mod).__name__
        if t in {"DCN_RCA", "DSC"}: mod.forward=types.MethodType(dcn_fwd_tri,mod)
        elif t in {"KADC4", "KADC4_m", "KADConv"}: mod.forward=types.MethodType(kadc4_fwd_tri,mod)
        elif (FUSE_CONV_BIAS_SILU and t == "Conv"
              and isinstance(getattr(mod, "act", None), torch.nn.SiLU)
              and getattr(getattr(mod, "conv", None), "bias", None) is not None):
            conv = mod.conv
            if (CONV1X1_TRITON and conv.kernel_size == (1, 1)
                    and conv.stride == (1, 1) and conv.groups == 1):
                mod.forward = types.MethodType(conv1x1_fwd_bias_silu, mod)
            else:
                mod.forward=types.MethodType(conv_fwd_bias_silu,mod)
    return m

# ---------- fused SE MLP: gap[N,C] -> Linear->ReLU->Linear->Sigmoid -> [N,C] ----------
@triton.jit
def k_se_mlp(gap_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
             C, Hd, BC: tl.constexpr, BH: tl.constexpr):
    n = tl.program_id(0)
    rc = tl.arange(0, BC); mc = rc < C
    rh = tl.arange(0, BH); mh = rh < Hd
    gap = tl.load(gap_ptr + n*C + rc, mask=mc, other=0.0)              # [BC]
    w1 = tl.load(w1_ptr + rh[:,None]*C + rc[None,:], mask=mh[:,None]&mc[None,:], other=0.0)  # [BH,BC]
    hid = tl.sum(w1 * gap[None,:], axis=1) + tl.load(b1_ptr+rh, mask=mh, other=0.0)          # [BH]
    hid = tl.maximum(hid, 0.0)
    w2 = tl.load(w2_ptr + rc[:,None]*Hd + rh[None,:], mask=mc[:,None]&mh[None,:], other=0.0) # [BC,BH]
    o = tl.sum(w2 * hid[None,:], axis=1) + tl.load(b2_ptr+rc, mask=mc, other=0.0)            # [BC]
    o = 1.0 / (1.0 + tl.exp(-o))
    tl.store(out_ptr + n*C + rc, o, mask=mc)

def se_mlp(gap, fc):
    N,C = gap.shape
    l1,l2 = fc[0], fc[2]
    Hd = l1.weight.shape[0]
    out = torch.empty(N,C, device=gap.device, dtype=gap.dtype)
    BC = triton.next_power_of_2(C); BH = triton.next_power_of_2(Hd)
    k_se_mlp[(N,)](gap.contiguous(), l1.weight.contiguous(), l1.bias.contiguous(),
                   l2.weight.contiguous(), l2.bias.contiguous(), out, C, Hd,
                   BC=BC, BH=BH, num_warps=SE_MLP_WARPS)
    return out

def kadc4_fwd_tri2(self, x):
    fl=[]
    att0 = None
    for i,(conv,fc,w) in enumerate(zip(self.convs,self.fc,self.weight)):
        x_d=torch.nn.functional.avg_pool2d(x,kernel_size=int(2**i)) if i>0 else x
        t=conv[1](conv[0](x_d))
        feature, gap = bnrelu_scale_gap(t, conv[2], w[0].weight)
        att = se_mlp(gap, fc)
        if i>0:
            feature = upsample_bilinear_att(feature, att, x.shape[2:])
        else:
            att0 = att
        fl.append(feature)
    return cat_residual_att0(fl, att0, x)

def patch(m):
    for _,mod in m.named_modules():
        t=type(mod).__name__
        if t in {"DCN_RCA", "DSC"}: mod.forward=types.MethodType(dcn_fwd_tri,mod)
        elif t in {"KADC4", "KADC4_m", "KADConv"}: mod.forward=types.MethodType(kadc4_fwd_tri2,mod)
        elif (FUSE_CONV_BIAS_SILU and FUSE_RESIDUAL_ADD
              and t == "RepNBottleneck" and getattr(mod, "add", False)
              and isinstance(getattr(mod.cv2, "act", None), torch.nn.SiLU)
              and getattr(getattr(mod.cv2, "conv", None), "bias", None) is not None):
            mod.forward = types.MethodType(repn_bottleneck_fused_residual, mod)
        elif (FUSE_CONV_BIAS_SILU and t == "Conv"
              and isinstance(getattr(mod, "act", None), torch.nn.SiLU)
              and getattr(getattr(mod, "conv", None), "bias", None) is not None):
            mod.forward=types.MethodType(conv_fwd_bias_silu,mod)
    return m
