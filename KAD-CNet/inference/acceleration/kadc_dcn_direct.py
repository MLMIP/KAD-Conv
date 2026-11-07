"""Direct FP32 modulated DCNv2 kernel for the KADC bs1 inference shapes.

This avoids torchvision's im2col workspace, zero-fill kernels, and tiny GEMMs.
It intentionally supports only the exact KADC deployment case: N=1, 3x3,
stride=1, padding=1, dilation=1, one weight/offset group.
"""

import os

import torch
import triton
import triton.language as tl


BLOCK = int(os.environ.get("KADC_DCN_BLOCK", "64"))
WARPS = int(os.environ.get("KADC_DCN_WARPS", "4"))
TILED = os.environ.get("KADC_DCN_TILED", "0") == "1"
SPATIAL_BLOCK = int(os.environ.get("KADC_DCN_SPATIAL_BLOCK", "16"))
CHANNEL_BLOCK = int(os.environ.get("KADC_DCN_CHANNEL_BLOCK", "8"))
IM2COL_BLOCK = int(os.environ.get("KADC_DCN_IM2COL_BLOCK", "128"))
SMALL_THRESHOLD = int(os.environ.get("KADC_DCN_SMALL_THRESHOLD", "0"))
SMALL_SPATIAL_BLOCK = int(os.environ.get("KADC_DCN_SMALL_SPATIAL_BLOCK", str(SPATIAL_BLOCK)))
SMALL_CHANNEL_BLOCK = int(os.environ.get("KADC_DCN_SMALL_CHANNEL_BLOCK", str(CHANNEL_BLOCK)))
SMALL_WARPS = int(os.environ.get("KADC_DCN_SMALL_WARPS", str(WARPS)))
TINY_THRESHOLD = int(os.environ.get("KADC_DCN_TINY_THRESHOLD", "0"))
TINY_SPATIAL_BLOCK = int(os.environ.get("KADC_DCN_TINY_SPATIAL_BLOCK", str(SMALL_SPATIAL_BLOCK)))
TINY_CHANNEL_BLOCK = int(os.environ.get("KADC_DCN_TINY_CHANNEL_BLOCK", str(SMALL_CHANNEL_BLOCK)))
TINY_WARPS = int(os.environ.get("KADC_DCN_TINY_WARPS", str(SMALL_WARPS)))
SHAPE_TILES = os.environ.get("KADC_DCN_SHAPE_TILES", "0") == "1"
FUSED_OFFSET_SPATIAL_BLOCK = int(os.environ.get("KADC_DCN_FUSED_OFFSET_SPATIAL_BLOCK", "8"))
FUSED_OFFSET_CHANNEL_BLOCK = int(os.environ.get("KADC_DCN_FUSED_OFFSET_CHANNEL_BLOCK", "32"))
FUSED_OFFSET_WARPS = int(os.environ.get("KADC_DCN_FUSED_OFFSET_WARPS", "4"))


@triton.jit
def _direct_dcn3x3_fp32(
    x_ptr,
    offset_ptr,
    mask_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    co = tl.program_id(0)
    spatial = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw = H * W
    active = spatial < hw
    oy = spatial // W
    ox = spatial - oy * W
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for ci in range(CIN):
        for k in range(9):
            ky = k // 3
            kx = k - ky * 3
            off_y = tl.load(offset_ptr + (2 * k) * hw + spatial, mask=active, other=0.0)
            off_x = tl.load(offset_ptr + (2 * k + 1) * hw + spatial, mask=active, other=0.0)
            modulation = tl.load(mask_ptr + k * hw + spatial, mask=active, other=0.0)

            iy = oy.to(tl.float32) + ky - 1.0 + off_y
            ix = ox.to(tl.float32) + kx - 1.0 + off_x
            valid_coord = active & (iy > -1.0) & (iy < H) & (ix > -1.0) & (ix < W)
            # Triton 2.2 has no tl.floor. Valid negative coordinates are only
            # in (-1, 0), whose mathematical floor is exactly -1.
            y_trunc = iy.to(tl.int32)
            x_trunc = ix.to(tl.int32)
            y0 = tl.where(iy < 0.0, -1, y_trunc)
            x0 = tl.where(ix < 0.0, -1, x_trunc)
            y1 = y0 + 1
            x1 = x0 + 1
            ly = iy - y0.to(tl.float32)
            lx = ix - x0.to(tl.float32)
            hy = 1.0 - ly
            hx = 1.0 - lx
            channel_base = ci * hw

            v00 = tl.load(
                x_ptr + channel_base + y0 * W + x0,
                mask=valid_coord & (y0 >= 0) & (x0 >= 0),
                other=0.0,
            )
            v01 = tl.load(
                x_ptr + channel_base + y0 * W + x1,
                mask=valid_coord & (y0 >= 0) & (x1 < W),
                other=0.0,
            )
            v10 = tl.load(
                x_ptr + channel_base + y1 * W + x0,
                mask=valid_coord & (y1 < H) & (x0 >= 0),
                other=0.0,
            )
            v11 = tl.load(
                x_ptr + channel_base + y1 * W + x1,
                mask=valid_coord & (y1 < H) & (x1 < W),
                other=0.0,
            )
            sampled = v00 * hy * hx + v01 * hy * lx + v10 * ly * hx + v11 * ly * lx
            kernel_weight = tl.load(weight_ptr + (co * CIN + ci) * 9 + k)
            acc += sampled * modulation * kernel_weight

    acc += tl.load(bias_ptr + co)
    tl.store(out_ptr + co * hw + spatial, acc, mask=active)


@triton.jit
def _direct_dcn3x3_tiled_fp32(
    x_ptr, input_bias_ptr, offset_ptr, offset_bias_ptr, offset_correction_ptr,
    mask_ptr, weight_ptr, bias_ptr, post_mul_ptr, post_add_ptr, post_scale_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr,
    WEIGHT_PACKED: tl.constexpr,
    OFFSET_BIAS_FUSED: tl.constexpr,
    INPUT_BIAS_FUSED: tl.constexpr,
    SAMPLE_INPUT_BIAS: tl.constexpr,
    FUSE_EPILOGUE: tl.constexpr,
):
    co = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
    spatial = tl.program_id(1) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    hw = H * W
    active_s = spatial < hw
    active_c = co < COUT
    oy = spatial // W
    ox = spatial - oy * W
    acc = tl.zeros((CHANNEL_BLOCK, SPATIAL_BLOCK), dtype=tl.float32)

    # One bilinear sample is shared by CHANNEL_BLOCK output channels.
    for k in range(9):
        ky = k // 3
        kx = k - ky * 3
        off_y = tl.load(offset_ptr + (2 * k) * hw + spatial, mask=active_s, other=0.0)
        off_x = tl.load(offset_ptr + (2 * k + 1) * hw + spatial, mask=active_s, other=0.0)
        if OFFSET_BIAS_FUSED:
            off_y += tl.load(offset_bias_ptr + 2 * k)
            off_x += tl.load(offset_bias_ptr + 2 * k + 1)
        if INPUT_BIAS_FUSED:
            off_y += tl.load(offset_correction_ptr + (2 * k) * hw + spatial,
                             mask=active_s, other=0.0)
            off_x += tl.load(offset_correction_ptr + (2 * k + 1) * hw + spatial,
                             mask=active_s, other=0.0)
        modulation = tl.load(mask_ptr + k * hw + spatial, mask=active_s, other=0.0)
        iy = oy.to(tl.float32) + ky - 1.0 + off_y
        ix = ox.to(tl.float32) + kx - 1.0 + off_x
        valid_coord = active_s & (iy > -1.0) & (iy < H) & (ix > -1.0) & (ix < W)
        y_trunc = iy.to(tl.int32)
        x_trunc = ix.to(tl.int32)
        y0 = tl.where(iy < 0.0, -1, y_trunc)
        x0 = tl.where(ix < 0.0, -1, x_trunc)
        y1 = y0 + 1
        x1 = x0 + 1
        ly = iy - y0.to(tl.float32)
        lx = ix - x0.to(tl.float32)
        hy = 1.0 - ly
        hx = 1.0 - lx
        for ci in range(CIN):
            channel_base = ci * hw
            v00 = tl.load(x_ptr + channel_base + y0 * W + x0,
                          mask=valid_coord & (y0 >= 0) & (x0 >= 0), other=0.0)
            v01 = tl.load(x_ptr + channel_base + y0 * W + x1,
                          mask=valid_coord & (y0 >= 0) & (x1 < W), other=0.0)
            v10 = tl.load(x_ptr + channel_base + y1 * W + x0,
                          mask=valid_coord & (y1 < H) & (x0 >= 0), other=0.0)
            v11 = tl.load(x_ptr + channel_base + y1 * W + x1,
                          mask=valid_coord & (y1 < H) & (x1 < W), other=0.0)
            if SAMPLE_INPUT_BIAS:
                input_bias = tl.load(input_bias_ptr + ci)
                v00 = tl.where(valid_coord & (y0 >= 0) & (x0 >= 0), v00 + input_bias, 0.0)
                v01 = tl.where(valid_coord & (y0 >= 0) & (x1 < W), v01 + input_bias, 0.0)
                v10 = tl.where(valid_coord & (y1 < H) & (x0 >= 0), v10 + input_bias, 0.0)
                v11 = tl.where(valid_coord & (y1 < H) & (x1 < W), v11 + input_bias, 0.0)
            sampled = (v00 * hy * hx + v01 * hy * lx +
                       v10 * ly * hx + v11 * ly * lx) * modulation
            if WEIGHT_PACKED:
                kernel_weight = tl.load(
                    weight_ptr + (ci * 9 + k) * COUT + co,
                    mask=active_c, other=0.0,
                )
            else:
                kernel_weight = tl.load(
                    weight_ptr + (co * CIN + ci) * 9 + k,
                    mask=active_c, other=0.0,
                )
            acc += kernel_weight[:, None] * sampled[None, :]
    acc += tl.load(bias_ptr + co, mask=active_c, other=0.0)[:, None]
    if FUSE_EPILOGUE:
        acc = acc * tl.load(post_mul_ptr + co, mask=active_c, other=0.0)[:, None]
        acc += tl.load(post_add_ptr + co, mask=active_c, other=0.0)[:, None]
        acc = tl.maximum(acc, 0.0)
        acc *= tl.load(post_scale_ptr + co, mask=active_c, other=0.0)[:, None]
    tl.store(out_ptr + co[:, None] * hw + spatial[None, :], acc,
             mask=active_c[:, None] & active_s[None, :])


@triton.jit
def _offset_and_direct_dcn3x3_fp32(
    x_ptr, mask_ptr, offset_weight_ptr, offset_bias_ptr,
    weight_ptr, bias_ptr, post_mul_ptr, post_add_ptr, post_scale_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, CIN: tl.constexpr, COUT: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr, CHANNEL_BLOCK: tl.constexpr,
    WEIGHT_PACKED: tl.constexpr, FUSE_EPILOGUE: tl.constexpr,
):
    """Fuse the trained 3x3 offset convolution into the direct DCNv2 kernel.

    KADC branches have COUT <= 22, so CHANNEL_BLOCK=32 makes one program own
    every output channel.  Offset values are consequently computed once per
    spatial tile rather than repeated across output-channel programs.
    """
    co = tl.arange(0, CHANNEL_BLOCK)
    spatial = tl.program_id(0) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    hw = H * W
    active_s = spatial < hw
    active_c = co < COUT
    oy = spatial // W
    ox = spatial - oy * W
    acc = tl.zeros((CHANNEL_BLOCK, SPATIAL_BLOCK), dtype=tl.float32)

    for k in range(9):
        offset_y = tl.full((SPATIAL_BLOCK,), tl.load(offset_bias_ptr + 2 * k), tl.float32)
        offset_x = tl.full((SPATIAL_BLOCK,), tl.load(offset_bias_ptr + 2 * k + 1), tl.float32)
        # Exact stride-1/pad-1 Conv2d used by DCN_RCA.conv_offset.
        for ci in range(CIN):
            for q in range(9):
                qy = q // 3
                qx = q - qy * 3
                source_y = oy + qy - 1
                source_x = ox + qx - 1
                valid_source = (
                    active_s & (source_y >= 0) & (source_y < H)
                    & (source_x >= 0) & (source_x < W)
                )
                value = tl.load(
                    x_ptr + ci * hw + source_y * W + source_x,
                    mask=valid_source, other=0.0,
                )
                offset_y += value * tl.load(
                    offset_weight_ptr + ((2 * k) * CIN + ci) * 9 + q
                )
                offset_x += value * tl.load(
                    offset_weight_ptr + ((2 * k + 1) * CIN + ci) * 9 + q
                )

        modulation = tl.load(mask_ptr + k * hw + spatial, mask=active_s, other=0.0)
        ky = k // 3
        kx = k - ky * 3
        iy = oy.to(tl.float32) + ky - 1.0 + offset_y
        ix = ox.to(tl.float32) + kx - 1.0 + offset_x
        valid_coord = active_s & (iy > -1.0) & (iy < H) & (ix > -1.0) & (ix < W)
        y_trunc = iy.to(tl.int32)
        x_trunc = ix.to(tl.int32)
        y0 = tl.where(iy < 0.0, -1, y_trunc)
        x0 = tl.where(ix < 0.0, -1, x_trunc)
        y1 = y0 + 1
        x1 = x0 + 1
        ly = iy - y0.to(tl.float32)
        lx = ix - x0.to(tl.float32)
        hy = 1.0 - ly
        hx = 1.0 - lx
        for ci in range(CIN):
            base = ci * hw
            v00 = tl.load(x_ptr + base + y0 * W + x0,
                          mask=valid_coord & (y0 >= 0) & (x0 >= 0), other=0.0)
            v01 = tl.load(x_ptr + base + y0 * W + x1,
                          mask=valid_coord & (y0 >= 0) & (x1 < W), other=0.0)
            v10 = tl.load(x_ptr + base + y1 * W + x0,
                          mask=valid_coord & (y1 < H) & (x0 >= 0), other=0.0)
            v11 = tl.load(x_ptr + base + y1 * W + x1,
                          mask=valid_coord & (y1 < H) & (x1 < W), other=0.0)
            sampled = (v00 * hy * hx + v01 * hy * lx +
                       v10 * ly * hx + v11 * ly * lx) * modulation
            if WEIGHT_PACKED:
                kernel_weight = tl.load(
                    weight_ptr + (ci * 9 + k) * COUT + co,
                    mask=active_c, other=0.0,
                )
            else:
                kernel_weight = tl.load(
                    weight_ptr + (co * CIN + ci) * 9 + k,
                    mask=active_c, other=0.0,
                )
            acc += kernel_weight[:, None] * sampled[None, :]

    acc += tl.load(bias_ptr + co, mask=active_c, other=0.0)[:, None]
    if FUSE_EPILOGUE:
        acc = acc * tl.load(post_mul_ptr + co, mask=active_c, other=0.0)[:, None]
        acc += tl.load(post_add_ptr + co, mask=active_c, other=0.0)[:, None]
        acc = tl.maximum(acc, 0.0)
        acc *= tl.load(post_scale_ptr + co, mask=active_c, other=0.0)[:, None]
    tl.store(out_ptr + co[:, None] * hw + spatial[None, :], acc,
             mask=active_c[:, None] & active_s[None, :])


@triton.jit
def _deform_im2col3x3_fp32(
    x_ptr, offset_ptr, mask_ptr, columns_ptr,
    H: tl.constexpr, W: tl.constexpr, CIN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    sample_plane = tl.program_id(0)
    ci = sample_plane // 9
    k = sample_plane - ci * 9
    spatial = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw = H * W
    active = spatial < hw
    oy = spatial // W
    ox = spatial - oy * W
    ky = k // 3
    kx = k - ky * 3
    off_y = tl.load(offset_ptr + (2 * k) * hw + spatial, mask=active, other=0.0)
    off_x = tl.load(offset_ptr + (2 * k + 1) * hw + spatial, mask=active, other=0.0)
    modulation = tl.load(mask_ptr + k * hw + spatial, mask=active, other=0.0)
    iy = oy.to(tl.float32) + ky - 1.0 + off_y
    ix = ox.to(tl.float32) + kx - 1.0 + off_x
    valid_coord = active & (iy > -1.0) & (iy < H) & (ix > -1.0) & (ix < W)
    y_trunc = iy.to(tl.int32)
    x_trunc = ix.to(tl.int32)
    y0 = tl.where(iy < 0.0, -1, y_trunc)
    x0 = tl.where(ix < 0.0, -1, x_trunc)
    y1 = y0 + 1
    x1 = x0 + 1
    ly = iy - y0.to(tl.float32)
    lx = ix - x0.to(tl.float32)
    hy = 1.0 - ly
    hx = 1.0 - lx
    channel_base = ci * hw
    v00 = tl.load(x_ptr + channel_base + y0 * W + x0,
                  mask=valid_coord & (y0 >= 0) & (x0 >= 0), other=0.0)
    v01 = tl.load(x_ptr + channel_base + y0 * W + x1,
                  mask=valid_coord & (y0 >= 0) & (x1 < W), other=0.0)
    v10 = tl.load(x_ptr + channel_base + y1 * W + x0,
                  mask=valid_coord & (y1 < H) & (x0 >= 0), other=0.0)
    v11 = tl.load(x_ptr + channel_base + y1 * W + x1,
                  mask=valid_coord & (y1 < H) & (x1 < W), other=0.0)
    sampled = (v00 * hy * hx + v01 * hy * lx +
               v10 * ly * hx + v11 * ly * lx) * modulation
    tl.store(columns_ptr + sample_plane * hw + spatial, sampled, mask=active)


def deform_conv2d_direct(input, offset, weight, bias, padding, mask, stride,
                         packed_weight=None, offset_bias=None,
                         input_bias=None, offset_correction=None,
                         input_already_biased=False,
                         post_mul=None, post_add=None, post_scale=None,
                         fused_offset_weight=None, fused_offset_bias=None):
    n, cin, h, w = input.shape
    cout, weight_cin, kh, kw = weight.shape
    stride_pair = stride if isinstance(stride, tuple) else (stride, stride)
    padding_pair = padding if isinstance(padding, tuple) else (padding, padding)
    if not (
        n == 1
        and input.dtype in (torch.float32, torch.float16)
        and weight.dtype == input.dtype
        and cin == weight_cin
        and kh == kw == 3
        and stride_pair == (1, 1)
        and padding_pair == (1, 1)
        and (fused_offset_weight is not None or offset.shape == (1, 18, h, w))
        and mask is not None
        and mask.shape == (1, 9, h, w)
    ):
        raise ValueError("direct DCN received an unsupported deployment shape")
    output = torch.empty((1, cout, h, w), device=input.device, dtype=input.dtype)
    if fused_offset_weight is not None:
        if input_bias is not None:
            raise ValueError("fused offset-conv path currently requires an explicitly biased input")
        if FUSED_OFFSET_CHANNEL_BLOCK < cout:
            raise ValueError("fused offset channel block must cover every KADC output channel")
        block = FUSED_OFFSET_SPATIAL_BLOCK
        _offset_and_direct_dcn3x3_fp32[(triton.cdiv(h * w, block),)](
            input.contiguous(), mask.contiguous(), fused_offset_weight.contiguous(),
            fused_offset_bias.contiguous(),
            packed_weight if packed_weight is not None else weight.contiguous(),
            bias.contiguous(),
            post_mul if post_mul is not None else bias,
            post_add if post_add is not None else bias,
            post_scale if post_scale is not None else bias,
            output, H=h, W=w, CIN=cin, COUT=cout,
            SPATIAL_BLOCK=block, CHANNEL_BLOCK=FUSED_OFFSET_CHANNEL_BLOCK,
            WEIGHT_PACKED=packed_weight is not None,
            FUSE_EPILOGUE=post_mul is not None,
            num_warps=FUSED_OFFSET_WARPS,
        )
        return output
    if TILED:
        # The real KADC-S checkpoint has nine recurring (C,H) shapes.  A raw
        # CUDA-Graph microbenchmark shows two tile families cover their optima:
        # 16x8 for the listed narrow low-resolution cases, 32x4 otherwise.
        # Unknown shapes retain the generic threshold dispatch so other model
        # sizes never silently inherit an unmeasured S-only tile.
        measured_shape = (cin, h) in {
            (12, 64), (10, 32), (16, 32), (10, 16), (16, 16),
            (22, 16), (16, 8), (21, 8), (21, 4),
        }
        if SHAPE_TILES and measured_shape:
            use_16x8 = (cin, h) in {(10, 16), (16, 16), (22, 16), (16, 8)}
            spatial_block = 16 if use_16x8 else 32
            channel_block = 8 if use_16x8 else 4
            warps = 2
        else:
            use_tiny = TINY_THRESHOLD > 0 and h <= TINY_THRESHOLD
            use_small = SMALL_THRESHOLD > 0 and h <= SMALL_THRESHOLD
            spatial_block = (TINY_SPATIAL_BLOCK if use_tiny else
                             SMALL_SPATIAL_BLOCK if use_small else SPATIAL_BLOCK)
            channel_block = (TINY_CHANNEL_BLOCK if use_tiny else
                             SMALL_CHANNEL_BLOCK if use_small else CHANNEL_BLOCK)
            warps = TINY_WARPS if use_tiny else SMALL_WARPS if use_small else WARPS
        _direct_dcn3x3_tiled_fp32[(triton.cdiv(cout, channel_block), triton.cdiv(h * w, spatial_block))](
            input.contiguous(), input_bias if input_bias is not None else bias,
            offset.contiguous(),
            offset_bias if offset_bias is not None else bias,
            offset_correction if offset_correction is not None else offset,
            mask.contiguous(),
            packed_weight if packed_weight is not None else weight.contiguous(),
            bias.contiguous(),
            post_mul if post_mul is not None else bias,
            post_add if post_add is not None else bias,
            post_scale if post_scale is not None else bias,
            output, H=h, W=w, CIN=cin, COUT=cout,
            SPATIAL_BLOCK=spatial_block, CHANNEL_BLOCK=channel_block,
            WEIGHT_PACKED=packed_weight is not None,
            OFFSET_BIAS_FUSED=offset_bias is not None,
            INPUT_BIAS_FUSED=input_bias is not None,
            SAMPLE_INPUT_BIAS=input_bias is not None and not input_already_biased,
            FUSE_EPILOGUE=post_mul is not None,
            num_warps=warps,
        )
        return output
    _direct_dcn3x3_fp32[(cout, triton.cdiv(h * w, BLOCK))](
        input.contiguous(),
        offset.contiguous(),
        mask.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        output,
        H=h,
        W=w,
        CIN=cin,
        COUT=cout,
        BLOCK_SIZE=BLOCK,
        num_warps=WARPS,
    )
    return output


def deform_conv2d_im2col(input, offset, weight, bias, padding, mask, stride):
    n, cin, h, w = input.shape
    cout, weight_cin, kh, kw = weight.shape
    stride_pair = stride if isinstance(stride, tuple) else (stride, stride)
    padding_pair = padding if isinstance(padding, tuple) else (padding, padding)
    if not (
        n == 1 and input.dtype in (torch.float32, torch.float16)
        and weight.dtype == input.dtype
        and cin == weight_cin and kh == kw == 3 and stride_pair == (1, 1)
        and padding_pair == (1, 1) and offset.shape == (1, 18, h, w)
        and mask is not None and mask.shape == (1, 9, h, w)
    ):
        raise ValueError("im2col DCN received an unsupported deployment shape")
    hw = h * w
    columns = torch.empty((cin * 9, hw), device=input.device, dtype=input.dtype)
    _deform_im2col3x3_fp32[(cin * 9, triton.cdiv(hw, IM2COL_BLOCK))](
        input.contiguous(), offset.contiguous(), mask.contiguous(), columns,
        H=h, W=w, CIN=cin, BLOCK_SIZE=IM2COL_BLOCK, num_warps=WARPS,
    )
    output = torch.addmm(bias[:, None], weight.reshape(cout, cin * 9), columns)
    return output.reshape(1, cout, h, w)
