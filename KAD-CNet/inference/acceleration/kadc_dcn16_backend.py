"""Selective CUDA dispatch for fixed batch-one KADC family DCN shapes."""

import os
from pathlib import Path
import atexit

import torch
from torch.utils.cpp_extension import load


_INSTALLED = False
_EXTENSION = None
_ORIGINAL = None
_STATS = {"calls": 0, "hits": 0, "gap_hits": 0, "offset_hits": 0,
          "rca_hits": 0, "se_hits": 0, "shape_hits": {}}
_FUSED_GAPS = {}
_FUSED_ATTENTIONS = {}
_SE_INPUTS = {}
_RCA_TAILS = {}
_SE_PARAMS = {}
_SHAPE_FAMILIES = {
    "n": {
        (6, 64), (5, 32), (8, 32), (5, 16), (8, 16),
        (12, 16), (8, 8), (10, 8), (10, 4),
    },
    "s": {
        (12, 64), (10, 32), (16, 32), (10, 16), (16, 16),
        (22, 16), (16, 8), (21, 8), (21, 4),
    },
    "m": {
        (12, 128), (10, 64), (20, 64), (10, 32), (20, 32),
        (30, 32), (20, 16), (30, 16), (40, 16), (30, 8),
        (40, 8), (40, 4),
    },
    "l": {
        (12, 128), (10, 64), (22, 64), (10, 32), (21, 32),
        (44, 32), (21, 16), (42, 16), (44, 16), (42, 8),
        (42, 4),
    },
    "x": {
        (12, 128), (10, 64), (22, 64), (10, 32), (21, 32),
        (44, 32), (21, 16), (42, 16), (44, 16), (86, 16),
        (42, 8), (85, 8), (42, 4), (85, 4),
    },
}


def _family():
    family = os.environ.get("KADC_DCN_FAMILY", "n").lower()
    if family not in _SHAPE_FAMILIES:
        raise ValueError(f"unsupported KADC_DCN_FAMILY={family!r}")
    return family


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    source_dir = Path(os.environ.get("KADC_DCN16_SOURCE_DIR", Path(__file__).resolve().parent))
    build_dir = Path(os.environ.get("KADC_DCN16_BUILD_DIR", "/dev/shm/kadc_dcn16_build_v3"))
    build_dir.mkdir(parents=True, exist_ok=True)
    # Build for the visible GPU unless the caller explicitly requests an arch.
    # The original 2080 Ti runner fixed this to 7.5; that silently produces the
    # wrong binary when the same source is moved to an A100 (compute capability
    # 8.0).  Kernel math is unchanged by this portability fix.
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    arch_tag = "".join(
        character if character.isalnum() else "_"
        for character in os.environ["TORCH_CUDA_ARCH_LIST"]
    )
    family = _family()
    family_define = f"-DKADC_FAMILY_{family.upper()}=1"
    cuda_flags = ["-O3", "-lineinfo", family_define]
    if os.environ.get("KADC_DCN_FAST_SIGMOID", "0") == "1":
        cuda_flags.append("-DKADC_FAST_SIGMOID=1")
    _EXTENSION = load(
        name=f"kadc_dcn16_{arch_tag}_{family}",
        sources=[str(source_dir / "kadc_dcn16.cpp"), str(source_dir / "kadc_dcn16_kernel.cu")],
        build_directory=str(build_dir),
        extra_cflags=["-O3", family_define],
        extra_cuda_cflags=cuda_flags,
        verbose=os.environ.get("KADC_DCN16_VERBOSE_BUILD", "0") == "1",
    )
    return _EXTENSION


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _parse_shape_set(text):
    result = set()
    for item in text.split(","):
        if not item:
            continue
        rendered_c, rendered_h = item.lower().split("h", 1)
        result.add((int(rendered_c[1:] if rendered_c.startswith("c") else rendered_c),
                    int(rendered_h)))
    return result


def _eligible(input, offset, weight, bias, padding, mask, stride,
              offset_bias, input_bias, offset_correction,
              input_already_biased, post_mul, post_add, post_scale):
    tensors = (
        input, offset, weight, bias, mask, offset_bias, offset_correction,
        post_mul, post_add, post_scale,
    )
    channels = input.shape[1] if input.ndim == 4 else -1
    height = input.shape[2] if input.ndim == 4 else -1
    target = os.environ.get("KADC_DCN_TARGET", "single")
    if target.startswith("all_"):
        selected_family = target.split("_", 1)[1]
        shape_selected = (channels, height) in _SHAPE_FAMILIES[selected_family]
    else:
        target_channels = int(os.environ.get("KADC_DCN_TARGET_C", "16"))
        target_height = int(os.environ.get("KADC_DCN_TARGET_H", "32"))
        shape_selected = channels == target_channels and height == target_height
    if os.environ.get("KADC_DCN_TRUST_INPUTS", "0") == "1":
        return (
            shape_selected and input.ndim == 4 and input.shape[0] == 1
            and input.shape[2] == input.shape[3]
            and input.is_cuda and input.dtype == torch.float32 and input.is_contiguous()
            and weight.is_cuda and weight.dtype == torch.float32 and weight.is_contiguous()
            and mask is not None and mask.is_cuda and mask.dtype == torch.float32
            and post_mul is not None and post_add is not None and post_scale is not None
        )
    return (
        shape_selected
        and input.shape == (1, channels, height, height)
        and input.dtype == torch.float32
        and offset.shape == (1, 18, height, height)
        and mask is not None and mask.shape == (1, 9, height, height)
        and weight.shape == (channels, channels, 3, 3)
        and _pair(padding) == (1, 1)
        and _pair(stride) == (1, 1)
        and offset_bias is not None and offset_bias.numel() == 18
        and input_bias is not None
        and offset_correction is not None
        and offset_correction.shape == (1, 18, height, height)
        and input_already_biased
        and post_mul is not None and post_add is not None and post_scale is not None
        and all(tensor.is_cuda and tensor.dtype == torch.float32 and tensor.is_contiguous()
                for tensor in tensors)
    )


def install(model=None):
    """Patch the selected family's verified shapes; all other calls fall back."""
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return _STATS
    extension = _load_extension()
    default_tile = int(os.environ.get("KADC_DCN16_TILE", "8"))
    if default_tile not in (4, 8, 16, 32):
        raise ValueError("KADC_DCN16_TILE must be 4, 8, 16, or 32")
    tile_map = {}
    for item in os.environ.get("KADC_DCN_TILE_MAP", "").split(","):
        if not item:
            continue
        shape, rendered_tile = item.split("=", 1)
        rendered_c, rendered_h = shape.lower().split("h", 1)
        channel_text = rendered_c[1:] if rendered_c.startswith("c") else rendered_c
        parsed_tile = int(rendered_tile)
        if parsed_tile not in (4, 8, 16, 32):
            raise ValueError(f"unsupported tile in KADC_DCN_TILE_MAP: {parsed_tile}")
        tile_map[(int(channel_text), int(rendered_h))] = parsed_tile
    fuse_gap = os.environ.get("KADC_DCN_FUSE_GAP", "0") == "1"
    fuse_se = os.environ.get("KADC_DCN_FUSE_SE", "0") == "1"
    native_se_gap = os.environ.get("KADC_DCN_NATIVE_SE_GAP", "0") == "1"
    native_se_min_c = int(os.environ.get("KADC_DCN_NATIVE_SE_MIN_C", "0"))
    se_mode = os.environ.get("KADC_DCN_SE_MODE", "reread")
    se_shapes = _parse_shape_set(os.environ.get("KADC_DCN_FUSE_SE_SHAPES", ""))
    if se_mode not in {"reread", "inline", "partial"}:
        raise ValueError("KADC_DCN_SE_MODE must be reread, inline, or partial")
    gap_shapes = _parse_shape_set(os.environ.get("KADC_DCN_FUSE_GAP_SHAPES", ""))
    if fuse_se:
        if model is None:
            raise ValueError("KADC_DCN_FUSE_SE=1 requires install(model)")
        for module in model.modules():
            if type(module).__name__ not in {"KADC4", "KADC4_2", "KADC4_m"}:
                continue
            for conv, fc in zip(module.convs, module.fc):
                dcn = conv[1]
                linear1, linear2 = fc[0], fc[2]
                if linear1.weight.shape[1] != dcn.weight.shape[0]:
                    raise ValueError("SE input channels do not match DCN output channels")
                _SE_PARAMS[dcn.weight.data_ptr()] = (
                    linear1.weight.contiguous(), linear1.bias.contiguous(),
                    linear2.weight.contiguous(), linear2.bias.contiguous(),
                )
        if not _SE_PARAMS:
            raise ValueError("no KADC DCN/SE parameter pairs were discovered")
    import kadc_dcn_direct

    _ORIGINAL = kadc_dcn_direct.deform_conv2d_direct
    debug_miss = os.environ.get("KADC_DCN_DEBUG_MISS", "0") == "1"
    miss_printed = [False]

    def dispatch(input, offset, weight, bias, padding, mask, stride,
                 packed_weight=None, offset_bias=None, input_bias=None,
                 offset_correction=None, input_already_biased=False,
                 post_mul=None, post_add=None, post_scale=None,
                 fused_offset_weight=None, fused_offset_bias=None):
        _STATS["calls"] += 1
        if _eligible(
            input, offset, weight, bias, padding, mask, stride,
            offset_bias, input_bias, offset_correction, input_already_biased,
            post_mul, post_add, post_scale,
        ):
            _STATS["hits"] += 1
            shape_key = f"c{input.shape[1]}h{input.shape[2]}"
            shape_pair = (input.shape[1], input.shape[2])
            _STATS["shape_hits"][shape_key] = _STATS["shape_hits"].get(shape_key, 0) + 1
            spatial_tile = tile_map.get((input.shape[1], input.shape[2]), default_tile)
            rca_tail = (_RCA_TAILS.pop(mask.data_ptr(), None)
                        if torch.is_tensor(mask) else None)
            if rca_tail is not None:
                se_params = _SE_PARAMS.get(weight.data_ptr()) if fuse_se else None
                if se_shapes and shape_pair not in se_shapes:
                    se_params = None
                if se_params is not None:
                    se_forward = (extension.forward_rca_inline_se_unsafe
                                  if se_mode in {"inline", "partial"}
                                  else extension.forward_rca_se_unsafe)
                    output, attention = se_forward(
                        input, offset, offset_bias, offset_correction, mask,
                        rca_tail[0], rca_tail[1], rca_tail[2], weight, bias,
                        post_mul, post_add, post_scale,
                        se_params[0], se_params[1], se_params[2], se_params[3],
                        spatial_tile,
                    )
                    _FUSED_ATTENTIONS[output.data_ptr()] = attention
                    _STATS["se_hits"] += 1
                else:
                    output, gap = extension.forward_rca_gap_unsafe(
                        input, offset, offset_bias, offset_correction, mask,
                        rca_tail[0], rca_tail[1], rca_tail[2], weight, bias,
                        post_mul, post_add, post_scale, spatial_tile,
                    )
                    _FUSED_GAPS[output.data_ptr()] = gap
                _STATS["rca_hits"] += 1
                return output
            if fused_offset_weight is not None:
                output, gap = extension.forward_fused_offset_gap_unsafe(
                    input, fused_offset_weight, fused_offset_bias, mask, weight, bias,
                    post_mul, post_add, post_scale, spatial_tile,
                )
                _STATS["offset_hits"] += 1
                _FUSED_GAPS[output.data_ptr()] = gap
                return output
            arguments = (
                input, offset, offset_bias, offset_correction, mask, weight, bias,
                post_mul, post_add, post_scale, spatial_tile,
            )
            if fuse_gap and (not gap_shapes or shape_pair in gap_shapes):
                gap_forward = (extension.forward_gap_unsafe
                               if os.environ.get("KADC_DCN_TRUST_INPUTS", "0") == "1"
                               else extension.forward_gap)
                output, gap = gap_forward(*arguments)
                _FUSED_GAPS[output.data_ptr()] = gap
                return output
            direct_forward = (extension.forward_unsafe
                              if os.environ.get("KADC_DCN_TRUST_INPUTS", "0") == "1"
                              else extension.forward)
            return direct_forward(*arguments)
        if debug_miss and not miss_printed[0]:
            miss_printed[0] = True
            print(
                "kadc_dcn16_miss "
                f"input_shape={tuple(input.shape)} input_dtype={input.dtype} "
                f"input_cuda={input.is_cuda} input_contiguous={input.is_contiguous()} "
                f"offset_shape={tuple(offset.shape)} weight_shape={tuple(weight.shape)} "
                f"mask_shape={None if mask is None else tuple(mask.shape)} "
                f"offset_bias={offset_bias is not None} input_bias={input_bias is not None} "
                f"offset_correction={offset_correction is not None} "
                f"input_already_biased={input_already_biased} "
                f"post_mul={post_mul is not None} post_add={post_add is not None} "
                f"post_scale={post_scale is not None} target={os.environ.get('KADC_DCN_TARGET')}",
                flush=True,
            )
        return _ORIGINAL(
            input=input, offset=offset, weight=weight, bias=bias, padding=padding,
            mask=mask, stride=stride, packed_weight=packed_weight,
            offset_bias=offset_bias, input_bias=input_bias,
            offset_correction=offset_correction,
            input_already_biased=input_already_biased,
            post_mul=post_mul, post_add=post_add, post_scale=post_scale,
            fused_offset_weight=fused_offset_weight,
            fused_offset_bias=fused_offset_bias,
        )

    kadc_dcn_direct.deform_conv2d_direct = dispatch
    # Some scheduling variants bind the symbol at module import time.
    try:
        import kadc_fused_parallel

        kadc_fused_parallel.deform_conv2d_direct = dispatch
    except (ImportError, AttributeError):
        pass
    if fuse_gap:
        import kadc_fused_sm75

        original_gap = kadc_fused_sm75.bnrelu_scale_gap

        def bnrelu_scale_gap_dispatch(value, bn, scale_weight):
            attention = _FUSED_ATTENTIONS.pop(value.data_ptr(), None)
            if attention is not None:
                _SE_INPUTS[attention.data_ptr()] = attention
                _STATS["gap_hits"] += 1
                return value, attention
            gap = _FUSED_GAPS.pop(value.data_ptr(), None)
            if gap is not None:
                _STATS["gap_hits"] += 1
                return value, gap
            return original_gap(value, bn, scale_weight)

        kadc_fused_sm75.bnrelu_scale_gap = bnrelu_scale_gap_dispatch
        try:
            import kadc_fused_parallel

            kadc_fused_parallel.bnrelu_scale_gap = bnrelu_scale_gap_dispatch
        except (ImportError, AttributeError):
            pass
        if fuse_se:
            original_se = kadc_fused_sm75.se_mlp

            def se_mlp_dispatch(gap, fc):
                attention = (_SE_INPUTS.pop(gap.data_ptr(), None)
                             if torch.is_tensor(gap) else None)
                if attention is not None:
                    return attention
                return original_se(gap, fc)

            kadc_fused_sm75.se_mlp = se_mlp_dispatch
            try:
                import kadc_fused_parallel

                kadc_fused_parallel.se_mlp = se_mlp_dispatch
            except (ImportError, AttributeError):
                pass
        elif native_se_gap:
            original_se = kadc_fused_sm75.se_mlp

            def native_se_mlp_dispatch(gap, fc):
                if (torch.is_tensor(gap) and gap.is_cuda and gap.dtype == torch.float32
                        and gap.dim() == 2 and gap.shape[0] == 1
                        and gap.shape[1] >= native_se_min_c):
                    linear1, linear2 = fc[0], fc[2]
                    return extension.se_from_gap_unsafe(
                        gap.contiguous(), linear1.weight.contiguous(),
                        linear1.bias.contiguous(), linear2.weight.contiguous(),
                        linear2.bias.contiguous(),
                    )
                return original_se(gap, fc)

            kadc_fused_sm75.se_mlp = native_se_mlp_dispatch
            try:
                import kadc_fused_parallel

                kadc_fused_parallel.se_mlp = native_se_mlp_dispatch
            except (ImportError, AttributeError):
                pass
    if os.environ.get("KADC_DCN_FUSE_RCA_TAIL", "0") == "1":
        import triton
        import kadc_fused_sm75

        original_rca = kadc_fused_sm75.rca_triton

        def rca_prepare_dispatch(x, rca, input_bias=None, biased_output=None,
                                 offset=None, offset_correction=None):
            if (x.shape[0] != 1 or (x.shape[1], x.shape[2]) not in _SHAPE_FAMILIES[_family()] or
                    offset is not None):
                return original_rca(x, rca, input_bias, biased_output,
                                    offset, offset_correction)
            n, channels, height, width = x.shape
            excite = rca.excite
            conv1, bn, conv2, conv3 = excite[0], excite[1], excite[3], excite[4]
            cache = getattr(rca, "_sm75_folded_constants", None)
            if cache is None:
                scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                w1 = (conv1.weight.reshape(channels, 11) * scale[:, None]).contiguous()
                b1 = ((conv1.bias - bn.running_mean) * scale + bn.bias).contiguous()
                w2 = conv2.weight.reshape(channels, 11).contiguous()
                pw_raw = conv3.weight.reshape(9, channels).contiguous()
                pw_bias = (conv3.bias + torch.mv(pw_raw, conv2.bias)).contiguous()
                pw_weight = pw_raw.transpose(0, 1).contiguous()
                cache = (w1, b1, w2, pw_weight, pw_bias)
                rca._sm75_folded_constants = cache
            w1, b1, w2, pw_weight, pw_bias = cache
            h1 = torch.empty_like(x)
            xw = torch.empty(n * channels * width, device=x.device, dtype=x.dtype)
            bh = triton.next_power_of_2(height)
            bw = triton.next_power_of_2(width)
            kadc_fused_sm75.k_reduce_hconv_bn_relu[(n * channels,)](
                x, input_bias if input_bias is not None else b1,
                biased_output if biased_output is not None else h1,
                xw, w1, b1, bn.running_mean, bn.running_var, bn.weight, bn.bias, h1,
                H=height, W=width, EPS=float(bn.eps), K=11, BH=bh, BW=bw,
                BN_FOLDED=True, INPUT_BIAS_FUSED=input_bias is not None,
                STORE_BIASED_INPUT=biased_output is not None,
                num_warps=kadc_fused_sm75.RCA_RH_WARPS,
            )
            _RCA_TAILS[h1.data_ptr()] = (w2, pw_weight, pw_bias)
            return h1

        kadc_fused_sm75.rca_triton = rca_prepare_dispatch
        # The parallel scheduler imports this symbol eagerly.  Keep its local
        # binding on the same fused-RCA implementation so split scheduling does
        # not silently fall back to the unfused mask path.
        try:
            import kadc_fused_parallel

            kadc_fused_parallel.rca_triton = rca_prepare_dispatch
        except (ImportError, AttributeError):
            pass
    _INSTALLED = True
    atexit.register(lambda: print(
        f"kadc_dcn16_calls={_STATS['calls']} hits={_STATS['hits']}", flush=True
    ))
    target = os.environ.get("KADC_DCN_TARGET", "single")
    print(
        f"kadc_dcn16_backend=installed target={target} default_tile={default_tile} "
        f"tile_map={tile_map} fuse_gap={fuse_gap} fuse_se={fuse_se} "
        f"se_mode={se_mode} se_shapes={se_shapes} native_se_gap={native_se_gap} "
        f"native_se_min_c={native_se_min_c} "
        f"se_pairs={len(_SE_PARAMS)} gap_shapes={gap_shapes}",
        flush=True,
    )
    return _STATS


def stats():
    return {
        "calls": _STATS["calls"],
        "hits": _STATS["hits"],
        "gap_hits": _STATS["gap_hits"],
        "offset_hits": _STATS["offset_hits"],
        "rca_hits": _STATS["rca_hits"],
        "se_hits": _STATS["se_hits"],
        "shape_hits": dict(_STATS["shape_hits"]),
    }
