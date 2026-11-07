#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor kadc_dcn16_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& mask,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile);

std::vector<torch::Tensor> kadc_dcn16_forward_gap_cuda(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& mask,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile);

std::vector<torch::Tensor> kadc_dcn16_forward_fused_offset_gap_cuda(
    const torch::Tensor& input,
    const torch::Tensor& fused_offset_weight,
    const torch::Tensor& fused_offset_bias,
    const torch::Tensor& mask,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile);

std::vector<torch::Tensor> kadc_dcn16_forward_rca_gap_cuda(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile);

std::vector<torch::Tensor> kadc_dcn16_forward_rca_se_cuda(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    const torch::Tensor& se_weight1,
    const torch::Tensor& se_bias1,
    const torch::Tensor& se_weight2,
    const torch::Tensor& se_bias2,
    std::int64_t spatial_tile);

std::vector<torch::Tensor> kadc_dcn16_forward_rca_inline_se_cuda(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    const torch::Tensor& se_weight1,
    const torch::Tensor& se_bias1,
    const torch::Tensor& se_weight2,
    const torch::Tensor& se_bias2,
    std::int64_t spatial_tile);

torch::Tensor kadc_se_from_gap_cuda(
    const torch::Tensor& gap,
    const torch::Tensor& weight1,
    const torch::Tensor& bias1,
    const torch::Tensor& weight2,
    const torch::Tensor& bias2);

namespace {

bool supported_shape(std::int64_t channels, std::int64_t height) {
#if defined(KADC_FAMILY_S)
  return (channels == 12 && height == 64) ||
         (channels == 10 && (height == 32 || height == 16)) ||
         (channels == 16 && (height == 32 || height == 16 || height == 8)) ||
         (channels == 22 && height == 16) ||
         (channels == 21 && (height == 8 || height == 4));
#elif defined(KADC_FAMILY_M)
  return (channels == 12 && height == 128) ||
         (channels == 10 && (height == 64 || height == 32)) ||
         (channels == 20 && (height == 64 || height == 32 || height == 16)) ||
         (channels == 30 && (height == 32 || height == 16 || height == 8)) ||
         (channels == 40 && (height == 16 || height == 8 || height == 4));
#elif defined(KADC_FAMILY_L)
  return (channels == 12 && height == 128) ||
         (channels == 10 && (height == 64 || height == 32)) ||
         (channels == 22 && height == 64) ||
         (channels == 21 && (height == 32 || height == 16)) ||
         (channels == 44 && (height == 32 || height == 16)) ||
         (channels == 42 && (height == 16 || height == 8 || height == 4));
#elif defined(KADC_FAMILY_X)
  return (channels == 12 && height == 128) ||
         (channels == 10 && (height == 64 || height == 32)) ||
         (channels == 22 && height == 64) ||
         (channels == 21 && (height == 32 || height == 16)) ||
         (channels == 44 && (height == 32 || height == 16)) ||
         (channels == 42 && (height == 16 || height == 8 || height == 4)) ||
         (channels == 86 && height == 16) ||
         (channels == 85 && (height == 8 || height == 4));
#else
  return (channels == 6 && height == 64) ||
         (channels == 5 && (height == 32 || height == 16)) ||
         (channels == 8 && (height == 32 || height == 16 || height == 8)) ||
         (channels == 12 && height == 16) ||
         (channels == 10 && (height == 8 || height == 4)) ||
         (channels == 16 && height == 32);
#endif
}

void check_cuda_float_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be FP32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate_inputs(
    const torch::Tensor& input, const torch::Tensor& offset,
    const torch::Tensor& offset_bias, const torch::Tensor& offset_correction,
    const torch::Tensor& mask, const torch::Tensor& weight,
    const torch::Tensor& bias, const torch::Tensor& post_mul,
    const torch::Tensor& post_add, const torch::Tensor& post_scale,
    std::int64_t spatial_tile) {
  check_cuda_float_contiguous(input, "input");
  check_cuda_float_contiguous(offset, "offset");
  check_cuda_float_contiguous(offset_bias, "offset_bias");
  check_cuda_float_contiguous(offset_correction, "offset_correction");
  check_cuda_float_contiguous(mask, "mask");
  check_cuda_float_contiguous(weight, "weight");
  check_cuda_float_contiguous(bias, "bias");
  check_cuda_float_contiguous(post_mul, "post_mul");
  check_cuda_float_contiguous(post_add, "post_add");
  check_cuda_float_contiguous(post_scale, "post_scale");
  TORCH_CHECK(input.dim() == 4 && input.size(0) == 1 &&
                  input.size(2) == input.size(3) &&
                  supported_shape(input.size(1), input.size(2)),
              "unsupported fixed KADC family DCNv2 shape");
  const auto height = input.size(2);
  TORCH_CHECK(offset.dim() == 4 && offset.size(0) == 1 && offset.size(1) == 18 &&
                  offset.size(2) == height && offset.size(3) == height,
              "offset must be [1,18,H,H]");
  TORCH_CHECK(offset_bias.numel() == 18, "offset_bias must contain 18 values");
  TORCH_CHECK(offset_correction.sizes() == offset.sizes(),
              "offset_correction must match offset");
  TORCH_CHECK(mask.dim() == 4 && mask.size(0) == 1 && mask.size(1) == 9 &&
                  mask.size(2) == height && mask.size(3) == height,
              "mask must be [1,9,H,H]");
  const auto channels = input.size(1);
  TORCH_CHECK(weight.dim() == 4 && weight.size(0) == channels &&
                  weight.size(1) == channels && weight.size(2) == 3 &&
                  weight.size(3) == 3,
              "weight must be [C,C,3,3]");
  TORCH_CHECK(bias.numel() == channels && post_mul.numel() == channels &&
                  post_add.numel() == channels && post_scale.numel() == channels,
              "bias and epilogue tensors must contain C values");
  TORCH_CHECK(input.device() == offset.device() && input.device() == offset_bias.device() &&
                  input.device() == offset_correction.device() && input.device() == mask.device() &&
                  input.device() == weight.device() && input.device() == bias.device() &&
                  input.device() == post_mul.device() && input.device() == post_add.device() &&
                  input.device() == post_scale.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(spatial_tile == 4 || spatial_tile == 8 ||
                  spatial_tile == 16 || spatial_tile == 32,
              "spatial_tile must be 4, 8, 16, or 32");
}

torch::Tensor forward(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& mask,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile) {
  validate_inputs(input, offset, offset_bias, offset_correction, mask, weight,
                  bias, post_mul, post_add, post_scale, spatial_tile);

  return kadc_dcn16_forward_cuda(
      input, offset, offset_bias, offset_correction, mask, weight, bias,
      post_mul, post_add, post_scale,
      spatial_tile);
}

std::vector<torch::Tensor> forward_gap(
    const torch::Tensor& input,
    const torch::Tensor& offset,
    const torch::Tensor& offset_bias,
    const torch::Tensor& offset_correction,
    const torch::Tensor& mask,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& post_mul,
    const torch::Tensor& post_add,
    const torch::Tensor& post_scale,
    std::int64_t spatial_tile) {
  validate_inputs(input, offset, offset_bias, offset_correction, mask, weight,
                  bias, post_mul, post_add, post_scale, spatial_tile);
  TORCH_CHECK(spatial_tile != 32, "forward_gap does not support tile32");
  return kadc_dcn16_forward_gap_cuda(
      input, offset, offset_bias, offset_correction, mask, weight, bias,
      post_mul, post_add, post_scale, spatial_tile);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward,
             "KADC family fixed-shape FP32 modulated DCNv2 forward (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("mask"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("spatial_tile") = 16);
  module.def("forward_gap", &forward_gap,
             "KADC family fixed-shape FP32 DCNv2 forward with fused GAP (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("mask"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("spatial_tile") = 16);
  // The Python dispatcher has already checked the fixed deployment contract.
  // These bindings remove repeated Tensor metadata validation from 126 calls.
  module.def("forward_unsafe", &kadc_dcn16_forward_cuda,
             "Unchecked fixed-shape FP32 DCNv2 forward (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("mask"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("spatial_tile"));
  module.def("forward_gap_unsafe", &kadc_dcn16_forward_gap_cuda,
             "Unchecked fixed-shape FP32 DCNv2 forward with GAP (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("mask"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("spatial_tile"));
  module.def("forward_fused_offset_gap_unsafe",
             &kadc_dcn16_forward_fused_offset_gap_cuda,
             "Unchecked fixed-shape fused offset-conv + DCNv2 + GAP (CUDA)",
             pybind11::arg("input"), pybind11::arg("fused_offset_weight"),
             pybind11::arg("fused_offset_bias"), pybind11::arg("mask"),
             pybind11::arg("weight"), pybind11::arg("bias"),
             pybind11::arg("post_mul"), pybind11::arg("post_add"),
             pybind11::arg("post_scale"), pybind11::arg("spatial_tile"));
  module.def("forward_rca_gap_unsafe", &kadc_dcn16_forward_rca_gap_cuda,
             "Unchecked RCA-tail + DCNv2 + GAP forward (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("rca_h1"), pybind11::arg("rca_vertical_weight"),
             pybind11::arg("rca_pointwise_weight"),
             pybind11::arg("rca_pointwise_bias"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("spatial_tile"));
  module.def("forward_rca_se_unsafe", &kadc_dcn16_forward_rca_se_cuda,
             "Unchecked RCA-tail + DCNv2 + GAP/SE forward (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("rca_h1"), pybind11::arg("rca_vertical_weight"),
             pybind11::arg("rca_pointwise_weight"),
             pybind11::arg("rca_pointwise_bias"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("se_weight1"), pybind11::arg("se_bias1"),
             pybind11::arg("se_weight2"), pybind11::arg("se_bias2"),
             pybind11::arg("spatial_tile"));
  module.def("forward_rca_inline_se_unsafe",
             &kadc_dcn16_forward_rca_inline_se_cuda,
             "Unchecked RCA-tail + DCNv2 + inline last-CTA SE forward (CUDA)",
             pybind11::arg("input"), pybind11::arg("offset"),
             pybind11::arg("offset_bias"), pybind11::arg("offset_correction"),
             pybind11::arg("rca_h1"), pybind11::arg("rca_vertical_weight"),
             pybind11::arg("rca_pointwise_weight"),
             pybind11::arg("rca_pointwise_bias"), pybind11::arg("weight"),
             pybind11::arg("bias"), pybind11::arg("post_mul"),
             pybind11::arg("post_add"), pybind11::arg("post_scale"),
             pybind11::arg("se_weight1"), pybind11::arg("se_bias1"),
             pybind11::arg("se_weight2"), pybind11::arg("se_bias2"),
             pybind11::arg("spatial_tile"));
  module.def("se_from_gap_unsafe", &kadc_se_from_gap_cuda,
             "Native FP32 SE MLP from an existing GAP vector (CUDA)",
             pybind11::arg("gap"), pybind11::arg("weight1"),
             pybind11::arg("bias1"), pybind11::arg("weight2"),
             pybind11::arg("bias2"));
}
