#include <torch/extension.h>

#include <cstdint>

torch::Tensor kadc_gemm_silu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

torch::Tensor kadc_add_bias_silu_cuda(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& bias,
    std::int64_t threads);

torch::Tensor kadc_bias_silu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& bias);

torch::Tensor kadc_avg_pool2x2_s1_cuda(const torch::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &kadc_gemm_silu_cuda,
             "Tiled FP32 1x1 convolution with bias and SiLU (CUDA)");
  module.def("add_bias_silu", &kadc_add_bias_silu_cuda,
             "Vectorized FP32 add, channel bias and SiLU (CUDA)",
             pybind11::arg("lhs"), pybind11::arg("rhs"),
             pybind11::arg("bias"), pybind11::arg("threads") = 256);
  module.def("bias_silu", &kadc_bias_silu_cuda,
             "Vectorized FP32 channel bias and SiLU (CUDA)");
  module.def("avg_pool2x2_s1", &kadc_avg_pool2x2_s1_cuda,
             "Vectorized FP32 2x2 stride-1 average pool (CUDA)");
}
