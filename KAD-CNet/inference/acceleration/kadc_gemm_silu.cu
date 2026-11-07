#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int TileM = 32;
constexpr int TileN = 32;
constexpr int TileK = 16;

template <int SpatialVectors>
__global__ void add_bias_silu_float4_kernel(
    const float4* __restrict__ lhs,
    const float4* __restrict__ rhs,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int vectors) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) {
    return;
  }
  const float4 left = lhs[index];
  const float4 right = rhs[index];
  const float channel_bias = __ldg(bias + index / SpatialVectors);
  float4 value;
  value.x = left.x + right.x + channel_bias;
  value.y = left.y + right.y + channel_bias;
  value.z = left.z + right.z + channel_bias;
  value.w = left.w + right.w + channel_bias;
  value.x = value.x / (1.0f + __expf(-value.x));
  value.y = value.y / (1.0f + __expf(-value.y));
  value.z = value.z / (1.0f + __expf(-value.z));
  value.w = value.w / (1.0f + __expf(-value.w));
  output[index] = value;
}

__global__ void add_bias_silu_float4_generic_kernel(
    const float4* __restrict__ lhs,
    const float4* __restrict__ rhs,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int vectors,
    int spatial_vectors) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) {
    return;
  }
  const float4 left = lhs[index];
  const float4 right = rhs[index];
  const float channel_bias = __ldg(bias + index / spatial_vectors);
  float4 value;
  value.x = left.x + right.x + channel_bias;
  value.y = left.y + right.y + channel_bias;
  value.z = left.z + right.z + channel_bias;
  value.w = left.w + right.w + channel_bias;
  value.x = value.x / (1.0f + __expf(-value.x));
  value.y = value.y / (1.0f + __expf(-value.y));
  value.z = value.z / (1.0f + __expf(-value.z));
  value.w = value.w / (1.0f + __expf(-value.w));
  output[index] = value;
}

__global__ void bias_silu_float4_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int vectors,
    int spatial_vectors) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) {
    return;
  }
  float4 value = input[index];
  const float channel_bias = __ldg(bias + index / spatial_vectors);
  value.x += channel_bias;
  value.y += channel_bias;
  value.z += channel_bias;
  value.w += channel_bias;
  value.x = value.x / (1.0f + __expf(-value.x));
  value.y = value.y / (1.0f + __expf(-value.y));
  value.z = value.z / (1.0f + __expf(-value.z));
  value.w = value.w / (1.0f + __expf(-value.w));
  output[index] = value;
}

__global__ void avg_pool2x2_s1_float4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels,
    int height) {
  constexpr int RowsPerBlock = 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y * RowsPerBlock + warp;
  const int channel = blockIdx.z;
  const int output_width = height - 1;
  const int col = (blockIdx.x * 32 + lane) * 4;
  if (channel >= channels || row >= output_width || col >= output_width) {
    return;
  }
  const int source = (channel * height + row) * height + col;
  const float4 top = *reinterpret_cast<const float4*>(input + source);
  const float4 bottom =
      *reinterpret_cast<const float4*>(input + source + height);
  const float top4 = col + 4 < height ? input[source + 4] : 0.0f;
  const float bottom4 =
      col + 4 < height ? input[source + height + 4] : 0.0f;
  const int target = (channel * output_width + row) * output_width + col;
  output[target] = (top.x + top.y + bottom.x + bottom.y) * 0.25f;
  if (col + 1 < output_width) {
    output[target + 1] = (top.y + top.z + bottom.y + bottom.z) * 0.25f;
  }
  if (col + 2 < output_width) {
    output[target + 2] = (top.z + top.w + bottom.z + bottom.w) * 0.25f;
  }
  if (col + 3 < output_width) {
    output[target + 3] = (top.w + top4 + bottom.w + bottom4) * 0.25f;
  }
}

__global__ void gemm_bias_silu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int output_channels,
    int spatial,
    int input_channels) {
  __shared__ float tile_weight[TileM][TileK + 1];
  __shared__ float tile_input[TileK][TileN + 1];

  const int tid = threadIdx.x;
  const int local_n = tid & 31;
  const int row_group = tid >> 5;
  const int output_col = blockIdx.x * TileN + local_n;
  const int output_base_row = blockIdx.y * TileM + row_group;
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k0 = 0; k0 < input_channels; k0 += TileK) {
    for (int index = tid; index < TileM * TileK; index += blockDim.x) {
      const int row = index / TileK;
      const int k = index - row * TileK;
      tile_weight[row][k] = __ldg(
          weight + (blockIdx.y * TileM + row) * input_channels + k0 + k);
    }
    for (int index = tid; index < TileK * TileN; index += blockDim.x) {
      const int k = index / TileN;
      const int col = index - k * TileN;
      tile_input[k][col] = __ldg(
          input + (k0 + k) * spatial + blockIdx.x * TileN + col);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TileK; ++k) {
      const float value = tile_input[k][local_n];
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        accum[item] = fmaf(
            tile_weight[row_group + item * 8][k], value, accum[item]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int row = output_base_row + item * 8;
    const float value = accum[item] + __ldg(bias + row);
    output[row * spatial + output_col] = value / (1.0f + expf(-value));
  }
}

}  // namespace

torch::Tensor kadc_gemm_silu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32 &&
                  input.is_contiguous(),
              "input must be contiguous CUDA FP32");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kFloat32 &&
                  weight.is_contiguous(),
              "weight must be contiguous CUDA FP32");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32 &&
                  bias.is_contiguous(),
              "bias must be contiguous CUDA FP32");
  TORCH_CHECK(input.dim() == 4 && input.size(0) == 1,
              "input must be [1,C,H,W]");
  TORCH_CHECK(weight.dim() == 4 && weight.size(2) == 1 && weight.size(3) == 1 &&
                  weight.size(1) == input.size(1),
              "weight must be [Cout,Cin,1,1]");
  const int input_channels = static_cast<int>(input.size(1));
  const int output_channels = static_cast<int>(weight.size(0));
  const int spatial = static_cast<int>(input.size(2) * input.size(3));
  TORCH_CHECK(input_channels % TileK == 0 && output_channels % TileM == 0 &&
                  spatial % TileN == 0 && bias.numel() == output_channels,
              "shape must be aligned to the fixed FP32 GEMM tiles");

  c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty(
      {1, output_channels, input.size(2), input.size(3)}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const dim3 grid(spatial / TileN, output_channels / TileM);
  gemm_bias_silu_kernel<<<grid, 256, 0, stream>>>(
      input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
      output.data_ptr<float>(), output_channels, spatial, input_channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor kadc_add_bias_silu_cuda(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const torch::Tensor& bias,
    std::int64_t requested_threads) {
  TORCH_CHECK(lhs.is_cuda() && lhs.scalar_type() == torch::kFloat32 &&
                  lhs.is_contiguous() && rhs.is_cuda() &&
                  rhs.scalar_type() == torch::kFloat32 && rhs.is_contiguous(),
              "inputs must be contiguous CUDA FP32");
  TORCH_CHECK(lhs.sizes() == rhs.sizes() && lhs.dim() == 4 && lhs.size(0) == 1,
              "inputs must be matching [1,C,H,W] tensors");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32 &&
                  bias.is_contiguous() && bias.numel() == lhs.size(1),
              "bias must be contiguous CUDA FP32 with C values");
  const int spatial = static_cast<int>(lhs.size(2) * lhs.size(3));
  TORCH_CHECK(spatial % 4 == 0, "spatial size must be float4 aligned");
  auto output = torch::empty_like(lhs);
  const int vectors = static_cast<int>(lhs.numel() / 4);
  const bool use_generic_index = requested_threads < 0;
  const auto absolute_threads =
      use_generic_index ? -requested_threads : requested_threads;
  TORCH_CHECK(absolute_threads >= 32 && absolute_threads <= 1024
                  && absolute_threads % 32 == 0,
              "absolute threads must be a warp-aligned value in [32,1024]");
  const int threads = static_cast<int>(absolute_threads);
  const auto stream = at::cuda::getCurrentCUDAStream(lhs.get_device()).stream();
  const auto* lhs4 = reinterpret_cast<const float4*>(lhs.data_ptr<float>());
  const auto* rhs4 = reinterpret_cast<const float4*>(rhs.data_ptr<float>());
  auto* output4 = reinterpret_cast<float4*>(output.data_ptr<float>());
  const int blocks = (vectors + threads - 1) / threads;
  if (use_generic_index) {
    add_bias_silu_float4_generic_kernel<<<blocks, threads, 0, stream>>>(
        lhs4, rhs4, bias.data_ptr<float>(), output4, vectors, spatial / 4);
  } else {
#define KADC_ADD_DISPATCH(SV) \
    if (spatial / 4 == (SV)) { \
      add_bias_silu_float4_kernel<(SV)><<<blocks, threads, 0, stream>>>( \
          lhs4, rhs4, bias.data_ptr<float>(), output4, vectors); \
    } else
    KADC_ADD_DISPATCH(4096)
    KADC_ADD_DISPATCH(1024)
    KADC_ADD_DISPATCH(256)
    KADC_ADD_DISPATCH(64)
    KADC_ADD_DISPATCH(16)
    KADC_ADD_DISPATCH(4)
    {
      TORCH_CHECK(false, "unsupported spatial size for templated add epilogue");
    }
#undef KADC_ADD_DISPATCH
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor kadc_bias_silu_cuda(
    const torch::Tensor& input,
    const torch::Tensor& bias) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32 &&
                  input.is_contiguous() && input.dim() == 4 && input.size(0) == 1,
              "input must be contiguous CUDA FP32 [1,C,H,W]");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32 &&
                  bias.is_contiguous() && bias.numel() == input.size(1),
              "bias must be contiguous CUDA FP32 with C values");
  const int spatial = static_cast<int>(input.size(2) * input.size(3));
  TORCH_CHECK(spatial % 4 == 0, "spatial size must be float4 aligned");
  auto output = torch::empty_like(input);
  const int vectors = static_cast<int>(input.numel() / 4);
  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  bias_silu_float4_kernel<<<(vectors + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const float4*>(input.data_ptr<float>()),
      bias.data_ptr<float>(),
      reinterpret_cast<float4*>(output.data_ptr<float>()),
      vectors, spatial / 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor kadc_avg_pool2x2_s1_cuda(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32 &&
                  input.is_contiguous() && input.dim() == 4 && input.size(0) == 1 &&
                  input.size(2) == input.size(3),
              "input must be contiguous CUDA FP32 [1,C,H,H]");
  const int channels = static_cast<int>(input.size(1));
  const int height = static_cast<int>(input.size(2));
  TORCH_CHECK(height >= 2 && height % 4 == 0,
              "input height must be a multiple of four");
  auto output = torch::empty(
      {1, channels, height - 1, height - 1}, input.options());
  constexpr int Threads = 128;
  constexpr int RowsPerBlock = 4;
  const dim3 grid(
      (height - 1 + 127) / 128,
      (height - 1 + RowsPerBlock - 1) / RowsPerBlock,
      channels);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  avg_pool2x2_s1_float4_kernel<<<grid, Threads, 0, stream>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), channels, height);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
