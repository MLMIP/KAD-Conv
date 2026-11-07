#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kKernelPoints = 9;

__device__ __forceinline__ float kadc_sigmoid(float value) {
#if defined(KADC_FAST_SIGMOID)
  return 1.0f / (1.0f + __expf(-value));
#else
  return 1.0f / (1.0f + expf(-value));
#endif
}

template <int Channels, int SpatialTile, bool CacheWeights>
struct SharedStorage {
  float weights[CacheWeights ? Channels * Channels * kKernelPoints : 1];
  float samples[Channels * kKernelPoints * SpatialTile];
  int y0[kKernelPoints * SpatialTile];
  int x0[kKernelPoints * SpatialTile];
  float ly[kKernelPoints * SpatialTile];
  float lx[kKernelPoints * SpatialTile];
  float modulation[kKernelPoints * SpatialTile];
  unsigned char valid[kKernelPoints * SpatialTile];
};

template <int Height>
__device__ __forceinline__ float load_input(
    const float* __restrict__ input, int channel, int y, int x,
    bool valid) {
  constexpr int Spatial = Height * Height;
  return valid ? __ldg(input + channel * Spatial + y * Height + x) : 0.0f;
}

template <int Channels, int Height, int SpatialTile, bool CacheWeights>
__device__ __forceinline__ float compute_output_value(
    const SharedStorage<Channels, SpatialTile, CacheWeights>& shared,
    int output_channel, int local_spatial,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ post_mul,
    const float* __restrict__ post_add,
    const float* __restrict__ post_scale) {
  float accumulator = 0.0f;
  if constexpr (CacheWeights) {
#pragma unroll
    for (int input_channel = 0; input_channel < Channels; ++input_channel) {
#pragma unroll
      for (int kernel_point = 0; kernel_point < kKernelPoints; ++kernel_point) {
        const int sample_index =
            (input_channel * kKernelPoints + kernel_point) * SpatialTile + local_spatial;
        const int weight_index =
            (output_channel * Channels + input_channel) * kKernelPoints + kernel_point;
        accumulator = fmaf(shared.samples[sample_index], shared.weights[weight_index],
                           accumulator);
      }
    }
  } else {
    // Large KADC variants exceed SM75's 48-KB block limit if the complete
    // CxCx3x3 tensor is staged. Threads in a spatial subgroup request the
    // same coefficient, so the read-only cache broadcasts these loads.
#pragma unroll 1
    for (int input_channel = 0; input_channel < Channels; ++input_channel) {
#pragma unroll
      for (int kernel_point = 0; kernel_point < kKernelPoints; ++kernel_point) {
        const int sample_index =
            (input_channel * kKernelPoints + kernel_point) * SpatialTile + local_spatial;
        const int weight_index =
            (output_channel * Channels + input_channel) * kKernelPoints + kernel_point;
        accumulator = fmaf(shared.samples[sample_index], __ldg(weight + weight_index),
                           accumulator);
      }
    }
  }
  accumulator += __ldg(bias + output_channel);
  accumulator = accumulator * __ldg(post_mul + output_channel) +
                __ldg(post_add + output_channel);
  accumulator = accumulator > 0.0f ? accumulator : 0.0f;
  return accumulator * __ldg(post_scale + output_channel);
}

template <int Channels, int Height, int SpatialTile, int Threads,
          bool ProduceGap, bool ProduceSE>
__global__ void dcn16x32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    const float* __restrict__ offset_bias,
    const float* __restrict__ offset_correction,
    const float* __restrict__ mask,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ post_mul,
    const float* __restrict__ post_add,
    const float* __restrict__ post_scale,
    const float* __restrict__ fused_offset_weight,
    const float* __restrict__ rca_h1,
    const float* __restrict__ rca_vertical_weight,
    const float* __restrict__ rca_pointwise_weight,
    const float* __restrict__ rca_pointwise_bias,
    float* __restrict__ output,
    float* __restrict__ gap,
    const float* __restrict__ se_weight1,
    const float* __restrict__ se_bias1,
    const float* __restrict__ se_weight2,
    const float* __restrict__ se_bias2,
    int se_hidden_channels,
    float* __restrict__ attention) {
  constexpr bool CacheWeights = Channels <= 32;
  __shared__ SharedStorage<Channels, SpatialTile, CacheWeights> shared;
  constexpr int Spatial = Height * Height;

  const int tid = threadIdx.x;
  const int spatial_base = blockIdx.x * SpatialTile;

  if constexpr (CacheWeights) {
    constexpr int kWeightElements = Channels * Channels * kKernelPoints;
    for (int index = tid; index < kWeightElements; index += Threads) {
      shared.weights[index] = __ldg(weight + index);
    }
  }

  if (rca_h1 != nullptr) {
    constexpr int kRcaKernel = 11;
    constexpr int kRcaRadius = kRcaKernel / 2;
    constexpr int kVerticalElements = Channels * SpatialTile;
    for (int index = tid; index < kVerticalElements; index += Threads) {
      const int channel = index / SpatialTile;
      const int local_spatial = index - channel * SpatialTile;
      const int spatial = spatial_base + local_spatial;
      const int row = spatial / Height;
      const int col = spatial - row * Height;
      float vertical = 0.0f;
#pragma unroll
      for (int kernel = 0; kernel < kRcaKernel; ++kernel) {
        const int source_row = row + kernel - kRcaRadius;
        const bool valid = spatial < Spatial && source_row >= 0 && source_row < Height;
        const float value = valid
            ? __ldg(rca_h1 + channel * Spatial + source_row * Height + col)
            : 0.0f;
        vertical = fmaf(value,
                        __ldg(rca_vertical_weight + channel * kRcaKernel + kernel),
                        vertical);
      }
      shared.samples[index] = vertical;
    }
    __syncthreads();
  }

  constexpr int kCoordinateElements = kKernelPoints * SpatialTile;
  for (int index = tid; index < kCoordinateElements; index += Threads) {
    const int kernel_point = index / SpatialTile;
    const int local_spatial = index - kernel_point * SpatialTile;
    const int spatial = spatial_base + local_spatial;
    const bool active = spatial < Spatial;
    const int output_y = spatial / Height;
    const int output_x = spatial - output_y * Height;
    const int kernel_y = kernel_point / 3;
    const int kernel_x = kernel_point - kernel_y * 3;

    float offset_y = 0.0f;
    float offset_x = 0.0f;
    if (active && fused_offset_weight != nullptr) {
#pragma unroll
      for (int input_channel = 0; input_channel < Channels; ++input_channel) {
#pragma unroll
        for (int offset_kernel = 0; offset_kernel < kKernelPoints; ++offset_kernel) {
          const int offset_kernel_y = offset_kernel / 3;
          const int offset_kernel_x = offset_kernel - offset_kernel_y * 3;
          const int input_y = output_y + offset_kernel_y - 1;
          const int input_x = output_x + offset_kernel_x - 1;
          const bool input_valid = input_y >= 0 && input_y < Height &&
                                   input_x >= 0 && input_x < Height;
          const float input_value = load_input<Height>(
              input, input_channel, input_y, input_x, input_valid);
          const int base = input_channel * kKernelPoints + offset_kernel;
          offset_y = fmaf(input_value,
                          __ldg(fused_offset_weight +
                                (2 * kernel_point) * Channels * kKernelPoints + base),
                          offset_y);
          offset_x = fmaf(input_value,
                          __ldg(fused_offset_weight +
                                (2 * kernel_point + 1) * Channels * kKernelPoints + base),
                          offset_x);
        }
      }
      offset_y += __ldg(offset_bias + 2 * kernel_point);
      offset_x += __ldg(offset_bias + 2 * kernel_point + 1);
    } else if (active) {
      offset_y = __ldg(offset + (2 * kernel_point) * Spatial + spatial) +
                 __ldg(offset_bias + 2 * kernel_point) +
                 __ldg(offset_correction + (2 * kernel_point) * Spatial + spatial);
      offset_x = __ldg(offset + (2 * kernel_point + 1) * Spatial + spatial) +
                 __ldg(offset_bias + 2 * kernel_point + 1) +
                 __ldg(offset_correction + (2 * kernel_point + 1) * Spatial + spatial);
    }
    const float sample_y = static_cast<float>(output_y + kernel_y - 1) + offset_y;
    const float sample_x = static_cast<float>(output_x + kernel_x - 1) + offset_x;
    const bool valid = active && sample_y > -1.0f && sample_y < static_cast<float>(Height) &&
                       sample_x > -1.0f && sample_x < static_cast<float>(Height);

    int y0 = static_cast<int>(sample_y);
    int x0 = static_cast<int>(sample_x);
    if (sample_y < 0.0f) y0 = -1;
    if (sample_x < 0.0f) x0 = -1;
    shared.y0[index] = y0;
    shared.x0[index] = x0;
    shared.ly[index] = sample_y - static_cast<float>(y0);
    shared.lx[index] = sample_x - static_cast<float>(x0);
    float modulation = 0.0f;
    if (active && rca_h1 != nullptr) {
      modulation = __ldg(rca_pointwise_bias + kernel_point);
#pragma unroll
      for (int channel = 0; channel < Channels; ++channel) {
        modulation = fmaf(shared.samples[channel * SpatialTile + local_spatial],
                          __ldg(rca_pointwise_weight + channel * kKernelPoints +
                                kernel_point),
                          modulation);
      }
      modulation = kadc_sigmoid(modulation);
    } else if (active) {
      modulation = __ldg(mask + kernel_point * Spatial + spatial);
    }
    shared.modulation[index] = modulation;
    shared.valid[index] = static_cast<unsigned char>(valid);
  }
  __syncthreads();

  constexpr int kSampleElements = Channels * kKernelPoints * SpatialTile;
  for (int index = tid; index < kSampleElements; index += Threads) {
    const int local_spatial = index % SpatialTile;
    const int channel_kernel = index / SpatialTile;
    const int kernel_point = channel_kernel % kKernelPoints;
    const int channel = channel_kernel / kKernelPoints;
    const int coordinate_index = kernel_point * SpatialTile + local_spatial;
    const int y0 = shared.y0[coordinate_index];
    const int x0 = shared.x0[coordinate_index];
    const int y1 = y0 + 1;
    const int x1 = x0 + 1;
    const float ly = shared.ly[coordinate_index];
    const float lx = shared.lx[coordinate_index];
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;
    const bool valid = shared.valid[coordinate_index] != 0;

    const float v00 = load_input<Height>(input, channel, y0, x0,
                                 valid && y0 >= 0 && x0 >= 0);
    const float v01 = load_input<Height>(input, channel, y0, x1,
                                 valid && y0 >= 0 && x1 < Height);
    const float v10 = load_input<Height>(input, channel, y1, x0,
                                 valid && y1 < Height && x0 >= 0);
    const float v11 = load_input<Height>(input, channel, y1, x1,
                                 valid && y1 < Height && x1 < Height);
    const float sampled = v00 * hy * hx + v01 * hy * lx +
                          v10 * ly * hx + v11 * ly * lx;
    shared.samples[index] = sampled * shared.modulation[coordinate_index];
  }
  __syncthreads();

  constexpr int kOutputElements = Channels * SpatialTile;
  if constexpr (ProduceGap) {
    static_assert(SpatialTile <= 16 && kOutputElements <= Threads,
                  "fused GAP requires one output per thread");
    const bool active_output = tid < kOutputElements;
    const int output_channel = tid / SpatialTile;
    const int local_spatial = tid % SpatialTile;
    const int spatial = spatial_base + local_spatial;
    float value = 0.0f;
    if (active_output && spatial < Spatial) {
      value = compute_output_value<Channels, Height, SpatialTile, CacheWeights>(
          shared, output_channel, local_spatial, weight, bias,
          post_mul, post_add, post_scale);
      output[output_channel * Spatial + spatial] = value;
    }
    // SpatialTile is a power of two and each channel occupies one aligned
    // subgroup.  Reduce once per tile/channel, reducing global atomics by
    // SpatialTile compared with an element-wise accumulation.
#pragma unroll
    for (int delta = SpatialTile / 2; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, delta, SpatialTile);
    }
    if (active_output && local_spatial == 0) {
      if constexpr (ProduceSE) {
        // One deterministic partial per (spatial tile, channel).  A following
        // one-CTA reducer consumes this compact buffer, avoiding both the GAP
        // memset and contended global atomics on the DCN critical path.
        gap[blockIdx.x * Channels + output_channel] = value;
      } else {
        atomicAdd(gap + output_channel, value / static_cast<float>(Spatial));
      }
    }
  } else {
    for (int index = tid; index < kOutputElements; index += Threads) {
      const int output_channel = index / SpatialTile;
      const int local_spatial = index - output_channel * SpatialTile;
      const float value = compute_output_value<Channels, Height, SpatialTile, CacheWeights>(
          shared, output_channel, local_spatial, weight, bias,
          post_mul, post_add, post_scale);
      const int spatial = spatial_base + local_spatial;
      if (spatial < Spatial) {
        output[output_channel * Spatial + spatial] = value;
      }
    }
  }
}

template <int Channels, int Height, int SpatialTile, int Threads,
          bool ProduceGap, bool ProduceSE = false>
void launch_kernel(
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
    const torch::Tensor& fused_offset_weight,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    const torch::Tensor& se_weight1,
    const torch::Tensor& se_bias1,
    const torch::Tensor& se_weight2,
    const torch::Tensor& se_bias2,
    torch::Tensor& output,
    torch::Tensor& gap,
    torch::Tensor& attention,
    cudaStream_t stream) {
  static_assert(sizeof(SharedStorage<Channels, SpatialTile, (Channels <= 32)>)
                    <= 48 * 1024,
                "shared-memory footprint exceeds the SM75 per-block limit");
  constexpr int Spatial = Height * Height;
  constexpr int blocks = (Spatial + SpatialTile - 1) / SpatialTile;
  dcn16x32_kernel<Channels, Height, SpatialTile, Threads, ProduceGap, ProduceSE>
      <<<blocks, Threads, 0, stream>>>(
      input.data_ptr<float>(), offset.data_ptr<float>(), offset_bias.data_ptr<float>(),
      offset_correction.data_ptr<float>(), mask.data_ptr<float>(),
      weight.data_ptr<float>(), bias.data_ptr<float>(), post_mul.data_ptr<float>(),
      post_add.data_ptr<float>(), post_scale.data_ptr<float>(),
      fused_offset_weight.defined() ? fused_offset_weight.data_ptr<float>() : nullptr,
      rca_h1.defined() ? rca_h1.data_ptr<float>() : nullptr,
      rca_vertical_weight.defined() ? rca_vertical_weight.data_ptr<float>() : nullptr,
      rca_pointwise_weight.defined() ? rca_pointwise_weight.data_ptr<float>() : nullptr,
      rca_pointwise_bias.defined() ? rca_pointwise_bias.data_ptr<float>() : nullptr,
      output.data_ptr<float>(), ProduceGap ? gap.data_ptr<float>() : nullptr,
      se_weight1.defined() ? se_weight1.data_ptr<float>() : nullptr,
      se_bias1.defined() ? se_bias1.data_ptr<float>() : nullptr,
      se_weight2.defined() ? se_weight2.data_ptr<float>() : nullptr,
      se_bias2.defined() ? se_bias2.data_ptr<float>() : nullptr,
      ProduceSE ? static_cast<int>(se_weight1.size(0)) : 0,
      ProduceSE ? attention.data_ptr<float>() : nullptr);
}

template <int Channels, int Height, bool ProduceGap, bool WideOutput = false,
          bool ProduceSE = false>
void launch_shape(
    const torch::Tensor& input, const torch::Tensor& offset,
    const torch::Tensor& offset_bias, const torch::Tensor& offset_correction,
    const torch::Tensor& mask, const torch::Tensor& weight,
    const torch::Tensor& bias, const torch::Tensor& post_mul,
    const torch::Tensor& post_add, const torch::Tensor& post_scale,
    const torch::Tensor& fused_offset_weight,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    const torch::Tensor& se_weight1,
    const torch::Tensor& se_bias1,
    const torch::Tensor& se_weight2,
    const torch::Tensor& se_bias2,
    torch::Tensor& output, torch::Tensor& gap, torch::Tensor& attention,
    std::int64_t spatial_tile, cudaStream_t stream) {
  if (spatial_tile == 4) {
    // H4/H8 otherwise expose too few CTAs to occupy Turing's 68 SMs.
    constexpr int Threads = (ProduceGap || WideOutput)
        ? (Channels * 4 <= 64 ? 64 : (Channels * 4 <= 128 ? 128 :
           (Channels * 4 <= 256 ? 256 : 512)))
        : 64;
    launch_kernel<Channels, Height, 4, Threads, ProduceGap, ProduceSE>(
        input, offset, offset_bias, offset_correction, mask, weight, bias,
        post_mul, post_add, post_scale, fused_offset_weight,
        rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
        se_weight1, se_bias1, se_weight2, se_bias2,
        output, gap, attention, stream);
  } else if (spatial_tile == 8) {
    constexpr int Threads = (ProduceGap || WideOutput)
        ? (Channels * 8 <= 64 ? 64 : (Channels * 8 <= 128 ? 128 :
           (Channels * 8 <= 256 ? 256 : (Channels * 8 <= 512 ? 512 : 1024))))
        : (Channels <= 8 ? 64 : 128);
    launch_kernel<Channels, Height, 8, Threads, ProduceGap, ProduceSE>(
        input, offset, offset_bias, offset_correction, mask, weight, bias,
        post_mul, post_add, post_scale, fused_offset_weight,
        rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
        se_weight1, se_bias1, se_weight2, se_bias2,
        output, gap, attention, stream);
  } else if (spatial_tile == 16) {
    constexpr int Threads = (ProduceGap || WideOutput)
        ? (Channels * 16 <= 128 ? 128 : (Channels * 16 <= 256 ? 256 :
           (Channels * 16 <= 512 ? 512 : 1024)))
        : (Channels <= 12 ? 128 : 256);
    if constexpr (Channels <= 22 || (Channels > 32 && Channels <= 64)) {
      launch_kernel<Channels, Height, 16, Threads, ProduceGap, ProduceSE>(
          input, offset, offset_bias, offset_correction, mask, weight, bias,
          post_mul, post_add, post_scale, fused_offset_weight,
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
          se_weight1, se_bias1, se_weight2, se_bias2,
          output, gap, attention, stream);
    } else {
      TORCH_CHECK(false, "tile16 is unavailable for this channel count");
    }
  } else {
    TORCH_CHECK(!ProduceGap, "fused GAP does not support tile32");
    if constexpr (Channels <= 16) {
      launch_kernel<Channels, Height, 32, 256, false, false>(
          input, offset, offset_bias, offset_correction, mask, weight, bias,
          post_mul, post_add, post_scale, fused_offset_weight,
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
          se_weight1, se_bias1, se_weight2, se_bias2,
          output, gap, attention, stream);
    } else {
      TORCH_CHECK(false, "tile32 is unavailable for this channel count");
    }
  }
}

template <int Channels, int Height, int Threads>
__global__ void gap_se_kernel(
    const float* __restrict__ feature,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    int hidden_channels,
    float* __restrict__ attention) {
  constexpr int Spatial = Height * Height;
  constexpr int Warps = Threads / 32;
  constexpr int MaxHidden = Channels < 32 ? 32 : Channels;
  __shared__ float channel_gap[Channels];
  __shared__ float hidden[MaxHidden];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // One warp reduces a channel at a time. This replaces the per-spatial-tile
  // global atomics and the zero-filled GAP tensor used by the previous path.
  for (int channel = warp; channel < Channels; channel += Warps) {
    float sum = 0.0f;
    for (int spatial = lane; spatial < Spatial; spatial += 32) {
      sum += __ldg(feature + channel * Spatial + spatial);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    if (lane == 0) {
      channel_gap[channel] = sum / static_cast<float>(Spatial);
    }
  }
  __syncthreads();

  for (int hidden_channel = tid; hidden_channel < hidden_channels;
       hidden_channel += Threads) {
    float value = __ldg(bias1 + hidden_channel);
#pragma unroll 1
    for (int channel = 0; channel < Channels; ++channel) {
      value = fmaf(__ldg(weight1 + hidden_channel * Channels + channel),
                   channel_gap[channel], value);
    }
    hidden[hidden_channel] = value > 0.0f ? value : 0.0f;
  }
  __syncthreads();

  for (int channel = tid; channel < Channels; channel += Threads) {
    float value = __ldg(bias2 + channel);
#pragma unroll 1
    for (int hidden_channel = 0; hidden_channel < hidden_channels;
         ++hidden_channel) {
      value = fmaf(__ldg(weight2 + channel * hidden_channels + hidden_channel),
                   hidden[hidden_channel], value);
    }
    attention[channel] = kadc_sigmoid(value);
  }
}

__global__ void partial_gap_se_kernel(
    const float* __restrict__ partial,
    int partial_blocks,
    int channels,
    int spatial,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    int hidden_channels,
    float* __restrict__ attention) {
  constexpr int Threads = 256;
  constexpr int Warps = Threads / 32;
  constexpr int MaxChannels = 96;
  __shared__ float channel_gap[MaxChannels];
  __shared__ float hidden[MaxChannels];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  for (int channel = warp; channel < channels; channel += Warps) {
    float sum = 0.0f;
    for (int block = lane; block < partial_blocks; block += 32) {
      sum += __ldg(partial + block * channels + channel);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, delta);
    }
    if (lane == 0) {
      channel_gap[channel] = sum / static_cast<float>(spatial);
    }
  }
  __syncthreads();

  for (int hidden_channel = tid; hidden_channel < hidden_channels;
       hidden_channel += Threads) {
    float value = __ldg(bias1 + hidden_channel);
#pragma unroll 1
    for (int channel = 0; channel < channels; ++channel) {
      value = fmaf(__ldg(weight1 + hidden_channel * channels + channel),
                   channel_gap[channel], value);
    }
    hidden[hidden_channel] = value > 0.0f ? value : 0.0f;
  }
  __syncthreads();

  for (int channel = tid; channel < channels; channel += Threads) {
    float value = __ldg(bias2 + channel);
#pragma unroll 1
    for (int hidden_channel = 0; hidden_channel < hidden_channels;
         ++hidden_channel) {
      value = fmaf(__ldg(weight2 + channel * hidden_channels + hidden_channel),
                   hidden[hidden_channel], value);
    }
    attention[channel] = kadc_sigmoid(value);
  }
}

void launch_partial_gap_se(
    const torch::Tensor& partial,
    int channels,
    int spatial,
    const torch::Tensor& weight1,
    const torch::Tensor& bias1,
    const torch::Tensor& weight2,
    const torch::Tensor& bias2,
    torch::Tensor& attention,
    cudaStream_t stream) {
  TORCH_CHECK(channels <= 96 && weight1.size(0) <= 96,
              "partial GAP/SE exceeds compiled channel capacity");
  constexpr int Threads = 256;
  partial_gap_se_kernel<<<1, Threads, 0, stream>>>(
      partial.data_ptr<float>(), static_cast<int>(partial.size(0)), channels,
      spatial, weight1.data_ptr<float>(), bias1.data_ptr<float>(),
      weight2.data_ptr<float>(), bias2.data_ptr<float>(),
      static_cast<int>(weight1.size(0)), attention.data_ptr<float>());
}

template <int Channels, int Threads>
__global__ void se_from_gap_fixed_kernel(
    const float* __restrict__ gap,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    float* __restrict__ attention) {
  __shared__ float hidden[Channels];
  const int tid = threadIdx.x;
  if (tid < Channels) {
    float value = __ldg(bias1 + tid);
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
      value = fmaf(__ldg(weight1 + tid * Channels + channel),
                   __ldg(gap + channel), value);
    }
    hidden[tid] = value > 0.0f ? value : 0.0f;
  }
  __syncthreads();
  if (tid < Channels) {
    float value = __ldg(bias2 + tid);
#pragma unroll
    for (int hidden_channel = 0; hidden_channel < Channels; ++hidden_channel) {
      value = fmaf(__ldg(weight2 + tid * Channels + hidden_channel),
                   hidden[hidden_channel], value);
    }
    attention[tid] = kadc_sigmoid(value);
  }
}

template <int Threads>
__global__ void se_from_gap_serial_kernel(
    const float* __restrict__ gap,
    int channels,
    const float* __restrict__ weight1,
    const float* __restrict__ bias1,
    const float* __restrict__ weight2,
    const float* __restrict__ bias2,
    int hidden_channels,
    float* __restrict__ attention) {
  constexpr int MaxHidden = 96;
  __shared__ float hidden[MaxHidden];

  const int tid = threadIdx.x;
  // The input already is the final GAP vector.  Compute each small FC output
  // in one thread and avoid the generic partial-GAP reconstruction phase.
  for (int hidden_channel = tid; hidden_channel < hidden_channels;
       hidden_channel += Threads) {
    float value = __ldg(bias1 + hidden_channel);
#pragma unroll 1
    for (int channel = 0; channel < channels; ++channel) {
      value = fmaf(__ldg(weight1 + hidden_channel * channels + channel),
                   __ldg(gap + channel), value);
    }
    hidden[hidden_channel] = value > 0.0f ? value : 0.0f;
  }
  __syncthreads();

  for (int channel = tid; channel < channels; channel += Threads) {
    float value = __ldg(bias2 + channel);
#pragma unroll 1
    for (int hidden_channel = 0; hidden_channel < hidden_channels;
         ++hidden_channel) {
      value = fmaf(__ldg(weight2 + channel * hidden_channels + hidden_channel),
                   hidden[hidden_channel], value);
    }
    attention[channel] = kadc_sigmoid(value);
  }
}

torch::Tensor se_from_gap_impl(
    const torch::Tensor& gap,
    const torch::Tensor& weight1,
    const torch::Tensor& bias1,
    const torch::Tensor& weight2,
    const torch::Tensor& bias2) {
  TORCH_CHECK(gap.dim() == 2 && gap.size(0) == 1,
              "gap must be [1,C]");
  TORCH_CHECK(gap.is_cuda() && gap.scalar_type() == torch::kFloat32 &&
                  gap.is_contiguous(),
              "gap must be contiguous CUDA FP32");
  auto attention = torch::empty_like(gap);
  const auto stream = at::cuda::getCurrentCUDAStream(gap.get_device()).stream();
  constexpr int Threads = 128;
  const int channels = static_cast<int>(gap.size(1));
  const int hidden_channels = static_cast<int>(weight1.size(0));
#define KADC_SE_FROM_GAP_DISPATCH(C) \
  if (channels == (C) && hidden_channels == (C)) { \
    se_from_gap_fixed_kernel<(C), Threads><<<1, Threads, 0, stream>>>( \
        gap.data_ptr<float>(), weight1.data_ptr<float>(), bias1.data_ptr<float>(), \
        weight2.data_ptr<float>(), bias2.data_ptr<float>(), \
        attention.data_ptr<float>()); \
  } else
  KADC_SE_FROM_GAP_DISPATCH(42)
  KADC_SE_FROM_GAP_DISPATCH(44)
  KADC_SE_FROM_GAP_DISPATCH(85)
  KADC_SE_FROM_GAP_DISPATCH(86)
  {
    se_from_gap_serial_kernel<Threads><<<1, Threads, 0, stream>>>(
        gap.data_ptr<float>(), channels,
        weight1.data_ptr<float>(), bias1.data_ptr<float>(),
        weight2.data_ptr<float>(), bias2.data_ptr<float>(), hidden_channels,
        attention.data_ptr<float>());
  }
#undef KADC_SE_FROM_GAP_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return attention;
}

void launch_gap_se(
    const torch::Tensor& feature,
    const torch::Tensor& weight1,
    const torch::Tensor& bias1,
    const torch::Tensor& weight2,
    const torch::Tensor& bias2,
    torch::Tensor& attention,
    cudaStream_t stream) {
  const int channels = static_cast<int>(feature.size(1));
  const int height = static_cast<int>(feature.size(2));
  const int hidden_channels = static_cast<int>(weight1.size(0));
  constexpr int Threads = 256;
#define KADC_SE_DISPATCH(C, H) \
  if (channels == (C) && height == (H)) { \
    gap_se_kernel<(C), (H), Threads><<<1, Threads, 0, stream>>>( \
        feature.data_ptr<float>(), weight1.data_ptr<float>(), \
        bias1.data_ptr<float>(), weight2.data_ptr<float>(), \
        bias2.data_ptr<float>(), hidden_channels, attention.data_ptr<float>()); \
  } else
#if defined(KADC_FAMILY_S)
  KADC_SE_DISPATCH(12, 64)
  KADC_SE_DISPATCH(10, 32)
  KADC_SE_DISPATCH(16, 32)
  KADC_SE_DISPATCH(10, 16)
  KADC_SE_DISPATCH(16, 16)
  KADC_SE_DISPATCH(22, 16)
  KADC_SE_DISPATCH(16, 8)
  KADC_SE_DISPATCH(21, 8)
  KADC_SE_DISPATCH(21, 4)
#elif defined(KADC_FAMILY_M)
  KADC_SE_DISPATCH(12, 128)
  KADC_SE_DISPATCH(10, 64)
  KADC_SE_DISPATCH(20, 64)
  KADC_SE_DISPATCH(10, 32)
  KADC_SE_DISPATCH(20, 32)
  KADC_SE_DISPATCH(30, 32)
  KADC_SE_DISPATCH(20, 16)
  KADC_SE_DISPATCH(30, 16)
  KADC_SE_DISPATCH(40, 16)
  KADC_SE_DISPATCH(30, 8)
  KADC_SE_DISPATCH(40, 8)
  KADC_SE_DISPATCH(40, 4)
#elif defined(KADC_FAMILY_L)
  KADC_SE_DISPATCH(12, 128)
  KADC_SE_DISPATCH(10, 64)
  KADC_SE_DISPATCH(22, 64)
  KADC_SE_DISPATCH(10, 32)
  KADC_SE_DISPATCH(21, 32)
  KADC_SE_DISPATCH(44, 32)
  KADC_SE_DISPATCH(21, 16)
  KADC_SE_DISPATCH(42, 16)
  KADC_SE_DISPATCH(44, 16)
  KADC_SE_DISPATCH(42, 8)
  KADC_SE_DISPATCH(42, 4)
#elif defined(KADC_FAMILY_X)
  KADC_SE_DISPATCH(12, 128)
  KADC_SE_DISPATCH(10, 64)
  KADC_SE_DISPATCH(22, 64)
  KADC_SE_DISPATCH(10, 32)
  KADC_SE_DISPATCH(21, 32)
  KADC_SE_DISPATCH(44, 32)
  KADC_SE_DISPATCH(21, 16)
  KADC_SE_DISPATCH(42, 16)
  KADC_SE_DISPATCH(44, 16)
  KADC_SE_DISPATCH(86, 16)
  KADC_SE_DISPATCH(42, 8)
  KADC_SE_DISPATCH(85, 8)
  KADC_SE_DISPATCH(42, 4)
  KADC_SE_DISPATCH(85, 4)
#else
  KADC_SE_DISPATCH(6, 64)
  KADC_SE_DISPATCH(5, 32)
  KADC_SE_DISPATCH(8, 32)
  KADC_SE_DISPATCH(5, 16)
  KADC_SE_DISPATCH(8, 16)
  KADC_SE_DISPATCH(12, 16)
  KADC_SE_DISPATCH(8, 8)
  KADC_SE_DISPATCH(10, 8)
  KADC_SE_DISPATCH(10, 4)
#endif
  {
    TORCH_CHECK(false, "unsupported GAP/SE CUDA dispatch shape");
  }
#undef KADC_SE_DISPATCH
}

}  // namespace

torch::Tensor kadc_se_from_gap_cuda(
    const torch::Tensor& gap,
    const torch::Tensor& weight1,
    const torch::Tensor& bias1,
    const torch::Tensor& weight2,
    const torch::Tensor& bias2) {
  return se_from_gap_impl(gap, weight1, bias1, weight2, bias2);
}

namespace {

std::vector<torch::Tensor> kadc_dcn16_forward_impl(
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
    const torch::Tensor& fused_offset_weight,
    const torch::Tensor& rca_h1,
    const torch::Tensor& rca_vertical_weight,
    const torch::Tensor& rca_pointwise_weight,
    const torch::Tensor& rca_pointwise_bias,
    std::int64_t spatial_tile,
    bool produce_gap,
    bool wide_output = false,
    bool produce_se = false,
    const torch::Tensor& se_weight1 = torch::Tensor(),
    const torch::Tensor& se_bias1 = torch::Tensor(),
    const torch::Tensor& se_weight2 = torch::Tensor(),
    const torch::Tensor& se_bias2 = torch::Tensor()) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  auto output = torch::empty_like(input);
  const char* native_memset_env = std::getenv("KADC_DCN_NATIVE_MEMSET");
  const bool native_gap_memset =
      native_memset_env != nullptr && std::strcmp(native_memset_env, "1") == 0;
  const auto partial_blocks =
      (input.size(2) * input.size(3) + spatial_tile - 1) / spatial_tile;
  auto gap = produce_gap
      ? (produce_se
             ? torch::empty({partial_blocks, input.size(1)}, input.options())
             : (native_gap_memset
                    ? torch::empty({1, input.size(1)}, input.options())
                    : torch::zeros({1, input.size(1)}, input.options())))
      : torch::empty({0}, input.options());
  auto attention = torch::empty({0}, input.options());
  if (produce_gap && !produce_se && native_gap_memset) {
    // torch::zeros emits a general FillFunctor kernel for every KADC branch.
    // A native memset graph node is sufficient for FP32 zero and is markedly
    // cheaper on SM75 while preserving the exact atomic-GAP computation.
    C10_CUDA_CHECK(cudaMemsetAsync(
        gap.data_ptr<float>(), 0,
        static_cast<std::size_t>(gap.numel()) * sizeof(float), stream));
  }

  const int channels = static_cast<int>(input.size(1));
  const int height = static_cast<int>(input.size(2));
#define KADC_DISPATCH(C, H) \
  if (channels == (C) && height == (H)) { \
    if (produce_se) { \
      launch_shape<(C), (H), true, false, true>(input, offset, offset_bias, offset_correction, mask, \
          weight, bias, post_mul, post_add, post_scale, fused_offset_weight, \
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias, \
          se_weight1, se_bias1, se_weight2, se_bias2, \
          output, gap, attention, spatial_tile, stream); \
    } else if (produce_gap) { \
      launch_shape<(C), (H), true, false, false>(input, offset, offset_bias, offset_correction, mask, \
          weight, bias, post_mul, post_add, post_scale, fused_offset_weight, \
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias, \
          se_weight1, se_bias1, se_weight2, se_bias2, \
          output, gap, attention, spatial_tile, stream); \
    } else if (wide_output) { \
      launch_shape<(C), (H), false, true, false>(input, offset, offset_bias, offset_correction, mask, \
          weight, bias, post_mul, post_add, post_scale, fused_offset_weight, \
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias, \
          se_weight1, se_bias1, se_weight2, se_bias2, \
          output, gap, attention, spatial_tile, stream); \
    } else { \
      launch_shape<(C), (H), false, false, false>(input, offset, offset_bias, offset_correction, mask, \
          weight, bias, post_mul, post_add, post_scale, fused_offset_weight, \
          rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias, \
          se_weight1, se_bias1, se_weight2, se_bias2, \
          output, gap, attention, spatial_tile, stream); \
    } \
  } else
#if defined(KADC_FAMILY_S)
  KADC_DISPATCH(12, 64)
  KADC_DISPATCH(10, 32)
  KADC_DISPATCH(16, 32)
  KADC_DISPATCH(10, 16)
  KADC_DISPATCH(16, 16)
  KADC_DISPATCH(22, 16)
  KADC_DISPATCH(16, 8)
  KADC_DISPATCH(21, 8)
  KADC_DISPATCH(21, 4)
#elif defined(KADC_FAMILY_M)
  KADC_DISPATCH(12, 128)
  KADC_DISPATCH(10, 64)
  KADC_DISPATCH(20, 64)
  KADC_DISPATCH(10, 32)
  KADC_DISPATCH(20, 32)
  KADC_DISPATCH(30, 32)
  KADC_DISPATCH(20, 16)
  KADC_DISPATCH(30, 16)
  KADC_DISPATCH(40, 16)
  KADC_DISPATCH(30, 8)
  KADC_DISPATCH(40, 8)
  KADC_DISPATCH(40, 4)
#elif defined(KADC_FAMILY_L)
  KADC_DISPATCH(12, 128)
  KADC_DISPATCH(10, 64)
  KADC_DISPATCH(22, 64)
  KADC_DISPATCH(10, 32)
  KADC_DISPATCH(21, 32)
  KADC_DISPATCH(44, 32)
  KADC_DISPATCH(21, 16)
  KADC_DISPATCH(42, 16)
  KADC_DISPATCH(44, 16)
  KADC_DISPATCH(42, 8)
  KADC_DISPATCH(42, 4)
#elif defined(KADC_FAMILY_X)
  KADC_DISPATCH(12, 128)
  KADC_DISPATCH(10, 64)
  KADC_DISPATCH(22, 64)
  KADC_DISPATCH(10, 32)
  KADC_DISPATCH(21, 32)
  KADC_DISPATCH(44, 32)
  KADC_DISPATCH(21, 16)
  KADC_DISPATCH(42, 16)
  KADC_DISPATCH(44, 16)
  KADC_DISPATCH(86, 16)
  KADC_DISPATCH(42, 8)
  KADC_DISPATCH(85, 8)
  KADC_DISPATCH(42, 4)
  KADC_DISPATCH(85, 4)
#else
  KADC_DISPATCH(6, 64)
  KADC_DISPATCH(5, 32)
  KADC_DISPATCH(8, 32)
  KADC_DISPATCH(5, 16)
  KADC_DISPATCH(8, 16)
  KADC_DISPATCH(12, 16)
  KADC_DISPATCH(8, 8)
  KADC_DISPATCH(10, 8)
  KADC_DISPATCH(10, 4)
  KADC_DISPATCH(16, 32)
#endif
  {
    TORCH_CHECK(false, "unsupported CUDA dispatch shape");
  }
#undef KADC_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, gap};
}

}  // namespace

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
    std::int64_t spatial_tile) {
  return kadc_dcn16_forward_impl(
      input, offset, offset_bias, offset_correction, mask, weight, bias,
      post_mul, post_add, post_scale, torch::Tensor(),
      torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
      spatial_tile, false)[0];
}

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
    std::int64_t spatial_tile) {
  return kadc_dcn16_forward_impl(
      input, offset, offset_bias, offset_correction, mask, weight, bias,
      post_mul, post_add, post_scale, torch::Tensor(),
      torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
      spatial_tile, true);
}

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
    std::int64_t spatial_tile) {
  // The offset/correction tensors are not dereferenced when a fused offset
  // weight is present; input is a defined same-device placeholder.
  return kadc_dcn16_forward_impl(
      input, input, fused_offset_bias, input, mask, weight, bias,
      post_mul, post_add, post_scale, fused_offset_weight,
      torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
      spatial_tile, true);
}

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
    std::int64_t spatial_tile) {
  return kadc_dcn16_forward_impl(
      input, offset, offset_bias, offset_correction, rca_h1, weight, bias,
      post_mul, post_add, post_scale, torch::Tensor(),
      rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
      spatial_tile, true);
}

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
    std::int64_t spatial_tile) {
  const auto hidden_channels = se_weight1.size(0);
  const auto max_hidden = std::max<std::int64_t>(32, input.size(1));
  TORCH_CHECK(hidden_channels > 0 && hidden_channels <= max_hidden,
              "SE hidden channels exceed compiled family capacity");
  TORCH_CHECK(se_weight1.dim() == 2 && se_weight1.size(1) == input.size(1),
              "SE weight1 must be [hidden,C]");
  TORCH_CHECK(se_bias1.numel() == hidden_channels,
              "SE bias1 must contain hidden values");
  TORCH_CHECK(se_weight2.dim() == 2 && se_weight2.size(0) == input.size(1) &&
                  se_weight2.size(1) == hidden_channels,
              "SE weight2 must be [C,hidden]");
  TORCH_CHECK(se_bias2.numel() == input.size(1),
              "SE bias2 must contain C values");

  auto result = kadc_dcn16_forward_impl(
      input, offset, offset_bias, offset_correction, rca_h1, weight, bias,
      post_mul, post_add, post_scale, torch::Tensor(),
      rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
      spatial_tile, false, true);
  auto attention = torch::empty({1, input.size(1)}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  launch_gap_se(result[0], se_weight1, se_bias1, se_weight2, se_bias2,
                attention, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {result[0], attention};
}

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
    std::int64_t spatial_tile) {
  const auto hidden_channels = se_weight1.size(0);
  const auto max_hidden = std::max<std::int64_t>(32, input.size(1));
  TORCH_CHECK(hidden_channels > 0 && hidden_channels <= max_hidden,
              "SE hidden channels exceed compiled family capacity");
  TORCH_CHECK(se_weight1.dim() == 2 && se_weight1.size(1) == input.size(1),
              "SE weight1 must be [hidden,C]");
  TORCH_CHECK(se_bias1.numel() == hidden_channels,
              "SE bias1 must contain hidden values");
  TORCH_CHECK(se_weight2.dim() == 2 && se_weight2.size(0) == input.size(1) &&
                  se_weight2.size(1) == hidden_channels,
              "SE weight2 must be [C,hidden]");
  TORCH_CHECK(se_bias2.numel() == input.size(1),
              "SE bias2 must contain C values");

  auto result = kadc_dcn16_forward_impl(
      input, offset, offset_bias, offset_correction, rca_h1, weight, bias,
      post_mul, post_add, post_scale, torch::Tensor(),
      rca_h1, rca_vertical_weight, rca_pointwise_weight, rca_pointwise_bias,
      spatial_tile, true, false, true,
      se_weight1, se_bias1, se_weight2, se_bias2);
  auto attention = torch::empty({1, input.size(1)}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  launch_partial_gap_se(
      result[1], static_cast<int>(input.size(1)),
      static_cast<int>(input.size(2) * input.size(3)),
      se_weight1, se_bias1, se_weight2, se_bias2, attention, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {result[0], attention};
}
