#include "src/models/qwen38_flash_next/kernels/rocm/expert_trace.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

__global__ void TraceExpertUsageKernel(const std::int32_t* ids,
                                       unsigned long long* trace,
                                       std::size_t capacity, std::uint32_t rows,
                                       std::uint32_t layer) {
  // One block per invocation preserves token-row order without atomics per row.
  __shared__ unsigned long long base, sequence;
  if (threadIdx.x == 0) {
    base = trace[0];
    sequence = trace[1]++;
    trace[0] += rows;
  }
  __syncthreads();
  if (base + rows > capacity)
    return;
  auto* records = reinterpret_cast<std::uint32_t*>(trace + 2);
  for (std::uint32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    auto* record = records + (base + row) * 14;
    record[0] = static_cast<std::uint32_t>(sequence);
    record[1] = static_cast<std::uint32_t>(sequence >> 32);
    record[2] = layer;
    record[3] = row;
    for (unsigned i = 0; i < 10; ++i)
      record[4 + i] = ids[row * 10 + i];
  }
}

void TraceExpertUsage(const std::int32_t* ids, unsigned long long* trace,
                      std::size_t capacity, std::uint32_t rows,
                      std::uint32_t layer, hipStream_t stream) {
  hipLaunchKernelGGL(TraceExpertUsageKernel, dim3(1), dim3(256), 0, stream, ids,
                     trace, capacity, rows, layer);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
