#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_TRACE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_TRACE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {

/// Records one routing invocation in stream order, including graph replays.
/// Header: uint64 row count and routing sequence; records: 14 uint32 words.
void TraceExpertUsage(const std::int32_t* ids, unsigned long long* trace,
                      std::size_t capacity, std::uint32_t rows,
                      std::uint32_t layer, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif
