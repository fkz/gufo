#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_CACHE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXPERT_CACHE_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

// Shared, byte-bounded LRU of unchanged GGUF expert payloads. Decode reads
// slots directly; prefill can use a reusable layer workspace. GPU events
// protect slots, tables and workspace after a CPU lease is released.
class ExpertCache {
public:
  static std::unique_ptr<ExpertCache> Create(
      const ModelWeights& weights, const MtpWeights* mtp,
      std::uint32_t mtp_shard_base, std::size_t budget,
      const hip::WeightAllocator& allocator,
      std::unique_ptr<hip::WeightUpload> upload,
      std::vector<DeviceLayer>& layers, DeviceLayer& draft, std::string* error);
  ~ExpertCache();
  std::size_t resident_bytes() const noexcept { return bytes_; }

  class Lease {
  public:
    ~Lease();
    // Called after shared-expert work is queued. Wait only for the routing
    // event, then overlap disk uploads with that independent GPU work.
    bool Ready(bool direct, std::string* error);
    bool Complete(bool success, std::string* error);

  private:
    friend class ExpertCache;
    Lease(ExpertCache& cache, std::uint32_t layer, hipStream_t stream);
    ExpertCache& cache_;
    std::unique_lock<std::mutex> lock_;
    std::uint32_t layer_;
    hipStream_t stream_;
    std::size_t count_{0};
    bool complete_{false};
  };

  std::unique_ptr<Lease> Begin(std::uint32_t layer,
                               const std::int32_t* device_ids,
                               std::size_t count, hipStream_t stream,
                               std::string* error);

private:
  ExpertCache() = default;
  struct Source {
    std::uint32_t shard;
    std::uint64_t offset;
    std::size_t bytes;
  };
  struct Layer {
    std::array<Source, 3> tensors;
  };
  struct Slot {
    std::size_t key;
    std::list<std::size_t>::iterator recency;
  };
  bool Fail(const std::string& message, std::string* error);
  bool Prepare(std::uint32_t layer, std::span<const std::int32_t> ids,
               hipStream_t stream, bool direct, std::string* error);
  std::mutex mutex_;
  std::unique_ptr<hip::WeightUpload> upload_;
  std::vector<Layer> layers_;
  std::vector<Slot> slots_;
  std::list<std::size_t> lru_;
  std::vector<std::size_t> slot_for_key_;
  std::array<void*, 3> workspace_{};
  void* storage_{nullptr};
  const void** pointers_device_{nullptr};
  hip::HostAllocations host_;
  const void** pointers_host_{nullptr};
  std::int32_t* ids_host_{nullptr};
  std::size_t ids_capacity_{0};
  hipEvent_t routed_{nullptr};
  hipEvent_t consumed_{nullptr};
  std::size_t stride_{0};
  std::size_t bytes_{0};
  std::uint32_t experts_{0};
  std::uint64_t hits_{0}, misses_{0}, read_bytes_{0};
  std::string failure_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm
#endif
