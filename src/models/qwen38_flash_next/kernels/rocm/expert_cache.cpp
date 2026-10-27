#include "src/models/qwen38_flash_next/kernels/rocm/expert_cache.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kTail = 4096;
constexpr std::size_t kMiB = 1ULL << 20;
}  // namespace

bool ExpertCache::Fail(const std::string& message, std::string* error) {
  if (failure_.empty())
    failure_ = "expert cache: " + message;
  if (error)
    *error = failure_;
  return false;
}

std::unique_ptr<ExpertCache> ExpertCache::Create(
    const ModelWeights& weights, const MtpWeights* mtp,
    std::uint32_t mtp_shard_base, std::size_t budget,
    const hip::WeightAllocator& allocator,
    std::unique_ptr<hip::WeightUpload> upload, std::vector<DeviceLayer>& layers,
    DeviceLayer& draft, std::string* error) {
  std::unique_ptr<ExpertCache> cache(new ExpertCache());
  auto& c = *cache;
  c.upload_ = std::move(upload);
  c.experts_ = weights.config.num_experts;
  std::array<std::size_t, 3> workspace_bytes{};
  const auto add = [&](const LayerWeights& w, std::uint32_t shard_base) {
    Layer layer;
    const std::array tensors{w.ffn_gate_exps, w.ffn_up_exps, w.ffn_down_exps};
    std::size_t stride = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& t = tensors[i];
      const auto size = t.RowBytes() * t.rows;
      layer.tensors[i] = {shard_base + t.shard, t.file_offset, size};
      workspace_bytes[i] = std::max(workspace_bytes[i], t.SizeBytes() + kTail);
      stride += size + kTail;
    }
    c.stride_ = std::max(c.stride_, stride);
    c.layers_.push_back(layer);
  };
  for (const auto& layer : weights.layers)
    add(layer, 0);
  if (mtp)
    add(mtp->block, mtp_shard_base);
  const auto pointer_bytes = 3 * c.experts_ * sizeof(void*);
  std::size_t overhead = hip::WeightUpload::StagingBytes() + 2 * pointer_bytes;
  for (auto size : workspace_bytes)
    overhead += size;
  // Pin every distinct expert in a routing invocation. A prefill can request
  // all experts of one layer, irrespective of its number of token rows.
  if (c.experts_ == 0 || c.stride_ == 0 || budget < overhead ||
      (budget - overhead) / c.stride_ < c.experts_) {
    c.Fail(
        "budget too small; need at least " +
            std::to_string((overhead + c.stride_ * c.experts_ + kMiB - 1) /
                           kMiB) +
            " MiB for one layer of pinned experts, workspace and I/O buffers",
        error);
    return nullptr;
  }
  const auto keys = c.layers_.size() * c.experts_;
  const auto capacity = std::min(keys, (budget - overhead) / c.stride_);
  c.slots_.resize(capacity);
  for (std::size_t i = 0; i < capacity; ++i) {
    c.lru_.push_back(i);
    c.slots_[i] = {kAbsent, std::prev(c.lru_.end())};
  }
  c.slot_for_key_.assign(keys, kAbsent);
  c.bytes_ = overhead + capacity * c.stride_;
  if (allocator.Allocate(&c.storage_, capacity * c.stride_) != hipSuccess) {
    c.Fail("backing allocation failed", error);
    return nullptr;
  }
  for (std::size_t i = 0; i < c.workspace_.size(); ++i) {
    if (allocator.Allocate(&c.workspace_[i], workspace_bytes[i]) !=
            hipSuccess ||
        hipMemset(c.workspace_[i], 0, workspace_bytes[i]) != hipSuccess) {
      c.Fail("layer workspace allocation failed", error);
      return nullptr;
    }
  }
  if (c.host_.Allocate(&c.pointers_host_, pointer_bytes) != hipSuccess ||
      allocator.Allocate(&c.pointers_device_, pointer_bytes) != hipSuccess ||
      hipEventCreateWithFlags(&c.routed_, hipEventDisableTiming) !=
          hipSuccess ||
      hipEventCreateWithFlags(&c.consumed_, hipEventDisableTiming) !=
          hipSuccess) {
    c.Fail("pointer table or dependency event allocation failed", error);
    return nullptr;
  }
  const auto bind = [&](DeviceLayer& d, const LayerWeights& w,
                        std::uint32_t index) {
    d.cache_layer = index;
    const std::array tensors{w.ffn_gate_exps, w.ffn_up_exps, w.ffn_down_exps};
    const std::array dest{&d.ffn_gate_exps, &d.ffn_up_exps, &d.ffn_down_exps};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& t = tensors[i];
      *dest[i] = {c.workspace_[i],
                  t.type,
                  static_cast<std::uint32_t>(t.cols),
                  static_cast<std::uint32_t>(t.rows),
                  static_cast<std::uint32_t>(t.experts),
                  c.pointers_device_ + i * c.experts_};
    }
  };
  for (std::size_t i = 0; i < layers.size(); ++i)
    bind(layers[i], weights.layers[i], static_cast<std::uint32_t>(i));
  if (mtp)
    bind(draft, mtp->block, static_cast<std::uint32_t>(layers.size()));
  std::cerr << "expert cache: " << capacity << " slots across "
            << c.layers_.size() << " layers, " << c.bytes_ / kMiB
            << " MiB reserved (" << overhead / kMiB
            << " MiB workspace and I/O); graph replay disabled\n";
  return cache;
}

ExpertCache::~ExpertCache() {
  // Finish workers before freeing their destinations, including failed loads.
  if (consumed_)
    (void)hipEventSynchronize(consumed_);
  upload_.reset();
  if (routed_)
    (void)hipEventDestroy(routed_);
  if (consumed_)
    (void)hipEventDestroy(consumed_);
  if (pointers_device_)
    (void)hipFree(pointers_device_);
  for (void* p : workspace_)
    if (p)
      (void)hipFree(p);
  if (storage_)
    (void)hipFree(storage_);
  if (hits_ || misses_)
    std::cerr << "expert cache: " << hits_ << " hits, " << misses_
              << " misses, " << read_bytes_ / kMiB << " MiB read\n";
}

ExpertCache::Lease::Lease(ExpertCache& cache, std::uint32_t layer,
                          hipStream_t stream)
    : cache_(cache), lock_(cache.mutex_), layer_(layer), stream_(stream) {}

ExpertCache::Lease::~Lease() {
  // Error/exception exits also drain readers and consumers before releasing
  // the shared workspace. A failed upload poisons the cache permanently.
  if (!complete_) {
    if (!cache_.upload_->Finish(nullptr))
      cache_.Fail("upload failed", nullptr);
    if (hipStreamSynchronize(stream_) != hipSuccess)
      cache_.Fail("consumer stream failed", nullptr);
  }
}

std::unique_ptr<ExpertCache::Lease> ExpertCache::Begin(
    std::uint32_t layer, const std::int32_t* device_ids, std::size_t count,
    hipStream_t stream, std::string* error) {
  std::unique_ptr<Lease> lease(new Lease(*this, layer, stream));
  if (!failure_.empty()) {
    Fail(failure_, error);
    return nullptr;
  }
  if (layer >= layers_.size() || count == 0) {
    Fail("invalid routing request", error);
    return nullptr;
  }
  if (count > ids_capacity_) {
    const auto capacity = std::max(count, ids_capacity_ * 2);
    if (host_.Allocate(&ids_host_, capacity * sizeof(std::int32_t)) !=
        hipSuccess) {
      Fail("routing staging allocation failed", error);
      return nullptr;
    }
    ids_capacity_ = capacity;
  }
  lease->count_ = count;
  // A peer stream may still be consuming the previous lease. Order the
  // routing event after that consumer before the CPU can evict its slots.
  if (hipStreamWaitEvent(stream, consumed_, 0) != hipSuccess ||
      hipMemcpyAsync(ids_host_, device_ids, count * sizeof(std::int32_t),
                     hipMemcpyDeviceToHost, stream) != hipSuccess ||
      hipEventRecord(routed_, stream) != hipSuccess) {
    Fail("routing download failed", error);
    return nullptr;
  }
  return lease;
}

bool ExpertCache::Lease::Ready(bool direct, std::string* error) {
  if (hipEventSynchronize(cache_.routed_) != hipSuccess)
    return cache_.Fail("routing event failed", error);
  return cache_.Prepare(layer_, {cache_.ids_host_, count_}, stream_, direct,
                        error);
}

bool ExpertCache::Lease::Complete(bool success, std::string* error) {
  const auto status = hipEventRecord(cache_.consumed_, stream_);
  complete_ = status == hipSuccess;
  if (status != hipSuccess)
    return cache_.Fail(
        "consumer stream failed: " + std::string(hipGetErrorString(status)),
        error);
  return success;
}

bool ExpertCache::Prepare(std::uint32_t layer,
                          std::span<const std::int32_t> ids, hipStream_t stream,
                          bool direct, std::string* error) {
  const auto& source = layers_[layer];
  const auto base = static_cast<std::size_t>(layer) * experts_;
  std::vector<bool> required(experts_, false);
  std::vector<bool> pinned(slots_.size(), false);
  for (auto id : ids) {
    if (id < 0 || static_cast<std::uint32_t>(id) >= experts_)
      return Fail("router returned an invalid expert", error);
    required[id] = true;
    const auto slot = slot_for_key_[base + id];
    if (slot != kAbsent)
      pinned[slot] = true;
  }
  // Touch in token-row/router-rank order. Repeated selections count as hits;
  // every distinct expert required by this invocation stays pinned.
  for (auto id : ids) {
    const auto key = base + id;
    auto slot = slot_for_key_[key];
    if (slot == kAbsent) {
      ++misses_;
      for (auto i : lru_) {
        if (!pinned[i]) {
          slot = i;
          break;
        }
      }
      if (slot == kAbsent)
        return Fail("no unpinned slot available", error);
      if (slots_[slot].key != kAbsent)
        slot_for_key_[slots_[slot].key] = kAbsent;
      slot_for_key_[key] = slot;
      slots_[slot].key = key;
      pinned[slot] = true;
      auto* dst = static_cast<char*>(storage_) + slot * stride_;
      for (const auto& tensor : source.tensors) {
        if (!upload_->Copy(tensor.shard, tensor.offset + id * tensor.bytes,
                           tensor.bytes, dst, error))
          return Fail(error ? *error : "upload failed", error);
        if (hipMemsetAsync(dst + tensor.bytes, 0, kTail, stream) != hipSuccess)
          return Fail("slot padding clear failed", error);
        dst += tensor.bytes + kTail;
        read_bytes_ += tensor.bytes;
      }
    } else {
      ++hits_;
    }
    lru_.splice(lru_.end(), lru_, slots_[slot].recency);
  }
  if (!upload_->Finish(error))
    return Fail(error ? *error : "upload failed", error);
  // Tables are tiny (12 KiB for 512 experts). The previous routing event
  // also drained the last table upload, so the host buffer can be reused.
  std::fill_n(pointers_host_, 3 * experts_, nullptr);
  for (std::uint32_t id = 0; id < experts_; ++id) {
    if (!required[id])
      continue;
    auto* src =
        static_cast<const char*>(storage_) + slot_for_key_[base + id] * stride_;
    for (std::size_t i = 0; i < source.tensors.size(); ++i) {
      pointers_host_[i * experts_ + id] = src;
      src += source.tensors[i].bytes + kTail;
    }
  }
  if (hipMemcpyAsync(pointers_device_, pointers_host_,
                     3 * experts_ * sizeof(void*), hipMemcpyHostToDevice,
                     stream) != hipSuccess)
    return Fail("pointer table upload failed", error);
  if (direct)
    return true;
  // Quantized kernels can read past an expert's final row. Clear those
  // tails before copying ANY weights: a tail may overlap the next selected
  // expert. Otherwise bytes left by another layer/encoding could decode as
  // NaN even where activation padding is zero.
  for (std::uint32_t id = 0; id < experts_; ++id) {
    if (!required[id])
      continue;
    for (std::size_t i = 0; i < source.tensors.size(); ++i) {
      auto* tail = static_cast<char*>(workspace_[i]) +
                   (id + 1) * source.tensors[i].bytes;
      if (hipMemsetAsync(tail, 0, kTail, stream) != hipSuccess)
        return Fail("workspace padding clear failed", error);
    }
  }
  // Preserve original IDs and tensor layouts, including all prefill paths.
  // Only selected expert regions are read by the unchanged routed kernels.
  for (std::uint32_t id = 0; id < experts_; ++id) {
    if (!required[id])
      continue;
    const auto slot = slot_for_key_[base + id];
    const auto* src = static_cast<const char*>(storage_) + slot * stride_;
    for (std::size_t i = 0; i < source.tensors.size(); ++i) {
      const auto bytes = source.tensors[i].bytes;
      auto* dst = static_cast<char*>(workspace_[i]) + id * bytes;
      if (hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream) !=
          hipSuccess)
        return Fail("workspace copy failed", error);
      src += bytes + kTail;
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
