#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/core/platform/mapped_file.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/expert_cache.hpp"
#include "tests/support/temporary_directory.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace r = gufo::models::qwen38_flash_next::rocm;
namespace {
void Hip(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Gpu {
  std::array<hipStream_t, 2> streams{};
  std::int32_t* ids{nullptr};
  float* observed{nullptr};
  Gpu() {
    for (auto& stream : streams)
      Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Hip(hipMalloc(&ids, 16 * sizeof(std::int32_t)));
    Hip(hipMalloc(&observed, sizeof(float)));
    std::array<std::int32_t, 16> host{};
    for (int i = 0; i < 16; ++i)
      host[i] = i;
    Hip(hipMemcpy(ids, host.data(), sizeof(host), hipMemcpyHostToDevice));
  }
  ~Gpu() {
    for (auto s : streams) {
      (void)hipStreamSynchronize(s);
      (void)hipStreamDestroy(s);
    }
    (void)hipFree(ids);
    (void)hipFree(observed);
  }
};
float Value(int layer, int matrix, int expert) {
  return 1000.0f * layer + 100.0f * matrix + expert + 0.25f;
}
void Hold(void* value) {
  auto* released = static_cast<std::atomic<bool>*>(value);
  while (!released->load())
    std::this_thread::yield();
}
}  // namespace

int main() {
  try {
    TemporaryDirectory files("gufo-expert-cache");
    const auto path = files.path() / "weights";
    constexpr int experts = 16, dim = 256;
    constexpr auto matrix_bytes = std::size_t(dim) * dim * sizeof(float);
    q::ModelWeights weights;
    weights.config.num_experts = experts;
    weights.layers.resize(3);
    std::ofstream out(path, std::ios::binary);
    std::uint64_t offset = 0;
    for (int layer = 0; layer < 3; ++layer) {
      auto& l = weights.layers[layer];
      const std::array tensors{&l.ffn_gate_exps, &l.ffn_up_exps,
                               &l.ffn_down_exps};
      for (int matrix = 0; matrix < 3; ++matrix) {
        *tensors[matrix] = {reinterpret_cast<void*>(1),
                            gufo::core::GgmlType::kF32,
                            dim,
                            dim,
                            experts,
                            offset,
                            0,
                            {}};
        for (int e = 0; e < experts; ++e) {
          std::vector<float> data(dim * dim, Value(layer, matrix, e));
          out.write(reinterpret_cast<const char*>(data.data()), matrix_bytes);
          offset += matrix_bytes;
        }
      }
    }
    out.close();
    auto mapped = gufo::platform::ReadOnlyMappedFile::Open(path);
    Require(mapped.valid(), "open fixture");
    const gufo::core::GgufMappedRegion region{nullptr, mapped.size(),
                                              mapped.PlatformHandle()};
    std::string error;
    auto uploader = gufo::hip::WeightUpload::Create({&region, 1}, &error);
    Require(bool(uploader), error);
    // Exactly one layer fits. Switching layers must evict every old entry.
    const auto overhead = gufo::hip::WeightUpload::StagingBytes() +
                          6 * experts * sizeof(void*) +
                          3 * (experts * matrix_bytes + 4096);
    const auto budget = overhead + experts * 3 * (matrix_bytes + 4096);
    const gufo::hip::WeightAllocator allocator(budget);
    std::vector<r::DeviceLayer> layers(3);
    r::DeviceLayer draft;
    auto cache =
        r::ExpertCache::Create(weights, nullptr, 1, budget, allocator,
                               std::move(uploader), layers, draft, &error);
    Require(bool(cache), error);
    Gpu gpu;
    const auto check = [&](int layer, bool direct, hipStream_t stream) {
      auto lease = cache->Begin(layer, gpu.ids, experts, stream, &error);
      Require(bool(lease), error);
      Require(lease->Ready(direct, &error), error);
      const std::array tensors{layers[layer].ffn_gate_exps,
                               layers[layer].ffn_up_exps,
                               layers[layer].ffn_down_exps};
      for (int m = 0; m < 3; ++m) {
        std::array<const void*, experts> pointers{};
        Hip(hipMemcpyAsync(pointers.data(), tensors[m].expert_pointers,
                           sizeof(pointers), hipMemcpyDeviceToHost, stream));
        Hip(hipStreamSynchronize(stream));
        for (int e = 0; e < experts; ++e) {
          float actual = 0;
          const auto* source = direct
                                   ? pointers[e]
                                   : static_cast<const char*>(tensors[m].data) +
                                         e * matrix_bytes;
          Hip(hipMemcpy(&actual, source, sizeof(float), hipMemcpyDeviceToHost));
          Require(actual == Value(layer, m, e),
                  "wrong expert payload after eviction");
        }
      }
      Require(lease->Complete(true, &error), error);
    };
    check(0, true, gpu.streams[0]);
    check(0, true, gpu.streams[0]);  // Hits without I/O.
    auto lease = cache->Begin(0, gpu.ids, experts, gpu.streams[0], &error);
    Require(bool(lease) && lease->Ready(true, &error), error);
    const void* previous = nullptr;
    Hip(hipMemcpyAsync(&previous, layers[0].ffn_gate_exps.expert_pointers,
                       sizeof(previous), hipMemcpyDeviceToHost,
                       gpu.streams[0]));
    Hip(hipStreamSynchronize(gpu.streams[0]));
    std::atomic<bool> released{false};
    Hip(hipLaunchHostFunc(gpu.streams[0], Hold, &released));
    Hip(hipMemcpyAsync(gpu.observed, previous, sizeof(float),
                       hipMemcpyDeviceToDevice, gpu.streams[0]));
    std::jthread release([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      released.store(true);
    });
    Require(lease->Complete(true, &error), error);
    Require(!released.load(), "Complete blocked on consumer execution");
    lease.reset();
    check(1, true, gpu.streams[1]);  // Must wait before evicting old slots.
    float actual = 0;
    Hip(hipMemcpy(&actual, gpu.observed, sizeof(float), hipMemcpyDeviceToHost));
    Require(actual == Value(0, 0, 0),
            "peer stream evicted an in-flight expert");
    check(2, false, gpu.streams[0]);  // Prefill workspace fallback.
    check(0, true, gpu.streams[1]);
    // Failure is sticky and every pending I/O is drained before freeing slots.
    std::filesystem::resize_file(path, 1);
    auto failed = cache->Begin(1, gpu.ids, experts, gpu.streams[0], &error);
    Require(bool(failed), error);
    Require(!failed->Ready(true, &error), "truncated shard must fail");
    failed.reset();
    Require(!cache->Begin(1, gpu.ids, experts, gpu.streams[1], &error),
            "failed cache must reject reuse");
    cache.reset();
    std::cout << "PASS: cache hits, eviction, prefill fallback, peer-stream "
                 "lifetime and I/O failure\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
