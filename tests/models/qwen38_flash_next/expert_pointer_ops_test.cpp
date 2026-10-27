#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {
void Hip(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("expert pointer projection mismatch");
}
struct Memory {
  std::vector<void*> owned;
  ~Memory() {
    for (auto p : owned)
      (void)hipFree(p);
  }
  template<class T = float>
  T* Make(std::size_t count) {
    void* p = nullptr;
    Hip(hipMalloc(&p, count * sizeof(T)));
    owned.push_back(p);
    return static_cast<T*>(p);
  }
};
std::uint32_t seed = 1777;
std::uint32_t Random() {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}
struct Weights {
  void* resident;
  void* scattered;
  const void** table;
  std::size_t expert_bytes;
};
Weights MakeWeights(Memory& memory, int type, int m, int k, int experts = 16) {
  const int block = type == 12 || type == 13 ? 256 : 32;
  const int bytes = type == 12 ? 144 : type == 13 ? 176 : type == 7 ? 24 : 34;
  const auto expert_bytes = std::size_t(m) * (k / block) * bytes;
  std::vector<std::uint8_t> packed(experts * expert_bytes);
  for (auto& b : packed)
    b = static_cast<std::uint8_t>(Random());
  for (std::size_t i = 0; i < packed.size(); i += bytes) {
    const __half d = __float2half(type == 8 ? 0.0005f : 0.0001f);
    std::memcpy(packed.data() + i, &d, sizeof(d));
    if (type != 8) {
      const __half minimum = __float2half(type == 7 ? -0.003f : 0.0002f);
      std::memcpy(packed.data() + i + 2, &minimum, sizeof(minimum));
    }
  }
  auto* resident = memory.Make<std::uint8_t>(packed.size());
  Hip(hipMemcpy(resident, packed.data(), packed.size(), hipMemcpyHostToDevice));
  // Gaps and a nontrivial permutation catch accidental contiguous indexing.
  const auto stride = expert_bytes + 4096;
  auto* scattered = memory.Make<std::uint8_t>(experts * stride);
  Hip(hipMemset(scattered, 0, experts * stride));
  std::vector<const void*> pointers(experts);
  for (int e = 0; e < experts; ++e) {
    auto* dst = scattered + ((e * 7 + 3) % experts) * stride;
    Hip(hipMemcpy(dst, packed.data() + e * expert_bytes, expert_bytes,
                  hipMemcpyHostToDevice));
    pointers[e] = dst;
  }
  auto** table = memory.Make<const void*>(experts);
  Hip(hipMemcpy(table, pointers.data(), pointers.size() * sizeof(void*),
                hipMemcpyHostToDevice));
  return {resident, scattered, table, expert_bytes};
}
void Same(const float* a, const float* b, std::size_t count) {
  std::vector<float> expected(count), actual(count);
  Hip(hipMemcpy(expected.data(), a, count * sizeof(float),
                hipMemcpyDeviceToHost));
  Hip(hipMemcpy(actual.data(), b, count * sizeof(float),
                hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < count; ++i) {
    Require(std::isfinite(actual[i]));
    if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
      std::cerr << "first differing value " << i << ": " << expected[i]
                << " vs " << actual[i] << '\n';
      Require(false);
    }
  }
}
void Check(int type, int tokens, bool gated, bool prequantized, int m) {
  constexpr int experts = 16;
  const int k = gated || type == 12 || type == 13 ? 2560 : 640;
  const int used = gated ? 10 : 1;
  Memory memory;
  const auto gate = MakeWeights(memory, type, m, k);
  const auto up = MakeWeights(memory, type, m, k);
  auto* x = memory.Make(std::size_t(tokens) * k);
  std::vector<float> input(std::size_t(tokens) * k);
  for (auto& v : input)
    v = (static_cast<int>(Random() % 2001) - 1000) / 4000.0f;
  Hip(hipMemcpy(x, input.data(), input.size() * sizeof(float),
                hipMemcpyHostToDevice));
  std::vector<std::int32_t> ids(tokens * used);
  for (int t = 0; t < tokens; ++t)
    for (int u = 0; u < used; ++u)
      ids[t * used + u] = (t * 3 + u * 7) % experts;
  if (!gated)
    ids.back() = -1;  // Inactive sorted down rows must still write zero.
  auto* d_ids = memory.Make<std::int32_t>(ids.size());
  Hip(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(ids[0]),
                hipMemcpyHostToDevice));
  const auto count = std::size_t(tokens) * used * m;
  auto* reference = memory.Make(count);
  auto* actual = memory.Make(count);
  auto* q8 = memory.Make<std::uint8_t>(qfn_mmq_q8_1_bytes(tokens, k));
  Require(qfn_mmq_quantize_q8_1(x, q8, tokens, k, nullptr) == 0);
  const auto run = [&](bool direct, float* out) {
    const auto pointers =
        direct ? QfnExpertPointers{gate.table, up.table} : QfnExpertPointers{};
    const auto* g = direct ? gate.scattered : gate.resident;
    const auto* u = direct ? up.scattered : up.resident;
    if (prequantized)
      return qfn_mmq_moe_gated_vec_preq(type, g, u, q8, d_ids, out, m, k,
                                        tokens, experts, used, nullptr,
                                        pointers);
    if (gated)
      return qfn_mmq_moe_gated_vec(type, g, u, x, d_ids, out, m, k, tokens,
                                   experts, used, nullptr, pointers);
    return qfn_mmq_moe_vec(type, g, x, d_ids, out, m, k, tokens, experts, used,
                           nullptr, nullptr, nullptr, pointers);
  };
  Require(run(false, reference) == 0);
  Require(run(true, actual) == 0);
  Hip(hipDeviceSynchronize());
  Same(reference, actual, count);
  if (!gated) {
    auto* reference_up = memory.Make(count);
    auto* actual_up = memory.Make(count);
    Require(qfn_mmq_moe_vec(type, gate.resident, x, d_ids, reference, m, k,
                            tokens, experts, used, nullptr, up.resident,
                            reference_up) == 0);
    Require(qfn_mmq_moe_vec(type, gate.scattered, x, d_ids, actual, m, k,
                            tokens, experts, used, nullptr, up.scattered,
                            actual_up, {gate.table, up.table}) == 0);
    Same(reference, actual, count);
    Same(reference_up, actual_up, count);
  }
  // Replay with changed table contents to catch cached kernel arguments.
  Hip(hipMemcpy(const_cast<void**>(gate.table), up.table,
                experts * sizeof(void*), hipMemcpyDeviceToDevice));
  if (!gated) {
    Require(qfn_mmq_moe_vec(type, up.resident, x, d_ids, reference, m, k,
                            tokens, experts, used, nullptr) == 0);
    Require(run(true, actual) == 0);
    Same(reference, actual, count);
  }
  std::cout << "PASS type=" << type << " tokens=" << tokens << " rows=" << m
            << " gated=" << gated << " prequantized=" << prequantized << '\n';
}
void Benchmark() {
  constexpr int experts = 64, used = 10, hidden = 2560, ff = 640, repeats = 100;
  Memory memory;
  auto gate = MakeWeights(memory, 12, ff, hidden, experts);
  auto up = MakeWeights(memory, 12, ff, hidden, experts);
  auto down = MakeWeights(memory, 7, hidden, ff, experts);
  const std::array weights{gate, up, down};
  std::array<void*, 3> workspace{};
  for (int i = 0; i < 3; ++i)
    workspace[i] =
        memory.Make<std::uint8_t>(experts * weights[i].expert_bytes + 4096);
  auto* x = memory.Make(hidden);
  std::vector<float> input(hidden, 0.125f);
  Hip(hipMemcpy(x, input.data(), hidden * sizeof(float),
                hipMemcpyHostToDevice));
  auto* ids = memory.Make<std::int32_t>(used);
  auto* intermediate = memory.Make(used * ff);
  auto* output = memory.Make(used * hidden);
  auto* reference = memory.Make(used * hidden);
  hipStream_t stream = nullptr;
  Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  const auto run = [&](bool direct, int iteration) {
    std::array<std::int32_t, used> selection{};
    for (int i = 0; i < used; ++i)
      selection[i] = (iteration * 11 + i * 7) % experts;
    Hip(hipMemcpyAsync(ids, selection.data(), sizeof(selection),
                       hipMemcpyHostToDevice, stream));
    if (!direct) {
      for (int i = 0; i < 3; ++i)
        for (auto expert : selection) {
          const auto bytes = weights[i].expert_bytes;
          Hip(hipMemcpyAsync(
              static_cast<char*>(workspace[i]) + expert * bytes,
              static_cast<const char*>(weights[i].resident) + expert * bytes,
              bytes, hipMemcpyDeviceToDevice, stream));
        }
    }
    Require(qfn_mmq_moe_gated_vec(
                12, direct ? gate.scattered : workspace[0],
                direct ? up.scattered : workspace[1], x, ids, intermediate, ff,
                hidden, 1, experts, used, stream,
                direct ? QfnExpertPointers{gate.table, up.table}
                       : QfnExpertPointers{}) == 0);
    Require(qfn_mmq_moe_vec(7, direct ? down.scattered : workspace[2],
                            intermediate, ids, output, hidden, ff, used,
                            experts, 1, stream, nullptr, nullptr,
                            direct ? QfnExpertPointers{down.table, nullptr}
                                   : QfnExpertPointers{}) == 0);
    Hip(hipStreamSynchronize(stream));
  };
  run(false, 0);
  Hip(hipMemcpy(reference, output, used * hidden * sizeof(float),
                hipMemcpyDeviceToDevice));
  run(true, 0);
  Same(reference, output, used * hidden);
  std::array<std::vector<double>, 2> milliseconds;
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (bool direct : {false, true}) {
      for (int i = 0; i < 5; ++i)
        run(direct, i);
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeats; ++i)
        run(direct, i);
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        repeats;
      milliseconds[direct].push_back(ms);
      std::cout << "cycle=" << cycle << " direct=" << direct << " ms=" << ms
                << '\n';
    }
  }
  for (auto& v : milliseconds)
    std::sort(v.begin(), v.end());
  std::cout << "MoE-only median per layer, rotated 64-expert Q4/Q5_1 weights: "
            << milliseconds[0][1] << " ms with weight copies, "
            << milliseconds[1][1]
            << " ms direct, ratio=" << milliseconds[0][1] / milliseconds[1][1]
            << '\n';
  Hip(hipStreamDestroy(stream));
}
}  // namespace

int main(int argc, char**) {
  try {
    Hip(hipSetDevice(0));
    if (argc > 1) {
      Benchmark();
      return 0;
    }
    for (int type : {12, 13, 8})
      for (int tokens : {1, 2, 3, 8})
        for (bool prequantized : {false, true})
          Check(type, tokens, true, prequantized, tokens == 3 ? 127 : 640);
    for (int tokens : {9, 33, 64})
      Check(12, tokens, true, false, 640);
    for (int type : {7, 8, 12, 13})
      for (int tokens : {1, 10, 20, 21, 80})
        Check(type, tokens, false, false, tokens == 21 ? 2591 : 2560);
    std::cout << "PASS: 47 expert pointer cases, exact finite outputs\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
