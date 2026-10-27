#include "src/models/qwen38_flash_next/kernels/rocm/expert_trace.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}
// Graph capture must not count as execution; replay must preserve ordering.
void CheckExpertTrace(unsigned rows) {
  namespace q = gufo::models::qwen38_flash_next::rocm;
  const unsigned capacity = 2 * rows;
  std::vector<std::int32_t> ids(rows * 10);
  for (unsigned i = 0; i < ids.size(); ++i)
    ids[i] = (100 + i) % 512;
  std::int32_t* device_ids = nullptr;
  unsigned long long* trace = nullptr;
  const auto bytes = 16 + capacity * 14 * sizeof(std::uint32_t);
  CheckHip(hipMalloc(&device_ids, ids.size() * sizeof(ids[0])), "trace ids");
  CheckHip(hipMalloc(&trace, bytes), "trace allocation");
  CheckHip(hipMemcpy(device_ids, ids.data(), ids.size() * sizeof(ids[0]),
                     hipMemcpyHostToDevice),
           "trace upload");
  CheckHip(hipMemset(trace, 0, bytes), "trace clear");
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "trace stream");
  CheckHip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
           "trace capture");
  q::TraceExpertUsage(device_ids, trace, capacity, rows, 7, stream);
  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  CheckHip(hipStreamEndCapture(stream, &graph), "trace capture end");
  CheckHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
           "trace graph");
  unsigned long long header[2]{};
  CheckHip(hipMemcpy(header, trace, sizeof(header), hipMemcpyDeviceToHost),
           "trace capture header");
  if (header[0] || header[1])
    throw std::runtime_error("graph capture recorded expert accesses");
  for (unsigned replay = 0; replay < 2; ++replay)
    CheckHip(hipGraphLaunch(exec, stream), "trace replay");
  CheckHip(hipStreamSynchronize(stream), "trace synchronize");
  CheckHip(hipMemcpy(header, trace, sizeof(header), hipMemcpyDeviceToHost),
           "trace header");
  if (header[0] != capacity || header[1] != 2)
    throw std::runtime_error("expert trace replay count mismatch");
  std::vector<std::uint32_t> records(capacity * 14);
  CheckHip(hipMemcpy(records.data(), trace + 2, records.size() * 4,
                     hipMemcpyDeviceToHost),
           "trace records");
  for (unsigned row = 0; row < capacity; ++row) {
    const auto* record = records.data() + row * 14;
    if (record[0] != row / rows || record[1] != 0 || record[2] != 7 ||
        record[3] != row % rows)
      throw std::runtime_error("expert trace order mismatch");
    for (unsigned i = 0; i < 10; ++i)
      if (record[4 + i] != static_cast<unsigned>(ids[(row % rows) * 10 + i]))
        throw std::runtime_error("expert trace selection mismatch");
  }
  CheckHip(hipGraphLaunch(exec, stream), "trace overflow");
  CheckHip(hipStreamSynchronize(stream), "trace overflow synchronize");
  CheckHip(hipMemcpy(header, trace, sizeof(header), hipMemcpyDeviceToHost),
           "trace overflow header");
  std::vector<std::uint32_t> after(records.size());
  CheckHip(hipMemcpy(after.data(), trace + 2, after.size() * 4,
                     hipMemcpyDeviceToHost),
           "trace overflow records");
  if (header[0] != 3 * rows || header[1] != 3 || after != records)
    throw std::runtime_error("expert trace overflow was not bounded");
  CheckHip(hipMemset(trace, 0, sizeof(*trace)), "trace drain");
  CheckHip(hipGraphLaunch(exec, stream), "trace after drain");
  CheckHip(hipStreamSynchronize(stream), "trace drain synchronize");
  CheckHip(hipMemcpy(after.data(), trace + 2, after.size() * 4,
                     hipMemcpyDeviceToHost),
           "trace drained records");
  if (after[0] != 3)
    throw std::runtime_error("expert trace sequence reset on drain");
  CheckHip(hipGraphExecDestroy(exec), "trace exec destroy");
  CheckHip(hipGraphDestroy(graph), "trace graph destroy");
  CheckHip(hipStreamDestroy(stream), "trace stream destroy");
  CheckHip(hipFree(trace), "trace free");
  CheckHip(hipFree(device_ids), "trace ids free");
}

}  // namespace
int main() {
  try {
    CheckExpertTrace(3);
    CheckExpertTrace(513);
    std::cout
        << "Expert trace ordering, graph replay, drain and overflow: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
