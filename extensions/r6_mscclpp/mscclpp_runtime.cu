// R6-M1 thin runtime: long-lived CudaIpc MemoryChannel over caller-owned memory.
// No scheduling, packing, collective, XML, ExecutionPlan, or NCCL is used here.

#include <cuda_runtime.h>

#include <cstdint>
#include <chrono>
#include <exception>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

struct Runtime {
  struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
  };
  struct HostIssueTrace {
    std::uint64_t wrapper_enter_ns = 0;
    std::uint64_t launch_call_ns = 0;
    std::uint64_t launch_return_ns = 0;
  };
  int rank;
  int remote_rank;
  int device;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::unique_ptr<mscclpp::Communicator> communicator;
  mscclpp::RegisteredMemory local_memory;
  mscclpp::RegisteredMemory remote_memory;
  mscclpp::Connection connection;
  mscclpp::Semaphore semaphore;
  std::unique_ptr<mscclpp::MemoryChannel> channel;
  mscclpp::MemoryChannelDeviceHandle handle;
  std::uint64_t put_calls = 0;
  std::uint64_t bytes_transferred = 0;
  std::uint64_t signals = 0;
  std::uint64_t waits = 0;
  cudaEvent_t origin_event = nullptr;
  std::uint64_t origin_host_ns = 0;
  std::vector<EventPair> put_events;
  std::vector<EventPair> wait_events;
  std::vector<HostIssueTrace> put_host_traces;

  Runtime(int rank_, int device_, void* ptr, std::size_t bytes, const char* endpoint)
      : rank(rank_), remote_rank(rank_ ^ 1), device(device_) {
    cudaError_t cuda_status = cudaSetDevice(device);
    if (cuda_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_status));
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, 2);
    bootstrap->initialize(endpoint);
    communicator = std::make_unique<mscclpp::Communicator>(bootstrap);
    constexpr auto transport = mscclpp::Transport::CudaIpc;
    connection = communicator->connect(
        {transport, {mscclpp::DeviceType::GPU, device}}, remote_rank).get();
    semaphore = communicator->buildSemaphore(connection, remote_rank).get();
    local_memory = communicator->registerMemory(ptr, bytes, transport);
    communicator->sendMemory(local_memory, remote_rank);
    remote_memory = communicator->recvMemory(remote_rank).get();
    channel = std::make_unique<mscclpp::MemoryChannel>(
        semaphore, remote_memory, local_memory);
    handle = channel->deviceHandle();
    auto status = cudaEventCreate(&origin_event);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventRecord(origin_event);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventSynchronize(origin_event);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    origin_host_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  ~Runtime() {
    for (auto value : put_events) {
      if (value.start) cudaEventDestroy(value.start);
      if (value.end) cudaEventDestroy(value.end);
    }
    for (auto value : wait_events) {
      if (value.start) cudaEventDestroy(value.start);
      if (value.end) cudaEventDestroy(value.end);
    }
    if (origin_event) cudaEventDestroy(origin_event);
  }
};

std::uint64_t monotonic_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

__global__ void put_and_signal_kernel(mscclpp::MemoryChannelDeviceHandle handle,
                                      std::uint64_t dst_offset,
                                      std::uint64_t src_offset,
                                      std::uint64_t bytes) {
  const std::uint32_t tid = threadIdx.x;
  // R6 slots contain an int64 count header, so source/destination offsets are
  // guaranteed 8-byte aligned but may differ modulo 16.  MSCCL++ documents
  // identical modulo-Alignment phases; select the valid 8-byte primitive.
  if (bytes != 0) handle.put<8>(dst_offset, src_offset, bytes, tid, blockDim.x);
  __syncthreads();
  if (tid == 0) handle.signal();
}

__global__ void wait_kernel(mscclpp::MemoryChannelDeviceHandle handle) {
  if (threadIdx.x == 0) handle.wait(/*maxSpinCount=*/-1);
}

template <typename Fn>
int protect(Fn&& fn) {
  try {
    fn();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    last_error = "unknown native exception";
    return -1;
  }
}

}  // namespace

extern "C" {

const char* r6_mscclpp_last_error() { return last_error.c_str(); }

void* r6_mscclpp_create(int rank, int device, void* ptr, std::size_t bytes,
                        const char* endpoint) {
  Runtime* runtime = nullptr;
  if (protect([&] { runtime = new Runtime(rank, device, ptr, bytes, endpoint); }) != 0) return nullptr;
  return runtime;
}

int r6_mscclpp_issue(void* opaque, std::uint64_t dst_offset,
                     std::uint64_t src_offset, std::uint64_t bytes,
                     std::uintptr_t stream_value) {
  const auto wrapper_enter_ns = monotonic_ns();
  return protect([&] {
    auto* runtime = static_cast<Runtime*>(opaque);
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    Runtime::EventPair events;
    auto status = cudaEventCreate(&events.start);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventCreate(&events.end);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventRecord(events.start, stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    const auto launch_call_ns = monotonic_ns();
    put_and_signal_kernel<<<1, 256, 0, stream>>>(runtime->handle, dst_offset, src_offset, bytes);
    const auto launch_return_ns = monotonic_ns();
    status = cudaEventRecord(events.end, stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    runtime->put_calls += (bytes != 0);
    runtime->bytes_transferred += bytes;
    runtime->signals += 1;
    runtime->put_events.push_back(events);
    runtime->put_host_traces.push_back(
        {wrapper_enter_ns, launch_call_ns, launch_return_ns});
  });
}

int r6_mscclpp_wait(void* opaque, std::uintptr_t stream_value) {
  return protect([&] {
    auto* runtime = static_cast<Runtime*>(opaque);
    auto stream = reinterpret_cast<cudaStream_t>(stream_value);
    Runtime::EventPair events;
    auto status = cudaEventCreate(&events.start);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventCreate(&events.end);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventRecord(events.start, stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    wait_kernel<<<1, 1, 0, stream>>>(runtime->handle);
    status = cudaEventRecord(events.end, stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    runtime->waits += 1;
    runtime->wait_events.push_back(events);
  });
}

int r6_mscclpp_synchronize(void* opaque, std::uintptr_t stream_value) {
  return protect([&] {
    auto* runtime = static_cast<Runtime*>(opaque);
    auto status = cudaSetDevice(runtime->device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_value));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  });
}

std::uint64_t r6_mscclpp_counter(void* opaque, int counter) {
  auto* runtime = static_cast<Runtime*>(opaque);
  if (counter == 0) return runtime->put_calls;
  if (counter == 1) return runtime->bytes_transferred;
  if (counter == 2) return runtime->signals;
  if (counter == 3) return runtime->waits;
  return 0;
}

std::uint64_t r6_mscclpp_origin_host_ns(void* opaque) {
  return static_cast<Runtime*>(opaque)->origin_host_ns;
}

int r6_mscclpp_event_timing(void* opaque, int kind, std::size_t index,
                            float* start_us, float* end_us) {
  return protect([&] {
    auto* runtime = static_cast<Runtime*>(opaque);
    const auto& values = kind == 0 ? runtime->put_events : runtime->wait_events;
    if (index >= values.size()) throw std::out_of_range("event index outside runtime trace");
    auto status = cudaEventElapsedTime(start_us, runtime->origin_event, values[index].start);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaEventElapsedTime(end_us, runtime->origin_event, values[index].end);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    *start_us *= 1000.0f;
    *end_us *= 1000.0f;
  });
}

int r6_mscclpp_issue_host_timing(void* opaque, std::size_t index,
                                 std::uint64_t* wrapper_enter_ns,
                                 std::uint64_t* launch_call_ns,
                                 std::uint64_t* launch_return_ns) {
  return protect([&] {
    auto* runtime = static_cast<Runtime*>(opaque);
    if (index >= runtime->put_host_traces.size()) {
      throw std::out_of_range("host issue trace index outside runtime trace");
    }
    const auto& value = runtime->put_host_traces[index];
    *wrapper_enter_ns = value.wrapper_enter_ns;
    *launch_call_ns = value.launch_call_ns;
    *launch_return_ns = value.launch_return_ns;
  });
}

void r6_mscclpp_destroy(void* opaque) { delete static_cast<Runtime*>(opaque); }

}  // extern "C"
