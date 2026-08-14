#include "gpu_scheduler.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rlccl::scheduler::cuda_backend {
namespace {

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(status_));                     \
    }                                                                           \
  } while (0)

__device__ __forceinline__ uint64_t global_time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ uint64_t plan_checksum(const uint8_t* bytes, uint64_t size) {
  uint64_t value = 0xCBF29CE484222325ULL;
  for (uint64_t index = 0; index < size; ++index) {
    uint8_t byte = (index >= 24 && index < 32) ? 0 : bytes[index];
    value ^= byte;
    value *= 0x100000001B3ULL;
  }
  return value;
}

__device__ void record_error(
    DeviceErrorLog* log, SchedulerErrorCode code, uint64_t action_id,
    const RevealRecord& reveal) {
  uint32_t index = atomicAdd(&log->count, 1U);
  if (index < log->capacity) {
    log->errors[index] = DeviceSchedulerError{
        static_cast<uint32_t>(code), 0, action_id,
        reveal.descriptor_id, reveal.reveal_epoch};
  }
}

__device__ __forceinline__ const CompiledRouteTemplate* routes(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const CompiledRouteTemplate*>(
      blob + header->route_templates_offset);
}

__device__ __forceinline__ const int32_t* pair_routes(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const int32_t*>(
      blob + header->rank_pair_to_route_offset);
}

__device__ __forceinline__ const uint64_t* capacities(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const uint64_t*>(blob + header->capacity_table_offset);
}

__device__ __forceinline__ const uint32_t* legality(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const uint32_t*>(blob + header->legality_flags_offset);
}

__device__ SchedulerErrorCode dynamic_guard(
    const CommittedAction& action,
    const RevealRecord& reveal,
    uint32_t pair,
    uint64_t available,
    int32_t route_index,
    bool force_duplicate,
    const uint8_t* blob,
    const CompiledPlanBlobHeader* header,
    const DeviceIncrementalState* state) {
  if (action.src_rank < 0 || static_cast<uint32_t>(action.src_rank) >= header->world_size)
    return kInvalidSourceRank;
  if (action.dst_rank < 0 || static_cast<uint32_t>(action.dst_rank) >= header->world_size)
    return kInvalidDestinationRank;
  if (action.token_count == 0) return kZeroTokenAction;
  if (force_duplicate || state->descriptor_epoch[action.descriptor_id] != UINT32_MAX)
    return kDuplicateDescriptor;
  if (action.reveal_epoch != reveal.reveal_epoch) return kStaleReveal;
  if (available <= state->committed_count[pair]) return kUnrevealedDemand;
  if (state->committed_count[pair] + action.token_count > available)
    return kFutureDemand;
  if (route_index < 0 || static_cast<uint32_t>(route_index) >= header->route_count)
    return kInvalidRoute;
  const auto& route = routes(blob, header)[route_index];
  if (!legality(blob, header)[route_index] ||
      route.src_rank != action.src_rank || route.dst_rank != action.dst_rank ||
      route.route_id != static_cast<int32_t>(action.route_id))
    return kInvalidRoute;
  if (action.token_count > capacities(blob, header)[route_index] ||
      action.bytes > static_cast<uint64_t>(header->max_tokens_per_peer) * header->record_bytes)
    return kBytesOverflow;
  if (action.src_offset + 8 + action.bytes > header->region_bytes ||
      action.dst_offset + 8 + action.bytes > 2 * header->region_bytes)
    return kOffsetOverflow;
  if ((action.src_offset % header->descriptor_stride) + 8 + action.bytes >
      header->descriptor_stride)
    return kOffsetOverflow;
  return kNoError;
}

__device__ void reject_reveal(
    DeviceErrorLog* errors, DeviceSchedulerCounters* counters,
    SchedulerErrorCode code, const RevealRecord& reveal,
    uint64_t action_id, SchedulerTiming* timing) {
  record_error(errors, code, action_id, reveal);
  atomicAdd(&counters->rejected_reveals, 1U);
  timing->t4_guard_complete = global_time_ns();
}

__global__ void publish_reveal_kernel(
    const int64_t* input_records, uint32_t record_count,
    DeviceRevealQueue* queue, SchedulerTiming* timing,
    DeviceSchedulerCounters* counters, uint64_t delay_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (uint32_t input = 0; input < record_count; ++input) {
    if (input && delay_cycles) {
      uint64_t start = clock64();
      while (clock64() - start < delay_cycles) __nanosleep(64);
    }
    const int64_t* row = input_records + static_cast<uint64_t>(input) * 8;
    RevealRecord reveal{
        static_cast<uint32_t>(row[0]), static_cast<uint32_t>(row[1]),
        static_cast<uint32_t>(row[2]), static_cast<uint32_t>(row[3]),
        static_cast<uint32_t>(row[4]), static_cast<uint32_t>(row[5]),
        static_cast<uint32_t>(row[6]), static_cast<uint32_t>(row[7])};
    timing[input].t0_router_complete = global_time_ns();
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> head_ref(queue->head);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_ref(queue->tail);
    uint32_t tail;
    while (true) {
      tail = tail_ref.load(cuda::memory_order_relaxed);
      uint32_t head = head_ref.load(cuda::memory_order_acquire);
      if (tail - head < queue->capacity) break;
      __nanosleep(64);
    }
    queue->records[tail % queue->capacity] = reveal;
    tail_ref.store(tail + 1, cuda::memory_order_release);
    timing[input].t1_reveal_published = global_time_ns();
    atomicAdd(&counters->producer_published, 1U);
  }
}

__global__ void gpu_scheduler_progress_kernel(
    const uint8_t* blob,
    const int32_t* dst_ranks,
    uint32_t assignment_count,
    uint32_t source_rank,
    uint32_t expected_reveals,
    DeviceRevealQueue* reveal_queue,
    DeviceActionQueue* action_queue,
    DeviceIncrementalState* state,
    DeviceErrorLog* errors,
    SchedulerTiming* timing,
    DeviceSchedulerCounters* counters) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const auto* header = reinterpret_cast<const CompiledPlanBlobHeader*>(blob);
  if (header->magic != kCompiledPlanMagic ||
      header->version != kCompiledPlanVersion ||
      header->header_bytes != sizeof(CompiledPlanBlobHeader) ||
      plan_checksum(blob, header->total_bytes) != header->checksum) {
    RevealRecord invalid{};
    record_error(errors, kPlanChecksum, 0, invalid);
    counters->rejected_reveals = expected_reveals;
    counters->processed_reveals = expected_reveals;
    return;
  }
  counters->uploaded_plan_checksum = header->checksum;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_head(reveal_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_tail(reveal_queue->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> action_head(action_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> action_tail(action_queue->tail);
  uint64_t last_progress_ns = global_time_ns();
  while (counters->processed_reveals < expected_reveals) {
    uint32_t head = reveal_head.load(cuda::memory_order_relaxed);
    uint32_t tail = reveal_tail.load(cuda::memory_order_acquire);
    if (head == tail) {
      if (global_time_ns() - last_progress_ns > 5'000'000'000ULL) {
        RevealRecord timeout_reveal{};
        timeout_reveal.descriptor_id = UINT32_MAX;
        record_error(errors, kInternal, state->next_action_id, timeout_reveal);
        counters->rejected_reveals += expected_reveals - counters->processed_reveals;
        counters->processed_reveals = expected_reveals;
        return;
      }
      __nanosleep(64);
      continue;
    }
    RevealRecord reveal = reveal_queue->records[head % reveal_queue->capacity];
    reveal_head.store(head + 1, cuda::memory_order_release);
    uint32_t sequence = counters->processed_reveals++;
    last_progress_ns = global_time_ns();
    SchedulerTiming* one_timing = &timing[sequence];
    one_timing->t2_reveal_consumed = global_time_ns();
    one_timing->chunk_id = reveal.chunk_id;
    one_timing->descriptor_id = reveal.descriptor_id;

    if (reveal.chunk_id >= header->max_chunks) {
      reject_reveal(errors, counters, kChunkRange, reveal, state->next_action_id, one_timing); continue;
    }
    if (reveal.descriptor_id >= header->max_descriptors) {
      reject_reveal(errors, counters, kDescriptorRange, reveal, state->next_action_id, one_timing); continue;
    }
    if (state->has_reveal_epoch && reveal.reveal_epoch <= state->last_reveal_epoch) {
      reject_reveal(errors, counters, kStaleReveal, reveal, state->next_action_id, one_timing); continue;
    }
    if (state->descriptor_epoch[reveal.descriptor_id] != UINT32_MAX) {
      reject_reveal(errors, counters, kDuplicateDescriptor, reveal, state->next_action_id, one_timing); continue;
    }
    if (static_cast<uint64_t>(reveal.assignment_begin) + reveal.assignment_count > assignment_count) {
      reject_reveal(errors, counters, kAssignmentRange, reveal, state->next_action_id, one_timing); continue;
    }

    uint32_t counts[kMaxWorldSize]{};
    SchedulerErrorCode bind_error = kNoError;
    for (uint32_t offset = 0; offset < reveal.assignment_count; ++offset) {
      int32_t dst = dst_ranks[reveal.assignment_begin + offset];
      if (dst < 0 || static_cast<uint32_t>(dst) >= header->world_size) {
        bind_error = kInvalidDestinationRank; break;
      }
      ++counts[dst];
    }
    if (bind_error != kNoError) {
      reject_reveal(errors, counters, bind_error, reveal, state->next_action_id, one_timing); continue;
    }
    uint32_t candidate_count = 0;
    for (uint32_t dst = 0; dst < header->world_size; ++dst)
      candidate_count += counts[dst] != 0;
    uint32_t action_head_value = action_head.load(cuda::memory_order_acquire);
    uint32_t action_tail_value = action_tail.load(cuda::memory_order_relaxed);
    if (action_tail_value - action_head_value + candidate_count > action_queue->capacity) {
      reject_reveal(errors, counters, kActionQueueOverflow, reveal, state->next_action_id, one_timing); continue;
    }

    CommittedAction candidates_local[kMaxWorldSize];
    uint32_t pair_indices[kMaxWorldSize];
    uint32_t built = 0;
    SchedulerErrorCode guard_error = kNoError;
    for (uint32_t dst = 0; dst < header->world_size; ++dst) {
      uint32_t raw_count = counts[dst];
      if (!raw_count) continue;
      uint32_t pair = source_rank * header->world_size + dst;
      int32_t route_index = pair_routes(blob, header)[pair];
      const CompiledRouteTemplate* route =
          route_index >= 0 && static_cast<uint32_t>(route_index) < header->route_count
              ? &routes(blob, header)[route_index] : nullptr;
      uint32_t count = raw_count;
      uint64_t available = state->revealed_count[pair] + raw_count;
      uint32_t action_source = source_rank;
      uint32_t action_epoch = reveal.reveal_epoch;
      bool force_duplicate = (reveal.flags & kInjectDuplicateAction) != 0;
      if (reveal.flags & kInjectFuture) ++count;
      if (reveal.flags & kInjectZeroTokenAction) count = 0;
      if (reveal.flags & kInjectUnrevealed) available = 0;
      if (reveal.flags & kInjectStaleAction) action_epoch = action_epoch ? action_epoch - 1 : 0;
      if (reveal.flags & kInjectInvalidRank) action_source = header->world_size;
      uint32_t route_id = route ? static_cast<uint32_t>(route->route_id) : UINT32_MAX;
      if (reveal.flags & kInjectInvalidRoute) ++route_id;
      uint64_t src_offset = static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride +
                            (route ? route->send_region_base : 0);
      uint64_t dst_offset = static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride +
                            (route ? route->recv_region_base : 0);
      if (reveal.flags & kInjectOffsetOverflow) src_offset = 2 * header->region_bytes;
      uint64_t action_bytes = static_cast<uint64_t>(count) * header->record_bytes;
      if (reveal.flags & kInjectBytesOverflow)
        action_bytes = static_cast<uint64_t>(header->max_tokens_per_peer + 1) * header->record_bytes;
      CommittedAction action{
          state->next_action_id + built, reveal.descriptor_id, reveal.chunk_id,
          action_epoch, static_cast<int32_t>(action_source), static_cast<int32_t>(dst),
          src_offset, dst_offset, count, 0,
          action_bytes,
          route_id, route ? route->flags : 0};
      guard_error = dynamic_guard(
          action, reveal, pair, available, route_index, force_duplicate,
          blob, header, state);
      if (guard_error != kNoError) break;
      candidates_local[built] = action;
      pair_indices[built] = pair;
      ++built;
    }
    one_timing->t3_binder_complete = global_time_ns();
    one_timing->t4_guard_complete = global_time_ns();
    if (guard_error != kNoError) {
      reject_reveal(errors, counters, guard_error, reveal, state->next_action_id, one_timing); continue;
    }

    for (uint32_t dst = 0; dst < header->world_size; ++dst) {
      uint32_t pair = source_rank * header->world_size + dst;
      state->revealed_count[pair] += counts[dst];
    }
    for (uint32_t index = 0; index < built; ++index) {
      const auto& action = candidates_local[index];
      uint32_t pair = pair_indices[index];
      state->committed_count[pair] += action.token_count;
      state->next_send_offset[pair] += action.bytes;
      state->next_recv_offset[pair] += action.bytes;
      action_queue->actions[action_tail_value % action_queue->capacity] = action;
      action_tail.store(++action_tail_value, cuda::memory_order_release);
    }
    state->next_action_id += built;
    state->descriptor_epoch[reveal.descriptor_id] = reveal.reveal_epoch;
    state->last_reveal_epoch = reveal.reveal_epoch;
    state->has_reveal_epoch = 1;
    counters->committed_actions += built;
    one_timing->t5_action_published = global_time_ns();
  }
}

template <typename T>
T* device_alloc(size_t count = 1) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, sizeof(T) * count));
  return pointer;
}

template <typename T>
void device_zero(T* pointer, size_t count = 1) {
  CUDA_CHECK(cudaMemset(pointer, 0, sizeof(T) * count));
}

}  // namespace

std::vector<torch::Tensor> run_gpu_scheduler_cuda(
    torch::Tensor plan_blob,
    torch::Tensor reveal_records,
    torch::Tensor dst_ranks,
    int64_t source_rank,
    int64_t reveal_queue_capacity,
    int64_t action_queue_capacity,
    int64_t producer_delay_cycles) {
  TORCH_CHECK(plan_blob.device().is_cpu() && plan_blob.scalar_type() == torch::kUInt8,
              "plan_blob must be a CPU uint8 tensor");
  TORCH_CHECK(plan_blob.is_contiguous(), "plan_blob must be contiguous");
  TORCH_CHECK(reveal_records.is_cuda() && reveal_records.scalar_type() == torch::kInt64,
              "reveal_records must be a CUDA int64 tensor");
  TORCH_CHECK(reveal_records.dim() == 2 && reveal_records.size(1) == 8,
              "reveal_records must have shape [N, 8]");
  TORCH_CHECK(dst_ranks.is_cuda() && dst_ranks.scalar_type() == torch::kInt32,
              "dst_ranks must be a CUDA int32 tensor");
  TORCH_CHECK(dst_ranks.dim() == 1, "dst_ranks must be one dimensional");
  TORCH_CHECK(reveal_queue_capacity > 0 && action_queue_capacity > 0,
              "queue capacities must be positive");
  const auto* host_header = reinterpret_cast<const CompiledPlanBlobHeader*>(plan_blob.data_ptr());
  TORCH_CHECK(host_header->magic == kCompiledPlanMagic, "invalid compiled plan magic");
  TORCH_CHECK(source_rank >= 0 && source_rank < host_header->world_size,
              "source rank outside compiled plan");
  uint32_t record_count = static_cast<uint32_t>(reveal_records.size(0));
  uint32_t pair_count = host_header->world_size * host_header->world_size;

  uint8_t* d_blob = device_alloc<uint8_t>(plan_blob.numel());
  RevealRecord* d_reveal_records = device_alloc<RevealRecord>(reveal_queue_capacity);
  CommittedAction* d_actions = device_alloc<CommittedAction>(action_queue_capacity);
  DeviceSchedulerError* d_errors = device_alloc<DeviceSchedulerError>(std::max<uint32_t>(record_count, 1));
  SchedulerTiming* d_timing = device_alloc<SchedulerTiming>(std::max<uint32_t>(record_count, 1));
  uint64_t* d_revealed = device_alloc<uint64_t>(pair_count);
  uint64_t* d_committed = device_alloc<uint64_t>(pair_count);
  uint64_t* d_send = device_alloc<uint64_t>(pair_count);
  uint64_t* d_recv = device_alloc<uint64_t>(pair_count);
  uint32_t* d_descriptor_epoch = device_alloc<uint32_t>(host_header->max_descriptors);
  DeviceRevealQueue* d_reveal_queue = device_alloc<DeviceRevealQueue>();
  DeviceActionQueue* d_action_queue = device_alloc<DeviceActionQueue>();
  DeviceIncrementalState* d_state = device_alloc<DeviceIncrementalState>();
  DeviceErrorLog* d_error_log = device_alloc<DeviceErrorLog>();
  DeviceSchedulerCounters* d_counters = device_alloc<DeviceSchedulerCounters>();

  CUDA_CHECK(cudaMemcpy(d_blob, plan_blob.data_ptr(), plan_blob.numel(), cudaMemcpyHostToDevice));
  device_zero(d_reveal_records, reveal_queue_capacity);
  device_zero(d_actions, action_queue_capacity);
  device_zero(d_errors, std::max<uint32_t>(record_count, 1));
  device_zero(d_timing, std::max<uint32_t>(record_count, 1));
  device_zero(d_revealed, pair_count); device_zero(d_committed, pair_count);
  device_zero(d_send, pair_count); device_zero(d_recv, pair_count);
  CUDA_CHECK(cudaMemset(d_descriptor_epoch, 0xFF, sizeof(uint32_t) * host_header->max_descriptors));

  DeviceRevealQueue reveal_queue{d_reveal_records, static_cast<uint32_t>(reveal_queue_capacity), 0, 0};
  DeviceActionQueue action_queue{d_actions, static_cast<uint32_t>(action_queue_capacity), 0, 0};
  DeviceIncrementalState state{
      d_revealed, d_committed, d_send, d_recv, d_descriptor_epoch,
      0, 0, 0};
  DeviceErrorLog error_log{d_errors, std::max<uint32_t>(record_count, 1), 0};
  DeviceSchedulerCounters counters{};
  CUDA_CHECK(cudaMemcpy(d_reveal_queue, &reveal_queue, sizeof(reveal_queue), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_action_queue, &action_queue, sizeof(action_queue), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_state, &state, sizeof(state), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_error_log, &error_log, sizeof(error_log), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_counters, &counters, sizeof(counters), cudaMemcpyHostToDevice));

  cudaStream_t scheduler_stream = nullptr;
  cudaStream_t producer_stream = at::cuda::getCurrentCUDAStream();
  CUDA_CHECK(cudaStreamCreateWithFlags(&scheduler_stream, cudaStreamNonBlocking));
  publish_reveal_kernel<<<1, 32, 0, producer_stream>>>(
      reveal_records.data_ptr<int64_t>(), record_count, d_reveal_queue,
      d_timing, d_counters, static_cast<uint64_t>(producer_delay_cycles));
  CUDA_CHECK(cudaGetLastError());
  gpu_scheduler_progress_kernel<<<1, 32, 0, scheduler_stream>>>(
      d_blob, dst_ranks.data_ptr<int32_t>(), static_cast<uint32_t>(dst_ranks.numel()),
      static_cast<uint32_t>(source_rank), record_count, d_reveal_queue,
      d_action_queue, d_state, d_error_log, d_timing, d_counters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(producer_stream));
  CUDA_CHECK(cudaStreamSynchronize(scheduler_stream));

  CUDA_CHECK(cudaMemcpy(&action_queue, d_action_queue, sizeof(action_queue), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&error_log, d_error_log, sizeof(error_log), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&state, d_state, sizeof(state), cudaMemcpyDeviceToHost));
  std::vector<CommittedAction> actions(action_queue.tail);
  std::vector<DeviceSchedulerError> errors(std::min(error_log.count, error_log.capacity));
  std::vector<SchedulerTiming> timings(record_count);
  std::vector<uint64_t> state_values(pair_count * 2);
  if (!actions.empty()) CUDA_CHECK(cudaMemcpy(actions.data(), d_actions, actions.size() * sizeof(CommittedAction), cudaMemcpyDeviceToHost));
  if (!errors.empty()) CUDA_CHECK(cudaMemcpy(errors.data(), d_errors, errors.size() * sizeof(DeviceSchedulerError), cudaMemcpyDeviceToHost));
  if (!timings.empty()) CUDA_CHECK(cudaMemcpy(timings.data(), d_timing, timings.size() * sizeof(SchedulerTiming), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(state_values.data(), d_revealed, pair_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(state_values.data() + pair_count, d_committed, pair_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, gpu_scheduler_progress_kernel));
  auto action_tensor = torch::empty({static_cast<int64_t>(actions.size()), 12}, torch::kInt64);
  auto action_access = action_tensor.accessor<int64_t, 2>();
  for (size_t i = 0; i < actions.size(); ++i) {
    const auto& a = actions[i];
    int64_t row[12] = {
        static_cast<int64_t>(a.action_id), a.descriptor_id, a.chunk_id,
        a.reveal_epoch, a.src_rank, a.dst_rank,
        static_cast<int64_t>(a.src_offset), static_cast<int64_t>(a.dst_offset),
        a.token_count, static_cast<int64_t>(a.bytes), a.route_id, a.flags};
    for (int column = 0; column < 12; ++column) action_access[i][column] = row[column];
  }
  auto error_tensor = torch::empty({static_cast<int64_t>(errors.size()), 4}, torch::kInt64);
  auto error_access = error_tensor.accessor<int64_t, 2>();
  for (size_t i = 0; i < errors.size(); ++i) {
    error_access[i][0] = errors[i].error_code;
    error_access[i][1] = static_cast<int64_t>(errors[i].action_id);
    error_access[i][2] = errors[i].descriptor_id;
    error_access[i][3] = errors[i].reveal_epoch;
  }
  auto timing_tensor = torch::empty({static_cast<int64_t>(timings.size()), 8}, torch::kInt64);
  auto timing_access = timing_tensor.accessor<int64_t, 2>();
  for (size_t i = 0; i < timings.size(); ++i) {
    uint64_t row[8] = {
        timings[i].t0_router_complete, timings[i].t1_reveal_published,
        timings[i].t2_reveal_consumed, timings[i].t3_binder_complete,
        timings[i].t4_guard_complete, timings[i].t5_action_published,
        timings[i].chunk_id, timings[i].descriptor_id};
    for (int column = 0; column < 8; ++column)
      timing_access[i][column] = static_cast<int64_t>(row[column]);
  }
  auto counter_tensor = torch::empty({12}, torch::kInt64);
  auto counter_access = counter_tensor.accessor<int64_t, 1>();
  const int64_t counter_values[12] = {
      static_cast<int64_t>(counters.processed_reveals),
      static_cast<int64_t>(counters.committed_actions),
      static_cast<int64_t>(error_log.count),
      static_cast<int64_t>(counters.producer_published),
      static_cast<int64_t>(counters.uploaded_plan_checksum),
      static_cast<int64_t>(host_header->checksum),
      static_cast<int64_t>(1), static_cast<int64_t>(32),
      static_cast<int64_t>(attributes.numRegs),
      static_cast<int64_t>(attributes.sharedSizeBytes),
      static_cast<int64_t>(attributes.localSizeBytes),
      static_cast<int64_t>(attributes.maxThreadsPerBlock)};
  for (int index = 0; index < 12; ++index) counter_access[index] = counter_values[index];
  auto state_tensor = torch::empty({2, static_cast<int64_t>(pair_count)}, torch::kInt64);
  auto state_access = state_tensor.accessor<int64_t, 2>();
  for (uint32_t i = 0; i < pair_count; ++i) {
    state_access[0][i] = static_cast<int64_t>(state_values[i]);
    state_access[1][i] = static_cast<int64_t>(state_values[pair_count + i]);
  }

  CUDA_CHECK(cudaStreamDestroy(scheduler_stream));
  cudaFree(d_blob); cudaFree(d_reveal_records); cudaFree(d_actions);
  cudaFree(d_errors); cudaFree(d_timing); cudaFree(d_revealed);
  cudaFree(d_committed); cudaFree(d_send); cudaFree(d_recv);
  cudaFree(d_descriptor_epoch); cudaFree(d_reveal_queue);
  cudaFree(d_action_queue); cudaFree(d_state); cudaFree(d_error_log);
  cudaFree(d_counters);
  return {action_tensor, error_tensor, timing_tensor, counter_tensor, state_tensor};
}

}  // namespace rlccl::scheduler::cuda_backend
