#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace py = pybind11;

namespace {

uint64_t monotonic_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct alignas(64) Slot {
  // 0 = idle, 1 = armed, 2 = ready, 3 = CUDA error.
  std::atomic<int> state{0};
  std::atomic<uintptr_t> event_handle{0};
  std::atomic<uint64_t> armed_ns{0};
  std::atomic<uint64_t> last_not_ready_query_start_ns{0};
  std::atomic<uint64_t> ready_ns{0};
  std::atomic<uint64_t> visibility_upper_bound_ns{0};
  std::atomic<uint64_t> success_query_duration_ns{0};
  std::atomic<uint64_t> max_query_duration_ns{0};
  std::atomic<uint64_t> poll_count{0};
  std::atomic<int> cuda_error{0};
  std::atomic<int> upper_bound_valid{0};
};

class BusyEventBridge {
 public:
  BusyEventBridge(size_t capacity, int cpu_core, int cuda_device)
      : capacity_(capacity),
        cpu_core_(cpu_core),
        cuda_device_(cuda_device),
        slots_(new Slot[capacity]) {
    if (capacity == 0) {
      throw std::invalid_argument("capacity must be positive");
    }
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&BusyEventBridge::poll_loop, this);
  }

  ~BusyEventBridge() { stop(); }

  BusyEventBridge(const BusyEventBridge&) = delete;
  BusyEventBridge& operator=(const BusyEventBridge&) = delete;

  void arm(size_t index, uintptr_t event_handle) {
    Slot& slot = checked(index);
    const int previous = slot.state.load(std::memory_order_acquire);
    if (previous == 1) {
      throw std::runtime_error("slot is already armed");
    }
    if (event_handle == 0) {
      throw std::invalid_argument("event handle must be nonzero");
    }
    slot.event_handle.store(event_handle, std::memory_order_relaxed);
    slot.armed_ns.store(monotonic_ns(), std::memory_order_relaxed);
    slot.last_not_ready_query_start_ns.store(0, std::memory_order_relaxed);
    slot.ready_ns.store(0, std::memory_order_relaxed);
    slot.visibility_upper_bound_ns.store(0, std::memory_order_relaxed);
    slot.success_query_duration_ns.store(0, std::memory_order_relaxed);
    slot.max_query_duration_ns.store(0, std::memory_order_relaxed);
    slot.poll_count.store(0, std::memory_order_relaxed);
    slot.cuda_error.store(0, std::memory_order_relaxed);
    slot.upper_bound_valid.store(0, std::memory_order_relaxed);
    slot.state.store(1, std::memory_order_release);
  }

  bool ready(size_t index) const {
    return checked_const(index).state.load(std::memory_order_acquire) == 2;
  }

  void wait_all(uint64_t timeout_ns) {
    const uint64_t start = monotonic_ns();
    py::gil_scoped_release release;
    while (true) {
      bool complete = true;
      for (size_t index = 0; index < capacity_; ++index) {
        const int state = slots_[index].state.load(std::memory_order_acquire);
        if (state == 3) {
          throw std::runtime_error("cudaEventQuery failed");
        }
        if (state == 1) {
          complete = false;
        }
      }
      if (complete) {
        return;
      }
      if (monotonic_ns() - start > timeout_ns) {
        throw std::runtime_error("event bridge timeout");
      }
    }
  }

  std::vector<py::dict> snapshot() const {
    std::vector<py::dict> output;
    output.reserve(capacity_);
    for (size_t index = 0; index < capacity_; ++index) {
      const Slot& slot = slots_[index];
      py::dict item;
      item["slot"] = index;
      item["state"] = slot.state.load(std::memory_order_acquire);
      item["armed_ns"] = slot.armed_ns.load(std::memory_order_relaxed);
      item["last_not_ready_query_start_ns"] =
          slot.last_not_ready_query_start_ns.load(std::memory_order_relaxed);
      item["ready_ns"] = slot.ready_ns.load(std::memory_order_relaxed);
      item["visibility_upper_bound_ns"] =
          slot.visibility_upper_bound_ns.load(std::memory_order_relaxed);
      item["upper_bound_valid"] =
          static_cast<bool>(slot.upper_bound_valid.load(std::memory_order_relaxed));
      item["success_query_duration_ns"] =
          slot.success_query_duration_ns.load(std::memory_order_relaxed);
      item["max_query_duration_ns"] =
          slot.max_query_duration_ns.load(std::memory_order_relaxed);
      item["poll_count"] = slot.poll_count.load(std::memory_order_relaxed);
      item["cuda_error"] = slot.cuda_error.load(std::memory_order_relaxed);
      output.push_back(std::move(item));
    }
    return output;
  }

  void reset_all() {
    for (size_t index = 0; index < capacity_; ++index) {
      Slot& slot = slots_[index];
      if (slot.state.load(std::memory_order_acquire) == 1) {
        throw std::runtime_error("cannot reset an armed slot");
      }
      slot.state.store(0, std::memory_order_release);
    }
  }

  bool pinned() const { return pinned_.load(std::memory_order_acquire); }
  int cpu_core() const { return cpu_core_; }
  int cuda_device() const { return cuda_device_; }
  size_t capacity() const { return capacity_; }

  void stop() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
      if (worker_.joinable()) {
        worker_.join();
      }
    }
  }

 private:
  Slot& checked(size_t index) {
    if (index >= capacity_) {
      throw std::out_of_range("slot index outside bridge capacity");
    }
    return slots_[index];
  }

  const Slot& checked_const(size_t index) const {
    if (index >= capacity_) {
      throw std::out_of_range("slot index outside bridge capacity");
    }
    return slots_[index];
  }

  void pin_worker() {
#ifdef __linux__
    if (cpu_core_ >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu_core_, &set);
      const int result = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      pinned_.store(result == 0, std::memory_order_release);
      return;
    }
#endif
    pinned_.store(false, std::memory_order_release);
  }

  void poll_loop() {
    pin_worker();
    const cudaError_t set_device_status = cudaSetDevice(cuda_device_);
    if (set_device_status != cudaSuccess) {
      for (size_t index = 0; index < capacity_; ++index) {
        slots_[index].cuda_error.store(static_cast<int>(set_device_status),
                                       std::memory_order_relaxed);
        slots_[index].state.store(3, std::memory_order_release);
      }
      return;
    }
    while (running_.load(std::memory_order_acquire)) {
      for (size_t index = 0; index < capacity_; ++index) {
        Slot& slot = slots_[index];
        if (slot.state.load(std::memory_order_acquire) != 1) {
          continue;
        }
        const uintptr_t raw = slot.event_handle.load(std::memory_order_relaxed);
        const cudaEvent_t event = reinterpret_cast<cudaEvent_t>(raw);
        const uint64_t query_start = monotonic_ns();
        const cudaError_t status = cudaEventQuery(event);
        const uint64_t query_end = monotonic_ns();
        const uint64_t duration = query_end - query_start;
        slot.poll_count.fetch_add(1, std::memory_order_relaxed);
        uint64_t old_max = slot.max_query_duration_ns.load(std::memory_order_relaxed);
        while (duration > old_max && !slot.max_query_duration_ns.compare_exchange_weak(
                                         old_max, duration, std::memory_order_relaxed)) {
        }
        if (status == cudaErrorNotReady) {
          slot.last_not_ready_query_start_ns.store(query_start,
                                                   std::memory_order_relaxed);
          continue;
        }
        if (status == cudaSuccess) {
          const uint64_t previous =
              slot.last_not_ready_query_start_ns.load(std::memory_order_relaxed);
          slot.ready_ns.store(query_end, std::memory_order_relaxed);
          slot.success_query_duration_ns.store(duration, std::memory_order_relaxed);
          if (previous != 0 && query_end >= previous) {
            // A NotReady result proves the event was incomplete at or after
            // that query's start.  Success proves completion by this query's
            // return.  Their all-host interval is therefore a conservative
            // upper bound on completion -> host-ready visibility.
            slot.visibility_upper_bound_ns.store(query_end - previous,
                                                 std::memory_order_relaxed);
            slot.upper_bound_valid.store(1, std::memory_order_relaxed);
          }
          slot.state.store(2, std::memory_order_release);
          continue;
        }
        slot.cuda_error.store(static_cast<int>(status), std::memory_order_relaxed);
        slot.state.store(3, std::memory_order_release);
      }
    }
  }

  size_t capacity_;
  int cpu_core_;
  int cuda_device_;
  std::unique_ptr<Slot[]> slots_;
  std::atomic<bool> running_{false};
  std::atomic<bool> pinned_{false};
  std::thread worker_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<BusyEventBridge>(module, "BusyEventBridge")
      .def(py::init<size_t, int, int>(), py::arg("capacity"),
           py::arg("cpu_core") = -1, py::arg("cuda_device") = 0)
      .def("arm", &BusyEventBridge::arm)
      .def("ready", &BusyEventBridge::ready)
      .def("wait_all", &BusyEventBridge::wait_all)
      .def("snapshot", &BusyEventBridge::snapshot)
      .def("reset_all", &BusyEventBridge::reset_all)
      .def("stop", &BusyEventBridge::stop)
      .def_property_readonly("pinned", &BusyEventBridge::pinned)
      .def_property_readonly("cpu_core", &BusyEventBridge::cpu_core)
      .def_property_readonly("cuda_device", &BusyEventBridge::cuda_device)
      .def_property_readonly("capacity", &BusyEventBridge::capacity);
}
