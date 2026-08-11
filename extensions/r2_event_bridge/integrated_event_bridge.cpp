#include <pybind11/pybind11.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>

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
  std::atomic<uint64_t> ready_ns{0};
  std::atomic<int> cuda_error{0};
};

class IntegratedEventBridge {
 public:
  IntegratedEventBridge(size_t capacity, int cpu_core, int cuda_device)
      : capacity_(capacity),
        cpu_core_(cpu_core),
        cuda_device_(cuda_device),
        slots_(new Slot[capacity]) {
    if (capacity == 0) {
      throw std::invalid_argument("capacity must be positive");
    }
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&IntegratedEventBridge::poll_loop, this);
  }

  ~IntegratedEventBridge() { stop(); }

  IntegratedEventBridge(const IntegratedEventBridge&) = delete;
  IntegratedEventBridge& operator=(const IntegratedEventBridge&) = delete;

  void arm(size_t index, uintptr_t event_handle) {
    Slot& slot = checked(index);
    if (slot.state.load(std::memory_order_acquire) == 1) {
      throw std::runtime_error("slot is already armed");
    }
    if (event_handle == 0) {
      throw std::invalid_argument("event handle must be nonzero");
    }
    slot.event_handle.store(event_handle, std::memory_order_relaxed);
    slot.ready_ns.store(0, std::memory_order_relaxed);
    slot.cuda_error.store(0, std::memory_order_relaxed);
    slot.state.store(1, std::memory_order_release);
  }

  bool ready(size_t index) const {
    return checked_const(index).state.load(std::memory_order_acquire) == 2;
  }

  uint64_t ready_ns(size_t index) const {
    const Slot& slot = checked_const(index);
    if (slot.state.load(std::memory_order_acquire) != 2) {
      throw std::runtime_error("slot is not ready");
    }
    return slot.ready_ns.load(std::memory_order_relaxed);
  }

  uint64_t wait_ready(size_t index, uint64_t timeout_ns) const {
    const Slot& slot = checked_const(index);
    const uint64_t start = monotonic_ns();
    py::gil_scoped_release release;
    while (true) {
      const int state = slot.state.load(std::memory_order_acquire);
      if (state == 2) {
        return slot.ready_ns.load(std::memory_order_relaxed);
      }
      if (state == 3) {
        throw std::runtime_error("cudaEventQuery failed");
      }
      if (monotonic_ns() - start > timeout_ns) {
        throw std::runtime_error("event bridge timeout");
      }
    }
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
    if (running_.compare_exchange_strong(expected, false,
                                         std::memory_order_acq_rel)) {
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
      const int result =
          pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
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
        const cudaEvent_t event = reinterpret_cast<cudaEvent_t>(
            slot.event_handle.load(std::memory_order_relaxed));
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) {
          continue;
        }
        if (status == cudaSuccess) {
          slot.ready_ns.store(monotonic_ns(), std::memory_order_relaxed);
          slot.state.store(2, std::memory_order_release);
          continue;
        }
        slot.cuda_error.store(static_cast<int>(status),
                              std::memory_order_relaxed);
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
  py::class_<IntegratedEventBridge>(module, "IntegratedEventBridge")
      .def(py::init<size_t, int, int>(), py::arg("capacity"),
           py::arg("cpu_core") = -1, py::arg("cuda_device") = 0)
      .def("arm", &IntegratedEventBridge::arm)
      .def("ready", &IntegratedEventBridge::ready)
      .def("ready_ns", &IntegratedEventBridge::ready_ns)
      .def("wait_ready", &IntegratedEventBridge::wait_ready)
      .def("reset_all", &IntegratedEventBridge::reset_all)
      .def("stop", &IntegratedEventBridge::stop)
      .def_property_readonly("pinned", &IntegratedEventBridge::pinned)
      .def_property_readonly("cpu_core", &IntegratedEventBridge::cpu_core)
      .def_property_readonly("cuda_device", &IntegratedEventBridge::cuda_device)
      .def_property_readonly("capacity", &IntegratedEventBridge::capacity);
}
