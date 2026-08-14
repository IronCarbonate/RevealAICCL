#include <torch/extension.h>

#include "gpu_scheduler.cuh"

namespace py = pybind11;
using rlccl::scheduler::cuda_backend::run_gpu_scheduler_cuda;

py::dict run_gpu_scheduler(
    torch::Tensor plan_blob,
    torch::Tensor reveal_records,
    torch::Tensor dst_ranks,
    int64_t source_rank,
    int64_t reveal_queue_capacity,
    int64_t action_queue_capacity,
    int64_t producer_delay_cycles) {
  auto output = run_gpu_scheduler_cuda(
      plan_blob, reveal_records, dst_ranks, source_rank,
      reveal_queue_capacity, action_queue_capacity, producer_delay_cycles);
  py::dict result;
  result["actions"] = output.at(0);
  result["errors"] = output.at(1);
  result["timings"] = output.at(2);
  result["counters"] = output.at(3);
  result["state"] = output.at(4);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "run_gpu_scheduler", &run_gpu_scheduler,
      py::arg("plan_blob"), py::arg("reveal_records"),
      py::arg("dst_ranks"), py::arg("source_rank"),
      py::arg("reveal_queue_capacity"), py::arg("action_queue_capacity"),
      py::arg("producer_delay_cycles") = 0,
      "Run one persistent GPU scheduler and a device-side reveal producer");
}
