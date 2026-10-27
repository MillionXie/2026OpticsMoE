// Host dispatch only. All original CUDAGraph::replay and CUDA event records stay.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py=pybind11;
double now_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void checked(cudaError_t code) {
  if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Plan {
  std::vector<std::shared_ptr<at::cuda::CUDAGraph>> graphs;
  std::vector<std::uintptr_t> events;
  std::size_t launches=0;
  Plan(std::vector<std::shared_ptr<at::cuda::CUDAGraph>> g,std::vector<std::uintptr_t> e)
      :graphs(std::move(g)),events(std::move(e)) {
    if(graphs.size()!=37 || events.size()!=74) throw std::runtime_error("Exactly 37 original graphs/74 events required");
    for(auto value:events) if(!value) throw std::runtime_error("Initialized original timing events required");
  }
  std::vector<std::array<double,2>> replay(std::uintptr_t expected_stream) {
    auto stream=c10::cuda::getCurrentCUDAStream().stream();
    if(reinterpret_cast<std::uintptr_t>(stream)!=expected_stream) throw std::runtime_error("Original current body stream differs");
    std::vector<std::array<double,2>> times;times.reserve(37);
    for(std::size_t i=0;i<37;++i) {
      double start=now_seconds();
      checked(cudaEventRecord(reinterpret_cast<cudaEvent_t>(events[2*i]),stream));
      graphs[i]->replay(); // preserve native allocator/RNG preparation, NOT raw graph-exec bypass
      checked(cudaEventRecord(reinterpret_cast<cudaEvent_t>(events[2*i+1]),stream));
      times.push_back({start,now_seconds()});
    }
    ++launches;
    return times;
  }
};
PYBIND11_MODULE(lgvq_native_dispatch_20261027,m) {
  m.def("now_seconds",&now_seconds);
  py::class_<Plan>(m,"Plan")
    .def(py::init<std::vector<std::shared_ptr<at::cuda::CUDAGraph>>,std::vector<std::uintptr_t>>())
    .def("replay",&Plan::replay,py::call_guard<py::gil_scoped_release>())
    .def_readonly("launches",&Plan::launches);
}
