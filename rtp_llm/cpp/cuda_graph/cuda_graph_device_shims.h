#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/cuda_graph/graph_capture_lifecycle.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ROCM
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::HIP
#else
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::CUDA
#endif

namespace py = pybind11;

namespace rtp_llm {
#if USING_ROCM
namespace rocm {
void setHipGraphCaptureEnabled(bool enabled);
}  // namespace rocm
#endif
namespace cuda_graph {

enum class GraphMemcpyKind {
    D2D,
    D2H,
    H2D,
};

#if USING_CUDA
using GraphPoolHandle = c10::cuda::MempoolId_t;
#else
struct GraphPoolHandle {};
#endif

#if USING_ROCM
using GraphStream      = at::hip::HIPStream;
using GraphStreamGuard = at::hip::HIPStreamGuard;
#else
using GraphStream      = at::cuda::CUDAStream;
using GraphStreamGuard = at::cuda::CUDAStreamGuard;
#endif

inline GraphStream toGraphStream(const torch::Stream& stream) {
#if USING_ROCM
    return at::hip::HIPStream(stream);
#else
    return at::cuda::CUDAStream(stream);
#endif
}

inline void setDevice(int rank) {
#if USING_ROCM
    auto result = hipSetDevice(rank);
    RTP_LLM_CHECK_WITH_INFO(result == hipSuccess, "hipSetDevice(%d) failed: %s", rank, hipGetErrorString(result));
    at::hip::set_device(rank);
#else
    check_cuda_value(cudaSetDevice(rank));
    at::cuda::set_device(rank);
#endif
}

inline GraphStream graphGetStreamFromPool(bool is_high_priority) {
#if USING_ROCM
    return at::hip::getStreamFromPool(is_high_priority);
#else
    return at::cuda::getStreamFromPool(is_high_priority);
#endif
}

inline GraphStream graphGetCurrentStream() {
#if USING_ROCM
    return at::hip::getCurrentHIPStream(at::hip::current_device());
#else
    return at::cuda::getCurrentCUDAStream(at::cuda::current_device());
#endif
}

inline void graphSetCurrentStream(GraphStream stream) {
#if USING_ROCM
    at::hip::setCurrentHIPStream(stream);
#else
    at::cuda::setCurrentCUDAStream(stream);
#endif
}

inline torch::Event makeGraphEvent() {
    return torch::Event(GRAPH_DEVICE_TYPE);
}

// Tell the caching allocator that this tensor's block is live on `stream`.
//
// The allocator binds each block to its allocating stream and may recycle it into
// the next allocation there as soon as the owning tensor dies -- event ordering on
// the consumer is not enough. Any tensor whose raw data_ptr() crosses to a second
// stream must be recorded on it, which is what the graph replay-prep fused copies
// (data_ptr() captured into a POD param struct) and the async MTP workers do.
//
// torch::Tensor::record_stream() is unusable on ROCm: it dispatches on device type
// and aborts for at::hip streams, so go to the allocator directly.
inline void recordTensorUseOnStream(const torch::Tensor& tensor, const GraphStream& stream) {
    if (!tensor.defined() || !tensor.is_cuda() || tensor.numel() == 0) {
        return;
    }
#if USING_ROCM
    c10::hip::HIPCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
#else
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
#endif
}

inline void recordTensorUseOnCurrentStream(const torch::Tensor& tensor) {
    recordTensorUseOnStream(tensor, graphGetCurrentStream());
}

GraphLifecycleContext acquire_graph_owner(uintptr_t owner_id);
void                  begin_capture_planning(const GraphLifecycleContext& ctx);
void                  cancel_capture_planning(const GraphLifecycleContext& ctx);
void                  prepare_capture_arena(const GraphLifecycleContext& ctx);
void                  release_graph_owner(const GraphLifecycleContext& ctx);
void                  enter_graph_capture(const GraphLifecycleContext* ctx);
void                  exit_graph_capture(const GraphLifecycleContext* ctx);
void                  finish_capture_session(const GraphLifecycleContext& ctx);
void                  graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, void* stream);
void                  graphDeviceSynchronize();
void                  graphMemGetInfo(size_t* free_bytes, size_t* total_bytes);
size_t                graphReservedBytes();
size_t                graphAllocatedBytes();
GraphPoolHandle       graphPoolHandle();
void                  graphCaptureBegin(at::cuda::CUDAGraph& graph, GraphPoolHandle pool);

}  // namespace cuda_graph
}  // namespace rtp_llm
