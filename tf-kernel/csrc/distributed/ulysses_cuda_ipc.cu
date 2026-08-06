#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>

namespace tf_kernel {
namespace {

constexpr size_t kIpcHandleBytes = sizeof(cudaIpcMemHandle_t);
constexpr size_t kSerializedIpcHandleBytes = kIpcHandleBytes + sizeof(uint64_t);

void check_driver(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* detail = nullptr;
  cuGetErrorString(result, &detail);
  TORCH_CHECK(false, operation, " failed: ", detail == nullptr ? "unknown CUDA driver error" : detail);
}

void check_ipc_handle(const torch::Tensor& handle) {
  TORCH_CHECK(handle.device().is_cpu(), "CUDA IPC handle must be on CPU");
  TORCH_CHECK(handle.scalar_type() == torch::kUInt8, "CUDA IPC handle must have dtype uint8");
  TORCH_CHECK(handle.is_contiguous(), "CUDA IPC handle must be contiguous");
  TORCH_CHECK(handle.numel() == kSerializedIpcHandleBytes, "invalid CUDA IPC handle size");
}

}  // namespace

torch::Tensor cuda_ipc_get_mem_handle(torch::Tensor tensor) {
  TORCH_CHECK(tensor.is_cuda(), "CUDA IPC can only export a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(tensor.device());

  CUdeviceptr allocation_base = 0;
  size_t allocation_size = 0;
  const auto tensor_pointer = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
  check_driver(cuMemGetAddressRange(&allocation_base, &allocation_size, tensor_pointer), "cuMemGetAddressRange");
  const auto offset = static_cast<uint64_t>(tensor_pointer - allocation_base);
  TORCH_CHECK(offset < allocation_size, "CUDA tensor pointer is outside its allocation");

  cudaIpcMemHandle_t handle{};
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(allocation_base)));
  auto result = torch::empty(
      {static_cast<int64_t>(kSerializedIpcHandleBytes)},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  std::memcpy(result.data_ptr(), &handle, kIpcHandleBytes);
  std::memcpy(static_cast<char*>(result.data_ptr()) + kIpcHandleBytes, &offset, sizeof(offset));
  return result;
}

int64_t cuda_ipc_open_mem_handle(torch::Tensor handle) {
  check_ipc_handle(handle);
  cudaIpcMemHandle_t ipc_handle{};
  uint64_t offset = 0;
  std::memcpy(&ipc_handle, handle.data_ptr(), kIpcHandleBytes);
  std::memcpy(&offset, static_cast<char*>(handle.data_ptr()) + kIpcHandleBytes, sizeof(offset));

  void* allocation_base = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&allocation_base, ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  const auto pointer = reinterpret_cast<uintptr_t>(allocation_base) + offset;
  return static_cast<int64_t>(pointer);
}

void cuda_ipc_close_mem_handle(int64_t pointer) {
  TORCH_CHECK(pointer != 0, "cannot close a null CUDA IPC pointer");
  CUdeviceptr allocation_base = 0;
  size_t allocation_size = 0;
  check_driver(
      cuMemGetAddressRange(
          &allocation_base, &allocation_size, static_cast<CUdeviceptr>(static_cast<uint64_t>(pointer))),
      "cuMemGetAddressRange");
  C10_CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(allocation_base)));
}

void ulysses_all_to_all_ce(
    torch::Tensor input,
    int64_t peer_output_pointer,
    int64_t rank,
    int64_t world_size,
    int64_t mode,
    int64_t peer) {
  TORCH_CHECK(input.is_cuda(), "Ulysses input must be CUDA");
  TORCH_CHECK(input.dim() == 4, "Ulysses input must be 4D, got ", input.dim(), "D");
  TORCH_CHECK(input.stride(3) == 1, "Ulysses head channels must be contiguous");
  TORCH_CHECK(input.stride(2) == input.size(3), "Ulysses heads must be contiguous within each sequence row");
  TORCH_CHECK(
      input.stride(1) >= input.size(2) * input.size(3),
      "Ulysses sequence rows must not overlap");
  TORCH_CHECK(
      input.size(0) <= 1 || input.stride(0) >= input.size(1) * input.stride(1),
      "Ulysses batches must not overlap");
  TORCH_CHECK(world_size > 1, "unsupported Ulysses world size: ", world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid Ulysses rank: ", rank);
  TORCH_CHECK(peer >= 0 && peer < world_size, "invalid Ulysses peer: ", peer);
  TORCH_CHECK(mode == 0 || mode == 1, "Ulysses mode must be 0 or 1, got ", mode);
  auto* target = reinterpret_cast<char*>(static_cast<uintptr_t>(peer_output_pointer));
  TORCH_CHECK(target != nullptr, "null Ulysses output pointer for peer ", peer);

  c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const size_t element_size = input.element_size();
  const int64_t batch = input.size(0);
  const int64_t sequence = input.size(1);
  const int64_t heads = input.size(2);
  const int64_t head_dim = input.size(3);
  auto* source = static_cast<char*>(input.data_ptr());

  if (mode == 0) {
    TORCH_CHECK(heads % world_size == 0, "head count must be divisible by the Ulysses world size");
    const int64_t local_heads = heads / world_size;
    const size_t width = static_cast<size_t>(local_heads * head_dim) * element_size;
    const size_t source_pitch = static_cast<size_t>(input.stride(1)) * element_size;
    const size_t target_pitch = width;
    const int64_t global_sequence = sequence * world_size;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      const int64_t source_elements =
          batch_index * input.stride(0) + peer * local_heads * head_dim;
      const int64_t target_elements =
          (batch_index * global_sequence + rank * sequence) * local_heads * head_dim;
      C10_CUDA_CHECK(cudaMemcpy2DAsync(
          target + target_elements * element_size,
          target_pitch,
          source + source_elements * element_size,
          source_pitch,
          width,
          sequence,
          cudaMemcpyDefault,
          stream));
    }
  } else {
    TORCH_CHECK(sequence % world_size == 0, "sequence length must be divisible by the Ulysses world size");
    const int64_t local_sequence = sequence / world_size;
    const int64_t global_heads = heads * world_size;
    const size_t width = static_cast<size_t>(heads * head_dim) * element_size;
    const size_t source_pitch = static_cast<size_t>(input.stride(1)) * element_size;
    const size_t target_pitch = static_cast<size_t>(global_heads * head_dim) * element_size;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      const int64_t source_elements =
          batch_index * input.stride(0) + peer * local_sequence * input.stride(1);
      const int64_t target_elements =
          batch_index * local_sequence * global_heads * head_dim + rank * heads * head_dim;
      C10_CUDA_CHECK(cudaMemcpy2DAsync(
          target + target_elements * element_size,
          target_pitch,
          source + source_elements * element_size,
          source_pitch,
          width,
          local_sequence,
          cudaMemcpyDefault,
          stream));
    }
  }
}


void ulysses_stream_barrier(
    std::vector<int64_t> peer_barrier_pointers,
    torch::Tensor local_barrier,
    int64_t rank,
    int64_t world_size,
    int64_t epoch) {
  TORCH_CHECK(local_barrier.is_cuda(), "Ulysses barrier must be CUDA");
  TORCH_CHECK(local_barrier.scalar_type() == torch::kInt64, "Ulysses barrier must have dtype int64");
  TORCH_CHECK(local_barrier.is_contiguous(), "Ulysses barrier must be contiguous");
  TORCH_CHECK(local_barrier.numel() == world_size, "Ulysses barrier has the wrong size");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid Ulysses rank: ", rank);
  TORCH_CHECK(epoch > 0, "Ulysses barrier epoch must be positive");
  TORCH_CHECK(
      peer_barrier_pointers.size() == static_cast<size_t>(world_size),
      "one Ulysses barrier pointer is required per peer");

  c10::cuda::CUDAGuard device_guard(local_barrier.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(local_barrier.get_device());
  const auto driver_stream = reinterpret_cast<CUstream>(stream);
  for (int64_t peer = 0; peer < world_size; ++peer) {
    const auto peer_base = static_cast<CUdeviceptr>(static_cast<uint64_t>(peer_barrier_pointers[peer]));
    check_driver(
        cuStreamWriteValue64(
            driver_stream,
            peer_base + rank * sizeof(uint64_t),
            static_cast<uint64_t>(epoch),
            CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue64");
  }
  const auto local_base = reinterpret_cast<CUdeviceptr>(local_barrier.data_ptr<int64_t>());
  for (int64_t peer = 0; peer < world_size; ++peer) {
    check_driver(
        cuStreamWaitValue64(
            driver_stream,
            local_base + peer * sizeof(uint64_t),
            static_cast<uint64_t>(epoch),
            CU_STREAM_WAIT_VALUE_EQ),
        "cuStreamWaitValue64");
  }
}
}  // namespace tf_kernel

TORCH_LIBRARY_FRAGMENT(tf_kernel, m) {
  m.def("cuda_ipc_get_mem_handle(Tensor tensor) -> Tensor");
  m.impl("cuda_ipc_get_mem_handle", &tf_kernel::cuda_ipc_get_mem_handle);
  m.def("cuda_ipc_open_mem_handle(Tensor handle) -> int");
  m.impl("cuda_ipc_open_mem_handle", &tf_kernel::cuda_ipc_open_mem_handle);
  m.def("cuda_ipc_close_mem_handle(int pointer) -> ()");
  m.impl("cuda_ipc_close_mem_handle", &tf_kernel::cuda_ipc_close_mem_handle);
  m.def(
      "ulysses_all_to_all_ce(Tensor input, int peer_output_pointer, int rank, int world_size, int mode, int peer) -> ()");
  m.impl("ulysses_all_to_all_ce", torch::kCUDA, &tf_kernel::ulysses_all_to_all_ce);
  m.def("ulysses_stream_barrier(int[] peer_barrier_pointers, Tensor local_barrier, int rank, int world_size, int epoch) -> ()");
  m.impl("ulysses_stream_barrier", torch::kCUDA, &tf_kernel::ulysses_stream_barrier);
}
