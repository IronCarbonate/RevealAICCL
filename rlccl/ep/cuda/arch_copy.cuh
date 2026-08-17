#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace rlccl::ep::cuda_backend {

template <int Arch>
struct DispatchCopyTraits;

template <>
struct DispatchCopyTraits<700> {
  static constexpr int kVectorBytes = 16;
  static constexpr bool kSupportsCpAsync = false;
  static constexpr bool kSupportsTma = false;
};

template <>
struct DispatchCopyTraits<800> {
  static constexpr int kVectorBytes = 16;
  static constexpr bool kSupportsCpAsync = true;
  static constexpr bool kSupportsTma = false;
};

template <>
struct DispatchCopyTraits<900> {
  static constexpr int kVectorBytes = 16;
  static constexpr bool kSupportsCpAsync = true;
  static constexpr bool kSupportsTma = true;
};

__device__ __forceinline__ void copy_feature_vectorized(
    void* destination, const void* source, uint32_t bytes) {
  auto* dst = static_cast<uint8_t*>(destination);
  const auto* src = static_cast<const uint8_t*>(source);
  uint32_t offset = 0;
  if ((reinterpret_cast<uintptr_t>(dst) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(src) % 16) == 0) {
    for (; offset + 16 <= bytes; offset += 16)
      *reinterpret_cast<uint4*>(dst + offset) =
          *reinterpret_cast<const uint4*>(src + offset);
  }
  for (; offset < bytes; ++offset) dst[offset] = src[offset];
}

}  // namespace rlccl::ep::cuda_backend
