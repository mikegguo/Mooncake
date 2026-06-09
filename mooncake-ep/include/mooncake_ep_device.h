#pragma once
// ============================================================================
// mooncake_ep_device.h — Unified CUDA/MUSA compatibility header
// ============================================================================
// Device-compatible types and macros only (no ATen / libtorch).
// ATen types (at::cuda::CUDAStream, torch::kCUDA, etc.) are used directly
// — torchada transparently maps them to MUSA equivalents at build time.
// ============================================================================

// NOTE: mooncake_ep_exception.cuh is NOT included here — it is always
// included by the translation unit before this header (via kernel .cu or
// mooncake_ep_buffer.h).  Including it here would cause EPException
// redefinition when torchada's simple_porting creates include_musa/ copies
// with .muh extensions.

#ifdef MOONCAKE_EP_USE_MUSA

// ---- MUSA platform --------------------------------------------------------
#include "cuda_alike.h"       // cuda* → musa* runtime API mapping
#include <musa_bf16.h>        // mt_bfloat16
#include <musa_runtime.h>     // musaStream_t, musaError_t, etc.

// -- bfloat16 -----------------------------------------------------------------
#ifdef __MUSA_ARCH__
using nv_bfloat16 = mt_bfloat16;
#elif defined(__MUSA__)
using nv_bfloat16 = mt_bfloat16;
#else
struct nv_bfloat16 { unsigned short __x; };
#endif

// -- FP8 stubs (MUSA has no FP8; templates compile but never instantiated) ---
using __nv_fp8_storage_t = uint8_t;
using __nv_fp8x2_storage_t = uint16_t;
#define __NV_SATFINITE 0
#define __NV_E4M3 0
#if defined(__CUDACC__) || defined(__MCC__)
__device__ __forceinline__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(
    float2, int, int) { return 0; }
#endif

// -- Device intrinsics -------------------------------------------------------
#ifndef __ldg
#define __ldg(ptr) (*(ptr))
#endif
#ifndef __activemask
#define __activemask() (0xffffffff)
#endif

#if defined(__CUDACC__) || defined(__MCC__)
__forceinline__ __device__ int get_lane_id() { return threadIdx.x % 32; }
#endif

// -- Kernel launch bounds (MUSA doesn't support __launch_bounds__) -----------
#define EP_LAUNCH_BOUNDS(max_threads, min_blocks)

// -- Launch config (MUSA: no cooperative launch) -----------------------------
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    dim3 _grid(num_sms);                                  \
    dim3 _block(num_threads);                             \
    cudaStream_t _stream = stream

#define LAUNCH_KERNEL(config, kernel, ...)                             \
    kernel<<<_grid, _block, 0, _stream>>>(__VA_ARGS__);               \
    {                                                                  \
        auto _err = musaGetLastError();                                \
        if (_err != musaSuccess) {                                     \
            fprintf(stderr, "[EP] kernel launch failed: %s\n",         \
                    musaGetErrorString(_err));                         \
        }                                                              \
    }

// -- Memory fence (MUSA needs explicit fences for peer visibility) -----------
#define EP_DEVICE_FENCE()  __threadfence_system()

#else  // !MOONCAKE_EP_USE_MUSA

// ---- CUDA platform ---------------------------------------------------------
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>

// -- Device intrinsics -------------------------------------------------------
#if defined(__CUDACC__) || defined(__MCC__)
__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}
#endif

// -- Kernel launch bounds ----------------------------------------------------
#define EP_LAUNCH_BOUNDS(max_threads, min_blocks) \
    __launch_bounds__(max_threads, min_blocks)

// -- Launch config (CUDA: cooperative launch) --------------------------------
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {                            \
        (num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1];                          \
    attr[0].id = cudaLaunchAttributeCooperative;          \
    attr[0].val.cooperative = 1;                          \
    cfg.attrs = attr;                                     \
    cfg.numAttrs = 1

#define LAUNCH_KERNEL(config, kernel, ...) \
    EP_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))

// -- Memory fence (no-op on CUDA) --------------------------------------------
#define EP_DEVICE_FENCE()  do {} while (0)

#endif  // MOONCAKE_EP_USE_MUSA

// ---- Shared macros (identical on both platforms) ---------------------------

// Unified error checking
#define EP_CHECK(cmd)                                              \
    do {                                                           \
        cudaError_t e = (cmd);                                     \
        if (e != cudaSuccess) {                                    \
            throw EPException("GPU", __FILE__, __LINE__,           \
                              cudaGetErrorString(e));              \
        }                                                          \
    } while (0)

// Unified device synchronization
#define EP_DEVICE_SYNCHRONIZE()  cudaDeviceSynchronize()
