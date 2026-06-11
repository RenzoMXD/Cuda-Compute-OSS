/*
 * cco_preload.c — CCO Tier-2 no-delegation backstop (LD_PRELOAD vendor-symbol trap).
 *
 * THE THREAT. CCO is a Triton-only competition: the compute must be in the miner's own @triton.jit
 * kernel, never delegated to a vendor library (cuBLAS / cuBLASLt / cuDNN). The in-Python guards
 * (static AST scan + the TorchFunction/TorchDispatch runtime trap) are necessary but NOT sufficient:
 * a kernel that shares the scorer's interpreter can POP the mode stack, delegate to a vendor GEMM,
 * push it back, and return — all invisible to Python (proven). No in-process guard can stop code
 * already inside the interpreter.
 *
 * THE BACKSTOP. This shared object is LD_PRELOAD'ed into the ISOLATED SCORING CHILD only. It
 * INTERPOSES the vendor COMPUTE entry points (GEMM / batched-GEMM / cuBLASLt matmul / cuDNN
 * convolution+backend-execute / RNN / batch-norm / pooling). PyTorch links these libraries into the
 * GLOBAL symbol scope (DT_NEEDED, and torch's own dlopen uses RTLD_GLOBAL — verified), so a preloaded
 * definition wins by symbol interposition for EVERY cross-library caller, including libtorch_cuda —
 * regardless of when/how libcublas is loaded. When the kernel (directly or via a delegated torch op
 * that slipped past the Python trap) reaches one of these symbols, we record the symbol to the file
 * at $CCO_DELEGATION_LOG and _exit(99). The parent (cco/isolate.py) maps exit-99 / that flag file to
 * a delegation verdict.
 *
 * WHY THIS IS ROBUST. The interposers take NO usable arguments and never call through — they flag and
 * _exit — so there is no ABI/signature matching (System V AMD64: the caller sets up/cleans args, the
 * callee may ignore them; we never return, so the return-type mismatch never matters) and nothing the
 * kernel can do to "handle" the call. The TRIGGER is the exit code (99), which the kernel cannot
 * forge (os/sys are banned, a real crash exits 1/134/139, never 99) and cannot suppress (the call is
 * already inside libcublas). The flag FILE carries which symbol; stderr would be forgeable (the kernel
 * could close fd 2 before delegating), so the file — whose path the kernel does not know — is primary.
 * The miner's kernel cannot re-resolve the real symbol (ctypes/cffi/dlopen/os/sys are all statically
 * banned). The PARENT is NOT preloaded (it legitimately computes the matmul ORACLE via cuBLAS).
 *
 * RESIDUAL (documented, Tier-3). A FUSED vendor kernel that is statically compiled into
 * libtorch_cuda.so and crosses NO cuBLAS/cuDNN symbol — flash / mem-efficient SDPA, row-wise/blockwise
 * fp8 _scaled_mm, _weight_int4pack_mm (all CUTLASS inside libtorch). The shim is blind to those; they
 * are guarded only by the (poppable) in-Python trap + the static AST ban. No current track needs them.
 *
 * NO false positives: a legit Triton kernel (incl. a tl.dot tensor-core GEMM) and torch's own CUDA
 * init/allocator path call NONE of the COMPUTE symbols below — only handle/descriptor management
 * (Create/Destroy/Set/Get/...), which we deliberately do NOT interpose. Empirically verified on
 * torch 2.8.0+cu128 / sm_120.
 *
 * Build (pure libc — no CUDA headers, we never call through):
 *     gcc -shared -fPIC -O2 -fvisibility=default -o cco_preload.so runtime/cco_preload.c
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

/* $CCO_DELEGATION_LOG read ONCE at load (constructor: single-threaded, getenv safe here) so cco_flag
 * — which can fire from inside torch's CUDA dispatch with locks held — uses only async-signal-safe
 * calls (open/write/_exit). */
static const char *g_log_path = 0;

__attribute__((constructor))
static void cco_init(void) {
    g_log_path = getenv("CCO_DELEGATION_LOG");
}

static void cco_flag(const char *sym) {
    /* sym is always a compile-time string literal (#sym from CCO_TRAP), so strlen is bounded. strlen
     * is a pure memory read (no locks / no malloc) -> safe to call from this async-signal context. */
    unsigned long n = strlen(sym);
    if (g_log_path) {
        int fd = open(g_log_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            (void)!write(fd, sym, n);
            (void)!write(fd, "\n", 1);
        }
    }
    (void)!write(2, "CCO_VENDOR_DELEGATION:", 22);   /* best-effort human log (forgeable; not relied on) */
    (void)!write(2, sym, n);
    (void)!write(2, "\n", 1);
    _exit(99);   /* unforgeable trigger; distinct from segv(139)/abort(134)/oom(137)/python-exc(1) */
}

/* Interpose purely by symbol NAME. extern "C" (C TU -> already unmangled) + default visibility so the
 * symbol enters the dynamic table and wins interposition. No-arg prototype is safe (we _exit). */
#define CCO_TRAP(sym) __attribute__((visibility("default"))) int sym(void) { cco_flag(#sym); return 0; }

/* ================= cuBLAS classic (libcublas) ================= */
/* fp16/bf16/tf32/int8 GEMM (the matmul track's main non-Lt path) + batched/grouped */
CCO_TRAP(cublasGemmEx)
CCO_TRAP(cublasGemmStridedBatchedEx)
CCO_TRAP(cublasGemmBatchedEx)
CCO_TRAP(cublasGemmGroupedBatchedEx)
/* fp32 GEMM. Interpose BOTH the _v2 name torch binds (cublas_v2.h macro) AND the legacy bare name. */
CCO_TRAP(cublasSgemm_v2)
CCO_TRAP(cublasSgemm)
CCO_TRAP(cublasSgemmStridedBatched)
CCO_TRAP(cublasSgemmBatched)
CCO_TRAP(cublasSgemmEx)
CCO_TRAP(cublasSgemm3m)
/* fp64 */
CCO_TRAP(cublasDgemm_v2)
CCO_TRAP(cublasDgemm)
CCO_TRAP(cublasDgemmStridedBatched)
CCO_TRAP(cublasDgemmBatched)
/* fp16 (pure-half accum) */
CCO_TRAP(cublasHgemm)
CCO_TRAP(cublasHgemmStridedBatched)
CCO_TRAP(cublasHgemmBatched)
/* complex */
CCO_TRAP(cublasCgemm_v2)
CCO_TRAP(cublasCgemm)
CCO_TRAP(cublasCgemm3m)
CCO_TRAP(cublasZgemm_v2)
CCO_TRAP(cublasZgemm)
CCO_TRAP(cublasZgemm3m)
CCO_TRAP(cublasCgemmStridedBatched)
CCO_TRAP(cublasZgemmStridedBatched)
/* GEMV (matmul -> matvec reshape escape hatch) */
CCO_TRAP(cublasSgemv_v2)
CCO_TRAP(cublasSgemv)
CCO_TRAP(cublasDgemv_v2)
CCO_TRAP(cublasDgemv)
CCO_TRAP(cublasCgemv_v2)
CCO_TRAP(cublasZgemv_v2)
CCO_TRAP(cublasSgemvStridedBatched)
CCO_TRAP(cublasSgemvBatched)

/* ================= cuBLASLt (libcublasLt) ================= */
/* The sole Lt compute entry: fp16/bf16 default path on Blackwell, fp8 _scaled_mm, int8 _int_mm,
 * gemm_and_bias, and strided/batched (via layout attrs) ALL funnel here. */
CCO_TRAP(cublasLtMatmul)

/* ================= cuDNN (libcudnn*) ================= */
/* cudnnBackendExecute: the cuDNN-9 graph executor — flash-SDPA (cuDNN backend) + graph conv/matmul. */
CCO_TRAP(cudnnBackendExecute)
CCO_TRAP(cudnnConvolutionForward)
CCO_TRAP(cudnnConvolutionBackwardData)
CCO_TRAP(cudnnConvolutionBackwardFilter)
CCO_TRAP(cudnnConvolutionBiasActivationForward)
CCO_TRAP(cudnnFusedOpsExecute)
CCO_TRAP(cudnnMultiHeadAttnForward)
CCO_TRAP(cudnnRNNForward)
CCO_TRAP(cudnnBatchNormalizationForwardInference)
CCO_TRAP(cudnnBatchNormalizationForwardTrainingEx)
CCO_TRAP(cudnnPoolingForward)

/* ================= other vendor compute (no current track; cheap future-proofing) ================= */
/* cuFFT */
CCO_TRAP(cufftExecC2C)
CCO_TRAP(cufftExecR2C)
CCO_TRAP(cufftExecC2R)
CCO_TRAP(cufftExecZ2Z)
CCO_TRAP(cufftExecD2Z)
CCO_TRAP(cufftExecZ2D)
CCO_TRAP(cufftXtExec)
/* cuSPARSE (sparse GEMM analogs) */
CCO_TRAP(cusparseSpMM)
CCO_TRAP(cusparseSDDMM)
CCO_TRAP(cusparseSpGEMM_compute)
CCO_TRAP(cusparseSpMV)
/* cuSOLVER (decompositions) */
CCO_TRAP(cusolverDnSgetrf)
CCO_TRAP(cusolverDnSpotrf)
CCO_TRAP(cusolverDnSgesvd)
