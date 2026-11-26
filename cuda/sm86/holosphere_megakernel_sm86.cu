// Bare-Metal NVIDIA Ampere (sm_86) switch-Free Uniform CellOp Megakernel vs.
// switch(opcode) Persistent Cooperative Megakernel VM & Siliquant A_16 Block-H4
// INT8 Tensor Core (mma.sync.m16n8k32 / dp4a) Prototype.
//
// Track S116 (PREREG-S116 + PREREG-S116-AMENDMENT1) / silikernels#2 Piece 1.
//
// Honest sm_86 Hardware Scope:
//   - Single consumer GA104M (RTX 3080 Laptop GPU, 48 SMs, 4 MiB L2, 4.6875 MiB SMEM,
//     12 MiB RF = 20.6875 MiB on-chip SRAM).
//   - Replaces datacenter GPUDirect RDMA / NVSHMEM with zero-copy host-pinned
//     cudaHostAllocMapped UTF-8 byte ring buffer + PTX ld.acquire.sys / st.release.sys.
//   - Uses native Ampere cp.async.ca.shared.global (LDGSTS double-buffered),
//     zero-swizzle 4-thread-group Block-H4 -> mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//     with 2-shuffle distributed [16,11,4] syndrome check, fused warp_absmax S^31->s8
//     quantization (cvt.rni.sat.s8.f32), branchless flags&1 1-in-8 KV-ring lookup (selp.b32),
//     mma.sync.aligned.m16n8k16 (HMMA.16816), dp4a.s32.s32 (IDP), and 48-CTA
//     cudaLaunchCooperativeKernel.
//   - Zero unsigned GSP microcode or raw SASS injection; 0 B stack / 0 B spills
//     verified on every kernel via nvcc --ptxas-options=-v & cuobjdump --dump-sass.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace holosphere_s116 {

namespace cg = cooperative_groups;

constexpr int S116_VM_BLOCK_THREADS = 128;    // 4 warps/CTA for switch(opcode) control
constexpr int S116_CELL_BLOCK_THREADS = 256;  // 8 warps/CTA for switch-free uniform CellOp
constexpr int S116_D_MODEL = 256;             // 64 float4 quads

// Multi-opcode control VM flags
constexpr uint32_t S116_FLAG_GRID_BARRIER = 1u << 16;
constexpr uint32_t S116_DEQUANT_MODE_INT4 = 1u << 0;
constexpr uint32_t S116_DEQUANT_FAIL_CLOSED_ZERO = 1u << 1;

// switch-free CellOp flags (bit 0 = branchless 1-in-8 covariant KV-ring lookup!)
constexpr uint16_t S116_CELL_IS_KV_RING = 1u << 0;
constexpr uint16_t S116_CELL_CKPT_SAVE = 1u << 1;
constexpr uint16_t S116_CELL_CKPT_ROLLBACK = 1u << 2;
constexpr uint16_t S116_CELL_FAIL_CLOSED_ZERO = 1u << 3;
constexpr uint16_t S116_CELL_GRID_BARRIER = 1u << 4;
constexpr uint16_t S116_CELL_FMNMX_PASS_B = 1u << 5;
constexpr uint16_t S116_CELL_L2_SCOREBOARD = 1u << 6;

enum S116Opcode : uint32_t {
    OP_HALT                  = 0u,
    OP_EMBED_BYTE            = 1u,
    OP_MMA_TILE              = 2u,
    OP_WARP_NORM_S3_S15_S31  = 3u,
    OP_SU2_HOLONOMY_STEP     = 4u,
    OP_CAYLEY2X2             = 5u,
    OP_SILIQUANT_A16_DEQUANT = 6u,
    OP_SAMPLE_ARGMAX         = 7u,
    OP_RING_COMMIT           = 8u,
};

struct alignas(16) S116Inst {
    uint32_t opcode;
    uint32_t dst;
    uint32_t src0;
    uint32_t src1;
};

// 32-byte aligned switch-free Single-Opcode CellOp descriptor (silikernels#2 Piece 1)
// flags & 1 == 0: src_tile_ptr points to 128 B A_16 Block-H4 tile (8 cols x 2 u64)
// flags & 1 == 1: src_tile_ptr aliases 128 B canonical s8 L2 KV-ring slice (exact-copy!)
struct alignas(32) S116CellOp {
    const uint4* src_tile_ptr;        // 128 B (8 x uint4): Block-H4 weights OR canonical s8 KV-ring
    const __half* weight_scale_ptr;   // 8 x FP16 col scales (1 scale group = 32 weights = 1 MMA K-tile!)
    uint16_t in_slot;                 // fixed-size activation slot in L2/global ring
    uint16_t out_slot;                // fixed-size output activation slot
    uint16_t norm_group;              // 4 (S^3), 16 (S^15), or 32 (S^31)
    uint16_t flags;                   // IS_KV_RING | CKPT_SAVE | CKPT_ROLLBACK | FAIL_CLOSED_ZERO | GRID_BARRIER
    float eigen_alpha;                // nGPT spherical tangent step
    float decay_eta;                  // persistent register fast-weight S_fast decay
};

struct alignas(64) S116MappedByteRing {
    uint32_t capacity_mask;
    uint32_t in_head;
    uint32_t in_tail;
    uint32_t out_head;
    uint32_t out_tail;
    uint32_t stop_flag;
    uint32_t active_cmd;
    uint32_t active_byte;
    uint32_t sampled_byte;
    uint32_t tokens_processed;
    uint32_t syndrome_errors;
    uint32_t reserved_pad;
    uint8_t* in_bytes;
    uint8_t* out_bytes;
};

struct S116VMWorkspace {
    const __nv_bfloat16* embed_table;
    __nv_bfloat16* mma_a_tiles;
    __nv_bfloat16* mma_b_tiles;
    float* mma_c_tiles;
    float4* act_slots;
    float4* su2_state;
    const float* su2_raw_omega;
    const float2* cayley_omega;
    const uint16_t* a16_codewords;
    const uint32_t* int4_packed;
    const float* quant_scales;
    const float* byte_logits;
    int32_t d_model;
    int32_t n_quads;
    float eps;
};

struct S116CellWorkspace {
    int8_t* act_s8_slots;             // [n_ctas * n_slots, 16, 32] row-major INT8 A tiles (512 B/slot)
    float* act_f32_slots;             // [n_ctas * n_slots, 16, 8]  row-major FP32 slot outputs (512 B/slot)
    float* act_absmax_slots;          // [n_ctas * n_slots] dynamic warp_absmax / 127.0f per slot
    int32_t* raw_imma_out;            // optional [n_slots, 16, 8] CTA-0 raw exact s32 IMMA dump (or nullptr)
    float4* su2_holonomy;             // [n_ctas * 32] live 16 B SU(2) unit-quaternion states
    float4* su2_ckpt_t0;              // [n_ctas * 32] 16 B Pi_{t_0} rollback checkpoints (Lean p5)
    const float* su2_raw_omega;       // [n_ctas * 32 * 3] local su(2) generators
    int32_t n_slots;                  // per-CTA slot stride (>= 1)
    float eps;                        // smooth norm floor
};

struct S116Cell8WarpWorkspace {
    int8_t* act_s8_in;                // [num_ctas, 8, 32] initial per-warp S^{31} int8 input
    float* act_f32_out;               // [num_ctas, 8, 32] final per-warp FP32 output
    float* act_absmax_out;            // [num_ctas, 8] per-warp dynamic warp_absmax / 127.0f
    const float* pass_a_scores;       // [num_ctas, 8, 32] Pass-A KV-ring similarity scores for FMNMX
    int32_t* pass_b_win_slot;         // [num_ctas, 8] winning slot index per warp (optional dump)
    int32_t* pass_b_raw_imma;         // [num_ctas, 8, 128] optional raw s32 IMMA dump (or nullptr)
    float4* su2_holonomy;             // [num_ctas * 256] live 16 B SU(2) states (8 warps x 32 lanes)
    float4* su2_ckpt_t0;              // [num_ctas * 256] 16 B Pi_{t_0} rollback checkpoints
    const float* su2_raw_omega;       // [num_ctas * 256 * 3] local su(2) generators
    int8_t* l2_cross_cta_buf;         // [num_ctas, 256] L2 cross-CTA int8 exchange (d_model=512..1024)
    uint32_t* l2_scoreboard;          // [max_layers] monotonic .gpu-scope L2 epoch counters
    int32_t num_active_ctas;          // 1 for d_model=256 (0 L2 sync!), 2..4 for d_model=512..1024
    float eps;                        // smooth norm floor
};

// ---------------------------------------------------------------------------
// Ampere sm_86 Inline PTX Primitives (.sys Ring Fences + .gpu L2 Scoreboard)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* ptr) {
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

__device__ __forceinline__ void st_release_sys_u32(uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.sys.global.u32 [%0], %1;\n" :: "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* ptr) {
    uint32_t val;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return val;
}

// OBS-ORDA-06 cure: .acq_rel.gpu on the atomic increment eliminates the last-arriving
// CTA acquire race (where the final CTA skips the spin loop and must still acquire prior L2 writes!).
__device__ __forceinline__ uint32_t atom_acq_rel_gpu_add_u32(uint32_t* ptr, uint32_t inc) {
    uint32_t old_val;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;\n" : "=r"(old_val) : "l"(ptr), "r"(inc) : "memory");
    return old_val;
}

// Zero-grid.sync() monotonic .gpu-scope L2 epoch barrier (wrap-safe signed compare + nanosleep 32).
__device__ __forceinline__ void sync_ctas_l2_gpu_scope(
    uint32_t* l2_step_arrived, uint32_t target_epoch
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t old_val = atom_acq_rel_gpu_add_u32(l2_step_arrived, 1u);
        uint32_t cur = old_val + 1u;
        while (static_cast<int32_t>(cur - target_epoch) < 0) {
            cur = ld_acquire_gpu_u32(l2_step_arrived);
            if (static_cast<int32_t>(cur - target_epoch) < 0) {
                asm volatile("nanosleep.u32 32;\n");
            }
        }
    }
    __syncthreads();
}

// Law-3 NaN-safe branchless warp-wide argmax: sanitizes NaN -> -INFINITY BEFORE
// the 5-step FMNMX + SEL butterfly so (win_score, win_slot) can NEVER desync!
__device__ __forceinline__ void warp_fmnmx_argmax_32(
    float local_score, int32_t local_slot, float& win_score, int32_t& win_slot
) {
    win_score = isnan(local_score) ? __int_as_float(0xFF800000u) : local_score;
    win_slot = local_slot;
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        const float other_score = __shfl_xor_sync(0xFFFFFFFFu, win_score, mask);
        const int32_t other_slot = __shfl_xor_sync(0xFFFFFFFFu, win_slot, mask);
        const bool take_other = (other_score > win_score) ||
                                (other_score == win_score && other_slot > win_slot);
        win_score = fmaxf(win_score, other_score);          // native sm_86 FMNMX!
        win_slot = take_other ? other_slot : win_slot;      // native sm_86 SEL!
    }
}

// Construct exact 1-hot 0x7F (127) int8 Operand A fragment for Pass-B IMMA.16832:
// Extracts 127 * V_ring[col, win_slot_mod32] bit-exactly with zero convex mixing!
__device__ __forceinline__ uint32_t pack_onehot_imma_a_frag(
    int32_t win_slot_mod32, int32_t tid_mod4, int32_t half_k16
) {
    const int32_t base_k = half_k16 * 16 + tid_mod4 * 4;
    const uint32_t b0 = (win_slot_mod32 == base_k + 0) ? 127u : 0u;
    const uint32_t b1 = (win_slot_mod32 == base_k + 1) ? 127u : 0u;
    const uint32_t b2 = (win_slot_mod32 == base_k + 2) ? 127u : 0u;
    const uint32_t b3 = (win_slot_mod32 == base_k + 3) ? 127u : 0u;
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ uint8_t ld_acquire_sys_u8(const uint8_t* ptr) {
    uint32_t val;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];\n" : "=r"(val) : "l"(ptr) : "memory");
    return static_cast<uint8_t>(val & 0xFFu);
}

__device__ __forceinline__ void st_release_sys_u8(uint8_t* ptr, uint8_t val) {
    const uint32_t v32 = static_cast<uint32_t>(val);
    asm volatile("st.release.sys.global.u8 [%0], %1;\n" :: "l"(ptr), "r"(v32) : "memory");
}

__device__ __forceinline__ void membar_sys_ptx() {
    asm volatile("membar.sys;\n" ::: "memory");
}

__device__ __forceinline__ void ptx_nanosleep(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;\n" :: "r"(ns));
}

__device__ __forceinline__ void cp_async_ca_16b(void* smem_dst, const void* gmem_src) {
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// Branchless predicated 32-bit select via PTX selp.b32 (zero warp divergence!)
__device__ __forceinline__ uint32_t selp_b32_ptx(uint32_t val_true, uint32_t val_false, uint32_t cond) {
    uint32_t out;
    asm(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        "setp.ne.u32 %%p, %3, 0;\n\t"
        "selp.b32 %0, %1, %2, %%p;\n\t"
        "}\n"
        : "=r"(out)
        : "r"(val_true), "r"(val_false), "r"(cond)
    );
    return out;
}

// Exact saturating FP32 -> signed INT8 round-to-nearest-even via PTX cvt.rni.sat.s8.f32
__device__ __forceinline__ int32_t cvt_rni_sat_s8_f32_ptx(float x) {
    int32_t out;
    asm("cvt.rni.sat.s8.f32 %0, %1;\n" : "=r"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ int dp4a_s32_ptx(int a, int b, int c) {
    int out;
    asm("dp4a.s32.s32 %0, %1, %2, %3;\n" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__ void mma_m16n8k32_s8_s32(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

// ---------------------------------------------------------------------------
// Register-Native Geometric & Zero-Swizzle Siliquant A_16 Block-H4 Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 q_identity() {
    return make_float4(1.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float4 q_mul(float4 p, float4 q) {
    return make_float4(
        fmaf(p.x, q.x, fmaf(-p.y, q.y, fmaf(-p.z, q.z, -p.w * q.w))),
        fmaf(p.x, q.y, fmaf( p.y, q.x, fmaf( p.z, q.w, -p.w * q.z))),
        fmaf(p.x, q.z, fmaf(-p.y, q.w, fmaf( p.z, q.x,  p.w * q.y))),
        fmaf(p.x, q.w, fmaf( p.y, q.z, fmaf(-p.z, q.y,  p.w * q.x)))
    );
}

__device__ __forceinline__ float4 q_norm_safe(float4 q, float eps) {
    const float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    if (n2 <= eps * eps) {
        return q_identity();
    }
    const float inv = rsqrtf(n2);
    return make_float4(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
}

__device__ __forceinline__ float4 su2_cayley_unit(float w1, float w2, float w3) {
    const float n2 = fmaf(w1, w1, fmaf(w2, w2, w3 * w3));
    const float inv_d = 1.0f / (1.0f + n2);
    const float s = 2.0f * inv_d;
    return make_float4((1.0f - n2) * inv_d, s * w1, s * w2, s * w3);
}

__device__ __forceinline__ float4 cayley2x2_quad(float4 x, float2 w) {
    const float w0_2 = w.x * w.x;
    const float d0 = 1.0f / (1.0f + w0_2);
    const float c0 = (1.0f - w0_2) * d0;
    const float s0 = (2.0f * w.x) * d0;

    const float w1_2 = w.y * w.y;
    const float d1 = 1.0f / (1.0f + w1_2);
    const float c1 = (1.0f - w1_2) * d1;
    const float s1 = (2.0f * w.y) * d1;

    return make_float4(
        fmaf(c0, x.x,  s0 * x.y),
        fmaf(-s0, x.x, c0 * x.y),
        fmaf(c1, x.z,  s1 * x.w),
        fmaf(-s1, x.z, c1 * x.w)
    );
}

// Extended Hamming [16, 11, 4] 5-bit syndrome in 5 hardware __popc instructions
__device__ __forceinline__ uint32_t a16_syndrome_u16(uint32_t cw) {
    const uint32_t s0 = static_cast<uint32_t>(__popc(cw & 0xFFFFu) & 1);
    const uint32_t s1 = static_cast<uint32_t>(__popc(cw & 0xAAAAu) & 1);
    const uint32_t s2 = static_cast<uint32_t>(__popc(cw & 0xCCCCu) & 1);
    const uint32_t s3 = static_cast<uint32_t>(__popc(cw & 0xF0F0u) & 1);
    const uint32_t s4 = static_cast<uint32_t>(__popc(cw & 0xFF00u) & 1);
    return s0 | (s1 << 1) | (s2 << 2) | (s3 << 3) | (s4 << 4);
}

// Distributed 4-thread-group [16,11,4] syndrome check in ONLY 2 register shuffles!
// Each thread tid_in_grp in 0..3 holds 1 uint16_t chunk (4 nibbles) of its groupID's uint64_t.
__device__ __forceinline__ uint32_t blockh4_syndrome_2shfl(uint32_t nibs16, int tid_in_grp) {
    const uint32_t p4 = ((nibs16 >> 0) & 1u)
                      | (((nibs16 >> 4) & 1u) << 1)
                      | (((nibs16 >> 8) & 1u) << 2)
                      | (((nibs16 >> 12) & 1u) << 3);
    uint32_t odd = p4 << (tid_in_grp * 4);
    odd |= __shfl_xor_sync(0xffffffffu, odd, 1);
    odd |= __shfl_xor_sync(0xffffffffu, odd, 2);
    return a16_syndrome_u16(odd);
}

__device__ __forceinline__ uint32_t a16_blockh4_syndrome_u64(uint64_t w64) {
    uint32_t odd = 0u;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        odd |= static_cast<uint32_t>((w64 >> (j * 4)) & 1ULL) << j;
    }
    return a16_syndrome_u16(odd);
}

// UB-safe sign-extending 4-point Sylvester H_4 butterfly -> packed 4xint8 word.
// Exact signed int8: |r_i| <= 28 for [-7,+7], <= 32 for [-8,+7], strictly <= 127!
__device__ __forceinline__ uint32_t blockh4_group_to_packed_s8(uint32_t nibs16) {
    const int32_t c0 = static_cast<int32_t>(static_cast<uint32_t>(nibs16 & 0xFu) << 28) >> 28;
    const int32_t c1 = static_cast<int32_t>(static_cast<uint32_t>((nibs16 >> 4) & 0xFu) << 28) >> 28;
    const int32_t c2 = static_cast<int32_t>(static_cast<uint32_t>((nibs16 >> 8) & 0xFu) << 28) >> 28;
    const int32_t c3 = static_cast<int32_t>(static_cast<uint32_t>((nibs16 >> 12) & 0xFu) << 28) >> 28;
    const int32_t s01 = c0 + c1;
    const int32_t d01 = c0 - c1;
    const int32_t s23 = c2 + c3;
    const int32_t d23 = c2 - c3;
    const int32_t r0 = s01 + s23;
    const int32_t r1 = d01 + d23;
    const int32_t r2 = s01 - s23;
    const int32_t r3 = d01 - d23;
    return static_cast<uint32_t>(
        (r0 & 0xFF) | ((r1 & 0xFF) << 8) | ((r2 & 0xFF) << 16) | ((r3 & 0xFF) << 24)
    );
}

__device__ __forceinline__ uint32_t pack_two_bf16(float lo, float hi) {
    const __nv_bfloat162 b2 = __float22bfloat162_rn(make_float2(lo, hi));
    return *reinterpret_cast<const uint32_t*>(&b2);
}

__device__ __forceinline__ float unpack_s4_nibble(uint32_t word, int idx) {
    const int32_t raw = static_cast<int32_t>((word >> (idx * 4)) & 0xFu);
    const int32_t signed_val = (raw >= 8) ? (raw - 16) : raw;
    return static_cast<float>(signed_val);
}

// Register/Warp-local smooth-floor normalization (norm_group in {4, 16, 32})
__device__ __forceinline__ float4 warp_local_norm_quad(
    float4 v, uint32_t norm_group, float eps
) {
    float sq = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w)));
    if (norm_group >= 16u) {
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    }
    if (norm_group >= 32u) {
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
    }
    const float inv = rsqrtf(fmaf(eps, eps, sq));
    return make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
}

// Fused warp_absmax S^{31}/S^{15}/S^3 normalization + full-7.9-bit s8 quantization!
// Computes BOTH sum(x^2) and max(|x|) in the EXACT SAME __shfl_xor_sync butterflies
// (0 extra shuffles, 0 SMEM) -> packs 4 exact s8 activations via cvt.rni.sat.s8.f32
// and outputs out_absmax_div_127 = (warp_absmax / 127.0f) for post-MMA epilogue folding!
__device__ __forceinline__ float4 warp_norm_and_quant_s8_quad(
    float4 v,
    uint32_t norm_group,
    float eps,
    uint32_t& out_packed_s8,
    float& out_absmax_div_127
) {
    float sq = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, v.w * v.w)));
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    if (norm_group >= 16u) {
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    }
    if (norm_group >= 32u) {
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    }
    const float inv = rsqrtf(fmaf(eps, eps, sq));
    const float4 y = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
    const float norm_amax = fmaxf(amax * inv, eps);
    const float q_scale = 127.0f / norm_amax;
    const int32_t q0 = cvt_rni_sat_s8_f32_ptx(y.x * q_scale);
    const int32_t q1 = cvt_rni_sat_s8_f32_ptx(y.y * q_scale);
    const int32_t q2 = cvt_rni_sat_s8_f32_ptx(y.z * q_scale);
    const int32_t q3 = cvt_rni_sat_s8_f32_ptx(y.w * q_scale);
    out_packed_s8 = static_cast<uint32_t>(
        (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24)
    );
    out_absmax_div_127 = norm_amax * (1.0f / 127.0f);
    return y;
}

__device__ __forceinline__ void cta_execute_mma_tile_m16n8k16(
    const __nv_bfloat16* __restrict__ g_a,
    const __nv_bfloat16* __restrict__ g_b_col,
    float* __restrict__ g_c,
    __nv_bfloat16* sh_a,
    __nv_bfloat16* sh_b,
    int tid,
    uint32_t accumulate_flag
) {
    if (tid < 32) {
        cp_async_ca_16b(sh_a + tid * 8, g_a + tid * 8);
    } else if (tid < 48) {
        const int b_chunk = tid - 32;
        cp_async_ca_16b(sh_b + b_chunk * 8, g_b_col + b_chunk * 8);
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    if (tid < 32) {
        const int lane = tid;
        const int group_id = lane >> 2;
        const int tid_in_grp = lane & 3;
        const int col0 = tid_in_grp * 2;
        const int col1 = col0 + 1;

        const uint32_t* sh_a_u32 = reinterpret_cast<const uint32_t*>(sh_a);
        const uint32_t* sh_b_u32 = reinterpret_cast<const uint32_t*>(sh_b);

        const uint32_t a0 = sh_a_u32[group_id * 8 + tid_in_grp];
        const uint32_t a1 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp];
        const uint32_t a2 = sh_a_u32[group_id * 8 + tid_in_grp + 4];
        const uint32_t a3 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp + 4];

        const uint32_t b0 = sh_b_u32[group_id * 8 + tid_in_grp];
        const uint32_t b1 = sh_b_u32[group_id * 8 + tid_in_grp + 4];

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        const int row0 = group_id;
        const int row1 = group_id + 8;
        if (accumulate_flag != 0u) {
            c0 = g_c[row0 * 8 + col0];
            c1 = g_c[row0 * 8 + col1];
            c2 = g_c[row1 * 8 + col0];
            c3 = g_c[row1 * 8 + col1];
        }

        float d0, d1, d2, d3;
        mma_m16n8k16_bf16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

        g_c[row0 * 8 + col0] = d0;
        g_c[row0 * 8 + col1] = d1;
        g_c[row1 * 8 + col0] = d2;
        g_c[row1 * 8 + col1] = d3;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Kernel 1 (PRIMARY): switch-Free Single-Opcode Uniform CellOp Megakernel
// Eliminates Together AI's Fixed Launch Envelope Trap!
//   - Zero-swizzle 4-thread-group Block-H4 -> IMMA.16832.S8.S8 (2x uint16_t/thread)
//   - 2-shuffle distributed [16,11,4] Extended Hamming syndrome check
//   - Branchless flags&1 (selp.b32) 1-in-8 canonical s8 KV-ring lookup
//   - Fused warp_absmax S^31 -> s8 activation quant (cvt.rni.sat.s8.f32)
//   - Persistent register S_fast + SU(2) holonomy + 16 B Pi_{t_0} rollback
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(S116_CELL_BLOCK_THREADS, 1)
void uniform_cell_megakernel_sm86(
    S116MappedByteRing* __restrict__ ring,
    const S116CellOp* __restrict__ schedule,
    int32_t num_ops,
    S116CellWorkspace ws
) {
    cg::grid_group grid = cg::this_grid();
    const int cta_id = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Double-buffered 256 B payload stages (16 x uint4 = 256 B: covers both 128 B Block-H4
    // and 256 B canonical 8x32 s8 KV-ring slice with zero swizzle!) + 512 B A tile + 32 B scales
    __shared__ alignas(16) uint4 sh_src_buf[2][16];  // 512 B total (2 double-buffered stages)
    __shared__ alignas(16) uint4 sh_a_s8[32];        // 512 B (16 x 32 int8 activations)
    __shared__ alignas(16) float sh_scales[8];       // 32 B (8 column scales)

    // Persistent plastic fast-weight accumulators held strictly in scalar registers (0 B spills!)
    float s_fast0 = 0.0f;
    float s_fast1 = 0.0f;
    float s_fast2 = 0.0f;
    float s_fast3 = 0.0f;

    while (true) {
        if (cta_id == 0 && tid == 0) {
            uint32_t cmd = 0u;
            uint32_t cur_byte = 0u;
            const uint32_t cap = ring->capacity_mask + 1u;
            while (cmd == 0u) {
                if (ld_acquire_sys_u32(&ring->stop_flag) != 0u) {
                    cmd = 2u;
                    break;
                }
                const uint32_t in_head = ld_acquire_sys_u32(&ring->in_head);
                const uint32_t in_tail = ring->in_tail;
                const uint32_t out_head = ring->out_head;
                const uint32_t out_tail = ld_acquire_sys_u32(&ring->out_tail);
                if (in_head != in_tail && (out_head - out_tail) < cap) {
                    cur_byte = static_cast<uint32_t>(ld_acquire_sys_u8(&ring->in_bytes[in_tail & ring->capacity_mask]));
                    cmd = 1u;
                    break;
                }
                ptx_nanosleep(64u);
            }
            st_release_sys_u32(&ring->active_byte, cur_byte);
            st_release_sys_u32(&ring->active_cmd, cmd);
        }
        grid.sync();

        if (ld_acquire_sys_u32(&ring->active_cmd) == 2u) {
            break;
        }
        const int64_t cta_slot_base = static_cast<int64_t>(cta_id) * static_cast<int64_t>(ws.n_slots);

        // Prime double-buffer stage 0 via sm_86 cp.async (8x16B for Block-H4 or 16x16B for KV-ring)
        if (num_ops > 0) {
            const int n_chunks0 = ((schedule[0].flags & S116_CELL_IS_KV_RING) != 0u) ? 16 : 8;
            if (tid < n_chunks0) {
                cp_async_ca_16b(&sh_src_buf[0][tid], schedule[0].src_tile_ptr + tid);
            }
        }
        cp_async_commit_group();

        // ZERO switch(opcode) branching! Uniform fixed-envelope CellOp loop.
        for (int32_t pc = 0; pc < num_ops; ++pc) {
            const int stage = pc & 1;
            const S116CellOp op = schedule[pc];
            const uint32_t is_kv_ring = static_cast<uint32_t>(op.flags & S116_CELL_IS_KV_RING);
            const int64_t in_slot_idx = cta_slot_base + static_cast<int64_t>(op.in_slot);
            const int64_t out_slot_idx = cta_slot_base + static_cast<int64_t>(op.out_slot);

            // Stage current INT8 activation tile (512 B = 32 x 16 B)
            const uint4* g_a_u4 = reinterpret_cast<const uint4*>(
                ws.act_s8_slots + in_slot_idx * 512
            );
            if (tid < 32) {
                cp_async_ca_16b(&sh_a_s8[tid], g_a_u4 + tid);
            }
            cp_async_commit_group();

            // 1. Prefetch NEXT specialist/KV tile via cp.async while waiting ONLY on current groups
            if (pc + 1 < num_ops) {
                const int n_chunks_next = ((schedule[pc + 1].flags & S116_CELL_IS_KV_RING) != 0u) ? 16 : 8;
                if (tid < n_chunks_next) {
                    cp_async_ca_16b(&sh_src_buf[stage ^ 1][tid], schedule[pc + 1].src_tile_ptr + tid);
                }
                cp_async_commit_group();
                cp_async_wait_group_1();
            } else {
                cp_async_wait_group_0();
            }

            if (tid < 8) {
                const float h4_factor = (is_kv_ring != 0u) ? 1.0f : 0.5f;
                const float act_scale = ws.act_absmax_slots[in_slot_idx];
                sh_scales[tid] = __half2float(op.weight_scale_ptr[tid]) * h4_factor * act_scale;
            }
            __syncthreads();

            // 2. Warp 0 executes zero-swizzle 4-thread-group Block-H4 OR branchless KV-ring -> IMMA.16832
            if (warp_id == 0) {
                const int group_id = lane >> 2;      // col 0..7
                const int tid_in_grp = lane & 3;     // k-group 0..3

                const uint16_t* sh_w16 = reinterpret_cast<const uint16_t*>(sh_src_buf[stage]);
                const uint32_t nibs_lo = static_cast<uint32_t>(sh_w16[group_id * 8 + tid_in_grp]);
                const uint32_t nibs_hi = static_cast<uint32_t>(sh_w16[group_id * 8 + 4 + tid_in_grp]);

                const uint32_t syn_lo = blockh4_syndrome_2shfl(nibs_lo, tid_in_grp);
                const uint32_t syn_hi = blockh4_syndrome_2shfl(nibs_hi, tid_in_grp);
                if (is_kv_ring == 0u && tid_in_grp == 0 && (syn_lo != 0u || syn_hi != 0u)) {
                    atomicAdd(&ring->syndrome_errors, (syn_lo != 0u ? 1u : 0u) + (syn_hi != 0u ? 1u : 0u));
                }

                uint32_t b0_h4 = blockh4_group_to_packed_s8(nibs_lo);
                uint32_t b1_h4 = blockh4_group_to_packed_s8(nibs_hi);
                if ((op.flags & S116_CELL_FAIL_CLOSED_ZERO) != 0u) {
                    b0_h4 = selp_b32_ptx(0u, b0_h4, syn_lo);
                    b1_h4 = selp_b32_ptx(0u, b1_h4, syn_hi);
                }

                const uint32_t* sh_kv_u32 = reinterpret_cast<const uint32_t*>(sh_src_buf[stage]);
                const uint32_t b0_kv = sh_kv_u32[group_id * 8 + tid_in_grp];
                const uint32_t b1_kv = sh_kv_u32[group_id * 8 + 4 + tid_in_grp];
                const uint32_t b0 = selp_b32_ptx(b0_kv, b0_h4, is_kv_ring);
                const uint32_t b1 = selp_b32_ptx(b1_kv, b1_h4, is_kv_ring);

                const uint32_t* sh_a_u32 = reinterpret_cast<const uint32_t*>(sh_a_s8);
                const uint32_t a0 = sh_a_u32[group_id * 8 + tid_in_grp];
                const uint32_t a1 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp];
                const uint32_t a2 = sh_a_u32[group_id * 8 + tid_in_grp + 4];
                const uint32_t a3 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp + 4];

                int32_t i0, i1, i2, i3;
                mma_m16n8k32_s8_s32(i0, i1, i2, i3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

                const int col0 = tid_in_grp * 2;
                const int col1 = col0 + 1;
                if (cta_id == 0 && ws.raw_imma_out != nullptr) {
                    int32_t* raw_dst = ws.raw_imma_out + static_cast<int64_t>(op.out_slot) * 128;
                    raw_dst[group_id * 8 + col0] = i0;
                    raw_dst[group_id * 8 + col1] = i1;
                    raw_dst[(group_id + 8) * 8 + col0] = i2;
                    raw_dst[(group_id + 8) * 8 + col1] = i3;
                }

                const float sc0 = sh_scales[col0];
                const float sc1 = sh_scales[col1];
                const float y0 = fmaf(static_cast<float>(i0), sc0, s_fast0);
                const float y1 = fmaf(static_cast<float>(i1), sc1, s_fast1);
                const float y2 = fmaf(static_cast<float>(i2), sc0, s_fast2);
                const float y3 = fmaf(static_cast<float>(i3), sc1, s_fast3);

                const float eta = op.decay_eta;
                const float alpha = op.eigen_alpha;
                s_fast0 = fmaf(eta, s_fast0, alpha * y0);
                s_fast1 = fmaf(eta, s_fast1, alpha * y1);
                s_fast2 = fmaf(eta, s_fast2, alpha * y2);
                s_fast3 = fmaf(eta, s_fast3, alpha * y3);

                const int fiber_idx = cta_id * 32 + lane;
                float4 u_cur = ws.su2_holonomy[fiber_idx];
                if ((op.flags & S116_CELL_CKPT_SAVE) != 0u) {
                    ws.su2_ckpt_t0[fiber_idx] = u_cur;
                }
                if ((op.flags & S116_CELL_CKPT_ROLLBACK) != 0u) {
                    u_cur = ws.su2_ckpt_t0[fiber_idx];
                } else {
                    const float4 du = su2_cayley_unit(
                        ws.su2_raw_omega[fiber_idx * 3 + 0],
                        ws.su2_raw_omega[fiber_idx * 3 + 1],
                        ws.su2_raw_omega[fiber_idx * 3 + 2]
                    );
                    u_cur = q_norm_safe(q_mul(u_cur, du), ws.eps);
                }
                ws.su2_holonomy[fiber_idx] = u_cur;

                const float4 y_trans = q_mul(u_cur, make_float4(y0, y1, y2, y3));
                uint32_t q_packed_s8 = 0u;
                float out_scale_div_127 = 1.0f;
                const float4 y_norm = warp_norm_and_quant_s8_quad(
                    y_trans, static_cast<uint32_t>(op.norm_group), ws.eps, q_packed_s8, out_scale_div_127
                );

                float* dst_f32 = ws.act_f32_slots + out_slot_idx * 128;
                dst_f32[group_id * 8 + col0] = y_norm.x;
                dst_f32[group_id * 8 + col1] = y_norm.y;
                dst_f32[(group_id + 8) * 8 + col0] = y_norm.z;
                dst_f32[(group_id + 8) * 8 + col1] = y_norm.w;

                uint32_t* dst_s8_u32 = reinterpret_cast<uint32_t*>(
                    ws.act_s8_slots + out_slot_idx * 512
                );
                dst_s8_u32[lane] = q_packed_s8;
                if (lane == 0) {
                    ws.act_absmax_slots[out_slot_idx] = out_scale_div_127;
                }
            }
            __syncthreads();

            if ((op.flags & S116_CELL_GRID_BARRIER) != 0u) {
                grid.sync();
            }
        }

        if (cta_id == 0 && tid == 0) {
            const uint32_t out_head = ring->out_head;
            const uint32_t in_tail = ring->in_tail;
            const float probe = ws.act_f32_slots[0];
            const uint32_t out_b = (static_cast<uint32_t>(__float_as_uint(probe) >> 16) ^ ld_acquire_sys_u32(&ring->active_byte)) & 0xFFu;
            ring->sampled_byte = out_b;
            st_release_sys_u8(&ring->out_bytes[out_head & ring->capacity_mask], static_cast<uint8_t>(out_b));
            membar_sys_ptx();
            st_release_sys_u32(&ring->out_head, out_head + 1u);
            st_release_sys_u32(&ring->in_tail, in_tail + 1u);
            ring->tokens_processed += 1u;
        }
        grid.sync();
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 (CONTROL): Multi-Opcode switch(opcode) Persistent Megakernel VM
// Directly measures Together AI's Fixed Launch Envelope register inflation!
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(S116_VM_BLOCK_THREADS, 1)
void holosphere_persistent_vm_sm86(
    S116MappedByteRing* __restrict__ ring,
    const S116Inst* __restrict__ program,
    int32_t program_len,
    S116VMWorkspace ws
) {
    cg::grid_group grid = cg::this_grid();
    const int cta_id = static_cast<int>(blockIdx.x);
    const int num_ctas = static_cast<int>(gridDim.x);
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ alignas(16) __nv_bfloat16 sh_a[16 * 16];
    __shared__ alignas(16) __nv_bfloat16 sh_b[8 * 16];
    __shared__ alignas(16) __nv_bfloat16 sh_embed[S116_D_MODEL];

    while (true) {
        if (cta_id == 0 && tid == 0) {
            uint32_t cmd = 0u;
            uint32_t cur_byte = 0u;
            const uint32_t cap = ring->capacity_mask + 1u;
            while (cmd == 0u) {
                if (ld_acquire_sys_u32(&ring->stop_flag) != 0u) {
                    cmd = 2u;
                    break;
                }
                const uint32_t in_head = ld_acquire_sys_u32(&ring->in_head);
                const uint32_t in_tail = ring->in_tail;
                const uint32_t out_head = ring->out_head;
                const uint32_t out_tail = ld_acquire_sys_u32(&ring->out_tail);
                if (in_head != in_tail && (out_head - out_tail) < cap) {
                    cur_byte = static_cast<uint32_t>(ld_acquire_sys_u8(&ring->in_bytes[in_tail & ring->capacity_mask]));
                    cmd = 1u;
                    break;
                }
                ptx_nanosleep(64u);
            }
            st_release_sys_u32(&ring->active_byte, cur_byte);
            st_release_sys_u32(&ring->active_cmd, cmd);
        }
        grid.sync();

        if (ld_acquire_sys_u32(&ring->active_cmd) == 2u) {
            break;
        }
        const uint32_t cur_byte = ld_acquire_sys_u32(&ring->active_byte) & 0xFFu;

        for (int32_t pc = 0; pc < program_len; ++pc) {
            const S116Inst inst = program[pc];
            const uint32_t op = inst.opcode;
            const uint32_t flags = inst.src1;

            if (op == OP_HALT) {
                break;
            } else if (op == OP_EMBED_BYTE) {
                if (cta_id == 0) {
                    const __nv_bfloat16* row_ptr = ws.embed_table + static_cast<int64_t>(cur_byte) * ws.d_model;
                    const int n_chunks = ws.d_model / 8;
                    if (tid < n_chunks && tid < (S116_D_MODEL / 8)) {
                        cp_async_ca_16b(sh_embed + tid * 8, row_ptr + tid * 8);
                    }
                    cp_async_commit_group();
                    cp_async_wait_group_0();
                    __syncthreads();
                    float4* dst_act = ws.act_slots + static_cast<int64_t>(inst.dst) * ws.n_quads;
                    if (tid < ws.n_quads && tid < (S116_D_MODEL / 4)) {
                        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(sh_embed + tid * 4);
                        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(sh_embed + tid * 4 + 2);
                        const float2 f0 = __bfloat1622float2(lo);
                        const float2 f1 = __bfloat1622float2(hi);
                        dst_act[tid] = make_float4(f0.x, f0.y, f1.x, f1.y);
                    }
                    __syncthreads();
                }
            } else if (op == OP_SILIQUANT_A16_DEQUANT) {
                const int total_blocks = static_cast<int>(inst.src0);
                const int blk_idx = cta_id * S116_VM_BLOCK_THREADS + tid;
                if (blk_idx < total_blocks) {
                    const uint32_t mode_int4 = flags & S116_DEQUANT_MODE_INT4;
                    const uint32_t fail_zero = flags & S116_DEQUANT_FAIL_CLOSED_ZERO;
                    const float sc = ws.quant_scales[blk_idx];
                    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(
                        ws.mma_b_tiles + static_cast<int64_t>(inst.dst) * 128 + static_cast<int64_t>(blk_idx) * 16
                    );
                    if (mode_int4 == 0u) {
                        const uint32_t cw = static_cast<uint32_t>(ws.a16_codewords[blk_idx]);
                        const uint32_t syn = a16_syndrome_u16(cw);
                        float eff_sc = sc;
                        if (syn != 0u) {
                            atomicAdd(&ring->syndrome_errors, 1u);
                            if (fail_zero != 0u) {
                                eff_sc = 0.0f;
                            }
                        }
                        #pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            const float v0 = ((cw >> (p * 2)) & 1u) ? eff_sc : -eff_sc;
                            const float v1 = ((cw >> (p * 2 + 1)) & 1u) ? eff_sc : -eff_sc;
                            out_u32[p] = pack_two_bf16(v0, v1);
                        }
                    } else {
                        const uint32_t w0 = ws.int4_packed[blk_idx * 2];
                        const uint32_t w1 = ws.int4_packed[blk_idx * 2 + 1];
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            out_u32[p] = pack_two_bf16(
                                unpack_s4_nibble(w0, p * 2) * sc,
                                unpack_s4_nibble(w0, p * 2 + 1) * sc
                            );
                            out_u32[p + 4] = pack_two_bf16(
                                unpack_s4_nibble(w1, p * 2) * sc,
                                unpack_s4_nibble(w1, p * 2 + 1) * sc
                            );
                        }
                    }
                }
            } else if (op == OP_MMA_TILE) {
                const int tile_count = static_cast<int>(inst.src0 & 0xFFFFu);
                const uint32_t acc_flag = flags & 1u;
                for (int t = cta_id; t < tile_count; t += num_ctas) {
                    const __nv_bfloat16* g_a = ws.mma_a_tiles + static_cast<int64_t>(t) * 256;
                    const __nv_bfloat16* g_b = ws.mma_b_tiles + static_cast<int64_t>(t) * 128;
                    float* g_c = ws.mma_c_tiles + static_cast<int64_t>(inst.dst + t) * 128;
                    cta_execute_mma_tile_m16n8k16(g_a, g_b, g_c, sh_a, sh_b, tid, acc_flag);
                }
            } else if (op == OP_WARP_NORM_S3_S15_S31) {
                if (cta_id == 0 && tid < ws.n_quads) {
                    float4* buf = ws.act_slots + static_cast<int64_t>(inst.dst) * ws.n_quads;
                    buf[tid] = warp_local_norm_quad(buf[tid], inst.src0, ws.eps);
                }
            } else if (op == OP_SU2_HOLONOMY_STEP) {
                if (cta_id == 0 && tid < ws.n_quads) {
                    float4* act_buf = ws.act_slots + static_cast<int64_t>(inst.dst) * ws.n_quads;
                    const float4 du = su2_cayley_unit(
                        ws.su2_raw_omega[tid * 3 + 0],
                        ws.su2_raw_omega[tid * 3 + 1],
                        ws.su2_raw_omega[tid * 3 + 2]
                    );
                    const float4 u_next = q_norm_safe(q_mul(ws.su2_state[tid], du), ws.eps);
                    ws.su2_state[tid] = u_next;
                    act_buf[tid] = q_mul(u_next, act_buf[tid]);
                }
            } else if (op == OP_CAYLEY2X2) {
                if (cta_id == 0 && tid < ws.n_quads) {
                    float4* act_buf = ws.act_slots + static_cast<int64_t>(inst.dst) * ws.n_quads;
                    act_buf[tid] = cayley2x2_quad(act_buf[tid], ws.cayley_omega[tid]);
                }
            } else if (op == OP_SAMPLE_ARGMAX) {
                if (cta_id == 0 && tid < 32) {
                    const int lane = tid;
                    float best_val = -3.402823466e+38f;
                    uint32_t best_idx = 0u;
                    #pragma unroll
                    for (int step = 0; step < 8; ++step) {
                        const uint32_t idx = static_cast<uint32_t>(lane + step * 32);
                        const float v = ws.byte_logits[idx];
                        if (v > best_val || (v == best_val && idx < best_idx)) {
                            best_val = v;
                            best_idx = idx;
                        }
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        const float other_val = __shfl_xor_sync(0xffffffffu, best_val, offset);
                        const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, offset);
                        if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                            best_val = other_val;
                            best_idx = other_idx;
                        }
                    }
                    if (lane == 0) {
                        ring->sampled_byte = best_idx & 0xFFu;
                    }
                }
            } else if (op == OP_RING_COMMIT) {
                if (cta_id == 0 && tid == 0) {
                    const uint32_t out_head = ring->out_head;
                    const uint32_t in_tail = ring->in_tail;
                    st_release_sys_u8(
                        &ring->out_bytes[out_head & ring->capacity_mask],
                        static_cast<uint8_t>(ring->sampled_byte & 0xFFu)
                    );
                    membar_sys_ptx();
                    st_release_sys_u32(&ring->out_head, out_head + 1u);
                    st_release_sys_u32(&ring->in_tail, in_tail + 1u);
                    ring->tokens_processed += 1u;
                }
            }

            if ((flags & S116_FLAG_GRID_BARRIER) != 0u || op == OP_RING_COMMIT) {
                grid.sync();
            }
        }
        grid.sync();
    }
}

// ---------------------------------------------------------------------------
// Standalone Unit-Testable Zero-Spill sm_86 Micro-Kernels (Kernels 3-7)
// ---------------------------------------------------------------------------

// Kernel 3: Zero-Swizzle 4-Thread-Group Block-H4 + 2-Shuffle Syndrome -> IMMA.16832.S8.S8
__global__ __launch_bounds__(128, 2)
void s116_blockh4_imma_m16n8k32_sm86(
    const int8_t* __restrict__ g_a_s8,          // [n_tiles, 16, 32] row-major s8
    const uint64_t* __restrict__ g_b_blockh4,   // [n_tiles, 8, 2]   col-major Block-H4 u64
    int32_t* __restrict__ g_d_s32,              // [n_tiles, 16, 8]  row-major exact s32
    uint32_t* __restrict__ syndrome_errors,
    int32_t n_tiles,
    uint32_t fail_closed_zero
) {
    __shared__ alignas(16) uint4 sh_a[32];      // 512 B
    __shared__ alignas(16) uint4 sh_b[8];       // 128 B
    const int tile = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (tile >= n_tiles) return;

    const uint4* a_src = reinterpret_cast<const uint4*>(g_a_s8 + static_cast<int64_t>(tile) * 512);
    const uint4* b_src = reinterpret_cast<const uint4*>(g_b_blockh4 + static_cast<int64_t>(tile) * 16);
    if (tid < 32) {
        cp_async_ca_16b(&sh_a[tid], a_src + tid);
    } else if (tid < 40) {
        cp_async_ca_16b(&sh_b[tid - 32], b_src + (tid - 32));
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    if (tid < 32) {
        const int group_id = tid >> 2;
        const int tid_in_grp = tid & 3;
        const uint16_t* sh_w16 = reinterpret_cast<const uint16_t*>(sh_b);
        const uint32_t nibs_lo = static_cast<uint32_t>(sh_w16[group_id * 8 + tid_in_grp]);
        const uint32_t nibs_hi = static_cast<uint32_t>(sh_w16[group_id * 8 + 4 + tid_in_grp]);
        const uint32_t syn_lo = blockh4_syndrome_2shfl(nibs_lo, tid_in_grp);
        const uint32_t syn_hi = blockh4_syndrome_2shfl(nibs_hi, tid_in_grp);
        if (tid_in_grp == 0 && syndrome_errors != nullptr && (syn_lo != 0u || syn_hi != 0u)) {
            atomicAdd(syndrome_errors, (syn_lo != 0u ? 1u : 0u) + (syn_hi != 0u ? 1u : 0u));
        }
        uint32_t b0 = blockh4_group_to_packed_s8(nibs_lo);
        uint32_t b1 = blockh4_group_to_packed_s8(nibs_hi);
        if (fail_closed_zero != 0u) {
            b0 = selp_b32_ptx(0u, b0, syn_lo);
            b1 = selp_b32_ptx(0u, b1, syn_hi);
        }
        const uint32_t* sh_a_u32 = reinterpret_cast<const uint32_t*>(sh_a);
        const uint32_t a0 = sh_a_u32[group_id * 8 + tid_in_grp];
        const uint32_t a1 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp];
        const uint32_t a2 = sh_a_u32[group_id * 8 + tid_in_grp + 4];
        const uint32_t a3 = sh_a_u32[(group_id + 8) * 8 + tid_in_grp + 4];

        int32_t d0, d1, d2, d3;
        mma_m16n8k32_s8_s32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

        int32_t* dst = g_d_s32 + static_cast<int64_t>(tile) * 128;
        const int col0 = tid_in_grp * 2;
        dst[group_id * 8 + col0] = d0;
        dst[group_id * 8 + col0 + 1] = d1;
        dst[(group_id + 8) * 8 + col0] = d2;
        dst[(group_id + 8) * 8 + col0 + 1] = d3;
    }
}

// Kernel 4: Batch-1 Decode A_16 Block-H4 -> dp4a.s32.s32 GEMV (exact s32!)
__global__ __launch_bounds__(256, 4)
void s116_blockh4_dp4a_gemv_sm86(
    const uint64_t* __restrict__ packed_w,
    const __half* __restrict__ scales,
    const int4* __restrict__ x_s8,
    float x_scale,
    float* __restrict__ y_out,
    uint32_t* __restrict__ syndrome_errors,
    int32_t M,
    int32_t K,
    uint32_t fail_closed_zero
) {
    const int warp_id = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (warp_id >= M) return;

    const int n_blk16 = K >> 4;
    const uint64_t* row_p = packed_w + static_cast<int64_t>(warp_id) * n_blk16;
    const __half* row_s = scales + static_cast<int64_t>(warp_id) * (K >> 5);
    float acc = 0.0f;
    uint32_t bad_cnt = 0u;

    for (int b = lane; b < n_blk16; b += 32) {
        const uint64_t w64 = __ldg(row_p + b);
        const int4 xv = __ldg(x_s8 + b);
        const uint32_t syn = a16_blockh4_syndrome_u64(w64);
        if (syn != 0u) {
            bad_cnt += 1u;
        }
        int idot = 0;
        if (syn == 0u || fail_closed_zero == 0u) {
            const uint32_t wp0 = blockh4_group_to_packed_s8(static_cast<uint32_t>(w64));
            const uint32_t wp1 = blockh4_group_to_packed_s8(static_cast<uint32_t>(w64 >> 16));
            const uint32_t wp2 = blockh4_group_to_packed_s8(static_cast<uint32_t>(w64 >> 32));
            const uint32_t wp3 = blockh4_group_to_packed_s8(static_cast<uint32_t>(w64 >> 48));
            idot = dp4a_s32_ptx(static_cast<int>(wp0), xv.x, idot);
            idot = dp4a_s32_ptx(static_cast<int>(wp1), xv.y, idot);
            idot = dp4a_s32_ptx(static_cast<int>(wp2), xv.z, idot);
            idot = dp4a_s32_ptx(static_cast<int>(wp3), xv.w, idot);
        }
        const float sc = __half2float(__ldg(row_s + (b >> 1))) * 0.5f * x_scale;
        acc = fmaf(static_cast<float>(idot), sc, acc);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
        bad_cnt += __shfl_down_sync(0xffffffffu, bad_cnt, off);
    }
    if (lane == 0) {
        y_out[warp_id] = acc;
        if (bad_cnt != 0u && syndrome_errors != nullptr) {
            atomicAdd(syndrome_errors, bad_cnt);
        }
    }
}

// Kernel 5: Standalone BF16 mma.sync.m16n8k16 tile
__global__ __launch_bounds__(128, 2)
void s116_mma_m16n8k16_tile_sm86(
    const __nv_bfloat16* __restrict__ g_a,
    const __nv_bfloat16* __restrict__ g_b_col,
    float* __restrict__ g_c,
    int32_t n_tiles,
    uint32_t accumulate_flag
) {
    __shared__ alignas(16) __nv_bfloat16 sh_a[16 * 16];
    __shared__ alignas(16) __nv_bfloat16 sh_b[8 * 16];
    const int tile = static_cast<int>(blockIdx.x);
    if (tile >= n_tiles) return;
    cta_execute_mma_tile_m16n8k16(
        g_a + static_cast<int64_t>(tile) * 256,
        g_b_col + static_cast<int64_t>(tile) * 128,
        g_c + static_cast<int64_t>(tile) * 128,
        sh_a,
        sh_b,
        static_cast<int>(threadIdx.x),
        accumulate_flag
    );
}

// Kernel 6: Standalone register/warp-local S^3 / S^15 / S^31 norm + fused warp_absmax s8 quant
__global__ __launch_bounds__(256, 4)
void s116_warp_norm_sm86(
    const float4* __restrict__ x_in,
    float4* __restrict__ y_out,
    uint32_t* __restrict__ y_s8_packed_out,
    float* __restrict__ absmax_div_127_out,
    int64_t n_quads,
    uint32_t norm_group,
    float eps
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float4 v = (idx < n_quads) ? x_in[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint32_t q_packed = 0u;
    float sc = 0.0f;
    const float4 out = warp_norm_and_quant_s8_quad(v, norm_group, eps, q_packed, sc);
    if (idx < n_quads) {
        y_out[idx] = out;
        if (y_s8_packed_out != nullptr) {
            y_s8_packed_out[idx] = q_packed;
        }
        if (absmax_div_127_out != nullptr) {
            absmax_div_127_out[idx] = sc;
        }
    }
}

// Kernel 7: Standalone SU(2) Cayley holonomy step + Beltrami Cayley2x2
__global__ __launch_bounds__(256, 4)
void s116_su2_cayley_step_sm86(
    const float4* __restrict__ u_prev,
    const float* __restrict__ raw_omega,
    const float4* __restrict__ act_in,
    const float2* __restrict__ cayley_omega,
    float4* __restrict__ u_next_out,
    float4* __restrict__ act_out,
    int64_t n_quads,
    float eps
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_quads) return;
    const float4 du = su2_cayley_unit(
        raw_omega[idx * 3 + 0], raw_omega[idx * 3 + 1], raw_omega[idx * 3 + 2]
    );
    const float4 u_next = q_norm_safe(q_mul(u_prev[idx], du), eps);
    u_next_out[idx] = u_next;
    act_out[idx] = cayley2x2_quad(q_mul(u_next, act_in[idx]), cayley_omega[idx]);
}

// ---------------------------------------------------------------------------
// Kernel 8 (Phase-2B PRIMARY): 8-Warp Zero-grid.sync() .gpu-Scope Uniform CellOp
//   - ALL 8/8 warps active in parallel (8 x S^{31} heads = 256 ch/CTA!)
//   - ZERO cooperative_groups::this_grid().sync()!
//     * Intra-CTA: 256 B cross-head int8 SMEM exchange + __syncthreads() (~20 cyc)
//     * Inter-CTA (d_model=512..1024): native .gpu-scope L2 acquire/release scoreboard
//   - Law-3 NaN-safe warp_fmnmx_argmax_32 + one-hot 0x7F Pass-B IMMA.16832
//   - ZERO HMMA.16816 instructions (0 IMMA<->HMMA pipe stalls!), 0 B stack/spills!
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(S116_CELL_BLOCK_THREADS, 1)
void uniform_cell_8warp_sm86(
    S116MappedByteRing* __restrict__ ring,
    const S116CellOp* __restrict__ schedule,
    int32_t num_ops,
    S116Cell8WarpWorkspace ws
) {
    const int cta_id = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;   // 0..7 (ALL 8 warps active!)
    const int lane = tid & 31;      // 0..31
    const int group_id = lane >> 2; // 0..7
    const int tid_in_grp = lane & 3;// 0..3

    // 4,680 B total SMEM (< 4.6% of 100 KiB on sm_86):
    //   - sh_src_buf[2][8][16] (4,096 B) double-buffered 8-warp Block-H4 / KV-ring tiles
    //   - sh_scales[8][8]        (256 B) per-warp 8-col scales
    //   - smem_x_s8[8][36]       (288 B) bank-conflict-free padded cross-head int8 exchange (OBS-ORDA-07)
    //   - smem_head_scale[8]      (32 B) per-head warp_absmax / 127.0f
    //   - sh_cmd_byte[2]           (8 B) CTA-local broadcast
    __shared__ alignas(16) uint4 sh_src_buf[2][8][16];
    __shared__ alignas(16) float sh_scales[8][8];
    __shared__ alignas(16) int8_t smem_x_s8[8][36];
    __shared__ alignas(16) float smem_head_scale[8];
    __shared__ uint32_t sh_cmd_byte[2];

    // Per-thread persistent plastic fast-weight accumulators (explicit scalars -> 0 B spills!)
    float s_fast0 = 0.0f;
    float s_fast1 = 0.0f;
    float s_fast2 = 0.0f;
    float s_fast3 = 0.0f;

    uint32_t token_epoch = 0u;
    const uint32_t n_ctas = static_cast<uint32_t>(ws.num_active_ctas > 0 ? ws.num_active_ctas : 1);

    // Seed bank-conflict-free smem_x_s8[8][36] from initial input once (1 byte/thread = 256 B)
    smem_x_s8[warp_id][lane] = ws.act_s8_in[cta_id * 256 + tid];
    if (lane == 0) {
        smem_head_scale[warp_id] = ws.act_absmax_out[cta_id * 8 + warp_id];
    }
    __syncthreads();

    while (true) {
        token_epoch += 1u;

        // Zero grid.sync(): CTA 0 polls host ring via .sys; if n_ctas > 1, signals via .acq_rel.gpu L2!
        if (cta_id == 0) {
            if (tid == 0) {
                uint32_t cmd = 0u;
                uint32_t cur_byte = 0u;
                const uint32_t cap = ring->capacity_mask + 1u;
                while (cmd == 0u) {
                    if (ld_acquire_sys_u32(&ring->stop_flag) != 0u) {
                        cmd = 2u;
                        break;
                    }
                    const uint32_t in_head = ld_acquire_sys_u32(&ring->in_head);
                    const uint32_t in_tail = ring->in_tail;
                    const uint32_t out_head = ring->out_head;
                    const uint32_t out_tail = ld_acquire_sys_u32(&ring->out_tail);
                    if (in_head != in_tail && (out_head - out_tail) < cap) {
                        cur_byte = static_cast<uint32_t>(ld_acquire_sys_u8(&ring->in_bytes[in_tail & ring->capacity_mask]));
                        cmd = 1u;
                        break;
                    }
                    ptx_nanosleep(64u);
                }
                sh_cmd_byte[0] = cmd;
                sh_cmd_byte[1] = cur_byte;
                if (n_ctas > 1u) {
                    st_release_sys_u32(&ring->active_byte, cur_byte);
                    st_release_sys_u32(&ring->active_cmd, cmd);
                    atom_acq_rel_gpu_add_u32(&ws.l2_scoreboard[0], 1u);
                }
            }
        } else if (tid == 0) {
            while (static_cast<int32_t>(ld_acquire_gpu_u32(&ws.l2_scoreboard[0]) - token_epoch) < 0) {
                ptx_nanosleep(32u);
            }
            sh_cmd_byte[0] = ld_acquire_sys_u32(&ring->active_cmd);
            sh_cmd_byte[1] = ld_acquire_sys_u32(&ring->active_byte);
        }
        __syncthreads();

        if (sh_cmd_byte[0] == 2u) {
            break;
        }

        // Prime double-buffer stage 0 across ALL 8 warps in parallel!
        if (num_ops > 0) {
            const uint32_t is_kv0 = static_cast<uint32_t>(schedule[0].flags & S116_CELL_IS_KV_RING);
            const int n_ch0 = (is_kv0 != 0u) ? 16 : 8;
            if (lane < n_ch0) {
                cp_async_ca_16b(
                    &sh_src_buf[0][warp_id][lane],
                    schedule[0].src_tile_ptr + static_cast<int64_t>(warp_id) * n_ch0 + lane
                );
            }
        }
        cp_async_commit_group();

        // ZERO switch(opcode) & ZERO grid.sync(): ALL 8 warps execute in lockstep!
        for (int32_t pc = 0; pc < num_ops; ++pc) {
            const int stage = pc & 1;
            const S116CellOp op = schedule[pc];
            const uint32_t is_kv_ring = static_cast<uint32_t>(op.flags & S116_CELL_IS_KV_RING);
            const uint32_t is_pass_b = static_cast<uint32_t>(op.flags & S116_CELL_FMNMX_PASS_B);

            // Prefetch next layer across ALL 8 warps simultaneously
            if (pc + 1 < num_ops) {
                const uint32_t is_kv_nxt = static_cast<uint32_t>(schedule[pc + 1].flags & S116_CELL_IS_KV_RING);
                const int n_ch_nxt = (is_kv_nxt != 0u) ? 16 : 8;
                if (lane < n_ch_nxt) {
                    cp_async_ca_16b(
                        &sh_src_buf[stage ^ 1][warp_id][lane],
                        schedule[pc + 1].src_tile_ptr + static_cast<int64_t>(warp_id) * n_ch_nxt + lane
                    );
                }
                cp_async_commit_group();
                cp_async_wait_group_1();
            } else {
                cp_async_wait_group_0();
            }

            if (lane < 8) {
                const float h4_factor = (is_kv_ring != 0u) ? (is_pass_b != 0u ? (1.0f / 127.0f) : 1.0f) : 0.5f;
                const float sc_in = (is_pass_b != 0u) ? 1.0f : smem_head_scale[(warp_id + lane) & 7];
                sh_scales[warp_id][lane] = __half2float(op.weight_scale_ptr[warp_id * 8 + lane]) * h4_factor * sc_in;
            }
            __syncthreads();

            // Zero-swizzle 4-thread-group Block-H4 + 2-shfl [16,11,4] OR canonical s8 KV-ring
            const uint16_t* sh_w16 = reinterpret_cast<const uint16_t*>(sh_src_buf[stage][warp_id]);
            const uint32_t nibs_lo = static_cast<uint32_t>(sh_w16[group_id * 8 + tid_in_grp]);
            const uint32_t nibs_hi = static_cast<uint32_t>(sh_w16[group_id * 8 + 4 + tid_in_grp]);

            const uint32_t syn_lo = blockh4_syndrome_2shfl(nibs_lo, tid_in_grp);
            const uint32_t syn_hi = blockh4_syndrome_2shfl(nibs_hi, tid_in_grp);
            if (is_kv_ring == 0u && tid_in_grp == 0 && (syn_lo != 0u || syn_hi != 0u)) {
                atomicAdd(&ring->syndrome_errors, (syn_lo != 0u ? 1u : 0u) + (syn_hi != 0u ? 1u : 0u));
            }

            uint32_t b0_h4 = blockh4_group_to_packed_s8(nibs_lo);
            uint32_t b1_h4 = blockh4_group_to_packed_s8(nibs_hi);
            if ((op.flags & S116_CELL_FAIL_CLOSED_ZERO) != 0u) {
                b0_h4 = selp_b32_ptx(0u, b0_h4, syn_lo);
                b1_h4 = selp_b32_ptx(0u, b1_h4, syn_hi);
            }

            const uint32_t* sh_kv_u32 = reinterpret_cast<const uint32_t*>(sh_src_buf[stage][warp_id]);
            const uint32_t b0 = selp_b32_ptx(sh_kv_u32[group_id * 8 + tid_in_grp], b0_h4, is_kv_ring);
            const uint32_t b1 = selp_b32_ptx(sh_kv_u32[group_id * 8 + 4 + tid_in_grp], b1_h4, is_kv_ring);

            // Bank-conflict-free vectorized 32-bit loads from smem_x_s8[8][36] (OBS-ORDA-07)!
            const int h_lo = (warp_id + group_id) & 7;
            const int h_hi = (warp_id + group_id + 4) & 7;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&smem_x_s8[h_lo][tid_in_grp * 4]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&smem_x_s8[h_hi][tid_in_grp * 4]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&smem_x_s8[h_lo][16 + tid_in_grp * 4]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&smem_x_s8[h_hi][16 + tid_in_grp * 4]);

            if (is_pass_b != 0u && ws.pass_a_scores != nullptr) {
                const float local_sc = ws.pass_a_scores[(cta_id * 8 + warp_id) * 32 + lane];
                float win_sc = 0.0f;
                int32_t win_slot = 0;
                warp_fmnmx_argmax_32(local_sc, lane, win_sc, win_slot);
                const uint32_t oh_lo = pack_onehot_imma_a_frag(win_slot, tid_in_grp, 0);
                const uint32_t oh_hi = pack_onehot_imma_a_frag(win_slot, tid_in_grp, 1);
                a0 = oh_lo;
                a1 = oh_lo;
                a2 = oh_hi;
                a3 = oh_hi;
                if (lane == 0 && ws.pass_b_win_slot != nullptr) {
                    ws.pass_b_win_slot[cta_id * 8 + warp_id] = win_slot;
                }
            }

            // Pure IMMA.16832.S8.S8 (ZERO HMMA.16816!)
            int32_t i0, i1, i2, i3;
            mma_m16n8k32_s8_s32(i0, i1, i2, i3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            const int col0 = tid_in_grp * 2;
            const int col1 = col0 + 1;
            if (ws.pass_b_raw_imma != nullptr && is_pass_b != 0u) {
                int32_t* raw_dst = ws.pass_b_raw_imma + static_cast<int64_t>(cta_id * 8 + warp_id) * 128;
                raw_dst[group_id * 8 + col0] = i0;
                raw_dst[group_id * 8 + col1] = i1;
                raw_dst[(group_id + 8) * 8 + col0] = i2;
                raw_dst[(group_id + 8) * 8 + col1] = i3;
            }

            const float sc0 = sh_scales[warp_id][col0];
            const float sc1 = sh_scales[warp_id][col1];
            const float y0 = fmaf(static_cast<float>(i0), sc0, s_fast0);
            const float y1 = fmaf(static_cast<float>(i1), sc1, s_fast1);
            const float y2 = fmaf(static_cast<float>(i2), sc0, s_fast2);
            const float y3 = fmaf(static_cast<float>(i3), sc1, s_fast3);

            s_fast0 = fmaf(op.decay_eta, s_fast0, op.eigen_alpha * y0);
            s_fast1 = fmaf(op.decay_eta, s_fast1, op.eigen_alpha * y1);
            s_fast2 = fmaf(op.decay_eta, s_fast2, op.eigen_alpha * y2);
            s_fast3 = fmaf(op.decay_eta, s_fast3, op.eigen_alpha * y3);

            // Covariant SU(2) holonomy + Lean-proved 16 B Pi_{t_0} rollback checkpoint
            const int fiber_idx = cta_id * 256 + tid;
            float4 u_cur = ws.su2_holonomy[fiber_idx];
            if ((op.flags & S116_CELL_CKPT_SAVE) != 0u) {
                ws.su2_ckpt_t0[fiber_idx] = u_cur;
            }
            if ((op.flags & S116_CELL_CKPT_ROLLBACK) != 0u) {
                u_cur = ws.su2_ckpt_t0[fiber_idx];
            } else {
                const float4 du = su2_cayley_unit(
                    ws.su2_raw_omega[fiber_idx * 3 + 0],
                    ws.su2_raw_omega[fiber_idx * 3 + 1],
                    ws.su2_raw_omega[fiber_idx * 3 + 2]
                );
                u_cur = q_norm_safe(q_mul(u_cur, du), ws.eps);
            }
            ws.su2_holonomy[fiber_idx] = u_cur;

            // Fused warp_absmax S^{31}/S^{15}/S^3 retraction + cvt.rni.sat.s8.f32
            const float4 y_trans = q_mul(u_cur, make_float4(y0, y1, y2, y3));
            uint32_t q_packed_s8 = 0u;
            float out_scale_div_127 = 1.0f;
            const float4 y_norm = warp_norm_and_quant_s8_quad(
                y_trans, static_cast<uint32_t>(op.norm_group), ws.eps, q_packed_s8, out_scale_div_127
            );

            // Publish retracted s8 head slice into bank-conflict-free smem_x_s8[8][36]!
            if (lane < 8) {
                *reinterpret_cast<uint32_t*>(&smem_x_s8[warp_id][lane * 4]) = q_packed_s8;
            }
            if (lane == 0) {
                smem_head_scale[warp_id] = out_scale_div_127;
            }
            if (pc + 1 == num_ops) {
                ws.act_f32_out[(cta_id * 8 + warp_id) * 32 + lane] = y_norm.x;
                if (lane == 0) {
                    ws.act_absmax_out[cta_id * 8 + warp_id] = out_scale_div_127;
                }
            }
            __syncthreads();  // ~20-cycle CTA-local cross-head exchange (NO grid.sync()!)

            // Optional .acq_rel.gpu L2 cross-CTA epoch barrier for d_model=512..1024 (OBS-ORDA-06!)
            // Double-buffered by (pc & 1) to eliminate any inter-layer write-after-read race!
            if (n_ctas > 1u && (op.flags & S116_CELL_L2_SCOREBOARD) != 0u && ws.l2_cross_cta_buf != nullptr) {
                int8_t* stage_buf = ws.l2_cross_cta_buf + static_cast<size_t>(pc & 1) * n_ctas * 256u;
                stage_buf[cta_id * 256 + tid] = smem_x_s8[warp_id][lane];
                sync_ctas_l2_gpu_scope(&ws.l2_scoreboard[pc + 1], token_epoch * n_ctas);
                const int peer_cta = (cta_id + 1) % static_cast<int>(n_ctas);
                smem_x_s8[warp_id][lane] ^= static_cast<int8_t>(stage_buf[peer_cta * 256 + tid] & 1);
                __syncthreads();
            }
        }

        // Synchronize multi-CTA token completion via .acq_rel.gpu L2 barrier (0 grid.sync()!)
        if (n_ctas > 1u) {
            sync_ctas_l2_gpu_scope(&ws.l2_scoreboard[num_ops + 1], token_epoch * n_ctas);
        }

        if (cta_id == 0 && tid == 0) {
            const uint32_t out_head = ring->out_head;
            const uint32_t in_tail = ring->in_tail;
            const float probe = ws.act_f32_out[0];
            const uint32_t out_b = (static_cast<uint32_t>(__float_as_uint(probe) >> 16) ^ sh_cmd_byte[1]) & 0xFFu;
            ring->sampled_byte = out_b;
            st_release_sys_u8(&ring->out_bytes[out_head & ring->capacity_mask], static_cast<uint8_t>(out_b));
            membar_sys_ptx();
            st_release_sys_u32(&ring->out_head, out_head + 1u);
            st_release_sys_u32(&ring->in_tail, in_tail + 1u);
            ring->tokens_processed += 1u;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 10 (OBS-ORDA-07): Weight-in-A (16x32 Block-H4) x 8-Branch B (32x8 s8)
// IMMA.16832.S8.S8 -> 2x output channels/MMA + free 8-way parallel Lean tactic branches!
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(128, 2)
void s116_blockh4_weight_in_a_imma_sm86(
    const uint64_t* __restrict__ w_a_blockh4,   // [n_tiles, 16, 2] row-major 16 weight rows x 32 s8
    const int8_t* __restrict__ x_b_branches,    // [n_tiles, 8, 32] col-major 8 parallel branches x 32 s8
    int32_t* __restrict__ out_d_s32,            // [n_tiles, 16, 8] 16 outputs x 8 branches exact s32
    uint32_t* __restrict__ syndrome_errors,
    int32_t n_tiles,
    uint32_t fail_closed_zero
) {
    __shared__ alignas(16) uint4 sh_w[16];      // 256 B (16 weight rows x 2 u64 Block-H4)
    __shared__ alignas(16) uint4 sh_x[16];      // 256 B (8 parallel branches x 32 s8)
    const int tile = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (tile >= n_tiles) return;

    const uint4* w_src = reinterpret_cast<const uint4*>(w_a_blockh4 + static_cast<int64_t>(tile) * 32);
    const uint4* x_src = reinterpret_cast<const uint4*>(x_b_branches + static_cast<int64_t>(tile) * 256);
    if (tid < 16) {
        cp_async_ca_16b(&sh_w[tid], w_src + tid);
    } else if (tid < 32) {
        cp_async_ca_16b(&sh_x[tid - 16], x_src + (tid - 16));
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    if (tid < 32) {
        const int group_id = tid >> 2;          // row 0..7 (and row+8 -> 8..15)
        const int tid_in_grp = tid & 3;         // k-chunk 0..3
        const uint16_t* sh_w16 = reinterpret_cast<const uint16_t*>(sh_w);

        const uint32_t r0_lo = static_cast<uint32_t>(sh_w16[group_id * 8 + tid_in_grp]);
        const uint32_t r1_lo = static_cast<uint32_t>(sh_w16[(group_id + 8) * 8 + tid_in_grp]);
        const uint32_t r0_hi = static_cast<uint32_t>(sh_w16[group_id * 8 + 4 + tid_in_grp]);
        const uint32_t r1_hi = static_cast<uint32_t>(sh_w16[(group_id + 8) * 8 + 4 + tid_in_grp]);

        const uint32_t s0l = blockh4_syndrome_2shfl(r0_lo, tid_in_grp);
        const uint32_t s1l = blockh4_syndrome_2shfl(r1_lo, tid_in_grp);
        const uint32_t s0h = blockh4_syndrome_2shfl(r0_hi, tid_in_grp);
        const uint32_t s1h = blockh4_syndrome_2shfl(r1_hi, tid_in_grp);
        if (tid_in_grp == 0 && syndrome_errors != nullptr) {
            const uint32_t errs = (s0l != 0u) + (s1l != 0u) + (s0h != 0u) + (s1h != 0u);
            if (errs != 0u) atomicAdd(syndrome_errors, errs);
        }

        uint32_t a0 = blockh4_group_to_packed_s8(r0_lo);
        uint32_t a1 = blockh4_group_to_packed_s8(r1_lo);
        uint32_t a2 = blockh4_group_to_packed_s8(r0_hi);
        uint32_t a3 = blockh4_group_to_packed_s8(r1_hi);
        if (fail_closed_zero != 0u) {
            a0 = selp_b32_ptx(0u, a0, s0l);
            a1 = selp_b32_ptx(0u, a1, s1l);
            a2 = selp_b32_ptx(0u, a2, s0h);
            a3 = selp_b32_ptx(0u, a3, s1h);
        }

        const uint32_t* sh_x_u32 = reinterpret_cast<const uint32_t*>(sh_x);
        const uint32_t b0 = sh_x_u32[group_id * 8 + tid_in_grp];
        const uint32_t b1 = sh_x_u32[group_id * 8 + 4 + tid_in_grp];

        int32_t d0, d1, d2, d3;
        mma_m16n8k32_s8_s32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

        int32_t* dst = out_d_s32 + static_cast<int64_t>(tile) * 128;
        const int col0 = tid_in_grp * 2;
        dst[group_id * 8 + col0] = d0;
        dst[group_id * 8 + col0 + 1] = d1;
        dst[(group_id + 8) * 8 + col0] = d2;
        dst[(group_id + 8) * 8 + col0 + 1] = d3;
    }
}

// ---------------------------------------------------------------------------
// Kernel 9: Standalone Law-3 NaN-Safe FMNMX + One-Hot 0x7F Pass-B IMMA.16832
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(128, 2)
void s116_fmnmx_onehot_imma_sm86(
    const float* __restrict__ pass_a_scores,    // [n_tiles, 32]
    const int8_t* __restrict__ v_ring_col_s8,   // [n_tiles, 8, 32] col-major canonical s8
    int32_t* __restrict__ out_win_slot,         // [n_tiles]
    int32_t* __restrict__ out_d_s32,            // [n_tiles, 16, 8] exact 127 * V_ring[:, win_slot]
    int32_t n_tiles
) {
    __shared__ alignas(16) uint4 sh_v[16];      // 256 B (8 cols x 32 s8)
    const int tile = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (tile >= n_tiles) return;

    const uint4* v_src = reinterpret_cast<const uint4*>(v_ring_col_s8 + static_cast<int64_t>(tile) * 256);
    if (tid < 16) {
        cp_async_ca_16b(&sh_v[tid], v_src + tid);
    }
    cp_async_commit_group();
    cp_async_wait_group_0();
    __syncthreads();

    if (tid < 32) {
        const int group_id = tid >> 2;
        const int tid_in_grp = tid & 3;
        float win_sc = 0.0f;
        int32_t win_slot = 0;
        warp_fmnmx_argmax_32(pass_a_scores[tile * 32 + tid], tid, win_sc, win_slot);
        if (tid == 0) {
            out_win_slot[tile] = win_slot;
        }

        const uint32_t a_lo = pack_onehot_imma_a_frag(win_slot, tid_in_grp, 0);
        const uint32_t a_hi = pack_onehot_imma_a_frag(win_slot, tid_in_grp, 1);
        const uint32_t* sh_v_u32 = reinterpret_cast<const uint32_t*>(sh_v);
        const uint32_t b0 = sh_v_u32[group_id * 8 + tid_in_grp];
        const uint32_t b1 = sh_v_u32[group_id * 8 + 4 + tid_in_grp];

        int32_t d0, d1, d2, d3;
        mma_m16n8k32_s8_s32(d0, d1, d2, d3, a_lo, a_lo, a_hi, a_hi, b0, b1, 0, 0, 0, 0);

        int32_t* dst = out_d_s32 + static_cast<int64_t>(tile) * 128;
        const int col0 = tid_in_grp * 2;
        dst[group_id * 8 + col0] = d0;
        dst[group_id * 8 + col0 + 1] = d1;
        dst[(group_id + 8) * 8 + col0] = d2;
        dst[(group_id + 8) * 8 + col0 + 1] = d3;
    }
}

}  // namespace holosphere_s116

// ---------------------------------------------------------------------------
// C ABI Host Entrypoints (Cooperative Launch + Standalone Micro-Kernels)
// ---------------------------------------------------------------------------

extern "C" {

int holosphere_s116_query_coop_occupancy(
    int* out_cell_blocks_per_sm,
    int* out_vm_blocks_per_sm
) {
    if (out_cell_blocks_per_sm == nullptr || out_vm_blocks_per_sm == nullptr) {
        return -1;
    }
    int c_blk = 0;
    int v_blk = 0;
    cudaError_t err1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &c_blk, holosphere_s116::uniform_cell_megakernel_sm86, holosphere_s116::S116_CELL_BLOCK_THREADS, 0
    );
    if (err1 != cudaSuccess) return static_cast<int>(err1);
    cudaError_t err2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &v_blk, holosphere_s116::holosphere_persistent_vm_sm86, holosphere_s116::S116_VM_BLOCK_THREADS, 0
    );
    *out_cell_blocks_per_sm = c_blk;
    *out_vm_blocks_per_sm = v_blk;
    return static_cast<int>(err2);
}

int holosphere_s116_launch_uniform_cell(
    holosphere_s116::S116MappedByteRing* ring,
    const holosphere_s116::S116CellOp* schedule,
    int32_t num_ops,
    holosphere_s116::S116CellWorkspace ws,
    int32_t num_ctas,
    void* stream_ptr
) {
    if (ring == nullptr || schedule == nullptr || num_ops <= 0 || num_ctas <= 0) {
        return -1;
    }
    void* args[] = { &ring, &schedule, &num_ops, &ws };
    const dim3 grid_dim(static_cast<unsigned>(num_ctas), 1u, 1u);
    const dim3 block_dim(static_cast<unsigned>(holosphere_s116::S116_CELL_BLOCK_THREADS), 1u, 1u);
    return static_cast<int>(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(holosphere_s116::uniform_cell_megakernel_sm86),
        grid_dim, block_dim, args, 0, static_cast<cudaStream_t>(stream_ptr)
    ));
}

int holosphere_s116_launch_persistent_vm(
    holosphere_s116::S116MappedByteRing* ring,
    const holosphere_s116::S116Inst* program,
    int32_t program_len,
    holosphere_s116::S116VMWorkspace ws,
    int32_t num_ctas,
    void* stream_ptr
) {
    if (ring == nullptr || program == nullptr || program_len <= 0 || num_ctas <= 0) {
        return -1;
    }
    void* args[] = { &ring, &program, &program_len, &ws };
    const dim3 grid_dim(static_cast<unsigned>(num_ctas), 1u, 1u);
    const dim3 block_dim(static_cast<unsigned>(holosphere_s116::S116_VM_BLOCK_THREADS), 1u, 1u);
    return static_cast<int>(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(holosphere_s116::holosphere_persistent_vm_sm86),
        grid_dim, block_dim, args, 0, static_cast<cudaStream_t>(stream_ptr)
    ));
}

int holosphere_s116_blockh4_imma_m16n8k32(
    const int8_t* g_a_s8,
    const uint64_t* g_b_blockh4,
    int32_t* g_d_s32,
    uint32_t* syndrome_errors,
    int32_t n_tiles,
    uint32_t fail_closed_zero,
    void* stream_ptr
) {
    if (n_tiles <= 0) return 0;
    holosphere_s116::s116_blockh4_imma_m16n8k32_sm86<<<
        dim3(static_cast<unsigned>(n_tiles)), dim3(128u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(g_a_s8, g_b_blockh4, g_d_s32, syndrome_errors, n_tiles, fail_closed_zero);
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_blockh4_dp4a_gemv(
    const uint64_t* packed_w,
    const void* scales_fp16,
    const void* x_s8,
    float x_scale,
    float* y_out,
    uint32_t* syndrome_errors,
    int32_t M,
    int32_t K,
    uint32_t fail_closed_zero,
    void* stream_ptr
) {
    if (M <= 0 || K <= 0 || (K & 31) != 0) return -1;
    const unsigned warps_per_block = 8u;
    const unsigned grid = (static_cast<unsigned>(M) + warps_per_block - 1u) / warps_per_block;
    holosphere_s116::s116_blockh4_dp4a_gemv_sm86<<<
        dim3(grid), dim3(256u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(
        packed_w, reinterpret_cast<const __half*>(scales_fp16),
        reinterpret_cast<const int4*>(x_s8), x_scale, y_out, syndrome_errors, M, K, fail_closed_zero
    );
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_mma_m16n8k16_f32_bf16(
    const void* g_a,
    const void* g_b_col,
    float* g_c,
    int32_t n_tiles,
    uint32_t accumulate_flag,
    void* stream_ptr
) {
    if (n_tiles <= 0) return 0;
    holosphere_s116::s116_mma_m16n8k16_tile_sm86<<<
        dim3(static_cast<unsigned>(n_tiles)), dim3(128u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(
        reinterpret_cast<const __nv_bfloat16*>(g_a),
        reinterpret_cast<const __nv_bfloat16*>(g_b_col),
        g_c, n_tiles, accumulate_flag
    );
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_warp_norm_quant_s8(
    const float* x_in,
    float* y_out,
    uint32_t* y_s8_packed_out,
    float* absmax_div_127_out,
    int64_t n_quads,
    uint32_t norm_group,
    float eps,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    if (norm_group != 4u && norm_group != 16u && norm_group != 32u) return -2;
    const unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    holosphere_s116::s116_warp_norm_sm86<<<
        dim3(grid), dim3(256u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(
        reinterpret_cast<const float4*>(x_in),
        reinterpret_cast<float4*>(y_out),
        y_s8_packed_out,
        absmax_div_127_out,
        n_quads, norm_group, eps
    );
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_su2_cayley_step_f32(
    const float* u_prev,
    const float* raw_omega,
    const float* act_in,
    const float* cayley_omega,
    float* u_next_out,
    float* act_out,
    int64_t n_quads,
    float eps,
    void* stream_ptr
) {
    if (n_quads <= 0) return 0;
    const unsigned grid = static_cast<unsigned>((n_quads + 255) / 256);
    holosphere_s116::s116_su2_cayley_step_sm86<<<
        dim3(grid), dim3(256u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(
        reinterpret_cast<const float4*>(u_prev),
        raw_omega,
        reinterpret_cast<const float4*>(act_in),
        reinterpret_cast<const float2*>(cayley_omega),
        reinterpret_cast<float4*>(u_next_out),
        reinterpret_cast<float4*>(act_out),
        n_quads, eps
    );
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_configure_l2_persistence(
    const void* weights_ptr,
    size_t want_bytes,
    void* stream_ptr,
    size_t* out_pinned_bytes
) {
    // OBS-ORDA-08: Query live GA104M max persisting L2 (2,883,584 B = 2.75 MiB) & clamp safely!
    int max_persist_l2 = 0;
    cudaError_t err = cudaDeviceGetAttribute(&max_persist_l2, cudaDevAttrMaxPersistingL2CacheSize, 0);
    if (err != cudaSuccess) return static_cast<int>(err);
    const size_t max_b = (max_persist_l2 > 0) ? static_cast<size_t>(max_persist_l2) : 0u;
    const size_t pin_b = (want_bytes < max_b) ? want_bytes : max_b;
    if (out_pinned_bytes != nullptr) *out_pinned_bytes = pin_b;
    if (pin_b == 0u || weights_ptr == nullptr) return 0;

    err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin_b);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(weights_ptr);
    attr.accessPolicyWindow.num_bytes = pin_b;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    return static_cast<int>(cudaStreamSetAttribute(
        static_cast<cudaStream_t>(stream_ptr), cudaStreamAttributeAccessPolicyWindow, &attr
    ));
}

int holosphere_s116_launch_uniform_cell_8warp(
    holosphere_s116::S116MappedByteRing* ring,
    const holosphere_s116::S116CellOp* schedule,
    int32_t num_ops,
    holosphere_s116::S116Cell8WarpWorkspace ws,
    int32_t num_ctas,
    void* stream_ptr
) {
    if (ring == nullptr || schedule == nullptr || num_ops <= 0 || num_ctas <= 0) {
        return -1;
    }
    // OBS-ORDA-06 fail-closed resident-CTA cap check prevents epoch-barrier deadlock!
    int blocks_per_sm = 0;
    int num_sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, holosphere_s116::uniform_cell_8warp_sm86,
            holosphere_s116::S116_CELL_BLOCK_THREADS, 0
        ) != cudaSuccess ||
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
        return -2;
    }
    if (num_ctas > blocks_per_sm * num_sms) {
        return -3;
    }
    holosphere_s116::uniform_cell_8warp_sm86<<<
        dim3(static_cast<unsigned>(num_ctas), 1u, 1u),
        dim3(static_cast<unsigned>(holosphere_s116::S116_CELL_BLOCK_THREADS), 1u, 1u),
        0,
        static_cast<cudaStream_t>(stream_ptr)
    >>>(ring, schedule, num_ops, ws);
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_blockh4_weight_in_a_imma(
    const uint64_t* w_a_blockh4,
    const int8_t* x_b_branches,
    int32_t* out_d_s32,
    uint32_t* syndrome_errors,
    int32_t n_tiles,
    uint32_t fail_closed_zero,
    void* stream_ptr
) {
    if (n_tiles <= 0) return 0;
    holosphere_s116::s116_blockh4_weight_in_a_imma_sm86<<<
        dim3(static_cast<unsigned>(n_tiles)), dim3(128u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(w_a_blockh4, x_b_branches, out_d_s32, syndrome_errors, n_tiles, fail_closed_zero);
    return static_cast<int>(cudaGetLastError());
}

int holosphere_s116_fmnmx_onehot_imma(
    const float* pass_a_scores,
    const int8_t* v_ring_col_s8,
    int32_t* out_win_slot,
    int32_t* out_d_s32,
    int32_t n_tiles,
    void* stream_ptr
) {
    if (n_tiles <= 0) return 0;
    holosphere_s116::s116_fmnmx_onehot_imma_sm86<<<
        dim3(static_cast<unsigned>(n_tiles)), dim3(128u), 0, static_cast<cudaStream_t>(stream_ptr)
    >>>(pass_a_scores, v_ring_col_s8, out_win_slot, out_d_s32, n_tiles);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
