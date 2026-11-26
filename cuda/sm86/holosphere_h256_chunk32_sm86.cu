// Track S117: Bare-Metal Ampere sm_86 Full-Stack Collapse Kernels
//   1. s117_h256_bent4_byte_inject_sm86    (W10-OBS-51/55: 0 B VRAM H_256 x bent4, 0/2048 const warps)
//   2. s117_gumbel_fmnmx_23bit_cellop_sm86 (W10-OBS-56: 23-bit finite Gumbel + FMNMX 8-warp tournament)
//   3. s117_sfast_geo_chunk32_fwd_sm86     (C=32 exact forward: SU(2) holonomy + S_fast + Cayley S^31)
//   4. s117_sfast_geo_chunk32_bwd_sm86     (Stage 1 C=32 bwd: 4 FP16 mini-anchors, 0 gamma^-1, eps-safe v_inv^3)
//   5. s117_su2_3line_gu_bwd_sm86          (Stage 3 C=32 bwd: exact 3-line non-Abelian gU + boundary projection)
//   6. s117_spec8_woodbury_commit_sm86     (W10-OBS-46/59 + OBS-ORDA-21/22: 6 KB Woodbury SMEM, FP32 S_prefix,
//                                           causal multi-step s_pred, exact 1<=win_len<=8 SU(2) replay, 4-shfl gather)
//   7. s117_sprefix_int8_shadow_imma_sm86  (OBS-ORDA-21/23: row-wise int8 S_prefix shadow quant + 2x IMMA.16832
//                                           + 4-shfl z_slow gather + mask {4,8,16} transpose-free S^31 branch norm)
//
// Target: nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --shared -Xcompiler -fPIC
// Strictly 0 B stack frames, 0 B spill stores, 0 B spill loads, 0 LDL/STL across all 7 kernels.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace s117_sm86 {

// ============================================================================
// Common Device Helpers (Warp Reductions, Cayley SU(2), 4-Shfl Gather)
// ============================================================================

__device__ __forceinline__ float warp_sum_32(float val) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_max_abs_32(float val) {
    float a = fabsf(val);
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, mask));
    }
    return a;
}

__device__ __forceinline__ float branch_sum_mask4_8_16(float val) {
    // Reduces strictly across groupID = lane >> 2 (0..7) keeping tid = lane & 3 FIXED!
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFFu, val, 16);
    return val;
}

// OBS-ORDA-21: Exact 4-shuffle z_slow gather from 2-tile IMMA.16832 accumulator layout.
// Shuffles all 4 registers from src_lane = ((lane & 7) << 2) | (b >> 1) BEFORE selecting
// by receiver's reg_slot = ((lane >> 4) << 1) | ((lane >> 3) & 1), fixing 75%-lane row corruption!
__device__ __forceinline__ float gather_z_slow_4shfl(
    float r0, float r1, float r2, float r3, int lane, int b
) {
    int   src_lane = ((lane & 7) << 2) | (b >> 1);
    float s0       = __shfl_sync(0xFFFFFFFFu, r0, src_lane);
    float s1       = __shfl_sync(0xFFFFFFFFu, r1, src_lane);
    float s2       = __shfl_sync(0xFFFFFFFFu, r2, src_lane);
    float s3       = __shfl_sync(0xFFFFFFFFu, r3, src_lane);
    int   reg_slot = ((lane >> 4) << 1) | ((lane >> 3) & 1);
    return (reg_slot == 0) ? s0 : ((reg_slot == 1) ? s1 : ((reg_slot == 2) ? s2 : s3));
}

__device__ __forceinline__ float4 cayley_su2(float3 w) {
    float w2  = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv = 1.0f / (1.0f + w2);
    return make_float4((1.0f - w2) * inv, 2.0f * w.x * inv, 2.0f * w.y * inv, 2.0f * w.z * inv);
}

__device__ __forceinline__ float4 quat_conj(float4 q) {
    return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float4 quat_mul(float4 a, float4 b) {
    return make_float4(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

__device__ __forceinline__ float4 quat_nr_polish(float4 q) {
    float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    float s  = fmaf(-0.5f, n2, 1.5f);
    return make_float4(q.x * s, q.y * s, q.z * s, q.w * s);
}

__device__ __forceinline__ float quat_dot(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

__device__ __forceinline__ float select_lane4(float4 q, int lane_mod4) {
    return (lane_mod4 == 0) ? q.x : ((lane_mod4 == 1) ? q.y : ((lane_mod4 == 2) ? q.z : q.w));
}

__device__ __forceinline__ float4 gather_spinor4(float val, int base_lane) {
    return make_float4(
        __shfl_sync(0xFFFFFFFFu, val, base_lane + 0),
        __shfl_sync(0xFFFFFFFFu, val, base_lane + 1),
        __shfl_sync(0xFFFFFFFFu, val, base_lane + 2),
        __shfl_sync(0xFFFFFFFFu, val, base_lane + 3)
    );
}

__device__ __forceinline__ int8_t quant_f32_to_s8(float x) {
    int32_t q;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(q) : "f"(x));
    return int8_t(q);
}

__device__ __forceinline__ uint32_t pack4_s8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return (uint32_t(uint8_t(b0))) |
           (uint32_t(uint8_t(b1)) << 8) |
           (uint32_t(uint8_t(b2)) << 16) |
           (uint32_t(uint8_t(b3)) << 24);
}

__device__ __forceinline__ void imma_m16n8k32_s8_zero(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    int zero = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(zero), "r"(zero), "r"(zero), "r"(zero)
    );
}

// ============================================================================
// Kernel 1: H_256 x Maiorana-McFarland bent4 Zero-VRAM Byte Injection
// ============================================================================

__device__ __forceinline__ int8_t h256_bent4_byte_inject_s8(uint8_t utf8_byte, int warp_id, int lane_id) {
    uint32_t d      = uint32_t((warp_id << 5) | lane_id);             // 0..255
    uint32_t db     = d ^ ((uint32_t(lane_id) & 7u) << 5);            // Lower-triangular GF(2)^8 bijection
    uint32_t bent4  = uint32_t(__popc((uint32_t(lane_id) & 3u) & ((uint32_t(lane_id) >> 2) & 3u))) & 1u;
    uint32_t parity = (uint32_t(__popc(uint32_t(utf8_byte) & db)) ^ bent4) & 1u;
    return parity ? int8_t(-127) : int8_t(127);
}

__global__ __launch_bounds__(256, 4)
void s117_h256_bent4_byte_inject_sm86(
    const uint8_t* __restrict__ byte_ids, // [N]
    int8_t* __restrict__ out_s8,          // [N, 256]
    int n_tokens
) {
    int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    int warp_id = threadIdx.x >> 5;       // 0..7
    int lane_id = threadIdx.x & 31;       // 0..31
    uint8_t b   = byte_ids[tok];
    out_s8[tok * 256 + threadIdx.x] = h256_bent4_byte_inject_s8(b, warp_id, lane_id);
}

// ============================================================================
// Kernel 2: 23-Bit Zero-Branch Finite Gumbel + FMNMX 8-Warp Tournament
// ============================================================================

__device__ __forceinline__ float gumbel_23bit_finite(uint32_t philox_u32, float temperature) {
    // 23-bit mantissa alignment: ((u32 >> 9) + 0.5f) * 2^-23 in [2^-24, 1 - 2^-24]
    // Strictly finite Gumbel in [-2.811541, +16.635532] over all 2^32 uint32_t inputs!
    float u = fmaf(float(philox_u32 >> 9), 1.1920928955078125e-7f, 5.9604644775390625e-8f);
    float lg2_u;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(lg2_u) : "f"(u));
    float neg_lg2_u = -lg2_u;
    float lg2_neg_lg2_u;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(lg2_neg_lg2_u) : "f"(neg_lg2_u));
    return temperature * fmaf(-0.69314718f, lg2_neg_lg2_u, 0.36651292f);
}

__device__ __forceinline__ void warp_fmnmx_argmax_32(float& val, uint32_t& idx) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float    other_val  = __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        uint32_t other_idx  = __shfl_xor_sync(0xFFFFFFFFu, idx, mask);
        float    best_val   = fmaxf(val, other_val); // Lowers directly to hardware FMNMX
        bool     take_other = (other_val > val) || ((other_val == val) && (other_idx < idx));
        val = best_val;
        idx = take_other ? other_idx : idx;
    }
}

__global__ __launch_bounds__(256, 4)
void s117_gumbel_fmnmx_23bit_cellop_sm86(
    const float* __restrict__ logits,       // [N, 256]
    const uint32_t* __restrict__ rng_u32,   // [N, 256]
    uint8_t* __restrict__ out_byte,         // [N]
    float* __restrict__ out_score,          // [N]
    float temperature,
    int n_tokens
) {
    __shared__ float   smem_val[8]; // 32 B SMEM for 8-warp FP32 scores
    __shared__ uint8_t smem_idx[8]; //  8 B SMEM for winning byte indices

    int tok = blockIdx.x;
    if (tok >= n_tokens) return;
    int tid     = threadIdx.x; // 0..255
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    float    score = logits[tok * 256 + tid] + gumbel_23bit_finite(rng_u32[tok * 256 + tid], temperature);
    uint32_t idx   = uint32_t(tid);
    warp_fmnmx_argmax_32(score, idx);

    if (lane_id == 0) {
        smem_val[warp_id] = score;
        smem_idx[warp_id] = uint8_t(idx);
    }
    __syncthreads();

    if (warp_id == 0) {
        float    w_val = (lane_id < 8) ? smem_val[lane_id] : -3.402823466e38f;
        uint32_t w_idx = (lane_id < 8) ? uint32_t(smem_idx[lane_id]) : 256u;
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov         = __shfl_xor_sync(0xFFFFFFFFu, w_val, mask);
            uint32_t oi         = __shfl_xor_sync(0xFFFFFFFFu, w_idx, mask);
            float    bv         = fmaxf(w_val, ov);
            bool     take_other = (ov > w_val) || ((ov == w_val) && (oi < w_idx));
            w_val = bv;
            w_idx = take_other ? oi : w_idx;
        }
        if (lane_id == 0) {
            out_byte[tok]  = uint8_t(w_idx);
            out_score[tok] = w_val;
        }
    }
}

// ============================================================================
// Kernel 3: Exact C = 32 Chunk Forward (SU(2) Holonomy + S_fast + Cayley S^31)
//   - Zero-transpose strided [B, T, D] support: off = b*stride_b + t*stride_t + h*32 + lane
//     (pass num_heads=H, stride_b=T*D, stride_t=D for 128 B coalesced [B, T, D] loads;
//      or num_heads=1, stride_b=1024, stride_t=32 for legacy [BH, 32, 32] layout)
// ============================================================================

__global__ __launch_bounds__(32, 8)
void s117_sfast_geo_chunk32_fwd_sm86(
    const float* __restrict__ x_chunk,      // [B, T, D] chunk slice (stride_b, stride_t)
    const float* __restrict__ z_slow_chunk, // [B, T, D] precomputed W_q * x_tilde
    const float3* __restrict__ w_su2_chunk, // [B, T, H, 8] (stride_b>>2, stride_t>>2)
    const __half* __restrict__ S_ckpt_0,    // [BH, 32, 32]
    const float4* __restrict__ U_ckpt_0,    // [BH, 8]
    float* __restrict__ y_out,              // [B, T, D]
    float* __restrict__ x_tilde_out,        // [B, T, D]
    __half* __restrict__ S_ckpt_end,        // [BH, 32, 32]
    float4* __restrict__ U_ckpt_end,        // [BH, 8]
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int apply_nr_polish,
    int num_heads,
    int stride_b,
    int stride_t,
    int total_bh
) {
    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx   = bh / num_heads;
    int h_idx   = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    __half2 S_col[16];
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        __half h0 = S_ckpt_0[bh * 1024 + (2 * p + 0) * 32 + lane];
        __half h1 = S_ckpt_0[bh * 1024 + (2 * p + 1) * 32 + lane];
        S_col[p]  = __halves2half2(h0, h1);
    }

    float4 U_cur = U_ckpt_0[bh * 8 + spin_id];
    float  t2    = cayley_t * cayley_t;
    float  inv_t = 1.0f / (1.0f + t2);
    float  c_st  = (1.0f - t2) * inv_t;
    float  s_st  = (2.0f * cayley_t) * inv_t;

    #pragma unroll 1
    for (int t = 0; t < 32; ++t) {
        int64_t t_base = bh_base + int64_t(t) * int64_t(stride_t);
        int64_t off    = t_base + lane;
        int64_t w_off  = (t_base >> 2) + spin_id;
        float   x_i    = x_chunk[off];
        float3  w_s    = w_su2_chunk[w_off];
        U_cur      = quat_mul(cayley_su2(w_s), U_cur);
        if (apply_nr_polish) U_cur = quat_nr_polish(U_cur);

        float4 xs  = gather_spinor4(x_i, base);
        float4 xt4 = quat_mul(quat_conj(U_cur), xs);
        float  xt  = select_lane4(xt4, lmod4);
        x_tilde_out[off] = xt;

        float s_pred = 0.0f;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            float2 sf = __half22float2(S_col[p]);
            float  p0 = warp_sum_32(sf.x * xt);
            float  p1 = warp_sum_32(sf.y * xt);
            if (lane == (2 * p + 0)) s_pred = p0;
            if (lane == (2 * p + 1)) s_pred = p1;
        }

        float z_i   = z_slow_chunk[off] + s_pred;
        float proj  = warp_sum_32(z_i * x_i);
        float v_i   = z_i - proj * x_i;
        float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
        float y_i   = c_st * x_i + s_st * (v_i * v_inv);
        float err_i = y_i - s_pred;
        y_out[off]  = y_i;

        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            float2 sf = __half22float2(S_col[p]);
            float  e0 = __shfl_sync(0xFFFFFFFFu, err_i, 2 * p + 0);
            float  e1 = __shfl_sync(0xFFFFFFFFu, err_i, 2 * p + 1);
            S_col[p]  = __floats2half2_rn(fmaf(gamma, sf.x, eta * e0 * xt), fmaf(gamma, sf.y, eta * e1 * xt));
        }
    }

    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        S_ckpt_end[bh * 1024 + (2 * p + 0) * 32 + lane] = __low2half(S_col[p]);
        S_ckpt_end[bh * 1024 + (2 * p + 1) * 32 + lane] = __high2half(S_col[p]);
    }
    if (lmod4 == 0) U_ckpt_end[bh * 8 + spin_id] = U_cur;
}

// Exact 5-mul binary exponentiation for gamma^t (0 <= t <= 31, ZERO gamma^-1 divisions, safe for gamma == 0)
__device__ __forceinline__ float pow_gamma_u5(float gamma, int t) {
    float res  = 1.0f;
    float base = gamma;
    if (t & 1)  res *= base;
    base *= base;
    if (t & 2)  res *= base;
    base *= base;
    if (t & 4)  res *= base;
    base *= base;
    if (t & 8)  res *= base;
    base *= base;
    if (t & 16) res *= base;
    return res;
}

// ============================================================================
// Kernel 4A: Choice A — Stage 1 C = 32 Closed-Form Woodbury Backward (W10-OBS-60)
//   - 1 warp/CTA (32 threads/CTA) -> EXACT 4,224 B SMEM/warp (6.0x smaller than 25,344 B!)
//   - Stores ONLY smem_xt[32][33] (2,112 B) + smem_err[32][33] (2,112 B) (0 S_anc, 0 S_win!)
//   - Phase 1 scopes S_col[32] so ptxas reclaims all 32 regs before Phase 2!
//   - Streams S_ckpt_0 via L1-coalesced __ldg (1 accum reg, 0 warp_sum_32 in S0^T v_t)
//   - Exact 5-mul pow_gamma_u5 (0 gamma^-1 divisions!) + zero-transpose stride_t = D
//   - Empirically verified on sm_86: 80-reg (24 w/SM) spills 272-384 B even w/ half2 gS_col;
//     16 warps/SM (<= 128 regs, 67,584 B SMEM) compiles at EXACT 0 B stack/spills, 0 LDL/STL!
// ============================================================================

__global__ __launch_bounds__(32, 16)
void s117_sfast_geo_chunk32_bwd_sm86(
    const float* __restrict__ x_chunk,         // [B, T, D] chunk slice (stride_b, stride_t)
    const float* __restrict__ x_tilde_chunk,   // [B, T, D]
    const float* __restrict__ z_slow_chunk,    // [B, T, D]
    const __half* __restrict__ S_ckpt_0,       // [BH, 32, 32] chunk-start checkpoint
    const float* __restrict__ grad_y_chunk,    // [B, T, D]
    const float* __restrict__ grad_S_in,       // [BH, 32, 32]
    float* __restrict__ grad_z_slow_out,       // [B, T, D]
    float* __restrict__ gx_tilde_sfast_out,    // [B, T, D]
    float* __restrict__ gx_geo_out,            // [B, T, D]
    float* __restrict__ grad_S_out,            // [BH, 32, 32]
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int num_heads,
    int stride_b,
    int stride_t,
    int total_bh
) {
    // EXACT 4,224 B/warp SMEM (16 CTAs/SM = 16 warps/SM @ 67,584 B <= 102,400 B!)
    __shared__ __half smem_xt[32][33];   // 2,112 B (+1 padded, 0 bank conflicts)
    __shared__ __half smem_err[32][33];  // 2,112 B (+1 padded, 0 bank conflicts)

    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;
    const __half* S0_head = S_ckpt_0 + int64_t(bh) * 1024;

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    // --- Phase 1 (t = 0 -> 31): Strictly scoped S_col[32] builds smem_xt & smem_err ---
    {
        float S_col[32];
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            S_col[r] = __half2float(__ldg(&S0_head[r * 32 + lane]));
        }

        #pragma unroll 1
        for (int t = 0; t < 32; ++t) {
            int64_t off    = bh_base + int64_t(t) * int64_t(stride_t) + lane;
            float   x_i    = x_chunk[off];
            float   xt     = x_tilde_chunk[off];
            float   s_pred = 0.0f;
            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                float pr = warp_sum_32(S_col[r] * xt);
                if (lane == r) s_pred = pr;
            }
            float z_i   = z_slow_chunk[off] + s_pred;
            float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
            float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
            float err_i = (c_st * x_i + s_st * (v_i * v_inv)) - s_pred;

            smem_xt[t][lane]  = __float2half(xt);
            smem_err[t][lane] = __float2half(err_i);

            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                float err_r = __shfl_sync(0xFFFFFFFFu, err_i, r);
                S_col[r]    = fmaf(gamma, S_col[r], eta * err_r * xt);
            }
        }
    } // S_col[32] dead here -> ptxas reclaims all 32 registers before Phase 2!

    // --- Phase 2 (t = 31 -> 0): Closed-form Woodbury adjoint with gS_col[32] ONLY ---
    float gS_col[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        gS_col[r] = grad_S_in[int64_t(bh) * 1024 + r * 32 + lane];
    }

    #pragma unroll 1
    for (int t = 31; t >= 0; --t) {
        int64_t off      = bh_base + int64_t(t) * int64_t(stride_t) + lane;
        float   x_i      = x_chunk[off];
        float   xt       = __half2float(smem_xt[t][lane]);
        float   err_i    = __half2float(smem_err[t][lane]);
        float   z_slow_i = z_slow_chunk[off];
        float   gy_i     = grad_y_chunk[off];
        float   g_t      = pow_gamma_u5(gamma, t);

        // 1. S_0 * x_t (L1-cached __ldg) & ge_t = eta * gS_t * x_t
        float s_pred_0 = 0.0f;
        float ge_i     = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float S0_rc = __half2float(__ldg(&S0_head[r * 32 + lane]));
            float sp0   = warp_sum_32(S0_rc * xt);
            float ge    = warp_sum_32(gS_col[r] * xt) * eta;
            if (lane == r) {
                s_pred_0 = sp0;
                ge_i     = ge;
            }
        }

        // 2. Closed-form Woodbury forward readout (0 gamma^-1!):
        float s_pred = g_t * s_pred_0;
        float w_fwd  = eta;
        #pragma unroll 1
        for (int k = t - 1; k >= 0; --k) {
            float xk    = __half2float(smem_xt[k][lane]);
            float ek    = __half2float(smem_err[k][lane]);
            float dot_x = warp_sum_32(xk * xt);
            s_pred      = fmaf(w_fwd * dot_x, ek, s_pred);
            w_fwd      *= gamma;
        }

        // 3. Exact Cayley S^31 geodesic backward (eps-safe v_inv^3 + exact gx_geo)
        float z_i      = z_slow_i + s_pred;
        float proj     = warp_sum_32(z_i * x_i);
        float v_i      = z_i - proj * x_i;
        float v_inv    = rsqrtf(warp_sum_32(v_i * v_i) + eps);
        float v_inv3   = v_inv * v_inv * v_inv;

        float gx_next  = gy_i + ge_i;
        float gu_i     = s_st * gx_next;
        float gv_i     = fmaf(-v_inv3 * warp_sum_32(gu_i * v_i), v_i, v_inv * gu_i);
        float gv_dot_x = warp_sum_32(gv_i * x_i);
        float gz_i     = gv_i - gv_dot_x * x_i;
        float gx_geo_i = fmaf(c_st, gx_next, -fmaf(proj, gv_i, gv_dot_x * z_i));

        grad_z_slow_out[off] = gz_i;
        gx_geo_out[off]      = gx_geo_i;

        // 4. Stream S_ckpt_0 via __ldg (1 accum reg s0t_v, 0 warp_sum_32!) + update gS_col
        float v_t   = gz_i - ge_i;
        float s0t_v = 0.0f;
        float gst_e = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float v_r   = __shfl_sync(0xFFFFFFFFu, v_t, r);
            float err_r = __shfl_sync(0xFFFFFFFFu, err_i, r);
            float S0_rc = __half2float(__ldg(&S0_head[r * 32 + lane]));
            s0t_v       = fmaf(S0_rc, v_r, s0t_v);
            gst_e       = fmaf(gS_col[r], eta * err_r, gst_e);
            gS_col[r]   = fmaf(gamma, gS_col[r], v_r * xt);
        }

        // 5. Closed-form Woodbury adjoint S_t^T v_t:
        float st_v  = g_t * s0t_v;
        float w_adj = eta;
        #pragma unroll 1
        for (int k = t - 1; k >= 0; --k) {
            float ek    = __half2float(smem_err[k][lane]);
            float xk    = __half2float(smem_xt[k][lane]);
            float dot_e = warp_sum_32(ek * v_t);
            st_v        = fmaf(w_adj * dot_e, xk, st_v);
            w_adj      *= gamma;
        }
        gx_tilde_sfast_out[off] = st_v + gst_e;
    }

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        grad_S_out[int64_t(bh) * 1024 + r * 32 + lane] = gS_col[r];
    }
}

// ============================================================================
// Kernel 4B: Choice B — Stage 1 C = 32 Woodbury Backward w/ Cached s_pred_fp16
//   - 1 warp/CTA (32 threads/CTA) -> EXACT 6,336 B SMEM/warp!
//   - Fits 16 CTAs/SM = 16 warps/SM (101,376 B <= 102,400 B GA104M SMEM)!
//   - Adds smem_spred[32][33] (2,112 B) -> ELIMINATES Phase-2 s_pred recompute
//     (cuts Phase-2 compute in HALF! Streams S_ckpt_0 ONCE via __ldg with 0 warp_sum_32!)
//   - Exact 5-mul pow_gamma_u5 (0 gamma^-1 divisions!) + zero-transpose stride_t = D
// ============================================================================

__global__ __launch_bounds__(32, 16)
void s117_sfast_geo_chunk32_bwd_choice_b_sm86(
    const float* __restrict__ x_chunk,         // [B, T, D] chunk slice (stride_b, stride_t)
    const float* __restrict__ x_tilde_chunk,   // [B, T, D]
    const float* __restrict__ z_slow_chunk,    // [B, T, D]
    const __half* __restrict__ S_ckpt_0,       // [BH, 32, 32] chunk-start checkpoint
    const float* __restrict__ grad_y_chunk,    // [B, T, D]
    const float* __restrict__ grad_S_in,       // [BH, 32, 32]
    float* __restrict__ grad_z_slow_out,       // [B, T, D]
    float* __restrict__ gx_tilde_sfast_out,    // [B, T, D]
    float* __restrict__ gx_geo_out,            // [B, T, D]
    float* __restrict__ grad_S_out,            // [BH, 32, 32]
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int num_heads,
    int stride_b,
    int stride_t,
    int total_bh
) {
    // EXACT 6,336 B/warp SMEM -> 16 CTAs/SM = 16 warps/SM @ 101,376 B <= 102,400 B!
    __shared__ __half smem_xt[32][33];     // 2,112 B
    __shared__ __half smem_err[32][33];    // 2,112 B
    __shared__ __half smem_spred[32][33];  // 2,112 B (cuts Phase-2 compute in half!)

    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;
    const __half* S0_head = S_ckpt_0 + int64_t(bh) * 1024;

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    // --- Phase 1 (t = 0 -> 31): Strictly scoped S_col[32] builds xt, err, AND s_pred ---
    {
        float S_col[32];
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            S_col[r] = __half2float(__ldg(&S0_head[r * 32 + lane]));
        }

        #pragma unroll 1
        for (int t = 0; t < 32; ++t) {
            int64_t off    = bh_base + int64_t(t) * int64_t(stride_t) + lane;
            float   x_i    = x_chunk[off];
            float   xt     = x_tilde_chunk[off];
            float   s_pred = 0.0f;
            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                float pr = warp_sum_32(S_col[r] * xt);
                if (lane == r) s_pred = pr;
            }
            float z_i   = z_slow_chunk[off] + s_pred;
            float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
            float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
            float err_i = (c_st * x_i + s_st * (v_i * v_inv)) - s_pred;

            smem_xt[t][lane]    = __float2half(xt);
            smem_err[t][lane]   = __float2half(err_i);
            smem_spred[t][lane] = __float2half(s_pred);

            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                float err_r = __shfl_sync(0xFFFFFFFFu, err_i, r);
                S_col[r]    = fmaf(gamma, S_col[r], eta * err_r * xt);
            }
        }
    } // S_col[32] dead here -> ptxas reclaims all 32 registers before Phase 2!

    // --- Phase 2 (t = 31 -> 0): Zero s_pred recompute! Single __ldg S_ckpt_0 stream ---
    float gS_col[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        gS_col[r] = grad_S_in[int64_t(bh) * 1024 + r * 32 + lane];
    }

    #pragma unroll 1
    for (int t = 31; t >= 0; --t) {
        int64_t off      = bh_base + int64_t(t) * int64_t(stride_t) + lane;
        float   x_i      = x_chunk[off];
        float   xt       = __half2float(smem_xt[t][lane]);
        float   err_i    = __half2float(smem_err[t][lane]);
        float   s_pred   = __half2float(smem_spred[t][lane]);
        float   z_slow_i = z_slow_chunk[off];
        float   gy_i     = grad_y_chunk[off];

        // 1. ge_t = eta * gS_t * x_t (zero S_ckpt_0 read needed here!)
        float ge_i = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float ge = warp_sum_32(gS_col[r] * xt) * eta;
            if (lane == r) ge_i = ge;
        }

        // 2. Exact Cayley S^31 geodesic backward (using cached s_pred directly!)
        float z_i      = z_slow_i + s_pred;
        float proj     = warp_sum_32(z_i * x_i);
        float v_i      = z_i - proj * x_i;
        float v_inv    = rsqrtf(warp_sum_32(v_i * v_i) + eps);
        float v_inv3   = v_inv * v_inv * v_inv;

        float gx_next  = gy_i + ge_i;
        float gu_i     = s_st * gx_next;
        float gv_i     = fmaf(-v_inv3 * warp_sum_32(gu_i * v_i), v_i, v_inv * gu_i);
        float gv_dot_x = warp_sum_32(gv_i * x_i);
        float gz_i     = gv_i - gv_dot_x * x_i;
        float gx_geo_i = fmaf(c_st, gx_next, -fmaf(proj, gv_i, gv_dot_x * z_i));

        grad_z_slow_out[off] = gz_i;
        gx_geo_out[off]      = gx_geo_i;

        // 3. Single L1-coalesced __ldg stream of S_ckpt_0 (1 accum reg s0t_v, 0 warp_sum_32!)
        float v_t   = gz_i - ge_i;
        float s0t_v = 0.0f;
        float gst_e = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float v_r   = __shfl_sync(0xFFFFFFFFu, v_t, r);
            float err_r = __shfl_sync(0xFFFFFFFFu, err_i, r);
            float S0_rc = __half2float(__ldg(&S0_head[r * 32 + lane]));
            s0t_v       = fmaf(S0_rc, v_r, s0t_v);
            gst_e       = fmaf(gS_col[r], eta * err_r, gst_e);
            gS_col[r]   = fmaf(gamma, gS_col[r], v_r * xt);
        }

        // 4. Closed-form Woodbury adjoint S_t^T v_t with 5-mul pow_gamma_u5:
        float g_t   = pow_gamma_u5(gamma, t);
        float st_v  = g_t * s0t_v;
        float w_adj = eta;
        #pragma unroll 1
        for (int k = t - 1; k >= 0; --k) {
            float ek    = __half2float(smem_err[k][lane]);
            float xk    = __half2float(smem_xt[k][lane]);
            float dot_e = warp_sum_32(ek * v_t);
            st_v        = fmaf(w_adj * dot_e, xk, st_v);
            w_adj      *= gamma;
        }
        gx_tilde_sfast_out[off] = st_v + gst_e;
    }

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        grad_S_out[int64_t(bh) * 1024 + r * 32 + lane] = gS_col[r];
    }
}

// ============================================================================
// Kernel 5: Stage 3 Exact 3-Line Non-Abelian SU(2) gU Backward (W10-OBS-58)
//   - Zero-transpose strided [B, T, D] support (stride_t = D -> 128 B coalesced!)
// ============================================================================

__device__ __forceinline__ float3 cayley_pullback_w(float3 w, float4 gq) {
    float w2      = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv     = 1.0f / (1.0f + w2);
    float dot_wv  = fmaf(w.x, gq.y, fmaf(w.y, gq.z, w.z * gq.w));
    float coeff_w = -4.0f * inv * inv * (gq.x + dot_wv);
    return make_float3(
        fmaf(2.0f * inv, gq.y, coeff_w * w.x),
        fmaf(2.0f * inv, gq.z, coeff_w * w.y),
        fmaf(2.0f * inv, gq.w, coeff_w * w.z)
    );
}

__global__ __launch_bounds__(32, 8)
void s117_su2_3line_gu_bwd_sm86(
    const float* __restrict__ x_chunk,          // [B, T, D] chunk slice (stride_b, stride_t)
    const float* __restrict__ gx_tilde_total,   // [B, T, D] = gx_tilde_sfast + grad_z_slow @ W_q
    const float* __restrict__ gx_geo_in,        // [B, T, D] from Stage 1
    const float3* __restrict__ w_su2_chunk,     // [B, T, H, 8]
    const float4* __restrict__ U_ckpt_end,      // [BH, 8] chunk-exit U_32
    const float4* __restrict__ grad_U_in,       // [BH, 8] incoming gU from chunk c+1
    float* __restrict__ grad_x_out,             // [B, T, D]
    float3* __restrict__ grad_w_su2_out,        // [B, T, H, 8]
    float4* __restrict__ grad_U_out,            // [BH, 8] outgoing gU to chunk c-1
    int project_boundary_tangent,
    int num_heads,
    int stride_b,
    int stride_t,
    int total_bh
) {
    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    float4 U_post = U_ckpt_end[bh * 8 + spin_id];
    float4 gU     = grad_U_in[bh * 8 + spin_id];

    #pragma unroll 1
    for (int t = 31; t >= 0; --t) {
        int64_t t_base = bh_base + int64_t(t) * int64_t(stride_t);
        int64_t off    = t_base + lane;
        int64_t w_off  = (t_base >> 2) + spin_id;
        float   x_i    = x_chunk[off];
        float   gxt_i  = gx_tilde_total[off];
        float   gxg_i  = gx_geo_in[off];
        float3  w_s    = w_su2_chunk[w_off];
        float3  neg_w  = make_float3(-w_s.x, -w_s.y, -w_s.z);
        float4  q_inv  = cayley_su2(neg_w);
        float4  U_prev = quat_nr_polish(quat_mul(q_inv, U_post));

        float4 xs   = gather_spinor4(x_i, base);
        float4 gxt4 = gather_spinor4(gxt_i, base);

        float4 gx_rot4  = quat_mul(U_post, gxt4);
        grad_x_out[off] = gxg_i + select_lane4(gx_rot4, lmod4);

        // Exact 3-line non-Abelian SU(2) adjoint recurrence (W10-OBS-58):
        float4 dU_local = quat_mul(xs, quat_conj(gxt4));
        gU              = make_float4(gU.x + dU_local.x, gU.y + dU_local.y, gU.z + dU_local.z, gU.w + dU_local.w);
        float4 gq       = quat_mul(gU, quat_conj(U_prev));
        gU              = quat_mul(q_inv, gU);

        float3 gw = cayley_pullback_w(w_s, gq);
        if (lmod4 == 0) {
            grad_w_su2_out[w_off] = gw;
        }
        U_post = U_prev;
    }

    if (project_boundary_tangent) {
        float radial = quat_dot(gU, U_post);
        gU = make_float4(
            fmaf(-radial, U_post.x, gU.x),
            fmaf(-radial, U_post.y, gU.y),
            fmaf(-radial, U_post.z, gU.z),
            fmaf(-radial, U_post.w, gU.w)
        );
    }
    if (lmod4 == 0) {
        grad_U_out[bh * 8 + spin_id] = gU;
    }
}

// ============================================================================
// Kernel 6: 8-Branch Speculative Woodbury Commit (W10-OBS-46/59, OBS-ORDA-21/22)
//   - 6,400 B SMEM Woodbury arena: x_tilde_k (int8, 2 KB) + e_k (fp16, 4 KB) + scales (256 B)
//   - Exact FP32 S_prefix_col[32] in registers (0 B stack, 0 B spills!)
//   - Strictly causal multi-step Woodbury s_pred (k < tau) matching token-by-token execution
//   - Exact 1 <= win_len <= 8 SU(2) U_prefix replay commit (0 B extra SMEM, normal .ca/.wb!)
// ============================================================================

__global__ __launch_bounds__(32, 4)
void s117_spec8_woodbury_commit_sm86(
    float* __restrict__ S_prefix_fp32_io,     // [H, 32, 32] exact FP32 committed S_prefix
    float4* __restrict__ U_ckpt_io,           // [H, 8]      L2-pinned committed U_holonomy (.ca/.wb)
    const int8_t* __restrict__ x_tilde_8x8,   // [H, 8, 8, 32] speculative branch x_tilde (int8)
    const float* __restrict__ x_scale_8x8,    // [H, 8, 8]     per-(branch, step) S^31 absmax/127
    const __half* __restrict__ e_k_8x8,       // [H, 8, 8, 32] speculative branch err vectors (fp16)
    const float3* __restrict__ w_branch_8x8,  // [H, 8, 8, 8]  branch Cayley omega for exact win_len replay
    float* __restrict__ causal_spred_out,     // [H, 32]       causal s_pred at step tau = win_len - 1
    float gamma,
    float eta,
    int b_win,                                // winning branch in 0..7
    int win_len,                              // accepted tactic length in 1..8
    int total_heads
) {
    // 6,400 B SMEM Woodbury arena (> 93 KB free on sm_86!)
    __shared__ int8_t smem_xt[8][8][32];
    __shared__ __half smem_ek[8][8][32];
    __shared__ float  smem_sc[8][8];

    int head_id = blockIdx.x;
    int lane    = threadIdx.x & 31;
    if (head_id >= total_heads) return;

    // 1. Stage 8x8 Woodbury delta list into 6 KB SMEM
    for (int b = 0; b < 8; ++b) {
        for (int k = 0; k < 8; ++k) {
            int off = ((head_id * 8 + b) * 8 + k) * 32 + lane;
            smem_xt[b][k][lane] = x_tilde_8x8[off];
            smem_ek[b][k][lane] = e_k_8x8[off];
        }
        if (lane < 8) {
            smem_sc[b][lane] = x_scale_8x8[(head_id * 8 + b) * 8 + lane];
        }
    }
    __syncwarp();

    // 2. Load exact FP32 S_prefix column into 32 registers/lane (0 B stack!)
    float S_col[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        S_col[r] = S_prefix_fp32_io[head_id * 1024 + r * 32 + lane];
    }

    int wl    = (win_len < 1) ? 1 : ((win_len > 8) ? 8 : win_len);
    int b_sel = b_win & 7;
    int tau   = wl - 1; // 0..7

    // 3. Strictly causal multi-step Woodbury s_pred at step tau (sums k = 0 .. tau - 1 ONLY!)
    //    s_pred(tau) = gamma^tau * (S_prefix * x_tau) + sum_{k=0}^{tau-1} (gamma^{tau-1-k} * eta * <x_k, x_tau>) * e_k
    float x_tau     = float(smem_xt[b_sel][tau][lane]) * smem_sc[b_sel][tau];
    float base_pred = 0.0f;
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        float pr = warp_sum_32(S_col[r] * x_tau);
        if (lane == r) base_pred = pr;
    }

    // Zero-stack reverse scalar power accumulation (0 local memory / 0 LDL / 0 STL!)
    float g_tau = 1.0f;
    for (int k = 0; k < tau; ++k) g_tau *= gamma;
    float s_pred_causal = g_tau * base_pred;

    float w_decay = eta;
    for (int k = tau - 1; k >= 0; --k) {
        float xk      = float(smem_xt[b_sel][k][lane]) * smem_sc[b_sel][k];
        float dot     = warp_sum_32(xk * x_tau);
        float ek      = __half2float(smem_ek[b_sel][k][lane]);
        s_pred_causal = fmaf(w_decay * dot, ek, s_pred_causal);
        w_decay      *= gamma;
    }
    causal_spred_out[head_id * 32 + lane] = s_pred_causal;

    // 4. Exact rank-wl commit into FP32 S_prefix (reverse sweep avoids array indexing -> 0 B stack!):
    //    S_prefix <- gamma^wl * S_prefix + sum_{k=0}^{wl-1} (gamma^{wl-1-k} * eta) * e_k * x_k^T
    float g_wl = g_tau * gamma;
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        S_col[r] *= g_wl;
    }
    float w_commit = eta;
    for (int k = wl - 1; k >= 0; --k) {
        float xk = float(smem_xt[b_sel][k][lane]) * smem_sc[b_sel][k];
        float wx = w_commit * xk;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            float er  = __half2float(smem_ek[b_sel][k][r]);
            S_col[r]  = fmaf(er, wx, S_col[r]);
        }
        w_commit *= gamma;
    }

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        S_prefix_fp32_io[head_id * 1024 + r * 32 + lane] = S_col[r];
    }

    // 5. Exact 1 <= win_len <= 8 SU(2) holonomy replay directly on L2-pinned U_ckpt_io (normal .ca/.wb!)
    if ((lane & 3) == 0) {
        int    spin_id = lane >> 2;
        float4 U_cur   = U_ckpt_io[head_id * 8 + spin_id];
        for (int k = 0; k < wl; ++k) {
            float3 w_k = w_branch_8x8[((head_id * 8 + b_sel) * 8 + k) * 8 + spin_id];
            U_cur      = quat_nr_polish(quat_mul(cayley_su2(w_k), U_cur));
        }
        U_ckpt_io[head_id * 8 + spin_id] = U_cur;
    }
}

// ============================================================================
// Kernel 7: 2-Op IMMA.16832 S_prefix int8 Shadow-Tile + 4-Shfl Gather + S^31 Norm
//   (OBS-ORDA-21 / OBS-ORDA-23 / W10-OBS-59)
//   - Row-wise quantizes FP32 S_prefix (32x32) once into int8 shadow tile + row scales
//   - Runs 2x mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (rows 0..15 & 16..31)
//   - Demonstrates exact 4-shuffle gather_z_slow_4shfl + mask {4,8,16} S^31 branch norm
// ============================================================================

__global__ __launch_bounds__(32, 8)
void s117_sprefix_int8_shadow_imma_sm86(
    const float* __restrict__ S_prefix_fp32,    // [H, 32, 32]
    const int8_t* __restrict__ X_branches_s8,   // [H, 8, 32] col-major per branch [branch, k]
    const float* __restrict__ X_branch_scale,   // [H, 8]
    int8_t* __restrict__ S_shadow_s8_out,       // [H, 32, 32]
    float* __restrict__ S_row_scale_out,        // [H, 32]
    float* __restrict__ Y_sprefix_out,          // [H, 8, 32] dequantized S_prefix @ X_branches
    float* __restrict__ Y_s31_norm_out,         // [H, 8, 32] mask {4,8,16} S^31 normalized branches
    int total_heads
) {
    __shared__ int8_t smem_Sq[32][32];  // 1,024 B
    __shared__ float  smem_Sr[32];      //   128 B
    __shared__ int8_t smem_Xb[8][32];   //   256 B
    __shared__ float  smem_Xs[8];       //    32 B

    int head_id = blockIdx.x;
    int lane    = threadIdx.x & 31;
    if (head_id >= total_heads) return;

    // 1. Row-wise int8 shadow quantization of S_prefix (1,024 elems -> 32 warp rows)
    for (int r = 0; r < 32; ++r) {
        float val    = S_prefix_fp32[head_id * 1024 + r * 32 + lane];
        float absmax = warp_max_abs_32(val);
        float scale  = (absmax > 1e-12f) ? (absmax * (1.0f / 127.0f)) : (1.0f / 127.0f);
        float inv_sc = (absmax > 1e-12f) ? (127.0f / absmax) : 0.0f;
        int8_t q     = quant_f32_to_s8(val * inv_sc);
        smem_Sq[r][lane] = q;
        S_shadow_s8_out[head_id * 1024 + r * 32 + lane] = q;
        if (lane == 0) {
            smem_Sr[r] = scale;
            S_row_scale_out[head_id * 32 + r] = scale;
        }
    }
    for (int b = 0; b < 8; ++b) {
        smem_Xb[b][lane] = X_branches_s8[(head_id * 8 + b) * 32 + lane];
    }
    if (lane < 8) smem_Xs[lane] = X_branch_scale[head_id * 8 + lane];
    __syncwarp();

    // 2. Pack Operand B (shared across both 16x8x32 tiles!) & run 2x IMMA.16832
    int group_id = lane >> 2; // 0..7
    int tid      = lane & 3;  // 0..3
    int k0       = tid * 4;
    int k1       = k0 + 16;

    uint32_t b0 = pack4_s8(smem_Xb[group_id][k0 + 0], smem_Xb[group_id][k0 + 1], smem_Xb[group_id][k0 + 2], smem_Xb[group_id][k0 + 3]);
    uint32_t b1 = pack4_s8(smem_Xb[group_id][k1 + 0], smem_Xb[group_id][k1 + 1], smem_Xb[group_id][k1 + 2], smem_Xb[group_id][k1 + 3]);

    // Tile 0: rows 0..15
    uint32_t a0_t0 = pack4_s8(smem_Sq[group_id + 0][k0 + 0], smem_Sq[group_id + 0][k0 + 1], smem_Sq[group_id + 0][k0 + 2], smem_Sq[group_id + 0][k0 + 3]);
    uint32_t a1_t0 = pack4_s8(smem_Sq[group_id + 8][k0 + 0], smem_Sq[group_id + 8][k0 + 1], smem_Sq[group_id + 8][k0 + 2], smem_Sq[group_id + 8][k0 + 3]);
    uint32_t a2_t0 = pack4_s8(smem_Sq[group_id + 0][k1 + 0], smem_Sq[group_id + 0][k1 + 1], smem_Sq[group_id + 0][k1 + 2], smem_Sq[group_id + 0][k1 + 3]);
    uint32_t a3_t0 = pack4_s8(smem_Sq[group_id + 8][k1 + 0], smem_Sq[group_id + 8][k1 + 1], smem_Sq[group_id + 8][k1 + 2], smem_Sq[group_id + 8][k1 + 3]);
    int32_t t0_d0, t0_d1, t0_d2, t0_d3;
    imma_m16n8k32_s8_zero(t0_d0, t0_d1, t0_d2, t0_d3, a0_t0, a1_t0, a2_t0, a3_t0, b0, b1);

    // Tile 1: rows 16..31
    uint32_t a0_t1 = pack4_s8(smem_Sq[group_id + 16][k0 + 0], smem_Sq[group_id + 16][k0 + 1], smem_Sq[group_id + 16][k0 + 2], smem_Sq[group_id + 16][k0 + 3]);
    uint32_t a1_t1 = pack4_s8(smem_Sq[group_id + 24][k0 + 0], smem_Sq[group_id + 24][k0 + 1], smem_Sq[group_id + 24][k0 + 2], smem_Sq[group_id + 24][k0 + 3]);
    uint32_t a2_t1 = pack4_s8(smem_Sq[group_id + 16][k1 + 0], smem_Sq[group_id + 16][k1 + 1], smem_Sq[group_id + 16][k1 + 2], smem_Sq[group_id + 16][k1 + 3]);
    uint32_t a3_t1 = pack4_s8(smem_Sq[group_id + 24][k1 + 0], smem_Sq[group_id + 24][k1 + 1], smem_Sq[group_id + 24][k1 + 2], smem_Sq[group_id + 24][k1 + 3]);
    int32_t t1_d0, t1_d1, t1_d2, t1_d3;
    imma_m16n8k32_s8_zero(t1_d0, t1_d1, t1_d2, t1_d3, a0_t1, a1_t1, a2_t1, a3_t1, b0, b1);

    // 3. Rescale even (2*tid) & odd (2*tid+1) branch accumulators into FP32
    float sr0 = smem_Sr[group_id +  0];
    float sr1 = smem_Sr[group_id +  8];
    float sr2 = smem_Sr[group_id + 16];
    float sr3 = smem_Sr[group_id + 24];
    float xs_e = smem_Xs[2 * tid + 0];
    float xs_o = smem_Xs[2 * tid + 1];

    float ev0 = float(t0_d0) * sr0 * xs_e;
    float ev1 = float(t0_d2) * sr1 * xs_e;
    float ev2 = float(t1_d0) * sr2 * xs_e;
    float ev3 = float(t1_d2) * sr3 * xs_e;

    float od0 = float(t0_d1) * sr0 * xs_o;
    float od1 = float(t0_d3) * sr1 * xs_o;
    float od2 = float(t1_d1) * sr2 * xs_o;
    float od3 = float(t1_d3) * sr3 * xs_o;

    // 4. Emit lane-aligned dequantized outputs via exact OBS-ORDA-21 4-shuffle gather
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        float y_b = ((b & 1) == 0) ? gather_z_slow_4shfl(ev0, ev1, ev2, ev3, lane, b)
                                   : gather_z_slow_4shfl(od0, od1, od2, od3, lane, b);
        Y_sprefix_out[(head_id * 8 + b) * 32 + lane] = y_b;
    }

    // 5. Transpose-free mask {4, 8, 16} S^31 branch normalization (0 cross-branch bleed!)
    float inv_e = rsqrtf(branch_sum_mask4_8_16(fmaf(ev0, ev0, fmaf(ev1, ev1, fmaf(ev2, ev2, ev3 * ev3)))) + 1e-12f);
    float inv_o = rsqrtf(branch_sum_mask4_8_16(fmaf(od0, od0, fmaf(od1, od1, fmaf(od2, od2, od3 * od3)))) + 1e-12f);

    float nev0 = ev0 * inv_e, nev1 = ev1 * inv_e, nev2 = ev2 * inv_e, nev3 = ev3 * inv_e;
    float nod0 = od0 * inv_o, nod1 = od1 * inv_o, nod2 = od2 * inv_o, nod3 = od3 * inv_o;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        float yn = ((b & 1) == 0) ? gather_z_slow_4shfl(nev0, nev1, nev2, nev3, lane, b)
                                  : gather_z_slow_4shfl(nod0, nod1, nod2, nod3, lane, b);
        Y_s31_norm_out[(head_id * 8 + b) * 32 + lane] = yn;
    }
}

// ============================================================================
// Kernel 8: Fused 4-IMMA Spec8 Kernel (W_slow + S_prefix int8 shadow-tile)
//   Answers silikernels#2#issuecomment-7179169347:
//   - Exact 1,056 B SMEM staging (int8_t smem_tile[32][33], +1 padded, 0 bank conflicts!)
//   - int zero = 0 bound via "r"(zero) in all 4 mma.sync.aligned.m16n8k32 ops
//   - Shared Operand B (b0, b1) across all 4 IMMA ops (2x W_slow + 2x S_prefix shadow)
//   - Receiver-indexed 4-shuffle gather_z_slow_4shfl + <= 64 regs @ 0 B spills!
// ============================================================================

__global__ __launch_bounds__(32, 8)
void s117_spec8_fused_4imma_sm86(
    const int8_t* __restrict__ W_slow_s8,       // [H, 32, 32] L2-pinned slow trunk (int8)
    const float* __restrict__ W_row_scale,      // [H, 32]
    const int8_t* __restrict__ S_shadow_s8,     // [H, 32, 32] int8 shadow S_prefix
    const float* __restrict__ S_row_scale,      // [H, 32]
    const int8_t* __restrict__ X_branches_s8,   // [H, 8, 32] 8 parallel speculative branches
    const float* __restrict__ X_branch_scale,   // [H, 8]
    float* __restrict__ Z_fused_out,            // [H, 8, 32] lane-aligned W_slow*X + gamma_tau*S_prefix*X
    float gamma_tau,
    int total_heads
) {
    // Exact 1,056 B SMEM staging (32 x 33 int8 = 1,056 B, reused across X -> W_slow -> S_prefix!)
    __shared__ int8_t smem_tile[32][33];

    int head_id = blockIdx.x;
    int lane    = threadIdx.x & 31;
    if (head_id >= total_heads) return;

    int group_id = lane >> 2; // 0..7
    int tid      = lane & 3;  // 0..3
    int k0       = tid * 4;
    int k1       = k0 + 16;

    // 1. Stage 8 branches into rows 0..7 of smem_tile and pack shared Operand B (b0, b1)
    #pragma unroll 1
    for (int b = 0; b < 8; ++b) {
        smem_tile[b][lane] = X_branches_s8[(head_id * 8 + b) * 32 + lane];
    }
    __syncwarp();
    uint32_t b0 = pack4_s8(smem_tile[group_id][k0 + 0], smem_tile[group_id][k0 + 1], smem_tile[group_id][k0 + 2], smem_tile[group_id][k0 + 3]);
    uint32_t b1 = pack4_s8(smem_tile[group_id][k1 + 0], smem_tile[group_id][k1 + 1], smem_tile[group_id][k1 + 2], smem_tile[group_id][k1 + 3]);
    float xs_e = X_branch_scale[head_id * 8 + (2 * tid + 0)];
    float xs_o = X_branch_scale[head_id * 8 + (2 * tid + 1)];

    // 2. Stage W_slow (32x32) into the SAME 1,056 B SMEM -> 2x IMMA.16832 (zero-bound C!)
    #pragma unroll 1
    for (int r = 0; r < 32; ++r) {
        smem_tile[r][lane] = W_slow_s8[head_id * 1024 + r * 32 + lane];
    }
    __syncwarp();
    float w_sc = W_row_scale[head_id * 32 + lane];
    float ws0  = __shfl_sync(0xFFFFFFFFu, w_sc, group_id +  0);
    float ws1  = __shfl_sync(0xFFFFFFFFu, w_sc, group_id +  8);
    float ws2  = __shfl_sync(0xFFFFFFFFu, w_sc, group_id + 16);
    float ws3  = __shfl_sync(0xFFFFFFFFu, w_sc, group_id + 24);

    int32_t d0, d1, d2, d3;
    imma_m16n8k32_s8_zero(
        d0, d1, d2, d3,
        pack4_s8(smem_tile[group_id + 0][k0 + 0], smem_tile[group_id + 0][k0 + 1], smem_tile[group_id + 0][k0 + 2], smem_tile[group_id + 0][k0 + 3]),
        pack4_s8(smem_tile[group_id + 8][k0 + 0], smem_tile[group_id + 8][k0 + 1], smem_tile[group_id + 8][k0 + 2], smem_tile[group_id + 8][k0 + 3]),
        pack4_s8(smem_tile[group_id + 0][k1 + 0], smem_tile[group_id + 0][k1 + 1], smem_tile[group_id + 0][k1 + 2], smem_tile[group_id + 0][k1 + 3]),
        pack4_s8(smem_tile[group_id + 8][k1 + 0], smem_tile[group_id + 8][k1 + 1], smem_tile[group_id + 8][k1 + 2], smem_tile[group_id + 8][k1 + 3]),
        b0, b1
    );
    float ev0 = float(d0) * ws0 * xs_e;
    float ev1 = float(d2) * ws1 * xs_e;
    float od0 = float(d1) * ws0 * xs_o;
    float od1 = float(d3) * ws1 * xs_o;

    imma_m16n8k32_s8_zero(
        d0, d1, d2, d3,
        pack4_s8(smem_tile[group_id + 16][k0 + 0], smem_tile[group_id + 16][k0 + 1], smem_tile[group_id + 16][k0 + 2], smem_tile[group_id + 16][k0 + 3]),
        pack4_s8(smem_tile[group_id + 24][k0 + 0], smem_tile[group_id + 24][k0 + 1], smem_tile[group_id + 24][k0 + 2], smem_tile[group_id + 24][k0 + 3]),
        pack4_s8(smem_tile[group_id + 16][k1 + 0], smem_tile[group_id + 16][k1 + 1], smem_tile[group_id + 16][k1 + 2], smem_tile[group_id + 16][k1 + 3]),
        pack4_s8(smem_tile[group_id + 24][k1 + 0], smem_tile[group_id + 24][k1 + 1], smem_tile[group_id + 24][k1 + 2], smem_tile[group_id + 24][k1 + 3]),
        b0, b1
    );
    float ev2 = float(d0) * ws2 * xs_e;
    float ev3 = float(d2) * ws3 * xs_e;
    float od2 = float(d1) * ws2 * xs_o;
    float od3 = float(d3) * ws3 * xs_o;

    // 3. Reuse the SAME 1,056 B SMEM for S_prefix int8 shadow-tile -> 2x IMMA.16832 (fused FMA!)
    #pragma unroll 1
    for (int r = 0; r < 32; ++r) {
        smem_tile[r][lane] = S_shadow_s8[head_id * 1024 + r * 32 + lane];
    }
    __syncwarp();
    float s_sc = S_row_scale[head_id * 32 + lane] * gamma_tau;
    float ss0  = __shfl_sync(0xFFFFFFFFu, s_sc, group_id +  0);
    float ss1  = __shfl_sync(0xFFFFFFFFu, s_sc, group_id +  8);
    float ss2  = __shfl_sync(0xFFFFFFFFu, s_sc, group_id + 16);
    float ss3  = __shfl_sync(0xFFFFFFFFu, s_sc, group_id + 24);

    imma_m16n8k32_s8_zero(
        d0, d1, d2, d3,
        pack4_s8(smem_tile[group_id + 0][k0 + 0], smem_tile[group_id + 0][k0 + 1], smem_tile[group_id + 0][k0 + 2], smem_tile[group_id + 0][k0 + 3]),
        pack4_s8(smem_tile[group_id + 8][k0 + 0], smem_tile[group_id + 8][k0 + 1], smem_tile[group_id + 8][k0 + 2], smem_tile[group_id + 8][k0 + 3]),
        pack4_s8(smem_tile[group_id + 0][k1 + 0], smem_tile[group_id + 0][k1 + 1], smem_tile[group_id + 0][k1 + 2], smem_tile[group_id + 0][k1 + 3]),
        pack4_s8(smem_tile[group_id + 8][k1 + 0], smem_tile[group_id + 8][k1 + 1], smem_tile[group_id + 8][k1 + 2], smem_tile[group_id + 8][k1 + 3]),
        b0, b1
    );
    ev0 = fmaf(float(d0), ss0 * xs_e, ev0);
    ev1 = fmaf(float(d2), ss1 * xs_e, ev1);
    od0 = fmaf(float(d1), ss0 * xs_o, od0);
    od1 = fmaf(float(d3), ss1 * xs_o, od1);

    imma_m16n8k32_s8_zero(
        d0, d1, d2, d3,
        pack4_s8(smem_tile[group_id + 16][k0 + 0], smem_tile[group_id + 16][k0 + 1], smem_tile[group_id + 16][k0 + 2], smem_tile[group_id + 16][k0 + 3]),
        pack4_s8(smem_tile[group_id + 24][k0 + 0], smem_tile[group_id + 24][k0 + 1], smem_tile[group_id + 24][k0 + 2], smem_tile[group_id + 24][k0 + 3]),
        pack4_s8(smem_tile[group_id + 16][k1 + 0], smem_tile[group_id + 16][k1 + 1], smem_tile[group_id + 16][k1 + 2], smem_tile[group_id + 16][k1 + 3]),
        pack4_s8(smem_tile[group_id + 24][k1 + 0], smem_tile[group_id + 24][k1 + 1], smem_tile[group_id + 24][k1 + 2], smem_tile[group_id + 24][k1 + 3]),
        b0, b1
    );
    ev2 = fmaf(float(d0), ss2 * xs_e, ev2);
    ev3 = fmaf(float(d2), ss3 * xs_e, ev3);
    od2 = fmaf(float(d1), ss2 * xs_o, od2);
    od3 = fmaf(float(d3), ss3 * xs_o, od3);

    // 4. Receiver-indexed 4-shuffle gather into lane-aligned output [H, 8, 32]
    #pragma unroll 1
    for (int b = 0; b < 8; ++b) {
        float zb = ((b & 1) == 0) ? gather_z_slow_4shfl(ev0, ev1, ev2, ev3, lane, b)
                                  : gather_z_slow_4shfl(od0, od1, od2, od3, lane, b);
        Z_fused_out[(head_id * 8 + b) * 32 + lane] = zb;
    }
}

// ============================================================================
// [16, 11, 4] Extended Hamming A_16 LSB SECDED + SiliSense .sys BER Canary
//   - 5-bit SECDED syndrome: bit 0 = overall parity, bits 1..4 = error bit pos j
//   - Single-bit error: (syndrome & 1u) != 0 -> inline LSB fix at j = (syndrome >> 1) & 0xF
//   - Double-bit error: syndrome != 0 && (syndrome & 1u) == 0 -> DED flagged to canary!
//   - Strictly .sys scope (atom.release.sys.global.add.u32) for cudaHostAllocMapped visibility
//   - Guarded by __ballot_sync(0xFFFFFFFFu, syndrome != 0u): ZERO PCIe traffic when clean!
// ============================================================================

__device__ __forceinline__ uint32_t a16_secded_syndrome(uint32_t w16) {
    uint32_t p0 = uint32_t(__popc(w16)) & 1u;
    uint32_t p1 = uint32_t(__popc(w16 & 0xAAAAu)) & 1u;
    uint32_t p2 = uint32_t(__popc(w16 & 0xCCCCu)) & 1u;
    uint32_t p3 = uint32_t(__popc(w16 & 0xF0F0u)) & 1u;
    uint32_t p4 = uint32_t(__popc(w16 & 0xFF00u)) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

__device__ __forceinline__ void silisense_ber_canary_add_sys(uint32_t* host_mapped_ber_ptr, uint32_t delta) {
    asm volatile(
        "atom.release.sys.global.add.u32 _, [%0], %1;"
        :
        : "l"(host_mapped_ber_ptr), "r"(delta)
        : "memory"
    );
}

// ============================================================================
// Kernel 9: Single-CTA 16-Warp d_model=512 Fused CellOp (OBS-ORDA-27)
//   - 16 warps = 512 threads in ONE CTA (0 cross-CTA .acq_rel.gpu L2 barriers!)
//   - <= 75 regs/thread (38,400 <= 65,536 regs/SM), 8,912 B SMEM, 0 B spills
//   - Fuses: H_256 x bent4 prologue -> 16-head smem_x_s8[16][36] cross-head
//            exchange + Cayley SU(2) + inline [16,11,4] SECDED & SiliSense 4 B
//            .sys BER canary (0 PCIe ops when clean!) + Hard-FMNMX 1-in-8 KV-ring
//            + Cayley S^31 -> 23-bit finite Gumbel + 8-warp FMNMX epilogue!
// ============================================================================

__global__ __launch_bounds__(512, 1)
void s117_uniform_cell_16warp_d512_sm86(
    const uint8_t* __restrict__ byte_in,          // [B]
    const int8_t* __restrict__ W_layer_s8,        // [L, 16, 32, 32] per-layer per-head int8 trunk
    const float* __restrict__ W_layer_scale,      // [L, 16, 32]
    const float3* __restrict__ w_su2_layer,       // [L, 16, 8]
    const float* __restrict__ kv_keys_init,       // [B, 16, 8, 32] 1-in-8 KV-ring keys
    const __half* __restrict__ kv_vals_init,      // [B, 16, 8, 32] 1-in-8 KV-ring values
    const float* __restrict__ lm_head_w,          // [256, 512] byte output projection
    const uint32_t* __restrict__ rng_u32,         // [B, 256]
    float4* __restrict__ U_state_io,              // [B, 16, 8] persistent SU(2) holonomy
    float* __restrict__ hidden_out,               // [B, 512] final S^31x16 state
    uint8_t* __restrict__ out_byte,               // [B]
    float* __restrict__ out_score,                // [B]
    uint32_t* __restrict__ host_ber_canary,       // 4 B cudaHostAllocMapped SiliSense BER counter (NULL-safe)
    float cayley_t,
    float temperature,
    int n_layers,
    int batch_size
) {
    // 9,552 B total SMEM (double-buffered smem_x_s8/scale -> 0 cross-warp WAR races, 0 cross-CTA barriers!)
    __shared__ int8_t  smem_x_s8[2][16][36];     // 1152 B: double-buffered 16-head exchange (+4 padded)
    __shared__ float   smem_x_scale[2][16];      //  128 B: double-buffered per-head S^31 absmax/127
    __shared__ __half  smem_kv_val[16][8][32];   // 8192 B: 16-head 1-in-8 KV-ring values
    __shared__ float   smem_ep_val[16];          //   64 B: epilogue 8-warp tournament scores
    __shared__ uint8_t smem_ep_idx[16];          //   16 B: epilogue winning byte indices

    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    int tid     = threadIdx.x; // 0..511
    int warp_id = tid >> 5;    // 0..15 (16 S^31 heads = d_model 512!)
    int lane    = tid & 31;    // 0..31
    int spin_id = lane >> 2;   // 0..7
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    // 1. Prologue: Zero-VRAM H_256 x bent4 byte injection across all 16 warps (d=512)
    uint8_t tok_b = byte_in[b_idx];
    int8_t  inj_q = h256_bent4_byte_inject_s8(tok_b, warp_id & 7, lane);
    if (warp_id >= 8) inj_q = int8_t(-inj_q); // Orthogonal second 256-dim half
    const float inv_sqrt32 = 0.176776695f;
    float x_i = (inj_q > 0) ? inv_sqrt32 : -inv_sqrt32;

    smem_x_s8[0][warp_id][lane] = inj_q;
    if (lane == 0) smem_x_scale[0][warp_id] = inv_sqrt32 * (1.0f / 127.0f);

    // Stage 8-slot KV-ring values into SMEM once per token
    #pragma unroll 1
    for (int slot = 0; slot < 8; ++slot) {
        smem_kv_val[warp_id][slot][lane] = kv_vals_init[((b_idx * 16 + warp_id) * 8 + slot) * 32 + lane];
    }
    __syncthreads();

    float4 U_cur = U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id];
    float  t2    = cayley_t * cayley_t;
    float  inv_t = 1.0f / (1.0f + t2);
    float  c_st  = (1.0f - t2) * inv_t;
    float  s_st  = (2.0f * cayley_t) * inv_t;

    // 2. Fused L-layer single-CTA CellOp loop (0 cross-CTA L2 barriers, 0 WAR races!)
    #pragma unroll 1
    for (int layer = 0; layer < n_layers; ++layer) {
        int cur_buf = layer & 1;
        int nxt_buf = cur_buf ^ 1;
        // A. Cayley SU(2) holonomy rotation on x_i -> x_tilde
        float3 w_s = w_su2_layer[(layer * 16 + warp_id) * 8 + spin_id];
        U_cur      = quat_nr_polish(quat_mul(cayley_su2(w_s), U_cur));
        float4 xs  = gather_spinor4(x_i, base);
        float  xt  = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);

        // B. Cross-head partner read via double-buffered smem_x_s8[cur_buf][partner]
        int   partner   = warp_id ^ (1 << (layer & 3));
        float x_partner = float(smem_x_s8[cur_buf][partner][lane]) * smem_x_scale[cur_buf][partner];
        float x_mix     = 0.75f * xt + 0.25f * x_partner;

        // C. Per-head int8 trunk dot product + inline [16, 11, 4] LSB SECDED BER Canary
        //    (ZERO PCIe atomic traffic when syndrome == 0u; .sys release atomic on fault!)
        float z_slow = 0.0f;
        const int8_t* w_base = W_layer_s8 + (layer * 16 + warp_id) * 1024;
        float w_sc = W_layer_scale[(layer * 16 + warp_id) * 32 + lane];
        uint32_t warp_ber_faults = 0u;
        #pragma unroll 1
        for (int r = 0; r < 32; ++r) {
            int8_t   w_raw  = w_base[r * 32 + lane];
            uint32_t lsb32  = __ballot_sync(0xFFFFFFFFu, (int(w_raw) & 1) != 0);
            uint32_t w16    = (lane < 16) ? (lsb32 & 0xFFFFu) : (lsb32 >> 16);
            uint32_t synd   = a16_secded_syndrome(w16);
            // Inline 1-bit LSB SECDED correction: j = (synd >> 1) & 0xF when (synd & 1) == 1
            if ((synd & 1u) != 0u) {
                uint32_t j = (synd >> 1) & 0xFu;
                if (uint32_t(lane & 15) == j) {
                    w_raw = int8_t(int(w_raw) ^ 1);
                }
            }
            if (synd != 0u && ((lane & 15) == 0)) {
                warp_ber_faults += 1u;
            }
            float dot_r = warp_sum_32(float(w_raw) * x_mix);
            if (lane == r) z_slow = dot_r * w_sc;
        }
        uint32_t err_mask = __ballot_sync(0xFFFFFFFFu, warp_ber_faults != 0u);
        if (err_mask != 0u && lane == 0 && host_ber_canary != nullptr) {
            silisense_ber_canary_add_sys(host_ber_canary, uint32_t(__popc(err_mask)));
        }

        // D. Hard FMNMX 1-in-8 KV-ring winner-take-all retrieval (0 Softmax!)
        float best_sc   = -3.402823466e38f;
        int   best_slot = 0;
        #pragma unroll 1
        for (int slot = 0; slot < 8; ++slot) {
            float k_val = kv_keys_init[((b_idx * 16 + warp_id) * 8 + slot) * 32 + lane];
            float sc    = warp_sum_32(k_val * x_mix);
            float next_best = fmaxf(best_sc, sc);
            if (sc > best_sc) best_slot = slot;
            best_sc = next_best;
        }
        float kv_v = __half2float(smem_kv_val[warp_id][best_slot][lane]);

        // E. Cayley S^31 geodesic update + S^31 re-quant into smem_x_s8[nxt_buf]
        float z_tot = z_slow + kv_v;
        float v_i   = z_tot - warp_sum_32(z_tot * x_i) * x_i;
        float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + 1e-12f);
        x_i         = c_st * x_i + s_st * (v_i * v_inv);

        float amax   = warp_max_abs_32(x_i);
        float inv_sc = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        smem_x_s8[nxt_buf][warp_id][lane] = quant_f32_to_s8(x_i * inv_sc);
        if (lane == 0) smem_x_scale[nxt_buf][warp_id] = amax * (1.0f / 127.0f);
        __syncthreads();
    }

    // Store final state & persistent SU(2) holonomy
    hidden_out[b_idx * 512 + tid] = x_i;
    if (lmod4 == 0) U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id] = U_cur;

    // 3. Epilogue: 23-bit finite Gumbel + 8-warp FMNMX byte argmax (warps 0..7)
    int final_buf = n_layers & 1;
    if (warp_id < 8) {
        float logit = 0.0f;
        const float* row_w = lm_head_w + tid * 512;
        #pragma unroll 1
        for (int h = 0; h < 16; ++h) {
            float sc_h = smem_x_scale[final_buf][h];
            #pragma unroll 4
            for (int c = 0; c < 32; ++c) {
                logit = fmaf(row_w[h * 32 + c], float(smem_x_s8[final_buf][h][c]) * sc_h, logit);
            }
        }
        float    score = logit + gumbel_23bit_finite(rng_u32[b_idx * 256 + tid], temperature);
        uint32_t idx   = uint32_t(tid);
        warp_fmnmx_argmax_32(score, idx);
        if (lane == 0) {
            smem_ep_val[warp_id] = score;
            smem_ep_idx[warp_id] = uint8_t(idx);
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float    w_val = (lane < 8) ? smem_ep_val[lane] : -3.402823466e38f;
        uint32_t w_idx = (lane < 8) ? uint32_t(smem_ep_idx[lane]) : 256u;
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov = __shfl_xor_sync(0xFFFFFFFFu, w_val, mask);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, w_idx, mask);
            float    bv = fmaxf(w_val, ov);
            bool     take_other = (ov > w_val) || ((ov == w_val) && (oi < w_idx));
            w_val = bv;
            w_idx = take_other ? oi : w_idx;
        }
        if (lane == 0) {
            out_byte[b_idx]  = uint8_t(w_idx);
            out_score[b_idx] = w_val;
        }
    }
}

// 2-op SWAR extraction of the 4 int8 LSBs (bits 0, 8, 16, 24) into bits 0..3
__device__ __forceinline__ uint32_t extract_lsb4_u32(uint32_t w4) {
    return ((w4 & 0x01010101u) * 0x01020408u) >> 24;
}

// Inline 1-bit [16, 11, 4] LSB SECDED check & bit-exact repair on 4 packed int32s (16 int8 weights)
__device__ __forceinline__ uint32_t a16_secded_fix_4words(
    uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3
) {
    uint32_t w16 = extract_lsb4_u32(w0)
                 | (extract_lsb4_u32(w1) << 4)
                 | (extract_lsb4_u32(w2) << 8)
                 | (extract_lsb4_u32(w3) << 12);
    uint32_t synd = a16_secded_syndrome(w16);
    if ((synd & 1u) != 0u) {
        uint32_t j    = (synd >> 1) & 0xFu;
        uint32_t mask = 1u << ((j & 3u) << 3);
        uint32_t widx = j >> 2;
        if (widx == 0u)      w0 ^= mask;
        else if (widx == 1u) w1 ^= mask;
        else if (widx == 2u) w2 ^= mask;
        else                 w3 ^= mask;
    }
    return synd;
}

// ============================================================================
// Kernel 10: Single-CTA 16-Warp d_model=512 V2 Megakernel (__dp4a + 0 B FWHT_256 LM Head!)
//   Eliminates OBS-S117-01 bottlenecks cleanly:
//   - Packed int32 [L, 16, 8, 32] weights -> 8 coalesced LDG.32 + 8 hardware __dp4a/layer
//     (ZERO warp_sum_32 shuffles in trunk matvec; 1 lane = 1 full output row!)
//   - Inline [16, 11, 4] LSB SECDED directly in registers on w0..w3 & w4..w7 (0 ballots in inner loop!)
//     + cold __ballot_sync -> @P0 RED.E.ADD.STRONG.SYS 4 B .sys BER canary (0 PCIe traffic when clean!)
//   - Pre-staged int8 KV-ring keys in SMEM -> parallel 1-in-8 Hard-FMNMX via __dp4a (0 global loads!)
//   - Tied Orthonormal FWHT_256 x bent4 Zero-VRAM LM Head (0 B weight memory reads!
//     5 in-register warp butterflies + 3 cross-warp SMEM butterflies = exact E_byte^T @ hidden!)
//   - Supports BOTH CellOp-only (run_fwht_lm_head=0, apples-to-apples vs S116 68.22 us/tok)
//     AND Full End-to-End (run_fwht_lm_head=1: H_256 inject + 12L CellOp + SECDED + FWHT_256 + Gumbel!)
// ============================================================================

__global__ __launch_bounds__(512, 1)
void s117_uniform_cell_16warp_d512_v2_sm86(
    const uint8_t* __restrict__ byte_in,          // [B]
    const uint32_t* __restrict__ W_packed_u32,    // [L, 16, 8, 32] coalesced 4x int8 packed weights
    const float* __restrict__ W_layer_scale,      // [L, 16, 32]
    const float3* __restrict__ w_su2_layer,       // [L, 16, 8]
    const uint32_t* __restrict__ kv_keys_u32,     // [B, 16, 8, 8] int8 packed 1-in-8 KV-ring keys
    const __half* __restrict__ kv_vals_init,      // [B, 16, 8, 32] 1-in-8 KV-ring values
    const uint32_t* __restrict__ rng_u32,         // [B, 256]
    float4* __restrict__ U_state_io,              // [B, 16, 8] persistent SU(2) holonomy
    float* __restrict__ hidden_out,               // [B, 512] final S^31x16 state
    float* __restrict__ logits_out,               // [B, 256] optional FWHT_256 logits (NULL-safe)
    uint8_t* __restrict__ out_byte,               // [B]
    float* __restrict__ out_score,                // [B]
    uint32_t* __restrict__ host_ber_canary,       // 4 B cudaHostAllocMapped SiliSense BER counter (NULL-safe)
    float cayley_t,
    float temperature,
    int n_layers,
    int run_fwht_lm_head,
    int batch_size
) {
    // 16,040 B total SMEM (double-buffered smem_x_s8/scale -> 0 cross-warp WAR races, 0 cross-CTA barriers!)
    __shared__ int8_t   smem_x_s8[2][16][32];     // 1,024 B: double-buffered cross-head int8 exchange
    __shared__ uint32_t smem_xmix_u32[16][8];     //   512 B: packed 4x int8 x_mix for __dp4a
    __shared__ float    smem_x_scale[2][16];      //   128 B: double-buffered per-head S^31 absmax/127
    __shared__ uint32_t smem_kv_key[16][8][8];    // 4,096 B: pre-staged int8 KV-ring keys
    __shared__ __half   smem_kv_val[16][8][32];   // 8,192 B: pre-staged fp16 KV-ring values
    __shared__ float    smem_fwht[16][32];        // 2,048 B: final hidden + 3-stage cross-warp FWHT_256
    __shared__ float    smem_ep_val[8];           //    32 B: 8-warp tournament scores
    __shared__ uint8_t  smem_ep_idx[8];           //     8 B: 8-warp tournament byte ids

    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    int tid     = threadIdx.x; // 0..511
    int warp_id = tid >> 5;    // 0..15
    int lane    = tid & 31;    // 0..31
    int spin_id = lane >> 2;   // 0..7
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    // 1. Prologue: Zero-VRAM H_256 x bent4 byte injection (d_model = 512)
    uint8_t tok_b = byte_in[b_idx];
    int8_t  inj_q = h256_bent4_byte_inject_s8(tok_b, warp_id & 7, lane);
    if (warp_id >= 8) inj_q = int8_t(-inj_q);
    const float inv_sqrt32 = 0.176776695f;
    float x_i = (inj_q > 0) ? inv_sqrt32 : -inv_sqrt32;

    smem_x_s8[0][warp_id][lane] = inj_q;
    if (lane == 0) smem_x_scale[0][warp_id] = inv_sqrt32 * (1.0f / 127.0f);

    // Stage 8-slot KV-ring keys (64 u32/head) & values (256 f16/head) into SMEM ONCE
    const uint32_t* kv_k_base = kv_keys_u32 + (b_idx * 16 + warp_id) * 64;
    smem_kv_key[warp_id][lane >> 3][lane & 7]            = kv_k_base[lane];
    smem_kv_key[warp_id][(lane + 32) >> 3][(lane + 32) & 7] = kv_k_base[lane + 32];
    #pragma unroll 1
    for (int slot = 0; slot < 8; ++slot) {
        smem_kv_val[warp_id][slot][lane] = kv_vals_init[((b_idx * 16 + warp_id) * 8 + slot) * 32 + lane];
    }
    __syncthreads();

    float4 U_cur = U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id];
    float  t2    = cayley_t * cayley_t;
    float  inv_t = 1.0f / (1.0f + t2);
    float  c_st  = (1.0f - t2) * inv_t;
    float  s_st  = (2.0f * cayley_t) * inv_t;

    // 2. Fused L-Layer Single-CTA __dp4a + SECDED + Parallel Hard-FMNMX Loop (0 WAR races, 0 L2 barriers!)
    #pragma unroll 1
    for (int layer = 0; layer < n_layers; ++layer) {
        int cur_buf = layer & 1;
        int nxt_buf = cur_buf ^ 1;
        // A. Cayley SU(2) holonomy rotation
        float3 w_s = w_su2_layer[(layer * 16 + warp_id) * 8 + spin_id];
        U_cur      = quat_nr_polish(quat_mul(cayley_su2(w_s), U_cur));
        float4 xs  = gather_spinor4(x_i, base);
        float  xt  = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);

        // B. Cross-head butterfly exchange via double-buffered smem_x_s8[cur_buf] + int8 pack
        int   partner   = warp_id ^ (1 << (layer & 3));
        float x_partner = float(smem_x_s8[cur_buf][partner][lane]) * smem_x_scale[cur_buf][partner];
        float x_mix     = fmaf(0.75f, xt, 0.25f * x_partner);

        float x_amax    = warp_max_abs_32(x_mix);
        float x_scale   = x_amax * (1.0f / 127.0f);
        float x_inv_sc  = (x_amax > 1e-12f) ? (127.0f / x_amax) : 0.0f;
        int8_t xq       = quant_f32_to_s8(x_mix * x_inv_sc);
        uint32_t xq_b   = uint32_t(uint8_t(xq));
        uint32_t xq_p0  = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 0);
        uint32_t xq_p1  = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 1);
        uint32_t xq_p2  = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 2);
        uint32_t xq_p3  = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 3);
        if (lmod4 == 0) {
            smem_xmix_u32[warp_id][lane >> 2] = xq_p0 | (xq_p1 << 8) | (xq_p2 << 16) | (xq_p3 << 24);
        }
        __syncwarp();

        uint32_t x0 = smem_xmix_u32[warp_id][0], x1 = smem_xmix_u32[warp_id][1];
        uint32_t x2 = smem_xmix_u32[warp_id][2], x3 = smem_xmix_u32[warp_id][3];
        uint32_t x4 = smem_xmix_u32[warp_id][4], x5 = smem_xmix_u32[warp_id][5];
        uint32_t x6 = smem_xmix_u32[warp_id][6], x7 = smem_xmix_u32[warp_id][7];

        // C. Coalesced 8x LDG.32 + Inline [16,11,4] SECDED (w0..w3, w4..w7) + 8x __dp4a!
        //    1 lane = 1 complete output row (ZERO warp_sum_32 shuffles!)
        const uint32_t* w_head = W_packed_u32 + (layer * 16 + warp_id) * 256;
        uint32_t w0 = __ldg(&w_head[0 * 32 + lane]);
        uint32_t w1 = __ldg(&w_head[1 * 32 + lane]);
        uint32_t w2 = __ldg(&w_head[2 * 32 + lane]);
        uint32_t w3 = __ldg(&w_head[3 * 32 + lane]);
        uint32_t w4 = __ldg(&w_head[4 * 32 + lane]);
        uint32_t w5 = __ldg(&w_head[5 * 32 + lane]);
        uint32_t w6 = __ldg(&w_head[6 * 32 + lane]);
        uint32_t w7 = __ldg(&w_head[7 * 32 + lane]);

        uint32_t synd_lo = a16_secded_fix_4words(w0, w1, w2, w3);
        uint32_t synd_hi = a16_secded_fix_4words(w4, w5, w6, w7);
        uint32_t err_mask = __ballot_sync(0xFFFFFFFFu, (synd_lo | synd_hi) != 0u);
        if (err_mask != 0u && lane == 0 && host_ber_canary != nullptr) {
            silisense_ber_canary_add_sys(host_ber_canary, uint32_t(__popc(err_mask)));
        }

        int acc = 0;
        acc = __dp4a(int(w0), int(x0), acc);
        acc = __dp4a(int(w1), int(x1), acc);
        acc = __dp4a(int(w2), int(x2), acc);
        acc = __dp4a(int(w3), int(x3), acc);
        acc = __dp4a(int(w4), int(x4), acc);
        acc = __dp4a(int(w5), int(x5), acc);
        acc = __dp4a(int(w6), int(x6), acc);
        acc = __dp4a(int(w7), int(x7), acc);
        float z_slow = float(acc) * W_layer_scale[(layer * 16 + warp_id) * 32 + lane] * x_scale;

        // D. Parallel 1-in-8 Hard-FMNMX KV-ring via __dp4a (lanes 0..7 score all 8 slots at once!)
        float kv_sc = -3.402823466e38f;
        uint32_t kv_idx = uint32_t(lane);
        if (lane < 8) {
            int kacc = 0;
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][0]), int(x0), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][1]), int(x1), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][2]), int(x2), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][3]), int(x3), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][4]), int(x4), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][5]), int(x5), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][6]), int(x6), kacc);
            kacc = __dp4a(int(smem_kv_key[warp_id][lane][7]), int(x7), kacc);
            kv_sc = float(kacc);
        }
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov = __shfl_xor_sync(0xFFFFFFFFu, kv_sc, mask);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, kv_idx, mask);
            float    bv = fmaxf(kv_sc, ov);
            bool     take = (ov > kv_sc) || ((ov == kv_sc) && (oi < kv_idx));
            kv_sc  = bv;
            kv_idx = take ? oi : kv_idx;
        }
        int   best_slot = int(__shfl_sync(0xFFFFFFFFu, kv_idx, 0)) & 7;
        float kv_v      = __half2float(smem_kv_val[warp_id][best_slot][lane]);

        // E. Cayley S^31 geodesic update + re-quant into double-buffered smem_x_s8[nxt_buf]
        float z_tot = z_slow + kv_v;
        float v_i   = z_tot - warp_sum_32(z_tot * x_i) * x_i;
        float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + 1e-12f);
        x_i         = fmaf(c_st, x_i, s_st * (v_i * v_inv));

        float amax   = warp_max_abs_32(x_i);
        float inv_sc = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        smem_x_s8[nxt_buf][warp_id][lane] = quant_f32_to_s8(x_i * inv_sc);
        if (lane == 0) smem_x_scale[nxt_buf][warp_id] = amax * (1.0f / 127.0f);
        __syncthreads();
    }

    // Store final state & persistent SU(2) holonomy
    hidden_out[b_idx * 512 + tid] = x_i;
    if (lmod4 == 0) U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id] = U_cur;

    if (!run_fwht_lm_head) return;

    // 3. Tied Orthonormal 0 B VRAM FWHT_256 x bent4 LM Head + 23-Bit Gumbel-FMNMX!
    //    Exact transpose E_byte^T @ hidden:
    //      z_k = (-1)^{bent4(k)} * (x_{pi(k)} - x_{256 + pi(k)}),  pi(k) = k ^ ((k & 7) << 5)
    //      logits = (1 / sqrt(32)) * H_256 @ z   (0 B weight reads!)
    smem_fwht[warp_id][lane] = x_i;
    __syncthreads();

    float v_fwht = 0.0f;
    if (warp_id < 8) {
        int      src_w = warp_id ^ (lane & 7);
        float    diff  = smem_fwht[src_w][lane] - smem_fwht[src_w + 8][lane];
        uint32_t b4    = uint32_t(__popc((lane & 3) & ((lane >> 2) & 3))) & 1u;
        v_fwht         = b4 ? -diff : diff;

        // Stages 0..4 (mask = 1..16): 100% in-register warp butterflies!
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v_fwht, mask);
            v_fwht  = (lane & mask) ? (o - v_fwht) : (v_fwht + o);
        }
    }
    __syncthreads();
    if (warp_id < 8) smem_fwht[warp_id][lane] = v_fwht;
    __syncthreads();

    // Stages 5..7 (wm = 1, 2, 4): 3 cross-warp butterflies in SMEM (all 16 warps hit __syncthreads!)
    #pragma unroll
    for (int wm = 1; wm <= 4; wm <<= 1) {
        if (warp_id < 8) {
            float o = smem_fwht[warp_id ^ wm][lane];
            v_fwht  = (warp_id & wm) ? (o - v_fwht) : (v_fwht + o);
        }
        __syncthreads();
        if (warp_id < 8) smem_fwht[warp_id][lane] = v_fwht;
        __syncthreads();
    }

    if (warp_id < 8) {
        float logit = v_fwht * inv_sqrt32;
        if (logits_out != nullptr) logits_out[b_idx * 256 + tid] = logit;
        float    score = logit + gumbel_23bit_finite(rng_u32[b_idx * 256 + tid], temperature);
        uint32_t idx   = uint32_t(tid);
        warp_fmnmx_argmax_32(score, idx);
        if (lane == 0) {
            smem_ep_val[warp_id] = score;
            smem_ep_idx[warp_id] = uint8_t(idx);
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float    w_val = (lane < 8) ? smem_ep_val[lane] : -3.402823466e38f;
        uint32_t w_idx = (lane < 8) ? uint32_t(smem_ep_idx[lane]) : 256u;
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov = __shfl_xor_sync(0xFFFFFFFFu, w_val, mask);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, w_idx, mask);
            float    bv = fmaxf(w_val, ov);
            bool     take_other = (ov > w_val) || ((ov == w_val) && (oi < w_idx));
            w_val = bv;
            w_idx = take_other ? oi : w_idx;
        }
        if (lane == 0) {
            out_byte[b_idx]  = uint8_t(w_idx);
            out_score[b_idx] = w_val;
        }
    }
}

} // namespace s117_sm86

// ============================================================================
// C-ABI Launch Wrappers
// ============================================================================

extern "C" {

int launch_s117_h256_bent4_byte_inject(const uint8_t* byte_ids, int8_t* out_s8, int n_tokens, cudaStream_t stream) {
    if (n_tokens <= 0) return 0;
    s117_sm86::s117_h256_bent4_byte_inject_sm86<<<n_tokens, 256, 0, stream>>>(byte_ids, out_s8, n_tokens);
    return int(cudaGetLastError());
}

int launch_s117_gumbel_fmnmx_23bit(
    const float* logits, const uint32_t* rng_u32, uint8_t* out_byte, float* out_score,
    float temperature, int n_tokens, cudaStream_t stream
) {
    if (n_tokens <= 0) return 0;
    s117_sm86::s117_gumbel_fmnmx_23bit_cellop_sm86<<<n_tokens, 256, 0, stream>>>(
        logits, rng_u32, out_byte, out_score, temperature, n_tokens
    );
    return int(cudaGetLastError());
}

int launch_s117_sfast_geo_chunk32_fwd(
    const float* x_chunk, const float* z_slow_chunk, const float* w_su2_chunk,
    const void* S_ckpt_0, const float* U_ckpt_0,
    float* y_out, float* x_tilde_out, void* S_ckpt_end, float* U_ckpt_end,
    float cayley_t, float gamma, float eta, float eps, int apply_nr_polish,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0) return 0;
    s117_sm86::s117_sfast_geo_chunk32_fwd_sm86<<<total_bh, 32, 0, stream>>>(
        x_chunk, z_slow_chunk, reinterpret_cast<const float3*>(w_su2_chunk),
        reinterpret_cast<const __half*>(S_ckpt_0), reinterpret_cast<const float4*>(U_ckpt_0),
        y_out, x_tilde_out, reinterpret_cast<__half*>(S_ckpt_end), reinterpret_cast<float4*>(U_ckpt_end),
        cayley_t, gamma, eta, eps, apply_nr_polish, num_heads, stride_b, stride_t, total_bh
    );
    return int(cudaGetLastError());
}

int launch_s117_sfast_geo_chunk32_bwd(
    const float* x_chunk, const float* x_tilde_chunk, const float* z_slow_chunk,
    const void* S_ckpt_0, const float* grad_y_chunk, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0) return 0;
    s117_sm86::s117_sfast_geo_chunk32_bwd_sm86<<<total_bh, 32, 0, stream>>>(
        x_chunk, x_tilde_chunk, z_slow_chunk, reinterpret_cast<const __half*>(S_ckpt_0),
        grad_y_chunk, grad_S_in, grad_z_slow_out, gx_tilde_sfast_out, gx_geo_out, grad_S_out,
        cayley_t, gamma, eta, eps, num_heads, stride_b, stride_t, total_bh
    );
    return int(cudaGetLastError());
}

int launch_s117_sfast_geo_chunk32_bwd_choice_b(
    const float* x_chunk, const float* x_tilde_chunk, const float* z_slow_chunk,
    const void* S_ckpt_0, const float* grad_y_chunk, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0) return 0;
    s117_sm86::s117_sfast_geo_chunk32_bwd_choice_b_sm86<<<total_bh, 32, 0, stream>>>(
        x_chunk, x_tilde_chunk, z_slow_chunk, reinterpret_cast<const __half*>(S_ckpt_0),
        grad_y_chunk, grad_S_in, grad_z_slow_out, gx_tilde_sfast_out, gx_geo_out, grad_S_out,
        cayley_t, gamma, eta, eps, num_heads, stride_b, stride_t, total_bh
    );
    return int(cudaGetLastError());
}

int launch_s117_su2_3line_gu_bwd(
    const float* x_chunk, const float* gx_tilde_total, const float* gx_geo_in,
    const float* w_su2_chunk, const float* U_ckpt_end, const float* grad_U_in,
    float* grad_x_out, float* grad_w_su2_out, float* grad_U_out,
    int project_boundary_tangent, int num_heads, int stride_b, int stride_t,
    int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0) return 0;
    s117_sm86::s117_su2_3line_gu_bwd_sm86<<<total_bh, 32, 0, stream>>>(
        x_chunk, gx_tilde_total, gx_geo_in,
        reinterpret_cast<const float3*>(w_su2_chunk), reinterpret_cast<const float4*>(U_ckpt_end),
        reinterpret_cast<const float4*>(grad_U_in), grad_x_out,
        reinterpret_cast<float3*>(grad_w_su2_out), reinterpret_cast<float4*>(grad_U_out),
        project_boundary_tangent, num_heads, stride_b, stride_t, total_bh
    );
    return int(cudaGetLastError());
}

int launch_s117_spec8_woodbury_commit(
    float* S_prefix_fp32_io, float* U_ckpt_io,
    const int8_t* x_tilde_8x8, const float* x_scale_8x8, const void* e_k_8x8,
    const float* w_branch_8x8, float* causal_spred_out,
    float gamma, float eta, int b_win, int win_len, int total_heads, cudaStream_t stream
) {
    if (total_heads <= 0) return 0;
    s117_sm86::s117_spec8_woodbury_commit_sm86<<<total_heads, 32, 0, stream>>>(
        S_prefix_fp32_io, reinterpret_cast<float4*>(U_ckpt_io),
        x_tilde_8x8, x_scale_8x8, reinterpret_cast<const __half*>(e_k_8x8),
        reinterpret_cast<const float3*>(w_branch_8x8), causal_spred_out,
        gamma, eta, b_win, win_len, total_heads
    );
    return int(cudaGetLastError());
}

int launch_s117_sprefix_int8_shadow_imma(
    const float* S_prefix_fp32, const int8_t* X_branches_s8, const float* X_branch_scale,
    int8_t* S_shadow_s8_out, float* S_row_scale_out,
    float* Y_sprefix_out, float* Y_s31_norm_out, int total_heads, cudaStream_t stream
) {
    if (total_heads <= 0) return 0;
    s117_sm86::s117_sprefix_int8_shadow_imma_sm86<<<total_heads, 32, 0, stream>>>(
        S_prefix_fp32, X_branches_s8, X_branch_scale, S_shadow_s8_out, S_row_scale_out,
        Y_sprefix_out, Y_s31_norm_out, total_heads
    );
    return int(cudaGetLastError());
}

int launch_s117_spec8_fused_4imma(
    const int8_t* W_slow_s8, const float* W_row_scale,
    const int8_t* S_shadow_s8, const float* S_row_scale,
    const int8_t* X_branches_s8, const float* X_branch_scale,
    float* Z_fused_out, float gamma_tau, int total_heads, cudaStream_t stream
) {
    if (total_heads <= 0) return 0;
    s117_sm86::s117_spec8_fused_4imma_sm86<<<total_heads, 32, 0, stream>>>(
        W_slow_s8, W_row_scale, S_shadow_s8, S_row_scale,
        X_branches_s8, X_branch_scale, Z_fused_out, gamma_tau, total_heads
    );
    return int(cudaGetLastError());
}

int launch_s117_uniform_cell_16warp_d512(
    const uint8_t* byte_in, const int8_t* W_layer_s8, const float* W_layer_scale,
    const float* w_su2_layer, const float* kv_keys_init, const void* kv_vals_init,
    const float* lm_head_w, const uint32_t* rng_u32, float* U_state_io,
    float* hidden_out, uint8_t* out_byte, float* out_score, uint32_t* host_ber_canary,
    float cayley_t, float temperature, int n_layers, int batch_size, cudaStream_t stream
) {
    if (batch_size <= 0) return 0;
    s117_sm86::s117_uniform_cell_16warp_d512_sm86<<<batch_size, 512, 0, stream>>>(
        byte_in, W_layer_s8, W_layer_scale, reinterpret_cast<const float3*>(w_su2_layer),
        kv_keys_init, reinterpret_cast<const __half*>(kv_vals_init),
        lm_head_w, rng_u32, reinterpret_cast<float4*>(U_state_io),
        hidden_out, out_byte, out_score, host_ber_canary,
        cayley_t, temperature, n_layers, batch_size
    );
    return int(cudaGetLastError());
}

int launch_s117_uniform_cell_16warp_d512_v2(
    const uint8_t* byte_in, const uint32_t* W_packed_u32, const float* W_layer_scale,
    const float* w_su2_layer, const uint32_t* kv_keys_u32, const void* kv_vals_init,
    const uint32_t* rng_u32, float* U_state_io, float* hidden_out, float* logits_out,
    uint8_t* out_byte, float* out_score, uint32_t* host_ber_canary,
    float cayley_t, float temperature, int n_layers, int run_fwht_lm_head,
    int batch_size, cudaStream_t stream
) {
    if (batch_size <= 0) return 0;
    s117_sm86::s117_uniform_cell_16warp_d512_v2_sm86<<<batch_size, 512, 0, stream>>>(
        byte_in, W_packed_u32, W_layer_scale, reinterpret_cast<const float3*>(w_su2_layer),
        kv_keys_u32, reinterpret_cast<const __half*>(kv_vals_init),
        rng_u32, reinterpret_cast<float4*>(U_state_io),
        hidden_out, logits_out, out_byte, out_score, host_ber_canary,
        cayley_t, temperature, n_layers, run_fwht_lm_head, batch_size
    );
    return int(cudaGetLastError());
}

} // extern "C"
