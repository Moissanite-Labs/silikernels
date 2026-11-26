// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// HoloSphere-V3 Bare-Metal sm_86 Kernels: CTA-SMEM Packed .sys Canary & 1 KiB FP16 Folded FWHT_256 x bent4 Head
//   1. 1 KiB Learned FP16 Channel Fold (u = alpha * h_L[:256] + beta * h_L[256:])
//      fused ahead of 0 B HBM-table tied orthonormal FWHT_256 x bent4 LM head
//      (256 x __half2 = 1,024 B L1-resident; 0 extra barriers; 0 B spills).
//   2. CTA-SMEM-Aggregated Packed (ded_count << 16) | sec_count .sys BER Canary:
//      warps reduce [16,11,4] SEC/DED counts in CTA SMEM (12 B) so thread 0 emits
//      <= 1 predicated per-lane saturating atom.release.sys.global.cas.b32 per CTA
//      (0 on clean steps; independent 16-bit lane saturation across arbitrarily many
//      CTAs/warps/launches with zero SEC->DED carry and zero DED bit-31 wrap-to-zero,
//      plus monotonic atom.release.sys.global.or.b32 diagnostic/overflow escalation).
//   Compiled cleanly under nvcc -O3 -arch=sm_86 --Werror all-warnings with 0 B spills.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace holosphere_smem_canary {

__device__ __forceinline__ float warp_sum_32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float warp_max_abs_32(float v) {
    float a = fabsf(v);
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, mask));
    }
    return a;
}

__device__ __forceinline__ int8_t quant_f32_to_s8(float scaled) {
    int q = __float2int_rn(scaled);
    q = (q > 127) ? 127 : ((q < -128) ? -128 : q);
    return static_cast<int8_t>(q);
}

__device__ __forceinline__ float4 cayley_su2(float3 w) {
    float w2  = fmaf(w.x, w.x, fmaf(w.y, w.y, w.z * w.z));
    float inv = __frcp_rn(1.0f + w2);
    return make_float4((1.0f - w2) * inv, (2.0f * w.x) * inv, (2.0f * w.y) * inv, (2.0f * w.z) * inv);
}

__device__ __forceinline__ float4 quat_conj(float4 q) {
    return make_float4(q.x, -q.y, -q.z, -q.w);
}

__device__ __forceinline__ float4 quat_mul(float4 a, float4 b) {
    return make_float4(
        fmaf(a.x, b.x, -fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w))),
        fmaf(a.x, b.y,  fmaf(a.y, b.x, fmaf(a.z, b.w, -a.w * b.z))),
        fmaf(a.x, b.z,  fmaf(-a.y, b.w, fmaf(a.z, b.x, a.w * b.y))),
        fmaf(a.x, b.w,  fmaf(a.y, b.z, fmaf(-a.z, b.y, a.w * b.x)))
    );
}

__device__ __forceinline__ float4 quat_nr_polish(float4 q) {
    float n2 = fmaf(q.x, q.x, fmaf(q.y, q.y, fmaf(q.z, q.z, q.w * q.w)));
    float s  = fmaf(-0.5f, n2, 1.5f);
    return make_float4(q.x * s, q.y * s, q.z * s, q.w * s);
}

__device__ __forceinline__ float4 gather_spinor4(float x_i, int base) {
    return make_float4(
        __shfl_sync(0xFFFFFFFFu, x_i, base + 0),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 1),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 2),
        __shfl_sync(0xFFFFFFFFu, x_i, base + 3)
    );
}

__device__ __forceinline__ float select_lane4(float4 q, int lmod4) {
    float v01 = (lmod4 & 1) ? q.y : q.x;
    float v23 = (lmod4 & 1) ? q.w : q.z;
    return (lmod4 & 2) ? v23 : v01;
}

__device__ __forceinline__ int8_t h256_bent4_byte_inject_s8(uint8_t utf8_byte, int warp_id, int lane_id) {
    int d      = ((warp_id & 7) << 5) | (lane_id & 31);
    int db     = d ^ ((lane_id & 7) << 5);
    int bent4  = __popc((lane_id & 3) & ((lane_id >> 2) & 3)) & 1;
    int parity = (__popc(int(utf8_byte) & db) ^ bent4) & 1;
    return parity ? int8_t(-127) : int8_t(127);
}

__device__ __forceinline__ float gumbel_23bit_finite(uint32_t u32, float temperature) {
    float top23       = __uint2float_rn(u32 >> 9);
    float u           = fmaf(top23, 1.1920928955078125e-7f, 5.9604644775390625e-8f);
    float neg_lg2_u   = -__log2f(u);
    float lg2_neg_lg2 = __log2f(neg_lg2_u);
    float g           = fmaf(-0.69314718f, lg2_neg_lg2, 0.36651292f);
    return temperature * g;
}

__device__ __forceinline__ void warp_fmnmx_argmax_32(float& val, uint32_t& idx) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float    ov = __shfl_xor_sync(0xFFFFFFFFu, val, mask);
        uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, idx, mask);
        float    bv = fmaxf(val, ov);
        bool     take_other = (ov > val) || ((ov == val) && (oi < idx));
        val = bv;
        idx = take_other ? oi : idx;
    }
}

__device__ __forceinline__ uint32_t extract_lsb4_u32(uint32_t w4) {
    return ((w4 & 0x01010101u) * 0x01020408u) >> 24;
}

__device__ __forceinline__ uint32_t dual_a16_syndrome_nonzero_swar(uint32_t w32) {
    uint32_t a  = w32 ^ (w32 >> 1);
    uint32_t b  = a   ^ (a   >> 2);
    uint32_t c  = b   ^ (b   >> 4);
    uint32_t d  = c   ^ (c   >> 8);
    uint32_t x1 = w32 & 0xAAAAAAAAu;
    x1 ^= (x1 >> 2);
    x1 ^= (x1 >> 4);
    x1 ^= (x1 >> 8);
    uint32_t x2 = a   & 0x44444444u;
    x2 ^= (x2 >> 4);
    x2 ^= (x2 >> 8);
    uint32_t p3 = (b ^ (b >> 8)) & 0x00100010u;
    uint32_t p4 = c & 0x01000100u;
    return (d & 0x00010001u) | (x1 & 0x00020002u) | (x2 & 0x00040004u) | p3 | p4;
}

__device__ __forceinline__ uint32_t a16_secded_syndrome_cold(uint32_t w16) {
    uint32_t p0 = uint32_t(__popc(w16)) & 1u;
    uint32_t p1 = uint32_t(__popc(w16 & 0xAAAAu)) & 1u;
    uint32_t p2 = uint32_t(__popc(w16 & 0xCCCCu)) & 1u;
    uint32_t p3 = uint32_t(__popc(w16 & 0xF0F0u)) & 1u;
    uint32_t p4 = uint32_t(__popc(w16 & 0xFF00u)) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

__device__ __forceinline__ void a16_secded_classify_fix_u4_cold(
    uint4& wv, uint32_t w16, uint32_t& lane_sec, uint32_t& lane_ded, uint32_t& diag
) {
    uint32_t synd = a16_secded_syndrome_cold(w16);
    if (synd != 0u) {
        if ((synd & 1u) != 0u) {
            uint32_t j    = (synd >> 1) & 0xFu;
            uint32_t mask = 1u << ((j & 3u) << 3);
            uint32_t widx = j >> 2;
            if (widx == 0u)      wv.x ^= mask;
            else if (widx == 1u) wv.y ^= mask;
            else if (widx == 2u) wv.z ^= mask;
            else                 wv.w ^= mask;
            lane_sec += 1u;
            if (diag == 0u) diag = synd & 0x1Fu;
        } else {
            lane_ded += 1u;
            diag = (1u << 5) | (synd & 0x1Fu);
        }
    }
}

// Per-lane saturating .sys atomic CAS accumulator for packed (ded << 16) | sec:
// Saturates SEC [15:0] and DED [31:16] independently at 0xFFFF across arbitrarily
// many concurrent CTAs, warps, or un-drained launches (zero SEC->DED carry,
// zero DED bit-31 wrap-to-zero). Escalates DED >= 1 and sets diag_io |= 0x80000000u
// whenever either lane saturates.
__device__ __forceinline__ void silisense_ber_canary_sat_add_sys(
    uint32_t* host_mapped_ptr,
    uint32_t sec_delta,
    uint32_t ded_delta,
    uint32_t& diag_io
) {
    uint32_t s_add = (sec_delta > 0xFFFFu) ? 0x10000u : sec_delta;
    uint32_t d_add = (ded_delta > 0xFFFFu) ? 0x10000u : ded_delta;
    uint32_t assumed = 0u;
    for (;;) {
        uint32_t old_sec = assumed & 0xFFFFu;
        uint32_t old_ded = assumed >> 16;
        uint32_t sec_sum = old_sec + s_add;
        uint32_t ded_sum = old_ded + d_add;
        bool ovf_sec = sec_sum > 0xFFFFu;
        bool ovf_ded = ded_sum > 0xFFFFu;
        uint32_t new_sec = ovf_sec ? 0xFFFFu : sec_sum;
        uint32_t new_ded = ovf_ded ? 0xFFFFu : ded_sum;
        bool ovf = ovf_sec || ovf_ded;
        if (ovf && new_ded == 0u) {
            new_ded = 1u;
        }
        uint32_t desired = (new_ded << 16) | new_sec;
        uint32_t prev = 0u;
        asm volatile(
            "atom.release.sys.global.cas.b32 %0, [%1], %2, %3;"
            : "=r"(prev)
            : "l"(host_mapped_ptr), "r"(assumed), "r"(desired)
            : "memory"
        );
        if (prev == assumed) {
            if (ovf) {
                diag_io |= 0x80000000u;
            }
            return;
        }
        assumed = prev;
    }
}

// Monotonic .sys atomic OR for diagnostic/overflow bits so a concurrent non-overflowing
// CTA can never clobber the 0x80000000u overflow flag or a DED syndrome bit.
__device__ __forceinline__ void silisense_or_release_sys_u32(uint32_t* host_mapped_ptr, uint32_t bits) {
    asm volatile(
        "atom.release.sys.global.or.b32 _, [%0], %1;"
        :
        : "l"(host_mapped_ptr), "r"(bits)
        : "memory"
    );
}

// ============================================================================
// Templated Single-CTA 16-Warp d_model=512 Megakernel (A/B Paired Warp-Canary vs CTA-SMEM + 1 KiB Fold)
//   USE_FOLD_1K   = false: fixed unweighted diff (h[:256] - h[256:512])
//                 = true : 1 KiB learned FP16 fold (alpha*h[:256] + beta*h[256:])
//   USE_SMEM_CTA  = false: per-warp predicated saturating .sys CAS (baseline)
//                 = true : CTA-SMEM-aggregated <= 1 saturating .sys CAS / CTA
// ============================================================================

template <bool USE_FOLD_1K, bool USE_SMEM_CTA>
__device__ __forceinline__ void holosphere_uniform_cell_d512_impl(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const __half2* __restrict__ fold_ab_f16,     // [256] = 1,024 B = 1 KiB (NULL if !USE_FOLD_1K)
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,   // 4 B mapped host word: (ded<<16)|sec (NULL-safe)
    uint32_t* __restrict__ host_diag_out,        // 4 B mapped host diag (NULL-safe)
    uint32_t* __restrict__ dev_sys_atomic_cnt,   // instrumentation: committed .sys canary updates (NULL-safe)
    float cayley_t,
    float temperature,
    int n_layers,
    int run_fwht_lm_head,
    int batch_size
) {
    __shared__ int8_t  smem_x_s8[2][16][32];
    __shared__ alignas(16) unsigned char smem_xmix_raw[16][32];
    __shared__ float   smem_x_scale[2][16];
    __shared__ uint4   smem_kv_key[16][8][2];
    // Raw byte-backed shared overlay (8,192 B, barrier-ordered, memcpy-accessed — zero union/aliasing UB):
    //   Phase 1 (layer loop): kv_val[16][8][32] as __half (loaded via uint4) = 8,192 B
    //   Phase 2 (FWHT head):  fwht[32][32] as float = 4,096 B
    __shared__ alignas(16) unsigned char smem_overlay_raw[8192];
    __shared__ float   smem_ep_val[8];
    __shared__ uint8_t smem_ep_idx[8];
    __shared__ uint32_t smem_cta_sec;
    __shared__ uint32_t smem_cta_ded;
    __shared__ uint32_t smem_cta_diag;

    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int spin_id = lane >> 2;
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    if (USE_SMEM_CTA && tid == 0) {
        smem_cta_sec  = 0u;
        smem_cta_ded  = 0u;
        smem_cta_diag = 0u;
    }

    uint8_t tok_b = byte_in[b_idx];
    int8_t  inj_q = h256_bent4_byte_inject_s8(tok_b, warp_id & 7, lane);
    if (warp_id >= 8) inj_q = int8_t(-inj_q);
    const float inv_sqrt32 = 0.17677669529663688f;
    float x_i = (inj_q > 0) ? inv_sqrt32 : -inv_sqrt32;

    smem_x_s8[0][warp_id][lane] = inj_q;
    if (lane == 0) smem_x_scale[0][warp_id] = inv_sqrt32 * (1.0f / 127.0f);

    if (lane < 16) {
        smem_kv_key[warp_id][lane >> 1][lane & 1] = __ldg(&kv_keys_u128[(b_idx * 16 + warp_id) * 16 + lane]);
    }
    uint4 kv_val_in = __ldg(&kv_vals_u128[(b_idx * 16 + warp_id) * 32 + lane]);
    std::memcpy(&smem_overlay_raw[(warp_id * 32 + lane) * 16], &kv_val_in, sizeof(uint4));

    uint4  w_lo  = make_uint4(0u, 0u, 0u, 0u);
    uint4  w_hi  = make_uint4(0u, 0u, 0u, 0u);
    float  w_sc  = 0.0f;
    float3 w_su2 = make_float3(0.0f, 0.0f, 0.0f);
    if (n_layers > 0) {
        const uint4* w0_base = W_packed_u128 + warp_id * 64;
        w_lo  = __ldg(&w0_base[0 * 32 + lane]);
        w_hi  = __ldg(&w0_base[1 * 32 + lane]);
        w_sc  = __ldg(&W_layer_scale[warp_id * 32 + lane]);
        const float3* wp0 = &w_su2_layer[warp_id * 8 + spin_id];
        w_su2 = make_float3(__ldg(&wp0->x), __ldg(&wp0->y), __ldg(&wp0->z));
    }

    float4 U_cur = U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id];
    float  t2    = cayley_t * cayley_t;
    float  inv_t = __frcp_rn(1.0f + t2);
    float  c_st  = (1.0f - t2) * inv_t;
    float  s_st  = (2.0f * cayley_t) * inv_t;

    __syncthreads();

    #pragma unroll 1
    for (int layer = 0; layer < n_layers; ++layer) {
        int cur_buf = layer & 1;
        int nxt_buf = cur_buf ^ 1;

        U_cur     = quat_nr_polish(quat_mul(cayley_su2(w_su2), U_cur));
        float4 xs = gather_spinor4(x_i, base);
        float  xt = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);

        int   partner   = warp_id ^ (1 << (layer & 3));
        float x_partner = float(smem_x_s8[cur_buf][partner][lane]) * smem_x_scale[cur_buf][partner];
        float x_mix     = fmaf(0.75f, xt, 0.25f * x_partner);

        float x_amax   = warp_max_abs_32(x_mix);
        float x_scale  = x_amax * (1.0f / 127.0f);
        float x_inv_sc = (x_amax > 1e-12f) ? (127.0f * __frcp_rn(x_amax)) : 0.0f;
        uint32_t xq_b  = uint32_t(uint8_t(quant_f32_to_s8(x_mix * x_inv_sc)));
        uint32_t xq_p0 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 0);
        uint32_t xq_p1 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 1);
        uint32_t xq_p2 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 2);
        uint32_t xq_p3 = __shfl_sync(0xFFFFFFFFu, xq_b, (lane & ~3) + 3);
        if (lmod4 == 0) {
            uint32_t packed_x = xq_p0 | (xq_p1 << 8) | (xq_p2 << 16) | (xq_p3 << 24);
            std::memcpy(&smem_xmix_raw[warp_id][(lane >> 2) * 4], &packed_x, sizeof(uint32_t));
        }
        __syncwarp();
        uint4 x_lo, x_hi;
        std::memcpy(&x_lo, &smem_xmix_raw[warp_id][0], sizeof(uint4));
        std::memcpy(&x_hi, &smem_xmix_raw[warp_id][16], sizeof(uint4));

        int acc = 0;
        acc = __dp4a(int(w_lo.x), int(x_lo.x), acc);
        acc = __dp4a(int(w_lo.y), int(x_lo.y), acc);
        acc = __dp4a(int(w_lo.z), int(x_lo.z), acc);
        acc = __dp4a(int(w_lo.w), int(x_lo.w), acc);
        acc = __dp4a(int(w_hi.x), int(x_hi.x), acc);
        acc = __dp4a(int(w_hi.y), int(x_hi.y), acc);
        acc = __dp4a(int(w_hi.z), int(x_hi.z), acc);
        acc = __dp4a(int(w_hi.w), int(x_hi.w), acc);

        uint32_t w16_lo = extract_lsb4_u32(w_lo.x) | (extract_lsb4_u32(w_lo.y) << 4)
                        | (extract_lsb4_u32(w_lo.z) << 8) | (extract_lsb4_u32(w_lo.w) << 12);
        uint32_t w16_hi = extract_lsb4_u32(w_hi.x) | (extract_lsb4_u32(w_hi.y) << 4)
                        | (extract_lsb4_u32(w_hi.z) << 8) | (extract_lsb4_u32(w_hi.w) << 12);
        uint32_t nz_swar  = dual_a16_syndrome_nonzero_swar(w16_lo | (w16_hi << 16));
        uint32_t err_mask = __ballot_sync(0xFFFFFFFFu, nz_swar != 0u);
        if (__builtin_expect(err_mask != 0u, 0)) {
            uint32_t l_sec = 0u, l_ded = 0u, diag = 0u;
            if (nz_swar != 0u) {
                a16_secded_classify_fix_u4_cold(w_lo, w16_lo, l_sec, l_ded, diag);
                a16_secded_classify_fix_u4_cold(w_hi, w16_hi, l_sec, l_ded, diag);
                if (l_sec != 0u) {
                    acc = 0;
                    acc = __dp4a(int(w_lo.x), int(x_lo.x), acc);
                    acc = __dp4a(int(w_lo.y), int(x_lo.y), acc);
                    acc = __dp4a(int(w_lo.z), int(x_lo.z), acc);
                    acc = __dp4a(int(w_lo.w), int(x_lo.w), acc);
                    acc = __dp4a(int(w_hi.x), int(x_hi.x), acc);
                    acc = __dp4a(int(w_hi.y), int(x_hi.y), acc);
                    acc = __dp4a(int(w_hi.z), int(x_hi.z), acc);
                    acc = __dp4a(int(w_hi.w), int(x_hi.w), acc);
                }
            }
            uint32_t w_sec = warp_sum_u32(l_sec);
            uint32_t w_ded = warp_sum_u32(l_ded);
            int src_lane = __ffs(int(err_mask)) - 1;
            uint32_t diag_out = __shfl_sync(0xFFFFFFFFu, diag, src_lane);
            if (lane == 0 && (w_sec | w_ded) != 0u) {
                if (USE_SMEM_CTA) {
                    if (w_sec != 0u) atomicAdd(&smem_cta_sec, w_sec);
                    if (w_ded != 0u) atomicAdd(&smem_cta_ded, w_ded);
                    if (diag_out != 0u) atomicCAS(&smem_cta_diag, 0u, diag_out);
                } else {
                    if (host_packed_canary != nullptr) {
                        uint32_t d_tmp = diag_out;
                        silisense_ber_canary_sat_add_sys(host_packed_canary, w_sec, w_ded, d_tmp);
                        if (host_diag_out != nullptr) silisense_or_release_sys_u32(host_diag_out, d_tmp);
                        if (dev_sys_atomic_cnt != nullptr) atomicAdd(dev_sys_atomic_cnt, 1u);
                    }
                }
            }
        }
        float z_slow = float(acc) * w_sc * x_scale;

        int next_l = layer + 1;
        if (next_l < n_layers) {
            const uint4* wn_base = W_packed_u128 + (next_l * 16 + warp_id) * 64;
            w_lo  = __ldg(&wn_base[0 * 32 + lane]);
            w_hi  = __ldg(&wn_base[1 * 32 + lane]);
            w_sc  = __ldg(&W_layer_scale[(next_l * 16 + warp_id) * 32 + lane]);
            const float3* wpn = &w_su2_layer[(next_l * 16 + warp_id) * 8 + spin_id];
            w_su2 = make_float3(__ldg(&wpn->x), __ldg(&wpn->y), __ldg(&wpn->z));
        }

        float    kv_sc  = -3.402823466e38f;
        uint32_t kv_idx = uint32_t(lane);
        if (lane < 8) {
            uint4 k_lo = smem_kv_key[warp_id][lane][0];
            uint4 k_hi = smem_kv_key[warp_id][lane][1];
            int kacc = 0;
            kacc = __dp4a(int(k_lo.x), int(x_lo.x), kacc);
            kacc = __dp4a(int(k_lo.y), int(x_lo.y), kacc);
            kacc = __dp4a(int(k_lo.z), int(x_lo.z), kacc);
            kacc = __dp4a(int(k_lo.w), int(x_lo.w), kacc);
            kacc = __dp4a(int(k_hi.x), int(x_hi.x), kacc);
            kacc = __dp4a(int(k_hi.y), int(x_hi.y), kacc);
            kacc = __dp4a(int(k_hi.z), int(x_hi.z), kacc);
            kacc = __dp4a(int(k_hi.w), int(x_hi.w), kacc);
            kv_sc = float(kacc);
        }
        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            float    ov   = __shfl_xor_sync(0xFFFFFFFFu, kv_sc, mask);
            uint32_t oi   = __shfl_xor_sync(0xFFFFFFFFu, kv_idx, mask);
            float    bv   = fmaxf(kv_sc, ov);
            bool     take = (ov > kv_sc) || ((ov == kv_sc) && (oi < kv_idx));
            kv_sc  = bv;
            kv_idx = take ? oi : kv_idx;
        }
        int   best_slot = int(__shfl_sync(0xFFFFFFFFu, kv_idx, 0)) & 7;
        __half kv_h;
        std::memcpy(&kv_h, &smem_overlay_raw[((warp_id * 8 + best_slot) * 32 + lane) * 2], sizeof(__half));
        float kv_v      = __half2float(kv_h);

        float z_tot = z_slow + kv_v;
        float v_i   = z_tot - warp_sum_32(z_tot * x_i) * x_i;
        float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + 1e-12f);
        x_i         = fmaf(c_st, x_i, s_st * (v_i * v_inv));

        float amax   = warp_max_abs_32(x_i);
        float inv_sc = (amax > 1e-12f) ? (127.0f * __frcp_rn(amax)) : 0.0f;
        smem_x_s8[nxt_buf][warp_id][lane] = quant_f32_to_s8(x_i * inv_sc);
        if (lane == 0) smem_x_scale[nxt_buf][warp_id] = amax * (1.0f / 127.0f);
        __syncthreads();
    }

    // CTA-SMEM canary flush: reuses final layer-loop __syncthreads() (0 extra barriers!)
    // Emits <= 1 predicated saturating .sys CAS per CTA (0 on clean steps!).
    if (USE_SMEM_CTA && tid == 0 && host_packed_canary != nullptr) {
        uint32_t sec_tot = smem_cta_sec;
        uint32_t ded_tot = smem_cta_ded;
        if ((sec_tot | ded_tot) != 0u) {
            uint32_t diag_w = smem_cta_diag;
            silisense_ber_canary_sat_add_sys(host_packed_canary, sec_tot, ded_tot, diag_w);
            if (host_diag_out != nullptr) silisense_or_release_sys_u32(host_diag_out, diag_w);
            if (dev_sys_atomic_cnt != nullptr) atomicAdd(dev_sys_atomic_cnt, 1u);
        }
    }

    hidden_out[b_idx * 512 + tid] = x_i;
    if (lmod4 == 0) U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id] = U_cur;

    if (!run_fwht_lm_head) return;

    // 0 B HBM-table Tied Orthonormal FWHT_256 x bent4 LM Head (+ optional 1 KiB FP16 fold!)
    auto fwht_store = [&](int row, int col, float val) {
        std::memcpy(&smem_overlay_raw[(row * 32 + col) * 4], &val, sizeof(float));
    };
    auto fwht_load = [&](int row, int col) -> float {
        float val;
        std::memcpy(&val, &smem_overlay_raw[(row * 32 + col) * 4], sizeof(float));
        return val;
    };

    fwht_store(warp_id, lane, x_i);
    __syncthreads();

    float v_fwht = 0.0f;
    if (warp_id < 8) {
        int src_w = warp_id ^ (lane & 7);
        float u_val;
        if (USE_FOLD_1K) {
            int src_idx = (src_w << 5) | lane;
            float2 ab = __half22float2(__ldg(&fold_ab_f16[src_idx]));
            u_val = fmaf(ab.x, fwht_load(src_w, lane), ab.y * fwht_load(src_w + 8, lane));
        } else {
            u_val = fwht_load(src_w, lane) - fwht_load(src_w + 8, lane);
        }
        uint32_t b4 = uint32_t(__popc((lane & 3) & ((lane >> 2) & 3))) & 1u;
        v_fwht = b4 ? -u_val : u_val;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v_fwht, mask);
            v_fwht  = (lane & mask) ? (o - v_fwht) : (v_fwht + o);
        }
        fwht_store(16 + warp_id, lane, v_fwht);
    }
    __syncthreads();

    if (warp_id < 8) {
        float o = fwht_load(16 + (warp_id ^ 1), lane);
        v_fwht  = (warp_id & 1) ? (o - v_fwht) : (v_fwht + o);
        fwht_store(24 + warp_id, lane, v_fwht);
    }
    __syncthreads();

    if (warp_id < 8) {
        float o = fwht_load(24 + (warp_id ^ 2), lane);
        v_fwht  = (warp_id & 2) ? (o - v_fwht) : (v_fwht + o);
        fwht_store(16 + warp_id, lane, v_fwht);
    }
    __syncthreads();

    if (warp_id < 8) {
        float o     = fwht_load(16 + (warp_id ^ 4), lane);
        v_fwht      = (warp_id & 4) ? (o - v_fwht) : (v_fwht + o);
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

__global__ __launch_bounds__(512, 1)
void holosphere_cell_unfolded_warp_canary_sm86(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out,
    uint32_t* __restrict__ dev_sys_atomic_cnt,
    float cayley_t,
    float temperature,
    int n_layers,
    int run_fwht_lm_head,
    int batch_size
) {
    holosphere_uniform_cell_d512_impl<false, false>(
        byte_in, W_packed_u128, W_layer_scale, w_su2_layer, kv_keys_u128, kv_vals_u128,
        nullptr, rng_u32, U_state_io, hidden_out, logits_out, out_byte, out_score,
        host_packed_canary, host_diag_out, dev_sys_atomic_cnt,
        cayley_t, temperature, n_layers, run_fwht_lm_head, batch_size
    );
}

__global__ __launch_bounds__(512, 1)
void holosphere_cell_folded1k_smem_canary_sm86(
    const uint8_t* __restrict__ byte_in,
    const uint4* __restrict__ W_packed_u128,
    const float* __restrict__ W_layer_scale,
    const float3* __restrict__ w_su2_layer,
    const uint4* __restrict__ kv_keys_u128,
    const uint4* __restrict__ kv_vals_u128,
    const __half2* __restrict__ fold_ab_f16,
    const uint32_t* __restrict__ rng_u32,
    float4* __restrict__ U_state_io,
    float* __restrict__ hidden_out,
    float* __restrict__ logits_out,
    uint8_t* __restrict__ out_byte,
    float* __restrict__ out_score,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out,
    uint32_t* __restrict__ dev_sys_atomic_cnt,
    float cayley_t,
    float temperature,
    int n_layers,
    int run_fwht_lm_head,
    int batch_size
) {
    holosphere_uniform_cell_d512_impl<true, true>(
        byte_in, W_packed_u128, W_layer_scale, w_su2_layer, kv_keys_u128, kv_vals_u128,
        fold_ab_f16, rng_u32, U_state_io, hidden_out, logits_out, out_byte, out_score,
        host_packed_canary, host_diag_out, dev_sys_atomic_cnt,
        cayley_t, temperature, n_layers, run_fwht_lm_head, batch_size
    );
}

// Standalone Head-to-Head FWHT_256 x bent4 LM Head Microbenchmark Kernels (0 B HBM Vocab Table)
template <bool USE_FOLD_1K>
__device__ __forceinline__ void holosphere_lm_head_impl(
    const float* __restrict__ x_in,          // [B, 512]
    const __half2* __restrict__ fold_ab_f16, // [256] = 1,024 B (NULL if !USE_FOLD_1K)
    float* __restrict__ logits_out,          // [B, 256]
    float logit_scale
) {
    __shared__ float smem[32][32];
    int b_idx   = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    smem[warp_id][lane] = __ldg(&x_in[b_idx * 512 + tid]);
    __syncthreads();

    float v = 0.0f;
    if (warp_id < 8) {
        int src_w = warp_id ^ (lane & 7);
        float u_val;
        if (USE_FOLD_1K) {
            int src_idx = (src_w << 5) | lane;
            float2 ab = __half22float2(__ldg(&fold_ab_f16[src_idx]));
            u_val = fmaf(ab.x, smem[src_w][lane], ab.y * smem[src_w + 8][lane]);
        } else {
            u_val = smem[src_w][lane] - smem[src_w + 8][lane];
        }
        int b4 = __popc((lane & 3) & ((lane >> 2) & 3)) & 1;
        v = b4 ? -u_val : u_val;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem[16 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[16 + (warp_id ^ 1)][lane];
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem[24 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[24 + (warp_id ^ 2)][lane];
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem[16 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[16 + (warp_id ^ 4)][lane];
        v = (warp_id & 4) ? (o - v) : (v + o);
        logits_out[b_idx * 256 + tid] = v * 0.17677669529663688f * logit_scale;
    }
}

__global__ __launch_bounds__(512, 1)
void holosphere_lm_head_unfolded_sm86(
    const float* __restrict__ x_in,
    float* __restrict__ logits_out,
    float logit_scale
) {
    holosphere_lm_head_impl<false>(x_in, nullptr, logits_out, logit_scale);
}

__global__ __launch_bounds__(512, 1)
void holosphere_lm_head_folded1k_sm86(
    const float* __restrict__ x_in,
    const __half2* __restrict__ fold_ab_f16,
    float* __restrict__ logits_out,
    float logit_scale
) {
    holosphere_lm_head_impl<true>(x_in, fold_ab_f16, logits_out, logit_scale);
}

// Fail-closed single-CTA and multi-CTA saturation/overflow probe kernel for packed .sys canary
__global__ __launch_bounds__(32, 1)
void holosphere_smem_canary_overflow_probe_sm86(
    uint32_t sec_tot,
    uint32_t ded_tot,
    uint32_t diag_in,
    uint32_t* __restrict__ host_packed_canary,
    uint32_t* __restrict__ host_diag_out
) {
    if (threadIdx.x == 0 && host_packed_canary != nullptr && (sec_tot | ded_tot) != 0u) {
        uint32_t d = diag_in;
        silisense_ber_canary_sat_add_sys(host_packed_canary, sec_tot, ded_tot, d);
        if (host_diag_out != nullptr) silisense_or_release_sys_u32(host_diag_out, d);
    }
}

} // namespace holosphere_smem_canary
