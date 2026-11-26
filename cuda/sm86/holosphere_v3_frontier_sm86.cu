// ============================================================================
// Track S117 V3 Frontier Bare-Metal Ampere (GA104M sm_86) Kernels
//   1. s117_uniform_cell_16warp_d512_v3_sm86:
//      - Target: < 20 us/tok (> 50,000 tok/s) at d_model = 512, L = 12, 1 CTA
//      - 2x coalesced LDG.128 (uint4) cross-layer prefetched weights + uint4 SMEM x_mix & KV-ring
//      - Speculative 8x __dp4a overlapped with 0-POPC dual-SWAR INT32 [16,11,4] SECDED
//        + cold-path exact repair & @P0 RED.E.ADD.STRONG.SYS 4 B .sys BER canary
//      - Double-buffered smem_x_s8[2][16][32] (0 WAR races, 0 cross-CTA barriers)
//      - Ping-pong 4-barrier 0 B VRAM tied orthonormal FWHT_256 x bent4 LM head + 23-bit Gumbel
//   2. s117_chunk32_bwd_wmma_sm86 (W10-OBS-61 Double-Woodbury Symmetric Gram HMMA):
//      - Target: > 100,000 tok/s (<= 327.6 ms / 12L) at B=4, T=8192, D=512 (stride_t = D)
//      - Row-owned S_row[32] & gS_row[32] -> ZERO SHFLs in S_t*xt & gS*xt (down from 10,240/chunk!)
//      - Exact Double-Woodbury Symmetric Gram M_{t,k} = gamma^{|t-k|-1}*eta*<v_{max}, e_{min}>:
//        12x HMMA.16816 (lower triangle G = V E^T) + 16x HMMA.16816 (W = M X_tilde)
//        computes BOTH S_t^T v_t AND eta*gS_{t+1}^T e_t Woodbury sums in 28 Tensor Core ops!
//      - Strict FP32 S_ckpt_0 & pow_gamma_u5 (Council Gate G2!), 0 B spills, 0 LDL/STL,
//        persistent multi-chunk n_chunks loop (1 launch/layer vs 256!)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace s117_v3 {

__device__ __forceinline__ float warp_sum_32(float v) {
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
// 0-POPC Dual-SWAR INT32 [16, 11, 4] SECDED Hot-Path Detector + Cold Repair
// ============================================================================

__device__ __forceinline__ uint32_t extract_lsb4_u32(uint32_t w4) {
    return ((w4 & 0x01010101u) * 0x01020408u) >> 24;
}

// Checks BOTH 16-bit [16,11,4] codewords packed in (w_lo in bits 0..15, w_hi in bits 16..31)
// in parallel on full-rate INT32 ALU (ZERO SFU POPC ops on the hot path!).
// Returns 0u iff BOTH w16_lo and w16_hi have exact 5-bit syndrome == 0.
__device__ __forceinline__ uint32_t dual_a16_syndrome_nonzero_swar(uint32_t w32) {
    uint32_t a  = w32 ^ (w32 >> 1);
    uint32_t b  = a   ^ (a   >> 2);
    uint32_t c  = b   ^ (b   >> 4);
    uint32_t d  = c   ^ (c   >> 8);           // bit 0 = p0_lo, bit 16 = p0_hi
    uint32_t x1 = w32 & 0xAAAAAAAAu;
    x1 ^= (x1 >> 2);
    x1 ^= (x1 >> 4);
    x1 ^= (x1 >> 8);                          // bit 1 = p1_lo, bit 17 = p1_hi
    uint32_t x2 = a   & 0x44444444u;
    x2 ^= (x2 >> 4);
    x2 ^= (x2 >> 8);                          // bit 2 = p2_lo, bit 18 = p2_hi
    uint32_t p3 = (b ^ (b >> 8)) & 0x00100010u; // bit 4 = p3_lo, bit 20 = p3_hi
    uint32_t p4 = c & 0x01000100u;              // bit 8 = p4_lo, bit 24 = p4_hi
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

__device__ __forceinline__ void a16_secded_fix_u4_cold(uint4& wv, uint32_t w16) {
    uint32_t synd = a16_secded_syndrome_cold(w16);
    if ((synd & 1u) != 0u) {
        uint32_t j    = (synd >> 1) & 0xFu;
        uint32_t mask = 1u << ((j & 3u) << 3);
        uint32_t widx = j >> 2;
        if (widx == 0u)      wv.x ^= mask;
        else if (widx == 1u) wv.y ^= mask;
        else if (widx == 2u) wv.z ^= mask;
        else                 wv.w ^= mask;
    }
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
// Kernel 1 (V3): Single-CTA 16-Warp d_model=512 Megakernel (< 20 us/tok Target!)
//   - Layout W_packed_u128[L][16][2][32] (uint4: w_lo = cols 0..15, w_hi = cols 16..31)
//     -> 2 coalesced LDG.128/layer, software-prefetched for layer+1 across compute & barrier!
//   - Speculative 8x __dp4a runs IMMEDIATELY overlapped with 0-POPC dual-SWAR INT32 SECDED;
//     cold __ballot_sync branch repairs 1-bit SEC bit-exact & fires @P0 RED.E.ADD.STRONG.SYS!
//   - uint4 prologue staging for KV-ring keys & values (2 LDG.128 total vs 10 scalar loads!)
//   - Ping-pong 4-barrier 0 B VRAM tied orthonormal FWHT_256 x bent4 LM head + 23-bit Gumbel!
// ============================================================================

__global__ __launch_bounds__(512, 1)
void s117_uniform_cell_16warp_d512_v3_sm86(
    const uint8_t* __restrict__ byte_in,          // [B]
    const uint4* __restrict__ W_packed_u128,      // [L, 16, 2, 32] coalesced 16-byte weight tiles
    const float* __restrict__ W_layer_scale,      // [L, 16, 32]
    const float3* __restrict__ w_su2_layer,       // [L, 16, 8]
    const uint4* __restrict__ kv_keys_u128,       // [B, 16, 16] (8 slots x 2 uint4 = 256 B/head)
    const uint4* __restrict__ kv_vals_u128,       // [B, 16, 32] (8 slots x 32 fp16 = 512 B/head)
    const uint32_t* __restrict__ rng_u32,         // [B, 256]
    float4* __restrict__ U_state_io,              // [B, 16, 8] persistent SU(2) holonomy
    float* __restrict__ hidden_out,               // [B, 512] final S^31x16 state
    float* __restrict__ logits_out,               // [B, 256] optional FWHT_256 logits (NULL-safe)
    uint8_t* __restrict__ out_byte,               // [B]
    float* __restrict__ out_score,                // [B]
    uint32_t* __restrict__ host_ber_canary,       // 4 B cudaHostAllocMapped .sys BER counter (NULL-safe)
    float cayley_t,
    float temperature,
    int n_layers,
    int run_fwht_lm_head,
    int batch_size
) {
    // 13,992 B total SMEM (13.7 KiB << 100 KiB; FWHT epilogue aliases dead smem_kv_val -> -2,048 B & -1 barrier!)
    __shared__ int8_t  smem_x_s8[2][16][32];       // 1,024 B: double-buffered cross-head exchange
    __shared__ uint4   smem_xmix_u128[16][2];      //   512 B: 2x uint4 packed x_mix for __dp4a
    __shared__ float   smem_x_scale[2][16];        //   128 B: double-buffered S^31 absmax/127
    __shared__ uint4   smem_kv_key[16][8][2];      // 4,096 B: uint4 KV-ring keys (2 LDS.128/slot!)
    alignas(16) __shared__ __half smem_kv_val[16][8][32]; // 8,192 B: fp16 KV values; aliased by FWHT epilogue!
    __shared__ float   smem_ep_val[8];             //    32 B
    __shared__ uint8_t smem_ep_idx[8];             //     8 B

    int b_idx = blockIdx.x;
    if (b_idx >= batch_size) return;

    int tid     = threadIdx.x; // 0..511
    int warp_id = tid >> 5;    // 0..15
    int lane    = tid & 31;    // 0..31
    int spin_id = lane >> 2;   // 0..7
    int base    = spin_id << 2;
    int lmod4   = lane & 3;

    // 1. Prologue: Zero-VRAM H_256 x bent4 byte injection + 2-LDG.128 KV staging + Layer-0 Prefetch!
    uint8_t tok_b = byte_in[b_idx];
    int8_t  inj_q = h256_bent4_byte_inject_s8(tok_b, warp_id & 7, lane);
    if (warp_id >= 8) inj_q = int8_t(-inj_q);
    const float inv_sqrt32 = 0.176776695f;
    float x_i = (inj_q > 0) ? inv_sqrt32 : -inv_sqrt32;

    smem_x_s8[0][warp_id][lane] = inj_q;
    if (lane == 0) smem_x_scale[0][warp_id] = inv_sqrt32 * (1.0f / 127.0f);

    // Stage ALL 8 KV-ring keys (lanes 0..15: 1x LDG.128) & ALL 8 KV-ring values (lanes 0..31: 1x LDG.128)!
    if (lane < 16) {
        smem_kv_key[warp_id][lane >> 1][lane & 1] = __ldg(&kv_keys_u128[(b_idx * 16 + warp_id) * 16 + lane]);
    }
    uint4* smem_kv_val_u4 = reinterpret_cast<uint4*>(&smem_kv_val[warp_id][0][0]);
    smem_kv_val_u4[lane] = __ldg(&kv_vals_u128[(b_idx * 16 + warp_id) * 32 + lane]);

    // Prefetch Layer 0 weights & SU(2) omega into registers BEFORE __syncthreads()!
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

    // 2. Software-Pipelined L-Layer Single-CTA Loop (0 cross-CTA barriers, 0 WAR races!)
    #pragma unroll 1
    for (int layer = 0; layer < n_layers; ++layer) {
        int cur_buf = layer & 1;
        int nxt_buf = cur_buf ^ 1;

        // A. Cayley SU(2) holonomy rotation (using prefetched w_su2)
        U_cur     = quat_nr_polish(quat_mul(cayley_su2(w_su2), U_cur));
        float4 xs = gather_spinor4(x_i, base);
        float  xt = select_lane4(quat_mul(quat_conj(U_cur), xs), lmod4);

        // B. Cross-head butterfly exchange + fast uint4 pack of x_mix
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
        uint32_t* xmix_u32 = reinterpret_cast<uint32_t*>(&smem_xmix_u128[warp_id][0]);
        if (lmod4 == 0) {
            xmix_u32[lane >> 2] = xq_p0 | (xq_p1 << 8) | (xq_p2 << 16) | (xq_p3 << 24);
        }
        __syncwarp();
        uint4 x_lo = smem_xmix_u128[warp_id][0];
        uint4 x_hi = smem_xmix_u128[warp_id][1];

        // C. Speculative 8x __dp4a IMMEDIATELY (0 SECDED latency stall!) overlapped with
        //    0-POPC Dual-SWAR INT32 [16,11,4] SECDED detection + cold-path bit-exact repair!
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
            if (nz_swar != 0u) {
                a16_secded_fix_u4_cold(w_lo, w16_lo);
                a16_secded_fix_u4_cold(w_hi, w16_hi);
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
            if (lane == 0 && host_ber_canary != nullptr) {
                silisense_ber_canary_add_sys(host_ber_canary, uint32_t(__popc(err_mask)));
            }
        }
        float z_slow = float(acc) * w_sc * x_scale;

        // Prefetch Layer (layer + 1) weights into w_lo/w_hi/w_sc/w_su2 RIGHT NOW (overlaps KV-ring + geodesic + barrier!)
        int next_l = layer + 1;
        if (next_l < n_layers) {
            const uint4* wn_base = W_packed_u128 + (next_l * 16 + warp_id) * 64;
            w_lo  = __ldg(&wn_base[0 * 32 + lane]);
            w_hi  = __ldg(&wn_base[1 * 32 + lane]);
            w_sc  = __ldg(&W_layer_scale[(next_l * 16 + warp_id) * 32 + lane]);
            const float3* wpn = &w_su2_layer[(next_l * 16 + warp_id) * 8 + spin_id];
            w_su2 = make_float3(__ldg(&wpn->x), __ldg(&wpn->y), __ldg(&wpn->z));
        }

        // D. Parallel 1-in-8 Hard-FMNMX KV-ring via 2x LDS.128 + 8x __dp4a (lanes 0..7)
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
        float kv_v      = __half2float(smem_kv_val[warp_id][best_slot][lane]);

        // E. Cayley S^31 geodesic update + re-quant into smem_x_s8[nxt_buf]
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

    hidden_out[b_idx * 512 + tid] = x_i;
    if (lmod4 == 0) U_state_io[(b_idx * 16 + warp_id) * 8 + spin_id] = U_cur;

    if (!run_fwht_lm_head) return;

    // 3. Ping-Pong 5-Barrier 0 B VRAM Tied Orthonormal FWHT_256 x bent4 LM Head + 23-Bit Gumbel!
    //    Reuses dead smem_kv_val (8,192 B): [0..15]=input x_i, [16..23]=ping, [24..31]=pong!
    float (*smem_fwht)[32] = reinterpret_cast<float (*)[32]>(&smem_kv_val[0][0][0]);
    smem_fwht[warp_id][lane] = x_i;
    __syncthreads(); // Barrier 1: input hidden[0..15] ready

    float v_fwht = 0.0f;
    if (warp_id < 8) {
        int      src_w = warp_id ^ (lane & 7);
        float    diff  = smem_fwht[src_w][lane] - smem_fwht[src_w + 8][lane];
        uint32_t b4    = uint32_t(__popc((lane & 3) & ((lane >> 2) & 3))) & 1u;
        v_fwht         = b4 ? -diff : diff;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v_fwht, mask);
            v_fwht  = (lane & mask) ? (o - v_fwht) : (v_fwht + o);
        }
        smem_fwht[16 + warp_id][lane] = v_fwht; // Disjoint ping [16..23]: saves 1 CTA barrier!
    }
    __syncthreads(); // Barrier 2: ping buffer [16..23] ready

    // Stage 5 (wm=1): read ping [16..23] -> write pong [24..31]
    if (warp_id < 8) {
        float o = smem_fwht[16 + (warp_id ^ 1)][lane];
        v_fwht  = (warp_id & 1) ? (o - v_fwht) : (v_fwht + o);
        smem_fwht[24 + warp_id][lane] = v_fwht;
    }
    __syncthreads(); // Barrier 3: pong buffer [24..31] ready

    // Stage 6 (wm=2): read pong [24..31] -> write ping [16..23]
    if (warp_id < 8) {
        float o = smem_fwht[24 + (warp_id ^ 2)][lane];
        v_fwht  = (warp_id & 2) ? (o - v_fwht) : (v_fwht + o);
        smem_fwht[16 + warp_id][lane] = v_fwht;
    }
    __syncthreads(); // Barrier 4: ping buffer [16..23] ready

    // Stage 7 (wm=4): read ping [16..23] -> final logit + Gumbel + warp argmax!
    if (warp_id < 8) {
        float o     = smem_fwht[16 + (warp_id ^ 4)][lane];
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
    __syncthreads(); // Barrier 5: 8-warp tournament ready

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

// ============================================================================
// Ampere sm_86 HMMA.16816 (FP16 -> FP32 Accumulate) PTX Primitive
// ============================================================================

__device__ __forceinline__ uint32_t pack_h2_u32(__half h0, __half h1) {
    __half2 h2 = __halves2half2(h0, h1);
    return *reinterpret_cast<const uint32_t*>(&h2);
}

__device__ __forceinline__ void hmma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

// ============================================================================
// Kernel 2 (W10-OBS-61): Double-Woodbury Symmetric Gram HMMA.16816 Chunk32 Backward
//   - Row-owned S_row[32] & gS_row[32] -> ZERO SHFLs in S_t*xt & gS*xt!
//   - Exact Double-Woodbury Symmetric Gram identity:
//       G = V E^T (12x HMMA.16816 lower triangle only!)
//       M_{t,k} = M_{k,t} = gamma^{|t-k|-1} * eta * G_{max(t,k), min(t,k)}  (M_{t,t} = 0)
//       W = M X_tilde (16x HMMA.16816) computes BOTH S_t^T v_t AND eta*gS_{t+1}^T e_t Woodbury sums!
//       gx_tilde_sfast[t] = gamma^t * (S_0^T v_t) + eta*gamma^{31-t} * (gS_end^T e_t) + W[t]
//   - Strict FP32 S_ckpt_0 & pow_gamma_u5 (Council Gate G2!), 0 B spills, 0 LDL/STL!
//   - Persistent multi-chunk loop (n_chunks = 1..256+) keeps gS_row[32] hot in registers across T=8192!
// ============================================================================

__global__ __launch_bounds__(32, 16)
void s117_chunk32_bwd_wmma_sm86(
    const float* __restrict__ x_seq,           // [B, T, D] (stride_b, stride_t)
    const float* __restrict__ x_tilde_seq,     // [B, T, D]
    const float* __restrict__ z_slow_seq,      // [B, T, D]
    const float* __restrict__ S_ckpts_f32,     // [n_chunks, BH, 32, 32] STRICT FP32 per Gate G2!
    const float* __restrict__ grad_y_seq,      // [B, T, D]
    const float* __restrict__ grad_S_in,       // [BH, 32, 32] FP32 incoming gS at end of chunk range
    float* __restrict__ grad_z_slow_out,       // [B, T, D]
    float* __restrict__ gx_tilde_sfast_out,    // [B, T, D]
    float* __restrict__ gx_geo_out,            // [B, T, D]
    float* __restrict__ grad_S_out,            // [BH, 32, 32] FP32 outgoing gS at start of chunk range
    float cayley_t,
    float gamma,
    float eta,
    float eps,
    int num_heads,
    int stride_b,
    int stride_t,
    int n_chunks,
    int total_bh
) {
    // 6,336 B SMEM/warp (16 warps/SM @ 101,376 B <= 102,400 B GA104M SMEM & <= 128 regs!)
    alignas(16) __shared__ __half smem_xt[32][33];    // 2,112 B: x_tilde[t, c]
    alignas(16) __shared__ __half smem_err[32][33];   // 2,112 B: Phase 1-3A err[t, c], Phase 3B FP32 W_half[16][33]!
    alignas(16) __shared__ __half smem_v[32][33];     // 2,112 B: Phase 1 s_pred, Phase 2 v_t in-place, Phase 3B M[t,k]!

    int bh   = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (bh >= total_bh) return;

    int b_idx       = bh / num_heads;
    int h_idx       = bh - b_idx * num_heads;
    int64_t bh_base = int64_t(b_idx) * int64_t(stride_b) + int64_t(h_idx) * 32;

    float t2    = cayley_t * cayley_t;
    float inv_t = 1.0f / (1.0f + t2);
    float c_st  = (1.0f - t2) * inv_t;
    float s_st  = (2.0f * cayley_t) * inv_t;

    int gid = lane >> 2; // 0..7 for HMMA.16816
    int tid = lane & 3;  // 0..3 for HMMA.16816

    // Persistent multi-chunk reverse sweep: gS_cur lives in grad_S_out between chunks (0 host syncs!)
    const float* gS_src_ptr = grad_S_in + int64_t(bh) * 1024;
    float*       gS_dst_ptr = grad_S_out + int64_t(bh) * 1024;

    #pragma unroll 1
    for (int ch = n_chunks - 1; ch >= 0; --ch) {
        int64_t ch_base = bh_base + int64_t(ch) * 32LL * int64_t(stride_t);
        const float* S0_head = S_ckpts_f32 + (int64_t(ch) * int64_t(total_bh) + int64_t(bh)) * 1024LL;

        // --- Phase 1 (t = 0 -> 31): Row-owned S_row[32] -> ZERO SHFLs in S_t*xt & S_t update! ---
        {
            float S_row[32];
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                S_row[c] = __ldg(&S0_head[lane * 32 + c]);
            }

            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                int64_t off = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                float   x_i = x_seq[off];
                float   xt  = x_tilde_seq[off];
                smem_xt[t][lane] = __float2half(xt);
                __syncwarp();

                // Row-owned S_t * xt: 0 SHFL instructions!
                float s_pred = 0.0f;
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    s_pred = fmaf(S_row[c], __half2float(smem_xt[t][c]), s_pred);
                }

                float z_i   = z_slow_seq[off] + s_pred;
                float v_i   = z_i - warp_sum_32(z_i * x_i) * x_i;
                float v_inv = rsqrtf(warp_sum_32(v_i * v_i) + eps);
                float err_i = (c_st * x_i + s_st * (v_i * v_inv)) - s_pred;

                smem_err[t][lane] = __float2half(err_i);
                smem_v[t][lane]   = __float2half(s_pred); // Phase 2 overwrites s_pred[t] in-place with v_t[t]!

                // Row-owned rank-1 update: lane already owns err_i! 0 SHFL instructions!
                float er = eta * err_i;
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    S_row[c] = fmaf(gamma, S_row[c], er * __half2float(smem_xt[t][c]));
                }
            }
        } // S_row[32] dead -> ptxas reclaims all 32 regs before Phase 1B!
        __syncwarp();

        // --- Phase 1B (Pre-Phase-2 Boundary Term): eta * gamma^{31-t} * (gS_end^T e_t) ---
        //   Computed BEFORE Phase 2 overwrites gS_dst_ptr (100% safe when gS_src_ptr == gS_dst_ptr
        //   for multi-chunk ch < n_chunks - 1!) and lets Phase 3B reuse dead smem_err for FP32 W!
        {
            float gSe_col[32];
            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                gSe_col[r] = gS_src_ptr[r * 32 + lane];
            }
            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                float acc = 0.0f;
                #pragma unroll
                for (int r = 0; r < 32; ++r) {
                    acc = fmaf(gSe_col[r], __half2float(smem_err[t][r]), acc);
                }
                int64_t off = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                gx_tilde_sfast_out[off] = (eta * pow_gamma_u5(gamma, 31 - t)) * acc;
            }
        } // gSe_col[32] dead -> ptxas reclaims all 32 regs before Phase 2!

        // --- Phase 2 (t = 31 -> 0): Row-owned gS_row[32] -> ZERO SHFLs in gS*xt & gS update! ---
        {
            float gS_row[32];
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                gS_row[c] = gS_src_ptr[lane * 32 + c];
            }

            #pragma unroll 1
            for (int t = 31; t >= 0; --t) {
                int64_t off    = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                float   x_i    = x_seq[off];
                float   s_pred = __half2float(smem_v[t][lane]);
                float   zs_i   = z_slow_seq[off];
                float   gy_i   = grad_y_seq[off];

                // Row-owned ge_i = eta * (gS * xt)[lane]: 0 SHFL instructions!
                float ge_i = 0.0f;
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    ge_i = fmaf(gS_row[c], __half2float(smem_xt[t][c]), ge_i);
                }
                ge_i *= eta;

                float z_i      = zs_i + s_pred;
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

                float v_t = gz_i - ge_i;
                smem_v[t][lane] = __float2half(v_t);

                // Row-owned gS update: lane already owns v_t! 0 SHFL instructions!
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    gS_row[c] = fmaf(gamma, gS_row[c], v_t * __half2float(smem_xt[t][c]));
                }
            }

            // Store final chunk-exit gS_0 (stays L1-hot if ch > 0!)
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                gS_dst_ptr[lane * 32 + c] = gS_row[c];
            }
        } // gS_row[32] dead -> ptxas reclaims all 32 regs before Phase 2B!
        __syncwarp();

        // --- Phase 2B (Boundary Term): Strict FP32 gamma^t * (S_0^T v_t) ---
        //   Computed BEFORE Phase 3A overwrites smem_v with symmetric M -> fits 6,336 B SMEM (16 w/SM)!
        {
            float S0_col[32];
            #pragma unroll
            for (int r = 0; r < 32; ++r) {
                S0_col[r] = __ldg(&S0_head[r * 32 + lane]);
            }
            #pragma unroll 1
            for (int t = 0; t < 32; ++t) {
                float acc = 0.0f;
                #pragma unroll
                for (int r = 0; r < 32; ++r) {
                    acc = fmaf(S0_col[r], __half2float(smem_v[t][r]), acc);
                }
                int64_t off = ch_base + int64_t(t) * int64_t(stride_t) + lane;
                gx_tilde_sfast_out[off] = fmaf(pow_gamma_u5(gamma, t), acc, gx_tilde_sfast_out[off]);
            }
        } // S0_col[32] dead -> ptxas reclaims all 32 regs before Phase 3A HMMA!
        __syncwarp();

        // --- Phase 3A: 12x HMMA.16816 Lower-Triangle G = V E^T held in 24 regs -> symmetric M in smem_v! ---
        //   All 6 lower-triangle tiles finish reading smem_v & smem_err BEFORE overwriting smem_v with M!
        {
            float g_d0[6], g_d1[6], g_d2[6], g_d3[6];
            #pragma unroll
            for (int tile = 0; tile < 6; ++tile) {
                int m0 = (tile < 2) ? 0 : 16;
                int n0 = (tile < 2) ? (tile * 8) : ((tile - 2) * 8);
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                #pragma unroll
                for (int k0 = 0; k0 < 32; k0 += 16) {
                    uint32_t a0 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 0], smem_v[m0 + gid + 0][k0 + 2 * tid + 1]);
                    uint32_t a1 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 0], smem_v[m0 + gid + 8][k0 + 2 * tid + 1]);
                    uint32_t a2 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 8], smem_v[m0 + gid + 0][k0 + 2 * tid + 9]);
                    uint32_t a3 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 8], smem_v[m0 + gid + 8][k0 + 2 * tid + 9]);
                    uint32_t b0 = pack_h2_u32(smem_err[n0 + gid][k0 + 2 * tid + 0], smem_err[n0 + gid][k0 + 2 * tid + 1]);
                    uint32_t b1 = pack_h2_u32(smem_err[n0 + gid][k0 + 2 * tid + 8], smem_err[n0 + gid][k0 + 2 * tid + 9]);
                    hmma_m16n8k16_f32_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
                }
                g_d0[tile] = d0; g_d1[tile] = d1; g_d2[tile] = d2; g_d3[tile] = d3;
            }
            __syncwarp(); // smem_v & smem_err are now 100% DEAD! Write symmetric M into smem_v!

            #pragma unroll
            for (int tile = 0; tile < 6; ++tile) {
                int m0 = (tile < 2) ? 0 : 16;
                int n0 = (tile < 2) ? (tile * 8) : ((tile - 2) * 8);
                int r0 = m0 + gid, r1 = m0 + gid + 8;
                int c0 = n0 + 2 * tid, c1 = c0 + 1;
                if (r0 > c0) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r0 - 1 - c0) * eta * g_d0[tile]);
                    smem_v[r0][c0] = mv; smem_v[c0][r0] = mv;
                } else if (r0 == c0) { smem_v[r0][c0] = __float2half(0.0f); }
                if (r0 > c1) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r0 - 1 - c1) * eta * g_d1[tile]);
                    smem_v[r0][c1] = mv; smem_v[c1][r0] = mv;
                } else if (r0 == c1) { smem_v[r0][c1] = __float2half(0.0f); }
                if (r1 > c0) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r1 - 1 - c0) * eta * g_d2[tile]);
                    smem_v[r1][c0] = mv; smem_v[c0][r1] = mv;
                } else if (r1 == c0) { smem_v[r1][c0] = __float2half(0.0f); }
                if (r1 > c1) {
                    __half mv = __float2half(pow_gamma_u5(gamma, r1 - 1 - c1) * eta * g_d3[tile]);
                    smem_v[r1][c1] = mv; smem_v[c1][r1] = mv;
                } else if (r1 == c1) { smem_v[r1][c1] = __float2half(0.0f); }
            }
        }
        __syncwarp();

        // --- Phase 3B: 16x HMMA.16816 W = M @ X_tilde -> FP32 smem_w16 overlaid on dead smem_err! ---
        {
            float (*smem_w16)[33] = reinterpret_cast<float (*)[33]>(&smem_err[0][0]);
            #pragma unroll
            for (int m0 = 0; m0 < 32; m0 += 16) {
                #pragma unroll
                for (int n0 = 0; n0 < 32; n0 += 8) {
                    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    #pragma unroll
                    for (int k0 = 0; k0 < 32; k0 += 16) {
                        uint32_t a0 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 0], smem_v[m0 + gid + 0][k0 + 2 * tid + 1]);
                        uint32_t a1 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 0], smem_v[m0 + gid + 8][k0 + 2 * tid + 1]);
                        uint32_t a2 = pack_h2_u32(smem_v[m0 + gid + 0][k0 + 2 * tid + 8], smem_v[m0 + gid + 0][k0 + 2 * tid + 9]);
                        uint32_t a3 = pack_h2_u32(smem_v[m0 + gid + 8][k0 + 2 * tid + 8], smem_v[m0 + gid + 8][k0 + 2 * tid + 9]);
                        uint32_t b0 = pack_h2_u32(smem_xt[k0 + 2 * tid + 0][n0 + gid], smem_xt[k0 + 2 * tid + 1][n0 + gid]);
                        uint32_t b1 = pack_h2_u32(smem_xt[k0 + 2 * tid + 8][n0 + gid], smem_xt[k0 + 2 * tid + 9][n0 + gid]);
                        hmma_m16n8k16_f32_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
                    }
                    int c0 = n0 + 2 * tid;
                    smem_w16[gid + 0][c0 + 0] = d0;
                    smem_w16[gid + 0][c0 + 1] = d1;
                    smem_w16[gid + 8][c0 + 0] = d2;
                    smem_w16[gid + 8][c0 + 1] = d3;
                }
                __syncwarp();

                #pragma unroll 1
                for (int dt = 0; dt < 16; ++dt) {
                    int64_t off = ch_base + int64_t(m0 + dt) * int64_t(stride_t) + lane;
                    gx_tilde_sfast_out[off] += smem_w16[dt][lane];
                }
                __syncwarp();
            }
        }
        // Next chunk (ch - 1) reads gS_0 from gS_dst_ptr (stays in L1 cache!)
        gS_src_ptr = gS_dst_ptr;
    }
}

} // namespace s117_v3

extern "C" {

int launch_s117_uniform_cell_16warp_d512_v3(
    const uint8_t* byte_in, const void* W_packed_u128, const float* W_layer_scale,
    const float* w_su2_layer, const void* kv_keys_u128, const void* kv_vals_u128,
    const uint32_t* rng_u32, float* U_state_io, float* hidden_out, float* logits_out,
    uint8_t* out_byte, float* out_score, uint32_t* host_ber_canary,
    float cayley_t, float temperature, int n_layers, int run_fwht_lm_head,
    int batch_size, cudaStream_t stream
) {
    if (batch_size <= 0) return 0;
    s117_v3::s117_uniform_cell_16warp_d512_v3_sm86<<<batch_size, 512, 0, stream>>>(
        byte_in, reinterpret_cast<const uint4*>(W_packed_u128), W_layer_scale,
        reinterpret_cast<const float3*>(w_su2_layer),
        reinterpret_cast<const uint4*>(kv_keys_u128),
        reinterpret_cast<const uint4*>(kv_vals_u128),
        rng_u32, reinterpret_cast<float4*>(U_state_io),
        hidden_out, logits_out, out_byte, out_score, host_ber_canary,
        cayley_t, temperature, n_layers, run_fwht_lm_head, batch_size
    );
    return int(cudaGetLastError());
}

int launch_s117_chunk32_bwd_wmma(
    const float* x_seq, const float* x_tilde_seq, const float* z_slow_seq,
    const float* S_ckpts_f32, const float* grad_y_seq, const float* grad_S_in,
    float* grad_z_slow_out, float* gx_tilde_sfast_out, float* gx_geo_out, float* grad_S_out,
    float cayley_t, float gamma, float eta, float eps,
    int num_heads, int stride_b, int stride_t, int n_chunks, int total_bh, cudaStream_t stream
) {
    if (total_bh <= 0 || num_heads <= 0 || n_chunks <= 0) return 0;
    s117_v3::s117_chunk32_bwd_wmma_sm86<<<total_bh, 32, 0, stream>>>(
        x_seq, x_tilde_seq, z_slow_seq, S_ckpts_f32, grad_y_seq, grad_S_in,
        grad_z_slow_out, gx_tilde_sfast_out, gx_geo_out, grad_S_out,
        cayley_t, gamma, eta, eps, num_heads, stride_b, stride_t, n_chunks, total_bh
    );
    return int(cudaGetLastError());
}

} // extern "C"
