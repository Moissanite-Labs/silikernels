/*
 * a16_v2_ampere_kernels.cuh — V2 Zero-Spill Ampere sm_86 Compute & Integrity Kernels
 *
 * Guarantees <= 48 registers, 0 B stack frame, 0 B spill stores/loads, 0 SASS LDL/STL:
 *   - 0-ULP canonical 128-lane binary tree reduction matching AVX-512 & CPU bit-for-bit
 *   - 1-LOP3 + 5-POPC Extended Hamming [16, 11, 4] SECDED SEC repair / DED SafeHalt
 *   - Fused H_4 + 22-bit radix-128 3-limb dp4a22 GEMV (4.5 bpw g=32 & 3.125 bpw g=128)
 *   - Batched B <= 16 IMMA K=64 (mma.sync.aligned.m16n8k32 s8/s8) & Split-K BF16 Tensor Core
 *   - Hoisted 256 B constant-memory 8x8 Cayley-Woodbury transform (c_wb_inv_256b[64])
 *   - IRIS golden-ratio Weyl + Murmur3 fmix32 state canary finalizer
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SILIKERNELS_A16_V2_AMPERE_KERNELS_CUH
#define SILIKERNELS_A16_V2_AMPERE_KERNELS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../../include/silikernels_v2_contracts.h"

namespace silikernels_v2 {

__constant__ float    c_wb_inv_256b[64];
__constant__ uint32_t c_iris_expected_sig_u32[16];

#define SILIK_V2_DEV __device__ __forceinline__

SILIK_V2_DEV float fp16_bits_to_f32_dev(uint16_t h) {
    __half hh;
    memcpy(&hh, &h, sizeof(hh));
    return __half2float(hh);
}

SILIK_V2_DEV uint32_t lop3_0x28_dev(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint32_t out;
    asm("lop3.b32 %0, %1, %2, %3, 0x60;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
#else
    return (a & (b ^ c));
#endif
}

SILIK_V2_DEV int32_t dp4a_s32_dev(int32_t src_c, int32_t a, int32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, src_c);
#else
    int32_t acc = src_c;
    for (int i = 0; i < 4; ++i) {
        int32_t ai = static_cast<int8_t>((a >> (8 * i)) & 0xFF);
        int32_t bi = static_cast<int8_t>((b >> (8 * i)) & 0xFF);
        acc += ai * bi;
    }
    return acc;
#endif
}

SILIK_V2_DEV uint32_t secded_syn_u16_dev(uint32_t odd16) {
    const uint32_t x  = odd16 & 0xFFFFu;
    const uint32_t p0 = static_cast<uint32_t>(__popc(x)) & 1u;
    const uint32_t s0 = static_cast<uint32_t>(__popc(x & 0xAAAAu)) & 1u;
    const uint32_t s1 = static_cast<uint32_t>(__popc(x & 0xCCCCu)) & 1u;
    const uint32_t s2 = static_cast<uint32_t>(__popc(x & 0xF0F0u)) & 1u;
    const uint32_t s3 = static_cast<uint32_t>(__popc(x & 0xFF00u)) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

SILIK_V2_DEV uint32_t secded_syn_w64_dev(uint64_t w64) {
    const uint32_t w0 = static_cast<uint32_t>(w64);
    const uint32_t w1 = static_cast<uint32_t>(w64 >> 32);
    const uint32_t px = lop3_0x28_dev(0x11111111u, w0, w1);
    const uint32_t p0 = static_cast<uint32_t>(__popc(px)) & 1u;
    const uint32_t s0 = static_cast<uint32_t>(__popc(px & 0x10101010u)) & 1u;
    const uint32_t s1 = static_cast<uint32_t>(__popc(px & 0x11001100u)) & 1u;
    const uint32_t s2 = static_cast<uint32_t>(__popc(px & 0x11110000u)) & 1u;
    const uint32_t s3 = static_cast<uint32_t>(__popc(w1 & 0x11111111u)) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

SILIK_V2_DEV void heal_u16_dev(uint32_t& w16, uint32_t& local_sec, uint32_t& local_ded) {
    const uint32_t syn = secded_syn_u16_dev(w16);
    if (syn != 0u) {
        if ((syn & 1u) != 0u) {
            w16 ^= (1u << ((syn >> 1) & 15u));
            local_sec += 1u;
        } else {
            local_ded += 1u;
        }
    }
}

SILIK_V2_DEV void heal_w64_dev(uint64_t& w64, uint32_t& local_sec, uint32_t& local_ded) {
    const uint32_t syn = secded_syn_w64_dev(w64);
    if (syn != 0u) {
        if ((syn & 1u) != 0u) {
            w64 ^= (1ULL << (((syn >> 1) & 15u) * 4u));
            local_sec += 1u;
        } else {
            local_ded += 1u;
        }
    }
}

SILIK_V2_DEV void commit_canary_counts_dev(
    HoloSphereRazorCanary64* canary,
    uint32_t local_sec,
    uint32_t local_ded
) {
    if (canary == nullptr) return;
    if (local_sec != 0u) atomicAdd(&canary->sec_count_u32, local_sec);
    if (local_ded != 0u) atomicAdd(&canary->ded_count_u32, local_ded);
}

SILIK_V2_DEV uint32_t iris_mix_word_dev(uint32_t state, uint32_t word) {
    uint32_t x = state ^ (word + SILIKERNELS_V2_IRIS_WEYL_MIX + (state << 6) + (state >> 2));
    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);
    return x;
}

SILIK_V2_DEV float tree_reduce_128_sm86(float val, float* sm_warp4) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wid  = tid >> 5;
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, stride);
    }
    if (lane == 0u) sm_warp4[wid] = val;
    __syncthreads();
    if (tid == 0u) {
        sm_warp4[0] = (sm_warp4[0] + sm_warp4[1]) + (sm_warp4[2] + sm_warp4[3]);
    }
    __syncthreads();
    return sm_warp4[0];
}

SILIK_V2_DEV void mma_m16n8k32_s8_dev(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(d0), "r"(d1), "r"(d2), "r"(d3)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

SILIK_V2_DEV void mma_m16n8k16_bf16_dev(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

SILIK_V2_DEV uint32_t unpack4_s8_nib_dev(uint32_t sub16) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint32_t b01 = ((sub16 & 0x000Fu) | ((sub16 & 0x00F0u) << 4) |
                    ((sub16 & 0x0F00u) << 8) | ((sub16 & 0xF000u) << 12));
    uint32_t out;
    asm("vsub4.s32.u32.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(b01), "r"(0x08080808u), "r"(0));
    return out;
#else
    int8_t c0 = static_cast<int8_t>((sub16 & 0x000Fu)) - 8;
    int8_t c1 = static_cast<int8_t>(((sub16 >> 4) & 0x000Fu)) - 8;
    int8_t c2 = static_cast<int8_t>(((sub16 >> 8) & 0x000Fu)) - 8;
    int8_t c3 = static_cast<int8_t>(((sub16 >> 12) & 0x000Fu)) - 8;
    return static_cast<uint32_t>(static_cast<uint8_t>(c0)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c1)) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c3)) << 24);
#endif
}

SILIK_V2_DEV uint32_t expand_4bits_to_bytes_dev(uint32_t nib4) {
    return ((nib4 & 0xFu) * 0x00204081u) & 0x01010101u;
}

SILIK_V2_DEV uint32_t unpack4_3bpw_s8_dev(uint32_t wp0_4, uint32_t wp1_4, uint32_t wp2_4) {
    const uint32_t b0 = expand_4bits_to_bytes_dev(wp0_4);
    const uint32_t b1 = expand_4bits_to_bytes_dev(wp1_4);
    const uint32_t b2 = expand_4bits_to_bytes_dev(wp2_4);
    return b0 | (b1 << 1) | (b2 * 0xFCu);
}

/* 1. Standalone 0-ULP canonical 128-lane tree reduction kernel */
__global__ __launch_bounds__(128, 10)
void a16_v2_tree_reduce_128_sm86(
    const float* __restrict__ in128,
    float* __restrict__ out1
) {
    __shared__ float sm_warp4[4];
    const float s = tree_reduce_128_sm86(in128[threadIdx.x], sm_warp4);
    if (threadIdx.x == 0u) out1[0] = s;
}

/* 2. Fused Block-H4 + 22-bit radix-128 3-limb activation quantizer */
__global__ __launch_bounds__(128, 10)
void a16_v2_quant22_h4_sm86(
    const float* __restrict__ x_in,
    int8_t* __restrict__ x_hi,
    int8_t* __restrict__ x_md,
    int8_t* __restrict__ x_lo,
    float* __restrict__ x_sc,
    int32_t* __restrict__ x_sumq,
    uint32_t in_dim
) {
    const uint32_t tid   = threadIdx.x;
    const uint32_t wid   = tid >> 5;
    const uint32_t lane  = tid & 31u;
    const uint32_t n_grp = in_dim >> 5;
    for (uint32_t g = blockIdx.x * 4u + wid; g < n_grp; g += gridDim.x * 4u) {
        const uint32_t idx = g * 32u + lane;
        float xv = x_in[idx];
        const float v1 = __shfl_xor_sync(0xffffffffu, xv, 1);
        xv = ((lane & 1u) == 0u) ? (xv + v1) : (v1 - xv);
        const float v2 = __shfl_xor_sync(0xffffffffu, xv, 2);
        xv = ((lane & 2u) == 0u) ? (xv + v2) : (v2 - xv);
        xv *= 0.5f;

        float amax = fabsf(xv);
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        const float sc = fmaxf(amax / 2064384.0f, 1.0e-12f);
        const int32_t q = __float2int_rn(xv / sc);
        const int32_t q_hi = (q >= 0) ? ((q + 8192) >> 14) : -((-q + 8192) >> 14);
        const int32_t r1   = q - (q_hi * 16384);
        const int32_t q_md = (r1 >= 0) ? ((r1 + 64) >> 7) : -((-r1 + 64) >> 7);
        const int32_t q_lo = r1 - (q_md * 128);

        int32_t qsum = q;
#pragma unroll
        for (int m = 16; m >= 1; m >>= 1) qsum += __shfl_xor_sync(0xffffffffu, qsum, m);
        if (lane == 0u) {
            x_sc[g] = sc;
            x_sumq[g] = qsum;
        }
        x_hi[idx] = static_cast<int8_t>(q_hi);
        x_md[idx] = static_cast<int8_t>(q_md);
        x_lo[idx] = static_cast<int8_t>(q_lo);
    }
}

/* 3. B=1 4.5 bpw g=32 dp4a22 GEMV with SECDED + 0-ULP tree_reduce_128 */
__global__ __launch_bounds__(128, 10)
void a16_v2_gemv_b1_45bpw_dp4a22_sm86(
    const uint64_t* __restrict__ w_u64,
    const uint16_t* __restrict__ w_sc,
    const int8_t* __restrict__ x_hi,
    const int8_t* __restrict__ x_md,
    const int8_t* __restrict__ x_lo,
    const float* __restrict__ x_sc,
    const int32_t* __restrict__ x_sumq,
    float* __restrict__ y_out,
    uint32_t out_dim,
    uint32_t in_dim,
    HoloSphereRazorCanary64* canary
) {
    __shared__ float sm_warp4[4];
    const uint32_t r = blockIdx.x;
    if (r >= out_dim) return;
    const uint32_t tid     = threadIdx.x;
    const uint32_t n_grp32 = in_dim >> 5;
    const uint64_t* rw     = w_u64 + static_cast<uint64_t>(r) * (n_grp32 * 2u);
    const uint16_t* rs     = w_sc + static_cast<uint64_t>(r) * n_grp32;

    float lane_acc = 0.0f;
    uint32_t local_sec = 0u, local_ded = 0u;
    for (uint32_t g = tid; g < n_grp32; g += 128u) {
        uint64_t w0 = rw[2u * g + 0u];
        uint64_t w1 = rw[2u * g + 1u];
        heal_w64_dev(w0, local_sec, local_ded);
        heal_w64_dev(w1, local_sec, local_ded);
        const uint64_t u0 = w0 ^ 0x8888888888888888ULL;
        const uint64_t u1 = w1 ^ 0x8888888888888888ULL;

        int32_t ih = 0, im = 0, il = 0;
        const int32_t* xh32 = reinterpret_cast<const int32_t*>(x_hi + g * 32u);
        const int32_t* xm32 = reinterpret_cast<const int32_t*>(x_md + g * 32u);
        const int32_t* xl32 = reinterpret_cast<const int32_t*>(x_lo + g * 32u);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint64_t uw = (h == 0) ? u0 : u1;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t sub16 = static_cast<uint32_t>(uw >> (c * 16));
                const int32_t w4 = static_cast<int32_t>(
                    (sub16 & 0x0Fu) |
                    (((sub16 >> 4) & 0x0Fu) << 8) |
                    (((sub16 >> 8) & 0x0Fu) << 16) |
                    (((sub16 >> 12) & 0x0Fu) << 24)
                );
                const int idx = h * 4 + c;
                ih = dp4a_s32_dev(ih, w4, xh32[idx]);
                im = dp4a_s32_dev(im, w4, xm32[idx]);
                il = dp4a_s32_dev(il, w4, xl32[idx]);
            }
        }
        const int64_t tot = (static_cast<int64_t>(ih) << 14) + (static_cast<int64_t>(im) << 7) +
                            static_cast<int64_t>(il) - 8LL * static_cast<int64_t>(x_sumq[g]);
        lane_acc += static_cast<float>(tot) * (fp16_bits_to_f32_dev(rs[g]) * x_sc[g]);
    }
    commit_canary_counts_dev(canary, local_sec, local_ded);
    const float out_v = tree_reduce_128_sm86(lane_acc, sm_warp4);
    if (tid == 0u) y_out[r] = out_v;
}

/* 4. B=1 3.125 bpw g=128 SoA bit-planed dp4a22 GEMV with SECDED */
__global__ __launch_bounds__(128, 10)
void a16_v2_gemv_b1_3125bpw_dp4a22_sm86(
    const uint8_t* __restrict__ w_planes,
    const uint16_t* __restrict__ w_sc,
    const int8_t* __restrict__ x_hi,
    const int8_t* __restrict__ x_md,
    const int8_t* __restrict__ x_lo,
    const float* __restrict__ x_sc,
    float* __restrict__ y_out,
    uint32_t out_dim,
    uint32_t in_dim,
    HoloSphereRazorCanary64* canary
) {
    __shared__ float sm_warp4[4];
    const uint32_t r = blockIdx.x;
    if (r >= out_dim) return;
    const uint32_t tid      = threadIdx.x;
    const uint32_t n_grp128 = in_dim >> 7;
    const uint8_t* rw       = w_planes + static_cast<uint64_t>(r) * n_grp128 * 48u;
    const uint16_t* rs      = w_sc + static_cast<uint64_t>(r) * n_grp128;

    float lane_acc = 0.0f;
    uint32_t local_sec = 0u, local_ded = 0u;
    for (uint32_t g = tid; g < n_grp128; g += 128u) {
        const uint8_t* gp = rw + static_cast<uint64_t>(g) * 48u;
        const float ws    = fp16_bits_to_f32_dev(rs[g]);
        const uint32_t base32 = g * 4u;
        for (uint32_t sg = 0u; sg < 4u; ++sg) {
            const uint32_t gi = base32 + sg;
            uint32_t p0_32 = *reinterpret_cast<const uint32_t*>(gp + sg * 4u);
            const uint32_t p1_32 = *reinterpret_cast<const uint32_t*>(gp + 16u + sg * 4u);
            const uint32_t p2_32 = *reinterpret_cast<const uint32_t*>(gp + 32u + sg * 4u);
            uint32_t h0 = p0_32 & 0xFFFFu, h1 = (p0_32 >> 16) & 0xFFFFu;
            heal_u16_dev(h0, local_sec, local_ded);
            heal_u16_dev(h1, local_sec, local_ded);
            p0_32 = h0 | (h1 << 16);

            int32_t ih = 0, im = 0, il = 0;
            const int32_t* xh32 = reinterpret_cast<const int32_t*>(x_hi + gi * 32u);
            const int32_t* xm32 = reinterpret_cast<const int32_t*>(x_md + gi * 32u);
            const int32_t* xl32 = reinterpret_cast<const int32_t*>(x_lo + gi * 32u);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const uint32_t sh = static_cast<uint32_t>(c * 4);
                const int32_t w4  = static_cast<int32_t>(unpack4_3bpw_s8_dev(p0_32 >> sh, p1_32 >> sh, p2_32 >> sh));
                ih = dp4a_s32_dev(ih, w4, xh32[c]);
                im = dp4a_s32_dev(im, w4, xm32[c]);
                il = dp4a_s32_dev(il, w4, xl32[c]);
            }
            const int64_t tot = (static_cast<int64_t>(ih) << 14) + (static_cast<int64_t>(im) << 7) + static_cast<int64_t>(il);
            lane_acc += static_cast<float>(tot) * (ws * x_sc[gi]);
        }
    }
    commit_canary_counts_dev(canary, local_sec, local_ded);
    const float out_v = tree_reduce_128_sm86(lane_acc, sm_warp4);
    if (tid == 0u) y_out[r] = out_v;
}

/* 5. Batched B <= 16 IMMA K=64 (m16n8k32) 4.5 bpw GEMM with SECDED */
__global__ __launch_bounds__(128, 10)
void a16_v2_gemm_b16_imma_k64_sm86(
    const uint64_t* __restrict__ w_u64,
    const uint16_t* __restrict__ w_sc,
    const int8_t* __restrict__ x_hi,
    const int8_t* __restrict__ x_md,
    const int8_t* __restrict__ x_lo,
    const float* __restrict__ x_sc,
    float* __restrict__ y_out,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t B,
    HoloSphereRazorCanary64* canary
) {
    __shared__ int8_t sm_w[64][64];
    __shared__ float  sm_sc[64][2];
    __shared__ int8_t sm_xh[16][64];
    __shared__ int8_t sm_xm[16][64];
    __shared__ int8_t sm_xl[16][64];
    __shared__ float  sm_xsc[16][2];

    const uint32_t tid = threadIdx.x;
    const uint32_t wid = tid >> 5, lane = tid & 31u, gid = lane >> 2, tid4 = lane & 3u;
    const uint32_t m_base  = blockIdx.x * 64u;
    const uint32_t n_grp32 = in_dim >> 5;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
    uint32_t local_sec = 0u, local_ded = 0u;

    for (uint32_t g0 = 0u; g0 < n_grp32; g0 += 2u) {
        const uint32_t mr = tid >> 1, gs = tid & 1u, row_idx = m_base + mr;
        if (row_idx < out_dim) {
            const uint32_t g = g0 + gs;
            ulonglong2 ww = __ldg(reinterpret_cast<const ulonglong2*>(
                w_u64 + static_cast<uint64_t>(row_idx) * (n_grp32 * 2u)) + g);
            uint64_t w0 = static_cast<uint64_t>(ww.x), w1 = static_cast<uint64_t>(ww.y);
            heal_w64_dev(w0, local_sec, local_ded);
            heal_w64_dev(w1, local_sec, local_ded);
            sm_sc[mr][gs] = fp16_bits_to_f32_dev(__ldg(&w_sc[static_cast<uint64_t>(row_idx) * n_grp32 + g]));
            const uint64_t u0 = w0 ^ 0x8888888888888888ULL;
            const uint64_t u1 = w1 ^ 0x8888888888888888ULL;
            uint32_t* dst32 = reinterpret_cast<uint32_t*>(&sm_w[mr][gs * 32u]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst32[k]     = unpack4_s8_nib_dev(static_cast<uint32_t>(u0 >> (k * 16)));
                dst32[4 + k] = unpack4_s8_nib_dev(static_cast<uint32_t>(u1 >> (k * 16)));
            }
        } else {
            sm_sc[mr][gs] = 0.0f;
            uint32_t* dst32 = reinterpret_cast<uint32_t*>(&sm_w[mr][gs * 32u]);
#pragma unroll
            for (int k = 0; k < 8; ++k) dst32[k] = 0u;
        }
        {
            const uint32_t bb = tid >> 3, wk = (tid & 7u) * 8u;
            if (bb < B) {
                const uint64_t off = static_cast<uint64_t>(bb) * in_dim + g0 * 32u + wk;
                *reinterpret_cast<uint2*>(&sm_xh[bb][wk]) = *reinterpret_cast<const uint2*>(x_hi + off);
                *reinterpret_cast<uint2*>(&sm_xm[bb][wk]) = *reinterpret_cast<const uint2*>(x_md + off);
                *reinterpret_cast<uint2*>(&sm_xl[bb][wk]) = *reinterpret_cast<const uint2*>(x_lo + off);
                if (wk == 0u) {
                    sm_xsc[bb][0] = __ldg(&x_sc[static_cast<uint64_t>(bb) * n_grp32 + g0]);
                    sm_xsc[bb][1] = __ldg(&x_sc[static_cast<uint64_t>(bb) * n_grp32 + g0 + 1u]);
                }
            } else {
                *reinterpret_cast<uint2*>(&sm_xh[bb][wk]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&sm_xm[bb][wk]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&sm_xl[bb][wk]) = make_uint2(0u, 0u);
                if (wk == 0u) { sm_xsc[bb][0] = 0.0f; sm_xsc[bb][1] = 0.0f; }
            }
        }
        __syncthreads();
        const uint32_t r0 = wid * 16u + gid;
#pragma unroll
        for (int gs = 0; gs < 2; ++gs) {
            const uint32_t ko = gs * 32u;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sm_w[r0][ko + tid4 * 4u]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sm_w[r0 + 8u][ko + tid4 * 4u]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sm_w[r0][ko + 16u + tid4 * 4u]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sm_w[r0 + 8u][ko + 16u + tid4 * 4u]);
            int32_t ih0 = 0, ih1 = 0, ih2 = 0, ih3 = 0;
            int32_t im0 = 0, im1 = 0, im2 = 0, im3 = 0;
            int32_t il0 = 0, il1 = 0, il2 = 0, il3 = 0;
            mma_m16n8k32_s8_dev(ih0, ih1, ih2, ih3, a0, a1, a2, a3,
                *reinterpret_cast<const uint32_t*>(&sm_xh[gid][ko + tid4 * 4u]),
                *reinterpret_cast<const uint32_t*>(&sm_xh[gid][ko + 16u + tid4 * 4u]));
            mma_m16n8k32_s8_dev(im0, im1, im2, im3, a0, a1, a2, a3,
                *reinterpret_cast<const uint32_t*>(&sm_xm[gid][ko + tid4 * 4u]),
                *reinterpret_cast<const uint32_t*>(&sm_xm[gid][ko + 16u + tid4 * 4u]));
            mma_m16n8k32_s8_dev(il0, il1, il2, il3, a0, a1, a2, a3,
                *reinterpret_cast<const uint32_t*>(&sm_xl[gid][ko + tid4 * 4u]),
                *reinterpret_cast<const uint32_t*>(&sm_xl[gid][ko + 16u + tid4 * 4u]));
            const float sc0 = sm_sc[r0][gs], sc1 = sm_sc[r0 + 8u][gs];
            const float xsc0 = sm_xsc[tid4 * 2u][gs], xsc1 = sm_xsc[tid4 * 2u + 1u][gs];
            d0 += static_cast<float>((ih0 << 14) + (im0 << 7) + il0) * (sc0 * xsc0);
            d1 += static_cast<float>((ih1 << 14) + (im1 << 7) + il1) * (sc0 * xsc1);
            d2 += static_cast<float>((ih2 << 14) + (im2 << 7) + il2) * (sc1 * xsc0);
            d3 += static_cast<float>((ih3 << 14) + (im3 << 7) + il3) * (sc1 * xsc1);
            if (B > 8u) {
                const uint32_t c2 = gid + 8u;
                ih0 = ih1 = ih2 = ih3 = im0 = im1 = im2 = im3 = il0 = il1 = il2 = il3 = 0;
                mma_m16n8k32_s8_dev(ih0, ih1, ih2, ih3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(&sm_xh[c2][ko + tid4 * 4u]),
                    *reinterpret_cast<const uint32_t*>(&sm_xh[c2][ko + 16u + tid4 * 4u]));
                mma_m16n8k32_s8_dev(im0, im1, im2, im3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(&sm_xm[c2][ko + tid4 * 4u]),
                    *reinterpret_cast<const uint32_t*>(&sm_xm[c2][ko + 16u + tid4 * 4u]));
                mma_m16n8k32_s8_dev(il0, il1, il2, il3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(&sm_xl[c2][ko + tid4 * 4u]),
                    *reinterpret_cast<const uint32_t*>(&sm_xl[c2][ko + 16u + tid4 * 4u]));
                const float xsc2 = sm_xsc[8u + tid4 * 2u][gs], xsc3 = sm_xsc[8u + tid4 * 2u + 1u][gs];
                d4 += static_cast<float>((ih0 << 14) + (im0 << 7) + il0) * (sc0 * xsc2);
                d5 += static_cast<float>((ih1 << 14) + (im1 << 7) + il1) * (sc0 * xsc3);
                d6 += static_cast<float>((ih2 << 14) + (im2 << 7) + il2) * (sc1 * xsc2);
                d7 += static_cast<float>((ih3 << 14) + (im3 << 7) + il3) * (sc1 * xsc3);
            }
        }
        __syncthreads();
    }
    commit_canary_counts_dev(canary, local_sec, local_ded);

    const uint32_t r0 = m_base + wid * 16u + gid, r1 = r0 + 8u, c0 = tid4 * 2u, c1 = c0 + 1u;
    if (r0 < out_dim && c0 < B) y_out[static_cast<uint64_t>(c0) * out_dim + r0] = d0;
    if (r0 < out_dim && c1 < B) y_out[static_cast<uint64_t>(c1) * out_dim + r0] = d1;
    if (r1 < out_dim && c0 < B) y_out[static_cast<uint64_t>(c0) * out_dim + r1] = d2;
    if (r1 < out_dim && c1 < B) y_out[static_cast<uint64_t>(c1) * out_dim + r1] = d3;
    if (B > 8u) {
        if (r0 < out_dim && (c0 + 8u) < B) y_out[static_cast<uint64_t>(c0 + 8u) * out_dim + r0] = d4;
        if (r0 < out_dim && (c1 + 8u) < B) y_out[static_cast<uint64_t>(c1 + 8u) * out_dim + r0] = d5;
        if (r1 < out_dim && (c0 + 8u) < B) y_out[static_cast<uint64_t>(c0 + 8u) * out_dim + r1] = d6;
        if (r1 < out_dim && (c1 + 8u) < B) y_out[static_cast<uint64_t>(c1 + 8u) * out_dim + r1] = d7;
    }
}

/* 6. Split-K BF16-pair Tensor Core GEMM (mma.sync.aligned.m16n8k16) */
__global__ __launch_bounds__(128, 10)
void a16_v2_gemm_b16_bf16_splitk_mma_sm86(
    const uint16_t* __restrict__ w_bf16,
    const uint16_t* __restrict__ x_hi_bf16,
    const uint16_t* __restrict__ x_lo_bf16,
    float* __restrict__ y_out,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t B,
    uint32_t split_k
) {
    __shared__ uint16_t sm_w[64][32];
    __shared__ uint16_t sm_xh[16][32];
    __shared__ uint16_t sm_xl[16][32];
    const uint32_t tid = threadIdx.x, wid = tid >> 5, lane = tid & 31u, gid = lane >> 2, tid4 = lane & 3u;
    const uint32_t m_base = blockIdx.x * 64u, sk = blockIdx.y;
    const uint32_t k_tiles = in_dim >> 5;
    const uint32_t t_per_sk = (k_tiles + split_k - 1u) / split_k;
    const uint32_t t0 = sk * t_per_sk, t1 = min(k_tiles, t0 + t_per_sk);
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;

    for (uint32_t kt = t0; kt < t1; ++kt) {
        const uint32_t k_off = kt * 32u;
        const uint32_t mr = tid >> 1, ks = (tid & 1u) * 16u, row = m_base + mr;
        if (row < out_dim) {
            *reinterpret_cast<uint4*>(&sm_w[mr][ks]) = __ldg(reinterpret_cast<const uint4*>(
                w_bf16 + static_cast<uint64_t>(row) * in_dim + k_off + ks));
            *reinterpret_cast<uint4*>(&sm_w[mr][ks + 8u]) = __ldg(reinterpret_cast<const uint4*>(
                w_bf16 + static_cast<uint64_t>(row) * in_dim + k_off + ks + 8u));
        } else {
            *reinterpret_cast<uint4*>(&sm_w[mr][ks]) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&sm_w[mr][ks + 8u]) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < 64u) {
            const uint32_t bb = tid >> 2, wk = (tid & 3u) * 8u;
            if (bb < B) {
                *reinterpret_cast<uint4*>(&sm_xh[bb][wk]) = __ldg(reinterpret_cast<const uint4*>(
                    x_hi_bf16 + static_cast<uint64_t>(bb) * in_dim + k_off + wk));
                *reinterpret_cast<uint4*>(&sm_xl[bb][wk]) = __ldg(reinterpret_cast<const uint4*>(
                    x_lo_bf16 + static_cast<uint64_t>(bb) * in_dim + k_off + wk));
            } else {
                *reinterpret_cast<uint4*>(&sm_xh[bb][wk]) = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(&sm_xl[bb][wk]) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();
        const uint32_t r0 = wid * 16u + gid;
#pragma unroll 1
        for (uint32_t ks_i = 0u; ks_i < 2u; ++ks_i) {
            const uint32_t k0 = ks_i * 16u + tid4 * 2u, k1 = k0 + 8u;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sm_w[r0][k0]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sm_w[r0 + 8u][k0]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sm_w[r0][k1]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sm_w[r0 + 8u][k1]);
            mma_m16n8k16_bf16_dev(d0, d1, d2, d3, a0, a1, a2, a3,
                *reinterpret_cast<const uint32_t*>(&sm_xh[gid][k0]),
                *reinterpret_cast<const uint32_t*>(&sm_xh[gid][k1]));
            mma_m16n8k16_bf16_dev(d0, d1, d2, d3, a0, a1, a2, a3,
                *reinterpret_cast<const uint32_t*>(&sm_xl[gid][k0]),
                *reinterpret_cast<const uint32_t*>(&sm_xl[gid][k1]));
            if (B > 8u) {
                const uint32_t c2 = gid + 8u;
                mma_m16n8k16_bf16_dev(d4, d5, d6, d7, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(&sm_xh[c2][k0]),
                    *reinterpret_cast<const uint32_t*>(&sm_xh[c2][k1]));
                mma_m16n8k16_bf16_dev(d4, d5, d6, d7, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(&sm_xl[c2][k0]),
                    *reinterpret_cast<const uint32_t*>(&sm_xl[c2][k1]));
            }
        }
        __syncthreads();
    }
    const uint32_t r0 = m_base + wid * 16u + gid, r1 = r0 + 8u, c0 = tid4 * 2u, c1 = c0 + 1u;
    auto write_val = [&](uint32_t r, uint32_t c, float v) {
        if (r < out_dim && c < B) {
            float* dst = y_out + static_cast<uint64_t>(c) * out_dim + r;
            if (split_k == 1u) *dst = v;
            else atomicAdd(dst, v);
        }
    };
    write_val(r0, c0, d0); write_val(r0, c1, d1); write_val(r1, c0, d2); write_val(r1, c1, d3);
    if (B > 8u) {
        write_val(r0, c0 + 8u, d4); write_val(r0, c1 + 8u, d5);
        write_val(r1, c0 + 8u, d6); write_val(r1, c1 + 8u, d7);
    }
}

/* 7. Rank-8 Cayley-Woodbury inverse M = I + alpha * V * U^T -> 256 B c_wb_inv_256b */
__global__ __launch_bounds__(128, 10)
void a16_v2_woodbury_inv8_sm86(
    const float* __restrict__ U8,
    const float* __restrict__ V8,
    float* __restrict__ out_inv64,
    uint32_t dim,
    float alpha
) {
    __shared__ float sm_aug[8][16];
    __shared__ float sm_warp4[4];
    const uint32_t tid = threadIdx.x;

    for (uint32_t r = 0u; r < 8u; ++r) {
        for (uint32_t c = 0u; c < 8u; ++c) {
            float partial = 0.0f;
            for (uint32_t d = tid; d < dim; d += 128u) {
                partial += V8[static_cast<uint64_t>(r) * dim + d] * U8[static_cast<uint64_t>(c) * dim + d];
            }
            const float dot = tree_reduce_128_sm86(partial, sm_warp4);
            if (tid == 0u) {
                sm_aug[r][c] = ((r == c) ? 1.0f : 0.0f) + alpha * dot;
                sm_aug[r][c + 8u] = (r == c) ? 1.0f : 0.0f;
            }
            __syncthreads();
        }
    }
    if (tid == 0u) {
        for (int col = 0; col < 8; ++col) {
            int piv = col;
            float best = fabsf(sm_aug[col][col]);
            for (int r = col + 1; r < 8; ++r) {
                const float v = fabsf(sm_aug[r][col]);
                if (v > best) { best = v; piv = r; }
            }
            if (piv != col) {
                for (int c = 0; c < 16; ++c) {
                    const float tmp = sm_aug[col][c];
                    sm_aug[col][c] = sm_aug[piv][c];
                    sm_aug[piv][c] = tmp;
                }
            }
            const float inv_d = 1.0f / fmaxf(fabsf(sm_aug[col][col]), 1.0e-12f) *
                                (sm_aug[col][col] >= 0.0f ? 1.0f : -1.0f);
            for (int c = 0; c < 16; ++c) sm_aug[col][c] *= inv_d;
            for (int r = 0; r < 8; ++r) {
                if (r == col) continue;
                const float f = sm_aug[r][col];
                for (int c = 0; c < 16; ++c) sm_aug[r][c] -= f * sm_aug[col][c];
            }
        }
    }
    __syncthreads();
    if (tid < 64u) {
        out_inv64[tid] = sm_aug[tid >> 3][(tid & 7u) + 8u];
    }
}

/* 8. Hoisted 256 B c_wb_inv_256b rank-8 Cayley-Woodbury apply */
__global__ __launch_bounds__(128, 10)
void a16_v2_cayley_woodbury_apply_sm86(
    float* __restrict__ h_io,
    const float* __restrict__ U8,
    const float* __restrict__ V8,
    uint32_t dim,
    float alpha
) {
    __shared__ float sm_z[8];
    __shared__ float sm_y[8];
    __shared__ float sm_warp4[4];
    const uint32_t tid = threadIdx.x;

    for (uint32_t r = 0u; r < 8u; ++r) {
        float acc = 0.0f;
        for (uint32_t d = tid; d < dim; d += 128u) {
            acc += V8[static_cast<uint64_t>(r) * dim + d] * h_io[d];
        }
        const float dot = tree_reduce_128_sm86(acc, sm_warp4);
        if (tid == 0u) sm_z[r] = dot;
        __syncthreads();
    }
    if (tid < 8u) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) s += c_wb_inv_256b[tid * 8u + c] * sm_z[c];
        sm_y[tid] = s;
    }
    __syncthreads();
    for (uint32_t d = tid; d < dim; d += 128u) {
        float delta = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) delta += U8[static_cast<uint64_t>(r) * dim + d] * sm_y[r];
        h_io[d] += alpha * delta;
    }
}

/* 9. IRIS runtime canary signature check */
__global__ __launch_bounds__(128, 10)
void a16_v2_iris_canary_finalize_sm86(
    const uint32_t* __restrict__ mem_u32,
    uint32_t n_words,
    uint32_t slot_idx,
    HoloSphereRazorCanary64* canary,
    uint32_t* __restrict__ out_mismatch
) {
    __shared__ uint32_t sm_lane[4];
    const uint32_t tid = threadIdx.x;
    uint32_t h = 0x5185A17Bu ^ (tid * SILIKERNELS_V2_IRIS_WEYL_MIX);
    for (uint32_t i = tid; i < n_words; i += 128u) {
        h = iris_mix_word_dev(h, mem_u32[i]);
    }
    const uint32_t lane = tid & 31u, wid = tid >> 5;
#pragma unroll
    for (int st = 1; st <= 16; st <<= 1) {
        const uint32_t other = __shfl_xor_sync(0xffffffffu, h, st);
        h = iris_mix_word_dev(h, other);
    }
    if (lane == 0u) sm_lane[wid] = h;
    __syncthreads();
    if (tid == 0u) {
        uint32_t s = iris_mix_word_dev(sm_lane[0], sm_lane[1]);
        s = iris_mix_word_dev(s, iris_mix_word_dev(sm_lane[2], sm_lane[3]));
        const uint32_t expected = c_iris_expected_sig_u32[slot_idx & 15u];
        uint32_t mismatch = 0u;
        if (expected != 0u && s != expected) mismatch = 1u;
        if (canary != nullptr && canary->ded_count_u32 > 0u) mismatch = 1u;
        if (out_mismatch != nullptr) *out_mismatch = mismatch;
    }
}

} // namespace silikernels_v2

#endif /* SILIKERNELS_A16_V2_AMPERE_KERNELS_CUH */
