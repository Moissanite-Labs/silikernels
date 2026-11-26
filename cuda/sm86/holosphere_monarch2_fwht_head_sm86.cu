// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// Bare-Metal NVIDIA Ampere sm_86 Zero-HBM-Table Structured Pre-Mixer + Byte-Bias
// Ahead of Orthonormal FWHT_256 x bent4 LM Head (Refs Moissanite-Labs/silikernels#2):
//   Kernels (0 B stack/register spill under nvcc -O3 -arch=sm_86 --Werror all-warnings,
//   zero unions, zero reinterpret_cast, raw byte-backed memcpy SMEM tiles):
//     1. holosphere_arm2b_bias_514b_sm86            : 2 B scalar + 512 B FP16 byte bias (514 B)
//     2. holosphere_arm5_bias_2p5kib_sm86           : 2 KiB Walsh conv + 512 B FP16 bias (2.5 KiB)
//     3. holosphere_arm7_bias_8p5kib_sm86           : 8 KiB 1-stage Monarch-16x16 + 512 B bias (8.5 KiB)
//     4. holosphere_arm9_monarch2_bias_16p5kib_sm86 : 16 KiB 2-stage Monarch-16x16 + 512 B bias (16.5 KiB,
//                                                     100% CTA SMEM-resident in 20,480 B, 0 B HBM vocab table)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace holosphere_monarch2_head {

__device__ __forceinline__ float fwht256_plain_8warp(
    float v, int warp_id, int lane, float (*smem)[32]
) {
    if (warp_id < 8) {
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem[warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[warp_id ^ 1][lane];
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem[8 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[8 + (warp_id ^ 2)][lane];
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem[warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem[warp_id ^ 4][lane];
        v = (warp_id & 4) ? (o - v) : (v + o);
        v *= 0.0625f;
    }
    return v;
}

__device__ __forceinline__ float fwht256_bent4_from_smem_8warp(
    const float* smem_in, int warp_id, int lane, float (*smem_buf)[32]
) {
    float v = 0.0f;
    if (warp_id < 8) {
        int src_w   = warp_id ^ (lane & 7);
        int src_idx = (src_w << 5) | lane;
        float u_val = smem_in[src_idx];
        int b4      = __popc((lane & 3) & ((lane >> 2) & 3)) & 1;
        v = b4 ? -u_val : u_val;
        #pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1) {
            float o = __shfl_xor_sync(0xFFFFFFFFu, v, mask);
            v = (lane & mask) ? (o - v) : (v + o);
        }
        smem_buf[warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_buf[warp_id ^ 1][lane];
        v = (warp_id & 1) ? (o - v) : (v + o);
        smem_buf[8 + warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_buf[8 + (warp_id ^ 2)][lane];
        v = (warp_id & 2) ? (o - v) : (v + o);
        smem_buf[warp_id][lane] = v;
    }
    __syncthreads();
    if (warp_id < 8) {
        float o = smem_buf[warp_id ^ 4][lane];
        v = (warp_id & 4) ? (o - v) : (v + o);
        v *= 0.17677669529663688f;
    }
    return v;
}

// 16-warp parallel 16x16 Monarch stage with implicit stride-16 transpose P_{16x16}
// Reads raw byte-backed B_raw[g][row][0..7] as __half2 via memcpy (zero union/aliasing UB),
// multiplies by z_in[c*16 + g], and writes z_out[g*16 + row].
__device__ __forceinline__ void monarch16_transpose_matvec_16warp(
    const unsigned char* B_raw,
    const float* z_in,
    float* z_out,
    int warp_id,
    int lane
) {
    int g      = warp_id;
    int row    = lane & 15;
    int half_k = lane >> 4;
    int c0     = half_k << 3;
    float acc  = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int h2_idx = ((g * 16 + row) * 8) + ((half_k << 2) + j);
        __half2 wh2;
        std::memcpy(&wh2, B_raw + (h2_idx * 4), sizeof(__half2));
        float2 w2 = __half22float2(wh2);
        float  x0 = z_in[((c0 + (j << 1) + 0) << 4) + g];
        float  x1 = z_in[((c0 + (j << 1) + 1) << 4) + g];
        acc = fmaf(w2.x, x0, fmaf(w2.y, x1, acc));
    }
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, 16);
    if (half_k == 0) {
        z_out[(g << 4) + row] = acc;
    }
}

// Kernel 1: Scalar + 512 B FP16 Byte-Unigram Bias (514 B total, 0 B HBM table) <<<B, 256>>>
__global__ __launch_bounds__(256, 2)
void holosphere_arm2b_bias_514b_sm86(
    const float* __restrict__ x_in,
    const __half* __restrict__ scale_f16,
    const __half* __restrict__ bias_f16,
    float* __restrict__ logits_out
) {
    __shared__ float smem_u[256];
    __shared__ float smem_buf[16][32];

    int b_idx   = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    smem_u[tid] = __ldg(&x_in[b_idx * 512 + tid]) - __ldg(&x_in[b_idx * 512 + 256 + tid]);
    __syncthreads();

    float logit = fwht256_bent4_from_smem_8warp(smem_u, warp_id, lane, smem_buf);
    float s     = __half2float(__ldg(&scale_f16[0]));
    float b     = __half2float(__ldg(&bias_f16[tid]));
    logits_out[b_idx * 256 + tid] = fmaf(logit, s, b);
}

// Kernel 2: 2 KiB Dyadic Walsh Conv + 512 B FP16 Bias (2.5 KiB total, 0 B HBM table) <<<B, 256>>>
__global__ __launch_bounds__(256, 2)
void holosphere_arm5_bias_2p5kib_sm86(
    const float* __restrict__ x_in,
    const __half2* __restrict__ ab_f16,
    const __half2* __restrict__ d12_f16,
    const __half* __restrict__ bias_f16,
    float* __restrict__ logits_out
) {
    __shared__ float smem_z2[256];
    __shared__ float smem_buf[16][32];

    int b_idx   = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    float lo   = __ldg(&x_in[b_idx * 512 + tid]);
    float hi   = __ldg(&x_in[b_idx * 512 + 256 + tid]);
    float2 ab  = __half22float2(__ldg(&ab_f16[tid]));
    float2 d12 = __half22float2(__ldg(&d12_f16[tid]));
    float b    = __half2float(__ldg(&bias_f16[tid]));

    float u  = fmaf(ab.x, lo, ab.y * hi);
    float z1 = fwht256_plain_8warp(u, warp_id, lane, smem_buf) * d12.x;
    __syncthreads();
    float z2 = fwht256_plain_8warp(z1, warp_id, lane, smem_buf) * d12.y;
    smem_z2[tid] = z2;
    __syncthreads();

    float logit = fwht256_bent4_from_smem_8warp(smem_z2, warp_id, lane, smem_buf);
    logits_out[b_idx * 256 + tid] = logit + b;
}

// Kernel 3: 8 KiB 1-Stage Monarch-16x16 + 512 B FP16 Bias (8.5 KiB total, 0 B HBM table) <<<B, 512>>>
__global__ __launch_bounds__(512, 1)
void holosphere_arm7_bias_8p5kib_sm86(
    const float* __restrict__ x_in,
    const uint4* __restrict__ B1_u4,
    const __half* __restrict__ bias_f16,
    float* __restrict__ logits_out
) {
    __shared__ alignas(16) unsigned char smem_B1_raw[8192];
    __shared__ float smem_z1[256];
    __shared__ float smem_z3[256];
    __shared__ float smem_buf[16][32];

    int b_idx   = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    uint4 b1_v = __ldg(&B1_u4[tid]);
    std::memcpy(&smem_B1_raw[tid * 16], &b1_v, sizeof(uint4));

    float u = 0.0f;
    float b = 0.0f;
    if (warp_id < 8) {
        u = __ldg(&x_in[b_idx * 512 + tid]) - __ldg(&x_in[b_idx * 512 + 256 + tid]);
        b = __half2float(__ldg(&bias_f16[tid]));
    }

    float z1 = fwht256_plain_8warp(u, warp_id, lane, smem_buf);
    if (warp_id < 8) smem_z1[tid] = z1;
    __syncthreads();

    monarch16_transpose_matvec_16warp(smem_B1_raw, smem_z1, smem_z3, warp_id, lane);
    __syncthreads();

    float logit = fwht256_bent4_from_smem_8warp(smem_z3, warp_id, lane, smem_buf);
    if (warp_id < 8) {
        logits_out[b_idx * 256 + tid] = logit + b;
    }
}

// Kernel 4: 16 KiB 2-Stage Monarch-16x16 + 512 B FP16 Bias (16.5 KiB total, 0 B HBM table) <<<B, 512>>>
//   Full 2-stage Monarch B_2 P_{16x16} B_1 P_{16x16} H_256(u) -> FWHT_256 x bent4 + bias
//   100% CTA SMEM-resident (20,480 B SMEM); loads all 16 KiB of B_1 + B_2 in 2 LDG.128/thread!
__global__ __launch_bounds__(512, 1)
void holosphere_arm9_monarch2_bias_16p5kib_sm86(
    const float* __restrict__ x_in,
    const uint4* __restrict__ B1_u4,        // [512] uint4 = 8,192 B
    const uint4* __restrict__ B2_u4,        // [512] uint4 = 8,192 B
    const __half* __restrict__ bias_f16,    // [256] __half = 512 B
    float* __restrict__ logits_out
) {
    __shared__ alignas(16) unsigned char smem_B1_raw[8192]; // 8,192 B
    __shared__ alignas(16) unsigned char smem_B2_raw[8192]; // 8,192 B
    __shared__ float smem_z1[256];                          // 1,024 B (reused for z5)
    __shared__ float smem_z3[256];                          // 1,024 B
    __shared__ float smem_buf[16][32];                      // 2,048 B

    int b_idx   = blockIdx.x;
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;

    uint4 b1_v = __ldg(&B1_u4[tid]);
    uint4 b2_v = __ldg(&B2_u4[tid]);
    std::memcpy(&smem_B1_raw[tid * 16], &b1_v, sizeof(uint4));
    std::memcpy(&smem_B2_raw[tid * 16], &b2_v, sizeof(uint4));

    float u = 0.0f;
    float b = 0.0f;
    if (warp_id < 8) {
        u = __ldg(&x_in[b_idx * 512 + tid]) - __ldg(&x_in[b_idx * 512 + 256 + tid]);
        b = __half2float(__ldg(&bias_f16[tid]));
    }

    float z1 = fwht256_plain_8warp(u, warp_id, lane, smem_buf);
    if (warp_id < 8) smem_z1[tid] = z1;
    __syncthreads();

    // Stage 1: z3 = B_1 @ P_{16x16} @ z1
    monarch16_transpose_matvec_16warp(smem_B1_raw, smem_z1, smem_z3, warp_id, lane);
    __syncthreads();

    // Stage 2: z5 = B_2 @ P_{16x16} @ z3 (reusing smem_z1 as z5)
    monarch16_transpose_matvec_16warp(smem_B2_raw, smem_z3, smem_z1, warp_id, lane);
    __syncthreads();

    float logit = fwht256_bent4_from_smem_8warp(smem_z1, warp_id, lane, smem_buf);
    if (warp_id < 8) {
        logits_out[b_idx * 256 + tid] = logit + b;
    }
}

} // namespace holosphere_monarch2_head
