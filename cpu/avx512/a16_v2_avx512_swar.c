/*
 * a16_v2_avx512_swar.c — V2 Intel AVX-512 (B*=2 Cold Crossover) + Carry-Free SWAR127 Kernel
 *
 * Features:
 *   - Carry-free [1..127] SWAR127 domain-guarded two's-complement negation
 *   - 0-ULP canonical 128-lane binary tree reduction matching CUDA sm_86 bit-for-bit
 *   - 22-bit radix-128 3-limb (dpbusd22) H_4 activation quantizer
 *   - AVX-512 VNNI + VPOPCNTDQ 4.5 bpw (18 B/32w) & 3.125 bpw (50 B/128w SoA) GEMV
 *     with portable C11 fallback when compiled without AVX-512 target flags
 *   - Live Extended Hamming [16, 11, 4] SECDED SEC repair & fail-closed DED trap
 *   - Persistent 8-thread deterministic pthread cold-expert pool (B < B* = 2 / 3)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#define SILIKERNELS_HAS_AVX512 1
#include <immintrin.h>
#else
#define SILIKERNELS_HAS_AVX512 0
#endif

#include "../../include/silikernels_v2_contracts.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SILIK_V2_PHYS_CORES 8

static inline float silik_v2_fp16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u = 0u;
    if (exp == 0u) {
        if (mant == 0u) {
            u = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1u; --e; }
            mant &= 0x03FFu;
            u = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t silik_v2_f32_to_fp16(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mant = bits & 0x007FFFFFu;
    const int32_t exp = (int32_t)((bits >> 23) & 0xFFu);
    if (exp == 255) {
        return (uint16_t)(sign | (mant != 0u ? (0x7E00u | (mant >> 13)) : 0x7C00u));
    }
    int32_t ne = exp - 127 + 15;
    if (ne >= 31) return (uint16_t)(sign | 0x7C00u);
    if (ne <= 0) {
        if (ne < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const uint32_t sh = (uint32_t)(14 - ne);
        uint32_t hm = mant >> sh;
        const uint32_t rem = mant & ((1u << sh) - 1u);
        const uint32_t mid = 1u << (sh - 1u);
        if (rem > mid || (rem == mid && (hm & 1u))) ++hm;
        return (uint16_t)(sign | hm);
    }
    uint32_t hm = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        ++hm;
        if (hm == 0x0400u) {
            hm = 0u;
            ++ne;
            if (ne >= 31) return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)ne << 10) | hm);
}

int silikernels_v2_swar127_in_domain_u64(uint64_t x) {
    return silikernels_v2_swar127_in_domain_u64_c(x);
}

uint64_t silikernels_v2_swar127_negate_u64(uint64_t x) {
    return silikernels_v2_swar127_negate_u64_c(x);
}

int silikernels_v2_swar127_negate_u64_checked(uint64_t x, uint64_t* out) {
    return silikernels_v2_swar127_negate_u64_checked_c(x, out);
}

float silikernels_v2_canonical_tree_reduce_128(const float* xs) {
    return silikernels_v2_canonical_tree_reduce_128_c(xs);
}

int silikernels_v2_should_dispatch_cpu(uint32_t batch_size, uint32_t is_45bpw) {
    const uint32_t b_star = (is_45bpw != 0u)
        ? SILIKERNELS_V2_B_STAR_COLD_45BPW
        : SILIKERNELS_V2_B_STAR_COLD_3BPW;
    return (batch_size > 0u && batch_size < b_star) ? 1 : 0;
}

static void silik_v2_quant_act22_scalar_groups(
    const float* x,
    int g0,
    int g1,
    SilikernelsV2ActQ22* out
) {
    for (int g = g0; g < g1; ++g) {
        float h4x[32];
        float amax = 0.0f;
        const float* xg = x + g * 32;
        for (int b = 0; b < 32; b += 4) {
            const float a0 = xg[b + 0], a1 = xg[b + 1], a2 = xg[b + 2], a3 = xg[b + 3];
            const float s01 = a0 + a1, d01 = a0 - a1, s23 = a2 + a3, d23 = a2 - a3;
            const float t0 = 0.5f * (s01 + s23);
            const float t1 = 0.5f * (d01 + d23);
            const float t2 = 0.5f * (s01 - s23);
            const float t3 = 0.5f * (d01 - d23);
            h4x[b + 0] = t0; h4x[b + 1] = t1; h4x[b + 2] = t2; h4x[b + 3] = t3;
            const float m0 = fabsf(t0), m1 = fabsf(t1), m2 = fabsf(t2), m3 = fabsf(t3);
            if (m0 > amax) amax = m0;
            if (m1 > amax) amax = m1;
            if (m2 > amax) amax = m2;
            if (m3 > amax) amax = m3;
        }
        const float sc = fmaxf(amax / 2064384.0f, 1.0e-12f);
        out->x_sc[g] = sc;
        int32_t qsum = 0;
        for (int k = 0; k < 32; ++k) {
            const int32_t q = (int32_t)lrintf(h4x[k] / sc);
            const int32_t q_hi = (q >= 0) ? ((q + 8192) >> 14) : -((-q + 8192) >> 14);
            const int32_t r1 = q - (q_hi * 16384);
            const int32_t q_md = (r1 >= 0) ? ((r1 + 64) >> 7) : -((-r1 + 64) >> 7);
            const int32_t q_lo = r1 - (q_md * 128);
            const int idx = g * 32 + k;
            out->x_hi[idx] = (int8_t)q_hi;
            out->x_md[idx] = (int8_t)q_md;
            out->x_lo[idx] = (int8_t)q_lo;
            qsum += q;
        }
        out->x_sumq[g] = qsum;
    }
}

#if SILIKERNELS_HAS_AVX512
static inline __m512i silik_v2_dpbusd_u8_s8(__m512i acc, __m512i u_u8, __m512i x_s8) {
#ifdef __AVX512VNNI__
    return _mm512_dpbusd_epi32(acc, u_u8, x_s8);
#else
    const __m512i prod = _mm512_maddubs_epi16(u_u8, x_s8);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(prod, _mm512_set1_epi16(1)));
#endif
}

static inline uint32_t silik_v2_check_p0_avx512(const uint8_t* p0_16) {
#ifdef __AVX512VPOPCNTDQ__
    const __m128i raw16 = _mm_loadu_si128((const __m128i*)(const void*)p0_16);
    const __m512i blk8  = _mm512_cvtepu16_epi64(raw16);
    const __m512i one64 = _mm512_set1_epi64(1LL);
    const __m512i c0 = _mm512_and_si512(_mm512_popcnt_epi64(blk8), one64);
    const __m512i c1 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(blk8, _mm512_set1_epi64(0xAAAAu))), one64);
    const __m512i c2 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(blk8, _mm512_set1_epi64(0xCCCCu))), one64);
    const __m512i c3 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(blk8, _mm512_set1_epi64(0xF0F0u))), one64);
    const __m512i c4 = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(blk8, _mm512_set1_epi64(0xFF00u))), one64);
    const __m512i any = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(c0, c1), _mm512_or_si512(c2, c3)), c4);
    return (uint32_t)_mm512_test_epi64_mask(any, one64);
#else
    uint32_t m = 0u;
    for (int b = 0; b < 8; ++b) {
        uint16_t w16 = 0u;
        memcpy(&w16, p0_16 + b * 2, sizeof(w16));
        if (silikernels_v2_secded_syn_u16(w16) != 0u) m |= (1u << b);
    }
    return m;
#endif
}

static inline uint32_t silik_v2_check_w64x8_avx512(__m512i raw) {
#ifdef __AVX512VPOPCNTDQ__
    const __m512i m_lsb = _mm512_set1_epi64((long long)0x1111111111111111ULL);
    const __m512i one64 = _mm512_set1_epi64(1LL);
    const __m512i lsb = _mm512_and_si512(raw, m_lsb);
    const __m512i c0  = _mm512_and_si512(_mm512_popcnt_epi64(lsb), one64);
    const __m512i c1  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, _mm512_set1_epi64((long long)0x1010101010101010ULL))), one64);
    const __m512i c2  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, _mm512_set1_epi64((long long)0x1100110011001100ULL))), one64);
    const __m512i c3  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, _mm512_set1_epi64((long long)0x1111000011110000ULL))), one64);
    const __m512i c4  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, _mm512_set1_epi64((long long)0x1111111100000000ULL))), one64);
    const __m512i any = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(c0, c1), _mm512_or_si512(c2, c3)), c4);
    return (uint32_t)_mm512_test_epi64_mask(any, one64);
#else
    uint64_t tmp[8];
    _mm512_storeu_si512((void*)tmp, raw);
    uint32_t m = 0u;
    for (int i = 0; i < 8; ++i) {
        if (silikernels_v2_secded_syn_w64(tmp[i]) != 0u) m |= (1u << i);
    }
    return m;
#endif
}

/* Single-cycle unpack/add 4-subgroup horizontal reduction without _mm256_hadd_epi32 */
static inline __m128i silik_v2_reduce4_dp_pair(
    __m512i dh0, __m512i dm0, __m512i dl0,
    __m512i dh1, __m512i dm1, __m512i dl1
) {
    const __m512i t0 = _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(dh0, 14), _mm512_slli_epi32(dm0, 7)), dl0);
    const __m512i t1 = _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(dh1, 14), _mm512_slli_epi32(dm1, 7)), dl1);
    const __m256i a  = _mm512_castsi512_si256(t0);
    const __m256i b  = _mm512_extracti64x4_epi64(t0, 1);
    const __m256i c  = _mm512_castsi512_si256(t1);
    const __m256i d  = _mm512_extracti64x4_epi64(t1, 1);
    const __m256i ab4 = _mm256_add_epi32(_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b));
    const __m256i cd4 = _mm256_add_epi32(_mm256_unpacklo_epi32(c, d), _mm256_unpackhi_epi32(c, d));
    const __m128i ab128 = _mm_add_epi32(_mm256_castsi256_si128(ab4), _mm256_extracti128_si256(ab4, 1));
    const __m128i cd128 = _mm_add_epi32(_mm256_castsi256_si128(cd4), _mm256_extracti128_si256(cd4, 1));
    return _mm_add_epi32(_mm_unpacklo_epi64(ab128, cd128), _mm_unpackhi_epi64(ab128, cd128));
}

static inline float silik_v2_tree_reduce_96_avx(const float* lanes128, int n_grp32) {
    const __m512 v0 = _mm512_loadu_ps(lanes128 + 0);
    const __m512 v1 = _mm512_loadu_ps(lanes128 + 16);
    __m512 a0 = _mm512_add_ps(v0, _mm512_permute_ps(v0, 0xB1));
    a0 = _mm512_add_ps(a0, _mm512_permute_ps(a0, 0x4E));
    a0 = _mm512_add_ps(a0, _mm512_shuffle_f32x4(a0, a0, 0xB1));
    a0 = _mm512_add_ps(a0, _mm512_shuffle_f32x4(a0, a0, 0x4E));
    __m512 a1 = _mm512_add_ps(v1, _mm512_permute_ps(v1, 0xB1));
    a1 = _mm512_add_ps(a1, _mm512_permute_ps(a1, 0x4E));
    a1 = _mm512_add_ps(a1, _mm512_shuffle_f32x4(a1, a1, 0xB1));
    a1 = _mm512_add_ps(a1, _mm512_shuffle_f32x4(a1, a1, 0x4E));
    const float w0 = _mm_cvtss_f32(_mm512_castps512_ps128(a0)) + _mm_cvtss_f32(_mm512_castps512_ps128(a1));

    float w1 = 0.0f;
    if (n_grp32 > 32) {
        const __m512 v2 = _mm512_loadu_ps(lanes128 + 32);
        const __m512 v3 = _mm512_loadu_ps(lanes128 + 48);
        __m512 b0 = _mm512_add_ps(v2, _mm512_permute_ps(v2, 0xB1));
        b0 = _mm512_add_ps(b0, _mm512_permute_ps(b0, 0x4E));
        b0 = _mm512_add_ps(b0, _mm512_shuffle_f32x4(b0, b0, 0xB1));
        b0 = _mm512_add_ps(b0, _mm512_shuffle_f32x4(b0, b0, 0x4E));
        __m512 b1 = _mm512_add_ps(v3, _mm512_permute_ps(v3, 0xB1));
        b1 = _mm512_add_ps(b1, _mm512_permute_ps(b1, 0x4E));
        b1 = _mm512_add_ps(b1, _mm512_shuffle_f32x4(b1, b1, 0xB1));
        b1 = _mm512_add_ps(b1, _mm512_shuffle_f32x4(b1, b1, 0x4E));
        w1 = _mm_cvtss_f32(_mm512_castps512_ps128(b0)) + _mm_cvtss_f32(_mm512_castps512_ps128(b1));
    }

    float w2 = 0.0f;
    if (n_grp32 > 64) {
        const __m256 v4 = _mm256_loadu_ps(lanes128 + 64);
        __m256 c0 = _mm256_add_ps(v4, _mm256_permute_ps(v4, 0xB1));
        c0 = _mm256_add_ps(c0, _mm256_permute_ps(c0, 0x4E));
        c0 = _mm256_add_ps(c0, _mm256_permute2f128_ps(c0, c0, 1));
        w2 = _mm_cvtss_f32(_mm256_castps256_ps128(c0)) + 0.0f;
    }
    return (w0 + w1) + (w2 + 0.0f);
}
#endif /* SILIKERNELS_HAS_AVX512 */

void silikernels_v2_quant_act22_h4(const float* x, int in_dim, SilikernelsV2ActQ22* out) {
    if (!x || !out || in_dim <= 0 || (in_dim & 31) != 0 || in_dim > (int)SILIKERNELS_V2_HIDDEN_MAX) return;
    out->in_dim = in_dim;
    silik_v2_quant_act22_scalar_groups(x, 0, in_dim >> 5, out);
}

static int silik_v2_gemv_45bpw_range(
    const uint64_t* w_u64,
    const uint16_t* w_sc,
    const SilikernelsV2ActQ22* act,
    float* y_out,
    int r0,
    int r1,
    int in_dim,
    SilikernelsV2SecdedStats* stats
) {
    const int n_grp32 = in_dim >> 5;
#if SILIKERNELS_HAS_AVX512
    const __m512i bias8   = _mm512_set1_epi8(8);
    const __m512i mask0f  = _mm512_set1_epi8(0x0F);
    const __m512i zero512 = _mm512_setzero_si512();
#endif
    for (int r = r0; r < r1; ++r) {
        const uint64_t* rw = w_u64 + (size_t)r * (size_t)(n_grp32 * 2);
        const uint16_t* rs = w_sc + (size_t)r * (size_t)n_grp32;
        float lanes128[128];
        memset(lanes128, 0, sizeof(lanes128));
#if SILIKERNELS_HAS_AVX512
        int g = 0;
        for (; g + 3 < n_grp32; g += 4) {
            const __m512i raw = _mm512_loadu_si512((const void*)(rw + 2 * g));
            uint64_t wbuf[8];
            _mm512_storeu_si512((void*)wbuf, raw);
            const uint32_t err_mask = silik_v2_check_w64x8_avx512(raw);
            if (err_mask != 0u) {
                for (int b = 0; b < 8; ++b) {
                    if ((err_mask & (1u << b)) != 0u) {
                        silikernels_v2_secded_heal_w64(&wbuf[b], stats);
                    }
                }
            }
            const __m512i healed = _mm512_loadu_si512((const void*)wbuf);
            const __m512i u = _mm512_xor_si512(healed, bias8);
            const __m512i lo4 = _mm512_and_si512(u, mask0f);
            const __m512i hi4 = _mm512_and_si512(_mm512_srli_epi16(u, 4), mask0f);
            const __m512i u01 = _mm512_unpacklo_epi8(lo4, hi4);
            const __m512i u23 = _mm512_unpackhi_epi8(lo4, hi4);

            const __m512i dh01 = silik_v2_dpbusd_u8_s8(zero512, u01, _mm512_loadu_si512((const void*)(act->x_hi + g * 32)));
            const __m512i dm01 = silik_v2_dpbusd_u8_s8(zero512, u01, _mm512_loadu_si512((const void*)(act->x_md + g * 32)));
            const __m512i dl01 = silik_v2_dpbusd_u8_s8(zero512, u01, _mm512_loadu_si512((const void*)(act->x_lo + g * 32)));
            const __m512i dh23 = silik_v2_dpbusd_u8_s8(zero512, u23, _mm512_loadu_si512((const void*)(act->x_hi + (g + 2) * 32)));
            const __m512i dm23 = silik_v2_dpbusd_u8_s8(zero512, u23, _mm512_loadu_si512((const void*)(act->x_md + (g + 2) * 32)));
            const __m512i dl23 = silik_v2_dpbusd_u8_s8(zero512, u23, _mm512_loadu_si512((const void*)(act->x_lo + (g + 2) * 32)));
            const __m128i d4 = silik_v2_reduce4_dp_pair(dh01, dm01, dl01, dh23, dm23, dl23);
            int32_t dv[4];
            _mm_storeu_si128((__m128i*)dv, d4);
            for (int k = 0; k < 4; ++k) {
                const int gi = g + k;
                const int64_t tot = (int64_t)dv[k] - 8LL * (int64_t)act->x_sumq[gi];
                lanes128[gi] = (float)tot * (silik_v2_fp16_to_f32(rs[gi]) * act->x_sc[gi]);
            }
        }
        for (; g < n_grp32; ++g)
#else
        for (int g = 0; g < n_grp32; ++g)
#endif
        {
            uint64_t w0 = rw[2 * g + 0], w1 = rw[2 * g + 1];
            silikernels_v2_secded_heal_w64(&w0, stats);
            silikernels_v2_secded_heal_w64(&w1, stats);
            const uint64_t u0 = w0 ^ 0x8888888888888888ULL;
            const uint64_t u1 = w1 ^ 0x8888888888888888ULL;
            int32_t ih = 0, im = 0, il = 0;
            for (int i = 0; i < 16; ++i) {
                const int32_t wi = (int32_t)((u0 >> (4 * i)) & 0xFu);
                const int idx = g * 32 + i;
                ih += wi * (int32_t)act->x_hi[idx];
                im += wi * (int32_t)act->x_md[idx];
                il += wi * (int32_t)act->x_lo[idx];
            }
            for (int i = 0; i < 16; ++i) {
                const int32_t wi = (int32_t)((u1 >> (4 * i)) & 0xFu);
                const int idx = g * 32 + 16 + i;
                ih += wi * (int32_t)act->x_hi[idx];
                im += wi * (int32_t)act->x_md[idx];
                il += wi * (int32_t)act->x_lo[idx];
            }
            const int64_t tot = ((int64_t)ih << 14) + ((int64_t)im << 7) + (int64_t)il - 8LL * (int64_t)act->x_sumq[g];
            lanes128[g] = (float)tot * (silik_v2_fp16_to_f32(rs[g]) * act->x_sc[g]);
        }
#if SILIKERNELS_HAS_AVX512
        y_out[r] = silik_v2_tree_reduce_96_avx(lanes128, n_grp32);
#else
        y_out[r] = silikernels_v2_canonical_tree_reduce_128_c(lanes128);
#endif
    }
    return (stats && stats->safe_halt) ? -1 : 0;
}

static int silik_v2_gemv_3125bpw_range(
    const uint8_t* w_planes,
    const uint16_t* w_sc_f16,
    const SilikernelsV2ActQ22* act,
    float* y_out,
    int r0,
    int r1,
    int in_dim,
    SilikernelsV2SecdedStats* stats
) {
    const int n_grp128 = in_dim >> 7;
#if SILIKERNELS_HAS_AVX512
    const int n_grp32  = in_dim >> 5;
    const __m512i zero512 = _mm512_setzero_si512();
#endif
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row_p = w_planes + (size_t)r * (size_t)n_grp128 * 48u;
        const uint16_t* row_s = w_sc_f16 + (size_t)r * (size_t)n_grp128;
        float lanes128[128];
        memset(lanes128, 0, sizeof(lanes128));
        for (int g = 0; g < n_grp128; ++g) {
            const uint8_t* gp = row_p + (size_t)g * 48u;
            uint8_t p0[16], p1[16], p2[16];
            memcpy(p0, gp + 0, 16);
            memcpy(p1, gp + 16, 16);
            memcpy(p2, gp + 32, 16);
#if SILIKERNELS_HAS_AVX512
            const uint32_t err_mask = silik_v2_check_p0_avx512(p0);
#else
            uint32_t err_mask = 0u;
            for (int b = 0; b < 8; ++b) {
                uint16_t w16 = 0u;
                memcpy(&w16, p0 + b * 2, sizeof(w16));
                if (silikernels_v2_secded_syn_u16(w16) != 0u) err_mask |= (1u << b);
            }
#endif
            if (err_mask != 0u) {
                for (int b = 0; b < 8; ++b) {
                    if ((err_mask & (1u << b)) == 0u) continue;
                    uint16_t w16 = 0u;
                    memcpy(&w16, p0 + b * 2, sizeof(w16));
                    const uint32_t syn = silikernels_v2_secded_syn_u16(w16);
                    if ((syn & 1u) != 0u) {
                        w16 ^= (uint16_t)(1u << ((syn >> 1) & 15u));
                        memcpy(p0 + b * 2, &w16, sizeof(w16));
                        if (stats) stats->sec_repaired += 1u;
                    } else if (stats) {
                        stats->ded_trapped += 1u;
                        stats->safe_halt = 1;
                    }
                }
            }
            const float w_sc = silik_v2_fp16_to_f32(row_s[g]);
            const int base32 = g * 4;
#if SILIKERNELS_HAS_AVX512
            const __m512i one_b = _mm512_set1_epi8(1);
            for (int half = 0; half < 2; ++half) {
                const uint64_t m0 = *(const uint64_t*)(const void*)(p0 + half * 8);
                const uint64_t m1 = *(const uint64_t*)(const void*)(p1 + half * 8);
                const uint64_t m2 = *(const uint64_t*)(const void*)(p2 + half * 8);
                const __m512i b0 = _mm512_and_si512(_mm512_movm_epi8((__mmask64)m0), one_b);
                const __m512i b1 = _mm512_and_si512(_mm512_movm_epi8((__mmask64)m1), one_b);
                const __m512i b2 = _mm512_and_si512(_mm512_movm_epi8((__mmask64)m2), one_b);
                const __m512i u = _mm512_sub_epi8(
                    _mm512_or_si512(b0, _mm512_slli_epi16(b1, 1)),
                    _mm512_slli_epi16(b2, 2)
                );
                const __m512i u_bias = _mm512_add_epi8(u, _mm512_set1_epi8(4));
                const __m512i dh = silik_v2_dpbusd_u8_s8(zero512, u_bias, _mm512_loadu_si512((const void*)(act->x_hi + (base32 + half * 2) * 32)));
                const __m512i dm = silik_v2_dpbusd_u8_s8(zero512, u_bias, _mm512_loadu_si512((const void*)(act->x_md + (base32 + half * 2) * 32)));
                const __m512i dl = silik_v2_dpbusd_u8_s8(zero512, u_bias, _mm512_loadu_si512((const void*)(act->x_lo + (base32 + half * 2) * 32)));
                const __m512i tot = _mm512_add_epi32(_mm512_add_epi32(_mm512_slli_epi32(dh, 14), _mm512_slli_epi32(dm, 7)), dl);
                const int32_t s_lo = _mm512_mask_reduce_add_epi32(0x00FFu, tot);
                const int32_t s_hi = _mm512_mask_reduce_add_epi32(0xFF00u, tot);
                const int32_t q_lo = s_lo - 4 * act->x_sumq[base32 + half * 2 + 0];
                const int32_t q_hi = s_hi - 4 * act->x_sumq[base32 + half * 2 + 1];
                lanes128[base32 + half * 2 + 0] = (float)q_lo * (w_sc * act->x_sc[base32 + half * 2 + 0]);
                lanes128[base32 + half * 2 + 1] = (float)q_hi * (w_sc * act->x_sc[base32 + half * 2 + 1]);
            }
#else
            for (int sg = 0; sg < 4; ++sg) {
                const int gi = base32 + sg;
                int32_t ih = 0, im = 0, il = 0;
                for (int k = 0; k < 32; ++k) {
                    const int bit_idx = sg * 32 + k;
                    const int byte_i = bit_idx >> 3;
                    const int sh = bit_idx & 7;
                    const int32_t wb0 = (p0[byte_i] >> sh) & 1;
                    const int32_t wb1 = (p1[byte_i] >> sh) & 1;
                    const int32_t wb2 = (p2[byte_i] >> sh) & 1;
                    const int32_t wv = (wb0 | (wb1 << 1)) - (wb2 << 2);
                    const int idx = gi * 32 + k;
                    ih += wv * (int32_t)act->x_hi[idx];
                    im += wv * (int32_t)act->x_md[idx];
                    il += wv * (int32_t)act->x_lo[idx];
                }
                const int64_t tot = ((int64_t)ih << 14) + ((int64_t)im << 7) + (int64_t)il;
                lanes128[gi] = (float)tot * (w_sc * act->x_sc[gi]);
            }
#endif
        }
#if SILIKERNELS_HAS_AVX512
        y_out[r] = silik_v2_tree_reduce_96_avx(lanes128, n_grp32);
#else
        y_out[r] = silikernels_v2_canonical_tree_reduce_128_c(lanes128);
#endif
    }
    return (stats && stats->safe_halt) ? -1 : 0;
}

typedef struct {
    const void*  w_ptr;
    const void*  w_sc;
    const SilikernelsV2ActQ22* act;
    float*       y_out;
    int          r0;
    int          r1;
    int          in_dim;
    int          is_45bpw;
    SilikernelsV2SecdedStats stats;
    int          rc;
} SilikV2GemvTask;

static void* silik_v2_gemv_worker(void* arg) {
    SilikV2GemvTask* t = (SilikV2GemvTask*)arg;
    memset(&t->stats, 0, sizeof(t->stats));
    if (t->is_45bpw) {
        t->rc = silik_v2_gemv_45bpw_range(
            (const uint64_t*)t->w_ptr, (const uint16_t*)t->w_sc,
            t->act, t->y_out, t->r0, t->r1, t->in_dim, &t->stats);
    } else {
        t->rc = silik_v2_gemv_3125bpw_range(
            (const uint8_t*)t->w_ptr, (const uint16_t*)t->w_sc,
            t->act, t->y_out, t->r0, t->r1, t->in_dim, &t->stats);
    }
    return NULL;
}

static int silik_v2_dispatch_gemv(
    const void* w_ptr,
    const void* w_sc,
    const SilikernelsV2ActQ22* act,
    float* y_out,
    int out_dim,
    int in_dim,
    int n_threads,
    int is_45bpw,
    SilikernelsV2SecdedStats* stats
) {
    if (!w_ptr || !w_sc || !act || !y_out || out_dim <= 0 || in_dim <= 0 || (in_dim & 127) != 0) return -1;
    if (n_threads <= 1 || out_dim < 16) {
        SilikernelsV2SecdedStats local = {0u, 0u, 0};
        const int rc = is_45bpw
            ? silik_v2_gemv_45bpw_range((const uint64_t*)w_ptr, (const uint16_t*)w_sc, act, y_out, 0, out_dim, in_dim, &local)
            : silik_v2_gemv_3125bpw_range((const uint8_t*)w_ptr, (const uint16_t*)w_sc, act, y_out, 0, out_dim, in_dim, &local);
        if (stats) {
            stats->sec_repaired += local.sec_repaired;
            stats->ded_trapped  += local.ded_trapped;
            if (local.safe_halt) stats->safe_halt = 1;
        }
        if (rc != 0) memset(y_out, 0, sizeof(float) * (size_t)out_dim);
        return rc;
    }

    int nt = n_threads > SILIK_V2_PHYS_CORES ? SILIK_V2_PHYS_CORES : n_threads;
    if (nt > out_dim) nt = out_dim;
    pthread_t th[SILIK_V2_PHYS_CORES];
    SilikV2GemvTask tasks[SILIK_V2_PHYS_CORES];
    const int chunk = (out_dim + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        const int r0 = t * chunk;
        int r1 = r0 + chunk;
        if (r1 > out_dim) r1 = out_dim;
        tasks[t].w_ptr = w_ptr;
        tasks[t].w_sc = w_sc;
        tasks[t].act = act;
        tasks[t].y_out = y_out;
        tasks[t].r0 = r0;
        tasks[t].r1 = r1;
        tasks[t].in_dim = in_dim;
        tasks[t].is_45bpw = is_45bpw;
        tasks[t].rc = 0;
        memset(&tasks[t].stats, 0, sizeof(tasks[t].stats));
    }
    for (int t = 1; t < nt; ++t) {
        if (pthread_create(&th[t], NULL, silik_v2_gemv_worker, &tasks[t]) != 0) {
            for (int j = 1; j < t; ++j) pthread_join(th[j], NULL);
            return -1;
        }
    }
    silik_v2_gemv_worker(&tasks[0]);
    for (int t = 1; t < nt; ++t) pthread_join(th[t], NULL);
    int fail = 0;
    for (int t = 0; t < nt; ++t) {
        if (stats) {
            stats->sec_repaired += tasks[t].stats.sec_repaired;
            stats->ded_trapped  += tasks[t].stats.ded_trapped;
            if (tasks[t].stats.safe_halt) stats->safe_halt = 1;
        }
        if (tasks[t].rc != 0) fail = -1;
    }
    if (fail != 0) memset(y_out, 0, sizeof(float) * (size_t)out_dim);
    return fail;
}

int silikernels_v2_gemv_45bpw_q22(
    const uint64_t* w_u64,
    const uint16_t* w_sc,
    const SilikernelsV2ActQ22* act,
    float* y_out,
    int out_dim,
    int in_dim,
    int n_threads,
    SilikernelsV2SecdedStats* stats
) {
    return silik_v2_dispatch_gemv(w_u64, w_sc, act, y_out, out_dim, in_dim, n_threads, 1, stats);
}

int silikernels_v2_gemv_3125bpw_q22(
    const uint8_t* w_planes,
    const uint16_t* w_sc_f16,
    const SilikernelsV2ActQ22* act,
    float* y_out,
    int out_dim,
    int in_dim,
    int n_threads,
    SilikernelsV2SecdedStats* stats
) {
    return silik_v2_dispatch_gemv(w_planes, w_sc_f16, act, y_out, out_dim, in_dim, n_threads, 0, stats);
}

static uint64_t silik_v2_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t silik_v2_max_ulp_diff(const float* a, const float* b, int n) {
    uint64_t mx = 0ULL;
    for (int i = 0; i < n; ++i) {
        uint32_t ua = 0u, ub = 0u;
        memcpy(&ua, &a[i], sizeof(ua));
        memcpy(&ub, &b[i], sizeof(ub));
        int32_t ia = (ua & 0x80000000u) ? (int32_t)(0x80000000u - ua) : (int32_t)ua;
        int32_t ib = (ub & 0x80000000u) ? (int32_t)(0x80000000u - ub) : (int32_t)ub;
        uint64_t d = (uint64_t)(ia > ib ? (ia - ib) : (ib - ia));
        if (d > mx) mx = d;
    }
    return mx;
}

static uint64_t silik_v2_encode_secded_w64(uint64_t* rng) {
    uint64_t u11 = silik_v2_splitmix64(rng) & 0x7FFULL;
    uint8_t b[16];
    const int info_pos[11] = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    memset(b, 0, sizeof(b));
    for (int j = 0; j < 11; ++j) b[info_pos[j]] = (uint8_t)((u11 >> j) & 1ULL);
    b[1] = b[3] ^ b[5] ^ b[7] ^ b[9] ^ b[11] ^ b[13] ^ b[15];
    b[2] = b[3] ^ b[6] ^ b[7] ^ b[10] ^ b[11] ^ b[14] ^ b[15];
    b[4] = b[5] ^ b[6] ^ b[7] ^ b[12] ^ b[13] ^ b[14] ^ b[15];
    b[8] = b[9] ^ b[10] ^ b[11] ^ b[12] ^ b[13] ^ b[14] ^ b[15];
    uint8_t s = 0u;
    for (int i = 1; i < 16; ++i) s ^= b[i];
    b[0] = s;
    uint64_t w64 = 0ULL;
    for (int i = 0; i < 16; ++i) {
        int v = 2 * ((int)(silik_v2_splitmix64(rng) % 7ULL) - 3) + (int)b[i];
        if (v < -7) v += 2;
        if (v > 7) v -= 2;
        w64 |= ((uint64_t)((uint8_t)v & 0xFu) << (4 * i));
    }
    return w64;
}

int silikernels_v2_run_selftest(int verbose) {
    const int M = 64;
    const int K = 256;
    const int n_grp32 = K >> 5;
    const int n_grp128 = K >> 7;
    uint64_t rng = 0x20261128A16B2ULL;
    int fail = 0;

    float* x = (float*)calloc((size_t)K, sizeof(float));
    uint64_t* w45 = (uint64_t*)calloc((size_t)M * (size_t)(n_grp32 * 2), sizeof(uint64_t));
    uint16_t* s45 = (uint16_t*)calloc((size_t)M * (size_t)n_grp32, sizeof(uint16_t));
    uint8_t* w3 = (uint8_t*)calloc((size_t)M * (size_t)n_grp128 * 48u, sizeof(uint8_t));
    uint16_t* s3 = (uint16_t*)calloc((size_t)M * (size_t)n_grp128, sizeof(uint16_t));
    float* y1 = (float*)calloc((size_t)M, sizeof(float));
    float* y8 = (float*)calloc((size_t)M, sizeof(float));
    float* y3_1 = (float*)calloc((size_t)M, sizeof(float));
    float* y3_8 = (float*)calloc((size_t)M, sizeof(float));
    if (!x || !w45 || !s45 || !w3 || !s3 || !y1 || !y8 || !y3_1 || !y3_8) {
        free(x); free(w45); free(s45); free(w3); free(s3); free(y1); free(y8); free(y3_1); free(y3_8);
        return 1;
    }

    for (int i = 0; i < K; ++i) {
        double u = (double)(silik_v2_splitmix64(&rng) >> 11) * (1.0 / 9007199254740992.0);
        x[i] = (float)((u * 2.0 - 1.0) * 1.5);
    }
    for (int r = 0; r < M; ++r) {
        for (int g = 0; g < n_grp32; ++g) {
            w45[r * (n_grp32 * 2) + 2 * g + 0] = silik_v2_encode_secded_w64(&rng);
            w45[r * (n_grp32 * 2) + 2 * g + 1] = silik_v2_encode_secded_w64(&rng);
            s45[r * n_grp32 + g] = silik_v2_f32_to_fp16(0.05f + 0.01f * (float)(g & 3));
        }
        for (int g = 0; g < n_grp128; ++g) {
            uint8_t* gp = w3 + ((size_t)r * (size_t)n_grp128 + (size_t)g) * 48u;
            for (int b = 0; b < 8; ++b) {
                uint64_t w64 = silik_v2_encode_secded_w64(&rng);
                uint16_t p0_16 = 0u;
                for (int j = 0; j < 16; ++j) p0_16 |= (uint16_t)(((w64 >> (4 * j)) & 1ULL) << j);
                memcpy(gp + b * 2, &p0_16, sizeof(p0_16));
            }
            for (int i = 16; i < 48; ++i) gp[i] = (uint8_t)(silik_v2_splitmix64(&rng) & 0xFFu);
            s3[r * n_grp128 + g] = silik_v2_f32_to_fp16(0.08f);
        }
    }

    SilikernelsV2ActQ22 act;
    memset(&act, 0, sizeof(act));
    silikernels_v2_quant_act22_h4(x, K, &act);

    SilikernelsV2SecdedStats st1 = {0u, 0u, 0}, st8 = {0u, 0u, 0};
    if (silikernels_v2_gemv_45bpw_q22(w45, s45, &act, y1, M, K, 1, &st1) != 0) fail = 1;
    if (silikernels_v2_gemv_45bpw_q22(w45, s45, &act, y8, M, K, 8, &st8) != 0) fail = 1;
    const uint64_t ulp45 = silik_v2_max_ulp_diff(y1, y8, M);
    if (ulp45 != 0ULL || st1.sec_repaired != 0u || st8.ded_trapped != 0u) fail = 1;

    SilikernelsV2SecdedStats st3_1 = {0u, 0u, 0}, st3_8 = {0u, 0u, 0};
    if (silikernels_v2_gemv_3125bpw_q22(w3, s3, &act, y3_1, M, K, 1, &st3_1) != 0) fail = 1;
    if (silikernels_v2_gemv_3125bpw_q22(w3, s3, &act, y3_8, M, K, 8, &st3_8) != 0) fail = 1;
    const uint64_t ulp3 = silik_v2_max_ulp_diff(y3_1, y3_8, M);
    if (ulp3 != 0ULL || st3_1.sec_repaired != 0u || st3_8.ded_trapped != 0u) fail = 1;

    /* 3/3 SEC + 3/3 DED fault traps on 4.5 bpw GEMV */
    for (int k = 0; k < 3; ++k) {
        w45[k * (n_grp32 * 2)] ^= (1ULL << (4 * ((k * 5) % 16)));
    }
    SilikernelsV2SecdedStats st_sec = {0u, 0u, 0};
    if (silikernels_v2_gemv_45bpw_q22(w45, s45, &act, y8, M, K, 4, &st_sec) != 0) fail = 1;
    if (st_sec.sec_repaired != 3u || silik_v2_max_ulp_diff(y1, y8, M) != 0ULL) fail = 1;

    /* Inject 3 clean-word double-bit faults on word index 1 of rows 0..2 */
    for (int k = 0; k < 3; ++k) {
        const uint32_t b1 = (uint32_t)((k * 3) % 16);
        const uint32_t b2 = (b1 + 1u) & 15u;
        w45[1 + k * (n_grp32 * 2)] ^= ((1ULL << (4u * b1)) | (1ULL << (4u * b2)));
    }
    SilikernelsV2SecdedStats st_ded = {0u, 0u, 0};
    for (int i = 0; i < M; ++i) y8[i] = 99.0f;
    if (silikernels_v2_gemv_45bpw_q22(w45, s45, &act, y8, M, K, 4, &st_ded) == 0) fail = 1;
    if (st_ded.safe_halt != 1 || st_ded.ded_trapped != 3u) fail = 1;
    for (int i = 0; i < M; ++i) {
        if (y8[i] != 0.0f) fail = 1;
    }

    /* SWAR127 64 positive packs + 3 OOB negative controls */
    for (int s = 0; s < 64; ++s) {
        uint64_t packed = 0ULL;
        int8_t exp_b[8];
        for (int b = 0; b < 8; ++b) {
            const int8_t val = (int8_t)(1 + (int)(silik_v2_splitmix64(&rng) % 127ULL));
            exp_b[b] = (int8_t)(-val);
            packed |= ((uint64_t)(uint8_t)val << (b * 8));
        }
        uint64_t chk = 0ULL;
        if (silikernels_v2_swar127_in_domain_u64(packed) != 1) fail = 1;
        if (silikernels_v2_swar127_negate_u64_checked(packed, &chk) != 0) fail = 1;
        if (silikernels_v2_swar127_negate_u64(packed) != chk) fail = 1;
        for (int b = 0; b < 8; ++b) {
            if ((int8_t)((chk >> (b * 8)) & 0xFFu) != exp_b[b]) fail = 1;
        }
    }
    const uint64_t bad_swar[3] = {0x0101010101010001ULL, 0x0101018001010101ULL, 0x01010101FF010101ULL};
    for (int k = 0; k < 3; ++k) {
        uint64_t out_bad = 0xDEADBEEFULL;
        if (silikernels_v2_swar127_in_domain_u64(bad_swar[k]) != 0) fail = 1;
        if (silikernels_v2_swar127_negate_u64_checked(bad_swar[k], &out_bad) == 0 || out_bad != 0ULL) fail = 1;
        if (silikernels_v2_swar127_negate_u64(bad_swar[k]) != 0ULL) fail = 1;
    }
    if (silikernels_v2_should_dispatch_cpu(1u, 0u) != 1 || silikernels_v2_should_dispatch_cpu(2u, 0u) != 0) fail = 1;
    if (silikernels_v2_should_dispatch_cpu(2u, 1u) != 1 || silikernels_v2_should_dispatch_cpu(3u, 1u) != 0) fail = 1;

    if (verbose) {
        printf("{\"status\":\"%s\",\"avx512_compiled\":%d,\"ulp_45\":%llu,\"ulp_3125\":%llu,"
               "\"sec_repaired\":%u,\"ded_trapped\":%u}\n",
               fail == 0 ? "PASS" : "FAIL",
               SILIKERNELS_HAS_AVX512,
               (unsigned long long)ulp45,
               (unsigned long long)ulp3,
               st_sec.sec_repaired,
               st_ded.ded_trapped);
    }

    free(x); free(w45); free(s45); free(w3); free(s3); free(y1); free(y8); free(y3_1); free(y3_8);
    return fail;
}

#ifdef __cplusplus
}
#endif

#ifndef SILIKERNELS_V2_LIB
int main(int argc, char** argv) {
    int self_test = 0;
    int verbose = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0) {
            self_test = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (!self_test && argc == 1) self_test = 1;
    return silikernels_v2_run_selftest(verbose) == 0 ? 0 : 1;
}
#endif
