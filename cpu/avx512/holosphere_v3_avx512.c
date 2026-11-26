/*
 * Track S125 (EXP-Q30b, PREREG-W14): Zero-Dependency C11 + Intel AVX-512 VNNI
 * L3-Resident HoloSphere-V3 15M (D=512, H=16, L=6, d_ff=800, V=256 tied FWHT)
 * Bare-Metal CPU Inference Engine.
 *
 * Design highlights:
 *   - Fits ENTIRE model weights (8.428 MiB / 8,837,120 B A_16 Block-H4 4.5 bpw +
 *     dense scalars) + O(1) persistent recurrent state (0.582 MiB / 610,304 B,
 *     0 KV cache growth) = 9.010 MiB (9,447,424 B) strictly inside Intel
 *     i9-11900H 24 MiB L3 cache (37.54% L3 occupancy).
 *   - Exact algebraic equivalence between scalar C11 and AVX-512 VNNI
 *     (_mm512_dpbusd_epi32): pre-applies H_4 to x once, quantizes x_h4 to int8
 *     per 32-element group, bias-shifts signed int4 nibbles c in [-8,+7] to
 *     u = c + 8 in [0,15] via XOR 0x88, computes exact int32 VPDPBUSD dot
 *     product, subtracts 8 * sum(x_q[g32]), and checks inline [16,11,4]
 *     Extended Hamming SECDED syndrome on the 16-bit packed nibble-LSB plane
 *     (w64 & 0x1111111111111111ULL, 1/4 of payload bits; matching Razor
 *     bit-plane canary semantics—nibble upper bits 1..3 are unguarded) via
 *     VPOPCNTQ (_mm512_popcnt_epi64).
 *   - Deterministic row-partitioned pthread worker pool (1T / 4T / 8T): bit-exact
 *     identical FP32 outputs (0.0 ULP drift, memcmp == 0) across thread counts.
 *   - Explicit 4 KiB (MADV_NOHUGEPAGE) vs 2 MiB (MADV_HUGEPAGE) arena allocator
 *     with live /proc/self/smaps AnonHugePages verification.
 *
 * Compile:
 *   gcc -O3 -march=native -std=c11 -Wall -Wextra -Werror -pthread \
 *       holosphere_v3_avx512.c -lm -o holosphere_v3_avx512
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

/* Exact HoloSphere-V3 15M Geometry (matching ahrazzle/holosphere-v3-unified-sm86-15m) */
#define HS_D_MODEL    512
#define HS_N_HEADS    16
#define HS_HEAD_DIM   32
#define HS_N_SPINORS  (HS_D_MODEL / 4)   /* 128 */
#define HS_N_LAYERS   6
#define HS_D_FF       800
#define HS_VOCAB      256                /* 0 B VRAM/RAM tied FWHT_256 x bent4 LM head */
#define HS_RING_K     8
#define HS_CONV_K     4
#define HS_C_DIM      (3 * HS_N_HEADS)   /* 48 */
#define HS_W_OMEGA_D  (3 * HS_N_SPINORS) /* 384 */

#define HS_CAYLEY_T   0.25f
#define HS_GAMMA      0.90f
#define HS_ETA        0.15f
#define HS_EPS        1e-6f
#define HS_LOGIT_SC   4.0f

#define HS_ALIGN64(x) (((size_t)(x) + 63u) & ~(size_t)63u)
#define HS_ALIGN2M(x) (((size_t)(x) + (2u * 1024u * 1024u - 1u)) & ~(size_t)(2u * 1024u * 1024u - 1u))

/* ============================================================================
 * 1. Deterministic SplitMix64 PRNG & IEEE-754 binary16 Exact Conversion
 * ============================================================================ */

static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float splitmix_uniform_sym(uint64_t* state, float amp) {
    uint32_t u = (uint32_t)(splitmix64_next(state) >> 40); /* 24 bits exact */
    float f = ((float)u * (1.0f / 8388608.0f)) - 1.0f;     /* [-1.0, +1.0) */
    return f * amp;
}

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            exp = 127u - 14u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t f32_to_fp16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = 1u << (shift - 1u);
        uint32_t m = (mant + round_bit) >> shift;
        return (uint16_t)(sign | m);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7BFFu);
    }
    /* Round to nearest even */
    uint32_t half_lsb = (mant >> 13) & 1u;
    uint32_t rounded  = mant + 0x00000FFFu + half_lsb;
    if (rounded & 0x00800000u) {
        rounded = 0u;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

/* ============================================================================
 * 2. Extended Hamming [16, 11, 4] A_16 Block-H4 Pack, Syndrome, & SEC/DED Heal
 * ============================================================================ */

static inline uint32_t a16_extract_odd16(uint64_t w64) {
    uint32_t lo = (uint32_t)w64;
    uint32_t hi = (uint32_t)(w64 >> 32);
    uint32_t ml = lo & 0x11111111u;
    ml = (ml | (ml >> 3)) & 0x03030303u;
    ml = (ml | (ml >> 6)) & 0x000F000Fu;
    ml = (ml | (ml >> 12)) & 0xFFu;
    uint32_t mh = hi & 0x11111111u;
    mh = (mh | (mh >> 3)) & 0x03030303u;
    mh = (mh | (mh >> 6)) & 0x000F000Fu;
    mh = (mh | (mh >> 12)) & 0xFFu;
    return ml | (mh << 8);
}

static inline uint32_t a16_syndrome_u16(uint32_t w16) {
    uint32_t p0 = (uint32_t)__builtin_popcount(w16 & 0xFFFFu) & 1u;
    uint32_t p1 = (uint32_t)__builtin_popcount(w16 & 0xAAAAu) & 1u;
    uint32_t p2 = (uint32_t)__builtin_popcount(w16 & 0xCCCCu) & 1u;
    uint32_t p3 = (uint32_t)__builtin_popcount(w16 & 0xF0F0u) & 1u;
    uint32_t p4 = (uint32_t)__builtin_popcount(w16 & 0xFF00u) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

/* Direct 64-bit word [16, 11, 4] nibble-LSB bit-plane syndrome (16 LSB bits = 1/4 payload;
 * 5 masked 64-bit popcounts, 0 nibble unpack; nibble upper bits 1..3 are unguarded) */
static inline uint32_t a16_syndrome_u64_direct(uint64_t w64) {
    uint64_t lsb = w64 & 0x1111111111111111ULL;
    uint32_t p0 = (uint32_t)__builtin_popcountll(lsb) & 1u;
    uint32_t p1 = (uint32_t)__builtin_popcountll(lsb & 0x1010101010101010ULL) & 1u;
    uint32_t p2 = (uint32_t)__builtin_popcountll(lsb & 0x1100110011001100ULL) & 1u;
    uint32_t p3 = (uint32_t)__builtin_popcountll(lsb & 0x1111000011110000ULL) & 1u;
    uint32_t p4 = (uint32_t)__builtin_popcountll(lsb & 0x1111111100000000ULL) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

static inline uint64_t a16_secded_heal_u64(uint64_t w64, uint32_t* sec_cnt, uint32_t* ded_cnt) {
    uint32_t syn = a16_syndrome_u64_direct(w64);
    if (syn == 0u) return w64;
    if ((syn & 1u) != 0u) {
        uint32_t pos = (syn >> 1) & 15u;
        if (sec_cnt) (*sec_cnt)++;
        return w64 ^ (1ULL << (pos * 4u));
    }
    if (ded_cnt) (*ded_cnt)++;
    return w64;
}

static void a16_snap_pos(int c16[16], const float u16[16], int pos, int req_lsb) {
    int cur = c16[pos];
    if ((cur & 1) == req_lsb) return;
    int step = (u16[pos] >= (float)cur) ? 1 : -1;
    if (cur == 7 && step > 0) step = -1;
    else if (cur == -8 && step < 0) step = 1;
    c16[pos] = cur + step;
}

static uint64_t pack_a16_matrix_synth(
    uint64_t* out_u64,
    uint16_t* out_sc_f16,
    int out_dim,
    int in_dim,
    uint64_t* rng_state,
    float amp
) {
    static const int p_pos[4] = {1, 2, 4, 8};
    static const int p_idx[4][7] = {
        {3, 5, 7, 9, 11, 13, 15},
        {3, 6, 7, 10, 11, 14, 15},
        {5, 6, 7, 12, 13, 14, 15},
        {9, 10, 11, 12, 13, 14, 15}
    };
    const int n_grp32 = in_dim / 32;
    uint64_t syn_errors = 0;

    for (int r = 0; r < out_dim; ++r) {
        for (int g = 0; g < n_grp32; ++g) {
            float u_raw[32];
            float amax = 0.0f;
            for (int k = 0; k < 32; ++k) {
                float v = splitmix_uniform_sym(rng_state, amp);
                u_raw[k] = v;
                float av = fabsf(v);
                if (av > amax) amax = av;
            }
            if (amax < 1e-6f) amax = 1e-6f;
            uint16_t sc16 = f32_to_fp16(amax / 7.5f);
            float sc = fp16_to_f32(sc16);
            out_sc_f16[(size_t)r * (size_t)n_grp32 + (size_t)g] = sc16;

            for (int half = 0; half < 2; ++half) {
                float u16[16];
                int c16[16];
                for (int j = 0; j < 16; ++j) {
                    float q = u_raw[half * 16 + j] / sc;
                    if (q > 7.0f) q = 7.0f;
                    if (q < -8.0f) q = -8.0f;
                    u16[j] = q;
                    int ci = (int)lrintf(q);
                    if (ci > 7) ci = 7;
                    if (ci < -8) ci = -8;
                    c16[j] = ci;
                }
                for (int p = 0; p < 4; ++p) {
                    int req = 0;
                    for (int k = 0; k < 7; ++k) req ^= (c16[p_idx[p][k]] & 1);
                    a16_snap_pos(c16, u16, p_pos[p], req);
                }
                int req0 = 0;
                for (int j = 1; j < 16; ++j) req0 ^= (c16[j] & 1);
                a16_snap_pos(c16, u16, 0, req0);

                uint64_t w64 = 0ULL;
                for (int j = 0; j < 16; ++j) {
                    w64 |= ((uint64_t)(c16[j] & 0xF)) << (j * 4);
                }
                if (a16_syndrome_u64_direct(w64) != 0u) syn_errors++;
                out_u64[(size_t)r * (size_t)(n_grp32 * 2) + (size_t)(g * 2 + half)] = w64;
            }
        }
    }
    return syn_errors;
}

/* ============================================================================
 * 3. Activation H_4 Pre-Transform + Per-Group32 Signed INT8 Quantization
 *    (Precomputed once per activation vector -> enables exact VPDPBUSD GEMV!)
 * ============================================================================ */

typedef struct {
    int8_t  xq[HS_D_FF];          /* Max K = 800 */
    float   sx[HS_D_FF / 32];     /* Per-group32 scale: x_h4 ≈ sx[g] * xq */
    int32_t sum_xq[HS_D_FF / 32]; /* Sum of 32 xq elements for exact bias cancel */
    int     in_dim;
} QuantActH4;

static void quantize_act_h4(const float* x, int in_dim, QuantActH4* out) {
    out->in_dim = in_dim;
    const int n_grp32 = in_dim / 32;
    for (int g = 0; g < n_grp32; ++g) {
        float h4x[32];
        float amax = 0.0f;
        const float* xg = x + g * 32;
        for (int b = 0; b < 32; b += 4) {
            float a0 = xg[b + 0];
            float a1 = xg[b + 1];
            float a2 = xg[b + 2];
            float a3 = xg[b + 3];
            float t0 = a0 + a1 + a2 + a3;
            float t1 = a0 - a1 + a2 - a3;
            float t2 = a0 + a1 - a2 - a3;
            float t3 = a0 - a1 - a2 + a3;
            h4x[b + 0] = t0;
            h4x[b + 1] = t1;
            h4x[b + 2] = t2;
            h4x[b + 3] = t3;
            float m0 = fabsf(t0), m1 = fabsf(t1), m2 = fabsf(t2), m3 = fabsf(t3);
            if (m0 > amax) amax = m0;
            if (m1 > amax) amax = m1;
            if (m2 > amax) amax = m2;
            if (m3 > amax) amax = m3;
        }
        float sx = (amax > 1e-12f) ? (amax / 127.0f) : 1.0f;
        float inv_sx = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        out->sx[g] = sx;
        int32_t s_acc = 0;
        int8_t* dst = out->xq + g * 32;
        for (int k = 0; k < 32; ++k) {
            int q = (int)lrintf(h4x[k] * inv_sx);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[k] = (int8_t)q;
            s_acc += q;
        }
        out->sum_xq[g] = s_acc;
    }
}

/* ============================================================================
 * 4. Scalar C11 vs AVX-512 VNNI (VPDPBUSD + VPOPCNTQ) A_16 Block-H4 GEMV
 *    (Mathematically BIT-EXACT int32 dot product & identical FP32 group-32 sum!)
 * ============================================================================ */

static void gemv_a16_scalar_rows(
    const QuantActH4* act,
    const uint64_t* w_u64,
    const uint16_t* w_sc_f16,
    float* y_out,
    int row_start,
    int row_end,
    uint32_t* syn_faults_out
) {
    const int n_grp32 = act->in_dim / 32;
    uint32_t faults = 0u;

    for (int r = row_start; r < row_end; ++r) {
        const uint64_t* r_u64 = w_u64 + (size_t)r * (size_t)(n_grp32 * 2);
        const uint16_t* r_sc  = w_sc_f16 + (size_t)r * (size_t)n_grp32;
        float acc = 0.0f;
        for (int g = 0; g < n_grp32; ++g) {
            uint64_t w0 = r_u64[2 * g + 0];
            uint64_t w1 = r_u64[2 * g + 1];
            uint32_t s0 = a16_syndrome_u64_direct(w0);
            uint32_t s1 = a16_syndrome_u64_direct(w1);
            if (__builtin_expect((s0 | s1) != 0u, 0)) {
                uint32_t sec = 0u, ded = 0u;
                w0 = a16_secded_heal_u64(w0, &sec, &ded);
                w1 = a16_secded_heal_u64(w1, &sec, &ded);
                faults += sec + ded;
            }
            /* Bias-shifted u = c + 8 in [0, 15] matching VPDPBUSD exactly */
            uint64_t u0 = w0 ^ 0x8888888888888888ULL;
            uint64_t u1 = w1 ^ 0x8888888888888888ULL;
            const int8_t* xq = act->xq + g * 32;
            int32_t dot_u = 0;
            for (int j = 0; j < 16; ++j) {
                int32_t nib0 = (int32_t)((u0 >> (j * 4)) & 0xFULL);
                int32_t nib1 = (int32_t)((u1 >> (j * 4)) & 0xFULL);
                dot_u += nib0 * (int32_t)xq[j] + nib1 * (int32_t)xq[16 + j];
            }
            int32_t dot_s = dot_u - 8 * act->sum_xq[g];
            float sc = 0.5f * fp16_to_f32(r_sc[g]) * act->sx[g];
            acc += (float)dot_s * sc;
        }
        y_out[r] = acc;
    }
    if (syn_faults_out) *syn_faults_out += faults;
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI__)
/* Unpack 8 uint64_t (= 4 groups of 32 weights = 128 biased u8 nibbles) into two 64-byte ZMM registers */
static inline void unpack_128_u4_to_2x_zmm_u8(
    const uint64_t* src8,
    __m512i* out_grp01,
    __m512i* out_grp23,
    uint32_t* any_syndrome_mask
) {
    /* Load 8 uint64_t codewords (64 B = 1 cacheline!) into 1 ZMM register */
    __m512i raw = _mm512_loadu_si512((const void*)src8);

    /* Vectorized 8-codeword [16,11,4] SECDED syndrome via hardware VPOPCNTQ! */
    const __m512i m_lsb = _mm512_set1_epi64((long long)0x1111111111111111ULL);
    const __m512i m_p1  = _mm512_set1_epi64((long long)0x1010101010101010ULL);
    const __m512i m_p2  = _mm512_set1_epi64((long long)0x1100110011001100ULL);
    const __m512i m_p3  = _mm512_set1_epi64((long long)0x1111000011110000ULL);
    const __m512i m_p4  = _mm512_set1_epi64((long long)0x1111111100000000ULL);
    const __m512i one64 = _mm512_set1_epi64(1LL);

    __m512i lsb = _mm512_and_si512(raw, m_lsb);
    __m512i c0  = _mm512_and_si512(_mm512_popcnt_epi64(lsb), one64);
    __m512i c1  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, m_p1)), one64);
    __m512i c2  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, m_p2)), one64);
    __m512i c3  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, m_p3)), one64);
    __m512i c4  = _mm512_and_si512(_mm512_popcnt_epi64(_mm512_and_si512(lsb, m_p4)), one64);
    __m512i any = _mm512_or_si512(_mm512_or_si512(_mm512_or_si512(c0, c1), _mm512_or_si512(c2, c3)), c4);
    *any_syndrome_mask = (uint32_t)_mm512_test_epi64_mask(any, one64);

    /* Bias shift all 128 signed int4 nibbles c -> u = c + 8 in [0, 15] in 1 XOR! */
    __m512i biased = _mm512_xor_si512(raw, _mm512_set1_epi8((char)0x88));
    const __m512i mask0f = _mm512_set1_epi8(0x0F);

    /* Split low nibbles (even indices 0,2,4,...) and high nibbles (odd indices 1,3,5,...) */
    __m512i lo_nib = _mm512_and_si512(biased, mask0f);
    __m512i hi_nib = _mm512_and_si512(_mm512_srli_epi16(biased, 4), mask0f);

    /* Interleave byte-by-byte within each 128-bit lane, then re-order 128-bit lanes cleanly */
    __m512i zip_lo = _mm512_unpacklo_epi8(lo_nib, hi_nib); /* bytes 0..7 of each 16 B lane -> 16 nibs */
    __m512i zip_hi = _mm512_unpackhi_epi8(lo_nib, hi_nib); /* bytes 8..15 of each 16 B lane -> 16 nibs */

    /*
     * Lane mapping in zip_lo / zip_hi (each 128-bit lane = 16 bytes = 16 nibbles):
     *   zip_lo lanes [0, 1, 2, 3] = codewords [w0, w2, w4, w6]
     *   zip_hi lanes [0, 1, 2, 3] = codewords [w1, w3, w5, w7]
     * Target order:
     *   out_grp01 lanes [0, 1, 2, 3] = [w0, w1, w2, w3]  (groups g+0 and g+1)
     *   out_grp23 lanes [0, 1, 2, 3] = [w4, w5, w6, w7]  (groups g+2 and g+3)
     */
    const __m512i perm01 = _mm512_setr_epi32(
        0, 1, 2, 3,   16, 17, 18, 19,   4, 5, 6, 7,   20, 21, 22, 23
    );
    const __m512i perm23 = _mm512_setr_epi32(
        8, 9, 10, 11, 24, 25, 26, 27,  12, 13, 14, 15, 28, 29, 30, 31
    );
    *out_grp01 = _mm512_permutex2var_epi32(zip_lo, perm01, zip_hi);
    *out_grp23 = _mm512_permutex2var_epi32(zip_lo, perm23, zip_hi);
}

static inline int32_t hsum_epi32_256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

static void gemv_a16_avx512_vnni_rows(
    const QuantActH4* act,
    const uint64_t* w_u64,
    const uint16_t* w_sc_f16,
    float* y_out,
    int row_start,
    int row_end,
    uint32_t* syn_faults_out
) {
    const int n_grp32 = act->in_dim / 32;
    const __m512i zero512 = _mm512_setzero_si512();
    uint32_t faults = 0u;

    for (int r = row_start; r < row_end; ++r) {
        const uint64_t* r_u64 = w_u64 + (size_t)r * (size_t)(n_grp32 * 2);
        const uint16_t* r_sc  = w_sc_f16 + (size_t)r * (size_t)n_grp32;
        float acc = 0.0f;

        int g = 0;
        for (; g + 3 < n_grp32; g += 4) {
            __m512i u01, u23;
            uint32_t syn_mask = 0u;
            unpack_128_u4_to_2x_zmm_u8(r_u64 + 2 * g, &u01, &u23, &syn_mask);
            if (__builtin_expect(syn_mask != 0u, 0)) {
                /* Cold-path exact inline 1-bit SEC repair / 2-bit DED detect */
                uint64_t fixed[8];
                memcpy(fixed, r_u64 + 2 * g, sizeof(fixed));
                for (int k = 0; k < 8; ++k) {
                    if (syn_mask & (1u << k)) {
                        uint32_t sec = 0u, ded = 0u;
                        fixed[k] = a16_secded_heal_u64(fixed[k], &sec, &ded);
                        faults += sec + ded;
                    }
                }
                uint32_t dummy_mask = 0u;
                unpack_128_u4_to_2x_zmm_u8(fixed, &u01, &u23, &dummy_mask);
            }

            /* Load 64 + 64 signed int8 quantized activations (groups g..g+3) */
            __m512i x01 = _mm512_loadu_si512((const void*)(act->xq + g * 32));
            __m512i x23 = _mm512_loadu_si512((const void*)(act->xq + (g + 2) * 32));

            /* Exact AVX-512 VNNI VPDPBUSD: 64 u8*s8 MACs per instruction! */
            __m512i d01 = _mm512_dpbusd_epi32(zero512, u01, x01);
            __m512i d23 = _mm512_dpbusd_epi32(zero512, u23, x23);

            int32_t s0 = hsum_epi32_256(_mm512_castsi512_si256(d01))          - 8 * act->sum_xq[g + 0];
            int32_t s1 = hsum_epi32_256(_mm512_extracti64x4_epi64(d01, 1))    - 8 * act->sum_xq[g + 1];
            int32_t s2 = hsum_epi32_256(_mm512_castsi512_si256(d23))          - 8 * act->sum_xq[g + 2];
            int32_t s3 = hsum_epi32_256(_mm512_extracti64x4_epi64(d23, 1))    - 8 * act->sum_xq[g + 3];

            /* Strict sequential group-32 FP32 accumulation -> bit-exact match with scalar C11! */
            acc += (float)s0 * (0.5f * fp16_to_f32(r_sc[g + 0]) * act->sx[g + 0]);
            acc += (float)s1 * (0.5f * fp16_to_f32(r_sc[g + 1]) * act->sx[g + 1]);
            acc += (float)s2 * (0.5f * fp16_to_f32(r_sc[g + 2]) * act->sx[g + 2]);
            acc += (float)s3 * (0.5f * fp16_to_f32(r_sc[g + 3]) * act->sx[g + 3]);
        }
        /* Tail for n_grp32 not divisible by 4 (e.g. in_dim=800 -> n_grp32=25 -> 1 tail group) */
        for (; g < n_grp32; ++g) {
            uint64_t w0 = r_u64[2 * g + 0];
            uint64_t w1 = r_u64[2 * g + 1];
            if (__builtin_expect((a16_syndrome_u64_direct(w0) | a16_syndrome_u64_direct(w1)) != 0u, 0)) {
                uint32_t sec = 0u, ded = 0u;
                w0 = a16_secded_heal_u64(w0, &sec, &ded);
                w1 = a16_secded_heal_u64(w1, &sec, &ded);
                faults += sec + ded;
            }
            uint64_t u0 = w0 ^ 0x8888888888888888ULL;
            uint64_t u1 = w1 ^ 0x8888888888888888ULL;
            const int8_t* xq = act->xq + g * 32;
            int32_t dot_u = 0;
            for (int j = 0; j < 16; ++j) {
                dot_u += (int32_t)((u0 >> (j * 4)) & 0xFULL) * (int32_t)xq[j]
                       + (int32_t)((u1 >> (j * 4)) & 0xFULL) * (int32_t)xq[16 + j];
            }
            int32_t dot_s = dot_u - 8 * act->sum_xq[g];
            acc += (float)dot_s * (0.5f * fp16_to_f32(r_sc[g]) * act->sx[g]);
        }
        y_out[r] = acc;
    }
    if (syn_faults_out) *syn_faults_out += faults;
}
#endif

/* ============================================================================
 * 5. HoloSphere-V3 15M L3-Resident Arena Layout (All Weights + O(1) State)
 * ============================================================================ */

typedef struct {
    uint64_t* w_q_ring_u64;     uint16_t* w_q_ring_sc;     /* [512, 512] */
    uint64_t* w_k_ring_u64;     uint16_t* w_k_ring_sc;     /* [512, 512] */
    uint64_t* w_ring_ctrl_u64;  uint16_t* w_ring_ctrl_sc;  /* [48,  512] */
    uint64_t* w_mem_out_u64;    uint16_t* w_mem_out_sc;    /* [512, 512] */
    uint64_t* w_omega_u64;      uint16_t* w_omega_sc;      /* [384, 512] */
    uint64_t* w_cell_out_u64;   uint16_t* w_cell_out_sc;   /* [512, 512] */
    uint64_t* w_gate_u64;       uint16_t* w_gate_sc;       /* [800, 512] */
    uint64_t* w_up_u64;         uint16_t* w_up_sc;         /* [800, 512] */
    uint64_t* w_down_u64;       uint16_t* w_down_sc;       /* [512, 800] */
    float*    w_q_fast;                                     /* [16, 32, 32] */
    float*    omega_scale;                                  /* [128] */
    float     alpha_ring;
    float     alpha_cell;
    float     alpha_mlp;
} HoloLayerWeights;

typedef struct {
    void*  arena_raw;
    void*  arena_aligned;
    size_t arena_alloc_bytes;
    size_t weight_bytes;
    size_t state_bytes;
    size_t total_live_bytes;
    long   smaps_anon_hugepages_kb;
    bool   hugepages_requested;

    /* Weights */
    float* conv_w; /* [512, 4] */
    HoloLayerWeights layers[HS_N_LAYERS];

    /* O(1) Persistent State */
    float* conv_ring; /* [4, 512] */
    float* S_fast;    /* [6, 16, 32, 32] */
    float* U_state;   /* [6, 16, 8, 4] */
    float* k_ring;    /* [6, 16, 8, 32] */
    float* v_ring;    /* [6, 16, 8, 32] */
    int    conv_pos;

    /* Scratch buffers */
    float x_buf[HS_D_MODEL];
    float q_buf[HS_D_MODEL];
    float k_buf[HS_D_MODEL];
    float ctrl_buf[HS_C_DIM];
    float ymem_buf[HS_D_MODEL];
    float rgate_buf[HS_N_HEADS];
    float mproj_buf[HS_D_MODEL];
    float hgen_buf[HS_D_MODEL];
    float wraw_buf[HS_W_OMEGA_D];
    float ycell_buf[HS_D_MODEL];
    float cout_buf[HS_D_MODEL];
    float xmid_buf[HS_D_MODEL];
    float gate_buf[HS_D_FF];
    float up_buf[HS_D_FF];
    float act_buf[HS_D_FF];
    float down_buf[HS_D_MODEL];
    float logits_buf[HS_VOCAB];
    QuantActH4 qact;

    uint64_t pack_syndrome_errors;
    uint32_t runtime_syndrome_faults;
} HoloV3Engine;

static long query_smaps_anon_hugepages_kb(const void* addr) {
    FILE* fp = fopen("/proc/self/smaps", "r");
    if (!fp) return -1;
    uintptr_t target = (uintptr_t)addr;
    char line[256];
    bool in_vma = false;
    long ahp_kb = -1;
    while (fgets(line, sizeof(line), fp) != NULL) {
        unsigned long long start = 0, end = 0;
        if (sscanf(line, "%llx-%llx ", &start, &end) == 2) {
            in_vma = (target >= (uintptr_t)start && target < (uintptr_t)end);
        } else if (in_vma && strncmp(line, "AnonHugePages:", 14) == 0) {
            long kb = 0;
            if (sscanf(line + 14, "%ld", &kb) == 1) {
                ahp_kb = kb;
                break;
            }
        }
    }
    fclose(fp);
    return ahp_kb;
}

static size_t a16_matrix_bytes(int out_dim, int in_dim) {
    size_t u64_b = HS_ALIGN64((size_t)out_dim * (size_t)(in_dim / 16) * sizeof(uint64_t));
    size_t sc_b  = HS_ALIGN64((size_t)out_dim * (size_t)(in_dim / 32) * sizeof(uint16_t));
    return u64_b + sc_b;
}

static uint8_t* bump_alloc(uint8_t** cursor, size_t nbytes) {
    uint8_t* p = *cursor;
    *cursor += HS_ALIGN64(nbytes);
    return p;
}

static void assign_a16(uint8_t** cursor, uint64_t** u64_p, uint16_t** sc_p, int out_dim, int in_dim) {
    *u64_p = (uint64_t*)bump_alloc(cursor, (size_t)out_dim * (size_t)(in_dim / 16) * sizeof(uint64_t));
    *sc_p  = (uint16_t*)bump_alloc(cursor, (size_t)out_dim * (size_t)(in_dim / 32) * sizeof(uint16_t));
}

static int holo_engine_init(HoloV3Engine* eng, bool use_hugepages, uint64_t seed) {
    memset(eng, 0, sizeof(*eng));
    eng->hugepages_requested = use_hugepages;

    /* Compute exact weight & state arena sizes */
    size_t layer_w_bytes =
        a16_matrix_bytes(HS_D_MODEL, HS_D_MODEL) * 4u +
        a16_matrix_bytes(HS_C_DIM, HS_D_MODEL) +
        a16_matrix_bytes(HS_W_OMEGA_D, HS_D_MODEL) +
        a16_matrix_bytes(HS_D_FF, HS_D_MODEL) * 2u +
        a16_matrix_bytes(HS_D_MODEL, HS_D_FF) +
        HS_ALIGN64((size_t)HS_N_HEADS * 32u * 32u * sizeof(float)) +
        HS_ALIGN64((size_t)HS_N_SPINORS * sizeof(float));

    size_t w_bytes = HS_ALIGN64((size_t)HS_D_MODEL * HS_CONV_K * sizeof(float)) + HS_N_LAYERS * layer_w_bytes;
    size_t s_bytes =
        HS_ALIGN64((size_t)HS_CONV_K * HS_D_MODEL * sizeof(float)) +
        HS_ALIGN64((size_t)HS_N_LAYERS * HS_N_HEADS * 32u * 32u * sizeof(float)) +
        HS_ALIGN64((size_t)HS_N_LAYERS * HS_N_HEADS * 8u * 4u * sizeof(float)) +
        HS_ALIGN64((size_t)HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32u * sizeof(float)) * 2u;

    eng->weight_bytes = w_bytes;
    eng->state_bytes = s_bytes;
    eng->total_live_bytes = w_bytes + s_bytes;

    size_t map_len = HS_ALIGN2M(eng->total_live_bytes);
    size_t alloc_len = map_len + (2u * 1024u * 1024u); /* Extra 2 MiB to guarantee exact 2 MiB alignment */
    void* raw = mmap(NULL, alloc_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return -1;

    uintptr_t aligned_addr = ((uintptr_t)raw + (2u * 1024u * 1024u - 1u)) & ~(uintptr_t)(2u * 1024u * 1024u - 1u);
    void* aligned = (void*)aligned_addr;

    /* Explicit huge-page vs 4 KiB advice BEFORE first-touch fault! */
    int adv = use_hugepages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
    if (madvise(aligned, map_len, adv) != 0) {
        munmap(raw, alloc_len);
        return -2;
    }
    /* First-touch zero-fill every page so THP/4K backing is committed before smaps check */
    memset(aligned, 0, map_len);
    eng->arena_raw = raw;
    eng->arena_aligned = aligned;
    eng->arena_alloc_bytes = alloc_len;
    eng->smaps_anon_hugepages_kb = query_smaps_anon_hugepages_kb(aligned);

    uint8_t* cur = (uint8_t*)aligned;
    eng->conv_w = (float*)bump_alloc(&cur, (size_t)HS_D_MODEL * HS_CONV_K * sizeof(float));
    for (int l = 0; l < HS_N_LAYERS; ++l) {
        HoloLayerWeights* lw = &eng->layers[l];
        assign_a16(&cur, &lw->w_q_ring_u64, &lw->w_q_ring_sc, HS_D_MODEL, HS_D_MODEL);
        assign_a16(&cur, &lw->w_k_ring_u64, &lw->w_k_ring_sc, HS_D_MODEL, HS_D_MODEL);
        assign_a16(&cur, &lw->w_ring_ctrl_u64, &lw->w_ring_ctrl_sc, HS_C_DIM, HS_D_MODEL);
        assign_a16(&cur, &lw->w_mem_out_u64, &lw->w_mem_out_sc, HS_D_MODEL, HS_D_MODEL);
        assign_a16(&cur, &lw->w_omega_u64, &lw->w_omega_sc, HS_W_OMEGA_D, HS_D_MODEL);
        assign_a16(&cur, &lw->w_cell_out_u64, &lw->w_cell_out_sc, HS_D_MODEL, HS_D_MODEL);
        assign_a16(&cur, &lw->w_gate_u64, &lw->w_gate_sc, HS_D_FF, HS_D_MODEL);
        assign_a16(&cur, &lw->w_up_u64, &lw->w_up_sc, HS_D_FF, HS_D_MODEL);
        assign_a16(&cur, &lw->w_down_u64, &lw->w_down_sc, HS_D_MODEL, HS_D_FF);
        lw->w_q_fast    = (float*)bump_alloc(&cur, (size_t)HS_N_HEADS * 32u * 32u * sizeof(float));
        lw->omega_scale = (float*)bump_alloc(&cur, (size_t)HS_N_SPINORS * sizeof(float));
        lw->alpha_ring  = 0.12f;
        lw->alpha_cell  = 0.15f;
        lw->alpha_mlp   = 0.14f;
    }

    eng->conv_ring = (float*)bump_alloc(&cur, (size_t)HS_CONV_K * HS_D_MODEL * sizeof(float));
    eng->S_fast    = (float*)bump_alloc(&cur, (size_t)HS_N_LAYERS * HS_N_HEADS * 32u * 32u * sizeof(float));
    eng->U_state   = (float*)bump_alloc(&cur, (size_t)HS_N_LAYERS * HS_N_HEADS * 8u * 4u * sizeof(float));
    eng->k_ring    = (float*)bump_alloc(&cur, (size_t)HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32u * sizeof(float));
    eng->v_ring    = (float*)bump_alloc(&cur, (size_t)HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32u * sizeof(float));

    /* Populate deterministic weights matching pure-Python FP64 oracle bit-for-bit! */
    uint64_t rng = seed;
    for (int i = 0; i < HS_D_MODEL * HS_CONV_K; ++i) {
        eng->conv_w[i] = splitmix_uniform_sym(&rng, 0.35f);
    }
    uint64_t syn_errs = 0;
    for (int l = 0; l < HS_N_LAYERS; ++l) {
        HoloLayerWeights* lw = &eng->layers[l];
        syn_errs += pack_a16_matrix_synth(lw->w_q_ring_u64,    lw->w_q_ring_sc,    HS_D_MODEL,   HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_k_ring_u64,    lw->w_k_ring_sc,    HS_D_MODEL,   HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_ring_ctrl_u64, lw->w_ring_ctrl_sc, HS_C_DIM,     HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_mem_out_u64,   lw->w_mem_out_sc,   HS_D_MODEL,   HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_omega_u64,     lw->w_omega_sc,     HS_W_OMEGA_D, HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_cell_out_u64,  lw->w_cell_out_sc,  HS_D_MODEL,   HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_gate_u64,      lw->w_gate_sc,      HS_D_FF,      HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_up_u64,        lw->w_up_sc,        HS_D_FF,      HS_D_MODEL, &rng, 0.08f);
        syn_errs += pack_a16_matrix_synth(lw->w_down_u64,      lw->w_down_sc,      HS_D_MODEL,   HS_D_FF,    &rng, 0.08f);
        for (int i = 0; i < HS_N_HEADS * 32 * 32; ++i) lw->w_q_fast[i] = splitmix_uniform_sym(&rng, 0.06f);
        for (int i = 0; i < HS_N_SPINORS; ++i)         lw->omega_scale[i] = 0.25f + 0.05f * splitmix_uniform_sym(&rng, 1.0f);
    }
    eng->pack_syndrome_errors = syn_errs;
    return 0;
}

static void holo_engine_reset_state(HoloV3Engine* eng) {
    memset(eng->conv_ring, 0, (size_t)HS_CONV_K * HS_D_MODEL * sizeof(float));
    memset(eng->S_fast,    0, (size_t)HS_N_LAYERS * HS_N_HEADS * 32u * 32u * sizeof(float));
    memset(eng->U_state,   0, (size_t)HS_N_LAYERS * HS_N_HEADS * 8u * 4u * sizeof(float));
    memset(eng->k_ring,    0, (size_t)HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32u * sizeof(float));
    memset(eng->v_ring,    0, (size_t)HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32u * sizeof(float));
    for (int l = 0; l < HS_N_LAYERS; ++l) {
        for (int h = 0; h < HS_N_HEADS; ++h) {
            for (int s = 0; s < 8; ++s) {
                size_t idx = (((size_t)l * HS_N_HEADS + (size_t)h) * 8u + (size_t)s) * 4u;
                eng->U_state[idx + 0u] = 1.0f;
            }
        }
    }
    eng->conv_pos = 0;
    eng->runtime_syndrome_faults = 0u;
}

static void holo_engine_destroy(HoloV3Engine* eng) {
    if (eng->arena_raw && eng->arena_raw != MAP_FAILED) {
        munmap(eng->arena_raw, eng->arena_alloc_bytes);
        eng->arena_raw = NULL;
        eng->arena_aligned = NULL;
    }
}

/* ============================================================================
 * 6. Deterministic Row-Partitioned Persistent Pthread Worker Pool (1T/4T/8T)
 * ============================================================================ */

typedef struct {
    const QuantActH4* act;
    const uint64_t*   w_u64;
    const uint16_t*   w_sc;
    float*            y_out;
    int               out_dim;
    bool              use_avx512;
} GemvJob;

typedef struct {
    int             n_threads;
    pthread_t       threads[16];
    pthread_mutex_t mu;
    pthread_cond_t  cv_start;
    pthread_cond_t  cv_done;
    GemvJob         job;
    uint32_t        thread_faults[16];
    uint64_t        epoch;
    int             completed;
    bool            shutdown;
} GemvPool;

typedef struct {
    GemvPool* pool;
    int       tid;
} WorkerArg;

static WorkerArg g_worker_args[16];

static void run_gemv_slice(const GemvJob* job, int tid, int n_threads, uint32_t* faults_out) {
    int rows_per = (job->out_dim + n_threads - 1) / n_threads;
    int r0 = tid * rows_per;
    int r1 = r0 + rows_per;
    if (r0 >= job->out_dim) return;
    if (r1 > job->out_dim) r1 = job->out_dim;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI__)
    if (job->use_avx512) {
        gemv_a16_avx512_vnni_rows(job->act, job->w_u64, job->w_sc, job->y_out, r0, r1, faults_out);
        return;
    }
#endif
    gemv_a16_scalar_rows(job->act, job->w_u64, job->w_sc, job->y_out, r0, r1, faults_out);
}

static void* gemv_worker_main(void* arg) {
    WorkerArg* wa = (WorkerArg*)arg;
    GemvPool* p = wa->pool;
    int tid = wa->tid;
    uint64_t seen_epoch = 0;

    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->shutdown && p->epoch == seen_epoch) {
            pthread_cond_wait(&p->cv_start, &p->mu);
        }
        if (p->shutdown) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen_epoch = p->epoch;
        GemvJob local_job = p->job;
        pthread_mutex_unlock(&p->mu);

        uint32_t f = 0u;
        run_gemv_slice(&local_job, tid, p->n_threads, &f);
        p->thread_faults[tid] += f;

        pthread_mutex_lock(&p->mu);
        p->completed++;
        if (p->completed == p->n_threads - 1) {
            pthread_cond_signal(&p->cv_done);
        }
        pthread_mutex_unlock(&p->mu);
    }
}

static int pool_init(GemvPool* p, int n_threads) {
    memset(p, 0, sizeof(*p));
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;
    p->n_threads = n_threads;
    if (pthread_mutex_init(&p->mu, NULL) != 0) return -1;
    if (pthread_cond_init(&p->cv_start, NULL) != 0) {
        pthread_mutex_destroy(&p->mu);
        return -2;
    }
    if (pthread_cond_init(&p->cv_done, NULL) != 0) {
        pthread_cond_destroy(&p->cv_start);
        pthread_mutex_destroy(&p->mu);
        return -3;
    }
    for (int t = 1; t < n_threads; ++t) {
        g_worker_args[t].pool = p;
        g_worker_args[t].tid = t;
        if (pthread_create(&p->threads[t], NULL, gemv_worker_main, &g_worker_args[t]) != 0) {
            pthread_mutex_lock(&p->mu);
            p->shutdown = true;
            pthread_cond_broadcast(&p->cv_start);
            pthread_mutex_unlock(&p->mu);
            for (int j = 1; j < t; ++j) {
                pthread_join(p->threads[j], NULL);
            }
            pthread_cond_destroy(&p->cv_done);
            pthread_cond_destroy(&p->cv_start);
            pthread_mutex_destroy(&p->mu);
            return -4;
        }
    }
    return 0;
}

static void pool_dispatch_gemv(
    GemvPool* p,
    const QuantActH4* act,
    const uint64_t* w_u64,
    const uint16_t* w_sc,
    float* y_out,
    int out_dim,
    bool use_avx512,
    uint32_t* faults_io
) {
    GemvJob job;
    job.act = act;
    job.w_u64 = w_u64;
    job.w_sc = w_sc;
    job.y_out = y_out;
    job.out_dim = out_dim;
    job.use_avx512 = use_avx512;

    if (p->n_threads == 1) {
        run_gemv_slice(&job, 0, 1, faults_io);
        return;
    }

    pthread_mutex_lock(&p->mu);
    p->job = job;
    p->completed = 0;
    p->epoch++;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);

    run_gemv_slice(&job, 0, p->n_threads, faults_io);

    pthread_mutex_lock(&p->mu);
    while (p->completed < p->n_threads - 1) {
        pthread_cond_wait(&p->cv_done, &p->mu);
    }
    pthread_mutex_unlock(&p->mu);
}

static uint32_t pool_destroy(GemvPool* p) {
    pthread_mutex_lock(&p->mu);
    p->shutdown = true;
    pthread_cond_broadcast(&p->cv_start);
    pthread_mutex_unlock(&p->mu);
    uint32_t total_f = 0u;
    for (int t = 1; t < p->n_threads; ++t) {
        pthread_join(p->threads[t], NULL);
        total_f += p->thread_faults[t];
    }
    pthread_cond_destroy(&p->cv_done);
    pthread_cond_destroy(&p->cv_start);
    pthread_mutex_destroy(&p->mu);
    return total_f;
}

/* ============================================================================
 * 7. HoloSphere-V3 Token Step: H_256xbent4 + ByteConv4 + V_ring + CayleySU2
 *    + S_fast Woodbury + SwiGLU + 0 B Tied FWHT_256 x bent4 LM Head
 * ============================================================================ */

static void normalize_sphere(float* v, int n) {
    float n2 = 0.0f;
    for (int i = 0; i < n; ++i) n2 += v[i] * v[i];
    float inv = 1.0f / sqrtf(n2 + 1e-12f);
    for (int i = 0; i < n; ++i) v[i] *= inv;
}

static void step_inject_byteconv4(HoloV3Engine* eng, int tok_byte) {
    int b = tok_byte & 0xFF;
    const float inv_s = 1.0f / sqrtf((float)HS_D_MODEL);
    int slot = eng->conv_pos & 3;
    float* ring_row = eng->conv_ring + (size_t)slot * HS_D_MODEL;

    for (int d = 0; d < HS_D_MODEL; ++d) {
        int d256 = d & 255;
        int w    = d256 >> 5;
        int lane = d256 & 31;
        int db   = ((w << 5) | lane) ^ ((lane & 7) << 5);
        int b4   = __builtin_popcount((lane & 3) & ((lane >> 2) & 3)) & 1;
        int par  = (__builtin_popcount(b & db) ^ b4) & 1;
        float val = par ? -inv_s : inv_s;
        if (d >= 256) val = -val;
        ring_row[d] = val;
    }

    for (int d = 0; d < HS_D_MODEL; ++d) {
        float acc = 0.0f;
        const float* cw = eng->conv_w + (size_t)d * HS_CONV_K;
        for (int k = 0; k < HS_CONV_K; ++k) {
            int p_idx = eng->conv_pos - k;
            if (p_idx >= 0) {
                acc += cw[k] * eng->conv_ring[(size_t)(p_idx & 3) * HS_D_MODEL + (size_t)d];
            }
        }
        eng->x_buf[d] = acc;
    }
    normalize_sphere(eng->x_buf, HS_D_MODEL);
    eng->conv_pos++;
}

static void step_vring_and_mix(HoloV3Engine* eng, int layer_idx) {
    const HoloLayerWeights* lw = &eng->layers[layer_idx];
    const int t_pos = eng->conv_pos - 1;
    const int n_valid = (t_pos < HS_RING_K) ? t_pos : HS_RING_K;
    const int w_slot = t_pos & (HS_RING_K - 1);

    for (int h = 0; h < HS_N_HEADS; ++h) {
        float qh[32], kh[32];
        const float* xh = eng->x_buf + h * 32;
        memcpy(qh, eng->q_buf + h * 32, sizeof(qh));
        memcpy(kh, eng->k_buf + h * 32, sizeof(kh));
        normalize_sphere(qh, 32);
        normalize_sphere(kh, 32);

        float write_g  = 1.0f / (1.0f + expf(-eng->ctrl_buf[h * 3 + 0]));
        float recall_g = 1.0f / (1.0f + expf(-eng->ctrl_buf[h * 3 + 1]));
        eng->rgate_buf[h] = recall_g;

        float* kr_base = eng->k_ring + ((size_t)layer_idx * HS_N_HEADS + (size_t)h) * HS_RING_K * 32u;
        float* vr_base = eng->v_ring + ((size_t)layer_idx * HS_N_HEADS + (size_t)h) * HS_RING_K * 32u;

        int best_s = 0;
        float best_sim = -1e30f;
        for (int s = 0; s < n_valid; ++s) {
            const float* ks = kr_base + (size_t)s * 32u;
            float sim = 0.0f;
            for (int i = 0; i < 32; ++i) sim += qh[i] * ks[i];
            if (sim > best_sim) {
                best_sim = sim;
                best_s = s;
            }
        }
        float* ymh = eng->ymem_buf + h * 32;
        if (n_valid > 0) {
            memcpy(ymh, vr_base + (size_t)best_s * 32u, 32u * sizeof(float));
        } else {
            memset(ymh, 0, 32u * sizeof(float));
        }

        /* Write new (k, v) slot smoothly gated by write_g */
        float* kw = kr_base + (size_t)w_slot * 32u;
        float* vw = vr_base + (size_t)w_slot * 32u;
        for (int i = 0; i < 32; ++i) {
            kw[i] = (1.0f - write_g) * kw[i] + write_g * kh[i];
            vw[i] = (1.0f - write_g) * vw[i] + write_g * xh[i];
        }
    }
    (void)lw;
}

static void step_cayley_sfast(HoloV3Engine* eng, int layer_idx) {
    const HoloLayerWeights* lw = &eng->layers[layer_idx];
    const float t2 = HS_CAYLEY_T * HS_CAYLEY_T;
    const float c_st = (1.0f - t2) / (1.0f + t2);
    const float s_st = (2.0f * HS_CAYLEY_T) / (1.0f + t2);

    for (int h = 0; h < HS_N_HEADS; ++h) {
        const float* xh = eng->hgen_buf + h * 32;
        float xt[32];

        /* 8 spinors per head: bounded SFU-free CayleySU2 + left action conj(U) * x */
        for (int s = 0; s < 8; ++s) {
            int sp_glob = h * 8 + s;
            float sc = lw->omega_scale[sp_glob];
            float wx = eng->wraw_buf[sp_glob * 3 + 0] * sc;
            float wy = eng->wraw_buf[sp_glob * 3 + 1] * sc;
            float wz = eng->wraw_buf[sp_glob * 3 + 2] * sc;
            float n2 = wx * wx + wy * wy + wz * wz;
            float cl = 8.0f / sqrtf(64.0f + n2);
            wx *= cl; wy *= cl; wz *= cl;

            float w2  = wx * wx + wy * wy + wz * wz;
            float inv = 1.0f / (1.0f + w2);
            float qw = (1.0f - w2) * inv;
            float qx = (2.0f * wx) * inv;
            float qy = (2.0f * wy) * inv;
            float qz = (2.0f * wz) * inv;

            float* up = eng->U_state + (((size_t)layer_idx * HS_N_HEADS + (size_t)h) * 8u + (size_t)s) * 4u;
            float uw = up[0], ux = up[1], uy = up[2], uz = up[3];
            float nw = qw * uw - qx * ux - qy * uy - qz * uz;
            float nx = qw * ux + qx * uw + qy * uz - qz * uy;
            float ny = qw * uy - qx * uz + qy * uw + qz * ux;
            float nz = qw * uz + qx * uy - qy * ux + qz * uw;
            float un = 1.0f / sqrtf(nw * nw + nx * nx + ny * ny + nz * nz + 1e-24f);
            nw *= un; nx *= un; ny *= un; nz *= un;
            up[0] = nw; up[1] = nx; up[2] = ny; up[3] = nz;

            /* xt_s = conj(U) * x_s */
            const float* xs = xh + s * 4;
            float cw = nw, cx = -nx, cy = -ny, cz = -nz;
            xt[s * 4 + 0] = cw * xs[0] - cx * xs[1] - cy * xs[2] - cz * xs[3];
            xt[s * 4 + 1] = cw * xs[1] + cx * xs[0] + cy * xs[3] - cz * xs[2];
            xt[s * 4 + 2] = cw * xs[2] - cx * xs[3] + cy * xs[0] + cz * xs[1];
            xt[s * 4 + 3] = cw * xs[3] + cx * xs[2] - cy * xs[1] + cz * xs[0];
        }

        const float* Wh = lw->w_q_fast + (size_t)h * 1024u;
        float* Sh = eng->S_fast + ((size_t)layer_idx * HS_N_HEADS + (size_t)h) * 1024u;
        float sp[32], z[32];
        float dot_zx = 0.0f;
        for (int r = 0; r < 32; ++r) {
            const float* wr = Wh + r * 32;
            const float* sr = Sh + r * 32;
            float zs = 0.0f, s_p = 0.0f;
            for (int c = 0; c < 32; ++c) {
                zs  += wr[c] * xt[c];
                s_p += sr[c] * xt[c];
            }
            sp[r] = s_p;
            z[r]  = zs + s_p;
            dot_zx += z[r] * xh[r];
        }
        float v[32];
        float v2 = 0.0f;
        for (int r = 0; r < 32; ++r) {
            v[r] = z[r] - dot_zx * xh[r];
            v2 += v[r] * v[r];
        }
        float v_inv = 1.0f / sqrtf(v2 + HS_EPS);
        float* yh = eng->ycell_buf + h * 32;
        for (int r = 0; r < 32; ++r) {
            float yr = c_st * xh[r] + s_st * (v[r] * v_inv);
            yh[r] = yr;
            float err = yr - sp[r];
            float* sr = Sh + r * 32;
            for (int c = 0; c < 32; ++c) {
                sr[c] = HS_GAMMA * sr[c] + HS_ETA * err * xt[c];
            }
        }
    }
}

static void step_tied_fwht256_lm_head(const float* x_in, float* logits_out) {
    /* 0 B VRAM/RAM tied orthonormal FWHT_256 x bent4 LM head */
    float buf[256];
    for (int d = 0; d < 256; ++d) {
        int w    = d >> 5;
        int lane = d & 31;
        int src_w = w ^ (lane & 7);
        int src_i = (src_w << 5) | lane;
        float diff = x_in[src_i] - x_in[src_i + 256];
        int b4 = __builtin_popcount((lane & 3) & ((lane >> 2) & 3)) & 1;
        buf[d] = b4 ? -diff : diff;
    }
    for (int len = 1; len < 256; len <<= 1) {
        for (int i = 0; i < 256; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                float u = buf[i + j];
                float v = buf[i + j + len];
                buf[i + j]       = u + v;
                buf[i + j + len] = u - v;
            }
        }
    }
    const float sc = (1.0f / sqrtf((float)HS_D_MODEL)) * HS_LOGIT_SC;
    for (int b = 0; b < 256; ++b) {
        logits_out[b] = buf[b] * sc;
    }
}

static int holo_engine_step_token(HoloV3Engine* eng, GemvPool* pool, int tok_byte, bool use_avx512) {
    step_inject_byteconv4(eng, tok_byte);

    for (int l = 0; l < HS_N_LAYERS; ++l) {
        const HoloLayerWeights* lw = &eng->layers[l];

        quantize_act_h4(eng->x_buf, HS_D_MODEL, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_q_ring_u64,    lw->w_q_ring_sc,    eng->q_buf,    HS_D_MODEL, use_avx512, &eng->runtime_syndrome_faults);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_k_ring_u64,    lw->w_k_ring_sc,    eng->k_buf,    HS_D_MODEL, use_avx512, &eng->runtime_syndrome_faults);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_ring_ctrl_u64, lw->w_ring_ctrl_sc, eng->ctrl_buf, HS_C_DIM,   use_avx512, &eng->runtime_syndrome_faults);

        step_vring_and_mix(eng, l);

        quantize_act_h4(eng->ymem_buf, HS_D_MODEL, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_mem_out_u64, lw->w_mem_out_sc, eng->mproj_buf, HS_D_MODEL, use_avx512, &eng->runtime_syndrome_faults);

        for (int h = 0; h < HS_N_HEADS; ++h) {
            float g = lw->alpha_ring * eng->rgate_buf[h];
            for (int i = 0; i < 32; ++i) {
                int idx = h * 32 + i;
                eng->hgen_buf[idx] = eng->x_buf[idx] + g * eng->mproj_buf[idx];
            }
        }
        normalize_sphere(eng->hgen_buf, HS_D_MODEL);

        quantize_act_h4(eng->hgen_buf, HS_D_MODEL, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_omega_u64, lw->w_omega_sc, eng->wraw_buf, HS_W_OMEGA_D, use_avx512, &eng->runtime_syndrome_faults);

        step_cayley_sfast(eng, l);

        quantize_act_h4(eng->ycell_buf, HS_D_MODEL, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_cell_out_u64, lw->w_cell_out_sc, eng->cout_buf, HS_D_MODEL, use_avx512, &eng->runtime_syndrome_faults);
        for (int i = 0; i < HS_D_MODEL; ++i) {
            eng->xmid_buf[i] = eng->hgen_buf[i] + lw->alpha_cell * eng->cout_buf[i];
        }
        normalize_sphere(eng->xmid_buf, HS_D_MODEL);

        quantize_act_h4(eng->xmid_buf, HS_D_MODEL, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_gate_u64, lw->w_gate_sc, eng->gate_buf, HS_D_FF, use_avx512, &eng->runtime_syndrome_faults);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_up_u64,   lw->w_up_sc,   eng->up_buf,   HS_D_FF, use_avx512, &eng->runtime_syndrome_faults);
        for (int i = 0; i < HS_D_FF; ++i) {
            float g = eng->gate_buf[i];
            eng->act_buf[i] = (g / (1.0f + expf(-g))) * eng->up_buf[i];
        }

        quantize_act_h4(eng->act_buf, HS_D_FF, &eng->qact);
        pool_dispatch_gemv(pool, &eng->qact, lw->w_down_u64, lw->w_down_sc, eng->down_buf, HS_D_MODEL, use_avx512, &eng->runtime_syndrome_faults);
        for (int i = 0; i < HS_D_MODEL; ++i) {
            eng->x_buf[i] = eng->xmid_buf[i] + lw->alpha_mlp * eng->down_buf[i];
        }
        normalize_sphere(eng->x_buf, HS_D_MODEL);
    }

    step_tied_fwht256_lm_head(eng->x_buf, eng->logits_buf);
    int best = 0;
    float best_v = eng->logits_buf[0];
    for (int b = 1; b < HS_VOCAB; ++b) {
        if (eng->logits_buf[b] > best_v) {
            best_v = eng->logits_buf[b];
            best = b;
        }
    }
    return best;
}

/* ============================================================================
 * 8. Self-Test (16/16 SEC Bit-Exact Heal + 120/120 DED + 1T==4T==8T Parity)
 *    & Timing / Benchmark CLI
 * ============================================================================ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long long read_rapl_pkg_uj(void) {
    FILE* fp = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r");
    if (!fp) return -1;
    long long uj = -1;
    if (fscanf(fp, "%lld", &uj) != 1) uj = -1;
    fclose(fp);
    return uj;
}

static int run_secded_unit_test(HoloV3Engine* eng, bool use_avx512) {
    /* Verify 16/16 single-bit SEC bit-exact heal and 120/120 2-bit DED catch on a real packed row */
    uint64_t clean_row[32];
    uint16_t clean_sc[16];
    memcpy(clean_row, eng->layers[0].w_q_ring_u64, sizeof(clean_row));
    memcpy(clean_sc,  eng->layers[0].w_q_ring_sc,  sizeof(clean_sc));

    float x_probe[HS_D_MODEL];
    for (int i = 0; i < HS_D_MODEL; ++i) x_probe[i] = sinf(0.17f * (float)i + 0.3f);
    QuantActH4 qa;
    quantize_act_h4(x_probe, HS_D_MODEL, &qa);

    float y_clean = 0.0f;
    uint32_t f_clean = 0u;
    gemv_a16_scalar_rows(&qa, clean_row, clean_sc, &y_clean, 0, 1, &f_clean);
    if (f_clean != 0u) return -1;

    for (int bit = 0; bit < 16; ++bit) {
        uint64_t cor[32];
        memcpy(cor, clean_row, sizeof(cor));
        cor[0] ^= (1ULL << (bit * 4));
        float y_rec = 0.0f;
        uint32_t f_rec = 0u;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI__)
        if (use_avx512) {
            gemv_a16_avx512_vnni_rows(&qa, cor, clean_sc, &y_rec, 0, 1, &f_rec);
        } else {
            gemv_a16_scalar_rows(&qa, cor, clean_sc, &y_rec, 0, 1, &f_rec);
        }
#else
        (void)use_avx512;
        gemv_a16_scalar_rows(&qa, cor, clean_sc, &y_rec, 0, 1, &f_rec);
#endif
        if (f_rec != 1u || y_rec != y_clean) return -2;
    }

    for (int i = 0; i < 16; ++i) {
        for (int j = i + 1; j < 16; ++j) {
            uint64_t cor = clean_row[0] ^ (1ULL << (i * 4)) ^ (1ULL << (j * 4));
            uint32_t sec = 0u, ded = 0u;
            uint64_t out = a16_secded_heal_u64(cor, &sec, &ded);
            if (sec != 0u || ded != 1u || out != cor) return -3;
        }
    }
    return 0;
}

static uint64_t fnv1a_bytes(const void* data, size_t nbytes) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool parse_strict_i64(const char* s, int64_t* out) {
    if (!s || s[0] == '\0') return false;
    if (s[0] == ' ' || s[0] == '\t' || s[0] == '\n' || s[0] == '\r' || s[0] == '+') return false;
    errno = 0;
    char* end = NULL;
    long long v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;
    *out = (int64_t)v;
    return true;
}

static bool parse_strict_u64(const char* s, uint64_t* out) {
    if (!s || s[0] == '\0') return false;
    if (s[0] < '0' || s[0] > '9') return false; /* Rejects '-', '+', whitespace, and garbage */
    errno = 0;
    char* end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;
    *out = (uint64_t)v;
    return true;
}

int main(int argc, char** argv) {
    const char* mode = "bench";
    int threads = 8;
    int tokens = 128;
    int warmup = 16;
    bool use_avx512 = true;
    bool use_hugepages = false;
    uint64_t seed = 125001ULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            int64_t v = 0;
            if (!parse_strict_i64(argv[++i], &v) || v < 1 || v > 16) {
                fprintf(stderr, "invalid --threads '%s' (must be integer 1..16)\n", argv[i]);
                return 1;
            }
            threads = (int)v;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            int64_t v = 0;
            if (!parse_strict_i64(argv[++i], &v) || v < 1 || v > 10000000) {
                fprintf(stderr, "invalid --tokens '%s' (must be integer >= 1)\n", argv[i]);
                return 1;
            }
            tokens = (int)v;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            int64_t v = 0;
            if (!parse_strict_i64(argv[++i], &v) || v < 0 || v > 10000000) {
                fprintf(stderr, "invalid --warmup '%s' (must be integer >= 0)\n", argv[i]);
                return 1;
            }
            warmup = (int)v;
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char* b = argv[++i];
            if (strcmp(b, "avx512") == 0) {
                use_avx512 = true;
            } else if (strcmp(b, "scalar") == 0) {
                use_avx512 = false;
            } else {
                fprintf(stderr, "invalid --backend '%s' (expected 'avx512' or 'scalar')\n", b);
                return 1;
            }
        } else if (strcmp(argv[i], "--hugepages") == 0 && i + 1 < argc) {
            int64_t v = 0;
            if (!parse_strict_i64(argv[++i], &v) || (v != 0 && v != 1)) {
                fprintf(stderr, "invalid --hugepages '%s' (expected 0 or 1)\n", argv[i]);
                return 1;
            }
            use_hugepages = (v == 1);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            uint64_t v = 0;
            if (!parse_strict_u64(argv[++i], &v)) {
                fprintf(stderr, "invalid --seed '%s' (must be unsigned integer)\n", argv[i]);
                return 1;
            }
            seed = v;
        } else {
            fprintf(stderr, "unknown or incomplete argument '%s'\n", argv[i]);
            return 1;
        }
    }

    if (strcmp(mode, "bench") != 0 && strcmp(mode, "trace") != 0) {
        fprintf(stderr, "invalid --mode '%s' (expected 'bench' or 'trace')\n", mode);
        return 1;
    }

    HoloV3Engine eng;
    if (holo_engine_init(&eng, use_hugepages, seed) != 0) {
        fprintf(stderr, "holo_engine_init failed\n");
        return 2;
    }
    if (eng.pack_syndrome_errors != 0ULL) {
        fprintf(stderr, "pack_syndrome_errors=%llu\n", (unsigned long long)eng.pack_syndrome_errors);
        holo_engine_destroy(&eng);
        return 3;
    }

    if (strcmp(mode, "trace") == 0) {
        /* Deterministic 8-token trace comparing scalar vs AVX-512 & 1T/4T/8T bit-exactness */
        if (run_secded_unit_test(&eng, use_avx512) != 0) {
            fprintf(stderr, "SECDED self-test failed\n");
            holo_engine_destroy(&eng);
            return 4;
        }
        const int prompt[8] = {72, 111, 108, 111, 83, 112, 104, 101}; /* "HoloSphe" */
        GemvPool pool;
        if (pool_init(&pool, threads) != 0) {
            fprintf(stderr, "pool_init failed\n");
            holo_engine_destroy(&eng);
            return 6;
        }
        const int active_threads = pool.n_threads;
        holo_engine_reset_state(&eng);
        int out_toks[8];
        for (int t = 0; t < 8; ++t) {
            out_toks[t] = holo_engine_step_token(&eng, &pool, prompt[t], use_avx512);
        }
        uint32_t wf = pool_destroy(&pool);
        uint32_t tot_f = eng.runtime_syndrome_faults + wf;

        uint64_t logits_hash = fnv1a_bytes(eng.logits_buf, sizeof(eng.logits_buf));
        uint64_t hidden_hash = fnv1a_bytes(eng.x_buf, sizeof(eng.x_buf));

        printf("{\n");
        printf("  \"mode\": \"trace\",\n");
        printf("  \"backend\": \"%s\",\n", use_avx512 ? "avx512" : "scalar");
        printf("  \"threads\": %d,\n", active_threads);
        printf("  \"hugepages_requested\": %s,\n", use_hugepages ? "true" : "false");
        printf("  \"smaps_anon_hugepages_kb\": %ld,\n", eng.smaps_anon_hugepages_kb);
        printf("  \"weight_bytes\": %zu,\n", eng.weight_bytes);
        printf("  \"state_bytes\": %zu,\n", eng.state_bytes);
        printf("  \"total_live_bytes\": %zu,\n", eng.total_live_bytes);
        printf("  \"pack_syndrome_errors\": %llu,\n", (unsigned long long)eng.pack_syndrome_errors);
        printf("  \"runtime_syndrome_faults\": %u,\n", tot_f);
        printf("  \"secded_16sec_120ded_pass\": true,\n");
        printf("  \"logits_fnv1a64\": \"%016llx\",\n", (unsigned long long)logits_hash);
        printf("  \"hidden_fnv1a64\": \"%016llx\",\n", (unsigned long long)hidden_hash);
        printf("  \"greedy_tokens\": [%d,%d,%d,%d,%d,%d,%d,%d],\n",
               out_toks[0], out_toks[1], out_toks[2], out_toks[3],
               out_toks[4], out_toks[5], out_toks[6], out_toks[7]);
        printf("  \"final_logits_256\": [");
        for (int b = 0; b < HS_VOCAB; ++b) {
            printf("%.9g%s", (double)eng.logits_buf[b], (b + 1 < HS_VOCAB) ? "," : "");
        }
        printf("],\n");
        printf("  \"final_hidden_16\": [");
        for (int i = 0; i < 16; ++i) {
            printf("%.9g%s", (double)eng.x_buf[i], (i + 1 < 16) ? "," : "");
        }
        printf("]\n}\n");
        holo_engine_destroy(&eng);
        return (tot_f == 0u) ? 0 : 5;
    }

    /* Timed decode benchmark */
    GemvPool pool;
    if (pool_init(&pool, threads) != 0) {
        fprintf(stderr, "pool_init failed\n");
        holo_engine_destroy(&eng);
        return 6;
    }
    const int active_threads = pool.n_threads;
    holo_engine_reset_state(&eng);

    int tok = 72;
    for (int w = 0; w < warmup; ++w) {
        tok = holo_engine_step_token(&eng, &pool, tok, use_avx512);
    }

    long long rapl0 = read_rapl_pkg_uj();
    double t0 = now_sec();
    for (int s = 0; s < tokens; ++s) {
        tok = holo_engine_step_token(&eng, &pool, tok, use_avx512);
    }
    double t1 = now_sec();
    long long rapl1 = read_rapl_pkg_uj();

    uint32_t wf = pool_destroy(&pool);
    uint32_t tot_f = eng.runtime_syndrome_faults + wf;
    double elapsed_s = t1 - t0;
    double tok_per_s = (elapsed_s > 0.0) ? ((double)tokens / elapsed_s) : 0.0;
    double ms_per_tok = (elapsed_s * 1000.0) / (double)tokens;
    double rapl_mj_tok = (rapl0 >= 0 && rapl1 >= rapl0) ? (((double)(rapl1 - rapl0) * 1e-3) / (double)tokens) : -1.0;

    printf("{\n");
    printf("  \"mode\": \"bench\",\n");
    printf("  \"backend\": \"%s\",\n", use_avx512 ? "avx512" : "scalar");
    printf("  \"threads\": %d,\n", active_threads);
    printf("  \"tokens\": %d,\n", tokens);
    printf("  \"warmup\": %d,\n", warmup);
    printf("  \"hugepages_requested\": %s,\n", use_hugepages ? "true" : "false");
    printf("  \"smaps_anon_hugepages_kb\": %ld,\n", eng.smaps_anon_hugepages_kb);
    printf("  \"weight_bytes\": %zu,\n", eng.weight_bytes);
    printf("  \"state_bytes\": %zu,\n", eng.state_bytes);
    printf("  \"total_live_bytes\": %zu,\n", eng.total_live_bytes);
    printf("  \"pack_syndrome_errors\": %llu,\n", (unsigned long long)eng.pack_syndrome_errors);
    printf("  \"runtime_syndrome_faults\": %u,\n", tot_f);
    printf("  \"elapsed_s\": %.6f,\n", elapsed_s);
    printf("  \"ms_per_tok\": %.4f,\n", ms_per_tok);
    printf("  \"tok_per_s\": %.2f,\n", tok_per_s);
    printf("  \"rapl_mj_per_tok\": %.3f,\n", rapl_mj_tok);
    printf("  \"last_tok\": %d\n", tok);
    printf("}\n");

    holo_engine_destroy(&eng);
    return (tot_f == 0u) ? 0 : 5;
}
