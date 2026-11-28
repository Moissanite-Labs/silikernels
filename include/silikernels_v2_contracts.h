/*
 * silikernels_v2_contracts.h — Shared V2 Ampere sm_86 & AVX-512 Contracts
 *
 * Provides:
 *   - Carry-free [1..127] SWAR127 domain-guarded two's-complement negation
 *   - 0-ULP canonical 128-lane binary tree reduction (CPU reference matching CUDA/AVX-512)
 *   - Extended Hamming [16, 11, 4] 64-bit word SECDED (SEC repair / DED fail-closed trap)
 *   - IEEE CRC32 (0xEDB88320) & IRIS (0x9E3779B9) telemetry mixer
 *   - 22-bit radix-128 3-limb activation struct & B*=2 CPU/GPU cold crossover threshold
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SILIKERNELS_V2_CONTRACTS_H
#define SILIKERNELS_V2_CONTRACTS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "razor_abi_4byte_sys.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SILIKERNELS_V2_IRIS_WEYL_MIX       0x9E3779B9u
#define SILIKERNELS_V2_WOODBURY_BYTES      256u
#define SILIKERNELS_V2_WOODBURY_FLOATS     64u
#define SILIKERNELS_V2_B_STAR_COLD_3BPW    2u
#define SILIKERNELS_V2_B_STAR_COLD_45BPW   3u
#define SILIKERNELS_V2_HIDDEN_MAX          2304u
#define SILIKERNELS_V2_GRP32_MAX           (SILIKERNELS_V2_HIDDEN_MAX / 32u)

typedef struct SilikernelsV2ActQ22 {
    int8_t  x_hi[SILIKERNELS_V2_HIDDEN_MAX];
    int8_t  x_md[SILIKERNELS_V2_HIDDEN_MAX];
    int8_t  x_lo[SILIKERNELS_V2_HIDDEN_MAX];
    float   x_sc[SILIKERNELS_V2_GRP32_MAX];
    int32_t x_sumq[SILIKERNELS_V2_GRP32_MAX];
    int     in_dim;
} SilikernelsV2ActQ22;

typedef struct SilikernelsV2SecdedStats {
    uint32_t sec_repaired;
    uint32_t ded_trapped;
    int      safe_halt;
} SilikernelsV2SecdedStats;

typedef struct SilikernelsV2ColdJob {
    const void*  w13_ptr;
    const void*  w13_sc;
    const void*  w2_ptr;
    const void*  w2_sc;
    float        weight;
    int32_t      tok_idx;
    int32_t      is_45bpw;
} SilikernelsV2ColdJob;

/* Carry-free [1..127] SWAR domain predicate across 8 packed bytes */
static inline int silikernels_v2_swar127_in_domain_u64_c(uint64_t x) {
    return (((x & 0x8080808080808080ULL) == 0ULL) &&
            (((x - 0x0101010101010101ULL) & (~x) & 0x8080808080808080ULL) == 0ULL))
        ? 1
        : 0;
}

/* Fail-closed SWAR127 negation: returns 0ULL if any byte is outside [1..127] */
static inline uint64_t silikernels_v2_swar127_negate_u64_c(uint64_t x) {
    if (!silikernels_v2_swar127_in_domain_u64_c(x)) return 0ULL;
    return (~x) + 0x0101010101010101ULL;
}

/* Explicit checked SWAR127 negation: returns 0 on valid input, -1 and *out=0ULL otherwise */
static inline int silikernels_v2_swar127_negate_u64_checked_c(uint64_t x, uint64_t* out) {
    if (!out) return -1;
    if (!silikernels_v2_swar127_in_domain_u64_c(x)) {
        *out = 0ULL;
        return -1;
    }
    *out = (~x) + 0x0101010101010101ULL;
    return 0;
}

static inline uint32_t silikernels_v2_popc32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(x);
#else
    uint32_t c = 0u;
    while (x != 0u) { c += (x & 1u); x >>= 1u; }
    return c;
#endif
}

static inline uint32_t silikernels_v2_secded_syn_u16(uint16_t odd16) {
    const uint32_t x  = (uint32_t)odd16;
    const uint32_t p0 = silikernels_v2_popc32(x & 0xFFFFu) & 1u;
    const uint32_t s0 = silikernels_v2_popc32(x & 0xAAAAu) & 1u;
    const uint32_t s1 = silikernels_v2_popc32(x & 0xCCCCu) & 1u;
    const uint32_t s2 = silikernels_v2_popc32(x & 0xF0F0u) & 1u;
    const uint32_t s3 = silikernels_v2_popc32(x & 0xFF00u) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

static inline uint32_t silikernels_v2_secded_syn_w64(uint64_t w64) {
    const uint32_t w0 = (uint32_t)w64;
    const uint32_t w1 = (uint32_t)(w64 >> 32);
    const uint32_t px = (w0 ^ w1) & 0x11111111u;
    const uint32_t p0 = silikernels_v2_popc32(px) & 1u;
    const uint32_t s0 = silikernels_v2_popc32(px & 0x10101010u) & 1u;
    const uint32_t s1 = silikernels_v2_popc32(px & 0x11001100u) & 1u;
    const uint32_t s2 = silikernels_v2_popc32(px & 0x11110000u) & 1u;
    const uint32_t s3 = silikernels_v2_popc32(w1 & 0x11111111u) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

static inline void silikernels_v2_secded_heal_w64(uint64_t* w64, SilikernelsV2SecdedStats* stats) {
    const uint32_t syn = silikernels_v2_secded_syn_w64(*w64);
    if (syn != 0u) {
        if ((syn & 1u) != 0u) {
            const uint32_t pos = (syn >> 1) & 15u;
            *w64 ^= (1ULL << (pos * 4u));
            if (stats) stats->sec_repaired += 1u;
        } else {
            if (stats) {
                stats->ded_trapped += 1u;
                stats->safe_halt = 1;
            }
        }
    }
}

static inline uint32_t silikernels_v2_crc32_ieee(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; ++i) {
        crc ^= (uint32_t)p[i];
        for (int k = 0; k < 8; ++k) {
            const uint32_t mask = (uint32_t)(-((int32_t)(crc & 1u)));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

static inline uint32_t silikernels_v2_iris_mix_word_u32(uint32_t state, uint32_t word) {
    uint32_t x = state ^ (word + SILIKERNELS_V2_IRIS_WEYL_MIX + (state << 6) + (state >> 2));
    x ^= (x >> 16);
    x *= 0x85EBCA6Bu;
    x ^= (x >> 13);
    x *= 0xC2B2AE35u;
    x ^= (x >> 16);
    return x;
}

/* Exact 0-ULP canonical 128-lane binary tree reduction matching CUDA tree_reduce_128_sm86 */
static inline float silikernels_v2_canonical_tree_reduce_128_c(const float* xs) {
    float warps[4];
    for (int w = 0; w < 4; ++w) {
        float lane[32];
        for (int i = 0; i < 32; ++i) lane[i] = xs[w * 32 + i];
        for (int stride = 1; stride <= 16; stride <<= 1) {
            float nxt[32];
            for (int i = 0; i < 32; ++i) nxt[i] = lane[i] + lane[i ^ stride];
            for (int i = 0; i < 32; ++i) lane[i] = nxt[i];
        }
        warps[w] = lane[0];
    }
    return (warps[0] + warps[1]) + (warps[2] + warps[3]);
}

#ifdef __cplusplus
}
#endif

#endif /* SILIKERNELS_V2_CONTRACTS_H */
