/*
 * Self-contained exact E_8^4 DP table generator for the AVX2 smoke tests.
 *
 * This is a C11 port of the siliquant harness builders
 *   studies/audit115r-e8had-simd-ballpack/c-bench/run_e8had_simd_bench.py
 *     ::_build_e8_4_tables_self_contained / _build_exact_reciprocal / _init_e8
 * (same recurrences, same symbol order, same reciprocal synthesis), so the
 * committed smoke tests can init libe8had_simd.so / libe8_stage1.so without
 * the Python harness. Functional tests use small m_max (<= 100) where every
 * intermediate fits in 128 bits and every divisor in 64 bits; the
 * overflow-rejection tests use m_max 1490/2048 in counts_only mode (init
 * rejects on the c4 width check before reading any other table).
 *
 * Only used by the test drivers in this directory; not part of any .so.
 */
#ifndef E8_TABLES_GEN_H
#define E8_TABLES_GEN_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 gen_u128;
typedef struct { uint64_t v[3]; } gen_u192; /* little-endian limbs */

static int gen_bitlen64(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }
static int gen_bitlen128(gen_u128 v) {
    if ((uint64_t)(v >> 64)) return 64 + gen_bitlen64((uint64_t)(v >> 64));
    return gen_bitlen64((uint64_t)v);
}

static gen_u192 u192_from_u128(gen_u128 a) {
    gen_u192 r;
    r.v[0] = (uint64_t)a;
    r.v[1] = (uint64_t)(a >> 64);
    r.v[2] = 0;
    return r;
}
static int u192_cmp(gen_u192 a, gen_u192 b) {
    for (int i = 2; i >= 0; --i)
        if (a.v[i] != b.v[i]) return a.v[i] < b.v[i] ? -1 : 1;
    return 0;
}
static gen_u192 u192_sub(gen_u192 a, gen_u192 b) {
    gen_u192 r; uint64_t borrow = 0;
    for (int i = 0; i < 3; ++i) {
        gen_u128 d = (gen_u128)a.v[i] - b.v[i] - borrow;
        r.v[i] = (uint64_t)d;
        borrow = (d >> 64) ? 1u : 0u;
    }
    return r;
}
/* a (u128) * b (u64) -> u192 */
static gen_u192 mul_u128_u64(gen_u128 a, uint64_t b) {
    gen_u192 r;
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64);
    gen_u128 p00 = (gen_u128)a0 * b;
    gen_u128 p10 = (gen_u128)a1 * b;
    r.v[0] = (uint64_t)p00;
    gen_u128 t = (p00 >> 64) + (uint64_t)p10;
    r.v[1] = (uint64_t)t;
    r.v[2] = (uint64_t)(p10 >> 64) + (uint64_t)(t >> 64);
    return r;
}
/* (1 << shift) as u192; shift must be < 192 */
static gen_u192 u192_shl1(uint32_t shift) {
    gen_u192 r = {{0, 0, 0}};
    r.v[shift >> 6] = 1ULL << (shift & 63u);
    return r;
}
/* a / d and a % d for u192 a, u64 d */
static gen_u192 div_u192_u64(gen_u192 a, uint64_t d, uint64_t *rem) {
    gen_u192 q = {{0, 0, 0}};
    uint64_t r = 0;
    for (int i = 2; i >= 0; --i) {
        gen_u128 cur = ((gen_u128)r << 64) | a.v[i];
        q.v[i] = (uint64_t)(cur / d);
        r = (uint64_t)(cur % d);
    }
    *rem = r;
    return q;
}

/* Port of _build_exact_reciprocal: universal Barrett reciprocal.
 * divisor must fit u64; the multiplier is returned as u192 (rec_root stores
 * 192-bit multipliers). Callers assert the width required by their table. */
static int gen_recip(uint64_t divisor, gen_u128 max_dividend, gen_u192 *m_out, uint32_t *s_out) {
    if (divisor == 0) return -1;
    if (divisor == 1) { gen_u192 one = {{1, 0, 0}}; *m_out = one; *s_out = 0; return 0; }
    if ((divisor & (divisor - 1)) == 0) {
        gen_u192 one = {{1, 0, 0}};
        *m_out = one; *s_out = (uint32_t)(gen_bitlen64(divisor) - 1); return 0;
    }
    int k = gen_bitlen128(max_dividend);
    if (k < 1) k = 1;
    for (int extra = 0; extra <= gen_bitlen64(divisor) + 3; ++extra) {
        uint32_t shift = (uint32_t)(k + extra);
        if (shift >= 192) return -1;
        gen_u192 two_s = u192_shl1(shift);
        uint64_t rem;
        gen_u192 q = div_u192_u64(two_s, divisor, &rem);
        gen_u192 mult = q;
        if (rem) { /* mult = q + 1 */
            uint64_t c = 1;
            for (int i = 0; i < 3 && c; ++i) {
                gen_u128 s = (gen_u128)mult.v[i] + c;
                mult.v[i] = (uint64_t)s;
                c = (uint64_t)(s >> 64);
            }
        }
        /* md = mult * divisor (4-limb product; if it exceeds 192 bits this
         * candidate cannot satisfy md - two_s < divisor, so skip it) */
        uint64_t md4[4] = {0, 0, 0, 0};
        {
            gen_u128 carry = 0;
            for (int i = 0; i < 3; ++i) {
                gen_u128 cur = (gen_u128)mult.v[i] * divisor + md4[i] + carry;
                md4[i] = (uint64_t)cur;
                carry = cur >> 64;
            }
            md4[3] = (uint64_t)carry;
        }
        if (md4[3] != 0) continue;
        gen_u192 md = {{md4[0], md4[1], md4[2]}};
        if (u192_cmp(md, two_s) >= 0) {
            gen_u192 diff = u192_sub(md, two_s); /* < divisor <= 2^64 */
            gen_u192 prod = mul_u128_u64(max_dividend, diff.v[0]);
            if (u192_cmp(prod, two_s) < 0) {
                *m_out = mult;
                *s_out = shift;
                return 0;
            }
        }
    }
    return -1;
}

typedef struct {
    int m_max, r_max, t_max;
    uint64_t count;            /* low 64 bits of c_root[m_max+1] */
    gen_u128 count128;         /* full c_root[m_max+1] */
    int nbytes;                /* bytes needed to hold count */
    uint64_t *c_root;          /* (m_max+2) * 3 limbs */
    uint64_t *s2, *w2;         /* (m_max+1) * 2 limbs (u128) */
    uint64_t *s1, *w1;         /* (m_max+1) */
    uint64_t *rec_root_m; uint32_t *rec_root_s; /* 3*(m_max+1), (m_max+1) */
    uint64_t *rec_b1_m; uint32_t *rec_b1_s;     /* 2*(m_max+1), (m_max+1) */
    uint64_t *rec_s1_m; uint32_t *rec_s1_s;     /* 2*(m_max+1), (m_max+1) */
    uint64_t *even_c8;         /* m_max+1 */
    uint32_t *even_c4;         /* 2*(r_max+1) */
    uint16_t *even_c2;         /* 2*(r_max+1) */
    uint16_t *even_off;        /* 2*(r_max+2) */
    int8_t *even_pts[2]; int even_n[2];
    uint32_t *odd_c4;          /* 2*(t_max+1) */
    uint16_t *odd_c2;          /* 2*(t_max+1) */
    uint16_t *odd_off;         /* 2*(t_max+2) */
    int8_t *odd_pts[2]; int odd_n[2];
    uint64_t *rec_ec4_m; uint32_t *rec_ec4_s;   /* 2*(r_max+1) each */
    uint32_t *rec_ec2_m; uint32_t *rec_ec2_s;
    uint64_t *rec_oc4_m; uint32_t *rec_oc4_s;   /* 2*(t_max+1) each */
    uint32_t *rec_oc2_m; uint32_t *rec_oc2_s;
} E8Tables;

static void e8tables_free(E8Tables *t) {
    if (!t) return;
    free(t->c_root); free(t->s2); free(t->w2); free(t->s1); free(t->w1);
    free(t->rec_root_m); free(t->rec_root_s);
    free(t->rec_b1_m); free(t->rec_b1_s);
    free(t->rec_s1_m); free(t->rec_s1_s);
    free(t->even_c8); free(t->even_c4); free(t->even_c2); free(t->even_off);
    free(t->even_pts[0]); free(t->even_pts[1]);
    free(t->odd_c4); free(t->odd_c2); free(t->odd_off);
    free(t->odd_pts[0]); free(t->odd_pts[1]);
    free(t->rec_ec4_m); free(t->rec_ec4_s); free(t->rec_ec2_m); free(t->rec_ec2_s);
    free(t->rec_oc4_m); free(t->rec_oc4_s); free(t->rec_oc2_m); free(t->rec_oc2_s);
    memset(t, 0, sizeof(*t));
}

/* counts_only: build just the c2/c4 count arrays (for overflow-rejection tests
 * at large m_max); all other arrays are zero-filled at the right sizes, because
 * the libraries reject such m_max on the narrowed-table check before reading
 * anything else. */
static int e8tables_build(E8Tables *t, int m_max, int counts_only) {
    memset(t, 0, sizeof(*t));
    const int r_max = 2 * m_max;
    const int t_max = m_max - 1;
    const size_t M1 = (size_t)m_max + 1, M2 = (size_t)m_max + 2;
    const size_t R1 = (size_t)r_max + 1, R2 = (size_t)r_max + 2;
    const size_t T1 = (size_t)t_max + 1, T2 = (size_t)t_max + 2;
    t->m_max = m_max; t->r_max = r_max; t->t_max = t_max;

    uint64_t *ec2[2], *oc2[2], *ec4[2], *oc4[2];
    for (int p = 0; p < 2; ++p) {
        ec2[p] = calloc(R1, sizeof(uint64_t));
        oc2[p] = calloc(T1, sizeof(uint64_t));
        ec4[p] = calloc(R1, sizeof(uint64_t));
        oc4[p] = calloc(T1, sizeof(uint64_t));
    }
    /* even symbols in harness order: [0, 1, -1, 2, -2, ...] up to k*k <= r_max */
    int kmax = 0; while ((long long)(kmax + 1) * (kmax + 1) <= r_max) ++kmax;
    int *ksym = malloc((2 * kmax + 2) * sizeof(int));
    int nks = 0;
    ksym[nks++] = 0;
    for (int k = 1; k <= kmax; ++k) { ksym[nks++] = k; ksym[nks++] = -k; }
    for (int a = 0; a < nks; ++a) {
        int s0 = ksym[a];
        for (int b = 0; b < nks; ++b) {
            int s1 = ksym[b];
            long long r = (long long)s0 * s0 + (long long)s1 * s1;
            if (r > r_max) continue;
            ec2[(s0 + s1) & 1][r]++;
        }
    }
    /* odd symbols in harness order: [0, -1, 1, -2, 2, ...] while c(c+1)/2 <= t_max */
    int cmax = 0; while ((long long)cmax * (cmax + 1) / 2 <= t_max) ++cmax;
    int *osym = malloc((2 * cmax + 2) * sizeof(int));
    int nos = 0;
    for (int c = 0; c < cmax; ++c) { osym[nos++] = c; osym[nos++] = -1 - c; }
    for (int a = 0; a < nos; ++a) {
        int s0 = osym[a];
        long long t0 = (long long)s0 * (s0 + 1) / 2;
        for (int b = 0; b < nos; ++b) {
            int s1 = osym[b];
            long long t1 = (long long)s1 * (s1 + 1) / 2;
            long long tt = t0 + t1;
            if (tt > t_max) continue;
            oc2[(s0 + s1) & 1][tt]++;
        }
    }
    /* c4 = c2 (*) c2 convolution, target parity pt = pa ^ pb */
    for (int pa = 0; pa < 2; ++pa) for (int pb = 0; pb < 2; ++pb) {
        int pt = pa ^ pb;
        for (int na = 0; na <= r_max; ++na) {
            if (!ec2[pa][na]) continue;
            for (int nb = 0; nb <= r_max - na; ++nb)
                if (ec2[pb][nb]) ec4[pt][na + nb] += ec2[pa][na] * ec2[pb][nb];
        }
        for (int na = 0; na <= t_max; ++na) {
            if (!oc2[pa][na]) continue;
            for (int nb = 0; nb <= t_max - na; ++nb)
                if (oc2[pb][nb]) oc4[pt][na + nb] += oc2[pa][na] * oc2[pb][nb];
        }
    }

    t->even_c4 = malloc(2 * R1 * sizeof(uint32_t));
    t->even_c2 = malloc(2 * R1 * sizeof(uint16_t));
    t->odd_c4 = malloc(2 * T1 * sizeof(uint32_t));
    t->odd_c2 = malloc(2 * T1 * sizeof(uint16_t));
    for (int p = 0; p < 2; ++p) {
        for (size_t i = 0; i < R1; ++i) {
            t->even_c4[p * R1 + i] = (uint32_t)ec4[p][i];
            t->even_c2[p * R1 + i] = (uint16_t)ec2[p][i];
        }
        for (size_t i = 0; i < T1; ++i) {
            t->odd_c4[p * T1 + i] = (uint32_t)oc4[p][i];
            t->odd_c2[p * T1 + i] = (uint16_t)oc2[p][i];
        }
    }
    if (counts_only) {
        t->c_root = calloc(M2 * 3, sizeof(uint64_t));
        t->s2 = calloc(M1 * 2, sizeof(uint64_t));
        t->w2 = calloc(M1 * 2, sizeof(uint64_t));
        t->s1 = calloc(M1, sizeof(uint64_t));
        t->w1 = calloc(M1, sizeof(uint64_t));
        t->rec_root_m = calloc(M1 * 3, sizeof(uint64_t)); t->rec_root_s = calloc(M1, sizeof(uint32_t));
        t->rec_b1_m = calloc(M1 * 2, sizeof(uint64_t)); t->rec_b1_s = calloc(M1, sizeof(uint32_t));
        t->rec_s1_m = calloc(M1 * 2, sizeof(uint64_t)); t->rec_s1_s = calloc(M1, sizeof(uint32_t));
        t->even_c8 = calloc(M1, sizeof(uint64_t));
        t->even_off = calloc(2 * R2, sizeof(uint16_t));
        t->odd_off = calloc(2 * T2, sizeof(uint16_t));
        for (int p = 0; p < 2; ++p) {
            t->even_pts[p] = calloc(2, 1); t->even_n[p] = 0;
            t->odd_pts[p] = calloc(2, 1); t->odd_n[p] = 0;
        }
        t->rec_ec4_m = calloc(2 * R1, sizeof(uint64_t)); t->rec_ec4_s = calloc(2 * R1, sizeof(uint32_t));
        t->rec_ec2_m = calloc(2 * R1, sizeof(uint32_t)); t->rec_ec2_s = calloc(2 * R1, sizeof(uint32_t));
        t->rec_oc4_m = calloc(2 * T1, sizeof(uint64_t)); t->rec_oc4_s = calloc(2 * T1, sizeof(uint32_t));
        t->rec_oc2_m = calloc(2 * T1, sizeof(uint32_t)); t->rec_oc2_s = calloc(2 * T1, sizeof(uint32_t));
        for (int p = 0; p < 2; ++p) { free(ec2[p]); free(oc2[p]); free(ec4[p]); free(oc4[p]); }
        free(ksym); free(osym);
        t->count = 0; t->nbytes = 1;
        return 0;
    }

    /* even_c8[m] = sum_p sum_a c4[p][a]*c4[p][2m-a]; odd_c8[m] over t8 = m-1 */
    uint64_t *even_c8 = calloc(M1, sizeof(uint64_t));
    uint64_t *odd_c8 = calloc(M1, sizeof(uint64_t));
    for (int m = 0; m <= m_max; ++m) {
        uint64_t tot = 0;
        int r8 = 2 * m;
        for (int p = 0; p < 2; ++p)
            for (int a = 0; a <= r8; ++a) tot += ec4[p][a] * ec4[p][r8 - a];
        even_c8[m] = tot;
    }
    for (int m = 1; m <= m_max; ++m) {
        uint64_t tot = 0;
        int t8 = m - 1;
        for (int p = 0; p < 2; ++p)
            for (int a = 0; a <= t8; ++a) tot += oc4[p][a] * oc4[p][t8 - a];
        odd_c8[m] = tot;
    }
    uint64_t *s1 = malloc(M1 * sizeof(uint64_t));
    for (size_t m = 0; m < M1; ++m) s1[m] = even_c8[m] + odd_c8[m];
    gen_u128 *s2 = calloc(M1, sizeof(gen_u128));
    for (size_t m1 = 0; m1 < M1; ++m1)
        for (size_t m2 = 0; m2 + m1 < M1; ++m2)
            s2[m1 + m2] += (gen_u128)s1[m1] * s1[m2];
    uint64_t *w1 = malloc(M1 * sizeof(uint64_t));
    gen_u128 *w2 = calloc(M1, sizeof(gen_u128));
    uint64_t acc1 = 0; gen_u128 acc2 = 0;
    for (size_t m = 0; m < M1; ++m) { acc1 += s1[m]; w1[m] = acc1; acc2 += s2[m]; w2[m] = acc2; }
    gen_u192 *c_root = calloc(M2, sizeof(gen_u192));
    {
        gen_u128 acc = 0;
        for (int m_l = 0; m_l <= m_max; ++m_l) {
            acc += s2[m_l] * w2[m_max - m_l]; /* fits u128 for m_max <= 100 */
            c_root[m_l + 1] = u192_from_u128(acc);
        }
    }
    gen_u192 last = c_root[m_max + 1];
    if (last.v[2]) { fprintf(stderr, "e8tables: count exceeds u128 at m_max=%d\n", m_max); return -1; }
    gen_u128 count128 = (gen_u128)last.v[0] | ((gen_u128)last.v[1] << 64);
    uint64_t count = (uint64_t)count128;
    int nbytes = (gen_bitlen128(count128) + 7) / 8;
    if (nbytes < 1) nbytes = 1;
    if (nbytes > 24) nbytes = 24;

    t->rec_root_m = malloc(M1 * 3 * sizeof(uint64_t)); t->rec_root_s = malloc(M1 * sizeof(uint32_t));
    t->rec_b1_m = malloc(M1 * 2 * sizeof(uint64_t)); t->rec_b1_s = malloc(M1 * sizeof(uint32_t));
    t->rec_s1_m = malloc(M1 * 2 * sizeof(uint64_t)); t->rec_s1_s = malloc(M1 * sizeof(uint32_t));
    for (size_t mr = 0; mr < M1; ++mr) {
        uint64_t max_s2 = 0, max_s1 = 0;
        for (size_t i = 0; i + mr < M1; ++i) {
            if (s2[i] > (gen_u128)UINT64_MAX) { fprintf(stderr, "e8tables: s2 exceeds u64\n"); return -1; }
            uint64_t v2 = (uint64_t)s2[i];
            if (v2 > max_s2) max_s2 = v2;
            if (s1[i] > max_s1) max_s1 = s1[i];
        }
        if (w2[mr] > (gen_u128)UINT64_MAX) { fprintf(stderr, "e8tables: w2 exceeds u64\n"); return -1; }
        uint64_t w2mr = (uint64_t)w2[mr], w1mr = w1[mr];
        gen_u192 m; uint32_t s;
        if (gen_recip(w2mr, (gen_u128)max_s2 * w2mr - 1, &m, &s) != 0) return -1;
        t->rec_root_m[3 * mr + 0] = m.v[0]; t->rec_root_m[3 * mr + 1] = m.v[1]; t->rec_root_m[3 * mr + 2] = m.v[2];
        t->rec_root_s[mr] = s;
        if (gen_recip(w1mr, (gen_u128)max_s1 * w1mr - 1, &m, &s) != 0) return -1;
        if (m.v[2]) return -1; /* rec_b1 is u128 */
        t->rec_b1_m[2 * mr + 0] = m.v[0]; t->rec_b1_m[2 * mr + 1] = m.v[1]; t->rec_b1_s[mr] = s;
        if (gen_recip(s1[mr], (gen_u128)max_s1 * s1[mr] - 1, &m, &s) != 0) return -1;
        if (m.v[2]) return -1; /* rec_s1 is u128 */
        t->rec_s1_m[2 * mr + 0] = m.v[0]; t->rec_s1_m[2 * mr + 1] = m.v[1]; t->rec_s1_s[mr] = s;
    }
    t->rec_ec4_m = malloc(2 * R1 * sizeof(uint64_t)); t->rec_ec4_s = malloc(2 * R1 * sizeof(uint32_t));
    t->rec_ec2_m = malloc(2 * R1 * sizeof(uint32_t)); t->rec_ec2_s = malloc(2 * R1 * sizeof(uint32_t));
    t->rec_oc4_m = malloc(2 * T1 * sizeof(uint64_t)); t->rec_oc4_s = malloc(2 * T1 * sizeof(uint32_t));
    t->rec_oc2_m = malloc(2 * T1 * sizeof(uint32_t)); t->rec_oc2_s = malloc(2 * T1 * sizeof(uint32_t));
    for (int p = 0; p < 2; ++p) {
        for (int nr = 0; nr <= r_max; ++nr) {
            size_t idx = (size_t)p * R1 + nr;
            uint64_t d4 = ec4[p][nr];
            uint64_t max_c4 = 0;
            for (int i = 0; i + nr <= r_max; ++i) if (ec4[p][i] > max_c4) max_c4 = ec4[p][i];
            gen_u192 m; uint32_t s;
            if (d4 > 0) {
                if (gen_recip(d4, (gen_u128)max_c4 * d4 - 1, &m, &s) != 0) return -1;
                if (m.v[1] || m.v[2]) return -1; /* rec_ec4 is u64 */
                t->rec_ec4_m[idx] = m.v[0]; t->rec_ec4_s[idx] = s;
            } else { t->rec_ec4_m[idx] = 0; t->rec_ec4_s[idx] = 0; }
            uint64_t d2 = ec2[p][nr];
            if (d2 > 0) {
                uint64_t ml = 0;
                for (int q = 0; q < 2; ++q)
                    for (int i = 0; i + nr <= r_max; ++i) if (ec2[q][i] > ml) ml = ec2[q][i];
                if (gen_recip(d2, (gen_u128)ml * d2 - 1, &m, &s) != 0) return -1;
                if (m.v[0] > 0xFFFFFFFFull || m.v[1] || m.v[2]) return -1; /* rec_ec2 is u32 */
                t->rec_ec2_m[idx] = (uint32_t)m.v[0]; t->rec_ec2_s[idx] = s;
            } else { t->rec_ec2_m[idx] = 0; t->rec_ec2_s[idx] = 0; }
        }
        for (int nr = 0; nr <= t_max; ++nr) {
            size_t idx = (size_t)p * T1 + nr;
            uint64_t d4 = oc4[p][nr];
            uint64_t max_c4 = 0;
            for (int i = 0; i + nr <= t_max; ++i) if (oc4[p][i] > max_c4) max_c4 = oc4[p][i];
            gen_u192 m; uint32_t s;
            if (d4 > 0) {
                if (gen_recip(d4, (gen_u128)max_c4 * d4 - 1, &m, &s) != 0) return -1;
                if (m.v[1] || m.v[2]) return -1; /* rec_oc4 is u64 */
                t->rec_oc4_m[idx] = m.v[0]; t->rec_oc4_s[idx] = s;
            } else { t->rec_oc4_m[idx] = 0; t->rec_oc4_s[idx] = 0; }
            uint64_t d2 = oc2[p][nr];
            if (d2 > 0) {
                uint64_t ml = 0;
                for (int q = 0; q < 2; ++q)
                    for (int i = 0; i + nr <= t_max; ++i) if (oc2[q][i] > ml) ml = oc2[q][i];
                if (gen_recip(d2, (gen_u128)ml * d2 - 1, &m, &s) != 0) return -1;
                if (m.v[0] > 0xFFFFFFFFull || m.v[1] || m.v[2]) return -1; /* rec_oc2 is u32 */
                t->rec_oc2_m[idx] = (uint32_t)m.v[0]; t->rec_oc2_s[idx] = s;
            } else { t->rec_oc2_m[idx] = 0; t->rec_oc2_s[idx] = 0; }
        }
    }

    /* pack into the struct */
    t->count = count; t->count128 = count128; t->nbytes = nbytes;
    t->c_root = malloc(M2 * 3 * sizeof(uint64_t));
    for (size_t i = 0; i < M2; ++i) {
        t->c_root[3 * i + 0] = c_root[i].v[0];
        t->c_root[3 * i + 1] = c_root[i].v[1];
        t->c_root[3 * i + 2] = c_root[i].v[2];
    }
    t->s2 = malloc(M1 * 2 * sizeof(uint64_t));
    t->w2 = malloc(M1 * 2 * sizeof(uint64_t));
    for (size_t i = 0; i < M1; ++i) {
        t->s2[2 * i + 0] = (uint64_t)s2[i]; t->s2[2 * i + 1] = (uint64_t)(s2[i] >> 64);
        t->w2[2 * i + 0] = (uint64_t)w2[i]; t->w2[2 * i + 1] = (uint64_t)(w2[i] >> 64);
    }
    t->s1 = s1; t->w1 = w1;
    t->even_c8 = even_c8;
    /* pts lists grouped by norm r (harness order inside each bucket) */
    t->even_off = malloc(2 * R2 * sizeof(uint16_t));
    t->odd_off = malloc(2 * T2 * sizeof(uint16_t));
    for (int p = 0; p < 2; ++p) {
        size_t tot_pairs = 0;
        for (int r = 0; r <= r_max; ++r) tot_pairs += ec2[p][r];
        int8_t *pb = malloc(2 * tot_pairs + 2);
        size_t plen = 0;
        uint16_t *off = t->even_off + (size_t)p * R2;
        off[0] = 0;
        for (int r = 0; r <= r_max; ++r) {
            for (int a = 0; a < nks; ++a) {
                int s0 = ksym[a];
                for (int b = 0; b < nks; ++b) {
                    int s1v = ksym[b];
                    if ((long long)s0 * s0 + (long long)s1v * s1v != r) continue;
                    if (((s0 + s1v) & 1) != p) continue;
                    pb[plen++] = (int8_t)(2 * s0);
                    pb[plen++] = (int8_t)(2 * s1v);
                }
            }
            off[r + 1] = (uint16_t)(plen / 2);
        }
        t->even_pts[p] = pb;
        t->even_n[p] = (int)(plen / 2);

        size_t otot_pairs = 0;
        for (int tt = 0; tt <= t_max; ++tt) otot_pairs += oc2[p][tt];
        int8_t *opb = malloc(2 * otot_pairs + 2);
        size_t oplen = 0;
        uint16_t *ooff = t->odd_off + (size_t)p * T2;
        ooff[0] = 0;
        for (int tt = 0; tt <= t_max; ++tt) {
            for (int a = 0; a < nos; ++a) {
                int s0 = osym[a];
                long long t0 = (long long)s0 * (s0 + 1) / 2;
                for (int b = 0; b < nos; ++b) {
                    int s1v = osym[b];
                    long long t1 = (long long)s1v * (s1v + 1) / 2;
                    if (t0 + t1 != tt) continue;
                    if (((s0 + s1v) & 1) != p) continue;
                    opb[oplen++] = (int8_t)(2 * s0 + 1);
                    opb[oplen++] = (int8_t)(2 * s1v + 1);
                }
            }
            ooff[tt + 1] = (uint16_t)(oplen / 2);
        }
        t->odd_pts[p] = opb;
        t->odd_n[p] = (int)(oplen / 2);
    }
    free(odd_c8); free(s2); free(w2); free(c_root); free(ksym); free(osym);
    for (int p = 0; p < 2; ++p) { free(ec2[p]); free(oc2[p]); free(ec4[p]); free(oc4[p]); }
    return 0;
}

/* FNV-1a 64-bit digest helper for cross-build (AVX2 vs scalar) comparison */
static uint64_t fnv1a_update(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

#endif /* E8_TABLES_GEN_H */
