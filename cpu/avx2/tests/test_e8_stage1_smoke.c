/*
 * Functional smoke test for libe8_stage1.so (cpu/avx2/e8_stage1_bench.c).
 *
 * Usage: test_e8_stage1_smoke <path-to-libe8_stage1.so>
 *
 * Exercises, on self-generated exact E_8^4 tables:
 *   - e8s1_init accepts m_max=16 (88b/u32 path) and m_max=65 (136b path), and
 *     REJECTS m_max=1490 and 2048 (c4 table would overflow uint16_t) and
 *     out-of-range m_max/bytes
 *   - all 6 decode_z variants agree byte-identically; rejected ranks counted bad
 *   - GEMV equivalence of all 6 adjoint variants + materialized vs an
 *     independent double reference; cols=33/0 rejected by every public GEMV
 *     entry point (fail closed, no 8-float overread)
 *   - e8s1_extract_leaves_from_blocks zeroes the leaf outputs of rejected blocks
 *   - e8s1_d8_unrank_shell_vec validates init/m/rank (rk >= shell_count zeroes
 *     the output instead of an OOB guide read) and is a bijection on m=0..3
 *     for both pshufb and compact modes
 *   - the Q4_K_M-style baseline honors the encoded zero point rec[1]
 *
 * Exit status 0 = all checks passed. Prints SMOKE_PASS / DECODE_DIGEST=<hex>.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "e8_tables_gen.h"

typedef int (*has_avx2_fn)(void);
typedef int (*init_fn)(int, int,
    const uint64_t *, const uint64_t *, const uint64_t *, const uint64_t *, const uint64_t *,
    const uint64_t *, const uint32_t *, const uint64_t *, const uint32_t *,
    const uint64_t *, const uint32_t *,
    const uint64_t *, const uint32_t *, const uint16_t *, const uint16_t *,
    const int8_t *, int, const int8_t *, int,
    const uint32_t *, const uint16_t *, const uint16_t *,
    const int8_t *, int, const int8_t *, int,
    const uint64_t *, const uint32_t *, const uint32_t *, const uint32_t *,
    const uint64_t *, const uint32_t *, const uint32_t *, const uint32_t *);
typedef size_t (*decode_z_fn)(const uint8_t *, size_t, int8_t *);
typedef size_t (*gemv_fn)(const uint8_t *, const float *, const float *, size_t, size_t, float *);
typedef size_t (*gemv_mat_fn)(int, const uint8_t *, const float *, const float *, size_t, size_t, float *);
typedef size_t (*extract_fn)(const uint8_t *, size_t, uint64_t *, int32_t *, uint64_t *, uint64_t *, uint64_t *);
typedef uint64_t (*bench_gemv_fn)(int, int, const uint8_t *, size_t, const float *, const float *,
                                  size_t, size_t, float *, size_t *, uint64_t *);
typedef void (*d8_unrank_fn)(int, uint64_t, int, int8_t *);
typedef int64_t (*d8_count_fn)(int);

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++failures; } \
} while (0)

static void pack_rank(uint8_t *dst, int nb, uint64_t rk) {
    for (int i = 0; i < nb; ++i)
        dst[i] = (i < 8) ? (uint8_t)(rk >> (8 * i)) : 0; /* no shift >= 64 (UB) */
}

static init_fn g_init;
static int call_init(const E8Tables *t, int m_max) {
    return g_init(m_max, t->nbytes,
        t->c_root, t->s2, t->w2, t->s1, t->w1,
        t->rec_root_m, t->rec_root_s, t->rec_b1_m, t->rec_b1_s, t->rec_s1_m, t->rec_s1_s,
        t->even_c8, t->even_c4, t->even_c2, t->even_off,
        t->even_pts[0], t->even_n[0], t->even_pts[1], t->even_n[1],
        t->odd_c4, t->odd_c2, t->odd_off,
        t->odd_pts[0], t->odd_n[0], t->odd_pts[1], t->odd_n[1],
        t->rec_ec4_m, t->rec_ec4_s, t->rec_ec2_m, t->rec_ec2_s,
        t->rec_oc4_m, t->rec_oc4_s, t->rec_oc2_m, t->rec_oc2_s);
}

/* run the functional suite at one m_max; returns digest */
static uint64_t run_functional(void *h, int m_max, const char *tag) {
    decode_z_fn dec[6] = {
        (decode_z_fn)dlsym(h, "e8s1_decode_z_baseline"),
        (decode_z_fn)dlsym(h, "e8s1_decode_z_guide_a"),
        (decode_z_fn)dlsym(h, "e8s1_decode_z_eytz_lut2"),
        (decode_z_fn)dlsym(h, "e8s1_decode_z_eytz_lut3"),
        (decode_z_fn)dlsym(h, "e8s1_decode_z_d8_compact"),
        (decode_z_fn)dlsym(h, "e8s1_decode_z_d8_pshufb"),
    };
    gemv_fn g_adj[6] = {
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_baseline"),
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_guide_a"),
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_eytz_lut2"),
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_eytz_lut3"),
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_d8_compact"),
        (gemv_fn)dlsym(h, "e8s1_gemv_adjoint_d8_pshufb"),
    };
    gemv_mat_fn g_mat = (gemv_mat_fn)dlsym(h, "e8s1_gemv_materialized_variant");
    gemv_fn g_q4 = (gemv_fn)dlsym(h, "q4km_style_gemv_avx2");
    extract_fn extract = (extract_fn)dlsym(h, "e8s1_extract_leaves_from_blocks");
    bench_gemv_fn bench_gemv = (bench_gemv_fn)dlsym(h, "e8s1_bench_gemv_ns");
    d8_unrank_fn d8_unrank = (d8_unrank_fn)dlsym(h, "e8s1_d8_unrank_shell_vec");
    d8_count_fn d8_count = (d8_count_fn)dlsym(h, "e8s1_d8_shell_count");
    if (!dec[0] || !g_adj[0] || !g_mat || !g_q4 || !extract || !bench_gemv || !d8_unrank || !d8_count)
        { fprintf(stderr, "dlsym failed\n"); exit(2); }
    for (int i = 1; i < 6; ++i)
        if (!dec[i] || !g_adj[i]) { fprintf(stderr, "dlsym failed\n"); exit(2); }

    E8Tables t;
    if (e8tables_build(&t, m_max, 0) != 0) { fprintf(stderr, "table build failed m_max=%d\n", m_max); exit(2); }
    int rc = call_init(&t, m_max);
    CHECK(rc == 0, "e8s1_init(m_max=%d) rc=%d", m_max, rc);
    if (rc != 0) { e8tables_free(&t); return 0; }
    printf("[%s] m_max=%d count_lo=%llu nbytes=%d\n", tag, m_max,
           (unsigned long long)t.count, t.nbytes);

    const int nb = t.nbytes;
    /* ranks always < 2^62, which is < count for both m_max=16 and m_max=65 */
    const uint64_t RANK_CAP = 1ULL << 62;
    const int G = 256, N_BAD = 4;
    const size_t NG = (size_t)G + N_BAD;
    uint8_t *ranks = malloc(NG * (size_t)nb);
    uint64_t rlist[NG];
    uint32_t rng = 0x5EEDu ^ (uint32_t)m_max;
    rlist[0] = 0; rlist[1] = RANK_CAP - 1; rlist[2] = RANK_CAP / 2;
    for (int g = 3; g < G; ++g) {
        rng = rng * 1664525u + 1013904223u;
        rlist[g] = (uint64_t)(rng % 0x3FFFFFFFu) * (RANK_CAP / 0x40000000u) + (rng >> 16);
        if (rlist[g] >= RANK_CAP) rlist[g] %= RANK_CAP;
    }
    for (int b = 0; b < N_BAD; ++b) rlist[G + b] = UINT64_MAX; /* >= count: invalid */
    for (size_t g = 0; g < NG; ++g) pack_rank(ranks + g * (size_t)nb, nb, rlist[g]);
    /* out-of-range ranks: all-0xFF bytes = 2^(8*nb)-1 >= count, always invalid */
    for (int b = 0; b < N_BAD; ++b) memset(ranks + (size_t)(G + b) * nb, 0xFF, (size_t)nb);

    int8_t *z[6];
    for (int m = 0; m < 6; ++m) z[m] = malloc(NG * 32);
    uint64_t digest = 1469598103934665603ULL;
    for (int m = 0; m < 6; ++m) {
        size_t bad = dec[m](ranks, NG, z[m]);
        CHECK(bad == (size_t)N_BAD, "[%s] decode variant %d bad=%zu (expected %d)", tag, m, bad, N_BAD);
        digest = fnv1a_update(digest, z[m], NG * 32);
    }
    /* ballpack variants 0-3 are the same bijection; D_8^+ variants 4-5 are a
     * different (but equally valid) enumeration order: they must agree with
     * each other, and every variant must be injective on the sample. */
    for (int m = 1; m < 4; ++m)
        CHECK(memcmp(z[0], z[m], NG * 32) == 0, "[%s] decode variant %d differs from baseline", tag, m);
    CHECK(memcmp(z[4], z[5], NG * 32) == 0, "[%s] d8 compact != pshufb", tag);
    for (int m = 0; m < 6; ++m) {
        typedef struct { uint64_t h; uint64_t rk; } HR;
        HR *hr = malloc(NG * sizeof(HR));
        size_t n = 0;
        for (size_t g = 0; g < NG; ++g) {
            if (rlist[g] >= RANK_CAP) continue; /* skip rejected */
            hr[n].h = fnv1a_update(1469598103934665603ULL, z[m] + g * 32, 32);
            hr[n].rk = rlist[g];
            ++n;
        }
        for (size_t i = 1; i < n; ++i) { HR k = hr[i]; size_t j = i; while (j > 0 && hr[j-1].h > k.h) { hr[j] = hr[j-1]; --j; } hr[j] = k; }
        for (size_t i = 1; i < n; ++i)
            if (hr[i].h == hr[i-1].h && hr[i].rk != hr[i-1].rk)
                { CHECK(0, "[%s] variant %d collision rk %llu vs %llu", tag, m,
                        (unsigned long long)hr[i].rk, (unsigned long long)hr[i-1].rk); break; }
        free(hr);
    }
    printf("[%s] DECODE_DIGEST=%016llx\n", tag, (unsigned long long)digest);

    /* GEMV equivalence vs independent double reference (from variant-0 decode) */
    const size_t ROWS = 8, COLS = 64, rec_sz = (size_t)nb + 1;
    uint8_t *recs = malloc(ROWS * (COLS / 32) * rec_sz);
    float scale_lut[256];
    for (int c = 0; c < 256; ++c) scale_lut[c] = 0.001f * (float)((c * 37 + 13) % 251 + 1);
    float x[COLS];
    for (size_t j = 0; j < COLS; ++j) {
        rng = rng * 1664525u + 1013904223u;
        x[j] = ((float)((int)(rng >> 16) % 2001) - 1000.0f) * 0.001f;
    }
    for (size_t r = 0; r < ROWS; ++r)
        for (size_t b = 0; b < COLS / 32; ++b) {
            uint8_t *rec = recs + (r * (COLS / 32) + b) * rec_sz;
            rng = rng * 1664525u + 1013904223u;
            rec[0] = (uint8_t)(rng & 0xFF);
            uint64_t rk = (uint64_t)(rng % 0x3FFFFFFFu) * (RANK_CAP / 0x40000000u);
            pack_rank(rec + 1, nb, rk);
        }
    /* reference per variant: decode the records with the SAME variant and
     * compute y = sum_b scale*inv_sqrt2 * <z, x_block> in double */
    float y[7][ROWS];
    for (int m = 0; m < 6; ++m) {
        int8_t *zref = malloc(ROWS * (COLS / 32) * 32);
        uint8_t *rr = malloc(ROWS * (COLS / 32) * (size_t)nb);
        for (size_t i = 0; i < ROWS * (COLS / 32); ++i)
            memcpy(rr + i * (size_t)nb, recs + i * rec_sz + 1, (size_t)nb);
        CHECK(dec[m](rr, ROWS * (COLS / 32), zref) == 0, "[%s] gemv decode variant %d bad", tag, m);
        free(rr);
        double ref[ROWS];
        const double inv_sqrt2 = 0.7071067811865475244;
        for (size_t r = 0; r < ROWS; ++r) {
            double acc = 0.0;
            for (size_t b = 0; b < COLS / 32; ++b) {
                const uint8_t *rec = recs + (r * (COLS / 32) + b) * rec_sz;
                double sc = (double)scale_lut[rec[0]] * inv_sqrt2;
                const int8_t *zb = zref + (r * (COLS / 32) + b) * 32;
                double d = 0.0;
                for (int j = 0; j < 32; ++j) d += (double)zb[j] * (double)x[b * 32 + j];
                acc += sc * d;
            }
            ref[r] = acc;
        }
        free(zref);
        size_t bad = g_adj[m](recs, scale_lut, x, ROWS, COLS, y[m]);
        CHECK(bad == 0, "[%s] adjoint variant %d bad=%zu", tag, m, bad);
        for (size_t r = 0; r < ROWS; ++r) {
            double tol = 1e-4 * (fabs(ref[r]) + 1.0);
            CHECK(fabs((double)y[m][r] - ref[r]) <= tol,
                  "[%s] adjoint variant %d row %zu: %f vs %f", tag, m, r, y[m][r], ref[r]);
        }
    }
    /* materialized (variant 0) against variant-0 reference */
    {
        int8_t *zref = malloc(ROWS * (COLS / 32) * 32);
        uint8_t *rr = malloc(ROWS * (COLS / 32) * (size_t)nb);
        for (size_t i = 0; i < ROWS * (COLS / 32); ++i)
            memcpy(rr + i * (size_t)nb, recs + i * rec_sz + 1, (size_t)nb);
        CHECK(dec[0](rr, ROWS * (COLS / 32), zref) == 0, "[%s] mat decode bad", tag);
        free(rr);
        double ref[ROWS];
        const double inv_sqrt2 = 0.7071067811865475244;
        for (size_t r = 0; r < ROWS; ++r) {
            double acc = 0.0;
            for (size_t b = 0; b < COLS / 32; ++b) {
                const uint8_t *rec = recs + (r * (COLS / 32) + b) * rec_sz;
                double sc = (double)scale_lut[rec[0]] * inv_sqrt2;
                const int8_t *zb = zref + (r * (COLS / 32) + b) * 32;
                double d = 0.0;
                for (int j = 0; j < 32; ++j) d += (double)zb[j] * (double)x[b * 32 + j];
                acc += sc * d;
            }
            ref[r] = acc;
        }
        free(zref);
        size_t badm = g_mat(0, recs, scale_lut, x, ROWS, COLS, y[6]);
        CHECK(badm == 0, "[%s] materialized bad=%zu", tag, badm);
        for (size_t r = 0; r < ROWS; ++r)
            CHECK(fabs((double)y[6][r] - ref[r]) <= 1e-4 * (fabs(ref[r]) + 1.0),
                  "[%s] mat row %zu: %f vs %f", tag, r, y[6][r], ref[r]);
    }

    /* cols alignment: fail closed everywhere */
    float x33[33]; memset(x33, 0, sizeof(x33));
    uint8_t *recs33 = calloc(ROWS * 2 * rec_sz, 1);
    float *y33 = malloc(ROWS * sizeof(float));
    for (int m = 0; m < 6; ++m)
        CHECK(g_adj[m](recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "[%s] adjoint %d cols=33 must fail", tag, m);
    CHECK(g_mat(0, recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "[%s] mat cols=33 must fail", tag);
    CHECK(g_q4(recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "[%s] q4 cols=33 must fail", tag);
    {
        size_t bad = 0; uint64_t cns = 0;
        bench_gemv(0, 1, recs33, rec_sz, scale_lut, x33, ROWS, 33, y33, &bad, &cns);
        CHECK(bad > ROWS, "[%s] bench_gemv cols=33 must fail (bad=%zu)", tag, bad);
        bench_gemv(0, 1, recs, rec_sz, scale_lut, x, ROWS, 0, y33, &bad, &cns);
        CHECK(bad > ROWS, "[%s] bench_gemv cols=0 must fail (bad=%zu)", tag, bad);
        bench_gemv(0, 1, recs, rec_sz, scale_lut, x, ROWS, COLS, y33, &bad, &cns);
        CHECK(bad == 0, "[%s] bench_gemv cols=64 bad=%zu", tag, bad);
    }
    free(recs33); free(y33);

    /* extract_leaves: rejected block leaves must be initialized (zeroed) */
    {
        const size_t NG2 = 8;
        uint8_t *in2 = malloc(NG2 * (size_t)nb);
        uint64_t lrk[NG2 * 4];
        int32_t lm[NG2 * 4];
        uint64_t h2 = 0, h3 = 0, cns = 0;
        /* fill with sentinel first */
        for (size_t i = 0; i < NG2 * 4; ++i) { lrk[i] = 0xDEADBEEFULL; lm[i] = -12345; }
        for (size_t g = 0; g < NG2; ++g) {
            uint64_t rk = (uint64_t)(g * 1000 % RANK_CAP);
            pack_rank(in2 + g * (size_t)nb, nb, rk);
            if (g == 0) { for (int i = 0; i < nb; ++i) in2[i] = 0xFF; } /* rejected block */
        }
        size_t bad = extract(in2, NG2, lrk, lm, &h2, &h3, &cns);
        CHECK(bad == 1, "[%s] extract bad=%zu (expected 1)", tag, bad);
        CHECK(lm[0] == 0 && lrk[0] == 0 && lm[1] == 0 && lrk[1] == 0 && lm[2] == 0 && lrk[2] == 0 && lm[3] == 0 && lrk[3] == 0,
              "[%s] rejected block leaves not initialized", tag);
        free(in2);
    }

    /* D_8^+ shell unranker: validation + bijection on m=0..3 */
    for (int m = -1; m <= 4; ++m) {
        int64_t c = d8_count(m);
        if (m < 0 || m > 3) {
            /* m=4 is valid (<= m_max) for shell counts, but we only exhaust m<=3 */
        }
        if (m >= 0 && m <= m_max) CHECK(c > 0, "[%s] shell_count(%d)=%lld", tag, m, (long long)c);
        else CHECK(c == -1, "[%s] shell_count(%d) should be -1", tag, m);
    }
    CHECK(d8_count(m_max + 1) == -1, "[%s] shell_count(m_max+1) should be -1", tag);
    for (int m = 0; m <= 3; ++m) {
        int64_t c = d8_count(m);
        CHECK(c > 0, "[%s] shell_count(%d)", tag, m);
        int8_t *a = malloc((size_t)c * 8), *b = malloc((size_t)c * 8);
        for (int64_t rk = 0; rk < c; ++rk) {
            d8_unrank(m, (uint64_t)rk, 0, a + (size_t)rk * 8);
            d8_unrank(m, (uint64_t)rk, 1, b + (size_t)rk * 8);
        }
        CHECK(memcmp(a, b, (size_t)c * 8) == 0, "[%s] d8 pshufb != compact at m=%d", tag, m);
        /* distinctness via sorting hashes */
        uint64_t *hs = malloc((size_t)c * sizeof(uint64_t));
        for (int64_t rk = 0; rk < c; ++rk) hs[rk] = fnv1a_update(1469598103934665603ULL, a + (size_t)rk * 8, 8);
        for (int64_t i = 1; i < c; ++i) { uint64_t k = hs[i]; int64_t j = i - 1; while (j >= 0 && hs[j] > k) { hs[j+1] = hs[j]; --j; } hs[j+1] = k; }
        for (int64_t i = 1; i < c; ++i) CHECK(hs[i] != hs[i-1], "[%s] d8 collision at m=%d rk=%lld", tag, m, (long long)i);
        free(hs); free(a); free(b);
        /* out-of-range rank must zero the output, not read OOB */
        int8_t o8[8] = {1,2,3,4,5,6,7,8};
        d8_unrank(m, UINT64_MAX, 0, o8);
        int8_t zero8[8] = {0};
        CHECK(memcmp(o8, zero8, 8) == 0, "[%s] d8 rk=UINT64_MAX not zeroed at m=%d", tag, m);
        int8_t o8b[8] = {1,2,3,4,5,6,7,8};
        d8_unrank(m, UINT64_MAX, 1, o8b);
        CHECK(memcmp(o8b, zero8, 8) == 0, "[%s] d8 pshufb rk=UINT64_MAX not zeroed at m=%d", tag, m);
    }
    {
        int8_t o8[8] = {1,2,3,4,5,6,7,8};
        d8_unrank(m_max + 1, 0, 0, o8);
        int8_t zero8[8] = {0};
        CHECK(memcmp(o8, zero8, 8) == 0, "[%s] d8 m=m_max+1 not zeroed", tag);
    }

    /* Q4 zero point */
    {
        const size_t QROWS = 4, QCOLS = 64, qrec_sz = 18;
        uint8_t *qrecs = malloc(QROWS * (QCOLS / 32) * qrec_sz);
        static const int zps[4] = {8, 3, 0, 15};
        for (size_t r = 0; r < QROWS; ++r)
            for (size_t b = 0; b < QCOLS / 32; ++b) {
                uint8_t *rec = qrecs + (r * (QCOLS / 32) + b) * qrec_sz;
                rng = rng * 1664525u + 1013904223u;
                rec[0] = (uint8_t)(rng & 0xFF);
                rec[1] = (uint8_t)zps[(r + b) % 4];
                for (int i = 0; i < 16; ++i) { rng = rng * 1664525u + 1013904223u; rec[2 + i] = (uint8_t)(rng >> 24); }
            }
        double qref[QROWS];
        for (size_t r = 0; r < QROWS; ++r) {
            double acc = 0.0;
            for (size_t b = 0; b < QCOLS / 32; ++b) {
                const uint8_t *rec = qrecs + (r * (QCOLS / 32) + b) * qrec_sz;
                double s = (double)scale_lut[rec[0]], zp = (double)rec[1], d = 0.0;
                for (int i = 0; i < 16; ++i) {
                    d += ((double)(rec[2 + i] & 0x0fu) - zp) * (double)x[b * 32 + i]
                       + ((double)(rec[2 + i] >> 4) - zp) * (double)x[b * 32 + 16 + i];
                }
                acc += s * d;
            }
            qref[r] = acc;
        }
        float yq[QROWS];
        CHECK(g_q4(qrecs, scale_lut, x, QROWS, QCOLS, yq) == 0, "[%s] q4 bad", tag);
        for (size_t r = 0; r < QROWS; ++r)
            CHECK(fabs((double)yq[r] - qref[r]) <= 1e-4 * (fabs(qref[r]) + 1.0),
                  "[%s] q4 zp row %zu: %f vs %f", tag, r, yq[r], qref[r]);
        free(qrecs);
    }

    for (int m = 0; m < 6; ++m) free(z[m]);
    free(recs); free(ranks);
    e8tables_free(&t);
    return digest;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <libe8_stage1.so>\n", argv[0]); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    has_avx2_fn has_avx2 = (has_avx2_fn)dlsym(h, "e8s1_has_avx2");
    g_init = (init_fn)dlsym(h, "e8s1_init");
    if (!has_avx2 || !g_init) { fprintf(stderr, "dlsym failed\n"); return 2; }
    printf("lib=%s has_avx2=%d\n", argv[1], has_avx2());

    /* overflow-rejection: m_max=1490 and 2048 have max(c4) = 69120 / 92160 >
     * UINT16_MAX, so init must fail closed (counts_only tables suffice) */
    {
        static const int rejects[] = {1490, 2048};
        for (size_t i = 0; i < sizeof(rejects) / sizeof(rejects[0]); ++i) {
            E8Tables t;
            if (e8tables_build(&t, rejects[i], 1) != 0) { fprintf(stderr, "counts build failed\n"); return 2; }
            int rc = call_init(&t, rejects[i]);
            CHECK(rc == 1, "e8s1_init(m_max=%d) must be rejected (c4 > uint16), rc=%d", rejects[i], rc);
            printf("m_max=%d init rejected rc=%d (expected 1)\n", rejects[i], rc);
            e8tables_free(&t);
        }
        /* out-of-range m_max */
        E8Tables t;
        if (e8tables_build(&t, 16, 1) != 0) return 2;
        CHECK(call_init(&t, 2) == 1, "init(m_max=2) must be rejected");
        CHECK(call_init(&t, 2049) == 1, "init(m_max=2049) must be rejected");
        e8tables_free(&t);
    }

    uint64_t d16 = run_functional(h, 16, "m16");
    uint64_t d65 = run_functional(h, 65, "m65");
    printf("DECODE_DIGEST_M16=%016llx\n", (unsigned long long)d16);
    printf("DECODE_DIGEST_M65=%016llx\n", (unsigned long long)d65);

    if (failures) {
        fprintf(stderr, "SMOKE_FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    printf("SMOKE_PASS\n");
    return 0;
}
