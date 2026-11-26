/*
 * Functional smoke test for libe8had_simd.so (cpu/avx2/e8had_simd_bench.c).
 *
 * Usage: test_e8had_smoke <path-to-libe8had_simd.so>
 *
 * Exercises, on a small self-generated exact E_8^4 table set (m_max=16):
 *   - e8had_init (accepts valid, rejects out-of-range m_max/bytes)
 *   - unrank/decode equivalence across all 4 decode modes (scalar / WHT-scalar /
 *     WHT-AVX2 / SIMD8-AVX2) with a shared FNV-1a digest printed for cross-build
 *     (AVX2 vs scalar .so) comparison by `make check-avx2-smoke` / CI
 *   - rejected (out-of-range) ranks are counted as bad, not decoded
 *   - GEMV equivalence of materialized / adjoint / batch4 against an independent
 *     double-precision reference computed from the scalar decode + WHT_8
 *   - cols=33 / cols=0 are rejected by every public GEMV API (fail closed, no
 *     8-float overread of the activation buffer)
 *   - the Q4_K_M-style baseline honors the encoded zero point rec[1]
 *     (tested with zero_pt 8, 3, 0 -- the old code hard-coded 8)
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
typedef uint64_t (*decode_bench_fn)(int, int, const uint8_t *, size_t, int8_t *, size_t *, uint64_t *);
typedef size_t (*gemv_fn)(const uint8_t *, const float *, const float *, size_t, size_t, float *);

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++failures; } \
} while (0)

/* scalar WHT_8 with /4, matching wht8_u_to_z_scalar in the library */
static void wht8_scalar(const int8_t *u, int8_t *z) {
    int a0 = u[0] + u[1], a1 = u[0] - u[1];
    int a2 = u[2] + u[3], a3 = u[2] - u[3];
    int a4 = u[4] + u[5], a5 = u[4] - u[5];
    int a6 = u[6] + u[7], a7 = u[6] - u[7];
    int b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2, b3 = a1 - a3;
    int b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6, b7 = a5 - a7;
    z[0] = (int8_t)((b0 + b4) >> 2); z[1] = (int8_t)((b1 + b5) >> 2);
    z[2] = (int8_t)((b2 + b6) >> 2); z[3] = (int8_t)((b3 + b7) >> 2);
    z[4] = (int8_t)((b0 - b4) >> 2); z[5] = (int8_t)((b1 - b5) >> 2);
    z[6] = (int8_t)((b2 - b6) >> 2); z[7] = (int8_t)((b3 - b7) >> 2);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <libe8had_simd.so>\n", argv[0]); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    has_avx2_fn has_avx2 = (has_avx2_fn)dlsym(h, "e8had_has_avx2");
    init_fn init = (init_fn)dlsym(h, "e8had_init");
    decode_bench_fn bench_decode = (decode_bench_fn)dlsym(h, "bench_decode_ns");
    gemv_fn g_materialized = (gemv_fn)dlsym(h, "e8had_gemv_materialized_avx2");
    gemv_fn g_adjoint = (gemv_fn)dlsym(h, "e8had_gemv_adjoint_avx2");
    gemv_fn g_batch4 = (gemv_fn)dlsym(h, "e8had_gemv_adjoint_batch4_avx2");
    gemv_fn g_q4 = (gemv_fn)dlsym(h, "q4km_style_gemv_avx2");
    if (!has_avx2 || !init || !bench_decode || !g_materialized || !g_adjoint || !g_batch4 || !g_q4) {
        fprintf(stderr, "dlsym failed\n");
        return 2;
    }
    printf("lib=%s has_avx2=%d\n", argv[1], has_avx2());

    const int M_MAX = 16;
    E8Tables t;
    if (e8tables_build(&t, M_MAX, 0) != 0) { fprintf(stderr, "table build failed\n"); return 2; }
    printf("m_max=%d count=%llu nbytes=%d\n", M_MAX, (unsigned long long)t.count, t.nbytes);

    /* --- init validation --- */
    int rc = init(M_MAX, t.nbytes,
        t.c_root, t.s2, t.w2, t.s1, t.w1,
        t.rec_root_m, t.rec_root_s, t.rec_b1_m, t.rec_b1_s, t.rec_s1_m, t.rec_s1_s,
        t.even_c8, t.even_c4, t.even_c2, t.even_off,
        t.even_pts[0], t.even_n[0], t.even_pts[1], t.even_n[1],
        t.odd_c4, t.odd_c2, t.odd_off,
        t.odd_pts[0], t.odd_n[0], t.odd_pts[1], t.odd_n[1],
        t.rec_ec4_m, t.rec_ec4_s, t.rec_ec2_m, t.rec_ec2_s,
        t.rec_oc4_m, t.rec_oc4_s, t.rec_oc2_m, t.rec_oc2_s);
    CHECK(rc == 0, "e8had_init rc=%d", rc);
    CHECK(init(0, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               NULL, NULL, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, 0, NULL, 0,
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == 1, "init(m_max=0) must be rejected");
    CHECK(init(2049, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
               NULL, NULL, NULL, NULL, NULL, 0, NULL, 0, NULL, NULL, NULL, NULL, 0, NULL, 0,
               NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == 1, "init(m_max=2049) must be rejected");
    /* re-init with the real tables for the functional tests */
    rc = init(M_MAX, t.nbytes,
        t.c_root, t.s2, t.w2, t.s1, t.w1,
        t.rec_root_m, t.rec_root_s, t.rec_b1_m, t.rec_b1_s, t.rec_s1_m, t.rec_s1_s,
        t.even_c8, t.even_c4, t.even_c2, t.even_off,
        t.even_pts[0], t.even_n[0], t.even_pts[1], t.even_n[1],
        t.odd_c4, t.odd_c2, t.odd_off,
        t.odd_pts[0], t.odd_n[0], t.odd_pts[1], t.odd_n[1],
        t.rec_ec4_m, t.rec_ec4_s, t.rec_ec2_m, t.rec_ec2_s,
        t.rec_oc4_m, t.rec_oc4_s, t.rec_oc2_m, t.rec_oc2_s);
    CHECK(rc == 0, "re-init rc=%d", rc);

    const int nb = t.nbytes;
    const uint64_t count = t.count;

    /* --- decode: ranks incl. boundary + 4 invalid ranks --- */
    const int G = 256, N_BAD = 4;
    uint8_t *ranks = malloc((size_t)(G + N_BAD) * nb);
    uint32_t rng = 0xC0FFEEu;
    uint64_t rlist[G + N_BAD];
    rlist[0] = 0; rlist[1] = count - 1; rlist[2] = count / 2;
    for (int g = 3; g < G; ++g) {
        rng = rng * 1664525u + 1013904223u;
        rlist[g] = (uint64_t)(rng % 0x100000007u) % count;
    }
    for (int b = 0; b < N_BAD; ++b) rlist[G + b] = count + (uint64_t)b + 1; /* out of range */
    for (int g = 0; g < G + N_BAD; ++g)
        for (int i = 0; i < nb; ++i) ranks[(size_t)g * nb + i] = (uint8_t)(rlist[g] >> (8 * i));

    const size_t NG = (size_t)G + N_BAD;
    int8_t *out[4];
    for (int m = 0; m < 4; ++m) out[m] = malloc(NG * 32);
    size_t bad[4]; uint64_t cns[4];
    uint64_t digest = 1469598103934665603ULL;
    for (int m = 0; m < 4; ++m) {
        bench_decode(m, 1, ranks, NG, out[m], &bad[m], &cns[m]);
        CHECK(bad[m] == (size_t)N_BAD, "decode mode %d bad=%zu (expected %d)", m, bad[m], N_BAD);
        digest = fnv1a_update(digest, out[m], NG * 32);
    }
    /* modes 1..3 (WHT scalar / WHT AVX2 / SIMD8) must be byte-identical */
    for (int m = 2; m < 4; ++m)
        CHECK(memcmp(out[1], out[m], NG * 32) == 0, "decode mode %d differs from mode 1", m);
    /* mode 0 is the raw u32; modes 1..3 are z = WHT_8(u)/4: check that relation */
    for (size_t g = 0; g < NG; ++g) {
        if (rlist[g] >= count) {
            int8_t zero[32] = {0};
            CHECK(memcmp(out[0] + g * 32, zero, 32) == 0, "rejected group %zu not zeroed (mode 0)", g);
            continue;
        }
        int8_t z[32];
        for (int b = 0; b < 4; ++b) wht8_scalar(out[0] + g * 32 + 8 * b, z + 8 * b);
        CHECK(memcmp(z, out[1] + g * 32, 32) == 0, "WHT relation broken at group %zu", g);
    }
    printf("DECODE_DIGEST=%016llx\n", (unsigned long long)digest);

    /* --- GEMV equivalence vs independent double reference --- */
    const size_t ROWS = 8, COLS = 64; /* 2 blocks of 32 per row */
    const size_t rec_sz = (size_t)nb + 1;
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
            uint64_t rk = (uint64_t)(rng % 0x100000007u) % count;
            for (int i = 0; i < nb; ++i) rec[1 + i] = (uint8_t)(rk >> (8 * i));
        }
    /* double reference from scalar decode (mode 0) + WHT_8 */
    int8_t *u_all = malloc(ROWS * (COLS / 32) * 32);
    {
        uint8_t *rr = malloc(ROWS * (COLS / 32) * nb);
        for (size_t i = 0; i < ROWS * (COLS / 32); ++i)
            memcpy(rr + i * nb, recs + i * rec_sz + 1, (size_t)nb);
        size_t b2 = 0; uint64_t c2 = 0;
        bench_decode(0, 1, rr, ROWS * (COLS / 32), u_all, &b2, &c2);
        CHECK(b2 == 0, "gemv decode bad=%zu", b2);
        free(rr);
    }
    double ref[ROWS];
    const double inv_sqrt2 = 0.7071067811865475244;
    for (size_t r = 0; r < ROWS; ++r) {
        double acc = 0.0;
        for (size_t b = 0; b < COLS / 32; ++b) {
            const uint8_t *rec = recs + (r * (COLS / 32) + b) * rec_sz;
            double s = (double)scale_lut[rec[0]] * inv_sqrt2;
            int8_t z[32];
            for (int q = 0; q < 4; ++q) wht8_scalar(u_all + (r * (COLS / 32) + b) * 32 + 8 * q, z + 8 * q);
            double d = 0.0;
            for (int j = 0; j < 32; ++j) d += (double)z[j] * (double)x[b * 32 + j];
            acc += s * d;
        }
        ref[r] = acc;
    }
    float y[3][ROWS];
    size_t badm = g_materialized(recs, scale_lut, x, ROWS, COLS, y[0]);
    size_t bada = g_adjoint(recs, scale_lut, x, ROWS, COLS, y[1]);
    size_t badb = g_batch4(recs, scale_lut, x, ROWS, COLS, y[2]);
    CHECK(badm == 0 && bada == 0 && badb == 0, "gemv bad counts %zu %zu %zu", badm, bada, badb);
    for (size_t r = 0; r < ROWS; ++r) {
        double tol = 1e-4 * (fabs(ref[r]) + 1.0);
        CHECK(fabs((double)y[0][r] - ref[r]) <= tol, "materialized row %zu: %f vs %f", r, y[0][r], ref[r]);
        CHECK(fabs((double)y[1][r] - ref[r]) <= tol, "adjoint row %zu: %f vs %f", r, y[1][r], ref[r]);
        CHECK(fabs((double)y[2][r] - ref[r]) <= tol, "batch4 row %zu: %f vs %f", r, y[2][r], ref[r]);
        CHECK(fabs((double)y[0][r] - (double)y[1][r]) <= 1e-3 * (fabs(ref[r]) + 1.0), "mat vs adj row %zu", r);
    }

    /* --- cols alignment: fail closed, no overread (ASan watches) --- */
    float x33[33]; memset(x33, 0, sizeof(x33));
    uint8_t *recs33 = calloc(ROWS * 2 * rec_sz, 1); /* 33 cols would need 2 blocks (64) */
    float y33[ROWS];
    CHECK(g_materialized(recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "materialized cols=33 must fail");
    CHECK(g_adjoint(recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "adjoint cols=33 must fail");
    CHECK(g_batch4(recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "batch4 cols=33 must fail");
    CHECK(g_q4(recs33, scale_lut, x33, ROWS, 33, y33) > ROWS, "q4 cols=33 must fail");
    CHECK(g_materialized(recs, scale_lut, x, ROWS, 0, y[0]) > ROWS, "materialized cols=0 must fail");
    CHECK(g_adjoint(recs, scale_lut, x, ROWS, 0, y[0]) > ROWS, "adjoint cols=0 must fail");
    CHECK(g_batch4(recs, scale_lut, x, ROWS, 0, y[0]) > ROWS, "batch4 cols=0 must fail");
    CHECK(g_q4(recs, scale_lut, x, ROWS, 0, y[0]) > ROWS, "q4 cols=0 must fail");
    free(recs33);

    /* --- Q4_K_M-style baseline: encoded zero point rec[1] must be honored --- */
    {
        const size_t QROWS = 4, QCOLS = 64, qrec_sz = 18;
        uint8_t *qrecs = malloc(QROWS * (QCOLS / 32) * qrec_sz);
        static const int zps[4] = {8, 3, 0, 15};
        for (size_t r = 0; r < QROWS; ++r)
            for (size_t b = 0; b < QCOLS / 32; ++b) {
                uint8_t *rec = qrecs + (r * (QCOLS / 32) + b) * qrec_sz;
                rng = rng * 1664525u + 1013904223u;
                rec[0] = (uint8_t)(rng & 0xFF);
                rec[1] = (uint8_t)zps[(r + b) % 4]; /* encoded zero point */
                for (int i = 0; i < 16; ++i) {
                    rng = rng * 1664525u + 1013904223u;
                    rec[2 + i] = (uint8_t)(rng >> 24);
                }
            }
        double qref[QROWS];
        for (size_t r = 0; r < QROWS; ++r) {
            double acc = 0.0;
            for (size_t b = 0; b < QCOLS / 32; ++b) {
                const uint8_t *rec = qrecs + (r * (QCOLS / 32) + b) * qrec_sz;
                double s = (double)scale_lut[rec[0]];
                double zp = (double)rec[1];
                double d = 0.0;
                for (int i = 0; i < 16; ++i) {
                    double q0 = (double)(rec[2 + i] & 0x0fu) - zp;
                    double q1 = (double)(rec[2 + i] >> 4) - zp;
                    d += q0 * (double)x[b * 32 + i] + q1 * (double)x[b * 32 + 16 + i];
                }
                acc += s * d;
            }
            qref[r] = acc;
        }
        float yq[QROWS];
        size_t qbad = g_q4(qrecs, scale_lut, x, QROWS, QCOLS, yq);
        CHECK(qbad == 0, "q4 bad=%zu", qbad);
        for (size_t r = 0; r < QROWS; ++r) {
            double tol = 1e-4 * (fabs(qref[r]) + 1.0);
            CHECK(fabs((double)yq[r] - qref[r]) <= tol,
                  "q4 zero_pt row %zu (zp=%d): %f vs ref %f", r, zps[r % 4], yq[r], qref[r]);
        }
        free(qrecs);
    }

    free(ranks);
    for (int m = 0; m < 4; ++m) free(out[m]);
    free(u_all); free(recs);
    e8tables_free(&t);

    if (failures) {
        fprintf(stderr, "SMOKE_FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    printf("SMOKE_PASS\n");
    return 0;
}
