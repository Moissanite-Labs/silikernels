/*
 * s71_proximity_bench.c
 *
 * Competing-Potential Proximity Barrier (64 B Cache-Line Opposite-Sign Force)
 * & Real Multicore AVX-512 A16_BlockH4 Quantized Matvec Unlock on Intel i9-11900H.
 *
 * Build (exact zero-warning reproducibility command):
 *   gcc -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mavx512vnni \
 *       -pthread -Wall -Wextra -Werror s71_proximity_bench.c -o s71_proximity_bench
 *
 * Modes:
 *   --self-test   Verify AVX-512 A16_BlockH4 bit-exact int64 parity & 0 syndrome errors (fast, no RAPL sweep)
 *   --run-live    Execute full palindromic ABBA live intel-rapl + rdtsc capture and emit JSON to stdout
 */

#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Constants matching Qwen2.5-3B down_proj (2048 x 11008 = 22,544,384 weights) ---- */
#define MAT_M 2048
#define MAT_K 11008
#define N_PHYS_CORES 8
#define N_STRIDES 9
#define N_GEMV_STRIDES 5
#define READ_BUF_BYTES (128ULL * 1024ULL * 1024ULL) /* 128 MiB > 5x 24 MiB L3 */
#define READ_STEPS 1048576ULL                       /* 1M lockstep DRAM steps x 8 cores = 8M reads */
#define READ_EPOCH 512ULL                           /* barrier every 512 steps for tight L3 coalescing */
#define WW_2C_OPS 16000000ULL
#define WW_8C_OPS 6000000ULL
#define GEMV_REPS_STRIDE 4
#define GEMV_REPS_OPT 32

static const uint64_t PROX_STRIDES[N_STRIDES] = {8, 16, 32, 64, 128, 256, 512, 4096, 65536};
static const uint64_t GEMV_STRIDES[N_GEMV_STRIDES] = {8, 16, 32, 64, 128};

/* ---- High-resolution timers & RAPL domain readers ---- */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t read_tsc(void) {
    unsigned int aux = 0;
    return __rdtscp(&aux);
}

static uint64_t read_u64_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 0ULL;
    unsigned long long val = 0ULL;
    int ok = fscanf(fp, "%llu", &val);
    fclose(fp);
    return (ok == 1) ? (uint64_t)val : 0ULL;
}

typedef struct {
    uint64_t pkg_uj;
    uint64_t core_uj;
    uint64_t uncore_uj;
    uint64_t psys_uj;
    uint64_t max_range_uj;
} RaplSnap;

static RaplSnap rapl_snapshot(void) {
    RaplSnap s;
    s.pkg_uj = read_u64_file("/sys/class/powercap/intel-rapl:0/energy_uj");
    s.core_uj = read_u64_file("/sys/class/powercap/intel-rapl:0:0/energy_uj");
    s.uncore_uj = read_u64_file("/sys/class/powercap/intel-rapl:0:1/energy_uj");
    s.psys_uj = read_u64_file("/sys/class/powercap/intel-rapl:1/energy_uj");
    s.max_range_uj = read_u64_file("/sys/class/powercap/intel-rapl:0/max_energy_range_uj");
    if (s.max_range_uj == 0ULL) s.max_range_uj = 262143328850ULL;
    return s;
}

static inline uint64_t wrap_diff(uint64_t after, uint64_t before, uint64_t max_range) {
    if (after >= before) return after - before;
    return (max_range - before) + after;
}

typedef struct {
    uint64_t ns;
    uint64_t tsc;
    uint64_t pkg_uj;
    uint64_t core_uj;
    uint64_t uncore_uj;
    uint64_t psys_uj;
    uint64_t total_ops;
    uint64_t checksum;
} Meas;

static void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/* ---- Pillar 1a: Write-Write MESI Coherence Repulsion Wall ---- */
typedef struct {
    uint8_t *buf;
    uint64_t stride;
    uint64_t ops_per_thread;
    pthread_barrier_t *bar_start;
    pthread_barrier_t *bar_end;
    int tid;
    uint64_t sink;
} WwArg;

static void *ww_worker(void *arg) {
    WwArg *a = (WwArg *)arg;
    pin_to_core(a->tid);
    uint64_t *slot = (uint64_t *)(void *)(a->buf + (uint64_t)a->tid * a->stride);
    pthread_barrier_wait(a->bar_start);
    uint64_t acc = 0ULL;
    for (uint64_t i = 0; i < a->ops_per_thread; ++i) {
        acc += __atomic_fetch_add(slot, 1ULL, __ATOMIC_RELAXED);
    }
    a->sink = acc;
    pthread_barrier_wait(a->bar_end);
    return NULL;
}

static Meas run_write_write(uint8_t *buf, int n_threads, uint64_t stride, uint64_t ops_per_t) {
    memset(buf, 0, (size_t)n_threads * 65536ULL + 4096ULL);
    pthread_t th[N_PHYS_CORES];
    WwArg args[N_PHYS_CORES];
    pthread_barrier_t bar_s, bar_e;
    pthread_barrier_init(&bar_s, NULL, (unsigned int)(n_threads + 1));
    pthread_barrier_init(&bar_e, NULL, (unsigned int)(n_threads + 1));

    for (int t = 0; t < n_threads; ++t) {
        args[t].buf = buf;
        args[t].stride = stride;
        args[t].ops_per_thread = ops_per_t;
        args[t].bar_start = &bar_s;
        args[t].bar_end = &bar_e;
        args[t].tid = t;
        args[t].sink = 0ULL;
        pthread_create(&th[t], NULL, ww_worker, &args[t]);
    }

    RaplSnap r0 = rapl_snapshot();
    uint64_t t0 = now_ns();
    uint64_t c0 = read_tsc();
    pthread_barrier_wait(&bar_s);
    pthread_barrier_wait(&bar_e);
    uint64_t c1 = read_tsc();
    uint64_t t1 = now_ns();
    RaplSnap r1 = rapl_snapshot();

    uint64_t chk = 0ULL;
    for (int t = 0; t < n_threads; ++t) {
        pthread_join(th[t], NULL);
        chk ^= args[t].sink;
    }
    pthread_barrier_destroy(&bar_s);
    pthread_barrier_destroy(&bar_e);

    Meas m;
    m.ns = t1 - t0;
    m.tsc = c1 - c0;
    m.pkg_uj = wrap_diff(r1.pkg_uj, r0.pkg_uj, r0.max_range_uj);
    m.core_uj = wrap_diff(r1.core_uj, r0.core_uj, r0.max_range_uj);
    m.uncore_uj = wrap_diff(r1.uncore_uj, r0.uncore_uj, r0.max_range_uj);
    m.psys_uj = wrap_diff(r1.psys_uj, r0.psys_uj, r0.max_range_uj);
    m.total_ops = (uint64_t)n_threads * ops_per_t;
    m.checksum = chk;
    return m;
}

/* ---- Pillar 1b: Read-Shared Coalescing Fusion Pull (128 MiB L3-exceeding buffer) ---- */
typedef struct {
    const uint8_t *buf;
    uint64_t stride;
    uint64_t steps;
    pthread_barrier_t *bar_start;
    pthread_barrier_t *bar_epoch;
    pthread_barrier_t *bar_end;
    int tid;
    uint64_t sink;
} ReadArg;

static void *read_worker(void *arg) {
    ReadArg *a = (ReadArg *)arg;
    pin_to_core(a->tid);
    const uint64_t line_mask = READ_BUF_BYTES - 64ULL;
    const uint64_t byte_mask = READ_BUF_BYTES - 8ULL;
    const uint64_t off_t = (uint64_t)a->tid * a->stride;
    uint64_t acc = 0ULL;

    pthread_barrier_wait(a->bar_start);
    for (uint64_t s = 0; s < a->steps; s += READ_EPOCH) {
        uint64_t end = s + READ_EPOCH;
        if (end > a->steps) end = a->steps;
        for (uint64_t i = s; i < end; ++i) {
            /* Odd multiplier mod 2^21 visits pseudo-random 64B cache lines across 128 MiB */
            uint64_t base = ((i * 1315423911ULL) << 6) & line_mask;
            uint64_t addr = (base + off_t) & byte_mask;
            const volatile uint64_t *ptr = (const volatile uint64_t *)(const void *)(a->buf + addr);
            acc += *ptr;
        }
        pthread_barrier_wait(a->bar_epoch);
    }
    a->sink = acc;
    pthread_barrier_wait(a->bar_end);
    return NULL;
}

static Meas run_read_shared(const uint8_t *buf, int n_threads, uint64_t stride, uint64_t steps) {
    pthread_t th[N_PHYS_CORES];
    ReadArg args[N_PHYS_CORES];
    pthread_barrier_t bar_s, bar_ep, bar_e;
    pthread_barrier_init(&bar_s, NULL, (unsigned int)(n_threads + 1));
    pthread_barrier_init(&bar_ep, NULL, (unsigned int)n_threads);
    pthread_barrier_init(&bar_e, NULL, (unsigned int)(n_threads + 1));

    for (int t = 0; t < n_threads; ++t) {
        args[t].buf = buf;
        args[t].stride = stride;
        args[t].steps = steps;
        args[t].bar_start = &bar_s;
        args[t].bar_epoch = &bar_ep;
        args[t].bar_end = &bar_e;
        args[t].tid = t;
        args[t].sink = 0ULL;
        pthread_create(&th[t], NULL, read_worker, &args[t]);
    }

    RaplSnap r0 = rapl_snapshot();
    uint64_t t0 = now_ns();
    uint64_t c0 = read_tsc();
    pthread_barrier_wait(&bar_s);
    pthread_barrier_wait(&bar_e);
    uint64_t c1 = read_tsc();
    uint64_t t1 = now_ns();
    RaplSnap r1 = rapl_snapshot();

    uint64_t chk = 0ULL;
    for (int t = 0; t < n_threads; ++t) {
        pthread_join(th[t], NULL);
        chk ^= args[t].sink;
    }
    pthread_barrier_destroy(&bar_s);
    pthread_barrier_destroy(&bar_ep);
    pthread_barrier_destroy(&bar_e);

    Meas m;
    m.ns = t1 - t0;
    m.tsc = c1 - c0;
    m.pkg_uj = wrap_diff(r1.pkg_uj, r0.pkg_uj, r0.max_range_uj);
    m.core_uj = wrap_diff(r1.core_uj, r0.core_uj, r0.max_range_uj);
    m.uncore_uj = wrap_diff(r1.uncore_uj, r0.uncore_uj, r0.max_range_uj);
    m.psys_uj = wrap_diff(r1.psys_uj, r0.psys_uj, r0.max_range_uj);
    m.total_ops = (uint64_t)n_threads * steps;
    m.checksum = chk;
    return m;
}

/* ---- Pillar 2: Native AVX-512 VNNI A16 [16,11,4] + BlockH4 Quantized Matvec ---- */
static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint32_t a16_syndrome_16(uint32_t odd_mask) {
    uint32_t p0 = (uint32_t)__builtin_popcount(odd_mask & 0xFFFFu) & 1u;
    uint32_t s0 = (uint32_t)__builtin_popcount(odd_mask & 0xAAAAu) & 1u;
    uint32_t s1 = (uint32_t)__builtin_popcount(odd_mask & 0xCCCCu) & 1u;
    uint32_t s2 = (uint32_t)__builtin_popcount(odd_mask & 0xF0F0u) & 1u;
    uint32_t s3 = (uint32_t)__builtin_popcount(odd_mask & 0xFF00u) & 1u;
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4);
}

/* Encode deterministic bounded [-7,7] weights into valid Extended Hamming [16,11,4] codewords */
static void build_a16_matrix(uint64_t *packed, int8_t *x_i8, uint64_t *syn_errs_out) {
    const size_t n_blk16 = ((size_t)MAT_M * (size_t)MAT_K) / 16ULL;
    uint64_t rng = 0x5371202611240071ULL;
    uint64_t syn_errs = 0ULL;

    for (size_t b = 0; b < n_blk16; ++b) {
        int8_t w[16];
        uint32_t odd = 0u;
        uint64_t r1 = splitmix64(&rng);
        uint64_t r2 = splitmix64(&rng);
        for (int j = 0; j < 16; ++j) {
            uint64_t nib = ((j < 8) ? (r1 >> (j * 8)) : (r2 >> ((j - 8) * 8))) & 0xFu;
            int v = (int)(nib % 15u) - 7; /* strictly in [-7, 7] */
            w[j] = (int8_t)v;
            odd |= ((uint32_t)(v & 1)) << j;
        }
        /* Fix parity positions 1, 2, 4, 8, and overall 0 to enforce syndrome == 0 */
        uint32_t s0 = (uint32_t)__builtin_popcount(odd & 0xAAAAu) & 1u;
        uint32_t s1 = (uint32_t)__builtin_popcount(odd & 0xCCCCu) & 1u;
        uint32_t s2 = (uint32_t)__builtin_popcount(odd & 0xF0F0u) & 1u;
        uint32_t s3 = (uint32_t)__builtin_popcount(odd & 0xFF00u) & 1u;
        int fix_idx[4] = {1, 2, 4, 8};
        uint32_t fix_s[4] = {s0, s1, s2, s3};
        for (int f = 0; f < 4; ++f) {
            if (fix_s[f]) {
                int idx = fix_idx[f];
                w[idx] = (w[idx] < 7) ? (int8_t)(w[idx] + 1) : (int8_t)(w[idx] - 1);
                odd ^= (1u << idx);
            }
        }
        uint32_t p0 = (uint32_t)__builtin_popcount(odd & 0xFFFFu) & 1u;
        if (p0) {
            w[0] = (w[0] < 7) ? (int8_t)(w[0] + 1) : (int8_t)(w[0] - 1);
            odd ^= 1u;
        }
        if (a16_syndrome_16(odd) != 0u) ++syn_errs;

        uint64_t word = 0ULL;
        for (int j = 0; j < 16; ++j) {
            word |= ((uint64_t)((uint8_t)w[j] & 0xFu)) << (j * 4);
        }
        packed[b] = word;
    }

    for (int c = 0; c < MAT_K; ++c) {
        int v = (int)(splitmix64(&rng) % 255ULL) - 127; /* [-127, 127] */
        x_i8[c] = (int8_t)v;
    }
    *syn_errs_out = syn_errs;
}

/* Exact scalar int64 reference for unscaled (H_4 W_q) * x_i8 */
static int64_t scalar_blockh4_row(const uint64_t *row_p, const int8_t *x_i8) {
    const int n_blk16 = MAT_K / 16;
    int64_t acc = 0;
    for (int b = 0; b < n_blk16; ++b) {
        uint64_t w64 = row_p[b];
        const int8_t *xb = x_i8 + b * 16;
        for (int g = 0; g < 4; ++g) {
            uint32_t nibs = (uint32_t)(w64 >> (g * 16));
            int64_t a0 = ((int32_t)(nibs & 0xFu) << 28) >> 28;
            int64_t a1 = ((int32_t)((nibs >> 4) & 0xFu) << 28) >> 28;
            int64_t a2 = ((int32_t)((nibs >> 8) & 0xFu) << 28) >> 28;
            int64_t a3 = ((int32_t)((nibs >> 12) & 0xFu) << 28) >> 28;
            int64_t u0 = a0 + a1 + a2 + a3;
            int64_t u1 = a0 - a1 + a2 - a3;
            int64_t u2 = a0 + a1 - a2 - a3;
            int64_t u3 = a0 - a1 - a2 + a3;
            acc += u0 * xb[g * 4 + 0] + u1 * xb[g * 4 + 1] +
                   u2 * xb[g * 4 + 2] + u3 * xb[g * 4 + 3];
        }
    }
    return acc;
}

/*
 * Native AVX-512 VNNI 64-weight (32-byte) A16_BlockH4 dot-product step:
 *   - Decodes 64 signed 4-bit nibbles via vpshufb LUT (even/odd split -> a0,a2 and a1,a3!)
 *   - Performs exact register-only H_4 butterfly across 16 quads in parallel:
 *       u = [s0+s1, d0+d1, s0-s1, d0-d1] in [-28, 28]
 *   - Shifts u + 32 in [4, 60] subset [0, 255] (exact unsigned uint8!)
 *   - Fires single-instruction _mm512_dpbusd_epi32(acc, u_plus_32, x_i8)
 */
static inline __m512i avx512_blockh4_tile64(
    const uint8_t *w32,
    const int8_t *x64,
    __m512i acc,
    __m256i lut256,
    __m256i mask0f,
    __m256i shuf_pair_swap,
    __m512i bias32) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)(const void *)w32);
    __m256i lo_n = _mm256_and_si256(raw, mask0f);
    __m256i hi_n = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0f);

    /* Sign-extend 4-bit nibbles into [-8, 7] via 16-byte in-lane LUT:
     * a_even = [a0, a2, a4, a6, ...] (32 bytes), a_odd = [a1, a3, a5, a7, ...] (32 bytes) */
    __m256i a_even = _mm256_shuffle_epi8(lut256, lo_n);
    __m256i a_odd  = _mm256_shuffle_epi8(lut256, hi_n);

    /* Stage 1 butterfly: s = [s0, s1, ...], d = [d0, d1, ...] */
    __m256i s = _mm256_add_epi8(a_even, a_odd);
    __m256i d = _mm256_sub_epi8(a_even, a_odd);

    /* Pair-swapped: s_sw = [s1, s0, ...], d_sw = [d1, d0, ...] */
    __m256i s_sw = _mm256_shuffle_epi8(s, shuf_pair_swap);
    __m256i d_sw = _mm256_shuffle_epi8(d, shuf_pair_swap);

    /* Stage 2 butterfly:
     * s_sum = [s0+s1, s1+s0, ...], s_dif = [s0-s1, s1-s0, ...]
     * Note lane 0 of s_sum = u0 = s0+s1, lane 0 of s_dif = u2 = s0-s1!
     * Blend even bytes from sum and odd bytes from diff -> [u0, u2, u0', u2', ...] */
    __m256i s_sum = _mm256_add_epi8(s, s_sw);
    __m256i s_dif = _mm256_sub_epi8(s, s_sw);
    __m256i d_sum = _mm256_add_epi8(d, d_sw);
    __m256i d_dif = _mm256_sub_epi8(d, d_sw);

    /* Put u0 at even byte, u2 at odd byte (s_dif shifted left by 1 byte = odd position) */
    const __m256i even_mask = _mm256_set1_epi16(0x00FF);
    __m256i u02 = _mm256_or_si256(_mm256_and_si256(s_sum, even_mask),
                                  _mm256_slli_epi16(_mm256_and_si256(s_dif, even_mask), 8));
    __m256i u13 = _mm256_or_si256(_mm256_and_si256(d_sum, even_mask),
                                  _mm256_slli_epi16(_mm256_and_si256(d_dif, even_mask), 8));

    /* Interleave u02 and u13 bytewise -> [u0, u1, u2, u3, ...] across 64 bytes!
     * unpacklo_epi8 produces lanes [0..7, 16..23] in 128-bit lanes; unpackhi produces [8..15, 24..31].
     * Reorder 128-bit lanes into exact [0..15, 16..31, 32..47, 48..63] order in ZMM! */
    __m256i ilo = _mm256_unpacklo_epi8(u02, u13); /* [lane0: bytes 0..15,  lane1: bytes 32..47] */
    __m256i ihi = _mm256_unpackhi_epi8(u02, u13); /* [lane0: bytes 16..31, lane1: bytes 48..63] */

    __m256i z_lo = _mm256_permute2x128_si256(ilo, ihi, 0x20); /* bytes 0..31 */
    __m256i z_hi = _mm256_permute2x128_si256(ilo, ihi, 0x31); /* bytes 32..63 */
    __m512i u64  = _mm512_inserti64x4(_mm512_castsi256_si512(z_lo), z_hi, 1);

    /* Shift into exact unsigned [4, 60] and fire AVX-512 VNNI vpdpbusd! */
    __m512i u64_u8 = _mm512_add_epi8(u64, bias32);
    __m512i xv = _mm512_loadu_si512((const void *)x64);
    return _mm512_dpbusd_epi32(acc, u64_u8, xv);
}

static inline int64_t avx512_blockh4_row_full(
    const uint64_t *row_p,
    const int8_t *x_i8,
    int64_t x_sum32) {
    const __m256i lut256 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const __m256i shuf_pair_swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i bias32 = _mm512_set1_epi8(32);

    const uint8_t *rp = (const uint8_t *)(const void *)row_p;
    __m512i acc = _mm512_setzero_si512();
    for (int k = 0; k < MAT_K; k += 64) {
        acc = avx512_blockh4_tile64(rp + (k >> 1), x_i8 + k, acc, lut256, mask0f, shuf_pair_swap, bias32);
    }
    int64_t sum = (int64_t)_mm512_reduce_add_epi32(acc);
    return sum - x_sum32;
}

/* ---- Pillar 2a: Split-K AVX-512 Matvec Sweeping Shared Accumulator Stride r ---- */
typedef struct {
    const uint64_t *packed;
    const int8_t *x_i8;
    int64_t x_sum32;
    uint8_t *shared_acc_buf;
    uint64_t acc_stride_bytes;
    int64_t *out_y;
    int reps;
    int tid;
    int n_threads;
    pthread_barrier_t *bar_start;
    pthread_barrier_t *bar_end;
    uint64_t chk;
} GemvStrideArg;

static void *gemv_stride_worker(void *arg) {
    GemvStrideArg *a = (GemvStrideArg *)arg;
    pin_to_core(a->tid);

    const __m256i lut256 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const __m256i shuf_pair_swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i bias32 = _mm512_set1_epi8(32);

    const int rows_per_t = MAT_M / a->n_threads;
    const int r0 = a->tid * rows_per_t;
    const int r1 = r0 + rows_per_t;
    const size_t row_bytes = MAT_K / 2;
    volatile int64_t *my_acc = (volatile int64_t *)(void *)(a->shared_acc_buf + (uint64_t)a->tid * a->acc_stride_bytes);

    pthread_barrier_wait(a->bar_start);
    uint64_t chk = 0ULL;
    for (int rep = 0; rep < a->reps; ++rep) {
        for (int r = r0; r < r1; ++r) {
            const uint8_t *rp = (const uint8_t *)(const void *)(a->packed + (size_t)r * (MAT_K / 16));
            (void)row_bytes;
            *my_acc = 0;
            /* Unpadded accumulator updated per 64-weight AVX-512 tile (exposes 64 B false-sharing barrier!) */
            for (int k = 0; k < MAT_K; k += 64) {
                __m512i v = avx512_blockh4_tile64(
                    rp + (k >> 1), a->x_i8 + k, _mm512_setzero_si512(),
                    lut256, mask0f, shuf_pair_swap, bias32);
                int64_t part = (int64_t)_mm512_reduce_add_epi32(v);
                *my_acc = *my_acc + part;
            }
            int64_t final_v = *my_acc - a->x_sum32;
            a->out_y[r] = final_v;
            chk ^= (uint64_t)final_v;
        }
    }
    a->chk = chk;
    pthread_barrier_wait(a->bar_end);
    return NULL;
}

static Meas run_gemv_stride(
    const uint64_t *packed, const int8_t *x_i8, int64_t x_sum32,
    uint8_t *shared_acc_buf, uint64_t stride, int64_t *out_y, int reps) {
    memset(shared_acc_buf, 0, 4096ULL);
    pthread_t th[N_PHYS_CORES];
    GemvStrideArg args[N_PHYS_CORES];
    pthread_barrier_t bar_s, bar_e;
    pthread_barrier_init(&bar_s, NULL, N_PHYS_CORES + 1);
    pthread_barrier_init(&bar_e, NULL, N_PHYS_CORES + 1);

    for (int t = 0; t < N_PHYS_CORES; ++t) {
        args[t].packed = packed;
        args[t].x_i8 = x_i8;
        args[t].x_sum32 = x_sum32;
        args[t].shared_acc_buf = shared_acc_buf;
        args[t].acc_stride_bytes = stride;
        args[t].out_y = out_y;
        args[t].reps = reps;
        args[t].tid = t;
        args[t].n_threads = N_PHYS_CORES;
        args[t].bar_start = &bar_s;
        args[t].bar_end = &bar_e;
        args[t].chk = 0ULL;
        pthread_create(&th[t], NULL, gemv_stride_worker, &args[t]);
    }

    RaplSnap r0 = rapl_snapshot();
    uint64_t t0 = now_ns();
    uint64_t c0 = read_tsc();
    pthread_barrier_wait(&bar_s);
    pthread_barrier_wait(&bar_e);
    uint64_t c1 = read_tsc();
    uint64_t t1 = now_ns();
    RaplSnap r1 = rapl_snapshot();

    uint64_t chk = 0ULL;
    for (int t = 0; t < N_PHYS_CORES; ++t) {
        pthread_join(th[t], NULL);
        chk ^= args[t].chk;
    }
    pthread_barrier_destroy(&bar_s);
    pthread_barrier_destroy(&bar_e);

    Meas m;
    m.ns = t1 - t0;
    m.tsc = c1 - c0;
    m.pkg_uj = wrap_diff(r1.pkg_uj, r0.pkg_uj, r0.max_range_uj);
    m.core_uj = wrap_diff(r1.core_uj, r0.core_uj, r0.max_range_uj);
    m.uncore_uj = wrap_diff(r1.uncore_uj, r0.uncore_uj, r0.max_range_uj);
    m.psys_uj = wrap_diff(r1.psys_uj, r0.psys_uj, r0.max_range_uj);
    m.total_ops = (uint64_t)reps * (uint64_t)MAT_M * (uint64_t)MAT_K;
    m.checksum = chk;
    return m;
}

/* ---- Pillar 2b: Dual-Force Optimal AVX-512 Kernel (Private Reg Acc + Read-Shared x_i8 vs Dispersed x) ---- */
typedef struct {
    const uint64_t *packed;
    const int8_t *x_i8;       /* contiguous 64B-aligned x_i8 OR dispersed 4096B-strided x_disp */
    int dispersed_x;          /* 0 = read-shared contiguous (fusion pull), 1 = 4096B-strided per 64B tile */
    int64_t x_sum32;
    int64_t *out_y;
    int reps;
    int tid;
    int n_threads;
    pthread_barrier_t *bar_start;
    pthread_barrier_t *bar_end;
    uint64_t chk;
} GemvOptArg;

static void *gemv_opt_worker(void *arg) {
    GemvOptArg *a = (GemvOptArg *)arg;
    pin_to_core(a->tid);

    const __m256i lut256 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    const __m256i shuf_pair_swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const __m512i bias32 = _mm512_set1_epi8(32);

    const int rows_per_t = MAT_M / a->n_threads;
    const int r0 = a->tid * rows_per_t;
    const int r1 = r0 + rows_per_t;

    pthread_barrier_wait(a->bar_start);
    uint64_t chk = 0ULL;
    for (int rep = 0; rep < a->reps; ++rep) {
        for (int r = r0; r < r1; ++r) {
            const uint8_t *rp = (const uint8_t *)(const void *)(a->packed + (size_t)r * (MAT_K / 16));
            __m512i acc = _mm512_setzero_si512();
            if (!a->dispersed_x) {
                for (int k = 0; k < MAT_K; k += 64) {
                    acc = avx512_blockh4_tile64(
                        rp + (k >> 1), a->x_i8 + k, acc,
                        lut256, mask0f, shuf_pair_swap, bias32);
                }
            } else {
                /* Each 64-byte x tile placed on a separate 4096 B page (destroys read-shared fusion!) */
                int tile = 0;
                for (int k = 0; k < MAT_K; k += 64, ++tile) {
                    acc = avx512_blockh4_tile64(
                        rp + (k >> 1), a->x_i8 + ((size_t)tile * 4096ULL), acc,
                        lut256, mask0f, shuf_pair_swap, bias32);
                }
            }
            int64_t final_v = (int64_t)_mm512_reduce_add_epi32(acc) - a->x_sum32;
            a->out_y[r] = final_v;
            chk ^= (uint64_t)final_v;
        }
    }
    a->chk = chk;
    pthread_barrier_wait(a->bar_end);
    return NULL;
}

static Meas run_gemv_opt(
    const uint64_t *packed, const int8_t *x_ptr, int dispersed_x,
    int64_t x_sum32, int64_t *out_y, int reps) {
    pthread_t th[N_PHYS_CORES];
    GemvOptArg args[N_PHYS_CORES];
    pthread_barrier_t bar_s, bar_e;
    pthread_barrier_init(&bar_s, NULL, N_PHYS_CORES + 1);
    pthread_barrier_init(&bar_e, NULL, N_PHYS_CORES + 1);

    for (int t = 0; t < N_PHYS_CORES; ++t) {
        args[t].packed = packed;
        args[t].x_i8 = x_ptr;
        args[t].dispersed_x = dispersed_x;
        args[t].x_sum32 = x_sum32;
        args[t].out_y = out_y;
        args[t].reps = reps;
        args[t].tid = t;
        args[t].n_threads = N_PHYS_CORES;
        args[t].bar_start = &bar_s;
        args[t].bar_end = &bar_e;
        args[t].chk = 0ULL;
        pthread_create(&th[t], NULL, gemv_opt_worker, &args[t]);
    }

    RaplSnap r0 = rapl_snapshot();
    uint64_t t0 = now_ns();
    uint64_t c0 = read_tsc();
    pthread_barrier_wait(&bar_s);
    pthread_barrier_wait(&bar_e);
    uint64_t c1 = read_tsc();
    uint64_t t1 = now_ns();
    RaplSnap r1 = rapl_snapshot();

    uint64_t chk = 0ULL;
    for (int t = 0; t < N_PHYS_CORES; ++t) {
        pthread_join(th[t], NULL);
        chk ^= args[t].chk;
    }
    pthread_barrier_destroy(&bar_s);
    pthread_barrier_destroy(&bar_e);

    Meas m;
    m.ns = t1 - t0;
    m.tsc = c1 - c0;
    m.pkg_uj = wrap_diff(r1.pkg_uj, r0.pkg_uj, r0.max_range_uj);
    m.core_uj = wrap_diff(r1.core_uj, r0.core_uj, r0.max_range_uj);
    m.uncore_uj = wrap_diff(r1.uncore_uj, r0.uncore_uj, r0.max_range_uj);
    m.psys_uj = wrap_diff(r1.psys_uj, r0.psys_uj, r0.max_range_uj);
    m.total_ops = (uint64_t)reps * (uint64_t)MAT_M * (uint64_t)MAT_K;
    m.checksum = chk;
    return m;
}

static Meas sum_meas(Meas a, Meas b) {
    Meas s;
    s.ns = a.ns + b.ns;
    s.tsc = a.tsc + b.tsc;
    s.pkg_uj = a.pkg_uj + b.pkg_uj;
    s.core_uj = a.core_uj + b.core_uj;
    s.uncore_uj = a.uncore_uj + b.uncore_uj;
    s.psys_uj = a.psys_uj + b.psys_uj;
    s.total_ops = a.total_ops + b.total_ops;
    s.checksum = a.checksum ^ b.checksum;
    return s;
}

static void print_meas_json(const char *label, uint64_t stride, Meas m, int trailing_comma) {
    printf("    {\"label\":\"%s\",\"stride_bytes\":%llu,\"total_ops\":%llu,\"ns\":%llu,\"tsc\":%llu,"
           "\"pkg_uj\":%llu,\"core_uj\":%llu,\"uncore_uj\":%llu,\"psys_uj\":%llu,\"checksum\":%llu}%s\n",
           label, (unsigned long long)stride, (unsigned long long)m.total_ops,
           (unsigned long long)m.ns, (unsigned long long)m.tsc,
           (unsigned long long)m.pkg_uj, (unsigned long long)m.core_uj,
           (unsigned long long)m.uncore_uj, (unsigned long long)m.psys_uj,
           (unsigned long long)m.checksum, trailing_comma ? "," : "");
}

int main(int argc, char **argv) {
    int self_test = 0;
    int run_live = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0) self_test = 1;
        else if (strcmp(argv[i], "--run-live") == 0) run_live = 1;
    }
    if (!self_test && !run_live) {
        fprintf(stderr, "Usage: %s --self-test | --run-live\n", argv[0]);
        return 2;
    }

    const size_t n_blk16 = ((size_t)MAT_M * (size_t)MAT_K) / 16ULL;
    uint64_t *packed = (uint64_t *)aligned_alloc(64, n_blk16 * sizeof(uint64_t));
    int8_t *x_i8 = (int8_t *)aligned_alloc(64, MAT_K);
    int64_t *y_ref = (int64_t *)aligned_alloc(64, (size_t)MAT_M * sizeof(int64_t));
    int64_t *y_avx = (int64_t *)aligned_alloc(64, (size_t)MAT_M * sizeof(int64_t));
    if (!packed || !x_i8 || !y_ref || !y_avx) {
        fprintf(stderr, "Allocation failed\n");
        return 3;
    }

    uint64_t syn_errs = 0ULL;
    build_a16_matrix(packed, x_i8, &syn_errs);
    int64_t x_sum32 = 0;
    for (int k = 0; k < MAT_K; ++k) x_sum32 += (int64_t)x_i8[k] * 32LL;

    /* Bit-exact verification across ALL 2048 rows (22,544,384 weights!) */
    uint64_t bit_mismatches = 0ULL;
    for (int r = 0; r < MAT_M; ++r) {
        const uint64_t *rp = packed + (size_t)r * (MAT_K / 16);
        y_ref[r] = scalar_blockh4_row(rp, x_i8);
        y_avx[r] = avx512_blockh4_row_full(rp, x_i8, x_sum32);
        if (y_ref[r] != y_avx[r]) ++bit_mismatches;
    }

    if (self_test) {
        printf("{\"self_test\":true,\"M\":%d,\"K\":%d,\"weights\":%llu,\"syndrome_errors\":%llu,\"avx512_int64_mismatches\":%llu}\n",
               MAT_M, MAT_K, (unsigned long long)((uint64_t)MAT_M * (uint64_t)MAT_K),
               (unsigned long long)syn_errs, (unsigned long long)bit_mismatches);
        free(packed); free(x_i8); free(y_ref); free(y_avx);
        return (syn_errs == 0ULL && bit_mismatches == 0ULL) ? 0 : 4;
    }

    if (syn_errs != 0ULL || bit_mismatches != 0ULL) {
        fprintf(stderr, "Parity check failed: syn=%llu mism=%llu\n",
                (unsigned long long)syn_errs, (unsigned long long)bit_mismatches);
        return 4;
    }

    /* Allocate Pillar 1 buffers + dispersed x_i8 buffer */
    uint8_t *ww_buf = (uint8_t *)aligned_alloc(4096, (size_t)N_PHYS_CORES * 65536ULL + 4096ULL);
    uint8_t *rd_buf = (uint8_t *)aligned_alloc(4096, READ_BUF_BYTES);
    uint8_t *acc_buf = (uint8_t *)aligned_alloc(4096, 4096ULL);
    const size_t n_tiles = MAT_K / 64;
    int8_t *x_disp = (int8_t *)aligned_alloc(4096, n_tiles * 4096ULL);
    if (!ww_buf || !rd_buf || !acc_buf || !x_disp) {
        fprintf(stderr, "Benchmark allocation failed\n");
        return 3;
    }
    /* Pre-fault all pages so page faults never contaminate RAPL or timing */
    for (size_t off = 0; off < READ_BUF_BYTES; off += 4096ULL) {
        rd_buf[off] = (uint8_t)((off >> 12) ^ 0x5Au);
        if (off + 64ULL < READ_BUF_BYTES) rd_buf[off + 64ULL] = (uint8_t)(off & 0xFFu);
    }
    memset(x_disp, 0, n_tiles * 4096ULL);
    for (size_t t = 0; t < n_tiles; ++t) {
        memcpy(x_disp + t * 4096ULL, x_i8 + t * 64ULL, 64ULL);
    }

    /* Quick warmup to stabilize CPU P-state & verify multicore bit-exactness */
    (void)run_gemv_opt(packed, x_i8, 0, x_sum32, y_avx, 2);
    uint64_t opt_mism = 0ULL;
    for (int r = 0; r < MAT_M; ++r) {
        if (y_avx[r] != y_ref[r]) ++opt_mism;
    }
    (void)run_gemv_opt(packed, x_disp, 1, x_sum32, y_avx, 1);
    for (int r = 0; r < MAT_M; ++r) {
        if (y_avx[r] != y_ref[r]) ++opt_mism;
    }

    /* ---- Palindromic ABBA sweeps (ascending + descending cancels linear thermal drift!) ---- */
    Meas ww2_a[N_STRIDES], ww2_b[N_STRIDES], ww2_sum[N_STRIDES];
    for (int i = 0; i < N_STRIDES; ++i) ww2_a[i] = run_write_write(ww_buf, 2, PROX_STRIDES[i], WW_2C_OPS);
    for (int i = N_STRIDES - 1; i >= 0; --i) {
        ww2_b[i] = run_write_write(ww_buf, 2, PROX_STRIDES[i], WW_2C_OPS);
        ww2_sum[i] = sum_meas(ww2_a[i], ww2_b[i]);
    }

    Meas ww8_a[N_STRIDES], ww8_b[N_STRIDES], ww8_sum[N_STRIDES];
    for (int i = 0; i < N_STRIDES; ++i) ww8_a[i] = run_write_write(ww_buf, N_PHYS_CORES, PROX_STRIDES[i], WW_8C_OPS);
    for (int i = N_STRIDES - 1; i >= 0; --i) {
        ww8_b[i] = run_write_write(ww_buf, N_PHYS_CORES, PROX_STRIDES[i], WW_8C_OPS);
        ww8_sum[i] = sum_meas(ww8_a[i], ww8_b[i]);
    }

    Meas rd8_a[N_STRIDES], rd8_b[N_STRIDES], rd8_sum[N_STRIDES];
    for (int i = 0; i < N_STRIDES; ++i) rd8_a[i] = run_read_shared(rd_buf, N_PHYS_CORES, PROX_STRIDES[i], READ_STEPS);
    for (int i = N_STRIDES - 1; i >= 0; --i) {
        rd8_b[i] = run_read_shared(rd_buf, N_PHYS_CORES, PROX_STRIDES[i], READ_STEPS);
        rd8_sum[i] = sum_meas(rd8_a[i], rd8_b[i]);
    }

    /* Pillar 2a: AVX-512 A16_BlockH4 shared-accumulator stride sweep (ABBA) */
    Meas gs_a[N_GEMV_STRIDES], gs_b[N_GEMV_STRIDES], gs_sum[N_GEMV_STRIDES];
    for (int i = 0; i < N_GEMV_STRIDES; ++i) {
        gs_a[i] = run_gemv_stride(packed, x_i8, x_sum32, acc_buf, GEMV_STRIDES[i], y_avx, GEMV_REPS_STRIDE);
    }
    for (int i = N_GEMV_STRIDES - 1; i >= 0; --i) {
        gs_b[i] = run_gemv_stride(packed, x_i8, x_sum32, acc_buf, GEMV_STRIDES[i], y_avx, GEMV_REPS_STRIDE);
        gs_sum[i] = sum_meas(gs_a[i], gs_b[i]);
    }
    for (int r = 0; r < MAT_M; ++r) {
        if (y_avx[r] != y_ref[r]) ++opt_mism;
    }

    /* Pillar 2b: Dual-Force Optimal vs Dispersed-Read Control (ABBA: A-B-B-A) */
    Meas opt_a1 = run_gemv_opt(packed, x_i8, 0, x_sum32, y_avx, GEMV_REPS_OPT);
    Meas disp_b1 = run_gemv_opt(packed, x_disp, 1, x_sum32, y_avx, GEMV_REPS_OPT);
    Meas disp_b2 = run_gemv_opt(packed, x_disp, 1, x_sum32, y_avx, GEMV_REPS_OPT);
    Meas opt_a2 = run_gemv_opt(packed, x_i8, 0, x_sum32, y_avx, GEMV_REPS_OPT);
    Meas opt_sum = sum_meas(opt_a1, opt_a2);
    Meas disp_sum = sum_meas(disp_b1, disp_b2);

    printf("{\n");
    printf("  \"schema\":\"aletheia.s71.proximity_barrier_silicon_raw.v1\",\n");
    printf("  \"protocol\":\"palindromic_abba_exact_integer\",\n");
    printf("  \"physical_cores\":%d,\n", N_PHYS_CORES);
    printf("  \"mat_m\":%d,\"mat_k\":%d,\"mat_weights\":%llu,\n",
           MAT_M, MAT_K, (unsigned long long)((uint64_t)MAT_M * (uint64_t)MAT_K));
    printf("  \"a16_syndrome_errors\":%llu,\"avx512_int64_mismatches\":%llu,\"multicore_int64_mismatches\":%llu,\n",
           (unsigned long long)syn_errs, (unsigned long long)bit_mismatches, (unsigned long long)opt_mism);
    printf("  \"write_write_2core\":[\n");
    for (int i = 0; i < N_STRIDES; ++i) print_meas_json("ww2", PROX_STRIDES[i], ww2_sum[i], i + 1 < N_STRIDES);
    printf("  ],\n");
    printf("  \"write_write_8core\":[\n");
    for (int i = 0; i < N_STRIDES; ++i) print_meas_json("ww8", PROX_STRIDES[i], ww8_sum[i], i + 1 < N_STRIDES);
    printf("  ],\n");
    printf("  \"read_shared_8core\":[\n");
    for (int i = 0; i < N_STRIDES; ++i) print_meas_json("rd8", PROX_STRIDES[i], rd8_sum[i], i + 1 < N_STRIDES);
    printf("  ],\n");
    printf("  \"avx512_a16_blockh4_stride_sweep\":[\n");
    for (int i = 0; i < N_GEMV_STRIDES; ++i) print_meas_json("gemv_stride", GEMV_STRIDES[i], gs_sum[i], i + 1 < N_GEMV_STRIDES);
    printf("  ],\n");
    printf("  \"avx512_a16_blockh4_dual_force\":[\n");
    print_meas_json("dual_force_aligned_reg_read_shared", 64ULL, opt_sum, 1);
    print_meas_json("aligned_reg_dispersed_read_4096B", 4096ULL, disp_sum, 0);
    printf("  ]\n");
    printf("}\n");

    free(packed); free(x_i8); free(y_ref); free(y_avx);
    free(ww_buf); free(rd_buf); free(acc_buf); free(x_disp);
    return (opt_mism == 0ULL) ? 0 : 4;
}
