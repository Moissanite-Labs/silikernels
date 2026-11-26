/*
 * Controlled 7-Regime Millisecond intel-rapl + Thermal Silicon Microkernel Harness
 *
 * Strictly zero audio, zero camera, zero display, zero network I/O.
 * Pins 4 worker threads to physical cores CPU 0..3 and 1 telemetry sampler thread
 * to CPU 7, sampling 20 exact integer sysfs channels at 50 ms (20 Hz):
 *   - 5 unlocked intel-rapl energy counters (package-0, core, uncore, psys, mmio:0)
 *   - 9 coretemp temperatures (Package id 0 + Core 0..7, millidegrees C)
 *   - 4 pinned-core frequencies (cpu0..cpu3 scaling_cur_freq, kHz)
 *   - 2 chassis fan speeds (cpu_fan, gpu_fan, RPM)
 *
 * Emits a canonical pure-integer JSON snapshot to stdout or --output path.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NUM_WORKERS 4
#define SAMPLER_CPU 7
#define NUM_REGIMES 7
#define NUM_CHANNELS 20
#define MAX_SAMPLES 2048

/* 64 MiB per worker = 256 MiB total (> 10x 24 MiB L3 cache on i9-11900H) */
#define DRAM_BYTES_PER_WORKER (64ULL * 1024ULL * 1024ULL)
#define CACHE_LINE_BYTES 64ULL
#define DRAM_LINES_PER_WORKER (DRAM_BYTES_PER_WORKER / CACHE_LINE_BYTES)

typedef struct {
    const char *id;
    const char *category;
    int regime_idx;
} regime_desc_t;

static const regime_desc_t REGIMES[NUM_REGIMES] = {
    {"R0_IDLE", "idle_baseline", 0},
    {"R1_BIJECTIVE_FEISTEL_LOW_H", "qfree_bijective_low_toggle", 1},
    {"R2_ERASE_DESTRUCTIVE_LOW_H", "mfree_irreversible_low_toggle", 2},
    {"R3_BIJECTIVE_FEISTEL_HIGH_H", "qfree_bijective_high_toggle", 3},
    {"R4_AVX512_FMA_L1", "avx512_fp64_fma_l1", 4},
    {"R5_AVX512_VNNI_L1", "avx512_int8_vnni_l1", 5},
    {"R6_DRAM_STREAM_L3_MISS", "dram_l3_miss_transport", 6},
};

typedef struct {
    const char *name;
    const char *unit;
    const char *modality;
    char path[256];
    uint64_t max_range;
} channel_desc_t;

static channel_desc_t CHANNELS[NUM_CHANNELS];

static int read_u64_file(const char *path, uint64_t *out) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(buf, &end, 10);
    if (errno != 0 || end == buf) return -1;
    *out = (uint64_t)v;
    return 0;
}

static int read_i64_file(const char *path, int64_t *out) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *end = NULL;
    errno = 0;
    long long v = strtoll(buf, &end, 10);
    if (errno != 0 || end == buf) return -1;
    *out = (int64_t)v;
    return 0;
}

static int find_hwmon_by_name(const char *target, char *out_dir, size_t out_sz) {
    for (int i = 0; i < 64; i++) {
        char p[128];
        snprintf(p, sizeof(p), "/sys/class/hwmon/hwmon%d/name", i);
        int fd = open(p, O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        char buf[64];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        close(fd);
        if (n <= 0) continue;
        buf[n] = '\0';
        while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) {
            buf[--n] = '\0';
        }
        if (strcmp(buf, target) == 0) {
            snprintf(out_dir, out_sz, "/sys/class/hwmon/hwmon%d", i);
            return 0;
        }
    }
    return -1;
}

static int init_channels(void) {
    char coretemp_dir[128];
    char asus_dir[128];
    if (find_hwmon_by_name("coretemp", coretemp_dir, sizeof(coretemp_dir)) != 0) {
        fprintf(stderr, "ERROR: coretemp hwmon not found\n");
        return -1;
    }
    if (find_hwmon_by_name("asus", asus_dir, sizeof(asus_dir)) != 0) {
        fprintf(stderr, "ERROR: asus hwmon not found\n");
        return -1;
    }

    const struct {
        const char *name;
        const char *dir;
    } rapl_specs[5] = {
        {"rapl_package_0_duj", "/sys/class/powercap/intel-rapl:0"},
        {"rapl_core_duj", "/sys/class/powercap/intel-rapl:0/intel-rapl:0:0"},
        {"rapl_uncore_duj", "/sys/class/powercap/intel-rapl:0/intel-rapl:0:1"},
        {"rapl_psys_duj", "/sys/class/powercap/intel-rapl:1"},
        {"rapl_mmio_0_duj", "/sys/class/powercap/intel-rapl-mmio:0"},
    };

    for (int i = 0; i < 5; i++) {
        CHANNELS[i].name = rapl_specs[i].name;
        CHANNELS[i].unit = "uJ_per_tick";
        CHANNELS[i].modality = "intel_rapl_energy";
        snprintf(CHANNELS[i].path, sizeof(CHANNELS[i].path), "%s/energy_uj", rapl_specs[i].dir);
        char mpath[280];
        snprintf(mpath, sizeof(mpath), "%s/max_energy_range_uj", rapl_specs[i].dir);
        uint64_t v = 0, mr = 0;
        if (read_u64_file(CHANNELS[i].path, &v) != 0 || read_u64_file(mpath, &mr) != 0 || mr == 0) {
            fprintf(stderr, "ERROR: cannot read RAPL domain %s\n", CHANNELS[i].path);
            return -1;
        }
        CHANNELS[i].max_range = mr;
    }

    static const char *temp_names[9] = {
        "temp_pkg_mc",
        "temp_core0_mc", "temp_core1_mc", "temp_core2_mc", "temp_core3_mc",
        "temp_core4_mc", "temp_core5_mc", "temp_core6_mc", "temp_core7_mc",
    };
    for (int i = 0; i < 9; i++) {
        int idx = 5 + i;
        CHANNELS[idx].name = temp_names[i];
        CHANNELS[idx].unit = "mC";
        CHANNELS[idx].modality = "cpu_die_thermal";
        snprintf(CHANNELS[idx].path, sizeof(CHANNELS[idx].path), "%s/temp%d_input", coretemp_dir, i + 1);
        CHANNELS[idx].max_range = 0;
        int64_t tv = 0;
        if (read_i64_file(CHANNELS[idx].path, &tv) != 0) {
            fprintf(stderr, "ERROR: cannot read %s\n", CHANNELS[idx].path);
            return -1;
        }
    }

    static const char *freq_names[4] = {
        "freq_cpu0_khz", "freq_cpu1_khz", "freq_cpu2_khz", "freq_cpu3_khz",
    };
    for (int i = 0; i < 4; i++) {
        int idx = 14 + i;
        CHANNELS[idx].name = freq_names[i];
        CHANNELS[idx].unit = "kHz";
        CHANNELS[idx].modality = "cpu_clock_freq";
        snprintf(CHANNELS[idx].path, sizeof(CHANNELS[idx].path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        CHANNELS[idx].max_range = 0;
        int64_t fv = 0;
        if (read_i64_file(CHANNELS[idx].path, &fv) != 0) {
            fprintf(stderr, "ERROR: cannot read %s\n", CHANNELS[idx].path);
            return -1;
        }
    }

    static const char *fan_names[2] = {"fan_cpu_rpm", "fan_gpu_rpm"};
    for (int i = 0; i < 2; i++) {
        int idx = 18 + i;
        CHANNELS[idx].name = fan_names[i];
        CHANNELS[idx].unit = "RPM";
        CHANNELS[idx].modality = "chassis_cooling_fan";
        snprintf(CHANNELS[idx].path, sizeof(CHANNELS[idx].path), "%s/fan%d_input", asus_dir, i + 1);
        CHANNELS[idx].max_range = 0;
        int64_t rv = 0;
        if (read_i64_file(CHANNELS[idx].path, &rv) != 0) {
            fprintf(stderr, "ERROR: cannot read %s\n", CHANNELS[idx].path);
            return -1;
        }
    }
    return 0;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void pin_to_cpu(int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
}

/* --- Inline algebraic self-verification of R1/R2/R3 step maps --- */
static inline void step_r1_bijective_low(uint64_t *a, uint64_t *b) {
    *a ^= ((*b & 1ULL) | 2ULL);
    *b ^= ((*a & 1ULL) | 2ULL);
}

static inline void inv_r1_bijective_low(uint64_t *a, uint64_t *b) {
    *b ^= ((*a & 1ULL) | 2ULL);
    *a ^= ((*b & 1ULL) | 2ULL);
}

static inline void step_r3_bijective_high(uint64_t *a, uint64_t *b) {
    *a ^= ~(*b & 1ULL);
    *b ^= ~(*a & 1ULL);
}

static inline void inv_r3_bijective_high(uint64_t *a, uint64_t *b) {
    *b ^= ~(*a & 1ULL);
    *a ^= ~(*b & 1ULL);
}

static inline void step_r2_erase_low(uint64_t *a, uint64_t *b, uint64_t ctr) {
    /* Destructive many-to-one projection (erases >= 2 bits/step) at matched 6-op low toggle */
    uint64_t t = (*a & *b) ^ (ctr & 3ULL);
    *a = t | 2ULL;
    *b = (*b & 1ULL) ^ (*a);
}

static int verify_step_algebra(void) {
    uint64_t cases[4][2] = {
        {0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL},
        {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL},
        {0ULL, 1ULL},
        {0xFFFFFFFFFFFFFFFFULL, 0ULL},
    };
    for (int i = 0; i < 4; i++) {
        uint64_t a = cases[i][0], b = cases[i][1];
        step_r1_bijective_low(&a, &b);
        inv_r1_bijective_low(&a, &b);
        if (a != cases[i][0] || b != cases[i][1]) return -1;

        a = cases[i][0]; b = cases[i][1];
        step_r3_bijective_high(&a, &b);
        inv_r3_bijective_high(&a, &b);
        if (a != cases[i][0] || b != cases[i][1]) return -1;
    }
    /* Explicit irreversibility collision for R2 */
    uint64_t a1 = 0x100ULL, b1 = 0x200ULL;
    uint64_t a2 = 0x300ULL, b2 = 0x400ULL;
    step_r2_erase_low(&a1, &b1, 1ULL);
    step_r2_erase_low(&a2, &b2, 1ULL);
    if (a1 != a2 || b1 != b2) return -1;
    return 0;
}

/* --- Worker thread state --- */
typedef struct {
    int worker_idx;
    int cpu_id;
    int regime_idx;
    _Atomic int *stop_flag;
    uint64_t *dram_next; /* 64 MiB cache-line pointer-chase ring (1M lines) */
    uint64_t ops_done;
    uint64_t erased_bits_done;
    uint64_t toggle_bits_done;
    uint64_t sink;
} worker_ctx_t;

static void init_dram_ring(uint64_t *ring, uint64_t n_lines, uint64_t seed) {
    /* Deterministic Sattolo single-cycle permutation across 64 MiB cache lines */
    uint64_t *perm = (uint64_t *)malloc(n_lines * sizeof(uint64_t));
    if (!perm) {
        fprintf(stderr, "ERROR: OOM allocating perm\n");
        exit(2);
    }
    for (uint64_t i = 0; i < n_lines; i++) perm[i] = i;
    uint64_t s = seed;
    for (uint64_t i = n_lines - 1; i > 0; i--) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        uint64_t r = (s * 2685821657736338717ULL) % i;
        uint64_t tmp = perm[i];
        perm[i] = perm[r];
        perm[r] = tmp;
    }
    /* Store next index at offset 0 of each 64-byte cache line (stride 8 uint64_t) */
    for (uint64_t i = 0; i < n_lines; i++) {
        uint64_t cur = perm[i];
        uint64_t nxt = perm[(i + 1 == n_lines) ? 0 : i + 1];
        ring[cur * 8ULL] = nxt;
        ring[cur * 8ULL + 1ULL] = cur ^ 0x5A5A5A5A5A5A5A5AULL;
    }
    free(perm);
}

static void *worker_main(void *arg) {
    worker_ctx_t *ctx = (worker_ctx_t *)arg;
    pin_to_cpu(ctx->cpu_id);
    int r = ctx->regime_idx;
    uint64_t ops = 0;
    uint64_t erased = 0;
    uint64_t toggles = 0;
    uint64_t sink = 0;

    if (r == 0) {
        /* R0_IDLE */
        struct timespec ts = {0, 10000000L}; /* 10 ms sleep */
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            nanosleep(&ts, NULL);
        }
    } else if (r == 1) {
        /* R1_BIJECTIVE_FEISTEL_LOW_H: 4 independent register pairs, 0 erased bits, ~3 bits/step */
        uint64_t a0 = 0x10ULL + (uint64_t)ctx->worker_idx, b0 = 0x21ULL;
        uint64_t a1 = 0x30ULL + (uint64_t)ctx->worker_idx, b1 = 0x41ULL;
        uint64_t a2 = 0x50ULL + (uint64_t)ctx->worker_idx, b2 = 0x61ULL;
        uint64_t a3 = 0x70ULL + (uint64_t)ctx->worker_idx, b3 = 0x81ULL;
        const uint64_t BATCH = 4096ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                step_r1_bijective_low(&a0, &b0);
                step_r1_bijective_low(&a1, &b1);
                step_r1_bijective_low(&a2, &b2);
                step_r1_bijective_low(&a3, &b3);
                __asm__ volatile("" : "+r"(a0), "+r"(b0), "+r"(a1), "+r"(b1),
                                      "+r"(a2), "+r"(b2), "+r"(a3), "+r"(b3));
            }
            ops += BATCH * 4ULL;
            toggles += BATCH * 4ULL * 3ULL;
        }
        sink = a0 ^ b0 ^ a1 ^ b1 ^ a2 ^ b2 ^ a3 ^ b3;
    } else if (r == 2) {
        /* R2_ERASE_DESTRUCTIVE_LOW_H: 4 independent register pairs, >=2 erased bits/step, ~3 bits/step */
        uint64_t a0 = 0x10ULL + (uint64_t)ctx->worker_idx, b0 = 0x21ULL;
        uint64_t a1 = 0x30ULL + (uint64_t)ctx->worker_idx, b1 = 0x41ULL;
        uint64_t a2 = 0x50ULL + (uint64_t)ctx->worker_idx, b2 = 0x61ULL;
        uint64_t a3 = 0x70ULL + (uint64_t)ctx->worker_idx, b3 = 0x81ULL;
        uint64_t ctr = 1ULL;
        const uint64_t BATCH = 4096ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                step_r2_erase_low(&a0, &b0, ctr);
                step_r2_erase_low(&a1, &b1, ctr + 1ULL);
                step_r2_erase_low(&a2, &b2, ctr + 2ULL);
                step_r2_erase_low(&a3, &b3, ctr + 3ULL);
                ctr++;
                __asm__ volatile("" : "+r"(a0), "+r"(b0), "+r"(a1), "+r"(b1),
                                      "+r"(a2), "+r"(b2), "+r"(a3), "+r"(b3));
            }
            ops += BATCH * 4ULL;
            erased += BATCH * 4ULL * 2ULL;
            toggles += BATCH * 4ULL * 3ULL;
        }
        sink = a0 ^ b0 ^ a1 ^ b1 ^ a2 ^ b2 ^ a3 ^ b3 ^ ctr;
    } else if (r == 3) {
        /* R3_BIJECTIVE_FEISTEL_HIGH_H: exact bijection, 0 erased bits, 127 bits/step */
        uint64_t a0 = 0x5555555555555550ULL + (uint64_t)ctx->worker_idx, b0 = 0xAAAAAAAAAAAAAAAAULL;
        uint64_t a1 = 0x5555555555555551ULL + (uint64_t)ctx->worker_idx, b1 = 0xAAAAAAAAAAAAAAABULL;
        uint64_t a2 = 0x5555555555555552ULL + (uint64_t)ctx->worker_idx, b2 = 0xAAAAAAAAAAAAAAACULL;
        uint64_t a3 = 0x5555555555555553ULL + (uint64_t)ctx->worker_idx, b3 = 0xAAAAAAAAAAAAAAADULL;
        const uint64_t BATCH = 4096ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                step_r3_bijective_high(&a0, &b0);
                step_r3_bijective_high(&a1, &b1);
                step_r3_bijective_high(&a2, &b2);
                step_r3_bijective_high(&a3, &b3);
                __asm__ volatile("" : "+r"(a0), "+r"(b0), "+r"(a1), "+r"(b1),
                                      "+r"(a2), "+r"(b2), "+r"(a3), "+r"(b3));
            }
            ops += BATCH * 4ULL;
            toggles += BATCH * 4ULL * 127ULL;
        }
        sink = a0 ^ b0 ^ a1 ^ b1 ^ a2 ^ b2 ^ a3 ^ b3;
    } else if (r == 4) {
        /* R4_AVX512_FMA_L1: 8 independent 512-bit FP64 FMA accumulators (128 FLOPs/iter) */
        __m512d v0 = _mm512_set1_pd(1.0000001);
        __m512d v1 = _mm512_set1_pd(1.0000002);
        __m512d v2 = _mm512_set1_pd(1.0000003);
        __m512d v3 = _mm512_set1_pd(1.0000004);
        __m512d v4 = _mm512_set1_pd(1.0000005);
        __m512d v5 = _mm512_set1_pd(1.0000006);
        __m512d v6 = _mm512_set1_pd(1.0000007);
        __m512d v7 = _mm512_set1_pd(1.0000008);
        __m512d mul = _mm512_set1_pd(0.999999999);
        __m512d add = _mm512_set1_pd(1e-9);
        const uint64_t BATCH = 2048ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                v0 = _mm512_fmadd_pd(v0, mul, add);
                v1 = _mm512_fmadd_pd(v1, mul, add);
                v2 = _mm512_fmadd_pd(v2, mul, add);
                v3 = _mm512_fmadd_pd(v3, mul, add);
                v4 = _mm512_fmadd_pd(v4, mul, add);
                v5 = _mm512_fmadd_pd(v5, mul, add);
                v6 = _mm512_fmadd_pd(v6, mul, add);
                v7 = _mm512_fmadd_pd(v7, mul, add);
                __asm__ volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3),
                                      "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7));
            }
            /* Count exact scalar FP64 FLOPs: 8 regs * 8 lanes * 2 FLOPs = 128 FLOPs/iter */
            ops += BATCH * 128ULL;
        }
        __m512d acc = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3)),
                                    _mm512_add_pd(_mm512_add_pd(v4, v5), _mm512_add_pd(v6, v7)));
        double out[8];
        _mm512_storeu_pd(out, acc);
        uint64_t bits = 0;
        memcpy(&bits, &out[0], sizeof(bits));
        sink = bits;
    } else if (r == 5) {
        /* R5_AVX512_VNNI_L1: 8 independent 512-bit INT8 vpdpbusd accumulators (1024 INT8 ops/iter) */
        __m512i c0 = _mm512_set1_epi32(1);
        __m512i c1 = _mm512_set1_epi32(2);
        __m512i c2 = _mm512_set1_epi32(3);
        __m512i c3 = _mm512_set1_epi32(4);
        __m512i c4 = _mm512_set1_epi32(5);
        __m512i c5 = _mm512_set1_epi32(6);
        __m512i c6 = _mm512_set1_epi32(7);
        __m512i c7 = _mm512_set1_epi32(8);
        __m512i u8 = _mm512_set1_epi8(0x55);
        __m512i s8 = _mm512_set1_epi8(0x2A);
        const uint64_t BATCH = 2048ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                c0 = _mm512_dpbusd_epi32(c0, u8, s8);
                c1 = _mm512_dpbusd_epi32(c1, u8, s8);
                c2 = _mm512_dpbusd_epi32(c2, u8, s8);
                c3 = _mm512_dpbusd_epi32(c3, u8, s8);
                c4 = _mm512_dpbusd_epi32(c4, u8, s8);
                c5 = _mm512_dpbusd_epi32(c5, u8, s8);
                c6 = _mm512_dpbusd_epi32(c6, u8, s8);
                c7 = _mm512_dpbusd_epi32(c7, u8, s8);
                __asm__ volatile("" : "+v"(c0), "+v"(c1), "+v"(c2), "+v"(c3),
                                      "+v"(c4), "+v"(c5), "+v"(c6), "+v"(c7));
            }
            /* 8 regs * 16 lanes * 4 MACs * 2 ops = 1024 INT8 ops/iter */
            ops += BATCH * 1024ULL;
        }
        __m512i acc = _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(c0, c1), _mm512_xor_si512(c2, c3)),
                                       _mm512_xor_si512(_mm512_xor_si512(c4, c5), _mm512_xor_si512(c6, c7)));
        uint64_t out[8];
        _mm512_storeu_si512((void *)out, acc);
        sink = out[0] ^ out[7];
    } else if (r == 6) {
        /* R6_DRAM_STREAM_L3_MISS: 4-way independent pointer-chase + write across 64 MiB/core (256 MiB total) */
        uint64_t *ring = ctx->dram_next;
        uint64_t p0 = 0ULL;
        uint64_t p1 = DRAM_LINES_PER_WORKER / 4ULL;
        uint64_t p2 = DRAM_LINES_PER_WORKER / 2ULL;
        uint64_t p3 = (3ULL * DRAM_LINES_PER_WORKER) / 4ULL;
        uint64_t acc = 0ULL;
        const uint64_t BATCH = 512ULL;
        while (!atomic_load_explicit(ctx->stop_flag, memory_order_relaxed)) {
            for (uint64_t k = 0; k < BATCH; k++) {
                uint64_t n0 = ring[p0 * 8ULL];
                uint64_t n1 = ring[p1 * 8ULL];
                uint64_t n2 = ring[p2 * 8ULL];
                uint64_t n3 = ring[p3 * 8ULL];
                ring[p0 * 8ULL + 1ULL] ^= n1;
                ring[p1 * 8ULL + 1ULL] ^= n2;
                ring[p2 * 8ULL + 1ULL] ^= n3;
                ring[p3 * 8ULL + 1ULL] ^= n0;
                acc ^= (n0 ^ n1 ^ n2 ^ n3);
                p0 = n0; p1 = n1; p2 = n2; p3 = n3;
            }
            /* Count cache lines touched (64 B RMW per miss) */
            ops += BATCH * 4ULL;
        }
        sink = acc ^ p0 ^ p1 ^ p2 ^ p3;
    }

    ctx->ops_done = ops;
    ctx->erased_bits_done = erased;
    ctx->toggle_bits_done = toggles;
    ctx->sink = sink;
    return NULL;
}

typedef struct {
    int regime_idx;
    uint64_t t_rel_us;
    uint64_t dt_us;
    int64_t vals[NUM_CHANNELS];
} sample_row_t;

typedef struct {
    uint64_t wall_ns;
    uint64_t total_ops;
    uint64_t total_erased_bits;
    uint64_t total_toggle_bits;
    uint64_t sink_xor;
    int n_samples;
} regime_summary_t;

int main(int argc, char **argv) {
    int samples_per_regime = 100; /* 100 * 50ms = 5.0 s per regime */
    int tick_ms = 50;
    int settle_ms = 1000;
    const char *out_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples_per_regime = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tick-ms") == 0 && i + 1 < argc) {
            tick_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--settle-ms") == 0 && i + 1 < argc) {
            settle_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s [--samples N] [--tick-ms MS] [--settle-ms MS] [--output PATH]\n", argv[0]);
            return 2;
        }
    }
    if (samples_per_regime <= 0 || samples_per_regime * NUM_REGIMES > MAX_SAMPLES || tick_ms <= 0 || settle_ms < 0) {
        fprintf(stderr, "ERROR: invalid timing args\n");
        return 2;
    }
    if (verify_step_algebra() != 0) {
        fprintf(stderr, "ERROR: inline step algebra verification failed\n");
        return 3;
    }
    if (init_channels() != 0) {
        return 4;
    }

    pin_to_cpu(SAMPLER_CPU);

    /* Allocate & initialize 64 MiB DRAM pointer-chase buffers BEFORE capture starts */
    uint64_t *dram_bufs[NUM_WORKERS];
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (posix_memalign((void **)&dram_bufs[w], 4096, DRAM_BYTES_PER_WORKER) != 0) {
            fprintf(stderr, "ERROR: posix_memalign failed\n");
            return 5;
        }
        init_dram_ring(dram_bufs[w], DRAM_LINES_PER_WORKER, 0x9E3779B97F4A7C15ULL + (uint64_t)w * 1315423911ULL);
    }

    sample_row_t *rows = (sample_row_t *)calloc((size_t)(samples_per_regime * NUM_REGIMES), sizeof(sample_row_t));
    if (!rows) {
        fprintf(stderr, "ERROR: OOM rows\n");
        return 6;
    }
    regime_summary_t summaries[NUM_REGIMES];
    memset(summaries, 0, sizeof(summaries));

    int total_rows = 0;
    uint64_t t0_ns = now_ns();
    uint64_t prev_rapl[5] = {0};

    /* Balanced palindromic ABBA schedule (forward R0..R6 then reverse R6..R0) cancels linear thermal/PL1 drift */
    static const int SCHEDULE[NUM_REGIMES * 2] = {0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0};
    int per_block_first = samples_per_regime / 2;
    int per_block_second = samples_per_regime - per_block_first;

    for (int blk = 0; blk < NUM_REGIMES * 2; blk++) {
        int r = SCHEDULE[blk];
        int blk_samples = (blk < NUM_REGIMES) ? per_block_first : per_block_second;
        if (blk_samples <= 0) continue;

        if (blk > 0 && settle_ms > 0) {
            struct timespec sts = {settle_ms / 1000, (settle_ms % 1000) * 1000000L};
            nanosleep(&sts, NULL);
        }

        _Atomic int stop_flag = 0;
        pthread_t th[NUM_WORKERS];
        worker_ctx_t ctxs[NUM_WORKERS];
        for (int w = 0; w < NUM_WORKERS; w++) {
            ctxs[w].worker_idx = w;
            ctxs[w].cpu_id = w; /* Physical cores 0, 1, 2, 3 */
            ctxs[w].regime_idx = r;
            ctxs[w].stop_flag = &stop_flag;
            ctxs[w].dram_next = dram_bufs[w];
            ctxs[w].ops_done = 0;
            ctxs[w].erased_bits_done = 0;
            ctxs[w].toggle_bits_done = 0;
            ctxs[w].sink = 0;
        }

        uint64_t r_start_ns = now_ns();
        for (int w = 0; w < NUM_WORKERS; w++) {
            if (pthread_create(&th[w], NULL, worker_main, &ctxs[w]) != 0) {
                fprintf(stderr, "ERROR: pthread_create failed\n");
                return 7;
            }
        }

        /* Brief 50 ms warmup inside regime so pipeline & RAPL counters stabilize */
        struct timespec wts = {0, (long)tick_ms * 1000000L};
        nanosleep(&wts, NULL);

        uint64_t prev_t = now_ns();
        for (int c = 0; c < 5; c++) {
            if (read_u64_file(CHANNELS[c].path, &prev_rapl[c]) != 0) return 8;
        }

        for (int s = 0; s < blk_samples; s++) {
            struct timespec tick_ts = {tick_ms / 1000, (tick_ms % 1000) * 1000000L};
            nanosleep(&tick_ts, NULL);
            uint64_t cur_t = now_ns();
            sample_row_t *row = &rows[total_rows++];
            row->regime_idx = r;
            row->t_rel_us = (cur_t - t0_ns) / 1000ULL;
            row->dt_us = (cur_t - prev_t) / 1000ULL;
            if (row->dt_us == 0) row->dt_us = 1;
            prev_t = cur_t;

            for (int c = 0; c < 5; c++) {
                uint64_t cur_e = 0;
                if (read_u64_file(CHANNELS[c].path, &cur_e) != 0) return 9;
                uint64_t diff = (cur_e >= prev_rapl[c])
                                    ? (cur_e - prev_rapl[c])
                                    : (cur_e + CHANNELS[c].max_range - prev_rapl[c]);
                prev_rapl[c] = cur_e;
                row->vals[c] = (int64_t)diff;
            }
            for (int c = 5; c < NUM_CHANNELS; c++) {
                int64_t iv = 0;
                if (read_i64_file(CHANNELS[c].path, &iv) != 0) return 10;
                row->vals[c] = iv;
            }
        }

        atomic_store_explicit(&stop_flag, 1, memory_order_release);
        for (int w = 0; w < NUM_WORKERS; w++) {
            pthread_join(th[w], NULL);
            summaries[r].total_ops += ctxs[w].ops_done;
            summaries[r].total_erased_bits += ctxs[w].erased_bits_done;
            summaries[r].total_toggle_bits += ctxs[w].toggle_bits_done;
            summaries[r].sink_xor ^= ctxs[w].sink;
        }
        summaries[r].wall_ns += (now_ns() - r_start_ns);
        summaries[r].n_samples += blk_samples;
    }

    FILE *fp = stdout;
    if (out_path != NULL) {
        fp = fopen(out_path, "wb");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot open output %s\n", out_path);
            return 11;
        }
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"harness_version\": \"rapl_landauer_microkernels/v1\",\n");
    fprintf(fp, "  \"num_workers\": %d,\n", NUM_WORKERS);
    fprintf(fp, "  \"worker_cpus\": [0, 1, 2, 3],\n");
    fprintf(fp, "  \"sampler_cpu\": %d,\n", SAMPLER_CPU);
    fprintf(fp, "  \"tick_ms\": %d,\n", tick_ms);
    fprintf(fp, "  \"settle_ms\": %d,\n", settle_ms);
    fprintf(fp, "  \"samples_per_regime\": %d,\n", samples_per_regime);
    fprintf(fp, "  \"dram_bytes_per_worker\": %" PRIu64 ",\n", (uint64_t)DRAM_BYTES_PER_WORKER);
    fprintf(fp, "  \"dram_total_bytes\": %" PRIu64 ",\n", (uint64_t)(DRAM_BYTES_PER_WORKER * NUM_WORKERS));
    fprintf(fp, "  \"step_algebra_self_checked\": true,\n");
    fprintf(fp, "  \"channels\": [\n");
    for (int c = 0; c < NUM_CHANNELS; c++) {
        fprintf(fp, "    {\"idx\": %d, \"name\": \"%s\", \"unit\": \"%s\", \"modality\": \"%s\", \"max_energy_range_uj\": %" PRIu64 "}%s\n",
                c, CHANNELS[c].name, CHANNELS[c].unit, CHANNELS[c].modality,
                CHANNELS[c].max_range, (c + 1 < NUM_CHANNELS) ? "," : "");
    }
    fprintf(fp, "  ],\n");
    fprintf(fp, "  \"regimes\": [\n");
    for (int r = 0; r < NUM_REGIMES; r++) {
        fprintf(fp, "    {\"idx\": %d, \"id\": \"%s\", \"category\": \"%s\", \"n_samples\": %d, \"wall_ns\": %" PRIu64 ", \"total_ops\": %" PRIu64 ", \"total_erased_bits\": %" PRIu64 ", \"total_toggle_bits\": %" PRIu64 ", \"sink_xor_hex\": \"%016" PRIx64 "\"}%s\n",
                r, REGIMES[r].id, REGIMES[r].category, summaries[r].n_samples,
                summaries[r].wall_ns, summaries[r].total_ops,
                summaries[r].total_erased_bits, summaries[r].total_toggle_bits,
                summaries[r].sink_xor, (r + 1 < NUM_REGIMES) ? "," : "");
    }
    fprintf(fp, "  ],\n");
    fprintf(fp, "  \"regime_labels\": [");
    for (int i = 0; i < total_rows; i++) {
        fprintf(fp, "%d%s", rows[i].regime_idx, (i + 1 < total_rows) ? "," : "");
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"t_rel_us\": [");
    for (int i = 0; i < total_rows; i++) {
        fprintf(fp, "%" PRIu64 "%s", rows[i].t_rel_us, (i + 1 < total_rows) ? "," : "");
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"dt_us\": [");
    for (int i = 0; i < total_rows; i++) {
        fprintf(fp, "%" PRIu64 "%s", rows[i].dt_us, (i + 1 < total_rows) ? "," : "");
    }
    fprintf(fp, "],\n");
    fprintf(fp, "  \"X_anonymous\": [\n");
    for (int i = 0; i < total_rows; i++) {
        fprintf(fp, "    [");
        for (int c = 0; c < NUM_CHANNELS; c++) {
            fprintf(fp, "%" PRId64 "%s", rows[i].vals[c], (c + 1 < NUM_CHANNELS) ? "," : "");
        }
        fprintf(fp, "]%s\n", (i + 1 < total_rows) ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    if (fp != stdout) fclose(fp);
    free(rows);
    for (int w = 0; w < NUM_WORKERS; w++) free(dram_bufs[w]);
    return 0;
}
