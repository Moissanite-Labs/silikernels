/*
 * Granular Live Silicon Micro-Probe:
 * Silicon-to-Universal Exact-Operator Correspondence & Obstruction Atlas.
 *
 * Strictly zero microphone, zero camera, zero display access.
 * Captures pure integer hardware observables directly on Intel i9-11900H:
 *   1. High-rate serialized RDTSC cycle latencies (131,072 samples x 3 regimes:
 *      L1 16 KiB quiescent, L3 16 MiB boundary, DRAM 64 MiB pointer-chase)
 *      with exact quantiles, power-of-2 log-bin histogram, and burst avalanches.
 *   2. 100 Hz (10 ms tick, 1,600 ticks = 16.0 s) synchronized 5x intel-rapl +
 *      coretemp + asus CPU/GPU fan RPM + CPU frequency time-series across
 *      quiescent -> ascending ramp -> peak/descending ramp -> passive relaxation.
 *   3. Allometric physical-core scaling sweep (p = 1, 2, 4, 6, 8 pinned distinct
 *      physical cores 0..p-1, 2.0 s each) measuring exact retired ops, 5x RAPL
 *      microjoules, active clock kHz, package temperature mC, and CPU fan RPM.
 *
 * Emits strict pure-integer JSON (zero floating-point values!).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static inline uint64_t rdtsc_serialized(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

static inline int64_t now_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static int read_int64_file(const char *path, int64_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[64];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char *end = NULL;
    long long v = strtoll(buf, &end, 10);
    if (end == buf) return -1;
    *out = (int64_t)v;
    return 0;
}

static int read_str_trim(const char *path, char *dst, size_t cap) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, dst, cap - 1);
    close(fd);
    if (n <= 0) return -1;
    dst[n] = '\0';
    while (n > 0 && (dst[n - 1] == '\n' || dst[n - 1] == '\r' || dst[n - 1] == ' ')) {
        dst[--n] = '\0';
    }
    return 0;
}

typedef struct {
    char rapl_path[5][128];
    int64_t rapl_max[5];
    int rapl_ok[5];
    char coretemp_pkg[320];
    char coretemp_c0[320];
    char coretemp_c1[320];
    char coretemp_c7[320];
    char fan_cpu[320];
    char fan_gpu[320];
} SensorPaths;

static void discover_sensors(SensorPaths *sp) {
    memset(sp, 0, sizeof(*sp));
    const char *rapl_dirs[5] = {
        "/sys/class/powercap/intel-rapl:0",
        "/sys/class/powercap/intel-rapl:0:0",
        "/sys/class/powercap/intel-rapl:0:1",
        "/sys/class/powercap/intel-rapl:1",
        "/sys/class/powercap/intel-rapl-mmio:0",
    };
    for (int i = 0; i < 5; ++i) {
        char mpath[160];
        snprintf(sp->rapl_path[i], sizeof(sp->rapl_path[i]), "%s/energy_uj", rapl_dirs[i]);
        snprintf(mpath, sizeof(mpath), "%s/max_energy_range_uj", rapl_dirs[i]);
        int64_t dummy = 0;
        if (read_int64_file(sp->rapl_path[i], &dummy) == 0 &&
            read_int64_file(mpath, &sp->rapl_max[i]) == 0) {
            sp->rapl_ok[i] = 1;
        }
    }

    DIR *d = opendir("/sys/class/hwmon");
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strncmp(ent->d_name, "hwmon", 5) != 0) continue;
        char npath[320], name[64];
        snprintf(npath, sizeof(npath), "/sys/class/hwmon/%s/name", ent->d_name);
        if (read_str_trim(npath, name, sizeof(name)) != 0) continue;
        if (strcmp(name, "coretemp") == 0) {
            snprintf(sp->coretemp_pkg, sizeof(sp->coretemp_pkg), "/sys/class/hwmon/%s/temp1_input", ent->d_name);
            snprintf(sp->coretemp_c0, sizeof(sp->coretemp_c0), "/sys/class/hwmon/%s/temp2_input", ent->d_name);
            snprintf(sp->coretemp_c1, sizeof(sp->coretemp_c1), "/sys/class/hwmon/%s/temp3_input", ent->d_name);
            snprintf(sp->coretemp_c7, sizeof(sp->coretemp_c7), "/sys/class/hwmon/%s/temp9_input", ent->d_name);
        } else if (strcmp(name, "asus") == 0) {
            snprintf(sp->fan_cpu, sizeof(sp->fan_cpu), "/sys/class/hwmon/%s/fan1_input", ent->d_name);
            snprintf(sp->fan_gpu, sizeof(sp->fan_gpu), "/sys/class/hwmon/%s/fan2_input", ent->d_name);
        }
    }
    closedir(d);
}

static void read_rapl_all(const SensorPaths *sp, int64_t out[5]) {
    for (int i = 0; i < 5; ++i) {
        out[i] = 0;
        if (sp->rapl_ok[i]) {
            read_int64_file(sp->rapl_path[i], &out[i]);
        }
    }
}

static int64_t rapl_diff(int64_t a, int64_t b, int64_t max_r) {
    if (b >= a) return b - a;
    return (max_r - a) + b;
}

static int64_t read_cpu_freq_khz(int cpu_id) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu_id);
    int64_t v = 0;
    if (read_int64_file(path, &v) != 0) return 0;
    return v;
}

/* =========================================================================
 * Experiment 1: Serialized RDTSC Micro-Jitter & Avalanche Bursts
 * ========================================================================= */

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

typedef struct {
    const char *name;
    size_t bytes;
    int steps_per_sample;
    int n_samples;
    uint64_t min_cyc, p10_cyc, p50_cyc, p90_cyc, p95_cyc, p99_cyc, p999_cyc, max_cyc;
    uint64_t total_cyc;
    uint64_t log2_hist[24];
    int n_avalanches_p95;
    int max_avalanche_len;
    uint64_t total_avalanche_excess_cyc;
    uint64_t max_avalanche_excess_cyc;
    uint64_t coarse_256[256]; /* 512-sample block sums -> exact 256-point series */
} JitterResult;

static void run_jitter_regime(const char *name, size_t bytes, int steps, int n_samples, JitterResult *res) {
    memset(res, 0, sizeof(*res));
    res->name = name;
    res->bytes = bytes;
    res->steps_per_sample = steps;
    res->n_samples = n_samples;

    /* Pin to physical core 0 */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);

    /* Allocate cache-line strided (64B) permutation cycle (Sattolo) defeating HW prefetcher */
    size_t n_nodes = bytes / 64u;
    if (n_nodes < 64u) n_nodes = 64u;
    uint32_t *buf = NULL;
    if (posix_memalign((void **)&buf, 64, n_nodes * 64u) != 0) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    uint32_t *perm = (uint32_t *)malloc(n_nodes * sizeof(uint32_t));
    for (size_t i = 0; i < n_nodes; ++i) perm[i] = (uint32_t)i;

    /* Deterministic xorshift64* Sattolo single-cycle permutation */
    uint64_t st = 0x564A1E7E10511C0FULL ^ (uint64_t)bytes;
    for (size_t i = n_nodes - 1; i > 0; --i) {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        uint64_t r = (st * 2685821657736338717ULL) % i;
        uint32_t tmp = perm[i];
        perm[i] = perm[r];
        perm[r] = tmp;
    }
    for (size_t i = 0; i < n_nodes; ++i) {
        buf[perm[i] * 16u] = perm[(i + 1) % n_nodes];
    }
    free(perm);

    uint64_t *samples = (uint64_t *)malloc((size_t)n_samples * sizeof(uint64_t));
    uint32_t cur = 0;

    /* Warmup */
    for (int w = 0; w < 2048; ++w) {
        cur = buf[cur * 16u];
    }

    int block_sz = n_samples / 256;
    for (int i = 0; i < n_samples; ++i) {
        uint64_t t0 = rdtsc_serialized();
        for (int s = 0; s < steps; ++s) {
            cur = buf[cur * 16u];
        }
        uint64_t t1 = rdtsc_serialized();
        uint64_t dt = (t1 >= t0) ? (t1 - t0) : 0ULL;
        samples[i] = dt;
        res->total_cyc += dt;
        int b = 0;
        uint64_t tmp = dt;
        while (tmp > 0 && b < 23) {
            tmp >>= 1;
            b++;
        }
        res->log2_hist[b]++;
        int blk = i / block_sz;
        if (blk < 256) res->coarse_256[blk] += dt;
    }
    /* Prevent compiler from eliminating pointer chase */
    volatile uint32_t sink = cur;
    (void)sink;

    uint64_t *sorted = (uint64_t *)malloc((size_t)n_samples * sizeof(uint64_t));
    memcpy(sorted, samples, (size_t)n_samples * sizeof(uint64_t));
    qsort(sorted, (size_t)n_samples, sizeof(uint64_t), cmp_u64);

    res->min_cyc = sorted[0];
    res->p10_cyc = sorted[(size_t)n_samples * 10 / 100];
    res->p50_cyc = sorted[(size_t)n_samples * 50 / 100];
    res->p90_cyc = sorted[(size_t)n_samples * 90 / 100];
    res->p95_cyc = sorted[(size_t)n_samples * 95 / 100];
    res->p99_cyc = sorted[(size_t)n_samples * 99 / 100];
    res->p999_cyc = sorted[(size_t)n_samples * 999 / 1000];
    res->max_cyc = sorted[n_samples - 1];

    /* Avalanche bursts: contiguous runs strictly above p95 threshold */
    uint64_t thresh = res->p95_cyc;
    int in_burst = 0;
    int cur_len = 0;
    uint64_t cur_excess = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (samples[i] > thresh) {
            if (!in_burst) {
                in_burst = 1;
                cur_len = 0;
                cur_excess = 0;
                res->n_avalanches_p95++;
            }
            cur_len++;
            uint64_t ex = samples[i] - thresh;
            cur_excess += ex;
            res->total_avalanche_excess_cyc += ex;
            if (cur_len > res->max_avalanche_len) res->max_avalanche_len = cur_len;
            if (cur_excess > res->max_avalanche_excess_cyc) res->max_avalanche_excess_cyc = cur_excess;
        } else {
            in_burst = 0;
        }
    }

    free(sorted);
    free(samples);
    free(buf);
}

/* =========================================================================
 * Worker Threads for Controlled Load (Exp 2 & Exp 3)
 * ========================================================================= */

static volatile int g_active_cores = 0;
static volatile int g_stop_workers = 0;
static volatile int g_count_ops = 0;
static uint64_t g_worker_ops[8];

static void *burn_worker(void *arg) {
    int cid = *(int *)arg;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cid, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);

    uint64_t a = 0x9E3779B97F4A7C15ULL + (uint64_t)cid;
    uint64_t b = 0xBF58476D1CE4E5B9ULL ^ (uint64_t)cid;
    uint64_t c = 0x94D049BB133111EBULL + (uint64_t)(cid * 7);
    uint64_t d = 0xD6E8FEB86659FD93ULL + (uint64_t)(cid * 13);

    while (!__atomic_load_n(&g_stop_workers, __ATOMIC_RELAXED)) {
        int lim = __atomic_load_n(&g_active_cores, __ATOMIC_RELAXED);
        if (cid < lim) {
            /* 1024 deterministic integer ALU ops per batch (64 iterations x 16 ops) */
            for (int k = 0; k < 64; ++k) {
                a = a * 6364136223846793005ULL + b;
                b = (b ^ (a >> 29)) + c;
                c = c * 2862933555777941757ULL + d;
                d = (d ^ (c >> 31)) + a;
                a += (b ^ c);
                b += (c ^ d);
                c += (d ^ a);
                d += (a ^ b);
            }
            if (__atomic_load_n(&g_count_ops, __ATOMIC_RELAXED)) {
                g_worker_ops[cid] += 1024ULL;
            }
        } else {
            struct timespec sl = {0, 1000000}; /* 1 ms sleep when inactive */
            nanosleep(&sl, NULL);
        }
    }
    volatile uint64_t sink = a ^ b ^ c ^ d;
    (void)sink;
    return NULL;
}

typedef struct {
    int64_t dt_us;
    int active_cores;
    int64_t rapl_uj[5];
    int64_t pkg_mc;
    int64_t c0_mc;
    int64_t c1_mc;
    int64_t c7_mc;
    int64_t cpu_fan_rpm;
    int64_t gpu_fan_rpm;
    int64_t cpu0_khz;
} TickSample;

typedef struct {
    int p_cores;
    int64_t wall_us;
    uint64_t total_ops;
    int64_t rapl_uj[5];
    int64_t mean_active_khz;
    int64_t pkg_mc_start;
    int64_t pkg_mc_end;
    int64_t cpu_fan_rpm_end;
} AlloSample;

int main(int argc, char **argv) {
    const char *out_path = (argc > 1) ? argv[1] : "granular_silicon_capture.json";

    SensorPaths sp;
    discover_sensors(&sp);

    /* Experiment 1: 3 x 131,072 RDTSC latency regimes */
    JitterResult jr[3];
    run_jitter_regime("l1_16kib_quiescent", 16u * 1024u, 64, 131072, &jr[0]);
    run_jitter_regime("l3_16mib_boundary",  16u * 1024u * 1024u, 16, 131072, &jr[1]);
    run_jitter_regime("dram_64mib_chase",   64u * 1024u * 1024u, 16, 131072, &jr[2]);

    /* Spawn 8 pinned worker threads on physical cores 0..7 (idle initially) */
    pthread_t th[8];
    int cids[8];
    __atomic_store_n(&g_active_cores, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_stop_workers, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_count_ops, 0, __ATOMIC_RELEASE);
    for (int i = 0; i < 8; ++i) {
        cids[i] = i;
        g_worker_ops[i] = 0;
        pthread_create(&th[i], NULL, burn_worker, &cids[i]);
    }

    /* Let system settle 0.5 s */
    struct timespec settle = {0, 500000000};
    nanosleep(&settle, NULL);

    /* Experiment 2: 1,600 ticks @ 100 Hz (10 ms) = 16.0 s multi-timescale + hysteresis loop
     *   0..399    (4.0 s): quiescent (0 cores)
     *   400..599  (2.0 s): ramp up 2 cores
     *   600..799  (2.0 s): ramp up 8 cores (peak heating)
     *   800..999  (2.0 s): sustained 8 cores
     *   1000..1199(2.0 s): ramp down 2 cores (symmetric mirror!)
     *   1200..1599(4.0 s): passive RC cooling & fan relaxation (0 cores)
     */
    const int N_TICKS = 1600;
    TickSample *ticks = (TickSample *)calloc((size_t)N_TICKS, sizeof(TickSample));
    int64_t prev_rapl[5], cur_rapl[5];
    read_rapl_all(&sp, prev_rapl);
    int64_t prev_ns = now_monotonic_ns();
    struct timespec next_ts;
    clock_gettime(CLOCK_MONOTONIC, &next_ts);

    for (int t = 0; t < N_TICKS; ++t) {
        int want = 0;
        if (t >= 400 && t < 600) want = 2;
        else if (t >= 600 && t < 1000) want = 8;
        else if (t >= 1000 && t < 1200) want = 2;
        else want = 0;
        __atomic_store_n(&g_active_cores, want, __ATOMIC_RELEASE);

        next_ts.tv_nsec += 10000000L; /* +10 ms */
        if (next_ts.tv_nsec >= 1000000000L) {
            next_ts.tv_sec += 1;
            next_ts.tv_nsec -= 1000000000L;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_ts, NULL);

        int64_t cur_ns = now_monotonic_ns();
        read_rapl_all(&sp, cur_rapl);
        ticks[t].dt_us = (cur_ns - prev_ns) / 1000LL;
        ticks[t].active_cores = want;
        for (int r = 0; r < 5; ++r) {
            ticks[t].rapl_uj[r] = rapl_diff(prev_rapl[r], cur_rapl[r], sp.rapl_max[r]);
            prev_rapl[r] = cur_rapl[r];
        }
        prev_ns = cur_ns;
        read_int64_file(sp.coretemp_pkg, &ticks[t].pkg_mc);
        read_int64_file(sp.coretemp_c0, &ticks[t].c0_mc);
        read_int64_file(sp.coretemp_c1, &ticks[t].c1_mc);
        read_int64_file(sp.coretemp_c7, &ticks[t].c7_mc);
        read_int64_file(sp.fan_cpu, &ticks[t].cpu_fan_rpm);
        read_int64_file(sp.fan_gpu, &ticks[t].gpu_fan_rpm);
        ticks[t].cpu0_khz = read_cpu_freq_khz(0);
    }
    __atomic_store_n(&g_active_cores, 0, __ATOMIC_RELEASE);
    nanosleep(&settle, NULL);

    /* Experiment 3: Allometric physical-core scaling sweep p in {1, 2, 4, 6, 8} (2.0 s each) */
    const int p_levels[5] = {1, 2, 4, 6, 8};
    AlloSample allo[5];
    memset(allo, 0, sizeof(allo));

    for (int idx = 0; idx < 5; ++idx) {
        int p = p_levels[idx];
        allo[idx].p_cores = p;
        for (int c = 0; c < 8; ++c) g_worker_ops[c] = 0;
        __atomic_store_n(&g_count_ops, 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_active_cores, p, __ATOMIC_RELEASE);
        /* 200 ms ramp settle into steady frequency */
        struct timespec warm = {0, 200000000};
        nanosleep(&warm, NULL);

        read_int64_file(sp.coretemp_pkg, &allo[idx].pkg_mc_start);
        int64_t r0[5], r1[5];
        read_rapl_all(&sp, r0);
        int64_t t0 = now_monotonic_ns();
        __atomic_store_n(&g_count_ops, 1, __ATOMIC_RELEASE);

        /* Sample active core frequencies 20 times over 2.0 s (every 100 ms) */
        int64_t freq_sum = 0;
        int freq_cnt = 0;
        for (int s = 0; s < 20; ++s) {
            struct timespec step_ts = {0, 100000000};
            nanosleep(&step_ts, NULL);
            for (int c = 0; c < p; ++c) {
                freq_sum += read_cpu_freq_khz(c);
                freq_cnt++;
            }
        }

        __atomic_store_n(&g_count_ops, 0, __ATOMIC_RELEASE);
        int64_t t1 = now_monotonic_ns();
        read_rapl_all(&sp, r1);
        read_int64_file(sp.coretemp_pkg, &allo[idx].pkg_mc_end);
        read_int64_file(sp.fan_cpu, &allo[idx].cpu_fan_rpm_end);
        __atomic_store_n(&g_active_cores, 0, __ATOMIC_RELEASE);

        allo[idx].wall_us = (t1 - t0) / 1000LL;
        for (int r = 0; r < 5; ++r) {
            allo[idx].rapl_uj[r] = rapl_diff(r0[r], r1[r], sp.rapl_max[r]);
        }
        uint64_t ops_sum = 0;
        for (int c = 0; c < p; ++c) ops_sum += g_worker_ops[c];
        allo[idx].total_ops = ops_sum;
        allo[idx].mean_active_khz = (freq_cnt > 0) ? (freq_sum / freq_cnt) : 0;

        /* 500 ms cool pause between levels */
        nanosleep(&settle, NULL);
    }

    __atomic_store_n(&g_stop_workers, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < 8; ++i) pthread_join(th[i], NULL);

    /* Write pure-integer JSON output */
    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema_version\": \"s64.granular_silicon_capture.v1\",\n");
    fprintf(fp, "  \"provenance_tier\": \"sampled\",\n");
    fprintf(fp, "  \"rapl_domains_available\": [%d,%d,%d,%d,%d],\n",
            sp.rapl_ok[0], sp.rapl_ok[1], sp.rapl_ok[2], sp.rapl_ok[3], sp.rapl_ok[4]);

    /* Exp 1 */
    fprintf(fp, "  \"exp1_rdtsc_jitter_regimes\": [\n");
    for (int i = 0; i < 3; ++i) {
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\",\n", jr[i].name);
        fprintf(fp, "      \"working_set_bytes\": %zu,\n", jr[i].bytes);
        fprintf(fp, "      \"steps_per_sample\": %d,\n", jr[i].steps_per_sample);
        fprintf(fp, "      \"n_samples\": %d,\n", jr[i].n_samples);
        fprintf(fp, "      \"min_cyc\": %llu,\n", (unsigned long long)jr[i].min_cyc);
        fprintf(fp, "      \"p10_cyc\": %llu,\n", (unsigned long long)jr[i].p10_cyc);
        fprintf(fp, "      \"p50_cyc\": %llu,\n", (unsigned long long)jr[i].p50_cyc);
        fprintf(fp, "      \"p90_cyc\": %llu,\n", (unsigned long long)jr[i].p90_cyc);
        fprintf(fp, "      \"p95_cyc\": %llu,\n", (unsigned long long)jr[i].p95_cyc);
        fprintf(fp, "      \"p99_cyc\": %llu,\n", (unsigned long long)jr[i].p99_cyc);
        fprintf(fp, "      \"p999_cyc\": %llu,\n", (unsigned long long)jr[i].p999_cyc);
        fprintf(fp, "      \"max_cyc\": %llu,\n", (unsigned long long)jr[i].max_cyc);
        fprintf(fp, "      \"total_cyc\": %llu,\n", (unsigned long long)jr[i].total_cyc);
        fprintf(fp, "      \"n_avalanches_gt_p95\": %d,\n", jr[i].n_avalanches_p95);
        fprintf(fp, "      \"max_avalanche_len\": %d,\n", jr[i].max_avalanche_len);
        fprintf(fp, "      \"total_avalanche_excess_cyc\": %llu,\n", (unsigned long long)jr[i].total_avalanche_excess_cyc);
        fprintf(fp, "      \"max_avalanche_excess_cyc\": %llu,\n", (unsigned long long)jr[i].max_avalanche_excess_cyc);
        fprintf(fp, "      \"log2_hist_24\": [");
        for (int b = 0; b < 24; ++b) fprintf(fp, "%s%llu", b ? "," : "", (unsigned long long)jr[i].log2_hist[b]);
        fprintf(fp, "],\n");
        fprintf(fp, "      \"coarse_256_block_sums\": [");
        for (int k = 0; k < 256; ++k) fprintf(fp, "%s%llu", k ? "," : "", (unsigned long long)jr[i].coarse_256[k]);
        fprintf(fp, "]\n");
        fprintf(fp, "    }%s\n", (i < 2) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    /* Exp 2: 1,600 x 100 Hz ticks packed compactly as column arrays */
    fprintf(fp, "  \"exp2_100hz_multitimescale\": {\n");
    fprintf(fp, "    \"n_ticks\": %d,\n", N_TICKS);
    fprintf(fp, "    \"tick_ms_nominal\": 10,\n");
    fprintf(fp, "    \"phase_bounds\": {\"quiescent\":[0,400],\"ramp_up_2c\":[400,600],\"peak_8c\":[600,1000],\"ramp_down_2c\":[1000,1200],\"relaxation_0c\":[1200,1600]},\n");
    const char *cols[13] = {
        "dt_us", "active_cores", "pkg_uj", "core_uj", "uncore_uj", "psys_uj", "mmio_uj",
        "pkg_mc", "core0_mc", "core1_mc", "core7_mc", "cpu_fan_rpm", "gpu_fan_rpm"
    };
    for (int c = 0; c < 13; ++c) {
        fprintf(fp, "    \"%s\": [", cols[c]);
        for (int t = 0; t < N_TICKS; ++t) {
            int64_t v = 0;
            switch (c) {
                case 0: v = ticks[t].dt_us; break;
                case 1: v = ticks[t].active_cores; break;
                case 2: v = ticks[t].rapl_uj[0]; break;
                case 3: v = ticks[t].rapl_uj[1]; break;
                case 4: v = ticks[t].rapl_uj[2]; break;
                case 5: v = ticks[t].rapl_uj[3]; break;
                case 6: v = ticks[t].rapl_uj[4]; break;
                case 7: v = ticks[t].pkg_mc; break;
                case 8: v = ticks[t].c0_mc; break;
                case 9: v = ticks[t].c1_mc; break;
                case 10: v = ticks[t].c7_mc; break;
                case 11: v = ticks[t].cpu_fan_rpm; break;
                case 12: v = ticks[t].gpu_fan_rpm; break;
            }
            fprintf(fp, "%s%lld", t ? "," : "", (long long)v);
        }
        fprintf(fp, "],\n");
    }
    fprintf(fp, "    \"cpu0_khz\": [");
    for (int t = 0; t < N_TICKS; ++t) fprintf(fp, "%s%lld", t ? "," : "", (long long)ticks[t].cpu0_khz);
    fprintf(fp, "]\n  },\n");

    /* Exp 3: Allometric core scaling */
    fprintf(fp, "  \"exp3_allometric_core_scaling\": [\n");
    for (int i = 0; i < 5; ++i) {
        fprintf(fp, "    {\"p_cores\":%d,\"wall_us\":%lld,\"total_ops\":%llu,\"pkg_uj\":%lld,\"core_uj\":%lld,\"uncore_uj\":%lld,\"psys_uj\":%lld,\"mmio_uj\":%lld,\"mean_active_khz\":%lld,\"pkg_mc_start\":%lld,\"pkg_mc_end\":%lld,\"cpu_fan_rpm_end\":%lld}%s\n",
                allo[i].p_cores, (long long)allo[i].wall_us, (unsigned long long)allo[i].total_ops,
                (long long)allo[i].rapl_uj[0], (long long)allo[i].rapl_uj[1], (long long)allo[i].rapl_uj[2],
                (long long)allo[i].rapl_uj[3], (long long)allo[i].rapl_uj[4],
                (long long)allo[i].mean_active_khz, (long long)allo[i].pkg_mc_start,
                (long long)allo[i].pkg_mc_end, (long long)allo[i].cpu_fan_rpm_end,
                (i < 4) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
    free(ticks);
    return 0;
}
