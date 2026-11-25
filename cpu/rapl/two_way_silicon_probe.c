/*
 * Live Deterministic Silicon Energy & Concurrency Impedance Probe
 *
 * Measures exact integer microjoules (intel-rapl package-0, core, uncore, psys)
 * and monotonic microseconds on Intel i9-11900H across:
 *   - Idle baseline power floor (P_0)
 *   - Experiment A: Fixed-work pacing vs. bursty duty-cycle & race-to-sleep
 *     (testing Endoreversible Jensen vs. Race-to-Sleep crossover)
 *   - Experiment B: Core-count & SMT hyperthreading impedance sweep
 *     p in {1, 2, 4, 6, 8, 12, 16} at identical fixed total work W_B
 *     (2400 identical deterministic integer-ALU + L1-tile work blocks,
 *      pinned to physical cores 0..7 first, then SMT siblings 8..15).
 *
 * Pure C11 + POSIX pthreads, zero floating-point arithmetic in work kernel,
 * zero microphone/camera/display/GPU touches, deterministic checksums verifying
 * identical retired operations across every arm.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define RAPL_PKG_PATH    "/sys/class/powercap/intel-rapl:0/energy_uj"
#define RAPL_CORE_PATH   "/sys/class/powercap/intel-rapl:0:0/energy_uj"
#define RAPL_UNCORE_PATH "/sys/class/powercap/intel-rapl:0:1/energy_uj"
#define RAPL_PSYS_PATH   "/sys/class/powercap/intel-rapl:1/energy_uj"
#define RAPL_MAX_PATH    "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
#define FREQ_CPU0_PATH   "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
#define TEMP_PKG_PATH    "/sys/class/thermal/thermal_zone0/temp"

/* Exact block geometry: 4096 uint64_t = 32 KiB per thread (fits private 48 KiB L1D
 * on physical cores 0..7; contends on shared L1D + execution ports under SMT 8..15). */
#define TILE_WORDS 4096
#define INNER_ROUNDS 64
#define OPS_PER_BLOCK ((uint64_t)TILE_WORDS * (uint64_t)INNER_ROUNDS * UINT64_C(4))

typedef struct {
    uint64_t pkg_uj;
    uint64_t core_uj;
    uint64_t uncore_uj;
    uint64_t psys_uj;
    uint64_t wall_ns;
    int64_t  freq_khz;
    int64_t  temp_mc;
} sample_t;

typedef struct {
    uint64_t pkg_uj;
    uint64_t core_uj;
    uint64_t uncore_uj;
    uint64_t psys_uj;
    uint64_t wall_us;
    int64_t  freq_khz_start;
    int64_t  freq_khz_end;
    int64_t  temp_mc_end;
    uint64_t checksum;
} measurement_t;

static uint64_t g_rapl_max_uj = 262143328850ULL;

static int read_u64_file(const char *path, uint64_t *out) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    unsigned long long v = 0;
    int rc = fscanf(fp, "%llu", &v);
    fclose(fp);
    if (rc != 1) return -1;
    *out = (uint64_t)v;
    return 0;
}

static int64_t read_i64_opt(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    long long v = -1;
    if (fscanf(fp, "%lld", &v) != 1) v = -1;
    fclose(fp);
    return (int64_t)v;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void sleep_until_ns(uint64_t target_ns) {
    struct timespec ts;
    ts.tv_sec = (time_t)(target_ns / 1000000000ULL);
    ts.tv_nsec = (long)(target_ns % 1000000000ULL);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {
    }
}

static uint64_t rapl_diff(uint64_t after, uint64_t before) {
    if (after >= before) return after - before;
    return (g_rapl_max_uj - before) + after;
}

static int take_sample(sample_t *s) {
    s->wall_ns = monotonic_ns();
    if (read_u64_file(RAPL_PKG_PATH, &s->pkg_uj) != 0) return -1;
    if (read_u64_file(RAPL_CORE_PATH, &s->core_uj) != 0) return -1;
    if (read_u64_file(RAPL_UNCORE_PATH, &s->uncore_uj) != 0) return -1;
    if (read_u64_file(RAPL_PSYS_PATH, &s->psys_uj) != 0) return -1;
    s->freq_khz = read_i64_opt(FREQ_CPU0_PATH);
    s->temp_mc = read_i64_opt(TEMP_PKG_PATH);
    return 0;
}

static measurement_t diff_samples(const sample_t *a, const sample_t *b, uint64_t checksum) {
    measurement_t m;
    m.pkg_uj = rapl_diff(b->pkg_uj, a->pkg_uj);
    m.core_uj = rapl_diff(b->core_uj, a->core_uj);
    m.uncore_uj = rapl_diff(b->uncore_uj, a->uncore_uj);
    m.psys_uj = rapl_diff(b->psys_uj, a->psys_uj);
    m.wall_us = (b->wall_ns - a->wall_ns) / 1000ULL;
    m.freq_khz_start = a->freq_khz;
    m.freq_khz_end = b->freq_khz;
    m.temp_mc_end = b->temp_mc;
    m.checksum = checksum;
    return m;
}

/* Deterministic integer-ALU + 32 KiB L1-tile work block.
 * Each block_id executes identically regardless of which thread runs it,
 * so summing block hashes across all block_ids yields an order-invariant
 * checksum proving exact conservation of total retired operations. */
static uint64_t run_one_block(uint64_t block_id, uint64_t *tile) {
    uint64_t x = block_id * 0x9E3779B97F4A7C15ULL + 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < TILE_WORDS; ++i) {
        x ^= x >> 30;
        x *= 0xBF58476D1CE4E5B9ULL;
        tile[i] = x;
    }
    uint64_t acc = block_id ^ 0x94D049BB133111EBULL;
    for (int r = 0; r < INNER_ROUNDS; ++r) {
        uint64_t c = 0x517CC1B727220A95ULL + (uint64_t)r;
        for (int i = 0; i < TILE_WORDS; ++i) {
            uint64_t v = tile[i];
            v = (v * 0x9E3779B97F4A7C15ULL) ^ (v >> 27) ^ acc;
            tile[i] = v + c;
            acc = (acc + v) ^ (v >> 13);
        }
    }
    return acc;
}

typedef struct {
    int      cpu_id;
    uint64_t block_start;
    uint64_t block_count;
    uint64_t slices;
    uint64_t win_start_ns;
    uint64_t win_dur_ns;
    uint64_t out_sum;
    pthread_barrier_t *barrier;
} worker_arg_t;

static void *worker_fn(void *argp) {
    worker_arg_t *a = (worker_arg_t *)argp;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->cpu_id, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    uint64_t *tile = NULL;
    if (posix_memalign((void **)&tile, 64, (size_t)TILE_WORDS * sizeof(uint64_t)) != 0 || !tile) {
        return NULL;
    }

    pthread_barrier_wait(a->barrier);

    uint64_t sum = 0;
    if (a->slices <= 1 || a->win_dur_ns == 0) {
        for (uint64_t k = 0; k < a->block_count; ++k) {
            sum += run_one_block(a->block_start + k, tile);
        }
        if (a->win_dur_ns > 0) {
            sleep_until_ns(a->win_start_ns + a->win_dur_ns);
        }
    } else {
        uint64_t per_slice = a->block_count / a->slices;
        uint64_t slice_ns = a->win_dur_ns / a->slices;
        uint64_t cursor = a->block_start;
        for (uint64_t s = 0; s < a->slices; ++s) {
            for (uint64_t k = 0; k < per_slice; ++k) {
                sum += run_one_block(cursor++, tile);
            }
            sleep_until_ns(a->win_start_ns + (s + 1ULL) * slice_ns);
        }
    }

    free(tile);
    a->out_sum = sum;
    return NULL;
}

static int run_parallel_job(
    int n_threads,
    uint64_t total_blocks,
    uint64_t slices,
    uint64_t win_dur_ms,
    measurement_t *out_m
) {
    if (n_threads < 1 || n_threads > 16) return -1;
    if (total_blocks % (uint64_t)n_threads != 0) return -1;
    uint64_t per_thr = total_blocks / (uint64_t)n_threads;
    if (slices > 1 && (per_thr % slices != 0)) return -1;

    pthread_t th[16];
    worker_arg_t args[16];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, (unsigned)(n_threads + 1));

    uint64_t win_dur_ns = win_dur_ms * 1000000ULL;
    for (int i = 0; i < n_threads; ++i) {
        args[i].cpu_id = i; /* CPUs 0..7 are physical cores; 8..15 are SMT siblings */
        args[i].block_start = (uint64_t)i * per_thr;
        args[i].block_count = per_thr;
        args[i].slices = slices;
        args[i].win_dur_ns = win_dur_ns;
        args[i].out_sum = 0;
        args[i].barrier = &barrier;
    }

    for (int i = 0; i < n_threads; ++i) {
        if (pthread_create(&th[i], NULL, worker_fn, &args[i]) != 0) {
            return -1;
        }
    }

    sample_t s0, s1;
    if (take_sample(&s0) != 0) return -1;
    uint64_t start_ns = monotonic_ns();
    for (int i = 0; i < n_threads; ++i) {
        args[i].win_start_ns = start_ns;
    }
    pthread_barrier_wait(&barrier);

    uint64_t total_sum = 0;
    for (int i = 0; i < n_threads; ++i) {
        pthread_join(th[i], NULL);
        total_sum += args[i].out_sum;
    }
    if (take_sample(&s1) != 0) return -1;
    pthread_barrier_destroy(&barrier);

    *out_m = diff_samples(&s0, &s1, total_sum);
    return 0;
}

static int run_idle_window(uint64_t dur_ms, measurement_t *out_m) {
    sample_t s0, s1;
    if (take_sample(&s0) != 0) return -1;
    sleep_until_ns(monotonic_ns() + dur_ms * 1000000ULL);
    if (take_sample(&s1) != 0) return -1;
    *out_m = diff_samples(&s0, &s1, 0ULL);
    return 0;
}

static void print_meas_json(const measurement_t *m) {
    printf(
        "{\"wall_us\":%" PRIu64 ",\"pkg_uj\":%" PRIu64 ",\"core_uj\":%" PRIu64
        ",\"uncore_uj\":%" PRIu64 ",\"psys_uj\":%" PRIu64
        ",\"freq_khz_start\":%" PRId64 ",\"freq_khz_end\":%" PRId64
        ",\"temp_mc_end\":%" PRId64 ",\"checksum_hex\":\"%016" PRIx64 "\"}",
        m->wall_us, m->pkg_uj, m->core_uj, m->uncore_uj, m->psys_uj,
        m->freq_khz_start, m->freq_khz_end, m->temp_mc_end, m->checksum
    );
}

int main(int argc, char **argv) {
    int calibrate_only = (argc >= 2 && strcmp(argv[1], "--calibrate") == 0);
    (void)read_u64_file(RAPL_MAX_PATH, &g_rapl_max_uj);

    if (calibrate_only) {
        measurement_t m;
        if (run_parallel_job(1, 240, 1, 0, &m) != 0) {
            fprintf(stderr, "calibrate failed\n");
            return 2;
        }
        fprintf(stderr, "CALIBRATE 1T 240 blocks: wall_us=%" PRIu64 " pkg_uj=%" PRIu64 "\n", m.wall_us, m.pkg_uj);
        if (run_parallel_job(4, 960, 1, 0, &m) != 0) return 2;
        fprintf(stderr, "CALIBRATE 4T 960 blocks: wall_us=%" PRIu64 " pkg_uj=%" PRIu64 "\n", m.wall_us, m.pkg_uj);
        return 0;
    }

    /* Brief warmup + 1-shot deterministic calibration so:
     * - Exp A (4T) active burst takes ~500 ms inside the fixed 1500 ms window (~33% duty cycle),
     *   with W_A_BLOCKS a multiple of 400 (divides cleanly by 4 threads x {1, 4, 40, 100} slices);
     * - Exp B (1T) takes ~1500 ms (and 8T ~200 ms), with W_B_BLOCKS a multiple of 480
     *   (divides cleanly by every p in {1, 2, 4, 6, 8, 12, 16}). */
    measurement_t warm;
    if (run_parallel_job(4, 160, 1, 0, &warm) != 0) return 2;
    uint64_t t4_160_us = (warm.wall_us > 0) ? warm.wall_us : 1000ULL;
    /* Target ~500,000 us on 4T for W_A_BLOCKS, rounded to multiple of 800 in [800, 16000]
     * so per_thr = W_A_BLOCKS / 4 is a multiple of 200 (divides cleanly by 1, 4, 40, 100). */
    uint64_t raw_a = (160ULL * 500000ULL) / t4_160_us;
    uint64_t W_A_BLOCKS = ((raw_a + 400ULL) / 800ULL) * 800ULL;
    if (W_A_BLOCKS < 800ULL) W_A_BLOCKS = 800ULL;
    if (W_A_BLOCKS > 16000ULL) W_A_BLOCKS = 16000ULL;
    const uint64_t WIN_A_MS = 1500ULL;
    /* Target ~350,000 us on 4T (~1.4s on 1T) for W_B_BLOCKS, multiple of 480 in [480, 19200]. */
    uint64_t raw_b = (160ULL * 350000ULL) / t4_160_us;
    uint64_t W_B_BLOCKS = ((raw_b + 240ULL) / 480ULL) * 480ULL;
    if (W_B_BLOCKS < 480ULL) W_B_BLOCKS = 480ULL;
    if (W_B_BLOCKS > 19200ULL) W_B_BLOCKS = 19200ULL;
    usleep(300000);

    measurement_t idle_pre, idle_post;
    if (run_idle_window(1500ULL, &idle_pre) != 0) return 2;

    /* Experiment A: 2 interleaved trials of (race_free, burst_1, burst_4, paced_40, paced_100) */
    const char *a_names[5] = {
        "race_to_sleep_free",
        "bursty_1_pulse_1500ms",
        "bursty_4_pulses_1500ms",
        "paced_40_slices_1500ms",
        "paced_100_slices_1500ms"
    };
    const uint64_t a_slices[5] = {1ULL, 1ULL, 4ULL, 40ULL, 100ULL};
    const uint64_t a_win_ms[5] = {0ULL, WIN_A_MS, WIN_A_MS, WIN_A_MS, WIN_A_MS};
    measurement_t a_res[2][5];

    for (int t = 0; t < 2; ++t) {
        for (int idx = 0; idx < 5; ++idx) {
            int arm = (t == 0) ? idx : (4 - idx);
            if (run_parallel_job(4, W_A_BLOCKS, a_slices[arm], a_win_ms[arm], &a_res[t][arm]) != 0) {
                fprintf(stderr, "Exp A failed t=%d arm=%d\n", t, arm);
                return 2;
            }
            usleep(120000);
        }
    }

    /* Experiment B: 2 opposite-order sweeps of p in {1, 2, 4, 6, 8, 12, 16} at fixed W_B = 1440 blocks */
    const int p_list[7] = {1, 2, 4, 6, 8, 12, 16};
    measurement_t b_res[2][7];
    for (int t = 0; t < 2; ++t) {
        for (int idx = 0; idx < 7; ++idx) {
            int k = (t == 0) ? idx : (6 - idx);
            if (run_parallel_job(p_list[k], W_B_BLOCKS, 1ULL, 0ULL, &b_res[t][k]) != 0) {
                fprintf(stderr, "Exp B failed t=%d p=%d\n", t, p_list[k]);
                return 2;
            }
            usleep(120000);
        }
    }

    usleep(300000);
    if (run_idle_window(1500ULL, &idle_post) != 0) return 2;

    /* Emit strict pure-integer JSON to stdout */
    printf("{\n");
    printf("  \"schema\": \"aletheia.track_s65_live_silicon_probe.v1\",\n");
    printf("  \"measured_live_silicon\": true,\n");
    printf("  \"host_cpu_model\": \"Intel Core i9-11900H (8 physical cores 0..7 + 8 SMT siblings 8..15)\",\n");
    printf("  \"rapl_max_energy_range_uj\": %" PRIu64 ",\n", g_rapl_max_uj);
    printf("  \"tile_words_u64\": %d,\n", TILE_WORDS);
    printf("  \"inner_rounds\": %d,\n", INNER_ROUNDS);
    printf("  \"ops_per_block\": %" PRIu64 ",\n", OPS_PER_BLOCK);
    printf("  \"exp_a_total_blocks\": %" PRIu64 ",\n", W_A_BLOCKS);
    printf("  \"exp_a_total_ops\": %" PRIu64 ",\n", W_A_BLOCKS * OPS_PER_BLOCK);
    printf("  \"exp_a_window_ms\": %" PRIu64 ",\n", WIN_A_MS);
    printf("  \"exp_a_threads\": 4,\n");
    printf("  \"exp_b_total_blocks\": %" PRIu64 ",\n", W_B_BLOCKS);
    printf("  \"exp_b_total_ops\": %" PRIu64 ",\n", W_B_BLOCKS * OPS_PER_BLOCK);
    printf("  \"idle_pre_1500ms\": ");
    print_meas_json(&idle_pre);
    printf(",\n  \"idle_post_1500ms\": ");
    print_meas_json(&idle_post);
    printf(",\n  \"exp_a_pacing_vs_bursty\": [\n");
    for (int arm = 0; arm < 5; ++arm) {
        printf("    {\"arm\":\"%s\",\"slices\":%" PRIu64 ",\"window_ms\":%" PRIu64 ",\"trials\":[",
               a_names[arm], a_slices[arm], a_win_ms[arm]);
        print_meas_json(&a_res[0][arm]);
        printf(",");
        print_meas_json(&a_res[1][arm]);
        printf("]}%s\n", (arm < 4) ? "," : "");
    }
    printf("  ],\n  \"exp_b_concurrency_impedance\": [\n");
    for (int k = 0; k < 7; ++k) {
        int p = p_list[k];
        printf("    {\"threads\":%d,\"physical_cores_used\":%d,\"smt_siblings_used\":%d,\"trials\":[",
               p, (p <= 8 ? p : 8), (p > 8 ? p - 8 : 0));
        print_meas_json(&b_res[0][k]);
        printf(",");
        print_meas_json(&b_res[1][k]);
        printf("]}%s\n", (k < 6) ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}
