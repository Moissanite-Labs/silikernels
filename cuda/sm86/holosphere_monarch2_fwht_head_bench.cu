// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// Bare-Metal NVIDIA Ampere sm_86 Live Silicon Benchmark & NaN-Safe FP64 Parity Harness
//   Measures (Latency & Numerical Parity Only; Zero Power/Energy Telemetry):
//     1. NaN-safe FP64 parity (< 1e-5, NaN-poisoned output buffers) for arm2b_bias (514 B),
//        arm5_bias (2.5 KiB), arm7_bias (8.5 KiB), and arm9_monarch2_bias (16.5 KiB)
//     2. Live RTX 3080 P0-warmed 3-trial interleaved CUDA-Graph latency (us/tok)
//     3. Fail-closed non-zero exit if any FP64 parity error is non-finite or >= 1e-5
// ============================================================================

#include "holosphere_monarch2_fwht_head_sm86.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA_ERR %s:%d %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(err__));                                     \
            std::exit(2);                                                                \
        }                                                                                \
    } while (0)

namespace {

static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static float rand_u(uint64_t& s, float lo, float hi) {
    double u = double(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
    return float(lo + (hi - lo) * u);
}

static std::vector<double> fwht256_plain_fp64(const std::vector<double>& u) {
    std::vector<double> out(256, 0.0);
    for (int k = 0; k < 256; ++k) {
        double acc = 0.0;
        for (int j = 0; j < 256; ++j) {
            int par = __builtin_popcount(k & j) & 1;
            acc += par ? -u[j] : u[j];
        }
        out[k] = acc * 0.0625;
    }
    return out;
}

static std::vector<double> fwht256_bent4_fp64(const std::vector<double>& u) {
    const double inv_sqrt32 = 1.0 / std::sqrt(32.0);
    std::vector<double> out(256, 0.0);
    for (int tok = 0; tok < 256; ++tok) {
        double acc = 0.0;
        for (int idx = 0; idx < 256; ++idx) {
            int lane = idx & 31;
            int db   = idx ^ ((lane & 7) << 5);
            int b4   = __builtin_popcount((lane & 3) & ((lane >> 2) & 3)) & 1;
            int par  = (__builtin_popcount(tok & db) ^ b4) & 1;
            acc += (par ? -1.0 : 1.0) * inv_sqrt32 * u[idx];
        }
        out[tok] = acc;
    }
    return out;
}

static std::vector<double> monarch16_stage_fp64(
    const std::vector<__half>& B, const std::vector<double>& z_in
) {
    std::vector<double> z_out(256, 0.0);
    for (int g = 0; g < 16; ++g) {
        for (int r = 0; r < 16; ++r) {
            double acc = 0.0;
            for (int c = 0; c < 16; ++c) {
                acc += double(__half2float(B[(g * 16 + r) * 16 + c])) * z_in[c * 16 + g];
            }
            z_out[g * 16 + r] = acc;
        }
    }
    return z_out;
}

static double max_abs_diff_nan_safe(const std::vector<double>& ref, const std::vector<float>& got) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i];
        double g = double(got[i]);
        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(m)) {
            return std::numeric_limits<double>::infinity();
        }
        double d = std::fabs(r - g);
        if (!std::isfinite(d)) {
            return std::numeric_limits<double>::infinity();
        }
        if (d > m) m = d;
    }
    return m;
}

static double med3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

} // namespace

int main() {
    uint64_t rng = 0x5156B2026ULL;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> h_x(512);
    double n2 = 0.0;
    for (int i = 0; i < 512; ++i) {
        h_x[i] = rand_u(rng, -1.0f, 1.0f);
        n2 += double(h_x[i]) * double(h_x[i]);
    }
    double inv_n = 1.0 / std::sqrt(n2);
    for (int i = 0; i < 512; ++i) h_x[i] = float(double(h_x[i]) * inv_n);

    __half h_scale = __float2half(1.75f);
    std::vector<__half>  h_bias(256);
    std::vector<__half2> h_ab(256), h_d12(256);
    for (int i = 0; i < 256; ++i) {
        h_bias[i] = __float2half(rand_u(rng, -3.0f, 3.0f));
        h_ab[i]   = __floats2half2_rn(rand_u(rng, 0.5f, 1.5f), rand_u(rng, -1.5f, -0.5f));
        h_d12[i]  = __floats2half2_rn(rand_u(rng, -1.5f, 1.5f), rand_u(rng, -1.5f, 1.5f));
    }
    std::vector<__half> h_B1(16 * 16 * 16), h_B2(16 * 16 * 16);
    for (size_t i = 0; i < h_B1.size(); ++i) {
        h_B1[i] = __float2half(rand_u(rng, -0.5f, 0.5f));
        h_B2[i] = __float2half(rand_u(rng, -0.5f, 0.5f));
    }

    float*   d_x = nullptr;
    __half*  d_sc = nullptr;
    __half*  d_bias = nullptr;
    __half2* d_ab = nullptr;
    __half2* d_d12 = nullptr;
    uint4*   d_B1 = nullptr;
    uint4*   d_B2 = nullptr;
    float*   d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sc, sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, 256 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ab, 256 * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc(&d_d12, 256 * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc(&d_B1, 16 * 16 * 16 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B2, 16 * 16 * 16 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), 512 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sc, &h_scale, sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), 256 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ab, h_ab.data(), 256 * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d12, h_d12.data(), 256 * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B1.data(), 16 * 16 * 16 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B2.data(), 16 * 16 * 16 * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<float> got(256);
    std::vector<double> u_diff(256);
    for (int i = 0; i < 256; ++i) u_diff[i] = double(h_x[i]) - double(h_x[256 + i]);

    auto poison_out = [&]() {
        CUDA_CHECK(cudaMemset(d_out, 0xFF, 256 * sizeof(float)));
    };

    // 1. Arm 2b + bias parity (NaN-poisoned output buffer)
    poison_out();
    holosphere_monarch2_head::holosphere_arm2b_bias_514b_sm86<<<1, 256>>>(d_x, d_sc, d_bias, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    auto ref_2b = fwht256_bent4_fp64(u_diff);
    double s_ref = double(__half2float(h_scale));
    for (int i = 0; i < 256; ++i) ref_2b[i] = ref_2b[i] * s_ref + double(__half2float(h_bias[i]));
    double err_2b = max_abs_diff_nan_safe(ref_2b, got);

    // 2. Arm 5 + bias parity
    poison_out();
    holosphere_monarch2_head::holosphere_arm5_bias_2p5kib_sm86<<<1, 256>>>(d_x, d_ab, d_d12, d_bias, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<double> u5(256);
    for (int i = 0; i < 256; ++i) {
        float2 ab = __half22float2(h_ab[i]);
        u5[i] = double(ab.x) * double(h_x[i]) + double(ab.y) * double(h_x[256 + i]);
    }
    auto z1_5 = fwht256_plain_fp64(u5);
    for (int i = 0; i < 256; ++i) z1_5[i] *= double(__half22float2(h_d12[i]).x);
    auto z2_5 = fwht256_plain_fp64(z1_5);
    for (int i = 0; i < 256; ++i) z2_5[i] *= double(__half22float2(h_d12[i]).y);
    auto ref_5 = fwht256_bent4_fp64(z2_5);
    for (int i = 0; i < 256; ++i) ref_5[i] += double(__half2float(h_bias[i]));
    double err_5 = max_abs_diff_nan_safe(ref_5, got);

    // 3. Arm 7 (1-stage Monarch) + bias parity
    poison_out();
    holosphere_monarch2_head::holosphere_arm7_bias_8p5kib_sm86<<<1, 512>>>(d_x, d_B1, d_bias, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    auto z1_m = fwht256_plain_fp64(u_diff);
    auto z3_m = monarch16_stage_fp64(h_B1, z1_m);
    auto ref_7 = fwht256_bent4_fp64(z3_m);
    for (int i = 0; i < 256; ++i) ref_7[i] += double(__half2float(h_bias[i]));
    double err_7 = max_abs_diff_nan_safe(ref_7, got);

    // 4. Arm 9 (2-stage Monarch) + bias parity
    poison_out();
    holosphere_monarch2_head::holosphere_arm9_monarch2_bias_16p5kib_sm86<<<1, 512>>>(d_x, d_B1, d_B2, d_bias, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    auto z5_m = monarch16_stage_fp64(h_B2, z3_m);
    auto ref_9 = fwht256_bent4_fp64(z5_m);
    for (int i = 0; i < 256; ++i) ref_9[i] += double(__half2float(h_bias[i]));
    double err_9 = max_abs_diff_nan_safe(ref_9, got);

    // P0 warmup + 3-trial interleaved CUDA-Graph timing (16,384 iters)
    cudaStream_t st;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaStreamCreate(&st));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    for (int w = 0; w < 2048; ++w) {
        holosphere_monarch2_head::holosphere_arm9_monarch2_bias_16p5kib_sm86<<<1, 512, 0, st>>>(d_x, d_B1, d_B2, d_bias, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));

    const int total_iters = 16384;
    const int graph_unroll = 128;
    const int n_launches = total_iters / graph_unroll;

    auto bench_arm = [&](int arm_id) -> double {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < graph_unroll; ++k) {
            if (arm_id == 0) {
                holosphere_monarch2_head::holosphere_arm2b_bias_514b_sm86<<<1, 256, 0, st>>>(d_x, d_sc, d_bias, d_out);
            } else if (arm_id == 1) {
                holosphere_monarch2_head::holosphere_arm5_bias_2p5kib_sm86<<<1, 256, 0, st>>>(d_x, d_ab, d_d12, d_bias, d_out);
            } else if (arm_id == 2) {
                holosphere_monarch2_head::holosphere_arm7_bias_8p5kib_sm86<<<1, 512, 0, st>>>(d_x, d_B1, d_bias, d_out);
            } else {
                holosphere_monarch2_head::holosphere_arm9_monarch2_bias_16p5kib_sm86<<<1, 512, 0, st>>>(d_x, d_B1, d_B2, d_bias, d_out);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(st, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 8; ++w) CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaEventRecord(ev0, st));
        for (int it = 0; it < n_launches; ++it) CUDA_CHECK(cudaGraphLaunch(ge, st));
        CUDA_CHECK(cudaEventRecord(ev1, st));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        CUDA_CHECK(cudaGraphExecDestroy(ge));
        CUDA_CHECK(cudaGraphDestroy(g));
        return (double(ms) * 1000.0) / double(total_iters);
    };

    double t2b[3], t5[3], t7[3], t9[3];
    for (int tr = 0; tr < 3; ++tr) {
        t2b[tr] = bench_arm(0);
        t5[tr]  = bench_arm(1);
        t7[tr]  = bench_arm(2);
        t9[tr]  = bench_arm(3);
    }

    std::printf("{\n");
    std::printf("  \"device\": \"%s\",\n", prop.name);
    std::printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    std::printf("  \"arm2b_bias_fp64_max_err\": %.9e,\n", err_2b);
    std::printf("  \"arm5_bias_fp64_max_err\": %.9e,\n", err_5);
    std::printf("  \"arm7_bias_fp64_max_err\": %.9e,\n", err_7);
    std::printf("  \"arm9_monarch2_bias_fp64_max_err\": %.9e,\n", err_9);
    std::printf("  \"arm2b_bias_us_per_tok\": %.6f,\n", med3(t2b[0], t2b[1], t2b[2]));
    std::printf("  \"arm5_bias_us_per_tok\": %.6f,\n", med3(t5[0], t5[1], t5[2]));
    std::printf("  \"arm7_bias_us_per_tok\": %.6f,\n", med3(t7[0], t7[1], t7[2]));
    std::printf("  \"arm9_monarch2_bias_us_per_tok\": %.6f\n", med3(t9[0], t9[1], t9[2]));
    std::printf("}\n");

    bool ok = std::isfinite(err_2b) && (err_2b < 1e-5)
           && std::isfinite(err_5)  && (err_5  < 1e-5)
           && std::isfinite(err_7)  && (err_7  < 1e-5)
           && std::isfinite(err_9)  && (err_9  < 1e-5);
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaStreamDestroy(st));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_sc));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_d12));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_out));
    return ok ? 0 : 3;
}
