// ============================================================================
// SPDX-License-Identifier: Apache-2.0
// Bare-Metal NVIDIA Ampere sm_86 Live Silicon Benchmark Harness (Latency, Parity, SEC/DED, .sys Atomics Only)
//   Measures:
//     1. Exact FP64 parity of 1 KiB FP16 channel-folded FWHT_256 x bent4 LM head (NaN-safe, poisoned buffers)
//     2. Exact 16/16 single-bit SEC heal (0.0 bit-exact hidden & logits + packed==16)
//     3. Exact 120/120 double-bit DED pair detection ((packed>>16)==120, (packed&0xFFFF)==0)
//     4. Exact fail-closed single-CTA AND multi-CTA saturation/overflow probes on GPU:
//        - Single-CTA SEC/DED overflow (70000,0) & (12,70000)
//        - Multi-CTA SEC carry defense: 2 CTAs x (40000, 0) -> 0x0001FFFF + ovf bit (0 SEC->DED carry!)
//        - Multi-CTA DED bit-31 wrap defense: 2 CTAs x (7, 0x8000) -> 0xFFFF000E + ovf bit (0 wrap-to-zero!)
//     5. Multi-warp brownout storm sweep (0%, 1%, 10%, 100% fault rates across 16 warps x 12 layers):
//        - Exact (ded << 16) | sec recovery (warp-level vs CTA-SMEM)
//        - Exact hardware .sys CAS atomic transaction count (warp vs CTA-SMEM)
//        - Exact CUDA-event & CUDA-Graph decode latency (us/tok)
//     6. Standalone LM-head latency overhead: unfolded vs 1 KiB FP16 fold (us/tok)
// ============================================================================

#include "holosphere_smem_canary_folded_fwht_sm86.cu"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static float rand_uniform(uint64_t& rng, float lo, float hi) {
    double u = double(splitmix64(rng) >> 11) * (1.0 / 9007199254740992.0);
    return float(lo + (hi - lo) * u);
}

static uint32_t host_a16_encode(uint32_t data11) {
    static const int dpos[11] = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    uint32_t w = 0u;
    for (int i = 0; i < 11; ++i) {
        if ((data11 >> i) & 1u) w |= (1u << dpos[i]);
    }
    uint32_t p1 = __builtin_popcount(w & 0xAAAAu) & 1u;
    uint32_t p2 = __builtin_popcount(w & 0xCCCCu) & 1u;
    uint32_t p3 = __builtin_popcount(w & 0xF0F0u) & 1u;
    uint32_t p4 = __builtin_popcount(w & 0xFF00u) & 1u;
    w |= (p1 << 1) | (p2 << 2) | (p3 << 4) | (p4 << 8);
    uint32_t p0 = __builtin_popcount(w) & 1u;
    w |= p0;
    return w;
}

static uint4 pack_16_s8_with_secded(const int8_t vals[16], uint32_t cw16) {
    uint8_t b[16];
    for (int i = 0; i < 16; ++i) {
        uint8_t u = uint8_t(vals[i]);
        u = uint8_t((u & 0xFEu) | ((cw16 >> i) & 1u));
        b[i] = u;
    }
    uint4 out;
    std::memcpy(&out, b, 16);
    return out;
}

static void flip_lsb_in_uint4(uint4& v, int idx16) {
    uint8_t b[16];
    std::memcpy(b, &v, 16);
    b[idx16] ^= 1u;
    std::memcpy(&v, b, 16);
}

// NaN-safe max absolute error update: any non-finite input propagates +inf
static double update_max_err_nan_safe(double cur_max, double ref_val, double got_val) {
    if (!std::isfinite(cur_max) || !std::isfinite(ref_val) || !std::isfinite(got_val)) {
        return std::numeric_limits<double>::infinity();
    }
    double err = std::fabs(ref_val - got_val);
    if (!std::isfinite(err)) {
        return std::numeric_limits<double>::infinity();
    }
    return (err > cur_max) ? err : cur_max;
}

static double dense_folded1k_fwht256_max_err(
    const std::vector<float>& h512,
    const std::vector<__half2>& ab256,
    const std::vector<float>& gpu_logits256,
    double logit_scale
) {
    const double inv_sqrt32 = 1.0 / std::sqrt(32.0);
    double max_err = 0.0;
    for (int tok = 0; tok < 256; ++tok) {
        double acc = 0.0;
        for (int w = 0; w < 8; ++w) {
            for (int lane = 0; lane < 32; ++lane) {
                int idx = w * 32 + lane;
                float2 abf = __half22float2(ab256[idx]);
                double u = double(abf.x) * double(h512[idx]) + double(abf.y) * double(h512[256 + idx]);
                int db = idx ^ ((lane & 7) << 5);
                int b4 = __builtin_popcount((lane & 3) & ((lane >> 2) & 3)) & 1;
                int par = (__builtin_popcount(tok & db) ^ b4) & 1;
                double e = (par ? -1.0 : 1.0) * inv_sqrt32;
                acc += e * u;
            }
        }
        acc *= logit_scale;
        max_err = update_max_err_nan_safe(max_err, acc, double(gpu_logits256[tok]));
    }
    return max_err;
}

struct StormMetrics {
    int rate_pct;
    uint32_t expected_sec;
    uint32_t expected_ded;
    uint32_t warp_sec;
    uint32_t warp_ded;
    uint32_t warp_sys_atomics;
    uint32_t smem_sec;
    uint32_t smem_ded;
    uint32_t smem_sys_atomics;
    double warp_us_per_tok;
    double smem_us_per_tok;
};

} // namespace

int main() {
    uint64_t rng = 0x501105EED2026ULL;
    const int L = 12;
    const int B = 1;
    const float cayley_t = 0.25f;
    const float temperature = 0.0f;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int host_native_atomic_attr = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&host_native_atomic_attr, cudaDevAttrHostNativeAtomicSupported, 0));

    std::vector<uint8_t> h_byte(B, 73);
    std::vector<uint4>   h_W_clean(L * 16 * 2 * 32);
    std::vector<float>   h_Wsc(L * 16 * 32);
    std::vector<float3>  h_wsu2(L * 16 * 8);
    std::vector<uint4>   h_kvk(B * 16 * 16);
    std::vector<uint4>   h_kvv(B * 16 * 32);
    std::vector<uint32_t> h_rng(B * 256);
    std::vector<float4>  h_U(B * 16 * 8);
    std::vector<__half2> h_ab(256);

    for (int i = 0; i < 256; ++i) {
        float a = rand_uniform(rng, 0.4f, 1.8f);
        float b = rand_uniform(rng, -1.8f, -0.4f);
        h_ab[i] = __floats2half2_rn(a, b);
    }

    for (int l = 0; l < L; ++l) {
        for (int w = 0; w < 16; ++w) {
            for (int lane = 0; lane < 32; ++lane) {
                int8_t lo_v[16], hi_v[16];
                for (int k = 0; k < 16; ++k) {
                    lo_v[k] = int8_t((int(splitmix64(rng) % 121) - 60) * 2);
                    hi_v[k] = int8_t((int(splitmix64(rng) % 121) - 60) * 2);
                }
                uint32_t cw_lo = host_a16_encode(uint32_t(splitmix64(rng) & 0x7FFu));
                uint32_t cw_hi = host_a16_encode(uint32_t(splitmix64(rng) & 0x7FFu));
                int base_idx = (l * 16 + w) * 64;
                h_W_clean[base_idx + 0 * 32 + lane] = pack_16_s8_with_secded(lo_v, cw_lo);
                h_W_clean[base_idx + 1 * 32 + lane] = pack_16_s8_with_secded(hi_v, cw_hi);
                h_Wsc[(l * 16 + w) * 32 + lane] = rand_uniform(rng, 0.002f, 0.015f);
            }
            for (int sp = 0; sp < 8; ++sp) {
                h_wsu2[(l * 16 + w) * 8 + sp] = make_float3(
                    rand_uniform(rng, -0.15f, 0.15f),
                    rand_uniform(rng, -0.15f, 0.15f),
                    rand_uniform(rng, -0.15f, 0.15f)
                );
            }
        }
    }
    for (size_t i = 0; i < h_kvk.size(); ++i) {
        h_kvk[i] = make_uint4(uint32_t(splitmix64(rng)), uint32_t(splitmix64(rng)),
                              uint32_t(splitmix64(rng)), uint32_t(splitmix64(rng)));
    }
    for (size_t i = 0; i < h_kvv.size(); ++i) {
        __half h8[8];
        for (int j = 0; j < 8; ++j) h8[j] = __float2half(rand_uniform(rng, -0.15f, 0.15f));
        std::memcpy(&h_kvv[i], h8, 16);
    }
    for (size_t i = 0; i < h_rng.size(); ++i) h_rng[i] = uint32_t(splitmix64(rng));
    for (size_t i = 0; i < h_U.size(); ++i) h_U[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

    uint8_t*  d_byte = nullptr;
    uint4*    d_W = nullptr;
    float*    d_Wsc = nullptr;
    float3*   d_wsu2 = nullptr;
    uint4*    d_kvk = nullptr;
    uint4*    d_kvv = nullptr;
    __half2*  d_ab = nullptr;
    uint32_t* d_rng = nullptr;
    float4*   d_U = nullptr;
    float*    d_hid = nullptr;
    float*    d_logits = nullptr;
    uint8_t*  d_outb = nullptr;
    float*    d_outs = nullptr;
    uint32_t* d_sys_cnt = nullptr;

    uint32_t* h_packed_canary = nullptr;
    uint32_t* d_packed_canary = nullptr;
    uint32_t* h_diag_canary = nullptr;
    uint32_t* d_diag_canary = nullptr;

    CUDA_CHECK(cudaMalloc(&d_byte, B));
    CUDA_CHECK(cudaMalloc(&d_W, h_W_clean.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_Wsc, h_Wsc.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wsu2, h_wsu2.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_kvk, h_kvk.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_kvv, h_kvv.size() * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_ab, h_ab.size() * sizeof(__half2)));
    CUDA_CHECK(cudaMalloc(&d_rng, h_rng.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_U, h_U.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_hid, B * 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, B * 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outb, B));
    CUDA_CHECK(cudaMalloc(&d_outs, B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sys_cnt, sizeof(uint32_t)));
    CUDA_CHECK(cudaHostAlloc(&h_packed_canary, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_packed_canary, h_packed_canary, 0));
    CUDA_CHECK(cudaHostAlloc(&h_diag_canary, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_diag_canary, h_diag_canary, 0));

    CUDA_CHECK(cudaMemcpy(d_byte, h_byte.data(), B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W_clean.data(), h_W_clean.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wsc, h_Wsc.data(), h_Wsc.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wsu2, h_wsu2.data(), h_wsu2.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvk, h_kvk.data(), h_kvk.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kvv, h_kvv.data(), h_kvv.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ab, h_ab.data(), h_ab.size() * sizeof(__half2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rng, h_rng.data(), h_rng.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Poison output buffers with NaN (0xFFFFFFFF) before every checked launch
    auto poison_outputs = [&]() {
        CUDA_CHECK(cudaMemset(d_hid, 0xFF, B * 512 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_logits, 0xFF, B * 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_outs, 0xFF, B * sizeof(float)));
    };

    // ------------------------------------------------------------------------
    // 1. Clean run + FP64 parity of 1 KiB FP16 channel-folded FWHT_256 x bent4
    // ------------------------------------------------------------------------
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison_outputs();
    holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs,
        d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, L, 1, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_hid_clean(512), h_logits_clean(256);
    CUDA_CHECK(cudaMemcpy(h_hid_clean.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logits_clean.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    uint32_t clean_packed = *h_packed_canary;
    uint32_t clean_sys_cnt = 0u;
    CUDA_CHECK(cudaMemcpy(&clean_sys_cnt, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    double folded_fp64_max_err = dense_folded1k_fwht256_max_err(h_hid_clean, h_ab, h_logits_clean, 1.0);

    // ------------------------------------------------------------------------
    // 2. 16/16 Single-Bit SEC Recovery (all 16 bit positions 0..15 simultaneously!)
    //    Injects 1 bit flip into each of the 16 warps in layer 0 -> 16 SEC in 1 CTA!
    // ------------------------------------------------------------------------
    std::vector<uint4> h_W_sec16 = h_W_clean;
    for (int bit = 0; bit < 16; ++bit) {
        flip_lsb_in_uint4(h_W_sec16[bit * 64 + 0], bit);
    }
    CUDA_CHECK(cudaMemcpy(d_W, h_W_sec16.data(), h_W_sec16.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison_outputs();
    holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs,
        d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, L, 1, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_hid_sec16(512), h_logits_sec16(256);
    CUDA_CHECK(cudaMemcpy(h_hid_sec16.data(), d_hid, 512 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logits_sec16.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    double sec16_hid_err = 0.0, sec16_log_err = 0.0;
    for (int i = 0; i < 512; ++i) {
        sec16_hid_err = update_max_err_nan_safe(sec16_hid_err, double(h_hid_clean[i]), double(h_hid_sec16[i]));
    }
    for (int i = 0; i < 256; ++i) {
        sec16_log_err = update_max_err_nan_safe(sec16_log_err, double(h_logits_clean[i]), double(h_logits_sec16[i]));
    }
    uint32_t sec16_packed = *h_packed_canary;
    uint32_t sec16_sys_cnt = 0u;
    CUDA_CHECK(cudaMemcpy(&sec16_sys_cnt, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // 3. 120/120 Double-Bit DED Pair Detection (all C(16,2)=120 pairs in 1 CTA!)
    //    Distribute 120 distinct 2-bit pairs across 16 warps x lanes 0..7 in layer 0
    // ------------------------------------------------------------------------
    std::vector<uint4> h_W_ded120 = h_W_clean;
    int pair_idx = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = i + 1; j < 16; ++j) {
            int w = pair_idx & 15;
            int ln = pair_idx >> 4; // 0..7
            flip_lsb_in_uint4(h_W_ded120[w * 64 + ln], i);
            flip_lsb_in_uint4(h_W_ded120[w * 64 + ln], j);
            ++pair_idx;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_W, h_W_ded120.data(), h_W_ded120.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
    poison_outputs();
    holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512>>>(
        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
        d_U, d_hid, d_logits, d_outb, d_outs,
        d_packed_canary, d_diag_canary, d_sys_cnt,
        cayley_t, temperature, L, 1, B
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t ded120_packed = *h_packed_canary;
    uint32_t ded120_sys_cnt = 0u;
    CUDA_CHECK(cudaMemcpy(&ded120_sys_cnt, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // 4. Fail-Closed Single-CTA AND Multi-CTA Saturation / Overflow Probes on GPU
    // ------------------------------------------------------------------------
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    holosphere_smem_canary::holosphere_smem_canary_overflow_probe_sm86<<<1, 32>>>(
        70000u, 0u, 0x15u, d_packed_canary, d_diag_canary
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t ovf_sec_packed = *h_packed_canary;
    uint32_t ovf_sec_diag = *h_diag_canary;

    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    holosphere_smem_canary::holosphere_smem_canary_overflow_probe_sm86<<<1, 32>>>(
        12u, 70000u, 0x22u, d_packed_canary, d_diag_canary
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t ovf_ded_packed = *h_packed_canary;
    uint32_t ovf_ded_diag = *h_diag_canary;

    // 4c. Multi-CTA SEC carry defense: 2 concurrent CTAs each adding sec=40000, ded=0 (sum=80000 > 65535)
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    holosphere_smem_canary::holosphere_smem_canary_overflow_probe_sm86<<<2, 32>>>(
        40000u, 0u, 0x15u, d_packed_canary, d_diag_canary
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t multi_cta_sec_carry_packed = *h_packed_canary;
    uint32_t multi_cta_sec_carry_diag = *h_diag_canary;

    // 4d. Multi-CTA DED bit-31 wrap-to-zero defense: 2 concurrent CTAs each adding sec=7, ded=0x8000 (sum=65536)
    *h_packed_canary = 0u;
    *h_diag_canary = 0u;
    holosphere_smem_canary::holosphere_smem_canary_overflow_probe_sm86<<<2, 32>>>(
        7u, 0x8000u, 0x22u, d_packed_canary, d_diag_canary
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t multi_cta_ded_wrap_packed = *h_packed_canary;
    uint32_t multi_cta_ded_wrap_diag = *h_diag_canary;

    // ------------------------------------------------------------------------
    // 5. Brownout Storm Sweep (0%, 1%, 10%, 100% fault rate across 16 warps x 12 layers)
    //    Measures Warp-Level vs CTA-SMEM: exact SEC/DED recovery, .sys atomics, latency (us/tok)
    // ------------------------------------------------------------------------
    const int rates[4] = {0, 1, 10, 100};
    const int storm_iters = 4096;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    std::vector<StormMetrics> storms;

    // Warm GPU clocks cleanly to P0 BEFORE any latency timing (eliminates P8->P0 ramp bias)
    CUDA_CHECK(cudaMemcpy(d_W, h_W_clean.data(), h_W_clean.size() * sizeof(uint4), cudaMemcpyHostToDevice));
    for (int w = 0; w < 512; ++w) {
        holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512>>>(
            d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
            d_U, d_hid, nullptr, d_outb, d_outs,
            d_packed_canary, nullptr, nullptr,
            cayley_t, temperature, L, 1, B
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int ri = 0; ri < 4; ++ri) {
        int rpct = rates[ri];
        std::vector<uint4> h_W_storm = h_W_clean;
        uint32_t exp_sec = 0u, exp_ded = 0u;
        uint64_t srng = 0xB0070000ULL + uint64_t(rpct) * 1009ULL;
        for (int l = 0; l < L; ++l) {
            for (int w = 0; w < 16; ++w) {
                for (int ln = 0; ln < 32; ++ln) {
                    int roll = int(splitmix64(srng) % 100ULL);
                    if (roll < rpct) {
                        // Exercise BOTH w_lo (+0) and w_hi (+32) fault paths across 75% SEC / 25% DED
                        int hi_off = ((splitmix64(srng) & 1ULL) != 0ULL) ? 32 : 0;
                        int base_idx = (l * 16 + w) * 64 + hi_off + ln;
                        if ((splitmix64(srng) & 3ULL) != 0ULL) {
                            flip_lsb_in_uint4(h_W_storm[base_idx], int(splitmix64(srng) & 15ULL));
                            exp_sec += 1u;
                        } else {
                            int b0 = int(splitmix64(srng) & 15ULL);
                            int b1 = (b0 + 1 + int(splitmix64(srng) % 15ULL)) & 15;
                            flip_lsb_in_uint4(h_W_storm[base_idx], b0);
                            flip_lsb_in_uint4(h_W_storm[base_idx], b1);
                            exp_ded += 1u;
                        }
                    }
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_W, h_W_storm.data(), h_W_storm.size() * sizeof(uint4), cudaMemcpyHostToDevice));

        // 5a. Single-step exact verification (Warp-Level)
        *h_packed_canary = 0u;
        *h_diag_canary = 0u;
        CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
        holosphere_smem_canary::holosphere_cell_unfolded_warp_canary_sm86<<<B, 512>>>(
            d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_rng,
            d_U, d_hid, d_logits, d_outb, d_outs,
            d_packed_canary, d_diag_canary, d_sys_cnt,
            cayley_t, temperature, L, 1, B
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t warp_pk = *h_packed_canary;
        uint32_t warp_sc = 0u;
        CUDA_CHECK(cudaMemcpy(&warp_sc, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // 5b. Single-step exact verification (CTA-SMEM)
        *h_packed_canary = 0u;
        *h_diag_canary = 0u;
        CUDA_CHECK(cudaMemset(d_sys_cnt, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float4), cudaMemcpyHostToDevice));
        holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512>>>(
            d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
            d_U, d_hid, d_logits, d_outb, d_outs,
            d_packed_canary, d_diag_canary, d_sys_cnt,
            cayley_t, temperature, L, 1, B
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t smem_pk = *h_packed_canary;
        uint32_t smem_sc = 0u;
        CUDA_CHECK(cudaMemcpy(&smem_sc, d_sys_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // 5c. CUDA-Graph latency benchmark (NULL instrumentation counter for true production speed)
        cudaStream_t st;
        CUDA_CHECK(cudaStreamCreate(&st));
        auto time_graph = [&](bool use_smem_fold) -> double {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            *h_packed_canary = 0u;
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            for (int k = 0; k < 64; ++k) {
                if (use_smem_fold) {
                    holosphere_smem_canary::holosphere_cell_folded1k_smem_canary_sm86<<<B, 512, 0, st>>>(
                        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_ab, d_rng,
                        d_U, d_hid, nullptr, d_outb, d_outs,
                        d_packed_canary, nullptr, nullptr,
                        cayley_t, temperature, L, 1, B
                    );
                } else {
                    holosphere_smem_canary::holosphere_cell_unfolded_warp_canary_sm86<<<B, 512, 0, st>>>(
                        d_byte, d_W, d_Wsc, d_wsu2, d_kvk, d_kvv, d_rng,
                        d_U, d_hid, nullptr, d_outb, d_outs,
                        d_packed_canary, nullptr, nullptr,
                        cayley_t, temperature, L, 1, B
                    );
                }
            }
            CUDA_CHECK(cudaStreamEndCapture(st, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            for (int w = 0; w < 8; ++w) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            int n_launches = storm_iters / 64;
            CUDA_CHECK(cudaEventRecord(ev0, st));
            for (int it = 0; it < n_launches; ++it) CUDA_CHECK(cudaGraphLaunch(ge, st));
            CUDA_CHECK(cudaEventRecord(ev1, st));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            CUDA_CHECK(cudaGraphExecDestroy(ge));
            CUDA_CHECK(cudaGraphDestroy(g));
            return (double(ms) * 1000.0) / double(n_launches * 64);
        };

        // Interleaved 3-trial median to eliminate thermal/clock drift bias
        double warp_trials[3], smem_trials[3];
        for (int tr = 0; tr < 3; ++tr) {
            warp_trials[tr] = time_graph(false);
            smem_trials[tr] = time_graph(true);
        }
        auto med3 = [](double a, double b, double c) -> double {
            if (a > b) { double t = a; a = b; b = t; }
            if (b > c) { double t = b; b = c; c = t; }
            if (a > b) { double t = a; a = b; b = t; }
            return b;
        };
        double warp_us = med3(warp_trials[0], warp_trials[1], warp_trials[2]);
        double smem_us = med3(smem_trials[0], smem_trials[1], smem_trials[2]);
        CUDA_CHECK(cudaStreamDestroy(st));

        storms.push_back(StormMetrics{
            rpct, exp_sec, exp_ded,
            warp_pk & 0xFFFFu, warp_pk >> 16, warp_sc,
            smem_pk & 0xFFFFu, smem_pk >> 16, smem_sc,
            warp_us, smem_us
        });
    }

    // ------------------------------------------------------------------------
    // 6. Standalone LM-Head Latency Microbenchmark (Unfolded vs 1 KiB FP16 Fold)
    // ------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_hid, h_hid_clean.data(), 512 * sizeof(float), cudaMemcpyHostToDevice));
    const int head_iters = 16384;
    cudaStream_t hst;
    CUDA_CHECK(cudaStreamCreate(&hst));
    auto time_head = [&](bool folded) -> double {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(hst, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < 128; ++k) {
            if (folded) {
                holosphere_smem_canary::holosphere_lm_head_folded1k_sm86<<<1, 512, 0, hst>>>(d_hid, d_ab, d_logits, 1.0f);
            } else {
                holosphere_smem_canary::holosphere_lm_head_unfolded_sm86<<<1, 512, 0, hst>>>(d_hid, d_logits, 1.0f);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(hst, &g));
        CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 8; ++w) CUDA_CHECK(cudaGraphLaunch(ge, hst));
        CUDA_CHECK(cudaStreamSynchronize(hst));
        int n_launches = head_iters / 128;
        CUDA_CHECK(cudaEventRecord(ev0, hst));
        for (int it = 0; it < n_launches; ++it) CUDA_CHECK(cudaGraphLaunch(ge, hst));
        CUDA_CHECK(cudaEventRecord(ev1, hst));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        CUDA_CHECK(cudaGraphExecDestroy(ge));
        CUDA_CHECK(cudaGraphDestroy(g));
        return (double(ms) * 1000.0) / double(n_launches * 128);
    };
    double hu_t[3], hf_t[3];
    for (int tr = 0; tr < 3; ++tr) {
        hu_t[tr] = time_head(false);
        hf_t[tr] = time_head(true);
    }
    auto med3h = [](double a, double b, double c) -> double {
        if (a > b) { double t = a; a = b; b = t; }
        if (b > c) { double t = b; b = c; c = t; }
        if (a > b) { double t = a; a = b; b = t; }
        return b;
    };
    double head_unfolded_us = med3h(hu_t[0], hu_t[1], hu_t[2]);
    double head_folded1k_us = med3h(hf_t[0], hf_t[1], hf_t[2]);
    CUDA_CHECK(cudaStreamDestroy(hst));

    // Verify standalone folded head FP64 parity against h_hid_clean (poisoned output buffer)
    CUDA_CHECK(cudaMemset(d_logits, 0xFF, 256 * sizeof(float)));
    holosphere_smem_canary::holosphere_lm_head_folded1k_sm86<<<1, 512>>>(d_hid, d_ab, d_logits, 1.25f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_stand_logits(256);
    CUDA_CHECK(cudaMemcpy(h_stand_logits.data(), d_logits, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    double stand_head_fp64_err = dense_folded1k_fwht256_max_err(h_hid_clean, h_ab, h_stand_logits, 1.25);

    // Emit canonical latency/parity/atomic telemetry JSON to stdout
    std::printf("{\n");
    std::printf("  \"device\": \"%s\",\n", prop.name);
    std::printf("  \"sm\": \"%d.%d\",\n", prop.major, prop.minor);
    std::printf("  \"host_native_atomic_supported\": %d,\n", (host_native_atomic_attr != 0) ? 1 : 0);
    std::printf("  \"clean_packed_u32\": %u,\n", clean_packed);
    std::printf("  \"clean_sys_atomics\": %u,\n", clean_sys_cnt);
    std::printf("  \"folded1k_cell_fp64_max_err\": %.9e,\n", folded_fp64_max_err);
    std::printf("  \"folded1k_head_fp64_max_err\": %.9e,\n", stand_head_fp64_err);
    std::printf("  \"sec16_packed_u32\": %u,\n", sec16_packed);
    std::printf("  \"sec16_sec_count\": %u,\n", sec16_packed & 0xFFFFu);
    std::printf("  \"sec16_ded_count\": %u,\n", sec16_packed >> 16);
    std::printf("  \"sec16_sys_atomics\": %u,\n", sec16_sys_cnt);
    std::printf("  \"sec16_hidden_max_abs_err\": %.9e,\n", sec16_hid_err);
    std::printf("  \"sec16_logits_max_abs_err\": %.9e,\n", sec16_log_err);
    std::printf("  \"ded120_packed_u32\": %u,\n", ded120_packed);
    std::printf("  \"ded120_sec_count\": %u,\n", ded120_packed & 0xFFFFu);
    std::printf("  \"ded120_ded_count\": %u,\n", ded120_packed >> 16);
    std::printf("  \"ded120_sys_atomics\": %u,\n", ded120_sys_cnt);
    std::printf("  \"overflow_sec70000_packed_u32\": %u,\n", ovf_sec_packed);
    std::printf("  \"overflow_sec70000_diag_u32\": %u,\n", ovf_sec_diag);
    std::printf("  \"overflow_ded70000_packed_u32\": %u,\n", ovf_ded_packed);
    std::printf("  \"overflow_ded70000_diag_u32\": %u,\n", ovf_ded_diag);
    std::printf("  \"multi_cta_sec_carry_packed_u32\": %u,\n", multi_cta_sec_carry_packed);
    std::printf("  \"multi_cta_sec_carry_diag_u32\": %u,\n", multi_cta_sec_carry_diag);
    std::printf("  \"multi_cta_ded_wrap_packed_u32\": %u,\n", multi_cta_ded_wrap_packed);
    std::printf("  \"multi_cta_ded_wrap_diag_u32\": %u,\n", multi_cta_ded_wrap_diag);
    std::printf("  \"standalone_head_unfolded_us\": %.6f,\n", head_unfolded_us);
    std::printf("  \"standalone_head_folded1k_us\": %.6f,\n", head_folded1k_us);
    std::printf("  \"standalone_head_overhead_us\": %.6f,\n", head_folded1k_us - head_unfolded_us);
    std::printf("  \"storms\": [\n");
    for (size_t i = 0; i < storms.size(); ++i) {
        const auto& s = storms[i];
        std::printf(
            "    {\"rate_pct\": %d, \"expected_sec\": %u, \"expected_ded\": %u, "
            "\"warp_sec\": %u, \"warp_ded\": %u, \"warp_sys_atomics\": %u, "
            "\"smem_sec\": %u, \"smem_ded\": %u, \"smem_sys_atomics\": %u, "
            "\"warp_us_per_tok\": %.6f, \"smem_us_per_tok\": %.6f}%s\n",
            s.rate_pct, s.expected_sec, s.expected_ded,
            s.warp_sec, s.warp_ded, s.warp_sys_atomics,
            s.smem_sec, s.smem_ded, s.smem_sys_atomics,
            s.warp_us_per_tok, s.smem_us_per_tok,
            (i + 1 < storms.size()) ? "," : ""
        );
    }
    std::printf("  ]\n");
    std::printf("}\n");

    // Fail-closed exit status if any parity, SEC/DED, single/multi-CTA overflow, or storm invariant failed
    bool ok = (clean_packed == 0u) && (clean_sys_cnt == 0u)
        && std::isfinite(folded_fp64_max_err) && (folded_fp64_max_err < 1e-5)
        && std::isfinite(stand_head_fp64_err) && (stand_head_fp64_err < 1e-5)
        && ((sec16_packed & 0xFFFFu) == 16u) && ((sec16_packed >> 16) == 0u) && (sec16_sys_cnt == 1u)
        && std::isfinite(sec16_hid_err) && (sec16_hid_err == 0.0)
        && std::isfinite(sec16_log_err) && (sec16_log_err == 0.0)
        && ((ded120_packed & 0xFFFFu) == 0u) && ((ded120_packed >> 16) == 120u) && (ded120_sys_cnt == 1u)
        && (ovf_sec_packed == 0x0001FFFFu) && (ovf_sec_diag == 0x80000015u)
        && (ovf_ded_packed == 0xFFFF000Cu) && (ovf_ded_diag == 0x80000022u)
        && (multi_cta_sec_carry_packed == 0x0001FFFFu) && (multi_cta_sec_carry_diag == 0x80000015u)
        && (multi_cta_ded_wrap_packed == 0xFFFF000Eu) && (multi_cta_ded_wrap_diag == 0x80000022u);
    for (const auto& s : storms) {
        if (s.smem_sec != s.expected_sec || s.smem_ded != s.expected_ded
            || s.warp_sec != s.expected_sec || s.warp_ded != s.expected_ded
            || (s.rate_pct == 0 && s.smem_sys_atomics != 0u)
            || (s.rate_pct > 0 && s.smem_sys_atomics != 1u)) {
            ok = false;
        }
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFreeHost(h_packed_canary));
    CUDA_CHECK(cudaFreeHost(h_diag_canary));
    CUDA_CHECK(cudaFree(d_byte));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Wsc));
    CUDA_CHECK(cudaFree(d_wsu2));
    CUDA_CHECK(cudaFree(d_kvk));
    CUDA_CHECK(cudaFree(d_kvv));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_hid));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_outb));
    CUDA_CHECK(cudaFree(d_outs));
    CUDA_CHECK(cudaFree(d_sys_cnt));
    return ok ? 0 : 3;
}
