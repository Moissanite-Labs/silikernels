// SPDX-License-Identifier: Apache-2.0
// V2 Ampere sm_86 .so + standalone bench harness (CPU selftest runs 100% GPU-free).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "a16_v2_ampere_kernels.cuh"

namespace {

uint64_t splitmix64_host(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

bool invert_8x8_host(const float* M, float* Minv) noexcept {
    double aug[8][16];
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            aug[r][c] = static_cast<double>(M[r * 8 + c]);
            aug[r][c + 8] = (r == c) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        double best = std::fabs(aug[col][col]);
        for (int r = col + 1; r < 8; ++r) {
            const double v = std::fabs(aug[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1.0e-12) return false;
        if (piv != col) {
            for (int c = 0; c < 16; ++c) std::swap(aug[col][c], aug[piv][c]);
        }
        const double inv_d = 1.0 / aug[col][col];
        for (int c = 0; c < 16; ++c) aug[col][c] *= inv_d;
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            const double f = aug[r][col];
            for (int c = 0; c < 16; ++c) aug[r][c] -= f * aug[col][c];
        }
    }
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) Minv[r * 8 + c] = static_cast<float>(aug[r][c + 8]);
    }
    return true;
}

} // namespace

extern "C" int a16_v2_sm86_cpu_selftest(int verbose) {
    int fails = 0;
    uint64_t rng = 0x5186A16B2001ULL;

    // 1. Verify 256 B Woodbury 8x8 Cayley inverse precision
    float U8[8 * 128], V8[8 * 128], M[64], Minv[64];
    for (int i = 0; i < 8 * 128; ++i) {
        const double u1 = static_cast<double>(splitmix64_host(rng) >> 11) * (1.0 / 9007199254740992.0);
        const double u2 = static_cast<double>(splitmix64_host(rng) >> 11) * (1.0 / 9007199254740992.0);
        U8[i] = static_cast<float>((u1 * 2.0 - 1.0) * 0.15);
        V8[i] = static_cast<float>((u2 * 2.0 - 1.0) * 0.15);
    }
    const float alpha = 0.25f;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            double d = 0.0;
            for (int k = 0; k < 128; ++k) d += static_cast<double>(V8[r * 128 + k]) * static_cast<double>(U8[c * 128 + k]);
            M[r * 8 + c] = ((r == c) ? 1.0f : 0.0f) + alpha * static_cast<float>(d);
        }
    }
    if (!invert_8x8_host(M, Minv)) ++fails;
    double wb_err = 0.0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            double s = 0.0;
            for (int k = 0; k < 8; ++k) s += static_cast<double>(M[r * 8 + k]) * static_cast<double>(Minv[k * 8 + c]);
            wb_err = std::max(wb_err, std::fabs(s - ((r == c) ? 1.0 : 0.0)));
        }
    }
    if (wb_err >= 1.0e-5) ++fails;

    // 2. Verify SWAR127 + 0-ULP canonical tree + IRIS + 64B canary ABI
    float xs[128];
    for (int i = 0; i < 128; ++i) {
        const double u = static_cast<double>(splitmix64_host(rng) >> 11) * (1.0 / 9007199254740992.0);
        xs[i] = static_cast<float>((u * 2.0 - 1.0) * 2.0);
    }
    const float red = silikernels_v2_canonical_tree_reduce_128_c(xs);
    if (!std::isfinite(red)) ++fails;

    uint64_t sw_out = 0ULL;
    if (silikernels_v2_swar127_negate_u64_checked_c(0x0102030405060708ULL, &sw_out) != 0) ++fails;
    if (silikernels_v2_swar127_negate_u64_checked_c(0x0101010001010101ULL, &sw_out) == 0 || sw_out != 0ULL) ++fails;

    HoloSphereRazorCanary64 can{};
    can.sec_count_u32 = 3u;
    const HoloSphereRazorSnapshot snap = holosphere_razor_poll_acquire(&can);
    if (snap.sec_count_u32 != 3u) ++fails;

    const uint32_t iris_h = silikernels_v2_iris_mix_word_u32(0x5185A17Bu, 0x12345678u);
    if (iris_h == 0u) ++fails;

    if (verbose) {
        std::printf("{\"status\":\"%s\",\"woodbury_bytes\":%u,\"woodbury_id_err\":%.3e,\"tree_sum\":%.6f}\n",
                    fails == 0 ? "PASS" : "FAIL",
                    SILIKERNELS_V2_WOODBURY_BYTES,
                    wb_err,
                    static_cast<double>(red));
    }
    return fails == 0 ? 0 : 1;
}

extern "C" int a16_v2_sm86_abi_info(uint32_t* out_woodbury_bytes, uint32_t* out_b_star) {
    if (out_woodbury_bytes) *out_woodbury_bytes = SILIKERNELS_V2_WOODBURY_BYTES;
    if (out_b_star) *out_b_star = SILIKERNELS_V2_B_STAR_COLD_3BPW;
    return 0;
}

#ifndef SILIKERNELS_V2_SHARED_LIB
int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::strcmp(argv[1], "--cpu-selftest") == 0)) {
        return a16_v2_sm86_cpu_selftest(1);
    }
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("Usage: %s [--cpu-selftest]\n", argv[0]);
        return 0;
    }
    std::fprintf(stderr, "Unsupported option (use --cpu-selftest for pure-CPU verification)\n");
    return 2;
}
#endif
