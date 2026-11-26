# Bare-Metal Intel AVX-512 VNNI L3-Resident `HoloSphere-V3 15M` CPU Engine (`Track S125` / `EXP-Q30b`)

Zero-dependency C11 + Intel AVX-512 (`VPDPBUSD`, `VPOPCNTQ`, `VPERMT2D`) inference engine implementing the full `HoloSphere-V3 15M` architecture (`D=512, H=16, head_dim=32, L=6, d_ff=800, V=256` tied `FWHT_256 x bent4` `0 B` LM head + `ByteConv4` + `V_ring` + SFU-free `CayleySU2` + `S_fast` Woodbury + `SwiGLU`) strictly inside the `24 MiB` L3 cache of an Intel Core `i9-11900H`.

## 1. Key Design & Honesty Scope

- **L3-Resident Arena (`9.010 MiB` = `37.541%` of `24 MiB` L3):** `8,837,120 B` (`8.428 MiB`) weights + `610,304 B` (`0.582 MiB`) $O(1)$ persistent recurrent state (`0 B/tok` KV growth) = **`9,447,424 B` (`9.010 MiB`)**, strictly under the `PREREG-W14` `< 12 MiB` gate (`PREREG-W14` `SHA-256 = 871900e5d6c860ac159f43ca6b4767e7c128d0b4f7748edf788d56e9cb535d3e`).
- **Bit-Exact `A_16 Block-H4` `VPDPBUSD` Equivalence:** Pre-applying symmetric orthogonal `H_4` once to activation `x` and quantizing per 32-element group to signed `int8` (`xq in [-127, +127]`) enables bias-shifting signed `int4` nibbles `c in [-8, +7]` to unsigned `u = c + 8 in [0, 15]` via one `vpxorq(..., 0x88)` and accumulating `64` exact `u8 * s8` MACs per `_mm512_dpbusd_epi32` (`vpdpbusd`), subtracting `8 * sum(xq[g32])` for bit-exact `0.0`-ULP parity between scalar C11 and AVX-512 VNNI across `1T`, `4T`, and `8T`.
- **`[16, 11, 4]` Nibble-LSB Bit-Plane SECDED Scope (Razor Parity):** Inline `_mm512_popcnt_epi64` (`vpopcntq`) verifies the `[16, 11, 4]` Extended Hamming syndrome over the **`16`-bit packed nibble-LSB control plane** (`w64 & 0x1111111111111111ULL`, i.e., `1/4` of payload bits, matching `holosphere#3` & `silikernels#4` Razor bit-plane canary semantics). Live self-test certifies `16/16` single-bit LSB bit-exact repair and `120/120` double-bit LSB detection; nibble upper bits `1..3` (`3/4` of weight payload bits) are unguarded payload bits.
- **Hardened Strict CLI:** Validates every integer flag via `endptr`/`errno`/sign-checked strict parsing (`--threads 1..16` emitting sanitized `pool.n_threads`, `--tokens >= 1`, `--warmup >= 0`, `--hugepages 0|1`, unsigned `--seed`), validates `--mode` (`bench|trace`) and `--backend` (`avx512|scalar`), checks `pthread_create` return codes with clean worker teardown, and returns exit code `5` on non-zero runtime syndrome faults in both `trace` and `bench` modes.

## 2. Live `i9-11900H` Bare-Metal Receipt Summary (`receipts/s125_avx512_l3_benchmark_results.json`)

Measured CPU-only (`CUDA_VISIBLE_DEVICES=""`) on `11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz` (`8C/16T`, `24 MiB` L3), `gcc (GCC) 16.2.1 20260810`, Linux `7.2.15-1-omarchy` (`THP=[always]`, `defrag=[madvise]`, explicit `MADV_NOHUGEPAGE` vs `MADV_HUGEPAGE` verified via `/proc/self/smaps`):

| Gate / Comparison | Measured Result | Verdict |
|---|---|---|
| Build & Sanitizers | `gcc -O3 -march=native -std=c11 -Wall -Wextra -Werror` (`0` warnings); `-fsanitize=undefined,address` (`0` errors); `objdump -d`: `vpdpbusd=4`, `vpopcntq=10`, `vpermt2d=7` | **`PASS`** |
| `G1` Exact FP64 Parity & Determinism | Independent full-`15M` FP64 oracle: `max_abs_logit_diff = 1.07937e-7`, `max_abs_hidden_diff = 1.5034e-8` (`<= 1e-5`), `8/8` greedy match; `0.0` ULP bit-exact across `scalar/avx512/1T/4T/8T/2M/UBSan` (`logits=6fc017bea5754bb1`, `hidden=3a65e5960be737a7`); `0` pack/runtime faults | **`CONFIRMED`** |
| `12`-Run A/A Control Noise Floor (`4 KiB`, `0 kB` THP) | `1T`: median `544.74 tok/s` (A/A peak `576.71`), `CV=3.198%`, `IQR=4.821%`, floor **`6.397%`** (`3/10` naive A/B/A false wins); `8T`: median `1,027.54 tok/s` (A/A peak `1,071.10`), `CV=9.279%`, `IQR=2.904%`, floor **`18.557%`** (`3/10` naive A/B/A false wins) | Calibrated |
| `G2` `scalar` vs `avx512` (`1T`, `4 KiB`, `6` triples) | `141.575 -> 545.750 tok/s` = **`3.8548x`** (`+285.48%` median, `+286.46%` paired, **`6/6` wins**, interleaved `1T` peak `578.24 tok/s`) | **`CONFIRMED`** |
| `G2` `scalar` vs `avx512` (`8T`, `4 KiB`, `6` triples) | `439.780 -> 1,021.970 tok/s` = **`2.3238x`** (`+132.38%` median, `+134.44%` paired, **`6/6` wins**, interleaved `8T` `4 KiB` peak `1,076.13 tok/s`) | **`CONFIRMED`** |
| `4 KiB` (`0 kB`) vs `2 MiB` (`2,048 kB` med / `4,096 kB` trace) (`1T`, `6` triples) | `569.475 -> 568.760 tok/s` (`-0.13%` median, `+0.07%` paired, `1/6` wins vs `6.397%` A/A floor; `4 KiB` peak `575.75` / `2 MiB` peak `573.64 tok/s`) | **`NEUTRAL / L3-ABSORBED`** |
| `4 KiB` (`0 kB`) vs `2 MiB` (`2,048 kB` med / `4,096 kB` trace) (`8T`, `6` triples) | `1,069.615 -> 1,057.405 tok/s` (`-1.14%` median, `+1.62%` paired, `1/6` wins vs `18.557%` A/A floor; `4 KiB` peak `1,104.90` / `2 MiB` peak `1,129.27 tok/s`) | **`NEUTRAL / L3-ABSORBED`** |

- **Non-Iso-Parameter Architectural Context:** For footprint context, `llm.asm/02-baremetal-os` (`Qwen3-0.6B`, `134.94 tok/s` KVM) spans `614 MiB` weights + `449 MiB` KV = `1,063 MiB` (`1.038 GiB`, `> 272,000` `4 KiB` pages) where `2 MiB` huge pages yield `+10%` by avoiding DRAM-scale TLB walks. Comparing `15M` `HoloSphere-V3` against `0.6B` `Qwen3-0.6B` is **strictly non-iso-parameter**; the architectural takeaway is that once a `9.010 MiB` recurrent model resides completely inside a `24 MiB` L3 cache (`~2,307` `4 KiB` pages), `2 MiB` `MADV_HUGEPAGE` backing is **`NEUTRAL / L3-ABSORBED`** within the empirical A/A noise floor (observed consistently at both `20%` and `100%` PMD backing), while `AVX-512 VNNI` delivers `3.85x` (`1T`) / `2.32x` (`8T`) speedup over scalar C11. On fragmented live Linux hosts (`buddyinfo` `zone Normal` `order-9 = 0`, `HugePages_Total = 0`), unprivileged `MADV_HUGEPAGE` promotion varies stochastically (`0%`–`100%`) across runs unless root pre-reserves `vm.nr_hugepages`.

## 3. Build & Verification

```bash
# Build native AVX-512 binary (0 warnings under -Wall -Wextra -Werror)
make cpu-holosphere-avx512

# Run 3-mode isolated verification (-I -B, -I -O -B, -I -B -W error) + R5 tamper suite
make verify-holosphere-avx512
```
