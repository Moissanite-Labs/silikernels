# Silikernels

**Bare-metal zero-spill Ampere `sm_86` PTX/SASS and Intel `AVX-512` (VBMI / VNNI / VPOPCNTDQ) kernel rewrites for `A_16` Extended Hamming `[16, 11, 4]` lattice quantization, Block-H4 Walsh–Hadamard fusion, and full-model LLM inference.**

Developed within **Moissanite Labs** alongside [Moissanite-Labs/siliquant](https://github.com/Moissanite-Labs/siliquant) and [Foundation-0/aletheia](https://github.com/Foundation-0/aletheia). Every kernel below compiles with **zero compiler warnings** and **zero register spills (`0 B` stack frame, `0 B` spill stores/loads, `0` SASS `LDL`/`STL`)** on NVIDIA Ampere `sm_86` (`RTX 3080 Laptop GPU`, `GA104M`, `8 GiB` `256-bit` GDDR6, `356.68 GB/s` STREAM Triad) and Intel `i9-11900H` (`8C/16T` Tiger Lake-H `AVX-512`), backed by verified Ring-0 `mlos` Zone-B provenance receipts.

---

## Verified Silicon Records

| Kernel Target | Architecture | Headline Verified Result | Verification & Provenance |
| :--- | :--- | :--- | :--- |
| **[`cuda/sm86/qwen25_sm86_a16_engine.cu`](cuda/sm86/qwen25_sm86_a16_engine.cu)** + **[`cpu/avx512/qwen25_avx512_a16_engine.c`](cpu/avx512/qwen25_avx512_a16_engine.c)** | Ampere `sm_86` PTX + `AVX-512` | **Full-model `Qwen/Qwen2.5-0.5B` (`357.8M` quantized weights):** **`571.37 tok/s`** 24-layer backbone (**`7.43×`** vs. PyTorch `BF16`, **`6.15×`** less energy), **`400.58 tok/s`** full E2E w/ `151,936`-vocab `BF16` `lm_head` (**`5.60×`** vs. `BF16` `71.58 tok/s`) at **`312.79 mJ/tok`** (**`4.91×`** less combined GPU+CPU energy), **`91.78 ms`** cold-to-hot load, **`31.70 tok/s`** zero-VRAM CPU fallback (`16/16` exact greedy GPU-vs-CPU match) | `16/16` kernels `0 B` spill, `0` syndrome errors · [`siliquant#88`](https://github.com/Moissanite-Labs/siliquant/pull/88) · [`receipts/s58_qwen25_live_engine_results.json`](receipts/s58_qwen25_live_engine_results.json) |
| **[`cuda/sm86/aletheion_sm86_ptx_foundry.cu`](cuda/sm86/aletheion_sm86_ptx_foundry.cu)** | Ampere `sm_86` PTX/SASS | **`646.51–648.64 Gweights/s`** (`102.03–102.41%` STREAM Triad, **`1.293 TB/s` FP16-equiv**) fused `4.5 bpw` `A_16` `LDG.128 + LOP3 + PRMT + dp4a.u32.s32` GEMV; live `5-POPC` syndrome verify at `643.29 Gw/s` (`0.4%` overhead); zero-shmem warp-shuffle `WHT_8`/`H_16` (`463.89 Gw/s`, `2.08×`); `O(1)` register-delta live repair (`478.46 Gw/s`, `1.74×`); batched `B=32` `cp.async + IMMA.16832` (`14.90 INT8 TOPS`, `11.52×`) | `15/15` kernels `0 B` spill, `0` verify fails · [`siliquant#83`](https://github.com/Moissanite-Labs/siliquant/pull/83) · [`receipts/s52_sm86_ptx_foundry_results.json`](receipts/s52_sm86_ptx_foundry_results.json) |
| **[`cpu/avx512/s71_proximity_bench.c`](cpu/avx512/s71_proximity_bench.c)** | Intel `AVX-512` VNNI | **`146.892 Gweights/s`** (`721.84 mJ/Gw`) multicore `A_16 + BlockH4` on `Qwen2.5-3B` `2048×11008` via `64 B` cache-line read-shared proximity fusion (**`6.982×` faster** / **`4.314×` more energy-efficient** than `8 B`, `-7.69%` `mJ/Gw` vs. `4096 B`) | `20/20` pure-`Init` `Lean 4.36.0` proofs in [`lean/ProximityBarrierSilicon.lean`](lean/ProximityBarrierSilicon.lean) · [`aletheia#71`](https://github.com/Foundation-0/aletheia/pull/71) · [`receipts/s71_proximity_avx512_results.json`](receipts/s71_proximity_avx512_results.json) |
| **[`cuda/sm86/sm86_s69_sorted_blockh4_kernel.cu`](cuda/sm86/sm86_s69_sorted_blockh4_kernel.cu)** | Ampere `sm_86` PTX | **`606.20 Gweights/s`** zero-cost folded `down_proj` + **`2.79×` faster / `3.25×` lower-energy** shared-RMSNorm fused gather for stratified column-permuted `Block-H4` | `9/9` kernels `0 B` spill · [`siliquant#93`](https://github.com/Moissanite-Labs/siliquant/pull/93) |
| **[`cuda/sm86/sm86_a16_blockh4_kernel.cu`](cuda/sm86/sm86_a16_blockh4_kernel.cu)** | Ampere `sm_86` PTX | **`592.31 Gweights/s`** (`93.08%` Triad, `639.73 mJ/tok`) fused `A_16 + Block-H4` GEMV | `7/7` kernels `0 B` spill · [`siliquant#91`](https://github.com/Moissanite-Labs/siliquant/pull/91) |
| **[`cpu/avx512/aletheion_avx512_a16_decoder.c`](cpu/avx512/aletheion_avx512_a16_decoder.c)** | Intel `AVX-512` VBMI/VNNI | **`60.95–80.21 Gweights/s`** fused `vpermb + vpdpbusd` VNNI GEMV + live `vpopcntq` verify; **`223.19 Mweights/s`** exact bit-identical `18 B/32 w` enumerative `hampack.c` (`22.3×` over `~10 Mw/s` scalar baseline) | `0` bit mismatches, `0` syndrome errors · [`siliquant#83`](https://github.com/Moissanite-Labs/siliquant/pull/83) · [`receipts/s52_avx512_a16_benchmark_results.json`](receipts/s52_avx512_a16_benchmark_results.json) |
| **[`cuda/sm86/rtx3080_live_llm_transcode_kernels.cu`](cuda/sm86/rtx3080_live_llm_transcode_kernels.cu)** & **[`cuda/sm86/rtx3080_qwen3b_kernels.cu`](cuda/sm86/rtx3080_qwen3b_kernels.cu)** | Ampere `sm_86` PTX | **`358.52 Gweights/s`** (**`727×`** speedup) cold `18 B` $\to$ hot SoA load-time transcode (`~1.41 s` for `72B`) and **`615.95 Gw/s` / `222 tok/s`** `Qwen2.5-3B` live-weight serving | `10/10` kernels `0 B` spill · [`siliquant#82`](https://github.com/Moissanite-Labs/siliquant/pull/82) · [`siliquant#86`](https://github.com/Moissanite-Labs/siliquant/pull/86) |
| **[`cuda/sm86/hf_reference_sm86_a16_blockh4.cu`](cuda/sm86/hf_reference_sm86_a16_blockh4.cu)** + **[`cpu/avx512/hf_reference_avx512_a16_blockh4.c`](cpu/avx512/hf_reference_avx512_a16_blockh4.c)** | Ampere `sm_86` + `AVX-512` | Self-contained standalone reference kernels bundled with [`siliquant-hf-v1`](https://huggingface.co/collections/ahrazzle/siliquant-stratified-lattice-quantized-llms-6b060e6988afd65942cd7768) public Hugging Face releases (`10` models live) | [`siliquant#100`](https://github.com/Moissanite-Labs/siliquant/pull/100) |
| **[`cuda/sm86/su2_holonomy_scan_sm86.cu`](cuda/sm86/su2_holonomy_scan_sm86.cu)** | Ampere `sm_86` PTX/SASS | **27 fused `SU(2)` quaternion holonomy, covariant `(Q,K,V)` transport, spherical SLERP/Hopf-$S^3$, `cayley_su2`, & Beltrami `Cayley2x2` kernels** (`float4` = 1 unit quaternion, zero-permute strided BHTD FP32/BF16, 3-float $\mathfrak{su}(2)$ additive backward suffix sum). Companion `GA104M` `RTX 3080` benchmarks: **`71.57 ms`** (`0.6313×` `RoPE-2D-nGPT`, `1.431 mJ/tok`), **`348.37 GB/s`** (`97.69%` Triad at `T=2048`), `Cayley2x2` **`9.77×`**, `Hopf-S^3+Cayley2x2` **`28.58×`** | `27/27` kernels `0 B` spill, `0` PTX `ld/st.local`, `0` SASS `LDL`/`STL` · [`receipts/su2_holonomy_sm86_ptxas_receipt.json`](receipts/su2_holonomy_sm86_ptxas_receipt.json) |

---

## HoloSphere-V3 Bare-Metal `sm_86` Megakernel, `Chunk32` Woodbury, & `W10-OBS-61` Frontier Kernels (`Refs #2`)

Extending `su2_holonomy_scan_sm86.cu` (`27/27`), the three HoloSphere `S116`/`S117`/`W10-OBS-61` kernel suites add **`23/23`** additional zero-spill Ampere `sm_86` kernels (**`50/50`** total HoloSphere `0 B`-spill kernels in `cuda/sm86/`) plus the cacheline-aligned **`64 B` `.sys` Razor telemetry & `16 B` Weft rollback ABI** ([`include/razor_abi_4byte_sys.h`](include/razor_abi_4byte_sys.h)):

| Kernel Target / Header | Scope | Live RTX 3080 (`GA104M`, `sm_86`) Zone-B Verified Result | Zero-Spill & Provenance |
| :--- | :--- | :--- | :--- |
| **[`cuda/sm86/holosphere_v3_frontier_sm86.cu`](cuda/sm86/holosphere_v3_frontier_sm86.cu)** | `W10-OBS-61` `HMMA.16816` Double-Woodbury Bwd + `v3_d512` `0`-`POPC` hot-path Dual-SWAR E2E | **`148,382.09 tok/s`** (`220.84 ms / 12L` at `B=4, T=8192, D=512`, `13,451.30 mJ` = `0.4105 mJ/tok` derived, **`2.4093×`** over Choice B `532.06 ms`; `128` regs / `6,336 B` SMEM, `28` `HMMA.16816`, `30` `SHFL`); `v3_d512` (`52` regs / `13,992 B` SMEM, `0`-`POPC` hot-path Dual-SWAR, `POPC: 14` outside per-layer hot path) CUDA-graph **`20.51 µs/tok`** CellOp (`T=8192`) / **`21.52 µs/tok`** E2E (`T=16384`, `46,467.86 tok/s`, **`1.3076 mJ/tok`**) & matched `T=8192` non-graph `21.24` / `22.25 µs/tok` (`44,946.85 tok/s`, `1.011 µs/tok` diff, `1.96e-7` `FWHT_256` err) w/ **`16/16` live `.sys` BER canary `0.0` bit-exact** (`0` clean PCIe writes, `1/1` DED) | `2/2` `0 B` spill, `0` `LDL`/`STL` · [`receipts/v3_frontier_ptxas_zero_spill_receipt.json`](receipts/v3_frontier_ptxas_zero_spill_receipt.json) · [`receipts/s117_v3_frontier_live_gpu_results.json`](receipts/s117_v3_frontier_live_gpu_results.json) · Zone-B [`7655e610…`](receipts/mlos_zone_b_s117_v3_frontier.json) |
| **[`cuda/sm86/holosphere_h256_chunk32_sm86.cu`](cuda/sm86/holosphere_h256_chunk32_sm86.cu)** | `Track S117` `C=32` `O(T/32)`-VRAM Woodbury Fwd/Bwd (Choice A `4,224 B` & Choice B `6,336 B`), `H_256 × bent4`, `23`-bit Gumbel, & `v2_d512` | Choice B **`61,587.61 tok/s`** (**`1.2543×`** over Choice A `49,102.91 tok/s`, `3.28e-6` `gS` parity, **`598.83 MiB`** live peak VRAM at `B=4, T=8192, D=512` w/ `torch.compile(fullgraph=True, backend="inductor")`); `v2_d512` `16`-warp **`26.23 µs/tok`** CellOp / **`27.43 µs/tok`** (`0.989 mJ/tok`, `9.091×` over `v1`, `1.83e-7` vs FP64 dense) w/ `16/16` `0.0` `.sys` BER canary | `11/11` `0 B` spill, `0` `LDL`/`STL`, `0` `MEMBAR_GL` · [`receipts/s117_ptxas_zero_spill_receipt.json`](receipts/s117_ptxas_zero_spill_receipt.json) · [`receipts/s117_live_rtx3080_zone_b_results.json`](receipts/s117_live_rtx3080_zone_b_results.json) · [`receipts/s117_live_v2_d512_zone_b_results.json`](receipts/s117_live_v2_d512_zone_b_results.json) · Zone-B [`de012df3…`](receipts/mlos_zone_b_s117_live.json) / [`2fccfe50…`](receipts/mlos_zone_b_s117_v2_d512.json) |
| **[`cuda/sm86/holosphere_megakernel_sm86.cu`](cuda/sm86/holosphere_megakernel_sm86.cu)** | `Track S116` `8`-Warp `d=256` Switch-Free `CellOp`, `ScopedAmpereL2Pin`, & `Weight-in-A` `IMMA.16832` | **`33.69 µs/tok`** `12L, d=256` `CellOp` (`0.462 mJ/tok`, **`9.12×`** over `48`-CTA switch VM, **`2.97×`** over `48`-launch async C++), **`ScopedAmpereL2Pin`** (`2.75 MiB` GA104M exact clamp) **`525.46 GB/s`** (`4`-pass polluted `T=8192`, `1.19×`–`1.49×`), **`Weight-in-A` `N=8` `1.039 µs/branch`** (`4.04 TOPS`, `1.71×` over `Weight-in-B`), **`0.0`** `16 B` $\Pi_{t_0}$ rollback drift | `10/10` `0 B` spill · [`receipts/s116_ptxas_zero_spill_receipt.json`](receipts/s116_ptxas_zero_spill_receipt.json) · [`receipts/s116_phase2b_gpu_results.json`](receipts/s116_phase2b_gpu_results.json) · Zone-B [`739f8708…`](receipts/mlos_zone_b_s116_phase2b.json) |
| **[`include/razor_abi_4byte_sys.h`](include/razor_abi_4byte_sys.h)** | `64 B` `.sys` BER Canary & `16 B` Weft Rollback ABI | Cacheline-aligned `sizeof=64, alignof=64` D2H mapped canary (`0` PCIe writes on clean cycles, up to `4` `.sys` release ops on faults) + `16 B` `HoloSpherePiRollback16`; `__ORDER_LITTLE_ENDIAN__`, `__atomic_load_n(..., __ATOMIC_ACQUIRE)` (zero C11/C++17 pointer-punning UB), pure-read `_MSC_VER` `_ReadWriteBarrier()` | Local `make verify-razor-abi` (`gcc -std=c11` & `g++ -std=c++17` `-O2 -Wall -Wextra -Werror -pedantic -fsanitize=undefined`); companion `nvcc sm_86` `12`-reg `0 B` spill probe verified in [`Moissanite-Labs/silisense#2`](https://github.com/Moissanite-Labs/silisense/pull/2) · paired w/ [`Moissanite-Labs/holosphere#3`](https://github.com/Moissanite-Labs/holosphere/pull/3) |

Verify all `50/50` HoloSphere zero-spill kernel receipts, `4/4` Zone-B hardware receipts, and the C11/C++17 UBSan Razor ABI without requiring a GPU:

```bash
make verify-holosphere
# On NVIDIA sm_86 with nvcc:
make check-holosphere-zero-spill
```

---

## Core Low-Level Techniques

1. **Direct Raw-Word 5-Bit Extended Hamming `[16, 11, 4]` Syndrome (`1 LOP3 + 5 POPC`):** Eliminates scalar bit-extraction loops via `lop3.b32(lo, hi, 0x11111111u, 0x28)` followed by five masked `__popc` reductions (and `vpopcntq` on AVX-512), reducing live syndrome verification overhead to **`0.4%–4.3%`** at zero register spill.
2. **Zero-Sign-Extension Bias-Shifted `dp4a.u32.s32` Unpack (`0.625 ops/weight`):** `lop3.b32(w, 0x0F0F0F0Fu, 0x08080808u, 0x6A)` converts signed 4-bit nibbles `[-8, +7]` into unsigned biased `u8 = n + 8`, packed via `prmt.b32` directly into `IDP.4A.U8.S8` (`dp4a`) without branchy sign extension.
3. **Zero-Shared-Memory Warp-Shuffle `WHT_8` / `Block-H4` / `H_16`:** Pure `__shfl_xor_sync` butterflies (`0 B` shared memory) fused directly inside the GEMV epilogue/prologue.
4. **Magic-Mantissa `PRMT` `FP16`/`BF16` Dequant + 4 MiB L2 Residency Pinning:** Eliminates `I2F` conversions in `HFMA2` / `BF16` paths and pins hot activation tiles via `cudaStreamAttributeAccessPolicyWindow`.
5. **Dual-Force `64 B` Cache-Line Proximity Layout (`AVX-512` VNNI):** Separates write-write MESI contention (`16.97×` slowdown) from read-shared `64 B` fusion (`6.98×` faster, `4.31×` lower energy) in `vpermb + vpdpbusd` multicore CPU GEMV.

---

## Building

```bash
# Build all Intel AVX-512 CPU targets (requires AVX-512 VBMI/VNNI/VPOPCNTDQ)
make cpu

# Build all NVIDIA Ampere sm_86 CUDA PTX/SASS targets (prints ptxas register & spill counts)
make cuda

# Build everything
make all
```

Override toolchain paths or GPU target architecture via `make NVCC=nvcc ARCH=sm_86 CC=gcc`.

---

## Repository Layout

```text
cuda/sm86/     Bare-metal Ampere sm_86 PTX/SASS CUDA kernels & full-model Qwen2.5 shared engine
cpu/avx512/    Bare-metal Intel AVX-512 VBMI/VNNI/VPOPCNTDQ CPU kernels & zero-VRAM Qwen2.5 engine
lean/          20 pure-Init Lean 4.36.0 theorems certifying the 64 B proximity-horizon laws (Track S71)
receipts/      Verified hardware benchmark, SASS (cuobjdump), and AVX-512 ISA inspection JSON receipts
```

---

## Honest Scope & Limitations (Aletheia Law 4)

- **Batched Prefill (`B=32`):** Custom `cp.async + IMMA` (`qwen25_sm86_a16_engine.cu`) reaches `8,383.5 tok/s` (`0.885×` cuBLAS `BF16` `9,473.7 tok/s`), trading slight prefill throughput for zero global weight-dequantization memory materialization.
- **Tied `lm_head` Footprint:** On `Qwen2.5-0.5B`, the untouched `151,936`-vocab `BF16` `lm_head` (`272.27 MB`) exceeds all `168` `4.5-bpw` `A_16` linear matrices combined (`201.28 MB`), dominating end-to-end vs. backbone latency (`400.58` vs. `571.37 tok/s`).
- **NVML Energy Counter Caveat:** On `GA104M` laptop silicon, `nvmlDeviceGetTotalEnergyConsumption` exhibited `>16×` idle counter drift and was rejected in favor of `100 Hz` trapezoidal `nvmlDeviceGetPowerUsage` (INA3221 rail sensor) integration paired with Linux `powercap` `intel-rapl` counters.
- **Hardware Tuning:** Inline PTX instructions (`lop3`, `prmt`, `dp4a`, `cp.async`, `mma.sync.m16n8k32`) target NVIDIA Ampere (`sm_86+`); `__nv_bfloat16` kernels in `su2_holonomy_scan_sm86.cu` require `ARCH >= sm_80`; CPU targets require `AVX-512F/BW/VL/VBMI/VNNI/VPOPCNTDQ` (with portable fallback in `hf_reference_avx512_a16_blockh4.c`).
- **SU(2) Holonomy Receipt Scope:** `receipts/su2_holonomy_sm86_ptxas_receipt.json` directly certifies offline `nvcc 13.4` (`-arch=sm_86`) compilation, `27/27` zero-spill `ptxas`, PTX (`ld.local=0, st.local=0`), and SASS (`LDL=0, STL=0`) inspection for `cuda/sm86/su2_holonomy_scan_sm86.cu` inside this repository; the companion `GA104M` `RTX 3080` step latency and `mJ/tok` figures were measured in the companion `Moissanite-Labs/holosphere` benchmark suite.
- **HoloSphere `S116`/`S117`/`V3` Receipt & Provenance Scope:** `receipts/s117_live_rtx3080_zone_b_results.json` records `cu_kernels_sha256 = a1a9780f…` from the initial 10-kernel `v1` Choice-A/B & `torch.compile` Zone-B run prior to appending the 11th `v2_d512` 16-warp kernel (`ade82cc9…`, whose source SHA-256 and `11/11` `0 B`-spill compile are certified in `receipts/s117_ptxas_zero_spill_receipt.json`, while `receipts/s117_live_v2_d512_zone_b_results.json` records the live `v2_d512` hardware latency/parity metrics and carries no `.cu` hash). All `4` `mlos-run-receipt/v0.1` Zone-B receipts (`git_head: null`) were recorded on `GA104M` (`RTX 3080 Laptop GPU`, driver `615.78.08`) prior to staging into this repository and pass canonical `receipt_sha256` verification via `verify/verify_holosphere_receipts.py`. In `receipts/s117_v3_frontier_live_gpu_results.json`, `v3_d512` CUDA-graph latency is recorded at `T=8192` for CellOp (`20.5056 µs/tok`) and `T=16384` for E2E (`21.5202 µs/tok`, `46,467.86 tok/s`, **`1.3076 mJ/tok`**, higher per-token energy than `v2_d512`'s `0.989 mJ/tok` due to higher sustained draw), while matched `T=8192` non-graph timings are `21.2373 µs/tok` CellOp vs. `22.2485 µs/tok` E2E (`1.0112 µs/tok` difference); both reflect `1`-CTA `16`-warp microbenchmarks (`B=1, L=12, d=512`), and `s117_chunk32_bwd_wmma_sm86` (`148,382.09 tok/s`, `13,451.30 mJ` over `B=4, T=8192, D=512`) measures the standalone `Chunk32` backward kernel rather than full PyTorch optimizer wall-clock.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
