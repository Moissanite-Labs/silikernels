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
- **Hardware Tuning:** Inline PTX instructions (`lop3`, `prmt`, `dp4a`, `cp.async`, `mma.sync.m16n8k32`) target NVIDIA Ampere (`sm_86+`); CPU targets require `AVX-512F/BW/VL/VBMI/VNNI/VPOPCNTDQ` (with portable fallback in `hf_reference_avx512_a16_blockh4.c`).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
