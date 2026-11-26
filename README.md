# Silikernels

**Bare-metal zero-spill Ampere `sm_86` PTX/SASS and Intel `AVX-512` (VBMI / VNNI / VPOPCNTDQ) kernel rewrites for `A_16` Extended Hamming `[16, 11, 4]` lattice quantization, Block-H4 Walsh–Hadamard fusion, and full-model LLM inference.**

Developed within **Moissanite Labs** alongside [Moissanite-Labs/siliquant](https://github.com/Moissanite-Labs/siliquant) and [Foundation-0/aletheia](https://github.com/Foundation-0/aletheia). Every kernel below that cites a `ptxas`/SASS receipt has **zero register spills (`0 B` stack frame, `0 B` spill stores/loads, `0` SASS `LDL`/`STL`)** on NVIDIA Ampere `sm_86` (`RTX 3080 Laptop GPU`, `GA104M`, `256-bit` GDDR6, `356.68 GB/s` STREAM Triad) and Intel `i9-11900H` (`8C/16T` Tiger Lake-H `AVX-512`). The HoloSphere GPU runs also carry Ring-0 `mlos` Zone-B provenance receipts.

---

## Verified Silicon Records

| Kernel Target | Architecture | Headline Verified Result | Verification & Provenance |
| :--- | :--- | :--- | :--- |
| **[`cuda/sm86/qwen25_sm86_a16_engine.cu`](cuda/sm86/qwen25_sm86_a16_engine.cu)** + **[`cpu/avx512/qwen25_avx512_a16_engine.c`](cpu/avx512/qwen25_avx512_a16_engine.c)** | Ampere `sm_86` PTX + `AVX-512` | **Full-model `Qwen/Qwen2.5-0.5B` (`357.8M` quantized weights):** **`571.37 tok/s`** 24-layer backbone (**`7.43×`** vs. PyTorch `BF16`), **`400.58 tok/s`** full E2E w/ `151,936`-vocab `BF16` `lm_head` (**`5.60×`** vs. `BF16` `71.58 tok/s`), **`91.78 ms`** cold-to-hot load, **`31.70 tok/s`** zero-VRAM CPU fallback (`16/16` exact greedy GPU-vs-CPU match) | `16/16` kernels `0 B` spill, `0` syndrome errors · [`siliquant#88`](https://github.com/Moissanite-Labs/siliquant/pull/88) · [`receipts/s58_qwen25_live_engine_results.json`](receipts/s58_qwen25_live_engine_results.json) |
| **[`cuda/sm86/aletheion_sm86_ptx_foundry.cu`](cuda/sm86/aletheion_sm86_ptx_foundry.cu)** | Ampere `sm_86` PTX/SASS | **`646.51 Gweights/s`** (`102.03%` STREAM Triad) fused `4.5 bpw` `A_16` `LDG.128 + LOP3 + PRMT + dp4a.u32.s32` GEMV; live `5-POPC` syndrome verify at `643.29 Gw/s` (`0.4%` overhead vs. the same-protocol `645.76 Gw/s` arm); zero-shmem warp-shuffle `WHT_8`/`H_16` (`463.89 Gw/s`); `O(1)` register-delta live repair (`478.46 Gw/s`); batched `B=32` `cp.async + IMMA.16832` (`14.90 INT8 TOPS`) | `15/15` kernels `0 B` spill, `0` verify fails · [`siliquant#83`](https://github.com/Moissanite-Labs/siliquant/pull/83) · [`receipts/s52_sm86_ptx_foundry_results.json`](receipts/s52_sm86_ptx_foundry_results.json) |
| **[`cpu/avx512/s71_proximity_bench.c`](cpu/avx512/s71_proximity_bench.c)** | Intel `AVX-512` VNNI | **`146.892 Gweights/s`** (`total_ops / ns` of `avx512_a16_blockh4_dual_force[0]`) multicore `A_16 + BlockH4` on `Qwen2.5-3B` `2048×11008` via `64 B` cache-line read-shared proximity fusion (**`6.982×` faster** per weight than the `8 B` row of `avx512_a16_blockh4_stride_sweep`, computed from the per-op `ns` of the two suites in the same receipt) | `20/20` pure-`Init` `Lean 4.36.0` proofs in [`lean/ProximityBarrierSilicon.lean`](lean/ProximityBarrierSilicon.lean) · [`aletheia#71`](https://github.com/Foundation-0/aletheia/pull/71) · [`receipts/s71_proximity_avx512_results.json`](receipts/s71_proximity_avx512_results.json) |
| **[`cpu/avx512/aletheion_avx512_a16_decoder.c`](cpu/avx512/aletheion_avx512_a16_decoder.c)** | Intel `AVX-512` VBMI/VNNI | **`60.95–80.21 Gweights/s`** fused `vpermb + vpdpbusd` VNNI GEMV + live `vpopcntq` verify; **`223.19 Mweights/s`** exact bit-identical `18 B/32 w` enumerative `hampack.c` | `0` bit mismatches, `0` syndrome errors · [`siliquant#83`](https://github.com/Moissanite-Labs/siliquant/pull/83) · [`receipts/s52_avx512_a16_benchmark_results.json`](receipts/s52_avx512_a16_benchmark_results.json) |
| **[`cuda/sm86/hf_reference_sm86_a16_blockh4.cu`](cuda/sm86/hf_reference_sm86_a16_blockh4.cu)** + **[`cpu/avx512/hf_reference_avx512_a16_blockh4.c`](cpu/avx512/hf_reference_avx512_a16_blockh4.c)** | Ampere `sm_86` + `AVX-512` | Self-contained standalone reference kernels bundled with [`siliquant-hf-v1`](https://huggingface.co/collections/ahrazzle/siliquant-stratified-lattice-quantized-llms-6b060e6988afd65942cd7768) public Hugging Face releases | [`siliquant#100`](https://github.com/Moissanite-Labs/siliquant/pull/100) |
| **[`cuda/sm86/su2_holonomy_scan_sm86.cu`](cuda/sm86/su2_holonomy_scan_sm86.cu)** | Ampere `sm_86` PTX/SASS | **27 fused `SU(2)` quaternion holonomy, covariant `(Q,K,V)` transport, spherical SLERP/Hopf-$S^3$, `cayley_su2`, & Beltrami `Cayley2x2` kernels** (`float4` = 1 unit quaternion, zero-permute strided BHTD FP32/BF16, 3-float $\mathfrak{su}(2)$ additive backward suffix sum). | `27/27` kernels `0 B` spill, `0` PTX `ld/st.local`, `0` SASS `LDL`/`STL` · [`receipts/su2_holonomy_sm86_ptxas_receipt.json`](receipts/su2_holonomy_sm86_ptxas_receipt.json) |

---

## HoloSphere-V3 Bare-Metal `sm_86` Megakernel, `Chunk32` Woodbury, & `W10-OBS-61` Frontier Kernels (`Refs #2`)

Extending `su2_holonomy_scan_sm86.cu` (`27/27`), the three HoloSphere `S116`/`S117`/`W10-OBS-61` kernel suites add **`23/23`** additional zero-spill Ampere `sm_86` kernels (**`50/50`** total HoloSphere `0 B`-spill kernels in `cuda/sm86/`) plus the cacheline-aligned **`64 B` `.sys` Razor telemetry & `16 B` Weft rollback ABI** ([`include/razor_abi_4byte_sys.h`](include/razor_abi_4byte_sys.h)):

| Kernel Target / Header | Scope | Live RTX 3080 (`GA104M`, `sm_86`) result (latency, parity, VRAM; no energy figures) | Zero-Spill & Provenance |
| :--- | :--- | :--- | :--- |
| **[`cuda/sm86/holosphere_v3_frontier_sm86.cu`](cuda/sm86/holosphere_v3_frontier_sm86.cu)** | `W10-OBS-61` `HMMA.16816` Double-Woodbury Bwd + `v3_d512` `0`-`POPC` hot-path Dual-SWAR E2E | **`148,153.42 tok/s`** (`221.18 ms / 12L` at `B=4, T=8192, D=512`, **`2.4056×`** over Choice B (`speedup_vs_choice_b_532ms`); `128` regs / `6,336 B` SMEM, `28` `HMMA.16816`, `30` `SHFL`); `v3_d512` (`52` regs / `13,992 B` SMEM, `0`-`POPC` hot-path Dual-SWAR, `POPC: 14` outside per-layer hot path) CUDA-graph **`20.50 µs/tok`** CellOp (`T=8192`) / **`21.52 µs/tok`** E2E (`T=16384`, `46,471.91 tok/s`) & matched `T=8192` non-graph `21.24` / `22.24 µs/tok` (`44,962.83 tok/s`, `1.0035 µs/tok` diff, `1.96e-7` `FWHT_256` err) w/ **`16/16` live `.sys` BER canary `0.0` bit-exact** (`0` clean PCIe writes, `1/1` DED) | `2/2` `0 B` spill, `0` `LDL`/`STL` · [`receipts/v3_frontier_ptxas_zero_spill_receipt.json`](receipts/v3_frontier_ptxas_zero_spill_receipt.json) · [`receipts/s117_v3_frontier_live_gpu_results.json`](receipts/s117_v3_frontier_live_gpu_results.json) · Zone-B [`feecc748…`](receipts/mlos_zone_b_s117_v3_frontier.json) |
| **[`cuda/sm86/holosphere_h256_chunk32_sm86.cu`](cuda/sm86/holosphere_h256_chunk32_sm86.cu)** | `Track S117` `C=32` `O(T/32)`-VRAM Woodbury Fwd/Bwd (Choice A `4,224 B` & Choice B `6,336 B`), `H_256 × bent4`, `23`-bit Gumbel, & `v2_d512` | Choice B **`61,509.22 tok/s`** (**`1.2535×`** over Choice A `49,071.12 tok/s`, `3.28e-6` `gS` parity, **`598.83 MiB`** live peak VRAM at `B=4, T=8192, D=512` w/ `torch.compile(fullgraph=True, backend="inductor")`); `v2_d512` `16`-warp **`26.21 µs/tok`** CellOp / **`27.41 µs/tok`** (`9.097×` over `v1`, `1.83e-7` vs FP64 dense) w/ `16/16` `0.0` `.sys` BER canary | `11/11` `0 B` spill, `0` `LDL`/`STL`, `0` `MEMBAR_GL` · [`receipts/s117_ptxas_zero_spill_receipt.json`](receipts/s117_ptxas_zero_spill_receipt.json) · [`receipts/s117_live_rtx3080_zone_b_results.json`](receipts/s117_live_rtx3080_zone_b_results.json) · [`receipts/s117_live_v2_d512_zone_b_results.json`](receipts/s117_live_v2_d512_zone_b_results.json) · Zone-B [`fef9d269…`](receipts/mlos_zone_b_s117_live.json) / [`699c3d05…`](receipts/mlos_zone_b_s117_v2_d512.json) |
| **[`cuda/sm86/holosphere_megakernel_sm86.cu`](cuda/sm86/holosphere_megakernel_sm86.cu)** | `Track S116` `8`-Warp `d=256` Switch-Free `CellOp`, `ScopedAmpereL2Pin`, & `Weight-in-A` `IMMA.16832` | **`33.37 µs/tok`** `12L, d=256` `CellOp` (**`10.09×`** over `48`-CTA switch VM, **`2.93×`** over `48`-launch async C++), **`ScopedAmpereL2Pin`** (`2.75 MiB` GA104M exact clamp) **`543.22 GB/s`** (`4`-pass polluted `T=8192`, `1.22×`–`1.39×`), **`Weight-in-A` `N=8` `1.057 µs/branch`** (`3.97 TOPS`, `1.69×` over `Weight-in-B`), **`0.0`** `16 B` $\Pi_{t_0}$ rollback drift | `10/10` `0 B` spill · [`receipts/s116_ptxas_zero_spill_receipt.json`](receipts/s116_ptxas_zero_spill_receipt.json) · [`receipts/s116_phase2b_gpu_results.json`](receipts/s116_phase2b_gpu_results.json) · Zone-B [`bec7b99b…`](receipts/mlos_zone_b_s116_phase2b.json) |
| **[`include/razor_abi_4byte_sys.h`](include/razor_abi_4byte_sys.h)** | `64 B` `.sys` BER Canary & `16 B` Weft Rollback ABI | Cacheline-aligned `sizeof=64, alignof=64` D2H mapped canary (`static_assert` in the header) + `16 B` `HoloSpherePiRollback16`; `__ORDER_LITTLE_ENDIAN__`, `__atomic_load_n(..., __ATOMIC_ACQUIRE)` (zero C11/C++17 pointer-punning UB), pure-read `_MSC_VER` `_ReadWriteBarrier()` | Local `make verify-razor-abi` (`gcc -std=c11` & `g++ -std=c++17` `-O2 -Wall -Wextra -Werror -pedantic -fsanitize=undefined`); ABI/layout-compatible with merged [`Moissanite-Labs/silisense#2`](https://github.com/Moissanite-Labs/silisense/pull/2) (`d621fd00`) · byte-identical to merged [`Moissanite-Labs/holosphere#3`](https://github.com/Moissanite-Labs/holosphere/pull/3) (`1c790ac2`) |

Verify all `50/50` HoloSphere zero-spill kernel receipts, `4/4` Zone-B hardware receipts, and the C11/C++17 UBSan Razor ABI without requiring a GPU:

```bash
make verify-holosphere
# On NVIDIA sm_86 with nvcc:
make check-holosphere-zero-spill
```

---

## Core Low-Level Techniques

1. **Direct Raw-Word 5-Bit Extended Hamming `[16, 11, 4]` Syndrome (`1 LOP3 + 5 POPC`):** Eliminates scalar bit-extraction loops via `lop3.b32(lo, hi, 0x11111111u, 0x28)` followed by five masked `__popc` reductions (and `vpopcntq` on AVX-512), reducing live syndrome verification overhead to **`0.4%–4.3%`** at zero register spill (`0.4%`: `s52_sm86_ptx_foundry_results.json` live-verify vs. same-protocol arm; `4.3%`: `s58_qwen25_live_engine_results.json` `b2_live_5popc_verify.verify_overhead_backbone_pct = 4.275`).
2. **Zero-Sign-Extension Bias-Shifted `dp4a.u32.s32` Unpack:** `lop3.b32(w, 0x0F0F0F0Fu, 0x08080808u, 0x6A)` converts signed 4-bit nibbles `[-8, +7]` into unsigned biased `u8 = n + 8`, packed via `prmt.b32` directly into `IDP.4A.U8.S8` (`dp4a`) without branchy sign extension.
3. **Zero-Shared-Memory Warp-Shuffle `WHT_8` / `Block-H4` / `H_16`:** Pure `__shfl_xor_sync` butterflies (`0 B` shared memory) fused directly inside the GEMV epilogue/prologue.
4. **Magic-Mantissa `PRMT` `FP16`/`BF16` Dequant + 4 MiB L2 Residency Pinning:** Eliminates `I2F` conversions in `HFMA2` / `BF16` paths and pins hot activation tiles via `cudaStreamAttributeAccessPolicyWindow`.
5. **Dual-Force `64 B` Cache-Line Proximity Layout (`AVX-512` VNNI):** Separates write-write MESI contention (`16.97×` slowdown) from read-shared `64 B` fusion (`6.98×` faster per weight; [`receipts/s71_proximity_avx512_results.json`](receipts/s71_proximity_avx512_results.json)) in `vpermb + vpdpbusd` multicore CPU GEMV.

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
- **Energy:** This README publishes no energy figures. The energy values recorded in `receipts/` come from an uncontrolled `UX582HS` laptop (`GA104M` GPU, `i9-11900H` CPU) and stay in the receipts only.
- **Hardware Tuning:** Inline PTX instructions (`lop3`, `prmt`, `dp4a`, `cp.async`, `mma.sync.m16n8k32`) target NVIDIA Ampere (`sm_86+`); `__nv_bfloat16` kernels in `su2_holonomy_scan_sm86.cu` require `ARCH >= sm_80`; CPU targets require `AVX-512F/BW/VL/VBMI/VNNI/VPOPCNTDQ` (with portable fallback in `hf_reference_avx512_a16_blockh4.c`).
- **SU(2) Holonomy Receipt Scope:** `receipts/su2_holonomy_sm86_ptxas_receipt.json` directly certifies offline `nvcc 13.4` (`-arch=sm_86`) compilation, `27/27` zero-spill `ptxas`, PTX (`ld.local=0, st.local=0`), and SASS (`LDL=0, STL=0`) inspection for `cuda/sm86/su2_holonomy_scan_sm86.cu` inside this repository; this README quotes no SU(2) timing figures, because no receipt in this repository records them.
- **HoloSphere `S116`/`S117`/`V3` Receipt & Provenance Scope:** All `4` live RTX 3080 (`GA104M`, driver `615.78.08`) result JSONs in `receipts/` (`s116_phase2b_gpu_results.json`, `s117_live_rtx3080_zone_b_results.json`, `s117_live_v2_d512_zone_b_results.json`, `s117_v3_frontier_live_gpu_results.json`) were freshly replayed directly against the committed `cuda/sm86/*.cu` sources at `git_head = 44eccad8dfa38bcc397c45420680d62d0020c585` under Ring-0 `mlos gpu-lease --mode train` + `mlos stamp-run --zone B --require-gpu`. `verify/verify_holosphere_receipts.py` fails closed unless every live result JSON carries exact `cu_sha256` (and `cu_kernels_sha256` on `s117_v1`) matching live `cuda/sm86/*.cu` bytes (`bb64939c…`, `ade82cc9…`, `61173b2c…`), `parent_commit_sha == 44eccad8…`, and `mlos_zone_b_receipt_sha256` matching its canonical Zone-B receipt (`bec7b99b…`, `fef9d269…`, `699c3d05…`, `feecc748…`). In `receipts/s117_v3_frontier_live_gpu_results.json`, `v3_d512` CUDA-graph latency is recorded at `T=8192` for CellOp (`20.5038 µs/tok`) and `T=16384` for E2E (`21.5184 µs/tok`, `46,471.91 tok/s`), while matched `T=8192` non-graph timings are `21.2371 µs/tok` CellOp vs. `22.2406 µs/tok` E2E (`1.0035 µs/tok` difference); both reflect the `16`-warp `d=512` kernel `s117_uniform_cell_16warp_d512_v3_sm86`, and `s117_chunk32_bwd_wmma_sm86` (`148,153.42 tok/s` over `B=4, T=8192, D=512`) measures the standalone `Chunk32` backward kernel rather than full PyTorch optimizer wall-clock.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
