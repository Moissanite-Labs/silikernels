# Bare-Metal `sm_86` CTA-SMEM Packed `.sys` BER Canary & `1 KiB` FP16 Folded `FWHT_256 x bent4` Head

Additive Ampere `sm_86` (`RTX 3080 Laptop GPU`, `GA104M`, `nvcc 13.4.92`) kernel upgrades for the 16-warp `d_model = 512` HoloSphere cell and `0 B` HBM-table `1/sqrt(32)`-scaled (`sqrt(8) * orthonormal`, Euclidean norm gain `2.828427`) `FWHT_256 x bent4` byte head (`Refs Moissanite-Labs/silikernels#2`).

- **Telemetry & Laptop Benchmark Policy:** `energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED` — zero uncontrolled laptop power or energy claims are published. All receipts record exact PTX/SASS register/spill counts, committed `.sys` atomic counts, FP32-vs-FP64 arithmetic parity on FP16-rounded fold weights, **uncontrolled RTX 3080 Laptop (`GA104M`)** latency and fault-storm contention readings (`us/tok`), and held-out UTF-8 BPB.
- **Self-Bound On-Disk Verification:** Every on-disk SHA-256 digest in [`receipts/smem_canary_folded_fwht_ptxas_receipt.json`](../../receipts/smem_canary_folded_fwht_ptxas_receipt.json) and [`receipts/smem_canary_folded_fwht_results.json`](../../receipts/smem_canary_folded_fwht_results.json) binds directly to committed files on disk in this repository (`holosphere_smem_canary_folded_fwht_sm86.cu`, `holosphere_smem_canary_folded_fwht_bench.cu`, `run_smem_canary_folded_fwht_bpb_study.py`, and the `ptxas` receipt), enforced with duplicate-key rejection, closed key allowlists, `--require-verdict CONFIRMED`, and exact canonical raw-byte checking. The external open-license UTF-8 corpus binaries (`train.bin`, `eval.bin`) are pinned by exact SHA-256 digests (`corpus_manifest_sha256`, `corpus_train_bin_sha256`, `corpus_eval_bin_sha256`) verified at study run time.

---

## 1. Zero-Spill `sm_86` `ptxas` & `cuobjdump -sass` Proof (`5 / 5` Kernels)

Compiled from [`cuda/sm86/holosphere_smem_canary_folded_fwht_sm86.cu`](holosphere_smem_canary_folded_fwht_sm86.cu) under `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror` (zero unions, zero `reinterpret_cast`, raw byte-backed `memcpy` SMEM overlay):

| Kernel (`holosphere_smem_canary::*`) | Regs | Barriers | SMEM (B) | Stack / Spill (B) | `LDL` / `STL` | `ATOMG.E.CAS.STRONG.SYS` | `RED.E.OR.STRONG.SYS` | `ATOMS` (SMEM) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `holosphere_cell_folded1k_smem_canary_sm86` | `55` | `1` | `14,004` (`+12 B`) | `0 / 0 / 0` | `0 / 0` | `1` | `1` | `3` |
| `holosphere_cell_unfolded_warp_canary_sm86` | `56` | `1` | `13,992` | `0 / 0 / 0` | `0 / 0` | `1` | `1` | `0` |
| `holosphere_lm_head_folded1k_sm86` | `19` (`+1`) | `1` | `4,096` (`+0 B`) | `0 / 0 / 0` | `0 / 0` | `0` | `0` | `0` |
| `holosphere_lm_head_unfolded_sm86` | `18` | `1` | `4,096` | `0 / 0 / 0` | `0 / 0` | `0` | `0` | `0` |
| `holosphere_smem_canary_overflow_probe_sm86` | `8` | `0` | `0` | `0 / 0 / 0` | `0 / 0` | `1` | `1` | `0` |

---

## 2. CTA-SMEM-Aggregated Per-Lane Saturating `.sys` CAS Canary

Rather than issuing per-warp PCIe `.sys` atomics against one mapped host cacheline during multi-warp error storms, each warp reduces its `[16, 11, 4]` Extended Hamming SEC/DED counts into `12 B` of CTA shared memory (`smem_cta_sec`, `smem_cta_ded`, `smem_cta_diag`). Thread `0` reuses the existing final layer-loop `__syncthreads()` (`0` extra barriers) and commits `<= 1` predicated per-lane saturating `atom.release.sys.global.cas.b32` (`ATOMG.E.CAS.STRONG.SYS`) update per CTA in drained single-step runs (`0` writes when `sec_tot | ded_tot == 0`), plus monotonic `atom.release.sys.global.or.b32` (`RED.E.OR.STRONG.SYS`) for diagnostic and overflow bits:

- **Committed Update Accounting & Tracked CUDA Follow-Ups:** `dev_sys_atomic_cnt` counts logical committed saturating updates (`smem_sys_atomics = 1`, yielding `55x`–`192x` fewer committed updates in drained single-step runs), and `smem_cta_diag` uses first-writer `atomicCAS(&smem_cta_diag, 0u, diag)` (first-lane/first-CAS diagnostic ordering within the CTA). Because the `.sys` CAS loop seeds `assumed = 0u`, an un-drained launch or multi-CTA storm where the host word is already nonzero issues a first CAS retry (`>= 2` `.sys` CAS transactions per CTA); seeding `assumed` with `ld.relaxed.sys` and merging CTA diagnostic bits with `atomicOr` (so an earlier warp's SEC diag cannot hide a later warp's DED diag bit) are tracked as CUDA follow-ups without altering the pinned `nvcc 13.4.92` kernel bytes here.
- **Zero Cross-CTA Carry or Wrap-to-Zero:** Unlike raw 32-bit `atom.add`, `silisense_ber_canary_sat_add_sys` saturates the low 16-bit SEC lane `[15:0]` and high 16-bit DED lane `[31:16]` independently at `0xFFFF` across arbitrarily many concurrent CTAs, warps, or un-drained launches — eliminating both SEC-to-DED bit-16 carry and DED bit-31 wrap-to-zero, while escalating `DED >= 1` and setting bit `31` (`0x80000000`) in `host_diag_out` whenever either lane saturates.
- **PCIe Atomic Capability (`host_native_atomic_supported = false`):** On the benchmarked `RTX 3080 Laptop GPU` host, `cudaDevAttrHostNativeAtomicSupported == 0` (`host_native_atomic_supported = false` in the receipt): `.sys` atomics are atomic across GPU threads/CTAs over PCIe (not against concurrent CPU writes), so the host drains mapped pinned memory only after explicit device synchronization. In uncontrolled RTX 3080 Laptop (`GA104M`) storm sweeps, 16 warps contending on one PCIe-mapped cacheline exhibit heavy latency degradation, while CTA-SMEM aggregation records `6.87x`–`9.75x` faster uncontrolled laptop storm latencies below.

### Live RTX 3080 (`GA104M`, `sm_86`) Functional Parity & Fail-Closed Saturation Verification

- **NaN-Safe FP32-vs-FP64 Arithmetic Parity on FP16-Rounded Fold Weights (NaN-Poisoned Output Buffers):** Fused cell + `1 KiB` FP16 fold max abs error `2.257e-7` (`2256895237/10000000000000000`); standalone `1 KiB` FP16 head max abs error `3.417e-7` (`683433099/2000000000000000`). This measures FP32 vs FP64 arithmetic parity given the already FP16-rounded fold weights (`__half22float2`), distinct from the `~4.3e-4` FP16 weight-rounding logit shift (end-to-end FP16-vs-BF16 BPB difference is measured separately below at `<= 2.51e-5` BPB).
- **Bit-Exact SEC/DED Recovery:** Clean step (`0` atomics, `packed = 0`); **`16 / 16`** single-bit SEC flips (`sec = 16`, `ded = 0`, `packed = 16`, `sys_atomics = 1`, `0.0` hidden and logit error); **`120 / 120`** double-bit DED pairs (`sec = 0`, `ded = 120`, `packed = 7864320 = 120 << 16`, `sys_atomics = 1`).
- **Single-CTA & Multi-CTA Fail-Closed Saturation Probes (Live GPU):**
  - Single-CTA SEC overflow (`<<<1, 32>>>`, `sec = 70000, ded = 0, diag = 0x15`): `packed = 0x0001FFFF` (`131071`), `diag = 0x80000015`.
  - Single-CTA DED overflow (`<<<1, 32>>>`, `sec = 12, ded = 70000, diag = 0x22`): `packed = 0xFFFF000C` (`4294901772`), `diag = 0x80000022`.
  - Multi-CTA SEC carry defense (`<<<2, 32>>>`, `2 x (sec = 40000, ded = 0)`): `packed = 0x0001FFFF` (`131071`), `diag = 0x80000015` (zero SEC-to-DED carry).
  - Multi-CTA DED wrap defense (`<<<2, 32>>>`, `2 x (sec = 7, ded = 0x8000)`): `packed = 0xFFFF000E` (`4294901774`), `diag = 0x80000022` (zero bit-31 wrap-to-zero; exact `sec = 14`, saturated `ded = 0xFFFF`).

### Uncontrolled RTX 3080 Laptop (`GA104M`) Readings — Latency & Fault-Storm Contention

> **Uncontrolled Laptop Measurement Scope:** Every `us/tok` latency, clean-step delta (`21.7707 -> 21.4898 us/tok`, `-0.2809 us/tok`, `1.0131x`), storm contention speedup (`6.87x`–`9.75x`), and standalone head latency (`1.2917 -> 1.4252 us/tok`, `+0.1335 us/tok`, `G2 <= 0.5 us/tok`) below is an **uncontrolled RTX 3080 Laptop (`GA104M`)** reading rather than a controlled datacenter benchmark.

| Fault-Storm Rate (`w_lo` + `w_hi`) | Injected `(SEC, DED)` | Per-Warp Committed `.sys` Updates | CTA-SMEM Committed `.sys` Updates | Committed Update Reduction | Per-Warp Uncontrolled Laptop (`us/tok`) | CTA-SMEM + `1 KiB` Fold Uncontrolled Laptop (`us/tok`) | Uncontrolled Laptop Speedup |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0%` (clean) | `(0, 0)` | `0` | `0` | `0.0x` (zero writes) | `21.7707` | `21.4898` (`-0.2809`) | `1.0131x` |
| `1%` storm | `(55, 17)` | `55` | `1` | **`55.0x`** | `218.2510` | `31.7643` | `6.8710x` |
| `10%` storm | `(459, 162)` | `185` | `1` | **`185.0x`** | `349.6432` | `35.8625` | `9.7495x` |
| `100%` storm | `(4646, 1498)` | `192` | `1` | **`192.0x`** | `368.2915` | `39.9738` | `9.2133x` |

- **Standalone `1 KiB` FP16 Head Latency (`holosphere_lm_head_folded1k_sm86` vs `unfolded`, uncontrolled RTX 3080 Laptop `GA104M`):** `1.2917 -> 1.4252 us/tok` (**`+0.1335 us/tok`**, `0 B` HBM vocab table reads). Note: CUDA-graph timing loops pass `nullptr` for `d_logits`, `d_diag_canary`, and `d_sys_cnt` to isolate hot-path decode + `.sys` atomic latency, while single-step correctness checks poison output buffers with NaN, validate all outputs via NaN-safe FP32-vs-FP64 comparison on FP16-rounded weights, and exit non-zero on any mismatch.

---

## 3. 5-Arm x 3-Seed UTF-8 BPB Study & Honest Steelman Attribution

Bound directly to [`cuda/sm86/run_smem_canary_folded_fwht_bpb_study.py`](run_smem_canary_folded_fwht_bpb_study.py) (on-disk SHA-256 verified in `source_bindings.bpb_study_script_sha256`). Iso-budget 4-layer `d_model = 512`, `d_ff = 800`, `n_heads = 16` (`9,115,648` trunk params) across `3` seeds `{42, 137, 2026}` (`300` steps x `4,096` tokens/step, evaluated on `32,768` held-out UTF-8 bytes from the external open-license byte corpus pinned by `corpus_manifest_sha256 = 280ff5af...`, `corpus_train_bin_sha256 = 528c6c79...`, `corpus_eval_bin_sha256 = 31fd132d...`):

| Head Configuration | Head Params | Head FP16 (B) | HBM Vocab Table (B) | Mean Held-Out BPB | Delta vs `arm3` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `arm1_prefix_fwht` (`u = h[:256]`, upper-half grad `= 0.0`) | `0` | `0` | `0` | `6.3894` | `-1.0175` (`3/3` wins) |
| `arm2_fixed_diff_fwht` (`u = h[:256] - h[256:]`) | `0` | `0` | `0` | `5.8068` | `-0.4349` (`3/3` wins) |
| `arm2b_fixed_diff_1scalar_fwht` (`2 B` **1-scalar steelman**) | `1` | `2` | `0` | `5.3761` | `-0.0042` (`2/3` wins, statistical tie) |
| `arm3_learned_1kib_fp16_fold_fwht` (`u = alpha*h[:256] + beta*h[256:]`) | `512` | `1,024` | `0` | **`5.3719`** | `0.0000` (FP16 quant diff `<= 2.51e-5`) |
| `arm4_dense_256x512` (dense ceiling) | `131,073` | `262,146` | `262,146` | `3.5269` | `+1.8450` (ceiling gap) |

**Honest Scope, Gate Contract & Steelman Takeaway:**
1. Gate `G2` (`verdict = CONFIRMED`) explicitly tests `arm3 < arm1` and `arm3 < arm2` on all `3/3` seeds plus uncontrolled RTX 3080 Laptop (`GA104M`) standalone head overhead `<= 0.5 us/tok` (`+0.1335 us/tok`). Only `arm3` applies an explicit FP16 head-weight round-trip against BF16 during evaluation (`fp16_quant_bpb_diff_q <= 2.51e-5`; seed 137 is `94669/3783181952 = 2.5024e-5`), whereas the baseline/reference arms evaluate unmodified BF16 head weights (`fp16_quant_bpb_diff_q = 0/1`).
2. As an un-gated steelman attribution control, the **`2 B` single-scalar logit-temperature control (`arm2b_fixed_diff_1scalar_fwht`, `5.3761` BPB)** captures **`99.0%`** (`-0.4307` of `-0.4349` BPB) of the gain over unscaled `arm2` — consistent with the head's `1/sqrt(32)` (`sqrt(8) * orthonormal`, gain `2.828427`) scale factor — while the remaining `511` per-channel diagonal weights add only `-0.0042` BPB (`2/3` seeds — a practical statistical tie) and leave `+1.8450` BPB behind a dense `[256, 512]` projection. Closing that remaining gap at low parameter budget requires structured non-diagonal cross-channel mixing before `FWHT_256 x bent4`.

---

## 4. Build & Offline Verification

```bash
# Pure-stdlib 3-mode isolated -I receipt + pure-Z oracle + tamper-refusal verification (no GPU required)
make -f cuda/sm86/holosphere_smem_canary_folded_fwht.mk verify-smem-canary

# On NVIDIA Ampere sm_86 with nvcc: compile (--Werror all-warnings) and verify 5/5 zero-spill kernels
make -f cuda/sm86/holosphere_smem_canary_folded_fwht.mk check-smem-canary-zero-spill
```
