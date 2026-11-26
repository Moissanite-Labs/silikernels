# Bare-Metal `sm_86` 2-Stage Monarch-`16x16` + `+512 B` Byte-Bias `FWHT_256 x bent4` LM Head (`0 B` HBM Table)

Additive Ampere `sm_86` (`uncontrolled RTX 3080 Laptop (GA104M)`, `nvcc 13.4.92`) structured zero-HBM-table output head kernels closing **`92.63%`** of the `262 KiB` dense vocabulary ceiling gap inside **`16,896 B` (`16.5 KiB`)** of CTA-SMEM-resident FP16 weights (`Refs Moissanite-Labs/silikernels#2`).

- **Telemetry & Uncontrolled Laptop Policy:** `energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED` — zero uncontrolled laptop power or energy claims are published. All decode latency (`us/tok`) figures below are **uncontrolled RTX 3080 Laptop (`GA104M`)** measurements self-attested in the committed JSON receipts alongside `ptxas`/SASS register and spill counts (`nvcc 13.4.92`), NaN-safe FP32-vs-FP64 parity on FP16-rounded head weights, and held-out UTF-8 BPB.
- **Transform Scale Specification (`1/sqrt(32)` vs Orthonormal `1/16`):** Plain `fwht256_plain_8warp` (`holosphere_monarch2_fwht_head_sm86.cu:48`) scales by orthonormal `1/16 = 0.0625f` (`||y||_2^2 = ||u||_2^2`). The output `bent4` transform `fwht256_bent4_from_smem_8warp` (`holosphere_monarch2_fwht_head_sm86.cu:86`), the FP64 bench reference (`holosphere_monarch2_fwht_head_bench.cu:58`), and the PyTorch BPB study script (`cuda/sm86/monarch2_fwht_head_bpb_study.py`) all scale by **`1/sqrt(32) = 0.17677669529663688f`** (`sqrt(8) * orthonormal`, yielding `sum y^2 = 8 * sum u^2` — matching the per-warp logit gain; note that line 4 of the unchanged `.cu` header comment reads `"Orthonormal FWHT_256 x bent4"`, while line 86 implements the `1/sqrt(32)` scale). All three scale constants are verified on disk by `verify/verify_monarch2_fwht_head.py`.
- **Self-Bound On-Disk Verification & Provenance:** Every SHA-256 digest in [`receipts/monarch2_fwht_head_ptxas_receipt.json`](../../receipts/monarch2_fwht_head_ptxas_receipt.json) and [`receipts/monarch2_fwht_head_results.json`](../../receipts/monarch2_fwht_head_results.json) binds directly to committed files on disk (`holosphere_monarch2_fwht_head_sm86.cu`, `holosphere_monarch2_fwht_head_bench.cu`, `monarch2_fwht_head_bpb_study.py`, and both receipts) plus the pinned external corpus SHA-256s (`holosphere-v3-flagship-corpus`), enforced with duplicate-key & integer-digit-cap rejection, closed key allowlists, `--require-verdict CONFIRMED`, and exact canonical raw-byte checking. Hardware `ptxas`/SASS counts (`nvcc 13.4.92`) and uncontrolled RTX 3080 Laptop (`GA104M`) latency/BPB numbers are self-attested in the receipts; `ptxas` zero-spill compilation can be re-checked with any local `nvcc` via `check-monarch2-head-zero-spill`, and offline CI verifies receipts, source bindings, scale constants, and pure-`Z` oracles without a GPU.

---

## 1. Zero-Spill `sm_86` `ptxas`, SASS, FP32-vs-FP64 Parity & Uncontrolled RTX 3080 Laptop (`GA104M`) Latency (`4 / 4` Kernels)

Compiled from [`cuda/sm86/holosphere_monarch2_fwht_head_sm86.cu`](holosphere_monarch2_fwht_head_sm86.cu) under `nvcc -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror` (zero unions, zero `reinterpret_cast`, raw `alignas(16)` byte-backed `memcpy` SMEM tiles), benchmarked on an **uncontrolled RTX 3080 Laptop (`GA104M`, `sm_86`)** with NaN-poisoned output buffers, NaN-safe FP32-vs-FP64 parity checks on the FP16-rounded head weights, and P0-warmed `3`-trial interleaved CUDA-Graph replay (`16,384` launches/trial; gate `<= 3.0 us/tok` on uncontrolled RTX 3080 Laptop `GA104M`):

| Kernel (`holosphere_monarch2_head::*`) | Head FP16 (B) | Regs | Barriers | SMEM (B) | Stack / Spill (B) | `LDL` / `STL` | FP32-vs-FP64 Max Err (FP16 Weights) | Uncontrolled RTX 3080 Laptop Latency (`us/tok`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `holosphere_arm2b_bias_514b_sm86` | `514` | `18` | `1` | `3,072` | `0 / 0 / 0` | `0 / 0` | `1.4137e-7` | `1.2614` |
| `holosphere_arm5_bias_2p5kib_sm86` | `2,560` | `22` | `1` | `3,072` | `0 / 0 / 0` | `0 / 0` | `1.3427e-7` | `1.8036` |
| `holosphere_arm7_bias_8p5kib_sm86` | `8,704` | `26` | `1` | `12,288` | `0 / 0 / 0` | `0 / 0` | `1.4565e-7` | `2.1912` |
| **`holosphere_arm9_monarch2_bias_16p5kib_sm86`** | **`16,896`** | **`27`** | **`1`** | **`20,480`** | **`0 / 0 / 0`** | **`0 / 0`** | **`1.6907e-7`** | **`2.6702`** |

All four kernels execute `H_256` (`1/16` orthonormal) and `FWHT_256 x bent4` (`1/sqrt(32)` = `sqrt(8) * orthonormal`) using warp-shuffle butterflies + a single CTA barrier, and `holosphere_arm9_monarch2_bias_16p5kib_sm86` loads all `16 KiB` of both Monarch-`16x16` stages (`B_1` + `B_2`) in `2` coalesced `LDG.128` (`uint4`) loads per thread (`512` threads/CTA), keeping `100%` of the head resident in CTA shared memory (`20,480 B`) with `0 B` HBM vocabulary table reads.

---

## 2. Controlled 6-Arm x 3-Seed UTF-8 BPB Showdown (Exact FP16 Head Eval)

Executed by the committed standalone script [`cuda/sm86/monarch2_fwht_head_bpb_study.py`](monarch2_fwht_head_bpb_study.py) (`bpb_study_script_sha256 = 1d1d662099a0cc48229c21d6d532c7fbfe62a28b96ff1e508cda20cb3e921341`) on an iso-budget 4-layer `d_model = 512`, `d_ff = 800`, `n_heads = 16` backbone across `3` seeds `{42, 137, 2026}` (`300` steps x `4,096` tokens/step, evaluated on `32,768` held-out UTF-8 bytes under exact FP16 head weights matching `sm_86` kernel inputs). External corpus identifier & SHA-256 bindings (`holosphere-v3-flagship-corpus`, raw binary not checked into this kernel repo):
- `corpus_sha256` (`flagship_corpus_manifest.json`): `a05e415f7d33ef32376ff3d9da1eede0e3f561b64094b5b91147b2696a99e9d4`
- `corpus_train_bin_sha256` (`train.bin`, `18,402,047 B`): `528c6c792d2a563d82d21ce213b26209441b39fa5e76b8595607bab2ca853b7b`
- `corpus_eval_bin_sha256` (`eval.bin`, `2,927,192 B`): `31fd132d888a21ecebe9e907b6c4b82e3e68e42b5a29e2839584447dcca631a0`

Unbiased `1`-scalar reference baseline (`2 B`, `0 B` HBM table): `5.3706 BPB` (`399633111796751/74411611509180`).

| Head Configuration | Head Params | Head FP16 (B) | HBM Table (B) | Seed 42 | Seed 137 | Seed 2026 | Mean BPB | Lift vs Unbiased Scalar | Lift vs `arm2b_bias` | Max FP16 Quant Diff |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arm2b_bias_514B` (`alpha * FWHT + b`) | `257` | `514` | `0` | `4.9695` | `4.9963` | `4.9866` | `4.9841` | `+0.3864` | `0.0000` | `0.011390` |
| `arm5_bias_2p5KiB` (`D_1 H_256 + b`) | `1,280` | `2,560` | `0` | `4.3025` | `4.3012` | `4.3453` | `4.3163` | `+1.0542` | `+0.6678` | `0.000137` |
| `arm7_bias_8p5KiB` (1-stage Monarch + `b`) | `4,352` | `8,704` | `0` | `4.2948` | `4.2813` | `4.3035` | `4.2932` | `+1.0774` | `+0.6909` | `0.000214` |
| `arm8a_lowrank16_bias_16p9KiB` (Rank-16 + `b`) | `8,448` | `16,896` | `16,896` | `4.4172` | `4.5355` | `4.6195` | `4.5241` | `+0.8465` | `+0.4600` | `0.000749` |
| **`arm9_monarch2_bias_16p5KiB`** (2-stage Monarch + `b`) | **`8,448`** | **`16,896`** | **`0`** | **`3.6516`** | **`3.6595`** | **`3.6742`** | **`3.6617`** | **`+1.7088`** | **`+1.3224`** | **`0.000275`** |
| `arm4_dense_bias_262KiB` (Dense `256x512` + `b`) | `131,329` | `262,658` | `262,658` | `3.5239` | `3.5123` | `3.5411` | `3.5258` | `+1.8448` | `+1.4583` | `0.000001` |

### Mechanistic Attribution & Scope
1. **Pure `+512 B` Byte-Unigram Bias:** Adding `256` FP16 byte biases to the unbiased scalar baseline lifts held-out BPB by `+0.3864` (`5.3706 -> 4.9841`).
2. **Full-Rank Structured Pre-Mixing vs Low-Rank Bottleneck:** Even `2.5 KiB` `arm5_bias` (`4.3163`) and `8.5 KiB` `arm7_bias` (`4.2932`) beat `16.9 KiB` rank-16 low-rank `arm8a` (`4.5241`) on `3/3` seeds because rank-16 confines logits to a 16-dimensional subspace of `R^256`.
3. **Why 2-Stage Monarch (`arm9`) Jumps from `4.2932` to `3.6617` (`92.63%` Dense Gap Closed):** A single Monarch-`16x16` stage (`B_1 P_{16x16}`) only reaches `16` of `256` input coordinates per output before `FWHT_256`, whereas completing the second `16x16` stage (`B_2 P_{16x16} B_1 P_{16x16}`) achieves exact **`256 / 256` all-to-all bipartite connectivity** (certified in pure `Z` alongside `6` fixed known-answer delta-to-signed-Hadamard columns by `verify_monarch2_fwht_head.py`). At exact `1:1` byte/parameter parity with rank-16 low-rank (`8,448` params / `16,896 B`), `arm9` beats `arm8a` by **`0.8623 BPB`** (`3/3` seeds), closes **`92.63%`** of the `1.8448 BPB` dense gap, and trails the `15.5x` larger `262,658 B` dense head by just **`0.1360 BPB`** with `0 B` HBM vocab table reads.
4. **Honest Scope:** Measurements are at `d_model = 512`, `V = 256` UTF-8 bytes, and `300` steps (`1.23M` tokens/seed); scaling to larger BPE vocabularies via factored radix-`64`/`128` Monarch-FWHT and longer training horizons remains open follow-on work.

---

## 3. Build & Offline Verification

```bash
# Pure-stdlib 3-mode isolated -I receipt + pure-Z oracle + scale-token + tamper-refusal verification (no GPU required)
make -f cuda/sm86/holosphere_monarch2_fwht_head.mk verify-monarch2-head

# On NVIDIA Ampere sm_86 with nvcc: compile (--Werror all-warnings) and verify 4/4 zero-spill kernels
make -f cuda/sm86/holosphere_monarch2_fwht_head.mk check-monarch2-head-zero-spill

# On CUDA with PyTorch and pinned holosphere-v3-flagship-corpus: reproduce the 6-arm x 3-seed UTF-8 BPB study
python3 cuda/sm86/monarch2_fwht_head_bpb_study.py --corpus-dir /path/to/holosphere-v3-flagship-corpus
```
