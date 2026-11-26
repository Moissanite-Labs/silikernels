#!/usr/bin/env python3
"""Track S125 (EXP-Q30b, PREREG-W14) pure-stdlib fail-closed verifier for Silikernels.

Verifies:
  1. Independent pure-stdlib algebraic certificates (zero shared C code):
     - Closed-form HoloSphere-V3 15M live arena footprint (< 12 MiB < 24 MiB L3),
     - Exact [16, 11, 4] Extended Hamming A_16 nibble-LSB plane pack + 16/16 SEC + 120/120 DED,
     - Exact H_4 VPDPBUSD +8 bias-shift equivalence over int4 x int8,
     - Exact CayleySU2 unit-norm preservation and FWHT_256 x bent4 orthonormality.
  2. Strict fail-closed audit of receipts/s125_avx512_l3_benchmark_results.json
     cryptographically bound to cpu/avx512/holosphere_v3_avx512.c SHA-256, pinned
     deterministic trace digests/tokens, PREREG-W14 SHA-256, full A/A & interleaved
     summary recomputation, and honest huge-page verdict enforcement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PREREG_W14_SHA256 = "871900e5d6c860ac159f43ca6b4767e7c128d0b4f7748edf788d56e9cb535d3e"
SCHEMA_VERSION = "s125-avx512-l3-v1"
EXPECTED_C_SOURCE_FILE = "cpu/avx512/holosphere_v3_avx512.c"
EXPECTED_C_SOURCE_SHA256 = "73019680c8399aa734e22fc1121c2a3577e42af1c85d4909a215b7ba7890352f"
EXPECTED_PROMPT = [72, 111, 108, 111, 83, 112, 104, 101]
EXPECTED_LOGITS_FNV1A64 = "6fc017bea5754bb1"
EXPECTED_HIDDEN_FNV1A64 = "3a65e5960be737a7"
EXPECTED_GREEDY_TOKENS = [149, 239, 169, 143, 89, 215, 50, 229]

HS_D_MODEL = 512
HS_N_HEADS = 16
HS_N_SPINORS = 128
HS_N_LAYERS = 6
HS_D_FF = 800
HS_RING_K = 8
HS_CONV_K = 4
HS_C_DIM = 48
HS_W_OMEGA_D = 384

LIMIT_LIVE_BYTES = 12 * 1024 * 1024
L3_CACHE_BYTES = 24 * 1024 * 1024
CAYLEY_NORM_TOL = 2.0 ** -44   # ~5.68e-14 (exact IEEE-754 binary64 boundary)
FWHT_ORTHO_TOL = 2.0 ** -36    # ~1.46e-11 (exact IEEE-754 binary64 boundary)


class S125Error(ValueError):
    """Fail-closed verification exception for Track S125."""


def require_ok(cond: bool, msg: str) -> None:
    if not cond:
        raise S125Error(msg)


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_real(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _no_dup_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in obj, f"Duplicate JSON key: {k}")
        obj[k] = v
    return obj


def _reject_non_finite(token: str) -> Any:
    raise S125Error(f"Non-finite JSON constant rejected: {token}")


def load_json_object_strict(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(
        raw,
        parse_constant=_reject_non_finite,
        object_pairs_hook=_no_dup_object_pairs,
    )
    require_ok(isinstance(data, dict), "Top-level JSON must be an object")
    return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 1. Pure-Stdlib Algebraic Certificates (Zero Shared C Code)
# ============================================================================

def _align64(n: int) -> int:
    return (n + 63) & ~63


def _a16_matrix_bytes(out_dim: int, in_dim: int) -> int:
    u64_b = _align64(out_dim * (in_dim // 16) * 8)
    sc_b = _align64(out_dim * (in_dim // 32) * 2)
    return u64_b + sc_b


def compute_exact_arena_bytes() -> tuple[int, int, int]:
    layer_w = (
        _a16_matrix_bytes(HS_D_MODEL, HS_D_MODEL) * 4
        + _a16_matrix_bytes(HS_C_DIM, HS_D_MODEL)
        + _a16_matrix_bytes(HS_W_OMEGA_D, HS_D_MODEL)
        + _a16_matrix_bytes(HS_D_FF, HS_D_MODEL) * 2
        + _a16_matrix_bytes(HS_D_MODEL, HS_D_FF)
        + _align64(HS_N_HEADS * 32 * 32 * 4)
        + _align64(HS_N_SPINORS * 4)
    )
    weight_bytes = _align64(HS_D_MODEL * HS_CONV_K * 4) + HS_N_LAYERS * layer_w
    state_bytes = (
        _align64(HS_CONV_K * HS_D_MODEL * 4)
        + _align64(HS_N_LAYERS * HS_N_HEADS * 32 * 32 * 4)
        + _align64(HS_N_LAYERS * HS_N_HEADS * 8 * 4 * 4)
        + _align64(HS_N_LAYERS * HS_N_HEADS * HS_RING_K * 32 * 4) * 2
    )
    return weight_bytes, state_bytes, weight_bytes + state_bytes


def _a16_syndrome_u64(w64: int) -> int:
    lsb = w64 & 0x1111111111111111
    p0 = lsb.bit_count() & 1
    p1 = (lsb & 0x1010101010101010).bit_count() & 1
    p2 = (lsb & 0x1100110011001100).bit_count() & 1
    p3 = (lsb & 0x1111000011110000).bit_count() & 1
    p4 = (lsb & 0x1111111100000000).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def _a16_secded_heal(w64: int) -> tuple[int, int, int]:
    syn = _a16_syndrome_u64(w64)
    if syn == 0:
        return w64, 0, 0
    if (syn & 1) != 0:
        pos = (syn >> 1) & 15
        return w64 ^ (1 << (pos * 4)), 1, 0
    return w64, 0, 1


def check_arena_budget(total_bytes: int) -> None:
    require_ok(
        _is_strict_int(total_bytes) and 0 < total_bytes < LIMIT_LIVE_BYTES,
        "Live arena footprint must be strictly within (0, 12 MiB)",
    )


def check_secded_probe(syn: int, sec_ok: int, ded_ok: int) -> None:
    require_ok(syn == 0, "A_16 nibble-LSB codeword syndrome != 0")
    require_ok(sec_ok == 16, "SECDED 16/16 single-bit LSB heal failed")
    require_ok(ded_ok == 120, "SECDED 120/120 double-bit LSB detect failed")


def check_secded_sec_step(healed: int, expected: int, sec: int, ded: int) -> None:
    require_ok(healed == expected and sec == 1 and ded == 0, "Single-bit LSB SEC step failed")


def check_secded_ded_step(healed: int, expected: int, sec: int, ded: int) -> None:
    require_ok(healed == expected and sec == 0 and ded == 1, "Double-bit LSB DED step failed")


def check_vpdpbusd_bias(c32: list[int], xq: list[int], claimed_dot: int) -> None:
    dot_signed = sum(c * x for c, x in zip(c32, xq))
    dot_biased = sum((c + 8) * x for c, x in zip(c32, xq)) - 8 * sum(xq)
    require_ok(
        dot_signed == dot_biased and dot_signed == claimed_dot,
        "VPDPBUSD +8 bias-shift identity mismatch",
    )


def check_cayley_unit(q: tuple[float, float, float, float]) -> float:
    q_norm_err = abs(sum(v * v for v in q) - 1.0)
    require_ok(q_norm_err < CAYLEY_NORM_TOL, "CayleySU2 quaternion norm drifted from 1.0")
    return q_norm_err


def check_fwht_orthonormal(cols: list[list[float]]) -> None:
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ip = sum(a * b for a, b in zip(ci, cj))
            target = 1.0 if i == j else 0.0
            require_ok(abs(ip - target) < FWHT_ORTHO_TOL, "FWHT_256 x bent4 orthonormality failed")


def verify_algebraic_certificates() -> dict[str, Any]:
    """Independent pure-stdlib check of footprint, A_16 nibble-LSB SEC/DED,
    VPDPBUSD bias, CayleySU2 norm, and FWHT_256 x bent4 orthonormality."""
    w_b, s_b, tot_b = compute_exact_arena_bytes()
    check_arena_budget(tot_b)

    c16 = [-7, 0, 0, 5, 0, -3, 2, 7, 0, -8, 4, -1, 6, -5, 3, -2]
    p_pos = (1, 2, 4, 8)
    p_idx = (
        (3, 5, 7, 9, 11, 13, 15),
        (3, 6, 7, 10, 11, 14, 15),
        (5, 6, 7, 12, 13, 14, 15),
        (9, 10, 11, 12, 13, 14, 15),
    )
    for pos, idxs in zip(p_pos, p_idx):
        req = 0
        for idx in idxs:
            req ^= c16[idx] & 1
        c16[pos] = req
    req0 = 0
    for j in range(1, 16):
        req0 ^= c16[j] & 1
    c16[0] = req0

    w64 = 0
    for j, val in enumerate(c16):
        w64 |= (val & 0xF) << (j * 4)
    syn = _a16_syndrome_u64(w64)

    sec_ok = 0
    for bit in range(16):
        healed, sec, ded = _a16_secded_heal(w64 ^ (1 << (bit * 4)))
        check_secded_sec_step(healed, w64, sec, ded)
        sec_ok += 1

    ded_ok = 0
    for i in range(16):
        for j in range(i + 1, 16):
            cor = w64 ^ (1 << (i * 4)) ^ (1 << (j * 4))
            healed, sec, ded = _a16_secded_heal(cor)
            check_secded_ded_step(healed, cor, sec, ded)
            ded_ok += 1
    check_secded_probe(syn, sec_ok, ded_ok)

    xq = [((k * 37 - 113) % 255) - 127 for k in range(32)]
    c32 = c16 + [((-v) if v != -8 else 7) for v in c16]
    dot_ref = sum(c * x for c, x in zip(c32, xq))
    check_vpdpbusd_bias(c32, xq, dot_ref)

    wx, wy, wz = 0.73, -1.21, 0.49
    w2 = wx * wx + wy * wy + wz * wz
    inv = 1.0 / (1.0 + w2)
    q = ((1.0 - w2) * inv, 2.0 * wx * inv, 2.0 * wy * inv, 2.0 * wz * inv)
    q_norm_err = check_cayley_unit(q)

    inv_s = 1.0 / math.sqrt(512.0)
    cols: list[list[float]] = []
    for b in (0, 72, 111, 255):
        col = []
        for d in range(512):
            d256 = d & 255
            w_ = d256 >> 5
            ln = d256 & 31
            db = ((w_ << 5) | ln) ^ ((ln & 7) << 5)
            b4 = ((ln & 3) & ((ln >> 2) & 3)).bit_count() & 1
            par = ((b & db).bit_count() ^ b4) & 1
            val = -inv_s if par else inv_s
            if d >= 256:
                val = -val
            col.append(val)
        cols.append(col)
    check_fwht_orthonormal(cols)

    return {
        "closed_form_weight_bytes": w_b,
        "closed_form_state_bytes": s_b,
        "closed_form_total_live_bytes": tot_b,
        "secded_lsb_plane_sec_16_of_16": True,
        "secded_lsb_plane_ded_120_of_120": True,
        "vpdpbusd_bias_exact": True,
        "cayley_unit_norm_err": q_norm_err,
        "fwht256_bent4_orthonormal": True,
    }


# ============================================================================
# 2. Fail-Closed Receipt Verification (All G1 / G2 / A-A / Huge-Page Invariants)
# ============================================================================

def _verify_aa_block(aa: Any, label: str) -> float:
    require_ok(isinstance(aa, dict), f"{label} must be an object")
    n_runs = aa.get("n_runs")
    runs = aa.get("tok_per_s_runs")
    require_ok(_is_strict_int(n_runs) and n_runs >= 10, f"{label} n_runs must be int >= 10")
    require_ok(isinstance(runs, list) and len(runs) == n_runs, f"{label} tok_per_s_runs length mismatch")
    require_ok(all(_is_finite_real(x) and float(x) > 0.0 for x in runs), f"{label} tok_per_s_runs invalid")
    require_ok(aa.get("smaps_anon_hugepages_kb_all_zero") is True, f"{label} 4 KiB smaps flag must be True")

    vals = [float(x) for x in runs]
    med = statistics.median(vals)
    mean = statistics.fmean(vals)
    stdev = statistics.stdev(vals)
    qs = statistics.quantiles(vals, n=4, method="inclusive")
    iqr_pct = 100.0 * (qs[2] - qs[0]) / med
    cv_pct = 100.0 * stdev / mean
    spread_pct = 100.0 * (max(vals) - min(vals)) / med
    floor_pct = max(2.0 * cv_pct, iqr_pct, 2.0)
    naive_wins = sum(1 for i in range(1, len(vals) - 1) if vals[i] > vals[i - 1] and vals[i] > vals[i + 1])

    require_ok(_is_finite_real(aa.get("median_tok_per_s")) and abs(float(aa["median_tok_per_s"]) - med) <= 0.01, f"{label} median_tok_per_s mismatch")
    require_ok(_is_finite_real(aa.get("mean_tok_per_s")) and abs(float(aa["mean_tok_per_s"]) - mean) <= 0.01, f"{label} mean_tok_per_s mismatch")
    require_ok(_is_finite_real(aa.get("stdev_tok_per_s")) and abs(float(aa["stdev_tok_per_s"]) - stdev) <= 0.02, f"{label} stdev_tok_per_s mismatch")
    require_ok(_is_finite_real(aa.get("cv_pct")) and abs(float(aa["cv_pct"]) - cv_pct) <= 0.02, f"{label} cv_pct mismatch")
    require_ok(_is_finite_real(aa.get("iqr_pct")) and abs(float(aa["iqr_pct"]) - iqr_pct) <= 0.02, f"{label} iqr_pct mismatch")
    require_ok(_is_finite_real(aa.get("max_rel_spread_pct")) and abs(float(aa["max_rel_spread_pct"]) - spread_pct) <= 0.02, f"{label} max_rel_spread_pct mismatch")
    require_ok(_is_finite_real(aa.get("noise_floor_pct")) and abs(float(aa["noise_floor_pct"]) - floor_pct) <= 0.02, f"{label} noise_floor_pct mismatch")
    require_ok(_is_strict_int(aa.get("aa_sliding_windows")) and aa["aa_sliding_windows"] == n_runs - 2, f"{label} aa_sliding_windows mismatch")
    require_ok(_is_strict_int(aa.get("aa_naive_middle_beats_both")) and aa["aa_naive_middle_beats_both"] == naive_wins, f"{label} aa_naive_middle_beats_both mismatch")
    return floor_pct


def _verify_interleaved_block(
    blk: Any,
    label: str,
    *,
    expect_ctl_huge: bool,
    expect_cand_huge: bool,
    min_speedup: float | None,
) -> dict[str, float | int]:
    require_ok(isinstance(blk, dict), f"{label} must be an object")
    n_triples = blk.get("n_triples")
    triples = blk.get("triples")
    require_ok(_is_strict_int(n_triples) and n_triples >= 5, f"{label} n_triples must be int >= 5")
    require_ok(isinstance(triples, list) and len(triples) == n_triples, f"{label} triples length mismatch")

    ctl_vals: list[float] = []
    cand_vals: list[float] = []
    paired_deltas: list[float] = []
    beats_both = 0
    beats_mean = 0

    for idx, tr in enumerate(triples):
        require_ok(isinstance(tr, dict), f"{label} triple[{idx}] must be an object")
        for k in ("ctl_pre_tok_s", "cand_tok_s", "ctl_post_tok_s"):
            require_ok(_is_finite_real(tr.get(k)) and float(tr[k]) > 0.0, f"{label} triple[{idx}].{k} invalid")
        t_pre = float(tr["ctl_pre_tok_s"])
        t_cand = float(tr["cand_tok_s"])
        t_post = float(tr["ctl_post_tok_s"])
        ctl_vals.extend([t_pre, t_post])
        cand_vals.append(t_cand)
        mid = 0.5 * (t_pre + t_post)
        paired_deltas.append(100.0 * (t_cand - mid) / mid)
        if t_cand > t_pre and t_cand > t_post:
            beats_both += 1
        if t_cand > mid:
            beats_mean += 1

    ctl_med = statistics.median(ctl_vals)
    cand_med = statistics.median(cand_vals)
    speedup = cand_med / ctl_med
    delta_pct = 100.0 * (cand_med - ctl_med) / ctl_med
    med_paired = statistics.median(paired_deltas)
    min_paired = min(paired_deltas)
    max_paired = max(paired_deltas)

    require_ok(_is_finite_real(blk.get("ctl_median_tok_per_s")) and abs(float(blk["ctl_median_tok_per_s"]) - ctl_med) <= 0.02, f"{label} ctl_median_tok_per_s mismatch")
    require_ok(_is_finite_real(blk.get("cand_median_tok_per_s")) and abs(float(blk["cand_median_tok_per_s"]) - cand_med) <= 0.02, f"{label} cand_median_tok_per_s mismatch")
    require_ok(_is_finite_real(blk.get("median_speedup")) and abs(float(blk["median_speedup"]) - speedup) <= 0.005, f"{label} median_speedup mismatch")
    require_ok(_is_finite_real(blk.get("delta_pct")) and abs(float(blk["delta_pct"]) - delta_pct) <= 0.05, f"{label} delta_pct mismatch")
    require_ok(_is_finite_real(blk.get("median_paired_delta_pct")) and abs(float(blk["median_paired_delta_pct"]) - med_paired) <= 0.05, f"{label} median_paired_delta_pct mismatch")
    require_ok(_is_finite_real(blk.get("min_paired_delta_pct")) and abs(float(blk["min_paired_delta_pct"]) - min_paired) <= 0.05, f"{label} min_paired_delta_pct mismatch")
    require_ok(_is_finite_real(blk.get("max_paired_delta_pct")) and abs(float(blk["max_paired_delta_pct"]) - max_paired) <= 0.05, f"{label} max_paired_delta_pct mismatch")
    require_ok(_is_strict_int(blk.get("triples_cand_beats_both_ctl")) and int(blk["triples_cand_beats_both_ctl"]) == beats_both, f"{label} triples_cand_beats_both_ctl mismatch")
    require_ok(_is_strict_int(blk.get("triples_cand_beats_mean_ctl")) and int(blk["triples_cand_beats_mean_ctl"]) == beats_mean, f"{label} triples_cand_beats_mean_ctl mismatch")

    ctl_min = blk.get("ctl_smaps_anon_hugepages_kb_min")
    ctl_max = blk.get("ctl_smaps_anon_hugepages_kb_max")
    cand_min = blk.get("cand_smaps_anon_hugepages_kb_min")
    cand_max = blk.get("cand_smaps_anon_hugepages_kb_max")
    for k, v in (("ctl_min", ctl_min), ("ctl_max", ctl_max), ("cand_min", cand_min), ("cand_max", cand_max)):
        require_ok(_is_strict_int(v) and v >= 0, f"{label} smaps {k} must be non-negative int")
    if expect_ctl_huge:
        require_ok(int(ctl_min) >= 2048, f"{label} expected ctl hugepages >= 2048 kB")
    else:
        require_ok(int(ctl_max) == 0, f"{label} expected ctl hugepages == 0 kB")
    if expect_cand_huge:
        require_ok(int(cand_min) >= 2048, f"{label} expected cand hugepages >= 2048 kB")
    else:
        require_ok(int(cand_max) == 0, f"{label} expected cand hugepages == 0 kB")

    if min_speedup is not None:
        require_ok(speedup >= min_speedup, f"{label} median_speedup {speedup:.4f} < {min_speedup}")

    return {
        "speedup": speedup,
        "delta_pct": delta_pct,
        "median_paired_delta_pct": med_paired,
        "beats_both": beats_both,
    }


def expected_hugepage_verdict(delta_pct: float, med_paired_pct: float, beats_both: int, noise_floor_pct: float) -> str:
    if delta_pct > noise_floor_pct and med_paired_pct > noise_floor_pct and beats_both >= 5:
        return "CONFIRMED"
    if delta_pct < -noise_floor_pct and med_paired_pct < -noise_floor_pct and beats_both <= 1:
        return "REGRESSION"
    return "NEUTRAL / L3-ABSORBED"


def verify_receipt_dict(receipt: dict[str, Any], c_source_path: Path) -> dict[str, Any]:
    alg = verify_algebraic_certificates()

    require_ok(isinstance(receipt, dict), "Receipt must be a JSON object")
    require_ok(receipt.get("schema_version") == SCHEMA_VERSION, "Unsupported schema_version")
    require_ok(receipt.get("track") == "S125", "track must be S125")
    require_ok(receipt.get("experiment_id") == "EXP-Q30b", "experiment_id must be EXP-Q30b")
    require_ok(receipt.get("prereg_w14_sha256") == PREREG_W14_SHA256, "PREREG-W14 SHA-256 mismatch")
    require_ok(receipt.get("cpu_only_enforced") is True, "cpu_only_enforced must be True")

    # Pinned source-code SHA-256 cryptographic binding
    require_ok(c_source_path.is_file(), "Missing cpu/avx512/holosphere_v3_avx512.c")
    actual_c_sha = sha256_file(c_source_path)
    require_ok(actual_c_sha == EXPECTED_C_SOURCE_SHA256, "Live C source SHA-256 drifted from pinned commit")
    require_ok(receipt.get("c_source_file") == EXPECTED_C_SOURCE_FILE, "c_source_file mismatch")
    require_ok(receipt.get("c_source_sha256") == actual_c_sha, "receipt c_source_sha256 mismatch")

    # Build, sanitizer, & objdump ISA proof
    b_isa = receipt.get("build_and_isa")
    require_ok(isinstance(b_isa, dict), "build_and_isa must be an object")
    require_ok(_is_strict_int(b_isa.get("gcc_o3_warnings")) and b_isa["gcc_o3_warnings"] == 0, "gcc_o3_warnings must be 0")
    require_ok(b_isa.get("ubsan_asan_pass") is True, "ubsan_asan_pass must be True")
    require_ok(_is_strict_int(b_isa.get("objdump_vpdpbusd_count")) and b_isa["objdump_vpdpbusd_count"] >= 1, "objdump_vpdpbusd_count must be >= 1")
    require_ok(_is_strict_int(b_isa.get("objdump_vpopcntq_count")) and b_isa["objdump_vpopcntq_count"] >= 1, "objdump_vpopcntq_count must be >= 1")
    require_ok(_is_strict_int(b_isa.get("objdump_vpermt2d_count")) and b_isa["objdump_vpermt2d_count"] >= 1, "objdump_vpermt2d_count must be >= 1")

    # L3-resident footprint exact binding
    fp = receipt.get("footprint")
    require_ok(isinstance(fp, dict), "footprint must be an object")
    require_ok(_is_strict_int(fp.get("weight_bytes")) and fp["weight_bytes"] == alg["closed_form_weight_bytes"], "weight_bytes mismatch")
    require_ok(_is_strict_int(fp.get("state_bytes")) and fp["state_bytes"] == alg["closed_form_state_bytes"], "state_bytes mismatch")
    require_ok(_is_strict_int(fp.get("total_live_bytes")) and fp["total_live_bytes"] == alg["closed_form_total_live_bytes"], "total_live_bytes mismatch")
    require_ok(fp.get("under_12_mib_gate") is True, "under_12_mib_gate must be True")
    require_ok(_is_strict_int(fp.get("smaps_4k_anon_hugepages_kb")) and fp["smaps_4k_anon_hugepages_kb"] == 0, "smaps_4k_anon_hugepages_kb must be 0")
    require_ok(_is_strict_int(fp.get("smaps_2m_anon_hugepages_kb")) and fp["smaps_2m_anon_hugepages_kb"] >= 2048, "smaps_2m_anon_hugepages_kb must be >= 2048")

    # G1 Pinned Parity, Multi-Thread Determinism, & Live SECDED
    par = receipt.get("parity_and_ecc")
    require_ok(isinstance(par, dict), "parity_and_ecc must be an object")
    require_ok(par.get("prompt_bytes") == EXPECTED_PROMPT, "prompt_bytes mismatch")
    require_ok(par.get("logits_fnv1a64") == EXPECTED_LOGITS_FNV1A64, "logits_fnv1a64 does not match pinned trace digest")
    require_ok(par.get("hidden_fnv1a64") == EXPECTED_HIDDEN_FNV1A64, "hidden_fnv1a64 does not match pinned trace digest")
    require_ok(par.get("greedy_tokens") == EXPECTED_GREEDY_TOKENS, "greedy_tokens does not match pinned trace tokens")
    require_ok(par.get("bit_exact_scalar_avx512_1t_4t_8t_2m_ubsan") is True, "bit_exact across configs must be True")
    require_ok(_is_strict_int(par.get("pack_syndrome_errors")) and par["pack_syndrome_errors"] == 0, "pack_syndrome_errors must be 0")
    require_ok(_is_strict_int(par.get("runtime_syndrome_faults")) and par["runtime_syndrome_faults"] == 0, "runtime_syndrome_faults must be 0")
    require_ok(par.get("secded_16sec_120ded_pass") is True, "secded_16sec_120ded_pass must be True")
    require_ok(_is_finite_real(par.get("fp64_oracle_max_abs_logit_diff")) and 0.0 <= float(par["fp64_oracle_max_abs_logit_diff"]) <= 1e-5, "fp64_oracle_max_abs_logit_diff must be in [0, 1e-5]")
    require_ok(_is_finite_real(par.get("fp64_oracle_max_abs_hidden_diff")) and 0.0 <= float(par["fp64_oracle_max_abs_hidden_diff"]) <= 1e-5, "fp64_oracle_max_abs_hidden_diff must be in [0, 1e-5]")
    require_ok(par.get("fp64_greedy_match_8_of_8") is True, "fp64_greedy_match_8_of_8 must be True")
    require_ok(par.get("fp64_gate_le_1e5_pass") is True, "fp64_gate_le_1e5_pass must be True")

    # Empirical A/A noise floor (>= 10 runs each)
    aa = receipt.get("aa_noise_floor")
    require_ok(isinstance(aa, dict), "aa_noise_floor must be an object")
    floor_1t = _verify_aa_block(aa.get("avx512_1t_4k"), "aa_noise_floor.avx512_1t_4k")
    floor_8t = _verify_aa_block(aa.get("avx512_8t_4k"), "aa_noise_floor.avx512_8t_4k")

    # Interleaved ctl / cand / ctl comparisons
    ib = receipt.get("interleaved_benchmarks")
    require_ok(isinstance(ib, dict), "interleaved_benchmarks must be an object")
    avx_1t = _verify_interleaved_block(
        ib.get("scalar_vs_avx512_1t_4k"),
        "scalar_vs_avx512_1t_4k",
        expect_ctl_huge=False,
        expect_cand_huge=False,
        min_speedup=1.5,
    )
    avx_8t = _verify_interleaved_block(
        ib.get("scalar_vs_avx512_8t_4k"),
        "scalar_vs_avx512_8t_4k",
        expect_ctl_huge=False,
        expect_cand_huge=False,
        min_speedup=1.5,
    )
    hp_1t = _verify_interleaved_block(
        ib.get("pages_4k_vs_2m_1t_avx512"),
        "pages_4k_vs_2m_1t_avx512",
        expect_ctl_huge=False,
        expect_cand_huge=True,
        min_speedup=None,
    )
    hp_8t = _verify_interleaved_block(
        ib.get("pages_4k_vs_2m_8t_avx512"),
        "pages_4k_vs_2m_8t_avx512",
        expect_ctl_huge=False,
        expect_cand_huge=True,
        min_speedup=None,
    )

    exp_v1t = expected_hugepage_verdict(
        float(hp_1t["delta_pct"]), float(hp_1t["median_paired_delta_pct"]), int(hp_1t["beats_both"]), floor_1t
    )
    exp_v8t = expected_hugepage_verdict(
        float(hp_8t["delta_pct"]), float(hp_8t["median_paired_delta_pct"]), int(hp_8t["beats_both"]), floor_8t
    )
    require_ok(ib["pages_4k_vs_2m_1t_avx512"].get("verdict") == exp_v1t, f"1T hugepage verdict mismatch (expected {exp_v1t})")
    require_ok(ib["pages_4k_vs_2m_8t_avx512"].get("verdict") == exp_v8t, f"8T hugepage verdict mismatch (expected {exp_v8t})")

    gates = receipt.get("gates")
    require_ok(isinstance(gates, dict), "gates must be an object")
    require_ok(gates.get("G1_parity_determinism_ecc") == "CONFIRMED", "G1 gate must be CONFIRMED")
    require_ok(gates.get("G2_l3_footprint_and_avx512_speedup") == "CONFIRMED", "G2 gate must be CONFIRMED")
    require_ok(gates.get("hugepage_2m_vs_4k_1t_verdict") == exp_v1t, "gates 1T hugepage verdict mismatch")
    require_ok(gates.get("hugepage_2m_vs_4k_8t_verdict") == exp_v8t, "gates 8T hugepage verdict mismatch")

    summary = {
        "status": "PASS",
        "track": "S125",
        "experiment_id": "EXP-Q30b",
        "prereg_w14_sha256": PREREG_W14_SHA256,
        "c_source_sha256": actual_c_sha,
        "logits_fnv1a64": EXPECTED_LOGITS_FNV1A64,
        "hidden_fnv1a64": EXPECTED_HIDDEN_FNV1A64,
        "total_live_bytes": fp["total_live_bytes"],
        "l3_occupancy_pct": round(100.0 * fp["total_live_bytes"] / L3_CACHE_BYTES, 3),
        "fp64_max_abs_logit_diff": par["fp64_oracle_max_abs_logit_diff"],
        "avx512_1t_speedup": round(float(avx_1t["speedup"]), 4),
        "avx512_8t_speedup": round(float(avx_8t["speedup"]), 4),
        "hugepage_1t_verdict": exp_v1t,
        "hugepage_8t_verdict": exp_v8t,
        "algebraic_certificates": alg,
    }
    summary["summary_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return summary


def verify_repo(repo_root: Path) -> dict[str, Any]:
    receipt_path = repo_root / "receipts" / "s125_avx512_l3_benchmark_results.json"
    c_source_path = repo_root / "cpu" / "avx512" / "holosphere_v3_avx512.c"
    require_ok(receipt_path.is_file(), "Missing receipts/s125_avx512_l3_benchmark_results.json")
    receipt = load_json_object_strict(receipt_path)
    return verify_receipt_dict(receipt, c_source_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Silikernels Track S125 (EXP-Q30b) receipt and certificates.")
    parser.add_argument("--check", action="store_true", help="Run fail-closed verification.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent, help="Silikernels root.")
    args = parser.parse_args(argv)
    summary = verify_repo(args.repo_root)
    if args.json:
        print(json.dumps(summary, sort_keys=True, indent=2))
    else:
        print(f"PASS S125 EXP-Q30b summary_sha256={summary['summary_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
