#!/usr/bin/env python3
"""Assertion-free R5 test suite for Silikernels Track S125 (EXP-Q30b) targeting
100.0% mutation kill rate under aletheia mutate --guards require_ok --max-mutants 0.
"""

from __future__ import annotations

import contextlib
import copy
import io
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parent
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

import verify_s125_avx512_l3 as mod


C_SRC = REPO_ROOT / "cpu" / "avx512" / "holosphere_v3_avx512.c"
RECEIPT_PATH = REPO_ROOT / "receipts" / "s125_avx512_l3_benchmark_results.json"


class TestFailure(RuntimeError):
    """Explicit test failure exception (zero bare assert statements)."""


def expect_true(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFailure(msg)


def expect_s125_error(fn: Callable[[], Any], label: str) -> None:
    try:
        fn()
    except mod.S125Error:
        return
    raise TestFailure(f"Expected S125Error for tamper case: {label}")


def _make_consistent_aa(runs: list[float], n_runs: int | None = None) -> dict[str, Any]:
    med = statistics.median(runs)
    mean = statistics.fmean(runs)
    stdev = statistics.stdev(runs)
    qs = statistics.quantiles(runs, n=4, method="inclusive")
    iqr_pct = 100.0 * (qs[2] - qs[0]) / med
    cv_pct = 100.0 * stdev / mean
    spread_pct = 100.0 * (max(runs) - min(runs)) / med
    floor_pct = max(2.0 * cv_pct, iqr_pct, 2.0)
    naive_wins = sum(1 for i in range(1, len(runs) - 1) if runs[i] > runs[i - 1] and runs[i] > runs[i + 1])
    return {
        "n_runs": len(runs) if n_runs is None else n_runs,
        "tok_per_s_runs": list(runs),
        "median_tok_per_s": med,
        "mean_tok_per_s": mean,
        "stdev_tok_per_s": stdev,
        "cv_pct": cv_pct,
        "iqr_pct": iqr_pct,
        "max_rel_spread_pct": spread_pct,
        "noise_floor_pct": floor_pct,
        "aa_sliding_windows": (len(runs) if n_runs is None else n_runs) - 2,
        "aa_naive_middle_beats_both": naive_wins,
        "smaps_anon_hugepages_kb_all_zero": True,
    }


def _make_consistent_interleaved(
    triples_spec: list[tuple[float, float, float]],
    *,
    n_triples: int | None = None,
    ctl_hp: int = 0,
    cand_hp: int = 0,
) -> dict[str, Any]:
    ctl_vals: list[float] = []
    cand_vals: list[float] = []
    paired: list[float] = []
    beats_both = 0
    beats_mean = 0
    triples_out: list[dict[str, Any]] = []
    for idx, (pre, cand, post) in enumerate(triples_spec):
        ctl_vals.extend([pre, post])
        cand_vals.append(cand)
        mid = 0.5 * (pre + post)
        p_d = 100.0 * (cand - mid) / mid
        paired.append(p_d)
        if cand > pre and cand > post:
            beats_both += 1
        if cand > mid:
            beats_mean += 1
        triples_out.append(
            {
                "triple_index": idx,
                "ctl_pre_tok_s": pre,
                "cand_tok_s": cand,
                "ctl_post_tok_s": post,
                "paired_delta_pct": p_d,
            }
        )
    ctl_med = statistics.median(ctl_vals)
    cand_med = statistics.median(cand_vals)
    speedup = cand_med / ctl_med
    delta_pct = 100.0 * (cand_med - ctl_med) / ctl_med
    return {
        "n_triples": len(triples_spec) if n_triples is None else n_triples,
        "ctl_median_tok_per_s": ctl_med,
        "cand_median_tok_per_s": cand_med,
        "median_speedup": speedup,
        "delta_pct": delta_pct,
        "median_paired_delta_pct": statistics.median(paired),
        "min_paired_delta_pct": min(paired),
        "max_paired_delta_pct": max(paired),
        "triples_cand_beats_both_ctl": beats_both,
        "triples_cand_beats_mean_ctl": beats_mean,
        "ctl_smaps_anon_hugepages_kb_min": ctl_hp,
        "ctl_smaps_anon_hugepages_kb_max": ctl_hp,
        "cand_smaps_anon_hugepages_kb_min": cand_hp,
        "cand_smaps_anon_hugepages_kb_max": cand_hp,
        "triples": triples_out,
    }


def test_live_repo_and_strict_json() -> dict[str, Any]:
    summary = mod.verify_repo(REPO_ROOT)
    expect_true(summary["status"] == "PASS", "Live repo verification must PASS")
    expect_true(summary["total_live_bytes"] == 9447424, "Unexpected total_live_bytes")
    expect_true(summary["avx512_1t_speedup"] >= 1.5, "1T speedup below gate")
    expect_true(summary["avx512_8t_speedup"] >= 1.5, "8T speedup below gate")

    with tempfile.TemporaryDirectory(prefix="s125_sk_r5_") as tmp_str:
        tmp = Path(tmp_str)
        expect_s125_error(lambda: mod.verify_repo(tmp), "missing receipt file")

        dup_path = tmp / "dup.json"
        dup_path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
        expect_s125_error(lambda: mod.load_json_object_strict(dup_path), "duplicate JSON key")

        nan_path = tmp / "nan.json"
        nan_path.write_text('{"a": NaN}\n', encoding="utf-8")
        expect_s125_error(lambda: mod.load_json_object_strict(nan_path), "NaN JSON constant")

        arr_path = tmp / "arr.json"
        arr_path.write_text("[1, 2, 3]\n", encoding="utf-8")
        expect_s125_error(lambda: mod.load_json_object_strict(arr_path), "non-object top-level JSON")

        # Tampered C source file whose SHA matches a forged receipt must STILL fail pinned SHA guard
        fake_c = tmp / "holosphere_v3_avx512.c"
        fake_c.write_text("/* drifted source */\n", encoding="utf-8")
        forged = mod.load_json_object_strict(RECEIPT_PATH)
        forged["c_source_sha256"] = mod.sha256_file(fake_c)
        expect_s125_error(lambda: mod.verify_receipt_dict(forged, fake_c), "pinned C source SHA drift")

    return mod.load_json_object_strict(RECEIPT_PATH)


def test_algebraic_and_boundary_controls() -> None:
    expect_s125_error(lambda: mod.check_arena_budget(mod.LIMIT_LIVE_BYTES), "arena budget >= 12 MiB")
    expect_s125_error(lambda: mod.check_arena_budget(True), "arena budget bool")
    expect_s125_error(lambda: mod.check_secded_probe(1, 16, 120), "secded syn != 0")
    expect_s125_error(lambda: mod.check_secded_probe(0, 15, 120), "secded sec != 16")
    expect_s125_error(lambda: mod.check_secded_probe(0, 16, 119), "secded ded != 120")
    expect_s125_error(lambda: mod.check_secded_sec_step(1, 2, 1, 0), "sec step healed mismatch")
    expect_s125_error(lambda: mod.check_secded_sec_step(2, 2, 0, 0), "sec step sec != 1")
    expect_s125_error(lambda: mod.check_secded_sec_step(2, 2, 1, 1), "sec step ded != 0")
    expect_s125_error(lambda: mod.check_secded_ded_step(1, 2, 0, 1), "ded step healed mismatch")
    expect_s125_error(lambda: mod.check_secded_ded_step(2, 2, 1, 1), "ded step sec != 0")
    expect_s125_error(lambda: mod.check_secded_ded_step(2, 2, 0, 0), "ded step ded != 1")
    expect_s125_error(lambda: mod.check_vpdpbusd_bias([1] * 32, [1] * 32, 31), "vpdpbusd wrong dot")
    expect_s125_error(lambda: mod.check_cayley_unit((1.0, 2.0 ** -22, 0.0, 0.0)), "cayley exact tol boundary")
    expect_s125_error(
        lambda: mod.check_fwht_orthonormal([[1.0, 0.0], [2.0 ** -36, 1.0]]),
        "fwht exact tol boundary",
    )

    # Exact huge-page verdict 3-branch + every strict boundary (F2 coverage)
    expect_true(mod.expected_hugepage_verdict(5.1, 5.1, 5, 5.0) == "CONFIRMED", "expected CONFIRMED at >=5")
    expect_true(mod.expected_hugepage_verdict(5.0, 5.1, 5, 5.0) == "NEUTRAL / L3-ABSORBED", "CONFIRMED delta strict >")
    expect_true(mod.expected_hugepage_verdict(5.1, 5.0, 5, 5.0) == "NEUTRAL / L3-ABSORBED", "CONFIRMED paired strict >")
    expect_true(mod.expected_hugepage_verdict(5.1, 5.1, 4, 5.0) == "NEUTRAL / L3-ABSORBED", "CONFIRMED beats >= 5")
    expect_true(mod.expected_hugepage_verdict(-5.1, -5.1, 1, 5.0) == "REGRESSION", "expected REGRESSION at <=1")
    expect_true(mod.expected_hugepage_verdict(-5.0, -5.1, 1, 5.0) == "NEUTRAL / L3-ABSORBED", "REGRESSION delta strict <")
    expect_true(mod.expected_hugepage_verdict(-5.1, -5.0, 1, 5.0) == "NEUTRAL / L3-ABSORBED", "REGRESSION paired strict <")
    expect_true(mod.expected_hugepage_verdict(-5.1, -5.1, 2, 5.0) == "NEUTRAL / L3-ABSORBED", "REGRESSION beats <= 1")

    # Exact t_cand == mid and t_cand == t_pre tie-boundary checks (F2 coverage)
    blk_ties = _make_consistent_interleaved(
        [(90.0, 100.0, 110.0), (100.0, 100.0, 90.0), (90.0, 100.0, 100.0), (100.0, 120.0, 100.0), (100.0, 120.0, 100.0)]
    )
    expect_true(blk_ties["triples_cand_beats_both_ctl"] == 2, "tie beats_both expected 2")
    expect_true(blk_ties["triples_cand_beats_mean_ctl"] == 4, "tie beats_mean expected 4")
    res_ties = mod._verify_interleaved_block(
        blk_ties, "tie_bounds", expect_ctl_huge=False, expect_cand_huge=False, min_speedup=None
    )
    expect_true(res_ties["beats_both"] == 2, "tie_bounds PASS")

    # Exercise CLI main() in both --json and text modes (kills return 0 / --json mutants)
    buf_json = io.StringIO()
    with contextlib.redirect_stdout(buf_json):
        rc_j = mod.main(["--check", "--json", "--repo-root", str(REPO_ROOT)])
    expect_true(rc_j == 0 and '"status": "PASS"' in buf_json.getvalue(), "main --json must return 0")
    buf_txt = io.StringIO()
    with contextlib.redirect_stdout(buf_txt):
        rc_t = mod.main(["--check", "--repo-root", str(REPO_ROOT)])
    expect_true(rc_t == 0 and "PASS S125 EXP-Q30b" in buf_txt.getvalue(), "main text must return 0")

    # Decoupled exact-boundary AA tests
    aa_9 = _make_consistent_aa([400.0 + i for i in range(9)])
    expect_s125_error(lambda: mod._verify_aa_block(aa_9, "aa_9"), "aa n_runs=9 < 10")
    aa_len = _make_consistent_aa([400.0 + i for i in range(12)], n_runs=11)
    expect_s125_error(lambda: mod._verify_aa_block(aa_len, "aa_len"), "aa len != n_runs")
    aa_neg = _make_consistent_aa([400.0] * 9 + [-1.0])
    expect_s125_error(lambda: mod._verify_aa_block(aa_neg, "aa_neg"), "aa negative run")

    # Decoupled exact-boundary interleaved tests
    blk_4 = _make_consistent_interleaved([(100.0, 200.0, 100.0)] * 4)
    expect_s125_error(
        lambda: mod._verify_interleaved_block(blk_4, "blk_4", expect_ctl_huge=False, expect_cand_huge=False, min_speedup=1.5),
        "n_triples=4 < 5",
    )
    blk_len = _make_consistent_interleaved([(100.0, 200.0, 100.0)] * 6, n_triples=7)
    expect_s125_error(
        lambda: mod._verify_interleaved_block(blk_len, "blk_len", expect_ctl_huge=False, expect_cand_huge=False, min_speedup=1.5),
        "triples len != n_triples",
    )
    blk_neg = _make_consistent_interleaved([(100.0, -1.0, 100.0)] + [(100.0, 200.0, 100.0)] * 4)
    expect_s125_error(
        lambda: mod._verify_interleaved_block(blk_neg, "blk_neg", expect_ctl_huge=False, expect_cand_huge=False, min_speedup=None),
        "triple cand <= 0",
    )

    # Exact positive boundary tests: ctl_min == 2048, cand_min == 2048, and speedup == 1.5 must PASS
    blk_bounds = _make_consistent_interleaved([(100.0, 150.0, 100.0)] * 5, ctl_hp=2048, cand_hp=2048)
    res_b = mod._verify_interleaved_block(
        blk_bounds, "exact_bounds", expect_ctl_huge=True, expect_cand_huge=True, min_speedup=1.5
    )
    expect_true(res_b["speedup"] == 1.5, "Exact 1.5x boundary must PASS")


def test_receipt_tamper_controls(base: dict[str, Any]) -> None:
    def check_tamper(mutator: Callable[[dict[str, Any]], None], label: str) -> None:
        r = copy.deepcopy(base)
        mutator(r)
        expect_s125_error(lambda: mod.verify_receipt_dict(r, C_SRC), label)

    expect_s125_error(lambda: mod.verify_receipt_dict([], C_SRC), "receipt not dict")  # type: ignore[arg-type]
    expect_s125_error(lambda: mod.verify_receipt_dict(base, VERIFY_DIR / "missing.c"), "missing c_source_path")
    check_tamper(lambda r: r.__setitem__("schema_version", "bad"), "schema_version")
    check_tamper(lambda r: r.__setitem__("track", "S124"), "track")
    check_tamper(lambda r: r.__setitem__("experiment_id", "EXP-Q30a"), "experiment_id")
    check_tamper(lambda r: r.__setitem__("prereg_w14_sha256", "0" * 64), "prereg_w14_sha256")
    check_tamper(lambda r: r.__setitem__("cpu_only_enforced", False), "cpu_only_enforced")
    check_tamper(lambda r: r.__setitem__("c_source_file", "other.c"), "c_source_file")
    check_tamper(lambda r: r.__setitem__("c_source_sha256", "a" * 64), "c_source_sha256")

    check_tamper(lambda r: r.__setitem__("build_and_isa", []), "build_and_isa type")
    check_tamper(lambda r: r["build_and_isa"].__setitem__("gcc_o3_warnings", 1), "gcc_o3_warnings")
    check_tamper(lambda r: r["build_and_isa"].__setitem__("ubsan_asan_pass", False), "ubsan_asan_pass")
    check_tamper(lambda r: r["build_and_isa"].__setitem__("objdump_vpdpbusd_count", 0), "objdump_vpdpbusd_count")
    check_tamper(lambda r: r["build_and_isa"].__setitem__("objdump_vpopcntq_count", 0), "objdump_vpopcntq_count")
    check_tamper(lambda r: r["build_and_isa"].__setitem__("objdump_vpermt2d_count", 0), "objdump_vpermt2d_count")

    check_tamper(lambda r: r.__setitem__("footprint", []), "footprint type")
    check_tamper(lambda r: r["footprint"].__setitem__("weight_bytes", 1), "weight_bytes")
    check_tamper(lambda r: r["footprint"].__setitem__("state_bytes", 1), "state_bytes")
    check_tamper(lambda r: r["footprint"].__setitem__("total_live_bytes", 1), "total_live_bytes")
    check_tamper(lambda r: r["footprint"].__setitem__("under_12_mib_gate", False), "under_12_mib_gate")
    check_tamper(lambda r: r["footprint"].__setitem__("smaps_4k_anon_hugepages_kb", 2048), "smaps_4k")
    check_tamper(lambda r: r["footprint"].__setitem__("smaps_2m_anon_hugepages_kb", 0), "smaps_2m")

    check_tamper(lambda r: r.__setitem__("parity_and_ecc", []), "parity_and_ecc type")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("prompt_bytes", [0] * 8), "prompt_bytes")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("logits_fnv1a64", "0000000000000000"), "pinned logits_fnv1a64")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("hidden_fnv1a64", "0000000000000000"), "pinned hidden_fnv1a64")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("greedy_tokens", [0] * 8), "pinned greedy_tokens")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("bit_exact_scalar_avx512_1t_4t_8t_2m_ubsan", False), "bit_exact")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("pack_syndrome_errors", 1), "pack_syndrome_errors")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("runtime_syndrome_faults", 1), "runtime_syndrome_faults")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("secded_16sec_120ded_pass", False), "secded_16sec_120ded_pass")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("fp64_oracle_max_abs_logit_diff", 2e-5), "fp64_logit_diff")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("fp64_oracle_max_abs_hidden_diff", 2e-5), "fp64_hidden_diff")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("fp64_greedy_match_8_of_8", False), "fp64_greedy_match")
    check_tamper(lambda r: r["parity_and_ecc"].__setitem__("fp64_gate_le_1e5_pass", False), "fp64_gate")

    check_tamper(lambda r: r.__setitem__("aa_noise_floor", []), "aa_noise_floor type")
    check_tamper(lambda r: r["aa_noise_floor"].__setitem__("avx512_1t_4k", []), "aa_1t type")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("smaps_anon_hugepages_kb_all_zero", False), "aa smaps")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("median_tok_per_s", 1.0), "aa median")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("mean_tok_per_s", 1.0), "aa mean")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("stdev_tok_per_s", 99.0), "aa stdev")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("cv_pct", 99.0), "aa cv_pct")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("iqr_pct", 99.0), "aa iqr_pct")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("max_rel_spread_pct", 99.0), "aa spread")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("noise_floor_pct", 99.0), "aa noise_floor")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("aa_sliding_windows", 0), "aa sliding")
    check_tamper(lambda r: r["aa_noise_floor"]["avx512_1t_4k"].__setitem__("aa_naive_middle_beats_both", 9), "aa naive wins")

    check_tamper(lambda r: r.__setitem__("interleaved_benchmarks", []), "interleaved type")
    check_tamper(lambda r: r["interleaved_benchmarks"].__setitem__("scalar_vs_avx512_1t_4k", []), "blk type")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["triples"].__setitem__(0, []), "triple not dict")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("ctl_median_tok_per_s", 1.0), "ctl_median")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("cand_median_tok_per_s", 1.0), "cand_median")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("median_speedup", 1.0), "median_speedup")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("delta_pct", 0.0), "delta_pct")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("median_paired_delta_pct", 0.0), "median_paired_delta_pct")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("min_paired_delta_pct", 0.0), "min_paired_delta_pct")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("max_paired_delta_pct", 0.0), "max_paired_delta_pct")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("triples_cand_beats_both_ctl", 0), "beats_both")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("triples_cand_beats_mean_ctl", 0), "beats_mean")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("ctl_smaps_anon_hugepages_kb_min", -1), "smaps negative")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("ctl_smaps_anon_hugepages_kb_max", 2048), "ctl huge unexpected")
    check_tamper(lambda r: r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"].__setitem__("cand_smaps_anon_hugepages_kb_max", 2048), "cand huge unexpected")
    check_tamper(lambda r: r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"].__setitem__("cand_smaps_anon_hugepages_kb_min", 0), "cand huge missing")

    blk_ctl_huge = copy.deepcopy(base["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"])
    expect_s125_error(
        lambda: mod._verify_interleaved_block(blk_ctl_huge, "direct_ctl_huge", expect_ctl_huge=True, expect_cand_huge=False, min_speedup=None),
        "expect_ctl_huge missing",
    )
    blk_slow = copy.deepcopy(base["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"])
    expect_s125_error(
        lambda: mod._verify_interleaved_block(blk_slow, "direct_min_speedup", expect_ctl_huge=False, expect_cand_huge=True, min_speedup=1.5),
        "min_speedup gate",
    )

    check_tamper(lambda r: r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"].__setitem__("verdict", "CONFIRMED"), "1T hugepage verdict tamper")
    check_tamper(lambda r: r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"].__setitem__("verdict", "CONFIRMED"), "8T hugepage verdict tamper")
    check_tamper(lambda r: r.__setitem__("gates", []), "gates type")
    check_tamper(lambda r: r["gates"].__setitem__("G1_parity_determinism_ecc", "FAILED"), "G1 gate")
    check_tamper(lambda r: r["gates"].__setitem__("G2_l3_footprint_and_avx512_speedup", "FAILED"), "G2 gate")
    check_tamper(lambda r: r["gates"].__setitem__("hugepage_2m_vs_4k_1t_verdict", "CONFIRMED"), "gates 1T verdict")
    check_tamper(lambda r: r["gates"].__setitem__("hugepage_2m_vs_4k_8t_verdict", "CONFIRMED"), "gates 8T verdict")


def main() -> int:
    base = test_live_repo_and_strict_json()
    test_algebraic_and_boundary_controls()
    test_receipt_tamper_controls(base)
    print("OK: Silikernels Track S125 R5 suite passed all live, algebraic, boundary, and tamper controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
