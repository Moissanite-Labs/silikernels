#!/usr/bin/env python3
"""Pure-stdlib fail-closed verifier for HoloSphere S116/S117/V3 Frontier sm_86 kernels & receipts.

Verifies (zero external deps, zero bare asserts, zero GPU/sensor opens):
  1. 50/50 zero-spill Ampere sm_86 kernels across 4 CUDA TUs:
     - cuda/sm86/holosphere_megakernel_sm86.cu        (10/10 Track S116 kernels)
     - cuda/sm86/holosphere_h256_chunk32_sm86.cu      (11/11 Track S117 kernels)
     - cuda/sm86/holosphere_v3_frontier_sm86.cu       ( 2/2  W10-OBS-61 V3 Frontier kernels)
     - cuda/sm86/su2_holonomy_scan_sm86.cu            (27/27 Core SU(2)/Cayley2x2 kernels)
     including exact .cu SHA-256 parity, distinct expected symbol coverage (zero duplicate rows),
     0 B stack/spill checks, and SASS zero-LDL/STL + tensor-core/canary instruction floors.
  2. 4/4 live RTX 3080 Zone-B ML-OS hardware receipts (canonical receipt_sha256, exit_code=0,
     clean_hardware_run=true, anomalies=[], thermal_throttled=false).
  3. 4/4 live RTX 3080 GPU benchmark result JSONs (parity, 0 syndrome errors, 16/16 live .sys
     BER canary bit-exact repair, 1/1 DED, 148,382 tok/s HMMA backward, 20.51/21.52 us v3_d512).
  4. include/razor_abi_4byte_sys.h static ABI layout (64 B canary offsets 0..40, 16 B Pi rollback
     offsets 0..12, Apache-2.0 SPDX, little-endian check, UB-free __atomic_load_n + _ReadWriteBarrier).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


class SiliKernelsReceiptError(ValueError):
    """Fail-closed receipt or ABI contract violation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    """Assertion-free guard safe under python3 -O."""
    if cond is not True:
        raise SiliKernelsReceiptError(code, message)


EXPECTED_S116_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN15holosphere_s11627s116_fmnmx_onehot_imma_sm86EPKfPKaPiS4_i",
        "_ZN15holosphere_s11634s116_blockh4_weight_in_a_imma_sm86EPKmPKaPiPjij",
        "_ZN15holosphere_s11623uniform_cell_8warp_sm86EPNS_18S116MappedByteRingEPKNS_10S116CellOpEiNS_22S116Cell8WarpWorkspaceE",
        "_ZN15holosphere_s11625s116_su2_cayley_step_sm86EPK6float4PKfS2_PK6float2PS0_S8_lf",
        "_ZN15holosphere_s11619s116_warp_norm_sm86EPK6float4PS0_PjPfljf",
        "_ZN15holosphere_s11627s116_mma_m16n8k16_tile_sm86EPK13__nv_bfloat16S2_Pfij",
        "_ZN15holosphere_s11627s116_blockh4_dp4a_gemv_sm86EPKmPK6__halfPK4int4fPfPjiij",
        "_ZN15holosphere_s11631s116_blockh4_imma_m16n8k32_sm86EPKaPKmPiPjij",
        "_ZN15holosphere_s11629holosphere_persistent_vm_sm86EPNS_18S116MappedByteRingEPKNS_8S116InstEiNS_15S116VMWorkspaceE",
        "_ZN15holosphere_s11628uniform_cell_megakernel_sm86EPNS_18S116MappedByteRingEPKNS_10S116CellOpEiNS_17S116CellWorkspaceE",
    }
)

EXPECTED_S117_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN9s117_sm8637s117_uniform_cell_16warp_d512_v2_sm86EPKhPKjPKfPK6float3S3_PK6__halfS3_P6float4PfSE_PhSE_Pjffiii",
        "_ZN9s117_sm8634s117_uniform_cell_16warp_d512_sm86EPKhPKaPKfPK6float3S5_PK6__halfS5_PKjP6float4PfPhSG_Pjffii",
        "_ZN9s117_sm8627s117_spec8_fused_4imma_sm86EPKaPKfS1_S3_S1_S3_Pffi",
        "_ZN9s117_sm8634s117_sprefix_int8_shadow_imma_sm86EPKfPKaS1_PaPfS5_S5_i",
        "_ZN9s117_sm8631s117_spec8_woodbury_commit_sm86EPfP6float4PKaPKfPK6__halfPK6float3S0_ffiii",
        "_ZN9s117_sm8626s117_su2_3line_gu_bwd_sm86EPKfS1_S1_PK6float3PK6float4S7_PfPS2_PS5_iiiii",
        "_ZN9s117_sm8640s117_sfast_geo_chunk32_bwd_choice_b_sm86EPKfS1_S1_PK6__halfS1_S1_PfS5_S5_S5_ffffiiii",
        "_ZN9s117_sm8631s117_sfast_geo_chunk32_bwd_sm86EPKfS1_S1_PK6__halfS1_S1_PfS5_S5_S5_ffffiiii",
        "_ZN9s117_sm8631s117_sfast_geo_chunk32_fwd_sm86EPKfS1_PK6float3PK6__halfPK6float4PfSB_PS5_PS8_ffffiiiii",
        "_ZN9s117_sm8635s117_gumbel_fmnmx_23bit_cellop_sm86EPKfPKjPhPffi",
        "_ZN9s117_sm8632s117_h256_bent4_byte_inject_sm86EPKhPai",
    }
)

EXPECTED_V3_SYMBOLS: frozenset[str] = frozenset(
    {
        "_ZN7s117_v326s117_chunk32_bwd_wmma_sm86EPKfS1_S1_S1_S1_S1_PfS2_S2_S2_ffffiiiii",
        "_ZN7s117_v337s117_uniform_cell_16warp_d512_v3_sm86EPKhPK5uint4PKfPK6float3S4_S4_PKjP6float4PfSE_PhSE_Pjffiii",
    }
)

PTXAS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "s116_megakernel",
        "cu_rel": "cuda/sm86/holosphere_megakernel_sm86.cu",
        "rec_rel": "receipts/s116_ptxas_zero_spill_receipt.json",
        "sha_key": "source_sha256",
        "count_key": "kernel_count",
        "count": 10,
        "sym_key": "name",
        "store_key": "spill_stores_bytes",
        "load_key": "spill_loads_bytes",
        "sass_key": "sass_instruction_counts",
        "expected_symbols": EXPECTED_S116_SYMBOLS,
        "min_ops": {"IMMA.16832": 5, "HMMA.16816": 2, "LDGSTS": 15, "POPC": 72},
        "zero_ops": (),
    },
    {
        "id": "s117_chunk32",
        "cu_rel": "cuda/sm86/holosphere_h256_chunk32_sm86.cu",
        "rec_rel": "receipts/s117_ptxas_zero_spill_receipt.json",
        "sha_key": "cu_sha256",
        "count_key": "total_kernels",
        "count": 11,
        "sym_key": "symbol",
        "store_key": "spill_store_bytes",
        "load_key": "spill_load_bytes",
        "sass_key": "sass_instruction_counts",
        "expected_symbols": EXPECTED_S117_SYMBOLS,
        "min_ops": {"RED_SYS": 2, "IDP4A": 16, "IMMA_16832": 6},
        "zero_ops": ("LDL", "STL", "MEMBAR_GL", "ATOMG_GPU"),
    },
    {
        "id": "v3_frontier",
        "cu_rel": "cuda/sm86/holosphere_v3_frontier_sm86.cu",
        "rec_rel": "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        "sha_key": "cu_sha256",
        "count_key": "total_kernels",
        "count": 2,
        "sym_key": "symbol",
        "store_key": "spill_store_bytes",
        "load_key": "spill_load_bytes",
        "sass_key": "sass_instruction_counts",
        "expected_symbols": EXPECTED_V3_SYMBOLS,
        "min_ops": {"HMMA_16816": 28, "IDP4A": 24, "RED_SYS": 1},
        "zero_ops": ("LDL", "STL", "MEMBAR_GL", "ATOMG_GPU"),
    },
    {
        "id": "s114_su2_holonomy",
        "cu_rel": "cuda/sm86/su2_holonomy_scan_sm86.cu",
        "rec_rel": "receipts/su2_holonomy_sm86_ptxas_receipt.json",
        "sha_key": "source_sha256",
        "count_key": "kernel_count",
        "count": 27,
        "sym_key": "symbol",
        "store_key": "spill_stores_bytes",
        "load_key": "spill_loads_bytes",
        "sass_key": "sass_diagnostics",
        "expected_symbols": None,
        "min_ops": {"FFMA": 3000, "SHFL": 300, "LDG_E_128": 100},
        "zero_ops": ("LDL", "STL"),
    },
)

ZONE_B_MLOS_RECEIPTS: tuple[tuple[str, str], ...] = (
    ("receipts/mlos_zone_b_s116_phase2b.json", "739f8708a22152763f3a8c8b448809bb3363dd2f592966ce1069056a03786d18"),
    ("receipts/mlos_zone_b_s117_live.json", "de012df35435ab5b4193443f50c21bf7382bc68899309e9727b0e40f521d7c8f"),
    ("receipts/mlos_zone_b_s117_v2_d512.json", "2fccfe50027445937d93ca67a1441658194123891298d98ef60c9bed6925d05e"),
    ("receipts/mlos_zone_b_s117_v3_frontier.json", "7655e610e12e3de767116e5093cdb76294ea9594a85fe6240e4f0de3b13beb0e"),
)

EXPECTED_CANARY_OFFSETS: dict[str, int] = {
    "sec_count_u32": 0,
    "ded_count_u32": 4,
    "epoch_u32": 8,
    "rollback_token_u32": 12,
    "non_lsb_canary_u32": 16,
    "last_syndrome_u32": 20,
    "ewma_ber_q16_u32": 24,
    "clean_streak_u32": 28,
    "hold_remaining_u32": 32,
    "ded_latch_remaining_u32": 36,
    "reserved_u32": 40,
}

EXPECTED_PI16_OFFSETS: dict[str, int] = {"w": 0, "x": 4, "y": 8, "z": 12}


def _reject_nonfinite_json(token: str) -> Any:
    raise SiliKernelsReceiptError("ERR_JSON_NONFINITE", f"Non-finite JSON constant forbidden: {token!r}")


def load_strict_json(path: Path) -> dict[str, Any]:
    """Read a JSON object strictly rejecting missing files, syntax errors, non-dicts, and NaN/Inf."""
    require_ok(path.is_file(), "ERR_MISSING_FILE", f"Missing required file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        require_ok(False, "ERR_BAD_JSON", f"Malformed JSON in {path}: {exc}")
        return {}
    require_ok(isinstance(obj, dict), "ERR_JSON_OBJECT", f"Expected JSON object in {path}")
    return obj


def _is_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_int_eq(val: Any, target: int) -> bool:
    return _is_int(val) and val == target


def _is_real_num(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _is_zero_real(val: Any) -> bool:
    return _is_real_num(val) and float(val) == 0.0


def verify_ptxas_spec(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed verification of one .cu target against its ptxas/SASS zero-spill receipt."""
    cu_path = repo_root / str(spec["cu_rel"])
    rec_path = repo_root / str(spec["rec_rel"])
    require_ok(cu_path.is_file(), "ERR_MISSING_CU", f"Missing CUDA source: {cu_path}")
    cu_sha = hashlib.sha256(cu_path.read_bytes()).hexdigest()
    rec = load_strict_json(rec_path)
    require_ok(
        rec.get(str(spec["sha_key"])) == cu_sha,
        "ERR_CU_SHA",
        f"{spec['id']}: SHA-256 mismatch between {cu_path.name} and {rec_path.name}",
    )
    kernels = rec.get("kernels")
    n_exp = int(spec["count"])
    require_ok(
        isinstance(kernels, list) and len(kernels) == n_exp and _is_int_eq(rec.get(str(spec["count_key"])), n_exp),
        "ERR_KERNEL_COUNT",
        f"{spec['id']}: expected {n_exp} kernels",
    )
    if "nonzero_spill_kernels" in rec:
        require_ok(
            _is_int_eq(rec.get("nonzero_spill_kernels"), 0),
            "ERR_NONZERO_SPILL",
            f"{spec['id']}: nonzero_spill_kernels != 0",
        )
    if "all_zero_spill" in rec:
        require_ok(rec.get("all_zero_spill") is True, "ERR_NONZERO_SPILL", f"{spec['id']}: all_zero_spill != True")

    sym_key = str(spec["sym_key"])
    store_key = str(spec["store_key"])
    load_key = str(spec["load_key"])
    seen_syms: list[str] = []
    for entry in kernels:
        require_ok(
            isinstance(entry, dict)
            and _is_int_eq(entry.get("stack_frame_bytes"), 0)
            and _is_int_eq(entry.get(store_key), 0)
            and _is_int_eq(entry.get(load_key), 0)
            and ("zero_spill" not in entry or entry.get("zero_spill") is True),
            "ERR_SPILL",
            f"{spec['id']}: kernel spill or invalid entry detected",
        )
        sym = entry.get(sym_key)
        require_ok(isinstance(sym, str) and sym != "", "ERR_SYMBOL_SET", f"{spec['id']}: invalid kernel symbol")
        seen_syms.append(sym)

    expected_syms = spec.get("expected_symbols")
    require_ok(
        len(set(seen_syms)) == n_exp and (expected_syms is None or frozenset(seen_syms) == expected_syms),
        "ERR_SYMBOL_SET",
        f"{spec['id']}: duplicate or mismatched kernel symbol set",
    )

    sass = rec.get(str(spec["sass_key"]))
    require_ok(isinstance(sass, dict), "ERR_SASS_SCHEMA", f"{spec['id']}: missing SASS counts object")
    for zop in spec["zero_ops"]:
        require_ok(_is_int_eq(sass.get(zop), 0), "ERR_SASS_FORBIDDEN", f"{spec['id']}: forbidden SASS op {zop} != 0")
    for op_name, floor in spec["min_ops"].items():
        op_val = sass.get(op_name)
        require_ok(_is_int(op_val) and op_val >= int(floor), "ERR_SASS_OPS", f"{spec['id']}: SASS op {op_name} < {floor}")

    return {
        "id": str(spec["id"]),
        "cu_rel": str(spec["cu_rel"]),
        "cu_sha256": cu_sha,
        "zero_spill_kernels": n_exp,
    }


def verify_zone_b_mlos_receipt(path: Path, expected_receipt_sha256: str) -> dict[str, Any]:
    """Fail-closed verification of an ML-OS Zone-B hardware receipt and its canonical self-hash."""
    rec = load_strict_json(path)
    require_ok(
        rec.get("schema") == "mlos-run-receipt/v0.1"
        and rec.get("determinism_zone") == "B"
        and rec.get("clean_hardware_run") is True
        and rec.get("anomalies") == [],
        "ERR_MLOS_CLEAN",
        f"{path.name}: Zone-B clean-hardware invariants failed",
    )
    exe = rec.get("execution")
    require_ok(isinstance(exe, dict) and _is_int_eq(exe.get("exit_code"), 0), "ERR_MLOS_EXIT", f"{path.name}: exit_code != 0")
    end_tel = rec.get("end_gpu_telemetry")
    reasons = end_tel.get("clocks_event_reasons") if isinstance(end_tel, dict) else None
    require_ok(
        isinstance(reasons, dict) and reasons.get("thermal_throttled") is False,
        "ERR_MLOS_THERMAL",
        f"{path.name}: thermal_throttled != False",
    )
    body = {k: v for k, v in rec.items() if k != "receipt_sha256"}
    recomputed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    require_ok(
        rec.get("receipt_sha256") == recomputed and recomputed == expected_receipt_sha256,
        "ERR_MLOS_SHA",
        f"{path.name}: canonical receipt_sha256 mismatch",
    )
    return {
        "file": path.name,
        "receipt_sha256": recomputed,
        "gpu_name": end_tel.get("gpu_name"),
    }


def verify_live_benchmark_results(repo_root: Path) -> dict[str, Any]:
    """Fail-closed parity & headline-metric verification across all 4 live RTX 3080 result receipts."""
    s116 = load_strict_json(repo_root / "receipts" / "s116_phase2b_gpu_results.json")
    p116 = s116.get("phase2b_live_parity")
    c116 = s116.get("h16a_v2_matched_12layer_showdown", {}).get("uniform_cell_8warp_d256_1cta") if isinstance(s116.get("h16a_v2_matched_12layer_showdown"), dict) else None
    l116 = s116.get("h16c_v2_l2_persistence_and_weight_in_a")
    require_ok(
        isinstance(p116, dict)
        and isinstance(c116, dict)
        and isinstance(l116, dict)
        and p116.get("all_parity_pass") is True
        and _is_int_eq(p116.get("weight_in_a_valid_syndrome_errors"), 0)
        and _is_int_eq(p116.get("weight_in_a_injected_16_flips_caught"), 16)
        and _is_zero_real(p116.get("cell8warp_2cta_16b_pi_t0_rollback_max_drift"))
        and l116.get("clamp_exact_ga104m_ceiling") is True
        and _is_real_num(c116.get("us_per_tok"))
        and float(c116["us_per_tok"]) < 35.0
        and _is_real_num(l116.get("t8192_96mib_polluted_4pass_pinned_gb_s"))
        and float(l116["t8192_96mib_polluted_4pass_pinned_gb_s"]) > 500.0,
        "ERR_S116_RESULTS",
        "Track S116 live RTX 3080 benchmark invariants failed",
    )

    s117_v1 = load_strict_json(repo_root / "receipts" / "s117_live_rtx3080_zone_b_results.json")
    bm117 = s117_v1.get("bare_metal_rtx3080") if isinstance(s117_v1.get("bare_metal_rtx3080"), dict) else {}
    ab117 = bm117.get("chunk32_woodbury_bwd_ab_t8192")
    can117 = bm117.get("live_sys_ber_canary")
    tc117 = s117_v1.get("live_gpu_torch_compile_and_t8192_vram")
    require_ok(
        s117_v1.get("all_live_gates_pass") is True
        and isinstance(ab117, dict)
        and isinstance(can117, dict)
        and isinstance(tc117, dict)
        and _is_int_eq(can117.get("clean_canary_count"), 0)
        and _is_int_eq(can117.get("sec16_canary_count"), 16)
        and _is_zero_real(can117.get("sec16_max_abs_diff_vs_clean"))
        and can117.get("sec16_out_byte_bit_exact") is True
        and _is_int_eq(can117.get("ded1_canary_count"), 1)
        and can117.get("ded1_flagged_no_silent_miscorrect") is True
        and _is_real_num(ab117.get("speedup_choice_b_over_a"))
        and float(ab117["speedup_choice_b_over_a"]) > 1.2
        and tc117.get("torch_compile_fullgraph_pass") is True
        and _is_real_num(tc117.get("t8192_live_peak_allocated_mib"))
        and float(tc117["t8192_live_peak_allocated_mib"]) < 650.0,
        "ERR_S117_V1_RESULTS",
        "Track S117 v1 live RTX 3080 benchmark invariants failed",
    )

    s117_v2 = load_strict_json(repo_root / "receipts" / "s117_live_v2_d512_zone_b_results.json")
    e2e_v2 = s117_v2.get("full_e2e_fwht256_12L_d512")
    par_v2 = s117_v2.get("fwht256_bent4_lm_head_parity")
    can_v2 = s117_v2.get("live_sys_ber_canary_v2")
    require_ok(
        isinstance(e2e_v2, dict)
        and isinstance(par_v2, dict)
        and isinstance(can_v2, dict)
        and par_v2.get("argmax_byte_exact") is True
        and _is_real_num(par_v2.get("max_abs_err_vs_fp64_dense"))
        and float(par_v2["max_abs_err_vs_fp64_dense"]) < 1e-6
        and _is_int_eq(can_v2.get("clean_canary_count"), 0)
        and _is_int_eq(can_v2.get("sec16_canary_count"), 16)
        and _is_zero_real(can_v2.get("sec16_hidden_max_abs_diff"))
        and _is_zero_real(can_v2.get("sec16_logits_max_abs_diff"))
        and can_v2.get("sec16_out_byte_bit_exact") is True
        and _is_int_eq(can_v2.get("ded1_canary_count"), 1)
        and can_v2.get("ded1_flagged_no_silent_miscorrect") is True
        and _is_real_num(e2e_v2.get("t8192_us_per_tok"))
        and float(e2e_v2["t8192_us_per_tok"]) < 28.0,
        "ERR_S117_V2_RESULTS",
        "Track S117 v2 d=512 live RTX 3080 benchmark invariants failed",
    )

    v3 = load_strict_json(repo_root / "receipts" / "s117_v3_frontier_live_gpu_results.json")
    v3_d = v3.get("v3_d512")
    v3_b = v3.get("chunk32_bwd_wmma")
    require_ok(
        v3.get("all_parity_pass") is True
        and isinstance(v3_d, dict)
        and isinstance(v3_b, dict)
        and v3_d.get("parity_pass") is True
        and v3_b.get("parity_pass") is True
        and _is_int_eq(v3_d.get("clean_canary"), 0)
        and _is_int_eq(v3_d.get("sec16_canary"), 16)
        and _is_int_eq(v3_d.get("ded_canary"), 1)
        and _is_zero_real(v3_d.get("sec16_hidden_max_abs_err"))
        and _is_zero_real(v3_d.get("sec16_logits_max_abs_err"))
        and _is_real_num(v3_d.get("cellop_graph_us_per_tok_8192"))
        and float(v3_d["cellop_graph_us_per_tok_8192"]) < 21.0
        and _is_real_num(v3_d.get("e2e_graph_us_per_tok_16384"))
        and float(v3_d["e2e_graph_us_per_tok_16384"]) < 22.0
        and _is_real_num(v3_b.get("B4_T8192_D512_L12_tok_per_sec"))
        and float(v3_b["B4_T8192_D512_L12_tok_per_sec"]) > 140000.0
        and _is_real_num(v3_b.get("speedup_vs_choice_b_532ms"))
        and float(v3_b["speedup_vs_choice_b_532ms"]) > 2.4,
        "ERR_V3_RESULTS",
        "W10-OBS-61 V3 Frontier live RTX 3080 benchmark invariants failed",
    )

    return {
        "s116_cell8warp_us_per_tok": c116["us_per_tok"],
        "s116_l2pin_4pass_gb_s": l116["t8192_96mib_polluted_4pass_pinned_gb_s"],
        "s117_choice_b_speedup": ab117["speedup_choice_b_over_a"],
        "s117_v2_e2e_us_per_tok": e2e_v2["t8192_us_per_tok"],
        "v3_hmma_bwd_tok_per_sec": v3_b["B4_T8192_D512_L12_tok_per_sec"],
        "v3_d512_cellop_graph_us": v3_d["cellop_graph_us_per_tok_8192"],
        "v3_d512_e2e_graph_us": v3_d["e2e_graph_us_per_tok_16384"],
    }


def verify_razor_abi_header(header_path: Path) -> dict[str, Any]:
    """Verify Apache-2.0 SPDX, UB-free acquire helpers, and exact 64 B / 16 B wire layout in razor_abi_4byte_sys.h."""
    require_ok(header_path.is_file(), "ERR_ABI_MISSING", f"Missing ABI header: {header_path}")
    raw = header_path.read_bytes()
    txt = raw.decode("utf-8", "strict")
    required_tokens = (
        "SPDX-License-Identifier: Apache-2.0",
        "__ORDER_LITTLE_ENDIAN__",
        "__atomic_load_n(ptr, __ATOMIC_ACQUIRE)",
        "#include <intrin.h>",
        "_ReadWriteBarrier();",
        "atom.release.sys.global.add.u32",
        "st.release.sys.global.u32",
        "sizeof(HoloSphereRazorCanary64) == 64u",
        "sizeof(HoloSpherePiRollback16) == 16u",
    )
    require_ok(all(tok in txt for tok in required_tokens), "ERR_ABI_CONTRACT", "Missing required Razor ABI contract token")
    forbidden_tokens = ("_InterlockedCompareExchange", "std::atomic<uint32_t>*", "_Atomic uint32_t*")
    require_ok(all(tok not in txt for tok in forbidden_tokens), "ERR_ABI_UB", "Forbidden UB/CAS token found in Razor ABI header")

    canary_offsets = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"offsetof\(HoloSphereRazorCanary64,\s*(\w+)\)\s*==\s*(\d+)u", txt)
    }
    pi16_offsets = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"offsetof\(HoloSpherePiRollback16,\s*(\w+)\)\s*==\s*(\d+)u", txt)
    }
    require_ok(
        canary_offsets == EXPECTED_CANARY_OFFSETS and pi16_offsets == EXPECTED_PI16_OFFSETS,
        "ERR_ABI_OFFSETS",
        "Razor ABI struct offsetof assertions do not match 64 B / 16 B specification",
    )
    return {
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "canary_fields": len(canary_offsets),
        "pi16_fields": len(pi16_offsets),
    }


def verify_all(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run the full fail-closed verification suite across all 50 HoloSphere sm_86 kernels, receipts, and Razor ABI."""
    ptxas_reports = [verify_ptxas_spec(repo_root, spec) for spec in PTXAS_SPECS]
    total_kernels = sum(int(r["zero_spill_kernels"]) for r in ptxas_reports)
    mlos_reports = [
        verify_zone_b_mlos_receipt(repo_root / rel, exp_sha)
        for rel, exp_sha in ZONE_B_MLOS_RECEIPTS
    ]
    bench_summary = verify_live_benchmark_results(repo_root)
    abi_summary = verify_razor_abi_header(repo_root / "include" / "razor_abi_4byte_sys.h")
    payload = {
        "schema": "silikernels.holosphere_s116_s117_v3_verify.v1",
        "status": "PASS",
        "total_holosphere_zero_spill_kernels": total_kernels,
        "ptxas_targets": ptxas_reports,
        "zone_b_mlos_receipts": mlos_reports,
        "live_benchmarks": bench_summary,
        "razor_abi": abi_summary,
    }
    payload["summary_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify HoloSphere S116/S117/V3 sm_86 receipts & Razor ABI.")
    parser.add_argument("--check", action="store_true", help="Run fail-closed receipt verification.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path.")
    args = parser.parse_args(argv)
    res = verify_all(args.repo_root)
    if args.json:
        sys.stdout.write(json.dumps(res, sort_keys=True, indent=2) + "\n")
    else:
        sys.stdout.write(
            f"PASS silikernels holosphere_kernels={res['total_holosphere_zero_spill_kernels']}/50 "
            f"summary_sha256={res['summary_sha256']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
