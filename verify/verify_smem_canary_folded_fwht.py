#!/usr/bin/env python3
"""Pure-Q / Pure-Z Fail-Closed Verifier for Ampere sm_86 CTA-SMEM Canary & Folded FWHT Head.

Zero-float, zero-ast.Div, pure-stdlib, self-bound on-disk verifier for:
  1. Strict canonical-JSON & duplicate-key-free loading with exact key allowlists and
     on-disk SHA-256 bindings across both CUDA sources and both JSON receipts (zero
     unbound external hashes; fails closed on any 1-byte tamper, duplicate key, extra
     key, whitespace/key-order edit, or trailing byte).
  2. Built-in pure-Q AST self-audit banning ast.Div, ast.Pow, ast.Assert, float/complex
     constants, float-producing imports/relative imports/wildcard imports, dunder float
     escapes (__truediv__, sys.modules, getattr, __builtins__), and aliased/str Fraction.
  3. sm_86 ptxas & cuobjdump SASS zero-spill / exact register, barrier, SMEM,
     ATOMG.E.CAS.STRONG.SYS, RED.E.OR.STRONG.SYS, and ATOMS_SMEM receipt verification
     (5/5 kernels, 0 B spills, 0 LDL/STL, --Werror all-warnings).
  4. Exact pure-Z [16,11,4] SECDED (16/16 SEC + 120/120 DED), single-CTA and multi-CTA
     per-lane saturating CAS accumulator oracle (zero SEC->DED carry, zero DED bit-31
     wrap-to-zero), and FWHT_256 x bent4 Parseval/gradient-routing oracles.
  5. Exact pure-Q recomputation of head overhead, exact storm speedup (warp_us / smem_us),
     exact FP16 quantization delta (abs(fp16 - bf16)), 5-arm x 3-seed UTF-8 BPB means/deltas
     with strict seed alignment, gates G1/G2/G3, and overall verdict (--require-verdict).
  6. Fail-closed uncontrolled-laptop energy/power telemetry guard (rejecting W/J/Wh/TDP,
     RAPL, NVML, nvidia-smi, j/tok, tok/J, per watt, energy per token, stock power,
     power-savings, mAh, homoglyphs, NUL/control chars, and invisible format marks).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence as AbcSequence, Set as AbcSet
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

SCHEMA_PTXAS = "silikernels.sm86.smem_canary_folded_fwht_ptxas_receipt.v1"
SCHEMA_RESULTS = "silikernels.sm86.smem_canary_folded_fwht_results.v1"
SCHEMA_VERIFY = "silikernels.sm86.smem_canary_folded_fwht_verification.v1"
ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
EXPECTED_COMPILE_FLAGS = (
    "-O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings "
    "--shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror"
)

KERNEL_CU_REL = "cuda/sm86/holosphere_smem_canary_folded_fwht_sm86.cu"
BENCH_CU_REL = "cuda/sm86/holosphere_smem_canary_folded_fwht_bench.cu"
PTXAS_RECEIPT_REL = "receipts/smem_canary_folded_fwht_ptxas_receipt.json"
RESULTS_RECEIPT_REL = "receipts/smem_canary_folded_fwht_results.json"

FOLDED_CELL_SYMBOL = (
    "_ZN22holosphere_smem_canary41holosphere_cell_folded1k_smem_canary_sm86E"
    "PKhPK5uint4PKfPK6float3S4_S4_PK7__half2PKjP6float4PfSH_PhSH_PjSJ_SJ_ffiii"
)
UNFOLDED_CELL_SYMBOL = (
    "_ZN22holosphere_smem_canary41holosphere_cell_unfolded_warp_canary_sm86E"
    "PKhPK5uint4PKfPK6float3S4_S4_PKjP6float4PfSE_PhSE_PjSG_SG_ffiii"
)
FOLDED_HEAD_SYMBOL = "_ZN22holosphere_smem_canary32holosphere_lm_head_folded1k_sm86EPKfPK7__half2Pff"
UNFOLDED_HEAD_SYMBOL = "_ZN22holosphere_smem_canary32holosphere_lm_head_unfolded_sm86EPKfPff"
OVERFLOW_PROBE_SYMBOL = "_ZN22holosphere_smem_canary42holosphere_smem_canary_overflow_probe_sm86EjjjPjS0_"

# Exact expected (registers, barriers, smem_bytes, atom_cas_sys_insns, red_or_sys_insns, atoms_smem_insns)
_EXPECTED_KERNEL_SPECS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    FOLDED_CELL_SYMBOL: (55, 1, 14004, 1, 1, 3),
    UNFOLDED_CELL_SYMBOL: (56, 1, 13992, 1, 1, 0),
    FOLDED_HEAD_SYMBOL: (19, 1, 4096, 0, 0, 0),
    UNFOLDED_HEAD_SYMBOL: (18, 1, 4096, 0, 0, 0),
    OVERFLOW_PROBE_SYMBOL: (8, 0, 0, 1, 1, 0),
}
_EXPECTED_KERNEL_SYMBOLS: frozenset[str] = frozenset(_EXPECTED_KERNEL_SPECS.keys())

# Strict closed key allowlists (zero extra/unbound keys or digests allowed)
_PTXAS_KEYS: frozenset[str] = frozenset(
    {
        "schema", "energy_power_status", "arch", "nvcc_version", "compile_flags",
        "cu_path", "cu_sha256", "total_kernels", "zero_spill_kernels",
        "nonzero_spill_kernels", "kernels", "receipt_sha256",
    }
)
_KERNEL_STAT_KEYS: frozenset[str] = frozenset(
    {
        "stack_bytes", "spill_store_bytes", "spill_load_bytes",
        "registers", "barriers", "smem_bytes",
        "ldl_insns", "stl_insns", "atom_cas_sys_insns", "red_or_sys_insns", "atoms_smem_insns",
    }
)
_RESULTS_KEYS: frozenset[str] = frozenset(
    {
        "schema", "energy_power_status", "hardware", "source_bindings",
        "silicon_parity_and_canary", "utf8_bpb_study", "bpb_deltas",
        "gates", "verdict", "result_sha256",
    }
)
_HARDWARE_KEYS: frozenset[str] = frozenset({"device", "sm", "host_native_atomic_supported", "nvcc_version"})
_SOURCE_BINDINGS_KEYS: frozenset[str] = frozenset(
    {
        "kernel_cu_path", "kernel_cu_sha256", "bench_cu_path", "bench_cu_sha256",
        "ptxas_receipt_path", "ptxas_receipt_sha256",
    }
)
_SILICON_KEYS: frozenset[str] = frozenset(
    {
        "clean_packed_u32", "clean_sys_atomics",
        "folded1k_cell_fp64_max_err_q", "folded1k_head_fp64_max_err_q",
        "sec16_packed_u32", "sec16_sec_count", "sec16_ded_count", "sec16_sys_atomics",
        "sec16_hidden_max_abs_err_q", "sec16_logits_max_abs_err_q",
        "ded120_packed_u32", "ded120_sec_count", "ded120_ded_count", "ded120_sys_atomics",
        "overflow_sec70000_packed_u32", "overflow_sec70000_diag_u32",
        "overflow_ded70000_packed_u32", "overflow_ded70000_diag_u32",
        "multi_cta_sec_carry_packed_u32", "multi_cta_sec_carry_diag_u32",
        "multi_cta_ded_wrap_packed_u32", "multi_cta_ded_wrap_diag_u32",
        "standalone_head_unfolded_us_q", "standalone_head_folded1k_us_q",
        "standalone_head_overhead_us_q", "storms",
    }
)
_STORM_KEYS: frozenset[str] = frozenset(
    {
        "rate_pct", "expected_sec", "expected_ded",
        "warp_sec", "warp_ded", "warp_sys_atomics",
        "smem_sec", "smem_ded", "smem_sys_atomics",
        "atomic_reduction_ratio_q", "warp_us_per_tok_q", "smem_us_per_tok_q", "latency_speedup_q",
    }
)
_BPB_STUDY_KEYS: frozenset[str] = frozenset(
    {
        "scope_note", "d_model", "d_ff", "n_heads", "n_layers",
        "batch_size", "seq_len", "steps_per_seed", "tokens_per_seed",
        "eval_chunks", "eval_bytes", "seeds",
        "butterfly_vs_matrix_fp64_max_err_q",
        "arm1_upper_half_grad_norm_q", "arm3_upper_half_grad_norm_q", "arms",
    }
)
_ARM_KEYS: frozenset[str] = frozenset(
    {"head_params", "head_fp16_bytes", "hbm_vocab_table_bytes", "total_params", "mean_eval_bpb_q", "seeds"}
)
_SEED_KEYS: frozenset[str] = frozenset(
    {"seed", "eval_bpb_q", "eval_bpb_bf16_q", "fp16_quant_bpb_diff_q", "eval_tokens", "train_wall_s_q"}
)
_ARM3_SEED_KEYS: frozenset[str] = _SEED_KEYS | frozenset({"alpha_rms_q", "beta_rms_q"})
_BPB_DELTAS_KEYS: frozenset[str] = frozenset(
    {
        "arm3_minus_arm1_prefix_q", "arm3_minus_arm2_fixed_diff_q",
        "arm3_minus_arm2b_1scalar_q", "arm3_minus_arm4_dense_q",
        "arm3_beats_arm1_all_seeds", "arm3_beats_arm2_all_seeds", "arm3_beats_arm2b_all_seeds",
    }
)
_GATES_KEYS: frozenset[str] = frozenset(
    {"G1_zero_spill_sm86", "G2_fold1k_bpb_and_latency", "G3_smem_canary_atomic_and_recovery"}
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_Q_STR_RE = re.compile(r"^-?[0-9]+/[1-9][0-9]*$")
_ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_FORBIDDEN_AST_FROM_MODULES: frozenset[str] = frozenset(
    {"math", "cmath", "decimal", "statistics", "numpy", "scipy", "torch", "operator", "importlib", "builtins", "ctypes"}
)
_FORBIDDEN_AST_IMPORT_MODULES: frozenset[str] = _FORBIDDEN_AST_FROM_MODULES | frozenset({"fractions"})
_FORBIDDEN_AST_NAMES: frozenset[str] = frozenset(
    {
        "float", "complex", "pow", "__import__", "eval", "exec", "compile",
        "getattr", "setattr", "globals", "locals", "vars", "builtins", "__builtins__",
    }
)
_FORBIDDEN_AST_ATTRS: frozenset[str] = (_FORBIDDEN_AST_NAMES - frozenset({"compile"})) | frozenset(
    {
        "from_float", "from_decimal", "truediv", "itruediv",
        "__truediv__", "__rtruediv__", "__itruediv__",
        "__floordiv__", "__rfloordiv__", "__float__", "__complex__",
        "__pow__", "__rpow__", "__ipow__",
        "modules", "__dict__", "__subclasses__", "__globals__", "__loader__", "__spec__",
    }
)

_FORBIDDEN_ENERGY_KEYS: frozenset[str] = frozenset(
    {
        "w", "pw", "nw", "uw", "mw", "kw", "gw", "tw",
        "wh", "whr", "whrs", "mwh", "kwh",
        "j", "pj", "nj", "uj", "mj", "kj", "gj", "tj",
        "tdp", "pwr", "watt", "watts", "joule", "joules",
        "millijoule", "millijoules", "energy", "power", "edp",
        "rapl", "nvml", "nvidiasmi", "mah",
    }
)
_FORBIDDEN_COMPACT_KEY_RE = re.compile(
    r"(?:(?:tera|giga|mega|kilo|milli|micro|nano|pico)?(?:watt|joule)"
    r"|power|energy|mjpertok|mjtok|jpertok|ujpertok|kwh|mwh|whpertok|tdp|rapl|nvml|nvidiasmi)"
)
_ENERGY_PFX = r"(?:p|n|u|\u03bc|m|k|g|t|pico|nano|micro|milli|kilo|mega|giga|tera)"
_ENERGY_UNIT = r"(?:watt[^a-z0-9]*hours?|watthours?|watts?|joules?|whrs?|wh|w|j|tdp)"
_ENERGY_SEP = r"[^a-z0-9]*"
_ENERGY_NUM = r"\d[\d.,]*(?:e[+-]?\d+)?"
_FORBIDDEN_ENERGY_VALUE_RE = re.compile(
    "(?:"
    rf"{_ENERGY_NUM}{_ENERGY_SEP}(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}(?![a-z0-9])(?!/\s)"
    rf"|(?<![a-z0-9])(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}\s*(?:[:=]\s*|\s+){_ENERGY_NUM}(?![a-z0-9])"
    rf"|(?<![a-z0-9])tdp[^a-z0-9]*{_ENERGY_NUM}"
    rf"|^[^a-z0-9]*(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}[^a-z0-9]*$"
    r"|(?:^|[^a-z0-9])(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:joules?|watts?|j|wh)|kwh|mwh|energy|power)[\s_-]*(?:per|/)[\s_-]*tok(?:en)?s?(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])tok(?:en)?s?[\s_-]*(?:per|/)[\s_-]*(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:j|joules?|watts?)|wh|kwh|mwh)(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])per[\s_-]+(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:watts?|joules?)|w|j)(?:$|[^a-z0-9])"
    r"|(?:rapl|nvml|nvidia[\s_-]*smi|(?:^|[^a-z0-9])mah(?:$|[^a-z0-9]))"
    r"|(?:^|[^a-z0-9])(?:less|lower|reduced|fewer)[\s_-]+(?:in[\s_-]+)?(?:energy|power)(?:$|[^a-z0-9])"
    r"|(?:^|[^a-z0-9])(?:energy|power)[\s_-]+(?:use|usage|reduction|savings?|dropped|consumption)(?:$|[^a-z0-9])"
    r"|(?:%|percent)[\s_-]+(?:of[\s_-]+)?(?:(?:stock|baseline|nominal|peak|idle|full|total|gpu|cpu)[\s_-]+)?(?:energy|power)(?:$|[^a-z0-9])"
    ")"
)

_EXPECTED_ARMS: Tuple[str, ...] = (
    "arm1_prefix_fwht",
    "arm2_fixed_diff_fwht",
    "arm2b_fixed_diff_1scalar_fwht",
    "arm3_learned_1kib_fp16_fold_fwht",
    "arm4_dense_256x512",
)
_EXPECTED_HEAD_PARAMS: Dict[str, Tuple[int, int, int]] = {
    "arm1_prefix_fwht": (0, 0, 0),
    "arm2_fixed_diff_fwht": (0, 0, 0),
    "arm2b_fixed_diff_1scalar_fwht": (1, 2, 0),
    "arm3_learned_1kib_fp16_fold_fwht": (512, 1024, 0),
    "arm4_dense_256x512": (131073, 262146, 262146),
}


class SmemCanaryVerifyError(ValueError):
    """Fail-closed typed error for sm_86 CTA-SMEM canary & folded FWHT verification."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise SmemCanaryVerifyError(code, message)


def is_strict_int(val: Any) -> bool:
    return type(val) is int


def is_hex64(val: Any) -> bool:
    return type(val) is str and bool(_HEX64_RE.match(val))


def _no_dup_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "DUPLICATE_JSON_KEY", f"duplicate key: {k!r}")
        out[k] = v
    return out


def _reject_non_finite(token: str) -> Any:
    raise SmemCanaryVerifyError("NON_FINITE_JSON", f"non-finite constant: {token!r}")


def load_strict_canonical_json(path: Path) -> Any:
    require_ok(path.is_file(), "MISSING_FILE", path.name)
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("ascii")
        parsed = json.loads(text, object_pairs_hook=_no_dup_object_pairs, parse_constant=_reject_non_finite)
    except SmemCanaryVerifyError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmemCanaryVerifyError("READ_OR_PARSE_ERROR", str(exc)) from exc
    expected_raw = (json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    require_ok(raw_bytes == expected_raw, "RAW_JSON_NOT_CANONICAL", f"non-canonical bytes in {path.name}")
    return parsed


def verify_pure_q_ast(source_text: str) -> None:
    """Enforce zero-float, zero-ast.Div, zero-ast.Pow, zero-assert, and zero-escape discipline."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        raise SmemCanaryVerifyError("AST_SYNTAX_ERROR", str(exc)) from exc
    for node in ast.walk(tree):
        require_ok(not isinstance(node, ast.Div), "AST_DIV_FORBIDDEN", "ast.Div forbidden")
        require_ok(not isinstance(node, ast.Pow), "AST_POW_FORBIDDEN", "ast.Pow forbidden")
        require_ok(not isinstance(node, ast.Assert), "AST_BARE_ASSERT", "ast.Assert forbidden")
        if isinstance(node, ast.Constant):
            tname = type(node.value).__name__
            require_ok(tname not in ("float", "complex"), "AST_FLOAT_LITERAL", f"literal {tname}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".", 1)[0]
                require_ok(root_mod not in _FORBIDDEN_AST_IMPORT_MODULES, "AST_FORBIDDEN_IMPORT", alias.name)
        if isinstance(node, ast.ImportFrom):
            require_ok(node.level == 0 and node.module is not None, "AST_RELATIVE_IMPORT", "relative import forbidden")
            mod = node.module.split(".", 1)[0]
            require_ok(mod not in _FORBIDDEN_AST_FROM_MODULES, "AST_FORBIDDEN_IMPORT", node.module)
            for alias in node.names:
                require_ok(alias.name != "*", "AST_WILDCARD_IMPORT", "wildcard import forbidden")
                if node.module == "fractions":
                    require_ok(alias.name == "Fraction" and alias.asname is None, "AST_FRACTION_ALIAS", alias.name)
        if isinstance(node, ast.Name):
            require_ok(node.id not in _FORBIDDEN_AST_NAMES, "AST_FORBIDDEN_NAME", node.id)
        if isinstance(node, ast.Attribute):
            require_ok(node.attr not in _FORBIDDEN_AST_ATTRS, "AST_FORBIDDEN_ATTR", node.attr)
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id == "Fraction") or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "Fraction"
            ):
                require_ok(len(node.args) == 2 and not node.keywords, "AST_FRACTION_ARGS", "Fraction requires 2 int args")
                for arg in node.args:
                    require_ok(
                        not isinstance(arg, ast.JoinedStr)
                        and not (isinstance(arg, ast.Constant) and type(arg.value) in (str, bytes)),
                        "AST_FRACTION_STR",
                        "Fraction(str) forbidden",
                    )


def canonical_json_bytes(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SmemCanaryVerifyError("BAD_JSON", str(exc)) from exc


def canonical_sha256(obj: Any, *, exclude_key: Optional[str] = None) -> str:
    payload = obj if exclude_key is None else {k: v for k, v in obj.items() if k != exclude_key}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    require_ok(path.is_file(), "MISSING_FILE", path.name)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            blk = fh.read(1 << 20)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def format_q(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def parse_q(raw: Any, *, code: str = "BAD_RATIONAL") -> Fraction:
    require_ok(type(raw) is str and bool(_Q_STR_RE.match(raw)), code, f"invalid rational: {raw!r}")
    num_s, den_s = raw.split("/", 1)
    q = Fraction(int(num_s), int(den_s))
    require_ok(format_q(q) == raw, code, f"non-canonical rational: {raw!r}")
    return q


def q_div(a: Fraction, b: Fraction) -> Fraction:
    require_ok(b != Fraction(0, 1), "ZERO_DIV", "division by zero")
    return Fraction(a.numerator * b.denominator, a.denominator * b.numerator)


def q_div_int(q: Fraction, n: int) -> Fraction:
    require_ok(is_strict_int(n) and n > 0, "BAD_DIVISOR", f"n={n!r}")
    return Fraction(q.numerator, q.denominator * n)


def contains_any_float(obj: Any) -> bool:
    if any(base.__name__ == "float" for base in type(obj).__mro__):
        return True
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if contains_any_float(k) or contains_any_float(v):
                return True
        return False
    if isinstance(obj, (AbcSequence, AbcSet)) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            if contains_any_float(item):
                return True
    return False


def _strip_nfkd_marks(raw: str) -> str:
    nfkd = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", raw))
    return "".join(ch for ch in nfkd if unicodedata.category(ch) not in ("Cf", "Mn", "Cc", "Cs", "Co"))


def _has_non_ascii_letter(norm: str) -> bool:
    return any(not ch.isascii() and ch.isalpha() and ch != "\u03bc" for ch in norm)


def is_forbidden_energy_value(raw: str) -> bool:
    norm = _strip_nfkd_marks(raw).casefold()
    return _has_non_ascii_letter(norm) or bool(_FORBIDDEN_ENERGY_VALUE_RE.search(norm))


def is_forbidden_energy_key_name(raw: str) -> bool:
    cleaned = _strip_nfkd_marks(raw)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", cleaned).replace("_", " ")
    if is_forbidden_energy_value(spaced):
        return True
    tokens = _ALNUM_TOKEN_RE.findall(spaced.casefold())
    for tok in tokens:
        if tok in _FORBIDDEN_ENERGY_KEYS:
            return True
    return bool(_FORBIDDEN_COMPACT_KEY_RE.search("".join(tokens)))


def has_forbidden_energy_claim(obj: Any, *, _path: Tuple[str, ...] = ()) -> bool:
    if isinstance(obj, str):
        return is_forbidden_energy_value(obj)
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str):
                child = _path + (k,)
                if child == ("energy_power_status",) and v == ENERGY_POLICY_STATUS:
                    continue
                if len(child) == 2 and child[0] == "kernels" and k in _EXPECTED_KERNEL_SYMBOLS:
                    if has_forbidden_energy_claim(v, _path=child):
                        return True
                    continue
                if is_forbidden_energy_key_name(k):
                    return True
                if has_forbidden_energy_claim(v, _path=child):
                    return True
            else:
                if has_forbidden_energy_claim(v, _path=_path):
                    return True
        return False
    if isinstance(obj, (AbcSequence, AbcSet)) and not isinstance(obj, (bytes, bytearray)):
        for item in obj:
            if has_forbidden_energy_claim(item, _path=_path):
                return True
    return False


# ============================================================================
# Pure-Z Exact Algebraic Oracles ([16,11,4] SECDED, Saturating CAS, FWHT_256)
# ============================================================================

def a16_encode_u16(data11: int) -> int:
    require_ok(is_strict_int(data11) and 0 <= data11 < 2048, "BAD_DATA11", f"data11={data11!r}")
    dpos = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    w = 0
    for i, pos in enumerate(dpos):
        if (data11 >> i) & 1:
            w |= 1 << pos
    p1 = (w & 0xAAAA).bit_count() & 1
    p2 = (w & 0xCCCC).bit_count() & 1
    p3 = (w & 0xF0F0).bit_count() & 1
    p4 = (w & 0xFF00).bit_count() & 1
    w |= (p1 << 1) | (p2 << 2) | (p3 << 4) | (p4 << 8)
    p0 = w.bit_count() & 1
    return w | p0


def a16_syndrome_u5(w16: int) -> int:
    require_ok(is_strict_int(w16) and 0 <= w16 <= 0xFFFF, "BAD_W16", f"w16={w16!r}")
    p0 = w16.bit_count() & 1
    p1 = (w16 & 0xAAAA).bit_count() & 1
    p2 = (w16 & 0xCCCC).bit_count() & 1
    p3 = (w16 & 0xF0F0).bit_count() & 1
    p4 = (w16 & 0xFF00).bit_count() & 1
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4)


def sat_add_packed_int(old_packed: int, sec_delta: int, ded_delta: int, diag_in: int) -> Tuple[int, int]:
    """Exact pure-Z model of silisense_ber_canary_sat_add_sys + monotonic OR diag."""
    require_ok(
        is_strict_int(old_packed)
        and is_strict_int(sec_delta)
        and is_strict_int(ded_delta)
        and is_strict_int(diag_in)
        and 0 <= old_packed <= 0xFFFFFFFF
        and sec_delta >= 0
        and ded_delta >= 0
        and 0 <= diag_in <= 0xFFFFFFFF,
        "BAD_PACK_ARGS",
        "sat_add_packed_int requires valid u32/non-negative ints",
    )
    old_sec = old_packed & 0xFFFF
    old_ded = old_packed >> 16
    s_add = 0x10000 if sec_delta > 0xFFFF else sec_delta
    d_add = 0x10000 if ded_delta > 0xFFFF else ded_delta
    sec_sum = old_sec + s_add
    ded_sum = old_ded + d_add
    ovf_sec = sec_sum > 0xFFFF
    ovf_ded = ded_sum > 0xFFFF
    new_sec = 0xFFFF if ovf_sec else sec_sum
    new_ded = 0xFFFF if ovf_ded else ded_sum
    diag_out = diag_in
    if ovf_sec or ovf_ded:
        if new_ded == 0:
            new_ded = 1
        diag_out |= 0x80000000
    return ((new_ded << 16) | new_sec, diag_out)


def verify_secded_and_smem_packer_oracle(
    *,
    encode_fn: Any = a16_encode_u16,
    syndrome_fn: Any = a16_syndrome_u5,
    sat_add_fn: Any = sat_add_packed_int,
) -> Dict[str, Any]:
    cw = encode_fn(0x5A3)
    require_ok(syndrome_fn(cw) == 0, "ORACLE_CLEAN_SYND", "clean codeword nonzero syndrome")

    sec_healed = 0
    for bit in range(16):
        corrupted = cw ^ (1 << bit)
        s = syndrome_fn(corrupted)
        require_ok((s & 1) == 1, "ORACLE_SEC_PARITY", f"bit={bit}")
        pos = (s >> 1) & 15
        healed = corrupted ^ (1 << pos)
        require_ok(healed == cw, "ORACLE_SEC_HEAL", f"bit={bit}")
        sec_healed += 1

    ded_caught = 0
    for i in range(16):
        for j in range(i + 1, 16):
            corrupted = cw ^ (1 << i) ^ (1 << j)
            s = syndrome_fn(corrupted)
            require_ok(s != 0 and (s & 1) == 0, "ORACLE_DED_DETECT", f"pair=({i},{j})")
            ded_caught += 1

    pk_clean, dg_clean = sat_add_fn(0, 0, 0, 0)
    pk_ovf_sec, dg_ovf_sec = sat_add_fn(0, 70000, 0, 0x15)
    pk_ovf_ded, dg_ovf_ded = sat_add_fn(0, 12, 70000, 0x22)
    # Multi-CTA accumulation tests: 2 CTAs x (40000, 0) and 2 CTAs x (7, 0x8000)
    mc1_pk, mc1_dg = sat_add_fn(0, 40000, 0, 0x15)
    mc_sec_pk, mc_sec_dg = sat_add_fn(mc1_pk, 40000, 0, mc1_dg | 0x15)
    mc2_pk, mc2_dg = sat_add_fn(0, 7, 0x8000, 0x22)
    mc_ded_pk, mc_ded_dg = sat_add_fn(mc2_pk, 7, 0x8000, mc2_dg | 0x22)

    require_ok(pk_clean == 0 and dg_clean == 0, "ORACLE_PACK_CLEAN", "clean mismatch")
    require_ok(
        pk_ovf_sec == ((1 << 16) | 0xFFFF) and dg_ovf_sec == (0x80000000 | 0x15),
        "ORACLE_PACK_OVF_SEC",
        "sec overflow mismatch",
    )
    require_ok(
        pk_ovf_ded == ((0xFFFF << 16) | 12) and dg_ovf_ded == (0x80000000 | 0x22),
        "ORACLE_PACK_OVF_DED",
        "ded overflow mismatch",
    )
    require_ok(
        mc_sec_pk == ((1 << 16) | 0xFFFF) and mc_sec_dg == (0x80000000 | 0x15),
        "ORACLE_MULTI_CTA_SEC",
        "multi-cta sec carry mismatch",
    )
    require_ok(
        mc_ded_pk == ((0xFFFF << 16) | 14) and mc_ded_dg == (0x80000000 | 0x22),
        "ORACLE_MULTI_CTA_DED",
        "multi-cta ded wrap mismatch",
    )
    return {
        "sec_healed": sec_healed,
        "ded_caught": ded_caught,
        "overflow_sec70000_packed_u32": pk_ovf_sec,
        "overflow_sec70000_diag_u32": dg_ovf_sec,
        "overflow_ded70000_packed_u32": pk_ovf_ded,
        "overflow_ded70000_diag_u32": dg_ovf_ded,
        "multi_cta_sec_carry_packed_u32": mc_sec_pk,
        "multi_cta_sec_carry_diag_u32": mc_sec_dg,
        "multi_cta_ded_wrap_packed_u32": mc_ded_pk,
        "multi_cta_ded_wrap_diag_u32": mc_ded_dg,
    }


def fwht256_bent4_int(u256: Sequence[int]) -> List[int]:
    """Exact pure-Z Walsh-Hadamard 256 x bent4 butterfly (matching sm_86 kernel up to 1/sqrt(32))."""
    require_ok(len(u256) == 256, "BAD_FWHT_LEN", f"len={len(u256)}")
    buf: List[int] = [0] * 256
    for idx in range(256):
        w = idx >> 5
        lane = idx & 31
        src_w = w ^ (lane & 7)
        src_idx = (src_w << 5) | lane
        b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
        val = u256[src_idx]
        buf[idx] = -val if b4 else val
    step = 1
    while step < 256:
        stride = step << 1
        for base in range(0, 256, stride):
            for off in range(step):
                a = buf[base + off]
                b = buf[base + off + step]
                buf[base + off] = a + b
                buf[base + off + step] = a - b
        step = stride
    return buf


def verify_fwht256_fold_oracle(*, fwht_fn: Any = fwht256_bent4_int) -> Dict[str, Any]:
    """Verify exact Parseval identity (sum y^2 == 256 * sum u^2) & upper-half gradient routing over Z."""
    h512 = [((i * 17 + 5) % 13) - 6 for i in range(512)]
    alpha = [((i * 3 + 1) % 5) + 1 for i in range(256)]
    beta = [-(((i * 7 + 2) % 5) + 1) for i in range(256)]

    u_fold = [alpha[i] * h512[i] + beta[i] * h512[256 + i] for i in range(256)]
    y_fold = fwht_fn(u_fold)
    energy_u = sum(x * x for x in u_fold)
    energy_y = sum(y * y for y in y_fold)
    require_ok(energy_y == 256 * energy_u, "ORACLE_PARSEVAL_FAIL", "Parseval mismatch")

    h512_pert = list(h512)
    h512_pert[300] += 3
    y_prefix_a = fwht_fn(h512[:256])
    y_prefix_b = fwht_fn(h512_pert[:256])
    prefix_upper_diff_l1 = sum(abs(a - b) for a, b in zip(y_prefix_a, y_prefix_b))

    u_fold_pert = [alpha[i] * h512_pert[i] + beta[i] * h512_pert[256 + i] for i in range(256)]
    y_fold_pert = fwht_fn(u_fold_pert)
    fold_upper_diff_l1 = sum(abs(a - b) for a, b in zip(y_fold, y_fold_pert))
    require_ok(prefix_upper_diff_l1 == 0 and fold_upper_diff_l1 > 0, "ORACLE_UPPER_ROUTING", "bad routing")
    return {
        "parseval_input_l2_sq": energy_u,
        "parseval_output_l2_sq": energy_y,
        "prefix_upper_half_delta_l1": prefix_upper_diff_l1,
        "fold1k_upper_half_delta_l1": fold_upper_diff_l1,
        "fold1k_param_count": 512,
        "fold1k_fp16_bytes": 1024,
    }


# ============================================================================
# Self-Bound On-Disk Receipt Verifier (PTXAS + Live Results + Gate Re-Grading)
# ============================================================================

def verify_ptxas_receipt(ptx: Any, *, expected_cu_sha: Optional[str] = None) -> Dict[str, Any]:
    require_ok(type(ptx) is dict, "PTXAS_TYPE", "ptxas receipt must be dict")
    require_ok(ptx.get("schema") == SCHEMA_PTXAS, "PTXAS_SCHEMA", "bad ptxas schema")
    require_ok(
        ptx.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_claim(ptx),
        "PTXAS_ENERGY_POLICY",
        "ptxas receipt violates energy policy",
    )
    require_ok(not contains_any_float(ptx), "PTXAS_FLOAT_FORBIDDEN", "ptxas receipt contains float")
    require_ok(frozenset(ptx.keys()) == _PTXAS_KEYS, "PTXAS_KEYS", "unexpected or missing ptxas keys")
    rsha = ptx["receipt_sha256"]
    require_ok(is_hex64(rsha), "PTXAS_SHA_FORMAT", "bad receipt_sha256")
    require_ok(canonical_sha256(ptx, exclude_key="receipt_sha256") == rsha, "PTXAS_SHA_MISMATCH", "sha mismatch")
    require_ok(ptx["arch"] == "sm_86", "PTXAS_ARCH", "arch != sm_86")
    require_ok(ptx["compile_flags"] == EXPECTED_COMPILE_FLAGS, "PTXAS_FLAGS", "compile_flags mismatch")
    require_ok(type(ptx["nvcc_version"]) is str and len(ptx["nvcc_version"]) > 0, "PTXAS_NVCC", "bad nvcc_version")
    require_ok(ptx["cu_path"] == KERNEL_CU_REL, "PTXAS_CU_PATH", "unexpected cu_path")
    cu_sha = ptx["cu_sha256"]
    require_ok(is_hex64(cu_sha), "PTXAS_CU_SHA", "bad cu_sha256")
    if expected_cu_sha is not None:
        require_ok(cu_sha == expected_cu_sha, "PTXAS_CU_SHA_DISK", "cu_sha256 mismatch vs disk")

    kernels = ptx["kernels"]
    require_ok(type(kernels) is dict and len(kernels) == 5, "PTXAS_KERNELS", "expected 5 kernels")
    require_ok(frozenset(kernels.keys()) == _EXPECTED_KERNEL_SYMBOLS, "PTXAS_REQUIRED_SYMBOLS", "kernel symbols mismatch")

    nonzero = 0
    for name, st in kernels.items():
        require_ok(type(st) is dict and frozenset(st.keys()) == _KERNEL_STAT_KEYS, "PTXAS_KSTAT", name)
        for field in _KERNEL_STAT_KEYS:
            require_ok(is_strict_int(st[field]) and st[field] >= 0, "PTXAS_FIELD_TYPE", f"{name}.{field}")
        if (
            st["stack_bytes"] != 0
            or st["spill_store_bytes"] != 0
            or st["spill_load_bytes"] != 0
            or st["ldl_insns"] != 0
            or st["stl_insns"] != 0
            or st["registers"] <= 0
        ):
            nonzero += 1
        exp_regs, exp_bar, exp_smem, exp_cas, exp_or, exp_atoms = _EXPECTED_KERNEL_SPECS[name]
        require_ok(
            st["registers"] == exp_regs
            and st["barriers"] == exp_bar
            and st["smem_bytes"] == exp_smem
            and st["atom_cas_sys_insns"] == exp_cas
            and st["red_or_sys_insns"] == exp_or
            and st["atoms_smem_insns"] == exp_atoms,
            "PTXAS_KERNEL_SPEC",
            f"unexpected ptxas/SASS spec for {name}",
        )
    require_ok(
        ptx["total_kernels"] == 5
        and ptx["zero_spill_kernels"] == 5
        and ptx["nonzero_spill_kernels"] == 0
        and nonzero == 0,
        "PTXAS_SPILL_FAIL",
        f"nonzero_spills={nonzero}",
    )
    return ptx


def verify_results_receipt(
    res: Any,
    ptx: Dict[str, Any],
    *,
    expected_bench_cu_sha: Optional[str] = None,
) -> Dict[str, Any]:
    require_ok(type(res) is dict, "RESULTS_TYPE", "results must be dict")
    require_ok(res.get("schema") == SCHEMA_RESULTS, "RESULTS_SCHEMA", "bad results schema")
    require_ok(
        res.get("energy_power_status") == ENERGY_POLICY_STATUS and not has_forbidden_energy_claim(res),
        "RESULTS_ENERGY_POLICY",
        "results violates uncontrolled-laptop energy policy",
    )
    require_ok(not contains_any_float(res), "RESULTS_FLOAT_FORBIDDEN", "results contains float")
    require_ok(frozenset(res.keys()) == _RESULTS_KEYS, "RESULTS_KEYS", "unexpected or missing results keys")

    rsha = res["result_sha256"]
    require_ok(is_hex64(rsha), "RESULTS_SHA_FORMAT", "bad result_sha256")
    require_ok(canonical_sha256(res, exclude_key="result_sha256") == rsha, "RESULTS_SHA_MISMATCH", "sha mismatch")

    hw = res["hardware"]
    require_ok(
        type(hw) is dict
        and frozenset(hw.keys()) == _HARDWARE_KEYS
        and type(hw["device"]) is str
        and len(hw["device"]) > 0
        and hw["sm"] == "8.6"
        and hw["nvcc_version"] == ptx["nvcc_version"]
        and type(hw["host_native_atomic_supported"]) is bool,
        "RESULTS_HARDWARE",
        "invalid hardware block",
    )

    sb = res["source_bindings"]
    require_ok(type(sb) is dict and frozenset(sb.keys()) == _SOURCE_BINDINGS_KEYS, "RESULTS_BINDINGS", "bad source_bindings")
    require_ok(
        sb["kernel_cu_path"] == KERNEL_CU_REL
        and sb["bench_cu_path"] == BENCH_CU_REL
        and sb["ptxas_receipt_path"] == PTXAS_RECEIPT_REL,
        "RESULTS_BINDING_PATHS",
        "unexpected source binding paths",
    )
    require_ok(sb["kernel_cu_sha256"] == ptx["cu_sha256"], "RESULTS_CU_BIND", "kernel cu sha mismatch")
    require_ok(sb["ptxas_receipt_sha256"] == ptx["receipt_sha256"], "RESULTS_PTX_BIND", "ptxas sha mismatch")
    require_ok(is_hex64(sb["bench_cu_sha256"]), "RESULTS_BENCH_SHA", "bad bench_cu_sha256")
    if expected_bench_cu_sha is not None:
        require_ok(sb["bench_cu_sha256"] == expected_bench_cu_sha, "RESULTS_BENCH_DISK", "bench cu disk mismatch")

    sil = res["silicon_parity_and_canary"]
    require_ok(type(sil) is dict and frozenset(sil.keys()) == _SILICON_KEYS, "RESULTS_SILICON", "bad silicon section")
    cell_err = parse_q(sil["folded1k_cell_fp64_max_err_q"], code="RESULTS_PARITY_Q")
    head_err = parse_q(sil["folded1k_head_fp64_max_err_q"], code="RESULTS_PARITY_Q")
    require_ok(Fraction(0, 1) <= cell_err <= Fraction(1, 100000), "RESULTS_CELL_PARITY", f"cell_err={cell_err}")
    require_ok(Fraction(0, 1) <= head_err <= Fraction(1, 100000), "RESULTS_HEAD_PARITY", f"head_err={head_err}")

    sec16_hid_err = parse_q(sil["sec16_hidden_max_abs_err_q"], code="RESULTS_SEC_ERR")
    sec16_log_err = parse_q(sil["sec16_logits_max_abs_err_q"], code="RESULTS_SEC_ERR")
    head_unf_q = parse_q(sil["standalone_head_unfolded_us_q"], code="RESULTS_HEAD_US")
    head_fld_q = parse_q(sil["standalone_head_folded1k_us_q"], code="RESULTS_HEAD_US")
    head_ovh_q = parse_q(sil["standalone_head_overhead_us_q"], code="RESULTS_HEAD_US")
    require_ok(head_unf_q > Fraction(0, 1) and head_fld_q > Fraction(0, 1), "RESULTS_HEAD_POS", "head latency must be > 0")
    require_ok(head_fld_q - head_unf_q == head_ovh_q, "RESULTS_HEAD_DELTA", "head overhead delta mismatch")

    storms = sil["storms"]
    require_ok(type(storms) is list and len(storms) == 4, "RESULTS_STORMS_LEN", "expected 4 storm rates")
    seen_rates: List[int] = []
    for st in storms:
        require_ok(type(st) is dict and frozenset(st.keys()) == _STORM_KEYS, "RESULTS_STORM_TYPE", "bad storm entry")
        rate = st["rate_pct"]
        require_ok(is_strict_int(rate), "RESULTS_STORM_RATE", "bad rate_pct")
        seen_rates.append(rate)
        for k in ("expected_sec", "expected_ded", "warp_sec", "warp_ded", "warp_sys_atomics", "smem_sec", "smem_ded", "smem_sys_atomics"):
            require_ok(is_strict_int(st[k]) and st[k] >= 0, "RESULTS_STORM_INT", k)
        red_q = parse_q(st["atomic_reduction_ratio_q"], code="RESULTS_STORM_RED")
        expected_red = Fraction(st["warp_sys_atomics"], st["smem_sys_atomics"]) if st["smem_sys_atomics"] > 0 else Fraction(0, 1)
        require_ok(red_q == expected_red, "RESULTS_STORM_RED_MISMATCH", f"rate={rate}")
        w_us = parse_q(st["warp_us_per_tok_q"], code="RESULTS_STORM_US")
        s_us = parse_q(st["smem_us_per_tok_q"], code="RESULTS_STORM_US")
        sp_q = parse_q(st["latency_speedup_q"], code="RESULTS_STORM_US")
        require_ok(w_us > Fraction(0, 1) and s_us > Fraction(0, 1), "RESULTS_STORM_POS", f"rate={rate}")
        require_ok(sp_q == q_div(w_us, s_us), "RESULTS_STORM_SPEEDUP_MISMATCH", f"rate={rate}")
    require_ok(seen_rates == [0, 1, 10, 100], "RESULTS_STORM_RATES", f"rates={seen_rates}")

    bpb = res["utf8_bpb_study"]
    require_ok(type(bpb) is dict and frozenset(bpb.keys()) == _BPB_STUDY_KEYS, "RESULTS_BPB", "bad utf8_bpb_study")
    expected_seeds = bpb["seeds"]
    require_ok(
        type(expected_seeds) is list
        and expected_seeds == [42, 137, 2026]
        and all(is_strict_int(s) for s in expected_seeds)
        and bpb["d_model"] == 512
        and is_strict_int(bpb["eval_bytes"])
        and bpb["eval_bytes"] == bpb["eval_chunks"] * bpb["seq_len"]
        and bpb["eval_bytes"] > 0
        and bpb["tokens_per_seed"] == bpb["batch_size"] * bpb["seq_len"] * bpb["steps_per_seed"],
        "RESULTS_BPB_META",
        "invalid bpb study metadata",
    )
    arms = bpb["arms"]
    require_ok(type(arms) is dict and tuple(sorted(arms.keys())) == tuple(sorted(_EXPECTED_ARMS)), "RESULTS_ARMS", "bad arms")
    but_err = parse_q(bpb["butterfly_vs_matrix_fp64_max_err_q"], code="RESULTS_BUT_ERR")
    g1_norm = parse_q(bpb["arm1_upper_half_grad_norm_q"], code="RESULTS_GRAD_Q")
    g3_norm = parse_q(bpb["arm3_upper_half_grad_norm_q"], code="RESULTS_GRAD_Q")
    require_ok(Fraction(0, 1) <= but_err <= Fraction(1, 100000000000), "RESULTS_BUT_BOUND", f"err={but_err}")
    require_ok(g1_norm == Fraction(0, 1) and g3_norm > Fraction(0, 1), "RESULTS_GRAD_ROUTING", "bad grad routing")

    arm_means: Dict[str, Fraction] = {}
    arm_seed_bpbs: Dict[str, List[Fraction]] = {}
    for arm_name in _EXPECTED_ARMS:
        entry = arms[arm_name]
        require_ok(type(entry) is dict and frozenset(entry.keys()) == _ARM_KEYS, "RESULTS_ARM_TYPE", arm_name)
        exp_p, exp_b, exp_hbm = _EXPECTED_HEAD_PARAMS[arm_name]
        require_ok(
            entry["head_params"] == exp_p
            and entry["head_fp16_bytes"] == exp_b
            and entry["hbm_vocab_table_bytes"] == exp_hbm
            and is_strict_int(entry["total_params"])
            and entry["total_params"] > 0,
            "RESULTS_ARM_BYTES",
            arm_name,
        )
        seeds_list = entry["seeds"]
        require_ok(type(seeds_list) is list and len(seeds_list) == 3, "RESULTS_ARM_SEEDS", arm_name)
        exp_seed_keys = _ARM3_SEED_KEYS if arm_name == "arm3_learned_1kib_fp16_fold_fwht" else _SEED_KEYS
        vals: List[Fraction] = []
        row_seeds: List[int] = []
        for row in seeds_list:
            require_ok(
                type(row) is dict
                and frozenset(row.keys()) == exp_seed_keys
                and is_strict_int(row["seed"])
                and is_strict_int(row["eval_tokens"])
                and row["eval_tokens"] == bpb["eval_bytes"],
                "RESULTS_SEED_ROW",
                arm_name,
            )
            row_seeds.append(row["seed"])
            v_fp16 = parse_q(row["eval_bpb_q"], code="RESULTS_SEED_BPB")
            v_bf16 = parse_q(row["eval_bpb_bf16_q"], code="RESULTS_SEED_BPB")
            qdiff = parse_q(row["fp16_quant_bpb_diff_q"], code="RESULTS_SEED_BPB")
            wall_s = parse_q(row["train_wall_s_q"], code="RESULTS_SEED_WALL")
            require_ok(v_fp16 > Fraction(0, 1) and v_bf16 > Fraction(0, 1) and wall_s > Fraction(0, 1), "RESULTS_SEED_BPB_POS", arm_name)
            require_ok(qdiff == abs(v_fp16 - v_bf16), "RESULTS_SEED_QDIFF_MISMATCH", arm_name)
            if arm_name == "arm3_learned_1kib_fp16_fold_fwht":
                a_rms = parse_q(row["alpha_rms_q"], code="RESULTS_ARM3_RMS")
                b_rms = parse_q(row["beta_rms_q"], code="RESULTS_ARM3_RMS")
                require_ok(a_rms > Fraction(0, 1) and b_rms > Fraction(0, 1), "RESULTS_ARM3_RMS_POS", arm_name)
            vals.append(v_fp16)
        require_ok(row_seeds == expected_seeds, "RESULTS_SEED_ORDER", arm_name)
        calc_mean = q_div_int(sum(vals, Fraction(0, 1)), len(vals))
        stored_mean = parse_q(entry["mean_eval_bpb_q"], code="RESULTS_MEAN_BPB")
        require_ok(stored_mean == calc_mean, "RESULTS_MEAN_MISMATCH", arm_name)
        arm_means[arm_name] = calc_mean
        arm_seed_bpbs[arm_name] = vals

    deltas = res["bpb_deltas"]
    require_ok(type(deltas) is dict and frozenset(deltas.keys()) == _BPB_DELTAS_KEYS, "RESULTS_DELTAS", "bad bpb_deltas")
    a1 = arm_means["arm1_prefix_fwht"]
    a2 = arm_means["arm2_fixed_diff_fwht"]
    a2b = arm_means["arm2b_fixed_diff_1scalar_fwht"]
    a3 = arm_means["arm3_learned_1kib_fp16_fold_fwht"]
    a4 = arm_means["arm4_dense_256x512"]
    require_ok(parse_q(deltas["arm3_minus_arm1_prefix_q"]) == a3 - a1, "RESULTS_DELTA_A1", "a3-a1 mismatch")
    require_ok(parse_q(deltas["arm3_minus_arm2_fixed_diff_q"]) == a3 - a2, "RESULTS_DELTA_A2", "a3-a2 mismatch")
    require_ok(parse_q(deltas["arm3_minus_arm2b_1scalar_q"]) == a3 - a2b, "RESULTS_DELTA_A2B", "a3-a2b mismatch")
    require_ok(parse_q(deltas["arm3_minus_arm4_dense_q"]) == a3 - a4, "RESULTS_DELTA_A4", "a3-a4 mismatch")

    beats_a1 = all(x < y for x, y in zip(arm_seed_bpbs["arm3_learned_1kib_fp16_fold_fwht"], arm_seed_bpbs["arm1_prefix_fwht"]))
    beats_a2 = all(x < y for x, y in zip(arm_seed_bpbs["arm3_learned_1kib_fp16_fold_fwht"], arm_seed_bpbs["arm2_fixed_diff_fwht"]))
    beats_a2b = all(x < y for x, y in zip(arm_seed_bpbs["arm3_learned_1kib_fp16_fold_fwht"], arm_seed_bpbs["arm2b_fixed_diff_1scalar_fwht"]))
    require_ok(
        deltas["arm3_beats_arm1_all_seeds"] is beats_a1
        and deltas["arm3_beats_arm2_all_seeds"] is beats_a2
        and deltas["arm3_beats_arm2b_all_seeds"] is beats_a2b,
        "RESULTS_BEATS_FLAGS",
        "per-seed win flags mismatch",
    )

    g1_calc = ptx["nonzero_spill_kernels"] == 0 and ptx["total_kernels"] == 5
    g2_calc = a3 < a1 and a3 < a2 and beats_a1 and beats_a2 and head_ovh_q <= Fraction(1, 2)
    g3_calc = (
        sil["clean_packed_u32"] == 0
        and sil["clean_sys_atomics"] == 0
        and sil["sec16_packed_u32"] == 16
        and sil["sec16_sec_count"] == 16
        and sil["sec16_ded_count"] == 0
        and sil["sec16_sys_atomics"] == 1
        and sec16_hid_err == Fraction(0, 1)
        and sec16_log_err == Fraction(0, 1)
        and sil["ded120_packed_u32"] == (120 << 16)
        and sil["ded120_sec_count"] == 0
        and sil["ded120_ded_count"] == 120
        and sil["ded120_sys_atomics"] == 1
        and sil["overflow_sec70000_packed_u32"] == ((1 << 16) | 0xFFFF)
        and sil["overflow_sec70000_diag_u32"] == (0x80000000 | 0x15)
        and sil["overflow_ded70000_packed_u32"] == ((0xFFFF << 16) | 12)
        and sil["overflow_ded70000_diag_u32"] == (0x80000000 | 0x22)
        and sil["multi_cta_sec_carry_packed_u32"] == ((1 << 16) | 0xFFFF)
        and sil["multi_cta_sec_carry_diag_u32"] == (0x80000000 | 0x15)
        and sil["multi_cta_ded_wrap_packed_u32"] == ((0xFFFF << 16) | 14)
        and sil["multi_cta_ded_wrap_diag_u32"] == (0x80000000 | 0x22)
        and all(
            st["smem_sec"] == st["expected_sec"]
            and st["smem_ded"] == st["expected_ded"]
            and st["warp_sec"] == st["expected_sec"]
            and st["warp_ded"] == st["expected_ded"]
            and ((st["rate_pct"] == 0 and st["smem_sys_atomics"] == 0) or (st["rate_pct"] > 0 and st["smem_sys_atomics"] == 1))
            for st in storms
        )
    )
    gates = res["gates"]
    require_ok(
        type(gates) is dict
        and frozenset(gates.keys()) == _GATES_KEYS
        and gates["G1_zero_spill_sm86"] is g1_calc
        and gates["G2_fold1k_bpb_and_latency"] is g2_calc
        and gates["G3_smem_canary_atomic_and_recovery"] is g3_calc,
        "RESULTS_GATES_MISMATCH",
        "recomputed gates mismatch",
    )
    if g1_calc and g2_calc and g3_calc:
        expected_verdict = "CONFIRMED"
    elif g1_calc and (g2_calc or g3_calc):
        expected_verdict = "PARTIAL"
    else:
        expected_verdict = "FALSIFIED"
    require_ok(res["verdict"] == expected_verdict, "RESULTS_VERDICT_MISMATCH", f"expected={expected_verdict}")
    return res


def verify_on_disk(repo_root: Path) -> Dict[str, Any]:
    verify_pure_q_ast(Path(__file__).resolve().read_text(encoding="utf-8"))
    kernel_cu = repo_root.joinpath(KERNEL_CU_REL)
    bench_cu = repo_root.joinpath(BENCH_CU_REL)
    ptx_path = repo_root.joinpath(PTXAS_RECEIPT_REL)
    res_path = repo_root.joinpath(RESULTS_RECEIPT_REL)

    kernel_sha = sha256_file(kernel_cu)
    bench_sha = sha256_file(bench_cu)
    ptx = load_strict_canonical_json(ptx_path)
    res = load_strict_canonical_json(res_path)

    verify_ptxas_receipt(ptx, expected_cu_sha=kernel_sha)
    verify_results_receipt(res, ptx, expected_bench_cu_sha=bench_sha)
    secded_oracle = verify_secded_and_smem_packer_oracle()
    fwht_oracle = verify_fwht256_fold_oracle()

    summary: Dict[str, Any] = {
        "schema": SCHEMA_VERIFY,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "status": "PASS",
        "verdict": res["verdict"],
        "gates": res["gates"],
        "kernel_cu_sha256": kernel_sha,
        "bench_cu_sha256": bench_sha,
        "ptxas_file_sha256": sha256_file(ptx_path),
        "results_file_sha256": sha256_file(res_path),
        "ptxas_receipt_sha256": ptx["receipt_sha256"],
        "result_sha256": res["result_sha256"],
        "secded_oracle": secded_oracle,
        "fwht_oracle": fwht_oracle,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pure-Q / Pure-Z Verifier for sm_86 CTA-SMEM Canary & Folded FWHT Head")
    parser.add_argument("--check", action="store_true", help="Verify on-disk receipts and oracles")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary")
    parser.add_argument("--repo-root", type=str, default=None, help="Override repository root directory")
    parser.add_argument("--require-verdict", type=str, default=None, help="Fail closed unless verdict matches this exact value")
    args = parser.parse_args(argv)

    root = Path(args.repo_root) if args.repo_root is not None else Path(__file__).resolve().parent.parent
    try:
        summary = verify_on_disk(root)
        if args.require_verdict is not None:
            require_ok(
                summary["verdict"] == args.require_verdict,
                "VERDICT_REQUIREMENT_FAILED",
                f"expected={args.require_verdict} got={summary['verdict']}",
            )
    except SmemCanaryVerifyError as exc:
        sys.stderr.write(json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"PASS {summary['verdict']} result_sha256={summary['result_sha256']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
