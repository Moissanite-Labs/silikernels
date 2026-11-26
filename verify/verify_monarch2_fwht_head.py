#!/usr/bin/env python3
"""Pure-Q / Pure-Z Fail-Closed Verifier for Ampere sm_86 Monarch-2 + Byte-Bias FWHT Head.

Zero-float, zero-ast.Div, zero-ast.FloorDiv, pure-stdlib, self-bound on-disk verifier for:
  1. Strict canonical-JSON & duplicate-key-free loading (with 1 MiB file cap, integer digit
     cap, pre-parse & iterative MAX_JSON_DEPTH=32, float rejection, root-to-leaf dir_fd O_NOFOLLOW |
     O_NONBLOCK + S_ISDIR/S_ISREG refusal across all verified files and Path(__file__), and
     single-fd read-once hashing) with exact key allowlists and on-disk SHA-256 bindings across
     both CUDA sources, the standalone UTF-8 BPB study script, pinned external corpus hashes
     cross-checked against the study script AST, and both JSON receipts.
  2. Built-in closed-allowlist pure-Q AST self-audit banning ast.Div, ast.FloorDiv, ast.Mod,
     ast.Pow, ast.Assert, f-string format_spec, float/complex constants, non-allowlisted
     imports/module attributes, arbitrary ast.parse calls, two-step/direct type() escapes,
     json aliasing/rebinding and unguarded json.loads, dunder/reflection/float escapes,
     local/attribute/def/class Fraction rebinding, and non-safe Fraction args.
  3. sm_86 ptxas & cuobjdump SASS zero-spill / exact register, barrier, SMEM, LDL, and
     STL receipt verification (4/4 kernels, 0 B spills, 0 LDL/STL, --Werror all-warnings,
     strict nvcc_version format, line-splice/raw-string/comment-stripped CUDA & bench
     preprocessor-free function-scoped scale & __global__ kernel-def binding, plus closed
     allowlist BPB study script AST audit & single-assignment corpus-pin/scale checks).
     Offline CI verifies receipts, source bindings, contextual scale tokens, and pure-Z
     oracles without a GPU; live nvcc zero-spill compilation is available locally via
     `check-monarch2-head-zero-spill`.
  4. Exact pure-Z algebraic oracles: H_256 involution (H_256^2 = 256 I), FWHT_256 x bent4
     unscaled Parseval identity (sum y^2 = 256 sum u^2 -> 8 * sum u^2 at 1/sqrt(32) scale),
     6 fixed known-answer delta -> signed-Hadamard multi-vector tests (rejecting sign-drop,
     permutation-reversal, output-rotation, negation, and warp-XOR bugs), and 1-stage (16/256)
     vs 2-stage (256/256) Monarch-16x16 bipartite reachability + exact FP16 byte count
     (16,896 B, 0 B HBM).
  5. Exact pure-Q recomputation of 6-arm x 3-seed UTF-8 BPB means (with strict seed order
     alignment and strict int metadata checks), FP16 quantization deltas, lifts over unbiased
     scalar & byte-bias baselines, matched-byte rank-16 low-rank beat, dense gap closure
     fraction, gates G1/G2 (including <= 3.0 us/tok across all 4 sm_86 arms), and overall
     verdict (--require-verdict).
  6. Fail-closed bounded-backtracking (Python >= 3.11 possessive digit quantifiers, ReDoS-free)
     uncontrolled-laptop energy/power telemetry guard (99/99 bad / 55/55 good).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence as AbcSequence, Set as AbcSet
import errno
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

SCHEMA_PTXAS = "silikernels.sm86.monarch2_fwht_head_ptxas_receipt.v1"
SCHEMA_RESULTS = "silikernels.sm86.monarch2_fwht_head_results.v1"
SCHEMA_VERIFY = "silikernels.sm86.monarch2_fwht_head_verification.v1"
ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
EXPECTED_COMPILE_FLAGS = (
    "-O3 -std=c++17 -arch=sm_86 --ptxas-options=-v --Werror all-warnings "
    "--shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror"
)

KERNEL_CU_REL = "cuda/sm86/holosphere_monarch2_fwht_head_sm86.cu"
BENCH_CU_REL = "cuda/sm86/holosphere_monarch2_fwht_head_bench.cu"
BPB_STUDY_SCRIPT_REL = "cuda/sm86/monarch2_fwht_head_bpb_study.py"
PTXAS_RECEIPT_REL = "receipts/monarch2_fwht_head_ptxas_receipt.json"
RESULTS_RECEIPT_REL = "receipts/monarch2_fwht_head_results.json"
EXPECTED_CORPUS_ID = "holosphere-v3-flagship-corpus"
MAX_FILE_BYTES = 1_048_576
MAX_JSON_INT_DIGITS = 1024
MAX_JSON_DEPTH = 32

_C_LINE_SPLICE_RE = re.compile(r"\\\r?\n")
_C_STRIP_RE = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|R"([^()\s\\]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
)
_C_PREPROC_COND_RE = re.compile(r"(?m)^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b")
_KERNEL_FN_SCALE_SPECS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("fwht256_plain_8warp", re.compile(r"\bv\s*\*=\s*0\.0625f\s*;")),
    ("fwht256_bent4_from_smem_8warp", re.compile(r"\bv\s*\*=\s*0\.17677669529663688f\s*;")),
)
_KERNEL_DEF_NAMES: Tuple[str, ...] = (
    "holosphere_arm2b_bias_514b_sm86",
    "holosphere_arm5_bias_2p5kib_sm86",
    "holosphere_arm7_bias_8p5kib_sm86",
    "holosphere_arm9_monarch2_bias_16p5kib_sm86",
)
_BENCH_FN_SCALE_SPECS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("fwht256_plain_fp64", re.compile(r"\bout\[k\]\s*=\s*acc\s*\*\s*0\.0625\s*;")),
    ("fwht256_bent4_fp64", re.compile(r"\binv_sqrt32\s*=\s*1\.0\s*/\s*std::sqrt\(\s*32\.0\s*\)\s*;")),
)
_ALLOWED_BPB_IMPORTS: Dict[str, Optional[str]] = {
    "argparse": None,
    "hashlib": None,
    "json": None,
    "math": None,
    "sys": None,
    "time": None,
    "torch": None,
    "torch.nn": "nn",
    "torch.nn.functional": "F",
}
_ALLOWED_BPB_FROM_IMPORTS: Dict[str, frozenset[str]] = {
    "__future__": frozenset({"annotations"}),
    "fractions": frozenset({"Fraction"}),
    "pathlib": frozenset({"Path"}),
    "typing": frozenset({"Any", "Dict", "List", "Optional", "Tuple"}),
}
_FORBIDDEN_BPB_SCRIPT_NAMES: frozenset[str] = frozenset(
    {
        "__import__", "eval", "exec", "compile", "getattr", "setattr", "delattr",
        "globals", "locals", "vars", "builtins", "__builtins__",
    }
)
_BPB_CORPUS_PIN_NAMES: Tuple[str, ...] = (
    "CORPUS_ID",
    "EXPECTED_CORPUS_MANIFEST_SHA256",
    "EXPECTED_TRAIN_BIN_SHA256",
    "EXPECTED_EVAL_BIN_SHA256",
)
_KAT_DELTA_INDICES: Tuple[int, ...] = (0, 5, 15, 37, 74, 250)

ARM2B_BIAS_SYMBOL = "_ZN24holosphere_monarch2_head31holosphere_arm2b_bias_514b_sm86EPKfPK6__halfS4_Pf"
ARM5_BIAS_SYMBOL = "_ZN24holosphere_monarch2_head32holosphere_arm5_bias_2p5kib_sm86EPKfPK7__half2S4_PK6__halfPf"
ARM7_BIAS_SYMBOL = "_ZN24holosphere_monarch2_head32holosphere_arm7_bias_8p5kib_sm86EPKfPK5uint4PK6__halfPf"
ARM9_MONARCH2_SYMBOL = "_ZN24holosphere_monarch2_head42holosphere_arm9_monarch2_bias_16p5kib_sm86EPKfPK5uint4S4_PK6__halfPf"

# Exact expected (registers, barriers, smem_bytes)
_EXPECTED_KERNEL_SPECS: Dict[str, Tuple[int, int, int]] = {
    ARM2B_BIAS_SYMBOL: (18, 1, 3072),
    ARM5_BIAS_SYMBOL: (22, 1, 3072),
    ARM7_BIAS_SYMBOL: (26, 1, 12288),
    ARM9_MONARCH2_SYMBOL: (27, 1, 20480),
}
_EXPECTED_KERNEL_SYMBOLS: frozenset[str] = frozenset(_EXPECTED_KERNEL_SPECS.keys())

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
        "registers", "barriers", "smem_bytes", "ldl_insns", "stl_insns",
    }
)
_RESULTS_KEYS: frozenset[str] = frozenset(
    {
        "schema", "energy_power_status", "hardware", "source_bindings",
        "unbiased_scalar_ref_bpb_q", "silicon_sm86_bench",
        "utf8_bpb_study", "bpb_summary", "gates", "verdict", "result_sha256",
    }
)
_HARDWARE_KEYS: frozenset[str] = frozenset({"device", "sm", "nvcc_version"})
_SOURCE_BINDINGS_KEYS: frozenset[str] = frozenset(
    {
        "kernel_cu_path", "kernel_cu_sha256", "bench_cu_path", "bench_cu_sha256",
        "bpb_study_script_path", "bpb_study_script_sha256",
        "corpus_id", "corpus_sha256", "corpus_train_bin_sha256", "corpus_eval_bin_sha256",
        "ptxas_receipt_path", "ptxas_receipt_sha256",
    }
)
_SILICON_KEYS: frozenset[str] = frozenset(
    {
        "arm2b_bias_fp64_max_err_q", "arm2b_bias_us_per_tok_q",
        "arm5_bias_fp64_max_err_q", "arm5_bias_us_per_tok_q",
        "arm7_bias_fp64_max_err_q", "arm7_bias_us_per_tok_q",
        "arm9_monarch2_bias_fp64_max_err_q", "arm9_monarch2_bias_us_per_tok_q",
    }
)
_BPB_STUDY_KEYS: frozenset[str] = frozenset(
    {
        "scope_note", "d_model", "d_ff", "n_heads", "n_layers",
        "batch_size", "seq_len", "steps_per_seed", "tokens_per_seed",
        "eval_chunks", "eval_bytes", "seeds", "h256_ortho_fp64_max_err_q", "arms",
    }
)
_BPB_POS_INT_KEYS: Tuple[str, ...] = (
    "d_model", "d_ff", "n_heads", "n_layers",
    "batch_size", "seq_len", "steps_per_seed",
    "eval_chunks", "eval_bytes", "tokens_per_seed",
)
_ARM_KEYS: frozenset[str] = frozenset(
    {"head_params", "head_fp16_bytes", "hbm_vocab_table_bytes", "total_params", "mean_eval_bpb_q", "seeds"}
)
_SEED_KEYS: frozenset[str] = frozenset(
    {"seed", "eval_bpb_q", "eval_bpb_bf16_q", "fp16_quant_bpb_diff_q", "eval_tokens", "train_wall_s_q"}
)
_BPB_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "bias_only_lift_over_ref2b_q",
        "arm5_lift_over_ref2b_q", "arm5_lift_over_2b_bias_q", "arm5_minus_arm8a_q", "arm5_lt_8a_all_seeds",
        "arm7_lift_over_ref2b_q", "arm7_lift_over_2b_bias_q", "arm7_minus_arm8a_q", "arm7_lt_2b_bias_all_seeds", "arm7_lt_8a_all_seeds",
        "arm9_lift_over_ref2b_q", "arm9_lift_over_2b_bias_q", "arm9_minus_arm8a_q", "arm9_lt_2b_bias_all_seeds", "arm9_lt_8a_all_seeds",
        "best_structured_arm", "best_lift_over_ref2b_q", "dense_gap_ref2b_minus_arm4_q",
        "best_gap_closed_fraction_q", "best_minus_arm4_dense_q",
    }
)
_GATES_KEYS: frozenset[str] = frozenset(
    {"G1_bpb_bias_and_lowrank_beat", "G2_sm86_zero_spill_latency_parity"}
)

_EXPECTED_ARMS: Tuple[str, ...] = (
    "arm2b_bias_514B",
    "arm5_bias_2p5KiB",
    "arm7_bias_8p5KiB",
    "arm9_monarch2_bias_16p5KiB",
    "arm8a_lowrank16_bias_16p9KiB",
    "arm4_dense_bias_262KiB",
)
# (head_params, head_fp16_bytes, hbm_vocab_table_bytes)
_EXPECTED_HEAD_SPECS: Dict[str, Tuple[int, int, int]] = {
    "arm2b_bias_514B": (257, 514, 0),
    "arm5_bias_2p5KiB": (1280, 2560, 0),
    "arm7_bias_8p5KiB": (4352, 8704, 0),
    "arm9_monarch2_bias_16p5KiB": (8448, 16896, 0),
    "arm8a_lowrank16_bias_16p9KiB": (8448, 16896, 16896),
    "arm4_dense_bias_262KiB": (131329, 262658, 262658),
}

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_Q_STR_RE = re.compile(r"^-?[0-9]+/[1-9][0-9]*$")
_NVCC_VERSION_RE = re.compile(r"^Cuda compilation tools, release [0-9]+\.[0-9]+, V[0-9]+\.[0-9]+\.[0-9]+$")
_ALNUM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_ALLOWED_AST_IMPORTS: frozenset[str] = frozenset(
    {"argparse", "ast", "errno", "hashlib", "json", "os", "re", "stat", "sys", "unicodedata"}
)
_ALLOWED_MODULE_ATTRS: Dict[str, frozenset[str]] = {
    "argparse": frozenset({"ArgumentParser"}),
    "ast": frozenset(
        {
            "AST", "AnnAssign", "Assert", "Assign", "AsyncFunctionDef", "Attribute",
            "AugAssign", "BinOp", "Call", "ClassDef", "Compare", "Constant", "Delete",
            "Div", "FloorDiv", "For", "FormattedValue", "FunctionDef", "Import",
            "ImportFrom", "Is", "IsNot", "List", "Mod", "Mult", "Name", "NamedExpr",
            "Pow", "Starred", "Tuple", "arg", "parse", "unparse", "walk",
        }
    ),
    "errno": frozenset({"ELOOP"}),
    "hashlib": frozenset({"sha256"}),
    "json": frozenset({"loads", "dumps"}),
    "os": frozenset({"O_NONBLOCK", "O_NOFOLLOW", "O_RDONLY", "O_WRONLY", "close", "devnull", "dup2", "fstat", "open", "read"}),
    "re": frozenset({"Pattern", "compile", "search", "sub"}),
    "stat": frozenset({"S_ISDIR", "S_ISREG"}),
    "sys": frozenset({"stderr", "stdout"}),
    "unicodedata": frozenset({"category", "normalize"}),
}
_ALLOWED_AST_FROM_IMPORTS: Dict[str, Dict[str, Optional[str]]] = {
    "__future__": {"annotations": None},
    "collections.abc": {"Mapping": None, "Sequence": "AbcSequence", "Set": "AbcSet"},
    "fractions": {"Fraction": None},
    "pathlib": {"Path": None},
    "typing": {"Any": None, "Dict": None, "List": None, "Optional": None, "Sequence": None, "Tuple": None},
}
_FORBIDDEN_AST_NAMES: frozenset[str] = frozenset(
    {
        "float", "complex", "pow", "divmod", "format", "literal_eval", "memoryview",
        "__import__", "eval", "exec", "compile",
        "getattr", "setattr", "globals", "locals", "vars", "builtins", "__builtins__",
    }
)
_FORBIDDEN_AST_ATTRS: frozenset[str] = (_FORBIDDEN_AST_NAMES - frozenset({"compile"})) | frozenset(
    {
        "from_float", "from_decimal", "fromhex", "as_integer_ratio", "truediv", "itruediv",
        "cast", "format_map",
        "__truediv__", "__rtruediv__", "__itruediv__",
        "__floordiv__", "__rfloordiv__", "__ifloordiv__", "__float__", "__complex__",
        "__pow__", "__rpow__", "__ipow__",
        "__getattribute__", "__getattr__", "__class__", "__bases__", "real", "imag",
        "modules", "__dict__", "__subclasses__", "__globals__", "__loader__", "__spec__",
    }
)

_FORBIDDEN_ENERGY_KEYS: frozenset[str] = frozenset(
    {
        "pw", "nw", "uw", "mw", "kw", "gw", "tw",
        "wh", "whr", "whrs", "mwh", "kwh",
        "pj", "nj", "uj", "mj", "kj", "gj", "tj",
        "tdp", "pwr", "watt", "watts", "joule", "joules",
        "millijoule", "millijoules", "energy", "power", "edp",
        "rapl", "nvml", "nvidiasmi", "mah",
    }
)
_FORBIDDEN_COMPACT_KEY_RE = re.compile(
    r"(?:(?:tera|giga|mega|kilo|milli|micro|nano|pico)?(?:watt|joule)"
    r"|(?:^|(?<!em))power(?!ment)|energy|mjpertok|mjtok|jpertok|ujpertok|kwh|mwh|whpertok|tdp|rapl|nvml|nvidiasmi)"
)
_ASCII_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_ENERGY_PFX = r"(?:p|n|u|\u03bc|m|k|g|t|pico|nano|micro|milli|kilo|mega|giga|tera)"
_ENERGY_UNIT = r"(?:watt[^a-z0-9]{0,4}hours?|watthours?|watts?|joules?|whrs?|wh|mah|w|j|tdp)"
_ENERGY_UNIT_NON_J = r"(?:watt[^a-z0-9]{0,4}hours?|watthours?|watts?|joules?|whrs?|wh|mah|w|tdp)"
_ENERGY_SEP = r"[^a-z0-9]{0,8}"
# Anchored possessive quantifiers (\d++, Python >= 3.11) guarantee bounded-backtracking scan (<5 ms on 8k-char runs)
_ENERGY_NUM = r"(?<![0-9])(?<!\d[.,])\d++(?:[.,]\d++){0,3}(?:e[+-]?\d++)?"
_SPELLED_TENS = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_SPELLED_ONES = r"(?:one|two|three|four|five|six|seven|eight|nine)"
_SPELLED_WORD = (
    rf"(?:{_SPELLED_TENS}(?:[\s-]{{0,2}}{_SPELLED_ONES})?|"
    rf"zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    rf"{_SPELLED_ONES}|hundred|thousand|million|billion|half|quarter)"
)
_FORBIDDEN_ENERGY_VALUE_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"{_ENERGY_NUM}(?:"
        rf"{_ENERGY_SEP}(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}"
        r"(?![a-z0-9])(?!/\s)(?!/\d)(?!/o(?![a-z0-9]))"
        r"|\s{0,4}w\s{0,4}/\s{0,4}tok(?:en)?s?(?![a-z0-9])"
        rf"|\s{{0,3}}(?:m|u|\u03bc|k)?(?:amps?|amperes?|a)(?![a-z0-9]).{{0,24}}?{_ENERGY_NUM}\s{{0,3}}(?:m|u|\u03bc|k)?(?:volts?|v)(?![a-z0-9])"
        rf"|\s{{0,3}}(?:m|u|\u03bc|k)?(?:volts?|v)(?![a-z0-9]).{{0,24}}?{_ENERGY_NUM}\s{{0,3}}(?:m|u|\u03bc|k)?(?:amps?|amperes?|a)(?![a-z0-9])"
        ")"
    ),
    re.compile(
        "(?:"
        rf"(?<![a-z0-9])(?:{_ENERGY_PFX}{_ENERGY_SEP}{_ENERGY_UNIT}|{_ENERGY_UNIT_NON_J})\s{{0,4}}(?:[:=]\s{{0,4}}|\s{{1,4}}){_ENERGY_NUM}"
        rf"|(?<![a-z0-9])j(?:\s{{0,4}}:\s{{0,4}}|=){_ENERGY_NUM}"
        rf"|(?<![a-z0-9])tdp[^a-z0-9]{{0,4}}{_ENERGY_NUM}"
        rf"|\A[^a-z0-9]{{0,4}}(?:{_ENERGY_PFX}{_ENERGY_SEP})?{_ENERGY_UNIT}[^a-z0-9]{{0,4}}\Z"
        rf"|(?<![a-z0-9]){_SPELLED_WORD}(?:[\s-]{{1,3}}{_SPELLED_WORD}){{0,3}}[\s-]{{1,3}}(?:(?:pico|nano|micro|milli|kilo|mega|giga|tera)[\s-]?)?(?:watt[\s-]?hours?|watthours?|watts?|joules?)(?![a-z0-9])"
        rf"|(?<![a-z0-9])(?:energy|power|wattage)\s{{0,4}}(?:[:=]\s{{0,4}}|\s{{1,4}}){_ENERGY_NUM}"
        r"|(?<![a-z0-9])(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:joules?|watts?|j|wh)|(?:p|n|u|\u03bc|m|k|g|t)w|kwh|mwh|energy|power)[\s_-]{0,4}(?:per|/)[\s_-]{0,4}tok(?:en)?s?(?![a-z0-9])"
        r"|(?<![a-z0-9])w(?:[\s_-]{1,4}per[\s_-]{1,4}|/)tok(?:en)?s?(?![a-z0-9])"
        r"|(?<![a-z0-9])tok(?:en)?s?[\s_-]{0,4}(?:per|/)[\s_-]{0,4}(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:j|w|joules?|watts?)|wh|kwh|mwh)(?![a-z0-9])"
        r"|(?<![a-z0-9])per[\s_-]{1,4}(?:(?:p|n|u|\u03bc|m|k|g|t)?(?:watts?|joules?)|w|j)(?![a-z0-9])"
        r"|(?:rapl|nvml|nvidia[\s_-]{0,4}smi|(?<![a-z0-9])mah(?![a-z0-9]))"
        r"|(?<![a-z0-9])(?:less|lower|reduced|fewer)[\s_-]{1,4}(?:in[\s_-]{1,4})?(?:energy|power)(?![a-z0-9])"
        r"|(?<![a-z0-9])(?:energy|power)[\s_-]{1,4}(?:use|usage|reduction|savings?|dropped|consumption)(?![a-z0-9])"
        r"|(?:%|(?<![a-z0-9])percent)[\s_-]{1,4}(?:of[\s_-]{1,4})?(?:(?:stock|baseline|nominal|peak|idle|full|total|gpu|cpu)[\s_-]{1,4})?(?:energy|power)(?![a-z0-9])"
        ")"
    ),
)


class Monarch2HeadVerifyError(ValueError):
    """Fail-closed typed error for sm_86 Monarch-2 + byte-bias FWHT head verification."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise Monarch2HeadVerifyError(code, message)


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
    raise Monarch2HeadVerifyError("NON_FINITE_JSON", f"non-finite constant: {token!r}")


def _reject_json_float(token: str) -> Any:
    raise Monarch2HeadVerifyError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {token!r}")


def _parse_strict_json_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "INT_DIGIT_LIMIT", f"digits={len(digits)}")
    return int(token)


def _open_nofollow_fd(path_str: str, *, name: str, dir_fd: Optional[int] = None) -> int:
    try:
        return os.open(path_str, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except FileNotFoundError as exc:
        raise Monarch2HeadVerifyError("MISSING_FILE", name) from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise Monarch2HeadVerifyError("SYMLINK_FORBIDDEN", name) from exc
        raise Monarch2HeadVerifyError("READ_OR_PARSE_ERROR", f"{name}: {type(exc).__name__}") from exc


def _read_fd_regular_bytes(fd: int, *, name: str, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    try:
        st = os.fstat(fd)
        require_ok(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE", name)
        chunks: List[bytes] = []
        total = 0
        while True:
            blk = os.read(fd, 65536)
            if not blk:
                break
            total += len(blk)
            require_ok(total <= max_bytes, "FILE_TOO_LARGE", name)
            chunks.append(blk)
        return b"".join(chunks)
    except Monarch2HeadVerifyError:
        raise
    except OSError as exc:
        raise Monarch2HeadVerifyError("READ_OR_PARSE_ERROR", f"{name}: {type(exc).__name__}") from exc
    finally:
        os.close(fd)


def _read_abs_path_nofollow(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    """TOCTOU-free kernel dir_fd O_NOFOLLOW walk from filesystem root down to leaf regular file."""
    parts = path.absolute().parts
    require_ok(len(parts) >= 2, "ROOT_PATH_FORBIDDEN", str(path))
    dfd = _open_nofollow_fd(parts[0], name=parts[0])
    try:
        for part in parts[1:-1]:
            require_ok(stat.S_ISDIR(os.fstat(dfd).st_mode), "NOT_DIRECTORY", part)
            next_dfd = _open_nofollow_fd(part, name=part, dir_fd=dfd)
            os.close(dfd)
            dfd = next_dfd
        require_ok(stat.S_ISDIR(os.fstat(dfd).st_mode), "NOT_DIRECTORY", parts[-1])
        leaf_fd = _open_nofollow_fd(parts[-1], name=parts[-1], dir_fd=dfd)
    finally:
        os.close(dfd)
    return _read_fd_regular_bytes(leaf_fd, name=parts[-1], max_bytes=max_bytes)


def _read_regular_bytes_nofollow(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    return _read_abs_path_nofollow(path, max_bytes=max_bytes)


def _read_rel_bytes_nofollow(repo_root: Path, rel: str, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    return _read_abs_path_nofollow(repo_root.joinpath(rel), max_bytes=max_bytes)


def _check_raw_json_nesting_depth(text: str, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Enforce bounded container nesting depth prior to json.loads via string-aware bracket/brace scan."""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in ("{", "["):
            depth += 1
            require_ok(depth <= max_depth, "JSON_DEPTH_LIMIT", f"depth={depth}")
        elif ch in ("}", "]"):
            if depth > 0:
                depth -= 1


def _check_json_depth(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    stack: List[Tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        require_ok(depth <= max_depth, "JSON_DEPTH_LIMIT", f"depth={depth}")
        if isinstance(cur, dict):
            for v in cur.values():
                stack.append((v, depth + 1))
        elif isinstance(cur, list):
            for item in cur:
                stack.append((item, depth + 1))


def parse_strict_canonical_json_bytes(raw_bytes: bytes, *, name: str) -> Any:
    try:
        text = raw_bytes.decode("ascii")
        _check_raw_json_nesting_depth(text)
        parsed = json.loads(
            text,
            object_pairs_hook=_no_dup_object_pairs,
            parse_constant=_reject_non_finite,
            parse_int=_parse_strict_json_int,
            parse_float=_reject_json_float,
        )
        _check_json_depth(parsed)
        expected_raw = (
            json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("ascii")
    except Monarch2HeadVerifyError:
        raise
    except RecursionError as exc:
        raise Monarch2HeadVerifyError("JSON_DEPTH_LIMIT", f"{name}: RecursionError") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise Monarch2HeadVerifyError("READ_OR_PARSE_ERROR", f"{name}: {type(exc).__name__}") from exc
    require_ok(raw_bytes == expected_raw, "RAW_JSON_NOT_CANONICAL", f"non-canonical bytes in {name}")
    return parsed


def load_strict_canonical_json(path: Path) -> Any:
    return parse_strict_canonical_json_bytes(_read_regular_bytes_nofollow(path), name=path.name)


def _is_safe_int_ast_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _is_safe_int_ast_expr(node.left) and _is_safe_int_ast_expr(node.right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and not node.keywords
    ):
        return True
    return isinstance(node, ast.Attribute) and node.attr in ("numerator", "denominator")


def verify_pure_q_ast(source_text: str) -> None:
    """Enforce closed-allowlist zero-float, zero-Div/FloorDiv/Mod/Pow, zero-assert, and zero-escape AST discipline."""
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Monarch2HeadVerifyError("AST_SYNTAX_ERROR", type(exc).__name__) from exc

    allowed_fraction_name_ids: set[int] = set()
    allowed_mod_name_ids: set[int] = set()
    allowed_type_name_ids: set[int] = set()
    allowed_ast_parse_call_ids: set[int] = set()
    annotation_node_ids: set[int] = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name in ("verify_pure_q_ast", "verify_cuda_scale_and_study_bindings"):
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Attribute)
                    and isinstance(sub.value.func.value, ast.Name)
                    and sub.value.func.value.id == "ast"
                    and sub.value.func.attr == "parse"
                    and len(sub.value.args) == 1
                    and isinstance(sub.value.args[0], ast.Name)
                    and sub.value.args[0].id in ("source_text", "bpb_script_src")
                    and not sub.value.keywords
                ):
                    allowed_ast_parse_call_ids.add(id(sub.value))
                    allowed_mod_name_ids.add(id(sub.value.func.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Fraction":
            allowed_fraction_name_ids.add(id(node.func))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
            and node.func.attr in ("loads", "dumps")
        ):
            allowed_mod_name_ids.add(id(node.func.value))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in _ALLOWED_MODULE_ATTRS
            and node.value.id != "json"
            and node.attr in _ALLOWED_MODULE_ATTRS[node.value.id]
            and not (node.value.id == "ast" and node.attr == "parse")
        ):
            allowed_mod_name_ids.add(id(node.value))
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Is, ast.IsNot))
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "type"
        ):
            allowed_type_name_ids.add(id(node.left.func))
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ("__name__", "__mro__")
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "type"
        ):
            allowed_type_name_ids.add(id(node.value.func))
        if isinstance(node, ast.arg) and node.annotation is not None:
            for sub in ast.walk(node.annotation):
                annotation_node_ids.add(id(sub))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            for sub in ast.walk(node.returns):
                annotation_node_ids.add(id(sub))
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            for sub in ast.walk(node.annotation):
                annotation_node_ids.add(id(sub))

    for node in ast.walk(tree):
        require_ok(not isinstance(node, ast.Div), "AST_DIV_FORBIDDEN", "ast.Div forbidden")
        require_ok(not isinstance(node, ast.FloorDiv), "AST_FLOORDIV_FORBIDDEN", "ast.FloorDiv forbidden")
        require_ok(not isinstance(node, ast.Mod), "AST_MOD_FORBIDDEN", "ast.Mod forbidden")
        require_ok(not isinstance(node, ast.Pow), "AST_POW_FORBIDDEN", "ast.Pow forbidden")
        require_ok(not isinstance(node, ast.Assert), "AST_BARE_ASSERT", "ast.Assert forbidden")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            require_ok(node.name != "Fraction", "AST_FRACTION_REBIND", "def/class Fraction forbidden")
        if isinstance(node, ast.FormattedValue):
            require_ok(node.format_spec is None, "AST_FORMAT_SPEC_FORBIDDEN", "f-string format_spec forbidden")
        if isinstance(node, ast.Constant):
            tname = type(node.value).__name__
            require_ok(tname not in ("float", "complex"), "AST_FLOAT_LITERAL", f"literal {tname}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                require_ok(
                    alias.name in _ALLOWED_AST_IMPORTS and alias.asname is None,
                    "AST_FORBIDDEN_IMPORT",
                    alias.name,
                )
        if isinstance(node, ast.ImportFrom):
            require_ok(node.level == 0 and node.module is not None, "AST_RELATIVE_IMPORT", "relative import forbidden")
            allowed_names = _ALLOWED_AST_FROM_IMPORTS.get(node.module)
            require_ok(allowed_names is not None, "AST_FORBIDDEN_IMPORT", node.module)
            for alias in node.names:
                require_ok(alias.name != "*", "AST_WILDCARD_IMPORT", "wildcard import forbidden")
                require_ok(
                    alias.name in allowed_names and alias.asname == allowed_names[alias.name],
                    "AST_FORBIDDEN_IMPORT",
                    f"{node.module}.{alias.name}",
                )
        if isinstance(node, ast.Name):
            require_ok(node.id not in _FORBIDDEN_AST_NAMES, "AST_FORBIDDEN_NAME", node.id)
            if node.id == "Fraction":
                require_ok(
                    id(node) in allowed_fraction_name_ids or id(node) in annotation_node_ids,
                    "AST_FRACTION_REBIND",
                    "Fraction may only be called directly or used in type annotations",
                )
            if node.id == "type":
                require_ok(
                    id(node) in allowed_type_name_ids,
                    "AST_TYPE_ESCAPE",
                    "type() only permitted in `type(x) is ...` or `type(x).__name__/__mro__`",
                )
            if node.id in _ALLOWED_AST_IMPORTS:
                require_ok(
                    id(node) in allowed_mod_name_ids,
                    "AST_FORBIDDEN_MOD_ATTR",
                    f"disallowed module ref or attribute on {node.id}",
                )
        if isinstance(node, ast.Attribute):
            require_ok(node.attr not in _FORBIDDEN_AST_ATTRS, "AST_FORBIDDEN_ATTR", node.attr)
            require_ok(node.attr != "Fraction", "AST_FRACTION_REBIND", "attribute Fraction forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Fraction":
                require_ok(
                    len(node.args) == 2
                    and not node.keywords
                    and not any(isinstance(arg, ast.Starred) for arg in node.args),
                    "AST_FRACTION_ARGS",
                    "Fraction requires 2 non-starred positional int args",
                )
                for arg in node.args:
                    require_ok(
                        _is_safe_int_ast_expr(arg),
                        "AST_FRACTION_ARG_TYPE",
                        "Fraction args must be integer expressions",
                    )
            if (isinstance(node.func, ast.Attribute) and node.func.attr in ("loads", "load")) or (
                isinstance(node.func, ast.Name) and node.func.id in ("loads", "load")
            ):
                kw_map = {kw.arg: kw.value for kw in node.keywords}
                pf = kw_map.get("parse_float")
                pc = kw_map.get("parse_constant")
                pi = kw_map.get("parse_int")
                require_ok(
                    isinstance(pf, ast.Name)
                    and pf.id == "_reject_json_float"
                    and isinstance(pc, ast.Name)
                    and pc.id == "_reject_non_finite"
                    and isinstance(pi, ast.Name)
                    and pi.id == "_parse_strict_json_int",
                    "AST_JSON_FLOAT_PARSE",
                    "json.loads requires strict parse_float/parse_constant/parse_int guards",
                )


def canonical_json_bytes(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Monarch2HeadVerifyError("BAD_JSON", str(exc)) from exc


def canonical_sha256(obj: Any, *, exclude_key: Optional[str] = None) -> str:
    payload = obj if exclude_key is None else {k: v for k, v in obj.items() if k != exclude_key}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes_nofollow(path)).hexdigest()


def format_q(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def parse_q(raw: Any, *, code: str = "BAD_RATIONAL") -> Fraction:
    require_ok(type(raw) is str and bool(_Q_STR_RE.match(raw)), code, f"invalid rational: {raw!r}")
    num_s, den_s = raw.split("/", 1)
    num_digits = num_s[1:] if num_s.startswith("-") else num_s
    require_ok(
        len(num_digits) <= MAX_JSON_INT_DIGITS and len(den_s) <= MAX_JSON_INT_DIGITS,
        code,
        "rational exceeds digit limit",
    )
    q = Fraction(int(num_s), int(den_s))
    require_ok(format_q(q) == raw, code, f"non-canonical rational: {raw!r}")
    return q


def q_div(a: Fraction, b: Fraction) -> Fraction:
    require_ok(b != Fraction(0, 1), "ZERO_DIV", "division by zero")
    return Fraction(a.numerator * b.denominator, a.denominator * b.numerator)


def q_div_int(q: Fraction, n: int) -> Fraction:
    require_ok(is_strict_int(n) and n > 0, "BAD_DIVISOR", f"n={n!r}")
    return Fraction(q.numerator, q.denominator * int(n))


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
    if raw.isascii():
        return _ASCII_CTRL_RE.sub("", raw)
    nfkd = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", raw))
    return "".join(ch for ch in nfkd if unicodedata.category(ch) not in ("Cf", "Mn", "Cc", "Cs", "Co"))


def _has_non_ascii_letter(norm: str) -> bool:
    return any(not ch.isascii() and ch.isalpha() and ch != "\u03bc" for ch in norm)


def is_forbidden_energy_value(raw: str) -> bool:
    norm = _strip_nfkd_marks(raw).casefold()
    return _has_non_ascii_letter(norm) or any(rx.search(norm) is not None for rx in _FORBIDDEN_ENERGY_VALUE_RES)


def is_forbidden_energy_key_name(raw: str) -> bool:
    cleaned = _strip_nfkd_marks(raw)
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", cleaned).replace("_", " ")
    if is_forbidden_energy_value(cleaned) or is_forbidden_energy_value(spaced):
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
# Pure-Z Exact Algebraic Oracles (H_256, FWHT_256 x bent4, 1-Stage vs 2-Stage Monarch)
# ============================================================================

def fwht256_plain_int(u256: Sequence[int]) -> List[int]:
    require_ok(len(u256) == 256, "BAD_FWHT_LEN", f"len={len(u256)}")
    buf = list(u256)
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


def fwht256_bent4_int(u256: Sequence[int]) -> List[int]:
    """Unscaled integer FWHT_256 x bent4 (CUDA kernel scales output by 1/sqrt(32), giving sum y^2 = 8 * sum u^2)."""
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
    return fwht256_plain_int(buf)


def monarch16_stage_allones_int(z_in: Sequence[int]) -> List[int]:
    """Exact pure-Z Monarch-16x16 stage B @ P_{16x16} @ z with all-ones 16x16 blocks."""
    require_ok(len(z_in) == 256, "BAD_MONARCH_LEN", f"len={len(z_in)}")
    z_out = [0] * 256
    for g in range(16):
        col_sum = sum(z_in[c * 16 + g] for c in range(16))
        for r in range(16):
            z_out[g * 16 + r] = col_sum
    return z_out


def _strip_c_comments_and_strings(src: str) -> str:
    joined = _C_LINE_SPLICE_RE.sub("", src)
    return _C_STRIP_RE.sub(" ", joined)


def _extract_c_function_body(code: str, func_name: str) -> Optional[str]:
    """Extract balanced-brace body of a C/CUDA function definition (zero cross-function leakage)."""
    m = re.search(rf"\b{func_name}\s*\([^;{{}}]*\)\s*\{{", code)
    if m is None:
        return None
    start = m.end() - 1
    depth = 0
    for idx in range(start, len(code)):
        ch = code[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code[start : idx + 1]
    return None


def _iter_assigned_names(target: ast.AST) -> List[str]:
    names: List[str] = []
    stack: List[ast.AST] = [target]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            names.append(cur.id)
        elif isinstance(cur, (ast.Tuple, ast.List)):
            stack.extend(cur.elts)
        elif isinstance(cur, ast.Starred):
            stack.append(cur.value)
    return names


def verify_cuda_scale_and_study_bindings(
    kernel_src: str, bench_src: str, bpb_script_src: str
) -> Tuple[Dict[str, Any], Tuple[str, str, str, str]]:
    """Verify stripped CUDA/bench preprocessor-free function scales, 4 __global__ defs, & BPB study AST."""
    k_code = _strip_c_comments_and_strings(kernel_src)
    b_code = _strip_c_comments_and_strings(bench_src)
    require_ok(_C_PREPROC_COND_RE.search(k_code) is None, "CU_DISABLED_CODE_FORBIDDEN", "kernel contains #if/#ifdef")
    require_ok(_C_PREPROC_COND_RE.search(b_code) is None, "BENCH_DISABLED_CODE_FORBIDDEN", "bench contains #if/#ifdef")
    for fn_name, rx in _KERNEL_FN_SCALE_SPECS:
        body = _extract_c_function_body(k_code, fn_name)
        require_ok(body is not None and rx.search(body) is not None, "CU_SCALE_TOKEN_MISSING", f"kernel missing {fn_name}")
    for kname in _KERNEL_DEF_NAMES:
        require_ok(
            re.search(rf"__global__[^{{;]*\bvoid\s+{kname}\s*\([^)]*\)\s*\{{", k_code) is not None,
            "CU_KERNEL_DEF_MISSING",
            f"kernel missing __global__ def {kname}",
        )
    for fn_name, rx in _BENCH_FN_SCALE_SPECS:
        body = _extract_c_function_body(b_code, fn_name)
        require_ok(body is not None and rx.search(body) is not None, "BENCH_SCALE_TOKEN_MISSING", f"bench missing {fn_name}")

    try:
        study_tree = ast.parse(bpb_script_src)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Monarch2HeadVerifyError("BPB_SCRIPT_SYNTAX_ERROR", type(exc).__name__) from exc

    pin_values: Dict[str, str] = {}
    top_level_pin_assign_ids: set[int] = set()
    run_bpb_fn: Optional[ast.FunctionDef] = None
    for stmt in study_tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            tgt = stmt.targets[0].id
            if tgt in _BPB_CORPUS_PIN_NAMES:
                require_ok(
                    tgt not in pin_values and isinstance(stmt.value, ast.Constant) and type(stmt.value.value) is str,
                    "BPB_SCRIPT_PIN_REBIND",
                    f"invalid or duplicate top-level pin {tgt}",
                )
                pin_values[tgt] = stmt.value.value
                top_level_pin_assign_ids.add(id(stmt))
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "run_bpb_study":
            run_bpb_fn = stmt

    run_bpb_def_count = 0
    for node in ast.walk(study_tree):
        require_ok(not isinstance(node, ast.Assert), "BPB_SCRIPT_BARE_ASSERT", "bpb script contains bare assert")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_bpb_study":
            run_bpb_def_count += 1
        if isinstance(node, ast.Name):
            require_ok(
                node.id not in _FORBIDDEN_BPB_SCRIPT_NAMES,
                "BPB_SCRIPT_FORBIDDEN_NAME",
                node.id,
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                require_ok(
                    alias.name in _ALLOWED_BPB_IMPORTS and alias.asname == _ALLOWED_BPB_IMPORTS[alias.name],
                    "BPB_SCRIPT_FORBIDDEN_IMPORT",
                    alias.name,
                )
        if isinstance(node, ast.ImportFrom):
            allowed_from = _ALLOWED_BPB_FROM_IMPORTS.get(node.module or "")
            require_ok(node.level == 0 and allowed_from is not None, "BPB_SCRIPT_FORBIDDEN_IMPORT", str(node.module))
            for alias in node.names:
                require_ok(
                    alias.name in allowed_from and alias.asname is None,
                    "BPB_SCRIPT_FORBIDDEN_IMPORT",
                    f"{node.module}.{alias.name}",
                )
        if isinstance(node, ast.Assign) and id(node) not in top_level_pin_assign_ids:
            for t in node.targets:
                for nm in _iter_assigned_names(t):
                    require_ok(nm not in _BPB_CORPUS_PIN_NAMES, "BPB_SCRIPT_PIN_REBIND", nm)
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For)):
            for nm in _iter_assigned_names(node.target):
                require_ok(nm not in _BPB_CORPUS_PIN_NAMES, "BPB_SCRIPT_PIN_REBIND", nm)
        if isinstance(node, ast.Delete):
            for t in node.targets:
                for nm in _iter_assigned_names(t):
                    require_ok(nm not in _BPB_CORPUS_PIN_NAMES, "BPB_SCRIPT_PIN_REBIND", nm)

    require_ok(
        run_bpb_fn is not None and run_bpb_def_count == 1,
        "BPB_SCRIPT_RUN_DEF_INVALID",
        f"expected 1 top-level run_bpb_study def, got {run_bpb_def_count}",
    )

    h256_valid_ids: set[int] = set()
    inv32_valid_ids: set[int] = set()
    for fn_stmt in run_bpb_fn.body:
        if isinstance(fn_stmt, ast.Assign) and len(fn_stmt.targets) == 1 and isinstance(fn_stmt.targets[0], ast.Name):
            fn_tgt = fn_stmt.targets[0].id
            if fn_tgt == "h256_fp64" and any(
                isinstance(sub, ast.Constant) and repr(sub.value) == "0.0625" for sub in ast.walk(fn_stmt.value)
            ):
                h256_valid_ids.add(id(fn_stmt))
            if fn_tgt == "inv_sqrt32" and ast.unparse(fn_stmt.value) == "1.0 / math.sqrt(32.0)":
                inv32_valid_ids.add(id(fn_stmt))

    h256_all_assigns = 0
    inv32_all_assigns = 0
    for sub in ast.walk(run_bpb_fn):
        bound_names: List[str] = []
        if isinstance(sub, ast.Assign):
            for t in sub.targets:
                bound_names.extend(_iter_assigned_names(t))
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For)):
            bound_names.extend(_iter_assigned_names(sub.target))
        elif isinstance(sub, ast.Delete):
            for t in sub.targets:
                bound_names.extend(_iter_assigned_names(t))
        for nm in bound_names:
            if nm == "h256_fp64":
                h256_all_assigns += 1
            if nm == "inv_sqrt32":
                inv32_all_assigns += 1

    require_ok(
        len(h256_valid_ids) == 1
        and h256_all_assigns == 1
        and len(inv32_valid_ids) == 1
        and inv32_all_assigns == 1,
        "BPB_SCRIPT_SCALE_TOKEN_MISSING",
        "bpb script must bind h256_fp64 and inv_sqrt32 exactly once directly in run_bpb_study",
    )
    require_ok(
        all(k in pin_values for k in _BPB_CORPUS_PIN_NAMES),
        "BPB_SCRIPT_CORPUS_PINS_MISSING",
        "bpb script missing pinned corpus constants",
    )
    corpus_pins = (
        pin_values["CORPUS_ID"],
        pin_values["EXPECTED_CORPUS_MANIFEST_SHA256"],
        pin_values["EXPECTED_TRAIN_BIN_SHA256"],
        pin_values["EXPECTED_EVAL_BIN_SHA256"],
    )
    return (
        {
            "plain_fwht_scale_q": "1/16",
            "bent4_fwht_scale_sq_q": "1/32",
            "bent4_norm_sq_multiplier": 8,
            "scale_tokens_verified": True,
            "bpb_script_ast_verified": True,
        },
        corpus_pins,
    )


def verify_cuda_scale_tokens(kernel_src: str, bench_src: str, bpb_script_src: str) -> Dict[str, Any]:
    """Verify contextual scaling literals & study-script AST across kernel, bench, and BPB script."""
    return verify_cuda_scale_and_study_bindings(kernel_src, bench_src, bpb_script_src)[0]


def verify_monarch2_and_fwht_oracle(
    *,
    plain_fn: Any = fwht256_plain_int,
    bent4_fn: Any = fwht256_bent4_int,
    stage_fn: Any = monarch16_stage_allones_int,
) -> Dict[str, Any]:
    """Certify H_256 involution, FWHT_256 x bent4 Parseval + 6 delta KAT columns, and Monarch reach."""
    u: List[int] = []
    rem = 3
    for _ in range(256):
        while rem >= 11:
            rem -= 11
        u.append(rem - 5)
        rem += 19
    h1 = plain_fn(u)
    h2 = plain_fn(h1)
    require_ok(h2 == [256 * x for x in u], "ORACLE_H256_INVOLUTION", "H_256^2 != 256 I")

    y = bent4_fn(u)
    energy_u = sum(x * x for x in u)
    energy_y = sum(v * v for v in y)
    require_ok(energy_y == 256 * energy_u, "ORACLE_PARSEVAL_FAIL", "Parseval mismatch")

    # Known-answer delta -> signed-Hadamard multi-vector tests (blocks sign-drop, reversal, rotation, negation, warp-XOR drop)
    for src_pos in _KAT_DELTA_INDICES:
        delta = [1 if i == src_pos else 0 for i in range(256)]
        got_col = bent4_fn(delta)
        lane = src_pos & 31
        db = src_pos ^ ((lane & 7) << 5)
        b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
        exp_col = [-1 if (((tok & db).bit_count() ^ b4) & 1) else 1 for tok in range(256)]
        require_ok(got_col == exp_col, "ORACLE_BENT4_KAT_FAIL", f"delta s={src_pos} mismatch")

    e0 = [1 if i == 0 else 0 for i in range(256)]
    z1 = stage_fn(e0)
    stage1_reach = sum(1 for v in z1 if v != 0)
    z2 = stage_fn(z1)
    stage2_reach = sum(1 for v in z2 if v != 0)
    require_ok(stage1_reach == 16 and stage2_reach == 256, "ORACLE_MONARCH2_REACH", f"reach=({stage1_reach},{stage2_reach})")
    return {
        "h256_involution_verified": True,
        "bent4_parseval_input_l2_sq": energy_u,
        "bent4_parseval_output_l2_sq": energy_y,
        "bent4_scaled_norm_sq_multiplier": 8,
        "bent4_kat_delta_vectors_verified": len(_KAT_DELTA_INDICES),
        "monarch1_reach_count": stage1_reach,
        "monarch2_full_reach_count": stage2_reach,
        "monarch2_bias_param_count": 8448,
        "monarch2_bias_fp16_bytes": 16896,
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
    require_ok(
        type(ptx["nvcc_version"]) is str and bool(_NVCC_VERSION_RE.match(ptx["nvcc_version"])),
        "PTXAS_NVCC",
        "bad nvcc_version",
    )
    require_ok(ptx["cu_path"] == KERNEL_CU_REL, "PTXAS_CU_PATH", "unexpected cu_path")
    cu_sha = ptx["cu_sha256"]
    require_ok(is_hex64(cu_sha), "PTXAS_CU_SHA", "bad cu_sha256")
    if expected_cu_sha is not None:
        require_ok(cu_sha == expected_cu_sha, "PTXAS_CU_SHA_DISK", "cu_sha256 mismatch vs disk")

    kernels = ptx["kernels"]
    require_ok(type(kernels) is dict and len(kernels) == 4, "PTXAS_KERNELS", "expected 4 kernels")
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
        exp_regs, exp_bar, exp_smem = _EXPECTED_KERNEL_SPECS[name]
        require_ok(
            st["registers"] == exp_regs and st["barriers"] == exp_bar and st["smem_bytes"] == exp_smem,
            "PTXAS_KERNEL_SPEC",
            f"unexpected ptxas spec for {name}",
        )
    require_ok(
        ptx["total_kernels"] == 4
        and ptx["zero_spill_kernels"] == 4
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
    expected_bpb_script_sha: Optional[str] = None,
    expected_corpus_pins: Optional[Tuple[str, str, str, str]] = None,
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
        and hw["nvcc_version"] == ptx["nvcc_version"],
        "RESULTS_HARDWARE",
        "invalid hardware block",
    )

    sb = res["source_bindings"]
    require_ok(type(sb) is dict and frozenset(sb.keys()) == _SOURCE_BINDINGS_KEYS, "RESULTS_BINDINGS", "bad source_bindings")
    require_ok(
        sb["kernel_cu_path"] == KERNEL_CU_REL
        and sb["bench_cu_path"] == BENCH_CU_REL
        and sb["bpb_study_script_path"] == BPB_STUDY_SCRIPT_REL
        and sb["ptxas_receipt_path"] == PTXAS_RECEIPT_REL,
        "RESULTS_BINDING_PATHS",
        "unexpected source binding paths",
    )
    require_ok(sb["kernel_cu_sha256"] == ptx["cu_sha256"], "RESULTS_CU_BIND", "kernel cu sha mismatch")
    require_ok(sb["ptxas_receipt_sha256"] == ptx["receipt_sha256"], "RESULTS_PTX_BIND", "ptxas sha mismatch")
    require_ok(is_hex64(sb["bench_cu_sha256"]), "RESULTS_BENCH_SHA", "bad bench_cu_sha256")
    if expected_bench_cu_sha is not None:
        require_ok(sb["bench_cu_sha256"] == expected_bench_cu_sha, "RESULTS_BENCH_DISK", "bench cu disk mismatch")
    require_ok(is_hex64(sb["bpb_study_script_sha256"]), "RESULTS_BPB_SCRIPT_SHA", "bad bpb_study_script_sha256")
    if expected_bpb_script_sha is not None:
        require_ok(
            sb["bpb_study_script_sha256"] == expected_bpb_script_sha,
            "RESULTS_BPB_SCRIPT_DISK",
            "bpb study script disk mismatch",
        )
    require_ok(sb["corpus_id"] == EXPECTED_CORPUS_ID, "RESULTS_CORPUS_ID", "unexpected corpus_id")
    require_ok(
        is_hex64(sb["corpus_sha256"])
        and is_hex64(sb["corpus_train_bin_sha256"])
        and is_hex64(sb["corpus_eval_bin_sha256"]),
        "RESULTS_CORPUS_SHA",
        "bad corpus sha256 bindings",
    )
    if expected_corpus_pins is not None:
        require_ok(
            (
                sb["corpus_id"],
                sb["corpus_sha256"],
                sb["corpus_train_bin_sha256"],
                sb["corpus_eval_bin_sha256"],
            )
            == expected_corpus_pins,
            "RESULTS_CORPUS_PIN_MISMATCH",
            "corpus pins mismatch vs bpb study script",
        )

    ref2b = parse_q(res["unbiased_scalar_ref_bpb_q"], code="RESULTS_REF2B")
    require_ok(ref2b > Fraction(0, 1), "RESULTS_REF2B_POS", "ref2b must be > 0")

    sil = res["silicon_sm86_bench"]
    require_ok(type(sil) is dict and frozenset(sil.keys()) == _SILICON_KEYS, "RESULTS_SILICON", "bad silicon_sm86_bench")
    err_2b = parse_q(sil["arm2b_bias_fp64_max_err_q"], code="RESULTS_SIL_Q")
    err_5 = parse_q(sil["arm5_bias_fp64_max_err_q"], code="RESULTS_SIL_Q")
    err_7 = parse_q(sil["arm7_bias_fp64_max_err_q"], code="RESULTS_SIL_Q")
    err_9 = parse_q(sil["arm9_monarch2_bias_fp64_max_err_q"], code="RESULTS_SIL_Q")
    us_2b = parse_q(sil["arm2b_bias_us_per_tok_q"], code="RESULTS_SIL_Q")
    us_5 = parse_q(sil["arm5_bias_us_per_tok_q"], code="RESULTS_SIL_Q")
    us_7 = parse_q(sil["arm7_bias_us_per_tok_q"], code="RESULTS_SIL_Q")
    us_9 = parse_q(sil["arm9_monarch2_bias_us_per_tok_q"], code="RESULTS_SIL_Q")
    for err_v in (err_2b, err_5, err_7, err_9):
        require_ok(Fraction(0, 1) <= err_v < Fraction(1, 100000), "RESULTS_PARITY_BOUND", f"err={err_v}")
    for us_v in (us_2b, us_5, us_7, us_9):
        require_ok(us_v > Fraction(0, 1), "RESULTS_US_POS", f"us={us_v}")

    bpb = res["utf8_bpb_study"]
    require_ok(type(bpb) is dict and frozenset(bpb.keys()) == _BPB_STUDY_KEYS, "RESULTS_BPB", "bad utf8_bpb_study")
    expected_seeds = bpb["seeds"]
    require_ok(
        type(expected_seeds) is list
        and expected_seeds == [42, 137, 2026]
        and all(is_strict_int(s) for s in expected_seeds)
        and all(is_strict_int(bpb[k]) and bpb[k] > 0 for k in _BPB_POS_INT_KEYS)
        and bpb["d_model"] == 512
        and bpb["eval_bytes"] == bpb["eval_chunks"] * bpb["seq_len"]
        and bpb["tokens_per_seed"] == bpb["batch_size"] * bpb["seq_len"] * bpb["steps_per_seed"],
        "RESULTS_BPB_META",
        "invalid bpb study metadata",
    )
    h256_err = parse_q(bpb["h256_ortho_fp64_max_err_q"], code="RESULTS_H256_ERR")
    require_ok(Fraction(0, 1) <= h256_err <= Fraction(1, 100000000000), "RESULTS_H256_BOUND", f"err={h256_err}")

    arms = bpb["arms"]
    require_ok(type(arms) is dict and frozenset(arms.keys()) == frozenset(_EXPECTED_ARMS), "RESULTS_ARMS", "bad arms")
    arm_means: Dict[str, Fraction] = {}
    arm_seeds: Dict[str, List[Fraction]] = {}
    for arm_name in _EXPECTED_ARMS:
        entry = arms[arm_name]
        require_ok(type(entry) is dict and frozenset(entry.keys()) == _ARM_KEYS, "RESULTS_ARM_TYPE", arm_name)
        exp_p, exp_b, exp_hbm = _EXPECTED_HEAD_SPECS[arm_name]
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
        vals: List[Fraction] = []
        row_seeds: List[int] = []
        for row in seeds_list:
            require_ok(
                type(row) is dict
                and frozenset(row.keys()) == _SEED_KEYS
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
            vals.append(v_fp16)
        require_ok(row_seeds == expected_seeds, "RESULTS_SEED_ORDER", arm_name)
        calc_mean = q_div_int(sum(vals, Fraction(0, 1)), len(vals))
        stored_mean = parse_q(entry["mean_eval_bpb_q"], code="RESULTS_MEAN_BPB")
        require_ok(stored_mean == calc_mean, "RESULTS_MEAN_MISMATCH", arm_name)
        arm_means[arm_name] = calc_mean
        arm_seeds[arm_name] = vals

    bs = res["bpb_summary"]
    require_ok(type(bs) is dict and frozenset(bs.keys()) == _BPB_SUMMARY_KEYS, "RESULTS_SUMMARY", "bad bpb_summary")
    a2b = arm_means["arm2b_bias_514B"]
    a5 = arm_means["arm5_bias_2p5KiB"]
    a7 = arm_means["arm7_bias_8p5KiB"]
    a9 = arm_means["arm9_monarch2_bias_16p5KiB"]
    a8a = arm_means["arm8a_lowrank16_bias_16p9KiB"]
    a4 = arm_means["arm4_dense_bias_262KiB"]

    a5_lt_8a = all(x < y for x, y in zip(arm_seeds["arm5_bias_2p5KiB"], arm_seeds["arm8a_lowrank16_bias_16p9KiB"]))
    a7_lt_2b = all(x < y for x, y in zip(arm_seeds["arm7_bias_8p5KiB"], arm_seeds["arm2b_bias_514B"]))
    a7_lt_8a = all(x < y for x, y in zip(arm_seeds["arm7_bias_8p5KiB"], arm_seeds["arm8a_lowrank16_bias_16p9KiB"]))
    a9_lt_2b = all(x < y for x, y in zip(arm_seeds["arm9_monarch2_bias_16p5KiB"], arm_seeds["arm2b_bias_514B"]))
    a9_lt_8a = all(x < y for x, y in zip(arm_seeds["arm9_monarch2_bias_16p5KiB"], arm_seeds["arm8a_lowrank16_bias_16p9KiB"]))

    cand_arms = ("arm5_bias_2p5KiB", "arm7_bias_8p5KiB", "arm9_monarch2_bias_16p5KiB")
    best_arm = min(cand_arms, key=lambda k: arm_means[k])
    best_lift = ref2b - arm_means[best_arm]
    dense_gap = ref2b - a4
    gap_closed = q_div(best_lift, dense_gap) if dense_gap > Fraction(0, 1) else Fraction(0, 1)

    require_ok(parse_q(bs["bias_only_lift_over_ref2b_q"]) == ref2b - a2b, "RESULTS_SUM_BIAS", "bias lift mismatch")
    require_ok(parse_q(bs["arm5_lift_over_ref2b_q"]) == ref2b - a5, "RESULTS_SUM_A5_REF", "a5 ref lift mismatch")
    require_ok(parse_q(bs["arm5_lift_over_2b_bias_q"]) == a2b - a5, "RESULTS_SUM_A5_2B", "a5 2b lift mismatch")
    require_ok(parse_q(bs["arm5_minus_arm8a_q"]) == a5 - a8a, "RESULTS_SUM_A5_8A", "a5-8a mismatch")
    require_ok(parse_q(bs["arm7_lift_over_ref2b_q"]) == ref2b - a7, "RESULTS_SUM_A7_REF", "a7 ref lift mismatch")
    require_ok(parse_q(bs["arm7_lift_over_2b_bias_q"]) == a2b - a7, "RESULTS_SUM_A7_2B", "a7 2b lift mismatch")
    require_ok(parse_q(bs["arm7_minus_arm8a_q"]) == a7 - a8a, "RESULTS_SUM_A7_8A", "a7-8a mismatch")
    require_ok(parse_q(bs["arm9_lift_over_ref2b_q"]) == ref2b - a9, "RESULTS_SUM_A9_REF", "a9 ref lift mismatch")
    require_ok(parse_q(bs["arm9_lift_over_2b_bias_q"]) == a2b - a9, "RESULTS_SUM_A9_2B", "a9 2b lift mismatch")
    require_ok(parse_q(bs["arm9_minus_arm8a_q"]) == a9 - a8a, "RESULTS_SUM_A9_8A", "a9-8a mismatch")
    require_ok(
        bs["arm5_lt_8a_all_seeds"] is a5_lt_8a
        and bs["arm7_lt_2b_bias_all_seeds"] is a7_lt_2b
        and bs["arm7_lt_8a_all_seeds"] is a7_lt_8a
        and bs["arm9_lt_2b_bias_all_seeds"] is a9_lt_2b
        and bs["arm9_lt_8a_all_seeds"] is a9_lt_8a,
        "RESULTS_SUM_FLAGS",
        "win flags mismatch",
    )
    require_ok(bs["best_structured_arm"] == best_arm, "RESULTS_SUM_BEST_ARM", "best arm mismatch")
    require_ok(parse_q(bs["best_lift_over_ref2b_q"]) == best_lift, "RESULTS_SUM_BEST_LIFT", "best lift mismatch")
    require_ok(parse_q(bs["dense_gap_ref2b_minus_arm4_q"]) == dense_gap, "RESULTS_SUM_DGAP", "dense gap mismatch")
    require_ok(parse_q(bs["best_gap_closed_fraction_q"]) == gap_closed, "RESULTS_SUM_GCLOSED", "gap closed mismatch")
    require_ok(parse_q(bs["best_minus_arm4_dense_q"]) == arm_means[best_arm] - a4, "RESULTS_SUM_BEST_A4", "best-a4 mismatch")

    g1_arm7 = (ref2b - a7) >= Fraction(6, 5) and (a2b - a7) >= Fraction(2, 5) and a7_lt_2b and a7_lt_8a
    g1_arm9 = (ref2b - a9) >= Fraction(6, 5) and (a2b - a9) >= Fraction(2, 5) and a9_lt_2b and a9_lt_8a
    g1_calc = g1_arm7 or g1_arm9

    g2_calc = (
        ptx["nonzero_spill_kernels"] == 0
        and ptx["total_kernels"] == 4
        and all(us_v <= Fraction(3, 1) for us_v in (us_2b, us_5, us_7, us_9))
    )
    gates = res["gates"]
    require_ok(
        type(gates) is dict
        and frozenset(gates.keys()) == _GATES_KEYS
        and gates["G1_bpb_bias_and_lowrank_beat"] is g1_calc
        and gates["G2_sm86_zero_spill_latency_parity"] is g2_calc,
        "RESULTS_GATES_MISMATCH",
        "recomputed gates mismatch",
    )
    if g1_calc and g2_calc:
        expected_verdict = "CONFIRMED"
    elif g2_calc:
        expected_verdict = "PARTIAL"
    else:
        expected_verdict = "FALSIFIED"
    require_ok(res["verdict"] == expected_verdict, "RESULTS_VERDICT_MISMATCH", f"expected={expected_verdict}")
    return res


def _decode_utf8_strict(raw_bytes: bytes, *, name: str) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Monarch2HeadVerifyError("READ_OR_PARSE_ERROR", f"{name}: UnicodeDecodeError") from exc


def verify_on_disk(repo_root: Path) -> Dict[str, Any]:
    self_path = Path(__file__)
    self_bytes = _read_regular_bytes_nofollow(self_path)
    verify_pure_q_ast(_decode_utf8_strict(self_bytes, name=self_path.name))

    # TOCTOU-free dir_fd O_NOFOLLOW traversal + single-fd read-once hashing for every verified input
    kernel_bytes = _read_rel_bytes_nofollow(repo_root, KERNEL_CU_REL)
    bench_bytes = _read_rel_bytes_nofollow(repo_root, BENCH_CU_REL)
    bpb_script_bytes = _read_rel_bytes_nofollow(repo_root, BPB_STUDY_SCRIPT_REL)
    ptx_bytes = _read_rel_bytes_nofollow(repo_root, PTXAS_RECEIPT_REL)
    res_bytes = _read_rel_bytes_nofollow(repo_root, RESULTS_RECEIPT_REL)

    kernel_sha = hashlib.sha256(kernel_bytes).hexdigest()
    bench_sha = hashlib.sha256(bench_bytes).hexdigest()
    bpb_script_sha = hashlib.sha256(bpb_script_bytes).hexdigest()
    ptx_file_sha = hashlib.sha256(ptx_bytes).hexdigest()
    res_file_sha = hashlib.sha256(res_bytes).hexdigest()

    scale_binding, corpus_pins = verify_cuda_scale_and_study_bindings(
        _decode_utf8_strict(kernel_bytes, name=Path(KERNEL_CU_REL).name),
        _decode_utf8_strict(bench_bytes, name=Path(BENCH_CU_REL).name),
        _decode_utf8_strict(bpb_script_bytes, name=Path(BPB_STUDY_SCRIPT_REL).name),
    )
    ptx = parse_strict_canonical_json_bytes(ptx_bytes, name=Path(PTXAS_RECEIPT_REL).name)
    res = parse_strict_canonical_json_bytes(res_bytes, name=Path(RESULTS_RECEIPT_REL).name)

    verify_ptxas_receipt(ptx, expected_cu_sha=kernel_sha)
    verify_results_receipt(
        res,
        ptx,
        expected_bench_cu_sha=bench_sha,
        expected_bpb_script_sha=bpb_script_sha,
        expected_corpus_pins=corpus_pins,
    )
    oracle = verify_monarch2_and_fwht_oracle()

    summary: Dict[str, Any] = {
        "schema": SCHEMA_VERIFY,
        "energy_power_status": ENERGY_POLICY_STATUS,
        "status": "PASS",
        "verdict": res["verdict"],
        "gates": res["gates"],
        "kernel_cu_sha256": kernel_sha,
        "bench_cu_sha256": bench_sha,
        "bpb_study_script_sha256": bpb_script_sha,
        "corpus_sha256": res["source_bindings"]["corpus_sha256"],
        "ptxas_file_sha256": ptx_file_sha,
        "results_file_sha256": res_file_sha,
        "ptxas_receipt_sha256": ptx["receipt_sha256"],
        "result_sha256": res["result_sha256"],
        "cuda_scale_binding": scale_binding,
        "monarch2_fwht_oracle": oracle,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _write_stream_or_epipe(stream: Any, text: str) -> bool:
    try:
        stream.write(text)
        stream.flush()
        return True
    except BrokenPipeError:
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, stream.fileno())
            finally:
                os.close(devnull_fd)
        except OSError:
            pass
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pure-Q / Pure-Z Verifier for sm_86 Monarch-2 + Byte-Bias FWHT Head")
    parser.add_argument("--check", action="store_true", help="Verify on-disk receipts and oracles")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary")
    parser.add_argument("--repo-root", type=str, default=None, help="Override repository root directory")
    parser.add_argument("--require-verdict", type=str, default=None, help="Fail closed unless verdict matches this exact value")
    args = parser.parse_args(argv)

    root = Path(args.repo_root) if args.repo_root is not None else Path(__file__).absolute().parent.parent
    try:
        summary = verify_on_disk(root)
        if args.require_verdict is not None:
            require_ok(
                summary["verdict"] == args.require_verdict,
                "VERDICT_REQUIREMENT_FAILED",
                f"expected={args.require_verdict} got={summary['verdict']}",
            )
    except Monarch2HeadVerifyError as exc:
        _write_stream_or_epipe(
            sys.stderr,
            json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True, allow_nan=False)
            + "\n",
        )
        return 2

    out_text = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.json
        else f"PASS {summary['verdict']} result_sha256={summary['result_sha256']}\n"
    )
    return 0 if _write_stream_or_epipe(sys.stdout, out_text) else 2


if __name__ == "__main__":
    raise SystemExit(main())
