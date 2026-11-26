#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib fail-closed verifier that parses README.md and cpu/avx512/README.md
and checks every quoted figure, receipt SHA-256 pin, and public-hygiene policy
directly against the committed JSON receipts in receipts/.

Zero external dependencies, zero bare assert statements (safe under python3 -O),
strict regular-file I/O (O_NOFOLLOW | O_NONBLOCK, S_ISREG, byte cap, ancestor
symlink checks), HTML-comment stripping + unique-match enforcement, and strict
JSON parsing (rejects duplicate keys, NaN/Inf, overflow floats like 1e999,
oversized integers, and excessive nesting depth).
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_INT_DIGITS = 1024

EXPECTED_EIGHT_PIN_FILES: frozenset[str] = frozenset(
    {
        "receipts/s116_phase2b_gpu_results.json",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        "receipts/mlos_zone_b_s116_phase2b.json",
        "receipts/mlos_zone_b_s117_live.json",
        "receipts/mlos_zone_b_s117_v2_d512.json",
        "receipts/mlos_zone_b_s117_v3_frontier.json",
    }
)

EXPECTED_CANONICAL_MLOS_FILES: frozenset[str] = frozenset(
    {
        "receipts/mlos_zone_b_s116_phase2b.json",
        "receipts/mlos_zone_b_s117_live.json",
        "receipts/mlos_zone_b_s117_v2_d512.json",
        "receipts/mlos_zone_b_s117_v3_frontier.json",
    }
)

_NUM_TOK = r"(?<![A-Za-z0-9_.])\d{1,12}(?:,\d{3}){0,4}(?:\.\d{1,12})?(?:[eE][+-]?\d{1,4})?"

FORBIDDEN_README_PATTERNS: tuple[tuple[str, str, int], ...] = (
    (
        "energy_units_ci",
        rf"{_NUM_TOK}[ \t]*(?:(?:m|u|\u00b5|\u03bc|k|g)?(?:j|wh)(?:/tok(?:en)?)?|watts?)\b",
        re.IGNORECASE,
    ),
    (
        "watt_units_cs",
        rf"{_NUM_TOK}[ \t]*(?:m|k|M|G)?W(?:/tok(?:en)?)?(?=\s|[,).;]|$)",
        0,
    ),
    (
        "spelled_energy_ci",
        r"\b(?:energy|power|wattage)(?:[ \t]*:[ \t]*|[ \t]+)\d"
        r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
        r"(?:[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))?[ \t]+(?:watts?|joules?|milliwatts?|millijoules?)\b",
        re.IGNORECASE,
    ),
    (
        "voltage_current_pair",
        rf"(?:{_NUM_TOK}[ \t]*(?:m?A|amps?)\b[^.\n]{{0,24}}{_NUM_TOK}[ \t]*(?:m?V|volts?)\b"
        rf"|{_NUM_TOK}[ \t]*(?:m?V|volts?)\b[^.\n]{{0,24}}{_NUM_TOK}[ \t]*(?:m?A|amps?)\b)",
        re.IGNORECASE,
    ),
    (
        "fp16_equiv_throughput_ci",
        rf"(?:{_NUM_TOK}[ \t]*(?:TB/s|GB/s|MB/s|[TGMPE]?FLOPS?|[TGMPE]?FLOP/s|[TGMPE]?TOPS)[ \t]*\(?(?:fp-?16|bf-?16)\b"
        rf"|\b(?:fp-?16|bf-?16)[ \t:(\-]*{_NUM_TOK}[ \t]*(?:TB/s|GB/s|MB/s|[TGMPE]?FLOPS?|[TGMPE]?FLOP/s|[TGMPE]?TOPS)\b)",
        re.IGNORECASE,
    ),
    (
        "html_numeric_entity",
        r"&#(?:\d{1,7}|[xX][0-9a-fA-F]{1,6});",
        0,
    ),
    (
        "raw_json_field_name",
        r"\bspeedup_vs_choice_b_532ms\b",
        0,
    ),
    (
        "removed_transcode_row",
        r"rtx3080_live_llm_transcode_kernels",
        0,
    ),
)

_COMPILED_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (rule_id, re.compile(pattern, flags=flags))
    for rule_id, pattern, flags in FORBIDDEN_README_PATTERNS
)
_FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n?$")
_FENCE_CLOSE_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})[ \t]*\r?\n?$")
_MD_BS_ESCAPE_RE = re.compile(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~ ])')
_UNICODE_WS_RE = re.compile(r"[\u00a0\u2007\u2009\u202f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u00ad\ufeff]")
_LEAN_THEOREM_RE = re.compile(r"^\s*(?:theorem|lemma)\s+\S+", re.MULTILINE)


class ReadmeFigureError(ValueError):
    """Fail-closed README figure, receipt pin, or policy violation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    """Assertion-free guard safe under python3 -O."""
    if cond is not True:
        raise ReadmeFigureError(code, message)


def _reject_nonfinite(token: str) -> Any:
    raise ReadmeFigureError("ERR_JSON_NONFINITE", f"Non-finite JSON constant forbidden: {token!r}")


def _parse_strict_float(token: str) -> float:
    val = float(token)
    require_ok(math.isfinite(val), "ERR_JSON_NONFINITE", f"Non-finite JSON float forbidden: {token!r}")
    return val


def _parse_strict_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "ERR_BAD_JSON", f"JSON integer exceeds {MAX_JSON_INT_DIGITS} digits")
    return int(token)


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        require_ok(k not in out, "ERR_JSON_DUP_KEY", f"Duplicate JSON key: {k!r}")
        out[k] = v
    return out


def _check_depth(obj: Any, depth: int = 0) -> None:
    require_ok(depth <= MAX_JSON_DEPTH, "ERR_JSON_DEPTH", f"JSON nesting exceeds {MAX_JSON_DEPTH}")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth(item, depth + 1)


def _is_symlink_at(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode)
    except OSError:
        return False


def _open_regular_nofollow(path: Path) -> int:
    """Open a file by walking every path component via dir_fd + O_NOFOLLOW (TOCTOU-free)."""
    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_absolute():
        parts = path.parts[1:]
    else:
        cwd = Path(os.getcwd())
        parts = (*cwd.parts[1:], *path.parts)
    require_ok(len(parts) > 0, "ERR_NOT_REGULAR", "Non-regular root path refused")
    try:
        cur_fd = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise ReadmeFigureError("ERR_MISSING_FILE", "Unable to open root directory") from exc
    try:
        for comp in parts[:-1]:
            if comp == ".":
                continue
            try:
                next_fd = os.open(comp, dir_flags, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR) and _is_symlink_at(comp, cur_fd):
                    raise ReadmeFigureError("ERR_SYMLINK", f"Symlink ancestor refused: {comp}") from exc
                if exc.errno == errno.ENOTDIR:
                    raise ReadmeFigureError("ERR_NOT_REGULAR", f"Non-directory ancestor refused: {comp}") from exc
                raise ReadmeFigureError("ERR_MISSING_FILE", f"Unable to traverse directory: {comp}") from exc
            try:
                os.close(cur_fd)
            except OSError:
                pass
            cur_fd = next_fd
        leaf = parts[-1]
        try:
            return os.open(leaf, file_flags, dir_fd=cur_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ReadmeFigureError("ERR_SYMLINK", f"Symlink refused: {leaf}") from exc
            raise ReadmeFigureError("ERR_MISSING_FILE", f"Unable to open regular file: {leaf}") from exc
    finally:
        try:
            os.close(cur_fd)
        except OSError:
            pass


def read_regular_bytes(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    """Read a regular non-symlink file via dir_fd O_NOFOLLOW | O_NONBLOCK with strict byte cap."""
    fd = _open_regular_nofollow(path)
    try:
        try:
            st = os.fstat(fd)
            require_ok(stat.S_ISREG(st.st_mode), "ERR_NOT_REGULAR", f"Non-regular file refused: {path.name}")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError as exc:
            raise ReadmeFigureError("ERR_READ_ERROR", f"Unable to read regular file: {path.name}") from exc
        data = b"".join(chunks)
        require_ok(len(data) <= max_bytes, "ERR_FILE_TOO_LARGE", f"File exceeds byte cap: {path.name}")
        return data
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _decode_utf8(raw: bytes, doc_name: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReadmeFigureError("ERR_BAD_UTF8", f"Invalid UTF-8 in {doc_name}") from exc


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load a regular JSON object rejecting symlinks, syntax errors, dup keys, NaN/Inf, overflow floats, oversized ints, and deep nesting."""
    raw = read_regular_bytes(path)
    try:
        obj = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite,
            parse_float=_parse_strict_float,
            parse_int=_parse_strict_int,
            object_pairs_hook=_reject_dup_keys,
        )
    except ReadmeFigureError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ReadmeFigureError("ERR_BAD_JSON", f"Malformed JSON in {path.name}") from exc
    require_ok(isinstance(obj, dict), "ERR_JSON_OBJECT", f"Expected top-level JSON object in {path.name}")
    _check_depth(obj, 0)
    return obj


def _parse_number(token: str) -> float:
    cleaned = token.replace(",", "")
    try:
        val = float(cleaned)
    except ValueError as exc:
        raise ReadmeFigureError("ERR_BAD_NUMBER", f"Unparseable numeric figure: {token!r}") from exc
    require_ok(math.isfinite(val), "ERR_NONFINITE_VAL", f"Non-finite parsed figure: {token!r}")
    return val


def _strip_comments_outside_inline_code(chunk: str, doc_name: str) -> str:
    """Linear O(n) HTML-comment stripper preserving inline backtick code spans."""
    out: list[str] = []
    n = len(chunk)
    i = 0
    unclosed_tick_lens: set[int] = set()
    while i < n:
        next_bt = chunk.find("`", i)
        next_cm = chunk.find("<!--", i)
        if next_bt == -1 and next_cm == -1:
            out.append(chunk[i:])
            break
        if next_bt != -1 and (next_cm == -1 or next_bt < next_cm):
            out.append(chunk[i:next_bt])
            k = next_bt
            while k < n and chunk[k] == "`":
                k += 1
            run = k - next_bt
            close_pos = -1
            if run not in unclosed_tick_lens:
                search = k
                while search < n:
                    nxt = chunk.find("`", search)
                    if nxt == -1:
                        break
                    m_end = nxt
                    while m_end < n and chunk[m_end] == "`":
                        m_end += 1
                    if m_end - nxt == run:
                        close_pos = m_end
                        break
                    search = m_end
                if close_pos == -1:
                    unclosed_tick_lens.add(run)
            if close_pos != -1:
                out.append(chunk[next_bt:close_pos])
                i = close_pos
            else:
                out.append(chunk[next_bt:k])
                i = k
        else:
            out.append(chunk[i:next_cm])
            end_cm = chunk.find("-->", next_cm + 4)
            require_ok(end_cm != -1, "ERR_README_COMMENT", f"Unclosed HTML comment in {doc_name}")
            i = end_cm + 3
    return "".join(out)


def _strip_html_comments(text: str, doc_name: str) -> str:
    """Remove HTML comments outside fenced/inline Markdown code in O(n) time and reject unclosed comments."""
    out: list[str] = []
    plain_buf: list[str] = []
    fence_char = ""
    fence_len = 0
    for line in text.splitlines(keepends=True):
        if fence_char:
            out.append(line)
            m_close = _FENCE_CLOSE_RE.match(line)
            if m_close is not None:
                tok = m_close.group(1)
                if tok[0] == fence_char and len(tok) >= fence_len:
                    fence_char = ""
                    fence_len = 0
            continue
        m_open = _FENCE_OPEN_RE.match(line)
        if m_open is not None:
            tok = m_open.group(1)
            info = m_open.group(2)
            if tok[0] == "~" or "`" not in info:
                if plain_buf:
                    out.append(_strip_comments_outside_inline_code("".join(plain_buf), doc_name))
                    plain_buf.clear()
                fence_char = tok[0]
                fence_len = len(tok)
                out.append(line)
                continue
        plain_buf.append(line)
    if plain_buf:
        out.append(_strip_comments_outside_inline_code("".join(plain_buf), doc_name))
    return "".join(out)


def _normalize_markdown_prose(visible_text: str) -> str:
    """Decode HTML character entities, CommonMark punctuation escapes, and NBSP/zero-width obfuscation."""
    decoded = html.unescape(visible_text)
    decoded = _MD_BS_ESCAPE_RE.sub(r"\1", decoded)
    decoded = _UNICODE_WS_RE.sub(" ", decoded)
    decoded = _ZERO_WIDTH_RE.sub("", decoded)
    return decoded


def _s71_ratio_dual_vs_stride8(r: dict[str, Any]) -> float:
    m = r["measurements"]
    df0 = m["avx512_a16_blockh4_dual_force"][0]
    sw8 = next(row for row in m["avx512_a16_blockh4_stride_sweep"] if row["stride_bytes"] == 8)
    return (df0["total_ops"] / df0["ns"]) / (sw8["total_ops"] / sw8["ns"])


def _s71_ww8_slowdown(r: dict[str, Any]) -> float:
    ww8 = r["measurements"]["write_write_8core"]
    ns8 = next(row["ns"] for row in ww8 if row["stride_bytes"] == 8)
    ns64 = next(row["ns"] for row in ww8 if row["stride_bytes"] == 64)
    return ns8 / ns64


def _s52_verify_overhead_pct(r: dict[str, Any]) -> float:
    arms = r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]
    base = float(arms["A16_preenc_dp4a_r2_l2pin"]["gweights_per_s"])
    ver = float(arms["A16_preenc_dp4a_live_verify_r2_l2pin"]["gweights_per_s"])
    return 100.0 * (base - ver) / base


def _s125_ctl_runs(r: dict[str, Any], key: str) -> list[float]:
    triples = r["interleaved_benchmarks"][key]["triples"]
    return [float(t["ctl_pre_tok_s"]) for t in triples] + [float(t["ctl_post_tok_s"]) for t in triples]


def _s125_cand_runs(r: dict[str, Any], key: str) -> list[float]:
    return [float(t["cand_tok_s"]) for t in r["interleaved_benchmarks"][key]["triples"]]


# Each spec: (id, unique anchored regex with 1 numeric capture group, receipt_rel, extractor(receipt)->float, tol)
FigureSpec = tuple[str, str, str, Callable[[dict[str, Any]], float], float]

README_FIGURE_SPECS: tuple[FigureSpec, ...] = (
    # Intro header
    (
        "intro_stream_triad",
        r"`([0-9][0-9.,]*)\s+GB/s`\s+STREAM Triad",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["device"]["stream_triad_gb_s"]),
        0.005,
    ),
    # Row 1: qwen25_sm86_a16_engine.cu + qwen25_avx512_a16_engine.c
    (
        "qwen25_quantized_weights_m",
        r"`Qwen/Qwen2\.5-0\.5B`\s+\(`([0-9][0-9.,]*)M`\s+quantized weights\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["model_identity"]["block_weight_count"]) / 1e6,
        0.05,
    ),
    (
        "qwen25_backbone_tok_s",
        r"\*\*`([0-9][0-9.,]*)\s+tok/s`\*\*\s+24-layer backbone",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_backbone_24layers"]["arm_b1_a16_sm86_tok_s"]),
        0.005,
    ),
    (
        "qwen25_backbone_speedup",
        r"24-layer backbone\s+\(\*\*`([0-9][0-9.,]*)×`\*\*\s+vs\.\s+PyTorch `BF16`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_backbone_24layers"]["speedup_b1_vs_a_bf16"]),
        0.005,
    ),
    (
        "qwen25_e2e_tok_s",
        r"\*\*`([0-9][0-9.,]*)\s+tok/s`\*\*\s+full E2E w/",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_full_e2e_with_bf16_lm_head"]["arm_b1_a16_sm86_tok_s"]),
        0.005,
    ),
    (
        "qwen25_vocab_size",
        r"full E2E w/\s+`([0-9][0-9,]*)`-vocab `BF16` `lm_head`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["model_identity"]["vocab_size"]),
        0.0,
    ),
    (
        "qwen25_e2e_speedup",
        r"`lm_head`\s+\(\*\*`([0-9][0-9.,]*)×`\*\*\s+vs\.\s+`BF16`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_full_e2e_with_bf16_lm_head"]["speedup_b1_vs_a_bf16"]),
        0.005,
    ),
    (
        "qwen25_e2e_bf16_tok_s",
        r"vs\.\s+`BF16`\s+`([0-9][0-9.,]*)\s+tok/s`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_full_e2e_with_bf16_lm_head"]["arm_a_bf16_gpu_tok_s"]),
        0.005,
    ),
    (
        "qwen25_cold_load_ms",
        r"\*\*`([0-9][0-9.,]*)\s+ms`\*\*\s+cold-to-hot load",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["arm_d_cold_18b_transcode_and_128tok_gen"]["total_cold_to_hot_vram_ms"]),
        0.005,
    ),
    (
        "qwen25_cpu_fallback_tok_s",
        r"\*\*`([0-9][0-9.,]*)\s+tok/s`\*\*\s+zero-VRAM CPU fallback",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_full_e2e_with_bf16_lm_head"]["arm_c1_a16_avx512_8t_tok_s"]),
        0.005,
    ),
    (
        "qwen25_greedy_match_count",
        r"`(\d+)/16`\s+exact greedy GPU-vs-CPU match",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["arm_c_avx512_cpu_only_a16"]["parity_vs_gpu_a16"]["greedy_16tok_match_vs_gpu_a16"]),
        0.0,
    ),
    (
        "qwen25_sass_kernel_count",
        r"`(\d+)/\d+`\s+kernels `0 B` spill in receipt",
        "receipts/s58_sm86_engine_sass_inspection.json",
        lambda r: float(r["kernel_count"]),
        0.0,
    ),
    # Row 2: aletheion_sm86_ptx_foundry.cu
    (
        "foundry_peak_gweights_s",
        r"\*\*`([0-9][0-9.,]*)\s+Gweights/s`\*\*\s+\(`[0-9.]+%`\s+STREAM Triad\)",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_preenc_dp4a_r1_l2off"]["gweights_per_s"]),
        0.005,
    ),
    (
        "foundry_peak_pct_triad",
        r"Gweights/s`\*\*\s+\(`([0-9][0-9.,]*)%`\s+STREAM Triad\)",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_preenc_dp4a_r1_l2off"]["pct_triad"]),
        0.005,
    ),
    (
        "foundry_live_verify_gw_s",
        r"live `5-POPC` syndrome verify at `([0-9][0-9.,]*)\s+Gw/s`",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_preenc_dp4a_live_verify_r2_l2pin"]["gweights_per_s"]),
        0.005,
    ),
    (
        "foundry_verify_overhead_pct",
        r"syndrome verify at `[0-9.]+\s+Gw/s`\s+\(`([0-9][0-9.,]*)%`\s+overhead",
        "receipts/s52_sm86_ptx_foundry_results.json",
        _s52_verify_overhead_pct,
        0.05,
    ),
    (
        "foundry_same_protocol_gw_s",
        r"vs\.\s+the same-protocol `([0-9][0-9.,]*)\s+Gw/s`\s+arm",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_preenc_dp4a_r2_l2pin"]["gweights_per_s"]),
        0.005,
    ),
    (
        "foundry_warp_shuffle_gw_s",
        r"zero-shmem warp-shuffle `WHT_8`/`H_16`\s+\(`([0-9][0-9.,]*)\s+Gw/s`\)",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_H16_warp_shfl_zero_shmem_r2_l2pin"]["gweights_per_s"]),
        0.005,
    ),
    (
        "foundry_live_repair_gw_s",
        r"`O\(1\)` register-delta live repair\s+\(`([0-9][0-9.,]*)\s+Gw/s`\)",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["arms"]["A16_live_repair_dp4a_r2_l2pin"]["gweights_per_s"]),
        0.005,
    ),
    (
        "foundry_b32_int8_tops",
        r"batched `B=32` `cp\.async \+ IMMA\.16832`\s+\(`([0-9][0-9.,]*)\s+INT8 TOPS`\)",
        "receipts/s52_sm86_ptx_foundry_results.json",
        lambda r: float(r["fused_gemv_4_5_bpw"]["8192x14336"]["batched_b32_cp_async_imma_gemm"]["int8_tops"]),
        0.005,
    ),
    (
        "foundry_sass_kernel_count",
        r"`(\d+)/\d+`\s+kernels `0 B` spill, `0` verify fails",
        "receipts/s52_sm86_ptx_sass_inspection.json",
        lambda r: float(r["kernels_compiled"]) if r.get("zero_spill_verified") is True else -1.0,
        0.0,
    ),
    # Row 3: s71_proximity_bench.c
    (
        "s71_dual_force_gweights_s",
        r"\*\*`([0-9][0-9.,]*)\s+Gweights/s`\*\*\s+\(`total_ops / ns`",
        "receipts/s71_proximity_avx512_results.json",
        lambda r: float(r["measurements"]["avx512_a16_blockh4_dual_force"][0]["total_ops"])
        / float(r["measurements"]["avx512_a16_blockh4_dual_force"][0]["ns"]),
        0.0005,
    ),
    (
        "s71_speedup_vs_stride8",
        r"\(\*\*`([0-9][0-9.,]*)×`\s+faster\*\*\s+per weight than the `8 B` row",
        "receipts/s71_proximity_avx512_results.json",
        _s71_ratio_dual_vs_stride8,
        0.0005,
    ),
    # Row 4: aletheion_avx512_a16_decoder.c
    (
        "avx512_vnni_min_gweights_s",
        r"\*\*`([0-9][0-9.,]*)–[0-9.]+\s+Gweights/s`\*\*\s+fused `vpermb \+ vpdpbusd`",
        "receipts/s52_avx512_a16_benchmark_results.json",
        lambda r: float(r["fused_vnni_gemv"]["a16_vnni_8t_gweights_s"]),
        0.005,
    ),
    (
        "avx512_vnni_max_gweights_s",
        r"\*\*`[0-9.]+–([0-9][0-9.,]*)\s+Gweights/s`\*\*\s+fused `vpermb \+ vpdpbusd`",
        "receipts/s52_avx512_a16_benchmark_results.json",
        lambda r: float(r["fused_vnni_gemv"]["a16_vnni_live_verify_8t_gweights_s"]),
        0.005,
    ),
    (
        "avx512_hampack_mweights_s",
        r"\*\*`([0-9][0-9.,]*)\s+Mweights/s`\*\*\s+exact bit-identical",
        "receipts/s52_avx512_a16_benchmark_results.json",
        lambda r: float(r["siliquant77_exact_enumerative_hampack"]["fast_u128_avx512_16t_mweights_s"]),
        0.005,
    ),
    # Row 6: su2_holonomy_scan_sm86.cu
    (
        "su2_kernel_count",
        r"`(\d+)/27`\s+kernels `0 B` spill, `0` PTX `ld/st\.local`",
        "receipts/su2_holonomy_sm86_ptxas_receipt.json",
        lambda r: float(r["kernel_count"]),
        0.0,
    ),
    # HoloSphere V3 Frontier row
    (
        "v3_bwd_tok_s",
        r"\*\*`([0-9][0-9.,]*)\s+tok/s`\*\*\s+\(`[0-9.]+\s+ms / 12L`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["chunk32_bwd_wmma"]["B4_T8192_D512_L12_tok_per_sec"]),
        0.005,
    ),
    (
        "v3_bwd_ms",
        r"\(`([0-9][0-9.,]*)\s+ms / 12L`\s+at `B=4, T=8192, D=512`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["chunk32_bwd_wmma"]["B4_T8192_D512_L12_ms"]),
        0.005,
    ),
    (
        "v3_bwd_speedup",
        r"\*\*`([0-9][0-9.,]*)×`\*\*\s+over Choice B;",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["chunk32_bwd_wmma"]["speedup_vs_choice_b_532ms"]),
        0.00005,
    ),
    (
        "v3_bwd_regs",
        r"over Choice B;\s+`(\d+)`\s+regs / `[0-9,]+\s+B`\s+SMEM",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][0]["registers"]),
        0.0,
    ),
    (
        "v3_bwd_smem",
        r"over Choice B;\s+`\d+`\s+regs / `([0-9][0-9,]*)\s+B`\s+SMEM",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][0]["smem_bytes"]),
        0.0,
    ),
    (
        "v3_bwd_hmma",
        r"`(\d+)`\s+`HMMA\.16816`",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][0]["sass"]["HMMA_16816"]),
        0.0,
    ),
    (
        "v3_bwd_shfl",
        r"`HMMA\.16816`,\s+`(\d+)`\s+`SHFL`\)",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][0]["sass"]["SHFL"]),
        0.0,
    ),
    (
        "v3_d512_regs",
        r"`v3_d512`\s+\(`(\d+)`\s+regs / `[0-9,]+\s+B`\s+SMEM",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][1]["registers"]),
        0.0,
    ),
    (
        "v3_d512_smem",
        r"`v3_d512`\s+\(`\d+`\s+regs / `([0-9][0-9,]*)\s+B`\s+SMEM",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][1]["smem_bytes"]),
        0.0,
    ),
    (
        "v3_d512_popc",
        r"`POPC:\s+(\d+)`\s+outside per-layer hot path",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernels"][1]["sass"]["POPC"]),
        0.0,
    ),
    (
        "v3_d512_cellop_graph_us",
        r"CUDA-graph \*\*`([0-9][0-9.,]*)\s+µs/tok`\*\*\s+CellOp",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["cellop_graph_us_per_tok_8192"]),
        0.005,
    ),
    (
        "v3_d512_e2e_graph_us",
        r"CellOp \(`T=8192`\) / \*\*`([0-9][0-9.,]*)\s+µs/tok`\*\*\s+E2E",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_graph_us_per_tok_16384"]),
        0.005,
    ),
    (
        "v3_d512_e2e_graph_tok_s",
        r"E2E \(`T=16384`,\s+`([0-9][0-9.,]*)\s+tok/s`\)",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_graph_tok_per_sec_16384"]),
        0.005,
    ),
    (
        "v3_d512_nongraph_cellop_us",
        r"matched `T=8192` non-graph `([0-9][0-9.,]*)`\s+/\s+`[0-9.]+\s+µs/tok`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["cellop_us_per_tok_8192"]),
        0.005,
    ),
    (
        "v3_d512_nongraph_e2e_us",
        r"matched `T=8192` non-graph `[0-9.]+`\s+/\s+`([0-9][0-9.,]*)\s+µs/tok`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_us_per_tok_8192"]),
        0.005,
    ),
    (
        "v3_d512_nongraph_tok_s",
        r"non-graph `[0-9.]+` / `[0-9.]+\s+µs/tok`\s+\(`([0-9][0-9.,]*)\s+tok/s`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_tok_per_sec_8192"]),
        0.005,
    ),
    (
        "v3_d512_nongraph_diff_us",
        r"`([0-9][0-9.,]*)\s+µs/tok`\s+diff,\s+`[0-9.e-]+`\s+`FWHT_256`\s+err",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_us_per_tok_8192"]) - float(r["v3_d512"]["cellop_us_per_tok_8192"]),
        0.00005,
    ),
    (
        "v3_d512_fwht_err",
        r"`([0-9.]+e-[0-9]+)`\s+`FWHT_256`\s+err",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["fwht256_max_abs_err"]),
        5e-10,
    ),
    (
        "v3_kernel_count",
        r"\|\s+`(\d+)/2`\s+`0 B` spill, `0` `LDL`/`STL`",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        lambda r: float(r["total_kernels"]),
        0.0,
    ),
    # HoloSphere S117 Chunk32 row
    (
        "s117_choice_b_tok_s",
        r"Choice B \*\*`([0-9][0-9.,]*)\s+tok/s`\*\*",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        lambda r: float(r["bare_metal_rtx3080"]["chunk32_woodbury_bwd_ab_t8192"]["choice_b_6336B_tok_per_s"]),
        0.005,
    ),
    (
        "s117_choice_b_speedup",
        r"\(\*\*`([0-9][0-9.,]*)×`\*\*\s+over Choice A",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        lambda r: float(r["bare_metal_rtx3080"]["chunk32_woodbury_bwd_ab_t8192"]["speedup_choice_b_over_a"]),
        0.00005,
    ),
    (
        "s117_choice_a_tok_s",
        r"over Choice A `([0-9][0-9.,]*)\s+tok/s`",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        lambda r: float(r["bare_metal_rtx3080"]["chunk32_woodbury_bwd_ab_t8192"]["choice_a_4224B_tok_per_s"]),
        0.005,
    ),
    (
        "s117_gs_parity",
        r"`([0-9.]+e-[0-9]+)`\s+`gS`\s+parity",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        lambda r: float(r["bare_metal_rtx3080"]["chunk32_woodbury_bwd_ab_t8192"]["gpu_a_vs_b_gS_max_abs_diff"]),
        5e-9,
    ),
    (
        "s117_t8192_vram_mib",
        r"\*\*`([0-9][0-9.,]*)\s+MiB`\*\*\s+live peak VRAM",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        lambda r: float(r["live_gpu_torch_compile_and_t8192_vram"]["t8192_live_peak_allocated_mib"]),
        0.005,
    ),
    (
        "s117_v2_cellop_us",
        r"`v2_d512`\s+`16`-warp \*\*`([0-9][0-9.,]*)\s+µs/tok`\*\*\s+CellOp",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        lambda r: float(r["cellop_only_12L_d512"]["t8192_us_per_tok"]),
        0.005,
    ),
    (
        "s117_v2_e2e_us",
        r"CellOp / \*\*`([0-9][0-9.,]*)\s+µs/tok`\*\*\s+\(`[0-9.]+×`\s+over `v1`",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        lambda r: float(r["full_e2e_fwht256_12L_d512"]["t8192_us_per_tok"]),
        0.005,
    ),
    (
        "s117_v2_speedup_vs_v1",
        r"`([0-9][0-9.,]*)×`\s+over `v1`",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        lambda r: float(r["full_e2e_fwht256_12L_d512"]["speedup_vs_v1_249_32us"]),
        0.0005,
    ),
    (
        "s117_v2_fp64_err",
        r"`([0-9.]+e-[0-9]+)`\s+vs FP64 dense",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        lambda r: float(r["fwht256_bent4_lm_head_parity"]["max_abs_err_vs_fp64_dense"]),
        5e-10,
    ),
    (
        "s117_kernel_count",
        r"`(\d+)/11`\s+`0 B` spill, `0` `LDL`/`STL`, `0` `MEMBAR_GL`",
        "receipts/s117_ptxas_zero_spill_receipt.json",
        lambda r: float(r["total_kernels"]),
        0.0,
    ),
    # HoloSphere S116 Megakernel row
    (
        "s116_cellop_us",
        r"\*\*`([0-9][0-9.,]*)\s+µs/tok`\*\*\s+`12L, d=256`\s+`CellOp`",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16a_v2_matched_12layer_showdown"]["uniform_cell_8warp_d256_1cta"]["us_per_tok"]),
        0.005,
    ),
    (
        "s116_speedup_vs_vm",
        r"\(\*\*`([0-9][0-9.,]*)×`\*\*\s+over `48`-CTA switch VM",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16a_v2_matched_12layer_showdown"]["uniform_cell_8warp_d256_1cta"]["speedup_vs_switch_vm_48cta"]),
        0.005,
    ),
    (
        "s116_speedup_vs_async",
        r"\*\*`([0-9][0-9.,]*)×`\*\*\s+over `48`-launch async C\+\+\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16a_v2_matched_12layer_showdown"]["uniform_cell_8warp_d256_1cta"]["speedup_vs_perop_12l_48"]),
        0.005,
    ),
    (
        "s116_l2pin_clamp_mib",
        r"`ScopedAmpereL2Pin`\*\*\s+\(`([0-9][0-9.,]*)\s+MiB`\s+GA104M exact clamp\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["max_persisting_l2_cache_mib"]),
        0.005,
    ),
    (
        "s116_l2pin_gb_s",
        r"GA104M exact clamp\)\s+\*\*`([0-9][0-9.,]*)\s+GB/s`\*\*",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["t8192_96mib_polluted_4pass_pinned_gb_s"]),
        0.005,
    ),
    (
        "s116_l2pin_speedup_lo",
        r"`4`-pass polluted `T=8192`,\s+`([0-9][0-9.,]*)×`–`[0-9.]+×`\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["t8192_4pass_pin_speedup"]),
        0.005,
    ),
    (
        "s116_l2pin_speedup_hi",
        r"`4`-pass polluted `T=8192`,\s+`[0-9.]+×`–`([0-9][0-9.,]*)×`\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["t8192_1pass_pin_speedup"]),
        0.005,
    ),
    (
        "s116_weight_in_a_us",
        r"`Weight-in-A`\s+`N=8`\s+`([0-9][0-9.,]*)\s+µs/branch`\*\*",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["weight_in_a_n8_us_per_branch"]),
        0.0005,
    ),
    (
        "s116_weight_in_a_tops",
        r"\(`([0-9][0-9.,]*)\s+TOPS`,\s+`[0-9.]+×`\s+over `Weight-in-B`\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["weight_in_a_n8_tops"]),
        0.005,
    ),
    (
        "s116_weight_in_a_speedup",
        r"TOPS`,\s+`([0-9][0-9.,]*)×`\s+over `Weight-in-B`\)",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["h16c_v2_l2_persistence_and_weight_in_a"]["weight_in_a_vs_b_tops_ratio"]),
        0.005,
    ),
    (
        "s116_rollback_drift",
        r"\*\*`([0-9][0-9.,]*)`\*\*\s+`16 B`\s+\$\\Pi_\{t_0\}\$\s+rollback drift",
        "receipts/s116_phase2b_gpu_results.json",
        lambda r: float(r["phase2b_live_parity"]["cell8warp_2cta_16b_pi_t0_rollback_max_drift"]),
        0.0,
    ),
    (
        "s116_kernel_count",
        r"`(\d+)/10`\s+`0 B` spill",
        "receipts/s116_ptxas_zero_spill_receipt.json",
        lambda r: float(r["kernel_count"]),
        0.0,
    ),
    # Core Techniques & Honest Scope (including all headline prose metrics)
    (
        "core_verify_overhead_min_pct",
        r"overhead to \*\*`([0-9][0-9.,]*)%–[0-9.]+%`\*\*",
        "receipts/s52_sm86_ptx_foundry_results.json",
        _s52_verify_overhead_pct,
        0.05,
    ),
    (
        "core_verify_overhead_max_pct",
        r"overhead to \*\*`[0-9.]+%–([0-9][0-9.,]*)%`\*\*",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["arm_b_sm86_ptx_a16_gpu"]["b2_live_5popc_verify"]["verify_overhead_backbone_pct"]),
        0.05,
    ),
    (
        "core_verify_overhead_min_note_pct",
        r"\(`([0-9][0-9.,]*)%`:\s+`s52_sm86_ptx_foundry_results\.json`",
        "receipts/s52_sm86_ptx_foundry_results.json",
        _s52_verify_overhead_pct,
        0.05,
    ),
    (
        "core_verify_overhead_max_note_pct",
        r";\s+`([0-9][0-9.,]*)%`:\s+`s58_qwen25_live_engine_results\.json`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["arm_b_sm86_ptx_a16_gpu"]["b2_live_5popc_verify"]["verify_overhead_backbone_pct"]),
        0.05,
    ),
    (
        "s58_verify_overhead_backbone_pct",
        r"verify_overhead_backbone_pct = ([0-9][0-9.,]*)`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["arm_b_sm86_ptx_a16_gpu"]["b2_live_5popc_verify"]["verify_overhead_backbone_pct"]),
        0.0005,
    ),
    (
        "s71_mesi_slowdown",
        r"write-write MESI contention\s+\(`([0-9][0-9.,]*)×`\s+slowdown\)",
        "receipts/s71_proximity_avx512_results.json",
        _s71_ww8_slowdown,
        0.005,
    ),
    (
        "s71_read_shared_speedup",
        r"read-shared `64 B` fusion\s+\(`([0-9][0-9.,]*)×`\s+faster per weight;",
        "receipts/s71_proximity_avx512_results.json",
        _s71_ratio_dual_vs_stride8,
        0.005,
    ),
    (
        "s58_prefill_b32_imma_tok_s",
        r"reaches `([0-9][0-9.,]*)\s+tok/s`\s+\(`[0-9.]+×`\s+cuBLAS `BF16`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["prefill_b32_all168_linear"]["arm_b3_a16_imma_tok_s"]),
        0.05,
    ),
    (
        "s58_prefill_b32_ratio",
        r"reaches `[0-9.,]+\s+tok/s`\s+\(`([0-9][0-9.,]*)×`\s+cuBLAS `BF16`",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["prefill_b32_all168_linear"]["speedup_b3_vs_a3"]),
        0.0005,
    ),
    (
        "s58_prefill_b32_bf16_tok_s",
        r"cuBLAS `BF16`\s+`([0-9][0-9.,]*)\s+tok/s`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["prefill_b32_all168_linear"]["arm_a3_bf16_pytorch_tok_s"]),
        0.05,
    ),
    (
        "s58_lm_head_mb",
        r"`BF16` `lm_head`\s+\(`([0-9][0-9.,]*)\s+MB`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: (float(r["model_identity"]["vocab_size"]) * float(r["model_identity"]["hidden_size"]) * 2.0) / 1e6,
        0.005,
    ),
    (
        "s58_linear_matrices_mb",
        r"`168`\s+`4\.5-bpw` `A_16` linear matrices combined\s+\(`([0-9][0-9.,]*)\s+MB`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["model_identity"]["cold_18b_bytes"]) / 1e6,
        0.005,
    ),
    (
        "scope_e2e_vs_backbone_e2e_tok_s",
        r"backbone latency \(`([0-9][0-9.,]*)`\s+vs\.\s+`[0-9.]+\s+tok/s`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_full_e2e_with_bf16_lm_head"]["arm_b1_a16_sm86_tok_s"]),
        0.005,
    ),
    (
        "scope_e2e_vs_backbone_bb_tok_s",
        r"backbone latency \(`[0-9.]+`\s+vs\.\s+`([0-9][0-9.,]*)\s+tok/s`\)",
        "receipts/s58_qwen25_live_engine_results.json",
        lambda r: float(r["pareto_latency_energy_summary"]["decode_backbone_24layers"]["arm_b1_a16_sm86_tok_s"]),
        0.005,
    ),
    (
        "scope_v3_cellop_graph_precise_us",
        r"for CellOp \(`([0-9][0-9.,]*)\s+µs/tok`\)",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["cellop_graph_us_per_tok_8192"]),
        0.00005,
    ),
    (
        "scope_v3_e2e_graph_precise_us",
        r"for E2E \(`([0-9][0-9.,]*)\s+µs/tok`,\s+`[0-9.,]+\s+tok/s`\)",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_graph_us_per_tok_16384"]),
        0.00005,
    ),
    (
        "scope_v3_e2e_graph_precise_tok_s",
        r"for E2E \(`[0-9.]+\s+µs/tok`,\s+`([0-9][0-9.,]*)\s+tok/s`\)",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_graph_tok_per_sec_16384"]),
        0.005,
    ),
    (
        "scope_v3_nongraph_cellop_precise_us",
        r"non-graph timings are `([0-9][0-9.,]*)\s+µs/tok`\s+CellOp",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["cellop_us_per_tok_8192"]),
        0.00005,
    ),
    (
        "scope_v3_nongraph_e2e_precise_us",
        r"CellOp vs\.\s+`([0-9][0-9.,]*)\s+µs/tok`\s+E2E",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_us_per_tok_8192"]),
        0.00005,
    ),
    (
        "scope_v3_nongraph_diff_precise_us",
        r"\(`([0-9][0-9.,]*)\s+µs/tok`\s+difference\)",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["v3_d512"]["e2e_us_per_tok_8192"]) - float(r["v3_d512"]["cellop_us_per_tok_8192"]),
        0.00005,
    ),
    (
        "scope_v3_bwd_tok_s",
        r"`s117_chunk32_bwd_wmma_sm86`\s+\(`([0-9][0-9.,]*)\s+tok/s`",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        lambda r: float(r["chunk32_bwd_wmma"]["B4_T8192_D512_L12_tok_per_sec"]),
        0.005,
    ),
    (
        "s58_sass_total_kernels",
        r"`\d+/(\d+)`\s+kernels `0 B` spill in receipt",
        "receipts/s58_sm86_engine_sass_inspection.json",
        lambda r: float(r["kernel_count"]),
        0.0,
    ),
    (
        "s52_sass_total_kernels",
        r"`\d+/(\d+)`\s+kernels `0 B` spill,\s+`0`\s+verify fails",
        "receipts/s52_sm86_ptx_sass_inspection.json",
        lambda r: float(r["kernels_compiled"]),
        0.0,
    ),
    (
        "intro_su2_zero_spill",
        r"\(`(\d+)/\d+`\s+SU\(2\)",
        "receipts/su2_holonomy_sm86_ptxas_receipt.json",
        lambda r: float(r["zero_spill_kernels"]),
        0.0,
    ),
    (
        "intro_su2_total",
        r"\(`\d+/(\d+)`\s+SU\(2\)",
        "receipts/su2_holonomy_sm86_ptxas_receipt.json",
        lambda r: float(r["kernel_count"]),
        0.0,
    ),
)

_S125_REC = "receipts/s125_avx512_l3_benchmark_results.json"

AVX512_README_FIGURE_SPECS: tuple[FigureSpec, ...] = (
    # Footprint & L3 occupancy
    (
        "s125_arena_mib_heading",
        r"L3-Resident Arena \(`([0-9][0-9.,]*)\s+MiB`\s+=",
        _S125_REC,
        lambda r: float(r["footprint"]["total_live_mib"]),
        0.0005,
    ),
    (
        "s125_l3_occupancy_pct",
        r"L3-Resident Arena \(`[0-9.]+\s+MiB`\s+=\s+`([0-9][0-9.,]*)%`\s+of `24 MiB` L3\)",
        _S125_REC,
        lambda r: float(r["footprint"]["l3_occupancy_pct"]),
        0.0005,
    ),
    (
        "s125_weight_bytes",
        r"holds `([0-9][0-9,]*)\s+B`\s+\(`[0-9.]+\s+MiB`\)\s+synthetic weights",
        _S125_REC,
        lambda r: float(r["footprint"]["weight_bytes"]),
        0.0,
    ),
    (
        "s125_weight_mib",
        r"holds `[0-9,]+\s+B`\s+\(`([0-9][0-9.,]*)\s+MiB`\)\s+synthetic weights",
        _S125_REC,
        lambda r: float(r["footprint"]["weight_mib"]),
        0.0005,
    ),
    (
        "s125_state_bytes",
        r"\+ `([0-9][0-9,]*)\s+B`\s+\(`[0-9.]+\s+MiB`\)\s+\$O\(1\)\$",
        _S125_REC,
        lambda r: float(r["footprint"]["state_bytes"]),
        0.0,
    ),
    (
        "s125_state_mib",
        r"\+ `[0-9,]+\s+B`\s+\(`([0-9][0-9.,]*)\s+MiB`\)\s+\$O\(1\)\$",
        _S125_REC,
        lambda r: float(r["footprint"]["state_mib"]),
        0.0005,
    ),
    (
        "s125_total_live_bytes",
        r"= \*\*`([0-9][0-9,]*)\s+B`\s+\(`[0-9.]+\s+MiB`\)\*\*",
        _S125_REC,
        lambda r: float(r["footprint"]["total_live_bytes"]),
        0.0,
    ),
    (
        "s125_total_live_mib",
        r"= \*\*`[0-9,]+\s+B`\s+\(`([0-9][0-9.,]*)\s+MiB`\)\*\*",
        _S125_REC,
        lambda r: float(r["footprint"]["total_live_mib"]),
        0.0005,
    ),
    # ISA counts (scope bullet & table)
    (
        "s125_objdump_vpdpbusd_scope",
        r"instruction counts \(`vpdpbusd=(\d+)`,",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpdpbusd_count"]),
        0.0,
    ),
    (
        "s125_objdump_vpopcntq_scope",
        r"instruction counts \(`vpdpbusd=\d+`, `vpopcntq=(\d+)`,",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpopcntq_count"]),
        0.0,
    ),
    (
        "s125_objdump_vpermt2d_scope",
        r"instruction counts \(`vpdpbusd=\d+`, `vpopcntq=\d+`, `vpermt2d=(\d+)`\)",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpermt2d_count"]),
        0.0,
    ),
    (
        "s125_objdump_vpdpbusd_table",
        r"`objdump -d`: `vpdpbusd=(\d+)`,",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpdpbusd_count"]),
        0.0,
    ),
    (
        "s125_objdump_vpopcntq_table",
        r"`objdump -d`: `vpdpbusd=\d+`, `vpopcntq=(\d+)`,",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpopcntq_count"]),
        0.0,
    ),
    (
        "s125_objdump_vpermt2d_table",
        r"`objdump -d`: `vpdpbusd=\d+`, `vpopcntq=\d+`, `vpermt2d=(\d+)`",
        _S125_REC,
        lambda r: float(r["build_and_isa"]["objdump_vpermt2d_count"]),
        0.0,
    ),
    # FP64 parity residuals
    (
        "s125_fp64_logit_diff",
        r"`max_abs_logit_diff = ([0-9.]+e-[0-9]+)`",
        _S125_REC,
        lambda r: float(r["parity_and_ecc"]["fp64_oracle_max_abs_logit_diff"]),
        5e-13,
    ),
    (
        "s125_fp64_hidden_diff",
        r"`max_abs_hidden_diff = ([0-9.]+e-[0-9]+)`",
        _S125_REC,
        lambda r: float(r["parity_and_ecc"]["fp64_oracle_max_abs_hidden_diff"]),
        5e-13,
    ),
    # 12-run A/A control noise floor (1T)
    (
        "s125_aa_1t_median",
        r"`1T`: median `([0-9][0-9.,]*)\s+tok/s`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["median_tok_per_s"]),
        0.005,
    ),
    (
        "s125_aa_1t_min",
        r"`1T`: median `[0-9.,]+\s+tok/s` \(min–max `([0-9][0-9.,]*)–[0-9.,]+`",
        _S125_REC,
        lambda r: min(float(x) for x in r["aa_noise_floor"]["avx512_1t_4k"]["tok_per_s_runs"]),
        0.005,
    ),
    (
        "s125_aa_1t_max",
        r"`1T`: median `[0-9.,]+\s+tok/s` \(min–max `[0-9.,]+–([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: max(float(x) for x in r["aa_noise_floor"]["avx512_1t_4k"]["tok_per_s_runs"]),
        0.005,
    ),
    (
        "s125_aa_1t_stdev",
        r"`1T`: median `[0-9.,]+\s+tok/s` \(min–max `[0-9.,]+–[0-9.,]+`, `stdev=([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["stdev_tok_per_s"]),
        0.005,
    ),
    (
        "s125_aa_1t_cv",
        r"`1T`: median `[0-9.,]+\s+tok/s` [^;]+`CV=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["cv_pct"]),
        0.0005,
    ),
    (
        "s125_aa_1t_iqr",
        r"`1T`: median `[0-9.,]+\s+tok/s` [^;]+`IQR=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["iqr_pct"]),
        0.0005,
    ),
    (
        "s125_aa_1t_spread",
        r"`1T`: median `[0-9.,]+\s+tok/s` [^;]+`max_rel_spread=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["max_rel_spread_pct"]),
        0.0005,
    ),
    (
        "s125_aa_1t_floor",
        r"`1T`: median `[0-9.,]+\s+tok/s` [^;]+floor \*\*`([0-9][0-9.,]*)%`\*\*",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_1t_4k"]["noise_floor_pct"]),
        0.0005,
    ),
    # 12-run A/A control noise floor (8T)
    (
        "s125_aa_8t_median",
        r"`8T`: median `([0-9][0-9.,]*)\s+tok/s`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["median_tok_per_s"]),
        0.005,
    ),
    (
        "s125_aa_8t_min",
        r"`8T`: median `[0-9.,]+\s+tok/s` \(min–max `([0-9][0-9.,]*)–[0-9.,]+`",
        _S125_REC,
        lambda r: min(float(x) for x in r["aa_noise_floor"]["avx512_8t_4k"]["tok_per_s_runs"]),
        0.005,
    ),
    (
        "s125_aa_8t_max",
        r"`8T`: median `[0-9.,]+\s+tok/s` \(min–max `[0-9.,]+–([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: max(float(x) for x in r["aa_noise_floor"]["avx512_8t_4k"]["tok_per_s_runs"]),
        0.005,
    ),
    (
        "s125_aa_8t_stdev",
        r"`8T`: median `[0-9.,]+\s+tok/s` \(min–max `[0-9.,]+–[0-9.,]+`, `stdev=([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["stdev_tok_per_s"]),
        0.005,
    ),
    (
        "s125_aa_8t_cv",
        r"`8T`: median `[0-9.,]+\s+tok/s` [^|]+`CV=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["cv_pct"]),
        0.0005,
    ),
    (
        "s125_aa_8t_iqr",
        r"`8T`: median `[0-9.,]+\s+tok/s` [^|]+`IQR=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["iqr_pct"]),
        0.0005,
    ),
    (
        "s125_aa_8t_spread",
        r"`8T`: median `[0-9.,]+\s+tok/s` [^|]+`max_rel_spread=([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["max_rel_spread_pct"]),
        0.0005,
    ),
    (
        "s125_aa_8t_floor",
        r"`8T`: median `[0-9.,]+\s+tok/s` [^|]+floor \*\*`([0-9][0-9.,]*)%`\*\*",
        _S125_REC,
        lambda r: float(r["aa_noise_floor"]["avx512_8t_4k"]["noise_floor_pct"]),
        0.0005,
    ),
    # G2 scalar vs avx512 (1T)
    (
        "s125_g2_1t_ctl_med",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| `([0-9][0-9.,]*) ->",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["ctl_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_g2_1t_cand_med",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| `[0-9.,]+ -> ([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["cand_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_g2_1t_speedup",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+ = \*\*`([0-9][0-9.,]*)x`\*\*",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["median_speedup"]),
        0.00005,
    ),
    (
        "s125_g2_1t_delta_pct",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+\(`\+([0-9][0-9.,]*)%` median",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_1t_paired_pct",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+`\+([0-9][0-9.,]*)%` paired",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["median_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_1t_paired_min",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+range `\+([0-9][0-9.,]*)%\.\.",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["min_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_1t_paired_max",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+\.\.\+([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["max_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_1t_cand_min",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+`avx512` runs `([0-9][0-9.,]*)–[0-9.,]+ tok/s`",
        _S125_REC,
        lambda r: min(_s125_cand_runs(r, "scalar_vs_avx512_1t_4k")),
        0.005,
    ),
    (
        "s125_g2_1t_cand_max",
        r"`G2` `scalar` vs `avx512` \(`1T`, `4 KiB`, `6` triples\) \| [^|]+`avx512` runs `[0-9.,]+–([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: max(_s125_cand_runs(r, "scalar_vs_avx512_1t_4k")),
        0.005,
    ),
    # G2 scalar vs avx512 (8T)
    (
        "s125_g2_8t_ctl_med",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| `([0-9][0-9.,]*) ->",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["ctl_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_g2_8t_cand_med",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| `[0-9.,]+ -> ([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["cand_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_g2_8t_speedup",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+ = \*\*`([0-9][0-9.,]*)x`\*\*",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["median_speedup"]),
        0.00005,
    ),
    (
        "s125_g2_8t_delta_pct",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+\(`\+([0-9][0-9.,]*)%` median",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_8t_paired_pct",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+`\+([0-9][0-9.,]*)%` paired",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["median_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_8t_paired_min",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+range `\+([0-9][0-9.,]*)%\.\.",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["min_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_8t_paired_max",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+\.\.\+([0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["max_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_g2_8t_cand_min",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+`avx512` runs `([0-9][0-9.,]*)–[0-9.,]+ tok/s`",
        _S125_REC,
        lambda r: min(_s125_cand_runs(r, "scalar_vs_avx512_8t_4k")),
        0.005,
    ),
    (
        "s125_g2_8t_cand_max",
        r"`G2` `scalar` vs `avx512` \(`8T`, `4 KiB`, `6` triples\) \| [^|]+`avx512` runs `[0-9.,]+–([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: max(_s125_cand_runs(r, "scalar_vs_avx512_8t_4k")),
        0.005,
    ),
    (
        "s125_g2_8t_ctl_min",
        r"vs `scalar` `([0-9][0-9.,]*)–[0-9.,]+ tok/s`",
        _S125_REC,
        lambda r: min(_s125_ctl_runs(r, "scalar_vs_avx512_8t_4k")),
        0.005,
    ),
    (
        "s125_g2_8t_ctl_max",
        r"vs `scalar` `[0-9.,]+–([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: max(_s125_ctl_runs(r, "scalar_vs_avx512_8t_4k")),
        0.005,
    ),
    # 4 KiB vs 2 MiB (1T)
    (
        "s125_hp_1t_ctl_med",
        r"trace\) \(`1T`, `6` triples\) \| `([0-9][0-9.,]*) ->",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["ctl_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_hp_1t_cand_med",
        r"trace\) \(`1T`, `6` triples\) \| `[0-9.,]+ -> ([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["cand_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_hp_1t_delta_pct",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+\(`(-?[0-9][0-9.,]*)%` median",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_1t_paired_pct",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+`\+?(-?[0-9][0-9.,]*)%` paired",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["median_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_1t_paired_min",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+range `(-?[0-9][0-9.,]*)%\.\.",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["min_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_1t_paired_max",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+\.\.\+?(-?[0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_1t_avx512"]["max_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_1t_ctl_min",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+`4 KiB` `([0-9][0-9.,]*)–[0-9.,]+`",
        _S125_REC,
        lambda r: min(_s125_ctl_runs(r, "pages_4k_vs_2m_1t_avx512")),
        0.005,
    ),
    (
        "s125_hp_1t_ctl_max",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+`4 KiB` `[0-9.,]+–([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: max(_s125_ctl_runs(r, "pages_4k_vs_2m_1t_avx512")),
        0.005,
    ),
    (
        "s125_hp_1t_cand_min",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+`2 MiB` `([0-9][0-9.,]*)–[0-9.,]+ tok/s`",
        _S125_REC,
        lambda r: min(_s125_cand_runs(r, "pages_4k_vs_2m_1t_avx512")),
        0.005,
    ),
    (
        "s125_hp_1t_cand_max",
        r"trace\) \(`1T`, `6` triples\) \| [^|]+`2 MiB` `[0-9.,]+–([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: max(_s125_cand_runs(r, "pages_4k_vs_2m_1t_avx512")),
        0.005,
    ),
    # 4 KiB vs 2 MiB (8T)
    (
        "s125_hp_8t_ctl_med",
        r"trace\) \(`8T`, `6` triples\) \| `([0-9][0-9.,]*) ->",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["ctl_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_hp_8t_cand_med",
        r"trace\) \(`8T`, `6` triples\) \| `[0-9.,]+ -> ([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["cand_median_tok_per_s"]),
        0.0005,
    ),
    (
        "s125_hp_8t_delta_pct",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+\(`(-?[0-9][0-9.,]*)%` median",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_8t_paired_pct",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+`\+?(-?[0-9][0-9.,]*)%` paired",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["median_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_8t_paired_min",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+range `(-?[0-9][0-9.,]*)%\.\.",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["min_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_8t_paired_max",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+\.\.\+?(-?[0-9][0-9.,]*)%`",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["pages_4k_vs_2m_8t_avx512"]["max_paired_delta_pct"]),
        0.005,
    ),
    (
        "s125_hp_8t_ctl_min",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+`4 KiB` `([0-9][0-9.,]*)–[0-9.,]+`",
        _S125_REC,
        lambda r: min(_s125_ctl_runs(r, "pages_4k_vs_2m_8t_avx512")),
        0.005,
    ),
    (
        "s125_hp_8t_ctl_max",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+`4 KiB` `[0-9.,]+–([0-9][0-9.,]*)`",
        _S125_REC,
        lambda r: max(_s125_ctl_runs(r, "pages_4k_vs_2m_8t_avx512")),
        0.005,
    ),
    (
        "s125_hp_8t_cand_min",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+`2 MiB` `([0-9][0-9.,]*)–[0-9.,]+ tok/s`",
        _S125_REC,
        lambda r: min(_s125_cand_runs(r, "pages_4k_vs_2m_8t_avx512")),
        0.005,
    ),
    (
        "s125_hp_8t_cand_max",
        r"trace\) \(`8T`, `6` triples\) \| [^|]+`2 MiB` `[0-9.,]+–([0-9][0-9.,]*) tok/s`",
        _S125_REC,
        lambda r: max(_s125_cand_runs(r, "pages_4k_vs_2m_8t_avx512")),
        0.005,
    ),
    # Prose takeaway speedups
    (
        "s125_prose_1t_speedup",
        r"delivers `([0-9][0-9.,]*)x` \(`1T`\)",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_1t_4k"]["median_speedup"]),
        0.005,
    ),
    (
        "s125_prose_8t_speedup",
        r"`1T`\) / `([0-9][0-9.,]*)x` \(`8T`\) speedup over scalar C11",
        _S125_REC,
        lambda r: float(r["interleaved_benchmarks"]["scalar_vs_avx512_8t_4k"]["median_speedup"]),
        0.005,
    ),
    (
        "s125_seed_intro",
        r"synthetic weights \(`seed=(\d+)`\), not trained",
        _S125_REC,
        lambda r: float(r["seed"]),
        0.0,
    ),
    (
        "s125_seed_sec2",
        r"synthetic weights \(`seed=(\d+)`, not trained",
        _S125_REC,
        lambda r: float(r["seed"]),
        0.0,
    ),
    (
        "s125_mmap_rounded_b",
        r"length up to `2 MiB` \(`([0-9,]+)\s+B`\s+=",
        _S125_REC,
        lambda r: float(((int(r["footprint"]["total_live_bytes"]) + 2097151) // 2097152) * 2097152),
        0.0,
    ),
    (
        "s125_mmap_rounded_mib",
        r"length up to `2 MiB` \(`[0-9,]+\s+B`\s+=\s+`([0-9.]+)\s+MiB`\)",
        _S125_REC,
        lambda r: float(((int(r["footprint"]["total_live_bytes"]) + 2097151) // 2097152) * 2097152) / (1024.0 * 1024.0),
        0.0005,
    ),
    (
        "s125_mmap_res_b",
        r"inside a `([0-9,]+)\s+B`\s+\(`[0-9.]+\s+MiB`\)\s+virtual `mmap`",
        _S125_REC,
        lambda r: float((((int(r["footprint"]["total_live_bytes"]) + 2097151) // 2097152) * 2097152) + 2 * 1024 * 1024),
        0.0,
    ),
    (
        "s125_mmap_res_mib",
        r"inside a `[0-9,]+\s+B`\s+\(`([0-9.]+)\s+MiB`\)\s+virtual `mmap`",
        _S125_REC,
        lambda r: float((((int(r["footprint"]["total_live_bytes"]) + 2097151) // 2097152) * 2097152) + 2 * 1024 * 1024) / (1024.0 * 1024.0),
        0.0005,
    ),
)

_PIN_LINE_RE = re.compile(
    r"^\s*-\s+`(receipts/[A-Za-z0-9_.-]+\.json)`:\s+`([0-9a-f]{64})`"
    r"(?:\s+\(canonical `receipt_sha256`:\s+`([0-9a-f]{64})`\))?\s*$",
    re.MULTILINE,
)


def verify_readme_policy(raw_text: str, visible_text: str, normalized_text: str, doc_name: str) -> None:
    """Reject forbidden energy figures, FP16-equiv throughput claims, HTML entities, raw field names, and removed rows."""
    views = tuple(dict.fromkeys((raw_text, visible_text, normalized_text)))
    for rule_id, compiled in _COMPILED_FORBIDDEN_PATTERNS:
        for view in views:
            hit = compiled.search(view)
            require_ok(
                hit is None,
                "ERR_README_POLICY",
                f"Forbidden pattern {rule_id!r} matched in {doc_name}: {hit.group(0) if hit else ''!r}",
            )


def _verify_specs_on_text(
    repo_root: Path,
    visible_text: str,
    specs: Sequence[FigureSpec],
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for spec_id, pattern, receipt_rel, extractor, tol in specs:
        matches = list(re.finditer(pattern, visible_text))
        require_ok(
            len(matches) == 1,
            "ERR_README_FIGURE_COUNT",
            f"Expected exactly 1 match for README figure {spec_id!r}, found {len(matches)}",
        )
        found_val = _parse_number(matches[0].group(1))
        if receipt_rel not in cache:
            cache[receipt_rel] = load_strict_json(repo_root / receipt_rel)
        try:
            expected_val = float(extractor(cache[receipt_rel]))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError, StopIteration, ZeroDivisionError) as exc:
            raise ReadmeFigureError("ERR_BAD_RECEIPT", f"Unable to extract {spec_id!r} from {receipt_rel}") from exc
        require_ok(
            math.isfinite(expected_val) and abs(found_val - expected_val) <= tol,
            "ERR_README_FIGURE",
            f"README figure {spec_id!r} mismatch: readme={found_val} vs receipt={expected_val} (tol={tol})",
        )
        checked.append({"id": spec_id, "readme": found_val, "receipt": expected_val})
    return checked


def verify_cross_receipt_prose_counts(
    repo_root: Path,
    root_norm: str,
    cache: dict[str, dict[str, Any]],
) -> int:
    """Verify aggregate kernel counts (23/23, 50/50) and Lean theorem count (20/20 + SHA-256) in README.md."""
    for rel in (
        "receipts/su2_holonomy_sm86_ptxas_receipt.json",
        "receipts/s116_ptxas_zero_spill_receipt.json",
        "receipts/s117_ptxas_zero_spill_receipt.json",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        "receipts/s71_proximity_avx512_results.json",
    ):
        if rel not in cache:
            cache[rel] = load_strict_json(repo_root / rel)

    n_su2 = int(cache["receipts/su2_holonomy_sm86_ptxas_receipt.json"]["zero_spill_kernels"])
    n_s116 = int(cache["receipts/s116_ptxas_zero_spill_receipt.json"]["kernel_count"])
    n_s117 = int(cache["receipts/s117_ptxas_zero_spill_receipt.json"]["zero_spill_kernels"])
    n_v3 = int(cache["receipts/v3_frontier_ptxas_zero_spill_receipt.json"]["total_kernels"])
    n_holo_new = n_s116 + n_s117 + n_v3
    n_holo_total = n_su2 + n_holo_new

    lean_raw = read_regular_bytes(repo_root / "lean" / "ProximityBarrierSilicon.lean")
    lean_sha = hashlib.sha256(lean_raw).hexdigest()
    require_ok(
        lean_sha == cache["receipts/s71_proximity_avx512_results.json"].get("lean_source_sha256"),
        "ERR_LEAN_SHA",
        "lean/ProximityBarrierSilicon.lean SHA-256 does not match s71 receipt",
    )
    n_lean = len(_LEAN_THEOREM_RE.findall(_decode_utf8(lean_raw, "lean/ProximityBarrierSilicon.lean")))

    ratio_specs: tuple[tuple[str, str, int], ...] = (
        ("intro_holo_50", r"; all `(\d+)/(\d+)` HoloSphere `sm_86` kernels", n_holo_total),
        ("intro_holo_23", r"\+ `(\d+)/(\d+)` `S116`/`S117`/`V3`\)", n_holo_new),
        ("sec_holo_23", r"kernel suites add \*\*`(\d+)/(\d+)`\*\* additional", n_holo_new),
        ("sec_holo_50", r"\(\*\*`(\d+)/(\d+)`\*\* total HoloSphere", n_holo_total),
        ("verify_holo_50", r"Verify all `(\d+)/(\d+)` HoloSphere zero-spill", n_holo_total),
        ("scope_holo_50", r"across all `(\d+)/(\d+)` HoloSphere `sm_86` kernels", n_holo_total),
        ("scope_holo_23", r"kernels \(`(\d+)/(\d+)` `S116`/`S117`/`V3` \+", n_holo_new),
        ("s71_lean_20", r"`(\d+)/(\d+)` pure-`Init` `Lean 4\.36\.0` proofs", n_lean),
    )
    for spec_id, pattern, expected in ratio_specs:
        matches = list(re.finditer(pattern, root_norm))
        require_ok(
            len(matches) == 1,
            "ERR_README_FIGURE_COUNT",
            f"Expected exactly 1 match for aggregate prose count {spec_id!r}, found {len(matches)}",
        )
        num = int(matches[0].group(1))
        den = int(matches[0].group(2))
        require_ok(
            num == expected and den == expected,
            "ERR_README_FIGURE",
            f"Aggregate prose count {spec_id!r} mismatch: {num}/{den} vs {expected}/{expected}",
        )
    return len(ratio_specs)


def verify_readme_receipt_pins(repo_root: Path, visible_text: str, cache: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Parse all 8 b3712675 receipt file SHA-256 pins from visible README.md text and check them against disk."""
    parsed_file_pins: dict[str, str] = {}
    parsed_canonical_pins: dict[str, str] = {}
    for m in _PIN_LINE_RE.finditer(visible_text):
        rel, file_sha, canon_sha = m.group(1), m.group(2), m.group(3)
        require_ok(rel not in parsed_file_pins, "ERR_PIN_DUP", f"Duplicate pin line in README.md for {rel}")
        parsed_file_pins[rel] = file_sha
        if canon_sha is not None:
            parsed_canonical_pins[rel] = canon_sha

    require_ok(
        frozenset(parsed_file_pins.keys()) == EXPECTED_EIGHT_PIN_FILES,
        "ERR_PIN_SET",
        f"README.md receipt pin set mismatch: got {len(parsed_file_pins)} pins",
    )
    require_ok(
        frozenset(parsed_canonical_pins.keys()) == EXPECTED_CANONICAL_MLOS_FILES,
        "ERR_CANON_PIN_SET",
        f"README.md canonical ML-OS receipt_sha256 pin set mismatch: got {len(parsed_canonical_pins)} pins",
    )

    for rel in sorted(EXPECTED_EIGHT_PIN_FILES):
        raw = read_regular_bytes(repo_root / rel)
        live_sha = hashlib.sha256(raw).hexdigest()
        require_ok(
            parsed_file_pins[rel] == live_sha,
            "ERR_PIN_MISMATCH",
            f"README.md SHA-256 pin for {rel} does not match disk ({parsed_file_pins[rel]} != {live_sha})",
        )
    for rel in sorted(EXPECTED_CANONICAL_MLOS_FILES):
        if rel not in cache:
            cache[rel] = load_strict_json(repo_root / rel)
        require_ok(
            cache[rel].get("receipt_sha256") == parsed_canonical_pins[rel],
            "ERR_CANON_PIN_MISMATCH",
            f"README.md canonical receipt_sha256 for {rel} does not match receipt JSON",
        )
    return parsed_file_pins


def verify_readme_figures(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Parse README.md and cpu/avx512/README.md and fail-closed verify every quoted figure, policy rule, and receipt pin."""
    root_raw = _decode_utf8(read_regular_bytes(repo_root / "README.md"), "README.md")
    root_visible = _strip_html_comments(root_raw, "README.md")
    root_norm = _normalize_markdown_prose(root_visible)
    verify_readme_policy(root_raw, root_visible, root_norm, "README.md")

    avx_raw = _decode_utf8(read_regular_bytes(repo_root / "cpu" / "avx512" / "README.md"), "cpu/avx512/README.md")
    avx_visible = _strip_html_comments(avx_raw, "cpu/avx512/README.md")
    avx_norm = _normalize_markdown_prose(avx_visible)
    verify_readme_policy(avx_raw, avx_visible, avx_norm, "cpu/avx512/README.md")

    cache: dict[str, dict[str, Any]] = {}
    root_checked = _verify_specs_on_text(repo_root, root_norm, README_FIGURE_SPECS, cache)
    avx_checked = _verify_specs_on_text(repo_root, avx_norm, AVX512_README_FIGURE_SPECS, cache)
    cross_checked = verify_cross_receipt_prose_counts(repo_root, root_norm, cache)
    pins = verify_readme_receipt_pins(repo_root, root_norm, cache)

    payload = {
        "schema": "silikernels.readme_figure_check.v2",
        "status": "PASS",
        "figures_checked": len(root_checked),
        "avx512_figures_checked": len(avx_checked),
        "cross_receipt_counts_checked": cross_checked,
        "total_figures_checked": len(root_checked) + len(avx_checked) + cross_checked,
        "receipt_pins_verified": len(pins),
        "receipts_loaded": len(cache),
    }
    payload["summary_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _safe_write_stderr(text: str) -> None:
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse README.md and cpu/avx512/README.md and verify every figure and pin against receipts/.")
    parser.add_argument("--check", action="store_true", help="Run fail-closed README figure verification.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON summary.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path.")
    args = parser.parse_args(argv)
    try:
        res = verify_readme_figures(args.repo_root)
    except (ReadmeFigureError, UnicodeDecodeError, OSError, RecursionError, OverflowError) as exc:
        msg = str(exc) if isinstance(exc, ReadmeFigureError) else f"[ERR_IO] {type(exc).__name__}"
        _safe_write_stderr(f"FAIL {msg}\n")
        return 2
    out_text = (
        json.dumps(res, sort_keys=True, indent=2) + "\n"
        if args.json
        else (
            f"PASS silikernels readme_figures={res['figures_checked']} "
            f"avx512_figures={res['avx512_figures_checked']} "
            f"cross_counts={res['cross_receipt_counts_checked']} "
            f"receipt_pins={res['receipt_pins_verified']} "
            f"summary_sha256={res['summary_sha256']}\n"
        )
    )
    try:
        sys.stdout.write(out_text)
        sys.stdout.flush()
    except OSError:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
