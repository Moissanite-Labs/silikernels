#!/usr/bin/env python3
"""Pure-CPU & hygiene unit tests for SiliKernels V2 contracts.

Strictly CPU-only (CUDA_VISIBLE_DEVICES="", zero PyTorch required, pure stdlib).
Zero bare asserts, bit-identical output across 3x -I modes (-B, -O -B, -B -W error).
"""

from __future__ import annotations

import ast
import ctypes
import json
import os
from pathlib import Path
import random
import re
import struct
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def _secded_syn_w64_py(w64: int) -> int:
    w0 = w64 & 0xFFFFFFFF
    w1 = (w64 >> 32) & 0xFFFFFFFF
    px = (w0 ^ w1) & 0x11111111
    p0 = px.bit_count() & 1
    s0 = (px & 0x10101010).bit_count() & 1
    s1 = (px & 0x11001100).bit_count() & 1
    s2 = (px & 0x11110000).bit_count() & 1
    s3 = (w1 & 0x11111111).bit_count() & 1
    return p0 | (s0 << 1) | (s1 << 2) | (s2 << 3) | (s3 << 4)


def _check_hygiene(failures: list[str]) -> int:
    slash = chr(47)
    rx_home = re.compile("(?:" + slash + "ho" + "me" + slash + "|" + slash + "Us" + "ers" + slash + r")[A-Za-z0-9._-]+")
    rx_secret = re.compile(r"(?:hf_[A-Za-z0-9]{20}|ghp_[A-Za-z0-9]{20}|gho_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16})")
    rx_internal = re.compile(
        r"(?:(?i:\btrack[-_ ]?s\d+[a-z0-9_-]*\b)"
        r"|\bEXP-Q\d+[A-Za-z0-9_-]*\b"
        r"|\bKG-[A-Z0-9][A-Z0-9_-]*\b"
        r"|\bSOP-\d+[A-Za-z0-9-]*\b"
        r"|\bCS-\d+[A-Z0-9-]*\b"
        r"|\bAUDIT-(?:S\d+|\d+)[A-Za-z0-9._-]*\b"
        r"|\bP2[23][0-9][a-z]*\b"
        r"|\bS97\b|\bS164\b|\bS18[0-9][a-z0-9]*\b"
        r"|expedi" + "tion" + slash + "track-"
        r"|Proj" + "ects" + slash + "expedi" + "tion"
        r"|(?<![/A-Za-z0-9_])#[345]\d{3}\b)"
    )
    v2_paths = [
        REPO_ROOT / "Makefile",
        REPO_ROOT / "README.md",
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "codeql.yml",
        REPO_ROOT / "include" / "silikernels_v2_contracts.h",
        REPO_ROOT / "cpu" / "avx512" / "a16_v2_avx512_swar.c",
        REPO_ROOT / "cuda" / "sm86" / "a16_v2_ampere.mk",
        REPO_ROOT / "cuda" / "sm86" / "a16_v2_ampere_kernels.cuh",
        REPO_ROOT / "cuda" / "sm86" / "a16_v2_ampere_kernels.cu",
        REPO_ROOT / "cuda" / "sm86" / "a16_v2_ampere_bench.cu",
        REPO_ROOT / "verify" / "verify_v2_sass_and_cpu.py",
        REPO_ROOT / "tests" / "test_silikernels_v2_cpu.py",
    ]
    checked = 0
    for p in v2_paths:
        _check(p.exists(), f"missing_v2_file:{p.name}", failures)
        if not p.exists():
            continue
        _check(not p.is_symlink(), f"symlink_forbidden:{p.name}", failures)
        raw = p.read_bytes()
        _check(b"\r" not in raw, f"crlf_forbidden:{p.name}", failures)
        txt = raw.decode("utf-8")
        _check(rx_home.search(txt) is None, f"home_leak:{p.name}", failures)
        _check(rx_secret.search(txt) is None, f"secret_leak:{p.name}", failures)
        if p.name == "README.md":
            # Scan V2-added README blocks strictly while preserving pre-existing repo notices on main
            m_sec = re.search(r"## V2 Zero-Spill.*?## Core Low-Level", txt, flags=re.S)
            m_law4 = re.search(r"\*\*V2 \(`a16_v2_\*`\).*?## License", txt, flags=re.S)
            _check(m_sec is not None and m_law4 is not None, "README missing V2 sections", failures)
            if m_sec is not None:
                _check(rx_internal.search(m_sec.group(0)) is None, "internal_id_leak:README.md:V2_sec", failures)
            if m_law4 is not None:
                _check(rx_internal.search(m_law4.group(0)) is None, "internal_id_leak:README.md:V2_law4", failures)
        else:
            _check(rx_internal.search(txt) is None, f"internal_id_leak:{p.name}", failures)
        if p.suffix == ".py":
            tree = ast.parse(txt, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    failures.append(f"bare_assert:{p.name}:{getattr(node, 'lineno', 0)}")
        checked += 1
    return checked


def main() -> int:
    failures: list[str] = []
    rng = random.Random(20261128)

    so_path = BUILD_DIR / "liba16_v2_portable.so"
    _check(so_path.exists(), f"missing {so_path}", failures)
    if so_path.exists():
        lib = ctypes.CDLL(str(so_path))
        lib.silikernels_v2_swar127_in_domain_u64.argtypes = [ctypes.c_uint64]
        lib.silikernels_v2_swar127_in_domain_u64.restype = ctypes.c_int
        lib.silikernels_v2_swar127_negate_u64.argtypes = [ctypes.c_uint64]
        lib.silikernels_v2_swar127_negate_u64.restype = ctypes.c_uint64
        lib.silikernels_v2_should_dispatch_cpu.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.silikernels_v2_should_dispatch_cpu.restype = ctypes.c_int
        lib.silikernels_v2_canonical_tree_reduce_128.argtypes = [ctypes.POINTER(ctypes.c_float)]
        lib.silikernels_v2_canonical_tree_reduce_128.restype = ctypes.c_float

        # Exhaustive single-byte SWAR127 domain scan [0..255] embedded in lane 3
        for bval in range(256):
            word = 0x0101010101010101 ^ (0x01 << 24) ^ ((bval & 0xFF) << 24)
            ok = lib.silikernels_v2_swar127_in_domain_u64(word)
            exp_ok = 1 if (1 <= bval <= 127) else 0
            _check(ok == exp_ok, f"SWAR127 byte={bval} domain mismatch", failures)
            neg = lib.silikernels_v2_swar127_negate_u64(word)
            if exp_ok == 1:
                got_byte = (neg >> 24) & 0xFF
                exp_neg = (-bval) & 0xFF
                _check(got_byte == exp_neg, f"SWAR127 neg byte={bval} mismatch", failures)
            else:
                _check(neg == 0, f"SWAR127 OOB byte={bval} failed to halt", failures)

        # B*=2 and B*=3 dispatch threshold checks
        _check(lib.silikernels_v2_should_dispatch_cpu(1, 0) == 1, "B*=2 accepts B=1", failures)
        _check(lib.silikernels_v2_should_dispatch_cpu(2, 0) == 0, "B*=2 rejects B=2", failures)
        _check(lib.silikernels_v2_should_dispatch_cpu(2, 1) == 1, "B*=3 accepts B=2 for 4.5bpw", failures)
        _check(lib.silikernels_v2_should_dispatch_cpu(3, 1) == 0, "B*=3 rejects B=3 for 4.5bpw", failures)

        # Sign-symmetry of 0-ULP tree reduction (negating all inputs negates bit pattern exactly)
        arr_t = ctypes.c_float * 128
        xs = [float(struct.unpack("<f", struct.pack("<f", rng.gauss(0.0, 2.0)))[0]) for _ in range(128)]
        s_pos = float(lib.silikernels_v2_canonical_tree_reduce_128(arr_t(*xs)))
        s_neg = float(lib.silikernels_v2_canonical_tree_reduce_128(arr_t(*[-v for v in xs])))
        _check(s_pos == -s_neg, "canonical tree reduce odd sign symmetry", failures)

    # Pure-Python SECDED-64 single-bit & double-bit syndrome separation check
    info_pos = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
    for trial in range(8):
        u11 = rng.randrange(2048)
        b = [0] * 16
        for j, pos in enumerate(info_pos):
            b[pos] = (u11 >> j) & 1
        b[1] = b[3] ^ b[5] ^ b[7] ^ b[9] ^ b[11] ^ b[13] ^ b[15]
        b[2] = b[3] ^ b[6] ^ b[7] ^ b[10] ^ b[11] ^ b[14] ^ b[15]
        b[4] = b[5] ^ b[6] ^ b[7] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
        b[8] = b[9] ^ b[10] ^ b[11] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
        b[0] = sum(b[1:]) & 1
        w64 = 0
        for idx in range(16):
            nib = (rng.randrange(8) * 2 + b[idx]) & 0xF
            w64 |= nib << (4 * idx)
        _check(_secded_syn_w64_py(w64) == 0, f"SECDED clean word {trial}", failures)
        p = trial % 16
        syn1 = _secded_syn_w64_py(w64 ^ (1 << (4 * p)))
        _check((syn1 & 1) == 1 and (syn1 >> 1) == p, f"SECDED SEC {trial}", failures)
        syn2 = _secded_syn_w64_py(w64 ^ (1 << (4 * p)) ^ (1 << (4 * ((p + 3) % 16))))
        _check(syn2 != 0 and (syn2 & 1) == 0, f"SECDED DED {trial}", failures)

    files_checked = _check_hygiene(failures)
    out = {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": files_checked,
        "swar127_domain_exhausted": 256,
        "secded_trials": 8,
        "failures": failures,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
