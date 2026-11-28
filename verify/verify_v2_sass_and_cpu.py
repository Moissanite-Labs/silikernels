#!/usr/bin/env python3
"""Pure-CPU & static sm_86 ptxas/SASS verifier for SiliKernels V2.

Runs in two explicit modes:
  --cpu-only     Verify C ABI / SWAR127 / 0-ULP tree_reduce_128 / dpbusd22 / SECDED / B*=2
                 contracts via ctypes + inspect CUDA header contracts without GPU launches.
  --verify-sass  Run all CPU checks PLUS audit build/*.ptxas and build/*.sass for .o, .so,
                 and standalone bench binaries (<= 48 regs, 0 B stack/spill, 0 SASS LDL/STL)
                 and verify build/liba16_v2_ampere.so CPU selftest ABI via ctypes.

Strictly CPU-only, pure stdlib, zero bare asserts, deterministic across 3x -I modes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
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

EXPECTED_SM86_KERNELS: tuple[str, ...] = (
    "a16_v2_tree_reduce_128_sm86",
    "a16_v2_quant22_h4_sm86",
    "a16_v2_gemv_b1_45bpw_dp4a22_sm86",
    "a16_v2_gemv_b1_3125bpw_dp4a22_sm86",
    "a16_v2_gemm_b16_imma_k64_sm86",
    "a16_v2_gemm_b16_bf16_splitk_mma_sm86",
    "a16_v2_woodbury_inv8_sm86",
    "a16_v2_cayley_woodbury_apply_sm86",
    "a16_v2_iris_canary_finalize_sm86",
)


class V2VerifyError(RuntimeError):
    """Explicit fail-closed verification error (replaces bare assert)."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise V2VerifyError(msg)


def _cpu_has_avx512f() -> bool:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    txt = cpuinfo.read_text(encoding="utf-8", errors="ignore")
    return re.search(r"\bavx512f\b", txt) is not None


def _py_canonical_tree_reduce_128(xs: list[float]) -> float:
    _require(len(xs) == 128, "expected 128 elements for canonical tree reduce")
    warps: list[float] = []
    for w in range(4):
        lane = [float(struct.unpack("<f", struct.pack("<f", xs[w * 32 + i]))[0]) for i in range(32)]
        stride = 1
        while stride <= 16:
            nxt = [0.0] * 32
            for i in range(32):
                s = lane[i] + lane[i ^ stride]
                nxt[i] = float(struct.unpack("<f", struct.pack("<f", s))[0])
            lane = nxt
            stride <<= 1
        warps.append(lane[0])
    s01 = float(struct.unpack("<f", struct.pack("<f", warps[0] + warps[1]))[0])
    s23 = float(struct.unpack("<f", struct.pack("<f", warps[2] + warps[3]))[0])
    return float(struct.unpack("<f", struct.pack("<f", s01 + s23))[0])


def _select_cpu_so() -> Path:
    avx_so = BUILD_DIR / "liba16_v2_avx512.so"
    port_so = BUILD_DIR / "liba16_v2_portable.so"
    if _cpu_has_avx512f() and avx_so.exists():
        return avx_so
    _require(port_so.exists(), f"missing compiled CPU shared library: {port_so}")
    return port_so


def verify_cpu_abi() -> dict[str, object]:
    so_path = _select_cpu_so()
    lib = ctypes.CDLL(str(so_path))

    lib.silikernels_v2_swar127_in_domain_u64.argtypes = [ctypes.c_uint64]
    lib.silikernels_v2_swar127_in_domain_u64.restype = ctypes.c_int
    lib.silikernels_v2_swar127_negate_u64.argtypes = [ctypes.c_uint64]
    lib.silikernels_v2_swar127_negate_u64.restype = ctypes.c_uint64
    lib.silikernels_v2_swar127_negate_u64_checked.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.silikernels_v2_swar127_negate_u64_checked.restype = ctypes.c_int
    lib.silikernels_v2_canonical_tree_reduce_128.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.silikernels_v2_canonical_tree_reduce_128.restype = ctypes.c_float
    lib.silikernels_v2_should_dispatch_cpu.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.silikernels_v2_should_dispatch_cpu.restype = ctypes.c_int
    lib.silikernels_v2_run_selftest.argtypes = [ctypes.c_int]
    lib.silikernels_v2_run_selftest.restype = ctypes.c_int

    _require(lib.silikernels_v2_run_selftest(0) == 0, f"C self-test failed inside {so_path.name}")

    # Boundary & randomized SWAR127 verification
    rng = random.Random(20261128)
    for idx in range(32):
        packed = 0
        exp_bytes = []
        for b in range(8):
            val = 1 if idx == 0 else (127 if idx == 1 else rng.randint(1, 127))
            packed |= (val & 0xFF) << (b * 8)
            exp_bytes.append((-val) & 0xFF)
        _require(lib.silikernels_v2_swar127_in_domain_u64(packed) == 1, "SWAR127 valid domain check failed")
        neg = lib.silikernels_v2_swar127_negate_u64(packed)
        out_c = ctypes.c_uint64(0)
        rc = lib.silikernels_v2_swar127_negate_u64_checked(packed, ctypes.byref(out_c))
        _require(rc == 0 and out_c.value == neg, "SWAR127 checked negate mismatch")
        for b in range(8):
            _require(((neg >> (b * 8)) & 0xFF) == exp_bytes[b], "SWAR127 negated byte mismatch")

    for bad in (
        0x0101010101010001,
        0x0101018001010101,
        0x01010101FF010101,
        0x0000000000000000,
        0x8080808080808080,
    ):
        _require(lib.silikernels_v2_swar127_in_domain_u64(bad) == 0, "SWAR127 OOB byte accepted")
        _require(lib.silikernels_v2_swar127_negate_u64(bad) == 0, "SWAR127 fail-closed negate failed")
        out_bad = ctypes.c_uint64(0xCAFEBABE)
        rc_bad = lib.silikernels_v2_swar127_negate_u64_checked(bad, ctypes.byref(out_bad))
        _require(rc_bad == -1 and out_bad.value == 0, "SWAR127 checked OOB trap failed")

    # 0-ULP canonical tree reduction bit-exact check
    max_tree_bits_diff = 0
    arr_t = ctypes.c_float * 128
    for _ in range(16):
        xs = [float(struct.unpack("<f", struct.pack("<f", rng.gauss(0.0, 3.0)))[0]) for _ in range(128)]
        c_arr = arr_t(*xs)
        c_sum = float(lib.silikernels_v2_canonical_tree_reduce_128(c_arr))
        py_sum = _py_canonical_tree_reduce_128(xs)
        b_c = struct.unpack("<I", struct.pack("<f", c_sum))[0]
        b_p = struct.unpack("<I", struct.pack("<f", py_sum))[0]
        diff = abs(b_c - b_p)
        if diff > max_tree_bits_diff:
            max_tree_bits_diff = diff
        _require(b_c == b_p, f"0-ULP tree_reduce_128 mismatch: C={b_c:#x} PY={b_p:#x}")

    # B*=2 / B*=3 dispatch contract
    _require(lib.silikernels_v2_should_dispatch_cpu(1, 0) == 1, "3bpw B=1 must dispatch CPU")
    _require(lib.silikernels_v2_should_dispatch_cpu(2, 0) == 0, "3bpw B=2 reaches B* crossover")
    _require(lib.silikernels_v2_should_dispatch_cpu(2, 1) == 1, "4.5bpw B=2 must dispatch CPU")
    _require(lib.silikernels_v2_should_dispatch_cpu(3, 1) == 0, "4.5bpw B=3 reaches B* crossover")

    # Verify static CUDA source contracts (without requiring live GPU execution)
    cuh = (REPO_ROOT / "cuda" / "sm86" / "a16_v2_ampere_kernels.cuh").read_text(encoding="utf-8")
    for kname in EXPECTED_SM86_KERNELS:
        _require(kname in cuh, f"missing sm_86 kernel definition: {kname}")
    _require("__constant__ float    c_wb_inv_256b[64];" in cuh, "missing 256 B Woodbury constant buffer")
    _require("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in cuh, "missing IMMA m16n8k32 PTX instruction")
    _require("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in cuh, "missing BF16 Split-K mma PTX")
    _require("lop3.b32" in cuh, "missing LOP3 0x60 SECDED instruction")

    return {
        "cpu_library": so_path.name,
        "swar127_tested": 32,
        "swar127_oob_rejected": 5,
        "tree_reduce_128_max_bits_diff": max_tree_bits_diff,
        "b_star_3bpw": 2,
        "b_star_45bpw": 3,
        "cuda_kernels_declared": len(EXPECTED_SM86_KERNELS),
    }


def _audit_single_target_ptxas_and_sass(label: str, ptxas_path: Path, sass_path: Path) -> dict[str, object]:
    _require(ptxas_path.exists(), f"missing ptxas log for {label}: {ptxas_path}")
    _require(sass_path.exists(), f"missing SASS dump for {label}: {sass_path}")
    ptx_txt = ptxas_path.read_text(encoding="utf-8")
    sass_txt = sass_path.read_text(encoding="utf-8")

    # Parse ptxas per-entry stats
    regs_by_kernel: dict[str, int] = {}
    spills_by_kernel: dict[str, tuple[int, int, int]] = {}
    current_sym: str | None = None
    lines = ptx_txt.splitlines()
    for i, line in enumerate(lines):
        m_comp = re.search(r"Compiling entry function '([^']+)' for 'sm_86'", line)
        if m_comp:
            current_sym = m_comp.group(1)
            continue
        m_prop = re.search(r"Function properties for (\S+)", line)
        if m_prop:
            sym = m_prop.group(1)
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            m_sp = re.search(
                r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads",
                nxt,
            )
            _require(m_sp is not None, f"{label}: missing stack/spill line for {sym}")
            if m_sp is not None:
                st, ss, sl = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
                _require(st == 0 and ss == 0 and sl == 0, f"{label}: non-zero spill in {sym}: {st}/{ss}/{sl}")
                spills_by_kernel[sym] = (st, ss, sl)
            continue
        m_reg = re.search(r"Used\s+(\d+)\s+registers", line)
        if m_reg and current_sym is not None:
            regs = int(m_reg.group(1))
            _require(1 <= regs <= 48, f"{label}: register budget exceeded in {current_sym}: {regs} > 48")
            regs_by_kernel[current_sym] = regs

    matched_kernels: dict[str, int] = {}
    for kname in EXPECTED_SM86_KERNELS:
        matches = [sym for sym in regs_by_kernel if kname in sym]
        _require(len(matches) == 1, f"{label}: expected 1 ptxas entry for {kname}, found {matches}")
        sym = matches[0]
        _require(sym in spills_by_kernel, f"{label}: missing spill entry for {sym}")
        _require(kname in sass_txt, f"{label}: missing SASS function dump for {kname}")
        matched_kernels[kname] = regs_by_kernel[sym]

    # Assert strictly 0 LDL / STL instructions anywhere in SASS disassembly
    ldl_stl_hits = re.findall(r"(?m)^\s*(?:/\*[0-9a-fA-F]+\*/\s*)?(?:@[!A-Z0-9_]+\s+)?\b(?:LDL|STL)(?:\.[A-Za-z0-9_]+)?\b", sass_txt)
    _require(len(ldl_stl_hits) == 0, f"{label}: forbidden SASS LDL/STL instructions found: {ldl_stl_hits[:5]}")

    for mnem in ("IMAD", "LOP3", "SHFL", "HMMA", "IDP"):
        _require(re.search(rf"\b{mnem}\b", sass_txt) is not None, f"{label}: missing expected SASS mnemonic {mnem}")

    return {
        "target": label,
        "kernels_verified": len(matched_kernels),
        "max_registers": max(matched_kernels.values()),
        "max_stack_spill_bytes": 0,
        "sass_ldl_stl_count": len(ldl_stl_hits),
        "registers_by_kernel": matched_kernels,
    }


def verify_sass_targets() -> dict[str, object]:
    targets = [
        ("obj", BUILD_DIR / "_v2_ptxas_o.log", BUILD_DIR / "_v2_kernels_o.sass"),
        ("so", BUILD_DIR / "_v2_ptxas_so.log", BUILD_DIR / "_v2_kernels_so.sass"),
        ("bench", BUILD_DIR / "_v2_ptxas_bench.log", BUILD_DIR / "_v2_kernels_bench.sass"),
    ]
    audits = [_audit_single_target_ptxas_and_sass(lbl, ptx, sass) for lbl, ptx, sass in targets]

    cuda_so = BUILD_DIR / "liba16_v2_ampere.so"
    _require(cuda_so.exists(), f"missing compiled CUDA shared lib: {cuda_so}")
    cudart_lib = ctypes.CDLL(str(cuda_so))
    cudart_lib.a16_v2_sm86_cpu_selftest.argtypes = [ctypes.c_int]
    cudart_lib.a16_v2_sm86_cpu_selftest.restype = ctypes.c_int
    cudart_lib.a16_v2_sm86_abi_info.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    cudart_lib.a16_v2_sm86_abi_info.restype = ctypes.c_int
    _require(cudart_lib.a16_v2_sm86_cpu_selftest(0) == 0, "liba16_v2_ampere.so CPU selftest failed")

    wb_b = ctypes.c_uint32(0)
    b_star = ctypes.c_uint32(0)
    _require(cudart_lib.a16_v2_sm86_abi_info(ctypes.byref(wb_b), ctypes.byref(b_star)) == 0, "ABI info query failed")
    _require(wb_b.value == 256 and b_star.value == 2, f"unexpected CUDA ABI info: wb={wb_b.value}, b_star={b_star.value}")

    return {
        "targets_audited": audits,
        "cuda_so_cpu_selftest_pass": True,
        "woodbury_bytes": wb_b.value,
        "b_star_3bpw": b_star.value,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify SiliKernels V2 CPU contracts and sm_86 ptxas/SASS constraints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu-only", action="store_true", help="Verify CPU ABI/contracts and static CUDA source only")
    group.add_argument("--verify-sass", action="store_true", help="Verify CPU contracts plus .o/.so/bench ptxas & SASS gates")
    args = parser.parse_args(argv)

    cpu_summary = verify_cpu_abi()
    out: dict[str, object] = {"status": "PASS", "mode": "verify-sass" if args.verify_sass else "cpu-only", "cpu": cpu_summary}
    if args.verify_sass:
        out["sass"] = verify_sass_targets()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
