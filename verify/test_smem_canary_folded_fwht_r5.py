#!/usr/bin/env python3
"""Causality, Tamper-Refusal, Pure-Q AST, & Energy-Policy Test Suite for CTA-SMEM Canary & Folded FWHT.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import contextlib
import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parent
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

try:
    import verify_smem_canary_folded_fwht as vh
except SystemExit as exc:
    raise RuntimeError(f"verify_smem_canary_folded_fwht exited on import: {exc.code}") from exc


def _resign(doc: dict, key: str) -> dict:
    doc = copy.deepcopy(doc)
    doc[key] = vh.canonical_sha256(doc, exclude_key=key)
    return doc


class TestSmemCanaryFoldedFwht(unittest.TestCase):
    def assertCode(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(vh.SmemCanaryVerifyError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_hardened_pure_q_ast_audit_and_negatives(self) -> None:
        src = VERIFY_DIR.joinpath("verify_smem_canary_folded_fwht.py").read_text(encoding="utf-8")
        vh.verify_pure_q_ast(src)
        vh.verify_pure_q_ast("from fractions import Fraction\nx = Fraction(-1, 2)\ny = Fraction(1 + 2, 3)\n")

        for code, bad_code in (
            ("AST_SYNTAX_ERROR", "def (\n"),
            ("AST_DIV_FORBIDDEN", "x = 1 / 2\n"),
            ("AST_DIV_FORBIDDEN", "x /= 2\n"),
            ("AST_FLOORDIV_FORBIDDEN", "x = 1 // 2\n"),
            ("AST_MOD_FORBIDDEN", "x = '%.2f' % Fraction(1, 3)\n"),
            ("AST_MOD_FORBIDDEN", "x = 5 % 2\n"),
            ("AST_POW_FORBIDDEN", "x = 2 ** -1\n"),
            ("AST_BARE_ASSERT", "assert True\n"),
            ("AST_FORMAT_SPEC_FORBIDDEN", "x = f'{1:.3f}'\n"),
            ("AST_FLOAT_LITERAL", "x = 1.5\n"),
            ("AST_FLOAT_LITERAL", "x = 1j\n"),
            ("AST_FRACTION_SHADOW", "def Fraction(): pass\n"),
            ("AST_FRACTION_SHADOW", "class Fraction: pass\n"),
            ("AST_FRACTION_SHADOW", "def f(Fraction): pass\n"),
            ("AST_FRACTION_SHADOW", "try:\n pass\nexcept Exception as Fraction:\n pass\n"),
            ("AST_FORBIDDEN_IMPORT", "import math\n"),
            ("AST_FORBIDDEN_IMPORT", "import decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import _decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import random\n"),
            ("AST_FORBIDDEN_IMPORT", "import fractions\n"),
            ("AST_FORBIDDEN_IMPORT", "import struct\n"),
            ("AST_FORBIDDEN_IMPORT", "import array\n"),
            ("AST_FORBIDDEN_IMPORT", "from math import sqrt\n"),
            ("AST_FORBIDDEN_IMPORT", "from _operator import truediv\n"),
            ("AST_FORBIDDEN_IMPORT", "from sys import modules\n"),
            ("AST_FORBIDDEN_IMPORT", "from json import loads\n"),
            ("AST_FORBIDDEN_IMPORT", "from importlib import import_module\n"),
            ("AST_RELATIVE_IMPORT", "from . import helper\n"),
            ("AST_RELATIVE_IMPORT", "from .foo import bar\n"),
            ("AST_WILDCARD_IMPORT", "from os import *\n"),
            ("AST_FRACTION_ALIAS", "from fractions import Fraction as F\n"),
            ("AST_FRACTION_ALIAS", "from fractions import gcd\n"),
            ("AST_FORBIDDEN_NAME", "x = float(1)\n"),
            ("AST_FORBIDDEN_NAME", "x = pow(2, -1)\n"),
            ("AST_FORBIDDEN_NAME", "x = __import__('math')\n"),
            ("AST_FORBIDDEN_NAME", "k = 'float'\nfn = getattr(__builtins__, k)\n"),
            ("AST_FORBIDDEN_NAME", "x = format(1, '.2f')\n"),
            ("AST_FORBIDDEN_NAME", "x = divmod(5, 2)\n"),
            ("AST_FORBIDDEN_NAME", "x = round(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = Fraction.from_float(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__truediv__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "fn = (1).__getattribute__('__truediv__')\n"),
            ("AST_FORBIDDEN_ATTR", "import sys\nm = sys.modules['math']\n"),
            ("AST_FORBIDDEN_ATTR", "x = b.float(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = json.load(f)\n"),
            ("AST_FORBIDDEN_ATTR", "x = s.format(1)\n"),
            ("AST_FRACTION_REF", "from fractions import Fraction\nQ = Fraction\nx = Q('1/2')\n"),
            ("AST_FRACTION_REF", "from fractions import Fraction\nxs = [Fraction]\n"),
            ("AST_FRACTION_REF", "x = mod.Fraction\n"),
            ("AST_LOADS_REF", "import json\nf = json.loads\n"),
            ("AST_LOADS_REF", "x = loads('1.5')\n"),
            ("AST_DYNAMIC_CALL", "from fractions import Fraction\nx = type(Fraction(1, 1))('1.5')\n"),
            ("AST_TYPE_ESCAPE", "from fractions import Fraction\nt = type(Fraction(1, 1))\n"),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.loads('1.5')\n"),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.loads('1.5', parse_float=None, parse_constant=c)\n"),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.loads('1.5', parse_float=f, parse_constant=None)\n"),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.loads('1.5', parse_float=float, parse_constant=c)\n"),
            ("AST_FRACTION_ARGS", "x = Fraction(1)\n"),
            ("AST_FRACTION_ARGS", "x = Fraction(1, 2, k=3)\n"),
            ("AST_FRACTION_ARGS", "x = mod.Fraction('1.5')\n"),
            ("AST_FRACTION_STR", "x = Fraction('1', 2)\n"),
            ("AST_FRACTION_STR", "x = Fraction(b'1', 2)\n"),
            ("AST_FRACTION_STR", "x = Fraction(f'{1}', 2)\n"),
            ("AST_FRACTION_STR", "x = Fraction(s, 2)\n"),
            ("AST_FRACTION_STR", "x = Fraction(True, 2)\n"),
            ("AST_FRACTION_STR", "x = mod.Fraction('1', 2)\n"),
        ):
            self.assertCode(code, vh.verify_pure_q_ast, bad_code)

        self.assertFalse(vh.contains_any_float({"a": [1, "2/3", {"b": (3, 4)}], "s": frozenset({"ok"})}))
        self.assertTrue(vh.contains_any_float({"a": [1, 1.5]}))
        self.assertTrue(vh.contains_any_float({1.2: "x"}))
        self.assertTrue(vh.contains_any_float({"a": (1.0,)}))

    def test_02_rational_kat_and_extended_energy_policy_guard(self) -> None:
        self.assertEqual(vh.parse_q("3/4"), Fraction(3, 4))
        self.assertEqual(vh.parse_q("-5/2"), Fraction(-5, 2))
        self.assertEqual(vh.parse_q("0/1"), Fraction(0, 1))
        self.assertEqual(vh.q_div(Fraction(3, 4), Fraction(2, 5)), Fraction(15, 8))
        self.assertCode("ZERO_DIV", vh.q_div, Fraction(1, 2), Fraction(0, 1))
        self.assertCode("BAD_RATIONAL", vh.parse_q, 1)
        self.assertCode("BAD_RATIONAL", vh.parse_q, "2/4")
        self.assertCode("BAD_RATIONAL", vh.parse_q, "1/0")
        self.assertCode("BAD_RATIONAL", vh.parse_q, ("9" * 1025) + "/1")
        self.assertCode("BAD_RATIONAL", vh.parse_q, "1/" + ("9" * 1025))
        self.assertEqual(vh.parse_q(("9" * 1024) + "/1"), Fraction(int("9" * 1024), 1))
        self.assertEqual(vh.parse_q("1/" + ("9" * 1024)), Fraction(1, int("9" * 1024)))
        self.assertCode("BAD_DIVISOR", vh.q_div_int, Fraction(1, 2), 0)
        self.assertCode("BAD_DIVISOR", vh.q_div_int, Fraction(1, 2), True)
        self.assertCode("BAD_JSON", vh.canonical_json_bytes, {"x": object()})
        self.assertFalse(vh.is_hex64("z" * 64))
        self.assertFalse(vh.is_hex64(123))

        # Independent FIPS 180-4 SHA-256 KAT ("abc")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "kat.bin"
            p.write_bytes(b"abc")
            self.assertEqual(
                vh.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

        # Reference 61 + extended RAPL/NVML/nvidia-smi/j-tok/NUL/stock-power/space-unit corpus (83 bad)
        bad_values = (
            "45.2 W", "45W", "45.2 W.", "45.2W!", "W?", "[W]", "45.2\u00d7W", "45.2-W", "45.2 [W]",
            "W: 45.2", "W=45.2", "45.2 kilowatts", "45.2\u200bW", "45.2 k\u200bW", "5 GW", "5 TW",
            "5 pW", "5 kilo watts", "5 kilo-watts", "5 m W", "15 watthours", "15 watt hours",
            "15 watt-hour", "5\u2013W", "12\u00b7W", "3.9e2 W", "1.5E-3W", "0.5 kWh", "1.2 J/tok",
            "joules per token", "MJ per token", "mJ/tok", "15 TDP", "TDP 15", "TDP",
            "70% less energy", "70 percent less power", "energy use dropped", "power savings",
            "5 \u051d", "\u0435nergy", "3 \u043a\u0412\u0442", "\uff11\uff12\uff37", "45 milliwatts",
            "1.2 megawatts", "12.5 kilojoules", "W: 45.2 drain", "45.2[W] drain", "J: 5",
            "cap_45_w", "45_watts", "limit_45_w", "draw_45w", "45 Whr", "130w_psu",
            "max_45w", "45w", "5 PW", "45.2 \u03c9", "\u051datt", "45.2\u2219W",
            "measured with RAPL", "nvml query", "nvidia-smi draw", "10 tok/J", "tokens per joule",
            "50 tok/kWh", "score per watt", "500 mAh", "45\u200cW", "45\u200dW", "45\u2060W",
            "45\ufeffW", "45\u00adW",
            "j/tok", "1 j/ tok", "tokens_per_joule", "x_nvml_y", "wa\u0000tt",
            "45.0% of stock power", "energy per token", "power-savings", "W 45.2",
            "forty-five watts", "energy: 12", "J/token", "W/tok", "tok_per_mwh",
            "nvmlDeviceGetTotalEnergyConsumption",
            "wattage: 45", "2.25 A at 20 V", "20 V at 2.25 A", "100 W/ token",
        )
        good_values = (
            "5 ms", "latency 12", "70 C", "78\u00b0C", "70/78/85 C knees", "5 C deadband",
            "5 deg C deadband", "135/2", "45.0%", "0.5 mm", "ac", "battery", "unknown",
            "temp_c", "gpu_temp_exceeded", "not_on_ac_power", "UNCONTROLLED_LAPTOP_NOT_CLAIMED",
            "grad_W_q", "stdlib_sha256_chain", "cpu_jiffies_total", "a" * 64, "129,279.00 tok/s",
            "469.41 us/tok", "8.113 MiB", "sm_86", "T10-TAG-61", "wmma", "window 20", "warp 32",
            "16 warps/SM", "h256", "rows 1024", "tdp_like_free_text_no", "T13-TAG-DECIDE",
            "J2000 epoch-like", "J2000.0", "worker T10", "2024-W05", "sprint-W12", "node-w10",
            "q4-w3", "W-2", "W12", "kW12", "bench w/ perf counters", "taskset -c 3 w/ 5 reps",
            "12 ms/tok", "sm_clock 2100 MHz", "0.9 V", "1024 MiB/s",
            "j = 3", "3 w/o pinning", "w/ tokens",
        )
        self.assertEqual(len(bad_values), 93)
        self.assertEqual(len(good_values), 53)
        for bad in bad_values:
            self.assertTrue(vh.is_forbidden_energy_value(bad), f"must reject {bad!r}")
        for good in good_values:
            self.assertFalse(vh.is_forbidden_energy_value(good), f"false positive on {good!r}")
        self.assertFalse(vh.is_forbidden_energy_key_name("empowerment_score"))
        self.assertTrue(vh.is_forbidden_energy_key_name("wattage"))

        # Linear-time ReDoS defense on long digit/dot/comma runs (< 0.1s)
        for payload in ("1" * 8000, "1." * 3000, "1" + "," * 4000):
            t0 = time.monotonic()
            self.assertFalse(vh.is_forbidden_energy_value(payload))
            self.assertLess(time.monotonic() - t0, 0.25)

        self.assertFalse(
            vh.has_forbidden_energy_claim(
                {
                    "energy_power_status": vh.ENERGY_POLICY_STATUS,
                    "kernels": {vh.FOLDED_CELL_SYMBOL: {"registers": 55}},
                    "ok": list(good_values),
                    "t": ("sm_86",),
                    "f": frozenset({"1024 B"}),
                    1: "clean",
                }
            )
        )
        for bad_obj in (
            {"mJ_per_tok": 1},
            {"avgPower": 45},
            {"po\u0301wer": 45},
            {"\u0435nergy": 1},
            {"max_45w": 1},
            {"pwr": 1},
            {"rapl": "ok"},
            {"nvml": "ok"},
            {"nvidia_smi": "ok"},
            {"nested": {"energy_power_status": vh.ENERGY_POLICY_STATUS}},
            {"kernels": {vh.FOLDED_CELL_SYMBOL: {"note": "45.2 W"}}},
            {"kernels": {"fake_pj_symbol": {"registers": 8}}},
            {"t": ("45.2 W",)},
            {1: "45.2 W"},
        ):
            self.assertTrue(vh.has_forbidden_energy_claim(bad_obj), f"must reject {bad_obj!r}")

    def test_03_pure_z_secded_smem_packer_and_fwht_oracles(self) -> None:
        sec_o = vh.verify_secded_and_smem_packer_oracle()
        self.assertEqual(sec_o["sec_healed"], 16)
        self.assertEqual(sec_o["ded_caught"], 120)
        self.assertEqual(sec_o["multi_cta_sec_carry_packed_u32"], 0x0001FFFF)
        self.assertEqual(sec_o["multi_cta_ded_wrap_packed_u32"], 0xFFFF000E)
        fw_o = vh.verify_fwht256_fold_oracle()
        self.assertEqual(fw_o["prefix_upper_half_delta_l1"], 0)
        self.assertGreater(fw_o["fold1k_upper_half_delta_l1"], 0)
        self.assertEqual(fw_o["fold1k_fp16_bytes"], 1024)

        self.assertCode("BAD_DATA11", vh.a16_encode_u16, -1)
        self.assertCode("BAD_DATA11", vh.a16_encode_u16, 2048)
        self.assertCode("BAD_DATA11", vh.a16_encode_u16, True)
        self.assertCode("BAD_W16", vh.a16_syndrome_u5, -1)
        self.assertCode("BAD_W16", vh.a16_syndrome_u5, 65536)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, -1, 0, 0, 0)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, 0x100000000, 0, 0, 0)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, 0, -1, 0, 0)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, 0, 0, -1, 0)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, 0, 0, 0, -1)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, 0, 0, 0, 0x100000000)
        self.assertCode("BAD_PACK_ARGS", vh.sat_add_packed_int, True, 0, 0, 0)
        pk_max, dg_max = vh.sat_add_packed_int(0, 0xFFFF, 0xFFFF, 0x10)
        self.assertEqual(pk_max, 0xFFFFFFFF)
        self.assertEqual(dg_max, 0x10)
        pk_ovf_both, dg_ovf_both = vh.sat_add_packed_int(0, 70000, 5, 0x10)
        self.assertEqual(pk_ovf_both, (5 << 16) | 0xFFFF)
        self.assertEqual(dg_ovf_both, 0x80000010)
        self.assertCode("BAD_FWHT_LEN", vh.fwht256_bent4_int, [0] * 16)

        cw_ref = vh.a16_encode_u16(0x5A3)
        self.assertCode("ORACLE_CLEAN_SYND", vh.verify_secded_and_smem_packer_oracle, syndrome_fn=lambda w: 1)
        self.assertCode(
            "ORACLE_SEC_PARITY",
            vh.verify_secded_and_smem_packer_oracle,
            syndrome_fn=lambda w: 0 if w == cw_ref else 2,
        )
        self.assertCode(
            "ORACLE_SEC_HEAL",
            vh.verify_secded_and_smem_packer_oracle,
            syndrome_fn=lambda w: 0 if w == cw_ref else (3 if w == (cw_ref ^ 1) else vh.a16_syndrome_u5(w)),
        )
        self.assertCode(
            "ORACLE_DED_DETECT",
            vh.verify_secded_and_smem_packer_oracle,
            syndrome_fn=lambda w: 0 if w == (cw_ref ^ 3) else vh.a16_syndrome_u5(w),
        )
        self.assertCode(
            "ORACLE_DED_DETECT",
            vh.verify_secded_and_smem_packer_oracle,
            syndrome_fn=lambda w: 1 if w == (cw_ref ^ 3) else vh.a16_syndrome_u5(w),
        )
        self.assertCode(
            "ORACLE_PACK_CLEAN",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (1, 0) if o == 0 and s == 0 and d == 0 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_PACK_CLEAN",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (0, 1) if o == 0 and s == 0 and d == 0 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_PACK_OVF_SEC",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (0, 0x80000015) if s == 70000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_PACK_OVF_SEC",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (((1 << 16) | 0xFFFF), 0) if s == 70000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_PACK_OVF_DED",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (0, 0x80000022) if d == 70000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_PACK_OVF_DED",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (((0xFFFF << 16) | 12), 0) if d == 70000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_MULTI_CTA_SEC",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (0, 0x80000015) if o == 40000 and s == 40000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_MULTI_CTA_SEC",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (((1 << 16) | 0xFFFF), 0) if o == 40000 and s == 40000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_MULTI_CTA_DED",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (14, 0x80000022) if o == ((0x8000 << 16) | 7) and d == 0x8000 else vh.sat_add_packed_int(o, s, d, g),
        )
        self.assertCode(
            "ORACLE_MULTI_CTA_DED",
            vh.verify_secded_and_smem_packer_oracle,
            sat_add_fn=lambda o, s, d, g: (((0xFFFF << 16) | 14), 0) if o == ((0x8000 << 16) | 7) and d == 0x8000 else vh.sat_add_packed_int(o, s, d, g),
        )
        # Planted wrong FWHTs: drop bent4 signs, reverse input permutation, rotate output, negate output
        def _fwht_drop_bent4(u):
            buf = [0] * 256
            for idx in range(256):
                w = idx >> 5
                lane = idx & 31
                src_idx = ((w ^ (lane & 7)) << 5) | lane
                buf[idx] = u[src_idx]
            step = 1
            while step < 256:
                stride = step << 1
                for base in range(0, 256, stride):
                    for off in range(step):
                        a, b = buf[base + off], buf[base + off + step]
                        buf[base + off] = a + b
                        buf[base + off + step] = a - b
                step = stride
            return buf

        self.assertCode("ORACLE_FWHT_KAT", vh.verify_fwht256_fold_oracle, fwht_fn=_fwht_drop_bent4)
        self.assertCode("ORACLE_FWHT_KAT", vh.verify_fwht256_fold_oracle, fwht_fn=lambda u: vh.fwht256_bent4_int(list(reversed(u))))
        self.assertCode("ORACLE_FWHT_KAT", vh.verify_fwht256_fold_oracle, fwht_fn=lambda u: (lambda out: out[1:] + out[:1])(vh.fwht256_bent4_int(u)))
        self.assertCode("ORACLE_FWHT_KAT", vh.verify_fwht256_fold_oracle, fwht_fn=lambda u: [-x for x in vh.fwht256_bent4_int(u)])

        call_p = {"n": 0}
        def _bad_parseval_fwht(u):
            call_p["n"] += 1
            if call_p["n"] <= 4:
                return vh.fwht256_bent4_int(u)
            return [0] * 256
        self.assertCode("ORACLE_PARSEVAL_FAIL", vh.verify_fwht256_fold_oracle, fwht_fn=_bad_parseval_fwht)
        call_n = {"n": 0}
        def _bad_prefix_fwht(u):
            call_n["n"] += 1
            out = vh.fwht256_bent4_int(u)
            if call_n["n"] == 7:
                return [x + 1 for x in out]
            return out
        self.assertCode("ORACLE_UPPER_ROUTING", vh.verify_fwht256_fold_oracle, fwht_fn=_bad_prefix_fwht)
        call_m = {"n": 0}
        def _bad_fold_fwht(u):
            call_m["n"] += 1
            if call_m["n"] == 8:
                return _bad_fold_fwht.first
            out = vh.fwht256_bent4_int(u)
            if call_m["n"] == 5:
                _bad_fold_fwht.first = out
            return out
        self.assertCode("ORACLE_UPPER_ROUTING", vh.verify_fwht256_fold_oracle, fwht_fn=_bad_fold_fwht)

    def test_04_live_on_disk_receipts_and_tamper_refusal(self) -> None:
        summary = vh.verify_on_disk(REPO_ROOT)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["verdict"], "CONFIRMED")

        ptx = vh.load_strict_canonical_json(REPO_ROOT.joinpath(vh.PTXAS_RECEIPT_REL))
        res = vh.load_strict_canonical_json(REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL))

        self.assertCode("PTXAS_TYPE", vh.verify_ptxas_receipt, [])
        for code, resign, mut in (
            ("PTXAS_SCHEMA", False, lambda d: d.__setitem__("schema", "bad")),
            ("PTXAS_ENERGY_POLICY", True, lambda d: d.__setitem__("energy_power_status", "BAD")),
            ("PTXAS_ENERGY_POLICY", True, lambda d: d.__setitem__("note", "45.2 W")),
            ("PTXAS_FLOAT_FORBIDDEN", True, lambda d: d.__setitem__("x", 1.5)),
            ("PTXAS_KEYS", True, lambda d: d.__setitem__("bogus_external_sha256", "0" * 64)),
            ("PTXAS_SHA_FORMAT", False, lambda d: d.__setitem__("receipt_sha256", "bad")),
            ("PTXAS_SHA_MISMATCH", False, lambda d: d.__setitem__("arch", "sm_90")),
            ("PTXAS_ARCH", True, lambda d: d.__setitem__("arch", "sm_90")),
            ("PTXAS_FLAGS", True, lambda d: d.__setitem__("compile_flags", "-O0")),
            ("PTXAS_NVCC", True, lambda d: d.__setitem__("nvcc_version", "")),
            ("PTXAS_CU_PATH", True, lambda d: d.__setitem__("cu_path", "wrong.cu")),
            ("PTXAS_CU_SHA", True, lambda d: d.__setitem__("cu_sha256", "bad")),
            ("PTXAS_KERNELS", True, lambda d: d.__setitem__("kernels", {})),
            ("PTXAS_REQUIRED_SYMBOLS", True, lambda d: (d["kernels"].pop(vh.FOLDED_HEAD_SYMBOL), d["kernels"].__setitem__("fake", copy.deepcopy(ptx["kernels"][vh.FOLDED_HEAD_SYMBOL])))),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"].__setitem__(vh.FOLDED_HEAD_SYMBOL, [])),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("extra", 0)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("registers", -1)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("registers", True)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("registers", 20)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vh.FOLDED_CELL_SYMBOL].__setitem__("atom_cas_sys_insns", 0)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vh.FOLDED_CELL_SYMBOL].__setitem__("red_or_sys_insns", 0)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vh.FOLDED_CELL_SYMBOL].__setitem__("atoms_smem_insns", 2)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("spill_store_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("stack_bytes", 8)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("spill_load_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("ldl_insns", 1)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("stl_insns", 1)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vh.FOLDED_HEAD_SYMBOL].__setitem__("registers", 0)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("nonzero_spill_kernels", 1)),
        ):
            p2 = copy.deepcopy(ptx)
            mut(p2)
            if resign:
                p2 = _resign(p2, "receipt_sha256")
            self.assertCode(code, vh.verify_ptxas_receipt, p2)
        self.assertCode("PTXAS_CU_SHA_DISK", vh.verify_ptxas_receipt, ptx, expected_cu_sha="0" * 64)

        self.assertCode("RESULTS_TYPE", vh.verify_results_receipt, [], ptx)
        for code, resign, mut in (
            ("RESULTS_SCHEMA", False, lambda d: d.__setitem__("schema", "bad")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("energy_power_status", "BAD")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("mj_per_tok", "1/2")),
            ("RESULTS_FLOAT_FORBIDDEN", True, lambda d: d.__setitem__("x", 0.5)),
            ("RESULTS_KEYS", True, lambda d: d.__setitem__("bogus_external_sha256", "0" * 64)),
            ("RESULTS_SHA_FORMAT", False, lambda d: d.__setitem__("result_sha256", "bad")),
            ("RESULTS_SHA_MISMATCH", False, lambda d: d.__setitem__("verdict", "FALSIFIED")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("host_native_atomic_supported", 0)),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("sm", "9.0")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("nvcc_version", "other")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("device", "")),
            ("RESULTS_BINDINGS", True, lambda d: d.__setitem__("source_bindings", [])),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_path", "wrong.cu")),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("bench_cu_path", "wrong.cu")),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_path", "wrong.py")),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("ptxas_receipt_path", "wrong.json")),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("corpus_external_binary_status", "BAD")),
            ("RESULTS_CU_BIND", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_sha256", "0" * 64)),
            ("RESULTS_PTX_BIND", True, lambda d: d["source_bindings"].__setitem__("ptxas_receipt_sha256", "0" * 64)),
            ("RESULTS_BENCH_SHA", True, lambda d: d["source_bindings"].__setitem__("bench_cu_sha256", "bad")),
            ("RESULTS_BPB_SCRIPT_SHA", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_sha256", "bad")),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_manifest_sha256", "0" * 64)),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_train_bin_sha256", "0" * 64)),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_eval_bin_sha256", "0" * 64)),
            ("RESULTS_SILICON", True, lambda d: d.__setitem__("silicon_parity_and_canary", [])),
            ("RESULTS_CELL_PARITY", True, lambda d: d["silicon_parity_and_canary"].__setitem__("folded1k_cell_fp64_max_err_q", "1/10")),
            ("RESULTS_HEAD_PARITY", True, lambda d: d["silicon_parity_and_canary"].__setitem__("folded1k_head_fp64_max_err_q", "1/10")),
            ("RESULTS_HEAD_POS", True, lambda d: d["silicon_parity_and_canary"].__setitem__("standalone_head_unfolded_us_q", "0/1")),
            ("RESULTS_HEAD_DELTA", True, lambda d: d["silicon_parity_and_canary"].__setitem__("standalone_head_overhead_us_q", "99/1")),
            ("RESULTS_STORMS_LEN", True, lambda d: d["silicon_parity_and_canary"].__setitem__("storms", [])),
            ("RESULTS_STORM_TYPE", True, lambda d: d["silicon_parity_and_canary"]["storms"].__setitem__(0, "bad")),
            ("RESULTS_STORM_RATE", True, lambda d: d["silicon_parity_and_canary"]["storms"][0].__setitem__("rate_pct", True)),
            ("RESULTS_STORM_INT", True, lambda d: d["silicon_parity_and_canary"]["storms"][0].__setitem__("smem_sec", -1)),
            ("RESULTS_STORM_RED_MISMATCH", True, lambda d: d["silicon_parity_and_canary"]["storms"][1].__setitem__("atomic_reduction_ratio_q", "999/1")),
            ("RESULTS_STORM_POS", True, lambda d: d["silicon_parity_and_canary"]["storms"][0].__setitem__("warp_us_per_tok_q", "0/1")),
            ("RESULTS_STORM_SPEEDUP_MISMATCH", True, lambda d: d["silicon_parity_and_canary"]["storms"][1].__setitem__("latency_speedup_q", "99/1")),
            ("RESULTS_STORM_RATES", True, lambda d: d["silicon_parity_and_canary"]["storms"][3].__setitem__("rate_pct", 50)),
            ("RESULTS_BPB", True, lambda d: d.__setitem__("utf8_bpb_study", [])),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("seeds", [42, 137, 999])),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("d_model", 256)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("eval_bytes", 1024)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("tokens_per_seed", 1)),
            ("RESULTS_ARMS", True, lambda d: d["utf8_bpb_study"].__setitem__("arms", {})),
            ("RESULTS_BUT_BOUND", True, lambda d: d["utf8_bpb_study"].__setitem__("butterfly_vs_matrix_fp64_max_err_q", "1/10")),
            ("RESULTS_GRAD_ROUTING", True, lambda d: d["utf8_bpb_study"].__setitem__("arm1_upper_half_grad_norm_q", "1/10")),
            ("RESULTS_GRAD_ROUTING", True, lambda d: d["utf8_bpb_study"].__setitem__("arm3_upper_half_grad_norm_q", "0/1")),
            ("RESULTS_ARM_TYPE", True, lambda d: d["utf8_bpb_study"]["arms"].__setitem__("arm1_prefix_fwht", [])),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm3_learned_1kib_fp16_fold_fwht"].__setitem__("head_fp16_bytes", 2048)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"].__setitem__("total_params", 0)),
            ("RESULTS_ARM_SEEDS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"].__setitem__("seeds", [])),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"]["seeds"].__setitem__(0, "bad")),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"]["seeds"][0].__setitem__("eval_tokens", 100)),
            ("RESULTS_SEED_ORDER", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"]["seeds"][0].__setitem__("seed", 99)),
            ("RESULTS_SEED_BPB_POS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"]["seeds"][0].__setitem__("eval_bpb_q", "0/1")),
            ("RESULTS_SEED_BPB_POS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"]["seeds"][0].__setitem__("train_wall_s_q", "0/1")),
            ("RESULTS_SEED_QDIFF_MISMATCH", True, lambda d: d["utf8_bpb_study"]["arms"]["arm3_learned_1kib_fp16_fold_fwht"]["seeds"][0].__setitem__("fp16_quant_bpb_diff_q", "1/2")),
            ("RESULTS_ARM3_RMS_POS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm3_learned_1kib_fp16_fold_fwht"]["seeds"][0].__setitem__("alpha_rms_q", "0/1")),
            ("RESULTS_MEAN_MISMATCH", True, lambda d: d["utf8_bpb_study"]["arms"]["arm1_prefix_fwht"].__setitem__("mean_eval_bpb_q", "99/1")),
            ("RESULTS_DELTAS", True, lambda d: d.__setitem__("bpb_deltas", [])),
            ("RESULTS_DELTA_A1", True, lambda d: d["bpb_deltas"].__setitem__("arm3_minus_arm1_prefix_q", "99/1")),
            ("RESULTS_DELTA_A2", True, lambda d: d["bpb_deltas"].__setitem__("arm3_minus_arm2_fixed_diff_q", "99/1")),
            ("RESULTS_DELTA_A2B", True, lambda d: d["bpb_deltas"].__setitem__("arm3_minus_arm2b_1scalar_q", "99/1")),
            ("RESULTS_DELTA_A4", True, lambda d: d["bpb_deltas"].__setitem__("arm3_minus_arm4_dense_q", "99/1")),
            ("RESULTS_BEATS_FLAGS", True, lambda d: d["bpb_deltas"].__setitem__("arm3_beats_arm1_all_seeds", not d["bpb_deltas"]["arm3_beats_arm1_all_seeds"])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("G2_fold1k_bpb_and_latency", not d["gates"]["G2_fold1k_bpb_and_latency"])),
            ("RESULTS_VERDICT_MISMATCH", True, lambda d: d.__setitem__("verdict", "FALSIFIED")),
        ):
            r2 = copy.deepcopy(res)
            mut(r2)
            if resign:
                r2 = _resign(r2, "result_sha256")
            self.assertCode(code, vh.verify_results_receipt, r2, ptx)

        self.assertCode("RESULTS_BENCH_DISK", vh.verify_results_receipt, res, ptx, expected_bench_cu_sha="0" * 64)
        self.assertCode("RESULTS_BPB_SCRIPT_DISK", vh.verify_results_receipt, res, ptx, expected_bpb_script_sha="0" * 64)

        for field, val in (
            ("folded1k_cell_fp64_max_err_q", "0/1"),
            ("folded1k_cell_fp64_max_err_q", "1/100000"),
            ("folded1k_head_fp64_max_err_q", "0/1"),
            ("folded1k_head_fp64_max_err_q", "1/100000"),
        ):
            r_bnd = copy.deepcopy(res)
            r_bnd["silicon_parity_and_canary"][field] = val
            r_bnd = _resign(r_bnd, "result_sha256")
            self.assertEqual(vh.verify_results_receipt(r_bnd, ptx)["verdict"], "CONFIRMED")
        r_but = copy.deepcopy(res)
        r_but["utf8_bpb_study"]["butterfly_vs_matrix_fp64_max_err_q"] = "1/100000000000"
        r_but = _resign(r_but, "result_sha256")
        self.assertEqual(vh.verify_results_receipt(r_but, ptx)["verdict"], "CONFIRMED")

        ptx_spill = copy.deepcopy(ptx)
        ptx_spill["nonzero_spill_kernels"] = 1
        self.assertCode("RESULTS_GATES_MISMATCH", vh.verify_results_receipt, res, ptx_spill)

        # Pin sec16_packed_u32, ded120_packed_u32, multi-CTA overflow probes, and smem_sys_atomics == 1 on nonzero storms
        for mut in (
            lambda d: d["silicon_parity_and_canary"].__setitem__("sec16_packed_u32", 0),
            lambda d: d["silicon_parity_and_canary"].__setitem__("ded120_packed_u32", 0),
            lambda d: d["silicon_parity_and_canary"].__setitem__("overflow_sec70000_diag_u32", 0x80000000),
            lambda d: d["silicon_parity_and_canary"].__setitem__("multi_cta_sec_carry_packed_u32", 0),
            lambda d: d["silicon_parity_and_canary"].__setitem__("multi_cta_ded_wrap_packed_u32", 14),
            lambda d: d["silicon_parity_and_canary"]["storms"][1].__setitem__("smem_sec", 0),
            lambda d: d["silicon_parity_and_canary"]["storms"][1].__setitem__("warp_sec", 0),
            lambda d: (
                d["silicon_parity_and_canary"]["storms"][1].__setitem__("smem_sys_atomics", 0),
                d["silicon_parity_and_canary"]["storms"][1].__setitem__("atomic_reduction_ratio_q", "0/1"),
            ),
        ):
            r_bad = copy.deepcopy(res)
            mut(r_bad)
            r_bad = _resign(r_bad, "result_sha256")
            self.assertCode("RESULTS_GATES_MISMATCH", vh.verify_results_receipt, r_bad, ptx)

        r_partial = copy.deepcopy(res)
        r_partial["silicon_parity_and_canary"]["clean_sys_atomics"] = 1
        r_partial["gates"]["G3_smem_canary_atomic_and_recovery"] = False
        r_partial["verdict"] = "PARTIAL"
        r_partial = _resign(r_partial, "result_sha256")
        self.assertEqual(vh.verify_results_receipt(r_partial, ptx)["verdict"], "PARTIAL")

        r_fals = copy.deepcopy(r_partial)
        unf_q = vh.parse_q(r_fals["silicon_parity_and_canary"]["standalone_head_unfolded_us_q"])
        r_fals["silicon_parity_and_canary"]["standalone_head_folded1k_us_q"] = vh.format_q(unf_q + Fraction(1, 1))
        r_fals["silicon_parity_and_canary"]["standalone_head_overhead_us_q"] = "1/1"
        r_fals["gates"]["G2_fold1k_bpb_and_latency"] = False
        r_fals["verdict"] = "FALSIFIED"
        r_fals = _resign(r_fals, "result_sha256")
        self.assertEqual(vh.verify_results_receipt(r_fals, ptx)["verdict"], "FALSIFIED")

    def test_05_cli_and_on_disk_raw_byte_tamper_handling(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(vh.main(["--check", "--require-verdict", "CONFIRMED", "--json"]), 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["status"], "PASS")

        err_v = io.StringIO()
        with contextlib.redirect_stderr(err_v):
            self.assertEqual(vh.main(["--check", "--require-verdict", "PARTIAL"]), 2)
        self.assertEqual(json.loads(err_v.getvalue())["code"], "VERDICT_REQUIREMENT_FAILED")

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            self.assertEqual(vh.main(["--check"]), 0)
        self.assertTrue(buf2.getvalue().startswith("PASS "))

        # Subprocess entrypoint check kills if __name__ == "__main__" != mutant
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(VERIFY_DIR / "verify_smem_canary_folded_fwht.py"),
                "--check",
                "--require-verdict",
                "CONFIRMED",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("PASS CONFIRMED result_sha256="), proc.stdout)

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self.assertCode("MISSING_FILE", vh.load_strict_canonical_json, tdp / "missing.json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(vh.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err.getvalue())["code"], "MISSING_FILE")

            for rel in (
                vh.KERNEL_CU_REL,
                vh.BENCH_CU_REL,
                vh.BPB_STUDY_SCRIPT_REL,
                vh.PTXAS_RECEIPT_REL,
                vh.RESULTS_RECEIPT_REL,
            ):
                dst = tdp.joinpath(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(REPO_ROOT.joinpath(rel).read_bytes())
            self.assertEqual(vh.verify_on_disk(tdp)["status"], "PASS")

            # Duplicate-key JSON bypass attempt fails closed!
            res_raw = REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL).read_text(encoding="utf-8")
            dup_raw = "{\n  \"energy_power_status\": \"45.2 W claimed via RAPL\",\n" + res_raw[2:]
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_text(dup_raw, encoding="utf-8")
            self.assertCode("DUPLICATE_JSON_KEY", vh.verify_on_disk, tdp)

            # Non-finite JSON constant (NaN) and JSON float literal fail closed!
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_text("{\"x\": NaN}\n", encoding="utf-8")
            self.assertCode("NON_FINITE_JSON", vh.verify_on_disk, tdp)
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_text("{\"x\": 1.5}\n", encoding="utf-8")
            self.assertCode("JSON_FLOAT_FORBIDDEN", vh.verify_on_disk, tdp)

            # 4400-digit integer JSON, bounded int boundary, and depth-64/65 cap fail closed with structured RC=2
            self.assertEqual(vh._parse_bounded_int("9" * vh.MAX_Q_DIGITS), int("9" * vh.MAX_Q_DIGITS))
            self.assertEqual(vh._parse_bounded_int("-" + "9" * vh.MAX_Q_DIGITS), -int("9" * vh.MAX_Q_DIGITS))
            self.assertCode("READ_OR_PARSE_ERROR", vh._parse_bounded_int, "9" * (vh.MAX_Q_DIGITS + 1))
            self.assertCode("READ_OR_PARSE_ERROR", vh._parse_bounded_int, "-" + "9" * (vh.MAX_Q_DIGITS + 1))
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_text("{\"x\": " + ("9" * 4400) + "}\n", encoding="utf-8")
            err_big = io.StringIO()
            with contextlib.redirect_stderr(err_big):
                self.assertEqual(vh.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err_big.getvalue())["code"], "READ_OR_PARSE_ERROR")

            depth_p = tdp / "depth.json"
            ok_nest = {}
            cur_d = ok_nest
            for _ in range(vh.MAX_JSON_DEPTH):
                nxt = {}
                cur_d["a"] = nxt
                cur_d = nxt
            depth_p.write_text(json.dumps(ok_nest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertIsInstance(vh.load_strict_canonical_json(depth_p), dict)
            depth_p.write_text(('{"a":' * (vh.MAX_JSON_DEPTH + 1)) + "{}" + ("}" * (vh.MAX_JSON_DEPTH + 1)) + "\n", encoding="utf-8")
            self.assertCode("READ_OR_PARSE_ERROR", vh.load_strict_canonical_json, depth_p)
            depth_p.write_text('{"a":' + ("[" * vh.MAX_JSON_DEPTH) + "{}" + ("]" * vh.MAX_JSON_DEPTH) + "}\n", encoding="utf-8")
            self.assertCode("READ_OR_PARSE_ERROR", vh.load_strict_canonical_json, depth_p)
            depth_p.write_text('{"a":' + ("[" * (vh.MAX_JSON_DEPTH + 1)) + "1" + ("]" * (vh.MAX_JSON_DEPTH + 1)) + "}\n", encoding="utf-8")
            self.assertCode("READ_OR_PARSE_ERROR", vh.load_strict_canonical_json, depth_p)
            with mock.patch.object(vh.json, "dumps", side_effect=RecursionError("deep")):
                self.assertCode("READ_OR_PARSE_ERROR", vh._parse_strict_canonical_json_bytes, b"{}\n", "x")
            depth_p.unlink()

            # Symlink refusal on root, intermediate dirs, receipts, hashed source inputs, and self-read!
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).unlink()
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).symlink_to(REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL))
            err_sym = io.StringIO()
            with contextlib.redirect_stderr(err_sym):
                self.assertEqual(vh.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err_sym.getvalue())["code"], "SYMLINK_FORBIDDEN")
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).unlink()
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL).read_bytes())

            tdp.joinpath(vh.KERNEL_CU_REL).unlink()
            tdp.joinpath(vh.KERNEL_CU_REL).symlink_to(REPO_ROOT.joinpath(vh.KERNEL_CU_REL))
            self.assertCode("SYMLINK_FORBIDDEN", vh.verify_on_disk, tdp)
            tdp.joinpath(vh.KERNEL_CU_REL).unlink()
            tdp.joinpath(vh.KERNEL_CU_REL).write_bytes(REPO_ROOT.joinpath(vh.KERNEL_CU_REL).read_bytes())

            # Symlink root, non-directory root, symlink ancestor dir, & symlink self-parent fail closed
            sym_root = tdp.parent / (tdp.name + "_symroot")
            sym_root.symlink_to(tdp)
            try:
                self.assertCode("SYMLINK_FORBIDDEN", vh.verify_on_disk, sym_root)
            finally:
                sym_root.unlink()
            self.assertCode("MISSING_FILE", vh.verify_on_disk, tdp / "no_such_root")
            plain_root = tdp / "plain.txt"
            plain_root.write_text("x", encoding="utf-8")
            self.assertCode("NOT_REGULAR_FILE", vh.verify_on_disk, plain_root)
            plain_root.unlink()

            sym_parent_root = tdp / "sym_parent_repo"
            sym_parent_root.mkdir()
            (sym_parent_root / "cuda").symlink_to(tdp / "cuda")
            self.assertCode("SYMLINK_FORBIDDEN", vh.verify_on_disk, sym_parent_root)

            sym_self_dir = tdp / "sym_verify_dir"
            sym_self_dir.symlink_to(VERIFY_DIR)
            self.assertCode("SYMLINK_FORBIDDEN", vh._read_self_source_nofollow, sym_self_dir / "verify_smem_canary_folded_fwht.py")
            self.assertCode("MISSING_FILE", vh._read_self_source_nofollow, tdp / "no_dir" / "v.py")
            bad_utf8 = tdp / "bad_self.py"
            bad_utf8.write_bytes(b"\xff\xfe")
            self.assertCode("READ_OR_PARSE_ERROR", vh._read_self_source_nofollow, bad_utf8)
            bad_utf8.unlink()
            sym_self_dir.unlink()

            # Non-regular dir/FIFO fail closed BEFORE os.open(), and PermissionError/fstat OSError are wrapped cleanly
            with mock.patch.object(vh.os, "open", side_effect=AssertionError("MUST_NOT_OPEN")):
                self.assertCode("NOT_REGULAR_FILE", vh.sha256_file, tdp)
            fifo_p = tdp / "fifo.bin"
            os.mkfifo(fifo_p)
            with mock.patch.object(vh.os, "open", side_effect=AssertionError("MUST_NOT_OPEN")):
                self.assertCode("NOT_REGULAR_FILE", vh.sha256_file, fifo_p)
            fifo_p.unlink()

            fifo_rel_root = tdp / "fifo_rel_repo"
            fifo_rel_root.mkdir()
            for rel in (vh.KERNEL_CU_REL, vh.BENCH_CU_REL, vh.BPB_STUDY_SCRIPT_REL, vh.PTXAS_RECEIPT_REL, vh.RESULTS_RECEIPT_REL):
                dst = fifo_rel_root.joinpath(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(REPO_ROOT.joinpath(rel).read_bytes())
            fifo_rel_root.joinpath(vh.KERNEL_CU_REL).unlink()
            os.mkfifo(fifo_rel_root.joinpath(vh.KERNEL_CU_REL))
            real_open = os.open
            leaf_opens: list[str] = []

            def _track_leaf(p, flags, *a, **kw):
                if "holosphere_smem_canary_folded_fwht_sm86.cu" in str(p):
                    leaf_opens.append(str(p))
                return real_open(p, flags, *a, **kw)

            with mock.patch.object(vh.os, "open", side_effect=_track_leaf):
                self.assertCode("NOT_REGULAR_FILE", vh.verify_on_disk, fifo_rel_root)
            self.assertEqual(leaf_opens, [])

            with mock.patch.object(vh.os, "open", side_effect=PermissionError("denied root")):
                self.assertCode("MISSING_FILE", vh._read_rel_bytes_under_root_nofollow, tdp, vh.KERNEL_CU_REL)
            with mock.patch.object(vh.os, "open", side_effect=PermissionError("denied file")):
                self.assertCode("MISSING_FILE", vh.sha256_file, tdp.joinpath(vh.KERNEL_CU_REL))
            tdp.joinpath(vh.KERNEL_CU_REL).unlink()
            self.assertCode("MISSING_FILE", vh.verify_on_disk, tdp)
            tdp.joinpath(vh.KERNEL_CU_REL).write_bytes(REPO_ROOT.joinpath(vh.KERNEL_CU_REL).read_bytes())

            def _deny_leaf(p, flags, *a, **kw):
                if "holosphere_smem_canary_folded_fwht_sm86.cu" in str(p):
                    raise PermissionError("denied leaf")
                return real_open(p, flags, *a, **kw)

            with mock.patch.object(vh.os, "open", side_effect=_deny_leaf):
                self.assertCode("MISSING_FILE", vh.verify_on_disk, tdp)
            with mock.patch.object(vh.os, "fstat", side_effect=OSError("io fail")):
                self.assertCode("READ_OR_PARSE_ERROR", vh.sha256_file, tdp.joinpath(vh.KERNEL_CU_REL))
            with mock.patch.object(vh.os, "fstat", return_value=os.lstat(str(tdp))):
                self.assertCode("NOT_REGULAR_FILE", vh.sha256_file, tdp.joinpath(vh.KERNEL_CU_REL))

            # BPB study corpus SHA-256 constant binding fails closed if altered or unparseable
            self.assertCode("BPB_STUDY_CORPUS_PIN_MISMATCH", vh._verify_bpb_study_corpus_pins, b"EXPECTED_TRAIN_BIN_SHA256 = '0'\n")
            self.assertCode("READ_OR_PARSE_ERROR", vh._verify_bpb_study_corpus_pins, b"\xff\xfe")
            self.assertCode("READ_OR_PARSE_ERROR", vh._verify_bpb_study_corpus_pins, b"def (\n")

            big_bin = tdp / "oversized.bin"
            big_bin.write_bytes(b"0" * (vh.MAX_FILE_BYTES + 1))
            self.assertCode("FILE_TOO_LARGE", vh.sha256_file, big_bin)
            big_bin.write_bytes(b"0" * vh.MAX_FILE_BYTES)
            self.assertEqual(len(vh.sha256_file(big_bin)), 64)
            big_bin.unlink()

            # Single-file symlink refusal on sha256_file
            sym_single = tdp / "sym_single.bin"
            sym_single.symlink_to(tdp.joinpath(vh.KERNEL_CU_REL))
            self.assertCode("SYMLINK_FORBIDDEN", vh.sha256_file, sym_single)
            sym_single.unlink()

            # Non-canonical whitespace / trailing-byte tamper fails closed!
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL).read_bytes() + b"\n")
            self.assertCode("RAW_JSON_NOT_CANONICAL", vh.verify_on_disk, tdp)

            # Corrupted JSON syntax fails closed via CLI!
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_text("{bad", encoding="utf-8")
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                self.assertEqual(vh.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err2.getvalue())["code"], "READ_OR_PARSE_ERROR")

            # Restore receipts and verify 1-byte tamper of either committed .cu file or BPB study script fails closed
            tdp.joinpath(vh.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vh.RESULTS_RECEIPT_REL).read_bytes())
            tdp.joinpath(vh.BENCH_CU_REL).write_bytes(REPO_ROOT.joinpath(vh.BENCH_CU_REL).read_bytes() + b"\n")
            self.assertCode("RESULTS_BENCH_DISK", vh.verify_on_disk, tdp)
            tdp.joinpath(vh.BENCH_CU_REL).write_bytes(REPO_ROOT.joinpath(vh.BENCH_CU_REL).read_bytes())
            tdp.joinpath(vh.BPB_STUDY_SCRIPT_REL).write_bytes(REPO_ROOT.joinpath(vh.BPB_STUDY_SCRIPT_REL).read_bytes() + b"\n")
            self.assertCode("RESULTS_BPB_SCRIPT_DISK", vh.verify_on_disk, tdp)
            tdp.joinpath(vh.BPB_STUDY_SCRIPT_REL).write_bytes(REPO_ROOT.joinpath(vh.BPB_STUDY_SCRIPT_REL).read_bytes())
            tdp.joinpath(vh.KERNEL_CU_REL).write_bytes(REPO_ROOT.joinpath(vh.KERNEL_CU_REL).read_bytes() + b"\n")
            self.assertCode("PTXAS_CU_SHA_DISK", vh.verify_on_disk, tdp)


if __name__ == "__main__":
    unittest.main()
