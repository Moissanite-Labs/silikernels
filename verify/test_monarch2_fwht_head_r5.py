#!/usr/bin/env python3
"""Causality, Tamper-Refusal, Pure-Q AST, & Energy-Policy Test Suite for Monarch-2 + Byte-Bias FWHT Head.

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
    import verify_monarch2_fwht_head as vm
except SystemExit as exc:
    raise RuntimeError(f"verify_monarch2_fwht_head exited on import: {exc.code}") from exc


def _resign(doc: dict, key: str) -> dict:
    doc = copy.deepcopy(doc)
    doc[key] = vm.canonical_sha256(doc, exclude_key=key)
    return doc


class TestMonarch2FwhtHead(unittest.TestCase):
    def assertCode(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(vm.Monarch2HeadVerifyError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_hardened_pure_q_ast_audit_and_negatives(self) -> None:
        src = VERIFY_DIR.joinpath("verify_monarch2_fwht_head.py").read_text(encoding="utf-8")
        vm.verify_pure_q_ast(src)
        vm.verify_pure_q_ast(
            "from __future__ import annotations\n"
            "import ast\n"
            "from collections.abc import Sequence as AbcSequence\n"
            "from fractions import Fraction\n"
            "import json\n"
            "import sys\n"
            "import unicodedata\n"
            "def verify_pure_q_ast(source_text: str) -> None:\n"
            "    _ = ast.parse(source_text)\n"
            "def f(a: Fraction, s: str) -> Fraction:\n"
            "    b: Fraction = Fraction(1, 2)\n"
            "    c = Fraction(int(s), a.denominator * b.numerator)\n"
            "    _ = type(a) is int\n"
            "    _ = type(a).__name__\n"
            "    _ = type(a).__mro__\n"
            "    _ = unicodedata.normalize('NFKC', s)\n"
            "    _ = sys.stdout\n"
            "    _ = json.loads(s, parse_float=_reject_json_float, parse_constant=_reject_non_finite, parse_int=_parse_strict_json_int)\n"
            "    _ = json.dumps(1)\n"
            "    return c\n"
        )

        for code, bad_code in (
            ("AST_SYNTAX_ERROR", "def (\n"),
            ("AST_SYNTAX_ERROR", "x = 1\x00\n"),
            ("AST_DIV_FORBIDDEN", "x = 1 / 2\n"),
            ("AST_DIV_FORBIDDEN", "x /= 2\n"),
            ("AST_FLOORDIV_FORBIDDEN", "x = 1 // 2\n"),
            ("AST_FLOORDIV_FORBIDDEN", "x //= 2\n"),
            ("AST_MOD_FORBIDDEN", "x = '%f' % 1\n"),
            ("AST_MOD_FORBIDDEN", "fmt = '%f'\nx = fmt % 1\n"),
            ("AST_MOD_FORBIDDEN", "x %= 1\n"),
            ("AST_POW_FORBIDDEN", "x = 2 ** -1\n"),
            ("AST_BARE_ASSERT", "assert True\n"),
            ("AST_FORMAT_SPEC_FORBIDDEN", "x = f'{1:.3f}'\n"),
            ("AST_TYPE_ESCAPE", "from fractions import Fraction\nT = type(Fraction(1, 1))\nx = T('1.5')\n"),
            ("AST_TYPE_ESCAPE", "from fractions import Fraction\nx = type(Fraction(1, 2))('0.5')\n"),
            ("AST_TYPE_ESCAPE", "f = type\n"),
            ("AST_FLOAT_LITERAL", "x = 1.5\n"),
            ("AST_FLOAT_LITERAL", "x = 1j\n"),
            ("AST_FORBIDDEN_IMPORT", "import math\n"),
            ("AST_FORBIDDEN_IMPORT", "import decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import _decimal\n"),
            ("AST_FORBIDDEN_IMPORT", "import fractions\n"),
            ("AST_FORBIDDEN_IMPORT", "import struct\n"),
            ("AST_FORBIDDEN_IMPORT", "import array\n"),
            ("AST_FORBIDDEN_IMPORT", "import time\n"),
            ("AST_FORBIDDEN_IMPORT", "import random\n"),
            ("AST_FORBIDDEN_IMPORT", "import json as j\n"),
            ("AST_FORBIDDEN_IMPORT", "from _operator import truediv\nx = truediv(1, 2)\n"),
            ("AST_FORBIDDEN_IMPORT", "from _operator import truediv as Fraction\nx = Fraction(1, 2)\n"),
            ("AST_FORBIDDEN_IMPORT", "from math import sqrt\n"),
            ("AST_FORBIDDEN_IMPORT", "from struct import unpack\n"),
            ("AST_FORBIDDEN_IMPORT", "from json import loads\n"),
            ("AST_FORBIDDEN_IMPORT", "from importlib import import_module\n"),
            ("AST_FORBIDDEN_IMPORT", "from fractions import Fraction as F\n"),
            ("AST_FORBIDDEN_IMPORT", "from fractions import gcd\n"),
            ("AST_FORBIDDEN_IMPORT", "from collections.abc import Sequence\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import json\nj = json\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import json\nf = json.loads\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import json\nx = json.JSONEncoder\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import unicodedata\nx = unicodedata.numeric('\\u00bd')\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import unicodedata\nu = unicodedata\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import sys\nx = sys.float_info.max\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import ast\nx = ast.parse('1.5').body[0].value.value\n"),
            ("AST_FORBIDDEN_MOD_ATTR", "import ast\np = ast.parse\n"),
            ("AST_FRACTION_ARG_TYPE", "import sys\nfrom fractions import Fraction\nx = Fraction(int(sys.float_info.max), 1)\n"),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.loads('1.5')\n"),
            (
                "AST_JSON_FLOAT_PARSE",
                "import json\nx = json.loads('1.5', parse_float=None, parse_constant=_reject_non_finite, parse_int=_parse_strict_json_int)\n",
            ),
            (
                "AST_JSON_FLOAT_PARSE",
                "import json\nx = json.loads('1.5', parse_float=int, parse_constant=_reject_non_finite, parse_int=_parse_strict_json_int)\n",
            ),
            (
                "AST_JSON_FLOAT_PARSE",
                "import json\nx = json.loads('1.5', parse_float=_reject_json_float, parse_constant=int, parse_int=_parse_strict_json_int)\n",
            ),
            (
                "AST_JSON_FLOAT_PARSE",
                "import json\nx = json.loads('1.5', parse_float=_reject_json_float, parse_constant=_reject_non_finite, parse_int=int)\n",
            ),
            ("AST_JSON_FLOAT_PARSE", "import json\nx = json.load(fp)\n"),
            ("AST_JSON_FLOAT_PARSE", "x = loads('1.5')\n"),
            ("AST_RELATIVE_IMPORT", "from . import helper\n"),
            ("AST_RELATIVE_IMPORT", "from .foo import bar\n"),
            ("AST_WILDCARD_IMPORT", "from fractions import *\n"),
            ("AST_FRACTION_REBIND", "from fractions import Fraction\nF = Fraction\nx = F('0.5')\n"),
            ("AST_FRACTION_REBIND", "from fractions import Fraction\nF: object = Fraction\nx = F('0.5')\n"),
            ("AST_FRACTION_REBIND", "from fractions import Fraction\nx = (F := Fraction)('0.5')\n"),
            ("AST_FRACTION_REBIND", "from fractions import Fraction\nx = [Fraction][0]('0.5')\n"),
            ("AST_FRACTION_REBIND", "def Fraction(a, b):\n    return a\n"),
            ("AST_FRACTION_REBIND", "class Fraction:\n    pass\n"),
            ("AST_FRACTION_REBIND", "x = mod.Fraction\n"),
            ("AST_FRACTION_REBIND", "x = mod.Fraction(1, 2)\n"),
            ("AST_FORBIDDEN_NAME", "x = float(1)\n"),
            ("AST_FORBIDDEN_NAME", "x = pow(2, -1)\n"),
            ("AST_FORBIDDEN_NAME", "x = divmod(5, 2)\n"),
            ("AST_FORBIDDEN_NAME", "x = format(1, '.3f')\n"),
            ("AST_FORBIDDEN_NAME", "x = literal_eval('0.5')\n"),
            ("AST_FORBIDDEN_NAME", "x = memoryview(bytes(8))\n"),
            ("AST_FORBIDDEN_NAME", "x = __import__('math')\n"),
            ("AST_FORBIDDEN_NAME", "k = 'float'\nx = getattr(__builtins__, k)(1)\n"),
            ("AST_FORBIDDEN_ATTR", "import ast\nx = ast.literal_eval('0.5')\n"),
            ("AST_FORBIDDEN_ATTR", "x = m.cast('d')\n"),
            ("AST_FORBIDDEN_ATTR", "x = '{:.3f}'.format(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = '{:.3f}'.format_map({'a': 1})\n"),
            ("AST_FORBIDDEN_ATTR", "x = Fraction.from_float(1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__truediv__(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__getattribute__('__truediv__')(2)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__getattr__('x')\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__class__\n"),
            ("AST_FORBIDDEN_ATTR", "x = int.__bases__\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).real\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).imag\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).fromhex('1')\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).as_integer_ratio()\n"),
            ("AST_FORBIDDEN_ATTR", "x = (2).__pow__(-1)\n"),
            ("AST_FORBIDDEN_ATTR", "x = (1).__float__()\n"),
            ("AST_FORBIDDEN_ATTR", "import sys\nm = sys.modules['math']\n"),
            ("AST_FORBIDDEN_ATTR", "x = b.float(1)\n"),
            ("AST_FRACTION_ARGS", "x = Fraction(1)\n"),
            ("AST_FRACTION_ARGS", "x = Fraction(1, 2, k=3)\n"),
            ("AST_FRACTION_ARGS", "a = (1,)\nb = 2\nx = Fraction(*a, b)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction('1', 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(b'1', 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(f'{1}', 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(True, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(a, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(1 + 2, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(1 * a, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(a * 1, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(int(a, 10), 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(int(x=1), 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(int(a.b), 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(foo(1), 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(q.foo, 2)\n"),
            ("AST_FRACTION_ARG_TYPE", "x = Fraction(1, a)\n"),
        ):
            self.assertCode(code, vm.verify_pure_q_ast, bad_code)

        self.assertFalse(vm.contains_any_float({"a": [1, "2/3", {"b": (3, 4)}], "s": frozenset({"ok"})}))
        self.assertTrue(vm.contains_any_float({"a": [1, 1.5]}))
        self.assertTrue(vm.contains_any_float({1.2: "x"}))
        self.assertTrue(vm.contains_any_float({"a": (1.0,)}))

    def test_02_rational_kat_and_extended_energy_policy_guard(self) -> None:
        self.assertEqual(vm.parse_q("3/4"), Fraction(3, 4))
        self.assertEqual(vm.parse_q("-5/2"), Fraction(-5, 2))
        self.assertEqual(vm.parse_q("0/1"), Fraction(0, 1))
        self.assertEqual(vm.q_div(Fraction(3, 4), Fraction(2, 5)), Fraction(15, 8))
        self.assertCode("ZERO_DIV", vm.q_div, Fraction(1, 2), Fraction(0, 1))
        self.assertCode("BAD_RATIONAL", vm.parse_q, 1)
        self.assertCode("BAD_RATIONAL", vm.parse_q, "2/4")
        self.assertCode("BAD_RATIONAL", vm.parse_q, "1/0")
        self.assertEqual(vm.parse_q(("9" * vm.MAX_JSON_INT_DIGITS) + "/1"), Fraction(int("9" * vm.MAX_JSON_INT_DIGITS), 1))
        self.assertEqual(vm.parse_q("-" + ("9" * vm.MAX_JSON_INT_DIGITS) + "/1"), Fraction(-int("9" * vm.MAX_JSON_INT_DIGITS), 1))
        self.assertEqual(vm.parse_q("1/" + ("9" * vm.MAX_JSON_INT_DIGITS)), Fraction(1, int("9" * vm.MAX_JSON_INT_DIGITS)))
        self.assertCode("BAD_RATIONAL", vm.parse_q, ("9" * (vm.MAX_JSON_INT_DIGITS + 1)) + "/1")
        self.assertCode("BAD_RATIONAL", vm.parse_q, "1/" + ("9" * (vm.MAX_JSON_INT_DIGITS + 1)))
        self.assertCode("BAD_RATIONAL", vm.parse_q, ("9" * 4400) + "/1")
        self.assertCode("BAD_DIVISOR", vm.q_div_int, Fraction(1, 2), 0)
        self.assertCode("BAD_DIVISOR", vm.q_div_int, Fraction(1, 2), True)
        self.assertCode("BAD_JSON", vm.canonical_json_bytes, {"x": object()})
        self.assertFalse(vm.is_hex64("z" * 64))
        self.assertFalse(vm.is_hex64(123))

        # Integer digit-cap boundaries (kills <= vs < and sign stripping)
        self.assertEqual(vm._parse_strict_json_int("9" * vm.MAX_JSON_INT_DIGITS), int("9" * vm.MAX_JSON_INT_DIGITS))
        self.assertEqual(vm._parse_strict_json_int("-" + "9" * vm.MAX_JSON_INT_DIGITS), -int("9" * vm.MAX_JSON_INT_DIGITS))
        self.assertCode("INT_DIGIT_LIMIT", vm._parse_strict_json_int, "9" * (vm.MAX_JSON_INT_DIGITS + 1))

        # Independent FIPS 180-4 SHA-256 KAT ("abc")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "kat.bin"
            p.write_bytes(b"abc")
            self.assertEqual(
                vm.sha256_file(p),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

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
            "forty-five watts", "fortyfive watts", "energy: 12", "energy 12", "power: 45", "power 45",
            "wattage: 45", "2.25 A at 20 V", "20 V @ 2.25 A", "5000mAh", "j=5", "J/token", "W/tok", "250 W/ token",
            "tok_per_mwh", "nvmlDeviceGetTotalEnergyConsumption",
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
            "j = 3", "3 w/o pinning", "w/ tokens", "3 w/5 reps", "empowerment_score",
        )
        self.assertEqual(len(bad_values), 99)
        self.assertEqual(len(good_values), 55)
        for bad in bad_values:
            self.assertTrue(vm.is_forbidden_energy_value(bad), f"must reject {bad!r}")
        for good in good_values:
            self.assertFalse(vm.is_forbidden_energy_value(good), f"false positive on {good!r}")
        self.assertTrue(vm.is_forbidden_energy_key_name("5000mAh"))
        self.assertTrue(vm.is_forbidden_energy_key_name("w"))
        self.assertTrue(vm.is_forbidden_energy_key_name("j"))
        self.assertFalse(vm.is_forbidden_energy_key_name("empowerment_score"))
        self.assertFalse(vm.is_forbidden_energy_key_name("grad_W_q"))

        # ReDoS regression guards: strict linear-time on pathological digit & punctuation runs
        for probe in ("1" * 8000, "1." * 3000, "1" + "," * 4000, "1" + "." * 10000, "1" + "." * 8000 + "x", "1" + " " * 8000):
            t0 = time.process_time()
            self.assertFalse(vm.is_forbidden_energy_value(probe))
            self.assertLess(time.process_time() - t0, 0.10)

        self.assertFalse(
            vm.has_forbidden_energy_claim(
                {
                    "energy_power_status": vm.ENERGY_POLICY_STATUS,
                    "kernels": {vm.ARM9_MONARCH2_SYMBOL: {"registers": 27}},
                    "ok": list(good_values),
                    "t": ("sm_86",),
                    "f": frozenset({"16896 B"}),
                    1: "clean",
                }
            )
        )
        for bad_obj in (
            {"mJ_per_tok": 1},
            {"avgPower": 45},
            {"milliwatts_avg": 1},
            {"kilojoules": 1},
            {"powerdraw": 1},
            {"wattage": 1},
            {"rapl_uj": 1},
            {"nvml_power": 1},
            {"po\u0301wer": 45},
            {"\u0435nergy": 1},
            {"max_45w": 1},
            {"pwr": 1},
            {"rapl": "ok"},
            {"nvml": "ok"},
            {"nvidia_smi": "ok"},
            {"nested": {"energy_power_status": vm.ENERGY_POLICY_STATUS}},
            {"kernels": {vm.ARM9_MONARCH2_SYMBOL: {"note": "45.2 W"}}},
            {"kernels": {"fake_pj_symbol": {"registers": 8}}},
            {"t": ("45.2 W",)},
            {1: "45.2 W"},
        ):
            self.assertTrue(vm.has_forbidden_energy_claim(bad_obj), f"must reject {bad_obj!r}")

    def test_03_pure_z_monarch2_fwht_oracles_and_scale_tokens(self) -> None:
        o = vm.verify_monarch2_and_fwht_oracle()
        self.assertTrue(o["h256_involution_verified"])
        self.assertEqual(o["bent4_scaled_norm_sq_multiplier"], 8)
        self.assertEqual(o["bent4_kat_delta_vectors_verified"], 6)
        self.assertEqual(o["monarch1_reach_count"], 16)
        self.assertEqual(o["monarch2_full_reach_count"], 256)
        self.assertEqual(o["monarch2_bias_fp16_bytes"], 16896)

        # Contextual scale-token, kernel-def, & BPB-script AST binding (comment-only fakes MUST fail!)
        k_txt = REPO_ROOT.joinpath(vm.KERNEL_CU_REL).read_text(encoding="utf-8")
        b_txt = REPO_ROOT.joinpath(vm.BENCH_CU_REL).read_text(encoding="utf-8")
        s_txt = REPO_ROOT.joinpath(vm.BPB_STUDY_SCRIPT_REL).read_text(encoding="utf-8")
        sc = vm.verify_cuda_scale_tokens(k_txt, b_txt, s_txt)
        self.assertEqual(sc["plain_fwht_scale_q"], "1/16")
        self.assertEqual(sc["bent4_fwht_scale_sq_q"], "1/32")
        self.assertEqual(sc["bent4_norm_sq_multiplier"], 8)
        self.assertTrue(sc["bpb_script_ast_verified"])
        self.assertCode(
            "CU_DISABLED_CODE_FORBIDDEN",
            vm.verify_cuda_scale_tokens,
            k_txt + "\n#if 0\nfloat x = 0.0f;\n#endif\n",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_DISABLED_CODE_FORBIDDEN",
            vm.verify_cuda_scale_tokens,
            k_txt + "\n#if 1-1\nfloat fake = 0.0f;\n#endif\n",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_DISABLED_CODE_FORBIDDEN",
            vm.verify_cuda_scale_tokens,
            k_txt + "\n#ifdef NEVER_DEFINED\nfloat fake2 = 0.0f;\n#endif\n",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_DISABLED_CODE_FORBIDDEN",
            vm.verify_cuda_scale_tokens,
            k_txt + "\n#i\\\nf 0\nfloat fake3 = 0.0f;\n#endif\n",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "BENCH_DISABLED_CODE_FORBIDDEN",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt + "\n#if 0\nint x = 0;\n#endif\n",
            s_txt,
        )
        self.assertCode(
            "CU_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            "// fwht256_plain_8warp() { v *= 0.0625f; }\n/* fwht256_bent4_from_smem_8warp() { v *= 0.17677669529663688f; } */",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            'const char *s = R"(fwht256_plain_8warp() { v *= 0.0625f; } fwht256_bent4_from_smem_8warp() { v *= 0.17677669529663688f; })";',
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            "float fwht256_plain_8warp() {}\nfloat other() { v *= 0.0625f; }",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            "float fwht256_plain_8warp() { v *= 0.0625f;",
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_KERNEL_DEF_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt.replace("holosphere_arm9_monarch2_bias_16p5kib_sm86", "renamed_kernel"),
            b_txt,
            s_txt,
        )
        self.assertCode(
            "CU_KERNEL_DEF_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt.replace("__global__", "static"),
            b_txt,
            s_txt,
        )
        self.assertCode(
            "BENCH_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            "/* fwht256_plain_fp64() { out[k] = acc * 0.0625; } fwht256_bent4_fp64() { inv_sqrt32 = 1.0 / std::sqrt(32.0); } */",
            s_txt,
        )
        self.assertCode(
            "BENCH_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            'const char *s = "fwht256_plain_fp64() { out[k] = acc * 0.0625; } fwht256_bent4_fp64() { inv_sqrt32 = 1.0 / std::sqrt(32.0); }";',
            s_txt,
        )
        self.assertCode("BPB_SCRIPT_SYNTAX_ERROR", vm.verify_cuda_scale_tokens, k_txt, b_txt, "def (\n")
        self.assertCode("BPB_SCRIPT_SYNTAX_ERROR", vm.verify_cuda_scale_tokens, k_txt, b_txt, "x = 1\x00\n")
        self.assertCode("BPB_SCRIPT_BARE_ASSERT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "assert True\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_NAME", vm.verify_cuda_scale_tokens, k_txt, b_txt, "x = __import__('os')\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_NAME", vm.verify_cuda_scale_tokens, k_txt, b_txt, "globals()['CORPUS_ID'] = 'evil'\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_IMPORT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "import pynvml\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_IMPORT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "import torch.nn as bad\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_IMPORT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "from subprocess import run\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_IMPORT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "from fractions import Fraction as F\n")
        self.assertCode("BPB_SCRIPT_FORBIDDEN_IMPORT", vm.verify_cuda_scale_tokens, k_txt, b_txt, "from . import helper\n")
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID = 123\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID = get_id()\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID = 'a'\nCORPUS_ID = 'b'\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID = '0'\nif False:\n    CORPUS_ID = 'real'\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID, _unused = ('evil', 0)\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "[*CORPUS_ID] = ['evil']\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "del CORPUS_ID\n",
        )
        self.assertCode(
            "BPB_SCRIPT_PIN_REBIND",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "CORPUS_ID: str = 'a'\n",
        )
        self.assertCode(
            "BPB_SCRIPT_RUN_DEF_INVALID",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            '"""0.0625 1.0 / math.sqrt(32.0)"""\n# h256_fp64 = [0.0625]\n',
        )
        self.assertCode(
            "BPB_SCRIPT_RUN_DEF_INVALID",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            s_txt + "\ndef run_bpb_study():\n    return 'tampered'\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    if False:\n        h256_fp64 = [0.0625]\n        inv_sqrt32 = 1.0 / math.sqrt(32.0)\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n    h256_fp64 = [999]\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n    inv_sqrt32 = 999\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n    del inv_sqrt32\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n    inv_sqrt32 += 1\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.5]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n",
        )
        self.assertCode(
            "BPB_SCRIPT_SCALE_TOKEN_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(16.0)\n",
        )
        self.assertCode(
            "BPB_SCRIPT_CORPUS_PINS_MISSING",
            vm.verify_cuda_scale_tokens,
            k_txt,
            b_txt,
            "def run_bpb_study():\n    h256_fp64 = [0.0625]\n    inv_sqrt32 = 1.0 / math.sqrt(32.0)\n",
        )

        self.assertCode("BAD_FWHT_LEN", vm.fwht256_plain_int, [0] * 16)
        self.assertCode("BAD_FWHT_LEN", vm.fwht256_bent4_int, [0] * 16)
        self.assertCode("BAD_MONARCH_LEN", vm.monarch16_stage_allones_int, [0] * 16)
        self.assertCode("ORACLE_H256_INVOLUTION", vm.verify_monarch2_and_fwht_oracle, plain_fn=lambda u: [0] * 256)
        self.assertCode("ORACLE_PARSEVAL_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=lambda u: [0] * 256)

        # Planted wrong isometric FWHTs (all satisfy Parseval; ALL MUST fail ORACLE_BENT4_KAT_FAIL!)
        def _drop_bent4_signs(u):
            buf = [0] * 256
            for idx in range(256):
                w, lane = idx >> 5, idx & 31
                buf[idx] = u[((w ^ (lane & 7)) << 5) | lane]
            return vm.fwht256_plain_int(buf)

        def _drop_warp_xor(u):
            buf = [0] * 256
            for idx in range(256):
                lane = idx & 31
                b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
                buf[idx] = -u[idx] if b4 else u[idx]
            return vm.fwht256_plain_int(buf)

        self.assertCode("ORACLE_BENT4_KAT_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=_drop_bent4_signs)
        self.assertCode("ORACLE_BENT4_KAT_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=_drop_warp_xor)
        self.assertCode("ORACLE_BENT4_KAT_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=lambda u: vm.fwht256_bent4_int(list(reversed(u))))
        self.assertCode("ORACLE_BENT4_KAT_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=lambda u: (lambda y: y[1:] + y[:1])(vm.fwht256_bent4_int(u)))
        self.assertCode("ORACLE_BENT4_KAT_FAIL", vm.verify_monarch2_and_fwht_oracle, bent4_fn=lambda u: [-v for v in vm.fwht256_bent4_int(u)])

        self.assertCode("ORACLE_MONARCH2_REACH", vm.verify_monarch2_and_fwht_oracle, stage_fn=lambda u: [0] * 256)
        calls = {"n": 0}
        def _stage1_only(u):
            calls["n"] += 1
            return vm.monarch16_stage_allones_int(u) if calls["n"] == 1 else [1] * 16 + [0] * 240
        self.assertCode("ORACLE_MONARCH2_REACH", vm.verify_monarch2_and_fwht_oracle, stage_fn=_stage1_only)

    def test_04_live_on_disk_receipts_and_tamper_refusal(self) -> None:
        summary = vm.verify_on_disk(REPO_ROOT)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["verdict"], "CONFIRMED")
        self.assertTrue(summary["cuda_scale_binding"]["scale_tokens_verified"])

        ptx = vm.load_strict_canonical_json(REPO_ROOT.joinpath(vm.PTXAS_RECEIPT_REL))
        res = vm.load_strict_canonical_json(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL))

        self.assertCode("PTXAS_TYPE", vm.verify_ptxas_receipt, [])
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
            ("PTXAS_NVCC", True, lambda d: d.__setitem__("nvcc_version", "nvcc 13.4")),
            ("PTXAS_CU_PATH", True, lambda d: d.__setitem__("cu_path", "wrong.cu")),
            ("PTXAS_CU_SHA", True, lambda d: d.__setitem__("cu_sha256", "bad")),
            ("PTXAS_KERNELS", True, lambda d: d.__setitem__("kernels", {})),
            ("PTXAS_REQUIRED_SYMBOLS", True, lambda d: (d["kernels"].pop(vm.ARM2B_BIAS_SYMBOL), d["kernels"].__setitem__("fake", copy.deepcopy(ptx["kernels"][vm.ARM2B_BIAS_SYMBOL])))),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"].__setitem__(vm.ARM2B_BIAS_SYMBOL, [])),
            ("PTXAS_KSTAT", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("extra", 0)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("registers", -1)),
            ("PTXAS_FIELD_TYPE", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("registers", True)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vm.ARM9_MONARCH2_SYMBOL].__setitem__("registers", 30)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vm.ARM9_MONARCH2_SYMBOL].__setitem__("smem_bytes", 1024)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("stack_bytes", 8)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("spill_store_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("spill_load_bytes", 4)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("ldl_insns", 1)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("stl_insns", 1)),
            ("PTXAS_KERNEL_SPEC", True, lambda d: d["kernels"][vm.ARM2B_BIAS_SYMBOL].__setitem__("registers", 0)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("total_kernels", 5)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("zero_spill_kernels", 3)),
            ("PTXAS_SPILL_FAIL", True, lambda d: d.__setitem__("nonzero_spill_kernels", 1)),
        ):
            p2 = copy.deepcopy(ptx)
            mut(p2)
            if resign:
                p2 = _resign(p2, "receipt_sha256")
            self.assertCode(code, vm.verify_ptxas_receipt, p2)
        self.assertCode("PTXAS_CU_SHA_DISK", vm.verify_ptxas_receipt, ptx, expected_cu_sha="0" * 64)

        self.assertCode("RESULTS_TYPE", vm.verify_results_receipt, [], ptx)
        for code, resign, mut in (
            ("RESULTS_SCHEMA", False, lambda d: d.__setitem__("schema", "bad")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("energy_power_status", "BAD")),
            ("RESULTS_ENERGY_POLICY", True, lambda d: d.__setitem__("mj_per_tok", "1/2")),
            ("RESULTS_FLOAT_FORBIDDEN", True, lambda d: d.__setitem__("x", 0.5)),
            ("RESULTS_KEYS", True, lambda d: d.__setitem__("bogus_external_sha256", "0" * 64)),
            ("RESULTS_SHA_FORMAT", False, lambda d: d.__setitem__("result_sha256", "bad")),
            ("RESULTS_SHA_MISMATCH", False, lambda d: d.__setitem__("verdict", "FALSIFIED")),
            ("RESULTS_HARDWARE", True, lambda d: d.__setitem__("hardware", [])),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("extra", 1)),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("device", 123)),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("sm", "9.0")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("nvcc_version", "other")),
            ("RESULTS_HARDWARE", True, lambda d: d["hardware"].__setitem__("device", "")),
            ("RESULTS_BINDINGS", True, lambda d: d.__setitem__("source_bindings", [])),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_path", "wrong.cu")),
            ("RESULTS_BINDING_PATHS", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_path", "wrong.py")),
            ("RESULTS_CU_BIND", True, lambda d: d["source_bindings"].__setitem__("kernel_cu_sha256", "0" * 64)),
            ("RESULTS_PTX_BIND", True, lambda d: d["source_bindings"].__setitem__("ptxas_receipt_sha256", "0" * 64)),
            ("RESULTS_BENCH_SHA", True, lambda d: d["source_bindings"].__setitem__("bench_cu_sha256", "bad")),
            ("RESULTS_BPB_SCRIPT_SHA", True, lambda d: d["source_bindings"].__setitem__("bpb_study_script_sha256", "bad")),
            ("RESULTS_CORPUS_ID", True, lambda d: d["source_bindings"].__setitem__("corpus_id", "other")),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_sha256", "bad")),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_train_bin_sha256", "bad")),
            ("RESULTS_CORPUS_SHA", True, lambda d: d["source_bindings"].__setitem__("corpus_eval_bin_sha256", "bad")),
            ("RESULTS_REF2B_POS", True, lambda d: d.__setitem__("unbiased_scalar_ref_bpb_q", "0/1")),
            ("RESULTS_SILICON", True, lambda d: d.__setitem__("silicon_sm86_bench", [])),
            ("RESULTS_PARITY_BOUND", True, lambda d: d["silicon_sm86_bench"].__setitem__("arm9_monarch2_bias_fp64_max_err_q", "1/100000")),
            ("RESULTS_US_POS", True, lambda d: d["silicon_sm86_bench"].__setitem__("arm9_monarch2_bias_us_per_tok_q", "0/1")),
            ("RESULTS_BPB", True, lambda d: d.__setitem__("utf8_bpb_study", [])),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("seeds", [42, 137, 999])),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("d_model", 256)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("eval_chunks", True)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("seq_len", True)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("batch_size", True)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("steps_per_seed", True)),
            ("RESULTS_BPB_META", True, lambda d: (d["utf8_bpb_study"].__setitem__("eval_chunks", 0), d["utf8_bpb_study"].__setitem__("eval_bytes", 0))),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("eval_bytes", 1024)),
            ("RESULTS_BPB_META", True, lambda d: d["utf8_bpb_study"].__setitem__("tokens_per_seed", 1)),
            ("RESULTS_H256_BOUND", True, lambda d: d["utf8_bpb_study"].__setitem__("h256_ortho_fp64_max_err_q", "1/10")),
            ("RESULTS_ARMS", True, lambda d: d["utf8_bpb_study"].__setitem__("arms", {})),
            ("RESULTS_ARM_TYPE", True, lambda d: d["utf8_bpb_study"]["arms"].__setitem__("arm2b_bias_514B", [])),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"].__setitem__("head_params", 0)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"].__setitem__("head_fp16_bytes", 1024)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"].__setitem__("hbm_vocab_table_bytes", 16)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"].__setitem__("total_params", True)),
            ("RESULTS_ARM_BYTES", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"].__setitem__("total_params", 0)),
            ("RESULTS_ARM_SEEDS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"].__setitem__("seeds", [])),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"].__setitem__(0, "bad")),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("extra", 1)),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("seed", True)),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("eval_tokens", True)),
            ("RESULTS_SEED_ROW", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("eval_tokens", 100)),
            ("RESULTS_SEED_ORDER", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("seed", 99)),
            ("RESULTS_SEED_BPB_POS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("eval_bpb_q", "0/1")),
            ("RESULTS_SEED_BPB_POS", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"]["seeds"][0].__setitem__("train_wall_s_q", "0/1")),
            ("RESULTS_SEED_QDIFF_MISMATCH", True, lambda d: d["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"]["seeds"][0].__setitem__("fp16_quant_bpb_diff_q", "1/2")),
            ("RESULTS_MEAN_MISMATCH", True, lambda d: d["utf8_bpb_study"]["arms"]["arm2b_bias_514B"].__setitem__("mean_eval_bpb_q", "99/1")),
            ("RESULTS_SUMMARY", True, lambda d: d.__setitem__("bpb_summary", [])),
            ("RESULTS_SUM_BIAS", True, lambda d: d["bpb_summary"].__setitem__("bias_only_lift_over_ref2b_q", "99/1")),
            ("RESULTS_SUM_A5_REF", True, lambda d: d["bpb_summary"].__setitem__("arm5_lift_over_ref2b_q", "99/1")),
            ("RESULTS_SUM_A5_2B", True, lambda d: d["bpb_summary"].__setitem__("arm5_lift_over_2b_bias_q", "99/1")),
            ("RESULTS_SUM_A5_8A", True, lambda d: d["bpb_summary"].__setitem__("arm5_minus_arm8a_q", "99/1")),
            ("RESULTS_SUM_A7_REF", True, lambda d: d["bpb_summary"].__setitem__("arm7_lift_over_ref2b_q", "99/1")),
            ("RESULTS_SUM_A7_2B", True, lambda d: d["bpb_summary"].__setitem__("arm7_lift_over_2b_bias_q", "99/1")),
            ("RESULTS_SUM_A7_8A", True, lambda d: d["bpb_summary"].__setitem__("arm7_minus_arm8a_q", "99/1")),
            ("RESULTS_SUM_A9_REF", True, lambda d: d["bpb_summary"].__setitem__("arm9_lift_over_ref2b_q", "99/1")),
            ("RESULTS_SUM_A9_2B", True, lambda d: d["bpb_summary"].__setitem__("arm9_lift_over_2b_bias_q", "99/1")),
            ("RESULTS_SUM_A9_8A", True, lambda d: d["bpb_summary"].__setitem__("arm9_minus_arm8a_q", "99/1")),
            ("RESULTS_SUM_FLAGS", True, lambda d: d["bpb_summary"].__setitem__("arm5_lt_8a_all_seeds", not d["bpb_summary"]["arm5_lt_8a_all_seeds"])),
            ("RESULTS_SUM_FLAGS", True, lambda d: d["bpb_summary"].__setitem__("arm7_lt_2b_bias_all_seeds", not d["bpb_summary"]["arm7_lt_2b_bias_all_seeds"])),
            ("RESULTS_SUM_FLAGS", True, lambda d: d["bpb_summary"].__setitem__("arm7_lt_8a_all_seeds", not d["bpb_summary"]["arm7_lt_8a_all_seeds"])),
            ("RESULTS_SUM_FLAGS", True, lambda d: d["bpb_summary"].__setitem__("arm9_lt_2b_bias_all_seeds", not d["bpb_summary"]["arm9_lt_2b_bias_all_seeds"])),
            ("RESULTS_SUM_FLAGS", True, lambda d: d["bpb_summary"].__setitem__("arm9_lt_8a_all_seeds", not d["bpb_summary"]["arm9_lt_8a_all_seeds"])),
            ("RESULTS_SUM_BEST_ARM", True, lambda d: d["bpb_summary"].__setitem__("best_structured_arm", "arm5_bias_2p5KiB")),
            ("RESULTS_SUM_BEST_LIFT", True, lambda d: d["bpb_summary"].__setitem__("best_lift_over_ref2b_q", "99/1")),
            ("RESULTS_SUM_DGAP", True, lambda d: d["bpb_summary"].__setitem__("dense_gap_ref2b_minus_arm4_q", "99/1")),
            ("RESULTS_SUM_GCLOSED", True, lambda d: d["bpb_summary"].__setitem__("best_gap_closed_fraction_q", "99/1")),
            ("RESULTS_SUM_BEST_A4", True, lambda d: d["bpb_summary"].__setitem__("best_minus_arm4_dense_q", "99/1")),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d.__setitem__("gates", [])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("extra", True)),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("G1_bpb_bias_and_lowrank_beat", not d["gates"]["G1_bpb_bias_and_lowrank_beat"])),
            ("RESULTS_GATES_MISMATCH", True, lambda d: d["gates"].__setitem__("G2_sm86_zero_spill_latency_parity", not d["gates"]["G2_sm86_zero_spill_latency_parity"])),
            ("RESULTS_VERDICT_MISMATCH", True, lambda d: d.__setitem__("verdict", "FALSIFIED")),
        ):
            r2 = copy.deepcopy(res)
            mut(r2)
            if resign:
                r2 = _resign(r2, "result_sha256")
            self.assertCode(code, vm.verify_results_receipt, r2, ptx)

        self.assertCode("RESULTS_BENCH_DISK", vm.verify_results_receipt, res, ptx, expected_bench_cu_sha="0" * 64)
        self.assertCode("RESULTS_BPB_SCRIPT_DISK", vm.verify_results_receipt, res, ptx, expected_bpb_script_sha="0" * 64)
        self.assertCode(
            "RESULTS_CORPUS_PIN_MISMATCH",
            vm.verify_results_receipt,
            res,
            ptx,
            expected_corpus_pins=(vm.EXPECTED_CORPUS_ID, "0" * 64, "0" * 64, "0" * 64),
        )

        # Boundary tests: parity=0/1, h256=1e-11, latency=3/1 (PASS) vs >3/1 on best OR non-best arm (FALSIFIED)
        r_bnd = copy.deepcopy(res)
        r_bnd["silicon_sm86_bench"]["arm9_monarch2_bias_fp64_max_err_q"] = "0/1"
        r_bnd["silicon_sm86_bench"]["arm9_monarch2_bias_us_per_tok_q"] = "3/1"
        r_bnd["utf8_bpb_study"]["h256_ortho_fp64_max_err_q"] = "1/100000000000"
        r_bnd = _resign(r_bnd, "result_sha256")
        self.assertEqual(vm.verify_results_receipt(r_bnd, ptx)["verdict"], "CONFIRMED")

        r_fals = copy.deepcopy(res)
        r_fals["silicon_sm86_bench"]["arm9_monarch2_bias_us_per_tok_q"] = "4/1"
        r_fals["gates"]["G2_sm86_zero_spill_latency_parity"] = False
        r_fals["verdict"] = "FALSIFIED"
        r_fals = _resign(r_fals, "result_sha256")
        self.assertEqual(vm.verify_results_receipt(r_fals, ptx)["verdict"], "FALSIFIED")

        r_fals_other = copy.deepcopy(res)
        r_fals_other["silicon_sm86_bench"]["arm2b_bias_us_per_tok_q"] = "4/1"
        r_fals_other["gates"]["G2_sm86_zero_spill_latency_parity"] = False
        r_fals_other["verdict"] = "FALSIFIED"
        r_fals_other = _resign(r_fals_other, "result_sha256")
        self.assertEqual(vm.verify_results_receipt(r_fals_other, ptx)["verdict"], "FALSIFIED")

        # Synthetic PARTIAL + dense_gap == 0 branch
        r_part = copy.deepcopy(res)
        a4_mean_s = r_part["utf8_bpb_study"]["arms"]["arm4_dense_bias_262KiB"]["mean_eval_bpb_q"]
        r_part["unbiased_scalar_ref_bpb_q"] = a4_mean_s
        ref_q = vm.parse_q(a4_mean_s)
        m = {k: vm.parse_q(r_part["utf8_bpb_study"]["arms"][k]["mean_eval_bpb_q"]) for k in vm._EXPECTED_ARMS}
        bs = r_part["bpb_summary"]
        bs["bias_only_lift_over_ref2b_q"] = vm.format_q(ref_q - m["arm2b_bias_514B"])
        bs["arm5_lift_over_ref2b_q"] = vm.format_q(ref_q - m["arm5_bias_2p5KiB"])
        bs["arm7_lift_over_ref2b_q"] = vm.format_q(ref_q - m["arm7_bias_8p5KiB"])
        bs["arm9_lift_over_ref2b_q"] = vm.format_q(ref_q - m["arm9_monarch2_bias_16p5KiB"])
        bs["best_lift_over_ref2b_q"] = vm.format_q(ref_q - m[bs["best_structured_arm"]])
        bs["dense_gap_ref2b_minus_arm4_q"] = "0/1"
        bs["best_gap_closed_fraction_q"] = "0/1"
        r_part["gates"]["G1_bpb_bias_and_lowrank_beat"] = False
        r_part["verdict"] = "PARTIAL"
        r_part = _resign(r_part, "result_sha256")
        self.assertEqual(vm.verify_results_receipt(r_part, ptx)["verdict"], "PARTIAL")

        # Synthetic G1 passed via arm7 alone (kills g1_arm7 or g1_arm9 BoolOp!)
        r_a7 = copy.deepcopy(res)
        for row in r_a7["utf8_bpb_study"]["arms"]["arm7_bias_8p5KiB"]["seeds"]:
            row["eval_bpb_q"] = "4/1"
            row["eval_bpb_bf16_q"] = "4/1"
            row["fp16_quant_bpb_diff_q"] = "0/1"
        r_a7["utf8_bpb_study"]["arms"]["arm7_bias_8p5KiB"]["mean_eval_bpb_q"] = "4/1"
        for row in r_a7["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"]["seeds"]:
            row["eval_bpb_q"] = "9/2"
            row["eval_bpb_bf16_q"] = "9/2"
            row["fp16_quant_bpb_diff_q"] = "0/1"
        r_a7["utf8_bpb_study"]["arms"]["arm9_monarch2_bias_16p5KiB"]["mean_eval_bpb_q"] = "9/2"
        ref_orig = vm.parse_q(r_a7["unbiased_scalar_ref_bpb_q"])
        m7 = {k: vm.parse_q(r_a7["utf8_bpb_study"]["arms"][k]["mean_eval_bpb_q"]) for k in vm._EXPECTED_ARMS}
        bs7 = r_a7["bpb_summary"]
        bs7["arm7_lift_over_ref2b_q"] = vm.format_q(ref_orig - m7["arm7_bias_8p5KiB"])
        bs7["arm7_lift_over_2b_bias_q"] = vm.format_q(m7["arm2b_bias_514B"] - m7["arm7_bias_8p5KiB"])
        bs7["arm7_minus_arm8a_q"] = vm.format_q(m7["arm7_bias_8p5KiB"] - m7["arm8a_lowrank16_bias_16p9KiB"])
        bs7["arm9_lift_over_ref2b_q"] = vm.format_q(ref_orig - m7["arm9_monarch2_bias_16p5KiB"])
        bs7["arm9_lift_over_2b_bias_q"] = vm.format_q(m7["arm2b_bias_514B"] - m7["arm9_monarch2_bias_16p5KiB"])
        bs7["arm9_minus_arm8a_q"] = vm.format_q(m7["arm9_monarch2_bias_16p5KiB"] - m7["arm8a_lowrank16_bias_16p9KiB"])
        bs7["arm9_lt_8a_all_seeds"] = False
        bs7["best_structured_arm"] = "arm7_bias_8p5KiB"
        bs7["best_lift_over_ref2b_q"] = vm.format_q(ref_orig - m7["arm7_bias_8p5KiB"])
        bs7["best_gap_closed_fraction_q"] = vm.format_q(vm.q_div(ref_orig - m7["arm7_bias_8p5KiB"], ref_orig - m7["arm4_dense_bias_262KiB"]))
        bs7["best_minus_arm4_dense_q"] = vm.format_q(m7["arm7_bias_8p5KiB"] - m7["arm4_dense_bias_262KiB"])
        r_a7["gates"]["G1_bpb_bias_and_lowrank_beat"] = True
        r_a7["verdict"] = "CONFIRMED"
        r_a7 = _resign(r_a7, "result_sha256")
        self.assertEqual(vm.verify_results_receipt(r_a7, ptx)["verdict"], "CONFIRMED")

        ptx_spill = copy.deepcopy(ptx)
        ptx_spill["nonzero_spill_kernels"] = 1
        self.assertCode("RESULTS_GATES_MISMATCH", vm.verify_results_receipt, res, ptx_spill)

    def test_05_cli_subprocess_and_on_disk_raw_byte_tamper_handling(self) -> None:
        # Real isolated subprocess test verifying `if __name__ == "__main__":` & PASS CONFIRMED stdout
        verifier_script = str(VERIFY_DIR.joinpath("verify_monarch2_fwht_head.py"))
        proc_ok = subprocess.run(
            [sys.executable, "-I", "-B", verifier_script, "--check", "--require-verdict", "CONFIRMED"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(proc_ok.returncode, 0)
        self.assertIn("PASS CONFIRMED", proc_ok.stdout)

        # Closed-pipe EPIPE handling (must exit cleanly with RC=2, NEVER RC=120 / BrokenPipeError)
        r_out, w_out = os.pipe()
        os.close(r_out)
        try:
            proc_epipe_out = subprocess.run(
                [sys.executable, "-I", "-B", verifier_script, "--check", "--json"],
                stdout=w_out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        finally:
            os.close(w_out)
        self.assertEqual(proc_epipe_out.returncode, 2)
        self.assertNotIn("BrokenPipeError", proc_epipe_out.stderr)

        r_err, w_err = os.pipe()
        os.close(r_err)
        try:
            proc_epipe_err = subprocess.run(
                [sys.executable, "-I", "-B", verifier_script, "--check", "--repo-root", "/nonexistent_s156b_root"],
                stdout=subprocess.PIPE,
                stderr=w_err,
                text=True,
                timeout=20,
            )
        finally:
            os.close(w_err)
        self.assertEqual(proc_epipe_err.returncode, 2)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(vm.main(["--check", "--require-verdict", "CONFIRMED", "--json"]), 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["status"], "PASS")

        err_v = io.StringIO()
        with contextlib.redirect_stderr(err_v):
            self.assertEqual(vm.main(["--check", "--require-verdict", "PARTIAL"]), 2)
        self.assertEqual(json.loads(err_v.getvalue())["code"], "VERDICT_REQUIREMENT_FAILED")

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            self.assertEqual(vm.main(["--check"]), 0)
        self.assertTrue(buf2.getvalue().startswith("PASS "))

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            self.assertCode("MISSING_FILE", vm.load_strict_canonical_json, tdp / "missing.json")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(vm.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err.getvalue())["code"], "MISSING_FILE")

            for rel in (
                vm.KERNEL_CU_REL,
                vm.BENCH_CU_REL,
                vm.BPB_STUDY_SCRIPT_REL,
                vm.PTXAS_RECEIPT_REL,
                vm.RESULTS_RECEIPT_REL,
            ):
                dst = tdp.joinpath(rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(REPO_ROOT.joinpath(rel).read_bytes())
            self.assertEqual(vm.verify_on_disk(tdp)["status"], "PASS")

            # Symlink, ancestor-symlink, non-regular (dir/FIFO), unreadable, & size-cap refusal
            res_copy = tdp / "res_real.json"
            res_copy.write_bytes(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_bytes())
            res_dst = tdp.joinpath(vm.RESULTS_RECEIPT_REL)
            res_dst.unlink()
            res_dst.symlink_to(res_copy)
            self.assertCode("SYMLINK_FORBIDDEN", vm.verify_on_disk, tdp)
            proc_sym = subprocess.run(
                [sys.executable, "-I", "-B", verifier_script, "--check", "--repo-root", str(tdp)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(proc_sym.returncode, 2)
            self.assertEqual(json.loads(proc_sym.stderr)["code"], "SYMLINK_FORBIDDEN")
            res_dst.unlink()
            res_dst.write_bytes(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_bytes())

            # Symlinked repo_root and symlinked intermediate directory fail closed
            root_link = tdp.parent / (tdp.name + "_link")
            try:
                root_link.symlink_to(tdp)
                self.assertCode("SYMLINK_FORBIDDEN", vm.verify_on_disk, root_link)
            finally:
                if root_link.is_symlink():
                    root_link.unlink()

            real_receipts = tdp / "receipts_real"
            tdp.joinpath("receipts").rename(real_receipts)
            tdp.joinpath("receipts").symlink_to(real_receipts)
            self.assertCode("SYMLINK_FORBIDDEN", vm.verify_on_disk, tdp)
            tdp.joinpath("receipts").unlink()
            real_receipts.rename(tdp / "receipts")

            # Non-directory repo_root / intermediate component & root "/" fail closed (root-to-leaf dir_fd walk)
            self.assertCode("ROOT_PATH_FORBIDDEN", vm.sha256_file, Path("/"))
            self.assertCode("MISSING_FILE", vm.sha256_file, Path("/__nonexistent_top_level_s156b__"))
            above_link = tdp / "above_sym"
            real_sub = tdp / "real_sub"
            real_sub.mkdir()
            above_link.symlink_to(real_sub)
            self.assertCode("SYMLINK_FORBIDDEN", vm.verify_on_disk, above_link / "nested_root")

            not_a_dir = tdp / "plain_file"
            not_a_dir.write_text("x", encoding="ascii")
            self.assertCode("NOT_DIRECTORY", vm.verify_on_disk, not_a_dir)

            tdp.joinpath("receipts").rename(real_receipts)
            tdp.joinpath("receipts").write_text("not_a_dir", encoding="ascii")
            self.assertCode("NOT_DIRECTORY", vm.verify_on_disk, tdp)
            tdp.joinpath("receipts").unlink()
            real_receipts.rename(tdp / "receipts")

            # Non-regular directory & FIFO (O_NONBLOCK prevents hang) fail closed
            self.assertCode("NOT_REGULAR_FILE", vm.load_strict_canonical_json, tdp / "receipts")
            fifo_p = tdp / "test.fifo"
            os.mkfifo(fifo_p)
            self.assertCode("NOT_REGULAR_FILE", vm.sha256_file, fifo_p)

            # Unreadable file & os.read OSError (O_WRONLY fd -> EBADF) fail closed with sanitized READ_OR_PARSE_ERROR
            unreadable_p = tdp / "unreadable.json"
            unreadable_p.write_text("{}\n", encoding="ascii")
            wronly_fd = os.open(str(unreadable_p), os.O_WRONLY)
            self.assertCode("READ_OR_PARSE_ERROR", vm._read_fd_regular_bytes, wronly_fd, name="unreadable.json")
            unreadable_p.chmod(0o000)
            try:
                self.assertCode("READ_OR_PARSE_ERROR", vm.load_strict_canonical_json, unreadable_p)
            finally:
                unreadable_p.chmod(0o600)

            # Pre-parse + iterative JSON depth cap exact boundary (MAX_JSON_DEPTH passes; +1 and deep-2000 fail JSON_DEPTH_LIMIT)
            d_ok = {"a": 1}
            for _ in range(vm.MAX_JSON_DEPTH - 1):
                d_ok = [d_ok]
            vm._check_json_depth(d_ok)
            self.assertCode("JSON_DEPTH_LIMIT", vm._check_json_depth, [d_ok])

            d_ok_bytes = (
                json.dumps(d_ok, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
            ).encode("ascii")
            self.assertEqual(vm.parse_strict_canonical_json_bytes(d_ok_bytes, name="ok32.json"), d_ok)

            vm._check_raw_json_nesting_depth("[" * vm.MAX_JSON_DEPTH + "]" * vm.MAX_JSON_DEPTH)
            vm._check_raw_json_nesting_depth("{" * vm.MAX_JSON_DEPTH + "}" * vm.MAX_JSON_DEPTH)
            self.assertCode(
                "JSON_DEPTH_LIMIT",
                vm._check_raw_json_nesting_depth,
                "[" * (vm.MAX_JSON_DEPTH + 1) + "]" * (vm.MAX_JSON_DEPTH + 1),
            )
            self.assertCode(
                "JSON_DEPTH_LIMIT",
                vm._check_raw_json_nesting_depth,
                "{" * (vm.MAX_JSON_DEPTH + 1) + "}" * (vm.MAX_JSON_DEPTH + 1),
            )
            vm._check_raw_json_nesting_depth("[" + "{}" * 40 + "]")
            vm._check_raw_json_nesting_depth("[" + "[]" * 40 + "]")
            self.assertCode(
                "JSON_DEPTH_LIMIT",
                vm._check_raw_json_nesting_depth,
                ("]" * 40) + ("[" * (vm.MAX_JSON_DEPTH + 1)) + ("]" * (vm.MAX_JSON_DEPTH + 1)),
            )
            str_ok_val = 'x"y\\z' + ("[" * 40) + ("{" * 40)
            str_ok_bytes = (
                json.dumps({"a": str_ok_val}, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
            ).encode("ascii")
            self.assertEqual(vm.parse_strict_canonical_json_bytes(str_ok_bytes, name="str_ok.json")["a"], str_ok_val)
            self.assertCode(
                "JSON_DEPTH_LIMIT",
                vm._check_raw_json_nesting_depth,
                '{"a": "x\\\\"' + ("[" * (vm.MAX_JSON_DEPTH + 1)) + ("]" * (vm.MAX_JSON_DEPTH + 1)) + "}",
            )

            with mock.patch.object(
                vm.json,
                "loads",
                side_effect=RuntimeError("json.loads must not run on over-depth JSON"),
            ):
                self.assertCode(
                    "JSON_DEPTH_LIMIT",
                    vm.parse_strict_canonical_json_bytes,
                    (b"[" * 2000) + (b"]" * 2000),
                    name="deep.json",
                )
                self.assertCode(
                    "JSON_DEPTH_LIMIT",
                    vm.parse_strict_canonical_json_bytes,
                    (b"{" * (vm.MAX_JSON_DEPTH + 1)) + (b"}" * (vm.MAX_JSON_DEPTH + 1)),
                    name="deep33_obj.json",
                )

            old_rec_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(150)
            try:
                self.assertEqual(vm.parse_strict_canonical_json_bytes(d_ok_bytes, name="ok32_lowrec.json"), d_ok)
                self.assertCode(
                    "JSON_DEPTH_LIMIT",
                    vm.parse_strict_canonical_json_bytes,
                    (b"[" * 2000) + (b"]" * 2000),
                    name="deep_lowrec.json",
                )
            finally:
                sys.setrecursionlimit(old_rec_limit)

            with mock.patch.object(vm.json, "loads", side_effect=RecursionError("simulated recursion limit")):
                self.assertCode("JSON_DEPTH_LIMIT", vm.parse_strict_canonical_json_bytes, b"{}\n", name="rec.json")

            # File size cap exact boundary (MAX_FILE_BYTES passes; MAX_FILE_BYTES + 1 fails FILE_TOO_LARGE)
            max_json_p = tdp / "max_ok.json"
            max_json_p.write_bytes(('{\n  "a": "' + ("x" * (vm.MAX_FILE_BYTES - 14)) + '"\n}\n').encode("ascii"))
            self.assertEqual(len(max_json_p.read_bytes()), vm.MAX_FILE_BYTES)
            self.assertEqual(len(vm.load_strict_canonical_json(max_json_p)["a"]), vm.MAX_FILE_BYTES - 14)
            too_big_p = tdp / "too_big.bin"
            too_big_p.write_bytes(b"a" * (vm.MAX_FILE_BYTES + 1))
            self.assertCode("FILE_TOO_LARGE", vm.sha256_file, too_big_p)

            # Float literal in JSON fails closed with JSON_FLOAT_FORBIDDEN
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_text("{\n  \"x\": 1.5\n}\n", encoding="ascii")
            self.assertCode("JSON_FLOAT_FORBIDDEN", vm.verify_on_disk, tdp)

            # Invalid UTF-8 bytes in CUDA source fails closed with READ_OR_PARSE_ERROR
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_bytes())
            tdp.joinpath(vm.KERNEL_CU_REL).write_bytes(b"\xff\xfe")
            self.assertCode("READ_OR_PARSE_ERROR", vm.verify_on_disk, tdp)
            tdp.joinpath(vm.KERNEL_CU_REL).write_bytes(REPO_ROOT.joinpath(vm.KERNEL_CU_REL).read_bytes())

            # 4,400-digit integer JSON MUST exit RC=2 structured JSON (never uncaught RC=1 ValueError!)
            huge_int_json = "{\n  \"x\": " + ("9" * 4400) + "\n}\n"
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_text(huge_int_json, encoding="ascii")
            self.assertCode("INT_DIGIT_LIMIT", vm.verify_on_disk, tdp)
            proc_huge = subprocess.run(
                [sys.executable, "-I", "-B", verifier_script, "--check", "--repo-root", str(tdp)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(proc_huge.returncode, 2)
            self.assertEqual(json.loads(proc_huge.stderr)["code"], "INT_DIGIT_LIMIT")

            # Duplicate-key JSON bypass attempt fails closed!
            res_raw = REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_text(encoding="utf-8")
            dup_raw = "{\n  \"energy_power_status\": \"45.2 W claimed via RAPL\",\n" + res_raw[2:]
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_text(dup_raw, encoding="utf-8")
            self.assertCode("DUPLICATE_JSON_KEY", vm.verify_on_disk, tdp)

            # Non-finite JSON constant (NaN) fails closed!
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_text("{\"x\": NaN}\n", encoding="utf-8")
            self.assertCode("NON_FINITE_JSON", vm.verify_on_disk, tdp)

            # Non-canonical whitespace / trailing-byte tamper fails closed!
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_bytes() + b"\n")
            self.assertCode("RAW_JSON_NOT_CANONICAL", vm.verify_on_disk, tdp)

            # Corrupted JSON syntax fails closed via CLI!
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_text("{bad", encoding="utf-8")
            err2 = io.StringIO()
            with contextlib.redirect_stderr(err2):
                self.assertEqual(vm.main(["--check", "--repo-root", str(tdp)]), 2)
            self.assertEqual(json.loads(err2.getvalue())["code"], "READ_OR_PARSE_ERROR")

            # Restore receipts and verify 1-byte tamper of .cu or BPB study script fails closed
            tdp.joinpath(vm.RESULTS_RECEIPT_REL).write_bytes(REPO_ROOT.joinpath(vm.RESULTS_RECEIPT_REL).read_bytes())
            tdp.joinpath(vm.BPB_STUDY_SCRIPT_REL).write_bytes(REPO_ROOT.joinpath(vm.BPB_STUDY_SCRIPT_REL).read_bytes() + b"\n")
            self.assertCode("RESULTS_BPB_SCRIPT_DISK", vm.verify_on_disk, tdp)
            tdp.joinpath(vm.BPB_STUDY_SCRIPT_REL).write_bytes(REPO_ROOT.joinpath(vm.BPB_STUDY_SCRIPT_REL).read_bytes())
            tdp.joinpath(vm.BENCH_CU_REL).write_bytes(REPO_ROOT.joinpath(vm.BENCH_CU_REL).read_bytes() + b"\n")
            self.assertCode("RESULTS_BENCH_DISK", vm.verify_on_disk, tdp)
            tdp.joinpath(vm.BENCH_CU_REL).write_bytes(REPO_ROOT.joinpath(vm.BENCH_CU_REL).read_bytes())
            tdp.joinpath(vm.KERNEL_CU_REL).write_bytes(REPO_ROOT.joinpath(vm.KERNEL_CU_REL).read_bytes() + b"\n")
            self.assertCode("PTXAS_CU_SHA_DISK", vm.verify_on_disk, tdp)


if __name__ == "__main__":
    unittest.main()
