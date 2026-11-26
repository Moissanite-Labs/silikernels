#!/usr/bin/env python3
"""Pure-stdlib assertion-free R5 test suite for verify_holosphere_receipts.py (100% mutant kill).

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFY_DIR = REPO_ROOT / "verify"
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

import verify_holosphere_receipts as vhr


def _copy_minimal_repo(dst: Path) -> None:
    for rel in (
        "cuda/sm86/holosphere_megakernel_sm86.cu",
        "cuda/sm86/holosphere_h256_chunk32_sm86.cu",
        "cuda/sm86/holosphere_v3_frontier_sm86.cu",
        "cuda/sm86/su2_holonomy_scan_sm86.cu",
        "include/razor_abi_4byte_sys.h",
        "receipts/s116_ptxas_zero_spill_receipt.json",
        "receipts/s116_phase2b_gpu_results.json",
        "receipts/mlos_zone_b_s116_phase2b.json",
        "receipts/s117_ptxas_zero_spill_receipt.json",
        "receipts/s117_live_rtx3080_zone_b_results.json",
        "receipts/s117_live_v2_d512_zone_b_results.json",
        "receipts/mlos_zone_b_s117_live.json",
        "receipts/mlos_zone_b_s117_v2_d512.json",
        "receipts/v3_frontier_ptxas_zero_spill_receipt.json",
        "receipts/s117_v3_frontier_live_gpu_results.json",
        "receipts/mlos_zone_b_s117_v3_frontier.json",
        "receipts/su2_holonomy_sm86_ptxas_receipt.json",
    ):
        src = REPO_ROOT / rel
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)


class TestVerifyHoloSphereReceiptsR5(unittest.TestCase):
    def test_01_live_repo_50_zero_spill_kernels_4_zone_b_receipts_and_cli(self) -> None:
        res = vhr.verify_all(REPO_ROOT)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["total_holosphere_zero_spill_kernels"], 50)
        self.assertEqual(len(res["ptxas_targets"]), 4)
        self.assertEqual(len(res["zone_b_mlos_receipts"]), 4)
        self.assertGreater(res["live_benchmarks"]["v3_hmma_bwd_tok_per_sec"], 148000.0)
        self.assertLess(res["live_benchmarks"]["v3_d512_cellop_graph_us"], 21.0)
        self.assertLess(res["live_benchmarks"]["v3_d512_e2e_graph_us"], 22.0)
        self.assertEqual(res["razor_abi"]["canary_fields"], 11)
        self.assertEqual(res["razor_abi"]["pi16_fields"], 4)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = vhr.main(["--check", "--json", "--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["summary_sha256"], res["summary_sha256"])

        buf_txt = io.StringIO()
        with redirect_stdout(buf_txt):
            rc_t = vhr.main(["--check", "--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc_t, 0)
        self.assertIn("holosphere_kernels=50/50", buf_txt.getvalue())

        # Subprocess -I -B deterministic parity
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(VERIFY_DIR / "verify_holosphere_receipts.py"), "--check", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["summary_sha256"], res["summary_sha256"])

    def test_02_strict_json_and_ptxas_tamper_controls_incl_duplicate_symbols(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="silikernels_r5_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        _copy_minimal_repo(td)

        # load_strict_json guards: missing, malformed, non-finite NaN/Inf, non-object []
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_nf:
            vhr.load_strict_json(td / "missing.json")
        self.assertEqual(cm_nf.exception.code, "ERR_MISSING_FILE")

        bad_json = td / "bad.json"
        for text, code in (
            ("{bad", "ERR_BAD_JSON"),
            ('{"x": NaN}', "ERR_JSON_NONFINITE"),
            ('{"x": Infinity}', "ERR_JSON_NONFINITE"),
            ("[]", "ERR_JSON_OBJECT"),
        ):
            bad_json.write_text(text, encoding="utf-8")
            with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_j:
                vhr.load_strict_json(bad_json)
            self.assertEqual(cm_j.exception.code, code)

        # Missing .cu source
        s117_spec = vhr.PTXAS_SPECS[1]
        s116_spec = vhr.PTXAS_SPECS[0]
        v3_spec = vhr.PTXAS_SPECS[2]
        cu_bak = (td / s117_spec["cu_rel"]).read_bytes()
        (td / s117_spec["cu_rel"]).unlink()
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_nc:
            vhr.verify_ptxas_spec(td, s117_spec)
        self.assertEqual(cm_nc.exception.code, "ERR_MISSING_CU")
        (td / s117_spec["cu_rel"]).write_bytes(cu_bak)

        # Tamper S117 ptxas receipt across EVERY guard (incl. 11x duplicate symbol, unhashable symbol, SASS ops)
        s117_path = td / s117_spec["rec_rel"]
        s117_orig = json.loads(s117_path.read_text(encoding="utf-8"))
        for code, mut in (
            ("ERR_CU_SHA", lambda r: r.__setitem__("cu_sha256", "0" * 64)),
            ("ERR_KERNEL_COUNT", lambda r: r["kernels"].pop()),
            ("ERR_KERNEL_COUNT", lambda r: r.__setitem__("total_kernels", 10)),
            ("ERR_NONZERO_SPILL", lambda r: r.__setitem__("nonzero_spill_kernels", 1)),
            ("ERR_SPILL", lambda r: r["kernels"][0].__setitem__("stack_frame_bytes", 8)),
            ("ERR_SPILL", lambda r: r["kernels"][0].__setitem__("stack_frame_bytes", False)),
            ("ERR_SPILL", lambda r: r["kernels"][0].__setitem__("spill_store_bytes", 8)),
            ("ERR_SPILL", lambda r: r["kernels"][0].__setitem__("spill_load_bytes", 8)),
            ("ERR_SPILL", lambda r: r["kernels"][0].__setitem__("zero_spill", False)),
            ("ERR_SPILL", lambda r: r["kernels"].__setitem__(0, "not_a_dict")),
            ("ERR_SYMBOL_SET", lambda r: r["kernels"][0].__setitem__("symbol", "")),
            ("ERR_SYMBOL_SET", lambda r: r["kernels"][0].__setitem__("symbol", ["unhashable"])),
            ("ERR_SYMBOL_SET", lambda r: r.__setitem__("kernels", [copy.deepcopy(r["kernels"][0])] * 11)),
            ("ERR_SYMBOL_SET", lambda r: r["kernels"][0].__setitem__("symbol", "wrong_symbol")),
            ("ERR_SASS_SCHEMA", lambda r: r.__setitem__("sass_instruction_counts", [])),
            ("ERR_SASS_FORBIDDEN", lambda r: r["sass_instruction_counts"].__setitem__("LDL", 1)),
            ("ERR_SASS_FORBIDDEN", lambda r: r["sass_instruction_counts"].__setitem__("LDL", False)),
            ("ERR_SASS_FORBIDDEN", lambda r: r["sass_instruction_counts"].__setitem__("MEMBAR_GL", 1)),
            ("ERR_SASS_OPS", lambda r: r["sass_instruction_counts"].__setitem__("RED_SYS", 1)),
            ("ERR_SASS_OPS", lambda r: r["sass_instruction_counts"].__setitem__("RED_SYS", True)),
            ("ERR_SASS_OPS", lambda r: r["sass_instruction_counts"].__setitem__("IDP4A", 15)),
        ):
            bad = copy.deepcopy(s117_orig)
            mut(bad)
            s117_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_t:
                vhr.verify_ptxas_spec(td, s117_spec)
            self.assertEqual(cm_t.exception.code, code)
        s117_path.write_text(json.dumps(s117_orig), encoding="utf-8")

        # Exact SASS floor boundary tests (RED_SYS == 2, IDP4A == 16, IMMA_16832 == 6 must PASS -> kills >= to >)
        s117_edge = copy.deepcopy(s117_orig)
        s117_edge["sass_instruction_counts"]["RED_SYS"] = 2
        s117_edge["sass_instruction_counts"]["IDP4A"] = 16
        s117_edge["sass_instruction_counts"]["IMMA_16832"] = 6
        s117_path.write_text(json.dumps(s117_edge), encoding="utf-8")
        self.assertEqual(vhr.verify_ptxas_spec(td, s117_spec)["zero_spill_kernels"], 11)
        s117_path.write_text(json.dumps(s117_orig), encoding="utf-8")

        # Tamper S116 all_zero_spill flag & V3 Frontier 2x duplicate symbol
        s116_path = td / s116_spec["rec_rel"]
        s116_orig = json.loads(s116_path.read_text(encoding="utf-8"))
        bad_s116 = copy.deepcopy(s116_orig)
        bad_s116["all_zero_spill"] = False
        s116_path.write_text(json.dumps(bad_s116), encoding="utf-8")
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_az:
            vhr.verify_ptxas_spec(td, s116_spec)
        self.assertEqual(cm_az.exception.code, "ERR_NONZERO_SPILL")
        s116_path.write_text(json.dumps(s116_orig), encoding="utf-8")

        v3_path = td / v3_spec["rec_rel"]
        v3_orig = json.loads(v3_path.read_text(encoding="utf-8"))
        bad_v3 = copy.deepcopy(v3_orig)
        bad_v3["kernels"] = [copy.deepcopy(v3_orig["kernels"][0])] * 2
        v3_path.write_text(json.dumps(bad_v3), encoding="utf-8")
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_v3d:
            vhr.verify_ptxas_spec(td, v3_spec)
        self.assertEqual(cm_v3d.exception.code, "ERR_SYMBOL_SET")

    def test_03_zone_b_mlos_live_benchmarks_and_razor_abi_tamper_controls(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="silikernels_r5_b_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        _copy_minimal_repo(td)

        # ML-OS Zone-B receipt tamper controls
        mlos_rel, mlos_sha = vhr.ZONE_B_MLOS_RECEIPTS[0]
        mlos_path = td / mlos_rel
        mlos_orig = json.loads(mlos_path.read_text(encoding="utf-8"))
        for code, mut in (
            ("ERR_MLOS_CLEAN", lambda r: r.__setitem__("schema", "bad")),
            ("ERR_MLOS_CLEAN", lambda r: r.__setitem__("determinism_zone", "A")),
            ("ERR_MLOS_CLEAN", lambda r: r.__setitem__("clean_hardware_run", False)),
            ("ERR_MLOS_CLEAN", lambda r: r.__setitem__("anomalies", ["throttle"])),
            ("ERR_MLOS_EXIT", lambda r: r["execution"].__setitem__("exit_code", 1)),
            ("ERR_MLOS_EXIT", lambda r: r["execution"].__setitem__("exit_code", False)),
            ("ERR_MLOS_THERMAL", lambda r: r["end_gpu_telemetry"]["clocks_event_reasons"].__setitem__("thermal_throttled", True)),
            ("ERR_MLOS_SHA", lambda r: r.__setitem__("receipt_sha256", "0" * 64)),
            ("ERR_MLOS_SHA", lambda r: r.__setitem__("profile_id", "tampered")),
        ):
            bad = copy.deepcopy(mlos_orig)
            mut(bad)
            mlos_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_m:
                vhr.verify_zone_b_mlos_receipt(mlos_path, mlos_sha)
            self.assertEqual(cm_m.exception.code, code)
        mlos_path.write_text(json.dumps(mlos_orig), encoding="utf-8")

        # Live benchmark JSON tamper controls across all 4 result files (including exact boundary values!)
        def tamper_bench(rel: str, code: str, mut: object) -> None:
            p = td / rel
            orig = json.loads(p.read_text(encoding="utf-8"))
            bad = copy.deepcopy(orig)
            mut(bad)  # type: ignore[operator]
            p.write_text(json.dumps(bad), encoding="utf-8")
            try:
                with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_b:
                    vhr.verify_live_benchmark_results(td)
                self.assertEqual(cm_b.exception.code, code)
            finally:
                p.write_text(json.dumps(orig), encoding="utf-8")

        tamper_bench("receipts/s116_phase2b_gpu_results.json", "ERR_S116_RESULTS", lambda r: r["phase2b_live_parity"].__setitem__("all_parity_pass", False))
        tamper_bench("receipts/s116_phase2b_gpu_results.json", "ERR_S116_RESULTS", lambda r: r["h16a_v2_matched_12layer_showdown"]["uniform_cell_8warp_d256_1cta"].__setitem__("us_per_tok", 35.0))
        tamper_bench("receipts/s116_phase2b_gpu_results.json", "ERR_S116_RESULTS", lambda r: r["h16a_v2_matched_12layer_showdown"]["uniform_cell_8warp_d256_1cta"].__setitem__("us_per_tok", False))
        tamper_bench("receipts/s116_phase2b_gpu_results.json", "ERR_S116_RESULTS", lambda r: r["h16c_v2_l2_persistence_and_weight_in_a"].__setitem__("t8192_96mib_polluted_4pass_pinned_gb_s", 500.0))

        tamper_bench("receipts/s117_live_rtx3080_zone_b_results.json", "ERR_S117_V1_RESULTS", lambda r: r.__setitem__("all_live_gates_pass", False))
        tamper_bench("receipts/s117_live_rtx3080_zone_b_results.json", "ERR_S117_V1_RESULTS", lambda r: r["bare_metal_rtx3080"]["live_sys_ber_canary"].__setitem__("ded1_canary_count", True))
        tamper_bench("receipts/s117_live_rtx3080_zone_b_results.json", "ERR_S117_V1_RESULTS", lambda r: r["bare_metal_rtx3080"]["chunk32_woodbury_bwd_ab_t8192"].__setitem__("speedup_choice_b_over_a", 1.2))
        tamper_bench("receipts/s117_live_rtx3080_zone_b_results.json", "ERR_S117_V1_RESULTS", lambda r: r["live_gpu_torch_compile_and_t8192_vram"].__setitem__("t8192_live_peak_allocated_mib", 650.0))

        tamper_bench("receipts/s117_live_v2_d512_zone_b_results.json", "ERR_S117_V2_RESULTS", lambda r: r["fwht256_bent4_lm_head_parity"].__setitem__("max_abs_err_vs_fp64_dense", 1e-6))
        tamper_bench("receipts/s117_live_v2_d512_zone_b_results.json", "ERR_S117_V2_RESULTS", lambda r: r["full_e2e_fwht256_12L_d512"].__setitem__("t8192_us_per_tok", 28.0))

        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r.__setitem__("all_parity_pass", False))
        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r["v3_d512"].__setitem__("sec16_hidden_max_abs_err", False))
        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r["v3_d512"].__setitem__("cellop_graph_us_per_tok_8192", 21.0))
        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r["v3_d512"].__setitem__("e2e_graph_us_per_tok_16384", 22.0))
        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r["chunk32_bwd_wmma"].__setitem__("B4_T8192_D512_L12_tok_per_sec", 140000.0))
        tamper_bench("receipts/s117_v3_frontier_live_gpu_results.json", "ERR_V3_RESULTS", lambda r: r["chunk32_bwd_wmma"].__setitem__("speedup_vs_choice_b_532ms", 2.4))

        # Direct helper non-finite & bool rejection guards
        self.assertFalse(vhr._is_real_num(float("nan")))
        self.assertFalse(vhr._is_real_num(float("inf")))
        self.assertFalse(vhr._is_real_num(False))
        self.assertFalse(vhr._is_zero_real(False))
        self.assertTrue(vhr._is_zero_real(0.0))

        # Razor ABI header tamper controls
        abi_path = td / "include" / "razor_abi_4byte_sys.h"
        abi_orig = abi_path.read_text(encoding="utf-8")
        abi_path.unlink()
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_am:
            vhr.verify_razor_abi_header(abi_path)
        self.assertEqual(cm_am.exception.code, "ERR_ABI_MISSING")

        abi_path.write_text(abi_orig.replace("_ReadWriteBarrier();", ""), encoding="utf-8")
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_ac:
            vhr.verify_razor_abi_header(abi_path)
        self.assertEqual(cm_ac.exception.code, "ERR_ABI_CONTRACT")

        abi_path.write_text(abi_orig + "\n/* _InterlockedCompareExchange */\n", encoding="utf-8")
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_au:
            vhr.verify_razor_abi_header(abi_path)
        self.assertEqual(cm_au.exception.code, "ERR_ABI_UB")

        abi_path.write_text(abi_orig.replace("offsetof(HoloSphereRazorCanary64, ded_count_u32) == 4u", "offsetof(HoloSphereRazorCanary64, ded_count_u32) == 8u"), encoding="utf-8")
        with self.assertRaises(vhr.SiliKernelsReceiptError) as cm_ao:
            vhr.verify_razor_abi_header(abi_path)
        self.assertEqual(cm_ao.exception.code, "ERR_ABI_OFFSETS")

        # AST zero bare asserts check across verify/*.py
        for pyf in sorted(VERIFY_DIR.glob("*.py")):
            tree = ast.parse(pyf.read_text(encoding="utf-8"), filename=str(pyf))
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert)


if __name__ == "__main__":
    unittest.main()
