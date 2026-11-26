#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib assertion-free R5 test suite for verify/check_readme_figures.py.

Verifies live README.md and cpu/avx512/README.md parsing against receipts/,
HTML-comment decoy & duplicate-match refusal, dynamic tampering of every single
README figure across both docs, case-insensitive energy/FP16-equiv/field-name/
transcode policy rules, all 8 receipt file SHA-256 & 4 canonical pins, FIFO/
byte-cap/ancestor-symlink/1e999/4400-digit-int/depth strict I/O defenses, and
zero bare assert statements.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parent
if str(VERIFY_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIR))

import check_readme_figures as crf


def _copy_docs_and_receipts(dst: Path) -> None:
    shutil.copy2(REPO_ROOT / "README.md", dst / "README.md")
    avx_dst = dst / "cpu" / "avx512"
    avx_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "cpu" / "avx512" / "README.md", avx_dst / "README.md")
    lean_dst = dst / "lean"
    lean_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "lean" / "ProximityBarrierSilicon.lean", lean_dst / "ProximityBarrierSilicon.lean")
    rec_dst = dst / "receipts"
    rec_dst.mkdir(parents=True, exist_ok=True)
    for p in (REPO_ROOT / "receipts").glob("*.json"):
        shutil.copy2(p, rec_dst / p.name)


class TestCheckReadmeFiguresR5(unittest.TestCase):
    def test_01_live_repo_pass_cli_modes_and_subprocess(self) -> None:
        res = crf.verify_readme_figures(REPO_ROOT)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["figures_checked"], len(crf.README_FIGURE_SPECS))
        self.assertEqual(res["avx512_figures_checked"], len(crf.AVX512_README_FIGURE_SPECS))
        self.assertEqual(res["cross_receipt_counts_checked"], 8)
        self.assertEqual(
            res["total_figures_checked"],
            len(crf.README_FIGURE_SPECS) + len(crf.AVX512_README_FIGURE_SPECS) + res["cross_receipt_counts_checked"],
        )
        self.assertEqual(res["receipt_pins_verified"], 8)
        self.assertGreaterEqual(res["receipts_loaded"], 12)

        buf_json = io.StringIO()
        with redirect_stdout(buf_json):
            rc_j = crf.main(["--check", "--json", "--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc_j, 0)
        parsed = json.loads(buf_json.getvalue())
        self.assertEqual(parsed["summary_sha256"], res["summary_sha256"])

        buf_txt = io.StringIO()
        with redirect_stdout(buf_txt):
            rc_t = crf.main(["--check", "--repo-root", str(REPO_ROOT)])
        self.assertEqual(rc_t, 0)
        self.assertIn(f"readme_figures={len(crf.README_FIGURE_SPECS)}", buf_txt.getvalue())
        self.assertIn(f"avx512_figures={len(crf.AVX512_README_FIGURE_SPECS)}", buf_txt.getvalue())
        self.assertIn("cross_counts=8", buf_txt.getvalue())
        self.assertIn("receipt_pins=8", buf_txt.getvalue())

        # Broken stdout/stderr (BrokenPipeError / OSError) must return RC=2 cleanly with zero traceback!
        class _BrokenStream:
            def write(self, _s: str) -> int:
                raise BrokenPipeError("broken pipe")

            def flush(self) -> None:
                raise BrokenPipeError("broken pipe")

        with patch.object(crf.sys, "stdout", _BrokenStream()):
            self.assertEqual(crf.main(["--check", "--json", "--repo-root", str(REPO_ROOT)]), 2)
        with patch.object(crf.sys, "stderr", _BrokenStream()):
            with patch.object(crf, "verify_readme_figures", side_effect=OSError("disk error")):
                self.assertEqual(crf.main(["--check", "--repo-root", str(REPO_ROOT)]), 2)
        err_io = io.StringIO()
        with redirect_stderr(err_io):
            with patch.object(crf, "verify_readme_figures", side_effect=RecursionError("deep")):
                self.assertEqual(crf.main(["--check", "--repo-root", str(REPO_ROOT)]), 2)
        self.assertIn("FAIL [ERR_IO] RecursionError", err_io.getvalue())

        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(VERIFY_DIR / "check_readme_figures.py"), "--check", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["summary_sha256"], res["summary_sha256"])

    def test_02_every_readme_and_avx512_figure_tamper_and_decoy_bypass_fail_closed(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="silikernels_readme_fig_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        _copy_docs_and_receipts(td)
        orig_root = (td / "README.md").read_text(encoding="utf-8")
        orig_avx = (td / "cpu" / "avx512" / "README.md").read_text(encoding="utf-8")

        def _tamper_all_specs(doc_path: Path, orig_text: str, specs: tuple[crf.FigureSpec, ...]) -> None:
            for spec_id, pattern, _receipt_rel, _extractor, tol in specs:
                matches = list(re.finditer(pattern, orig_text))
                self.assertEqual(len(matches), 1, f"Spec {spec_id} must match uniquely in {doc_path.name}")
                start, end = matches[0].span(1)
                raw_tok = matches[0].group(1)
                val = float(raw_tok.replace(",", ""))
                if "e-" in raw_tok:
                    tampered_tok = f"{val + max(tol * 20.0, 1e-6):.2e}"
                elif "." in raw_tok:
                    tampered_tok = f"{val + max(tol * 10.0, 1.0):.4f}"
                else:
                    tampered_tok = str(int(val) + 1)
                doc_path.write_text(orig_text[:start] + tampered_tok + orig_text[end:], encoding="utf-8")
                with self.assertRaises(crf.ReadmeFigureError, msg=f"Tampered {spec_id} must fail") as cm:
                    crf.verify_readme_figures(td)
                self.assertEqual(cm.exception.code, "ERR_README_FIGURE", f"{spec_id}: {cm.exception}")
            doc_path.write_text(orig_text, encoding="utf-8")

        _tamper_all_specs(td / "README.md", orig_root, crf.README_FIGURE_SPECS)
        _tamper_all_specs(td / "cpu" / "avx512" / "README.md", orig_avx, crf.AVX512_README_FIGURE_SPECS)

        # Finding 1 (HIGH): HTML comment decoy above corrupted rendered figure MUST fail closed (ERR_README_FIGURE)!
        decoy_corrupt = (
            "<!-- **`33.37 µs/tok`** `12L, d=256` `CellOp` -->\n"
            + orig_root.replace("**`33.37 µs/tok`** `12L, d=256` `CellOp`", "**`999999.99 µs/tok`** `12L, d=256` `CellOp`", 1)
        )
        (td / "README.md").write_text(decoy_corrupt, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_decoy:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_decoy.exception.code, "ERR_README_FIGURE")

        # Visible duplicate figure line MUST fail closed (ERR_README_FIGURE_COUNT, len == 2 != 1)!
        dup_visible = "**`33.37 µs/tok`** `12L, d=256` `CellOp`\n" + orig_root
        (td / "README.md").write_text(dup_visible, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_dupv:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_dupv.exception.code, "ERR_README_FIGURE_COUNT")

        # Unclosed HTML comment MUST fail closed (ERR_README_COMMENT), in linear O(n) time even on 20k markers!
        (td / "README.md").write_text(orig_root + "\n<!-- unclosed comment\n", encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_unc:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_unc.exception.code, "ERR_README_COMMENT")
        with self.assertRaises(crf.ReadmeFigureError) as cm_dos:
            crf._strip_html_comments("<!--" * 20000, "dos.md")
        self.assertEqual(cm_dos.exception.code, "ERR_README_COMMENT")

        # Literal <!-- inside fenced code blocks (``` and ~~~~ with nested ```) and inline code spans MUST PASS!
        fenced_ok = (
            orig_root
            + "\n```text\nliteral <!-- inside backtick fence\n```\n"
            + "~~~~text\n```\nliteral <!-- inside tilde fence\n~~~~\n"
            + "Inline code `<!-- ok` and unmatched ` backtick plus <!-- removed --> text.\n"
        )
        (td / "README.md").write_text(fenced_ok, encoding="utf-8")
        self.assertEqual(crf.verify_readme_figures(td)["status"], "PASS")
        # Also verify backtick-in-info-string does not open a backtick fence (CommonMark rule)
        with self.assertRaises(crf.ReadmeFigureError) as cm_info:
            crf._strip_html_comments("```bad`info\n<!-- unclosed\n```\n", "info.md")
        self.assertEqual(cm_info.exception.code, "ERR_README_COMMENT")
        # Verify nested shorter closing fence inside 4-tilde fence does NOT close early (kills len >= fence_len & char == fence_char BoolOp!)
        self.assertIn("<!-- keep", crf._strip_html_comments("~~~~\n~~~\n<!-- keep\n~~~~\n", "nested_len.md"))
        self.assertIn("<!-- keep", crf._strip_html_comments("~~~~\n````\n<!-- keep\n~~~~\n", "nested_ch.md"))
        self.assertEqual(crf._strip_html_comments("prefix <!-- drop --> suffix", "plain.md"), "prefix  suffix")
        self.assertEqual(crf._strip_html_comments("`a` ``<!--`` <!-- drop --> tail", "inline.md"), "`a` ``<!--``  tail")

        # Cross-receipt aggregate counts (23/23, 50/50, 20/20 Lean + SHA-256) tamper tests!
        (td / "README.md").write_text(orig_root.replace("; all `50/50` HoloSphere", "; all `49/50` HoloSphere", 1), encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_c1:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_c1.exception.code, "ERR_README_FIGURE")
        (td / "README.md").write_text(orig_root.replace("; all `50/50` HoloSphere", "; all `50/49` HoloSphere", 1), encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_c2:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_c2.exception.code, "ERR_README_FIGURE")
        (td / "README.md").write_text(orig_root.replace("; all `50/50` HoloSphere", "; all HoloSphere", 1), encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_c3:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_c3.exception.code, "ERR_README_FIGURE_COUNT")
        (td / "README.md").write_text(orig_root, encoding="utf-8")
        lean_orig = (td / "lean" / "ProximityBarrierSilicon.lean").read_bytes()
        (td / "lean" / "ProximityBarrierSilicon.lean").write_bytes(lean_orig + b"\n-- tampered\n")
        with self.assertRaises(crf.ReadmeFigureError) as cm_ls:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_ls.exception.code, "ERR_LEAN_SHA")
        (td / "lean" / "ProximityBarrierSilicon.lean").write_bytes(lean_orig)

        # Invalid UTF-8 README.md MUST fail closed with ERR_BAD_UTF8 and CLI RC=2 (zero Traceback!)
        (td / "README.md").write_bytes(b"\xff\xfe\xfd")
        with self.assertRaises(crf.ReadmeFigureError) as cm_utf8:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_utf8.exception.code, "ERR_BAD_UTF8")
        proc_utf8 = subprocess.run(
            [sys.executable, "-I", "-B", str(VERIFY_DIR / "check_readme_figures.py"), "--check", "--repo-root", str(td)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_utf8.returncode, 2)
        self.assertIn("ERR_BAD_UTF8", proc_utf8.stderr)
        self.assertNotIn("Traceback", proc_utf8.stderr)

        # Missing figure fails with ERR_README_FIGURE_COUNT and CLI returns rc == 2
        (td / "README.md").write_text("# Silikernels\nEmpty stub.\n", encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_miss:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_miss.exception.code, "ERR_README_FIGURE_COUNT")

        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc_err = crf.main(["--check", "--repo-root", str(td)])
        self.assertEqual(rc_err, 2)
        self.assertIn("ERR_README_FIGURE_COUNT", err_buf.getvalue())

        # Exact tolerance boundary check (abs(found - expected) == tol must PASS -> kills <= to <)
        (td / "README.md").write_text(orig_root, encoding="utf-8")
        s52_path = td / "receipts" / "s52_sm86_ptx_foundry_results.json"
        s52_obj = json.loads(s52_path.read_text(encoding="utf-8"))
        s52_obj["device"]["stream_triad_gb_s"] = 356.685
        s52_path.write_text(json.dumps(s52_obj), encoding="utf-8")
        self.assertEqual(crf.verify_readme_figures(td)["status"], "PASS")
        s52_obj["device"]["stream_triad_gb_s"] = 356.686
        s52_path.write_text(json.dumps(s52_obj), encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_tol:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_tol.exception.code, "ERR_README_FIGURE")
        # 400-digit integer inside <=1024-digit JSON cap must raise ERR_BAD_RECEIPT (never uncaught OverflowError!)
        s52_obj["device"]["stream_triad_gb_s"] = 10**400
        s52_path.write_text(json.dumps(s52_obj), encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_ovf:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_ovf.exception.code, "ERR_BAD_RECEIPT")

    def test_03_policy_rules_pins_strict_io_json_and_zero_bare_asserts(self) -> None:
        td = Path(tempfile.mkdtemp(prefix="silikernels_readme_pol_"))
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        _copy_docs_and_receipts(td)
        orig_readme = (td / "README.md").read_text(encoding="utf-8")

        # Forbidden policy patterns (including Delta-2 HTML entities, Markdown escapes, reversed FP16, FLOP/s, and V*A pairs!)
        for bad_line in (
            "\nMeasured 1.25 mJ/tok on laptop.\n",
            "\nMeasured 140.5 mJ total energy.\n",
            "\nMeasured 67.3 W power draw.\n",
            "\nMeasured 450 mW power draw.\n",
            "\nMeasured 5 watts steady.\n",
            "\nMeasured 5 MJ total.\n",
            "\nMeasured 5 J total.\n",
            "\nMeasured 5 kWh total.\n",
            "\nMeasured 5e3 mJ total.\n",
            "\nMeasured 5 \u03bcJ total.\n",
            "\nMeasured 5<!-- split --> mJ/tok.\n",
            "\nMeasured 5&#109;J total.\n",
            "\nDuplicate: &#50;&#48;&#48; tok/s.\n",
            "\nMeasured 5&mu;J total.\n",
            "\nMeasured 5\\.0 mJ total.\n",
            "\nMeasured 5\u200bmJ total.\n",
            "\nMeasured forty-five watts.\n",
            "\nRecorded energy: 12 units.\n",
            "\nRail draw 2.25 A at 20 V.\n",
            "\nRail draw 20 V @ 2.25 A.\n",
            "\nThroughput 1.29 TB/s fp16.\n",
            "\nThroughput 1.29 TB/s (FP16).\n",
            "\nThroughput 1.29 TB/s FP-16.\n",
            "\nThroughput 500 MB/s FP16.\n",
            "\nThroughput FP16 1.29 TB/s.\n",
            "\nCompute 10 TFLOPS FP16.\n",
            "\nCompute FP16 10 TFLOP/s.\n",
            "\nField speedup_vs_choice_b_532ms quoted.\n",
            "\n| rtx3080_live_llm_transcode_kernels |\n",
        ):
            (td / "README.md").write_text(orig_readme + bad_line, encoding="utf-8")
            with self.assertRaises(crf.ReadmeFigureError, msg=f"Must reject: {bad_line!r}") as cm_pol:
                crf.verify_readme_figures(td)
            self.assertEqual(cm_pol.exception.code, "ERR_README_POLICY")
        # Markdown backslash-escaped duplicate figure line MUST also fail closed via normalized matching!
        (td / "README.md").write_text("**`33\\.37 µs/tok`** `12L, d=256` `CellOp`\n" + orig_readme, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_escdup:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_escdup.exception.code, "ERR_README_FIGURE_COUNT")
        (td / "README.md").write_text(orig_readme, encoding="utf-8")

        # Pin tamper: duplicate pin, missing pin, missing canonical pin, file sha mismatch, canonical sha mismatch
        dup_pin = "\n  - `receipts/s116_phase2b_gpu_results.json`: `3acde86f141e5af59cd766bdb442d2c3e0507a1b9b6cab59b5a343f9e7ad528a`\n"
        (td / "README.md").write_text(orig_readme + dup_pin, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_dup:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_dup.exception.code, "ERR_PIN_DUP")

        missing_pin = orig_readme.replace(
            "  - `receipts/s116_phase2b_gpu_results.json`: `3acde86f141e5af59cd766bdb442d2c3e0507a1b9b6cab59b5a343f9e7ad528a`\n",
            "",
        )
        (td / "README.md").write_text(missing_pin, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_ps:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_ps.exception.code, "ERR_PIN_SET")

        missing_canon = orig_readme.replace(
            " (canonical `receipt_sha256`: `bec7b99bd4023dac4953e806f59c948f596cf69f470e36c9ed37a44270c917d8`)",
            "",
        )
        (td / "README.md").write_text(missing_canon, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_cs:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_cs.exception.code, "ERR_CANON_PIN_SET")

        bad_file_pin = orig_readme.replace(
            "3acde86f141e5af59cd766bdb442d2c3e0507a1b9b6cab59b5a343f9e7ad528a",
            "0" * 64,
        )
        (td / "README.md").write_text(bad_file_pin, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_pm:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_pm.exception.code, "ERR_PIN_MISMATCH")

        bad_canon_pin = orig_readme.replace(
            "bec7b99bd4023dac4953e806f59c948f596cf69f470e36c9ed37a44270c917d8",
            "0" * 64,
        )
        (td / "README.md").write_text(bad_canon_pin, encoding="utf-8")
        with self.assertRaises(crf.ReadmeFigureError) as cm_cm:
            crf.verify_readme_figures(td)
        self.assertEqual(cm_cm.exception.code, "ERR_CANON_PIN_MISMATCH")
        (td / "README.md").write_text(orig_readme, encoding="utf-8")

        # Strict I/O & symlink defenses: symlink root, symlink ancestor above root, symlink subdirs, FIFO (O_NONBLOCK), byte cap (<= vs <)
        sym_root = td.parent / (td.name + "_symroot")
        os.symlink(td, sym_root)
        self.addCleanup(lambda: sym_root.unlink(missing_ok=True))
        with self.assertRaises(crf.ReadmeFigureError) as cm_sr:
            crf.verify_readme_figures(sym_root)
        self.assertEqual(cm_sr.exception.code, "ERR_SYMLINK")

        # Intermediate ancestor symlink above repo_root
        td_anc = Path(tempfile.mkdtemp(prefix="silikernels_anc_"))
        self.addCleanup(lambda: shutil.rmtree(td_anc, ignore_errors=True))
        nested_real = td_anc / "real_parent" / "repo"
        nested_real.mkdir(parents=True)
        _copy_docs_and_receipts(nested_real)
        os.symlink(td_anc / "real_parent", td_anc / "sym_parent")
        with self.assertRaises(crf.ReadmeFigureError) as cm_anc:
            crf.verify_readme_figures(td_anc / "sym_parent" / "repo")
        self.assertEqual(cm_anc.exception.code, "ERR_SYMLINK")

        td_symrec = Path(tempfile.mkdtemp(prefix="silikernels_symrec_"))
        self.addCleanup(lambda: shutil.rmtree(td_symrec, ignore_errors=True))
        shutil.copy2(REPO_ROOT / "README.md", td_symrec / "README.md")
        (td_symrec / "cpu" / "avx512").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "cpu" / "avx512" / "README.md", td_symrec / "cpu" / "avx512" / "README.md")
        os.symlink(td / "receipts", td_symrec / "receipts")
        with self.assertRaises(crf.ReadmeFigureError) as cm_srd:
            crf.verify_readme_figures(td_symrec)
        self.assertEqual(cm_srd.exception.code, "ERR_SYMLINK")

        # Direct read_regular_bytes grandparent symlink, relative path, root "/", non-dir ancestor, missing dir, & os.fstat OSError!
        with self.assertRaises(crf.ReadmeFigureError) as cm_gp_direct:
            crf.read_regular_bytes(td_anc / "sym_parent" / "repo" / "README.md")
        self.assertEqual(cm_gp_direct.exception.code, "ERR_SYMLINK")
        with self.assertRaises(crf.ReadmeFigureError) as cm_root_slash:
            crf.read_regular_bytes(Path("/"))
        self.assertEqual(cm_root_slash.exception.code, "ERR_NOT_REGULAR")
        with self.assertRaises(crf.ReadmeFigureError) as cm_notdir:
            crf.read_regular_bytes(td / "README.md" / "child")
        self.assertEqual(cm_notdir.exception.code, "ERR_NOT_REGULAR")
        with self.assertRaises(crf.ReadmeFigureError) as cm_missdir:
            crf.read_regular_bytes(td / "no_such_dir" / "file.json")
        self.assertEqual(cm_missdir.exception.code, "ERR_MISSING_FILE")
        old_cwd = os.getcwd()
        try:
            os.chdir(td)
            self.assertGreater(len(crf.read_regular_bytes(Path("./README.md"))), 100)
        finally:
            os.chdir(old_cwd)
        with patch.object(crf.os, "lstat", side_effect=OSError("ENOENT")):
            self.assertIs(crf._is_symlink_at("x", -1), False)
        with patch.object(crf.os, "open", side_effect=OSError("root open fail")):
            with self.assertRaises(crf.ReadmeFigureError) as cm_ro:
                crf.read_regular_bytes(td / "README.md")
            self.assertEqual(cm_ro.exception.code, "ERR_MISSING_FILE")

        # FIFO must fail immediately without hanging (O_NONBLOCK + S_ISREG)
        fifo_p = td / "pipe.json"
        os.mkfifo(fifo_p)
        with self.assertRaises(crf.ReadmeFigureError) as cm_fifo:
            crf.read_regular_bytes(fifo_p)
        self.assertEqual(cm_fifo.exception.code, "ERR_NOT_REGULAR")

        with self.assertRaises(crf.ReadmeFigureError) as cm_nr:
            crf.read_regular_bytes(td / "receipts")
        self.assertEqual(cm_nr.exception.code, "ERR_NOT_REGULAR")

        # Exact byte-cap boundary: 16 B passes (kills <= to <), 17 B fails ERR_FILE_TOO_LARGE, os.fstat/os.read OSError -> ERR_READ_ERROR
        cap_f = td / "cap.bin"
        cap_f.write_bytes(b"a" * 16)
        self.assertEqual(len(crf.read_regular_bytes(cap_f, max_bytes=16)), 16)
        with patch.object(crf.os, "fstat", side_effect=OSError("EBADF")):
            with self.assertRaises(crf.ReadmeFigureError) as cm_fst:
                crf.read_regular_bytes(cap_f, max_bytes=16)
            self.assertEqual(cm_fst.exception.code, "ERR_READ_ERROR")
        with patch.object(crf.os, "read", side_effect=OSError("EIO")):
            with self.assertRaises(crf.ReadmeFigureError) as cm_eio:
                crf.read_regular_bytes(cap_f, max_bytes=16)
            self.assertEqual(cm_eio.exception.code, "ERR_READ_ERROR")
        cap_f.write_bytes(b"a" * 17)
        with self.assertRaises(crf.ReadmeFigureError) as cm_cap:
            crf.read_regular_bytes(cap_f, max_bytes=16)
        self.assertEqual(cm_cap.exception.code, "ERR_FILE_TOO_LARGE")

        target_f = td / "valid.json"
        target_f.write_text('{"a": 1}\n', encoding="utf-8")
        sym_f = td / "sym.json"
        os.symlink(target_f, sym_f)
        with self.assertRaises(crf.ReadmeFigureError) as cm_sym:
            crf.load_strict_json(sym_f)
        self.assertEqual(cm_sym.exception.code, "ERR_SYMLINK")

        with self.assertRaises(crf.ReadmeFigureError) as cm_nf:
            crf.load_strict_json(td / "missing.json")
        self.assertEqual(cm_nf.exception.code, "ERR_MISSING_FILE")

        # Strict JSON checks: dup key, NaN, overflow float 1e999, syntax, 1025-digit & 4400-digit int, array, exact 1024-digit passes!
        exact_int_f = td / "exact_int.json"
        exact_int_f.write_text('{"big": -' + ("9" * crf.MAX_JSON_INT_DIGITS) + "}\n", encoding="utf-8")
        self.assertEqual(crf.load_strict_json(exact_int_f)["big"], -(10**crf.MAX_JSON_INT_DIGITS - 1))

        bad_f = td / "bad.json"
        for text, code in (
            ('{"a": 1, "a": 2}', "ERR_JSON_DUP_KEY"),
            ('{"a": NaN}', "ERR_JSON_NONFINITE"),
            ('{"a": 1e999}', "ERR_JSON_NONFINITE"),
            ('{"a": -1e999}', "ERR_JSON_NONFINITE"),
            ("{bad", "ERR_BAD_JSON"),
            ('{"big": ' + ("9" * (crf.MAX_JSON_INT_DIGITS + 1)) + "}", "ERR_BAD_JSON"),
            ('{"big": ' + ("9" * 4400) + "}", "ERR_BAD_JSON"),
            ("[1, 2]", "ERR_JSON_OBJECT"),
        ):
            bad_f.write_text(text, encoding="utf-8")
            with self.assertRaises(crf.ReadmeFigureError) as cm_j:
                crf.load_strict_json(bad_f)
            self.assertEqual(cm_j.exception.code, code)

        # Depth boundary: MAX_JSON_DEPTH passes (kills <= to <), MAX_JSON_DEPTH + 1 fails (both dict and list)
        ok_depth: object = 1
        for _ in range(crf.MAX_JSON_DEPTH):
            ok_depth = [ok_depth]
        crf._check_depth(ok_depth, 0)
        too_deep_list: object = 1
        for _ in range(crf.MAX_JSON_DEPTH + 1):
            too_deep_list = [too_deep_list]
        with self.assertRaises(crf.ReadmeFigureError) as cm_dl:
            crf._check_depth(too_deep_list, 0)
        self.assertEqual(cm_dl.exception.code, "ERR_JSON_DEPTH")

        too_deep_dict: dict[str, object] = {}
        cur = too_deep_dict
        for _ in range(crf.MAX_JSON_DEPTH + 1):
            nxt: dict[str, object] = {}
            cur["k"] = nxt
            cur = nxt
        with self.assertRaises(crf.ReadmeFigureError) as cm_dd:
            crf._check_depth(too_deep_dict, 0)
        self.assertEqual(cm_dd.exception.code, "ERR_JSON_DEPTH")

        # _parse_number rejects both non-finite and malformed tokens with ReadmeFigureError (zero uncaught ValueError!)
        with self.assertRaises(crf.ReadmeFigureError) as cm_pnf:
            crf._parse_number("nan")
        self.assertEqual(cm_pnf.exception.code, "ERR_NONFINITE_VAL")
        with self.assertRaises(crf.ReadmeFigureError) as cm_pbad:
            crf._parse_number("not_a_number")
        self.assertEqual(cm_pbad.exception.code, "ERR_BAD_NUMBER")

        # Zero bare assert statements in check_readme_figures.py and this test file
        for pyf in (VERIFY_DIR / "check_readme_figures.py", Path(__file__).resolve()):
            tree = ast.parse(pyf.read_text(encoding="utf-8"), filename=str(pyf))
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert)


if __name__ == "__main__":
    unittest.main()
