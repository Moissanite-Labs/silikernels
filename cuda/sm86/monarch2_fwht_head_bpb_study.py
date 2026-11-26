#!/usr/bin/env python3
"""Standalone 6-Arm x 3-Seed UTF-8 Byte-LM Training & Exact FP16 Head Evaluation Script.

Reproduces the controlled iso-budget UTF-8 bits-per-byte (BPB) study comparing:
  1. arm2b_bias_514B              : 2 B logit scalar + 512 B FP16 byte-unigram bias (514 B, 0 B HBM table)
  2. arm5_bias_2p5KiB             : 2 KiB Walsh-Hadamard conv + 512 B FP16 bias (2,560 B, 0 B HBM table)
  3. arm7_bias_8p5KiB             : 8 KiB 1-stage Monarch-16x16 + 512 B FP16 bias (8,704 B, 0 B HBM table)
  4. arm9_monarch2_bias_16p5KiB   : 16 KiB 2-stage Monarch-16x16 + 512 B FP16 bias (16,896 B, 0 B HBM table)
  5. arm8a_lowrank16_bias_16p9KiB : 16 KiB rank-16 low-rank + 512 B FP16 bias (16,896 B HBM)
  6. arm4_dense_bias_262KiB       : 262 KiB dense 256x512 linear + 512 B FP16 bias ceiling (262,658 B HBM)

Mathematical Scale Alignment:
  - Plain 256-point Hadamard (`h256_mat`) uses orthonormal scale `1/16 = 0.0625` (`||H u||_2^2 = ||u||_2^2`).
  - Bent-4 256-point FWHT head (`fwht_mat`) uses scale `1/sqrt(32) = 0.17677669529663688`
    (`sqrt(8) * orthonormal`, so `||y||_2^2 = 8 * ||u||_2^2`), matching both
    `cuda/sm86/holosphere_monarch2_fwht_head_sm86.cu` and
    `cuda/sm86/holosphere_monarch2_fwht_head_bench.cu` bit-for-bit.

Telemetry Policy (`UNCONTROLLED_LAPTOP_NOT_CLAIMED`):
  - Zero device power or energy telemetry is read, logged, or emitted by this script.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
CORPUS_ID = "holosphere-v3-flagship-corpus"
EXPECTED_CORPUS_MANIFEST_SHA256 = "a05e415f7d33ef32376ff3d9da1eede0e3f561b64094b5b91147b2696a99e9d4"
EXPECTED_TRAIN_BIN_SHA256 = "528c6c792d2a563d82d21ce213b26209441b39fa5e76b8595607bab2ca853b7b"
EXPECTED_EVAL_BIN_SHA256 = "31fd132d888a21ecebe9e907b6c4b82e3e68e42b5a29e2839584447dcca631a0"
UNBIASED_SCALAR_REF_BPB_Q = "399633111796751/74411611509180"


class Monarch2BpbStudyError(RuntimeError):
    """Fail-closed typed error for the Monarch-2 UTF-8 BPB study runner."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise Monarch2BpbStudyError(code, message)


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


def to_q_str(val: float, max_den: int = 1_000_000) -> str:
    require_ok(math.isfinite(val), "NON_FINITE", f"val={val!r}")
    f = Fraction(val).limit_denominator(max_den)
    return f"{f.numerator}/{f.denominator}"


def parse_q(raw: str) -> Fraction:
    num_s, den_s = raw.split("/", 1)
    return Fraction(int(num_s), int(den_s))


def format_q(val: Fraction) -> str:
    return f"{val.numerator}/{val.denominator}"


def run_bpb_study(
    corpus_dir: Path,
    *,
    steps: int = 300,
    batch_size: int = 8,
    seq_len: int = 512,
    eval_chunks: int = 64,
    seeds: Optional[List[int]] = None,
    verify_pinned_corpus: bool = True,
) -> Dict[str, Any]:
    """Run the 6-arm x 3-seed iso-budget UTF-8 byte-LM study with exact FP16 head eval."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    seed_list = [42, 137, 2026] if seeds is None else list(seeds)
    require_ok(torch.cuda.is_available(), "NO_CUDA", "CUDA device required for GPU training")
    device = torch.device("cuda:0")
    d_model, n_heads, d_ff, n_layers = 512, 16, 800, 4

    man_path = corpus_dir.joinpath("flagship_corpus_manifest.json")
    tr_path = corpus_dir.joinpath("train.bin")
    ev_path = corpus_dir.joinpath("eval.bin")
    require_ok(man_path.is_file() and tr_path.is_file() and ev_path.is_file(), "CORPUS_MISSING", corpus_dir.name)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man_sha = sha256_file(man_path)
    tr_raw = tr_path.read_bytes()
    ev_raw = ev_path.read_bytes()
    tr_sha = hashlib.sha256(tr_raw).hexdigest()
    ev_sha = hashlib.sha256(ev_raw).hexdigest()
    require_ok(tr_sha == man["train"]["sha256"], "TRAIN_SHA_MISMATCH", tr_sha)
    require_ok(ev_sha == man["eval"]["sha256"], "EVAL_SHA_MISMATCH", ev_sha)
    require_ok(man["strict_sha256_doc_overlap_count"] == 0, "CORPUS_OVERLAP", "overlap != 0")
    if verify_pinned_corpus:
        require_ok(man_sha == EXPECTED_CORPUS_MANIFEST_SHA256, "CORPUS_MANIFEST_PIN_MISMATCH", man_sha)
        require_ok(tr_sha == EXPECTED_TRAIN_BIN_SHA256, "TRAIN_PIN_MISMATCH", tr_sha)
        require_ok(ev_sha == EXPECTED_EVAL_BIN_SHA256, "EVAL_PIN_MISMATCH", ev_sha)

    # Orthonormal 256-point Hadamard (scale 1/16 = 0.0625)
    h256_fp64 = torch.tensor(
        [[(-0.0625 if ((k & j).bit_count() & 1) else 0.0625) for j in range(256)] for k in range(256)],
        dtype=torch.float64,
    )
    h256_cpu = h256_fp64.float()
    h256_ortho_err = float((h256_fp64 @ h256_fp64 - torch.eye(256, dtype=torch.float64)).abs().max().item())
    require_ok(h256_ortho_err < 1e-14, "H256_ORTHO_FAIL", f"err={h256_ortho_err}")

    # Bent-4 signed-Hadamard head transform scaled by 1/sqrt(32) = sqrt(8) * orthonormal
    inv_sqrt32 = 1.0 / math.sqrt(32.0)
    m_rows = []
    for tok in range(256):
        row = []
        for d in range(256):
            lane = d & 31
            db = d ^ ((lane & 7) << 5)
            b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
            par = ((tok & db).bit_count() ^ b4) & 1
            row.append((-1.0 if par else 1.0) * inv_sqrt32)
        m_rows.append(row)
    fwht_bent_cpu = torch.tensor(m_rows, dtype=torch.float64).float()

    inj_rows = []
    for tok in range(257):
        b_u8 = tok & 0xFF
        sp_tag = (tok >> 8) & 3
        row512 = []
        for d in range(512):
            bank = d >> 8
            d256 = d & 255
            lane = d256 & 31
            db = d256 ^ ((lane & 7) << 5)
            b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
            sp_flip = sp_tag & ((d256 >> 5) & 1)
            par = ((b_u8 & db).bit_count() ^ b4 ^ sp_flip) & 1
            sgn = -1.0 if bank == 1 else 1.0
            row512.append((-1.0 if par else 1.0) * inv_sqrt32 * sgn)
        inj_rows.append(row512)
    inj_mat_cpu = torch.tensor(inj_rows, dtype=torch.float32)

    def _unit_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return t * torch.rsqrt((t * t).sum(dim=-1, keepdim=True) + eps * eps)

    class CausalDepthwiseByteConv4(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(dim, 1, 4))
            with torch.no_grad():
                self.weight[:, 0, 3] = 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xt = x.transpose(1, 2)
            out = F.conv1d(F.pad(xt, (3, 0)), self.weight, groups=xt.shape[1]).transpose(1, 2)
            return _unit_norm(out)

    class HoloSphereBlock(nn.Module):
        def __init__(self, dim: int, heads: int, ff: int) -> None:
            super().__init__()
            self.heads = heads
            self.hdim = dim // heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
            self.alpha_attn = nn.Parameter(torch.full((dim,), 0.1))
            self.w_gate = nn.Linear(dim, ff, bias=False)
            self.w_up = nn.Linear(dim, ff, bias=False)
            self.w_down = nn.Linear(ff, dim, bias=False)
            self.alpha_ff = nn.Parameter(torch.full((dim,), 0.1))

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            bsz, seq, dim = x.shape
            qkv = self.qkv(x).view(bsz, seq, 3, self.heads, self.hdim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            q1, q2 = q[..., 0::2], q[..., 1::2]
            k1, k2 = k[..., 0::2], k[..., 1::2]
            q = _unit_norm(torch.stack((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1).flatten(-2))
            k = _unit_norm(torch.stack((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1).flatten(-2))
            y = _unit_norm(
                self.out_proj(
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    .transpose(1, 2)
                    .contiguous()
                    .view(bsz, seq, dim)
                )
            )
            x = _unit_norm(x + self.alpha_attn.abs().clamp(max=0.95) * (y - x))
            ff = _unit_norm(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
            return _unit_norm(x + self.alpha_ff.abs().clamp(max=0.95) * (ff - x))

    class HoloSphereMonarch2HeadLM(nn.Module):
        def __init__(self, head_kind: str) -> None:
            super().__init__()
            self.head_kind = head_kind
            self.register_buffer("inj_table", inj_mat_cpu.clone(), persistent=False)
            self.register_buffer("h256_mat", h256_cpu.clone(), persistent=False)
            self.register_buffer("fwht_mat", fwht_bent_cpu.clone(), persistent=False)
            self.byte_conv = CausalDepthwiseByteConv4(d_model)
            self.blocks = nn.ModuleList([HoloSphereBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

            hdim = d_model // n_heads
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hdim, 2, dtype=torch.float32) / float(hdim)))
            freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
            self.register_buffer("rot_cos", freqs.cos().view(1, 1, seq_len, hdim // 2), persistent=False)
            self.register_buffer("rot_sin", freqs.sin().view(1, 1, seq_len, hdim // 2), persistent=False)

            # Shared +512 B FP16 byte-unigram bias across all 6 arms
            self.byte_bias = nn.Parameter(torch.zeros(256))

            if head_kind == "arm2b_bias_514B":
                self.logit_scale = nn.Parameter(torch.tensor(1.0))
            elif head_kind == "arm5_bias_2p5KiB":
                self.fold_alpha = nn.Parameter(torch.ones(256))
                self.fold_beta = nn.Parameter(-torch.ones(256))
                self.d1 = nn.Parameter(torch.ones(256))
                self.d2 = nn.Parameter(torch.ones(256))
            elif head_kind == "arm7_bias_8p5KiB":
                self.monarch_b1 = nn.Parameter(torch.eye(16).unsqueeze(0).repeat(16, 1, 1).clone())
            elif head_kind == "arm9_monarch2_bias_16p5KiB":
                self.monarch_b1 = nn.Parameter(torch.eye(16).unsqueeze(0).repeat(16, 1, 1).clone())
                self.monarch_b2 = nn.Parameter(torch.eye(16).unsqueeze(0).repeat(16, 1, 1).clone())
            elif head_kind == "arm8a_lowrank16_bias_16p9KiB":
                self.lr_down = nn.Linear(256, 16, bias=False)
                self.lr_up = nn.Linear(16, 256, bias=False)
                with torch.no_grad():
                    self.lr_up.weight.mul_(8.0)
            elif head_kind == "arm4_dense_bias_262KiB":
                self.lm_head = nn.Linear(d_model, 256, bias=False)
                self.logit_scale = nn.Parameter(torch.tensor(8.0))
            else:
                raise Monarch2BpbStudyError("BAD_HEAD_KIND", head_kind)

        def head_spec(self) -> Tuple[int, int, int]:
            specs = {
                "arm2b_bias_514B": (257, 514, 0),
                "arm5_bias_2p5KiB": (1280, 2560, 0),
                "arm7_bias_8p5KiB": (4352, 8704, 0),
                "arm9_monarch2_bias_16p5KiB": (8448, 16896, 0),
                "arm8a_lowrank16_bias_16p9KiB": (8448, 16896, 16896),
                "arm4_dense_bias_262KiB": (131329, 262658, 262658),
            }
            return specs[self.head_kind]

        def forward(self, tok_ids: torch.Tensor, *, fp16_head_eval: bool = False) -> torch.Tensor:
            x = self.byte_conv(F.embedding(tok_ids, self.inj_table))
            seq = tok_ids.shape[1]
            cos = self.rot_cos[:, :, :seq, :].to(x.dtype)
            sin = self.rot_sin[:, :, :seq, :].to(x.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = _unit_norm(x)

            def _p(t: torch.Tensor) -> torch.Tensor:
                return t.half().to(x.dtype) if fp16_head_eval else t.to(x.dtype)

            bias = _p(self.byte_bias)
            if self.head_kind == "arm2b_bias_514B":
                u = x[..., :256] - x[..., 256:]
                return F.linear(u, self.fwht_mat.to(u.dtype)) * _p(self.logit_scale) + bias

            if self.head_kind == "arm5_bias_2p5KiB":
                u = x[..., :256] * _p(self.fold_alpha) + x[..., 256:] * _p(self.fold_beta)
                hmat = self.h256_mat.to(u.dtype)
                z1 = F.linear(u, hmat) * _p(self.d1)
                z2 = F.linear(z1, hmat) * _p(self.d2)
                return F.linear(z2, self.fwht_mat.to(u.dtype)) + bias

            if self.head_kind == "arm7_bias_8p5KiB":
                bsz, sq, _ = x.shape
                u = x[..., :256] - x[..., 256:]
                z1 = F.linear(u, self.h256_mat.to(u.dtype))
                z2 = z1.view(bsz, sq, 16, 16).transpose(-1, -2).contiguous()
                z3 = torch.einsum("grc,bsgc->bsgr", _p(self.monarch_b1), z2).reshape(bsz, sq, 256)
                return F.linear(z3, self.fwht_mat.to(z3.dtype)) + bias

            if self.head_kind == "arm9_monarch2_bias_16p5KiB":
                bsz, sq, _ = x.shape
                u = x[..., :256] - x[..., 256:]
                z1 = F.linear(u, self.h256_mat.to(u.dtype))
                z2 = z1.view(bsz, sq, 16, 16).transpose(-1, -2).contiguous()
                z3 = torch.einsum("grc,bsgc->bsgr", _p(self.monarch_b1), z2)
                z4 = z3.transpose(-1, -2).contiguous()
                z5 = torch.einsum("grc,bsgc->bsgr", _p(self.monarch_b2), z4).reshape(bsz, sq, 256)
                return F.linear(z5, self.fwht_mat.to(z5.dtype)) + bias

            if self.head_kind == "arm8a_lowrank16_bias_16p9KiB":
                u = x[..., :256] - x[..., 256:]
                return F.linear(F.linear(u, _p(self.lr_down.weight)), _p(self.lr_up.weight)) + bias

            return self.lm_head(x) * _p(self.logit_scale) + bias

    ev_batch_all = torch.stack(
        [
            torch.tensor([256] + list(ev_raw[ci * seq_len : (ci + 1) * seq_len]), dtype=torch.long)
            for ci in range(eval_chunks)
        ],
        dim=0,
    )

    def _eval_bpb(mod: HoloSphereMonarch2HeadLM, *, fp16_head_eval: bool) -> Tuple[float, int]:
        mod.eval()
        tot_nll, tot_toks = 0.0, 0
        with torch.no_grad():
            for i in range(0, ev_batch_all.shape[0], batch_size):
                b = ev_batch_all[i : i + batch_size].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = mod(b[:, :-1], fp16_head_eval=fp16_head_eval)
                tot_nll += float(
                    F.cross_entropy(lg.float().reshape(-1, 256), b[:, 1:].reshape(-1), reduction="sum").item()
                )
                tot_toks += int(b[:, 1:].numel())
        return ((tot_nll / float(tot_toks)) / math.log(2.0), tot_toks)

    arm_names = [
        "arm2b_bias_514B",
        "arm5_bias_2p5KiB",
        "arm7_bias_8p5KiB",
        "arm9_monarch2_bias_16p5KiB",
        "arm8a_lowrank16_bias_16p9KiB",
        "arm4_dense_bias_262KiB",
    ]
    max_start = len(tr_raw) - seq_len - 1
    backbone_init_by_seed: Dict[int, Dict[str, torch.Tensor]] = {}
    train_starts_by_seed: Dict[int, List[List[int]]] = {}
    for sd in seed_list:
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        ref_m = HoloSphereMonarch2HeadLM("arm2b_bias_514B")
        backbone_init_by_seed[sd] = {
            k: v.detach().cpu().clone()
            for k, v in ref_m.state_dict().items()
            if k.startswith("byte_conv.") or k.startswith("blocks.")
        }
        g_dat = torch.Generator(device="cpu").manual_seed(sd * 7919 + 17)
        train_starts_by_seed[sd] = torch.randint(0, max_start, (steps, batch_size), generator=g_dat).tolist()

    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm in arm_names:
        per_seed: List[Dict[str, Any]] = []
        head_params, head_bytes, hbm_bytes = (0, 0, 0)
        total_params = 0
        for sd in seed_list:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            model = HoloSphereMonarch2HeadLM(arm).to(device)
            cur_sd = model.state_dict()
            for k, v in backbone_init_by_seed[sd].items():
                cur_sd[k].copy_(v)
            model.load_state_dict(cur_sd, strict=True)
            head_params, head_bytes, hbm_bytes = model.head_spec()
            total_params = int(sum(p.numel() for p in model.parameters()))
            require_ok(total_params == 9_115_648 + head_params, "PARAM_COUNT_MISMATCH", f"{arm}: {total_params}")

            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.01)
            t0 = time.perf_counter()
            model.train()
            starts_sd = train_starts_by_seed[sd]
            for st in range(steps):
                w_steps = max(1, steps // 10)
                if st < w_steps:
                    lr_now = 1.5e-3 * float(st + 1) / float(w_steps)
                else:
                    prog = float(st - w_steps) / float(max(1, steps - w_steps))
                    lr_now = 1.5e-3 * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
                for pg in opt.param_groups:
                    pg["lr"] = lr_now

                rows = [[256] + list(tr_raw[p : p + seq_len]) for p in starts_sd[st]]
                batch = torch.tensor(rows, dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(batch[:, :-1], fp16_head_eval=False)
                    loss = F.cross_entropy(lg.float().reshape(-1, 256), batch[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - t0
            bpb_bf16, n_eval_toks = _eval_bpb(model, fp16_head_eval=False)
            bpb_fp16, _ = _eval_bpb(model, fp16_head_eval=True)
            q_fp16 = parse_q(to_q_str(bpb_fp16, 100_000))
            q_bf16 = parse_q(to_q_str(bpb_bf16, 100_000))
            per_seed.append(
                {
                    "seed": int(sd),
                    "eval_bpb_q": format_q(q_fp16),
                    "eval_bpb_bf16_q": format_q(q_bf16),
                    "fp16_quant_bpb_diff_q": format_q(abs(q_fp16 - q_bf16)),
                    "eval_tokens": int(n_eval_toks),
                    "train_wall_s_q": to_q_str(wall_s, 1000),
                }
            )
            del model, opt
            torch.cuda.empty_cache()

        mean_bpb_q = sum((parse_q(r["eval_bpb_q"]) for r in per_seed), Fraction(0, 1)) / Fraction(len(per_seed), 1)
        arm_results[arm] = {
            "head_params": head_params,
            "head_fp16_bytes": head_bytes,
            "hbm_vocab_table_bytes": hbm_bytes,
            "total_params": total_params,
            "mean_eval_bpb_q": format_q(mean_bpb_q),
            "seeds": per_seed,
        }

    return {
        "energy_power_status": ENERGY_POLICY_STATUS,
        "corpus_id": CORPUS_ID,
        "corpus_sha256": man_sha,
        "corpus_train_bin_sha256": tr_sha,
        "corpus_eval_bin_sha256": ev_sha,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps_per_seed": steps,
        "tokens_per_seed": steps * batch_size * seq_len,
        "eval_chunks": eval_chunks,
        "eval_bytes": eval_chunks * seq_len,
        "seeds": seed_list,
        "h256_ortho_fp64_max_err_q": to_q_str(h256_ortho_err, 10**12),
        "arms": arm_results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="6-Arm x 3-Seed UTF-8 Byte-LM Study Runner for Monarch-2 + Byte-Bias FWHT Head"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing flagship_corpus_manifest.json, train.bin, and eval.bin",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--eval-chunks", type=int, default=64)
    parser.add_argument("--skip-corpus-pin", action="store_true", help="Skip pinned corpus SHA-256 check")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    try:
        study = run_bpb_study(
            args.corpus_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_chunks=args.eval_chunks,
            seeds=[42, 137, 2026],
            verify_pinned_corpus=not args.skip_corpus_pin,
        )
    except Monarch2BpbStudyError as exc:
        sys.stderr.write(json.dumps({"status": "ERROR", "code": exc.code, "message": exc.message}, sort_keys=True) + "\n")
        return 2

    payload = json.dumps(study, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
