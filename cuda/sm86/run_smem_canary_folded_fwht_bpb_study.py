#!/usr/bin/env python3
"""Controlled 5-Arm x 3-Seed UTF-8 Byte-LM BPB Study for 1 KiB FP16 Folded FWHT_256 x bent4 Head.

Standalone PyTorch CUDA training and evaluation script bound by SHA-256 into
`receipts/smem_canary_folded_fwht_results.json`.

Evaluates five iso-budget byte-level LM head variants on a 4-layer d_model=512,
d_ff=800, n_heads=16 HoloSphere backbone across seeds {42, 137, 2026} with
identical per-seed trunk initialization and identical training-window draws:
  - arm1_prefix_fwht: 0 B head params (u = h[:256]; drops h[256:512])
  - arm2_fixed_diff_fwht: 0 B head params (u = h[:256] - h[256:])
  - arm2b_fixed_diff_1scalar_fwht: 2 B (1 FP16 scalar logit temperature steelman)
  - arm3_learned_1kib_fp16_fold_fwht: 1,024 B (512 FP16 params; u = alpha*h[:256] + beta*h[256:])
  - arm4_dense_256x512: 262,146 B dense linear reference ceiling (131,073 params)

The tied FWHT_256 x bent4 projection scales the 256-point Hadamard transform by
1/sqrt(32) (= sqrt(8) * orthonormal; Euclidean gain 2.828427...), matching
`cuda/sm86/holosphere_smem_canary_folded_fwht_sm86.cu` bit-for-bit.

Telemetry policy: `energy_power_status = UNCONTROLLED_LAPTOP_NOT_CLAIMED`.
Zero hardware power or energy telemetry is queried or emitted by this script.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ENERGY_POLICY_STATUS = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"
EXPECTED_CORPUS_MANIFEST_SHA256 = "280ff5af8bdac00b5db5d9b0dea7e57c05e4c854b576387ddf18c5d1042b7691"
EXPECTED_TRAIN_BIN_SHA256 = "528c6c792d2a563d82d21ce213b26209441b39fa5e76b8595607bab2ca853b7b"
EXPECTED_EVAL_BIN_SHA256 = "31fd132d888a21ecebe9e907b6c4b82e3e68e42b5a29e2839584447dcca631a0"


class BpbStudyError(RuntimeError):
    """Fail-closed typed error for the 5-arm UTF-8 BPB study runner."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise BpbStudyError(code, message)


def format_q(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


def to_q(val: float, max_den: int = 1_000_000) -> Fraction:
    require_ok(math.isfinite(val), "NON_FINITE_FLOAT", f"value={val!r}")
    return Fraction.from_float(float(val)).limit_denominator(max_den)


def to_q_str(val: float, max_den: int = 1_000_000) -> str:
    return format_q(to_q(val, max_den))


def verify_and_load_corpus(
    corpus_dir: Path,
    *,
    enforce_pinned_digests: bool = True,
) -> Tuple[Dict[str, Any], bytes, bytes, str, str]:
    """Load train.bin, eval.bin, and flagship_corpus_manifest.json with strict SHA-256 checks."""
    man_path = corpus_dir / "flagship_corpus_manifest.json"
    tr_path = corpus_dir / "train.bin"
    ev_path = corpus_dir / "eval.bin"
    require_ok(man_path.is_file() and tr_path.is_file() and ev_path.is_file(), "CORPUS_MISSING", corpus_dir.name)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    tr_raw = tr_path.read_bytes()
    ev_raw = ev_path.read_bytes()
    tr_sha = hashlib.sha256(tr_raw).hexdigest()
    ev_sha = hashlib.sha256(ev_raw).hexdigest()
    require_ok(tr_sha == man["train"]["sha256"], "TRAIN_SHA_MISMATCH", tr_sha)
    require_ok(ev_sha == man["eval"]["sha256"], "EVAL_SHA_MISMATCH", ev_sha)
    require_ok(man["strict_sha256_doc_overlap_count"] == 0, "CORPUS_OVERLAP", "overlap != 0")
    if enforce_pinned_digests:
        require_ok(
            str(man["manifest_sha256"]) == EXPECTED_CORPUS_MANIFEST_SHA256,
            "CORPUS_MANIFEST_PIN_MISMATCH",
            str(man["manifest_sha256"]),
        )
        require_ok(tr_sha == EXPECTED_TRAIN_BIN_SHA256, "TRAIN_PIN_MISMATCH", tr_sha)
        require_ok(ev_sha == EXPECTED_EVAL_BIN_SHA256, "EVAL_PIN_MISMATCH", ev_sha)
    return (man, tr_raw, ev_raw, tr_sha, ev_sha)


def run_pytorch_bpb_and_parity_study(
    *,
    corpus_dir: Path,
    seeds: Sequence[int] = (42, 137, 2026),
    steps: int = 300,
    batch_size: int = 8,
    seq_len: int = 512,
    eval_chunks: int = 64,
    d_model: int = 512,
    n_heads: int = 16,
    n_layers: int = 4,
    d_ff: int = 800,
    enforce_pinned_digests: bool = True,
) -> Dict[str, Any]:
    """Run the 5-arm x 3-seed UTF-8 BPB training & eval study and return canonical rational JSON."""
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    require_ok(torch.cuda.is_available(), "CUDA_UNAVAILABLE", "CUDA GPU required to run training study")
    device = torch.device("cuda:0")

    man, tr_raw, ev_raw, tr_sha, ev_sha = verify_and_load_corpus(
        corpus_dir, enforce_pinned_digests=enforce_pinned_digests
    )

    # Exact parameter-free 1/sqrt(32)-scaled (sqrt(8) * orthonormal) H_256 x bent4 matrix M[tok, d]
    # Matching cuda/sm86/holosphere_smem_canary_folded_fwht_sm86.cu bit-for-bit.
    inv_sqrt32 = 1.0 / math.sqrt(32.0)
    m_rows: List[List[float]] = []
    for tok in range(256):
        row: List[float] = []
        for d in range(256):
            lane = d & 31
            db = d ^ ((lane & 7) << 5)
            b4 = ((lane & 3) & ((lane >> 2) & 3)).bit_count() & 1
            par = ((tok & db).bit_count() ^ b4) & 1
            row.append((-1.0 if par else 1.0) * inv_sqrt32)
        m_rows.append(row)
    fwht_mat_fp64 = torch.tensor(m_rows, dtype=torch.float64)
    fwht_mat_cpu = fwht_mat_fp64.float()

    # Exact H_256 x bent4 byte injection table (257 x 512: bytes 0..255 + BOS=256)
    inj_rows: List[List[float]] = []
    for tok in range(257):
        b_u8 = tok & 0xFF
        sp_tag = (tok >> 8) & 3
        row512: List[float] = []
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

    # Verify exact butterfly vs dense FP64 parity and upper-half gradient routing
    gen_chk = torch.Generator(device="cpu").manual_seed(2026)
    h_chk = torch.randn((4, 512), generator=gen_chk, dtype=torch.float64)
    h_chk = h_chk / (h_chk.norm(dim=-1, keepdim=True) + 1e-12)
    a_chk = torch.empty(256, dtype=torch.float32).uniform_(0.5, 1.5, generator=gen_chk).half()
    b_chk = torch.empty(256, dtype=torch.float32).uniform_(-1.5, -0.5, generator=gen_chk).half()

    def _fwht256_butterfly_fp64(u256: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(256, dtype=torch.long)
        lane = idx & 31
        w = idx >> 5
        src_w = w ^ (lane & 7)
        src_idx = (src_w << 5) | lane
        b4_raw = (lane & 3) & ((lane >> 2) & 3)
        b4 = (b4_raw ^ (b4_raw >> 1)) & 1
        b4_sgn = torch.where(b4 == 1, -1.0, 1.0).to(torch.float64)
        v = u256[:, src_idx] * b4_sgn
        v = v.view(-1, 256)
        step = 1
        while step < 256:
            v = v.view(-1, 256 // (2 * step), 2, step)
            a_p = v[:, :, 0, :]
            b_p = v[:, :, 1, :]
            v = torch.stack((a_p + b_p, a_p - b_p), dim=2).reshape(-1, 256)
            step <<= 1
        return v * (1.0 / math.sqrt(32.0))

    u_ref = h_chk[:, :256] * a_chk.double() + h_chk[:, 256:] * b_chk.double()
    lg_mat = u_ref @ fwht_mat_fp64.T
    lg_but = _fwht256_butterfly_fp64(u_ref)
    butterfly_vs_mat_err = float((lg_mat - lg_but).abs().max().item())
    require_ok(butterfly_vs_mat_err < 1e-12, "BUTTERFLY_PARITY_FAIL", f"err={butterfly_vs_mat_err}")

    h_p1 = h_chk.float().clone().detach().requires_grad_(True)
    loss_p1 = (h_p1[:, :256] @ fwht_mat_cpu.T).pow(2).sum()
    loss_p1.backward()
    arm1_upper_half_grad_norm = float(h_p1.grad[:, 256:].norm().item())

    h_p3 = h_chk.float().clone().detach().requires_grad_(True)
    u_p3 = h_p3[:, :256] * a_chk.float() + h_p3[:, 256:] * b_chk.float()
    loss_p3 = (u_p3 @ fwht_mat_cpu.T).pow(2).sum()
    loss_p3.backward()
    arm3_upper_half_grad_norm = float(h_p3.grad[:, 256:].norm().item())
    require_ok(arm1_upper_half_grad_norm == 0.0 and arm3_upper_half_grad_norm > 0.1, "GRAD_FLOW_CHECK", "bad grad flow")

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
            xp = F.pad(xt, (3, 0))
            out = F.conv1d(xp, self.weight, groups=xt.shape[1]).transpose(1, 2)
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
            q = torch.stack((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1).flatten(-2)
            k = torch.stack((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1).flatten(-2)
            q = _unit_norm(q)
            k = _unit_norm(k)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = self.out_proj(y.transpose(1, 2).contiguous().view(bsz, seq, dim))
            y = _unit_norm(y)
            x = _unit_norm(x + self.alpha_attn.abs().clamp(max=0.95) * (y - x))
            ff = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
            ff = _unit_norm(ff)
            x = _unit_norm(x + self.alpha_ff.abs().clamp(max=0.95) * (ff - x))
            return x

    class HoloSphereByteLM(nn.Module):
        def __init__(self, head_kind: str) -> None:
            super().__init__()
            self.head_kind = head_kind
            self.register_buffer("inj_table", inj_mat_cpu.clone(), persistent=False)
            self.register_buffer("fwht_mat", fwht_mat_cpu.clone(), persistent=False)
            self.byte_conv = CausalDepthwiseByteConv4(d_model)
            self.blocks = nn.ModuleList([HoloSphereBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

            hdim = d_model // n_heads
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hdim, 2, dtype=torch.float32) / float(hdim)))
            pos = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            self.register_buffer("rot_cos", freqs.cos().view(1, 1, seq_len, hdim // 2), persistent=False)
            self.register_buffer("rot_sin", freqs.sin().view(1, 1, seq_len, hdim // 2), persistent=False)

            if head_kind in ("arm1_prefix_fwht", "arm2_fixed_diff_fwht"):
                pass
            elif head_kind == "arm2b_fixed_diff_1scalar_fwht":
                self.logit_scale = nn.Parameter(torch.tensor(1.0))
            elif head_kind == "arm3_learned_1kib_fp16_fold_fwht":
                self.fold_alpha = nn.Parameter(torch.ones(256))
                self.fold_beta = nn.Parameter(-torch.ones(256))
            elif head_kind == "arm4_dense_256x512":
                self.lm_head = nn.Linear(d_model, 256, bias=False)
                self.logit_scale = nn.Parameter(torch.tensor(8.0))
            else:
                raise BpbStudyError("BAD_HEAD_KIND", head_kind)

        def head_param_count_and_bytes(self) -> Tuple[int, int]:
            if self.head_kind in ("arm1_prefix_fwht", "arm2_fixed_diff_fwht"):
                return (0, 0)
            if self.head_kind == "arm2b_fixed_diff_1scalar_fwht":
                return (1, 2)
            if self.head_kind == "arm3_learned_1kib_fp16_fold_fwht":
                return (512, 1024)
            return (256 * d_model + 1, (256 * d_model + 1) * 2)

        def forward(self, tok_ids: torch.Tensor, *, fp16_fold_eval: bool = False) -> torch.Tensor:
            x = F.embedding(tok_ids, self.inj_table)
            x = self.byte_conv(x)
            seq = tok_ids.shape[1]
            cos = self.rot_cos[:, :, :seq, :].to(x.dtype)
            sin = self.rot_sin[:, :, :seq, :].to(x.dtype)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = _unit_norm(x)
            if self.head_kind == "arm1_prefix_fwht":
                u = x[..., :256]
                return F.linear(u, self.fwht_mat.to(u.dtype))
            if self.head_kind == "arm2_fixed_diff_fwht":
                u = x[..., :256] - x[..., 256:]
                return F.linear(u, self.fwht_mat.to(u.dtype))
            if self.head_kind == "arm2b_fixed_diff_1scalar_fwht":
                u = x[..., :256] - x[..., 256:]
                return F.linear(u, self.fwht_mat.to(u.dtype)) * self.logit_scale.to(u.dtype)
            if self.head_kind == "arm3_learned_1kib_fp16_fold_fwht":
                a = self.fold_alpha.half().to(x.dtype) if fp16_fold_eval else self.fold_alpha.to(x.dtype)
                b = self.fold_beta.half().to(x.dtype) if fp16_fold_eval else self.fold_beta.to(x.dtype)
                u = x[..., :256] * a + x[..., 256:] * b
                return F.linear(u, self.fwht_mat.to(u.dtype))
            return self.lm_head(x) * self.logit_scale.to(x.dtype)

    ev_tensors: List[torch.Tensor] = []
    for ci in range(eval_chunks):
        sl = ev_raw[ci * seq_len : (ci + 1) * seq_len]
        ev_tensors.append(torch.tensor([256] + list(sl), dtype=torch.long))
    ev_batch_all = torch.stack(ev_tensors, dim=0)

    def _eval_bpb(mod: HoloSphereByteLM, *, fp16_fold_eval: bool = False) -> Tuple[float, float, int]:
        mod.eval()
        tot_nll = 0.0
        tot_toks = 0
        with torch.no_grad():
            for i in range(0, ev_batch_all.shape[0], batch_size):
                b = ev_batch_all[i : i + batch_size].to(device)
                inp = b[:, :-1]
                tgt = b[:, 1:]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = mod(inp, fp16_fold_eval=fp16_fold_eval)
                nll = F.cross_entropy(lg.float().reshape(-1, 256), tgt.reshape(-1), reduction="sum")
                tot_nll += float(nll.item())
                tot_toks += int(tgt.numel())
        bpb = (tot_nll / float(tot_toks)) / math.log(2.0)
        return (bpb, tot_nll, tot_toks)

    arm_names = (
        "arm1_prefix_fwht",
        "arm2_fixed_diff_fwht",
        "arm2b_fixed_diff_1scalar_fwht",
        "arm3_learned_1kib_fp16_fold_fwht",
        "arm4_dense_256x512",
    )
    arm_results: Dict[str, Dict[str, Any]] = {}
    max_start = len(tr_raw) - seq_len - 1

    backbone_init_by_seed: Dict[int, Dict[str, torch.Tensor]] = {}
    train_starts_by_seed: Dict[int, List[List[int]]] = {}
    for sd in seeds:
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        ref_m = HoloSphereByteLM("arm1_prefix_fwht")
        backbone_init_by_seed[sd] = {k: v.detach().cpu().clone() for k, v in ref_m.state_dict().items()}
        g_dat = torch.Generator(device="cpu").manual_seed(sd * 7919 + 17)
        train_starts_by_seed[sd] = torch.randint(0, max_start, (steps, batch_size), generator=g_dat).tolist()

    for arm in arm_names:
        per_seed: List[Dict[str, Any]] = []
        head_params = 0
        head_bytes_f16 = 0
        total_params = 0
        for sd in seeds:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            model = HoloSphereByteLM(arm).to(device)
            cur_sd = model.state_dict()
            for k, v in backbone_init_by_seed[sd].items():
                if k in cur_sd:
                    cur_sd[k].copy_(v)
            model.load_state_dict(cur_sd, strict=True)
            head_params, head_bytes_f16 = model.head_param_count_and_bytes()
            total_params = int(sum(p.numel() for p in model.parameters()))

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
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(inp)
                    loss = F.cross_entropy(lg.float().reshape(-1, 256), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            torch.cuda.synchronize(device)
            wall_s = time.perf_counter() - t0
            bpb_bf16, _, n_eval_toks = _eval_bpb(model, fp16_fold_eval=False)
            bpb_fp16 = bpb_bf16
            q_bf16 = to_q(bpb_bf16, 100_000)
            q_fp16 = q_bf16
            row_dict: Dict[str, Any] = {
                "seed": int(sd),
                "eval_tokens": int(n_eval_toks),
                "train_wall_s_q": to_q_str(wall_s, 1000),
            }
            if arm == "arm3_learned_1kib_fp16_fold_fwht":
                bpb_fp16, _, _ = _eval_bpb(model, fp16_fold_eval=True)
                q_fp16 = to_q(bpb_fp16, 100_000)
                alpha_rms = float(model.fold_alpha.detach().float().pow(2).mean().sqrt().item())
                beta_rms = float(model.fold_beta.detach().float().pow(2).mean().sqrt().item())
                row_dict["alpha_rms_q"] = to_q_str(alpha_rms, 10_000)
                row_dict["beta_rms_q"] = to_q_str(beta_rms, 10_000)

            row_dict["eval_bpb_q"] = format_q(q_fp16)
            row_dict["eval_bpb_bf16_q"] = format_q(q_bf16)
            row_dict["fp16_quant_bpb_diff_q"] = format_q(abs(q_fp16 - q_bf16))
            per_seed.append(row_dict)
            del model, opt
            torch.cuda.empty_cache()

        mean_bpb_q = sum(Fraction(r["eval_bpb_q"]) for r in per_seed) / Fraction(len(per_seed))
        arm_results[arm] = {
            "head_params": head_params,
            "head_fp16_bytes": head_bytes_f16,
            "total_params": total_params,
            "hbm_vocab_table_bytes": 0 if arm != "arm4_dense_256x512" else head_bytes_f16,
            "mean_eval_bpb_q": format_q(mean_bpb_q),
            "seeds": per_seed,
        }

    return {
        "energy_power_status": ENERGY_POLICY_STATUS,
        "corpus_manifest_sha256": str(man["manifest_sha256"]),
        "corpus_train_bin_sha256": tr_sha,
        "corpus_eval_bin_sha256": ev_sha,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "steps_per_seed": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_seed": steps * batch_size * seq_len,
        "eval_chunks": eval_chunks,
        "eval_bytes": eval_chunks * seq_len,
        "seeds": [int(s) for s in seeds],
        "butterfly_vs_matrix_fp64_max_err_q": to_q_str(butterfly_vs_mat_err, 10**12),
        "arm1_upper_half_grad_norm_q": to_q_str(arm1_upper_half_grad_norm, 1_000_000),
        "arm3_upper_half_grad_norm_q": to_q_str(arm3_upper_half_grad_norm, 1_000_000),
        "arms": arm_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Controlled 5-arm x 3-seed UTF-8 byte-LM BPB study for 1 KiB FP16 folded FWHT_256 x bent4 head."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True, help="Directory with train.bin, eval.bin, flagship_corpus_manifest.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--eval-chunks", type=int, default=64)
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to write study JSON output")
    parser.add_argument("--allow-custom-corpus", action="store_true", help="Skip pinned flagship corpus SHA-256 checks")
    parser.add_argument("--verify-corpus-only", action="store_true", help="Verify corpus SHA-256 pins and exit without GPU")
    args = parser.parse_args(argv)

    if args.verify_corpus_only:
        man, _, _, tr_sha, ev_sha = verify_and_load_corpus(
            args.corpus_dir, enforce_pinned_digests=not args.allow_custom_corpus
        )
        print(
            json.dumps(
                {
                    "status": "CORPUS_OK",
                    "corpus_manifest_sha256": str(man["manifest_sha256"]),
                    "corpus_train_bin_sha256": tr_sha,
                    "corpus_eval_bin_sha256": ev_sha,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    study = run_pytorch_bpb_and_parity_study(
        corpus_dir=args.corpus_dir,
        seeds=(42, 137, 2026),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_chunks=args.eval_chunks,
        enforce_pinned_digests=not args.allow_custom_corpus,
    )
    rendered = json.dumps(study, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if args.out_json is not None:
        args.out_json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
