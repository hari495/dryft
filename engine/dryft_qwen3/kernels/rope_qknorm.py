"""Fused per-head q/k RMSNorm + RoPE + KV-cache write (Tier 2.2), in Triton.

One program per (row, head). Heads ``[0, nH)`` are query heads: norm, rotate,
write to ``q_out[row, h, :]``. Heads ``[nH, nH+nKV)`` are kv heads: norm +
rotate k and write it to ``k_cache[row, kvh, pos[row], :]``; copy v to
``v_cache`` unchanged.

Numerics mirror HF exactly, including bf16 rounding between every op:
    n  = bf16( bf16(x_f32 * rsqrt(mean(x^2) + eps)) * w )          # Qwen3RMSNorm
    o1 = bf16( bf16(n1 * cos) + bf16(-n2 * sin) )                    # x*cos + rotate_half(x)*sin
    o2 = bf16( bf16(n2 * cos) + bf16( n1 * sin) )
with ``cos``/``sin`` the bf16 tables ``[L, D]`` (``cat(freqs, freqs)`` so the
two halves are equal; only the first ``D/2`` columns are read).

STATUS: enabled (R2); selftest runs in Model.__init__ on the target GPU and
the torch path is used if it fails.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _norm_rope(x1, x2, w1, w2, c, s, eps, D: tl.constexpr):
    """x1/x2: the two halves of one head (fp32 values of bf16 inputs)."""
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
    inv = tl.math.rsqrt(var + eps)
    n1 = _bf16(_bf16(x1 * inv) * w1)
    n2 = _bf16(_bf16(x2 * inv) * w2)
    o1 = _bf16(_bf16(n1 * c) + _bf16(-n2 * s))
    o2 = _bf16(_bf16(n2 * c) + _bf16(n1 * s))
    return o1, o2


@triton.jit
def _qknorm_rope_kv_kernel(
    qkv_ptr, wq_ptr, wk_ptr, cos_ptr, sin_ptr, pos_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    stride_qkv_row,
    stride_kc_b, stride_kc_h, stride_kc_l,
    stride_vc_b, stride_vc_h, stride_vc_l,
    eps,
    NH: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D // 2)
    pos = tl.load(pos_ptr + row).to(tl.int64)
    c = tl.load(cos_ptr + pos * D + d).to(tl.float32)
    s = tl.load(sin_ptr + pos * D + d).to(tl.float32)
    base = qkv_ptr + row.to(tl.int64) * stride_qkv_row
    if head < NH:
        src = base + head * D
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + D // 2 + d).to(tl.float32)
        w1 = tl.load(wq_ptr + d).to(tl.float32)
        w2 = tl.load(wq_ptr + D // 2 + d).to(tl.float32)
        o1, o2 = _norm_rope(x1, x2, w1, w2, c, s, eps, D)
        dst = q_out_ptr + (row.to(tl.int64) * NH + head) * D
        tl.store(dst + d, o1.to(tl.bfloat16))
        tl.store(dst + D // 2 + d, o2.to(tl.bfloat16))
    else:
        kvh = head - NH
        src = base + NH * D + kvh * D
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + D // 2 + d).to(tl.float32)
        w1 = tl.load(wk_ptr + d).to(tl.float32)
        w2 = tl.load(wk_ptr + D // 2 + d).to(tl.float32)
        o1, o2 = _norm_rope(x1, x2, w1, w2, c, s, eps, D)
        kdst = k_cache_ptr + row.to(tl.int64) * stride_kc_b + kvh * stride_kc_h + pos * stride_kc_l
        tl.store(kdst + d, o1.to(tl.bfloat16))
        tl.store(kdst + D // 2 + d, o2.to(tl.bfloat16))
        vsrc = base + NH * D + NKV * D + kvh * D
        v1 = tl.load(vsrc + d)
        v2 = tl.load(vsrc + D // 2 + d)
        vdst = v_cache_ptr + row.to(tl.int64) * stride_vc_b + kvh * stride_vc_h + pos * stride_vc_l
        tl.store(vdst + d, v1)
        tl.store(vdst + D // 2 + d, v2)


def qknorm_rope_kvwrite(
    qkv: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    pos: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, eps: float, nH: int, nKV: int, D: int,
) -> torch.Tensor:
    """qkv ``[M, (nH+2nKV)*D]`` bf16; pos int64 ``[M]``; k/v_cache ``[B_cap, nKV, L_cap, D]``
    (any strides, D contiguous). Returns rotated q ``[M, nH, D]`` bf16 and
    writes k/v for row ``m`` at ``[m, :, pos[m], :]``."""
    M = qkv.shape[0]
    assert qkv.stride(1) == 1 and cos.is_contiguous() and sin.is_contiguous()
    assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    q_out = torch.empty((M, nH, D), dtype=torch.bfloat16, device=qkv.device)
    _qknorm_rope_kv_kernel[(M, nH + nKV)](
        qkv, wq, wk, cos, sin, pos, q_out, k_cache, v_cache,
        qkv.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        eps, NH=nH, NKV=nKV, D=D, num_warps=1,
    )
    return q_out


# ----------------------------------------------------------------- reference
def _rmsnorm_ref(x, w, eps):
    xf = x.to(torch.float32)
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rope_ref(x, c, s):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * c) + (rot * s)


def reference(qkv, wq, wk, cos, sin, pos, nH, nKV, D, eps):
    M = qkv.shape[0]
    q = qkv[:, : nH * D].view(M, nH, D)
    k = qkv[:, nH * D : (nH + nKV) * D].view(M, nKV, D)
    v = qkv[:, (nH + nKV) * D :].view(M, nKV, D)
    c = cos[pos][:, None, :]
    s = sin[pos][:, None, :]
    return _rope_ref(_rmsnorm_ref(q, wq, eps), c, s), _rope_ref(_rmsnorm_ref(k, wk, eps), c, s), v


def selftest(device: str = "cuda") -> None:
    torch.manual_seed(0)
    for (M, nH, nKV, D, Lcap) in ((1, 32, 8, 128, 512), (16, 32, 8, 128, 2048), (32, 32, 8, 128, 8192)):  # production constexpr set only
        qkv = (torch.randn(M, (nH + 2 * nKV) * D, device=device) * 2).to(torch.bfloat16)
        wq = (1 + 0.1 * torch.randn(D, device=device)).to(torch.bfloat16)
        wk = (1 + 0.1 * torch.randn(D, device=device)).to(torch.bfloat16)
        inv = 1.0 / (5e6 ** (torch.arange(0, D, 2, device=device).float() / D))
        fr = torch.arange(Lcap, device=device).float()[:, None] * inv[None]
        emb = torch.cat((fr, fr), -1)
        cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        pos = torch.randint(0, Lcap, (M,), device=device, dtype=torch.int64)
        kc = torch.zeros((M + 2, nKV, Lcap, D), dtype=torch.bfloat16, device=device)
        vc = torch.zeros_like(kc)
        q = qknorm_rope_kvwrite(qkv, wq, wk, cos, sin, pos, kc[:M], vc[:M], 1e-6, nH, nKV, D)
        q_ref, k_ref, v_ref = reference(qkv, wq, wk, cos, sin, pos, nH, nKV, D, 1e-6)
        rows = torch.arange(M, device=device)
        k_got = kc[rows, :, pos]  # [M, nKV, D]
        v_got = vc[rows, :, pos]
        for name, got, want in (("q", q, q_ref), ("k", k_got, k_ref), ("v", v_got, v_ref)):
            diff = (got.float() - want.float()).abs().max().item()
            tol = 2 * 2**-8 * want.float().abs().max().item()
            assert diff <= tol, f"{name} mismatch M={M} D={D}: max diff {diff} > {tol}"
        # nothing else in the cache was touched
        kc[rows, :, pos] = 0
        vc[rows, :, pos] = 0
        assert not kc.any() and not vc.any(), "kernel wrote outside its slot"
