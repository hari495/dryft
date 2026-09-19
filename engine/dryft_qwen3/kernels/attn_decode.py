"""Split-KV GQA decode attention (Tier 3.1), Flash-Decoding style, in Triton.

Grid ``(B, nKV, S)``: each program serves the ``G`` query heads that share one
kv head over one KV segment, using ``tl.dot`` with the query block padded to
16 rows (tensor-core minimum). Partial ``(m, l, acc)`` per split go to fp32
scratch; ``_reduce`` combines the splits and writes bf16 ``[B, nH, D]``.

Only ``seq_lens[b] + 1`` keys are read per row (the current token's K/V has
already been written at position ``seq_lens[b]``), so cost tracks the real
context length instead of the bucket length.

Numerics: scores fp32 (bf16 x bf16 products, fp32 accumulate), online
softmax fp32, P rounded to bf16 for the PV dot (as flash attention does),
fp32 accumulation, one bf16 rounding at the end.

STATUS: written without hardware access; FLAGS["TRITON_ATTN_DECODE"] stays
off until ``selftest()`` passes on an H100.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BLOCK_N = 64
MAX_SPLITS = 32
QPAD = 16


@triton.jit
def _attn_split_kernel(
    q_ptr, k_ptr, v_ptr, lens_ptr, acc_ptr, ml_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kl,
    stride_vb, stride_vh, stride_vl,
    seg, scale,
    G: tl.constexpr, NH: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr, BLOCK_N: tl.constexpr, QPAD: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    n_valid = tl.load(lens_ptr + b).to(tl.int32) + 1
    start = (s * seg).to(tl.int32)
    end = tl.minimum(start + seg, n_valid).to(tl.int32)

    offs_m = tl.arange(0, QPAD)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < G
    q_ptrs = q_ptr + b.to(tl.int64) * stride_qb + (kvh * G + offs_m)[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)  # [QPAD, D] bf16

    m_i = tl.full([QPAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([QPAD], dtype=tl.float32)
    acc = tl.zeros([QPAD, D], dtype=tl.float32)
    k_base = k_ptr + b.to(tl.int64) * stride_kb + kvh * stride_kh
    v_base = v_ptr + b.to(tl.int64) * stride_vb + kvh * stride_vh
    for n0 in range(start, end, BLOCK_N):
        n = n0 + offs_n
        n_mask = n < end
        k = tl.load(k_base + n[:, None].to(tl.int64) * stride_kl + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        v = tl.load(v_base + n[:, None].to(tl.int64) * stride_vl + offs_d[None, :], mask=n_mask[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale  # [QPAD, BLOCK_N] fp32
        scores = tl.where(n_mask[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    h = kvh * G + offs_m  # query head ids
    out_ptrs = acc_ptr + ((b.to(tl.int64) * NH + h)[:, None] * S_MAX + s) * D + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None])
    ml_base = ml_ptr + ((b.to(tl.int64) * NH + h) * S_MAX + s) * 2
    tl.store(ml_base, m_i, mask=m_mask)
    tl.store(ml_base + 1, l_i, mask=m_mask)


@triton.jit
def _attn_reduce_kernel(
    acc_ptr, ml_ptr, out_ptr, n_splits,
    NH: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, S_MAX)
    offs_d = tl.arange(0, D)
    s_mask = offs_s < n_splits
    row = b.to(tl.int64) * NH + h
    m = tl.load(ml_ptr + (row * S_MAX + offs_s) * 2, mask=s_mask, other=float("-inf"))
    l = tl.load(ml_ptr + (row * S_MAX + offs_s) * 2 + 1, mask=s_mask, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)  # empty splits: exp(-inf) = 0
    l_tot = tl.sum(w * l, axis=0)
    acc = tl.load(acc_ptr + (row * S_MAX + offs_s)[:, None] * D + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], axis=0) / l_tot
    tl.store(out_ptr + row * D + offs_d, o.to(tl.bfloat16))


class _Scratch:
    """fp32 partials. Grown, never shrunk; retired buffers are kept alive so
    CUDA graphs captured against them stay valid."""

    acc: torch.Tensor | None = None
    ml: torch.Tensor | None = None
    retired: list[torch.Tensor] = []

    @classmethod
    def get(cls, B: int, nH: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        need = max(B, 32) * nH * MAX_SPLITS * D
        if cls.acc is None or cls.acc.numel() < need or cls.acc.device != device:
            if cls.acc is not None:
                cls.retired += [cls.acc, cls.ml]
            cls.acc = torch.zeros(need, dtype=torch.float32, device=device)
            cls.ml = torch.zeros(need // D * 2, dtype=torch.float32, device=device)
        return cls.acc, cls.ml


def num_splits(B: int, nKV: int, Lb: int) -> tuple[int, int]:
    """(splits, tokens per split) so that B*nKV*splits >= 2*132 programs."""
    want = max(1, min(MAX_SPLITS, -(-264 // (B * nKV))))
    seg = -(-Lb // want)
    seg = -(-seg // BLOCK_N) * BLOCK_N
    splits = -(-Lb // seg)
    return splits, seg


def attn_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor,
                B: int, Lb: int, scale: float) -> torch.Tensor:
    """q ``[B, nH, D]`` bf16; caches ``[B_cap, nKV, L_cap, D]``; seq_lens int64
    ``[B_cap]`` = position of the current token (keys ``0..seq_lens`` are
    read). Returns ``[B, nH, D]`` bf16."""
    _, nH, D = q.shape
    nKV = k_cache.shape[1]
    G = nH // nKV
    assert q.is_contiguous() and k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    splits, seg = num_splits(B, nKV, Lb)
    acc, ml = _Scratch.get(B, nH, D, q.device)
    out = torch.empty((B, nH, D), dtype=torch.bfloat16, device=q.device)
    _attn_split_kernel[(B, nKV, splits)](
        q, k_cache, v_cache, seq_lens, acc, ml,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        seg, scale,
        G=G, NH=nH, D=D, S_MAX=MAX_SPLITS, BLOCK_N=BLOCK_N, QPAD=QPAD, num_warps=4, num_stages=3,
    )
    _attn_reduce_kernel[(B, nH)](acc, ml, out, splits, NH=nH, D=D, S_MAX=MAX_SPLITS, num_warps=4)
    return out


# ----------------------------------------------------------------- reference
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor, B: int, scale: float):
    """fp32 math attention over keys 0..seq_lens[b]."""
    _, nH, D = q.shape
    nKV = k_cache.shape[1]
    G = nH // nKV
    out = torch.empty((B, nH, D), dtype=torch.float32, device=q.device)
    for b in range(B):
        n = int(seq_lens[b]) + 1
        k = k_cache[b, :, :n].float()  # [nKV, n, D]
        v = v_cache[b, :, :n].float()
        qb = q[b].float().view(nKV, G, D)
        sc = torch.einsum("hgd,hnd->hgn", qb, k) * scale
        p = torch.softmax(sc, dim=-1)
        out[b] = torch.einsum("hgn,hnd->hgd", p, v).reshape(nH, D)
    return out.to(torch.bfloat16)


def selftest(device: str = "cuda") -> None:
    torch.manual_seed(0)
    for (B, nH, nKV, D, Lcap, Lb, lens) in (
        (1, 32, 8, 128, 1024, 1024, [0]),
        (1, 32, 8, 128, 1024, 1024, [512]),
        (1, 32, 8, 128, 1024, 1024, [1023]),
        (4, 32, 8, 128, 2048, 2048, [1, 513, 2047, 64]),
        (16, 32, 8, 128, 512, 512, list(range(3, 3 + 16 * 31, 31))),
        (3, 4, 2, 16, 64, 64, [0, 17, 63]),
    ):
        q = torch.randn(B, nH, D, device=device).to(torch.bfloat16)
        kc = torch.randn(B + 1, nKV, Lcap, D, device=device).to(torch.bfloat16)
        vc = torch.randn(B + 1, nKV, Lcap, D, device=device).to(torch.bfloat16)
        seq_lens = torch.tensor(lens + [0], dtype=torch.int64, device=device)
        scale = 1.0 / math.sqrt(D)
        got = attn_decode(q, kc, vc, seq_lens, B, Lb, scale)
        want = reference(q, kc, vc, seq_lens, B, scale)
        diff = (got.float() - want.float()).abs().max().item()
        tol = 4 * 2**-8 * want.float().abs().max().item() + 1e-3
        assert diff <= tol, f"attn mismatch B={B} D={D} lens={lens[:4]}: max diff {diff} > {tol}"
        assert torch.isfinite(got.float()).all()
