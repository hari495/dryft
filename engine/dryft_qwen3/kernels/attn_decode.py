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

STATUS: enabled (R2); selftest runs in Model.__init__ on the target GPU and
the torch path is used if it fails.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

BLOCK_N = 64
MAX_SPLITS = 32


def _qpad(G: int, QL: int) -> int:
    return max(16, triton.next_power_of_2(G * QL))


@triton.jit
def _attn_split_kernel(
    q_ptr, k_ptr, v_ptr, lens_ptr, acc_ptr, ml_ptr,
    stride_qb, stride_qj, stride_qh,
    stride_kb, stride_kh, stride_kl,
    stride_vb, stride_vh, stride_vl,
    seg, scale,
    G: tl.constexpr, QL: tl.constexpr, NH: tl.constexpr, D: tl.constexpr, S_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr, QPAD: tl.constexpr,
):
    """Rows of the query block: r = g * QL + j for head g of this kv group and
    chain position j; row r attends keys 0 .. lens[b] + j (the chain's K/V is
    already in the cache at lens[b] + 1 .. lens[b] + QL - 1)."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)
    lens_b = tl.load(lens_ptr + b).to(tl.int32)
    n_valid = lens_b + QL  # keys 0 .. lens_b + QL - 1 exist for the last chain row
    start = (s * seg).to(tl.int32)
    end = tl.minimum(start + seg, n_valid).to(tl.int32)

    offs_m = tl.arange(0, QPAD)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < G * QL
    g_of = tl.where(m_mask, offs_m // QL, 0)
    j_of = tl.where(m_mask, offs_m % QL, 0)
    row_limit = lens_b + j_of  # last key index row r may see
    q_ptrs = q_ptr + b.to(tl.int64) * stride_qb + j_of[:, None] * stride_qj + (kvh * G + g_of)[:, None] * stride_qh + offs_d[None, :]
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
        allowed = n_mask[None, :] & (n[None, :] <= row_limit[:, None])
        scores = tl.where(allowed, scores, float("-inf"))
        m_blk = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_blk)
        # A chain row may have no allowed key in this block and none so far
        # (m_new = -inf): keep it empty instead of computing (-inf) - (-inf).
        has_any = m_new != float("-inf")
        m_safe = tl.where(has_any, m_new, 0.0)
        alpha = tl.where(has_any, tl.exp(m_i - m_safe), 1.0)
        p = tl.exp(scores - m_safe[:, None])  # -inf scores -> 0
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    # partial row index matches the output layout: (b * QL + j) * NH + h
    h = kvh * G + g_of
    prow = (b.to(tl.int64) * QL + j_of) * NH + h
    out_ptrs = acc_ptr + (prow[:, None] * S_MAX + s) * D + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None])
    ml_base = ml_ptr + (prow * S_MAX + s) * 2
    tl.store(ml_base, m_i, mask=m_mask)
    tl.store(ml_base + 1, l_i, mask=m_mask)


@triton.jit
def _attn_reduce_kernel(
    acc_ptr, ml_ptr, out_ptr, n_splits,
    D: tl.constexpr, S_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)  # (b * QL + j) * NH + h
    offs_s = tl.arange(0, S_MAX)
    offs_d = tl.arange(0, D)
    s_mask = offs_s < n_splits
    m = tl.load(ml_ptr + (row * S_MAX + offs_s) * 2, mask=s_mask, other=float("-inf"))
    l = tl.load(ml_ptr + (row * S_MAX + offs_s) * 2 + 1, mask=s_mask, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)  # empty splits: exp(-inf) = 0
    l_tot = tl.sum(w * l, axis=0)
    acc = tl.load(acc_ptr + (row * S_MAX + offs_s)[:, None] * D + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], axis=0) / l_tot
    tl.store(out_ptr + row * D + offs_d, o.to(tl.bfloat16))


MAX_QL = 8  # scratch is sized for chains up to this many query rows per sequence


class _Scratch:
    """fp32 partials. Grown, never shrunk; retired buffers are kept alive so
    CUDA graphs captured against them stay valid."""

    acc: torch.Tensor | None = None
    ml: torch.Tensor | None = None
    retired: list[torch.Tensor] = []

    @classmethod
    def get(cls, rows: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        need = max(rows, 32 * 32 * MAX_QL) * MAX_SPLITS * D
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
                B: int, Lb: int, scale: float, QL: int = 1) -> torch.Tensor:
    """q ``[B * QL, nH, D]`` bf16 (row ``b * QL + j`` = chain position ``j`` of
    sequence ``b``); caches ``[B_cap, nKV, L_cap, D]``; seq_lens int64
    ``[B_cap]`` = position of chain token 0. Row ``j`` attends keys
    ``0 .. seq_lens[b] + j``. Returns ``[B * QL, nH, D]`` bf16."""
    _, nH, D = q.shape
    nKV = k_cache.shape[1]
    G = nH // nKV
    assert q.is_contiguous() and q.shape[0] == B * QL
    assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    splits, seg = num_splits(B, nKV, Lb + QL)
    acc, ml = _Scratch.get(B * QL * nH, D, q.device)
    out = torch.empty((B * QL, nH, D), dtype=torch.bfloat16, device=q.device)
    _attn_split_kernel[(B, nKV, splits)](
        q, k_cache, v_cache, seq_lens, acc, ml,
        q.stride(0) * QL, q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        seg, scale,
        G=G, QL=QL, NH=nH, D=D, S_MAX=MAX_SPLITS, BLOCK_N=BLOCK_N, QPAD=_qpad(G, QL),
        num_warps=4, num_stages=3,
    )
    _attn_reduce_kernel[(B * QL * nH,)](acc, ml, out, splits, D=D, S_MAX=MAX_SPLITS, num_warps=4)
    return out


# ----------------------------------------------------------------- reference
def reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, seq_lens: torch.Tensor,
              B: int, scale: float, QL: int = 1):
    """fp32 math attention; chain row j of sequence b sees keys 0..seq_lens[b]+j."""
    _, nH, D = q.shape
    nKV = k_cache.shape[1]
    G = nH // nKV
    out = torch.empty((B * QL, nH, D), dtype=torch.float32, device=q.device)
    for b in range(B):
        for j in range(QL):
            n = int(seq_lens[b]) + 1 + j
            k = k_cache[b, :, :n].float()  # [nKV, n, D]
            v = v_cache[b, :, :n].float()
            qb = q[b * QL + j].float().view(nKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qb, k) * scale
            p = torch.softmax(sc, dim=-1)
            out[b * QL + j] = torch.einsum("hgn,hnd->hgd", p, v).reshape(nH, D)
    return out.to(torch.bfloat16)


def selftest(device: str = "cuda") -> None:
    torch.manual_seed(0)
    # Production constexpr sets only (G=4, NH=32, D=128, QL in {1, 5}): every
    # extra set is a compile at load time. Odd lengths and length 1 cover the
    # masking; the QL=5 cases cover chain rows and empty blocks.
    for (B, nH, nKV, D, Lcap, Lb, lens, QL) in (
        (1, 32, 8, 128, 1024, 1024, [0], 1),
        (1, 32, 8, 128, 1024, 1024, [512], 1),
        (1, 32, 8, 128, 1024, 1024, [1023], 1),
        (4, 32, 8, 128, 2048, 2048, [1, 513, 2047, 64], 1),
        (16, 32, 8, 128, 512, 512, list(range(3, 3 + 16 * 31, 31)), 1),
        (2, 32, 8, 128, 8192, 8192, [8191, 4096], 1),
        (1, 32, 8, 128, 1024, 1024, [0], 5),
        (1, 32, 8, 128, 1024, 1024, [61], 5),
        (4, 32, 8, 128, 2048, 2048, [1, 513, 2040, 64], 5),
        (16, 32, 8, 128, 512, 512, list(range(3, 3 + 16 * 31, 31)), 5),
    ):
        q = torch.randn(B * QL, nH, D, device=device).to(torch.bfloat16)
        kc = torch.randn(B + 1, nKV, Lcap + QL, D, device=device).to(torch.bfloat16)
        vc = torch.randn(B + 1, nKV, Lcap + QL, D, device=device).to(torch.bfloat16)
        seq_lens = torch.tensor(lens + [0], dtype=torch.int64, device=device)
        scale = 1.0 / math.sqrt(D)
        got = attn_decode(q, kc, vc, seq_lens, B, Lb, scale, QL=QL)
        want = reference(q, kc, vc, seq_lens, B, scale, QL=QL)
        diff = (got.float() - want.float()).abs().max().item()
        tol = 4 * 2**-8 * want.float().abs().max().item() + 1e-3
        assert diff <= tol, f"attn mismatch B={B} QL={QL} lens={lens[:4]}: max diff {diff} > {tol}"
        assert torch.isfinite(got.float()).all()
