"""Fused ``residual add + RMSNorm`` (Tier 2.1) and plain RMSNorm, in Triton.

Semantics mirror ``Qwen3RMSNorm`` plus the bf16 residual add around it:
    res_new = bf16(res + x)                     # HF: residual + hidden_states (bf16 add)
    y       = bf16( bf16(res_new_f32 * rsqrt(mean(res_new_f32^2) + eps)) * w )
i.e. normalise in fp32, round to bf16, THEN multiply by the weight and round
once more. Reference implementations in torch live beside each kernel; run
``selftest()`` (``python agent/verify.py --kernels``) on the target GPU.

STATUS: enabled (R1); selftest runs in Model.__init__ on the target GPU and
the torch path is used if it fails.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(
    res_ptr, x_ptr, w_ptr, res_out_ptr, y_ptr,
    n_cols, eps,
    HAS_X: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row.to(tl.int64) * n_cols + cols
    r = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_X:
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        r = (r + x).to(tl.bfloat16).to(tl.float32)  # bf16 residual add, rounded once
        tl.store(res_out_ptr + offs, r.to(tl.bfloat16), mask=mask)
    var = tl.sum(r * r, axis=0) / n_cols
    normed = (r * tl.math.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (normed * w).to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_rows_kernel(
    x_ptr, w_ptr, y_ptr,
    n_rows, n_cols, eps,
    R: tl.constexpr, BLOCK: tl.constexpr,
):
    """R rows per program; for narrow rows (per-head q/k norm, n = 128)."""
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    offs = rows[:, None].to(tl.int64) * n_cols + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / n_cols
    normed = (x * tl.math.rsqrt(var + eps)[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (normed * w[None, :]).to(tl.bfloat16), mask=mask)


ROWS_PER_PROGRAM = 16  # for n_cols <= 256


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``[..., H]`` bf16 -> same shape bf16; matches ``Qwen3RMSNorm``."""
    x2 = x.contiguous().view(-1, x.shape[-1])
    y = torch.empty_like(x2)
    rows, n = x2.shape
    block = triton.next_power_of_2(n)
    if n <= 256:
        grid = (triton.cdiv(rows, ROWS_PER_PROGRAM),)
        _rmsnorm_rows_kernel[grid](x2, w, y, rows, n, eps, R=ROWS_PER_PROGRAM, BLOCK=block, num_warps=4)
    else:
        _add_rmsnorm_kernel[(rows,)](
            x2, x2, w, x2, y, n, eps, HAS_X=False, BLOCK=block, num_warps=8 if block >= 2048 else 4
        )
    return y.view(x.shape)


def add_rmsnorm(res: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(res + x, rmsnorm(res + x) * w)`` both bf16 ``[M, H]``."""
    res2 = res.contiguous().view(-1, res.shape[-1])
    x2 = x.contiguous().view(-1, x.shape[-1])
    res_out = torch.empty_like(res2)
    y = torch.empty_like(res2)
    rows, n = res2.shape
    block = triton.next_power_of_2(n)
    _add_rmsnorm_kernel[(rows,)](
        res2, x2, w, res_out, y, n, eps, HAS_X=True, BLOCK=block, num_warps=8 if block >= 2048 else 4
    )
    return res_out.view(res.shape), y.view(res.shape)


# ----------------------------------------------------------------- reference
def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return w * (xf * torch.rsqrt(var + eps)).to(x.dtype)


def add_rmsnorm_ref(res, x, w, eps):
    r = res + x
    return r, rmsnorm_ref(r, w, eps)


def selftest(device: str = "cuda") -> None:
    torch.manual_seed(0)
    tol = 2 * 2**-8  # one bf16 ulp of slack, relative to the largest output
    # Shapes the model uses (hidden 2560 rows; 128-wide head rows) plus odd
    # row counts for masking. Each distinct (kernel, BLOCK) costs a compile
    # at load, so stay on the production BLOCK sizes.
    shapes = ((1, 2560), (32, 2560), (7, 2560), (8192, 2560), (1, 128), (5, 128), (33, 128), (1000, 128))
    for rows, n in shapes:
        res = (torch.randn(rows, n, device=device) * 3).to(torch.bfloat16)
        x = torch.randn(rows, n, device=device).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(n, device=device)).to(torch.bfloat16)
        y = rmsnorm(res, w, 1e-6)
        y_ref = rmsnorm_ref(res, w, 1e-6)
        err = (y.float() - y_ref.float()).abs().max().item()
        assert err <= tol * y_ref.float().abs().max().item(), f"rmsnorm mismatch rows={rows} n={n}: {err}"
        if n > 256:  # add_rmsnorm only ever sees the hidden width
            r2, y2 = add_rmsnorm(res, x, w, 1e-6)
            r_ref, y2_ref = add_rmsnorm_ref(res, x, w, 1e-6)
            assert torch.equal(r2, r_ref), f"residual add mismatch rows={rows} n={n}"
            err = (y2.float() - y2_ref.float()).abs().max().item()
            assert err <= tol * y2_ref.float().abs().max().item(), f"add_rmsnorm mismatch rows={rows} n={n}: {err}"
    # strided input (a q slice out of the fused qkv row) must be handled by the wrapper
    qkv = torch.randn(4, 6144, device=device).to(torch.bfloat16)
    w = torch.ones(128, device=device, dtype=torch.bfloat16)
    q = qkv[:, :4096].view(4, 32, 128)
    y, y_ref = rmsnorm(q, w, 1e-6).float(), rmsnorm_ref(q, w, 1e-6).float()
    assert (y - y_ref).abs().max().item() <= tol * y_ref.abs().max().item(), "strided q slice"
