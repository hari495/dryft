"""Fused ``residual add + RMSNorm`` (Tier 2.1) and plain RMSNorm, in Triton.

Semantics mirror ``Qwen3RMSNorm`` plus the bf16 residual add around it:
    res_new = bf16(res + x)                     # HF: residual + hidden_states (bf16 add)
    y       = bf16( bf16(res_new_f32 * rsqrt(mean(res_new_f32^2) + eps)) * w )
i.e. normalise in fp32, round to bf16, THEN multiply by the weight and round
once more. Reference implementations in torch live beside each kernel; run
``selftest()`` (``python agent/verify.py --kernels``) on the target GPU.

STATUS: written without hardware access; FLAGS["TRITON_RMSNORM"] stays off
until selftest passes on an H100.
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


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``[M, H]`` bf16 -> ``[M, H]`` bf16; matches ``Qwen3RMSNorm``."""
    x2 = x.contiguous().view(-1, x.shape[-1])
    y = torch.empty_like(x2)
    rows, n = x2.shape
    block = triton.next_power_of_2(n)
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
    for rows, n in ((1, 2560), (32, 2560), (5, 128), (16, 16), (7, 4096)):
        res = (torch.randn(rows, n, device=device) * 3).to(torch.bfloat16)
        x = torch.randn(rows, n, device=device).to(torch.bfloat16)
        w = (1 + 0.1 * torch.randn(n, device=device)).to(torch.bfloat16)
        y = rmsnorm(res, w, 1e-6)
        y_ref = rmsnorm_ref(res, w, 1e-6)
        assert torch.equal(y, y_ref) or (y.float() - y_ref.float()).abs().max() <= 2 * 2**-8 * y_ref.float().abs().max(), \
            f"rmsnorm mismatch rows={rows} n={n}: {(y.float() - y_ref.float()).abs().max()}"
        r2, y2 = add_rmsnorm(res, x, w, 1e-6)
        r_ref, y2_ref = add_rmsnorm_ref(res, x, w, 1e-6)
        assert torch.equal(r2, r_ref), f"residual add mismatch rows={rows} n={n}"
        assert (y2.float() - y2_ref.float()).abs().max() <= 2 * 2**-8 * y2_ref.float().abs().max(), \
            f"add_rmsnorm mismatch rows={rows} n={n}"
