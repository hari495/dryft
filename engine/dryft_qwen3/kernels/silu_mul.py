"""Fused ``silu(gate) * up`` (Tier 2.3, elementwise half), in Triton.

Input is the fused gate_up GEMM output ``[M, 2I]`` (gate columns first, then
up); output ``[M, I]`` bf16. Numerics mirror HF's two bf16 ops:
    y = bf16( bf16(silu_f32(g)) * u )        with silu(x) = x / (1 + exp(-x))
Replaces two torch kernels (silu, mul) and one full pass over the ``[M, I]``
intermediate; in prefill at T = 8192 that is ~160 MB of traffic per layer.

STATUS: enabled (R3); selftest runs in Model.__init__ on the target GPU and
the torch path is used if it fails.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _silu_mul_kernel(gu_ptr, y_ptr, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = gu_ptr + row * stride_row
    g = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """``[..., 2I]`` bf16 -> ``[..., I]`` bf16."""
    n2 = gu.shape[-1]
    n = n2 // 2
    gu2 = gu.reshape(-1, n2)
    assert gu2.stride(1) == 1
    rows = gu2.shape[0]
    y = torch.empty((rows, n), dtype=gu.dtype, device=gu.device)
    _silu_mul_kernel[(rows, triton.cdiv(n, BLOCK))](gu2, y, n, gu2.stride(0), BLOCK=BLOCK, num_warps=4)
    return y.view(*gu.shape[:-1], n)


# ----------------------------------------------------------------- reference
def silu_mul_ref(gu: torch.Tensor) -> torch.Tensor:
    g, u = gu.split(gu.shape[-1] // 2, dim=-1)
    return F.silu(g) * u


def selftest(device: str = "cuda") -> None:
    torch.manual_seed(0)
    tol = 2 * 2**-8
    for rows, n in ((1, 9728), (32, 9728), (7, 9728), (4096, 9728), (3, 2048)):
        gu = (torch.randn(rows, 2 * n, device=device) * 3).to(torch.bfloat16)
        y, want = silu_mul(gu).float(), silu_mul_ref(gu).float()
        err = (y - want).abs().max().item()
        assert err <= tol * want.abs().max().item(), f"silu_mul mismatch rows={rows} n={n}: {err}"
    gu = (torch.randn(2, 5, 2 * 9728, device=device) * 3).to(torch.bfloat16)
    y, want = silu_mul(gu).float(), silu_mul_ref(gu).float()
    assert (y - want).abs().max().item() <= tol * want.abs().max().item(), "3-d input"
