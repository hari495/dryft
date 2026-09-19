"""Forward passes: ``prefill`` (many tokens, cuBLAS + SDPA) and ``decode_step``
(one token per row, static shapes, graph-capturable).

Every arithmetic step mirrors ``transformers/models/qwen3/modeling_qwen3.py``
(4.51.3) including its bf16 rounding boundaries:
  * RMSNorm: fp32 reduce, normalise, cast to bf16, THEN multiply by weight.
  * q_norm / k_norm per head, before RoPE.
  * RoPE: cos/sin built in fp32 and cast to bf16 before the multiply; the
    rotation itself is bf16 arithmetic (``x*cos + rotate_half(x)*sin``).
  * softmax in fp32 (inside SDPA), scale 1/sqrt(head_dim).
  * MLP ``down(silu(gate) * up)`` with bf16 rounding after each op.
  * residual stream bf16; adds in bf16.
Reorder freely, never reformulate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cache import DecodeState
from .config import ModelConfig, flag, has_triton, log
from .weights import Weights


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return w * xn.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def _silu_mul(gu: torch.Tensor) -> torch.Tensor:
    g, u = gu.split(gu.shape[-1] // 2, dim=-1)
    return F.silu(g) * u


def rope_tables(cfg: ModelConfig, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin ``[length, head_dim]`` in bf16, computed exactly like HF's
    default rope init + ``Qwen3RotaryEmbedding.forward`` (fp32, then cast)."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64, device=device).to(torch.float32) / d))
    pos = torch.arange(length, dtype=torch.float32, device=device)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


class Model:
    def __init__(self, cfg: ModelConfig, weights: Weights, device: torch.device, rope_len: int):
        self.cfg = cfg
        self.w = weights
        self.device = device
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.rope_len = rope_len
        self.cos, self.sin = rope_tables(cfg, rope_len, device)
        self.triton = has_triton() and device.type == "cuda"
        # Each Triton kernel earns its place at load time: selftest on this GPU
        # or the torch path is used for it. Official runs hide stderr and one
        # wrong token fails a workload, so a kernel that cannot prove itself
        # here never runs (AGENTS.md section 6, step 2).
        self._k_rms = self._load_kernel("TRITON_RMSNORM", "rmsnorm")
        self._k_rope = self._load_kernel("TRITON_ROPE", "rope_qknorm")
        self._k_attn = self._load_kernel("TRITON_ATTN_DECODE", "attn_decode")
        self._k_silu = self._load_kernel("TRITON_SILU_MUL", "silu_mul")
        self.prefill_gqa = flag("PREFILL_ENABLE_GQA") and self._flash_gqa_works()

    def _flash_gqa_works(self) -> bool:
        """True iff SDPA runs GQA on the flash backend here (never the math
        backend, which would materialise B*H*T*T scores at prefill)."""
        if self.device.type != "cuda":
            return False
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            cfg = self.cfg
            q = torch.randn(1, cfg.num_heads, 2048, cfg.head_dim, device=self.device, dtype=torch.bfloat16)
            kv = torch.randn(1, cfg.num_kv_heads, 2, 4096, cfg.head_dim, device=self.device, dtype=torch.bfloat16)
            k, v = kv[:, :, 0, :2048], kv[:, :, 1, :2048]  # strided like the cache views
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
            k32 = torch.repeat_interleave(k, cfg.group, dim=1).contiguous()
            v32 = torch.repeat_interleave(v, cfg.group, dim=1).contiguous()
            ref = F.scaled_dot_product_attention(q, k32, v32, is_causal=True, scale=self.scale)
            torch.cuda.synchronize(self.device)
            ok = bool(torch.isfinite(o).all()) and (o.float() - ref.float()).abs().max().item() <= 2 * 2**-8 * ref.float().abs().max().item()
        except Exception as exc:  # noqa: BLE001 - any failure means "repeat K/V"
            log(f"prefill: flash GQA unavailable ({exc!r}); using repeat_interleave path")
            return False
        log(f"prefill: flash GQA {'ok' if ok else 'MISMATCH -> repeat_interleave path'}")
        return ok

    def _load_kernel(self, flag_name: str, module_name: str):
        if not (self.triton and flag(flag_name)):
            return None
        import importlib
        import time

        t0 = time.time()
        try:
            mod = importlib.import_module(f".kernels.{module_name}", __package__)
            mod.selftest(str(self.device))
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - any failure means "use torch"
            log(f"warning: {module_name} selftest FAILED in {time.time() - t0:.1f}s; torch path for {flag_name}: {exc!r}")
            return None
        log(f"{module_name}: selftest ok in {time.time() - t0:.1f}s")
        return mod

    # ------------------------------------------------------------------ prefill
    @torch.no_grad()
    def prefill(self, state: DecodeState, rows: torch.Tensor, ids: torch.Tensor, rows_are_prefix: bool = False) -> torch.Tensor:
        """Run ``ids`` ``[B, T]`` (equal lengths) through the model, write K/V
        into cache rows ``rows`` at positions ``0..T-1`` and return the greedy
        next token ``[B]``. Does not touch ``state.seq_lens``.
        ``rows_are_prefix``: ``rows == 0..B-1`` (lets the fused rope kernel
        address the cache directly)."""
        cfg = self.cfg
        B, T = ids.shape
        Hq, Hkv, D, nH, nKV, I = cfg.q_dim, cfg.kv_dim, cfg.head_dim, cfg.num_heads, cfg.num_kv_heads, cfg.intermediate
        x = F.embedding(ids, self.w.embed)  # [B, T, H]
        use_gqa = self.prefill_gqa
        tri_rms, tri_rope, tri_silu = self._k_rms, self._k_rope, self._k_silu
        norm = tri_rms.rmsnorm if tri_rms is not None else rmsnorm
        silu_mul = tri_silu.silu_mul if tri_silu is not None else _silu_mul
        # The fused rope kernel maps row m -> (m // T, m % T): only valid when
        # the cache rows are 0..B-1 in order.
        fused_rope = tri_rope is not None and rows_are_prefix and T <= state.cos.shape[0]
        if not fused_rope:
            cos = state.cos[:T][None, None]  # [1, 1, T, D]
            sin = state.sin[:T][None, None]

        def add_norm(res: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if tri_rms is not None:
                return tri_rms.add_rmsnorm(res, y, w, cfg.eps)
            r = res + y
            return r, rmsnorm(r, w, cfg.eps)

        layers = self.w.layers
        h = norm(x, layers[0].w_in, cfg.eps)
        for li, L in enumerate(layers):
            qkv = F.linear(h, L.w_qkv)  # [B, T, Hq + 2 Hkv]
            if fused_rope:
                q = tri_rope.qknorm_rope_kvwrite_prefill(
                    qkv, L.w_qn, L.w_kn, state.cos, state.sin,
                    state.k_cache[li], state.v_cache[li], cfg.eps, nH, nKV, D,
                )  # [B, nH, T, D]; K/V written into the cache
                k = state.k_cache[li, :B, :, :T]  # [B, nKV, T, D] views, D contiguous
                v = state.v_cache[li, :B, :, :T]
            else:
                q = qkv[..., :Hq].view(B, T, nH, D)
                k = qkv[..., Hq : Hq + Hkv].view(B, T, nKV, D)
                v = qkv[..., Hq + Hkv :].view(B, T, nKV, D)
                q = norm(q, L.w_qn, cfg.eps).transpose(1, 2)  # [B, nH, T, D]
                k = norm(k, L.w_kn, cfg.eps).transpose(1, 2)  # [B, nKV, T, D]
                v = v.transpose(1, 2)
                q = rope(q, cos, sin).contiguous()
                k = rope(k, cos, sin)
                state.k_cache[li, rows, :, :T] = k
                state.v_cache[li, rows, :, :T] = v
            if use_gqa:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=True)
            else:
                k32 = torch.repeat_interleave(k, cfg.group, dim=1)
                v32 = torch.repeat_interleave(v, cfg.group, dim=1)
                o = F.scaled_dot_product_attention(q, k32, v32, is_causal=True, scale=self.scale)
            o = o.transpose(1, 2).reshape(B, T, Hq)
            x, h = add_norm(x, F.linear(o, L.w_o), L.w_post)
            mlp = F.linear(silu_mul(F.linear(h, L.w_gu)), L.w_down)
            if li + 1 < len(layers):
                x, h = add_norm(x, mlp, layers[li + 1].w_in)
            else:
                x = x + mlp
        last = norm(x[:, -1], self.w.norm, cfg.eps)  # [B, H]
        logits = F.linear(last, self.w.lm_head)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------- decode step
    @torch.no_grad()
    def decode_step(self, state: DecodeState, Bb: int, Lb: int) -> None:
        """One greedy step for rows ``[0, Bb)`` attending over cache ``[0, Lb)``.

        Reads ``state.ids`` and ``state.seq_lens``; writes K/V at
        ``seq_lens``, then ``state.next_ids``, ``state.ids`` and increments
        ``state.seq_lens``. All shapes are static in ``(Bb, Lb)``; all inputs
        are device tensors, so the call is CUDA-graph capturable.
        """
        cfg = self.cfg
        Hq, Hkv, D, nH, nKV, I, G = (
            cfg.q_dim,
            cfg.kv_dim,
            cfg.head_dim,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.intermediate,
            cfg.group,
        )
        ids = state.ids[:Bb]
        pos = state.seq_lens[:Bb]
        rows = state.row_idx[:Bb]
        layers = self.w.layers
        eps = cfg.eps
        tri_rms, tri_rope, tri_attn = self._k_rms, self._k_rope, self._k_attn
        norm = tri_rms.rmsnorm if tri_rms is not None else rmsnorm
        silu_mul = self._k_silu.silu_mul if self._k_silu is not None else _silu_mul

        def add_norm(res: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """(res + y, rmsnorm(res + y) * w): the residual add and the norm
            that follows it, fused in Triton when enabled."""
            if tri_rms is not None:
                return tri_rms.add_rmsnorm(res, y, w, eps)
            r = res + y
            return r, rmsnorm(r, w, eps)

        x = F.embedding(ids, self.w.embed)  # [Bb, H] residual stream
        h = norm(x, layers[0].w_in, eps)
        if tri_rope is None:
            cos = state.cos[pos][:, None, :]  # [Bb, 1, D]
            sin = state.sin[pos][:, None, :]
            row_i = rows[:, None]
            head_i = state.head_idx[None, :]
            pos_i = pos[:, None]
        if tri_attn is None:
            mask = (state.pos_range[:Lb][None, :] <= pos[:, None])[:, None, None, :]  # [Bb, 1, 1, Lb]
        for li, L in enumerate(layers):
            qkv = F.linear(h, L.w_qkv)  # [Bb, Hq + 2 Hkv]
            if tri_rope is not None:
                q = tri_rope.qknorm_rope_kvwrite(
                    qkv, L.w_qn, L.w_kn, state.cos, state.sin, pos,
                    state.k_cache[li], state.v_cache[li], eps, nH, nKV, D,
                )
            else:
                q = norm(qkv[:, :Hq].view(Bb, nH, D), L.w_qn, eps)
                k = norm(qkv[:, Hq : Hq + Hkv].view(Bb, nKV, D), L.w_kn, eps)
                v = qkv[:, Hq + Hkv :].view(Bb, nKV, D)
                q = rope(q, cos, sin)
                k = rope(k, cos, sin)
                state.k_cache[li, row_i, head_i, pos_i] = k
                state.v_cache[li, row_i, head_i, pos_i] = v
            if tri_attn is not None:
                o = tri_attn.attn_decode(q, state.k_cache[li], state.v_cache[li], state.seq_lens, Bb, Lb, self.scale)
            else:
                kc = state.k_cache[li, :Bb, :, :Lb]  # [Bb, nKV, Lb, D]
                vc = state.v_cache[li, :Bb, :, :Lb]
                qg = q.view(Bb, nKV, G, D)  # q head h = kv*G + g  ==  h // G
                o = F.scaled_dot_product_attention(qg, kc, vc, attn_mask=mask, scale=self.scale)
            x, h = add_norm(x, F.linear(o.reshape(Bb, Hq), L.w_o), L.w_post)
            w_next = layers[li + 1].w_in if li + 1 < len(layers) else self.w.norm
            x, h = add_norm(x, F.linear(silu_mul(F.linear(h, L.w_gu)), L.w_down), w_next)
        logits = F.linear(h, self.w.lm_head)  # h == final norm output; [Bb, V] bf16
        nxt = logits.argmax(dim=-1)
        state.next_ids[:Bb].copy_(nxt)
        state.ids[:Bb].copy_(nxt)
        state.seq_lens[:Bb].add_(1)
