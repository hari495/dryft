"""Static KV cache and the fixed device buffers a decode step reads/writes.

Cache layout: ``[layers, B_cap, kv_heads, L_cap, head_dim]`` bf16 (head-major
per sequence). ``cache[l, :B, :, :L]`` is exactly the ``[B, H, L, D]`` view
SDPA wants, so the torch decode path reads the cache with zero copies. The
Triton kernels take strides and are layout-agnostic.

``seq_lens[b]`` is the number of valid cached tokens of row ``b`` and also the
absolute position of the token being decoded. Kernels read it from the device
tensor so one CUDA graph serves every position.
"""

from __future__ import annotations

import torch

from .config import ModelConfig


class DecodeState:
    def __init__(
        self,
        cfg: ModelConfig,
        b_cap: int,
        l_cap: int,
        device: torch.device,
        pinned: bool,
        rope: tuple[torch.Tensor, torch.Tensor],
        ring_depth: int = 4,
    ):
        self.cfg = cfg
        # cos/sin tables [>= l_cap, head_dim] bf16. Held by the state (not the
        # model) so a fallback state can carry longer tables without
        # replacing tensors that captured graphs point at.
        self.cos, self.sin = rope
        assert self.cos.shape[0] >= l_cap and self.sin.shape[0] >= l_cap
        self.b_cap = b_cap
        self.l_cap = l_cap
        self.device = device
        shape = (cfg.num_layers, b_cap, cfg.num_kv_heads, l_cap, cfg.head_dim)
        # Zero-initialised so a masked-out (never written) slot can never hold
        # NaN: 0 * NaN in P @ V would poison a live row.
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.seq_lens = torch.zeros(b_cap, dtype=torch.int64, device=device)
        self.ids = torch.zeros(b_cap, dtype=torch.int64, device=device)
        self.next_ids = torch.zeros(b_cap, dtype=torch.int64, device=device)
        self.row_idx = torch.arange(b_cap, dtype=torch.int64, device=device)
        self.head_idx = torch.arange(cfg.num_kv_heads, dtype=torch.int64, device=device)
        self.pos_range = torch.arange(l_cap, dtype=torch.int64, device=device)
        self.pinned = pinned
        self.ring: torch.Tensor | None = None
        if pinned:
            self.ring = torch.zeros((ring_depth, b_cap), dtype=torch.int64, pin_memory=True)

    def bytes(self) -> int:
        return 2 * self.k_cache.numel() * self.k_cache.element_size()

    def reset_rows(self, b: int) -> None:
        """Forget every sequence. Cache contents may stay: masked by seq_lens."""
        self.seq_lens[:b].zero_()
        self.ids[:b].zero_()
        self.next_ids[:b].zero_()
