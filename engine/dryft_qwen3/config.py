"""Model constants, config assertions, bucket tables and feature flags.

Everything shape-related that the engine bakes into buffers or CUDA graphs is
decided here, once, so the rest of the engine reads numbers instead of
guessing them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace

import torch

# --------------------------------------------------------------------------
# Feature flags. Every optimisation is gated so it can be A/B'd in one process
# and switched off without re-engineering. Defaults are the shipped behaviour.
# --------------------------------------------------------------------------
FLAGS: dict[str, bool] = {
    # Capture one CUDA graph per (batch bucket, length bucket) for the decode
    # step and replay it. Falls back to eager per bucket if validation fails.
    "CUDA_GRAPHS": True,
    # Keep one decode step in flight; D2H copies land in a pinned ring and the
    # host waits on an event for step t-1 only.
    "PIPELINE": True,
    # Prefill: pass enable_gqa=True to SDPA (flash backend, K/V read straight
    # from the cache views) instead of materialising 32 K/V heads. Model.__init__
    # probes that flash really takes the GQA call and otherwise repeats K/V.
    "PREFILL_ENABLE_GQA": True,
    # Triton kernels (Tier 2/3). Each enabled kernel selftests on the GPU in
    # Model.__init__ and silently falls back to the torch path if it fails.
    # R1 (2026-09-19): fused residual+RMSNorm on, decode and prefill -> 524.
    # R2: fused qk-norm+RoPE+KV-write and split-KV decode attention on -> 793.
    # R3: rope kernel in prefill too, fused silu*mul, flash GQA prefill.
    "TRITON_RMSNORM": True,
    "TRITON_ROPE": True,
    "TRITON_ATTN_DECODE": True,
    "TRITON_SILU_MUL": True,
}


def flag(name: str) -> bool:
    return bool(FLAGS[name])


# --------------------------------------------------------------------------
# Pinned checkpoint facts (AGENTS.md §3). Asserted against config.json at load.
# --------------------------------------------------------------------------
EXPECTED_CONFIG: dict[str, object] | None = {
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "hidden_act": "silu",
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000,
    "rope_scaling": None,
    "max_position_embeddings": 262144,
    "attention_bias": False,
}


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    eps: float
    rope_theta: float
    tie_embeddings: bool
    max_position: int

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def group(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def bytes_per_token_kv(self) -> int:
        return self.num_layers * 2 * self.num_kv_heads * self.head_dim * 2

    @property
    def decode_bytes_per_step(self) -> int:
        """Weight bytes streamed per decode step (bf16, tied lm_head)."""
        per_layer = (
            self.qkv_dim * self.hidden
            + self.hidden * self.q_dim
            + 3 * self.intermediate * self.hidden
        )
        return 2 * (self.num_layers * per_layer + self.vocab * self.hidden)


def load_model_config(model_path: str) -> ModelConfig:
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    if EXPECTED_CONFIG is not None:
        bad = []
        for key, want in EXPECTED_CONFIG.items():
            got = raw.get(key, None)
            if key == "rms_norm_eps":
                ok = abs(float(got) - float(want)) < 1e-12
            else:
                ok = got == want
            if not ok:
                bad.append(f"{key}: expected {want!r}, config.json has {got!r}")
        if bad:
            raise RuntimeError(
                "config.json does not match the pinned Qwen3-4B-Instruct-2507 checkpoint:\n  "
                + "\n  ".join(bad)
            )
    # Architectural invariants the engine relies on regardless of size.
    assert raw.get("hidden_act") == "silu"
    assert raw.get("attention_bias", False) is False
    assert raw.get("rope_scaling", None) is None
    assert not raw.get("use_sliding_window", False)
    assert raw["num_attention_heads"] % raw["num_key_value_heads"] == 0
    head_dim = raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
    return ModelConfig(
        num_layers=int(raw["num_hidden_layers"]),
        hidden=int(raw["hidden_size"]),
        num_heads=int(raw["num_attention_heads"]),
        num_kv_heads=int(raw["num_key_value_heads"]),
        head_dim=int(head_dim),
        intermediate=int(raw["intermediate_size"]),
        vocab=int(raw["vocab_size"]),
        eps=float(raw["rms_norm_eps"]),
        rope_theta=float(raw["rope_theta"]),
        tie_embeddings=bool(raw.get("tie_word_embeddings", False)),
        max_position=int(raw["max_position_embeddings"]),
    )


# --------------------------------------------------------------------------
# Engine settings: capacities and buckets. Tests shrink these for tiny models.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    # Static KV capacity. 32 x 8192 tokens x 147 KB = 37.7 GB.
    b_max: int = 32
    l_max: int = 8192
    # Decode graphs are captured per (batch bucket, length bucket). A request
    # picks the smallest bucket >= its need; anything larger takes the eager
    # fallback with a dynamically allocated cache.
    batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    len_buckets: tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
    # Pinned host ring for the pipelined yield loop.
    ring_depth: int = 4
    # cuBLAS warmup M values for prefill GEMMs.
    prefill_warmup_m: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384)
    # Dummy generate shapes run once in __init__ (batch, prompt, out).
    warmup_shapes: tuple[tuple[int, int, int], ...] = ((1, 512, 4), (4, 2048, 4), (16, 512, 4))
    # Seconds of GPU burn before the constructor returns (clock stabilisation).
    burn_seconds: float = 2.0
    # Number of eager-vs-graph validation replays per captured graph.
    graph_validate: bool = True
    # Per-shape prompt/decode budget guard: init aborts capture beyond this.
    init_budget_seconds: float = 240.0


SETTINGS = Settings()


def with_settings(**kw) -> Settings:
    return replace(SETTINGS, **kw)


def pick_bucket(buckets: tuple[int, ...], need: int) -> int | None:
    for b in buckets:
        if b >= need:
            return b
    return None


# --------------------------------------------------------------------------
# Device / runtime facts.
# --------------------------------------------------------------------------
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def log(*parts: object) -> None:
    """Engine diagnostics go to stderr; the harness reads tokens elsewhere."""
    print("[engine]", *parts, file=sys.stderr, flush=True)
