"""Safetensors loading into flat bf16 tensors, with QKV and gate/up fusion.

Layout after load (all bf16, contiguous, on the engine device):
  embed            [V, H]           also the lm_head (tied)
  layers[i].w_in   [H]              input_layernorm
  layers[i].w_qkv  [Hq + 2 Hkv, H]  rows = q_proj ; k_proj ; v_proj
  layers[i].w_qn   [D]              q_norm
  layers[i].w_kn   [D]              k_norm
  layers[i].w_o    [H, Hq]
  layers[i].w_post [H]              post_attention_layernorm
  layers[i].w_gu   [2 I, H]         rows = gate_proj ; up_proj
  layers[i].w_down [H, I]
  norm             [H]
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open

from .config import ModelConfig, log


@dataclass
class LayerWeights:
    w_in: torch.Tensor
    w_qkv: torch.Tensor
    w_qn: torch.Tensor
    w_kn: torch.Tensor
    w_o: torch.Tensor
    w_post: torch.Tensor
    w_gu: torch.Tensor
    w_down: torch.Tensor


@dataclass
class Weights:
    embed: torch.Tensor
    layers: list[LayerWeights]
    norm: torch.Tensor

    @property
    def lm_head(self) -> torch.Tensor:
        return self.embed


def _shard_files(model_path: str) -> list[str]:
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({os.path.join(model_path, v) for v in weight_map.values()})
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {model_path}")
    return files


def load_weights(model_path: str, cfg: ModelConfig, device: torch.device) -> Weights:
    dev = str(device)
    raw: dict[str, torch.Tensor] = {}
    for path in _shard_files(model_path):
        with safe_open(path, framework="pt", device=dev) as f:
            for name in f.keys():
                raw[name] = f.get_tensor(name)

    def take(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        t = raw.pop(name)
        if tuple(t.shape) != shape:
            raise RuntimeError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous()

    H, Hq, Hkv, I, D, V = cfg.hidden, cfg.q_dim, cfg.kv_dim, cfg.intermediate, cfg.head_dim, cfg.vocab
    embed = take("model.embed_tokens.weight", (V, H))
    if "lm_head.weight" in raw:
        head = take("lm_head.weight", (V, H))
        if not cfg.tie_embeddings:
            raise RuntimeError("untied lm_head is not supported by this engine")
        if not torch.equal(head, embed):
            raise RuntimeError("lm_head.weight present and differs from embed_tokens while tie_word_embeddings=true")
        del head

    layers: list[LayerWeights] = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        q = take(p + "self_attn.q_proj.weight", (Hq, H))
        k = take(p + "self_attn.k_proj.weight", (Hkv, H))
        v = take(p + "self_attn.v_proj.weight", (Hkv, H))
        w_qkv = torch.cat([q, k, v], dim=0).contiguous()
        del q, k, v
        g = take(p + "mlp.gate_proj.weight", (I, H))
        u = take(p + "mlp.up_proj.weight", (I, H))
        w_gu = torch.cat([g, u], dim=0).contiguous()
        del g, u
        layers.append(
            LayerWeights(
                w_in=take(p + "input_layernorm.weight", (H,)),
                w_qkv=w_qkv,
                w_qn=take(p + "self_attn.q_norm.weight", (D,)),
                w_kn=take(p + "self_attn.k_norm.weight", (D,)),
                w_o=take(p + "self_attn.o_proj.weight", (H, Hq)),
                w_post=take(p + "post_attention_layernorm.weight", (H,)),
                w_gu=w_gu,
                w_down=take(p + "mlp.down_proj.weight", (H, I)),
            )
        )
    norm = take("model.norm.weight", (H,))
    if raw:
        log(f"warning: {len(raw)} unused tensors in checkpoint: {sorted(raw)[:5]}...")
    return Weights(embed=embed, layers=layers, norm=norm)
