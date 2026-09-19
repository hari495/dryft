"""Triton kernels for the decode step, imported by ``model.py`` when the
matching ``FLAGS`` are on and CUDA + triton are available.

  rmsnorm.py      fused residual add + RMSNorm            FLAGS["TRITON_RMSNORM"]
  rope_qknorm.py  q/k head norm + RoPE + KV-cache write   FLAGS["TRITON_ROPE"]
  attn_decode.py  split-KV GQA decode attention           FLAGS["TRITON_ATTN_DECODE"]

Each module carries a pure-torch ``reference``/``*_ref`` and a ``selftest()``;
``python agent/verify.py --kernels`` runs them on the GPU.
"""
