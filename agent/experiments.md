# Experiment ledger

Calibration table (local vs official). Fill the official columns from
`./bin/dryft result <run-id>` after each deliberate run.

| date | commit | local geomean | official geomean | hidden per-workload | notes |
|------|--------|--------------:|-----------------:|---------------------|-------|
| 2026-09-19 | (uncommitted) | n/a (no local GPU) | not run | – | Tier 1 torch engine; verified on tiny CPU model only |

Environment note: the development machine is an Apple M4 Pro (no CUDA, no
triton wheel). Everything below marked **CPU-verified** was checked against
the HF reference on a random tiny Qwen3 (`agent/verify.py --tiny --all`);
everything marked **UNVERIFIED** has never executed on hardware. Per AGENTS.md
§6 ("if no local H100"), GPU-only paths are flag-gated and the first official
run doubles as the bench.

---

### 2026-09-19  [T0] verify.py / bench.py / profiler.py / prompts.py            KEEP
flag: –     commit: –
verify: tooling only. `--tiny` builds a 2-layer random Qwen3 (H=64, 4/2 heads,
D=16, V=256) at /tmp/dryft-tiny-qwen3 and runs every synthetic prompt set on
CPU. Replay-margin check confirmed to flag a corrupted token (margin 8.4).
bench: mirrors §8 (warmup + N samples, median tok/s, TTFT, TPOT, spread,
peak mem, BW); `--baseline` times `agent/baseline_engine.py` (the unmodified
starter) into `agent/baseline.json` for the 1.10x gates.
notes: `agent/profile.py` renamed to `agent/profiler.py` — a module named
`profile` shadows the stdlib and breaks `import transformers`.
next: T1.

### 2026-09-19  [T1.1–1.6] own forward pass, static KV, graphs, pipeline       KEEP (flags default on)
flag: CUDA_GRAPHS=1 PIPELINE=1     commit: –
verify (tiny, CPU): 100% on random, copyheavy_synth, ragged, long, shapes
(all bucket edges ±1, out∈{1,2}, batch buckets ±1, fallback B=9 and L=65),
eos_early_synth; run-twice identical; b1→b8→b1 identical; max margin 0.000.
bench: not measurable locally.
CPU-verified: weights.py (fused Wqkv/Wgu), cache.py, model.prefill (grouped
by prompt length, is_causal SDPA, last-token lm_head), model.decode_step
(GQA as [B, nKV, G, D] queries over the head-major cache, mask from
seq_lens), engine.generate eager path incl. fallback.
UNVERIFIED (CUDA only): graphs.py capture/validate, pinned ring + event
pipeline in engine._run, _warm_cublas, _burn, init timing.
design deviations from AGENTS.md (recorded in §9.1/§9.2):
  * KV cache is head-major `[layers, B, nKV, L, D]` (not token-major) so the
    torch SDPA path reads `[B, H, L, D]` views with zero copies.
  * Decode graphs are per (batch bucket, length bucket) — 6 × 10 = 60 graphs;
    the torch SDPA path must attend over a static length, so the bucket picks
    the smallest L ≥ prompt + max_new_tokens. The Triton attention kernel
    reads only `seq_lens` and makes the length bucket irrelevant.
notes: `gc.freeze(); gc.disable()` after init. Engine logs go to stderr only.
next: first official public run to get baseline + T1 numbers; then flip
TRITON_* flags one at a time after `verify.py --kernels` passes on the H100.

### 2026-09-19  [T2.1, T2.2, T3.1] Triton kernels                              WRITTEN, OFF
flag: TRITON_RMSNORM=0 TRITON_ROPE=0 TRITON_ATTN_DECODE=0     commit: –
verify: UNVERIFIED — no triton on this machine (macOS). Each module has a
torch reference and `selftest()`; `python agent/verify.py --kernels` runs
them. Wired into `Model.decode_step` behind the flags; the decode step was
restructured so the residual add fuses with the following norm
(`add_norm(res, y, w)`), which is identical arithmetic in the torch path
(re-verified 100% on tiny).
expected: −2 kernels/layer (rmsnorm), −~10 kernels/layer (rope+norm+kv
write), attention reads only `seq_lens+1` keys instead of the length bucket;
b1 attention splits 32 ways to fill 132 SMs.
risk: first-run compile/launch errors in Triton 3.1 syntax; check
`selftest()` output before enabling. Scratch for split partials is grown-not-
shrunk and retired buffers are kept alive so captured graphs stay valid.
next: on GPU — `verify.py --kernels`, then enable one flag, `verify.py --all`,
`bench.py --public --extra`, record here.
