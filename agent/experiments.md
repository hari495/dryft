# Experiment ledger

Calibration table (local vs official). Fill the official columns from
`./bin/dryft result <run-id>` after each deliberate run.

| date | commit | local geomean | official geomean | hidden per-workload | notes |
|------|--------|--------------:|-----------------:|---------------------|-------|
| 2026-09-19 | 0fb3a60 | n/a (no local GPU) | **405.0** (native=100) | hidden not shown; public: b1 104 tok/s, b4 206, b16 1337 | T1 torch path, graphs+pipeline; all gates pass; leaders 633/624 |

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

### 2026-09-19  [official run d2b0b294] commit 0fb3a60                          RESULT
flags: CUDA_GRAPHS=1 PIPELINE=1 TRITON_*=0
score: 405.0 (geomean over 6 hidden workloads, native = 100). Leaderboard top: 633, 624, then ~300.
public (ours vs native, 5 samples each):
  b1x512x32    104 tok/s  total 306 ms (ref 805)   TTFT 22/28 ms = 0.81x   TPOT  9.14/25.1 ms = 0.36x  spread 0.1%
  b4x2048x32   206 tok/s  total 620 ms (ref 1040)  TTFT 213/201 ms = 1.05x TPOT 13.17/27.0 ms = 0.48x  spread 0.5%
  b16x512x128 1337 tok/s  total 1530 ms (ref 3930) TTFT 202/192 ms = 1.05x TPOT 10.45/29.3 ms = 0.35x  spread 0.3%
peak memory 49.2 GiB (52.83 GB) on all three -> within the 64 GB self-budget.
H100 80GB HBM3, driver 580.95, harness 0.2.0, gVisor.
reading:
  * Correct on all 9 workloads; graphs evidently captured (TPOT 9 ms at b1 is
    not achievable with ~1800 eager launches/step). Engine stderr is hidden on
    official runs; a public run of the same submission is queued for the log.
  * TPOT 9.1 ms at b1 = 0.88 TB/s achieved. Floor is 2.4 ms. We are
    launch/fusion-bound (Tier 2/3), exactly as §4 predicts for an unfused
    torch step. Tier 2 kernels are the next lever: expect 2-3x on TPOT.
  * TTFT is the gate closest to failing: 1.05x on b4x2048 and b16x512 (limit
    1.10). Prefill overhead vs HF: repeat_interleave K/V copies per layer
    (32-head materialisation), cache writes, fused-GEMM shapes. Also the first
    decode step is launched *before* the first yield. Fix: yield token 0
    before launching step 1 (costs one idle gap, ~0 TPOT), try
    PREFILL_ENABLE_GQA=1 (no K/V copy) and measure TTFT on b4x2048.
  * TPOT ratios 0.35-0.48x leave huge headroom; spread <1%.
next: (1) TTFT hygiene above; (2) verify.py --kernels on GPU is impossible from
here -> enable TRITON_RMSNORM alone in one push and read the result (a wrong
kernel fails correctness; a broken launch fails init); (3) then TRITON_ROPE,
TRITON_ATTN_DECODE.

### 2026-09-19  [official run d12c4e80] commit 07d4783 (same engine as 0fb3a60)   NOISE SAMPLE
harness: first end-to-end `./autoresearch.sh` run (CPU verify -> lint -> push main -> wait).
score: 399.2 vs 405.0 for the identical engine -> run-to-run noise ~1.5 % on score.
public: b1 102.4 (was 104.4), b4x2048 202.5 (206.2), b16 1319 (1338): -1..-2 %.
native reference moved more than we did: b16 total 3428 ms vs 3931 ms (-13 %),
so the TPOT *ratio* at b16 read 0.42x vs 0.36x with our TPOT within 2 %.
=> gate ratios carry ~10-15 % noise from the native side; keep TTFT <= 1.0x
   by design, not by margin. Deltas < 2 % on score are noise.
timing: leased 08:13:35, measuring 08:14:06, done 08:23:14 -> 9.1 min on the runner.
