# AGENTS.md — Qwen3-4B decode engine for the Dryft H100 benchmark

You are a coding/research agent working on `engine/`. Your job: make
`Engine.generate` produce the same greedy tokens as the baseline, faster,
on one NVIDIA H100, within the rules below. Read this file completely
before touching code. When this file and your intuition disagree, this
file wins; when this file and the official Dryft guide disagree, the
guide wins — and you update this file.

Table of contents
 0. Ten rules that are never broken
 1. What is being scored
 2. Interface contract
 3. Model facts (assert at load)
 4. Performance model
 5. Repository layout
 6. Workflow loop
 7. Verification protocol
 8. Benchmark protocol
 9. Engineering rules
10. Technique catalog (ordered)
11. Latency & stability gates
12. Memory plan
13. Warmup plan
14. Failure modes & debugging
15. Experiment ledger format
16. Questions for organizers
17. Don'ts
Appendix A. Research background (why the plan looks like this)
Appendix B. Reading list

---------------------------------------------------------------------------
## 0. Ten rules that are never broken

1. Output must be the baseline's greedy token at every position. The
   checker tolerates ≤2 logits of drift for numerical noise; never
   deliberately exploit that tolerance. No quantization, no approximate
   math, no "close enough" attention, no lossy speculation.
2. `generate` yields exactly `max_new_tokens` lists of `len(input_ids)`
   ints each, never stops at EOS, never returns early, never raises.
3. Only Python/Triton source in `engine/`. No weights, no `.so`, no
   `.cubin`, no pickles, no credentials, no network, no `pip`, no
   downloading. Weights come only from `model_path`.
4. Runtime is fixed: Python 3.11, CUDA 12.4, torch 2.5.1, triton 3.1.0,
   transformers 4.51.3, safetensors 0.5.3, tokenizers 0.21.1. Import
   nothing else (no flash_attn, vllm, xformers, flashinfer, apex,
   cutlass). Stdlib is fine. Prefer torch over numpy.
5. Everything expensive happens in `__init__` (≤300 s budget): weight
   load, buffer allocation, Triton compiles, CUDA-graph capture, cuBLAS
   warmup. `generate` does zero allocation, zero compilation, zero
   capture.
6. No CPU↔GPU sync inside the decode loop except the one required to
   yield a token list, and that one is pipelined (Section 9.3).
7. Every change passes `python agent/verify.py --all` (correctness) and
   `python agent/bench.py --gate` (latency/stability) before commit to
   `main`. `./bin/dryft validate engine` must also pass.
8. `main` = the connected branch: every push to it starts an **official,
   ranked** run (the platform offers no public mode this round and no
   local H100 exists, so the official run IS the benchmark). A worse run
   never lowers the team's best score; the only cost is a 10–15 min GPU
   slot. Therefore: `./autoresearch.sh` is the one way to push — it
   gates on CPU verify + platform lint first — and each run must answer
   one deliberate, batched question (e.g. "all Tier-2 kernels on"),
   bisecting only on failure. Never push `main` by hand mid-edit.
9. Peak GPU memory < 64 GB (self-imposed; hard limit is 72 GB = 90 % of
   80 GB).
10. Log every experiment in `agent/experiments.md` (Section 15), even
    failures. Especially failures.

---------------------------------------------------------------------------
## 1. What is being scored

- Score per workload = `batch × output_tokens / median_wall_seconds`
  over 5 samples, **including prefill**. Leaderboard = geometric mean
  over the hidden workloads (100, 200, 400 tok/s → 200). The published
  challenge definition (`./bin/dryft challenges`) lists **nine**
  workloads: 3 public + 6 hidden. A 2× gain
  anywhere is worth the same; don't specialize for one batch size.
- Gates (all must pass, per workload, else the workload fails):
  TTFT ≤ 1.10× baseline; TPOT ≤ 1.10× baseline; timing spread across
  the 5 samples ≤ 25 %; peak memory ≤ 90 % of GPU; every token correct
  (baseline greedy choice or within 2 logits; they replay OUR tokens
  through the baseline so a near-tie never cascades; one failing
  position fails the workload).
- Public workloads (never ranked, but they shape your buckets):
  `b1 × 512in × 32out`, `b4 × 2048in × 32out`, `b16 × 512in × 128out`.
  Hidden workloads are unknown; assume batch ∈ {1..32}, input ∈
  {128..4096}, output ∈ {16..512}. Engine must be *correct* for
  anything and *fast* for the bucketed range.
- One fresh `Engine` per workload. Load/warmup untimed but budgeted:
  300 s to load + warm up, 300 s per sample. The clock runs outside
  your process.
- A pre-GPU lint refuses archives that don't parse, don't export
  `Engine` with the two methods, or import modules the runtime lacks.

Implications you must internalize:
- Geometric mean + 32-token outputs ⇒ prefill is a large share of two
  of the public workloads. Prefill speed matters, not just decode.
- TTFT gate ⇒ our prefill may not be slower than 1.1× the baseline's.
  Baseline prefill is already cuBLAS + SDPA; do not add synchronous
  work (n-gram indexing, allocation, graph capture) before the first
  yield.
- Spread gate ⇒ data-dependent speedups (speculation) that vary a lot
  across the 5 samples can FAIL the whole workload even while being
  faster. Every speculative feature ships with a variance guard
  (Section 10, item 5.7).

---------------------------------------------------------------------------
## 2. Interface contract (do not change)

```python
class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield one list of token ids per step, one id per sequence,
        exactly max_new_tokens times. Greedy; do not stop at end-of-sequence."""
```
- `input_ids` are token ids (no tokenizer needed; never import one).
- Assume equal prompt lengths within a batch (public workloads are
  "fixed batches"), but handle unequal lengths correctly via
  left-alignment + per-sequence lengths (Section 9.1). Padded positions
  must be invisible to attention.
- `max_new_tokens` may be 1.
- Yield order = step order; element i of each yielded list belongs to
  sequence i of `input_ids`.

---------------------------------------------------------------------------
## 3. Model facts (assert these at load; fail loudly if wrong)

Checkpoint: `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, BF16 weights.

```
num_hidden_layers      36        hidden_size          2560
num_attention_heads    32        num_key_value_heads  8   (GQA group = 4)
head_dim               128       (explicit in config; NOT hidden/heads = 80)
intermediate_size      9728      hidden_act           silu (SwiGLU MLP)
vocab_size             151936    tie_word_embeddings  true  (lm_head = embed)
rms_norm_eps           1e-6      rope_theta           5_000_000
rope_scaling           null      max_position         262144
attention_bias         false     q_norm / k_norm      RMSNorm over head_dim,
                                                       applied BEFORE RoPE
```
Read `config.json` at `model_path` and assert every value above.

Weight names follow HF `Qwen3ForCausalLM`:
`model.embed_tokens.weight`,
`model.layers.{i}.input_layernorm.weight`,
`model.layers.{i}.post_attention_layernorm.weight`,
`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
`model.layers.{i}.self_attn.{q_norm,k_norm}.weight`,
`model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
`model.norm.weight`. `lm_head.weight` may be absent (tied) — use embed.
Shapes: q_proj [4096, 2560], k_proj/v_proj [1024, 2560], o_proj
[2560, 4096], gate/up [9728, 2560], down [2560, 9728], q_norm/k_norm
[128].

Reference semantics — match `transformers/models/qwen3/modeling_qwen3.py`
in 4.51.3 exactly; read it once, fully:
- RMSNorm: cast to fp32, `x * rsqrt(mean(x²) + eps)`, cast back to
  input dtype, THEN multiply by weight (`weight * hidden.to(dtype)`).
- Attention: qkv projections → reshape to heads → q_norm/k_norm per
  head (fp32 inside) → RoPE → attention with scale 1/sqrt(128) → o_proj.
- RoPE: rotate-half convention (`cat(-x2, x1)`), `inv_freq =
  theta^(-2i/d)`, position ids 0..L-1, cos/sin computed in fp32 then
  cast to bf16 before multiply (HF: `cos.to(x.dtype)`). Mirror this
  order or your logits drift more than necessary.
- Softmax in fp32 (SDPA does this internally).
- MLP: `down(silu(gate(x)) * up(x))`.
- Residual stream is bf16 in HF. Keep it bf16; accumulate in fp32
  inside kernels; round once at store.
- Final: `model.norm` → lm_head on last position only → argmax
  (ties: torch.argmax returns the first max index; match that).

Byte budget per decode step (bf16): attention 26.2 M params/layer, MLP
74.7 M/layer ⇒ 100.9 M/layer × 36 = 3.63 B + lm_head 0.389 B ≈ 4.02 B
params ≈ **8.04 GB read per step**. H100 SXM HBM3 ≈ 3.35 TB/s ⇒
**≈ 2.4 ms/step floor**; realistic target 2.9–3.2 ms (75–85 % of BW).
(If the machine is an H100 PCIe at 2.0 TB/s the floor is ~4 ms; check
`torch.cuda.get_device_name()` and memory clock in bench output.)
KV cache: 36 × 2 × 8 × 128 × 2 B = **147 KB/token**.

---------------------------------------------------------------------------
## 4. Performance model — know where the time goes before optimizing

| Workload      | Ideal decode | Ideal prefill | Ideal total → tok/s | Bottleneck               |
|---------------|-------------:|--------------:|--------------------:|--------------------------|
| b1 ×512 ×32   | 31 × 2.6 ms  | ~8 ms         | ~90 ms → ~355       | launch overhead, then BW |
| b4 ×2048 ×32  | 31 × 2.8 ms  | ~110 ms       | ~200 ms → ~640      | prefill GEMM + attention |
| b16 ×512 ×128 | 127 × 3.3 ms | ~35 ms        | ~455 ms → ~4500     | BW, KV attention reads   |

(First output token comes from prefill, so decode steps = out − 1.)
Achieved bandwidth = 8.04 GB / step_time. Print it in every bench run.
If step_time > 4 ms at b ≤ 16 you are launch/fusion-bound, not BW-bound.
An HF-`generate`-style baseline typically runs 30–50 ms/step with
500+ kernel launches per step; the first ~10× is pure overhead removal.

Decode at M = batch ≤ 32 is memory-bound for every GEMM (H100 ridge
≈ 300 FLOP/byte; even M ≈ 100 during speculative verify is below it).
Prefill at M = B×L ≥ 512 is compute-bound: use cuBLAS + flash SDPA.

### 4.1 Measured (official run d2b0b294, commit 0fb3a60, 2026-09-19)

H100 80GB HBM3 (SXM, 3.35 TB/s), gVisor sandbox, harness 0.2.0. Torch
decode path with CUDA graphs + pipelined yield; all `TRITON_*` flags off.
**Score 405** (hidden geomean, native = 100). Leaders: 633, 624, then ~300.

| Workload      | ours tok/s | native | total ms | TTFT ms (×native) | TPOT ms (×native) | achieved BW |
|---------------|-----------:|-------:|---------:|------------------:|------------------:|------------:|
| b1 ×512 ×32   |    104     |   40   |   306    |  23 (0.81×)       |  9.15 (0.36×)     |  0.88 TB/s  |
| b4 ×2048 ×32  |    206     |  123   |   621    | 214 (1.06×)       | 13.17 (0.49×)     |  0.61 TB/s  |
| b16 ×512 ×128 |   1338     |  521   |  1531    | 202 (1.05×)       | 10.45 (0.36×)     |  0.91 TB/s  |

Spread < 1 % everywhere; peak 49.2 GiB. All 9 workloads correct.

What the numbers say:
- **TPOT is 3–4× off the floor** (2.4 ms at b1; ~2.9 ms at b16 incl. KV
  reads). At ~55 torch kernels/layer ≈ 2000 kernels/step, even inside a
  graph each ~3–4 µs kernel costs more than the bytes it moves. This is
  the launch/fusion-bound regime of §4: Tier 2 + 3 fusion is worth
  ~2–2.5× on TPOT and, since hidden workloads are decode-heavy (405 ≫
  the public geomean speedup of 2.2×), roughly the same on the score.
- **b4 ×2048 decode is 4 ms/step slower than b1** with the same weights:
  that is the masked-SDPA decode path over the padded `[.., :3072]`
  bucket (mask build + math/efficient backend at q_len 1 + strided KV
  reads). The split-KV Triton kernel reads `seq_lens + 1` keys and
  removes the length-bucket dependence entirely.
- **TTFT is the gate nearest failure (1.06× of 1.10×)** on both
  prefill-heavy shapes. Our prefill ≈ HF's because both are unfused:
  per layer, RMSNorm ×2 (~8 passes over `[T, 2560]`), q/k-norm + RoPE
  (~14 passes over `[T, 4096]`), `silu(g)*u` (3 passes over
  `[T, 9728]`) ≈ 3–4 GB of elementwise traffic per layer at T = 8192,
  ≈ 35–40 ms of the 214 ms. Running the Tier-2 kernels in prefill too
  should give TTFT ≈ 0.85–0.9× and buys real headroom under the gate.
- Prefill GEMMs at T = 8192 are ~50 % of peak tensor throughput; that is
  cuBLAS territory and not worth touching before Tier 5.

Targets after Tier 2 + 3 (same workloads): TPOT ≈ 4 ms → b1 ≈ 220 tok/s,
b4 ×2048 ≈ 400, b16 ≈ 3000; projected score ≈ 800–900, i.e. clear of the
current 633. Speculation (Tier 5) is then the moat: it is the only
technique that beats the 2.4 ms/step floor.

---------------------------------------------------------------------------
## 5. Repository layout

```
engine/                     # THE ONLY THING SUBMITTED
  engine.py                 # Engine: load → warmup → generate (thin)      [done]
  dryft_qwen3/              # everything else lives in a uniquely named package: the
                            # archive root shares sys.path with the harness, so bare
                            # names like config/model/cache could collide
    config.py               # model constants + asserts, buckets, FLAGS       [done]
    weights.py              # safetensors load, fusion (qkv, gate_up)         [done]
    model.py                # prefill(), decode_step(); verify_step() later   [done]
    cache.py                # static KV cache, per-seq lengths, rope tables   [done]
    graphs.py               # CUDA-graph capture/replay per bucket            [done, unverified]
    kernels/
      rmsnorm.py            # fused residual + rmsnorm                        [written, OFF]
      rope_qknorm.py        # qk-norm + rope + kv-cache write                 [written, OFF]
      attn_decode.py        # split-KV GQA decode attention (+ reduce)        [written, OFF]
      attn_verify.py        # multi-query (chain/tree) verify attention
      gemm_splitk.py        # skinny GEMMs, silu*up epilogue
      lm_head.py            # (optional) fused last-token logits + argmax
    spec/
      ngram.py              # prompt-lookup drafting
      recycle.py            # token-recycling adjacency drafting
      tree.py               # draft tree → packed tokens, positions, mask
      accept.py             # greedy acceptance, ragged advance
    tuned/                  # hard-coded kernel configs per shape (JSON), generated offline
agent/                      # NEVER SUBMITTED
  verify.py                 # correctness vs HF reference; logit-margin report  [done]
  bench.py                  # timing harness mirroring official scoring + gates [done]
  baseline_engine.py        # the unmodified starter, timed by bench --baseline  [done]
  profiler.py               # torch.profiler per-kernel table ("profile" shadows stdlib) [done]
  prompts.py                # prompt-set builder (synthetic + tokenized text)    [done]
  tune.py                   # offline autotune → engine/dryft_qwen3/tuned/*.json
  prompts/                  # dumped text prompt sets (Section 7.2)
  experiments.md            # ledger                                            [done]
  loop.py                   # optional automated research loop
AGENTS.md
CLAUDE.md                   # "@AGENTS.md"
```
Keep `engine/` free of anything that isn't imported at runtime.
`dryft validate` lints every file in the folder. JSON in `tuned/` is fine
(it's source, not a binary). Inside the package use relative imports;
`engine.py` imports `from dryft_qwen3 import config` etc.

---------------------------------------------------------------------------
## 6. Workflow — the loop

The bench is the platform (no local GPU; official runs only). Each
iteration costs one 10–15 min run, so each run answers one batched
question, and everything that can fail cheaply fails locally first.

Each iteration:
1. Pick the highest-ranked unblocked item from Section 10 (or the
   ledger's "next" list). Batch what can be batched into one run: a
   *set* of flags whose failure modes are distinguishable from the
   report (e.g. wrong tokens vs slower TPOT vs init crash).
2. Implement behind a flag in `engine/dryft_qwen3/config.py:FLAGS` so a
   regression is one line to revert. Kernels ship with a torch
   reference and a `selftest()`; `Engine.__init__` must run every
   enabled kernel's selftest on the GPU and **fall back to the torch
   path** if one fails — a wrong kernel must never fail a workload,
   because the first time any Triton kernel executes on a GPU *is* an
   official run (T2.0 below wires this; nothing else ships before it).
3. Local gates (seconds): `python agent/verify.py --tiny --all` (CPU
   exact-match vs HF on a random tiny Qwen3 — Triton paths are
   *not* exercised here), `./bin/dryft validate engine`.
4. `./autoresearch.sh` → snapshots the working tree onto `origin/main`,
   waits, prints `METRIC score=…` plus public tok/s, TTFT/TPOT ratios,
   spread, peak memory (`agent/official_run.py`). Engine stderr is
   hidden on official runs: any diagnostic you need must be encoded in
   *timing* (e.g. a fallback makes TPOT identical to the previous run).
5. Keep iff: score improves AND TTFT/TPOT ratios stay ≤ 1.0 with ≥ 5 %
   headroom under the 1.10 gate AND spread < 10 % AND peak < 64 GB.
   Otherwise flip the flag off, record, move on. Two consecutive runs
   of the same engine differ by < 2 % (noise sample in the ledger);
   treat smaller deltas as noise.
6. Record every run in `agent/experiments.md` (Section 15) with the run
   id. Commit to the working branch with `[<area>] <what> : <score
   before→after>`.

Commands:
```
./autoresearch.sh                                 # gates + ONE official run (the bench)
python agent/official_run.py --run-id <RUN_ID>    # re-print a finished run's metrics
python agent/verify.py --tiny --all               # CPU exact-match gate on a random tiny Qwen3
python agent/verify.py --all --model $CKPT        # same gate on the real checkpoint (needs a GPU)
python agent/verify.py --kernels                  # per-kernel unit tests (needs CUDA + triton)
python agent/bench.py --public --extra            # local timing (needs a GPU)
python agent/profiler.py --workload b1_512_32 --steps 8 --eager  # kernel table (needs a GPU)
./bin/dryft validate engine                       # platform lint, locally
./bin/dryft runs | ./bin/dryft result <RUN_ID>    # run history / full report
```
Platform facts (2026-09-19): API at `https://htn.dryft.ai` (`.env` holds
`DRYFT_API` + `DRYFT_TOKEN`; gitignored). CLI upload (`dryft submit`)
returns 405 and `POST …/runs` rejects `mode != official` (422):
submissions exist only through the connected repo `hari495/dryft`
(branch `main`, engine folder `engine`), and every push is an official
ranked run. A newer push cancels a still-queued older run. Runs have a
15 min limit once a runner picks them up (9 workloads × (our init +
native load + 2×5 samples) fit in ~10 min today; init must stay
< 40 s/workload). Round ends 2026-09-20T12:00Z.
Local dev environment: `uv venv --python 3.11 .venv && uv pip install
--python .venv/bin/python torch==2.5.1 transformers==4.51.3
safetensors==0.5.3 tokenizers==0.21.1 "numpy<2.2"`; run the commands
above with `.venv/bin/python`. `--flag NAME=0|1` on verify/bench
overrides `FLAGS` for A/B runs.

---------------------------------------------------------------------------
## 7. Verification protocol (`agent/verify.py`)

7.1 Reference: `transformers.Qwen3ForCausalLM.from_pretrained(model_path,
torch_dtype=torch.bfloat16, attn_implementation=<same as starter>)`,
greedy: `do_sample=False`, `num_beams=1`, `max_new_tokens=N`,
`min_new_tokens=N`, no EOS stopping (`eos_token_id=None` or a
`StoppingCriteria` that never fires). Read the starter engine to learn
exactly which attention implementation and generate settings the
baseline uses, and match them. Record the answer in Section 16.

7.2 Prompt sets (token ids; store under `agent/prompts/` as JSON):
- `natural`: chat-templated real text, 128–2048 tokens, EN/ZH/code/math.
- `copyheavy`: summarization / extraction / code-edit prompts (where
  prompt-lookup wins).
- `random`: uniformly random token ids (flat logits, many near-ties,
  worst case for the 2-logit rule; prompt-lookup useless). Hidden
  workloads may be synthetic — this set is not optional.
- `eos_early`: prompts whose answer ends in <10 tokens; verify tokens
  after EOS still match.
- `ragged`: batch with unequal prompt lengths.
- `long`: 4096-token inputs, 512 outputs, b ∈ {1, 8}.
- `shapes`: every bucket boundary ±1 (Section 9.2), plus
  `max_new_tokens ∈ {1, 2}`.

7.3 Checks, per prompt set:
- Exact token match vs reference for every sequence and position.
- Replay check mirroring the official rule: teacher-force OUR tokens
  through the reference, report per-position
  `logit[ref_argmax] − logit[our_token]` (0 when equal). Assert max ≤
  1.0 (self-imposed headroom under the official 2.0). Print the
  histogram; growth of the tail is an early warning of numerics drift.
- Run each case twice in one process → outputs identical (catches
  uninitialized memory, graph-replay staleness, races in split-K
  atomics).
- Run a b1 case, then a b16 case, then the same b1 case again →
  identical (catches cross-bucket state leaks).

7.4 When a mismatch appears: bisect by flag, then by layer (dump hidden
states from both models at layer i and report max abs / rel diff).
Typical culprits: RMSNorm order (weight before/after cast), RoPE
convention or theta, q/k-norm after instead of before RoPE, cache
position off-by-one, GQA head mapping (q head h uses kv head h // 4),
attention scale, bf16 softmax, lm_head reading a stale or un-normed
input, bf16 accumulation in a Triton reduction (always accumulate
fp32), stale rows from a previous bucket's dummy sequences.

---------------------------------------------------------------------------
## 8. Benchmark protocol (`agent/bench.py`)

- Mirror the official formula: fresh `Engine` per workload; one warmup
  call (untimed; confirm with organizers that the harness does the
  same), then 5 timed samples; wall-clock around full generator
  consumption (`for toks in engine.generate(...)`) with
  `torch.cuda.synchronize()` before start and after the last yield;
  median → tok/s; report TTFT (time to first yield), TPOT
  ((total − TTFT)/(out − 1)), spread ((max − min)/median), peak mem
  (`max_memory_allocated` and `max_memory_reserved`), bytes/step and
  achieved BW.
- Measure the baseline (starter engine, unmodified) once per shape and
  cache it in `agent/baseline.json`: gates are relative to it. Print
  `ours/baseline` for TTFT and TPOT on every run.
- `--extra` shapes for hidden-workload guessing: b2×1024×64,
  b8×256×256, b8×2048×64, b32×128×32, b1×4096×128, b16×1024×256.
- Lock clocks if permitted (`nvidia-smi -lgc`), otherwise run a 5 s GPU
  burn before timing so the 25 % spread isn't clock ramp. Note the
  official harness probably does neither; keep spread low without it.
- Disable Python GC during the timed region (`gc.freeze()` +
  `gc.disable()` in `__init__`; `gc.collect()` only between samples).
  Try `torch.set_num_threads(1)` if it reduces jitter.

Profiling (`agent/profiler.py`): run 8 decode steps under
`torch.profiler` (CUDA activities) and print a table: kernel name |
count/step | µs/step | % step | bytes moved (lookup by kernel role) |
achieved GB/s. Gaps between kernels = launch overhead; count them.
Targets: ≤ 6 kernels per layer, ≤ 230 kernels per step, CPU time per
step < 50 µs when graphs are on.

---------------------------------------------------------------------------
## 9. Engineering rules

### 9.1 Static shapes & state
- Static KV cache allocated in `__init__`:
  `K, V: [layers, B_max, kv_heads, L_max, head_dim] bf16` (head-major per
  sequence: `cache[l, :B, :, :L]` is the `[B, H, L, D]` view SDPA wants,
  so the torch decode path reads the cache with zero copies; the Triton
  kernels take strides and don't care). Default `B_max=32, L_max=8192` →
  2 × 36 × 32 × 8192 × 2 KB ≈ 37.7 GB. If memory is tight for a chosen
  bucket set, prefer a paged layout
  (`[layers, num_blocks, block=16, kv_heads, hd]` + block-table tensor)
  so any (B, L) fits one pool without recapture. Paged is v2; static
  rows are v1. (Implemented: `engine/dryft_qwen3/cache.py`.)
- Per-sequence `seq_lens: int64[B_max]` on device (int64 so it doubles as a torch index tensor). Every kernel reads
  lengths from this tensor — never from Python ints — so one CUDA graph
  serves all positions.
- Fixed device buffers for: current input ids `[B_max, Kmax+1]`,
  positions `[B_max, Kmax+1]`, hidden/residual `[B_max×(Kmax+1), 2560]`,
  normed activation, qkv `[.., 6144]`, attention output `[.., 4096]`,
  MLP intermediate `[.., 9728]` (or `[.., 19456]` before the silu*mul
  fusion), logits `[B_max, 151936]` fp32 (or skip if fused argmax),
  split-KV scratch `[B, heads, splits, hd+2]` fp32, next-token ids
  `[B_max]`, pinned CPU output ring `[depth, B_max]`.
- Fallback path: if B > B_max or L_in + out > L_max, run the eager
  (non-graph) path with a dynamically allocated cache. Correct, slower,
  never crashes. Test it.

### 9.2 Buckets & graphs
- Decode graphs per (batch_bucket, len_bucket, q_len): batch_bucket ∈
  {1, 2, 4, 8, 16, 32}; len_bucket ∈ {256, 512, 768, 1024, 1536, 2048,
  3072, 4096, 6144, 8192} = smallest ≥ prompt + max_new_tokens (the torch
  SDPA decode path attends over a static `[.., :Lb]` slice masked by
  `seq_lens`; the Triton split-KV kernel reads only `seq_lens + 1` keys,
  so once it is on, the length bucket only bounds the mask); q_len ∈ {1}
  initially, {1, 1+k for k in spec tiers} later. Pad batch rows up to the
  bucket with dummy sequences (length 1, token 0); their cost is ~0 at
  memory-bound M. Never let dummy rows' outputs leak into yields; never
  let dummy rows write into live rows' cache. (Implemented:
  `engine/dryft_qwen3/graphs.py`; a graph that fails post-capture validation is
  dropped and its bucket runs eager.)
- Graph capture recipe (in `__init__`): warm up the exact callable 3×
  on a side stream (compiles Triton, initializes cuBLAS handles and
  workspace, runs lazy init) → `torch.cuda.synchronize()` →
  `with torch.cuda.graph(g, pool=shared_pool): out = step(...)`.
  All graphs share one memory pool. Verify after capture: replay once,
  compare against eager on the same inputs.
- Inside captured code: no Python-int arguments that change between
  steps (Triton bakes scalars as constexpr or as kernel args at
  capture; pass tensors), no `.item()`, no `print(tensor)`, no
  allocation, no `torch.cuda.synchronize`, no autotune, no side streams
  unless joined with `wait_stream` both ways.
- Prefill is NOT graph-captured in v1 (variable L). Warm up cuBLAS for
  prefill M values {256, 512, 1024, 2048, 4096, 8192, 16384} in
  `__init__` so the first real prefill doesn't pay heuristic selection.

### 9.3 The decode loop (no CPU on the critical path)
```
tokens[0] = argmax(prefill(...))          # first output, from prefill
launch decode(0)                          # consumes tokens[0] on device
for t in range(1, out):
    if t + 1 < out: launch decode(t)      # graph replay, async
    event[t-1].synchronize()              # wait only for step t-1
    yield pinned_out[t-1].tolist()        # D2H already done async
yield the final step similarly
```
Always keep one step in flight; copy next-token ids to a pinned CPU
buffer with `non_blocking=True` right after each step and record an
event. The GPU never waits for Python. Per-step yield cadence is
preserved (TPOT measurement stays honest). With speculation, yields
come from a per-sequence token buffer (Section 10, item 5.6). Use a
ring of ≥ 2 pinned slots so the in-flight copy never overwrites a slot
Python hasn't read.

### 9.4 Triton kernel rules
- Accumulate in fp32; store bf16. `tl.dot` needs M, N, K ≥ 16 — pad or
  use broadcast-multiply + `tl.sum` for tiny M (memory-bound anyway).
- No `@triton.autotune` at runtime. Tune offline (`agent/tune.py`),
  store configs in `engine/tuned/*.json` keyed by (kernel, M, N, K or
  L), load at import. Unknown shape → nearest key.
- `num_stages` 3–4, `num_warps` 4–8 for bandwidth kernels; use
  `tl.multiple_of` / `tl.max_contiguous` hints on pointers with known
  alignment (all dims here are multiples of 128).
- Every kernel has a pure-torch reference in the same file and a unit
  test (`agent/verify.py --kernels`) at bucket shapes and at odd
  lengths (e.g. L = 513, 2047) for masking correctness.
- Split-K with fp32 atomics is allowed (non-deterministic sums are
  within tolerance) but prefer a deterministic two-pass reduce when
  cost is equal; determinism makes debugging tractable.
- Compile budget: each distinct (kernel, constexpr set) costs 1–4 s in
  `__init__`. Keep total distinct compiles ≤ 60. Print compile count
  and time.
- Debug with `TRITON_INTERPRET=1` on tiny shapes before touching the
  GPU.

### 9.5 Prefill rules
- cuBLAS for all GEMMs (M = B×L large). Fuse QKV into one GEMM
  ([2560 → 4096+1024+1024]) and gate+up into one ([2560 → 19456]).
- Attention: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
  in bf16 with K/V expanded to 32 heads via `repeat_interleave`, or
  `enable_gqa=True` (in torch 2.5.1 that flag routes only to the flash
  backend — benchmark both; flash should win). No explicit mask for
  equal-length batches; for ragged batches use per-sequence calls or a
  varlen Triton kernel (v2). Never let SDPA fall to the math backend
  (materializes B×H×L×L scores) — confirm via profiler.
- Write K/V into the cache with one fused copy per layer (or inside
  the RoPE kernel).
- lm_head on the LAST position only. Never compute logits for all
  prompt positions (8192 × 151936 × 4 B = 5 GB and ~40 ms wasted).
- Chunked prefill (e.g. 2048-token chunks) only if memory requires; it
  does not speed anything up here.
- Overlap: build any speculation index (n-gram table) on the CPU in a
  thread while the GPU runs prefill, or on the GPU after the first
  yield. Nothing synchronous before TTFT.

---------------------------------------------------------------------------
## 10. Technique catalog (ordered; do them in this order unless blocked)

Legend — Gain: expected effect on geomean vs previous tier. Each item
lists its acceptance test.

### Status (2026-09-19) and the next runs, in order

Done: Tier 0 (tooling), Tier 1 (own forward, static cache, graphs,
pipelined yield, fused prefill GEMMs) → score 405. Written but never
executed on a GPU: `kernels/rmsnorm.py`, `kernels/rope_qknorm.py`,
`kernels/attn_decode.py` (flags off). Not started: Tier 4, Tier 5.

Run plan (one official run each; ~12 min; bisect only on failure):

| # | Batch pushed | Reads from the report | Expected |
|---|---|---|---|
| R1 | **T2.0** selftest-or-fallback in `__init__` for every Triton kernel; `TRITON_RMSNORM=1` (decode + prefill) | correct; TPOT ↓ ≈ 1 ms/step at b1 (−14 kernels/layer); TTFT b4×2048 ↓ ≈ 10 ms. If TPOT unchanged → kernel fell back → bisect its selftest tolerance | b1 ≈ 118, score ≈ 450 |
| R2 | `TRITON_ROPE=1` (decode + prefill; writes K/V in the kernel) | TPOT ↓ ≈ 1.5 ms (−20 kernels/layer); TTFT ↓ ≈ 15 ms | score ≈ 520 |
| R3 | `TRITON_ATTN_DECODE=1` (split-KV, reads `seq_lens+1` keys) | b4×2048 TPOT 13.2 → ≈ 9 ms; all TPOT ↓ (−4 kernels/layer, no mask build) | score ≈ 580 |
| R4 | **T2.3** `silu*mul` fused kernel (prefill + decode) + **T2.4** o/down GEMM epilogue-adds via Triton split-K GEMM for M ≤ 32 (`kernels/gemm_splitk.py`), or plain torch `addmm` into the residual if the GEMM kernel isn't ready | kernels/layer → ~8; TPOT → ≈ 5 ms | score ≈ 700 |
| R5 | **T4.1** cuBLAS-vs-split-K dispatch for the four decode GEMMs at M ∈ {1,2,4,8,16,32}: measure *inside* `__init__` on the real GPU (torch.cuda.Event timing, < 2 s total) and pick per shape — this is the only autotune we can do without a local GPU; it happens before the timed region and is allowed (rule 5) | TPOT → ≈ 4 ms at b1 | score ≈ 800 |
| R6 | **T5.1** PLD chain drafting, k = 3, with 5.7's guard; needs q_len = 1+k graphs and `attn_verify` (chain = causal over `[n, n]`, so the split-KV kernel with q_len > 1 suffices) | tokens/step ↑ on copy-heavy hidden prompts; TPOT ratio must stay ≤ 1.0; spread ≤ 10 % | +10–40 % where prompts repeat |
| R7 | **T5.2** Token Recycling tree (n ≈ 16–24 nodes), then **5.3** hybrid | uniform ≈ 1.3–1.6× at b ≤ 8; less at b16–32 | +20–40 % |

Research notes behind the ordering (2025 literature; details in
Appendix A.4/A.7):
- Fusion first: at ~2000 kernels/step the GPU is idle most of the step.
  Every kernel removed at M = 1 is worth ≈ 3–4 µs × 36 layers; the
  Tier-2/3 set removes ~45 kernels/layer ≈ 5–6 ms/step. No speculation
  scheme pays that well.
- Skinny GEMMs: published Triton split-K results on H100 at M ≤ 16 are
  1.2–1.9× over data-parallel Triton and up to ~1.7–1.9× over cuBLAS
  *FP8/FP16 GEMM*; plain cuBLAS bf16 GEMV at M = 1 is already ~70 % of
  peak, so expect ≤ 1.2× on `down_proj` (K = 9728) and `o_proj`, and
  measure before shipping (R5). GEMV-style (no `tl.dot`) wins at M = 1.
- Training-free speculation, Spec-Bench numbers (Vicuna/Llama, greedy):
  PLD mean accepted tokens ≈ 1.7–1.8 (1.5–1.7× speedup) on general
  text, up to 3.2 on code/edits; Token Recycling ≈ 2.7–2.8 (≈ 2×,
  uniform across tasks, < 2 MB state); SuffixDecoding ≈ 6–8 only on
  agentic traces with long repeats. Hidden prompts are unknown, so TR's
  uniformity matters more than PLD's peak; PLD first because it is ~40
  lines and needs only chain verify.
- Verify cost: at M = B·(1+k) ≤ ~128 every GEMM is still memory-bound,
  so a verify step ≈ 1.05–1.3× a decode step; attention grows with
  (1+k) query rows but reads the same K/V once per kv head.
- Prefill: TTFT is 1.05–1.06× native and the gate is 1.10. R1–R4 also
  run in prefill and should bring it to ≈ 0.85×; nothing else touches
  prefill until Tier 5 is stable.

### Tier 0 — Instrument (day 0)
- Run starter unchanged; record baseline tok/s, TTFT, TPOT per public
  shape in the ledger. Profile one decode step; count kernels and gaps.
- Write `verify.py` and `bench.py` before any optimization.

### Tier 1 — Kill overhead (expected 5–10× at b1, 3–6× at b16)
1.1 Own forward pass. Load safetensors directly into flat bf16 tensors;
    pre-fuse `Wqkv` and `Wgate_up`; drop `nn.Module` overhead.
    Test: verify 100 %.
1.2 Static KV cache + per-sequence lengths (9.1). Test: ragged set.
1.3 CUDA graph per batch bucket for the decode step (9.2). Test: replay
    equals eager on 3 random inputs; bench shows ≤ ~50 µs CPU per step.
1.4 One-step-in-flight yield loop (9.3). Test: TPOT unchanged or
    better; tokens identical.
1.5 Prefill hygiene (9.5): fused GEMMs, flash SDPA, last-token logits,
    cuBLAS warmup. Test: TTFT/baseline ≤ 0.9 on b4×2048.
1.6 Argmax path: bf16 logits → fp32 argmax (or fp32 GEMM output for the
    lm_head at small M; cost 151936 × B × 4 B — negligible). Test: logit
    margin histogram unchanged.
Exit criterion: step time ≤ 5 ms at b1 (≥ 1.6 TB/s achieved).

### Tier 2 — Fusion in Triton (expected 1.3–1.6×)
Target ≤ 6 kernels/layer.
2.1 `rmsnorm_residual`: `res = res + x; y = rmsnorm(res) * w` in one
    pass, fp32 math, outputs both (bf16). Removes 2 kernels/layer.
2.2 `qknorm_rope_kvwrite`: input qkv `[T, 6144]`; per-head RMSNorm on q
    (32 heads) and k (8 heads) → RoPE (rotate-half; cos/sin table
    `[L_max, 64]` fp32 precomputed in `__init__`, cast to bf16 at use to
    mirror HF) → write k, v to cache at `(seq, pos)` → write q
    `[T, 32, 128]`. Removes ~5 kernels/layer.
2.3 `silu_mul` fused into the gate_up GEMM epilogue: store W_gate_up
    with rows interleaved per N-tile so a tile holds matching gate/up
    columns and writes `silu(g) * u` at half width. Requires no split-K
    on that GEMM, or split-K + separate epilogue. Benchmark both.
2.4 `o_proj + residual` and `down_proj + residual` epilogues: add into
    the bf16 residual buffer in the GEMM epilogue (fp32 accumulate,
    round once). Needs no split-K on those GEMMs or a two-pass reduce
    whose second pass does the add.
2.5 Input RMSNorm folded into the following GEMM's A-load for small M
    (each program loads its whole 2560-wide row, computes rms, scales
    on the fly). Removes the norm kernel; combine with 2.1.
Test for every kernel: unit test vs torch reference (max abs err at
bf16-ulp level), then full verify. Bench: kernels/step, achieved BW.

### Tier 3 — Attention kernels (expected 1.1–1.3×, larger at long L)
3.1 `attn_decode` split-KV GQA: grid `(B, kv_heads, S splits)`; each
    program handles the 4 query heads of one kv head × one KV segment;
    loads K tiles `[BLOCK_N, 128]`, scores for 4 (× q_len) queries,
    online softmax in fp32, accumulates `[4, 128]`; writes partial
    `(m, l, acc)` to scratch; second kernel `attn_reduce` combines S
    splits and writes bf16 `[B, 32, 128]`. Choose S so that
    B × 8 × S ≥ 2 × 132 SMs — e.g. S = 32 at b1, S = 4 at b16 — but S
    is fixed per graph; read `seq_lens` for masking; segments beyond
    length exit early. Test: odd lengths, L = 1, L = L_max; compare to
    fp32 math attention.
3.2 `attn_verify` (for speculation): same kernel with q_len = n nodes
    and an ancestor mask `[n, n]` for in-flight tokens (tree) plus
    causal over the cache. Keep the split-KV structure — a naive
    one-program-per-(seq, head) verify kernel makes verify cost grow
    linearly with prefix length and kills speculation at long L.
3.3 Prefill attention: keep flash SDPA unless profiling shows > 25 % of
    prefill in attention at 2048; then a Triton flash-forward kernel
    with GQA-aware K/V loading is next. Low priority.

### Tier 4 — GEMM tuning for skinny M (expected 1.05–1.2×)
4.1 Per-shape dispatch table: for each (M ∈ {1, 2, 4, 8, 16, 32,
    (1+k)·B}, N, K) in the model, benchmark cuBLAS (`torch.matmul`) vs
    Triton split-K (split ∈ {1, 2, 4, 8, 16}) vs GEMV-style (no
    `tl.dot`, M = 1) and record the winner in `engine/tuned/gemm.json`.
    Expect split-K to win on `down_proj` (K = 9728) and on `o_proj`
    (K = 4096) at M ≤ 4; lm_head (N = 151936) likely stays cuBLAS. Run
    once offline; ship the JSON.
4.2 Weight layout: store weights so the winning kernel reads
    contiguously (e.g. `[N, K]` row-major for `x @ W.T` with
    K-contiguous loads). Reorder at load time; costs nothing at runtime.
4.3 lm_head: try fused GEMV + running argmax per program + final tiny
    reduce (never materialize logits) at M ≤ 4. Mostly a launch-count
    win.

### Tier 5 — Exact speculative decoding (the only way past the BW floor)
All variants: drafts come from zero-weight sources; verification is one
forward with `1+k` (chain) or `n` (tree) query tokens per sequence;
acceptance is greedy-exact (accept draft_i iff draft_i == argmax at its
parent); the bonus token = argmax at the last accepted node. Output is
provably identical to greedy decoding. Speedup ≈ (1 + E[accepted]) /
(verify_step_time / decode_step_time). At M = B·(1+k) ≤ ~100 the verify
GEMMs are still memory-bound, so verify_step ≈ 1.05–1.3× a decode step;
attention cost grows with (1+k).

5.1 **Prompt-lookup (PLD) chain drafting.** Index all n-grams (n = 3..1)
    of prompt + generated tokens per sequence in a CPU dict (runs while
    the GPU executes). Draft k ∈ {3, 5, 7} continuation tokens from the
    longest matching n-gram. Cost ≈ 0 GPU. Gain 1.5–3× on copy-heavy
    prompts (summarization, extraction, code edit), ~0.95–1.0× on random
    tokens (verify overhead only). Ship with the adaptive guard (5.7).
    Needs graphs for q_len = 1+k per batch bucket.
5.2 **Token Recycling (TR).** Keep a per-sequence adjacency matrix
    `A[vocab, top_k=8]` (int32, ~5 MB); after every verify, write the
    top-8 next-token candidates for each processed position into
    `A[tok]`. Draft a tree (BFS from last token, budget n ≈ 16–48 nodes,
    greedy by rank) and verify with `attn_verify`. Cost: `topk(8)` over
    151936 logits per processed position, tree build on CPU (~100 µs),
    tree-mask attention. Reported 1.5–2× on 7B+ at b1; expect less on
    4B/H100 because verify overhead is relatively larger. Uniform
    across prompt types (good for the spread gate).
5.3 **Spine + branches (hybrid tree).** PLD chain as the spine (high
    acceptance; context-matched tokens are accepted several times more
    often than TR transition tokens), TR candidates as branches at each
    spine node. One tree, one verify. Recommended end state for b ≤ 4.
5.4 **Lookahead (Jacobi) decoding.** Only if 5.1–5.3 underdeliver and
    b = 1 dominates the hidden set; it burns FLOPs that at b ≥ 8 are no
    longer free. Low priority.
5.5 **Self-speculation by layer skipping.** Draft with a subset of the
    36 layers, verify with all. With draft cost fraction c and
    per-token acceptance α, speedup = (1 − α^{k+1})/(1 − α) / (k·c + 1).
    Published gains come from 13B–70B models where redundancy is high;
    a 4B model likely gives poor α at useful c. Measure α for
    c ∈ {0.25, 0.4} on the `natural` set before writing any kernel;
    drop if α < 0.75 at c = 0.3. Expected outcome: drop.
5.6 **Batching + raggedness.** Each sequence drafts its own k (or its
    own tree). Verify processes `[B, n]` tokens with per-sequence
    positions. Acceptance is per sequence; `seq_lens` advances raggedly.
    Write draft K/V at positions `seq_len .. seq_len+n−1`; advance
    `seq_len` only by the accepted count; rejected entries are simply
    overwritten next step (attention masks by `seq_len`, so stale
    entries are never read). Yield buffer `out_buf[B][max_new_tokens]`
    filled raggedly; yield step t when all sequences have ≥ t+1 tokens.
    Sequences that reach `max_new_tokens` stop drafting (k = 0; still
    occupy a row). Cap each sequence's draft to `remaining − 1` tokens.
    Do NOT pad all sequences to the max accepted length — that wastes
    30–50 % at b ≥ 8; per-sequence lengths in the attention kernel make
    raggedness free.
5.7 **Variance & gate guard (mandatory).** Track an EMA of accepted
    length per sequence. Speculative tiers k ∈ {0, 3, 7}; pick the
    largest k whose predicted speedup > 1.05 given the measured
    verify/decode time ratio for this batch bucket; drop to k = 0 when
    EMA acceptance < 1.0 tokens for 8 consecutive steps; re-probe with
    k = 3 every 32 steps. Each (bucket, k) has its own captured graph.
    Log acceptance stats per sample in bench; if projected per-sample
    spread from speculation exceeds 20 %, cap k lower for that workload
    class. Until organizers answer Section 16 Q1–2, assume
    TPOT = (total − TTFT)/(out − 1) and spread = (max − min)/median.

### Tier 6 — Stretch (only after Tiers 1–5 are stable and submitted)
6.1 Persistent "layer megakernel" in Triton: one launch per layer (or
    per 2 layers) using a persistent grid ≤ #SMs × occupancy and
    atomic-counter barriers between phases. High deadlock risk (all
    programs must be co-resident); validate occupancy via
    `kernel.n_regs` / `n_spills` before trusting a barrier. Gains:
    removes ~4 kernel tails per layer. Full megakernels (as in the
    Hazy Research / Mirage work) need CUDA, which we cannot ship; the
    Triton approximation captures part of the win.
6.2 KV cache layout that lets `attn_decode` read K transposed
    `[128, L]` for better coalescing of the score dot; measure.
6.3 Deterministic split-K reduce vs atomics: pick by measured time.
6.4 Prefill for b4×2048: Triton flash-forward with GQA-aware K/V loads
    if SDPA's flash path is the bottleneck (see 3.3).

### Explicitly rejected (don't spend time)
- Any quantization (weights, KV, activations), FP8, int8 — banned.
- Trained drafters (Medusa/EAGLE/MTP heads) — cannot ship weights.
- torch.compile for the decode path in torch 2.5.1: unpredictable
  compile time inside the 300 s budget, graph breaks with custom Triton
  ops, and reduce-overhead's cudagraph trees fight manual capture.
  Manual graphs + hand kernels are strictly more controllable. (Trying
  `torch.compile` for prefill elementwise fusion is allowed but must
  finish compiling in `__init__`.)
- Exploiting the "within 2 logits" tolerance to accept non-argmax
  tokens. It changes outputs; treat as banned.
- Multi-GPU, CPU offload, disaggregated prefill — one H100, one
  process.

---------------------------------------------------------------------------
## 11. Latency & stability gates — how to not fail while being fast

- TTFT: prefill only. Do: fused GEMMs, flash SDPA, last-token logits,
  warm cuBLAS. Don't: build indexes, allocate, capture, or `.item()`
  before the first yield. Measure `ours/baseline` on b4×2048 every run.
- TPOT: per-step. Speculation makes *steps* slower but *tokens*
  faster; TPOT as (total − TTFT)/(out − 1) improves. If the harness
  instead measures max inter-yield gap, chain length k must keep
  verify_step ≤ 1.1× baseline_step — with a ≥ 5× faster engine this is
  trivially met.
- Spread: causes are clock ramp, first-call effects, GC, Python
  jitter, and data-dependent speculation. Mitigate with warmup burn
  (in `__init__`), one step in flight, `gc.disable()`, and 5.7.
- Memory: report `max_memory_reserved`; the organizer likely reads
  device-level usage. Keep the CUDA caching allocator from
  fragmenting: allocate all big buffers first, in `__init__`, then
  capture graphs with a shared pool.

---------------------------------------------------------------------------
## 12. Memory plan (H100 80 GB; budget 64 GB)

| Item                                   | Size      |
|----------------------------------------|----------:|
| Weights bf16 (tied lm_head)            | 8.0 GB    |
| KV cache B_max=32 × L_max=8192         | 37.7 GB   |
| Activations/scratch (B_max × (1+7))    | < 1 GB    |
| Logits fp32 [32, 151936]               | 19 MB     |
| Prefill transient (b4×2048): qkv, mlp  | ~1.5 GB   |
| CUDA graph pools (12–18 graphs)        | ~1 GB     |
| cuBLAS workspace, Triton, CUDA context | ~1.5 GB   |
| **Total**                              | **≈ 51 GB** |

If hidden shapes demand more (e.g. b64 or L > 8192), switch to paged
KV with a fixed 40 GB pool rather than raising B_max × L_max.

---------------------------------------------------------------------------
## 13. Warmup plan inside `__init__` (budget 300 s; target < 120 s)

1. Read and assert `config.json`. Load safetensors (mmap) → GPU,
   fuse/reorder weights (≈ 10 s).
2. Allocate cache and all static buffers.
3. Compile Triton kernels by calling each once at every bucket shape
   on a side stream (≈ 30–60 s; print count and time).
4. cuBLAS warmup for prefill M set and decode M set.
5. Capture graphs: for each batch bucket × q_len tier (≈ 12–18 graphs,
   ≈ 0.2 s each). Validate each against eager once.
6. Run one full dummy `generate` per bucket (b1, b4, b16 with realistic
   lengths) to touch every path, including speculation and fallback.
7. GPU burn 3–5 s to stabilize clocks.
8. `torch.cuda.synchronize()`; `gc.collect()`; `gc.freeze()`;
   `gc.disable()`; print peak memory and total init time.
If init exceeds 200 s locally, reduce buckets/tiers before shipping.

---------------------------------------------------------------------------
## 14. Failure modes & debugging

- `cudaErrorStreamCaptureUnsupported` / capture invalidated: something
  synced during capture (Triton autotune, `.item()`, `print(tensor)`,
  cuBLAS first-call, allocator growth). Warm up more; remove syncs.
- Graph replays produce stale/garbage tokens: an input buffer was
  replaced instead of `copy_()`-ed into; or a Python int changed. All
  graph inputs are fixed tensors written in place.
- Illegal memory access in Triton: run with `TRITON_INTERPRET=1` on
  tiny shapes; check masks at odd lengths; check `seq_lens` upper
  bounds; check `tl.dot` alignment and `[16, 16, 16]` minimums.
- Token mismatch only at long L: RoPE table too short, cos/sin
  precision (must be fp32 → bf16 like HF), or split-KV reduce missing a
  segment.
- Mismatch only on `random` set: numerics drift on near-ties; check the
  logit-margin histogram; fix fp32 accumulation order / RMSNorm order.
- Mismatch only after b16 ran: cross-bucket state leak (dummy rows,
  stale `seq_lens`, adjacency matrix not reset).
- Spread > 25 % locally: check clocks, GC, first-sample effects,
  speculation variance (turn spec off to bisect).
- Official `infra_error` / `harness_error`: retry once, then ask on
  Slack with submission link, full Run ID, error, time.
- Official `latency_limit`: compare TTFT/TPOT ratios; usually prefill
  regressions or spec chains too long.
- Official `unstable_timing`: see spread above; speculation is the
  usual suspect.
- `lint_failed`: forbidden import or non-parsing file; `dryft validate`
  reproduces it locally. Check for stray `import numpy` or test files
  inside `engine/`.

---------------------------------------------------------------------------
## 15. Experiment ledger format (`agent/experiments.md`)

```
### 2026-09-20  [T2.2] fused qknorm+rope+kvwrite            KEEP
flag: FUSED_ROPE=1     commit: 3f1c9e2
verify: 100% (natural, copyheavy, random, eos_early, ragged, long, shapes); max margin 0.31
bench (tok/s):   b1: 341→352   b4: 611→633   b16: 4210→4390   geomean +3.6%
gates: TTFT 0.71/0.71/0.72×   TPOT 0.19/0.20/0.22×   spread 4%/3%/5%   peak 50.9 GB
kernels/step: 260 → 224      achieved BW: 2.58 → 2.71 TB/s
notes: q-norm epilogue needed fp32 rsqrt; bf16 rsqrt caused 2 mismatches on random set.
next: T2.3 silu*mul epilogue (needs no-split-K gate_up)
```
Keep a table at the top: date | commit | local geomean | official
geomean | hidden per-workload | notes. Use it to calibrate local vs
official numbers.

---------------------------------------------------------------------------
## 16. Questions to ask the organizers (Slack) — record answers here

1. How is TPOT computed: (total − TTFT)/(out − 1), or max/mean of
   inter-yield gaps? (Determines speculation chain limits.)
   ANSWER (from report v2): `tpotMs` is consistent with (total − TTFT)/
   (out − 1): b1 306 ms, TTFT 23, TPOT 9.15 × 31 = 284. Not asked.
2. How is timing spread computed: (max − min)/median over total time?
   ANSWER: report exposes p10/p50/p90/mean/stddev over 5 samples; the
   gate text says "spread across the five samples ≤ 25 %". Not asked.
3. Are prompt lengths equal within a hidden batch? Are hidden prompts
   natural text or synthetic/random ids? (Decides whether PLD is worth
   building.)
   ANSWER: ______
4. Is there a warmup call before the 5 timed samples, at the workload
   shape?
   ANSWER: yes — `warmupIterations: 1` in every shape's report.
5. Which attention implementation / generate flags does the baseline
   use (eager vs sdpa)? Is the baseline the unmodified starter engine?
   ANSWER: report has `referenceMs/referenceTtftMs/referenceTpotMs`
   from "native Qwen" timed in the same container; starter uses
   `attn_implementation="sdpa"`, `use_cache=True`, `logits_to_keep=1`.
6. Is peak memory measured via torch allocator or device-level
   (nvidia-smi)?
   ANSWER: `peakMemoryBytes` = 52 830 994 432 on every shape, identical
   to the byte, so it is the allocator's reserved peak (our static
   buffers), not a device sample. Budget stays 64 GB.
7. Are `torch.compile`-generated Triton kernels acceptable (compiled
   at runtime, not shipped)?
   ANSWER: ______
8. Is the GPU an H100 SXM (3.35 TB/s) or PCIe (2.0 TB/s)?
   ANSWER: `deviceName: NVIDIA H100 80GB HBM3` (SXM), driver 580.95,
   gVisor sandbox (`Linux-4.19.0-gvisor`), Python 3.11.5.

---------------------------------------------------------------------------
## 17. Don'ts (quick reference)

Don't import anything outside torch/triton/transformers/safetensors/
stdlib. Don't download. Don't read `~/.cache`. Don't write files at
runtime except `/tmp` (Triton cache is fine). Don't quantize. Don't
stop at EOS. Don't return early. Don't allocate in `generate`. Don't
use Python ints as Triton args inside graphs. Don't autotune at
runtime. Don't compute logits for prompt positions. Don't push to
`main` without `verify --all`, `bench --gate`, and `dryft validate`.
Don't ship the agent, notes, or tokens in `engine/`. Don't exploit the
2-logit tolerance. Don't trust a speedup that isn't in the ledger with
gates.

---------------------------------------------------------------------------
## Appendix A. Research background (why the plan looks like this)

A.1 Where batch-1 decode time really goes. Measurements on H100 by
Hazy Research (the "megakernel" line of work) and follow-ups by
Cohere / Mirage show mainstream engines (vLLM, SGLang) reach only about
half of HBM bandwidth at batch 1 on Llama-1B/8B-class models. The cause
is not the kernels' inner loops but the ~100+ launches per forward,
each with a setup/teardown bubble in which no bytes move. Fusing the
entire forward into one persistent kernel recovers most of that
(reported 1.25–1.5× over vLLM in bf16 on H100). We cannot ship CUDA,
so we take the Triton-achievable share: CUDA graphs (remove CPU launch
cost), aggressive fusion (≤ 6 kernels/layer), and later a persistent
per-layer Triton kernel.

A.2 Skinny GEMMs. For M ≤ 16 and N, K in the thousands, cuBLAS often
underfills the H100 (132 SMs). Published Triton split-K work reports
~2× average speedup over plain Triton tiles on H100 for Llama-shaped
GEMMs with split_k ≈ 8; GEMV-style kernels that skip `tl.dot` and do
K-dimension reductions win at M = 1. Benchmark per shape; don't assume.

A.3 Decode attention. Flash-Decoding (split-KV) is the standard: at
small batch the number of (seq, head) programs is far below SM count,
so the sequence dimension must be split and reduced. vLLM's Triton
split-KV backend reports up to ~3× at batch 1 over a 2-D kernel. The
same applies to the verify kernel in speculation — a naive
one-program-per-(seq, head) verify kernel makes verify cost scale with
prefix length and can make speculation net-negative.

A.4 Training-free speculation menu. Because no draft weights can be
shipped, the candidates are: Prompt Lookup Decoding (n-gram match on
prompt+output; free; excellent on copy-heavy tasks, useless on random
tokens), Token Recycling (adjacency matrix of top-k candidates from
past logits; uniform gains; tree verification; reported ~1.5–2× on 7B+
at batch 1), Lookahead / Jacobi decoding (1.5–2.3× reported vs HF greedy
but FLOP-hungry, so it fades at larger batch), SuffixDecoding (suffix
tree over prior outputs; strongest for repetitive agentic traces),
hybrid trees (context-matched PLD tokens are accepted several times
more often than TR transition tokens, so PLD forms a deep spine and TR
forms wide branches), and self-speculation via layer skipping (Draft &
Verify; up to ~1.7× on 70B, much less on small models because the gain
correlates with model redundancy).

A.5 Batched speculation. The "ragged tensor" problem — sequences
accept different numbers of tokens — is why most engines only run
speculation at small batch. Padding to the max accepted length wastes
30–50 % of verify work at batch ≥ 8 when per-token acceptance is below
~0.9. Because we own the KV layout and kernels, per-sequence lengths in
attention make raggedness free. SGLang's adaptive scheme (EMA of
accepted length → switch among a few pre-captured speculative-length
tiers) is the model for our guard in 5.7.

A.6 torch 2.5.1 / Triton 3.1 specifics. `enable_gqa=True` in SDPA only
dispatches to the flash backend in 2.5; `repeat_interleave` + SDPA may
pick cuDNN/efficient backends — benchmark both. CUDA-graph capture
fails if anything synchronizes (Triton autotune does), so warm up
every callable before capture and never autotune at runtime. Side
streams inside capture must be joined via `wait_stream` in both
directions. Graph replays require all inputs to be the same tensors
written in place.

A.7 What the 2025 literature adds (surveyed 2026-09-19, after run 1).
- Training-free speculation, greedy, Spec-Bench (Vicuna/Llama class,
  H100): Prompt Lookup MAT ≈ 1.7–1.8 → 1.5–1.7× (summarization/RAG
  best, conversation/translation worst); Token Recycling (Luo et al.,
  arXiv 2408.08696, ACL 2025) MAT ≈ 2.7–2.8 → ≈ 2.0–2.1×, the best pure
  train-free method on mixed tasks, < 2 MB state, tree of ~60 nodes in
  the paper (use 16–32 here: our verify overhead is larger on a 4B
  model); SuffixDecoding (Oliaro et al., arXiv 2411.04975, NeurIPS
  2025) MAT 6–8 but only on agentic traces with long verbatim repeats —
  irrelevant unless hidden prompts turn out copy-heavy; SAM-Decoding +
  TR ≈ 2.3× (arXiv 2411.10666). Hybrids (PLD spine + TR branches) are
  consistently ≥ either alone. Reported speedups are batch-1; at b = 16–
  32 the verify GEMMs approach the compute ridge and gains shrink, so
  cap the per-sequence node budget by batch bucket (5.7).
- Skinny GEMM on H100 (Meta/IBM arXiv 2402.00025; PyTorch blogs on
  Llama-3 FP8 TK-GEMM and MoE GEMMs): split-K ≈ 8 is the sweet spot at
  M ≤ 16; gains vs data-parallel Triton 1.2–1.9×, vs cuBLAS 1.1–1.9×
  depending on dtype (largest for FP8/W4, smallest for plain bf16 where
  cuBLAS GEMV already runs ~70 % of HBM peak). Hopper wgmma pads M to 64
  rows, so at M = 1 a `tl.dot` kernel wastes ~98 % of its MMA fill —
  prefer broadcast-multiply + `tl.sum` (GEMV style) for M ≤ 4 and let
  the tile loop stream weights. Lesson for us: measure per shape in
  `__init__`, keep cuBLAS wherever it wins, fuse the residual add into
  whichever kernel wins.
- Overhead: with graphs on, the residual cost is per-kernel execution
  tail (~2–4 µs) not launch latency; at ~2000 kernels/step that is most
  of our 9 ms. This is why fusion (Tier 2/3) outranks everything else
  on the roadmap, and why a persistent per-layer kernel (Tier 6) is the
  eventual end state if Tiers 2–5 leave headroom.

---------------------------------------------------------------------------
## Appendix B. Reading list (study these; do not copy code that isn't
## pure Python/Triton)

- gpt-fast (PyTorch Labs): ~1000-line pure-PyTorch reference for static
  KV cache + CUDA graphs on Llama-class models. Closest existing
  blueprint for Tiers 1–2.
- Flash-Decoding (Dao et al., PyTorch blog): split-KV attention for
  q_len = 1. Blueprint for Tier 3.
- vLLM `triton_attn` / unified Triton attention, SGLang Triton decode
  attention: GQA-aware split-KV kernels in Triton; study head-group
  loading and the reduce kernel.
- Triton split-K GEMM writeups (PyTorch blog, GemLite): skinny-M GEMM
  design, GEMV-style kernels for M = 1.
- Hazy Research "megakernel" posts and Mirage Persistent Kernel:
  overhead analysis; persistent-grid barrier patterns (Tier 6).
- Prompt Lookup Decoding (Saxena, 2023); LLMA "Inference with
  Reference"; Token Recycling (2024); Lookahead Decoding (Fu et al.,
  2024); SuffixDecoding (2024); Goose / hybrid-tree drafting (2025);
  Draft & Verify (Zhang et al., 2023) and LayerSkip (Meta, 2024).
- SGLang speculative decoding docs (adaptive speculative length via
  EMA tiers) and EAGLE-3 tree-verification for tree-mask layout.
- HF `modeling_qwen3.py` (transformers 4.51.3): the reference
  semantics for RMSNorm order, q/k-norm placement, RoPE.
- PyTorch CUDA Graphs documentation and the `torch.cuda.graph` /
  `graph_pool_handle` API; Triton 3.1 docs for `tl.dot` constraints,
  `num_stages`, `multiple_of` hints.
