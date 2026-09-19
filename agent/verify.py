"""Correctness gate: engine tokens vs the native reference, plus the judge's
teacher-forced replay margin (AGENTS.md §7).

    python agent/verify.py --all [--model PATH]      # every prompt set, real checkpoint
    python agent/verify.py --tiny --all              # random tiny Qwen3 on CPU (no GPU needed)
    python agent/verify.py --set random --set ragged
    python agent/verify.py --kernels                 # Triton kernel unit tests (needs CUDA)

Reference = the unmodified starter loop: Qwen3ForCausalLM bf16, sdpa, greedy,
``logits_to_keep=1``, never stopping at EOS. Ragged batches are referenced per
sequence (the reference has no padding path; left-aligned positions match).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE_DIR = os.path.join(ROOT, "engine")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, HERE)

import prompts as P  # noqa: E402

MARGIN_LIMIT = 1.0  # self-imposed; official tie margin is 2.0


# --------------------------------------------------------------------------- setup
def make_tiny_model(path: str, seed: int = 0) -> str:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    if os.path.exists(os.path.join(path, "model.safetensors")):
        return path
    cfg = Qwen3Config(
        vocab_size=P.TINY.vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, hidden_act="silu",
        max_position_embeddings=262144, rms_norm_eps=1e-6, tie_word_embeddings=True,
        rope_theta=5_000_000, attention_bias=False, use_sliding_window=False,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():  # random init is ~N(0, 0.02); inflate so logits are not flat
        for p in model.parameters():
            p.mul_(4.0)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    return path


def load_engine(model_path: str, scale: P.Scale):
    from dryft_qwen3 import config

    if scale is P.TINY:
        config.EXPECTED_CONFIG = None
        config.SETTINGS = replace(
            config.SETTINGS, b_max=scale.b_max, l_max=scale.l_max, batch_buckets=scale.batch_buckets,
            len_buckets=scale.len_buckets, warmup_shapes=((1, 16, 4), (4, 32, 4)), burn_seconds=0.0,
            prefill_warmup_m=(16, 64),
        )
    import engine

    return engine.Engine(model_path)


def load_reference(model_path: str, device: torch.device):
    from transformers import Qwen3ForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Qwen3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.eval().to(device)


# ----------------------------------------------------------------------- reference
@torch.inference_mode()
def reference_generate(model, input_ids: list[list[int]], n: int, device) -> list[list[int]]:
    """The starter engine's loop, verbatim semantics. Returns [n][B]."""
    lens = {len(s) for s in input_ids}
    if len(lens) != 1:
        cols = [reference_generate(model, [s], n, device) for s in input_ids]
        return [[cols[b][t][0] for b in range(len(input_ids))] for t in range(n)]
    current = torch.tensor(input_ids, dtype=torch.int64, device=device)
    cache = None
    out = []
    for _ in range(n):
        o = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        current = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = o.past_key_values
        out.append(current[:, 0].tolist())
    return out


@torch.inference_mode()
def replay_margins(model, prompt: list[int], ours: list[int], device) -> list[float]:
    """Judge rule: teacher-force OUR tokens; margin_t = logit[argmax] - logit[ours_t]."""
    ids = torch.tensor([prompt + ours[:-1]], dtype=torch.int64, device=device)
    logits = model(input_ids=ids, return_dict=True).logits[0].float()
    L = len(prompt)
    pos = logits[L - 1 : L - 1 + len(ours)]
    best = pos.max(dim=-1).values
    mine = pos.gather(1, torch.tensor(ours, device=device)[:, None])[:, 0]
    return (best - mine).tolist()


def run_engine(eng, input_ids, n) -> list[list[int]]:
    out = []
    for step in eng.generate(input_ids, n):
        out.append(list(step))
    return out


# -------------------------------------------------------------------------- checks
def check_case(eng, ref, case: P.Case, device, do_margin: bool) -> tuple[bool, str, float]:
    B, n = len(case.input_ids), case.max_new_tokens
    ours = run_engine(eng, case.input_ids, n)
    if len(ours) != n or any(len(row) != B for row in ours):
        return False, f"shape: got {len(ours)} steps of {[len(r) for r in ours[:3]]}, want {n} x {B}", 0.0
    if any(not isinstance(t, int) for row in ours for t in row):
        return False, "non-int token yielded", 0.0
    again = run_engine(eng, case.input_ids, n)
    if again != ours:
        return False, "second run differs from first (stale state / race)", 0.0
    want = reference_generate(ref, case.input_ids, n, device)
    mism = [(t, b) for t in range(n) for b in range(B) if ours[t][b] != want[t][b]]
    max_margin = 0.0
    if do_margin:
        for b in range(B):
            col = [ours[t][b] for t in range(n)]
            max_margin = max(max_margin, max(replay_margins(ref, case.input_ids[b], col, device)))
    if mism:
        t, b = mism[0]
        return False, f"{len(mism)} mismatches; first at step {t} seq {b}: ours {ours[t][b]} ref {want[t][b]}; max margin {max_margin:.3f}", max_margin
    if max_margin > MARGIN_LIMIT:
        return False, f"replay margin {max_margin:.3f} > {MARGIN_LIMIT}", max_margin
    return True, f"ok (max margin {max_margin:.3f})", max_margin


def check_cross_bucket(eng, sets: list[P.PromptSet]) -> tuple[bool, str]:
    cases = [c for s in sets for c in s.cases]
    small = next((c for c in cases if len(c.input_ids) == 1), None)
    big = max(cases, key=lambda c: len(c.input_ids), default=None)
    if small is None or big is None or big is small:
        return True, "skipped (no b1 + big pair)"
    a = run_engine(eng, small.input_ids, small.max_new_tokens)
    run_engine(eng, big.input_ids, big.max_new_tokens)
    b = run_engine(eng, small.input_ids, small.max_new_tokens)
    return (a == b), ("ok" if a == b else "b1 output changed after a larger batch ran (state leak)")


def kernel_tests() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        print("kernels: triton not importable here; skipping (run on the H100)")
        return True
    if not torch.cuda.is_available():
        print("kernels: no CUDA; skipping")
        return True
    ok = True
    from dryft_qwen3.kernels import attn_decode, rmsnorm, rope_qknorm, silu_mul

    for mod in (rmsnorm, rope_qknorm, attn_decode, silu_mul):
        t0 = time.time()
        try:
            mod.selftest()
            print(f"kernels: {mod.__name__} ok ({time.time() - t0:.1f}s)")
        except Exception as e:
            ok = False
            print(f"kernels: {mod.__name__} FAILED: {e!r}")
    return ok


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DRYFT_MODEL_PATH"))
    ap.add_argument("--tiny", action="store_true", help="random tiny Qwen3 (CPU-friendly)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--no-margin", action="store_true", help="skip the teacher-forced replay check")
    ap.add_argument("--flag", action="append", default=[], help="FLAG=0|1 overrides, e.g. --flag CUDA_GRAPHS=0")
    args = ap.parse_args()

    if args.kernels and not (args.all or args.set):
        return 0 if kernel_tests() else 1

    scale = P.TINY if args.tiny else P.FULL
    if args.tiny:
        model_path = make_tiny_model(os.path.join(os.environ.get("TMPDIR", "/tmp"), "dryft-tiny-qwen3"))
    elif args.model:
        model_path = args.model
    else:
        print("need --model PATH (or DRYFT_MODEL_PATH) or --tiny")
        return 2

    from dryft_qwen3 import config

    for kv in args.flag:
        k, v = kv.split("=")
        config.FLAGS[k] = bool(int(v))
    device = config.device()
    print(f"device {device}; flags {config.FLAGS}")

    t0 = time.time()
    eng = load_engine(model_path, scale)
    print(f"engine init {time.time() - t0:.1f}s")
    ref = load_reference(model_path, device)
    sets = P.load_or_build(scale, model_path, None if args.all else args.set)
    if not sets:
        print("no prompt sets selected; use --all or --set NAME")
        return 2

    all_ok = True
    worst = 0.0
    for s in sets:
        n_ok = 0
        for i, case in enumerate(s.cases):
            t1 = time.time()
            ok, msg, m = check_case(eng, ref, case, device, not args.no_margin)
            worst = max(worst, m)
            n_ok += ok
            all_ok &= ok
            shape = f"b{len(case.input_ids)}x{len(case.input_ids[0])}x{case.max_new_tokens}"
            status = "PASS" if ok else "FAIL"
            print(f"  {s.name:<16} {i:>2} {shape:<16} {status} {msg}  [{time.time() - t1:.1f}s] {case.note}")
        print(f"{s.name}: {n_ok}/{len(s.cases)} cases")
    ok, msg = check_cross_bucket(eng, sets)
    all_ok &= ok
    print(f"cross-bucket: {'PASS' if ok else 'FAIL'} {msg}")
    if args.kernels:
        all_ok &= kernel_tests()
    print(f"\nverify: {'ALL PASS' if all_ok else 'FAILURES'}; worst replay margin {worst:.3f} (limit {MARGIN_LIMIT}, official 2.0)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
