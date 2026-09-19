"""Per-kernel table for a few decode steps (AGENTS.md §8, profiling).

    python agent/profiler.py --workload b1_512_32 --steps 8 [--model PATH] [--eager]

Prints: kernel name | count/step | us/step | % of step, plus kernels/step,
CUDA time/step, CPU time/step and the implied achieved bandwidth. Use
``--eager`` to profile the un-captured decode step (graph replays show up as
one opaque node otherwise); ``--prefill`` profiles the prefill instead.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, HERE)

from bench import EXTRA, PUBLIC  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DRYFT_MODEL_PATH"))
    ap.add_argument("--workload", default="b1_512_32")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--eager", action="store_true", help="disable CUDA graphs so kernels are visible")
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    if not args.model:
        print("need --model or DRYFT_MODEL_PATH")
        return 2
    if not torch.cuda.is_available():
        print("profile needs CUDA")
        return 2

    from dryft_qwen3 import config

    if args.eager:
        config.FLAGS["CUDA_GRAPHS"] = False
    import engine

    eng = engine.Engine(args.model)
    cfg = eng.cfg
    name, b, inp, out = next(w for w in PUBLIC + EXTRA if w[0] == args.workload)
    rng = random.Random(0)
    ids = [[rng.randrange(cfg.vocab) for _ in range(inp)] for _ in range(b)]
    n_steps = 1 if args.prefill else args.steps + 1

    gen = eng.generate(ids, n_steps)
    next(gen)  # prefill + first token, outside the profiled window unless --prefill
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if args.prefill:
            for _ in eng.generate(ids, 1):
                pass
        else:
            for _ in gen:
                pass
        torch.cuda.synchronize()

    steps = 1 if args.prefill else args.steps
    by_name: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    cuda_total = 0.0
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            by_name[ev.name][0] += 1
            by_name[ev.name][1] += ev.time_range.elapsed_us()
            cuda_total += ev.time_range.elapsed_us()
    rows = sorted(by_name.items(), key=lambda kv: -kv[1][1])
    n_kernels = sum(c for c, _ in by_name.values())
    print(f"{args.workload}: {steps} step(s); {n_kernels / steps:.0f} kernels/step; CUDA {cuda_total / steps / 1e3:.2f} ms/step")
    if not args.prefill and cuda_total > 0:
        bw = cfg.decode_bytes_per_step / (cuda_total / steps * 1e-6) / 1e12
        print(f"implied bandwidth {bw:.2f} TB/s ({cfg.decode_bytes_per_step / 1e9:.2f} GB weights per step)")
    print(f"{'kernel':<70} {'count/step':>10} {'us/step':>10} {'%':>6}")
    for name_, (count, us) in rows[: args.top]:
        print(f"{name_[:70]:<70} {count / steps:>10.1f} {us / steps:>10.1f} {100 * us / max(cuda_total, 1e-9):>6.1f}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return 0


if __name__ == "__main__":
    sys.exit(main())
