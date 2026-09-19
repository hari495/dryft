"""Timing harness mirroring the official scoring and gates (AGENTS.md §8).

    python agent/bench.py --public --extra [--model PATH]   # per-workload table
    python agent/bench.py --gate                            # pass/fail summary, exit code
    python agent/bench.py --baseline                        # measure the starter engine -> agent/baseline.json
    python agent/bench.py --tiny --public                   # CPU smoke run on the tiny model

Per workload: one untimed warmup call, then N timed samples with fresh random
prompts; wall clock around full generator consumption with
``torch.cuda.synchronize()`` at both ends. Reports median tok/s
(``batch * out / median_seconds``), TTFT, TPOT = (total - TTFT)/(out - 1),
spread = (max - min)/median, peak memory, achieved HBM bandwidth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE_DIR = os.path.join(ROOT, "engine")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, HERE)

import prompts as P  # noqa: E402

BASELINE_JSON = os.path.join(HERE, "baseline.json")
PUBLIC = [("b1_512_32", 1, 512, 32), ("b4_2048_32", 4, 2048, 32), ("b16_512_128", 16, 512, 128)]
EXTRA = [
    ("b2_1024_64", 2, 1024, 64), ("b8_256_256", 8, 256, 256), ("b8_2048_64", 8, 2048, 64),
    ("b32_128_32", 32, 128, 32), ("b1_4096_128", 1, 4096, 128), ("b16_1024_256", 16, 1024, 256),
]
TINY_WORKLOADS = [("t1_16_8", 1, 16, 8), ("t4_32_8", 4, 32, 8), ("t8_16_12", 8, 16, 12)]
LATENCY_GATE = 1.10
SPREAD_GATE = 0.25
MEM_GATE = 0.90


@dataclass
class Result:
    name: str
    batch: int
    inp: int
    out: int
    totals: list[float]
    ttfts: list[float]
    peak_alloc: int
    peak_reserved: int

    @property
    def median(self) -> float:
        return statistics.median(self.totals)

    @property
    def tok_s(self) -> float:
        return self.batch * self.out / self.median

    @property
    def ttft(self) -> float:
        return statistics.median(self.ttfts)

    @property
    def tpot(self) -> float:
        if self.out <= 1:
            return 0.0
        return (self.median - self.ttft) / (self.out - 1)

    @property
    def spread(self) -> float:
        return (max(self.totals) - min(self.totals)) / self.median

    def to_json(self) -> dict:
        return {
            "batch": self.batch, "inp": self.inp, "out": self.out, "median_s": self.median, "ttft_s": self.ttft,
            "tpot_s": self.tpot, "tok_s": self.tok_s, "spread": self.spread, "peak_alloc": self.peak_alloc,
            "peak_reserved": self.peak_reserved,
        }


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_one(eng, ids: list[list[int]], out: int) -> tuple[float, float]:
    sync()
    t0 = time.perf_counter()
    ttft = None
    n = 0
    for _ in eng.generate(ids, out):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    sync()
    total = time.perf_counter() - t0
    assert n == out, f"engine yielded {n} steps, expected {out}"
    return total, ttft


def run_workload(eng, name: str, b: int, inp: int, out: int, vocab: int, samples: int, seed: int) -> Result:
    rng = random.Random(seed)
    warm = [[rng.randrange(vocab) for _ in range(inp)] for _ in range(b)]
    time_one(eng, warm, out)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    totals, ttfts = [], []
    for _ in range(samples):
        ids = [[rng.randrange(vocab) for _ in range(inp)] for _ in range(b)]
        gc.collect()
        total, ttft = time_one(eng, ids, out)
        totals.append(total)
        ttfts.append(ttft)
    pa = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    pr = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    return Result(name, b, inp, out, totals, ttfts, pa, pr)


def load_ours(model_path: str, tiny: bool):
    from dryft_qwen3 import config

    if tiny:
        config.EXPECTED_CONFIG = None
        config.SETTINGS = replace(
            config.SETTINGS, b_max=P.TINY.b_max, l_max=P.TINY.l_max, batch_buckets=P.TINY.batch_buckets,
            len_buckets=P.TINY.len_buckets, warmup_shapes=((1, 16, 4),), burn_seconds=0.0, prefill_warmup_m=(16, 64),
        )
    import engine

    return engine.Engine(model_path), config.load_model_config(model_path)


def load_baseline(model_path: str, tiny: bool):
    import baseline_engine
    from dryft_qwen3 import config

    if tiny:
        config.EXPECTED_CONFIG = None
    if not torch.cuda.is_available():
        # starter hard-codes cuda:0; patch for a CPU smoke run
        src = open(baseline_engine.__file__).read().replace('"cuda:0"', '"cpu"')
        ns: dict = {}
        exec(compile(src, "baseline_engine_cpu", "exec"), ns)
        return ns["Engine"](model_path), config.load_model_config(model_path)
    return baseline_engine.Engine(model_path), config.load_model_config(model_path)


def fmt_row(r: Result, base: dict | None, bytes_per_step: int, dev_total: int) -> tuple[str, bool]:
    ok = True
    cols = [f"{r.name:<14}", f"{r.tok_s:9.1f} tok/s", f"total {r.median * 1e3:8.1f} ms", f"ttft {r.ttft * 1e3:7.1f} ms", f"tpot {r.tpot * 1e3:6.2f} ms"]
    if base:
        rt = r.ttft / base["ttft_s"] if base["ttft_s"] else 0.0
        rp = r.tpot / base["tpot_s"] if base["tpot_s"] else 0.0
        flag_t = "!" if rt > LATENCY_GATE else " "
        flag_p = "!" if rp > LATENCY_GATE else " "
        ok &= rt <= LATENCY_GATE and rp <= LATENCY_GATE
        cols.append(f"ttft/base {rt:4.2f}x{flag_t} tpot/base {rp:4.2f}x{flag_p} speed {base['median_s'] / r.median:5.2f}x")
    flag_s = "!" if r.spread > SPREAD_GATE else " "
    ok &= r.spread <= SPREAD_GATE
    cols.append(f"spread {100 * r.spread:4.1f}%{flag_s}")
    if r.tpot > 0:
        cols.append(f"BW {bytes_per_step / r.tpot / 1e12:4.2f} TB/s")
    if dev_total:
        frac = r.peak_reserved / dev_total
        ok &= frac <= MEM_GATE
        cols.append(f"mem {r.peak_alloc / 2**30:5.1f}/{r.peak_reserved / 2**30:5.1f} GiB{'!' if frac > MEM_GATE else ''}")
    return "  ".join(cols), ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DRYFT_MODEL_PATH"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--gate", action="store_true", help="public shapes, print gates only, exit 1 on failure")
    ap.add_argument("--baseline", action="store_true", help="time the starter engine and save agent/baseline.json")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--flag", action="append", default=[], help="FLAG=0|1 overrides")
    ap.add_argument("--workload", action="append", default=[], help="name,b,in,out")
    args = ap.parse_args()

    if args.tiny:
        import verify

        model_path = verify.make_tiny_model(os.path.join(os.environ.get("TMPDIR", "/tmp"), "dryft-tiny-qwen3"))
        workloads = TINY_WORKLOADS
    else:
        if not args.model:
            print("need --model PATH or DRYFT_MODEL_PATH (or --tiny)")
            return 2
        model_path = args.model
        workloads = []
        if args.public or args.gate or not (args.extra or args.workload):
            workloads += PUBLIC
        if args.extra:
            workloads += EXTRA
    for w in args.workload:
        name, b, i, o = w.split(",")
        workloads.append((name, int(b), int(i), int(o)))

    from dryft_qwen3 import config

    for kv in args.flag:
        k, v = kv.split("=")
        config.FLAGS[k] = bool(int(v))

    dev_total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        print(f"device {torch.cuda.get_device_name(0)}, {dev_total / 2**30:.0f} GiB")
    print(f"flags {config.FLAGS}; seed {args.seed}; samples {args.samples}")

    t0 = time.time()
    eng, cfg = load_baseline(model_path, args.tiny) if args.baseline else load_ours(model_path, args.tiny)
    print(f"engine init {time.time() - t0:.1f}s")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    baseline_json = BASELINE_JSON.replace(".json", "_tiny.json") if args.tiny else BASELINE_JSON
    base_all = {}
    if os.path.exists(baseline_json) and not args.baseline:
        with open(baseline_json) as f:
            base_all = json.load(f)

    results: list[Result] = []
    all_ok = True
    for name, b, inp, out in workloads:
        r = run_workload(eng, name, b, inp, out, cfg.vocab, args.samples, args.seed)
        results.append(r)
        row, ok = fmt_row(r, base_all.get(name), cfg.decode_bytes_per_step, dev_total)
        all_ok &= ok
        print(row)

    pub = [r for r in results if r.name in {p[0] for p in PUBLIC} or args.tiny]
    if pub:
        gm = math.exp(sum(math.log(r.tok_s) for r in pub) / len(pub))
        print(f"geomean(public) {gm:.1f} tok/s")
    if args.baseline:
        base_all.update({r.name: r.to_json() for r in results})
        with open(baseline_json, "w") as f:
            json.dump(base_all, f, indent=1, sort_keys=True)
        print(f"saved baseline to {baseline_json}")
    if args.gate:
        print(f"gates: {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
