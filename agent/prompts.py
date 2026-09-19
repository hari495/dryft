"""Prompt sets for verification (AGENTS.md §7.2), as token ids.

Synthetic sets need only the vocab size and are built deterministically from a
seed. ``natural``/``copyheavy`` text sets need the checkpoint's tokenizer
(``tokenizer.json`` under ``model_path``) and are chat-templated for Qwen3.

    python agent/prompts.py --model /path/to/ckpt --write   # dump agent/prompts/*.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "prompts")

# Qwen3 special ids (from tokenizer_config.json of the pinned checkpoint).
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643


@dataclass
class Case:
    input_ids: list[list[int]]
    max_new_tokens: int
    note: str = ""


@dataclass
class PromptSet:
    name: str
    cases: list[Case] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"name": self.name, "cases": [c.__dict__ for c in self.cases]}

    @staticmethod
    def from_json(d: dict) -> "PromptSet":
        return PromptSet(d["name"], [Case(**c) for c in d["cases"]])


@dataclass(frozen=True)
class Scale:
    """Shape knobs so the same sets run against the tiny CPU model."""

    vocab: int
    long_in: int = 4096
    long_out: int = 512
    long_b: tuple[int, ...] = (1, 8)
    mid_in: int = 512
    big_in: int = 2048
    out: int = 32
    len_buckets: tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
    batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    b_max: int = 32
    l_max: int = 8192


TINY = Scale(
    vocab=256, long_in=48, long_out=12, long_b=(1, 3), mid_in=16, big_in=32, out=8,
    len_buckets=(32, 64), batch_buckets=(1, 2, 4, 8), b_max=8, l_max=64,
)
FULL = Scale(vocab=151936)


def _rand_ids(rng: random.Random, b: int, n: int, vocab: int) -> list[list[int]]:
    return [[rng.randrange(vocab) for _ in range(n)] for _ in range(b)]


def _copyheavy_ids(rng: random.Random, b: int, n: int, vocab: int) -> list[list[int]]:
    """Prompts made of a few repeated segments: n-gram rich (prompt lookup wins)."""
    out = []
    for _ in range(b):
        seg = [rng.randrange(vocab) for _ in range(max(4, n // 8))]
        seq: list[int] = []
        while len(seq) < n:
            k = rng.randrange(len(seg))
            seq.extend(seg[k:] + seg[:k])
        out.append(seq[:n])
    return out


def synthetic_sets(scale: Scale, seed: int = 0) -> list[PromptSet]:
    rng = random.Random(seed)
    V = scale.vocab
    sets: list[PromptSet] = []

    s = PromptSet("random")
    s.cases += [
        Case(_rand_ids(rng, 1, scale.mid_in, V), scale.out, "b1 public shape"),
        Case(_rand_ids(rng, 4, scale.big_in, V), scale.out, "b4 public shape"),
        Case(_rand_ids(rng, 16 if scale.b_max >= 16 else scale.b_max, scale.mid_in, V), max(4, scale.out // 2), "b16 public shape"),
        Case(_rand_ids(rng, 2, scale.mid_in // 2, V), scale.out, "b2 short"),
    ]
    sets.append(s)

    s = PromptSet("copyheavy_synth")
    s.cases += [
        Case(_copyheavy_ids(rng, 1, scale.mid_in, V), scale.out * 2 if scale.out * 2 + scale.mid_in <= scale.l_max else scale.out),
        Case(_copyheavy_ids(rng, 4, scale.mid_in, V), scale.out),
    ]
    sets.append(s)

    s = PromptSet("ragged")
    base = scale.mid_in
    s.cases += [
        Case(_rand_ids(rng, 1, base, V) + _rand_ids(rng, 1, base // 2, V) + _rand_ids(rng, 1, base - 3, V), scale.out, "3 lengths"),
        Case(_rand_ids(rng, 2, base // 4, V) + _rand_ids(rng, 2, base, V), max(2, scale.out // 4), "2 lengths"),
    ]
    sets.append(s)

    s = PromptSet("long")
    for b in scale.long_b:
        s.cases.append(Case(_rand_ids(rng, b, scale.long_in, V), scale.long_out, f"b{b}"))
    sets.append(s)

    s = PromptSet("shapes")
    for Lb in scale.len_buckets:
        for delta in (-1, 0, 1):
            total = Lb + delta
            if total < 3 or total > scale.l_max + 1:
                continue
            for out in (1, 2):
                n = total - out
                if n < 1:
                    continue
                s.cases.append(Case(_rand_ids(rng, 1, n, V), out, f"total={total} out={out}"))
    for Bb in scale.batch_buckets:
        for b in (Bb, Bb + 1):
            if b > scale.b_max + 1:
                continue
            s.cases.append(Case(_rand_ids(rng, b, 8, V), 3, f"batch={b}"))
    # Beyond every bucket: exercises the eager fallback.
    s.cases.append(Case(_rand_ids(rng, scale.b_max + 1, 8, V), 2, "fallback batch"))
    sets.append(s)

    s = PromptSet("eos_early_synth")
    # Prompts that end with an im_end/im_start pair so EOS is likely soon;
    # generation must continue past it. Only meaningful for the real vocab.
    if V > IM_END:
        for b in (1, 3):
            ids = _rand_ids(rng, b, 24, V)
            for seq in ids:
                seq[-3:] = [IM_END, IM_START, ENDOFTEXT]
            s.cases.append(Case(ids, 24, "ends with im_end"))
    else:
        s.cases.append(Case(_rand_ids(rng, 2, 8, V), 8, "tiny stand-in"))
    sets.append(s)
    return sets


NATURAL_TEXTS = [
    ("en", "Explain, step by step, why the sky appears blue during the day but red at sunset. Keep it under 200 words."),
    ("code", "Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs and returns them sorted. Include a docstring and three test cases."),
    ("math", "Solve for x: 3x^2 - 12x + 9 = 0. Show every step, then verify both roots."),
    ("zh", "请用中文简要介绍一下量子计算的基本原理，并举一个实际应用的例子。"),
]

COPYHEAVY_TEXTS = [
    ("summarize", "Summarize the following text in three sentences, quoting the key phrases exactly:\n\n" + " ".join(
        f"Item {i}: the {['red', 'green', 'blue', 'amber'][i % 4]} sensor reported {100 + 7 * i} units at station {chr(65 + i % 26)}{i}." for i in range(120))),
    ("extract", "Extract every email address from the log below and list them one per line in order of appearance.\n\n" + "\n".join(
        f"2026-09-{1 + i % 28:02d} INFO user{i}@example{i % 5}.org logged in from 10.0.{i % 255}.{(i * 7) % 255}" for i in range(80))),
    ("codeedit", "Rename the variable `total_count` to `n_items` everywhere in this file and return the full file:\n\n" + "\n".join(
        [f"def f{i}(xs):\n    total_count = 0\n    for x in xs:\n        total_count += x * {i}\n    return total_count\n" for i in range(30)])),
]


def chat_template(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def text_sets(model_path: str, out_tokens: int = 64) -> list[PromptSet]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))

    def enc(text: str) -> list[int]:
        return tok.encode(chat_template(text), add_special_tokens=False).ids

    natural = PromptSet("natural")
    for tag, text in NATURAL_TEXTS:
        natural.cases.append(Case([enc(text)], out_tokens, tag))
    natural.cases.append(Case([enc(t) for _, t in NATURAL_TEXTS[:2]] * 2, out_tokens, "b4 mixed (ragged)"))
    copy = PromptSet("copyheavy")
    for tag, text in COPYHEAVY_TEXTS:
        copy.cases.append(Case([enc(text)], out_tokens * 2, tag))
    eos = PromptSet("eos_early")
    for text in ("Reply with exactly one word: yes or no. Is 7 prime?", "What is 2+2? Answer with only the number."):
        eos.cases.append(Case([enc(text)], 32, "short answer"))
    return [natural, copy, eos]


def load_or_build(scale: Scale, model_path: str | None, names: list[str] | None = None) -> list[PromptSet]:
    """Synthetic sets for ``scale``; plus text sets when running the real vocab
    (from ``agent/prompts/*.json`` if dumped, else tokenized on the fly)."""
    sets = synthetic_sets(scale)
    if scale.vocab >= 151936:
        dumped = sorted(f for f in os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []
        if dumped:
            for fn in dumped:
                if fn.endswith(".json"):
                    with open(os.path.join(OUT_DIR, fn), "r", encoding="utf-8") as f:
                        sets.append(PromptSet.from_json(json.load(f)))
        elif model_path and os.path.exists(os.path.join(model_path, "tokenizer.json")):
            sets += text_sets(model_path)
    if names:
        sets = [s for s in sets if s.name in names]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DRYFT_MODEL_PATH"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("--model or DRYFT_MODEL_PATH required to tokenize text sets")
    sets = text_sets(args.model)
    for s in sets:
        print(s.name, [(len(c.input_ids), len(c.input_ids[0]), c.max_new_tokens) for c in s.cases])
    if args.write:
        os.makedirs(OUT_DIR, exist_ok=True)
        for s in sets:
            with open(os.path.join(OUT_DIR, f"{s.name}.json"), "w", encoding="utf-8") as f:
                json.dump(s.to_json(), f)
        print(f"wrote {len(sets)} sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
