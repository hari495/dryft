"""Qwen3-4B greedy decode engine for the Dryft benchmark.

``__init__`` does everything expensive: weight load + fusion, static KV cache
and buffers, cuBLAS warmup, CUDA-graph capture per bucket, dummy generations,
a short GPU burn. ``generate`` allocates nothing on the bucketed path, keeps
one decode step in flight and yields from a pinned host ring.

Interface (fixed by the platform):
    Engine(model_path).generate(input_ids: list[list[int]], max_new_tokens) ->
    yields exactly ``max_new_tokens`` lists of ``len(input_ids)`` ints.
"""

from __future__ import annotations

import gc
import time
from collections import defaultdict

import torch

from dryft_qwen3 import config
from dryft_qwen3.cache import DecodeState
from dryft_qwen3.config import flag, load_model_config, log, pick_bucket
from dryft_qwen3.model import Model, rope_tables
from dryft_qwen3.weights import load_weights


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            # Keep SDPA on flash (prefill) / mem-efficient + math (masked decode):
            # the cuDNN backend is the one most likely to misbehave under graph
            # capture, and it is not needed for either shape.
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.settings = config.SETTINGS
        self.device = config.device()
        self.cuda = self.device.type == "cuda"
        self.cfg = load_model_config(model_path)
        self.weights = load_weights(model_path, self.cfg, self.device)
        t_load = time.time()
        log(f"weights loaded in {t_load - t0:.1f}s on {self.device}")

        s = self.settings
        self.model = Model(self.cfg, self.weights, self.device, rope_len=s.l_max)
        self.state = DecodeState(
            self.cfg, s.b_max, s.l_max, self.device, pinned=self.cuda,
            rope=(self.model.cos, self.model.sin), ring_depth=s.ring_depth,
        )
        self.graphs = None
        if self.cuda:
            self.events = [torch.cuda.Event() for _ in range(s.ring_depth)]
            deadline = t0 + s.init_budget_seconds
            self._warm_cublas()
            if flag("CUDA_GRAPHS"):
                from dryft_qwen3.graphs import DecodeGraphs

                self.graphs = DecodeGraphs(self.model, self.state, s)
                self.graphs.capture_all(deadline)
            self._warm_generate(deadline)
            self._burn(s.burn_seconds)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            log(f"peak allocated {peak:.1f} GiB, reserved {reserved:.1f} GiB, kv cache {self.state.bytes() / 2**30:.1f} GiB")
        gc.collect()
        gc.freeze()
        gc.disable()
        log(f"init done in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------- warmup
    def _warm_cublas(self) -> None:
        cfg, w = self.cfg, self.weights
        L0 = w.layers[0]
        with torch.inference_mode():
            for M in self.settings.prefill_warmup_m:
                if M > self.settings.b_max * self.settings.l_max:
                    continue
                h = torch.zeros((M, cfg.hidden), dtype=torch.bfloat16, device=self.device)
                torch.nn.functional.linear(h, L0.w_qkv)
                torch.nn.functional.linear(torch.zeros((M, cfg.q_dim), dtype=torch.bfloat16, device=self.device), L0.w_o)
                torch.nn.functional.linear(h, L0.w_gu)
                torch.nn.functional.linear(torch.zeros((M, cfg.intermediate), dtype=torch.bfloat16, device=self.device), L0.w_down)
                del h
            for Bb in self.settings.batch_buckets:
                h = torch.zeros((Bb, cfg.hidden), dtype=torch.bfloat16, device=self.device)
                torch.nn.functional.linear(h, w.lm_head).argmax(dim=-1)
        torch.cuda.synchronize()

    def _warm_generate(self, deadline: float) -> None:
        for (b, l, out) in self.settings.warmup_shapes:
            if time.time() > deadline:
                log("warmup generations skipped: init budget reached")
                return
            if b > self.settings.b_max or l + out > self.settings.l_max:
                continue
            ids = [[(7 * i + 13 * j) % self.cfg.vocab for j in range(l)] for i in range(b)]
            for _ in self.generate(ids, out):
                pass
        torch.cuda.synchronize()

    def _burn(self, seconds: float) -> None:
        if seconds <= 0:
            return
        a = torch.randn((4096, 4096), dtype=torch.bfloat16, device=self.device)
        t_end = time.time() + seconds
        while time.time() < t_end:
            for _ in range(20):
                a = a @ a * 1e-3
            torch.cuda.synchronize()
        del a

    # --------------------------------------------------------------- generate
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield one list of token ids per step, one id per sequence,
        exactly max_new_tokens times. Greedy; do not stop at end-of-sequence."""
        B = len(input_ids)
        if B == 0 or max_new_tokens <= 0:
            return
        lens = [len(s) for s in input_ids]
        need_len = max(lens) + max_new_tokens
        s = self.settings
        Bb = pick_bucket(s.batch_buckets, B)
        Lb = pick_bucket(s.len_buckets, need_len)
        if Bb is not None and Lb is not None and Bb <= self.state.b_cap and Lb <= self.state.l_cap:
            state, graphs = self.state, self.graphs
        else:
            # Out-of-bucket request: eager path with a right-sized cache.
            log(f"fallback path for B={B}, L={need_len}")
            Bb, Lb = B, need_len
            rope = (self.model.cos, self.model.sin)
            if need_len > self.model.rope_len:
                rope = rope_tables(self.cfg, need_len, self.device)
            state = DecodeState(self.cfg, Bb, Lb, self.device, pinned=self.cuda, rope=rope, ring_depth=s.ring_depth)
            graphs = None
        yield from self._run(state, graphs, Bb, Lb, input_ids, lens, max_new_tokens)

    def _run(self, state: DecodeState, graphs, Bb: int, Lb: int, input_ids, lens, max_new_tokens: int):
        B = len(input_ids)
        model = self.model
        with torch.inference_mode():
            state.reset_rows(Bb)
            # Prefill, grouped by prompt length (one group for fixed batches).
            groups: dict[int, list[int]] = defaultdict(list)
            for r, n in enumerate(lens):
                groups[n].append(r)
            for n, rows_list in groups.items():
                rows = torch.tensor(rows_list, dtype=torch.int64, device=self.device)
                ids = torch.tensor([input_ids[r] for r in rows_list], dtype=torch.int64, device=self.device)
                if n == 0:
                    raise ValueError("empty prompt")
                nxt = model.prefill(state, rows, ids)
                state.ids[rows] = nxt
                state.seq_lens[rows] = n

            use_graph = graphs is not None and graphs.has(Bb, Lb)
            step = (lambda: graphs.replay(Bb, Lb)) if use_graph else (lambda: model.decode_step(state, Bb, Lb))

            if self.cuda and flag("PIPELINE") and state.ring is not None:
                ring, events = state.ring, self.events
                R = ring.shape[0]
                stream = torch.cuda.current_stream()
                ring[0, :B].copy_(state.ids[:B], non_blocking=True)
                events[0].record(stream)
                for t in range(max_new_tokens):
                    if t + 1 < max_new_tokens:
                        step()
                        slot = (t + 1) % R
                        ring[slot, :B].copy_(state.next_ids[:B], non_blocking=True)
                        events[slot].record(stream)
                    events[t % R].synchronize()
                    yield ring[t % R, :B].tolist()
            else:
                yield state.ids[:B].tolist()
                for _ in range(1, max_new_tokens):
                    step()
                    yield state.next_ids[:B].tolist()
