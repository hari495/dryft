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
from collections import defaultdict, deque

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
        # Speculation needs the fused rope + attention kernels on CUDA (the
        # torch chain path exists for CPU verification only).
        self.spec_ok = flag("SPECULATE") and (
            not self.cuda or (self.model._k_rope is not None and self.model._k_attn is not None)
        )
        self.spec_ratio: dict[tuple[int, int], float] = {}
        if self.cuda:
            self.events = [torch.cuda.Event() for _ in range(s.ring_depth)]
            deadline = t0 + s.init_budget_seconds
            self._warm_cublas()
            if flag("CUDA_GRAPHS"):
                from dryft_qwen3.graphs import DecodeGraphs

                self.graphs = DecodeGraphs(self.model, self.state, s)
                spec_ql = {Bb: k + 1 for Bb, k in s.spec_drafts.items() if k > 0} if self.spec_ok else None
                self.graphs.capture_all(deadline, spec_ql)
                if spec_ql:
                    self._measure_spec_ratio(spec_ql)
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

    def _measure_spec_ratio(self, spec_ql: dict[int, int]) -> None:
        """Break-even tokens/step for speculation per bucket = verify-step time /
        draft-free step time, measured on this GPU with graph replays."""
        g = self.graphs
        for (Bb, Lb, QL), _ in list(g.graphs.items()):
            if QL <= 1 or not g.has(Bb, Lb, 1):
                continue
            t1 = min(g.time_step(Bb, Lb, 1), g.time_step(Bb, Lb, 1))
            tk = min(g.time_step(Bb, Lb, QL), g.time_step(Bb, Lb, QL))
            self.spec_ratio[(Bb, Lb)] = tk / max(t1, 1e-6)
            log(f"spec B={Bb} L={Lb} QL={QL}: step {t1:.2f} ms, verify {tk:.2f} ms -> break-even {tk / t1:.2f} tok/step")

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
        s = self.settings
        Bb = pick_bucket(s.batch_buckets, B)
        # Speculation needs >= 2 steps to pay for anything and room in the
        # cache: rows may run ahead of the slowest one by a few chains.
        K = s.spec_drafts.get(Bb, 0) if (self.spec_ok and max_new_tokens >= 3 and Bb is not None) else 0
        QL = K + 1 if K > 0 else 0
        need_len = max(lens) + max_new_tokens + (2 * QL + 2 if QL else 0)
        Lb = pick_bucket(s.len_buckets, need_len)
        if Bb is not None and Lb is not None and Bb <= self.state.b_cap and Lb <= self.state.l_cap:
            state, graphs = self.state, self.graphs
            if QL and graphs is not None and not graphs.has(Bb, Lb, QL):
                QL = 0  # no verify graph for this bucket: plain decode
        else:
            # Out-of-bucket request: eager path with a right-sized cache.
            log(f"fallback path for B={B}, L={need_len}")
            Bb, Lb = B, need_len
            rope = (self.model.cos, self.model.sin)
            if need_len > self.model.rope_len:
                rope = rope_tables(self.cfg, need_len, self.device)
            state = DecodeState(self.cfg, Bb, Lb, self.device, pinned=self.cuda, rope=rope, ring_depth=s.ring_depth)
            graphs = None
        yield from self._run(state, graphs, Bb, Lb, QL, input_ids, lens, max_new_tokens)

    def _prefill_all(self, state: DecodeState, input_ids, lens, max_new_tokens: int) -> None:
        """Prefill every prompt (grouped by length), seed the speculative
        history and counters. Leaves ``state.ids`` = first generated token."""
        model = self.model
        groups: dict[int, list[int]] = defaultdict(list)
        for r, n in enumerate(lens):
            groups[n].append(r)
        for n, rows_list in groups.items():
            if n == 0:
                raise ValueError("empty prompt")
            rows = torch.tensor(rows_list, dtype=torch.int64, device=self.device)
            ids = torch.tensor([input_ids[r] for r in rows_list], dtype=torch.int64, device=self.device)
            # Fixed batches prefill rows 0..B-1 in order; the fused rope
            # kernel relies on that mapping, ragged groups take the torch path.
            nxt = model.prefill(state, rows, ids, rows_are_prefix=rows_list == list(range(len(rows_list))))
            state.ids[rows] = nxt
            state.seq_lens[rows] = n
            state.hist[rows, :n] = ids
            state.hist[rows, n] = nxt
        B = len(input_ids)
        state.produced[:B].fill_(1)
        state.max_new.fill_(max_new_tokens)
        state.tok_in[:B, 0].copy_(state.ids[:B])

    def _run(self, state: DecodeState, graphs, Bb: int, Lb: int, QL: int, input_ids, lens, max_new_tokens: int):
        B = len(input_ids)
        N = max_new_tokens
        model = self.model
        s = self.settings
        with torch.inference_mode():
            state.reset_rows(Bb)
            self._prefill_all(state, input_ids, lens, N)
            first = state.ids[:B].tolist()  # the one sync prefill needs anyway
            yield first
            if N == 1:
                return

            def launch(ql: int) -> None:
                if graphs is not None and graphs.has(Bb, Lb, ql):
                    graphs.replay(Bb, Lb, ql)
                elif ql == 0:
                    model.decode_step(state, Bb, Lb)
                else:
                    model.verify_step(state, Bb, Lb, ql)

            if QL == 0:
                # ---- plain decode: one token per row per step
                if self.cuda and flag("PIPELINE") and state.ring is not None:
                    ring, events = state.ring, self.events
                    R = ring.shape[0]
                    stream = torch.cuda.current_stream()
                    launch(0)
                    ring[1 % R, :B].copy_(state.next_ids[:B], non_blocking=True)
                    events[1 % R].record(stream)
                    for t in range(1, N):
                        if t + 1 < N:
                            launch(0)
                            slot = (t + 1) % R
                            ring[slot, :B].copy_(state.next_ids[:B], non_blocking=True)
                            events[slot].record(stream)
                        events[t % R].synchronize()
                        yield ring[t % R, :B].tolist()
                else:
                    for _ in range(1, N):
                        launch(0)
                        yield state.next_ids[:B].tolist()
                return

            # ---- speculative: chains of QL tokens, ragged acceptance per row
            model.draft(state, Bb, Lb, QL)  # first drafts from the prompt (eager, once)
            queues = [deque() for _ in range(B)]
            emitted = 1
            ql = QL  # current step kind; drops to 0 if the guard trips
            window_tokens = 0  # new tokens over the guard window (active rows)
            window_rows = 0
            window_steps = 0
            ratio = self.spec_ratio.get((Bb, Lb), 1.15) * s.spec_margin
            produced = [1] * B  # host mirror of state.produced[:B]

            def absorb(tokens: list[list[int]], width: int) -> None:
                nonlocal window_tokens, window_rows, window_steps
                for b in range(B):
                    row = tokens[b]
                    got = 0
                    for j in range(width):
                        t = row[j]
                        if t < 0:
                            break
                        queues[b].append(t)
                        got += 1
                    if produced[b] < N:
                        window_tokens += got
                        window_rows += 1
                    produced[b] += got
                window_steps += 1

            def drain():
                nonlocal emitted
                while emitted < N and all(queues):
                    emitted += 1
                    yield [q.popleft() for q in queues]

            def decide() -> None:
                nonlocal ql, window_tokens, window_rows, window_steps
                if ql != QL or window_steps < s.spec_window:
                    return
                per_row_step = window_tokens / max(1, window_rows)
                if per_row_step < ratio:
                    # Verify costs more than it returns on these prompts: finish on
                    # the draft-free chain graph (same bookkeeping, frozen rows stay put).
                    ql = 1
                window_tokens = window_rows = window_steps = 0

            if self.cuda and flag("PIPELINE") and state.ring_spec is not None:
                ring, events = state.ring_spec, self.events
                R = ring.shape[0]
                stream = torch.cuda.current_stream()
                inflight: deque[tuple[int, int]] = deque()  # (slot, width)
                step_i = 0

                def push() -> None:
                    nonlocal step_i
                    launch(ql)
                    slot = step_i % R
                    width = ql
                    ring[slot, :B].copy_(state.out_step[:B], non_blocking=True)  # contiguous rows
                    events[slot].record(stream)
                    inflight.append((slot, width))
                    step_i += 1

                push()
                while emitted < N:
                    if len(inflight) < 2:  # keep one step in flight behind the one we wait on
                        push()
                    slot, width = inflight.popleft()
                    events[slot].synchronize()
                    absorb(ring[slot, :B, :width].tolist(), width)
                    decide()
                    yield from drain()
            else:
                while emitted < N:
                    launch(ql)
                    absorb(state.out_step[:B, :ql].tolist(), ql)
                    decide()
                    yield from drain()
