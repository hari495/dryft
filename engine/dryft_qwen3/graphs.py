"""CUDA-graph capture/replay of ``Model.decode_step`` per (batch, length) bucket.

Recipe (AGENTS.md §9.2): warm the exact callable 3x on a side stream, sync,
capture into one shared memory pool, then validate every graph by replaying
it on inputs whose eager result is known. A graph that fails validation is
dropped and its bucket runs eager; the engine stays correct either way.
"""

from __future__ import annotations

import time

import torch

from .cache import DecodeState
from .config import Settings, log
from .model import Model


class DecodeGraphs:
    def __init__(self, model: Model, state: DecodeState, settings: Settings):
        self.model = model
        self.state = state
        self.settings = settings
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def has(self, Bb: int, Lb: int) -> bool:
        return (Bb, Lb) in self.graphs

    def replay(self, Bb: int, Lb: int) -> None:
        self.graphs[(Bb, Lb)].replay()

    def capture_all(self, deadline: float) -> None:
        t0 = time.time()
        n_ok = 0
        for Lb in self.settings.len_buckets:
            if Lb > self.state.l_cap:
                continue
            for Bb in self.settings.batch_buckets:
                if Bb > self.state.b_cap:
                    continue
                if time.time() > deadline:
                    log(f"graph capture stopped at bucket (B={Bb}, L={Lb}): init budget reached")
                    return
                if self._capture_one(Bb, Lb):
                    n_ok += 1
        log(f"captured {n_ok} decode graphs in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ internals
    def _seed_inputs(self, Bb: int, Lb: int, gen: torch.Generator) -> None:
        st = self.state
        vocab = self.model.cfg.vocab
        lens = torch.randint(1, max(2, Lb - 4), (Bb,), generator=gen, dtype=torch.int64)
        ids = torch.randint(0, vocab, (Bb,), generator=gen, dtype=torch.int64)
        st.seq_lens[:Bb].copy_(lens.to(st.device))
        st.ids[:Bb].copy_(ids.to(st.device))

    def _capture_one(self, Bb: int, Lb: int) -> bool:
        st, model = self.state, self.model
        gen = torch.Generator().manual_seed(1234 + Bb * 131 + Lb)
        self._seed_inputs(Bb, Lb, gen)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._seed_inputs(Bb, Lb, gen)
                model.decode_step(st, Bb, Lb)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self._seed_inputs(Bb, Lb, gen)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self.pool):
                model.decode_step(st, Bb, Lb)
        except Exception as e:  # capture failure: run this bucket eager
            torch.cuda.synchronize()
            log(f"graph capture FAILED for (B={Bb}, L={Lb}): {e!r}")
            return False
        torch.cuda.synchronize()

        if self.settings.graph_validate and not self._validate(g, Bb, Lb, gen):
            log(f"graph validation FAILED for (B={Bb}, L={Lb}); bucket runs eager")
            return False
        self.graphs[(Bb, Lb)] = g
        return True

    def _validate(self, g: torch.cuda.CUDAGraph, Bb: int, Lb: int, gen: torch.Generator) -> bool:
        st, model = self.state, self.model
        ok = True
        for _ in range(2):
            self._seed_inputs(Bb, Lb, gen)
            lens = st.seq_lens[:Bb].clone()
            ids = st.ids[:Bb].clone()
            model.decode_step(st, Bb, Lb)
            torch.cuda.synchronize()
            want = st.next_ids[:Bb].clone()
            want_len = st.seq_lens[:Bb].clone()
            st.seq_lens[:Bb].copy_(lens)
            st.ids[:Bb].copy_(ids)
            st.next_ids[:Bb].zero_()
            g.replay()
            torch.cuda.synchronize()
            ok &= bool(torch.equal(want, st.next_ids[:Bb])) and bool(torch.equal(want_len, st.seq_lens[:Bb]))
        return ok
