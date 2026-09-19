"""CUDA-graph capture/replay of the decode and speculative steps per bucket.

Keys are ``(Bb, Lb, QL)``: ``QL == 0`` is ``Model.decode_step`` (plain one
token per row), ``QL >= 1`` is ``Model.verify_step`` with a chain of ``QL``
tokens per row (``QL == 1`` = no drafts but with the speculative bookkeeping,
used after the guard turns speculation off mid-generation).

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
        self.graphs: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()

    def has(self, Bb: int, Lb: int, QL: int = 0) -> bool:
        return (Bb, Lb, QL) in self.graphs

    def replay(self, Bb: int, Lb: int, QL: int = 0) -> None:
        self.graphs[(Bb, Lb, QL)].replay()

    def capture_all(self, deadline: float, spec_ql: dict[int, int] | None = None) -> None:
        """Capture the plain step for every bucket and, where ``spec_ql[Bb] > 1``,
        the chain graphs ``QL = spec_ql[Bb]`` and ``QL = 1`` (the guard's
        fallback) as well."""
        t0 = time.time()
        n_ok = 0
        for Lb in self.settings.len_buckets:
            if Lb > self.state.l_cap:
                continue
            for Bb in self.settings.batch_buckets:
                if Bb > self.state.b_cap:
                    continue
                ql = (spec_ql or {}).get(Bb, 0)
                kinds = (0, 1, ql) if ql > 1 else (0,)
                for QL in kinds:
                    if time.time() > deadline:
                        log(f"graph capture stopped at bucket (B={Bb}, L={Lb}, QL={QL}): init budget reached")
                        return
                    if self._capture_one(Bb, Lb, QL):
                        n_ok += 1
        log(f"captured {n_ok} graphs in {time.time() - t0:.1f}s")

    def time_step(self, Bb: int, Lb: int, QL: int, reps: int = 8) -> float:
        """Mean ms per replay of graph ``(Bb, Lb, QL)`` on seeded inputs whose
        positions leave room for ``reps`` chains."""
        st = self.state
        gen = torch.Generator().manual_seed(99 + Bb + Lb + QL)
        self._seed_inputs(Bb, Lb, QL, gen)
        st.seq_lens[:Bb].clamp_(max=max(1, Lb // 2))
        if QL > 0:
            st.produced[:Bb].zero_()
            st.max_new.fill_(10**6)
        g = self.graphs[(Bb, Lb, QL)]
        g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps

    # ------------------------------------------------------------------ internals
    def _step(self, Bb: int, Lb: int, QL: int) -> None:
        if QL == 0:
            self.model.decode_step(self.state, Bb, Lb)
        else:
            self.model.verify_step(self.state, Bb, Lb, QL)

    def _seed_inputs(self, Bb: int, Lb: int, QL: int, gen: torch.Generator) -> None:
        """Random but valid state: positions leave room for the chain, the
        history holds a repetitive pattern so the drafts sometimes hit."""
        st = self.state
        vocab = self.model.cfg.vocab
        span = max(2, Lb - 4 - 2 * max(QL, 1))
        lens = torch.randint(1, span, (Bb,), generator=gen, dtype=torch.int64)
        ids = torch.randint(0, vocab, (Bb,), generator=gen, dtype=torch.int64)
        st.seq_lens[:Bb].copy_(lens.to(st.device))
        st.ids[:Bb].copy_(ids.to(st.device))
        if QL > 0:
            hist = torch.randint(0, 50, (Bb, st.hist.shape[1]), generator=gen, dtype=torch.int64)
            st.hist[:Bb].copy_(hist.to(st.device))
            st.hist[:Bb].scatter_(1, st.seq_lens[:Bb, None], st.ids[:Bb, None])
            st.tok_in[:Bb, 0].copy_(st.ids[:Bb])
            st.tok_in[:Bb, 1:QL].copy_(torch.randint(0, 50, (Bb, QL - 1), generator=gen, dtype=torch.int64).to(st.device))
            st.produced[:Bb].copy_(torch.randint(1, 8, (Bb,), generator=gen, dtype=torch.int64).to(st.device))
            st.max_new.fill_(6)  # some rows frozen, some active

    def _capture_one(self, Bb: int, Lb: int, QL: int) -> bool:
        st = self.state
        gen = torch.Generator().manual_seed(1234 + Bb * 131 + Lb + 7 * QL)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._seed_inputs(Bb, Lb, QL, gen)
                self._step(Bb, Lb, QL)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self._seed_inputs(Bb, Lb, QL, gen)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self.pool):
                self._step(Bb, Lb, QL)
        except Exception as e:  # capture failure: run this bucket eager
            torch.cuda.synchronize()
            log(f"graph capture FAILED for (B={Bb}, L={Lb}, QL={QL}): {e!r}")
            return False
        torch.cuda.synchronize()

        if self.settings.graph_validate and not self._validate(g, Bb, Lb, QL, gen):
            log(f"graph validation FAILED for (B={Bb}, L={Lb}, QL={QL}); bucket runs eager")
            return False
        self.graphs[(Bb, Lb, QL)] = g
        return True

    def _snapshot(self, Bb: int, QL: int) -> list[torch.Tensor]:
        st = self.state
        outs = [st.seq_lens[:Bb], st.ids[:Bb]]
        if QL == 0:
            outs.append(st.next_ids[:Bb])
        else:
            outs += [st.out_step[:Bb, :QL], st.n_new[:Bb], st.produced[:Bb], st.tok_in[:Bb, :QL], st.hist[:Bb]]
        return [t.clone() for t in outs]

    def _validate(self, g: torch.cuda.CUDAGraph, Bb: int, Lb: int, QL: int, gen: torch.Generator) -> bool:
        st = self.state
        ok = True
        for _ in range(2):
            self._seed_inputs(Bb, Lb, QL, gen)
            before = self._snapshot(Bb, QL)
            self._step(Bb, Lb, QL)
            torch.cuda.synchronize()
            want = self._snapshot(Bb, QL)
            # restore the inputs exactly, replay, compare every output
            st.seq_lens[:Bb].copy_(before[0])
            st.ids[:Bb].copy_(before[1])
            if QL == 0:
                st.next_ids[:Bb].zero_()
            else:
                st.tok_in[:Bb, :QL].copy_(before[5])
                st.produced[:Bb].copy_(before[4])
                st.hist[:Bb].copy_(before[6])
                st.out_step[:Bb, :QL].fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            got = self._snapshot(Bb, QL)
            ok &= all(bool(torch.equal(a, b)) for a, b in zip(want, got))
        return ok
