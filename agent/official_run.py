"""Push the working tree to the connected branch, wait for the official run,
print its metrics.

There is no local H100, and this round only accepts ``mode=official`` runs,
so the platform *is* the benchmark. One call = one official run:

    python agent/official_run.py                 # push HEAD's working tree, wait, report
    python agent/official_run.py --run-id ID     # just report an existing run
    python agent/official_run.py --no-push       # wait for the newest run on the current tree

The working tree is snapshotted as a commit on top of ``origin/<branch>``
(the local branches are untouched), pushed, and the run the platform starts
for that commit is polled to a terminal state. Output lines:

    METRIC score=405.0          official geomean over hidden workloads (native = 100)
    METRIC tps_b1_512_32=...    public shapes, tok/s
    METRIC ttft_ratio_max=...   worst ours/native TTFT over public shapes (gate 1.10)
    METRIC tpot_ratio_max=...   worst ours/native TPOT (gate 1.10)
    METRIC spread_max_pct=...   worst (p90 - p10) / p50 over public shapes
    METRIC peak_mem_gib=...
    ASI run_id=... commit=... state=...

Exit status is 0 only when the run succeeded and produced a score.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import Dryft, TERMINAL  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args: str, env: dict | None = None) -> str:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.check_output(["git", *args], cwd=REPO, env=merged, text=True).strip()


def snapshot_and_push(branch: str, message: str) -> str:
    """Commit the working tree (tracked + untracked, .gitignore honoured) on
    top of ``origin/<branch>`` and push it. Returns the commit sha."""
    git("fetch", "-q", "origin", branch)
    parent = git("rev-parse", f"origin/{branch}")
    with tempfile.TemporaryDirectory() as tmp:
        index = os.path.join(tmp, "index")
        env = {"GIT_INDEX_FILE": index}
        git("read-tree", "HEAD", env=env)
        git("add", "-A", ".", env=env)
        tree = git("write-tree", env=env)
    if tree == git("rev-parse", f"{parent}^{{tree}}"):
        print(f"[official_run] tree identical to origin/{branch}; pushing a fresh commit anyway", file=sys.stderr)
    commit = git("commit-tree", tree, "-p", parent, "-m", message)
    git("push", "-q", "origin", f"{commit}:refs/heads/{branch}")
    print(f"[official_run] pushed {commit[:10]} -> origin/{branch} (parent {parent[:10]})", file=sys.stderr)
    return commit


def default_message() -> str:
    """``autoresearch: <HEAD> (+dirty) FLAGS: A=1 B=0`` — the flag set is the
    experiment's identity when the tree is pushed uncommitted."""
    head = git("rev-parse", "--short", "HEAD")
    dirty = "+dirty" if git("status", "--porcelain", "--", "engine") else ""
    flags = []
    with open(os.path.join(REPO, "engine", "dryft_qwen3", "config.py"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*"([A-Z_]+)":\s*(True|False),', line)
            if m:
                flags.append(f"{m.group(1)}={int(m.group(2) == 'True')}")
    return f"autoresearch: {head}{dirty} FLAGS: {' '.join(flags)}"


def find_run(api: Dryft, commit: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        for item in api.runs():
            if item.get("commitSha") == commit:
                return item
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no run appeared for {commit[:10]} within {timeout:.0f}s")
        time.sleep(5)


def wait_run(api: Dryft, run_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        current = api.run(run_id)
        state = current.get("state")
        if state != last:
            print(f"[official_run] {run_id[:8]} {state} at {time.strftime('%H:%M:%S')}", file=sys.stderr)
            last = state
        if state in TERMINAL:
            return current
        if time.monotonic() >= deadline:
            raise TimeoutError(f"run {run_id} still {state} after {timeout:.0f}s")
        time.sleep(15)


def shape_key(shape: dict, batch: dict | None) -> str:
    """``b1_512_32`` from the challenge's public shape list, else the case id."""
    if batch:
        return f"b{batch['batch']}_{batch['prompt']}_{batch['output']}"
    return shape["id"].replace("-", "_")


def report(run: dict, public_shapes: list[dict]) -> int:
    result = run.get("result") or {}
    state = run.get("state")
    asi = {
        "run_id": run.get("id"),
        "commit": (run.get("commitSha") or "")[:10],
        "state": state,
        "failure": result.get("failureCode") or run.get("errorCode") or "none",
    }
    shapes = result.get("shapes") or []
    ttft_ratio = tpot_ratio = spread = mem = 0.0
    public_tps: list[float] = []
    for i, shape in enumerate(shapes):
        mm = shape.get("modelMetrics") or {}
        key = shape_key(shape, public_shapes[i] if i < len(public_shapes) else None)
        tps = shape.get("tokensPerSecond")
        status = shape.get("caseStatus")
        line = f"  {key:<16} {status:<8}"
        if tps:
            public_tps.append(tps)
            print(f"METRIC tps_{key}={tps:.1f}")
            line += f" {tps:8.1f} tok/s  total {shape['metricMs']:.0f} ms (native {mm.get('referenceMs', 0):.0f})"
        if mm.get("referenceTtftMs"):
            r = mm["ttftMs"] / mm["referenceTtftMs"]
            ttft_ratio = max(ttft_ratio, r)
            line += f"  ttft {mm['ttftMs']:.0f} ms ({r:.2f}x)"
        if mm.get("referenceTpotMs"):
            r = mm["tpotMs"] / mm["referenceTpotMs"]
            tpot_ratio = max(tpot_ratio, r)
            line += f"  tpot {mm['tpotMs']:.2f} ms ({r:.2f}x)"
        if shape.get("p50Ms"):
            s = (shape["p90Ms"] - shape["p10Ms"]) / shape["p50Ms"] * 100
            spread = max(spread, s)
            line += f"  spread {s:.1f}%"
        if shape.get("peakMemoryBytes"):
            mem = max(mem, shape["peakMemoryBytes"] / 2**30)
        if shape.get("caseMessage"):
            line += f"  :: {shape['caseMessage']}"
        print(line, file=sys.stderr)

    score = result.get("score")
    if public_tps:
        print(f"METRIC public_geomean={math.exp(sum(map(math.log, public_tps)) / len(public_tps)):.1f}")
        print(f"METRIC ttft_ratio_max={ttft_ratio:.3f}")
        print(f"METRIC tpot_ratio_max={tpot_ratio:.3f}")
        print(f"METRIC spread_max_pct={spread:.2f}")
        print(f"METRIC peak_mem_gib={mem:.1f}")
    if score is not None:
        print(f"METRIC score={score:.2f}")
    print("ASI " + " ".join(f"{k}={v}" for k, v in asi.items()))
    if result.get("failureMessage"):
        print(f"[official_run] failure: {result['failureMessage']}", file=sys.stderr)
    if run.get("errorMessage"):
        print(f"[official_run] error: {run['errorMessage']}", file=sys.stderr)
    return 0 if (state == "succeeded" and score is not None) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="main", help="connected branch that triggers official runs")
    ap.add_argument("--message", default=None, help="commit message for the snapshot")
    ap.add_argument("--run-id", default=None, help="report an existing run instead of pushing")
    ap.add_argument("--no-push", action="store_true", help="wait for the newest run instead of pushing")
    ap.add_argument("--timeout", type=float, default=2700, help="seconds to wait for a terminal state")
    args = ap.parse_args()

    api = Dryft()
    try:
        public_shapes = api.benchmark().get("publicShapes") or []
    except Exception as exc:  # noqa: BLE001 - labels are cosmetic
        print(f"[official_run] benchmark definition unavailable: {exc}", file=sys.stderr)
        public_shapes = []

    if args.run_id:
        run = wait_run(api, args.run_id, args.timeout)
    elif args.no_push:
        items = api.runs()
        if not items:
            print("[official_run] no runs exist", file=sys.stderr)
            return 1
        run = wait_run(api, items[0]["id"], args.timeout)
    else:
        message = args.message or os.environ.get("AUTORESEARCH_DESC") or default_message()
        commit = snapshot_and_push(args.branch, message)
        run = find_run(api, commit, timeout=300)
        run = wait_run(api, run["id"], args.timeout)
    return report(run, public_shapes)


if __name__ == "__main__":
    sys.exit(main())
