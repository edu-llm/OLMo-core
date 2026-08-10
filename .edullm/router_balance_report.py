"""Turn an arm's training log into the curve the decision needs, on the machine that produced it.

    python .edullm/router_balance_report.py /scratch/rb/rb-g1e4.log [...]

WHY THIS RUNS IN THE CONTAINER RATHER THAN ON A LAPTOP. Six arms of up to 1,500 steps produce a
log far too long to read back through `Block: read a run's log`, which tails a fixed number of
lines, and the machine that could read the whole thing out of S3 is the one nobody on this lane
holds a credential for. Summarising in place means the last few hundred lines of the container's
own log carry every number in the report.

WHAT IT READS. `SpeedMonitorCallback` records `throughput/device/BPS`, which is 1/step_time, and
`throughput/device/MFU`. `MoERouter.compute_metrics` records `train/block N/load imbalance` for
each of the sixteen blocks, as max-over-experts divided by mean-over-experts of the token counts
that block routed, reduced with `ReduceType.max` across ranks. The evaluator records held-out CE
loss under a key ending `/CE loss` that does not begin `train/`.

WHY A TRAILING WINDOW AND NOT A WHOLE-ARM AVERAGE. Every arm here is deliberately measured while
a controller is still moving, so a mean over all steps is a mean over a transient and says
nothing about where the arm ended up. The steady-state row is the last `--window` steps before
the arm stopped; the curve above it is what says whether that row is a plateau or a waypoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STEP = re.compile(r"\[step=(\d+)/")
METRIC = re.compile(r"^\s+(\S.*?)=(\S+)\s*$")
IMBALANCE = re.compile(r"^train/block (\d+)/load imbalance$")


def _number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


class Record:
    __slots__ = ("step", "metrics")

    def __init__(self, step: int) -> None:
        self.step = step
        self.metrics: Dict[str, float] = {}

    @property
    def seconds_per_step(self) -> Optional[float]:
        bps = self.metrics.get("throughput/device/BPS")
        return None if not bps else 1.0 / bps

    @property
    def mfu(self) -> Optional[float]:
        return self.metrics.get("throughput/device/MFU")

    def imbalance(self) -> Optional[Tuple[float, float, float]]:
        """Mean, min and max of the sixteen per-block load imbalances."""
        values = [v for k, v in self.metrics.items() if IMBALANCE.match(k)]
        if not values:
            return None
        return statistics.fmean(values), min(values), max(values)


def parse(path: Path) -> Tuple[List[Record], List[Tuple[int, float]], Optional[dict]]:
    records: List[Record] = []
    val: List[Tuple[int, float]] = []
    summary: Optional[dict] = None
    current: Optional[Record] = None

    for raw in path.read_text(errors="replace").splitlines():
        found = STEP.search(raw)
        if found:
            current = Record(int(found.group(1)))
            records.append(current)
            continue
        if raw.lstrip().startswith("{") and '"run_id"' in raw:
            with_json = raw[raw.index("{") :]
            try:
                summary = json.loads(with_json)
            except ValueError:
                pass
            continue
        if current is None:
            continue
        matched = METRIC.match(raw)
        if not matched:
            continue
        name, text = matched.group(1), matched.group(2)
        value = _number(text)
        if value is None or math.isnan(value):
            continue
        current.metrics[name] = value
        if name.endswith("/CE loss") and not name.startswith("train/"):
            val.append((current.step, value))

    return [r for r in records if r.metrics], val, summary


def _window(records: List[Record], last: int, width: int) -> Dict[str, Optional[float]]:
    chosen = [r for r in records if last - width < r.step <= last and r.seconds_per_step]
    if not chosen:
        return {"n": 0, "sps": None, "mfu": None, "imb": None}
    imbalances = [r.imbalance()[0] for r in chosen if r.imbalance()]
    return {
        "n": len(chosen),
        "sps": statistics.fmean([r.seconds_per_step for r in chosen]),
        "mfu": statistics.fmean([r.mfu for r in chosen if r.mfu is not None]),
        "imb": statistics.fmean(imbalances) if imbalances else None,
        "sps_sd": statistics.stdev([r.seconds_per_step for r in chosen])
        if len(chosen) > 1
        else 0.0,
    }


def report(path: Path, every: int, width: int) -> Optional[Dict[str, object]]:
    records, val, summary = parse(path)
    if not records:
        print(f"\n### {path.name}: no metric records found")
        return None

    last = max(r.step for r in records)
    print(f"\n### {path.name}   {last} steps")

    print("  step |  s/step |    MFU |  imbalance (mean [min-max] over 16 blocks)")
    for mark in range(every, last + 1, every):
        near = [r for r in records if mark - every < r.step <= mark and r.seconds_per_step]
        if not near:
            continue
        sps = statistics.fmean([r.seconds_per_step for r in near])
        mfus = [r.mfu for r in near if r.mfu is not None]
        imbs = [r.imbalance() for r in near if r.imbalance()]
        mean_imb = statistics.fmean([i[0] for i in imbs]) if imbs else float("nan")
        lo = min(i[1] for i in imbs) if imbs else float("nan")
        hi = max(i[2] for i in imbs) if imbs else float("nan")
        print(
            f"  {mark:>4} | {sps:>7.3f} | {statistics.fmean(mfus) if mfus else float('nan'):>5.2f}% |"
            f"  {mean_imb:>5.2f} [{lo:.2f}-{hi:.2f}]"
        )

    steady = _window(records, last, width)
    print(
        f"  steady state, last {width} steps (n={steady['n']}): "
        f"{steady['sps']:.4f} s/step +/- {steady['sps_sd']:.4f}, "
        f"{steady['mfu']:.2f}% MFU, imbalance {steady['imb']:.3f}"
        if steady["n"]
        else "  steady state: no records in the window"
    )
    if val:
        print("  held-out CE loss: " + ", ".join(f"step {s}: {v:.4f}" for s, v in val))
    else:
        print("  held-out CE loss: none recorded")
    if summary:
        print(
            f"  summary json: val_loss={summary.get('val_loss')} "
            f"train_loss_last={summary.get('train_loss_last')} run_id={summary.get('run_id')}"
        )

    return {
        "arm": path.stem,
        "steps": last,
        "steady": steady,
        "val": val,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--every", type=int, default=50, help="Curve granularity, in steps.")
    parser.add_argument("--window", type=int, default=100, help="Trailing steady-state width.")
    parser.add_argument("--json", action="store_true")
    opts = parser.parse_args()

    out = []
    for path in opts.logs:
        if not path.exists():
            print(f"\n### {path}: missing")
            continue
        got = report(path, opts.every, opts.window)
        if got:
            out.append(got)

    if len(out) > 1:
        anchor = next((o for o in out if "anchor" in str(o["arm"])), None)
        print("\n### against the anchor")
        print("  arm                  s/step     MFU   imbalance   vs anchor   held-out CE")
        for entry in out:
            steady = entry["steady"]
            if not steady["n"]:
                continue
            if anchor and anchor["steady"]["n"] and anchor is not entry:
                gain = 100 * (anchor["steady"]["sps"] / steady["sps"] - 1)
                against = f"{gain:+.1f}%"
            else:
                against = "--"
            last_val = entry["val"][-1][1] if entry["val"] else float("nan")
            print(
                f"  {str(entry['arm']):<18} {steady['sps']:>7.3f}  {steady['mfu']:>5.2f}%"
                f"   {steady['imb']:>7.3f}   {against:>9}   {last_val:>10.4f}"
            )

    if opts.json:
        print("\nROUTER_BALANCE_JSON " + json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
