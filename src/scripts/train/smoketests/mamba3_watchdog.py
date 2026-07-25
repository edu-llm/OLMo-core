"""
Out-of-process watchdog for a Mamba-3 training run.

The in-process sentinel (:mod:`mamba3_sentinel`) cannot report the failures that matter most on
a leased box, because they are exactly the cases where the trainer stops being able to report
anything:

- the process is **wedged** -- a deadlocked collective, a stuck dataloader worker, or a driver
  hang. ``nvidia-smi`` shows 100% utilization and the step counter never advances. This is the
  single most expensive silent failure on metered hardware.
- the process **died** and nothing noticed, leaving the GPU idle for hours.
- checkpoints are being written locally but **never reaching durable storage**, so everything
  disappears when instance storage goes away.

Safety properties, by construction:

- It **never terminates anything.** There is no ``stop``, ``terminate``, ``reboot``, ``kill``,
  or scaling call anywhere in this file, and no process signalling. It only reads state and
  writes notifications. A watchdog that can stop an instance is a strictly worse failure mode
  than the one it is guarding against.
- The only optional side effect is an HTTP POST to a notification webhook you pass in.
- ``--durable-dir`` is checked for *freshness only*; this script never performs the copy, so it
  cannot itself corrupt or race the thing it is verifying.

Usage::

    python mamba3_watchdog.py --run-dir /mnt/nvme/runs/mine --gpu-index 3 \\
        --durable-dir /mnt/nvme/runs/mine/synced --deadline-utc 2026-07-27T09:00:00

Exit codes: ``0`` if it was asked to stop, ``2`` if it exits while alerts are outstanding.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class Watchdog:
    """Polls local run state and emits notifications. Read-only with respect to the workload."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.alerts_path = self.run_dir / "watchdog-alerts.jsonl"
        self.heartbeat_path = self.run_dir / "heartbeat.json"
        self._last_step: Optional[int] = None
        self._last_step_change = _now()
        self._fired: Dict[str, float] = {}
        self._outstanding = 0
        self._start = _now()
        self._seen_sentinel_lines = 0
        self._deadline: Optional[dt.datetime] = (
            dt.datetime.fromisoformat(args.deadline_utc).replace(tzinfo=dt.timezone.utc)
            if args.deadline_utc
            else None
        )

    # -- notification -------------------------------------------------------------------

    def notify(self, kind: str, message: str, **detail: Any) -> None:
        """Emit an alert, rate-limited per kind so one condition cannot drown out another."""
        last = self._fired.get(kind)
        if last is not None and _now() - last < self.args.renotify_seconds:
            return
        self._fired[kind] = _now()
        self._outstanding += 1

        record = {
            "ts": _now(),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "message": message,
            **detail,
        }
        line = json.dumps(record)
        print(f"[watchdog][ALERT] {kind}: {message}", file=sys.stderr, flush=True)
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.alerts_path.open("a") as f:
                f.write(line + "\n")
        except OSError as exc:
            print(f"[watchdog] could not write alert file: {exc}", file=sys.stderr, flush=True)

        if self.args.webhook:
            try:
                req = urllib.request.Request(
                    self.args.webhook,
                    data=json.dumps({"text": f"[{kind}] {message}"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=10).close()
            except (urllib.error.URLError, OSError, ValueError) as exc:
                print(f"[watchdog] webhook failed: {exc}", file=sys.stderr, flush=True)

    def clear(self, kind: str) -> None:
        if self._fired.pop(kind, None) is not None:
            self._outstanding = max(0, self._outstanding - 1)
            print(f"[watchdog] recovered: {kind}", flush=True)

    # -- individual checks --------------------------------------------------------------

    def check_heartbeat(self) -> Optional[Dict[str, Any]]:
        """Stale heartbeat with the process alive is the wedged-trainer signature."""
        if not self.heartbeat_path.exists():
            # Only complain once the run has had time to start.
            if _now() - self._start > self.args.startup_grace_seconds:
                self.notify(
                    "no_heartbeat",
                    f"no heartbeat at {self.heartbeat_path} after "
                    f"{_fmt_age(_now() - self._start)}; is the sentinel callback attached?",
                )
            return None
        try:
            beat = json.loads(self.heartbeat_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None  # mid-write; the sentinel writes atomically so this is transient

        age = _now() - float(beat.get("ts", 0))
        if age > self.args.stale_seconds:
            self.notify(
                "stale_heartbeat",
                f"heartbeat is {_fmt_age(age)} old at step {beat.get('step')} "
                f"(status={beat.get('status')}). The trainer is not stepping.",
                age_seconds=round(age),
                step=beat.get("step"),
            )
        else:
            self.clear("stale_heartbeat")

        status = beat.get("status")
        if status == "error":
            self.notify("trainer_error", f"trainer reported an error: {beat.get('error')}")
        return beat

    def check_step_progress(self, beat: Optional[Dict[str, Any]]) -> None:
        """A heartbeat that keeps ticking while the step number never moves is still a hang."""
        if not beat:
            return
        step = beat.get("step")
        if not isinstance(step, int) or step < 0:
            return
        if self._last_step is None or step != self._last_step:
            self._last_step = step
            self._last_step_change = _now()
            self.clear("no_step_progress")
            return
        stalled = _now() - self._last_step_change
        if stalled > self.args.stale_seconds and beat.get("status") == "training":
            self.notify(
                "no_step_progress",
                f"step counter stuck at {step} for {_fmt_age(stalled)} while status is still "
                f"'training'. GPU utilization alone will not reveal this.",
                step=step,
                stalled_seconds=round(stalled),
            )

    def check_sentinel_alerts(self) -> None:
        """Surface anything the in-process sentinel recorded."""
        path = self.run_dir / "alerts.jsonl"
        if not path.exists():
            return
        try:
            lines = path.read_text().strip().splitlines()
        except OSError:
            return
        for line in lines[self._seen_sentinel_lines :]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("critical"):
                self.notify(
                    f"sentinel_{rec.get('kind', 'unknown')}",
                    f"[step {rec.get('step')}] {rec.get('message')}",
                )
        self._seen_sentinel_lines = len(lines)

    def check_gpu(self) -> None:
        """
        Cross-check utilization against progress on *our* GPU only.

        Deliberately scoped to a single index: other people's processes on the other GPUs are
        none of this watchdog's business, and reporting on them invites someone to act on it.
        """
        if self.args.gpu_index is None or not shutil.which("nvidia-smi"):
            return
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.args.gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
            util_s, mem_s = (p.strip() for p in out.split(","))
            util, mem = int(util_s), int(mem_s)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            print(f"[watchdog] nvidia-smi unavailable: {exc}", flush=True)
            return

        stalled = _now() - self._last_step_change
        if util > 80 and stalled > self.args.stale_seconds:
            self.notify(
                "gpu_busy_no_progress",
                f"GPU {self.args.gpu_index} is at {util}% ({mem} MiB) but the step counter has "
                f"not moved in {_fmt_age(stalled)}. This is burning the lease for nothing.",
                utilization=util,
            )
        elif util < 5 and self._last_step is not None:
            self.notify(
                "gpu_idle",
                f"GPU {self.args.gpu_index} is idle ({util}%, {mem} MiB) but a run was active. "
                f"The trainer may have exited.",
                utilization=util,
            )
        else:
            self.clear("gpu_busy_no_progress")
            self.clear("gpu_idle")

    def check_durability(self) -> None:
        """
        Local checkpoints that never reach durable storage are the highest-consequence silent
        failure on a leased box: everything looks fine right up to the moment the disk is gone.
        """
        if not self.args.durable_dir:
            return
        durable = Path(self.args.durable_dir)
        newest = 0.0
        if durable.exists():
            for p in durable.rglob("*"):
                if p.is_file():
                    newest = max(newest, p.stat().st_mtime)
        age = _now() - newest if newest else float("inf")
        if age > self.args.durable_max_age_seconds:
            human = "never" if newest == 0 else _fmt_age(age)
            self.notify(
                "durable_copy_stale",
                f"durable copy at {durable} was last updated {human} ago. Anything not copied "
                f"off local instance storage is lost when the lease ends.",
                age_seconds=None if newest == 0 else round(age),
            )
        else:
            self.clear("durable_copy_stale")

    def check_deadline(self) -> None:
        """Warn as the lease deadline approaches, while there is still time to evacuate."""
        if not self._deadline:
            return
        remaining = (self._deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
        for threshold, label in ((0, "PASSED"), (1800, "30m"), (3600, "1h"), (7200, "2h")):
            if remaining <= threshold:
                self.notify(
                    f"deadline_{label}",
                    f"{'DEADLINE PASSED' if threshold == 0 else label + ' until deadline'} "
                    f"({remaining / 3600:+.1f}h). Confirm everything you need is in durable "
                    f"storage.",
                    remaining_hours=round(remaining / 3600, 2),
                )
                break

    # -- loop ---------------------------------------------------------------------------

    def run(self) -> int:
        self._start = _now()
        print(
            f"[watchdog] watching {self.run_dir} every {self.args.interval_seconds}s "
            f"(gpu={self.args.gpu_index}); this process never stops or terminates anything",
            flush=True,
        )
        try:
            while True:
                beat = self.check_heartbeat()
                self.check_step_progress(beat)
                self.check_sentinel_alerts()
                self.check_gpu()
                self.check_durability()
                self.check_deadline()
                self._print_status(beat)
                time.sleep(self.args.interval_seconds)
        except KeyboardInterrupt:
            print("[watchdog] stopping on request", flush=True)
            return 0 if not self._outstanding else 2

    def _print_status(self, beat: Optional[Dict[str, Any]]) -> None:
        """Periodic ETA line, so a long run reports progress rather than going quiet."""
        if not beat:
            return
        parts = [
            dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ"),
            f"step={beat.get('step')}",
            f"status={beat.get('status')}",
        ]
        if (loss := beat.get("recent_loss")) is not None:
            parts.append(f"loss={loss:.4f}")
        if (rate := beat.get("recent_skip_rate")) is not None:
            parts.append(f"skip_rate={rate:.0%}")
        if self._deadline:
            left = (self._deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
            parts.append(f"deadline_in={left:.1f}h")
        parts.append(f"alerts={self._outstanding}")
        print("[watchdog] " + "  ".join(parts), flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-dir", required=True, help="directory holding heartbeat.json")
    p.add_argument("--gpu-index", type=int, default=None, help="the single GPU index you claimed")
    p.add_argument("--durable-dir", default=None, help="local mirror of what has been shipped off")
    p.add_argument("--deadline-utc", default=None, help="ISO8601 UTC lease deadline")
    p.add_argument("--webhook", default=None, help="optional JSON POST endpoint")
    p.add_argument("--interval-seconds", type=int, default=60)
    p.add_argument("--stale-seconds", type=int, default=900)
    p.add_argument("--startup-grace-seconds", type=int, default=600)
    p.add_argument("--durable-max-age-seconds", type=int, default=3600)
    p.add_argument("--renotify-seconds", type=int, default=1800)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    return Watchdog(parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
