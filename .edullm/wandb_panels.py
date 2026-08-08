#!/usr/bin/env python3
"""Check that a hyper-connection run logged what the pre-registration needs, and build the view.

Two modes, and the first is the one that matters:

``--verify``  compares the keys a run actually logged against the keys each part of the
              pre-registration rests on, and names what is missing. A missing key is not a
              cosmetic problem here: the decision rule is "measure sigma from the baseline's
              three seeds, claim nothing under 2 sigma, report downstream alongside BPB", and
              every clause of that is a metric that either arrived or did not.

``--report``  builds a W&B report over an experiment group, with one panel section per
              question the module asks. Panels are only created for keys the runs actually
              have, because a panel over an invented key renders an empty chart that looks
              like a flat result.

    python .edullm/wandb_panels.py --verify --group hyper-connections-370m
    python .edullm/wandb_panels.py --report --group hyper-connections-370m
"""

import argparse
import fnmatch
import os
import re
from typing import Dict, List, Sequence, Tuple

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: What each part of the module needs, and why. ``required`` means a missing key blocks a
#: claim rather than costing a chart.
EXPECTED: List[Tuple[str, str, bool, Sequence[str]]] = [
    (
        "the guard",
        "Identical lane norms mean the mechanism is inert, and then no downstream number is "
        "interpretable in either direction.",
        True,
        ["hc/min lane norm spread", "hc/block */lane norm spread", "hc/block */lane * norm"],
    ),
    (
        "stability",
        "Parcae found diverging runs learn a spectral radius at or above 1; mHC's claim is "
        "that the Birkhoff constraint keeps the composite across depth well conditioned.",
        True,
        [
            "hc/block */rho(A_r) attention",
            "hc/block */rho(A_r) feed_forward",
            "hc/composite condition number",
            "hc/composite spectral radius",
        ],
    ),
    (
        "hidden-state scale",
        "RMSNorm readouts are scale-invariant, so cross-entropy cannot see hidden-state scale "
        "at all and a stack can drive norms into the thousands invisibly.",
        True,
        ["hc/block */hidden norm"],
    ),
    (
        "the result",
        "The noise floor and every hypothesis are stated on a held-out metric. Training loss "
        "will not do: --seed moves the shuffle, so its variance across seeds is partly a "
        "different sample of the corpus.",
        True,
        ["eval/*/CE loss", "eval/*/BPB", "train/CE loss"],
    ),
    (
        "throughput and cost",
        "What fills the cost table, and what says whether the arms are really iso-FLOP.",
        True,
        [
            "throughput/device/MFU",
            "throughput/device/TPS",
            "throughput/total tokens",
        ],
    ),
    (
        "the weight-decay split",
        "Arm 5 differs from arm 2 only in the optimizer, so separate LR groups are the "
        "cheapest visible confirmation that the split is actually on.",
        False,
        ["optim/LR (group 0)", "optim/LR (group 1)"],
    ),
    (
        "downstream",
        "The decision rule says report downstream alongside BPB. Not produced in-loop by "
        "design -- the downstream evaluator fetches from the public internet, which does not "
        "belong inside a run whose claim is that it read a sealed corpus. Expected to arrive "
        "from a separate job over saved checkpoints.",
        False,
        ["eval/downstream/*"],
    ),
]


def observed_keys(entity: str, project: str, group: str) -> Dict[str, List[str]]:
    """
    Collect the metric keys each run in a group logged.

    :returns: Run display name -> sorted metric keys.
    """
    import wandb

    api = wandb.Api(timeout=120)
    runs = api.runs(f"{entity}/{project}", filters={"group": group}, per_page=100)[:100]
    return {
        run.name: sorted(k for k in run.summary_metrics if not k.startswith("_")) for run in runs
    }


def matched(keys: Sequence[str], pattern: str) -> List[str]:
    return [k for k in keys if fnmatch.fnmatch(k, pattern)]


def verify(entity: str, project: str, group: str) -> int:
    """
    Print the expected-versus-observed table.

    :returns: A process exit status: non-zero when a required family is missing entirely, so
        this can gate a submission rather than only inform one.
    """
    per_run = observed_keys(entity, project, group)
    if not per_run:
        print(f"No runs in {entity}/{project} with group '{group}'.")
        return 2

    keys = sorted({k for ks in per_run.values() for k in ks})
    print(f"{entity}/{project}  group={group}")
    print(f"{len(per_run)} run(s), {len(keys)} distinct metric keys\n")

    missing_required = 0
    for section, why, required, patterns in EXPECTED:
        print(f"[{section}]")
        for pattern in patterns:
            hits = matched(keys, pattern)
            mark = "ok " if hits else ("MISSING" if required else "absent ")
            detail = f"{len(hits):3d} key(s)" if hits else "none"
            print(f"   {mark:8s} {pattern:34s} {detail}")
            if not hits and required:
                missing_required += 1
        if not all(matched(keys, p) for p in patterns):
            print(f"   why it matters: {why}")
        print()

    print(
        "VERDICT: everything the pre-registration rests on is present."
        if not missing_required
        else f"VERDICT: {missing_required} required metric family/families missing. "
        "Claims resting on them cannot be made from these runs."
    )
    return 0 if not missing_required else 1


def _as_regex(pattern: str) -> str:
    """
    Turn a shell glob from :data:`EXPECTED` into the regex a LinePlot wants.

    Hand-rolled rather than ``fnmatch.translate``, which emits atomic groups ``(?>...)``. The
    panel regex is evaluated by the W&B frontend in JavaScript, which has no atomic groups and
    would reject the whole pattern -- leaving an empty chart that reads as a flat result.
    """
    return "^" + "".join(".*" if ch == "*" else re.escape(ch) for ch in pattern) + "$"


def build_report(entity: str, project: str, group: str) -> str:
    """
    Build a report over the group, with one section per question the module asks.

    Panels are built per *family* with ``metric_regex`` rather than per key, so sixteen blocks
    of lane norms is one readable chart instead of sixty-four series, and so a run with a
    different layer count still lands in the right panel. Only families the runs actually have
    get a panel: an empty chart over an invented key reads as a flat result, which is the one
    way a plot here could actively mislead.
    """
    import wandb_workspaces.reports.v2 as wr

    per_run = observed_keys(entity, project, group)
    if not per_run:
        raise SystemExit(f"No runs in {entity}/{project} with group '{group}' to report on.")
    keys = sorted({k for ks in per_run.values() for k in ks})

    def runset() -> "wr.Runset":
        return wr.Runset(entity=entity, project=project, name=group)

    blocks: List[object] = [
        wr.H1(f"Hyper-connections at 370M — {group}"),
        wr.P(
            "Explaining a sign inversion rather than reproducing a method. ByteDance measured "
            "-0.030 loss and +1.3 downstream at n=4 on OLMo-1B over 500B tokens; a controlled "
            "replication measured -0.020 downstream at 1.2B dense and divergence at 3B. Each "
            "arm isolates one documented difference between those two setups. Read the guard "
            "section first: if the lanes never differentiate, nothing below it means anything."
        ),
    ]

    for section, why, _required, patterns in EXPECTED:
        panels = [
            wr.LinePlot(title=pattern, x="Step", metric_regex=_as_regex(pattern))
            for pattern in patterns
            if matched(keys, pattern)
        ]
        if not panels:
            continue
        blocks.append(wr.H2(section))
        blocks.append(wr.P(why))
        blocks.append(wr.PanelGrid(runsets=[runset()], panels=panels))

    report = wr.Report(
        entity=entity,
        project=project,
        title=f"Hyper-connections 370M — {group}",
        description="Built from the keys these runs actually logged.",
        width="fluid",
        blocks=blocks,  # type: ignore[arg-type]
    )
    report.save(draft=True)
    return report.url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", required=True, help="The experiment slug.")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", action="store_true")
    opts = parser.parse_args()

    if not opts.verify and not opts.report:
        opts.verify = True

    status = 0
    if opts.verify:
        status = verify(opts.entity, opts.project, opts.group)
    if opts.report:
        print("report:", build_report(opts.entity, opts.project, opts.group))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
