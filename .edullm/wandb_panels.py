#!/usr/bin/env python3
"""Check that a hyper-connection submission logged what the pre-registration needs, and build
the view.

Two modes, and the first is the one that matters:

``--verify``  asks whether ONE NAMED SUBMISSION is ready to analyse, cell by cell, against the
              families each part of the pre-registration rests on. A missing key is not a
              cosmetic problem here: the decision rule is "measure sigma from the baseline's
              seeds, claim nothing under 2 sigma, report downstream alongside BPB", and every
              clause of that is a metric that either arrived or did not.

``--report``  builds a W&B report over an experiment group, with one panel section per
              question the module asks. Panels are only created for keys the runs actually
              have, because a panel over an invented key renders an empty chart that looks
              like a flat result.

    python .edullm/wandb_panels.py --verify --run run_019fe2f4-f528 --cells 5 --arm baseline
    python .edullm/wandb_panels.py --report --group hyper-connections-370m

WHAT ``--verify`` USED TO ANSWER, AND WHY THAT WAS WORSE THAN NOTHING. It took a ``--group``,
unioned the metric keys of every run in it, and reported a family present if the union held
one. The group is the whole module -- every probe, every rehearsal, every arm, months of them
-- so the ``hc/*`` families were satisfied by old ``faithful`` probes while the runs actually
being gated were five ``baseline`` cells that cannot log a lane metric at all. It printed
"everything the pre-registration rests on is present" and exited 0 on that. It also keyed its
results by ``run.name``, which every cell of a fan-out shares, so a five-cell submission
reported as "1 run(s)" and a cell missing a family was hidden behind its four siblings.

A verifier that passes for the wrong reason is worse than none, because it is the thing
standing between the tranche and an analysis over a missing metric family. So this one is
scoped to a submission, addresses cells by run id, and knows which families each arm is
supposed to have.
"""

import argparse
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hyper_connection_arms import ARMS  # noqa: E402
from stage_gate import arms_consistent_with  # noqa: E402

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: A family that every arm must log.
EVERY = "every"

#: A family only an arm with lanes can log, and which on an arm without lanes is not merely
#: excused but forbidden: ``train_hyper_connections.train`` attaches the monitor only when
#: ``arm.hyper_connections`` is not None, so an ``hc/*`` key on a baseline cell means that cell
#: did not run the arm it was submitted as.
LANES = "lanes"


@dataclass(frozen=True)
class Family:
    """One group of metric keys, and the terms on which it is expected."""

    section: str
    """What it is called in the output and in the report."""

    why: str
    """What is lost if it is missing. Printed only when something is."""

    scope: str
    """:data:`EVERY` or :data:`LANES`."""

    required: bool
    """Whether an absence blocks an analysis or only costs a chart."""

    patterns: Tuple[str, ...]
    """Shell globs over metric keys."""


#: What each part of the module needs, and on which arms.
EXPECTED: Tuple[Family, ...] = (
    Family(
        "the guard",
        "Identical lane norms mean the mechanism is inert, and then no downstream number is "
        "interpretable in either direction.",
        LANES,
        True,
        ("hc/min lane norm spread", "hc/block */lane norm spread", "hc/block */lane * norm"),
    ),
    Family(
        "stability",
        "Parcae found diverging runs learn a spectral radius at or above 1; mHC's claim is "
        "that the Birkhoff constraint keeps the composite across depth well conditioned.",
        LANES,
        True,
        (
            "hc/block */rho(A_r) attention",
            "hc/block */rho(A_r) feed_forward",
            "hc/composite condition number",
            "hc/composite spectral radius",
        ),
    ),
    Family(
        "hidden-state scale",
        "RMSNorm readouts are scale-invariant, so cross-entropy cannot see hidden-state scale "
        "at all and a stack can drive norms into the thousands invisibly.",
        LANES,
        True,
        ("hc/block */hidden norm",),
    ),
    Family(
        "the result",
        "The noise floor and every hypothesis are stated on a held-out metric. Training loss "
        "will not do: --seed moves the shuffle, so its variance across seeds is partly a "
        "different sample of the corpus.",
        EVERY,
        True,
        ("eval/*/CE loss", "eval/*/BPB", "train/CE loss"),
    ),
    Family(
        "throughput and cost",
        "What fills the cost table, and what says whether the arms are really iso-FLOP.",
        EVERY,
        True,
        ("throughput/device/MFU", "throughput/device/TPS", "throughput/total tokens"),
    ),
    Family(
        "the weight-decay split",
        "Arm 5 differs from arm 2 only in the optimizer. NOTE THAT THIS DOES NOT DISCRIMINATE: "
        "the baseline logs two LR groups as well, because OLMo-core already splits decay off "
        "the norms and biases. So it confirms that groups exist and not that the lane split is "
        "the reason, which is why it is advisory on every arm rather than required on the "
        "arms with lanes.",
        EVERY,
        False,
        ("optim/LR (group 0)", "optim/LR (group 1)"),
    ),
    Family(
        "downstream",
        "The decision rule says report downstream alongside BPB. Not produced in-loop by "
        "design -- the downstream evaluator fetches from the public internet, which does not "
        "belong inside a run whose claim is that it read a sealed corpus. Expected to arrive "
        "from a separate job over saved checkpoints.",
        EVERY,
        False,
        ("eval/downstream/*",),
    ),
)


@dataclass(frozen=True)
class Cell:
    """One fan-out cell of a submission, with the keys it logged and the arm it really ran."""

    index: int
    state: str
    step: int
    keys: Tuple[str, ...]
    arms: Tuple[str, ...]
    """Arms this cell's own logged config is consistent with. See ``stage_gate``."""

    @property
    def has_lanes(self) -> Optional[bool]:
        """
        Whether this cell ran an arm with a residual-lane stream.

        ``None`` when its config matches arms that disagree about it, which cannot currently
        happen -- every ambiguity the config leaves is between two arms that both have lanes or
        neither -- but is not something to assume in a verifier.
        """
        lanes = {ARMS[a].hyper_connections is not None for a in self.arms if a in ARMS}
        return lanes.pop() if len(lanes) == 1 else None


def matched(keys: Sequence[str], pattern: str) -> List[str]:
    return [k for k in keys if fnmatch.fnmatch(k, pattern)]


def read_cells(entity: str, project: str, run_id: str, cells: int) -> List[Cell]:
    """
    Read one submission's cells out of W&B, addressed by run id.

    BY ID AND NOT BY NAME. A fan-out's cells share their display name and differ only in the
    ``-cell-<index>`` suffix of the id, so a dict keyed on the name keeps one of the five and
    silently drops the rest -- which is how the old verdict came to say "1 run(s)" for a
    five-cell submission.

    :param entity: W&B entity.
    :param project: W&B project.
    :param run_id: The platform run id of the submission.
    :param cells: The fan-out size.

    :returns: One :class:`Cell` per cell that has logged anything, in cell order.
    """
    import wandb

    api = wandb.Api(timeout=120)
    found: List[Cell] = []
    for index in range(cells):
        try:
            run = api.run(f"{entity}/{project}/{run_id}-cell-{index}")
        except Exception:
            continue
        found.append(
            Cell(
                index=index,
                state=run.state,
                step=int(run.summary.get("_step") or 0),
                keys=tuple(sorted(k for k in run.summary.keys() if not str(k).startswith("_"))),
                arms=arms_consistent_with(run.config),
            )
        )
    return found


def applies(family: Family, cell: Cell) -> bool:
    """Whether this family is expected on this cell, given the arm the cell actually ran."""
    if family.scope == EVERY:
        return True
    return cell.has_lanes is True


def verify(entity: str, project: str, run_id: str, cells: int, arm: Optional[str] = None) -> int:
    """
    Ask whether one submission is ready to analyse, and print the evidence per cell.

    :param entity: W&B entity.
    :param project: W&B project.
    :param run_id: The platform run id of the submission being gated.
    :param cells: How many cells the submission asked for. A cell that never started is a
        missing replicate and is a failure, not an absence.
    :param arm: The arm the submission was supposed to run. Checked against what each cell's
        own config says. Optional, because the config already names it; supplying it turns a
        description into an assertion.

    :returns: 0 when every required family is present on every cell, 1 when one is missing or a
        cell logged a family its arm cannot have, 2 when there is nothing to judge yet.
    """
    found = read_cells(entity, project, run_id, cells)
    print(f"{entity}/{project}  submission={run_id}")
    if not found:
        print(f"No cells of {run_id} have logged anything yet.")
        return 2
    print(f"{len(found)} of {cells} cell(s) reporting\n")

    failures: List[str] = []
    if len(found) < cells:
        failures.append(
            f"only {len(found)} of {cells} cells have logged anything -- an absent cell is a "
            "missing replicate, and the arm mean would be short one seed"
        )

    for cell in found:
        named = ", ".join(cell.arms) or "no arm in the table"
        note = ""
        if arm is not None and arm not in cell.arms:
            note = f"  <-- submitted as '{arm}'"
            failures.append(
                f"cell {cell.index} logged a config consistent with {named}, not '{arm}'"
            )
        print(
            f"  cell {cell.index}  {cell.state:<9} step {cell.step:>6}  "
            f"{len(cell.keys):3d} keys  arm: {named}{note}"
        )
    print()

    for family in EXPECTED:
        expected_on = [c for c in found if applies(family, c)]
        forbidden_on = [c for c in found if family.scope == LANES and c.has_lanes is False]

        short: Dict[int, List[str]] = {}
        for cell in expected_on:
            absent = [p for p in family.patterns if not matched(cell.keys, p)]
            if absent:
                short[cell.index] = absent
        leaked = {
            c.index: [p for p in family.patterns if matched(c.keys, p)]
            for c in forbidden_on
            if any(matched(c.keys, p) for p in family.patterns)
        }

        if not expected_on:
            print(
                f"[{family.section}]  n/a   no cell here runs an arm with lanes, so these keys "
                "are not expected and their absence is not evidence of anything"
            )
        elif short:
            mark = "MISSING" if family.required else "absent "
            print(f"[{family.section}]  {mark}  on cell(s) {sorted(short)}")
            for index, absent in sorted(short.items()):
                print(f"     cell {index}: {', '.join(absent)}")
            print(f"     why it matters: {family.why}")
            if family.required:
                failures.append(
                    f"'{family.section}' is missing on cell(s) {sorted(short)} of {len(expected_on)} "
                    "that should have it"
                )
        else:
            print(
                f"[{family.section}]  ok    all {len(family.patterns)} pattern(s) on all "
                f"{len(expected_on)} cell(s) that should have them"
            )

        if leaked:
            print(f"     LEAKED  cell(s) {sorted(leaked)} logged lane metrics without lanes")
            failures.append(
                f"cell(s) {sorted(leaked)} logged '{family.section}' keys although their config "
                "has no hyper-connections -- the monitor is only attached to an arm with lanes, "
                "so this cell did not run the arm it was submitted as"
            )

    print()
    if failures:
        print(f"VERDICT: {run_id} is NOT ready to analyse.")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"VERDICT: {run_id} has every required family on all {len(found)} cell(s).")
    return 0


def observed_keys(entity: str, project: str, group: str) -> Dict[str, List[str]]:
    """
    Collect the metric keys each run in a group logged, keyed by run id.

    FOR ``--report`` AND NOT FOR ``--verify``. A union over a whole group is the right input to
    a panel -- a chart wants every series the group can offer -- and exactly the wrong input to
    a gate, which is the confusion this module used to be built on. Keyed by id rather than by
    display name so that a fan-out contributes all of its cells.

    :returns: Run id -> sorted metric keys.
    """
    import wandb

    api = wandb.Api(timeout=120)
    runs = api.runs(f"{entity}/{project}", filters={"group": group}, per_page=100)[:100]
    return {run.id: sorted(k for k in run.summary_metrics if not k.startswith("_")) for run in runs}


def _as_regex(pattern: str) -> str:
    """
    Turn a shell glob from :data:`EXPECTED` into the regex a LinePlot wants.

    Hand-rolled rather than ``fnmatch.translate``, which emits atomic groups ``(?>...)``. The
    panel regex is evaluated by the W&B frontend in JavaScript, which has no atomic groups and
    would reject the whole pattern -- leaving an empty chart that reads as a flat result.
    """
    return "^" + "".join(".*" if ch == "*" else re.escape(ch) for ch in pattern) + "$"


def build_report(entity: str, project: str, group: str, dry_run: bool = False) -> str:
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

    for family in EXPECTED:
        panels = [
            wr.LinePlot(title=pattern, x="Step", metric_regex=_as_regex(pattern))
            for pattern in family.patterns
            if matched(keys, pattern)
        ]
        if not panels:
            continue
        blocks.append(wr.H2(family.section))
        blocks.append(wr.P(family.why))
        blocks.append(wr.PanelGrid(runsets=[runset()], panels=panels))

    report = wr.Report(
        entity=entity,
        project=project,
        title=f"Hyper-connections 370M — {group}",
        description="Built from the keys these runs actually logged.",
        width="fluid",
        blocks=blocks,  # type: ignore[arg-type]
    )

    if dry_run:
        # Constructing a Report is local until save() runs, so this shows exactly what would
        # be published without leaving a draft behind in a shared project.
        sections = [b.text for b in blocks if isinstance(b, wr.H2)]
        panels = sum(len(b.panels) for b in blocks if isinstance(b, wr.PanelGrid))
        print(f"would publish: {len(sections)} section(s), {panels} panel(s)")
        for block in blocks:
            if isinstance(block, wr.H2):
                print(f"  [{block.text}]")
            elif isinstance(block, wr.PanelGrid):
                for panel in block.panels:
                    print(f"      panel  {panel.title}")
        return "(dry run, nothing saved)"

    report.save(draft=True)
    return report.url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", help="The experiment slug. Required for --report.")
    parser.add_argument(
        "--run",
        help="The platform run id of the submission to verify. Required for --verify: a "
        "group spans every probe and every arm this module has ever run, and a gate over "
        "that answers a question nobody asked.",
    )
    parser.add_argument("--cells", type=int, default=5, help="The submission's fan-out size.")
    parser.add_argument(
        "--arm",
        help="The arm the submission was supposed to run, checked against each cell's own "
        "logged config. Optional: the config already names the arm, and supplying this turns "
        "a description of what ran into an assertion about it.",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --report, print the sections and panels that would be published without "
        "saving anything.",
    )
    opts = parser.parse_args()

    if not opts.verify and not opts.report:
        opts.verify = True
    if opts.verify and not opts.run:
        parser.error("--verify needs --run: see its help for why a group will not do")
    if opts.report and not opts.group:
        parser.error("--report needs --group")
    if opts.arm and opts.arm not in ARMS:
        parser.error(f"--arm {opts.arm!r} is not in the arm table: {', '.join(ARMS)}")

    status = 0
    if opts.verify:
        status = verify(opts.entity, opts.project, opts.run, opts.cells, opts.arm)
    if opts.report:
        print("report:", build_report(opts.entity, opts.project, opts.group, opts.dry_run))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
