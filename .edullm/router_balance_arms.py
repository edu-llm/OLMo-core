"""The arms of the router-balance sweep, and the gate that proves each one reaches the module.

    python .edullm/router_balance_arms.py --verify        # build every arm, assert every field
    python .edullm/router_balance_arms.py --commands      # print the torchrun line for each

ONE FILE OWNS THE TABLE SO THAT THE GATE AND THE RUNNER CANNOT DISAGREE. The failure this is
built against is not a typo, it is a setting that looks applied and is not: ``run_name`` is a
positional with ``nargs="?"``, so a bare ``model.block...=0.1`` appended to the command binds to
the name and the field keeps its default. The run then trains for forty minutes and reports a
null result for an arm that was never run. ``--verify`` builds each arm's config through
``build_config`` -- the same function the real invocation uses, from the same argv -- and asserts
the value on the built *module*, not on the config that describes it.

WHY THE BUILT MODULE AND NOT THE CONFIG. ``MoERouter.__init__`` reads ``bias_gamma`` and registers
``score_bias`` there, so a value that arrives after construction leaves the field set, the log
honest and the mechanism absent. ``verify_router_overrides.py`` makes the same argument for the
three overrides it covers; this file makes it for the arms actually dispatched.

WHAT THE ARMS SHARE, AND WHY EACH SHARED NUMBER IS THE NUMBER IT IS.

``--steps 6000`` is the learning-rate schedule and is NOT where any arm stops; ``--hard-stop-steps``
is. Sizing ``--steps`` to an arm's own length would decay that arm's learning rate to nothing over
its last few hundred steps, and on a router ablation a frozen optimiser looks exactly like an
imbalance that has settled -- which is the one question this sweep exists to answer. With 6000 the
rate is still 87% of peak at step 1500, so every arm runs at a near-flat rate and a short arm is a
prefix of a long one.

``--warmup-steps 100`` rather than the base run's 2000, because a probe on the 2000-step ramp sits
at a few percent of peak throughout, the router barely moves, and the imbalance stays wherever
initialisation put it. 100 is what the earlier probe used and it is what puts the measurement in
the 3.6-6.5 imbalance band the real run reports.

``--rank-microbatch-size 8192`` and not the base run's 16384, because at 16384 the *imbalanced*
arm dies of CUDA out-of-memory at step 32: a dropless MoE sizes its buffers by the busiest
expert's token count, so imbalance costs memory as well as time. A number measured at one
microbatch is not comparable with one measured at the other, and this file records which was used.

``--global-batch-size 524288`` is eight ranks times 65,536 tokens a rank, which is exactly the
base run's per-rank load (4,194,304 over 64 ranks). The bias controller takes one step per
optimiser step either way, so a step here is a step there for the purpose of asking how long the
controller needs -- with the caveat that this probe's update sees an eighth of the tokens, so its
sign is noisier and its equilibrium is if anything slightly worse than the real run's.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Everything every arm holds fixed. The three dataset arguments cannot come from the
#: environment on this lane -- the block bootstrap sets the run id, the prefixes and the data
#: bucket and stops there -- so they are written out.
COMMON: List[str] = [
    "--model-factory", "olmoe_7b_32x4",
    "--dataset-id", "pretrain/reservoir-dolma2",
    "--dataset-version", "v1",
    "--dataset-tokenizer", "tokenizer/dolma2-bpe",
    # The corpus the base run reads, chosen over regmix-10b for two reasons: imbalance is a
    # property of the token distribution, so it should be measured on the distribution the run
    # will see; and this one declares a held-out split, which is the only reason any arm here
    # can report a loss at all. regmix-10b declares none.
    "--require-val",
    "--sequence-length", "4096",
    "--global-batch-size", "524288",
    "--rank-microbatch-size", "8192",
    "--steps", "6000",
    "--warmup-steps", "100",
    "--learning-rate", "4e-4",
    "--param-dtype", "bfloat16",
    "--data-seed", "0",
    # Matched-step held-out loss, which is the only quality signal this sweep produces. Eight
    # batches is 524,288 tokens, one training batch's worth, forward-only -- about a third of a
    # step's time. Every arm long enough gets a number at 500 and at 1000 on identical data, and
    # `eval_on_finish` adds one wherever the arm stops.
    "--eval-interval", "500",
    "--eval-batches", "8",
    "--moe-shard-degree", "8",
    "--moe-num-replicas", "1",
    "--fsdp-prefetch-factor", "2",
]

#: Overrides applied after the run name, never before it. Disabling the checkpointer is not
#: cosmetic: `CheckpointerCallback.post_train` saves unconditionally and an earlier smoke run
#: spent 340 seconds inside that one synchronous save, which is not part of a throughput number.
OVERRIDES: List[str] = ["trainer.callbacks.checkpointer.enabled=false"]

DEFAULT_LB = 0.01
DEFAULT_Z = 0.001


@dataclass(frozen=True)
class Arm:
    name: str
    stop: int
    why: str
    gamma: Optional[float] = None
    lb: float = DEFAULT_LB
    z: float = DEFAULT_Z

    def args(self) -> List[str]:
        out = list(COMMON) + ["--hard-stop-steps", str(self.stop)]
        if self.gamma is not None:
            out += ["--moe-router-bias-gamma", repr(self.gamma)]
        if self.lb != DEFAULT_LB:
            out += ["--moe-lb-loss-weight", repr(self.lb)]
        if self.z != DEFAULT_Z:
            out += ["--moe-z-loss-weight", repr(self.z)]
        return out

    def argv(self) -> List[str]:
        """The full argv, run name first. The name has to be first and has to be there."""
        return [self.name] + self.args() + OVERRIDES

    def command(self, nproc: int = 8) -> str:
        return " ".join(
            shlex.quote(word)
            for word in [
                "torchrun", "--standalone", f"--nproc-per-node={nproc}",
                ".edullm/train_on_corpus.py",
                *self.argv(),
            ]
        )


#: IN PRIORITY ORDER, BECAUSE THE FLEET MAY TAKE THE MACHINE BACK. Each arm is written to stand
#: on its own against the anchor, so stopping after any of them leaves a readable result rather
#: than half of one.
ARMS: List[Arm] = [
    Arm(
        name="rb-anchor",
        stop=600,
        gamma=None,
        why="The control, on this machine. Every ratio in the report is between two arms that "
        "ran on one node, because the imbalanced arm is straggler-bound and so amplifies "
        "whatever is slowest on the box -- a cross-machine ratio measures the machine.",
    ),
    Arm(
        name="rb-g1e4",
        stop=1500,
        gamma=0.0001,
        why="The incumbent recommendation, run four times longer than the probe that found it. "
        "At 250 steps it was +8.4% and still improving; the question this arm exists for is "
        "where it stops.",
    ),
    Arm(
        name="rb-g3e4",
        stop=1200,
        gamma=0.0003,
        why="Between 0.0001 and 0.001, and the most likely winner. The bias moves by gamma a "
        "step, so three times the gamma reaches a given bias in a third of the steps: if the "
        "controller's destination depends on gamma only through gamma x steps, this arm at 500 "
        "is the 0.0001 arm at 1500 and it sees the plateau first. If instead it lands worse, "
        "the thrashing that ruins 0.001 has already started by here and 0.0001 is near optimal.",
    ),
    Arm(
        name="rb-g1e4-lb10",
        stop=1000,
        gamma=0.0001,
        lb=0.1,
        why="The two mechanisms together, which may compose or fight. The prediction from the "
        "code is that they compose without interfering: the auxiliary loss is a dot product of "
        "the post-bias assignment counts with the mean scores, so a bias that has already "
        "flattened the counts leaves that term near constant and its gradient small.",
    ),
    Arm(
        name="rb-lb10",
        stop=800,
        lb=0.1,
        why="The other mechanism alone, at ten times the recipe's weight. IT CHANGES THE LOSS "
        "FUNCTION, which bias_gamma deliberately does not, so it is a heavier trade and the "
        "held-out number matters more here than anywhere else in the table.",
    ),
    Arm(
        name="rb-g5e5",
        stop=1200,
        gamma=0.00005,
        why="Below 0.0001, to bracket it from underneath. Half the gamma is twice the steps to "
        "the same bias, so this arm at 1200 should sit on the 0.0001 arm's curve at 600 if the "
        "destination is set by gamma x steps. Two curves that collapse under that rescaling "
        "say the optimum is the largest gamma that has not begun to thrash, not the smallest "
        "one anybody tried.",
    ),
]


def _expected(arm: Arm) -> Dict[str, object]:
    return {
        "bias_gamma": arm.gamma,
        "lb_loss_weight": arm.lb,
        "z_loss_weight": arm.z,
        "uniform_expert_assignment": False,
        "rank_microbatch_size": 8192,
        "run_name": arm.name,
    }


def verify(only: Optional[str] = None) -> int:
    """Build each arm the way the dispatch will and assert the settings survived to the module."""
    import torch

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import train_on_corpus as toc  # noqa: E402

    failures: List[str] = []
    for arm in ARMS:
        if only is not None and arm.name != only:
            continue
        before = len(failures)
        opts, overrides = toc.build_parser().parse_known_args(arm.argv())

        # The trap itself, checked first and by name. If the run name were eaten by an override
        # every other assertion below would still pass, because the config would be the default
        # one and the default one is a legitimate config.
        if opts.run_name != arm.name:
            failures.append(f"{arm.name}: run_name resolved to {opts.run_name!r}")
            continue
        if sorted(overrides) != sorted(OVERRIDES):
            failures.append(f"{arm.name}: leftover overrides are {overrides!r}")
            continue

        config = toc.build_config(opts, overrides)
        with torch.device("meta"):
            model = config.model.build(init_device="meta")
        moe = model.blocks["0"].feed_forward_moe
        router = moe.router

        got: Dict[str, object] = {
            "bias_gamma": router.bias_gamma,
            # Read off the router, which is where the value lands after MoEBase divides it by
            # the layer count -- so the assertion has to undo that division rather than compare
            # against what was typed.
            "lb_loss_weight": None
            if router.lb_loss_weight is None
            else round(router.lb_loss_weight * 16, 12),
            "z_loss_weight": None
            if router.z_loss_weight is None
            else round(router.z_loss_weight * 16, 12),
            "uniform_expert_assignment": router.uniform_expert_assignment,
            "rank_microbatch_size": config.train_module.rank_microbatch_size,
            "run_name": opts.run_name,
        }
        for key, want in _expected(arm).items():
            if got[key] != want:
                failures.append(f"{arm.name}: {key} is {got[key]!r}, wanted {want!r}")

        # The buffer, which is the mechanism rather than the field. Present exactly when gamma is.
        has_buffer = router.score_bias is not None
        if has_buffer != (arm.gamma is not None):
            failures.append(
                f"{arm.name}: score_bias buffer {'present' if has_buffer else 'absent'} "
                f"with gamma {arm.gamma!r}"
            )

        # The checkpointer really is off, because a 340-second synchronous save inside a
        # throughput window would move every number in the report.
        checkpointer = config.trainer.callbacks.get("checkpointer")
        if checkpointer is None or getattr(checkpointer, "enabled", True):
            failures.append(f"{arm.name}: the checkpointer is not disabled")

        # The learning-rate schedule is the shared one and not this arm's length.
        if config.trainer.max_duration.value != 6000:
            failures.append(f"{arm.name}: max_duration is {config.trainer.max_duration}")
        if config.trainer.hard_stop is None or config.trainer.hard_stop.value != arm.stop:
            failures.append(f"{arm.name}: hard_stop is {config.trainer.hard_stop}")

        print(
            f"{'ok  ' if len(failures) == before else 'FAIL'}  {arm.name:<14} "
            f"gamma={arm.gamma!r:<9} lb={arm.lb:<6} "
            f"stop={arm.stop:<5} buffer={'yes' if has_buffer else 'no'}"
        )

    if failures:
        print("\nARMS_DO_NOT_LAND", file=sys.stderr)
        for line in failures:
            print(f"  FAIL  {line}", file=sys.stderr)
        return 1
    print("\nOVERRIDES_LAND: every arm reaches the module it names.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--commands", action="store_true")
    parser.add_argument("--names", action="store_true")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--nproc", type=int, default=8)
    opts = parser.parse_args()

    if opts.names:
        print("\n".join(a.name for a in ARMS if opts.arm in (None, a.name)))
        return 0
    if opts.commands:
        for arm in ARMS:
            if opts.arm in (None, arm.name):
                print(arm.command(opts.nproc))
        return 0
    if opts.verify:
        return verify(opts.arm)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
