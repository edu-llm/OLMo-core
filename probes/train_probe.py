"""Train one probe arm and measure length generalization.

Protocol follows arXiv:2502.10297 Appendix C.2: train on short sequences, evaluate
on progressively longer ones, and report per-length token accuracy. The headline
number is accuracy at lengths well beyond the training range.

``--beta-regime`` is **required** and has no default. ``strict`` gives
``beta = sigmoid(l) in (0, 1)``; ``reflection`` gives ``beta = 2*sigmoid(l) in (0, 2)``,
which admits negative erase eigenvalues. The DP2 program's defining constraint is
strict beta, and strict and reflection results may never be pooled -- so the
regime is an explicit argument recorded in the result record, not a default.

Four independent seed streams are required and are never derived from one
another: model initialization, the training data/curriculum stream, the
task-instance generator, and the held-out evaluation bank. Supply them with
``--bundle-id`` (which applies the canonical 100000/200000/300000/400000 offset
map) or individually with ``--seed-{init,data,task,eval}``.

Usage:
    python train_probe.py --manifest <run-manifest.json> --out <result.json>
    python train_probe.py --arm DP2-strict --task s5_words --bundle-id 1101
    python train_probe.py --mixer kda --task s5_words --beta-regime strict --bundle-id 1101
    python train_probe.py --mixer kda_hh --num-householder 1 --task s5_words \
        --beta-regime strict --bundle-id 1101 --match-non-embedding 1400524

The first form is canonical: under ``--manifest`` the manifest is the source of
truth and free-form overrides are rejected. The remaining forms stay available
for interactive work and are what the calibration sweeps use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional

import torch
from torch import nn

from model import ProbeModel, solve_ffn_dim
from tasks import TASKS

# Canonical seed map (runbook Phase 0-1 §5.5): one bundle ID expands into four
# independent streams. The offsets are 100000 apart, and bundle IDs are 4-digit,
# so the four streams of every bundle are pairwise distinct and no two bundles
# collide either.
SEED_STREAM_OFFSETS = {
    "init": 100_000,
    "data": 200_000,
    "task": 300_000,
    "eval": 400_000,
}

# Beta regimes. The value is the 'allow_neg_eigval' flag the OLMo-core mixer
# configs take; 'strict' keeps beta in (0, 1) so the erase eigenvalue
# 1 - beta*||k||^2 stays positive, 'reflection' doubles it into (0, 2).
BETA_REGIMES = {"strict": False, "reflection": True}

# Canonical arm IDs (runbook §3.2). The manifest 'arm' field accepts these verbatim
# and nothing else -- informal short forms such as "tied-K" are rejected.
#
# Two arms are declared here but deliberately NOT implemented. Both need a
# *per-factor* beta parameterization, which lives in the OLMo-core R-factor path
# (recurrent.py builds one '(B, T, R*n_heads)' beta from a single 'w_b' and applies
# one regime flag to all of it), not in this harness. They raise rather than
# silently mapping onto DP2-strict, which is what an omitted entry would do.
ARMS: dict[str, dict] = {
    "R1": dict(mixer="kda_hh", num_householder=1, beta_regime="strict"),
    # The capacity control for DP2-strict: R=1, but with the DP2 parameter delta spent in FFN
    # width so the two differ *only* in arity. 'match_arm' is part of the arm definition
    # rather than a flag the caller supplies, because an R1-P run launched without it is
    # byte-for-byte an R1 run wearing a different name -- it trains, it succeeds, and it
    # records 'arm': 'R1-P' while controlling for nothing. That failure is invisible in the
    # record unless someone thinks to compare param_ledger across two files.
    "R1-P": dict(mixer="kda_hh", num_householder=1, beta_regime="strict", match_arm="DP2-strict"),
    "DP2-strict": dict(mixer="kda_hh", num_householder=2, beta_regime="strict"),
    "Reflection": dict(mixer="kda_hh", num_householder=2, beta_regime="reflection"),
    # R=1 under the reflection regime: the fourth cell of the (beta regime x R) square, and the
    # only one that had no canonical id. The 155-record archive contains it under the name
    # 'R1-refl' -- set there with explicit --num-householder/--beta-regime flags rather than an
    # arm id -- which is why it is spelled that way here rather than 'R1-reflection'.
    #
    # It exists because the R-vs-regime interaction is not identified without it. 'Reflection'
    # minus 'DP2-strict' confounds the two factors: they differ in R *and* in beta range. Only
    # (Reflection - R1-refl) against (DP2-strict - R1) separates them, and the first of those
    # contrasts needs this arm. Naming it makes that contrast reproducible from an arm id
    # instead of from a pair of flags a caller has to remember to set together.
    "R1-refl": dict(mixer="kda_hh", num_householder=1, beta_regime="reflection"),
    # R1-P's reflection twin, and the reason R1-P alone does not close the capacity confound.
    #
    # The quantity the sweep measures is an *interaction*: (R effect under reflection) minus
    # (R effect under strict). Adding a capacity control to only one of those two contrasts
    # does not make the interaction capacity-controlled -- it makes the two contrasts
    # differently constructed, which is worse than leaving both uncontrolled, because the
    # difference between them then mixes the capacity correction with the effect. Both
    # regimes need their control or neither does.
    "R1-refl-P": dict(
        mixer="kda_hh", num_householder=1, beta_regime="reflection", match_arm="Reflection"
    ),
    "DP2-budgeted": dict(unimplemented="needs per-factor beta b=2*sigmoid(l_b), pi=sigmoid(l_pi)"),
    "R1-2step-tiedK": dict(unimplemented="needs k2=k1 forced at the recurrence boundary"),
}


def apply_arm(args: argparse.Namespace) -> None:
    """Resolve ``args.arm`` into the concrete mixer/factor/regime settings, in place.

    :param args: Parsed arguments carrying ``arm``.
    :raises SystemExit: If the arm ID is unknown or is one of the two arms whose
        implementation requires a per-factor beta parameterization.
    """
    if args.arm is None:
        return
    if args.arm not in ARMS:
        raise SystemExit(
            f"unknown arm '{args.arm}'; canonical IDs are {sorted(ARMS)} "
            "(informal short forms such as 'tied-K' are rejected)"
        )
    settings = ARMS[args.arm]
    if "unimplemented" in settings:
        raise SystemExit(f"arm '{args.arm}' is not implemented: {settings['unimplemented']}")
    for key, value in settings.items():
        # An arm's settings ARE the arm. Letting a command-line flag survive here would mean
        # two runs could both record 'arm': 'R1-P' while having matched different targets, and
        # the record would not show which. Refuse instead of picking a winner.
        existing = getattr(args, key, None)
        if key == "match_arm" and existing is not None and existing != value:
            raise SystemExit(
                f"--match-arm {existing!r} conflicts with arm '{args.arm}', which is defined "
                f"as matching {value!r}. Drop the flag, or use --mixer/--num-householder/"
                f"--beta-regime directly instead of an arm id."
            )
        setattr(args, key, value)


def load_manifest(path: str, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Build the run configuration from a frozen manifest.

    The manifest, not the shell command, is the source of truth (runbook §5.2.1).
    Only ``--manifest`` and ``--out`` may appear on the command line alongside it;
    any other flag is rejected rather than merged, because a manifest that can be
    overridden from the shell is not a record of what ran.

    :param path: Path to the run manifest JSON.
    :param parser: The argument parser, used for its defaults.
    :returns: A fully populated namespace.
    :raises SystemExit: On an unknown manifest key or a missing required key.
    """
    with open(path) as fh:
        manifest = json.load(fh)

    args = parser.parse_args([])  # defaults only
    known = {a.dest for a in parser._actions}
    # Manifest bookkeeping fields that are recorded but are not runner settings.
    passthrough = {
        "run_id", "phase", "source", "environment", "prereg_digest", "retry",
        "expect_eval_bank_sha256", "expect_non_embedding", "expect_source_revision",
        "task_settings", "notes",
    }
    for key, value in manifest.items():
        dest = key.replace("-", "_")
        if dest in passthrough:
            continue
        if dest not in known:
            raise SystemExit(f"manifest key '{key}' is not a runner setting")
        setattr(args, dest, value)
    for required in ("arm", "task"):
        if getattr(args, required, None) is None:
            raise SystemExit(f"manifest is missing required key '{required}'")
    args.manifest = path
    args._manifest_data = manifest
    return args


def check_manifest_expectations(
    manifest: dict, *, eval_digest: Optional[str], non_embedding: Optional[int]
) -> None:
    """Fail the run **before training** on any manifest expectation mismatch.

    Runbook §5.2.1: "A run must fail before training if its evaluation-bank
    checksum, source revision, or expected parameter ledger does not match the
    manifest." Each expectation is optional in the manifest but binding when present.

    :param manifest: The loaded manifest.
    :param eval_digest: The realized evaluation-bank sha256, or ``None`` if not built yet.
    :param non_embedding: The realized non-embedding parameter count.
    :raises SystemExit: On any mismatch.
    """
    expected_source = manifest.get("expect_source_revision")
    if expected_source is not None:
        actual = probe_source_revision()
        if actual != expected_source:
            raise SystemExit(
                f"source revision mismatch: manifest expects {expected_source}, tree is {actual}"
            )
    expected_params = manifest.get("expect_non_embedding")
    if expected_params is not None and non_embedding is not None:
        if int(expected_params) != int(non_embedding):
            raise SystemExit(
                f"parameter ledger mismatch: manifest expects non_embedding="
                f"{expected_params}, model has {non_embedding}"
            )
    expected_bank = manifest.get("expect_eval_bank_sha256")
    if expected_bank is not None and eval_digest is not None:
        if expected_bank != eval_digest:
            raise SystemExit(
                f"evaluation-bank checksum mismatch: manifest expects {expected_bank}, "
                f"generated {eval_digest}"
            )


def probe_source_revision() -> str:
    """Return the ``probes/`` git revision, with a dirty marker.

    Warns loudly on failure rather than returning ``"unknown"`` quietly. A results file whose
    provenance field reads ``"unknown"`` looks like data but is not: it cannot be traced back to
    the source that produced it, which is exactly what the runbook's source-passport requirement
    exists to prevent. The most common cause is not a broken git but a *copy* of ``probes/`` made
    without its ``.git`` directory -- e.g. ``tar --exclude='.git'`` when staging to a compute
    node -- which fails this way silently and is easy to miss in a results dump.

    Note the manifest guard in :func:`enforce_manifest_expectations` will reject ``"unknown"``
    against an ``expect_source_revision`` field, but only when a manifest is in use; this warning
    covers the free-form path too.

    :returns: e.g. ``"93b60d7"`` or ``"93b60d7-dirty"``, or ``"unknown"`` if git fails.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", here, "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except Exception as exc:
        # No git here. On the eduLLM platform that is the normal case rather than a fault: the
        # image is built with .git excluded from the build context on purpose (the root
        # .dockerignore documents why -- actions/checkout leaves the run's token in
        # .git/config), so `git rev-parse` inside the container cannot work and never will.
        #
        # The platform passes the commit it built from as EDULLM_COMMIT_SHA, which is *stronger*
        # provenance than a local git rev: it is the immutable 40-hex the image digest was built
        # from, recorded by the submission pipeline rather than read from a mutable worktree.
        # Preferring it here is what keeps a platform run from writing "unknown" into a field
        # analysis then discards -- analyze_sigma.py rejects records whose
        # probe_source_revision is None or "unknown", so a run that trained correctly and cost
        # real money would be silently dropped at the aggregation step.
        platform_sha = os.environ.get("EDULLM_COMMIT_SHA", "").strip()
        if platform_sha:
            print(
                f"NOTE: no usable git in {here} ({type(exc).__name__}), which is expected in a "
                f"platform image built without .git. Recording provenance from "
                f"EDULLM_COMMIT_SHA instead.",
                file=sys.stderr,
            )
            return platform_sha[:7] if len(platform_sha) >= 7 else platform_sha
        print(
            f"WARNING: cannot determine probes/ git revision ({type(exc).__name__}: {exc}) and "
            f"EDULLM_COMMIT_SHA is unset. Recording 'unknown'. This run has NO source "
            f"provenance and must not be used for a manifest-grade result. Checked: {here}",
            file=sys.stderr,
        )
        return "unknown"


def build_mixer_factory(
    name: str,
    num_householder: int,
    n_heads: int,
    head_dim: int,
    backend: str = "triton",
    *,
    allow_neg_eigval: bool,
):
    """Return a ``(d_model, layer_idx) -> Module`` factory for the named mixer.

    :param name: One of ``kda``, ``gdn``, ``kda_hh``.
    :param num_householder: R, the number of Householder/delta factors per token
        (``kda_hh`` only).
    :param n_heads: Number of mixer heads.
    :param head_dim: Per-head key dimension.
    :param backend: ``triton`` or ``torch``.
    :param allow_neg_eigval: Beta regime, keyword-only and with no default so no
        call site can silently inherit the reflection regime. ``False`` is strict
        beta in (0, 1); ``True`` doubles beta into (0, 2).
    :returns: The factory callable.
    """
    from olmo_core.nn.attention import GatedDeltaNetConfig, KimiDeltaAttentionConfig
    from olmo_core.nn.transformer.init import InitMethod

    def factory(d_model: int, layer_idx: int) -> nn.Module:
        common = dict(n_heads=n_heads, head_dim=head_dim, expand_v=1.0)
        if name == "kda":
            cfg = KimiDeltaAttentionConfig(**common, allow_neg_eigval=allow_neg_eigval)
        elif name == "gdn":
            cfg = GatedDeltaNetConfig(**common, allow_neg_eigval=allow_neg_eigval)
        elif name == "kda_hh":
            # Householder-KDA: the novel arm. Imported lazily so the other arms
            # remain runnable before the kernel exists.
            from olmo_core.nn.attention import KimiDeltaHouseholderConfig

            # The triton backward is ~400x faster than the torch reference and is validated
            # against it (per-gradient relative error < 2e-2). Overridable so the two backends
            # can be compared at a fixed seed.
            cfg = KimiDeltaHouseholderConfig(
                **common,
                num_householder=num_householder,
                allow_neg_eigval=allow_neg_eigval,
                backend=backend,
            )
        else:
            raise ValueError(f"unknown mixer '{name}'")
        mixer = cfg.build(d_model, layer_idx=layer_idx, n_layers=1, init_device="cuda")

        # Initialize the mixer explicitly. THIS CALL IS LOAD-BEARING AND WAS MISSING.
        #
        # 'build()' only constructs; every OLMo-core mixer allocates its gate parameters with
        # 'torch.empty' -- KimiDeltaHouseholder does so at recurrent.py:1162-1163 for 'A_log'
        # and 'dt_bias' -- and relies on a separate 'init_weights' pass to fill them. In a real
        # OLMo-core model that pass comes from 'Transformer.init_weights'
        # (nn/transformer/init.py), but ProbeModel is a plain nn.Module, so nothing here called
        # it. Every probe run before this fix therefore trained 'A_log' and 'dt_bias' on
        # whatever those pages happened to hold.
        #
        # Why that is not merely untidy: uninitialized memory is only harmless if it is
        # *identical across arms*. Fresh pages read as zero, but CUDA's caching allocator
        # recycles freed blocks, and this file frees a whole model during the
        # '--match-non-embedding' solve above -- so the arm that runs a solve can see different
        # bytes than the arm that does not. That is an arm-dependent difference in the decay
        # gate, which is exactly the quantity under study.
        #
        # No generator is passed: 'main' seeds the global RNG with the 'init' stream
        # immediately before each 'build_model' call, and 'init_weights' draws from that same
        # global RNG, so the init remains a pure function of 'seeds["init"]'.
        #
        # 'num_blocks=1' matches the 'n_layers=1' already passed to 'build' above. Both are
        # deliberate: they select the depth-independent branch of 'init_weights', so a mixer's
        # init does not change when the probe's layer count does. Only 'InitMethod.llama' and
        # 'llama_depth' consume these, and 'normal' is what ProbeModel's own layers use.
        mixer.init_weights(
            init_method=InitMethod.normal,
            d_model=d_model,
            block_idx=layer_idx,
            num_blocks=1,
        )
        return mixer

    return factory


def resolve_seeds(args: argparse.Namespace) -> dict[str, int]:
    """Resolve the four seed streams from ``--bundle-id`` and/or explicit overrides.

    :param args: Parsed arguments carrying ``bundle_id`` and ``seed_{init,data,task,eval}``.
    :returns: Mapping ``{"init":…, "data":…, "task":…, "eval":…}``.
    :raises ValueError: If a stream is unspecified, or if two streams collide.
    """
    seeds: dict[str, int] = {}
    for stream, offset in SEED_STREAM_OFFSETS.items():
        explicit = getattr(args, f"seed_{stream}")
        if explicit is not None:
            seeds[stream] = explicit
        elif args.bundle_id is not None:
            seeds[stream] = offset + args.bundle_id
        else:
            raise ValueError(
                f"seed stream '{stream}' is unset: pass --bundle-id or --seed-{stream}"
            )
    if len(set(seeds.values())) != len(seeds):
        raise ValueError(
            f"seed streams must be pairwise distinct (they are the whole point), got {seeds}"
        )
    return seeds


def build_eval_bank(
    task: str, lengths: list[int], batch: int, seed_eval: int
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Materialize the held-out evaluation bank once, from the evaluation seed only.

    Building it up front (rather than regenerating inside :func:`evaluate`) is what
    makes it a *bank*: the same tensors can be checksummed, replayed, and compared
    against the training stream for instance collisions.

    :param task: Task key in :data:`tasks.TASKS`.
    :param lengths: Evaluation sequence lengths.
    :param batch: Number of sequences per length.
    :param seed_eval: The evaluation seed stream. Not derived from any other stream.
    :returns: Mapping from length to ``(inputs, targets)``, both CPU int64.
    """
    spec = TASKS[task]
    bank: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for length in lengths:
        gen = torch.Generator().manual_seed(seed_eval * 100_003 + length)
        bank[length] = spec["fn"](batch, length, gen)
    return bank


def bank_checksum(bank: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> str:
    """sha256 over the bank's tensors in ascending length order.

    :param bank: As returned by :func:`build_eval_bank`.
    :returns: Hex digest.
    """
    h = hashlib.sha256()
    for length in sorted(bank):
        x, y = bank[length]
        h.update(str(length).encode())
        h.update(x.contiguous().numpy().tobytes())
        h.update(y.contiguous().numpy().tobytes())
    return h.hexdigest()


def _row_digests(x: torch.Tensor) -> set[bytes]:
    """sha1 digest of every row of an int64 ``[B, T]`` tensor."""
    arr = x.contiguous().numpy()
    return {hashlib.sha1(arr[i].tobytes()).digest() for i in range(arr.shape[0])}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    bank: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> dict[int, float]:
    """Score the model on a fixed evaluation bank.

    :param model: The probe model.
    :param bank: Immutable evaluation bank from :func:`build_eval_bank`.
    :param device: Device string.
    :returns: mapping from sequence length to mean token accuracy.
    """
    model.eval()
    out: dict[int, float] = {}
    for length in sorted(bank):
        x, y = bank[length]
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        # Tasks may mark positions as ignored with -100 (MQAR scores only query positions).
        # Averaging over all positions would make an unsolved MQAR look ~90% correct.
        correct = logits.float().argmax(-1) == y
        valid = y != -100
        out[length] = (correct & valid).sum().item() / max(1, int(valid.sum()))
    model.train()
    return out


@torch.no_grad()
def measure_beta(model: nn.Module, x: torch.Tensor, allow_neg_eigval: bool) -> dict[str, float]:
    """Measure the realized beta and its pre-sigmoid logit on a real batch.

    Reproduces ``recurrent.py``'s ``beta = w_b(x).sigmoid()`` (times 2 under
    reflection) by hooking each mixer's ``w_b`` projection, so the reported range
    is the tensor the recurrence actually consumes rather than the range the flag
    is *supposed* to imply. Runbook §4.6 asserts ``0 <= beta <= 1`` for strict arms
    -- not ``0 < beta < 1``, because sigmoid saturates to exactly 1.0 in low
    precision.

    :param model: The probe model.
    :param x: A token batch ``[B, T]``.
    :param allow_neg_eigval: Whether the doubling is in effect.
    :returns: min/max/mean of beta and of the logit, plus the doubling factor.
    :raises RuntimeError: If no ``w_b`` projection was found, so the measurement
        would silently report nothing.
    """
    logits: list[torch.Tensor] = []
    handles = []
    for module in model.modules():
        w_b = getattr(module, "w_b", None)
        if isinstance(w_b, nn.Module):
            handles.append(w_b.register_forward_hook(lambda _m, _i, o: logits.append(o.detach())))
    if not handles:
        raise RuntimeError("no 'w_b' beta projection found on any mixer; cannot measure beta")
    try:
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(x)
        model.train()
    finally:
        for handle in handles:
            handle.remove()

    logit = torch.cat([t.float().reshape(-1) for t in logits])
    scale = 2.0 if allow_neg_eigval else 1.0
    beta = logit.sigmoid() * scale
    return {
        "beta_min": beta.min().item(),
        "beta_max": beta.max().item(),
        "beta_mean": beta.mean().item(),
        "logit_min": logit.min().item(),
        "logit_max": logit.max().item(),
        "logit_mean": logit.mean().item(),
        "beta_scale": scale,
        "n_beta_values": beta.numel(),
    }


def build_model(
    args: argparse.Namespace,
    spec: dict,
    *,
    ffn_dim: Optional[int],
    allow_neg_eigval: bool,
    device: str,
) -> ProbeModel:
    """Construct a :class:`model.ProbeModel` from parsed arguments."""
    return ProbeModel(
        build_mixer_factory(
            args.mixer,
            args.num_householder,
            args.n_heads,
            args.head_dim,
            args.backend,
            allow_neg_eigval=allow_neg_eigval,
        ),
        in_vocab=spec["in_vocab"],
        out_vocab=spec["out_vocab"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        ffn_dim=ffn_dim,
    ).to(device)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    ``--mixer``, ``--task`` and ``--beta-regime`` are *not* marked ``required``
    here because they may instead come from ``--manifest`` (or, for the first and
    third, from ``--arm``). :func:`main` enforces their presence after resolution.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=None,
        help="Frozen run manifest JSON; the source of truth. Only --out may accompany it.",
    )
    p.add_argument(
        "--arm",
        default=None,
        choices=sorted(ARMS),
        help="Canonical arm ID (runbook §3.2). Sets mixer, factor count and beta regime.",
    )
    p.add_argument("--mixer", choices=["kda", "gdn", "kda_hh"])
    p.add_argument("--num-householder", type=int, default=1)
    p.add_argument("--task", choices=sorted(TASKS))
    p.add_argument(
        "--beta-regime",
        choices=sorted(BETA_REGIMES),
        help="strict: beta in (0,1). reflection: beta in (0,2). Never pool the two.",
    )
    p.add_argument(
        "--bundle-id",
        type=int,
        default=None,
        help="Seed bundle. Expands to the four canonical streams via SEED_STREAM_OFFSETS.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=(
            "Identifier recorded as the record's 'run_id' when no manifest is in use. A "
            "manifest's own 'run_id' always wins; this only fills the field on the free-form "
            "path, where it was previously always null."
        ),
    )
    for stream in SEED_STREAM_OFFSETS:
        p.add_argument(
            f"--seed-{stream}",
            type=int,
            default=None,
            help=f"Override the '{stream}' seed stream (otherwise from --bundle-id).",
        )
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train-min", type=int, default=3)
    p.add_argument("--train-max", type=int, default=40)
    p.add_argument("--eval-lengths", type=int, nargs="+", default=[40, 64, 128, 256, 512])
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--ffn-dim",
        type=int,
        default=None,
        help="Inner SwiGLU FFN width. Omitted = no FFN, which is the pre-FFN model exactly.",
    )
    p.add_argument(
        "--match-non-embedding",
        type=int,
        default=None,
        help="Solve --ffn-dim so the non-embedding parameter count matches this target (R1-P).",
    )
    p.add_argument(
        "--match-arm",
        default=None,
        choices=sorted(ARMS),
        help=(
            "Resolve --match-non-embedding by building this arm and reading its ledger, "
            "instead of hardcoding a count that is only correct for one geometry. The named "
            "arm must be in the same beta regime as this run."
        ),
    )
    p.add_argument(
        "--param-tolerance",
        type=float,
        default=0.005,
        help="Fractional tolerance on the --match-non-embedding mismatch. Runbook P1.0: 0.5%%.",
    )
    # NOTE FOR AN LR SWEEP: this is the *peak* of a OneCycle schedule, not a constant rate
    # (see the scheduler construction in 'main'). Every run therefore ends at ~0 regardless of
    # this value, which is exactly why the final sampled loss says nothing about convergence.
    # Changing it changes the whole trajectory, which is the intended treatment; do not read a
    # cell's '--lr' as the rate that was in force at any particular step.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backend", default="triton", choices=["triton", "torch"])
    p.add_argument(
        "--param-ledger-only",
        action="store_true",
        help="Build the model, emit the parameter ledger, and exit without training.",
    )
    p.add_argument(
        "--no-collision-check",
        action="store_true",
        help="Skip the train/eval instance-collision check (runbook §5.3 requirement 4).",
    )
    p.add_argument("--out", default=None)
    return p


def main() -> None:
    p = build_parser()
    argv = sys.argv[1:]
    if "--manifest" in argv:
        # Manifest mode: only --manifest and --out are tolerated on the command line.
        # A manifest that can be silently overridden from the shell is not a record.
        preview = p.parse_args(argv)
        allowed = {"--manifest", "--out"}
        stray = [a for a in argv if a.startswith("--") and a not in allowed]
        if stray:
            raise SystemExit(
                f"--manifest is the source of truth; free-form overrides are rejected: {stray}"
            )
        out = preview.out
        args = load_manifest(preview.manifest, p)
        if out is not None:
            args.out = out
    else:
        args = p.parse_args(argv)
        args._manifest_data = {}

    apply_arm(args)
    for required in ("mixer", "task", "beta_regime"):
        if getattr(args, required, None) is None:
            raise SystemExit(f"--{required.replace('_', '-')} is required (or supply --arm/--manifest)")

    manifest_data = getattr(args, "_manifest_data", {}) or {}
    # Source-revision expectation is checkable before anything is built.
    check_manifest_expectations(manifest_data, eval_digest=None, non_embedding=None)

    seeds = resolve_seeds(args)
    allow_neg_eigval = BETA_REGIMES[args.beta_regime]
    device = "cuda"
    spec = TASKS[args.task]

    # Model init draws from the global torch RNG inside the OLMo-core mixer builds,
    # so the init stream is set here and nowhere else.
    torch.manual_seed(seeds["init"])

    # '--match-arm' resolves to a parameter target by *building* the named arm's FFN-free
    # model and reading its ledger, rather than making the caller type the number.
    #
    # The number is geometry-dependent: 1,400,524 is DP2-strict's non-embedding count at
    # d_model=256, 3 layers, 4 heads, head_dim=64 and nothing else. A hardcoded target
    # silently stops being a match the moment any of those changes, and it stops in the
    # direction that still runs and still reports 'completed' -- the arm keeps its name while
    # no longer controlling for what it is named after. Deriving it means the control cannot
    # drift away from the thing it controls for.
    if args.match_arm is not None:
        if args.match_non_embedding is not None:
            raise SystemExit("pass at most one of --match-arm and --match-non-embedding")
        if ARMS[args.match_arm].get("match_arm"):
            # A control cannot be matched to another control. The target is built FFN-free to
            # read its intrinsic cost, so matching to an arm that is itself FFN-matched would
            # read the count it has *before* its own match is applied -- silently targeting
            # the wrong number while looking like it worked.
            raise SystemExit(
                f"--match-arm {args.match_arm!r} is itself parameter-matched (to "
                f"{ARMS[args.match_arm]['match_arm']!r}). Match to the arm whose capacity is "
                f"being controlled for, not to another control."
            )
        target_args = argparse.Namespace(**vars(args))
        target_args.arm = args.match_arm
        apply_arm(target_args)  # raises on an unimplemented target
        if target_args.beta_regime != args.beta_regime:
            # Matching across regimes would fold a capacity control and a regime contrast
            # into one arm, which is what makes the result unreadable rather than merely
            # imprecise. Both regimes have their own R=2 arm; name the right one.
            raise SystemExit(
                f"--match-arm {args.match_arm!r} is in the {target_args.beta_regime!r} regime "
                f"but this run is {args.beta_regime!r}. A capacity control must match an arm "
                f"in its own regime, or the comparison confounds capacity with beta range."
            )
        torch.manual_seed(seeds["init"])
        target_model = build_model(
            target_args,
            spec,
            ffn_dim=None,
            allow_neg_eigval=BETA_REGIMES[target_args.beta_regime],
            device=device,
        )
        args.match_non_embedding = target_model.parameter_ledger()["non_embedding"]
        del target_model
        torch.cuda.empty_cache()
        print(
            f"MATCH_ARM {args.match_arm} R={target_args.num_householder} "
            f"non_embedding={args.match_non_embedding}",
            flush=True,
        )

    ffn_dim = args.ffn_dim
    ffn_solve: Optional[dict] = None
    if args.match_non_embedding is not None:
        if ffn_dim is not None:
            raise SystemExit("pass at most one of --ffn-dim and --match-non-embedding")
        # Build the FFN-free model first to read its actual non-embedding cost, then
        # solve. Reading it rather than deriving it analytically means the solve is
        # correct whatever the mixer's own parameter count turns out to be.
        torch.manual_seed(seeds["init"])
        base = build_model(
            args, spec, ffn_dim=None, allow_neg_eigval=allow_neg_eigval, device=device
        )
        base_non_embedding = base.parameter_ledger()["non_embedding"]
        del base
        torch.cuda.empty_cache()
        ffn_dim = solve_ffn_dim(
            args.match_non_embedding,
            base_non_embedding,
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
        ffn_solve = {
            "target_non_embedding": args.match_non_embedding,
            "base_non_embedding": base_non_embedding,
            "solved_ffn_dim": ffn_dim,
        }
        torch.manual_seed(seeds["init"])

    model = build_model(
        args, spec, ffn_dim=ffn_dim, allow_neg_eigval=allow_neg_eigval, device=device
    )
    ledger = model.parameter_ledger()
    n_params = ledger["total"]

    if ffn_solve is not None:
        mismatch = ledger["non_embedding"] - ffn_solve["target_non_embedding"]
        ffn_solve["achieved_non_embedding"] = ledger["non_embedding"]
        ffn_solve["mismatch_abs"] = mismatch
        ffn_solve["mismatch_pct"] = 100.0 * mismatch / ffn_solve["target_non_embedding"]
        ffn_solve["tolerance_pct"] = 100.0 * args.param_tolerance
        ffn_solve["within_tolerance"] = (
            abs(mismatch) <= args.param_tolerance * ffn_solve["target_non_embedding"]
        )
        print("FFN_SOLVE " + json.dumps(ffn_solve), flush=True)
        # Refuse rather than train. 'within_tolerance' was computed and printed but never
        # acted on, so a parameter-matched arm whose match missed still recorded
        # 'outcome: completed' -- and the only thing that arm exists to establish is that the
        # match held. A downstream reader comparing it to DP2-strict would be reading a
        # capacity difference as a regime difference, which is the exact confound the arm was
        # added to remove. Failing here costs one cell; passing costs the conclusion.
        if not ffn_solve["within_tolerance"]:
            raise SystemExit(
                f"parameter match missed: solved ffn_dim={ffn_dim} gives "
                f"{ffn_solve['achieved_non_embedding']} non-embedding parameters against a "
                f"target of {ffn_solve['target_non_embedding']} "
                f"({ffn_solve['mismatch_pct']:+.3f}%, tolerance "
                f"±{ffn_solve['tolerance_pct']:.3f}%). The FFN width is an integer, so a "
                f"target this close to the FFN-free cost may not be reachable at all; raise "
                f"--param-tolerance only if the residual is defensible for the comparison."
            )

    print("PARAM_LEDGER " + json.dumps(ledger), flush=True)
    # Before training: the parameter ledger must match what the manifest expects.
    check_manifest_expectations(
        manifest_data, eval_digest=None, non_embedding=ledger["non_embedding"]
    )
    if args.param_ledger_only:
        record = {
            "run_id": manifest_data.get("run_id") or args.run_id,
            "arm": args.arm,
            "probe_source_revision": probe_source_revision(),
            "mixer": args.mixer,
            "num_householder": args.num_householder,
            "task": args.task,
            "beta_regime": args.beta_regime,
            "allow_neg_eigval": allow_neg_eigval,
            "seeds": seeds,
            "ffn_dim": ffn_dim,
            "ffn_solve": ffn_solve,
            "param_ledger": ledger,
            "n_params": n_params,
            "outcome": "param_ledger_only",
        }
        print("RESULT " + json.dumps(record), flush=True)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(record, fh, indent=2)
        return

    # The evaluation bank is frozen before training, from the evaluation seed only.
    eval_bank = build_eval_bank(args.task, args.eval_lengths, args.batch, seeds["eval"])
    eval_digest = bank_checksum(eval_bank)
    eval_rows = set()
    if not args.no_collision_check:
        for x, _ in eval_bank.values():
            eval_rows |= _row_digests(x)
    print(f"EVAL_BANK sha256={eval_digest} lengths={sorted(eval_bank)}", flush=True)
    # Still before training: the bank must replay to the checksum the manifest names.
    check_manifest_expectations(
        manifest_data, eval_digest=eval_digest, non_embedding=ledger["non_embedding"]
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

    # Two independent training-side streams: 'data' drives the length curriculum
    # (which sequence shapes are seen, in which order) and 'task' drives the
    # instance content. Deriving one from the other would make them a single
    # stream wearing two names.
    data_gen = torch.Generator().manual_seed(seeds["data"])
    task_gen = torch.Generator().manual_seed(seeds["task"])
    collisions = 0
    loss_trace: list[tuple[int, float]] = []
    # A trailing window of per-step losses, reduced at the end into 'loss_summary'.
    #
    # WHY THE TRACE ALONE IS NOT ENOUGH, AND HOW IT MISLEADS. 'loss_trace' samples one
    # minibatch every 500 steps, so its last entry is a single batch drawn at the final step.
    # Under OneCycle the LR there is ~4e-9 -- the weights are frozen -- so a high value is a
    # hard batch, not a model that failed to converge. Reading that entry as a convergence
    # signal produced a false 'unconverged' finding on this very probe: runs that reached
    # ~0.0000 by step 1000 were classified as failures because one late batch read 0.82.
    # A mean over the tail cannot be spoofed by one draw.
    tail_losses: list[float] = []
    tail_window = max(1, args.steps // 20)
    nonfinite_steps = 0
    started = time.time()
    for step in range(args.steps):
        # Curriculum over lengths: sample uniformly in [train_min, train_max].
        length = int(torch.randint(args.train_min, args.train_max + 1, (1,), generator=data_gen).item())
        x, y = spec["fn"](args.batch, length, task_gen)
        if eval_rows:
            collisions += len(_row_digests(x) & eval_rows)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.float().reshape(-1, spec["out_vocab"]), y.reshape(-1)
        )
        if not torch.isfinite(loss):
            nonfinite_steps += 1
        if step >= args.steps - tail_window:
            tail_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == args.steps - 1:
            loss_trace.append((step, round(loss.item(), 4)))
            print(f"step {step:5d}  loss {loss.item():.4f}  len {length}", flush=True)

    finite_tail = sorted(x for x in tail_losses if x == x and abs(x) != float("inf"))
    loss_summary = {
        "tail_window_steps": tail_window,
        "tail_mean": sum(finite_tail) / len(finite_tail) if finite_tail else float("nan"),
        "tail_median": finite_tail[len(finite_tail) // 2] if finite_tail else float("nan"),
        "tail_max": finite_tail[-1] if finite_tail else float("nan"),
        "tail_min": finite_tail[0] if finite_tail else float("nan"),
        # The best the run ever reached at a sampled step, ignoring the first two samples so a
        # warmup value cannot stand in for it. This is the number that answers "did it fit at
        # all", which is a different question from "where did it end up".
        "trace_min_after_warmup": min((v for _, v in loss_trace[2:]), default=float("nan")),
    }
    print("LOSS_SUMMARY " + json.dumps(loss_summary), flush=True)

    acc = evaluate(model, eval_bank, device)
    beta_stats = measure_beta(model, eval_bank[max(eval_bank)][0].to(device), allow_neg_eigval)
    # Runbook §4.6: assert on the realized tensor, with the closed bound -- sigmoid
    # saturates to exactly 1.0 in low precision, so 0 < beta < 1 is the wrong assertion.
    beta_ok = 0.0 <= beta_stats["beta_min"] and beta_stats["beta_max"] <= beta_stats["beta_scale"]

    record = {
        # The manifest wins where one is in use -- it is the source of truth per runbook
        # §5.2.1 -- and '--run-id' fills the field on the free-form path, where it was
        # previously always null even when the caller knew the id.
        "run_id": manifest_data.get("run_id") or args.run_id,
        "arm": args.arm,
        "manifest": args.manifest,
        "probe_source_revision": probe_source_revision(),
        "mixer": args.mixer,
        "num_householder": args.num_householder,
        "task": args.task,
        "beta_regime": args.beta_regime,
        "allow_neg_eigval": allow_neg_eigval,
        "seeds": seeds,
        "bundle_id": args.bundle_id,
        "ffn_dim": ffn_dim,
        "ffn_solve": ffn_solve,
        "match_arm": args.match_arm,
        "param_ledger": ledger,
        "n_params": n_params,
        "backend": args.backend,
        "steps": args.steps,
        # The peak learning rate. Recorded because it is a *treatment* the moment a sweep
        # varies it, and until now the record could not tell two LR arms apart: every other
        # field of a 1e-3 run and a 3e-3 run of the same arm and bundle is identical, so an
        # aggregator keying on (arm, task, bundle) would see them as duplicate cells rather
        # than as two points on a curve.
        "lr": args.lr,
        "batch": args.batch,
        "train_range": [args.train_min, args.train_max],
        "eval_bank_sha256": eval_digest,
        "eval_collisions": None if args.no_collision_check else collisions,
        "accuracy_by_length": acc,
        "beta_stats": beta_stats,
        "beta_range_ok": beta_ok,
        "loss_trace": loss_trace,
        "loss_summary": loss_summary,
        "nonfinite_loss_steps": nonfinite_steps,
        "outcome": "completed" if beta_ok and nonfinite_steps == 0 else "invalid",
        "wall_seconds": round(time.time() - started, 1),
    }
    print("RESULT " + json.dumps(record), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2)


if __name__ == "__main__":
    main()
