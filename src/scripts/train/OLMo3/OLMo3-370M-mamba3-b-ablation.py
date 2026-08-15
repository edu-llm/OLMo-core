"""
The Mamba-3 ``b=2`` vs ``b=3`` ablation at 370M on 10B Dolma tokens.

**What this asks.** Mamba-3's transition is a block-diagonal product of ``2x2`` rotations. SO(2) is
abelian, so the layer's transition monoid is solvable and it is confined to TC^0. SO(3) is not, and
it contains ``A_5``, whose word problem is NC^1-complete -- so widening the block to ``b=3`` is the
minimal edit that lifts the layer out of TC^0. Whether that expressivity buys anything in language
modelling is an open question the literature has not settled, and this is the run that asks it at a
scale where the answer would mean something.

**What makes it an experiment rather than two runs.** The two arms differ in exactly one config
field, ``rotation_block_size``, and :func:`verify_arms` proves it by diffing the two model configs
leaf by leaf and refusing anything else. Everything a difference in loss could otherwise be blamed
on is pinned: one learning rate, one data mixture, one token budget, one seed per replicate shared
by both arms, one optimizer, one precision, one code path.

Two consequences of the treatment are not confounders and are recorded rather than removed:

- ``b=3`` carries **+589,824 parameters**, 0.16% of the non-embedding model, because SO(3) needs
  three angles per block where SO(2) needs one. It is irreducible. Under standard loss scaling it
  is worth about 1.2e-4 nats, three orders of magnitude under the seed spread on these arms.
  ``--param-match ffn`` narrows the
  treatment arm's MLP to erase it exactly, which is what the paper does for its own MIMO variants,
  at the cost of making the two arms' FFN GEMM shapes differ.
- ``b=3`` computes a genuine non-commutative prefix product where ``b=2`` collapses to a ``cumsum``.
  That is the treatment, not an artifact. It is what makes ``b=3`` slower, so throughput is an
  endpoint of this experiment and not a nuisance -- report loss at fixed tokens and wall-clock
  separately, and do not read one as the other.

**The architecture is the published one.** Both arms build
:meth:`Mamba3Config.mamba3_faithful_olmo3_370M`, not the older ``mamba3_olmo3_370M``, which departs
from published Mamba-3 SISO in seven ways beyond the rotation and whose ``b=2`` number therefore
is not a Mamba-3 number. Three deviations remain, all shared by both arms: ``d_state=192`` (128
cannot express ``b=3``), a group-shared rotation timescale (per-head is 18.8x the rotation cost at
``b=3``), and a learned ``A_log`` baseline under the token-dependent decay. See
``MAMBA3_B2_VS_B3.md`` for the full audit.

Data, optimizer, scheduler, trainer and the training loop are the dense
``OLMo3-370M-dolma2mix.py``'s, imported and not modified: the dolma2 source mixture on S3,
sequence length 4096, global batch 786,432 tokens.

**Two scales.** ``--scale 370M`` is the experiment as specified, at the ladder's 10B tokens.
``--scale 190M`` is the same architecture on the OLMo-3-190M shell at 20 tokens per non-embedding
parameter, which is about a seventh of the work. Both fit the runtime bound of the eight-A100 node
this organization provisions: rescaling the 30,442 tok/s/device the August wave measured for the
faithful arm puts a ``b=3`` cell near 12.5 h at 370M and 3.7 h at 190M, and ``b=2`` can only be
faster because it replaces the prefix product with a ``cumsum``. Prefer 370M; take 190M when the
budget buys more seeds that way, or for a second scale to check a result against.

Usage::

    # What will run, and the proof that it is a one-field difference. No network, no GPU.
    python src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py plan

    # Same checks on their own, written to a file to keep beside the results.
    python src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py verify --manifest-out arms.json

    # One cell. Run every cell `plan` prints.
    torchrun --standalone --nproc-per-node=8 \\
        src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py train mamba3-b2-r0 \\
        --arm b2 --replicate 0 --save-folder s3://<bucket>/mamba3-b2-r0

GPU runs on this platform go through ``edullm``; ``.edullm/run-b-ablation.yaml`` is the spec for
this wave. See "Running on GPUs" in ``CLAUDE.md``.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import rich

from olmo_core.data import TokenizerConfig
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.mamba3.config import (
    FAITHFUL_190M_INTERMEDIATE_SIZE,
    FAITHFUL_370M_INTERMEDIATE_SIZE,
)
from olmo_core.nn.mamba3.mixer import ROTATION_TIMESCALES, Mamba3MixerConfig
from olmo_core.nn.utils import no_weight_decay_param_names
from olmo_core.optim import OptimGroupOverride
from olmo_core.train.callbacks import Mamba3BackendMonitorCallback

# The dense script owns the data, optimizer, scheduler, trainer and training loop, and is not
# modified by this one. Its filename is not a valid module name, so it is loaded by path and
# registered in `sys.modules` before exec so its dataclasses can resolve their own module.
_DOLMA2_PATH = Path(__file__).with_name("OLMo3-370M-dolma2mix.py")
_spec = importlib.util.spec_from_file_location("olmo3_370m_dolma2mix", _DOLMA2_PATH)
assert _spec is not None and _spec.loader is not None
dolma2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dolma2
_spec.loader.exec_module(dolma2)

sys.path.insert(0, str(_DOLMA2_PATH.parent.parent / "smoketests"))
from mamba3_sentinel import Mamba3SentinelCallback  # noqa: E402  # isort: skip

log = dolma2.log

# --- The two arms ----------------------------------------------------------------------------

#: Arm names, control first.
ARMS = ("b2", "b3")

#: The baseline. Named, because it decides whose recipe the other arm inherits -- most visibly the
#: shared learning rate, which is derived once from this arm and never re-derived for the other.
CONTROL_ARM = "b2"

#: The whole treatment.
ARM_BLOCK_SIZE = {"b2": 2, "b3": 3}

#: Reserved seeds, one per replicate, shared by both arms of that replicate. Replicate 0 reuses the
#: dense ladder's seed so the first pair sits on the same data order as the repository's other 370M
#: runs. Fixed and finite on purpose: a plan that can invent a seed cannot be pre-registered.
SEEDS = (dolma2.DEFAULT_SEED, 210007, 220014, 230021, 240028)

#: How to handle the parameters ``b=3`` necessarily adds. ``off`` leaves the FFN identical and lets
#: the treatment arm be 0.16% larger; ``ffn`` narrows the treatment arm's MLP to match exactly.
PARAM_MATCH_MODES = ("off", "ffn")
DEFAULT_PARAM_MATCH = "off"

#: Pinned, not derived. The ladder formula reads the parameter count, so leaving it to derive would
#: hand the two arms different learning rates off the back of the 0.16% parameter gap -- a silent
#: confounder in a comparison whose entire claim is that one field differs. 3e-4 rather than the
#: ladder's ~7.8e-4 because that is where this architecture has actually trained at this scale;
#: the August wave ran the faithful arm at 3e-4, and the retrospective on the runs that plateaued
#: recommended sweeping 1e-4 / 2e-4 / 3e-4. Both arms move together if you change it.
DEFAULT_LEARNING_RATE = 3e-4

#: Named here rather than left to ``MAMBA3_ROTATION_SCAN_IMPL``, so it lands in the saved config and
#: the startup banner. In the environment alone, a relaunch that lost the export silently fell back
#: to the chunked scan -- 2.2x slower, and it raises nothing.
ROTATION_SCAN_IMPL = "quaternion"

#: The ``b=2`` cumsum and the ``b=3`` quaternion prefix product are both exact, so the arms can and
#: must share one SSD backend.
SSD_BACKEND = "official_fast"

#: The two model scales, each named for the OLMo-3 reference it is parameter-matched to.
#:
#: 370M is the experiment as specified and the default. 190M exists because a 370M cell at the
#: ladder's 10B tokens does not fit the runtime bound of the eight-A100 node this organization
#: provisions, and ``b=3`` is the slower arm, so the pair would straddle the bound -- which is
#: worse than both being slow, because the two arms could not then share one request. Runtime goes
#: as parameters times tokens, so at a fixed tokens-per-parameter ratio it is quadratic in the
#: parameter count and 190M at Chinchilla is about a quarter of the work.
SCALES = ("370M", "190M")
DEFAULT_SCALE = "370M"

#: Each scale's OLMo-3 reference: the preset that builds it, its non-embedding parameter count, and
#: the token budget it trains on.
#:
#: The two budgets are chosen on different grounds and that is deliberate. 370M keeps the ladder's
#: 10B, which is 1.35x Chinchilla and what this repository's dense and Gated DeltaNet runs at this
#: size used, so the arms can be read beside them. 190M is Chinchilla-optimal at 20 tokens per
#: non-embedding parameter, because the ratio is the thing shrinking the model buys: the August
#: eight-arm wave ran at 1.54 tokens per parameter, and its own retrospective named that the single
#: largest reason its result was not interpretable.
#:
#: Counts are hardcoded rather than computed so that a change to a dense reference shows up as a
#: failing check here instead of silently re-baselining the ablation.
SCALE_REFERENCES = {
    "370M": {"non_embedding_params": 371_262_464, "token_budget": dolma2.DEFAULT_TOKEN_BUDGET},
    "190M": {"non_embedding_params": 190_354_176, "token_budget": 20 * 190_293_192},
}

#: How far either arm may sit from that reference before the label is wrong. The published expand
#: factor of 2 makes the mixer far wider than the layer it replaces, so the feed-forward width is
#: solved to land here; the default lands at +0.049%.
MAX_REFERENCE_DRIFT = 0.005

#: The leaf config keys the two arms are allowed to differ in. Anything else is a confounder and
#: :func:`verify_arms` refuses it.
TREATMENT_FIELD = "block.mamba3.sequence_mixer.rotation_block_size"
PARAM_MATCH_FIELDS = (
    "block.mamba3.feed_forward.hidden_size",
    "block.attn.feed_forward.hidden_size",
)


class ArmContractError(RuntimeError):
    """Raised when the two arms differ in something other than the treatment."""


def expected_config_difference(param_match: str) -> set:
    """
    Leaf config keys the two arms may differ in under a given parameter-matching mode.

    :param param_match: One of :data:`PARAM_MATCH_MODES`.
    """
    if param_match == "ffn":
        return {TREATMENT_FIELD, *PARAM_MATCH_FIELDS}
    return {TREATMENT_FIELD}


# --- Building the arms -----------------------------------------------------------------------


def build_model_config(
    arm: str,
    *,
    scale: str = DEFAULT_SCALE,
    intermediate_size: Optional[int] = None,
    param_match: str = DEFAULT_PARAM_MATCH,
    d_state: int = 192,
    rotation_timescale: str = "group_mean",
) -> Mamba3Config:
    """
    Build one arm's model config.

    :param arm: One of :data:`ARMS`.
    :param scale: One of :data:`SCALES`.
    :param intermediate_size: Feed-forward width of the control arm, or ``None`` for the scale's
        solved default. The treatment arm's may be narrowed from it under ``param_match="ffn"``.
    :param param_match: One of :data:`PARAM_MATCH_MODES`.
    :param d_state: SSM state size. Must admit both block sizes.
    :param rotation_timescale: One of
        :data:`~olmo_core.nn.mamba3.mixer.ROTATION_TIMESCALES`, shared by both arms.

    :raises SystemExit: On an unknown arm or scale, or an inexpressible configuration.
    """
    if arm not in ARM_BLOCK_SIZE:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")
    if scale not in SCALES:
        raise SystemExit(f"unknown scale {scale!r}; expected one of {SCALES}")
    if param_match not in PARAM_MATCH_MODES:
        raise SystemExit(
            f"unknown --param-match {param_match!r}; expected one of {PARAM_MATCH_MODES}"
        )

    preset = _PRESET_BY_SCALE[scale]
    if intermediate_size is None:
        intermediate_size = _DEFAULT_INTERMEDIATE_SIZE_BY_SCALE[scale]

    width = intermediate_size
    if param_match == "ffn" and arm != CONTROL_ARM:
        width = _matched_intermediate_size(
            arm,
            scale=scale,
            intermediate_size=intermediate_size,
            d_state=d_state,
            rotation_timescale=rotation_timescale,
        )

    return preset(
        vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        rotation_block_size=ARM_BLOCK_SIZE[arm],
        d_state=d_state,
        intermediate_size=width,
        rotation_timescale=rotation_timescale,
        rotation_scan_impl=ROTATION_SCAN_IMPL,
        ssd_backend=SSD_BACKEND,
        prefer_official_kernel=True,
    )


_PRESET_BY_SCALE = {
    "370M": Mamba3Config.mamba3_faithful_olmo3_370M,
    "190M": Mamba3Config.mamba3_faithful_olmo3_190M,
}
_DEFAULT_INTERMEDIATE_SIZE_BY_SCALE = {
    "370M": FAITHFUL_370M_INTERMEDIATE_SIZE,
    "190M": FAITHFUL_190M_INTERMEDIATE_SIZE,
}


def _matched_intermediate_size(
    arm: str, *, scale: str, intermediate_size: int, d_state: int, rotation_timescale: str
) -> int:
    """
    Feed-forward width that makes ``arm`` weigh exactly what the control arm weighs.

    Solved against the built configs rather than from a formula, and refused outright if the
    surcharge is not a whole number of feed-forward units -- an approximate "parameter match" is
    worse than an honest mismatch, because it reads as exact.
    """
    common: dict[str, Any] = dict(
        scale=scale, d_state=d_state, rotation_timescale=rotation_timescale
    )
    control = build_model_config(
        CONTROL_ARM, intermediate_size=intermediate_size, param_match="off", **common
    )
    treated = build_model_config(
        arm, intermediate_size=intermediate_size, param_match="off", **common
    )
    # What a unit of feed-forward width is worth, measured at this scale rather than assumed:
    # three SwiGLU matrices of width `d_model` across every layer, which differs between the two
    # scales and would be a silent off-by-a-factor if it were written down once.
    narrower = build_model_config(
        CONTROL_ARM, intermediate_size=intermediate_size - 1, param_match="off", **common
    )
    per_unit = control.num_params - narrower.num_params
    surcharge = treated.num_params - control.num_params
    units, remainder = divmod(surcharge, per_unit)
    if remainder:
        raise SystemExit(
            f"cannot parameter-match {arm} at {scale}: its {surcharge:,} extra parameters are not "
            f"a whole number of feed-forward units ({per_unit:,}). Run with --param-match off and "
            f"report the difference instead."
        )
    return intermediate_size - units


# --- Proving the arms differ in one thing ------------------------------------------------------


def _flatten(value: Any, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted leaf keys."""
    if isinstance(value, dict):
        flat: dict = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, (list, tuple)):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
        return flat
    return {prefix: value}


def arm_config_difference(control: Mamba3Config, treated: Mamba3Config) -> dict:
    """
    Every leaf config field on which two arms disagree.

    Comparing serialized configs rather than a hand-written list of fields is the point: a field
    added to the mixer next month is covered without anybody remembering to cover it.

    :returns: ``{dotted key: (control value, treated value)}``. Keys present in only one config
        appear with ``None`` on the missing side.
    """
    left = _flatten(control.as_config_dict())
    right = _flatten(treated.as_config_dict())
    return {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def verify_arms(
    configs: dict, *, param_match: str = DEFAULT_PARAM_MATCH, scale: str = DEFAULT_SCALE
) -> dict:
    """
    Check the comparison's contract and return the manifest that records it.

    :param configs: ``{arm name: model config}``, which must cover :data:`ARMS`.
    :param param_match: One of :data:`PARAM_MATCH_MODES`.
    :param scale: One of :data:`SCALES`, which decides the reference to check against.

    :raises ArmContractError: If the arms differ in anything but the treatment, or if the parameter
        counts do not match what the mode promises.

    :returns: A JSON-serializable manifest.
    """
    missing = set(ARMS) - set(configs)
    if missing:
        raise ArmContractError(f"missing arm(s): {sorted(missing)}")

    if scale not in SCALES:
        raise ArmContractError(f"unknown scale {scale!r}; expected one of {SCALES}")
    reference = SCALE_REFERENCES[scale]["non_embedding_params"]

    control = configs[CONTROL_ARM]
    manifest: dict = {
        "experiment": f"mamba3-{scale.lower()}-b2-vs-b3",
        "scale": scale,
        "param_match": param_match,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "rotation_scan_impl": ROTATION_SCAN_IMPL,
        "ssd_backend": SSD_BACKEND,
        "reference_non_embedding_params": reference,
        "token_budget": SCALE_REFERENCES[scale]["token_budget"],
        "arms": {},
        "config_difference": {},
    }

    for arm in ARMS:
        config = configs[arm]
        assert isinstance(config.block, dict)
        mixer = config.block["mamba3"].sequence_mixer
        assert isinstance(mixer, Mamba3MixerConfig)
        manifest["arms"][arm] = {
            "rotation_block_size": mixer.rotation_block_size,
            "num_params": config.num_params,
            "num_non_embedding_params": config.num_non_embedding_params,
            "intermediate_size": config.block["mamba3"].feed_forward.hidden_size,
            "d_state": mixer.d_state,
            "rotation_timescale": mixer.rotation_timescale,
            "drift_from_reference": round(
                (config.num_non_embedding_params - reference) / reference, 6
            ),
        }

    # Exactly one treatment arm, so one diff. Asserted rather than assumed, because the manifest
    # below carries a single `config_difference` and a third arm would silently vanish from it.
    (treatment_arm,) = [arm for arm in ARMS if arm != CONTROL_ARM]
    allowed = expected_config_difference(param_match)
    difference = arm_config_difference(control, configs[treatment_arm])
    unexpected = set(difference) - allowed
    if unexpected:
        detail = ", ".join(f"{key}={difference[key]}" for key in sorted(unexpected))
        raise ArmContractError(
            f"{CONTROL_ARM} and {treatment_arm} differ in {len(unexpected)} field(s) beyond the "
            f"treatment, which would confound the comparison: {detail}"
        )
    if TREATMENT_FIELD not in difference:
        raise ArmContractError(
            f"{CONTROL_ARM} and {treatment_arm} do not differ in {TREATMENT_FIELD}: both arms "
            f"would train the same model"
        )
    # Lists, not tuples: the manifest is written as JSON and has to round-trip.
    manifest["config_difference"] = {key: list(value) for key, value in difference.items()}

    counts = {arm: configs[arm].num_params for arm in ARMS}
    surcharge = counts["b3"] - counts["b2"]
    if param_match == "ffn" and surcharge != 0:
        raise ArmContractError(
            f"--param-match ffn promised equal parameter counts but the arms differ by "
            f"{surcharge:,}"
        )
    if param_match == "off" and surcharge <= 0:
        raise ArmContractError(
            f"b=3 must carry the extra angle projection, but the surcharge is {surcharge:,}"
        )
    manifest["parameter_surcharge"] = surcharge
    manifest["parameter_surcharge_fraction_of_total"] = round(surcharge / counts["b2"], 6)
    manifest["parameter_surcharge_fraction_of_non_embedding"] = round(
        surcharge / configs["b2"].num_non_embedding_params, 6
    )

    for arm, row in manifest["arms"].items():
        if abs(row["drift_from_reference"]) > MAX_REFERENCE_DRIFT:
            raise ArmContractError(
                f"arm {arm} is {row['drift_from_reference']:.2%} from the {scale} reference; "
                f"adjust --intermediate-size"
            )
    return manifest


# --- The plan --------------------------------------------------------------------------------


def build_plan(opts) -> list:
    """
    Every cell of the wave, in the order they should run.

    Arm-major, so a wave that is cut short still has complete replicates of the control. Both arms
    of a replicate share a seed, which under the dense recipe pairs data order, the mixture draw
    and initialization all at once.
    """
    cells: list = []
    scale = opts.scale.lower()
    for arm in ARMS:
        for replicate in range(opts.replicates):
            cells.append(
                {
                    "index": len(cells),
                    "arm": arm,
                    "replicate": replicate,
                    "seed": SEEDS[replicate],
                    "rotation_block_size": ARM_BLOCK_SIZE[arm],
                    "run_name": f"mamba3-{scale}-{arm}-r{replicate}",
                }
            )
    return cells


def resolve_learning_rate(opts, arm: str) -> float:
    """
    The learning rate for an arm, which is the same learning rate for every arm.

    Takes ``arm`` and ignores it, deliberately: the signature is where the invariant is stated, and
    a future change that makes the rate arm-dependent has to go through this function to do it.
    """
    del arm
    return opts.lr


# --- Training one cell -----------------------------------------------------------------------


def build_config(opts, overrides):
    """
    Build the dense experiment config, then swap in this arm's Mamba-3 model.

    The dense builder owns everything that is not the model. Three things are corrected after it
    runs: the model, the learning rate (which it derives from the parameter count and which must
    not move between arms), and the weight-decay exemption for the SSM timescale parameters, which
    it does not know about.
    """
    dolma2_opts, dolma2_overrides = _dense_opts(opts, overrides)
    config = dolma2.build_config(dolma2_opts, dolma2_overrides)

    model_config = build_model_config(
        opts.arm,
        scale=opts.scale,
        intermediate_size=opts.intermediate_size,
        param_match=opts.param_match,
        d_state=opts.d_state,
        rotation_timescale=opts.rotation_timescale,
    )
    config.model = model_config

    # The dense builder derived this from the dense model's parameter count. Overwrite it: the
    # comparison needs one rate across both arms, and the ladder formula would not give one.
    config.train_module.optim.lr = resolve_learning_rate(opts, opts.arm)

    meta_model = model_config.build(init_device="meta")
    no_decay = sorted(no_weight_decay_param_names(meta_model))
    if no_decay:
        # A_log, dt_bias and D set the recurrence's timescale rather than its capacity; decaying
        # them pulls |A| toward 1 and dt toward softplus(0), squeezing the memory horizon from both
        # ends. `fixed_fields` pins it so a resume cannot restore the old non-zero value.
        optim = config.train_module.optim
        optim.group_overrides = [
            *(optim.group_overrides or []),
            OptimGroupOverride(params=no_decay, opts=dict(weight_decay=0.0)),
        ]
        if "weight_decay" not in optim.fixed_fields:
            optim.fixed_fields = (*optim.fixed_fields, "weight_decay")

    # Backstop against the ablation's one unrecoverable failure: an arm that trains as the other
    # one. The sentinel re-checks the block size on the built model at `pre_train` and then watches
    # grad norm, skip rate, plateau and decay horizon each step.
    config.trainer = config.trainer.with_callback(
        "mamba3_sentinel",
        Mamba3SentinelCallback(
            run_dir=dolma2_opts.work_dir,
            expected_rotation_block_size=ARM_BLOCK_SIZE[opts.arm],
            sequence_length=config.train_module.max_sequence_length,
            cancel_on_alert=True,
        ),
    )
    config.trainer = config.trainer.with_callback(
        "mamba3_backend_monitor",
        Mamba3BackendMonitorCallback(expected_backend=SSD_BACKEND),
    )

    # The scan is named identically on both arms so the config diff stays one field wide, but at
    # b=2 it describes nothing: SO(2) prefix products collapse to a cumsum and the branch
    # short-circuits before the implementation is read. Say so, rather than let a b=2 log line and
    # a b=2 saved config both claim a quaternion scan ran.
    scan = ROTATION_SCAN_IMPL
    if ARM_BLOCK_SIZE[opts.arm] == 2:
        scan = f"{ROTATION_SCAN_IMPL} (inert at b=2; SO(2) uses a cumsum of angles)"
    log.info(
        "Mamba-3 b-ablation scale=%s arm=%s b=%d (%s) params=%s non-emb=%s tokens=%s lr=%.3e "
        "seed=%d param_match=%s scan=%s",
        opts.scale,
        opts.arm,
        ARM_BLOCK_SIZE[opts.arm],
        "TC^0 control" if opts.arm == CONTROL_ARM else "NC^1 treatment",
        f"{model_config.num_params:,}",
        f"{model_config.num_non_embedding_params:,}",
        f"{opts.token_budget:,}",
        config.train_module.optim.lr,
        dolma2_opts.seed,
        opts.param_match,
        scan,
    )
    return config


def _dense_opts(opts, overrides):
    """
    Run the dense script's own parser over this cell's arguments.

    Round-tripping through ``dolma2.parse_args`` rather than assembling a namespace by hand means
    every dense default and every dense validation applies unchanged, including the one that
    refuses a non-durable save folder.
    """
    argv = [
        opts.run_name,
        "--seed",
        str(SEEDS[opts.replicate]),
        "--lr",
        repr(resolve_learning_rate(opts, opts.arm)),
        "--token-budget",
        str(opts.token_budget),
        "--data-config",
        opts.data_config,
        *opts.dense_args,
    ]
    saved = sys.argv
    sys.argv = [saved[0], *argv, *overrides]
    try:
        return dolma2.parse_args()
    finally:
        sys.argv = saved


# --- CLI -------------------------------------------------------------------------------------


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scale",
        choices=SCALES,
        default=DEFAULT_SCALE,
        help="Model size, named for the OLMo-3 reference it is parameter-matched to. Each scale "
        "carries its own token budget; see --token-budget.",
    )
    parser.add_argument(
        "--param-match",
        choices=PARAM_MATCH_MODES,
        default=DEFAULT_PARAM_MATCH,
        help="How to handle the parameters b=3 necessarily adds. 'off' keeps the FFN identical "
        "and lets b=3 be 0.16%% larger; 'ffn' narrows the b=3 MLP to match exactly.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Feed-forward width of the control arm. Defaults to the width solved for the scale's "
        "reference parameter count.",
    )
    parser.add_argument(
        "--d-state",
        type=int,
        default=192,
        help="SSM state size, shared by both arms. Must be divisible by 2 and 3.",
    )
    parser.add_argument(
        "--rotation-timescale",
        choices=ROTATION_TIMESCALES,
        default="group_mean",
        help="Shared by both arms. 'per_head' is the published semantics and costs 18.8x on the "
        "b=3 rotation; 'group_mean' is the recorded deviation this wave runs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Pinned, and shared by both arms. Do not let this be derived per arm.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Defaults to the scale's own budget: 10B at 370M, which is the ladder's recipe and "
        "1.35x Chinchilla, and 20 tokens per non-embedding parameter at 190M.",
    )
    parser.add_argument("--data-config", type=str, default=dolma2.DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--replicates",
        type=_replicate_count,
        default=1,
        help=f"Seeds per arm, at most {len(SEEDS)}. Two 10B-token runs is the minimum experiment; "
        f"more is better and costs proportionally.",
    )


def _replicate_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--replicates must be an integer, got {value!r}")
    if not 1 <= count <= len(SEEDS):
        raise argparse.ArgumentTypeError(
            f"--replicates must be between 1 and {len(SEEDS)} (the reserved seeds), got {count}"
        )
    return count


def parse_args(argv: Optional[list] = None):
    """
    Parse this script's own arguments.

    ``train`` additionally accepts every dense flag (``--save-folder``, ``--work-dir``,
    ``--rank-microbatch-size``, ``--no-wandb``, dotted config overrides, ...) and passes them
    through untouched; ``--seed`` and ``--lr`` are set by the plan and are not the caller's to give.
    """
    parser = argparse.ArgumentParser(
        prog="OLMo3-370M-mamba3-b-ablation.py",
        description="Mamba-3 b=2 vs b=3 at 370M on 10B Dolma tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the wave and the single-field proof.")
    _shared_arguments(plan)
    plan.add_argument("--manifest-out", type=str, default=None)

    verify = subparsers.add_parser("verify", help="Check the contract and exit.")
    _shared_arguments(verify)
    verify.add_argument("--manifest-out", type=str, default=None)

    train = subparsers.add_parser("train", help="Train one cell (launch under torchrun).")
    _shared_arguments(train)
    train.add_argument("run_name", type=str)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--replicate", type=int, default=0)
    train.add_argument(
        "--dry-run", action="store_true", help="Build and print the config, then exit."
    )

    opts, dense_args = parser.parse_known_args(argv)
    opts.dense_args = dense_args
    if opts.token_budget is None:
        opts.token_budget = SCALE_REFERENCES[opts.scale]["token_budget"]
    if opts.command == "train":
        if not 0 <= opts.replicate < len(SEEDS):
            parser.error(f"--replicate must be between 0 and {len(SEEDS) - 1}")
        _refuse_reserved_dense_flags(parser, dense_args)
        if opts.dry_run:
            opts.dense_args = [*opts.dense_args, "--dry-run"]
    return opts


#: Dense flags this script sets itself and a caller must not. Unlike `--lr` or `--token-budget`,
#: which this script's own parser declares and therefore swallows, `--seed` reaches the dense parser
#: untouched -- and argparse keeps the *last* occurrence, so a passed-through one would quietly
#: replace the replicate's. That unpairs the two arms and nothing downstream can see it happen: by
#: the time the config exists the seed looks like an ordinary part of the recipe.
_RESERVED_DENSE_FLAGS = {"--seed": "--replicate"}


def _refuse_reserved_dense_flags(parser: argparse.ArgumentParser, dense_args: list) -> None:
    for argument in dense_args:
        name = argument.split("=", 1)[0]
        if name in _RESERVED_DENSE_FLAGS:
            parser.error(
                f"{name} is set by the replicate and is not the caller's to give: both arms of a "
                f"replicate must share it. Use {_RESERVED_DENSE_FLAGS[name]} to choose which "
                f"reserved seed the cell runs on."
            )


def _arm_configs(opts) -> dict:
    return {
        arm: build_model_config(
            arm,
            scale=opts.scale,
            intermediate_size=opts.intermediate_size,
            param_match=opts.param_match,
            d_state=opts.d_state,
            rotation_timescale=opts.rotation_timescale,
        )
        for arm in ARMS
    }


def _write_manifest(manifest: dict, path: Optional[str]) -> None:
    if path is None:
        return
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest written to {path}")


def _print_manifest(manifest: dict) -> None:
    rich.print(manifest)
    print("\nOne field differs, and it is the treatment:")
    for key, (control, treated) in manifest["config_difference"].items():
        print(f"  {key}: {CONTROL_ARM}={control!r} -> b3={treated!r}")


def main():
    opts = parse_args()

    if opts.command in ("plan", "verify"):
        manifest = verify_arms(_arm_configs(opts), param_match=opts.param_match, scale=opts.scale)
        _print_manifest(manifest)
        if opts.command == "plan":
            print(f"\n{2 if opts.replicates == 1 else opts.replicates * 2} cells:")
            for cell in build_plan(opts):
                print(
                    f"  [{cell['index']}] torchrun --standalone --nproc-per-node=8 {sys.argv[0]} "
                    f"train {cell['run_name']} --arm {cell['arm']} "
                    f"--replicate {cell['replicate']} --save-folder s3://<bucket>/{cell['run_name']}"
                )
            print(
                f"\n{opts.token_budget:,} tokens per cell from {opts.data_config}, "
                f"lr {opts.lr:g} on both arms."
                f"\n\nAny flag you add, add to every cell. The contract this script checks covers "
                f"\nthe model and the recipe it builds, not what you type after it -- a "
                f"--rank-microbatch-size"
                f"\nor an --eval-data given to one arm and not the other is a difference nothing "
                f"here will catch."
                f"\nEach cell writes its full resolved config beside its checkpoints, so the two "
                f"can be diffed"
                f"\nafter the fact as well as before."
            )
        _write_manifest(manifest, opts.manifest_out)
        return

    # `verify_arms` runs before the run does, not after: an arm that fails the contract must cost a
    # process start, not a GPU-hour and a result nobody can use.
    verify_arms(_arm_configs(opts), param_match=opts.param_match, scale=opts.scale)

    if opts.dry_run:
        rich.print(build_config(opts, []))
        print("\nDry run OK -- config built. Drop --dry-run and launch under torchrun to train.")
        return

    dolma2.prepare_training_environment()
    try:
        dolma2.train(build_config(opts, []))
    finally:
        dolma2.teardown_training_environment()


if __name__ == "__main__":
    main()
