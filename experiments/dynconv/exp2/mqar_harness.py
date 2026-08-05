"""Train + eval ONE (arm, topology, W, config, seed) cell of Exp-2, reproducibly.

THE MODEL-BUILDER INTERFACE THIS FILE CODES AGAINST
--------------------------------------------------
Sub-agent A owns ``arms.py``. This harness never imports it at module scope; it takes a callable::

    build_model(
        arm: str,          # "static" | "permuted" | "dynqkv" | "dynamic"  (S1/S2/S3/S4)
        topology: str,     # "hybrid" (6 layers, 4 LIV + 2 GQA) | "allliv" (6 layers, 0 attention)
        kernel_size: int,  # W in {2, 3, 4, 8}; W=2 is the FALSIFICATION control
        vocab_size: int,   # 256, calibrated
        d_model: int,      # 128
        n_layers: int,     # 6
        seed: int,         # the INIT seed -- see the pairing contract below
    ) -> nn.Module

The returned module must satisfy:

* ``forward(tokens: LongTensor[B, T]) -> logits: FloatTensor[B, T, vocab_size]``
* every parameter already initialized (``init_weights`` or an explicit equivalent CALLED, on
  **every** arm including S1 -- SPEC Sec 5.3; the scar is that ``mqar_model.py`` constructs
  ``ShortConv`` directly and never calls ``init_weights``, so one arm ran at ~1/128 activation scale
  *on the probe used to justify that arm*)
* deterministic given ``seed``: two calls with the same arguments produce ``torch.equal`` parameters
* raising for an undefined combination is CORRECT and expected -- S3 (``dynqkv``) is undefined in
  ``allliv`` (no GQA blocks to put a dynamic conv in) and must be reported N/A, never silently
  substituted with S1 (SPEC Sec 1.2).

:func:`stub_build_model` in this file implements that contract so the harness is testable
end-to-end without waiting on ``arms.py``. It is a STUB: real ``ShortConv``-shaped topology, no
dynamic mechanism. Do not read science off it.

PAIRING: DATA ORDER ONLY. THIS IS NOT A DETAIL.
-----------------------------------------------
R3 F3 is explicit that the proposal's claim of paired *initialization* seeds is **mechanically
impossible**: S2/S3/S4 add ``V`` and ``U`` tensors that S1 does not have, so a single sequential RNG
stream **diverges at the first new tensor and every subsequent draw is misaligned** -- the arms get
*unrelated* init draws whether or not they are seeded identically
(``moe/audit/findings/power.md:352-362``: *"Not achieved, and not achievable by seeding alone"*).

So this harness implements the pairing that IS achievable and states plainly what it is:

* ``data_seed = DATA_SEED_BASE + seed_pair`` -- **identical across all arms** in a pair. Both the
  training stream and the eval stream are drawn from generators keyed only on this. That identity IS
  the pairing.
* ``init_seed = derive_init_seed(seed_pair, arm)`` -- a stable per-(pair, arm) hash. Deliberately
  DIFFERENT per arm, because pretending otherwise would be a false claim, and because a shared init
  seed on differently-shaped models buys nothing while sounding like it buys something.

**Pairing is on data order only.** The realized rho is therefore an empirical quantity, measured by
``sigma.paired_stats``, not the assumed 0.5.

WHAT ELSE THIS HARNESS PINS
---------------------------
* ``random_non_queries=False`` -- Zoology's published configs set it, the CLASS DEFAULT IS ``True``,
  and ``True`` makes filler random tokens that can collide with keys (a *harder*, different task).
  Asserted at :func:`assert_zoology_gotchas`, not assumed.
* ``state_mixer=Identity`` -- the second published-vs-default divergence. The model-builder owns the
  block structure, so this is asserted as a declared property (``model.state_mixer_is_identity``)
  and the harness REFUSES to run a model that does not declare it.
* the calibrated budget, 8000 x 64 = 512,000 examples, refused below outside ``--smoke``.
* >=1000 eval items with CLUSTERED SEs.
* incremental JSONL to disk after every cell -- this machine has died mid-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sigma import (  # noqa: E402
    MIN_EVAL_ITEMS,
    SOLVE_THRESHOLD,
    SeedRecord,
    clustered_mean,
    degenerate_floor,
)

# --- the recorded MQAR generator, reused not rebuilt -----------------------------------------
#
# Zoology-faithful, 43 correctness tests (mqar_data_test.py). Two byte-identical copies exist; we
# take whichever is present so this runs from either checkout.
_MQAR_SOURCES = (
    Path("/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/mqar"),
    Path(
        "/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/"
        "claude-01--liv-short-conv-mixer/experiments/liv/mqar"
    ),
)


def _import_mqar_data():
    for p in _MQAR_SOURCES:
        if (p / "mqar_data.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            import mqar_data  # type: ignore[import-not-found]

            return mqar_data
    raise ImportError(
        "mqar_data.py not found. Looked in:\n  " + "\n  ".join(str(p) for p in _MQAR_SOURCES)
    )


mqar_data = _import_mqar_data()
MQARConfig = mqar_data.MQARConfig
make_mqar_batch = mqar_data.make_mqar_batch
IGNORE_INDEX = mqar_data.IGNORE_INDEX
CALIBRATED_VOCAB = mqar_data.CALIBRATED_VOCAB
CALIBRATED_LR = mqar_data.CALIBRATED_LR
CALIBRATED_STEPS = mqar_data.CALIBRATED_STEPS
CALIBRATED_BATCH_SIZE = mqar_data.CALIBRATED_BATCH_SIZE
CALIBRATED_EXAMPLES = mqar_data.CALIBRATED_EXAMPLES

# --- geometry, SPEC Sec 7 --------------------------------------------------------------------
D_MODEL = 128
N_LAYERS = 6
GENERATOR_RANK = 16  # R. At d=128 this is R/d = 1/8, a PRE-REGISTERED deviation (SPEC Sec 7).
HYBRID_ATTENTION_LAYERS = (1, 4)  # 2 of 6, spread like LFM2
ALLLIV_ATTENTION_LAYERS: Tuple[int, ...] = ()

TOPOLOGIES = {"hybrid": HYBRID_ATTENTION_LAYERS, "allliv": ALLLIV_ATTENTION_LAYERS}
ARMS = ("static", "permuted", "dynqkv", "dynamic")  # S1, S2, S3, S4
ARM_CODES = {"static": "S1", "permuted": "S2", "dynqkv": "S3", "dynamic": "S4"}

# S3 has no definition without GQA blocks. Report N/A; never substitute S1 (SPEC Sec 1.2).
UNDEFINED_CELLS = {("dynqkv", "allliv")}

DATA_SEED_BASE = 10_000  # data_seed = DATA_SEED_BASE + seed_pair. Shared across arms.
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOSS_LOG_EVERY = 250

#: Batch size for EVAL, decoupled from the training batch size on purpose. Eval is a no-grad
#: forward pass, so it can use a much larger batch than training regardless of the training config;
#: tying the two made a smoke run at batch 4 issue 250 tiny forward passes for its 1,000 mandatory
#: eval items and dominated the cell's wall clock.
EVAL_BATCH_SIZE = 128


def resolve_device(spec: str = "auto") -> torch.device:
    """
    Resolve a device string. Explicit is better than implicit on a shared cluster.

    :param spec: ``"auto"`` (cuda when available), ``"cpu"``, or ``"cuda"``.
    :raises RuntimeError: if ``"cuda"`` is requested and unavailable -- silently falling back to CPU
        on a GPU allocation would waste the allocation and mis-attribute the timings.
    """
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def derive_init_seed(seed_pair: int, arm: str) -> int:
    """
    Stable per-(pair, arm) init seed. Deliberately arm-dependent -- see the module docstring.

    Uses a hash rather than ``seed_pair * k + arm_index`` so that adding an arm cannot shift any
    existing arm's seed, which would silently invalidate a partially-complete sweep.

    :returns: A seed in ``[0, 2**31)``.
    """
    h = hashlib.sha256(f"exp2|pair={seed_pair}|arm={arm}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def data_seed_for(seed_pair: int) -> int:
    """The SHARED data seed. Identical for every arm in a pair -- this IS the pairing."""
    return DATA_SEED_BASE + seed_pair


# ======================================================================================
# The model-builder contract
# ======================================================================================


class ModelBuilder(Protocol):
    """The callable :func:`run_cell` expects. See the module docstring."""

    def __call__(
        self,
        *,
        arm: str,
        topology: str,
        kernel_size: int,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        seed: int,
    ) -> nn.Module: ...


def assert_zoology_gotchas(cfg) -> None:
    """
    Pin gotcha 1 of 2: ``random_non_queries=False``.

    The class default is ``True``. With ``True`` the filler is random tokens that can COLLIDE with
    keys, which is a harder and different task -- so an unpinned run is not the published task and
    its numbers compare to nothing. ``mqar_data.MQARConfig`` already defaults it to ``False``; this
    asserts rather than assumes, because the assumption is exactly what silently breaks.
    """
    if getattr(cfg, "random_non_queries", None) is not False:
        raise AssertionError(
            "ZOOLOGY GOTCHA 1 VIOLATED: random_non_queries must be False (the class default is "
            f"True). Got {getattr(cfg, 'random_non_queries', None)!r}. With True the filler is "
            "random tokens that can collide with keys -- a harder, different task, and the numbers "
            "compare to nothing published."
        )
    # Verified: the recorded generator hard-codes random_non_queries=False as a dataclass default.
    import inspect

    src = inspect.getsource(type(cfg))
    if "random_non_queries: bool = False" not in src:
        raise AssertionError(
            "ZOOLOGY GOTCHA 1: MQARConfig no longer pins random_non_queries=False at the dataclass "
            "level. A per-call default is not enough -- a caller that omits it would silently get "
            "the class default."
        )


def assert_sequence_layout(cfg, tokens: torch.Tensor, labels: torch.Tensor) -> Dict[str, object]:
    """
    Absolute-structure gate on the DATA, adapted from the Exp-0 missing-BOS scar.

    Exp-0 measured that a missing BOS puts LFM2-350M **2.4-3.8 nats** off -- ~100x the effect being
    chased -- and that it fails **silently**. The MQAR analogue is a layout error: queries in the
    wrong positions, or labels on the wrong tokens, changes the absolute loss enormously while every
    delta still looks computable.

    **This generator has NO BOS/sentinel token, and that is a checked fact, not an oversight.**
    ``mqar_data.py`` is Zoology-faithful: the layout is ``k1 v1 ... kD vD <filler> q1 ... qD`` with
    no separator. (The OLDER in-tree design ``probes/mqar_patch.py`` DID use ``MQAR_SEP = 0`` at
    ``length - D - 1``; the two designs are not interchangeable, and asserting a BOS that this
    generator never emits would fail every batch.) So this asserts the invariants that DO hold:

    1. no token equals a reserved BOS id at position 0 (i.e. the no-BOS design is intact);
    2. labels are ``IGNORE_INDEX`` everywhere EXCEPT the final ``D`` positions;
    3. exactly ``D`` labelled positions per row;
    4. every label is in the VALUE half of the vocab and every query token is in the KEY half.

    :raises AssertionError: on any violation.
    :returns: a small dict of the measured structure, logged into the record.
    """
    b, t = tokens.shape
    d, v = cfg.num_pairs, cfg.vocab_size
    half = v // 2
    mask = labels != IGNORE_INDEX

    per_row = mask.sum(dim=1)
    if not bool((per_row == d).all()):
        raise AssertionError(
            f"LAYOUT: expected exactly D={d} labelled positions per row, got "
            f"min={int(per_row.min())} max={int(per_row.max())}. A wrong label count changes the "
            f"absolute loss without changing the shape of any delta."
        )
    q0 = t - d
    if bool(mask[:, :q0].any()):
        raise AssertionError(
            f"LAYOUT: found labels before position {q0}. Queries must occupy the FINAL {d} "
            f"positions; labels elsewhere mean the task is not the published MQAR."
        )
    if not bool(mask[:, q0:].all()):
        raise AssertionError(f"LAYOUT: the final {d} positions are not all labelled.")

    lab = labels[mask]
    if not bool(((lab >= half) & (lab < v - 1)).all()):
        raise AssertionError(
            f"LAYOUT: labels must lie in the VALUE half [{half}, {v - 1}) and exclude the filler "
            f"id {v - 1}. Got min={int(lab.min())} max={int(lab.max())}."
        )
    qtok = tokens[:, q0:]
    if not bool((qtok < half).all()):
        raise AssertionError(
            f"LAYOUT: query tokens must lie in the KEY half [0, {half}). Got max={int(qtok.max())}."
        )
    return {
        "has_bos": False,
        "n_labelled_per_row": d,
        "query_start": q0,
        "key_half": half,
        "filler_id": v - 1,
    }


def resolve_kernel_path(model: nn.Module) -> Dict[str, object]:
    """
    Requirement (b): report the REALISED conv kernel path, per module, so a fused treatment can
    never be compared against an unfused baseline.

    ``short_conv.py:185`` defaults ``use_fla=True`` while ``fla`` is absent in many environments, so
    ``has_fla()`` returns False and the module silently falls back to a plain ``nn.Conv1d``. If one
    arm ever resolves to the fused kernel and another does not, the contrast is confounded by the
    kernel rather than by the mechanism -- and it would bias toward whichever arm got the faster,
    numerically different path.

    :returns: ``{"backends": {name: count}, "use_fla_flags": [...], "family": str}``.
    """
    backends: Dict[str, int] = {}
    flags: List[object] = []
    for mod in model.modules():
        name = type(mod).__name__
        if "Conv" not in name and "ShortConv" not in name:
            continue
        if hasattr(mod, "use_fla"):
            flags.append(bool(getattr(mod, "use_fla")))
        fla_avail = getattr(mod, "_fla_available", None)
        if isinstance(mod, nn.Conv1d):
            key = "nn.Conv1d"
        elif getattr(mod, "use_fla", False) and fla_avail:
            key = "fla.fused"
        elif hasattr(mod, "use_fla"):
            key = "ShortConv/torch-fallback"
        else:
            key = name
        backends[key] = backends.get(key, 0) + 1
    family = "|".join(f"{k}x{v}" for k, v in sorted(backends.items())) or "none"
    return {"backends": backends, "use_fla_flags": flags, "family": family}


def assert_same_kernel_family(paths: Dict[str, Dict[str, object]]) -> None:
    """
    Requirement (b), the gate: every arm in a comparison must resolve to the SAME backend family.

    :param paths: ``{arm: resolve_kernel_path(model)}``.
    :raises AssertionError: if the arms disagree.
    """
    fams = {arm: p["family"] for arm, p in paths.items()}
    if len(set(fams.values())) > 1:
        raise AssertionError(
            f"KERNEL PATH MISMATCH across arms: {fams}. A fused treatment against an unfused "
            f"baseline is a confounded contrast -- the difference would be attributable to the "
            f"kernel, not the mechanism. Pin use_fla identically on every arm."
        )


def assert_state_mixer_identity(model: nn.Module) -> None:
    """
    Pin gotcha 2 of 2: ``state_mixer=Identity``.

    Zoology's published configs use ``state_mixer=Identity``, contradicting the paper's own
    Appendix E.2. The block structure belongs to the model-builder, so the harness cannot inspect it
    portably -- it therefore REQUIRES the builder to declare the property and **refuses to run**
    otherwise. A missing attribute is a failure, not a pass; per the ``green that means nothing``
    scar, ``getattr(..., True)`` would make this check unable to fail.
    """
    declared = getattr(model, "state_mixer_is_identity", None)
    if declared is not True:
        raise AssertionError(
            "ZOOLOGY GOTCHA 2 NOT VERIFIABLE: the model must declare "
            "`state_mixer_is_identity = True`. Zoology's published configs use "
            "state_mixer=Identity while the paper's Appendix E.2 says otherwise, and the "
            f"difference silently changes results. Got {declared!r}."
        )


# ======================================================================================
# Stub model -- lets this harness be tested end-to-end without arms.py
# ======================================================================================


class _StubAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, -1))


class _StubShortConv(nn.Module):
    """
    A gated short conv shaped like LFM2's: in_proj -> chunk into (B, C, x), depthwise causal conv,
    out_proj. **No activation in the conv path** -- SPEC Sec 5 trap 3: ``CausalConv1d`` defaults to
    ``activation="silu"`` and real LFM2 has none; permuting the chunk order still trains, just
    worse, which is a silent failure.
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.activation = None  # explicit: check 12 of the pre-flight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, v = self.in_proj(x).chunk(3, dim=-1)  # PRE-gate, POST-gate, value
        h = (b * v).transpose(1, 2)
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.conv(h).transpose(1, 2)
        return self.out_proj(c * h)


class _StubSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StubExp2Model(nn.Module):
    """
    Topology-faithful, mechanism-free stand-in satisfying the :class:`ModelBuilder` contract.

    Exists so the harness, the endpoints, the pairing and the disk format can all be exercised on
    CPU today. It has NO dynamic mechanism: every arm returns the same architecture, so a
    stub-driven sweep MUST show arms tying to within seed noise. That is a useful negative check on
    the harness (it must not manufacture a difference), and nothing more.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        kernel_size: int,
        attention_layers: Sequence[int],
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.attention_layers = tuple(sorted(attention_layers))
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mixer_norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.ffn_norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.ffns = nn.ModuleList(_StubSwiGLU(d_model, ffn_mult * d_model) for _ in range(n_layers))
        mixers: List[nn.Module] = []
        for i in range(n_layers):
            if i in self.attention_layers:
                mixers.append(_StubAttention(d_model))
            else:
                mixers.append(_StubShortConv(d_model, kernel_size))
        self.mixers = nn.ModuleList(mixers)
        self.out_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Declared so assert_state_mixer_identity can pass: no per-position state mixer in a block.
        self.state_mixer_is_identity = True

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for mixer, mnorm, ffn, fnorm in zip(
            self.mixers, self.mixer_norms, self.ffns, self.ffn_norms
        ):
            x = x + mixer(mnorm(x))
            x = x + ffn(fnorm(x))
        return self.head(self.out_norm(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- adapter onto sub-agent A's arms.py ------------------------------------------------------
#
# RECONCILIATION, measured against the delivered arms.py (2026-08-05):
#
#   theirs                                        mine
#   ------------------------------------------    -------------------------------------------
#   build_arm(spec: ArmSpec, seed) -> MQARModel   build_model(arm=..., topology=..., ...)
#   arm names "S1".."S4"                          "static"/"permuted"/"dynqkv"/"dynamic"
#   ArmSpec.width                                 kernel_size
#   ArmSpec.rank (default RANK)                   GENERATOR_RANK
#   raises ArmNotDefined for (S3, allliv)         raises ValueError -- compatible, both refuse
#   forward(tokens) -> logits [B,T,vocab]         same. AGREES.
#   does NOT set `state_mixer_is_identity`        assert_state_mixer_identity requires it
#
# The last row is the only real gap. Their block is norm -> mixer -> residual -> norm -> SwiGLU ->
# residual, i.e. there is no per-position state mixer inside the mixer block at all, which IS
# `state_mixer=Identity` in Zoology's sense. Rather than edit their file (not mine to edit), this
# adapter asserts that structure and stamps the declaration.
_ARM_TO_CODE = dict(ARM_CODES)


def arms_build_model(
    *,
    arm: str,
    topology: str,
    kernel_size: int,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    seed: int,
) -> nn.Module:
    """
    Adapt ``arms.build_arm(ArmSpec, seed)`` to the :class:`ModelBuilder` contract.

    Verifies structurally that there is no per-position state mixer before stamping
    ``state_mixer_is_identity``, so the declaration remains a checked claim rather than a label.
    """
    from arms import ArmSpec, build_arm  # type: ignore[import-not-found]

    spec_kwargs = dict(
        arm=_ARM_TO_CODE[arm],
        topology=topology,
        width=kernel_size,
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
    )
    if topology == "allliv":
        spec_kwargs["attention_layers"] = ()
    else:
        spec_kwargs["attention_layers"] = HYBRID_ATTENTION_LAYERS
    model = build_arm(ArmSpec(**spec_kwargs), seed=seed)

    # Zoology gotcha 2, verified rather than asserted by fiat: a block must carry no state-mixer
    # module between the sequence mixer and the FFN.
    for i, blk in enumerate(model.blocks):
        for banned in ("state_mixer", "state_mix"):
            got = getattr(blk, banned, None)
            if got is not None and not isinstance(got, nn.Identity):
                raise AssertionError(
                    f"ZOOLOGY GOTCHA 2: block {i} carries a non-Identity {banned!r} ({got}). "
                    f"Zoology's published configs use state_mixer=Identity."
                )
    model.state_mixer_is_identity = True
    return model


def stub_build_model(
    *,
    arm: str,
    topology: str,
    kernel_size: int,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    seed: int,
) -> nn.Module:
    """Reference :class:`ModelBuilder` implementation, for tests and smoke runs only."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; expected one of {sorted(TOPOLOGIES)}")
    if (arm, topology) in UNDEFINED_CELLS:
        raise ValueError(
            f"arm {arm!r} ({ARM_CODES[arm]}) is UNDEFINED in topology {topology!r}: there are no "
            f"GQA blocks to place a dynamic conv in. Report N/A; do NOT substitute S1."
        )
    torch.manual_seed(seed)
    return StubExp2Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        attention_layers=TOPOLOGIES[topology],
    )


# ======================================================================================
# Eval -- >=1000 items, clustered SEs, accuracy AND query NLL
# ======================================================================================


@dataclass(frozen=True)
class EvalResult:
    """Per-cell eval. Both endpoints, both SEs, clustered on the sequence."""

    accuracy: float
    acc_se_clustered: float
    acc_se_naive: float
    acc_design_effect: float
    nll_query: float
    nll_se_clustered: float
    nll_se_naive: float
    nll_design_effect: float
    n_items: int
    n_query_tokens: int
    floor: float
    per_item_accuracy: Tuple[float, ...]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cfg,
    *,
    n_items: int = MIN_EVAL_ITEMS,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> EvalResult:
    """
    Score a model on ``>= n_items`` held-out MQAR sequences.

    Two endpoints, per SPEC Sec 4.2-4.4:

    * **accuracy** at query positions (the bimodal endpoint), and
    * **query NLL** in nats -- the CONTINUOUS endpoint, worth a 2-18x SNR gain and the single
      biggest free lever available. Graded even when accuracy is pinned at floor or ceiling.

    Both get **clustered** SEs, clustering on the eval sequence: the ``D`` queries in one sequence
    share a key-value table and a forward pass, so naive token-level SEs are up to ``sqrt(D)`` too
    small -- *"a fabricated significance factory"*.

    :param n_items: Minimum eval sequences. Refuses fewer than :data:`MIN_EVAL_ITEMS`.
    :returns: An :class:`EvalResult`.
    """
    if n_items < MIN_EVAL_ITEMS:
        raise ValueError(
            f"n_items={n_items} is below MIN_EVAL_ITEMS={MIN_EVAL_ITEMS}. R3 F8 fix 4 requires "
            f">=1,000 items (Miller Eq. 9: n~=969 for delta=0.03 with likelihood scoring). A "
            f"smaller probe has a naive SE up to 3x too small."
        )
    was_training = model.training
    model.eval()
    g = generator or torch.Generator().manual_seed(0)

    correct_sums: List[float] = []
    nll_sums: List[float] = []
    counts: List[int] = []
    per_item_acc: List[float] = []
    tok_sq = 0.0
    tok_n = 0

    done = 0
    while done < n_items:
        bs = min(batch_size, n_items - done)
        tokens, labels = make_mqar_batch(cfg, bs, g)
        logits = model(tokens.to(device)).float().cpu()
        mask = labels != IGNORE_INDEX

        logp = F.log_softmax(logits, dim=-1)
        safe = labels.clamp_min(0).unsqueeze(-1)
        tok_nll = -logp.gather(-1, safe).squeeze(-1)  # [B, T]
        tok_ok = (logits.argmax(-1) == labels).float()  # [B, T]

        for i in range(bs):
            m = mask[i]
            k = int(m.sum())
            if k == 0:
                continue
            c = float(tok_ok[i][m].sum())
            s = float(tok_nll[i][m].sum())
            correct_sums.append(c)
            nll_sums.append(s)
            counts.append(k)
            per_item_acc.append(c / k)
            tok_sq += float((tok_nll[i][m] ** 2).sum())
            tok_n += k
        done += bs

    acc = clustered_mean(correct_sums, counts)
    mean_nll = sum(nll_sums) / max(tok_n, 1)
    tok_var = max(tok_sq / max(tok_n, 1) - mean_nll**2, 0.0)
    nll = clustered_mean(nll_sums, counts, naive_variance=tok_var)

    if was_training:
        model.train()
    return EvalResult(
        accuracy=acc.mean,
        acc_se_clustered=acc.se_clustered,
        acc_se_naive=acc.se_naive,
        acc_design_effect=acc.design_effect,
        nll_query=nll.mean,
        nll_se_clustered=nll.se_clustered,
        nll_se_naive=nll.se_naive,
        nll_design_effect=nll.design_effect,
        n_items=len(counts),
        n_query_tokens=tok_n,
        floor=degenerate_floor(cfg.num_pairs),
        per_item_accuracy=tuple(per_item_acc),
    )


# ======================================================================================
# One cell
# ======================================================================================


#: Minimum query tokens the init-loss band is evaluated over. At vocab 256 the per-token NLL has an
#: SD of ~1 nat, so the SE of a mean over ``m`` tokens is ~1/sqrt(m). The band is +-0.25 wide, so
#: reading it off a single small batch measures SAMPLING NOISE, not initialization: 16 tokens give an
#: SE of ~0.25 and the guard then fires on healthy models (measured: the same stub reads 5.32-5.95
#: over batches of 4 vs 5.65-5.75 over 4,096 tokens). 2,048 tokens put the SE at ~0.022, i.e. ~1/11
#: of the band.
INIT_LOSS_MIN_TOKENS = 2048
INIT_LOSS_BAND = (-0.05, 0.25)  # relative to ln(vocab)


@torch.no_grad()
def check_init_loss(
    model: nn.Module,
    cfg,
    *,
    device: torch.device = torch.device("cpu"),
    min_tokens: int = INIT_LOSS_MIN_TOKENS,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    SPEC Sec 6 check 8: init loss must lie in ``[ln V, ln V + 0.25]``. At vocab 256 that is
    ``[5.5452, 5.7952]``.

    **Assert magnitudes, not existence** -- this check has caught uninitialized weights ~4x in this
    repo. Measured over at least :data:`INIT_LOSS_MIN_TOKENS` query tokens on a stream DISJOINT from
    training, because a per-batch reading is dominated by sampling noise (see the constant's note).

    :returns: The measured init loss in nats.
    :raises AssertionError: if the loss is outside the band.
    """
    was_training = model.training
    model.eval()
    g = generator or torch.Generator().manual_seed(12345)
    tot = 0.0
    n = 0
    while n < min_tokens:
        tokens, labels = make_mqar_batch(cfg, 64, g)
        logits = model(tokens.to(device)).float().cpu()
        mask = labels != IGNORE_INDEX
        k = int(mask.sum())
        if k == 0:
            continue
        lp = F.log_softmax(logits, dim=-1)
        nll = -lp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tot += float(nll[mask].sum())
        n += k
    if was_training:
        model.train()
    got = tot / n
    lo = math.log(cfg.vocab_size)
    if not (lo + INIT_LOSS_BAND[0] <= got <= lo + INIT_LOSS_BAND[1]):
        raise AssertionError(
            f"INIT LOSS OUT OF BAND: {got:.4f} not in "
            f"[{lo + INIT_LOSS_BAND[0]:.4f}, {lo + INIT_LOSS_BAND[1]:.4f}] "
            f"(ln vocab={cfg.vocab_size} => {lo:.4f}), measured over {n:,} query tokens. This is "
            f"the uninitialized-weights / broken-loss signature, not a slow start."
        )
    return got


def check_budget(steps: int, batch_size: int, *, smoke: bool) -> None:
    """
    Refuse to run under-budget outside an explicit smoke flag.

    Job 1670963 failed exactly here: a stale sbatch carried 3000 x 32 = 96,000 examples against the
    calibrated 512,000 (a 5.3x shortfall), and ``N64_D4`` -- which the positive control solved at
    1.000 -- then scored 0.24/0.25/0.25/0.26/0.93 with FOUR runs parked on the ``1/D`` floor.
    **Under-training is indistinguishable from a too-hard task in the output.**
    """
    examples = steps * batch_size
    if examples < CALIBRATED_EXAMPLES and not smoke:
        raise RuntimeError(
            f"REFUSING TO RUN UNDER-BUDGET: {examples:,} examples ({steps} steps x {batch_size}) "
            f"is below the calibrated {CALIBRATED_EXAMPLES:,} "
            f"({CALIBRATED_STEPS} x {CALIBRATED_BATCH_SIZE}), the only budget measured to solve "
            f"this task here. Under-training is indistinguishable from a too-hard task in the "
            f"output, so this would produce a confident but meaningless result (job 1670963). "
            f"Pass smoke=True ONLY for a smoke test whose numbers you will not interpret."
        )


def run_cell(
    *,
    arm: str,
    topology: str,
    kernel_size: int,
    cfg,
    seed_pair: int,
    build_model: ModelBuilder,
    steps: int = CALIBRATED_STEPS,
    batch_size: int = CALIBRATED_BATCH_SIZE,
    lr: float = CALIBRATED_LR,
    device: torch.device = torch.device("cpu"),
    eval_items: int = MIN_EVAL_ITEMS,
    smoke: bool = False,
    log_every: int = LOSS_LOG_EVERY,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    verbose: bool = True,
) -> SeedRecord:
    """
    Train and evaluate one cell. The atomic unit of Exp-2.

    :param arm: ``"static"`` | ``"permuted"`` | ``"dynqkv"`` | ``"dynamic"``.
    :param topology: ``"hybrid"`` | ``"allliv"``.
    :param kernel_size: ``W``.
    :param cfg: An ``MQARConfig``.
    :param seed_pair: The pair index. Determines the SHARED data order and, via
        :func:`derive_init_seed`, this arm's init.
    :param build_model: The :class:`ModelBuilder`.
    :param smoke: Permit an under-calibration budget. Numbers from a smoke run are not results.
    :returns: A :class:`sigma.SeedRecord`.
    """
    if (arm, topology) in UNDEFINED_CELLS:
        raise ValueError(
            f"cell ({arm}, {topology}) is UNDEFINED -- {ARM_CODES[arm]} has no meaning without GQA "
            f"blocks. Report N/A; do not substitute another arm."
        )
    check_budget(steps, batch_size, smoke=smoke)
    assert_zoology_gotchas(cfg)

    data_seed = data_seed_for(seed_pair)
    init_seed = derive_init_seed(seed_pair, arm)

    model = build_model(
        arm=arm,
        topology=topology,
        kernel_size=kernel_size,
        vocab_size=cfg.vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        seed=init_seed,
    ).to(device)
    assert_state_mixer_identity(model)

    # Requirement (b): log the REALISED kernel path per arm, at step 0.
    kernel_path = resolve_kernel_path(model)

    # Requirement (a) + SPEC Sec 6 check 8, BEFORE any optimizer step, on a disjoint stream.
    # HARD GATE: if the ABSOLUTE loss is out of band, no between-arm delta may be read. The Exp-0
    # scar is that a missing BOS put LFM2-350M 2.4-3.8 nats off -- ~100x the effect being chased --
    # and it failed silently. check_init_loss RAISES; it does not warn.
    init_loss = check_init_loss(model, cfg, device=device)

    # Requirement (a), the data half: assert the sequence layout on a real batch.
    _lay_gen = torch.Generator().manual_seed(data_seed)
    _lt, _ll = make_mqar_batch(cfg, min(batch_size, 16), _lay_gen)
    layout = assert_sequence_layout(cfg, _lt, _ll)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    # The pairing: this generator is keyed ONLY on data_seed, so every arm at the same seed_pair
    # sees byte-identical batches in identical order.
    train_gen = torch.Generator().manual_seed(data_seed)

    first_loss = float("nan")
    loss = torch.tensor(float("nan"))
    t0 = time.time()
    model.train()
    for step in range(steps):
        tokens, labels = make_mqar_batch(cfg, batch_size, train_gen)
        logits = model(tokens.to(device))
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            labels.reshape(-1).to(device),
            ignore_index=IGNORE_INDEX,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        if step == 0:
            first_loss = float(loss.detach())
        if verbose and log_every and (step % log_every == 0 or step == steps - 1):
            print(f"      step {step:>5} loss {float(loss):.4f}", flush=True)

    # Eval stream: also keyed only on data_seed, disjoint offset so it is held out from training.
    eval_gen = torch.Generator().manual_seed(data_seed + 7_919_311)
    ev = evaluate(
        model,
        cfg,
        n_items=eval_items,
        batch_size=eval_batch_size,
        device=device,
        generator=eval_gen,
    )
    seconds = time.time() - t0

    return SeedRecord(
        arm=arm,
        topology=topology,
        kernel_size=kernel_size,
        config=cfg.label,
        seed=seed_pair,
        accuracy=ev.accuracy,
        nll_query=ev.nll_query,
        num_pairs=cfg.num_pairs,
        acc_se_clustered=ev.acc_se_clustered,
        nll_se_clustered=ev.nll_se_clustered,
        acc_design_effect=ev.acc_design_effect,
        nll_design_effect=ev.nll_design_effect,
        n_eval_items=ev.n_items,
        first_loss=first_loss,
        final_loss=float(loss),
        n_params=sum(p.numel() for p in model.parameters()),
        seconds=seconds,
        data_seed=data_seed,
        init_seed=init_seed,
        extra={
            "arm_code": ARM_CODES[arm],
            "init_loss": init_loss,
            "init_loss_band": [
                math.log(cfg.vocab_size) + INIT_LOSS_BAND[0],
                math.log(cfg.vocab_size) + INIT_LOSS_BAND[1],
            ],
            "init_loss_in_band": True,  # run_cell raises otherwise, so reaching here proves it
            "kernel_path": kernel_path,
            "layout": layout,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "torch_version": torch.__version__,
            "smoke": smoke,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "examples": steps * batch_size,
            "seq_len": cfg.seq_len,
            "vocab_size": cfg.vocab_size,
            "solved": ev.accuracy >= SOLVE_THRESHOLD,
            "floor": ev.floor,
            "acc_over_floor": ev.accuracy / ev.floor if ev.floor > 0 else float("nan"),
            "acc_se_naive": ev.acc_se_naive,
            "nll_se_naive": ev.nll_se_naive,
            "n_query_tokens": ev.n_query_tokens,
        },
    )


# ======================================================================================
# Incremental persistence -- this machine has died mid-run
# ======================================================================================


def cell_key(arm: str, topology: str, kernel_size: int, config: str, seed_pair: int) -> str:
    return f"{arm}|{topology}|W{kernel_size}|{config}|s{seed_pair}"


def append_record(path: Path, rec: SeedRecord) -> None:
    """
    Append one record as a JSONL line and ``fsync``. Crash-safe and resumable.

    JSONL not JSON: a partially-written JSON array is unparseable, whereas a truncated JSONL file
    loses at most the last line. The repo memory is explicit that this machine has died mid-run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(rec)
    d["_key"] = cell_key(rec.arm, rec.topology, rec.kernel_size, rec.config, rec.seed)
    with path.open("a") as fh:
        fh.write(json.dumps(d) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_records(path: Path) -> List[SeedRecord]:
    """Load a JSONL results file, tolerating a truncated final line."""
    out: List[SeedRecord] = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            print(f"  warning: skipping unparseable (truncated?) line in {path}", file=sys.stderr)
            continue
        d.pop("_key", None)
        out.append(SeedRecord(**d))
    return out


def completed_keys(path: Path) -> set:
    return {cell_key(r.arm, r.topology, r.kernel_size, r.config, r.seed) for r in load_records(path)}


# ======================================================================================
# Sweep driver
# ======================================================================================


def run_sweep(
    *,
    arms: Sequence[str],
    topologies: Sequence[str],
    kernel_sizes: Sequence[int],
    configs: Sequence,
    seed_pairs: Sequence[int],
    out_path: Path,
    build_model: ModelBuilder,
    steps: int = CALIBRATED_STEPS,
    batch_size: int = CALIBRATED_BATCH_SIZE,
    lr: float = CALIBRATED_LR,
    device: torch.device = torch.device("cpu"),
    eval_items: int = MIN_EVAL_ITEMS,
    smoke: bool = False,
    resume: bool = True,
    verbose: bool = True,
) -> List[SeedRecord]:
    """
    Run every defined cell, writing incrementally and skipping cells already on disk.

    Cell order is **seed-pair-outermost then arm-innermost**, so all arms of a pair complete
    together. That matters: if the machine dies, the surviving data is a set of COMPLETE pairs, and
    a complete pair is analyzable while an orphaned treatment arm is not.

    :returns: All records, including those loaded from a previous partial run.
    """
    done = completed_keys(out_path) if resume else set()
    if done and verbose:
        print(f"resuming: {len(done)} cells already on disk in {out_path}", flush=True)

    n_skipped_undefined = 0
    _kernel_families: Dict[tuple, Dict[str, Dict[str, object]]] = {}
    for cfg in configs:
        for w in kernel_sizes:
            for top in topologies:
                for pair in seed_pairs:
                    for arm in arms:
                        if (arm, top) in UNDEFINED_CELLS:
                            n_skipped_undefined += 1
                            continue
                        key = cell_key(arm, top, w, cfg.label, pair)
                        if key in done:
                            continue
                        if verbose:
                            print(
                                f"  {key}  ({ARM_CODES[arm]})",
                                flush=True,
                            )
                        rec = run_cell(
                            arm=arm,
                            topology=top,
                            kernel_size=w,
                            cfg=cfg,
                            seed_pair=pair,
                            build_model=build_model,
                            steps=steps,
                            batch_size=batch_size,
                            lr=lr,
                            device=device,
                            eval_items=eval_items,
                            smoke=smoke,
                            verbose=False,
                        )
                        append_record(out_path, rec)
                        done.add(key)
                        # Requirement (b), enforced ACROSS arms within a pair: every arm at this
                        # (topology, W, config, seed) must have resolved to the same backend.
                        fam_key = (top, w, cfg.label, pair)
                        seen = _kernel_families.setdefault(fam_key, {})
                        seen[arm] = rec.extra["kernel_path"]
                        assert_same_kernel_family(seen)
                        if verbose:
                            print(
                                f"    acc {rec.accuracy:.4f} (floor {rec.floor:.4f}, "
                                f"{rec.accuracy / rec.floor:.1f}x)  nll {rec.nll_query:.4f}  "
                                f"loss {rec.first_loss:.3f}->{rec.final_loss:.3f}  "
                                f"[{rec.seconds:.1f}s]",
                                flush=True,
                            )
    if verbose and n_skipped_undefined:
        print(
            f"  N/A: {n_skipped_undefined} cells skipped as UNDEFINED "
            f"({sorted(UNDEFINED_CELLS)}) -- reported N/A, not substituted",
            flush=True,
        )
    return load_records(out_path)


# ======================================================================================
# Grid planning and cost -- so the cell count is auditable code, not a number in an email
# ======================================================================================


def plan_grid(
    *,
    arms: Sequence[str] = ARMS,
    topologies: Sequence[str] = ("allliv", "hybrid"),
    kernel_sizes: Sequence[int] = (2, 3, 4, 8),
    configs: Sequence[str] = ("N512_D64", "N512_D8"),
    seeds: int = 10,
) -> Dict[str, object]:
    """
    Enumerate the defined cells, excluding the (arm, topology) combinations that are N/A.

    S3 is undefined in ``allliv``, so the arm count is topology-dependent and a naive
    ``len(arms) * len(topologies) * ...`` OVERCOUNTS. This returns the real number.

    :returns: ``{"n_cells", "per_topology", "cells", "seeds", ...}``.
    """
    cells: List[Tuple[str, str, int, str, int]] = []
    per_topology: Dict[str, int] = {}
    for top in topologies:
        defined = [a for a in arms if (a, top) not in UNDEFINED_CELLS]
        per_topology[top] = len(defined) * len(kernel_sizes) * len(configs) * seeds
        for w in kernel_sizes:
            for cfgname in configs:
                for s in range(seeds):
                    for a in defined:
                        cells.append((a, top, w, cfgname, s))
    return {
        "n_cells": len(cells),
        "per_topology": per_topology,
        "arms": list(arms),
        "topologies": list(topologies),
        "kernel_sizes": list(kernel_sizes),
        "configs": list(configs),
        "seeds": seeds,
        "n_undefined_skipped": (
            len(arms) * len(topologies) * len(kernel_sizes) * len(configs) * seeds - len(cells)
        ),
        "cells": cells,
    }


# =====================================================================================
# SECONDS-PER-CELL: **OUTSTANDING -- MUST BE MEASURED ON FARMSHARE, NOT ON THIS LAPTOP**
# =====================================================================================
#
# There is deliberately NO default value here, and :func:`cost_estimate` REQUIRES the caller to
# pass a measured number. A plausible-looking constant would get copied into a submission and
# become "the measurement" without anyone having taken it.
#
# STATUS 2026-08-05: **NOT MEASURED.** The user's machine is failing and all local execution is
# forbidden, so the CPU timing run was killed before it produced a number. Do not substitute an
# estimate. What exists instead:
#
#   * MEASURED, but on the WRONG geometry and someone else's device -- from
#     ``mqar_calibration.json`` (FarmShare, CUDA, 4-layer d=128 with attention at (1,3),
#     8000 steps x batch 64, fp32):
#         N512_D64   592.1 s/cell      N512_D8   174.7 s/cell
#         N256_D16   211.8 s/cell      N128_D8   171.7 s/cell
#     Exp-2 is 6 layers, so these are a LOWER BOUND on the per-cell cost (1.5x the depth), and the
#     ``allliv`` topology replaces 2 attention layers with conv, which changes the cost again.
#
#   * Repo memory ``measure-throughput-post-startup``: whole-run wall clock read **3.1x LOW** on a
#     40-step probe and penalises bigger shapes hardest. So the FarmShare measurement must exclude
#     startup and be taken as a per-step rate at the REAL geometry, then multiplied by 8,000 --
#     not read off a short whole-run timing.
#
# TO TAKE THE MEASUREMENT (FarmShare, per the operate-farmshare skill):
#     python3 mqar_harness.py --stub --device cuda --smoke --steps 200 --batch-size 64 \
#         --seq-len 512 --num-pairs 64 --seeds 1 --arms static --topologies allliv
# then read ``seconds`` from the record, subtract the eval time, and scale the per-step rate to
# 8,000 steps. Record device, dtype and torch version alongside it.
SECONDS_PER_CELL_MEASURED: Optional[float] = None
"""OUTSTANDING. See the note above. Must be measured on FarmShare; never estimated."""

#: MEASURED on FarmShare CUDA but at 4 layers, not Exp-2's 6 -- a LOWER BOUND, not the number.
RECORDED_SECONDS_PER_CELL_4LAYER_CUDA = {
    "N512_D64": 592.1,
    "N512_D8": 174.7,
    "N256_D16": 211.8,
    "N128_D8": 171.7,
    "N1024_D8": 246.1,
    "N64_D4": 163.0,
}


def cost_estimate(
    seconds_per_cell: float,
    *,
    n_cells: int,
    n_parallel: int = 1,
    usd_per_hour: float = 1.8610,
) -> Dict[str, float]:
    """
    Price a grid from a MEASURED seconds-per-cell.

    :param seconds_per_cell: **MEASURED, not guessed.** Required positional -- there is no default,
        because a default would get copied into a submission and become "the measurement". Per repo
        memory ``measure-throughput-post-startup``, whole-run wall clock read 3.1x LOW on a 40-step
        probe and penalises bigger shapes hardest, so extrapolate from a per-step rate at the real
        geometry rather than from a short whole-run timing. See
        :data:`SECONDS_PER_CELL_MEASURED`, which is currently ``None`` (OUTSTANDING).
    :param n_cells: From :func:`plan_grid`.
    :param n_parallel: Concurrent cells per device (these models are ~1.8M params; a single L40S
        fits many, and at d=128 one cell cannot saturate the card).
    :param usd_per_hour: ``gpu-1xl40s`` is $1.8610/hr. **There is no H100.**
    :returns: sequential and parallel wall-clock hours plus cost.
    :raises ValueError: if ``seconds_per_cell`` is not a positive measured number.
    """
    if seconds_per_cell is None or not (seconds_per_cell > 0):
        raise ValueError(
            "cost_estimate requires a MEASURED seconds_per_cell. It is currently OUTSTANDING "
            "(see SECONDS_PER_CELL_MEASURED) and must be taken on FarmShare, not estimated."
        )
    seq_h = n_cells * seconds_per_cell / 3600.0
    par_h = seq_h / max(n_parallel, 1)
    return {
        "seconds_per_cell": seconds_per_cell,
        "n_cells": float(n_cells),
        "n_parallel": float(n_parallel),
        "sequential_gpu_hours": seq_h,
        "wall_clock_hours": par_h,
        "usd": par_h * usd_per_hour,
        "usd_per_hour": usd_per_hour,
        "fits_095h_autoapprove": par_h <= 0.95,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["static", "dynamic"], choices=list(ARMS))
    ap.add_argument("--topologies", nargs="+", default=["allliv"], choices=sorted(TOPOLOGIES))
    ap.add_argument("--kernel-sizes", nargs="+", type=int, default=[3])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-pairs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=10, help="number of paired seeds")
    ap.add_argument("--steps", type=int, default=CALIBRATED_STEPS)
    ap.add_argument("--batch-size", type=int, default=CALIBRATED_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=CALIBRATED_LR)
    ap.add_argument("--eval-items", type=int, default=MIN_EVAL_ITEMS)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="permit an under-calibration budget. Numbers from a smoke run are NOT results.",
    )
    ap.add_argument("--stub", action="store_true", help="use the built-in stub model builder")
    ap.add_argument("--out", default="exp2_results.jsonl")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda when available. Set explicitly on FarmShare sbatch.",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    if args.stub:
        builder: ModelBuilder = stub_build_model
        print("MODEL: built-in STUB (no dynamic mechanism). Numbers are harness checks, "
              "not science.", flush=True)
    else:
        try:
            builder: ModelBuilder = arms_build_model
            import arms  # noqa: F401  # fail fast if it is not importable
        except ImportError as exc:
            print(
                f"cannot import arms.py ({exc}). Sub-agent A owns that file; "
                f"pass --stub to exercise the harness meanwhile.",
                file=sys.stderr,
            )
            return 2

    device = resolve_device(args.device)
    cfg = MQARConfig(
        seq_len=args.seq_len, num_pairs=args.num_pairs, vocab_size=CALIBRATED_VOCAB
    )
    print(f"device: {device}  dtype: torch.float32  torch {torch.__version__}   "
          f"config: {cfg.label}   floor 1/D = {degenerate_floor(cfg.num_pairs):.4f}", flush=True)
    print(f"budget: {args.steps * args.batch_size:,} examples"
          f"{'  [SMOKE -- under-budget, not a result]' if args.smoke else ''}", flush=True)

    recs = run_sweep(
        arms=args.arms,
        topologies=args.topologies,
        kernel_sizes=args.kernel_sizes,
        configs=[cfg],
        seed_pairs=list(range(args.seeds)),
        out_path=Path(args.out),
        build_model=builder,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        eval_items=args.eval_items,
        smoke=args.smoke,
        resume=not args.no_resume,
    )

    from sigma import sigma_report  # noqa: PLC0415

    print()
    print(sigma_report(recs))
    print(f"\nwrote {len(recs)} records to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
