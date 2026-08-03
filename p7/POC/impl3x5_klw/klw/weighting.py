"""Impl 3's per-token loss weighting, reimplemented from IMPL3_HANDOFF §4.1.

James's ``common/weighting.py`` is **not** in the handoff bundle — it ships ``common/kl.py``,
``common/system_instructions.py`` and the eval tree, but neither ``weighting.py`` nor
``sft_train.py``. §4.1 specifies the objective completely, so this is a reimplementation
against the spec rather than a port, and every constant below is quoted from it.

For every loss-bearing **pedagogy** token *t* a scalar "distance from base" *s_t* is computed,
standardised once and globally with a robust z-score, then turned into a multiplier on that
token's cross-entropy::

    m_t = N_ped · softmax_ped(−z(s_t)/T)      for pedagogy tokens
    m_t = 1                                    for general (replay) tokens

Two signal variants:

===========  ==========================================  ==============================
variant      signal                                      cost
===========  ==========================================  ==============================
``a``        s_t = −log π₀(y_t | ctx)                     one frozen-base forward pass
``b``        s_t = KL(π₀(·|ctx_t) ‖ π_SFT(·|ctx_t))       needs a vanilla-SFT reference too
===========  ==========================================  ==============================

Four properties this module exists to guarantee, each of which is asserted in
``acceptance_checks_klw.py`` rather than trusted:

1. **mean-1 over pedagogy tokens.** ``N_ped ·`` is what makes this hold, and it is what
   preserves the pedagogy:general loss ratio and the effective learning rate. Without it a
   temperature sweep is also an LR sweep.
2. **T → ∞ recovers vanilla SFT exactly.** James verified this with ``b-T451`` reproducing
   his SFT baseline to within 0.002 on every metric and recommends running the equivalent;
   arm ``bT451`` is that control, and :func:`multipliers` makes the limit exact in the sense
   that ``m → 1`` uniformly.
3. **The softmax is global**, over every pedagogy token in the dataset — not per row and not
   per batch. So the normalisation depends on the whole corpus, which is why the cache is
   keyed on the training file's content.
4. **Temperature is not in the cache key.** One precompute serves an entire temperature
   sweep; only :func:`multipliers` sees T.

What is *not* inherited from James, and matters when reading these runs: the standardisation
is global over **this** corpus. Impl 5's pedagogy targets are ~37% base-model text, where
both signals are systematically smaller, so the same numeric T does not produce the same
multiplier distribution it produced on gold. ``bT1`` here is not ``b-T1`` there. See
:func:`describe` — its ``ess`` field is the number to quote when comparing across corpora.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

#: Consistency constant making MAD a consistent estimator of σ for a normal distribution.
#: IMPL3_HANDOFF §4.1: "a robust z-score (median / MAD × 1.4826)".
MAD_TO_SIGMA = 1.4826

#: Multiplier for a general/replay token. §4.1: "general replay tokens always get 1.0".
GENERAL_MULTIPLIER = 1.0

VARIANTS = ("a", "b")


# --------------------------------------------------------------------------- #
# the two-step reduction: signal -> robust z -> multiplier
# --------------------------------------------------------------------------- #
def robust_stats(signal: np.ndarray) -> tuple[float, float]:
    """``(median, 1.4826 · MAD)`` over *all* pedagogy tokens, computed once and globally.

    Raises when MAD is 0 rather than dividing by it. That happens only if over half the
    pedagogy tokens share one signal value exactly, which for a continuous signal means the
    precompute produced a constant — a bug worth stopping on, not smoothing over.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("empty signal: nothing to standardise")
    if not np.isfinite(s).all():
        bad = int((~np.isfinite(s)).sum())
        raise ValueError(f"signal has {bad} non-finite values of {s.size}")
    median = float(np.median(s))
    mad = float(np.median(np.abs(s - median)))
    if mad <= 0.0:
        raise ValueError(
            f"MAD is 0 (median {median:.6g}) — over half the pedagogy tokens share one "
            f"signal value, so the signal is degenerate and z would be undefined"
        )
    return median, MAD_TO_SIGMA * mad


def robust_z(signal: np.ndarray, median: float, scale: float) -> np.ndarray:
    """``(s − median) / (1.4826 · MAD)``, in float64."""
    return (np.asarray(signal, dtype=np.float64) - median) / scale


def multipliers(z: np.ndarray, temperature: float) -> np.ndarray:
    """``N_ped · softmax(−z/T)`` over the whole pedagogy stream — i.e. mean exactly 1.

    ``N_ped · softmax(x)_t`` is algebraically ``exp(x_t) / mean(exp(x))``, which is what is
    computed: the max-shift that makes the exponential safe cancels between numerator and
    denominator, so the result is shift-invariant and needs no separate normalisation pass.

    Low T concentrates weight on tokens the base already finds *easy* (low surprise, low KL),
    which is the "stay close to base" pressure. It concentrates hard: at T=0.5 James's
    variant-a runs ended at NLL 2.743, *above* the base model's 1.416, because almost all the
    gradient landed on almost none of the tokens. :func:`describe` reports the effective
    sample size so that regime is visible before a GPU-hour is spent in it.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    a = -np.asarray(z, dtype=np.float64) / float(temperature)
    a -= a.max()                      # softmax is shift-invariant; this keeps exp() in range
    u = np.exp(a)
    mean_u = u.mean()
    if not np.isfinite(mean_u) or mean_u <= 0.0:
        raise ValueError(
            f"softmax underflowed at T={temperature}: the weight is concentrated on so few "
            f"tokens that the mean of exp(−z/T) is {mean_u!r}. Raise T."
        )
    return u / mean_u


def describe(m: np.ndarray) -> dict:
    """Diagnostics for a multiplier vector. ``ess`` is the one that matters.

    ``ess`` is the normalised effective sample size ``(Σm)² / (N · Σm²)`` — the fraction of
    pedagogy tokens that effectively carry gradient. It is 1.0 for vanilla SFT and falls
    toward 0 as T drops. It is also the only diagnostic here that is comparable *across
    corpora*, which is what makes it the right thing to quote when asking whether ``bT1`` on
    Impl 5's targets applies the same pressure that ``b-T1`` applied on gold.
    """
    m = np.asarray(m, dtype=np.float64).ravel()
    n = m.size
    ess = float(m.sum() ** 2 / (n * np.square(m).sum())) if n else 0.0
    nz = m[m > 0]
    # Shannon entropy of m/N read as a distribution, in nats, normalised by log N.
    p = nz / nz.sum() if nz.size else nz
    ent = float(-(p * np.log(p)).sum() / np.log(n)) if n > 1 and p.size else 0.0
    return {
        "n": int(n),
        "mean": float(m.mean()) if n else 0.0,
        "min": float(m.min()) if n else 0.0,
        "max": float(m.max()) if n else 0.0,
        "p50": float(np.percentile(m, 50)) if n else 0.0,
        "p99": float(np.percentile(m, 99)) if n else 0.0,
        "ess": ess,
        "entropy_frac": ent,
        "frac_below_0.01": float((m < 0.01).mean()) if n else 0.0,
        "frac_above_10": float((m > 10.0).mean()) if n else 0.0,
    }


# --------------------------------------------------------------------------- #
# ragged per-row storage
# --------------------------------------------------------------------------- #
@dataclass
class SignalCache:
    """Per-token signals for one training file, ragged, in the file's own row order.

    ``values[offsets[i]:offsets[i + 1]]`` are the signals for row ``i``'s loss-bearing tokens
    **in label order** — the k-th value belongs to the k-th unmasked label position. General
    rows are stored as zero-length spans, because they are never reweighted.

    ``row_hash`` is a digest of the row's ``input_ids`` and label mask. Training re-tokenises
    from the same file with the same function, so the hashes must match; if they do not, the
    multipliers would be applied to different tokens than they were computed for and the run
    would be quietly meaningless. That check is the reason this field exists.
    """

    variant: str
    values: np.ndarray            # float32, concatenated
    offsets: np.ndarray           # int64, len n_rows + 1
    row_hash: np.ndarray          # uint64, len n_rows
    is_pedagogy: np.ndarray       # bool,   len n_rows
    meta: dict

    @property
    def n_rows(self) -> int:
        return int(self.offsets.size - 1)

    def row(self, i: int) -> np.ndarray:
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def pedagogy_values(self) -> np.ndarray:
        """Every pedagogy signal, flat. General rows contribute nothing by construction."""
        return self.values

    def save(self, path) -> None:
        import json
        np.savez(
            path,
            variant=np.array(self.variant),
            values=self.values.astype(np.float32),
            offsets=self.offsets.astype(np.int64),
            row_hash=self.row_hash.astype(np.uint64),
            is_pedagogy=self.is_pedagogy.astype(bool),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path) -> "SignalCache":
        import json
        z = np.load(path, allow_pickle=False)
        return cls(
            variant=str(z["variant"].item()),
            values=z["values"],
            offsets=z["offsets"],
            row_hash=z["row_hash"],
            is_pedagogy=z["is_pedagogy"],
            meta=json.loads(str(z["meta"].item())),
        )


def row_digest(input_ids: Sequence[int], labels: Sequence[int]) -> np.uint64:
    """A 64-bit digest of one tokenised row, over the ids and the *mask* of the labels.

    The mask rather than the label values: labels are input_ids shifted and masked, so the
    values add nothing, while the mask is exactly the thing that has to agree between
    precompute and training.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(np.asarray(input_ids, dtype=np.int32).tobytes())
    h.update(np.asarray([t != -100 for t in labels], dtype=np.bool_).tobytes())
    return np.frombuffer(h.digest(), dtype=np.uint64)[0]


def content_key(*parts: str) -> str:
    """Cache key over (training-file content, variant, base model, reference adapter).

    Deliberately **not** including the temperature — §4.1: "note temperature is not in the
    key, so one precompute serves an entire temperature sweep".
    """
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def file_digest(path, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def signal_key(variant: str, train_file, base_model: str, reference=None,
               max_len: int | None = None) -> str:
    """The cache key for one variant. **The single definition** — both the precompute and the
    trainer call this rather than assembling a key themselves.

    They used to assemble it separately, and it diverged immediately: the precompute folded the
    reference adapter's digest into one key shared by both variants, while the trainer left it
    out for variant a because variant a has no reference. Same data, two keys, and the symptom
    was "missing signal cache" for an arm whose cache had just been written. ``smoke_klw.py``
    caught it; one function is what stops it recurring.

    **Variant a's key must not include the reference**, and this is semantics rather than
    convenience: ``s_t = −log π₀(y_t|ctx)`` does not depend on π_SFT, so a variant-a cache stays
    valid no matter which reference happens to be on disk. Variant b's key must include it —
    §4.1: "keep this fixed, since changing it changes both the signal and the precompute cache
    key."

    Temperature is never part of the key (§4.1), so one cache serves a whole sweep.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    parts = [file_digest(train_file), base_model]
    if variant == "b":
        ref_sf = Path(reference, "adapter_model.safetensors") if reference else None
        parts.append(file_digest(ref_sf) if (ref_sf and ref_sf.exists()) else "no-reference")
    else:
        parts.append("no-reference-needed")
    parts.append(str(max_len))
    return content_key(*parts)


# --------------------------------------------------------------------------- #
# cache -> per-row multiplier vectors
# --------------------------------------------------------------------------- #
def build_row_multipliers(cache: SignalCache, temperature: float) -> tuple[list[np.ndarray], dict]:
    """``([m per row], diagnostics)``, standardising and normalising over the whole corpus.

    The whole-corpus reduction happens here, once, before training — the trainer never sees
    a signal, only a finished multiplier per label position. That keeps the global softmax
    genuinely global: computing it inside the training loop would silently make it per-batch.
    """
    median, scale = robust_stats(cache.values)
    z = robust_z(cache.values, median, scale)
    m = multipliers(z, temperature)
    diag = {
        "variant": cache.variant,
        "temperature": float(temperature),
        "signal_median": median,
        "signal_mad_scaled": scale,
        "signal_min": float(cache.values.min()),
        "signal_max": float(cache.values.max()),
        "multiplier": describe(m),
    }
    rows = [m[cache.offsets[i]:cache.offsets[i + 1]].astype(np.float32)
            for i in range(cache.n_rows)]
    return rows, diag


def scatter_to_labels(labels: Sequence[int], row_m: np.ndarray,
                      general: bool = False) -> list[float]:
    """Lay a row's multipliers back out over its full token axis, aligned to ``labels``.

    Returns a list as long as ``labels``: ``m`` at each unmasked position in label order,
    ``GENERAL_MULTIPLIER`` at every unmasked position of a general row, and 0.0 at masked
    positions (where cross-entropy contributes nothing anyway, so the value is arbitrary —
    0.0 is chosen so that a padding bug shows up as a *missing* contribution rather than a
    plausible one).

    The alignment this function implements is the whole correctness question, and
    ``acceptance_checks_klw.py`` check W3 verifies it end-to-end through the collator's
    padding and the causal shift rather than here.
    """
    out = [0.0] * len(labels)
    k = 0
    for i, t in enumerate(labels):
        if t == -100:
            continue
        if general:
            out[i] = GENERAL_MULTIPLIER
        else:
            if k >= len(row_m):
                raise ValueError(
                    f"row has more unmasked labels than cached signals ({k + 1} > "
                    f"{len(row_m)}) — the cache was built from different tokenisation"
                )
            out[i] = float(row_m[k])
        k += 1
    if not general and k != len(row_m):
        raise ValueError(f"row has {k} unmasked labels but {len(row_m)} cached signals")
    return out
