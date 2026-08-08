#!/usr/bin/env python3
"""
Build the frozen gap-band masks the sliced evaluation reads: run 2's recall-versus-distance
endpoint.

WHY THIS EXISTS. Run 1 measured aggregate held-out cross-entropy over 975,077,376 tokens and came
back with the primary endpoint UNRESOLVED: pooled sigma-hat 0.02042 nats at df 12, MDE 0.0636
nats, against a literature target of 0.010-0.030. Six arms, no contrast significant under any
corrected test. That is not evidence the mixers are equivalent; it is evidence the aggregate is
the wrong instrument. An aggregate CE averages the handful of positions where a linear-attention
mixer's finite state actually has to *retrieve* something together with the overwhelming mass of
locally-predictable positions, where every arm in the bake-off is identical by construction. The
published evidence on these operators is a RECALL gap -- one study reports a 20-point recall
difference at near-identical perplexity. Diluting a 20-point recall gap across a corpus where
~90% of positions do not test recall is how you get 0.0051 nats and a p of 0.765.

THE ARITHMETIC THAT MAKES THIS THE HIGHEST-VALUE ADDITION TO RUN 2, and the honest version of it.
Slicing does NOT shrink sigma. Run 1's sigma-hat is seed noise -- init draw plus kernel
nondeterminism -- and the blocking analysis showed the shared data seed explains 1.8% of
within-arm variance, so 98.2% of it is a component no amount of extra tokens touches. A band mean
therefore carries roughly the SAME sigma as the aggregate, about 0.02 nats, plus a token-sampling
term this file's floors keep small. What slicing buys is a bigger NUMERATOR: if an effect of
0.010-0.030 nats is diluted over all positions and a tenth of positions carry it, the effect
inside those positions is ~0.1-0.3 nats, which is 1.6x to 4.7x the MDE that defeated run 1. The
endpoint is worth building because it moves the effect, not because it moves the noise. State it
that way in any writeup; "slicing gives more power" is the wrong sentence and invites the
sigma-pooling error this repo has already paid for once.

--------------------------------------------------------------------------------------------
WHAT "RECALL-CRITICAL AT DISTANCE d" MEANS HERE. THIS IS THE DEFINITION. ARGUE WITH THIS.
--------------------------------------------------------------------------------------------

Definition version: see :data:`DEFINITION_VERSION`. It is recorded in every manifest, and a
manifest built under a different version is a different endpoint that must not be pooled with
this one.

A scored position ``p`` (a TARGET -- the token the model must predict) is labelled by the distance
back to the most recent earlier point in the model's OWN VISIBLE CONTEXT at which the same
**bigram** was completed:

    ``p`` is recall-critical at distance ``d = p - q`` when ``q`` is the largest index with
    ``off + 1 <= q < p`` and ``(tok[q-1], tok[q]) == (tok[p-1], tok[p])``,

where ``off`` is the start of the evaluation window that contains ``p`` as a target. If no such
``q`` exists, ``p`` is labelled band 0 -- "no literal antecedent this model could have copied".

The bands are the boundaries in :data:`BAND_BIT`, and a band's name is the UPPER edge of a
right-closed interval:

    ====== ==================================================
    band   labels a scored position whose distance ``d`` is
    ====== ==================================================
    0      no visible antecedent at all
    32     ``1 <= d <= 32``
    256    ``32 < d <= 256``
    1024   ``256 < d <= 1024``
    4096   ``1024 < d <= 4096``
    ====== ==================================================

Exactly one bit is set per scored position. The bands PARTITION the scored set, so the band
counts sum to the aggregate count -- which is a checkable invariant and is checked, in
:func:`assign_bands` and again in :func:`verify_build`.

WHY UPPER-CLOSED AND DISJOINT, AND NOT THE OTHER TWO READINGS. Two rival readings exist and both
are defensible until you write down what they cost:

* *Nested / cumulative* ("band b = every position with d >= b"). The consumer's own log line
  formats these as ``gap>%-5s`` (``train_core6_arm.py:2969-2975``), which reads as this. It is
  the wrong choice: the bands are then supersets of one another, so band 4096's tokens are also
  band 32's, adjacent-band differences are confounded, and the counts cannot sum to the
  aggregate. It also makes the top band nearly empty -- see the next bullet.
* *Lower-closed disjoint* ("band 4096 = ``4096 <= d``"). Structurally near-empty. At
  ``sequence_length`` 4096 the largest distance a target can have to a visible antecedent is
  ``seq_len - 1 = 4095``, because the antecedent bigram needs both of its tokens inside the
  window. A band with a handful of tokens reports ``ce: null`` from
  ``band_ce_from_totals`` and the endpoint silently loses its most interesting cell. That is the
  "empty comparison set reports success" failure in its purest form.

The upper-closed disjoint reading is the one that leaves every band populated, partitions the
scored set, and satisfies the constraint in ``core6_arms.py:111-112`` -- ``SWA_WINDOW = 1024``,
"Must stay **below** the evaluation slice gap." Under this reading every distance in band 4096
lies in ``(1024, 4095]``, strictly outside a 1024-span sliding window. So arm ``S14`` cannot see
ANY band-4096 antecedent, which makes band 4096 the band where a windowed arm must degrade. That
is not a coincidence to note in passing: it is a POSITIVE CONTROL this study already owns, and it
is listed under falsification below.

The consumer's ``gap>`` log format is therefore cosmetically wrong for these masks; it should
read ``gap<=``. It is a format string in a file this script does not own, it has no effect on any
number, and it is reported rather than edited.

KNOWN BIASES. Every one of these is a real limit on what a band CE means.

1. **Band 0 is not "easy".** It is "no literal bigram antecedent was visible". It mixes genuinely
   novel text with text whose support is semantic, morphological, or a repeat of something longer
   ago than the window. Do not read band 0 as a locally-predictable control; read it as the
   complement of the recall probe.
2. **Band 0's size is a function of ``sequence_length``.** The visibility rule is deliberately
   the model's own: an antecedent one token before the window start is invisible to the model and
   is labelled band 0 here too. Change ``sequence_length`` and the partition changes. That is why
   the manifest records it and why masks built at one sequence length must never be scored at
   another. The consumer does NOT check this -- see UNDERSPECIFIED below.
3. **Distance is confounded with frequency, and therefore with difficulty.** High-frequency
   bigrams ("of the") recur at short distances, so band 32 is enriched in function words and band
   4096 in rare, topic-specific pairs. CONSEQUENCE, stated precisely because it decides what the
   endpoint can and cannot claim: comparing two ARMS *within* a band is unaffected -- every arm
   is scored on the same frozen mask over the same tokens, so composition cancels exactly.
   Comparing band LEVELS to each other is confounded, and the per-band CE curve is NOT a
   difficulty-versus-distance curve. Only the arm-minus-arm contrast inside a band is licensed.
4. **Bigrams only.** A trigram or a rarer key would be a more specific recall probe and would
   move mass out of the short bands. Bigram is the cheapest defensible choice and it is the unit
   induction heads are studied on; it is a floor on recall structure, not a census of it.
5. **Nothing here is causal.** A position labelled distance 300 is a position where a copyable
   antecedent existed 300 back. Whether the model used it is unobserved. The label is an
   OPPORTUNITY for recall, not a demonstration of it.

WHAT WOULD FALSIFY A CONCLUSION DRAWN FROM THIS. Four handles, cheapest first:

1. **The shuffle control.** Rebuild with ``--shuffle-labels SEED``: the same per-shard band
   counts, assigned to positions chosen by a pinned PRNG rather than by distance. Every band then
   has its true size and no distance information. If an arm ranking survives the shuffle, it is
   not about distance -- it is about band size, per-band token composition, or an artifact of the
   scoring path. This is the control to run FIRST on any positive result.
2. **The ``S14`` positive control.** ``SWA_WINDOW = 1024``, so ``S14`` provably cannot reach any
   band-4096 antecedent. If ``S14`` does not degrade in band 4096 relative to the other arms, and
   degrade there more than in bands 32/256, then the band labels do not mean what this file says
   they mean, and no other conclusion from these masks is safe.
3. **Key order.** Rebuild at ``--ngram 3``. An induction-style recall effect should get LARGER
   with a more specific key; an effect that vanishes was about bigram frequency.
4. **Window length.** Rebuild at a different ``--sequence-length`` and rescore. Band 0's mass and
   the top band's ceiling both move. A conclusion that flips is a conclusion about the evaluation
   window, not about the operator.

--------------------------------------------------------------------------------------------
THE ON-DISK FORMAT, DERIVED FROM THE CODE THAT READS IT
--------------------------------------------------------------------------------------------

Every clause below is pinned to the consumer, ``.edullm/train_core6_arm.py``. Nothing here is a
convention this script invented except where it says so.

* **One mask file per corpus shard**, ``uint8``, exactly ONE BYTE PER CORPUS TOKEN, flat at the
  mask prefix. ``evaluate_sliced`` does ``np.memmap(mp, dtype=np.uint8, mode="r")``
  (``train_core6_arm.py:1475``) and the length check is ``os.path.getsize(mp)`` against
  ``_shard_token_count(vp, dtype=np.uint32)`` (``:1409-1414``). One byte per token, no header,
  and byte order is a non-question at width 1.
* **The mask indexes the TARGET, not the input.** The read is
  ``mask[off + 1 : off + seq_len + 1]`` (``:1488``), while the inputs for that window are corpus
  positions ``off .. off + seq_len - 1``. So byte ``i`` labels the token AT corpus index ``i``,
  scored on the step that predicts it. Corpus position 0 is never a target and the tail past the
  last complete window is never read; this script writes ``0x00`` there.
* **Windows are non-overlapping and start at multiples of ``seq_len``.** ``_shard_windows``
  (``:1121-1150``) uses ``off = w * seq_len`` for ``w`` in ``range((n - 1) // seq_len)``. So the
  window that scores target ``p`` is ``off = ((p - 1) // seq_len) * seq_len``, and each scored
  position belongs to exactly one window.
* **Bits, not enums.** The select is ``(flat & bit) != 0`` per band (``:1498-1502``), so a byte
  may legally carry several bands. This script sets exactly one, for the reason above.
* **``slice_manifest.json`` at the mask prefix.** ``fetch_slice_inputs`` copies
  ``f"{base}/slice_manifest.json"`` (``:1203``) and reads:

  - ``bands`` -- must equal ``sorted(BAND_BIT)`` == ``[0, 32, 256, 1024, 4096]`` (``:1207``).
  - ``shards`` -- a list, iterated with ``enumerate`` and sharded across ranks by
    ``index % world_size == rank`` (``:1213-1215``). ORDER IS LOAD-BEARING: it decides which rank
    reads which shard. This script sorts by ``s3_key`` so the order is a property of the corpus.
  - ``shards[i]["s3_key"]`` -- fetched as ``s3://edullm-data/{s3_key}`` (``:1227``). Absent is an
    explicit refusal (``:1222-1226``).
  - ``shards[i]["mask"]`` -- fetched as ``{base}/{mask}`` (``:1228``). FLAT at the prefix, so mask
    names must be globally unique; this script asserts that.
  - ``shards[i]["shard"]`` -- a NAME, used only for local filenames and messages (``:1220``).
  - ``shards[i]["tokens"]`` -- checked as ``os.path.getsize(shard) // 4 == tokens`` (``:1232``).
    The ``// 4`` is hard-coded, which is what makes ``uint32`` a requirement rather than a
    preference; this script refuses any other width.
  - ``shards[i]["sha256"]`` -- of the MASK bytes, compared against
    ``sha256(mask).hexdigest()[:len(entry["sha256"])]`` (``:1237-1240``). This script writes the
    full 64 hex characters; see UNDERSPECIFIED for why a short one is dangerous.
  - ``c_mass`` and ``realized_mass`` -- LOGGED ONLY, never checked (``:1253-1254``).
    ``realized_mass`` is formatted as ``100 * value`` with ``%.3f%%``, so it is a fraction in
    ``[0, 1]``. Their meaning is defined by this file: see :data:`DEFAULT_C_MASS`.

* Everything else in the manifest is provenance this script adds; the consumer ignores unknown
  keys.

**Determinism.** No RNG is used on the default path. The output is a function of (corpus bytes,
``sequence_length``, ``--ngram``, :data:`DEFINITION_VERSION`) and nothing else: integer-only
arithmetic, shard list sorted by ``s3_key``, ``json.dumps(sort_keys=True)``, and NO wall-clock
time, hostname, or absolute path anywhere in ``slice_manifest.json``. Two people building from
the same corpus release get byte-identical masks AND a byte-identical manifest -- which is the
property that makes "compare arm A's run-2 band CE against arm B's" mean anything. Timing and the
machine go in ``slice_build_log.json``, a sidecar the consumer never opens. ``--shuffle-labels``
takes an explicit seed and is recorded in the manifest, so even the control arm is reproducible.

--------------------------------------------------------------------------------------------
THINGS IN THE CONSUMER PATH THAT LOOK WRONG OR UNDERSPECIFIED (reported, not edited)
--------------------------------------------------------------------------------------------

1. **The digest check is fail-open on an empty ``sha256``.** ``digest[:len(entry["sha256"])] ==
   entry["sha256"]`` (``:1238-1240``): a manifest with ``"sha256": ""`` compares ``"" == ""`` and
   passes for ANY mask bytes. A one-character digest passes 1 in 16 wrong masks. The manifest
   decides how much of the manifest gets checked. This script always writes 64 characters and
   :func:`verify_build` refuses anything shorter, but a hostile or truncated manifest is not
   caught on the consumer side.
2. **The shard content is never verified.** ``fetch_slice_inputs``'s docstring says the digest is
   "the one that catches a shard that is the right size and the wrong content", but the digest it
   checks is of the MASK; the shard gets only ``getsize // 4 == tokens`` (``:1232``). A
   same-length shard of different content passes. This script records ``shard_sha256`` per entry
   so the check becomes possible, and nothing on the consumer side reads it yet.
3. **``sequence_length`` is never compared.** The band partition depends on the evaluation
   window, and ``fetch_slice_inputs`` does not compare the manifest's ``sequence_length`` against
   ``opts.sequence_length``. Masks built at 4096 and scored at 2048 attribute every token to a
   band computed against a context the model never saw, and produce a completely plausible table.
   The field is written; the check does not exist. **This is the most dangerous of the four.**
4. **Only band NAMES are checked, not the bit layout.** ``manifest.get("bands") !=
   sorted(BAND_BIT)`` (``:1207``) compares ``[0, 32, 256, 1024, 4096]``. Two builds that agree on
   the names and disagree on which bit each band owns pass this check and mislabel every token.
   This script writes ``band_bit`` in full so the stronger check is one line away.

--------------------------------------------------------------------------------------------
WHERE THIS RUNS, AND WHAT IT COSTS. IT DOES NOT RUN ON A LAPTOP.
--------------------------------------------------------------------------------------------

It is CPU work over 975M tokens: ~3.9 GB of shard bytes down from ``s3://edullm-data`` and ~975
MB of masks back up. **Venue: an AWS Batch CPU job through the ``edullm`` CLI**, for three
reasons that are not about size. (a) The shards are in ``s3://edullm-data`` and the Batch workload
role is the principal that holds ``s3:GetObject`` there; a FarmShare node would need credentials
this project does not put on FarmShare. (b) The output has to land on S3 for
``--slice-mask-uri`` to fetch it, and the airlock means a job role is the writer. (c) 3.9 GB
should not transit a laptop or FarmShare's DTN to end up back in the same account.

**Estimated runtime, and where the number comes from.** The inner loop is one dict lookup, one
integer shift-or, one comparison chain and one dict store per token; the per-window dict reset
caps live keys at ``seq_len``. The test suite's own fixtures label at **2.2-2.4M tok/s** in
CPython 3.11 (printed per shard on stderr), which over 975,077,376 tokens is **~7 minutes of
single-core CPU**. A Batch vCPU is slower than this laptop and real text exercises the
dict-hit path the fixtures barely touch, so budget a band rather than a point: at 0.7M tok/s it
is **23 minutes single-core**, at 0.35M **46 minutes**. The work is embarrassingly parallel over
the 39 shards (100 MB and 25,001,984 tokens each), so ``--jobs 8`` puts a whole build at
**roughly 1-6 minutes of labelling** plus ~3.9 GB of download -- call it **10-20 minutes of wall
clock**, transfer-dominated. Declare an hour; it is a CPU profile and cheap.

The measured rate is not a promise about the Batch host, which is why the builder prints its own
tokens/sec after every shard: the estimate becomes a measurement inside the first minute of the
real run, early enough to kill a build that is an order of magnitude off rather than discovering
it at the timeout.

Submit it the same way every other job here is submitted -- ``edullm check`` then
``edullm submit``, with the command in a committed ``.edullm/*.yaml``. A sketch of the command
this script needs, for whoever owns that file (this script does not own it and does not write it)::

    python scripts/build_slice_masks.py build \\
      --dataset-id reservoir-dolma2 --dataset-version v1 \\
      --sequence-length 4096 --jobs 8 \\
      --out "$EDULLM_CHECKPOINT_DIR/slice-masks-v1"

and then, once, from a host that may write the mask prefix::

    python scripts/build_slice_masks.py verify <local-dir>   # before upload
    # upload, then point the training runs at it:
    #   --slice-mask-uri s3://.../slice-masks-v1

Usage::

    build_slice_masks.py build --out DIR (--dataset-id ID --dataset-version V | --shard-list F)
                               --sequence-length 4096 [--jobs N] [--ngram 2]
                               [--c-mass 0.05] [--min-band-tokens 25000000]
                               [--shuffle-labels SEED] [--limit-shards N]
    build_slice_masks.py verify DIR

Exit codes: 0 built or verified, 1 refused with an explanation on stderr, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------------
# THE CONTRACT. Every constant here is quoted from the consumer and checked against it.
# --------------------------------------------------------------------------------------------

#: Gap bands, and the bit each one occupies in the frozen mask files. MUST match
#: ``.edullm/train_core6_arm.py``'s ``BAND_BIT`` exactly -- the masks are written once and are
#: read there as bytes. :func:`consumer_band_bit` parses that file and
#: :func:`assert_bands_match_consumer` refuses a build when the two disagree, so this is not a
#: comment asking to be kept in sync.
BAND_BIT: Dict[int, int] = {0: 1, 32: 2, 256: 4, 1024: 8, 4096: 16}

#: The band names in the order the consumer compares them: ``sorted(BAND_BIT)``.
BANDS: Tuple[int, ...] = tuple(sorted(BAND_BIT))

#: The distance bands, smallest first. Band 0 is not a distance; it is "no visible antecedent".
POSITIVE_BANDS: Tuple[int, ...] = tuple(b for b in BANDS if b != 0)

#: The definition this build implements, recorded in every manifest.
#:
#: Bump it for ANY change to what a band means -- the key (bigram), the visibility rule (the
#: model's own window), the boundary side (upper-closed), or the band set. Two manifests with
#: different versions describe different endpoints and their band CEs must not be compared. The
#: string is the wire format of that promise.
DEFINITION_VERSION = "bigram-antecedent-upper-closed-v1"

#: The corpus width the consumer hard-codes. ``fetch_slice_inputs`` checks
#: ``os.path.getsize(shard) // 4 == tokens`` and ``evaluate_sliced`` memmaps ``np.uint32``, so a
#: corpus of any other width would be read as a different, in-range-looking token stream.
REQUIRED_DTYPE = "uint32"
REQUIRED_ITEMSIZE = 4
REQUIRED_BYTE_ORDER = "little"

#: The minimum coverage this endpoint needs to be worth scoring, as a fraction of scored
#: positions carrying a NON-ZERO band (i.e. having a visible antecedent). This is what
#: ``c_mass`` means in the manifest -- ``c`` for criterion -- and ``realized_mass`` is what the
#: build achieved against it.
#:
#: 0.05 is a FLOOR, NOT AN ESTIMATE, and it is deliberately far below any plausible value for
#: natural text: it exists to catch a build that has gone structurally wrong (the wrong dtype, a
#: shard of unique ids, a visibility rule that never fires), not to certify a good one. Nobody
#: has measured this fraction on ``reservoir-dolma2``, and this file cannot -- it does not run
#: where the data is. The realized number is printed and recorded, so the first real build
#: replaces the guess with a measurement; if it comes back near 0.05, that is a finding about the
#: corpus and not a passing gate.
DEFAULT_C_MASS = 0.05

#: Per-band minimum token count, below which the band is refused rather than shipped.
#:
#: DERIVED, not picked. Per-token CE has a standard deviation around 2.5 nats at this loss level.
#: Tokens inside a document are correlated, so take an effective sample size of ``n / 100`` --
#: a deliberately pessimistic 100-token correlation length. The sampling standard error of a band
#: mean is then ``2.5 / sqrt(n / 100)``, and requiring that to sit at a quarter of run 1's
#: measured seed noise (0.0204 nats), i.e. under 0.005 nats, gives
#: ``n >= 100 * (2.5 / 0.005) ** 2 = 25,000,000`` tokens. Over the 975,077,376-token held-out set
#: that is 2.56% per band.
#:
#: Enforced with a REFUSAL and a named per-band list, not a warning and not a mean. A band with
#: too few tokens is a cell whose arm-to-arm contrast is noise, and an EMPTY band is worse: it
#: reports ``ce: null`` from ``band_ce_from_totals`` and silently vanishes from the endpoint
#: while every other number looks fine.
DEFAULT_MIN_BAND_TOKENS = 25_000_000

#: Where the consumer lives, relative to the repository root, for the band-layout cross-check.
CONSUMER_RELATIVE_PATH = Path(".edullm") / "train_core6_arm.py"

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_NAME = "slice_manifest.json"
BUILD_LOG_NAME = "slice_build_log.json"
MASK_SUFFIX = ".mask.u8"


class Refused(Exception):
    """A build or verification that must not produce an artifact.

    Raised rather than logged, and never caught inside this module. A mask set that is wrong in
    any of the ways below is worse than no mask set: it produces a full per-band table on every
    arm, in range, that nobody can tell from a correct one.
    """


# --------------------------------------------------------------------------------------------
# The band rule. One function, called by everything, so there is one place to mutate.
# --------------------------------------------------------------------------------------------


def band_of_gap(gap: int) -> int:
    """The band a visible antecedent at distance ``gap`` belongs to.

    Upper-closed: ``gap == 32`` is band 32, ``gap == 33`` is band 256. See the module docstring
    for why this side and not the other.

    :raises ValueError: On ``gap < 1`` (a token cannot be its own antecedent) or on a gap past
        the top band, which is unrepresentable rather than merely unusual. Both are caller bugs;
        :func:`assert_sequence_length_is_representable` makes the second unreachable for a build
        that got past its configuration.
    """
    if gap < 1:
        raise ValueError(f"gap {gap} is not a distance; the nearest antecedent is 1 back")
    for boundary in POSITIVE_BANDS:
        if gap <= boundary:
            return boundary
    raise ValueError(
        f"gap {gap} exceeds the top band {POSITIVE_BANDS[-1]}, so it has no bit to occupy"
    )


def band_lower_bounds() -> Dict[int, int]:
    """The smallest gap each positive band can hold: band 32 -> 1, 256 -> 33, 1024 -> 257, ...

    The lower edge is the previous band's upper edge plus one, which is what "upper-closed
    interval" means. Derived from :data:`POSITIVE_BANDS` rather than written out, so it cannot
    disagree with :func:`band_of_gap`; :func:`test_the_lower_bounds_are_the_inverse_of_band_of_gap`
    checks the two against each other.
    """
    bounds: Dict[int, int] = {}
    previous = 0
    for boundary in POSITIVE_BANDS:
        bounds[boundary] = previous + 1
        previous = boundary
    return bounds


def assert_sequence_length_is_representable(seq_len: int) -> None:
    """Refuse a sequence length whose largest possible gap has no band.

    The furthest a visible antecedent can sit is ``seq_len - 1`` (the antecedent's own first
    token has to be inside the window too), so a window longer than ``top band + 1`` produces
    distances with nowhere to go. Refusing here beats discovering it as a ``ValueError`` a few
    hundred million tokens into a paid job -- and beats the alternative some readers reach for,
    which is to clamp the overflow into the top band and quietly redefine it.
    """
    if seq_len < 2:
        raise Refused(
            f"--sequence-length {seq_len} yields no scorable target; a window needs at least one "
            "input and one target"
        )
    top = POSITIVE_BANDS[-1]
    if seq_len - 1 > top:
        raise Refused(
            f"--sequence-length {seq_len} allows gaps up to {seq_len - 1}, past the top band "
            f"{top}, so some positions would have no bit to occupy. Add a band to BAND_BIT (and "
            "to the consumer's, and bump DEFINITION_VERSION) rather than clamping into the top "
            "band, which would silently change what band "
            f"{top} means."
        )


# --------------------------------------------------------------------------------------------
# THE DEFINITION, as executable code. Pure integers, no third-party import, one implementation.
# --------------------------------------------------------------------------------------------


@dataclass
class BandAssignment:
    """The labelling of one shard, and the accounting that proves it covers what it should."""

    mask: bytes
    #: Per-band scored-position counts, keys exactly :data:`BANDS`.
    counts: Dict[int, int]
    #: Positions this shard's windows actually score: ``windows * seq_len``.
    scored: int
    #: Positions never read by the consumer: index 0 plus the tail past the last full window.
    unscored: int
    #: Complete windows of ``seq_len`` inputs plus one target.
    windows: int

    @property
    def with_antecedent(self) -> int:
        """Scored positions in a distance band, i.e. every band except 0."""
        return sum(self.counts[b] for b in POSITIVE_BANDS)


def window_count(n_tokens: int, seq_len: int) -> int:
    """Complete windows in a shard, matching ``_shard_windows``'s ``(tokens.size - 1) // seq_len``.

    Spelled out once and shared, because the ``- 1`` is one of the three off-by-ones the consumer
    warns about: a window needs ``seq_len`` inputs AND one more token to be its last target, so 96
    tokens at ``seq_len`` 32 is 2 windows, not 3. ``n // seq_len`` reads one token past the end.
    """
    return max((n_tokens - 1) // seq_len, 0)


def _label_windows(blocks: "Any", *, n_tokens: int, seq_len: int, ngram: int) -> BandAssignment:
    """THE DEFINITION, AS THE ONLY IMPLEMENTATION OF IT.

    Every caller -- the in-memory one a test uses and the streaming one the corpus build uses --
    goes through this function, so there is exactly one copy of the rule. A fast second copy
    would be a second thing to keep right, and the failure mode of two copies disagreeing is a
    labelling that is correct on the test data and wrong on the corpus: the test passes, and it
    was never evidence about the code that ran.

    ``blocks`` yields ``(off, block)`` where ``block`` is the ``seq_len + 1`` tokens starting at
    corpus index ``off`` -- the window's inputs plus its final target -- and ``off`` advances by
    ``seq_len``. Indexed ``block[local]`` for ``local in 0..seq_len``, so it may be a list, an
    ``array('I')``, or any sequence; nothing here needs numpy or torch.

    THE PER-WINDOW DICTIONARY RESET IS THE VISIBILITY RULE, NOT AN OPTIMISATION. ``last`` is
    created inside the loop, so it only ever holds keys whose span lies inside the current window
    -- an antecedent one token before ``off`` is invisible here exactly as it is invisible to the
    model. Hoisting it out would let a position "recall" something the model never saw, and the
    resulting table would look entirely normal. It also caps the dictionary at ``seq_len`` keys,
    which is why a 25M-token shard costs kilobytes -- but correctness is the reason.

    :returns: A :class:`BandAssignment` whose ``mask`` is one byte per token.
    :raises Refused: If the band counts do not partition the scored set -- see the final check.
    """
    if ngram < 2:
        raise Refused(f"ngram {ngram} is not a key: a recall probe needs a context token")
    assert_sequence_length_is_representable(seq_len)

    mask = bytearray(n_tokens)  # 0x00 everywhere: unscored positions stay unlabelled.
    counts = {b: 0 for b in BANDS}
    zero_bit = BAND_BIT[0]
    bits = {b: BAND_BIT[b] for b in POSITIVE_BANDS}
    # A position needs `ngram` tokens ending at it, all inside the window, so the earliest
    # position of a window that can carry a key is at local offset `ngram - 1`.
    first_keyed_offset = ngram - 1
    windows = 0

    for off, block in blocks:
        windows += 1
        last: Dict[int, int] = {}
        for local in range(1, seq_len + 1):
            q = off + local
            if local < first_keyed_offset:
                # Scored by the consumer, so it must carry a bit, but it has no room behind it
                # for a full key inside the window -- so there is nothing it could have copied.
                mask[q] = zero_bit
                counts[0] += 1
                continue
            key = 0
            for back in range(ngram - 1, -1, -1):
                key = (key << 32) | int(block[local - back])
            previous = last.get(key)
            if previous is None:
                mask[q] = zero_bit
                counts[0] += 1
            else:
                band = band_of_gap(q - previous)
                mask[q] = bits[band]
                counts[band] += 1
            last[key] = q

    expected_windows = window_count(n_tokens, seq_len)
    if windows != expected_windows:
        raise Refused(
            f"labelled {windows} window(s) but {n_tokens:,} token(s) at sequence length "
            f"{seq_len} is {expected_windows}; the mask and the consumer's read would cover "
            "different positions"
        )
    scored = windows * seq_len
    # THE PARTITION, ASSERTED. Exactly one bit per scored position means the band counts must sum
    # to the scored count -- the same equality the consumer's `agg_n` and summed `band_n` have to
    # satisfy. If this ever fails, the endpoint's denominators are wrong and every band CE with
    # it, so it refuses here rather than shipping a table.
    total = sum(counts.values())
    if total != scored:
        raise Refused(
            f"band counts sum to {total:,} but {scored:,} positions are scored, so the bands do "
            "not partition the scored set and the per-band denominators are wrong"
        )
    return BandAssignment(
        mask=bytes(mask),
        counts=counts,
        scored=scored,
        unscored=n_tokens - scored,
        windows=windows,
    )


def assign_bands(tokens: Sequence[int], seq_len: int, *, ngram: int = 2) -> BandAssignment:
    """Label one shard held in memory. Calls :func:`_label_windows`; adds no rule of its own.

    The entry point a test uses on a hand-built token list where the right answer is known. It is
    the same code path the corpus build takes, differing only in where the window bytes come
    from, so a test that passes here is evidence about the builder.
    """

    def blocks():
        for w in range(window_count(len(tokens), seq_len)):
            off = w * seq_len
            yield off, tokens[off : off + seq_len + 1]

    return _label_windows(blocks(), n_tokens=len(tokens), seq_len=seq_len, ngram=ngram)


def shuffle_labels(
    assignment: BandAssignment, seq_len: int, *, seed: int, n_tokens: int
) -> BandAssignment:
    """The falsification control: the same band SIZES, assigned to positions at random.

    Destroys the distance information and keeps everything else -- per-shard band counts, the
    scored set, the file length, the partition property. An arm ranking that survives this is not
    a ranking about distance. Seeded from an explicit integer that lands in the manifest, so the
    control is as reproducible as the real thing.
    """
    positions: List[int] = []
    for w in range(assignment.windows):
        off = w * seq_len
        positions.extend(range(off + 1, off + seq_len + 1))
    rng = random.Random(seed)
    rng.shuffle(positions)

    mask = bytearray(n_tokens)
    cursor = 0
    for band in BANDS:  # sorted, so the consumption order does not depend on dict order
        bit = BAND_BIT[band]
        for index in range(cursor, cursor + assignment.counts[band]):
            mask[positions[index]] = bit
        cursor += assignment.counts[band]
    if cursor != assignment.scored:
        raise Refused(
            f"shuffle consumed {cursor:,} of {assignment.scored:,} scored positions; the control "
            "would not have the same band sizes as the real labelling"
        )
    return BandAssignment(
        mask=bytes(mask),
        counts=dict(assignment.counts),
        scored=assignment.scored,
        unscored=assignment.unscored,
        windows=assignment.windows,
    )


# --------------------------------------------------------------------------------------------
# Shard I/O. stdlib only: `array` is native-order, which is what the consumer memmaps.
# --------------------------------------------------------------------------------------------


def assert_host_can_write_native_masks() -> None:
    """Refuse a host whose native integers are not what the consumer will read back.

    ``evaluate_sliced`` memmaps ``np.uint32`` in NATIVE order and ``corpus_from_manifest``
    already refuses a corpus whose declared order differs from the host's. So a shard decoded on
    a big-endian builder yields different, in-range-looking ids and a mask that is plausible and
    wrong everywhere.
    """
    if sys.byteorder != REQUIRED_BYTE_ORDER:
        raise Refused(
            f"this host is {sys.byteorder}-endian and the corpus is {REQUIRED_BYTE_ORDER}-endian; "
            "every token would decode to a different in-range id and the masks would be wrong "
            "without being detectably wrong"
        )
    if array("I").itemsize != REQUIRED_ITEMSIZE:
        raise Refused(
            f"array('I') is {array('I').itemsize} bytes on this host, not {REQUIRED_ITEMSIZE}; "
            "the shard would be decoded at the wrong width"
        )


def read_window(handle, off: int, count: int) -> array:
    """``count`` tokens starting at token offset ``off``, as a native ``uint32`` array.

    Read through a handle rather than loading the shard, so peak memory is a window and not 100
    MB per worker. A short read is a truncated shard and is refused: the alternative is a final
    window scored against fewer tokens than the consumer will read for it.
    """
    handle.seek(off * REQUIRED_ITEMSIZE)
    raw = handle.read(count * REQUIRED_ITEMSIZE)
    if len(raw) != count * REQUIRED_ITEMSIZE:
        raise Refused(
            f"short read at token offset {off}: wanted {count} token(s), got "
            f"{len(raw) // REQUIRED_ITEMSIZE}"
        )
    out = array("I")
    out.frombytes(raw)
    return out


def shard_token_count(path: str) -> int:
    """Tokens on disk, from the file size, the way the consumer counts them.

    ``fetch_slice_inputs`` computes ``os.path.getsize(shard) // 4`` and ``_shard_token_count``
    divides by ``np.dtype(np.uint32).itemsize``. A file whose size is not a whole number of
    tokens is refused rather than floored, because flooring is how a truncated download becomes a
    slightly-short corpus that nothing notices.
    """
    size = os.path.getsize(path)
    if size % REQUIRED_ITEMSIZE:
        raise Refused(
            f"{path} is {size} bytes, not a whole number of {REQUIRED_ITEMSIZE}-byte tokens; it "
            "is truncated or is not a u32le shard"
        )
    return size // REQUIRED_ITEMSIZE


def assign_bands_from_file(path: str, seq_len: int, *, ngram: int, n_tokens: int) -> BandAssignment:
    """Label a shard on disk, one window of tokens in memory at a time.

    Calls :func:`_label_windows` -- the same and only rule :func:`assign_bands` calls -- and
    supplies its window bytes from a file handle instead of a list. Nothing about the labelling
    lives here. Peak memory is one window, so a 100 MB shard costs the mask plus ``seq_len``
    tokens rather than the shard.

    Each window is a fresh ``seek``+``read`` of ``seq_len + 1`` tokens, which is one seek per
    4096 tokens and is not the bottleneck; the per-token arithmetic is.
    """

    def blocks():
        with open(path, "rb") as handle:
            for w in range(window_count(n_tokens, seq_len)):
                off = w * seq_len
                # seq_len + 1 tokens: the window's inputs plus its final target.
                yield off, read_window(handle, off, seq_len + 1)

    return _label_windows(blocks(), n_tokens=n_tokens, seq_len=seq_len, ngram=ngram)


# --------------------------------------------------------------------------------------------
# The consumer cross-check. Parsed out of the consumer's source, not restated here.
# --------------------------------------------------------------------------------------------


def consumer_band_bit(path: Path) -> Dict[int, int]:
    """``BAND_BIT`` as the consumer declares it, read by parsing rather than by importing.

    Importing ``train_core6_arm`` pulls in torch, olmo_core and the dataset reader, none of which
    a mask build needs and none of which a laptop should load. ``ast`` reads the literal.

    :raises Refused: If the file is missing or does not declare a usable ``BAND_BIT``. NOT a
        skip: an absent artifact makes this check unperformed, and an unperformed check that
        reports success is how a mask set gets built against a bit layout nobody compared.
    """
    if not path.exists():
        raise Refused(
            f"{path} is not here, so the band layout could not be compared against the code that "
            "reads the masks. That check is not optional -- pass --consumer with the path, or "
            "--no-consumer-check to record in the manifest that it was skipped and why"
        )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: List[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id != "BAND_BIT":
                continue
            if node.value is None:
                continue
            try:
                literal = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise Refused(f"{path} declares BAND_BIT but it is not a literal: {exc}") from exc
            if not isinstance(literal, dict) or not literal:
                raise Refused(f"{path} declares BAND_BIT as {type(literal).__name__}, not a dict")
            return {int(k): int(v) for k, v in literal.items()}
    raise Refused(f"{path} declares no module-level BAND_BIT to compare against")


def assert_bands_match_consumer(path: Path) -> Dict[int, int]:
    """Refuse a build whose bit layout differs from the consumer's in ANY way.

    Compares the whole mapping, not just ``sorted(BAND_BIT)`` -- which is all the consumer itself
    checks at ``train_core6_arm.py:1207``. Agreeing on the band names while disagreeing on which
    bit each owns passes that check and mislabels every token in the corpus.
    """
    theirs = consumer_band_bit(path)
    if theirs != BAND_BIT:
        raise Refused(
            f"BAND_BIT here is {BAND_BIT} and {path} declares {theirs}. The masks are written "
            "once and read as bytes, so a disagreement on the bit layout attributes every token "
            "to the wrong band while the consumer's own band-name check still passes"
        )
    return theirs


def source_digest(path: Path) -> str:
    """SHA-256 of a source file, so an artifact can name the code that produced it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Shard naming. Unique by construction, because the mask prefix is FLAT.
# --------------------------------------------------------------------------------------------


def shard_name_from_key(s3_key: str) -> str:
    """A unique, human-readable name for a corpus key.

    ``fetch_slice_inputs`` fetches masks as ``{prefix}/{entry['mask']}`` -- one flat namespace --
    so two shards that produced the same mask name would have one silently overwrite the other at
    upload time and the run would score a third of its tokens against another topic's labels.

    The documented convention is ``<source>__<shard>.u32le.bin``, and the consumer's own
    docstrings record why that form is dangerous: ``val-00212.u32le.bin`` exists under 24 topic
    directories, so a two-component name collides. This joins EVERY path component after the
    bucket with ``__``, which is unique because S3 keys are. :func:`assert_names_are_unique`
    checks it rather than trusting the argument.
    """
    parts = [p for p in s3_key.strip("/").split("/") if p]
    if not parts:
        raise Refused(f"{s3_key!r} has no path components to name a shard after")
    return "__".join(parts)


def mask_name_for(shard_name: str) -> str:
    """``<shard name with its .u32le.bin dropped>.mask.u8``."""
    stem = shard_name
    for suffix in (".u32le.bin", ".bin", ".npy"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + MASK_SUFFIX


def assert_names_are_unique(entries: Sequence[Dict[str, Any]]) -> None:
    """Refuse duplicate mask names, shard names, or s3 keys before anything is written."""
    for field_name in ("mask", "shard", "s3_key"):
        seen: Dict[str, int] = {}
        for entry in entries:
            value = entry[field_name]
            seen[value] = seen.get(value, 0) + 1
        clashes = sorted(name for name, count in seen.items() if count > 1)
        if clashes:
            raise Refused(
                f"{len(clashes)} duplicate {field_name} value(s): {clashes[:5]}. The mask prefix "
                "is flat, so a repeated name means one mask overwrites another and a shard is "
                "scored against the wrong labels"
            )


# --------------------------------------------------------------------------------------------
# The shard list: where the 39 held-out objects come from.
# --------------------------------------------------------------------------------------------


@dataclass
class ShardSpec:
    """One held-out object: its corpus key, and where its bytes are (or will be) locally."""

    s3_key: str
    local: Optional[str] = None


@dataclass
class CorpusSpec:
    """The corpus release the masks are built against, plus its declared physical layout."""

    dataset_id: str
    dataset_version: str
    dtype: str
    byte_order: str
    header_bytes: int
    shards: List[ShardSpec] = field(default_factory=list)

    def assert_readable_the_way_the_consumer_reads_it(self) -> None:
        """Refuse any layout the consumer's hard-coded ``uint32`` / offset-0 read would corrupt."""
        if self.dtype != REQUIRED_DTYPE:
            raise Refused(
                f"{self.dataset_id}/{self.dataset_version} is {self.dtype} and the consumer "
                f"hard-codes {REQUIRED_DTYPE} (`getsize // 4`, `np.memmap(dtype=np.uint32)`); "
                "the shard would be decoded at the wrong width"
            )
        if self.byte_order != REQUIRED_BYTE_ORDER:
            raise Refused(
                f"{self.dataset_id}/{self.dataset_version} is {self.byte_order}-endian and the "
                f"consumer memmaps in native order on a {REQUIRED_BYTE_ORDER}-endian host"
            )
        if self.header_bytes:
            raise Refused(
                f"{self.dataset_id}/{self.dataset_version} declares {self.header_bytes} header "
                "byte(s) and both this builder and the consumer read from offset zero, so the "
                "header would be labelled and scored as tokens"
            )


def corpus_from_shard_list(path: Path) -> CorpusSpec:
    """Read a recorded shard list, so a rebuild does not depend on resolving the catalog again.

    The format is what :func:`corpus_from_reader` writes into the build log, which is what makes
    a rebuild on a host without the dataset reader byte-identical to the original::

        {"dataset_id": ..., "dataset_version": ..., "dtype": "uint32",
         "byte_order": "little", "header_bytes": 0,
         "shards": [{"s3_key": "...", "local": "optional/local/path"}, ...]}
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("dataset_id", "dataset_version", "dtype", "byte_order", "shards")
        if key not in payload
    ]
    if missing:
        raise Refused(f"{path} is missing {', '.join(missing)}; it cannot describe a corpus")
    shards = [
        ShardSpec(s3_key=entry["s3_key"], local=entry.get("local")) for entry in payload["shards"]
    ]
    if not shards:
        raise Refused(f"{path} lists no shards")
    return CorpusSpec(
        dataset_id=str(payload["dataset_id"]),
        dataset_version=str(payload["dataset_version"]),
        dtype=str(payload["dtype"]),
        byte_order=str(payload["byte_order"]),
        header_bytes=int(payload.get("header_bytes", 0)),
        shards=shards,
    )


def corpus_from_reader(dataset_id: str, dataset_version: str) -> CorpusSpec:
    """Resolve the corpus's OWN held-out objects through the dataset reader.

    Same source of truth the training run uses: ``read.val``, which is a property over
    ``read.splits`` filtered by the reader's ``is_trainable``. NOT reconstructed from shard
    names, for the reason ``corpus_from_manifest`` documents at length -- ``val-00212.u32le.bin``
    exists under 24 topic directories, and a key rebuilt from a name fetches a real, readable
    shard of the wrong topic.

    Imported here rather than at module scope so that everything above can be exercised on a host
    without the reader installed.
    """
    try:
        from edullm_data.read import dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:
        raise Refused(
            f"the dataset reader is not importable here ({exc}), so the held-out shard list "
            "cannot be resolved. Pass --shard-list with a recorded list instead, or run this "
            "where the reader is installed (the research image)"
        ) from exc

    s3 = Boto3S3.default()
    version = dataset_version
    if version in ("", "latest"):
        resolved = resolve_latest(dataset_id, s3=s3)
        if resolved is None:
            raise Refused(f"no published version of {dataset_id}")
        version = resolved
    read = dataset_paths(dataset_id, version, s3=s3)

    val_paths = list(getattr(read, "val", None) or [])
    if not val_paths:
        raise Refused(
            f"{dataset_id}/{version} declares no held-out split, so there is nothing to label. "
            "The sliced endpoint scores the corpus's own val partition"
        )
    if len(set(val_paths)) != len(val_paths):
        raise Refused(
            f"{dataset_id}/{version} lists a held-out object more than once; those tokens would "
            "be labelled twice and weighted twice in every band mean"
        )
    return CorpusSpec(
        dataset_id=dataset_id,
        dataset_version=version,
        dtype=str(getattr(read, "dtype", "") or ""),
        byte_order=str(getattr(read, "byte_order", "") or REQUIRED_BYTE_ORDER),
        header_bytes=int(getattr(read, "header_bytes", 0) or 0),
        shards=[ShardSpec(s3_key=_key_of(uri)) for uri in val_paths],
    )


def _key_of(uri: str) -> str:
    """``s3://edullm-data/a/b/c`` -> ``a/b/c``.

    The consumer refetches as ``s3://edullm-data/{s3_key}`` (``train_core6_arm.py:1227``), so the
    bucket is fixed there and a key from any other bucket cannot be expressed in the manifest.
    Refused rather than silently rewritten.
    """
    text = uri.strip()
    if not text.startswith("s3://"):
        return text.lstrip("/")
    without = text[len("s3://") :]
    bucket, _, key = without.partition("/")
    if bucket != "edullm-data":
        raise Refused(
            f"{uri} is in bucket {bucket!r}, but the consumer refetches every shard from "
            "s3://edullm-data/{s3_key} and cannot express another bucket"
        )
    return key


def ensure_local(spec: ShardSpec, cache_dir: Path) -> str:
    """The shard's bytes on this host, downloading them once if needed.

    Named by a digest of the key, not by its basename: two topics genuinely contain a
    ``val-00212.u32le.bin`` and the second download would overwrite the first, halving the token
    count in a way that looks like a short corpus.
    """
    if spec.local:
        if not os.path.exists(spec.local):
            raise Refused(f"{spec.local} (for {spec.s3_key}) is not on this host")
        return spec.local
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256(spec.s3_key.encode("utf-8")).hexdigest()[:16]
    target = cache_dir / f"{stamp}-{os.path.basename(spec.s3_key)}"
    if target.exists():
        return str(target)
    try:
        import boto3
    except ImportError as exc:
        raise Refused(
            f"boto3 is not importable, so {spec.s3_key} cannot be fetched. Run this where the "
            "workload role and boto3 are (an AWS Batch CPU job), or stage the shards and point "
            "--shard-list at them with a `local` field per entry"
        ) from exc
    partial = target.with_suffix(target.suffix + ".partial")
    boto3.client("s3").download_file("edullm-data", spec.s3_key, str(partial))
    # Rename only after the body is complete, so an interrupted download cannot be mistaken for a
    # cached shard on the next attempt -- which would be scored as a short corpus.
    os.replace(partial, target)
    return str(target)


# --------------------------------------------------------------------------------------------
# The build.
# --------------------------------------------------------------------------------------------


@dataclass
class BuiltShard:
    """One finished mask, with everything the manifest and the accounting need."""

    index: int
    s3_key: str
    shard: str
    mask: str
    tokens: int
    mask_sha256: str
    shard_sha256: str
    counts: Dict[int, int]
    scored: int
    unscored: int
    windows: int
    seconds: float


def build_one(
    spec: ShardSpec,
    *,
    index: int,
    out_dir: Path,
    cache_dir: Path,
    seq_len: int,
    ngram: int,
    shuffle_seed: Optional[int],
) -> BuiltShard:
    """Label one shard and write its mask. Independent of every other shard, hence parallelisable.

    Determinism does not depend on the order these complete in: the caller sorts by ``index``,
    and ``index`` comes from the ``s3_key``-sorted list.
    """
    started = time.monotonic()
    local = ensure_local(spec, cache_dir)
    n_tokens = shard_token_count(local)
    if n_tokens <= seq_len:
        raise Refused(
            f"{spec.s3_key} holds {n_tokens:,} token(s), which yields no complete window of "
            f"{seq_len}; the consumer would read no mask bytes for it and the shard would be "
            "declared but unscored"
        )

    assignment = assign_bands_from_file(local, seq_len, ngram=ngram, n_tokens=n_tokens)
    if shuffle_seed is not None:
        # Per-shard seed derived from the declared one AND the key, so two shards do not receive
        # the same permutation and the whole control is still reproducible from one integer.
        derived = int.from_bytes(
            hashlib.sha256(f"{shuffle_seed}:{spec.s3_key}".encode("utf-8")).digest()[:8], "big"
        )
        assignment = shuffle_labels(assignment, seq_len, seed=derived, n_tokens=n_tokens)

    if len(assignment.mask) != n_tokens:
        raise Refused(
            f"mask for {spec.s3_key} is {len(assignment.mask):,} byte(s) for {n_tokens:,} "
            "token(s); the consumer's length check would refuse it"
        )

    shard_name = shard_name_from_key(spec.s3_key)
    mask_name = mask_name_for(shard_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / mask_name
    partial = mask_path.with_suffix(mask_path.suffix + ".partial")
    partial.write_bytes(assignment.mask)
    os.replace(partial, mask_path)

    return BuiltShard(
        index=index,
        s3_key=spec.s3_key,
        shard=shard_name,
        mask=mask_name,
        tokens=n_tokens,
        mask_sha256=hashlib.sha256(assignment.mask).hexdigest(),
        shard_sha256=_file_digest(local),
        counts=dict(assignment.counts),
        scored=assignment.scored,
        unscored=assignment.unscored,
        windows=assignment.windows,
        seconds=time.monotonic() - started,
    )


def _file_digest(path: str) -> str:
    """SHA-256 of a file, read in blocks so a 100 MB shard is not held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_bands_are_live(
    counts: Dict[int, int], *, min_band_tokens: int, total_scored: int
) -> None:
    """Refuse a mask set with a band too thin to carry an arm contrast, NAMING the bands.

    PER BAND, NOT ON A MEAN. A mean over five bands stays comfortable while one of them is dead:
    this project has already shipped a component at 3.2e-4 -- provably inert -- under an
    aggregate that read 0.186 and looked healthy. So the check is per band and the failure names
    the offenders and their counts.

    An EMPTY band is called out separately because it is not a small version of the same problem.
    ``band_ce_from_totals`` returns ``ce: null`` for ``n == 0``, so an empty band does not show up
    as a bad number; it shows up as a missing row, in a table where every other row is fine.
    """
    empty = [b for b in BANDS if counts.get(b, 0) == 0]
    thin = [b for b in BANDS if 0 < counts.get(b, 0) < min_band_tokens]
    if empty:
        raise Refused(
            "band(s) "
            + ", ".join(str(b) for b in empty)
            + " have NO labelled token, so they would report `ce: null` and vanish from the "
            "endpoint rather than show up as a bad number. Counts: "
            + json.dumps({str(b): counts.get(b, 0) for b in BANDS})
        )
    if thin:
        detail = ", ".join(f"{b}: {counts[b]:,}" for b in thin)
        raise Refused(
            f"band(s) below the {min_band_tokens:,}-token floor -- {detail} -- over "
            f"{total_scored:,} scored position(s). At that count the band's own sampling error is "
            "comparable to the seed noise the arm contrast has to beat. Widen the corpus, widen "
            "the band, or lower --min-band-tokens knowingly (it is recorded in the manifest)"
        )


def assert_mass_clears_the_criterion(
    *, with_antecedent: int, total_scored: int, c_mass: float
) -> float:
    """Refuse a build whose recall coverage is too low to be an endpoint, and return the mass.

    ``realized_mass`` is the fraction of scored positions with a visible antecedent -- the
    positions the recall bands are made of. ``c_mass`` is the declared criterion it must clear.
    Fires on exactly the structural failures worth catching: a corpus of unique ids, a visibility
    rule that never matches, a dtype read at the wrong width.
    """
    if total_scored <= 0:
        raise Refused("no position was scored, so there is no mass to measure")
    realized = with_antecedent / total_scored
    if realized < c_mass:
        raise Refused(
            f"realized mass {realized:.6f} ({with_antecedent:,} of {total_scored:,} scored "
            f"position(s) have a visible antecedent) is below the declared criterion "
            f"c_mass={c_mass}. The recall bands would carry too little of the corpus to move any "
            "endpoint. This is the shape of failure a wrong dtype or a broken visibility rule "
            "produces, so check those before lowering the criterion"
        )
    return realized


def manifest_from_builds(
    builds: Sequence[BuiltShard],
    *,
    corpus: CorpusSpec,
    seq_len: int,
    ngram: int,
    c_mass: float,
    min_band_tokens: int,
    shuffle_seed: Optional[int],
    consumer_checked: Optional[str],
    builder_sha256: str,
) -> Dict[str, Any]:
    """The manifest, assembled so that two builds from one corpus are byte-identical.

    NO TIMESTAMP, NO HOSTNAME, NO ABSOLUTE PATH. Those are the three fields that would make every
    build differ and turn "the masks are frozen" into a claim nobody can check. They go in
    :data:`BUILD_LOG_NAME`, which the consumer never opens.

    ``shards`` is in the order the builds arrive, which the caller has sorted by ``s3_key``. That
    order is load-bearing beyond aesthetics: the consumer shards work across ranks by
    ``index % world_size``, so a reordered manifest gives every rank a different subset. The
    union is the same and the numbers are the same, but the digest of the manifest is not, and a
    reordering is exactly the silent change this file exists to prevent.
    """
    entries: List[Dict[str, Any]] = []
    for build in builds:
        entries.append(
            {
                # --- the five fields the consumer reads. See the module docstring for cites.
                "s3_key": build.s3_key,
                "shard": build.shard,
                "mask": build.mask,
                "tokens": build.tokens,
                "sha256": build.mask_sha256,
                # --- provenance and accounting the consumer ignores today.
                "shard_sha256": build.shard_sha256,
                "band_counts": {str(b): build.counts[b] for b in BANDS},
                "scored": build.scored,
                "unscored": build.unscored,
                "windows": build.windows,
            }
        )
    assert_names_are_unique(entries)

    totals = {b: sum(build.counts[b] for build in builds) for b in BANDS}
    scored = sum(build.scored for build in builds)
    tokens = sum(build.tokens for build in builds)
    with_antecedent = sum(totals[b] for b in POSITIVE_BANDS)

    return {
        # THE FIELD THE CONSUMER COMPARES, and it must be `sorted(BAND_BIT)`.
        "bands": list(BANDS),
        "shards": entries,
        "c_mass": c_mass,
        "realized_mass": with_antecedent / scored if scored else 0.0,
        # --- everything below is provenance. Unknown keys are ignored by the consumer.
        "band_bit": {str(b): BAND_BIT[b] for b in BANDS},
        "band_semantics": (
            "Band names are UPPER edges of right-closed distance intervals over the gap back to "
            "the most recent earlier completion of the same bigram inside the model's own "
            "evaluation window. Band 0 means no visible antecedent. Exactly one bit is set per "
            "scored position, so the band counts partition the scored set."
        ),
        "definition_version": DEFINITION_VERSION,
        "dataset_id": corpus.dataset_id,
        "dataset_version": corpus.dataset_version,
        "dtype": corpus.dtype,
        "byte_order": corpus.byte_order,
        # THE FIELD THE CONSUMER DOES NOT CHECK AND MUST. Masks built at one window and scored at
        # another attribute every token to a band computed against a context the model never saw.
        "sequence_length": seq_len,
        "ngram": ngram,
        "min_band_tokens": min_band_tokens,
        "shuffle_labels_seed": shuffle_seed,
        "is_shuffled_control": shuffle_seed is not None,
        "consumer_band_bit_sha256": consumer_checked,
        "builder_sha256": builder_sha256,
        "totals": {
            "shards": len(entries),
            "tokens": tokens,
            "scored": scored,
            "unscored": tokens - scored,
            "band_counts": {str(b): totals[b] for b in BANDS},
            "with_antecedent": with_antecedent,
        },
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """One writer, one encoding, so two builds produce identical bytes.

    ``sort_keys`` removes insertion order from the output, ``indent=2`` and the trailing newline
    make it readable and diffable, and ``ensure_ascii`` keeps the bytes independent of locale.
    """
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------------------------
# Verification: the path that rejects a truncated or altered mask set.
# --------------------------------------------------------------------------------------------


def verify_build(directory: Path, *, min_band_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Re-check a built mask directory the way the consumer will, and then harder.

    Run before upload and again after, because the two failure modes it catches -- a truncated
    file and a file whose bytes changed -- are both invisible in a listing. Everything is
    recomputed from the bytes; nothing is taken from the manifest and compared to itself.

    :raises Refused: With every problem found, not just the first, so one pass fixes a build.
    """
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise Refused(f"{manifest_path} does not exist; this is not a built mask directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: List[str] = []
    if manifest.get("bands") != list(BANDS):
        problems.append(
            f"manifest bands {manifest.get('bands')} != this build's {list(BANDS)}; the consumer "
            "refuses this at train_core6_arm.py:1207"
        )
    declared_bits = manifest.get("band_bit")
    expected_bits = {str(b): BAND_BIT[b] for b in BANDS}
    if declared_bits != expected_bits:
        problems.append(f"manifest band_bit {declared_bits} != this build's {expected_bits}")
    if not manifest.get("sequence_length"):
        problems.append(
            "manifest declares no sequence_length, so nothing can check that the masks are "
            "scored at the window they were built for"
        )
    if manifest.get("definition_version") != DEFINITION_VERSION:
        problems.append(
            f"manifest definition_version {manifest.get('definition_version')!r} != "
            f"{DEFINITION_VERSION!r}; these are different endpoints and must not be pooled"
        )

    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise Refused(f"{manifest_path} lists no shards" + _joined(problems))

    seq_len = int(manifest.get("sequence_length") or 0)
    totals = {b: 0 for b in BANDS}
    scored_total = 0
    for index, entry in enumerate(entries):
        label = entry.get("mask", f"entry {index}")
        for key in ("s3_key", "shard", "mask", "tokens", "sha256"):
            if key not in entry:
                problems.append(f"{label}: manifest entry has no {key!r}, which the consumer reads")
        if "mask" not in entry or "tokens" not in entry or "sha256" not in entry:
            continue

        # A short digest makes the consumer's `digest[:len(sha256)]` comparison weaker in
        # proportion, and an EMPTY one makes it pass for any bytes at all.
        digest_declared = str(entry["sha256"])
        if len(digest_declared) != 64:
            problems.append(
                f"{label}: sha256 is {len(digest_declared)} character(s), not 64. The consumer "
                "compares only that many, so a short digest weakens the check and an empty one "
                "disables it"
            )

        mask_path = directory / entry["mask"]
        if not mask_path.exists():
            problems.append(f"{label}: mask file is missing")
            continue
        size = mask_path.stat().st_size
        if size != int(entry["tokens"]):
            problems.append(
                f"{label}: {size:,} mask byte(s) for {int(entry['tokens']):,} declared token(s). "
                "One byte per token is the format; the consumer refuses this as a mask/shard "
                "length mismatch"
            )
            continue
        body = mask_path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if digest_declared and digest[: len(digest_declared)] != digest_declared:
            problems.append(f"{label}: sha256 {digest} != manifest {digest_declared}")

        observed = _count_bits(body)
        stray = sorted(observed["stray_bits"])
        if stray:
            problems.append(
                f"{label}: byte value(s) {stray[:5]} carry bits outside {sorted(BAND_BIT.values())}"
            )
        if observed["multi_bit"]:
            problems.append(
                f"{label}: {observed['multi_bit']:,} byte(s) set more than one band bit, so the "
                "bands do not partition the scored set and the counts double-count"
            )
        declared_counts = entry.get("band_counts")
        if declared_counts is not None:
            recomputed = {str(b): observed["counts"][b] for b in BANDS}
            if {k: int(v) for k, v in declared_counts.items()} != recomputed:
                problems.append(
                    f"{label}: band_counts {declared_counts} != recounted from the bytes "
                    f"{recomputed}"
                )
        if seq_len:
            expected_scored = window_count(int(entry["tokens"]), seq_len) * seq_len
            got_scored = sum(observed["counts"].values())
            if got_scored != expected_scored:
                problems.append(
                    f"{label}: {got_scored:,} labelled position(s) but a {seq_len}-token window "
                    f"scores {expected_scored:,} of {int(entry['tokens']):,}. A labelled position "
                    "the consumer never reads, or a read position with no label, both shift the "
                    "band denominators"
                )
            # THE COUNT ALONE CANNOT SEE EVERY RELABEL, WHICH IS WHY REACHABILITY IS CHECKED TOO.
            # Halving a declared `sequence_length` leaves the labelled COUNT unchanged whenever the
            # shard divides evenly -- 8,193 tokens is 4 windows at 2048 and 8 at 1024, and
            # 4 * 2048 == 8 * 1024 -- and the labelled POSITIONS are unchanged as well, because a
            # window's targets are contiguous (`off + 1 .. off + seq_len`) so every position from 1
            # to `scored` is some window's target at any window size. There is no unlabelled
            # interior position to look for.
            #
            # What DOES change is which bands are reachable. The furthest an antecedent can sit
            # inside a `seq_len` window is `seq_len - 1`, so a band whose smallest distance exceeds
            # that cannot hold a single token -- and a populated unreachable band proves the masks
            # were built at a different window than the manifest declares. Relabel a 2048-built
            # mask set as 1024 and band 4096 (distances 1025 and up) is populated but impossible.
            #
            # PARTIAL BY CONSTRUCTION, AND SAID SO RATHER THAN OVERSOLD: a relabel that both
            # divides evenly AND leaves every populated band reachable is not caught here. It is
            # caught by the mask digest whenever the manifest was not rewritten wholesale, which
            # is the case this can still see.
            unreachable_lower = seq_len  # a gap of `seq_len` or more is unrepresentable
            for band, lower in band_lower_bounds().items():
                if lower >= unreachable_lower and observed["counts"][band] > 0:
                    problems.append(
                        f"{label}: band {band} holds {observed['counts'][band]:,} token(s), but at "
                        f"sequence length {seq_len} the furthest visible antecedent is "
                        f"{seq_len - 1} and band {band} starts at {lower}. These masks were built "
                        "at a different window than the manifest declares, so every band is "
                        "computed against a context the model never saw"
                    )
                    break
            scored_total += expected_scored
        for band in BANDS:
            totals[band] += observed["counts"][band]

    floor = min_band_tokens
    if floor is None:
        floor = int(manifest.get("min_band_tokens") or 0)
    if floor:
        try:
            assert_bands_are_live(totals, min_band_tokens=floor, total_scored=scored_total)
        except Refused as exc:
            problems.append(str(exc))

    if problems:
        raise Refused(f"{directory} did not verify:" + _joined(problems))
    return {
        "manifest": str(manifest_path),
        "shards": len(entries),
        "band_counts": {str(b): totals[b] for b in BANDS},
        "scored": scored_total,
    }


def _joined(problems: Sequence[str]) -> str:
    return "".join(f"\n  - {p}" for p in problems)


def _count_bits(body: bytes) -> Dict[str, Any]:
    """Recount a mask's bands from its bytes, and report anything that is not a single band bit.

    Uses one histogram over 256 values rather than a scan per band, so a 25 MB mask costs one
    pass. Anything outside :data:`BAND_BIT`'s values is reported rather than ignored: a byte with
    a stray bit is a byte the consumer would silently attribute to whichever band that bit
    belongs to, and a byte with two band bits double-counts.
    """
    histogram = [0] * 256
    for value in body:
        histogram[value] += 1
    valid = {BAND_BIT[b]: b for b in BANDS}
    counts = {b: 0 for b in BANDS}
    stray: List[int] = []
    multi = 0
    for value, count in enumerate(histogram):
        if not count or value == 0:
            continue
        if value in valid:
            counts[valid[value]] += count
            continue
        # Not a single recognised bit. Say which of the two problems it is.
        if value & ~sum(BAND_BIT.values()):
            stray.append(value)
        else:
            multi += count
    return {"counts": counts, "stray_bits": stray, "multi_bit": multi}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> int:
    """Resolve the corpus, label every shard, refuse anything unusable, write the artifacts."""
    assert_host_can_write_native_masks()
    assert_sequence_length_is_representable(args.sequence_length)

    consumer_checked: Optional[str] = None
    if args.no_consumer_check:
        print(
            "WARNING: --no-consumer-check: the band bit layout was NOT compared against the code "
            "that reads these masks. The manifest records this as null.",
            file=sys.stderr,
        )
    else:
        consumer_path = Path(args.consumer) if args.consumer else REPO_ROOT / CONSUMER_RELATIVE_PATH
        assert_bands_match_consumer(consumer_path)
        consumer_checked = source_digest(consumer_path)

    if args.shard_list:
        corpus = corpus_from_shard_list(Path(args.shard_list))
    else:
        if not args.dataset_id:
            raise Refused("pass --dataset-id/--dataset-version, or --shard-list")
        corpus = corpus_from_reader(args.dataset_id, args.dataset_version)
    corpus.assert_readable_the_way_the_consumer_reads_it()

    # SORTED BY KEY, ALWAYS. The manifest order decides which rank reads which shard, so it has
    # to be a property of the corpus rather than of whatever order a listing came back in.
    shards = sorted(corpus.shards, key=lambda s: s.s3_key)
    if args.limit_shards:
        # A deliberately visible knob: a partial build cannot clear the per-band floor at the
        # default, so a subset build has to lower it on purpose and the manifest records both.
        shards = shards[: args.limit_shards]
        print(
            f"NOTE: --limit-shards {args.limit_shards} builds a SUBSET. Its band counts and "
            "realized mass are not the full held-out set's, and a mask set built this way must "
            "not be used for an arm comparison against one built from the whole corpus.",
            file=sys.stderr,
        )
    if not shards:
        raise Refused("no shard to build after resolution")

    out_dir = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_shards"
    specs = list(enumerate(shards))
    started = time.monotonic()

    def report(done: BuiltShard, completed: int) -> None:
        """THE ESTIMATE BECOMES A MEASUREMENT AFTER THE FIRST SHARD.

        The module docstring's runtime band is derived from a rate this laptop measured, which is
        not a promise about the Batch host. This is the real rate, printed early enough that an
        operator can kill a build that is an order of magnitude off rather than finding out at the
        job timeout. On BOTH paths -- ``--jobs 8`` is the recommended mode and a silent recommended
        mode is how a wedged build looks identical to a slow one.
        """
        rate = done.tokens / done.seconds if done.seconds > 0 else 0.0
        print(
            f"[{completed}/{len(specs)}] {done.shard}: {done.tokens:,} tokens in "
            f"{done.seconds:.1f}s ({rate / 1e6:.2f}M tok/s), bands "
            + json.dumps({str(b): done.counts[b] for b in BANDS}),
            file=sys.stderr,
            flush=True,
        )

    builds: List[BuiltShard] = []
    if args.jobs > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # `fork` EXPLICITLY, NOT THE PLATFORM DEFAULT. Under `spawn` the child re-imports the
        # parent's main module to unpickle `build_one`, which fails outright whenever this file was
        # loaded by anything other than a plain import -- a test harness, a wrapper script -- and
        # fails as a BrokenProcessPool rather than as a readable error. The work forked here is
        # pure-integer, single-threaded, holds no lock and touches no handle the parent is using,
        # which is the case where fork is safe. Falls back to the default where fork does not
        # exist, so `--jobs` degrades rather than breaking.
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork") if "fork" in methods else None
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
            futures = {
                pool.submit(
                    build_one,
                    spec,
                    index=index,
                    out_dir=out_dir,
                    cache_dir=cache_dir,
                    seq_len=args.sequence_length,
                    ngram=args.ngram,
                    shuffle_seed=args.shuffle_labels,
                ): index
                for index, spec in specs
            }
            # `as_completed` rather than iteration over the dict, so a slow first shard does not
            # hide the progress of seven finished ones and the first measured rate arrives as soon
            # as any shard lands. This is a LATENCY-only choice -- the output is identical either
            # way because `builds` is re-sorted by index below -- so no test asserts it. A worker's
            # exception still propagates from `.result()` on both forms.
            for future in as_completed(futures):
                done = future.result()
                builds.append(done)
                report(done, len(builds))
    else:
        for index, spec in specs:
            done = build_one(
                spec,
                index=index,
                out_dir=out_dir,
                cache_dir=cache_dir,
                seq_len=args.sequence_length,
                ngram=args.ngram,
                shuffle_seed=args.shuffle_labels,
            )
            builds.append(done)
            report(done, len(builds))

    # Order is by index, never by completion, so --jobs cannot change a byte of the output.
    builds.sort(key=lambda b: b.index)

    totals = {b: sum(bd.counts[b] for bd in builds) for b in BANDS}
    scored = sum(bd.scored for bd in builds)
    with_antecedent = sum(totals[b] for b in POSITIVE_BANDS)

    realized = assert_mass_clears_the_criterion(
        with_antecedent=with_antecedent, total_scored=scored, c_mass=args.c_mass
    )
    assert_bands_are_live(totals, min_band_tokens=args.min_band_tokens, total_scored=scored)

    manifest = manifest_from_builds(
        builds,
        corpus=corpus,
        seq_len=args.sequence_length,
        ngram=args.ngram,
        c_mass=args.c_mass,
        min_band_tokens=args.min_band_tokens,
        shuffle_seed=args.shuffle_labels,
        consumer_checked=consumer_checked,
        builder_sha256=source_digest(Path(__file__).resolve()),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / MANIFEST_NAME, manifest)

    # The volatile half, kept OUT of the manifest so the manifest can be byte-identical across
    # builds. Nothing reads this; it is for whoever has to explain a build later.
    write_json(
        out_dir / BUILD_LOG_NAME,
        {
            "built_at_unix": int(time.time()),
            "seconds": time.monotonic() - started,
            "host": platform.node(),
            "python": sys.version,
            "argv": sys.argv[1:],
            "jobs": args.jobs,
            "per_shard_seconds": {b.shard: b.seconds for b in builds},
            "shard_list": {
                "dataset_id": corpus.dataset_id,
                "dataset_version": corpus.dataset_version,
                "dtype": corpus.dtype,
                "byte_order": corpus.byte_order,
                "header_bytes": corpus.header_bytes,
                "shards": [{"s3_key": s.s3_key} for s in shards],
            },
        },
    )

    # Verify what was just written, from the bytes, before telling anyone it is done.
    verify_build(out_dir, min_band_tokens=args.min_band_tokens)

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "definition_version": DEFINITION_VERSION,
                "dataset": f"{corpus.dataset_id}/{corpus.dataset_version}",
                "sequence_length": args.sequence_length,
                "shards": len(builds),
                "tokens": sum(b.tokens for b in builds),
                "scored": scored,
                "realized_mass": realized,
                "c_mass": args.c_mass,
                "band_counts": {str(b): totals[b] for b in BANDS},
                "verified": True,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def verify(args: argparse.Namespace) -> int:
    print(json.dumps(verify_build(Path(args.directory)), sort_keys=True, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_slice_masks",
        description="Build the frozen gap-band masks the sliced evaluation reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="label every held-out shard and write the mask set")
    b.add_argument("--out", required=True, help="output directory for the masks and the manifest")
    b.add_argument("--dataset-id", default="", help="corpus to resolve the held-out split from")
    b.add_argument("--dataset-version", default="latest")
    b.add_argument(
        "--shard-list",
        default="",
        help="a recorded shard list instead of resolving the catalog; see corpus_from_shard_list",
    )
    b.add_argument(
        "--sequence-length",
        type=int,
        required=True,
        help="the evaluation window. NOT optional and NOT defaulted: the band partition is a "
        "function of it, and masks built at one window and scored at another are wrong in a way "
        "the consumer does not check",
    )
    b.add_argument(
        "--ngram", type=int, default=2, help="antecedent key order; 2 is the shipped one"
    )
    b.add_argument("--c-mass", type=float, default=DEFAULT_C_MASS)
    b.add_argument("--min-band-tokens", type=int, default=DEFAULT_MIN_BAND_TOKENS)
    b.add_argument(
        "--shuffle-labels",
        type=int,
        default=None,
        metavar="SEED",
        help="build the zero-information CONTROL: same band sizes, positions permuted by this "
        "seed. An arm ranking that survives it is not about distance",
    )
    b.add_argument("--jobs", type=int, default=1, help="shards to label in parallel")
    b.add_argument("--limit-shards", type=int, default=0, help="build a subset; see the warning")
    b.add_argument("--cache-dir", default="", help="where downloaded shards are staged")
    b.add_argument("--consumer", default="", help=f"path to {CONSUMER_RELATIVE_PATH}")
    b.add_argument(
        "--no-consumer-check",
        action="store_true",
        help="skip the band-layout comparison and record the skip in the manifest as null",
    )
    b.set_defaults(func=build)

    v = sub.add_parser("verify", help="recheck a built mask directory from its bytes")
    v.add_argument("directory")
    v.set_defaults(func=verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Refused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
