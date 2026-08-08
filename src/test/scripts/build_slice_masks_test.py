"""
Tests for scripts/build_slice_masks.py.

EVERY TEST CALLS THE BUILDER'S OWN FUNCTION on a token array small enough that the right answer
is derivable by hand and written into the test as a literal. None of them re-derives the builder's
formula: a test that recomputes what the code computes passes when the code changes, which is how
this repo shipped guards that could not fire.

EVERY TEST NAMES THE MUTATION THAT BREAKS IT, in its docstring, as an exact source edit. A test
whose failure mode you cannot name is not evidence. The mutations are collected in
``test_the_named_mutations_are_all_reachable`` so the list cannot drift away from the source.

NO TORCH, NO NUMPY, NO REAL DATA, NO S3. Pure integers and a tmp_path. Nothing here loads a model
or reads a corpus, and the module under test imports nothing heavier than the standard library.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_slice_masks",
    str(Path(__file__).resolve().parents[3] / "scripts" / "build_slice_masks.py"),
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("could not load scripts/build_slice_masks.py")
bsm = importlib.util.module_from_spec(_SPEC)
sys.modules["build_slice_masks"] = bsm
_SPEC.loader.exec_module(bsm)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSUMER = REPO_ROOT / ".edullm" / "train_core6_arm.py"

#: Every band bit, by name, so a test can assert a byte's value rather than recompute the mapping.
NO_ANTECEDENT = 1
B32 = 2
B256 = 4
B1024 = 8
B4096 = 16


# ==============================================================================================
# THE CONTRACT AGAINST THE CONSUMER. If these fail, nothing else in this file matters.
# ==============================================================================================


def test_the_band_bits_are_exactly_the_consumers():
    """
    THE MASKS ARE WRITTEN ONCE AND READ AS BYTES, so the bit layout is the whole contract.

    Checks the WHOLE MAPPING, not ``sorted(BAND_BIT)`` -- which is all the consumer itself
    compares at ``train_core6_arm.py:1207``. Two builds that agree on the band NAMES and disagree
    on which bit each owns pass the consumer's own check and mislabel every token in the corpus.

    FAILURE MUTATION: change any value in ``BAND_BIT`` in build_slice_masks.py -- e.g.
    ``{0: 1, 32: 2, 256: 4, 1024: 8, 4096: 16}`` to ``... 4096: 32``. Also fails if someone edits
    the consumer's copy instead, which is the direction that actually matters.
    """
    assert bsm.BAND_BIT == {0: 1, 32: 2, 256: 4, 1024: 8, 4096: 16}
    assert CONSUMER.exists(), (
        "the consumer is an ARTIFACT of this repo, not an optional dependency; a skip here would "
        "leave the bit layout uncompared while the suite read green"
    )
    assert bsm.consumer_band_bit(CONSUMER) == bsm.BAND_BIT
    # And the function that refuses on a mismatch returns rather than raises on a match.
    assert bsm.assert_bands_match_consumer(CONSUMER) == bsm.BAND_BIT


def test_a_disagreeing_consumer_is_refused_rather_than_reported(tmp_path):
    """
    PROVES THE PREVIOUS TEST CAN FAIL, by handing the checker a consumer that disagrees.

    Without this, ``assert_bands_match_consumer`` could be a function that never raises and the
    test above would still pass. This is the mutation, applied as data instead of to the source.

    FAILURE MUTATION: make ``assert_bands_match_consumer`` compare ``sorted(theirs) !=
    sorted(BAND_BIT)`` instead of ``theirs != BAND_BIT`` -- the permuted layout below has the same
    sorted keys, so the weaker check passes it.
    """
    fake = tmp_path / "consumer.py"
    # Same band NAMES, permuted bits: exactly what the consumer's own check cannot see.
    fake.write_text("BAND_BIT = {0: 1, 32: 4, 256: 2, 1024: 8, 4096: 16}\n")
    with pytest.raises(bsm.Refused, match="bit layout"):
        bsm.assert_bands_match_consumer(fake)


def test_a_missing_consumer_is_a_refusal_and_not_a_skip(tmp_path):
    """
    A SKIP COUNTS AS A PASS, and an absent artifact is not an absent dependency.

    ``consumer_band_bit`` on a path that does not exist must refuse. The tempting alternative --
    return None and let the caller carry on -- makes an unperformed check indistinguishable from a
    passed one.

    FAILURE MUTATION: replace the ``if not path.exists(): raise Refused(...)`` block in
    ``consumer_band_bit`` with ``return dict(BAND_BIT)``. Every build then self-certifies.
    """
    with pytest.raises(bsm.Refused, match="not here"):
        bsm.consumer_band_bit(tmp_path / "absent.py")
    # A file that exists but declares no BAND_BIT is the same failure, not a pass.
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1\n")
    with pytest.raises(bsm.Refused, match="no module-level BAND_BIT"):
        bsm.consumer_band_bit(empty)


# ==============================================================================================
# THE BAND RULE. Hand-computed answers, one per band, plus every boundary.
# ==============================================================================================


def test_the_gap_to_band_mapping_is_upper_closed_at_every_boundary():
    """
    THE BOUNDARY SIDE, ONE ASSERTION PER EDGE, AS LITERALS.

    Documented rule: a band name is the UPPER edge of a right-closed interval, so gap 32 is band
    32 and gap 33 is band 256. These pairs are transcribed from the module docstring's table, not
    computed from ``POSITIVE_BANDS``.

    FAILURE MUTATION: change ``if gap <= boundary`` to ``if gap < boundary`` in ``band_of_gap``.
    Every ``(boundary, boundary)`` pair below then lands one band too high. The ``+1`` cases catch
    the opposite mutation (``<=`` to ``<`` on the wrong side, or a ``-1`` on the boundary).
    """
    for gap, expected in (
        (1, 32),
        (31, 32),
        (32, 32),  # exactly 32 -> band 32
        (33, 256),  # one past -> next band
        (255, 256),
        (256, 256),  # exactly 256 -> band 256
        (257, 1024),
        (1023, 1024),
        (1024, 1024),  # exactly 1024 -> band 1024
        (1025, 4096),
        (4095, 4096),
        (4096, 4096),  # exactly 4096 -> band 4096
    ):
        assert bsm.band_of_gap(gap) == expected, f"gap {gap} should be band {expected}"


def test_a_gap_that_no_band_can_hold_is_refused_rather_than_clamped():
    """
    AN OUT-OF-RANGE GAP MUST NOT SILENTLY BECOME THE TOP BAND.

    Clamping is the tempting fix and it redefines band 4096 from "1024 < d <= 4096" to
    "1024 < d", which changes the endpoint without changing the manifest.

    FAILURE MUTATION: replace ``band_of_gap``'s final ``raise ValueError`` with
    ``return POSITIVE_BANDS[-1]``. Also proves ``gap < 1`` is rejected: a token cannot be its own
    antecedent, and gap 0 would mean the same position matched itself.
    """
    with pytest.raises(ValueError, match="exceeds the top band"):
        bsm.band_of_gap(4097)
    with pytest.raises(ValueError, match="not a distance"):
        bsm.band_of_gap(0)


def _tokens_with_one_antecedent_at(gap: int, seq_len: int, *, unique_base: int = 1000) -> List[int]:
    """A single window in which EXACTLY ONE position has a bigram antecedent, ``gap`` back.

    Construction, so the expected answer is readable rather than derived: every token is a
    distinct id, so no bigram can repeat by accident; then the bigram ending at position ``q`` is
    copied to end at ``q + gap``. That makes exactly one bigram occur twice, at distance ``gap``.

    ``q`` sits at ``seq_len - gap`` so that ``q + gap == seq_len``, the window's last target --
    which also pins that the last target IS scored.
    """
    n = seq_len + 1
    tokens = [unique_base + i for i in range(n)]
    q = seq_len - gap
    assert 1 <= q, f"gap {gap} does not fit in a {seq_len}-token window"
    # The repeat: (tok[q-1], tok[q]) reappears as (tok[q+gap-1], tok[q+gap]).
    tokens[q + gap - 1] = tokens[q - 1]
    tokens[q + gap] = tokens[q]
    return tokens


@pytest.mark.parametrize(
    "gap,expected_band,expected_bit",
    [
        (1, 32, B32),
        (32, 32, B32),
        (33, 256, B256),
        (256, 256, B256),
        (257, 1024, B1024),
        (1024, 1024, B1024),
        (1025, 4096, B4096),
        (4095, 4096, B4096),
    ],
)
def test_one_antecedent_at_each_distance_lands_in_the_documented_band(
    gap, expected_band, expected_bit
):
    """
    ONE POSITION, ONE KNOWN DISTANCE, ONE EXPECTED BYTE -- for every band and both sides of
    every boundary.

    The construction puts exactly one repeated bigram in the window at distance ``gap``, so the
    correct labelling is: one position carries ``expected_bit``, every other scored position
    carries band 0's bit, and the marked position is ``seq_len`` (the window's last target).

    FAILURE MUTATION: any off-by-one in the distance. Change ``band_of_gap(q - previous)`` to
    ``band_of_gap(q - previous + 1)`` in ``_label_windows`` and the ``gap == 32``, ``256``,
    ``1024`` and ``4095`` cases move a band; change it to ``q - previous - 1`` and the
    ``gap == 1`` case raises "not a distance" while ``33``, ``257`` and ``1025`` move down.
    """
    seq_len = 4096
    tokens = _tokens_with_one_antecedent_at(gap, seq_len)
    out = bsm.assign_bands(tokens, seq_len)

    assert out.windows == 1
    assert out.scored == seq_len
    assert out.counts[expected_band] == 1, f"gap {gap} should put one token in band {expected_band}"
    assert out.counts[0] == seq_len - 1, "every other scored position has no antecedent"
    for other in bsm.BANDS:
        if other not in (0, expected_band):
            assert out.counts[other] == 0, f"band {other} should be empty"

    # The BYTE, at the position whose distance we constructed.
    assert out.mask[seq_len] == expected_bit
    # Position 0 is never a target and is never read by the consumer, so it stays unlabelled.
    assert out.mask[0] == 0


def test_a_token_with_no_antecedent_lands_in_band_zero():
    """
    BAND 0 IS "NO VISIBLE ANTECEDENT", and a window of all-distinct ids has no repeated bigram at
    all -- so every scored position must carry band 0's bit and nothing else.

    FAILURE MUTATION: delete the ``mask[q] = zero_bit; counts[0] += 1`` pair in the
    ``previous is None`` branch of ``_label_windows``. The mask bytes go to ``0x00``, the counts
    stop summing to ``scored``, and the partition check in that same function refuses -- so this
    fails as a ``Refused`` rather than as a wrong number, which is the correct behaviour.
    """
    seq_len = 64
    tokens = list(range(7000, 7000 + seq_len + 1))  # all distinct, so no bigram repeats
    out = bsm.assign_bands(tokens, seq_len)
    assert out.counts[0] == seq_len
    assert out.with_antecedent == 0
    assert set(out.mask[1 : seq_len + 1]) == {NO_ANTECEDENT}


def test_the_bands_partition_the_scored_set():
    """
    THE INVARIANT THE ENDPOINT'S DENOMINATORS REST ON: exactly one bit per scored position, so
    the band counts sum to the scored count -- the same equality the consumer's ``agg_n`` and its
    summed ``band_n`` have to satisfy.

    Uses a token stream with MANY repeats at mixed distances, so several bands are populated at
    once and the sum is a real check rather than a tautology over one band.

    FAILURE MUTATION: change ``mask[q] = bits[band]`` to ``mask[q] |= bits[band] | zero_bit``.
    Every position then carries two bits, the consumer's per-band ``(flat & bit) != 0`` counts it
    twice, and ``sum(counts) != scored``. ``_count_bits`` reports it as ``multi_bit``.
    """
    seq_len = 512
    # A small alphabet, so bigrams repeat constantly and at a spread of distances.
    tokens = [(i * 37) % 23 for i in range(seq_len * 3 + 1)]
    out = bsm.assign_bands(tokens, seq_len)
    assert out.windows == 3
    assert out.scored == 3 * seq_len
    assert sum(out.counts.values()) == out.scored
    assert out.with_antecedent > 0, "the fixture must populate at least one distance band"
    # Every labelled byte is exactly one recognised bit, checked through the verifier's own
    # counter rather than by a second implementation here.
    observed = bsm._count_bits(out.mask)
    assert observed["stray_bits"] == []
    assert observed["multi_bit"] == 0
    assert {b: observed["counts"][b] for b in bsm.BANDS} == out.counts


def test_an_antecedent_outside_the_window_is_invisible():
    """
    THE VISIBILITY RULE. A bigram that recurs across a window boundary must be band 0, because
    the model scoring that window never saw the earlier copy -- windows are non-overlapping and
    start at multiples of ``seq_len`` (``_shard_windows``, ``train_core6_arm.py:1141-1149``).

    Construction: two windows of otherwise-distinct ids, with window 1's bigram at local offset 5
    copied from window 0's at local offset 5. The distance is exactly ``seq_len``, which WOULD be
    a real band if the antecedent were visible -- so a builder that ignored the boundary would
    label it band 4096 here rather than band 0.

    FAILURE MUTATION: hoist ``last: Dict[int, int] = {}`` out of the ``for off, block in blocks``
    loop in ``_label_windows`` (i.e. make the dictionary shard-global). The marked position then
    reports band 4096 and ``counts[4096] == 1``, which is a position "recalling" something the
    model could not see.
    """
    seq_len = 1024
    n = 2 * seq_len + 1
    tokens = [5000 + i for i in range(n)]
    # window 0 covers 0..1024, window 1 covers 1024..2048.
    src = 5  # local offset inside window 0
    dst = seq_len + 5  # the same local offset inside window 1
    tokens[dst - 1] = tokens[src - 1]
    tokens[dst] = tokens[src]

    out = bsm.assign_bands(tokens, seq_len)
    assert out.windows == 2
    assert out.mask[dst] == NO_ANTECEDENT, "a cross-window repeat is not a visible antecedent"
    assert out.counts[4096] == 0, "distance seq_len would be band 4096 if it were visible"
    assert out.counts[0] == out.scored


def test_the_key_is_a_bigram_and_not_a_bare_token_repeat():
    """
    THE KEY IS THE BIGRAM THE POSITION COMPLETES, WHICH IS THE DEFINITION'S ACTUAL CLAIM.

    A repeated TOKEN whose preceding token differs is NOT a bigram antecedent and must be band 0.
    A unigram rule would label it, which is a materially different (and much looser) endpoint.

    FAILURE MUTATION: change ``for back in range(ngram - 1, -1, -1)`` to ``for back in range(1)``
    in ``_label_windows``, making the key the bare token. The repeated token below then reports
    band 32.
    """
    seq_len = 32
    tokens = [9000 + i for i in range(seq_len + 1)]
    # tok[20] repeats tok[10]'s VALUE, but tok[19] != tok[9], so the bigram does not repeat.
    tokens[20] = tokens[10]
    out = bsm.assign_bands(tokens, seq_len)
    assert out.mask[20] == NO_ANTECEDENT
    assert out.counts[0] == seq_len

    # And with the preceding token made to match too, it DOES become a bigram antecedent at
    # distance 10 -- which proves the assertion above is about the bigram and not about nothing.
    tokens[19] = tokens[9]
    out2 = bsm.assign_bands(tokens, seq_len)
    assert out2.mask[20] == B32
    assert out2.counts[32] == 1


def test_the_antecedent_is_the_most_recent_one():
    """
    "MOST RECENT EARLIER OCCURRENCE" -- so three copies of a bigram label the third by its
    distance to the SECOND, not to the first.

    FAILURE MUTATION: change ``last[key] = q`` to ``last.setdefault(key, q)`` in
    ``_label_windows``. The third copy below then measures 200 back instead of 100 and reports
    band 256 instead of band 128-in-256... which is why the distances are chosen to straddle a
    boundary: 100 is band 256 and 200 is also band 256, so THAT pair would not catch it. The
    fixture uses 20 and 300 instead, which land in band 32 and band 1024 respectively.
    """
    seq_len = 1024
    tokens = [3000 + i for i in range(seq_len + 1)]
    first = 100
    second = first + 300  # 400
    third = second + 20  # 420
    for target in (second, third):
        tokens[target - 1] = tokens[first - 1]
        tokens[target] = tokens[first]

    out = bsm.assign_bands(tokens, seq_len)
    assert out.mask[second] == B1024, "300 back is band 1024"
    assert out.mask[third] == B32, "20 back (to the SECOND copy) is band 32, not 320 to the first"
    assert out.counts[32] == 1
    assert out.counts[1024] == 1


def test_the_window_count_drops_the_tail_the_consumer_never_reads():
    """
    THE OFF-BY-ONE THAT ONLY SHOWS UP ON A CLEAN DIVISION, matching the consumer's own test at
    ``src/test/edullm_train_core6_arm_test.py``: a window needs ``seq_len`` inputs AND one more
    token to be its last target, so 96 tokens at ``seq_len`` 32 is 2 windows, not 3.

    The pairs are transcribed from that test so the two files agree by shared literal rather than
    by shared formula. Positions past the last complete window get ``0x00`` -- they are never read
    by the consumer, and labelling them would inflate nothing but would falsify the accounting.

    FAILURE MUTATION: change ``window_count``'s ``(n_tokens - 1) // seq_len`` to
    ``n_tokens // seq_len``. The ``96 -> 2`` and ``64 -> 1`` cases fail, and the builder would
    read one token past the end of a real shard.
    """
    for n_tokens, expected in ((96, 2), (97, 3), (95, 2), (32, 0), (33, 1), (64, 1), (65, 2)):
        assert bsm.window_count(n_tokens, 32) == expected, f"{n_tokens} tokens"

    # And the assignment agrees, including the unlabelled tail.
    seq_len, n_tokens = 32, 96 + 5
    tokens = list(range(400, 400 + n_tokens))
    out = bsm.assign_bands(tokens, seq_len)
    assert out.windows == 3 and out.scored == 96
    assert out.unscored == n_tokens - 96
    assert set(out.mask[97:]) == {0}, "the tail past the last full window is never labelled"
    assert out.mask[96] != 0, "the last target of the last full window IS labelled"


def test_a_sequence_length_past_the_top_band_is_refused():
    """
    A WINDOW LONGER THAN THE TOP BAND + 1 PRODUCES DISTANCES WITH NO BIT TO OCCUPY, and the fix
    must be a refusal rather than a clamp into band 4096.

    ``seq_len`` 4097 allows gap 4096 (fine) but a 8192-token window allows gap 8191 (not fine).
    The boundary case ``seq_len == 4097`` is exactly representable and must be ACCEPTED, which is
    what stops the guard being over-eager.

    FAILURE MUTATION: change ``if seq_len - 1 > top`` to ``if seq_len > top`` in
    ``assert_sequence_length_is_representable``. ``seq_len == 4097`` then refuses even though
    every gap it can produce has a band -- a guard that fires where it must not.
    """
    bsm.assert_sequence_length_is_representable(4096)
    bsm.assert_sequence_length_is_representable(4097)  # max gap 4096, exactly representable
    with pytest.raises(bsm.Refused, match="past the top band"):
        bsm.assert_sequence_length_is_representable(4098)
    with pytest.raises(bsm.Refused, match="no scorable target"):
        bsm.assert_sequence_length_is_representable(1)


def test_an_ngram_of_one_is_refused():
    """
    ``--ngram 1`` IS A UNIGRAM RULE, which is a different endpoint and not this one.

    FAILURE MUTATION: change ``if ngram < 2`` to ``if ngram < 1`` in ``_label_windows``. A
    unigram build then succeeds and ships under this definition's version string.
    """
    with pytest.raises(bsm.Refused, match="not a key"):
        bsm.assign_bands(list(range(100)), 32, ngram=1)


def test_a_trigram_key_is_stricter_than_a_bigram_key():
    """
    THE FALSIFICATION CONTROL FROM THE DOCSTRING, as a property: a higher-order key can only
    label FEWER positions, because a repeated trigram is also a repeated bigram.

    The fixture has a bigram that repeats with a DIFFERENT token before it, so the bigram rule
    labels it and the trigram rule must not. That makes the inequality strict here rather than
    vacuously equal.

    FAILURE MUTATION: change ``key = (key << 32) | ...`` to ``key = key | ...`` in
    ``_label_windows``. Without the shift, tokens commute into the key and a trigram key collides
    with permutations of itself, so the trigram build labels MORE positions than the bigram build
    and this inequality inverts.
    """
    seq_len = 256
    tokens = [200 + i for i in range(seq_len + 1)]
    src, dst = 40, 140
    tokens[dst - 1] = tokens[src - 1]
    tokens[dst] = tokens[src]
    # tokens[dst-2] is left distinct from tokens[src-2], so the TRIGRAM does not repeat.

    bigram = bsm.assign_bands(tokens, seq_len, ngram=2)
    trigram = bsm.assign_bands(tokens, seq_len, ngram=3)
    assert bigram.with_antecedent == 1
    assert trigram.with_antecedent == 0
    assert trigram.scored == bigram.scored, "the scored set does not depend on the key order"


# ==============================================================================================
# DETERMINISM AND PROVENANCE.
# ==============================================================================================


def _shard_list(tmp_path: Path, shards: Sequence[Sequence[int]]) -> Path:
    """Write raw u32le shards and the ``--shard-list`` JSON that points at them.

    Raw headerless bytes, because that is what the corpus is and what the consumer memmaps -- a
    real ``.npy`` would shift every read by its 64-byte header and still train happily.
    """
    from array import array

    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, tokens in enumerate(shards):
        # Two different "topics", so the keys collide on basename the way the real corpus does.
        key = f"reservoir/topic{index}/val-00212.u32le.bin"
        path = tmp_path / f"shard{index}.u32le.bin"
        buf = array("I", list(tokens))
        path.write_bytes(buf.tobytes())
        entries.append({"s3_key": key, "local": str(path)})
    listing = tmp_path / "shards.json"
    listing.write_text(
        json.dumps(
            {
                "dataset_id": "reservoir-dolma2",
                "dataset_version": "v1",
                "dtype": "uint32",
                "byte_order": "little",
                "header_bytes": 0,
                "shards": entries,
            }
        )
    )
    return listing


#: The window the end-to-end fixtures build at. NOT 64, and that is a correctness constraint on
#: the FIXTURE rather than a size preference: the furthest gap a window can hold is
#: ``seq_len - 1``, so at 64 the bands 256/1024/4096 are UNREACHABLE and ``assert_bands_are_live``
#: refuses every build. A fixture in a regime where a guard cannot be satisfied tests the regime,
#: not the code.
FIXTURE_SEQ_LEN = 2048

#: One planted distance per positive band, chosen inside ``(lower, upper]`` and inside the window.
#: These are what make every band reachable in the fixture regime.
FIXTURE_GAPS = {32: 10, 256: 100, 1024: 500, 4096: 2000}


def _all_bands_populated(seq_len: int, *, base: int, extra_windows: int = 0) -> List[int]:
    """Tokens whose labelling reaches all five bands, by construction rather than by luck.

    ONE PLANT PER WINDOW, which is not tidiness: an antecedent is only visible inside its own
    window, so two plants sharing a window can have one plant's copy serve as the other's
    antecedent and the planted distance is then not the measured one. One window each makes the
    expected count exactly 1 per positive band, known by hand.

    Ids are otherwise distinct, so nothing repeats by accident and band 0 is populated too.
    """
    gaps = sorted(FIXTURE_GAPS.values())
    n_windows = len(gaps) + extra_windows
    n = seq_len * n_windows + 1
    tokens = [base + i for i in range(n)]
    for window, gap in enumerate(gaps):
        off = window * seq_len
        src = off + 4
        dst = src + gap
        assert dst <= off + seq_len, f"gap {gap} must fit inside one window of {seq_len}"
        tokens[dst - 1] = tokens[src - 1]
        tokens[dst] = tokens[src]
    return tokens


def _small_build(tmp_path: Path, out_name: str, *, extra: Optional[List[str]] = None) -> Path:
    """A complete two-shard build at :data:`FIXTURE_SEQ_LEN`, with the floors lowered to match.

    ``--min-band-tokens 1`` and ``--c-mass 0.0`` are declared on purpose: a toy build cannot clear
    the production floors (25M tokens per band), and lowering them is exactly the knowing act the
    builder records in the manifest. A test that quietly built with the production floors would be
    testing the floors. The floor is lowered to 1 and not to 0, so the liveness check still has to
    pass -- which is why the fixture has to reach every band.
    """
    seq_len = FIXTURE_SEQ_LEN
    # Different shard lengths, so a bug that assumed uniform shards shows up.
    a = _all_bands_populated(seq_len, base=100_000, extra_windows=1)
    b = _all_bands_populated(seq_len, base=500_000)
    listing = _shard_list(tmp_path / out_name, [a, b])
    out = tmp_path / out_name / "masks"
    argv = [
        "build",
        "--out",
        str(out),
        "--shard-list",
        str(listing),
        "--sequence-length",
        str(seq_len),
        "--min-band-tokens",
        "1",
        "--c-mass",
        "0.0",
        "--consumer",
        str(CONSUMER),
    ]
    argv.extend(extra or [])
    assert bsm.main(argv) == 0
    return out


def test_two_builds_from_the_same_corpus_are_byte_identical(tmp_path):
    """
    THE FROZEN-ARTIFACT PROPERTY. Masks that silently changed between runs would make every
    arm-to-arm comparison invalid without anyone noticing, so this compares BYTES: every mask file
    and the manifest itself, across two independent invocations into two different directories.

    The manifest is the harder half -- a timestamp, a hostname or an absolute path in it would
    make every build differ while every mask stayed identical.

    FAILURE MUTATION: add ``"built_at_unix": int(time.time())`` (or ``platform.node()``, or
    ``str(out_dir)``) to the dict returned by ``manifest_from_builds``. The mask digests still
    match and the manifest comparison fails.

    THE ABSENCE OF A VOLATILE FIELD IS ALSO ASSERTED DIRECTLY, because comparing two builds run
    seconds apart is a WEAK test of it: a whole-second timestamp is usually identical across two
    builds in the same test, so the comparison passes and the claim goes unchecked. The explicit
    scan below is what makes the claim evidence.
    """
    first = _small_build(tmp_path, "run1")
    second = _small_build(tmp_path, "run2")

    left = sorted(p.name for p in first.glob("*" + bsm.MASK_SUFFIX))
    right = sorted(p.name for p in second.glob("*" + bsm.MASK_SUFFIX))
    assert left == right and left, "both builds must produce the same, non-empty mask set"
    for name in left:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    assert (first / bsm.MANIFEST_NAME).read_bytes() == (second / bsm.MANIFEST_NAME).read_bytes()

    # No volatile field ANYWHERE in the manifest, checked by walking it rather than by hoping the
    # two builds straddled a second boundary. The two output directories differ, so an absolute
    # path would show up as a substring of one of them.
    text = (first / bsm.MANIFEST_NAME).read_text()
    for forbidden in ("built_at_unix", "seconds", "host", "argv", str(first), "/private/var"):
        assert forbidden not in text, f"{forbidden!r} in the manifest makes every build differ"

    # The volatile half lives in a SEPARATE file the consumer never opens, and it is expected to
    # differ. Asserted present, so the provenance was not simply dropped instead of relocated.
    log = json.loads((first / bsm.BUILD_LOG_NAME).read_text())
    for expected in ("built_at_unix", "host", "argv", "shard_list"):
        assert expected in log, f"{expected} must still be recorded, in the sidecar"


def test_write_json_canonicalises_key_order(tmp_path):
    """
    ``write_json`` MUST SORT, and running a build twice cannot show it: dict insertion order is
    already deterministic for one code path, so two builds match with or without ``sort_keys``.
    Mutating it therefore leaves ``test_two_builds_...`` green -- which is why that test's
    docstring does NOT claim it, and why this one exists instead.

    What sorting actually buys: the bytes stop depending on the order
    ``manifest_from_builds`` happens to insert its keys, so reordering that dict literal -- a
    change no reviewer would think twice about -- cannot silently change every published digest.
    Asserted as a property, on two dicts that differ only in insertion order.

    FAILURE MUTATION: change ``sort_keys=True`` to ``sort_keys=False`` in ``write_json``. The two
    files below then differ, and the mutation is caught here rather than nowhere.
    """
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    bsm.write_json(a, {"zebra": 1, "apple": {"n": 2, "m": 3}})
    bsm.write_json(b, {"apple": {"m": 3, "n": 2}, "zebra": 1})
    assert a.read_bytes() == b.read_bytes(), "insertion order must not reach the bytes"
    assert a.read_text().endswith("\n"), "a trailing newline keeps the file diffable"


def test_parallel_and_sequential_builds_agree_byte_for_byte(tmp_path):
    """
    ``--jobs`` MUST NOT CHANGE A BYTE. Completion order under a process pool is not the shard
    order, and the manifest's order decides which rank reads which shard
    (``index % world_size``, ``train_core6_arm.py:1214``).

    THIS ALSO EXERCISES THE PROCESS POOL AT ALL, which is not a given: under the ``spawn`` start
    method the worker re-imports the parent's main module to unpickle ``build_one``, and this file
    is loaded through ``importlib`` rather than as ``__main__``, so a default-context pool dies
    with a ``BrokenProcessPool`` and no readable reason. That failure was found here.

    FAILURE MUTATION: delete ``builds.sort(key=lambda b: b.index)`` in ``build``. Under ``--jobs
    2`` the manifest's shard order becomes completion order, the masks stay identical, and this
    manifest comparison fails. Changing ``mp_context=context`` back to the platform default breaks
    it on any host whose default is ``spawn``.
    """
    serial = _small_build(tmp_path, "serial")
    parallel = _small_build(tmp_path, "parallel", extra=["--jobs", "2"])
    assert (serial / bsm.MANIFEST_NAME).read_bytes() == (parallel / bsm.MANIFEST_NAME).read_bytes()
    for entry in json.loads((serial / bsm.MANIFEST_NAME).read_text())["shards"]:
        assert (serial / entry["mask"]).read_bytes() == (parallel / entry["mask"]).read_bytes()


def test_both_build_paths_report_a_measured_rate(capsys, tmp_path):
    """
    A SILENT BUILD LOOKS THE SAME WEDGED AS SLOW, and ``--jobs 8`` is the RECOMMENDED mode -- so
    the progress line has to exist on the parallel path too, not only on the sequential one it was
    first written for. The line is what turns the docstring's estimated runtime into a measurement
    inside the first minute of a real 39-shard build.

    FAILURE MUTATION: delete the ``report(done, len(builds))`` call inside the ``as_completed``
    loop in ``build`` (the parallel branch). The sequential half of this test still passes, which
    is exactly why both halves are asserted.
    """
    for name, extra in (("seq", []), ("par", ["--jobs", "2"])):
        _small_build(tmp_path, name, extra=extra)
        captured = capsys.readouterr()
        assert "tok/s" in captured.err, f"{name}: no measured rate was printed"
        # One line per shard, so an operator can see which shard is slow rather than only a total.
        assert captured.err.count("tok/s") == 2, f"{name}: expected one line per shard"


def test_the_manifest_carries_what_the_consumer_reads_and_what_it_should(tmp_path):
    """
    THE FIVE FIELDS ``fetch_slice_inputs`` READS, plus the provenance that makes the build
    reproducible and the ``sequence_length`` the consumer ought to check and does not.

    The digest is asserted to be the FULL 64 characters, because the consumer compares
    ``digest[:len(entry["sha256"])]`` -- a short digest weakens the check in proportion and an
    empty one passes for any bytes at all.

    FAILURE MUTATION: change ``"sha256": build.mask_sha256`` to
    ``"sha256": build.mask_sha256[:8]`` in ``manifest_from_builds``. The length assertion fails
    here, and ``verify_build`` refuses it too. Dropping ``"sequence_length"`` fails the same way.
    """
    out = _small_build(tmp_path, "fields")
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())

    # The one field the consumer compares by equality, at train_core6_arm.py:1207.
    assert manifest["bands"] == [0, 32, 256, 1024, 4096]
    assert manifest["band_bit"] == {"0": 1, "32": 2, "256": 4, "1024": 8, "4096": 16}
    assert manifest["sequence_length"] == FIXTURE_SEQ_LEN
    # THE FIXTURE REACHES EVERY BAND, which is what makes the liveness floor a live check here
    # rather than a guard that is unsatisfiable in this regime and therefore untested.
    assert all(int(v) > 0 for v in manifest["totals"]["band_counts"].values()), manifest["totals"][
        "band_counts"
    ]
    assert manifest["definition_version"] == bsm.DEFINITION_VERSION
    assert manifest["dataset_id"] == "reservoir-dolma2"
    assert manifest["dataset_version"] == "v1"
    assert 0.0 <= manifest["realized_mass"] <= 1.0, "logged as 100 * value with %.3f%%"
    assert manifest["is_shuffled_control"] is False
    assert manifest["shuffle_labels_seed"] is None
    assert manifest["consumer_band_bit_sha256"], "the band-layout check must be recorded"
    assert manifest["builder_sha256"]

    for entry in manifest["shards"]:
        for key in ("s3_key", "shard", "mask", "tokens", "sha256"):
            assert key in entry, f"the consumer reads {key!r}"
        assert len(entry["sha256"]) == 64, "a short digest weakens the consumer's check"
        assert entry["mask"].endswith(bsm.MASK_SUFFIX)
        # One byte per token, which is the whole format.
        assert (out / entry["mask"]).stat().st_size == entry["tokens"]
        assert hashlib.sha256((out / entry["mask"]).read_bytes()).hexdigest() == entry["sha256"]
        # Provenance the consumer ignores today but which makes its own docstring's claim true.
        assert len(entry["shard_sha256"]) == 64
        assert sum(int(v) for v in entry["band_counts"].values()) == entry["scored"]

    totals = manifest["totals"]
    assert totals["shards"] == len(manifest["shards"])
    assert totals["scored"] == sum(e["scored"] for e in manifest["shards"])
    assert totals["tokens"] == sum(e["tokens"] for e in manifest["shards"])


def test_two_shards_with_the_same_basename_get_distinct_mask_names(tmp_path):
    """
    THE DOCUMENTED TRAP: ``val-00212.u32le.bin`` exists under 24 topic directories, and the mask
    prefix is FLAT (``{base}/{entry['mask']}``, ``train_core6_arm.py:1228``). Two shards that
    produced the same mask name would have one overwrite the other at upload and a third of the
    val set would be scored against another topic's labels -- with a completely plausible CE.

    The fixture's two shards deliberately share a basename.

    FAILURE MUTATION: change ``shard_name_from_key`` to ``return parts[-1]`` (the basename) or to
    ``"__".join(parts[-2:])`` (the documented two-component convention). Both collide on this
    fixture, and ``assert_names_are_unique`` refuses -- so the build fails rather than silently
    overwriting.
    """
    out = _small_build(tmp_path, "collide")
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())
    keys = [e["s3_key"] for e in manifest["shards"]]
    basenames = [k.rsplit("/", 1)[-1] for k in keys]
    assert len(set(basenames)) < len(basenames), "the fixture must contain a repeated basename"
    names = [e["mask"] for e in manifest["shards"]]
    assert len(set(names)) == len(names), "mask names must be unique in a flat prefix"

    # And the collision-prone naming is refused rather than tolerated.
    with pytest.raises(bsm.Refused, match="duplicate mask"):
        bsm.assert_names_are_unique(
            [
                {"mask": "m.mask.u8", "shard": "a", "s3_key": "x/a"},
                {"mask": "m.mask.u8", "shard": "b", "s3_key": "x/b"},
            ]
        )


def test_the_shard_order_is_the_key_order(tmp_path):
    """
    THE MANIFEST ORDER IS LOAD-BEARING: the consumer assigns shard ``i`` to rank
    ``i % world_size``. A list in listing order rather than key order gives every rank a different
    subset -- same union, same numbers, different manifest bytes.

    FAILURE MUTATION: change ``sorted(corpus.shards, key=lambda s: s.s3_key)`` in ``build`` to
    ``list(corpus.shards)``. The fixture below is written in reverse key order, so the manifest
    comes out unsorted and this fails.
    """
    from array import array

    seq_len = FIXTURE_SEQ_LEN
    entries = []
    # Deliberately DESCENDING keys, so listing order != sorted order.
    for index, key in enumerate(["reservoir/z/val-1.u32le.bin", "reservoir/a/val-2.u32le.bin"]):
        path = tmp_path / f"s{index}.u32le.bin"
        tokens = _all_bands_populated(seq_len, base=10_000 + 100_000 * index)
        path.write_bytes(array("I", tokens).tobytes())
        entries.append({"s3_key": key, "local": str(path)})
    listing = tmp_path / "shards.json"
    listing.write_text(
        json.dumps(
            {
                "dataset_id": "d",
                "dataset_version": "v1",
                "dtype": "uint32",
                "byte_order": "little",
                "header_bytes": 0,
                "shards": entries,
            }
        )
    )
    out = tmp_path / "ordered"
    assert (
        bsm.main(
            [
                "build",
                "--out",
                str(out),
                "--shard-list",
                str(listing),
                "--sequence-length",
                str(seq_len),
                "--min-band-tokens",
                "1",
                "--c-mass",
                "0.0",
                "--consumer",
                str(CONSUMER),
            ]
        )
        == 0
    )
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())
    keys = [e["s3_key"] for e in manifest["shards"]]
    assert keys == sorted(keys), "the manifest order must be a property of the corpus"


def test_the_shuffled_control_keeps_the_band_sizes_and_is_reproducible(tmp_path):
    """
    THE ZERO-INFORMATION CONTROL. Same per-band counts, positions permuted -- so an arm ranking
    that survives it is not about distance. And it must be reproducible from the declared seed,
    or the control is not a control.

    FAILURE MUTATION: change ``rng = random.Random(seed)`` in ``shuffle_labels`` to
    ``random.Random()``. The two same-seed builds below stop matching. Removing the
    ``cursor != assignment.scored`` check lets a permutation that dropped positions ship with
    different band sizes, which would confound the control with a coverage change.
    """
    real = _small_build(tmp_path, "real")
    ctrl_a = _small_build(tmp_path, "ctrl_a", extra=["--shuffle-labels", "4242"])
    ctrl_b = _small_build(tmp_path, "ctrl_b", extra=["--shuffle-labels", "4242"])

    m_real = json.loads((real / bsm.MANIFEST_NAME).read_text())
    m_a = json.loads((ctrl_a / bsm.MANIFEST_NAME).read_text())
    m_b = json.loads((ctrl_b / bsm.MANIFEST_NAME).read_text())

    assert m_a["is_shuffled_control"] is True and m_a["shuffle_labels_seed"] == 4242
    assert m_a["totals"]["band_counts"] == m_real["totals"]["band_counts"], "same band SIZES"
    assert m_a["shards"][0]["sha256"] != m_real["shards"][0]["sha256"], "different POSITIONS"
    # Same seed, same bytes.
    assert m_a["shards"][0]["sha256"] == m_b["shards"][0]["sha256"]
    for entry in m_a["shards"]:
        assert (ctrl_a / entry["mask"]).read_bytes() == (ctrl_b / entry["mask"]).read_bytes()


# ==============================================================================================
# REJECTION. A mask set that is wrong must not verify.
# ==============================================================================================


def test_a_truncated_mask_is_rejected(tmp_path):
    """
    THE FAILURE THE CONSUMER CALLS A MASK/SHARD LENGTH MISMATCH, caught before upload rather than
    eleven hours into a paid run. One byte per token is the format; a short file means every
    position after the truncation point is unlabelled and the band denominators are wrong.

    FAILURE MUTATION: delete the ``if size != int(entry["tokens"])`` branch in ``verify_build``.
    A truncated mask then verifies, and the consumer refuses it later -- after the GPU hours.
    """
    out = _small_build(tmp_path, "truncate")
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())
    victim = out / manifest["shards"][0]["mask"]
    body = victim.read_bytes()
    victim.write_bytes(body[: len(body) - 1])
    with pytest.raises(bsm.Refused, match="mask byte"):
        bsm.verify_build(out)


def test_a_mask_of_the_right_length_and_the_wrong_bytes_is_rejected(tmp_path):
    """
    THE DIGEST'S JOB. A same-length mask of different content passes every structural check, and
    the SHA-256 is the only thing that catches it. This is also the check the consumer performs,
    so a build that fails here would fail there.

    FAILURE MUTATION: change ``verify_build``'s digest comparison to
    ``digest[:0] == digest_declared[:0]`` (or drop the branch). The flipped byte below then
    verifies. Note the flip is at a SCORED position and changes one band bit into another, so
    neither the length nor the stray-bit check would catch it alone.
    """
    out = _small_build(tmp_path, "tamper")
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())
    victim = out / manifest["shards"][0]["mask"]
    body = bytearray(victim.read_bytes())
    body[1] = bsm.BAND_BIT[4096] if body[1] != bsm.BAND_BIT[4096] else bsm.BAND_BIT[0]
    victim.write_bytes(bytes(body))
    with pytest.raises(bsm.Refused, match="sha256"):
        bsm.verify_build(out)


def test_a_manifest_whose_digest_is_short_or_empty_is_rejected(tmp_path):
    """
    THE CONSUMER'S DIGEST CHECK IS FAIL-OPEN ON AN EMPTY DIGEST -- ``digest[:len("")] == ""`` is
    ``"" == ""`` and passes for ANY bytes (``train_core6_arm.py:1237-1240``). The manifest decides
    how much of the manifest gets checked, so the builder's verifier must insist on all 64
    characters.

    FAILURE MUTATION: delete the ``len(digest_declared) != 64`` branch in ``verify_build``. Both
    manifests below then verify -- the empty one while performing no content check at all.
    """
    out = _small_build(tmp_path, "shortdigest")
    path = out / bsm.MANIFEST_NAME
    for bad in ("", "ab"):
        manifest = json.loads(path.read_text())
        manifest["shards"][0]["sha256"] = bad
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        with pytest.raises(bsm.Refused, match="not 64"):
            bsm.verify_build(out)


def test_a_mask_with_a_stray_or_doubled_bit_is_rejected(tmp_path):
    """
    A BYTE THAT IS NOT EXACTLY ONE BAND BIT breaks the partition: the consumer's
    ``(flat & bit) != 0`` attributes a doubled byte to two bands and counts its CE twice, and a
    stray bit outside the layout is a byte nobody's accounting covers.

    FAILURE MUTATION: delete the ``stray`` / ``multi_bit`` branches in ``verify_build``, or make
    ``_count_bits`` fold unknown values into band 0. Both bytes below then verify.
    """
    out = _small_build(tmp_path, "stray")
    manifest = json.loads((out / bsm.MANIFEST_NAME).read_text())
    entry = manifest["shards"][0]
    victim = out / entry["mask"]
    pristine = victim.read_bytes()

    # A stray bit: 0x20 is outside BAND_BIT's values.
    body = bytearray(pristine)
    body[1] = 0x20
    victim.write_bytes(bytes(body))
    entry["sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    (out / bsm.MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="outside"):
        bsm.verify_build(out)

    # Two band bits in one byte.
    body = bytearray(pristine)
    body[1] = bsm.BAND_BIT[0] | bsm.BAND_BIT[32]
    victim.write_bytes(bytes(body))
    entry["sha256"] = hashlib.sha256(bytes(body)).hexdigest()
    (out / bsm.MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="more than one band bit"):
        bsm.verify_build(out)


def test_a_manifest_with_the_wrong_bands_or_no_sequence_length_is_rejected(tmp_path):
    """
    THE TWO MANIFEST-LEVEL DEFECTS: a band list the consumer would refuse
    (``train_core6_arm.py:1207``), and a missing ``sequence_length`` -- which the consumer does
    NOT check and which decides what every band means.

    FAILURE MUTATION: delete the ``manifest.get("bands") != list(BANDS)`` branch, or the
    ``not manifest.get("sequence_length")`` branch, in ``verify_build``.
    """
    out = _small_build(tmp_path, "badmanifest")
    path = out / bsm.MANIFEST_NAME
    pristine = path.read_text()

    manifest = json.loads(pristine)
    manifest["bands"] = [0, 32, 256, 1024]
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="manifest bands"):
        bsm.verify_build(out)

    manifest = json.loads(pristine)
    del manifest["sequence_length"]
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="no sequence_length"):
        bsm.verify_build(out)

    manifest = json.loads(pristine)
    manifest["definition_version"] = "something-else-v9"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="definition_version"):
        bsm.verify_build(out)


def test_a_mask_scored_at_a_different_window_is_rejected(tmp_path):
    """
    THE MOST DANGEROUS UNCHECKED THING IN THE CONSUMER PATH, caught here instead.

    TWO CASES, BECAUSE THE COUNT CHECK ALONE IS BLIND TO ONE OF THEM. Halving a declared window
    changes neither the labelled COUNT nor the labelled POSITIONS when the shard divides evenly
    (8,193 tokens is 4 windows at 2048 and 8 at 1024, and 4*2048 == 8*1024; and a window's targets
    are contiguous, so every position from 1 to ``scored`` is some window's target at any window
    size). What it does change is which bands are REACHABLE -- the furthest antecedent a
    ``seq_len`` window can hold is ``seq_len - 1`` -- so a populated band whose smallest distance
    exceeds that is proof of a relabel.

    FAILURE MUTATION (case 1, halved): delete the ``lower >= unreachable_lower`` loop in
    ``verify_build``. The halved manifest then verifies with the count check still in place --
    which is exactly the blind spot, and is why the loop exists.
    FAILURE MUTATION (case 2, non-dividing): delete the ``got_scored != expected_scored`` branch.
    """
    out = _small_build(tmp_path, "window")
    path = out / bsm.MANIFEST_NAME
    pristine = path.read_text()

    # Case 1: halved. Same labelled count AND same labelled positions; only reachability moves.
    manifest = json.loads(pristine)
    halved = FIXTURE_SEQ_LEN // 2
    manifest["sequence_length"] = halved
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    assert all(
        bsm.window_count(e["tokens"], FIXTURE_SEQ_LEN) * FIXTURE_SEQ_LEN
        == bsm.window_count(e["tokens"], halved) * halved
        for e in manifest["shards"]
    ), (
        "the fixture must divide evenly at both windows, or this case would be caught by the "
        "count check and the reachability check would go untested"
    )
    # And the fixture must actually populate a band the halved window cannot reach, or the check
    # would pass for the wrong reason -- an empty comparison set reporting success.
    assert manifest["totals"]["band_counts"]["4096"] > 0
    assert bsm.band_lower_bounds()[4096] >= halved
    with pytest.raises(bsm.Refused, match="furthest visible antecedent"):
        bsm.verify_build(out)

    # Case 2: a window that does not divide the shard, so the labelled count moves too.
    manifest = json.loads(pristine)
    manifest["sequence_length"] = 1500
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    with pytest.raises(bsm.Refused, match="labelled position"):
        bsm.verify_build(out)


def test_the_lower_bounds_are_the_inverse_of_band_of_gap():
    """
    ``band_lower_bounds`` AND ``band_of_gap`` MUST AGREE, since the relabel check trusts the first
    and the labelling uses the second. Checked by round-trip on both sides of every edge, not by
    restating the table.

    FAILURE MUTATION: change ``bounds[boundary] = previous + 1`` to ``= previous`` in
    ``band_lower_bounds``. The ``lower - 1`` assertion below then finds band 256's claimed lower
    bound of 32 mapping to band 32, so the two disagree.
    """
    bounds = bsm.band_lower_bounds()
    assert set(bounds) == set(bsm.POSITIVE_BANDS)
    for band, lower in bounds.items():
        assert bsm.band_of_gap(lower) == band, f"gap {lower} should be the first of band {band}"
        assert bsm.band_of_gap(band) == band, f"gap {band} should be the last of band {band}"
        if lower > 1:
            assert bsm.band_of_gap(lower - 1) != band, f"gap {lower - 1} must be a lower band"


def test_an_empty_band_is_refused_by_name_and_not_averaged_away(tmp_path):
    """
    AN EMPTY BAND REPORTS ``ce: null`` FROM ``band_ce_from_totals`` -- a missing row, not a bad
    number, in a table where everything else is fine. And a MEAN over five bands stays
    comfortable while one of them is dead, which is the aggregate-masks-a-dead-component failure.

    So the check is per band, and the message must NAME the offenders.

    FAILURE MUTATION: change ``assert_bands_are_live`` to test
    ``statistics.mean(counts.values()) >= min_band_tokens``. The counts below have a healthy mean
    and a dead band, and the mean version passes them.
    """
    counts = {0: 900_000_000, 32: 40_000_000, 256: 30_000_000, 1024: 30_000_000, 4096: 0}
    with pytest.raises(bsm.Refused, match=r"band\(s\) 4096 have NO labelled token"):
        bsm.assert_bands_are_live(counts, min_band_tokens=1_000, total_scored=1_000_000_000)

    thin = {**counts, 4096: 5}
    with pytest.raises(bsm.Refused) as excinfo:
        bsm.assert_bands_are_live(thin, min_band_tokens=1_000, total_scored=1_000_000_000)
    assert "4096: 5" in str(excinfo.value), "the failure must name the band and its count"

    # And it passes when every band clears the floor, so it is not a guard that always fires.
    bsm.assert_bands_are_live(
        {**counts, 4096: 2_000}, min_band_tokens=1_000, total_scored=1_000_000_000
    )


def test_the_mass_criterion_fires_on_a_corpus_with_no_recall_structure():
    """
    ``c_mass`` IS A STRUCTURAL FLOOR, not a quality bar: it catches a corpus of unique ids, a
    visibility rule that never matches, or a dtype read at the wrong width. All three present as
    almost no position having a visible antecedent.

    FAILURE MUTATION: change ``if realized < c_mass`` to ``if realized < 0`` in
    ``assert_mass_clears_the_criterion``. The zero-mass case below then passes, and a build in
    which the definition never fired ships a table of band-0-only results.
    """
    with pytest.raises(bsm.Refused, match="below the declared criterion"):
        bsm.assert_mass_clears_the_criterion(
            with_antecedent=0, total_scored=1_000_000, c_mass=bsm.DEFAULT_C_MASS
        )
    got = bsm.assert_mass_clears_the_criterion(
        with_antecedent=500_000, total_scored=1_000_000, c_mass=bsm.DEFAULT_C_MASS
    )
    assert got == pytest.approx(0.5)
    with pytest.raises(bsm.Refused, match="no position was scored"):
        bsm.assert_mass_clears_the_criterion(
            with_antecedent=0, total_scored=0, c_mass=bsm.DEFAULT_C_MASS
        )


def test_a_corpus_of_the_wrong_width_or_with_a_header_is_refused():
    """
    THE CONSUMER HARD-CODES ``uint32`` AND OFFSET ZERO (``getsize // 4``,
    ``np.memmap(dtype=np.uint32)``, and OLMo-core memmaps from byte 0). A uint16 corpus decodes to
    in-range ids and a header is labelled and scored as tokens -- both silent.

    FAILURE MUTATION: delete any of the three branches in
    ``assert_readable_the_way_the_consumer_reads_it``.
    """
    ok = bsm.CorpusSpec("d", "v1", "uint32", "little", 0, [bsm.ShardSpec("k")])
    ok.assert_readable_the_way_the_consumer_reads_it()
    with pytest.raises(bsm.Refused, match="hard-codes uint32"):
        bsm.CorpusSpec(
            "d", "v1", "uint16", "little", 0, []
        ).assert_readable_the_way_the_consumer_reads_it()
    with pytest.raises(bsm.Refused, match="endian"):
        bsm.CorpusSpec(
            "d", "v1", "uint32", "big", 0, []
        ).assert_readable_the_way_the_consumer_reads_it()
    with pytest.raises(bsm.Refused, match="header"):
        bsm.CorpusSpec(
            "d", "v1", "uint32", "little", 64, []
        ).assert_readable_the_way_the_consumer_reads_it()


def test_a_shard_that_is_not_a_whole_number_of_tokens_is_refused(tmp_path):
    """
    A TRUNCATED DOWNLOAD IS NOT A SHORT CORPUS. ``getsize // 4`` floors, which is how a
    partially-transferred shard becomes a slightly-short shard that nothing notices.

    FAILURE MUTATION: change ``shard_token_count`` to ``return os.path.getsize(path) // 4``
    without the remainder check.
    """
    path = tmp_path / "odd.u32le.bin"
    path.write_bytes(b"\x01\x02\x03\x04\x05")  # 5 bytes: one token and a fragment
    with pytest.raises(bsm.Refused, match="whole number"):
        bsm.shard_token_count(str(path))
    path.write_bytes(b"\x01\x02\x03\x04" * 3)
    assert bsm.shard_token_count(str(path)) == 3


def test_a_shard_too_short_for_one_window_is_refused(tmp_path):
    """
    A SHARD THAT YIELDS NO WINDOW WOULD BE DECLARED IN THE MANIFEST AND SCORED FOR NOTHING: the
    consumer reads no mask bytes for it, but its ``tokens`` still enters the declared total.

    FAILURE MUTATION: delete the ``if n_tokens <= seq_len`` branch in ``build_one``. The build
    then succeeds with an all-zero mask, and every band count for that shard is 0.
    """
    from array import array

    seq_len = 64
    short = tmp_path / "short.u32le.bin"
    short.write_bytes(array("I", list(range(seq_len))).tobytes())  # seq_len tokens: 0 windows
    listing = tmp_path / "shards.json"
    listing.write_text(
        json.dumps(
            {
                "dataset_id": "d",
                "dataset_version": "v1",
                "dtype": "uint32",
                "byte_order": "little",
                "header_bytes": 0,
                "shards": [{"s3_key": "d/t/short.u32le.bin", "local": str(short)}],
            }
        )
    )
    assert (
        bsm.main(
            [
                "build",
                "--out",
                str(tmp_path / "out"),
                "--shard-list",
                str(listing),
                "--sequence-length",
                str(seq_len),
                "--min-band-tokens",
                "1",
                "--c-mass",
                "0.0",
                "--consumer",
                str(CONSUMER),
            ]
        )
        == 1
    ), "a shard with no complete window must refuse, exit 1"


def test_a_key_from_another_bucket_is_refused():
    """
    THE CONSUMER REFETCHES AS ``s3://edullm-data/{s3_key}`` (``train_core6_arm.py:1227``), so a
    key from any other bucket cannot be expressed in the manifest and must not be silently
    rewritten into one that resolves to the wrong object.

    FAILURE MUTATION: change ``_key_of`` to ``return without.partition("/")[2]`` unconditionally.
    A shard from another bucket then lands in the manifest as though it were in ``edullm-data``.
    """
    assert bsm._key_of("s3://edullm-data/a/b/c.u32le.bin") == "a/b/c.u32le.bin"
    with pytest.raises(bsm.Refused, match="cannot express another bucket"):
        bsm._key_of("s3://some-other-bucket/a/b/c.u32le.bin")


def test_verify_on_a_directory_with_no_manifest_refuses(tmp_path):
    """
    AN EMPTY DIRECTORY MUST NOT VERIFY. "Nothing to check" reporting success is the
    empty-comparison-set failure.

    FAILURE MUTATION: change ``verify_build``'s missing-manifest branch to ``return {}``.
    """
    with pytest.raises(bsm.Refused, match="not a built mask directory"):
        bsm.verify_build(tmp_path)


def test_the_verify_subcommand_exits_nonzero_on_a_bad_build(tmp_path):
    """
    THE EXIT CODE IS THE ONLY THING A BATCH JOB OR A SHELL ``&&`` READS.

    FAILURE MUTATION: change ``main``'s ``except Refused`` handler to ``return 0``. Every
    rejection test above still passes -- they call the functions directly -- while the CLI reports
    success on a refused build. That is the gap this test closes.
    """
    out = _small_build(tmp_path, "cli")
    assert bsm.main(["verify", str(out)]) == 0
    manifest = out / bsm.MANIFEST_NAME
    payload = json.loads(manifest.read_text())
    payload["bands"] = [1, 2, 3]
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2))
    assert bsm.main(["verify", str(out)]) == 1


# ==============================================================================================
# THE META-TEST: the named mutations must all point at code that exists.
# ==============================================================================================


def test_the_named_mutations_are_all_reachable():
    """
    EVERY FAILURE MUTATION NAMED ABOVE MUST NAME REAL SOURCE TEXT.

    A docstring that says "change ``if gap <= boundary``" is worthless if no such line exists --
    the mutation is unreachable, the test's failure mode is unnamed, and the test is decoration.
    This greps the builder for each mutated expression, so a refactor that renames one of them
    breaks THIS test rather than silently invalidating the claim in the other one's docstring.

    Not a substitute for running the mutations. It is the cheap half: it proves the target exists.
    """
    source = (REPO_ROOT / "scripts" / "build_slice_masks.py").read_text(encoding="utf-8")
    for fragment in (
        "if gap <= boundary",  # boundary side
        "band_of_gap(q - previous)",  # distance arithmetic
        "last[key] = q",  # most-recent antecedent
        "last: Dict[int, int] = {}",  # per-window visibility reset
        "key = (key << 32) | int(block[local - back])",  # the bigram key
        "for back in range(ngram - 1, -1, -1)",  # key order
        "(n_tokens - 1) // seq_len",  # the window off-by-one
        "if seq_len - 1 > top",  # representable window
        "if ngram < 2",  # unigram refusal
        "builds.sort(key=lambda b: b.index)",  # jobs-independent order
        "sorted(corpus.shards, key=lambda s: s.s3_key)",  # manifest order
        "sort_keys=True",  # deterministic manifest bytes
        'if size != int(entry["tokens"])',  # truncation
        "len(digest_declared) != 64",  # digest length
        "if realized < c_mass",  # mass criterion
        "if n_tokens <= seq_len",  # no-window shard
        "rng = random.Random(seed)",  # reproducible control
        "if total != scored",  # the partition
    ):
        assert fragment in source, (
            f"a test docstring names the mutation {fragment!r} but no such text is in the "
            "builder, so that test's failure mode is unnamed"
        )


def test_nothing_in_the_builder_imports_a_heavy_dependency():
    """
    THIS MUST RUN WHERE THERE IS NO TORCH, NO NUMPY AND NO CUDA -- it is a CPU labeller, and its
    tests are part of the fast suite. A stray ``import numpy`` at module scope would make the
    builder unimportable in a metadata-only environment and would drag the test suite with it.

    FAILURE MUTATION: add ``import numpy`` (or ``import torch``) at the top of
    build_slice_masks.py.
    """
    assert "numpy" not in sys.modules or bsm.__name__ == "build_slice_masks"
    source = (REPO_ROOT / "scripts" / "build_slice_masks.py").read_text(encoding="utf-8")
    import ast as _ast

    tree = _ast.parse(source)
    forbidden = {"torch", "numpy", "np", "pandas", "scipy"}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, alias.name
        elif isinstance(node, _ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden, node.module
