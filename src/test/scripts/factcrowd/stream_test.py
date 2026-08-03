"""
What the token stream guarantees: exact exposure counts, a correct offset index, and exact ranges.

Everything here is checked against a brute-force reference -- the whole stream rendered biography by
biography -- rather than against the index's own arithmetic. An offset index that agrees with itself
and disagrees with the renderer would produce a corpus that trains fine and states the wrong facts at
the wrong positions, and nothing downstream would notice.

The exposure count is the invariant with the most riding on it. Demand is computed assuming every
entity appears exactly ``exposures`` times; if the stream order dropped or duplicated one, the cell
would sit at a demand nobody had computed.
"""

import numpy as np
import pytest
from factcrowd.corpus import entities as E
from factcrowd.corpus import render as Rn
from factcrowd.corpus import stream as St
from factcrowd.corpus import values as V
from factcrowd.corpus import vocab as Vo

from olmo_core.exceptions import OLMoConfigurationError

DOMAIN_TOKENS = ("<facts>", "<mano>", "<brevo>", "<related>")
SEED = 1234


def make_renderer(n_table_entities: int = 400) -> Rn.Renderer:
    """A renderer over the bioS schema, big enough for the slices these tests build."""
    literals = Rn.literal_words_of(Rn.BIOS_TEMPLATES)
    schema = V.bios_schema(reserved=tuple(literals) + Vo.SPECIALS + DOMAIN_TOKENS)
    vocabulary = Vo.Vocabulary.build(
        schema.schema, literal_words=literals, domain_tokens=DOMAIN_TOKENS
    )
    table = E.EntityTable.build(schema.schema, n_table_entities, SEED)
    return Rn.Renderer(
        table, schema, vocabulary, Rn.BIOS_TEMPLATES, domain_token=DOMAIN_TOKENS[0], seed=7
    )


def brute_force(bio_stream: St.BioStream, renderer: Rn.Renderer) -> np.ndarray:
    """The whole stream, rendered one biography at a time. The reference every test compares to."""
    pieces = []
    for index in range(bio_stream.n_bios):
        entity_ids, exposures = bio_stream.assignment(np.array([index]))
        tokens, _ = renderer.render(int(entity_ids[0]), int(exposures[0]))
        pieces.append(tokens)
    return np.concatenate(pieces)


# --- the exposure invariant --------------------------------------------------------------------------


@pytest.mark.parametrize("n_entities, exposures", [(50, 4), (97, 7), (200, 3), (13, 200)])
def test_every_entity_appears_exactly_the_stated_number_of_times(n_entities, exposures, tmp_path):
    """
    Demand assumes exactly ``exposures`` appearances per entity, so this is what makes it true.

    Structural rather than sampled: the order is a permutation per epoch, so a count that came out
    wrong would mean the permutation is not one.
    """
    renderer = make_renderer(max(400, n_entities))
    bio_stream = St.BioStream(
        renderer, n_entities=n_entities, exposures=exposures, work_dir=tmp_path
    )
    entity_ids, exposure_indices = bio_stream.assignment(np.arange(bio_stream.n_bios))

    counts = np.bincount(entity_ids.astype(np.int64), minlength=n_entities)
    assert counts.tolist() == [exposures] * n_entities
    assert (
        sorted(np.bincount(exposure_indices.astype(np.int64)).tolist()) == [n_entities] * exposures
    )


def test_each_epoch_is_a_permutation_of_the_entity_set(tmp_path):
    """One appearance per entity per epoch, which is what makes the total count exact."""
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=200, exposures=5, work_dir=tmp_path)
    for epoch in range(5):
        entity_ids, exposures = bio_stream.assignment(np.arange(epoch * 200, (epoch + 1) * 200))
        assert sorted(int(e) for e in entity_ids) == list(range(200))
        assert set(int(x) for x in exposures) == {epoch}


def test_an_entity_gets_different_neighbours_in_different_epochs(tmp_path):
    """
    The confound the per-epoch permutation exists to remove.

    Without it, entity ``e`` would sit between ``e-1`` and ``e+1`` in all 200 of its exposures, and a
    model could learn the sequence rather than the facts.
    """
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=200, exposures=8, work_dir=tmp_path)
    entity_ids, _ = bio_stream.assignment(np.arange(bio_stream.n_bios))

    followers = {}
    for epoch in range(8):
        block = entity_ids[epoch * 200 : (epoch + 1) * 200]
        position = int(np.where(block == 0)[0][0])
        if position + 1 < block.size:
            followers[epoch] = int(block[position + 1])
    assert len(set(followers.values())) > 1, followers


# --- the offset index against a brute-force reference -----------------------------------------------


def test_num_tokens_is_the_real_sum_of_lengths(tmp_path):
    """Recomputed from the rendered stream, not from the index that reports it."""
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=120, exposures=3, work_dir=tmp_path)
    assert bio_stream.num_tokens == brute_force(bio_stream, renderer).size


@pytest.mark.parametrize("chunk", [1, 7, 64, 512])
def test_locate_agrees_with_a_brute_force_scan_at_every_token(chunk, tmp_path):
    """
    Checked at every token position, not sampled, and across chunk sizes including degenerate ones.

    ``locate`` is a search plus a scan, so its two halves can disagree at a chunk boundary -- which is
    exactly where a sampled test would miss it.
    """
    renderer = make_renderer()
    bio_stream = St.BioStream(
        renderer, n_entities=40, exposures=3, work_dir=tmp_path / str(chunk), chunk=chunk
    )
    entity_ids, exposures = bio_stream.assignment(np.arange(bio_stream.n_bios))
    lengths = renderer.lengths_of(entity_ids, exposures)
    starts = np.concatenate([[0], np.cumsum(lengths)])

    for token_index in range(bio_stream.num_tokens):
        expected_bio = int(np.searchsorted(starts, token_index, side="right")) - 1
        assert bio_stream.locate(token_index) == (expected_bio, int(starts[expected_bio]))


def test_the_index_is_cached_and_keyed_by_fingerprint(tmp_path):
    """
    A changed schema, template set or seed must get a different file rather than a reused index.

    Reusing one would place every biography at the wrong offset while every number stayed plausible.
    """
    renderer = make_renderer()
    first = St.BioStream(renderer, n_entities=60, exposures=2, work_dir=tmp_path)
    assert first.index_path.is_file()
    assert first.index_path.with_suffix(".json").is_file()

    again = St.BioStream(renderer, n_entities=60, exposures=2, work_dir=tmp_path)
    assert again.index_path == first.index_path
    assert again.num_tokens == first.num_tokens

    other = St.BioStream(renderer, n_entities=60, exposures=2, work_dir=tmp_path, seed=99)
    assert other.index_path != first.index_path


def test_write_index_false_does_not_write(tmp_path):
    """Non-primary ranks compute the index in memory rather than racing on the file."""
    renderer = make_renderer()
    bio_stream = St.BioStream(
        renderer, n_entities=60, exposures=2, work_dir=tmp_path, write_index=False
    )
    assert bio_stream.num_tokens > 0
    assert not bio_stream.index_path.is_file()


# --- ranges, against the same reference -------------------------------------------------------------


def test_the_whole_stream_reads_back_exactly(tmp_path):
    """One range covering everything must equal the brute-force render."""
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=60, exposures=3, work_dir=tmp_path)
    expected = brute_force(bio_stream, renderer)
    np.testing.assert_array_equal(bio_stream.tokens(0, bio_stream.num_tokens), expected)


@pytest.mark.parametrize("width", [1, 2, 13, 128, 512])
def test_consecutive_ranges_reconstruct_the_stream(width, tmp_path):
    """
    Chunking at every width, including widths far shorter and longer than a biography.

    This is what ``ConcatAndChunkInstanceSource`` does, so a range that began or ended mid-biography
    incorrectly would corrupt exactly the instances the trainer sees.
    """
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=40, exposures=3, work_dir=tmp_path)
    expected = brute_force(bio_stream, renderer)

    pieces = []
    for start in range(0, bio_stream.num_tokens, width):
        end = min(start + width, bio_stream.num_tokens)
        piece = bio_stream.tokens(start, end)
        assert piece.size == end - start
        pieces.append(piece)
    np.testing.assert_array_equal(np.concatenate(pieces), expected)


def test_random_ranges_match_the_reference(tmp_path):
    """Random access is what a shuffled instance order produces, and it is a different code path."""
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=50, exposures=3, work_dir=tmp_path)
    expected = brute_force(bio_stream, renderer)
    rng = np.random.default_rng(0)

    for _ in range(300):
        start = int(rng.integers(0, bio_stream.num_tokens - 1))
        end = int(rng.integers(start + 1, min(start + 700, bio_stream.num_tokens) + 1))
        np.testing.assert_array_equal(bio_stream.tokens(start, end), expected[start:end])


def test_a_range_spanning_the_final_biography_is_exact(tmp_path):
    """The end of the stream is where an off-by-one in the fill loop would show."""
    renderer = make_renderer()
    bio_stream = St.BioStream(renderer, n_entities=30, exposures=2, work_dir=tmp_path)
    expected = brute_force(bio_stream, renderer)
    total = bio_stream.num_tokens
    for start in range(total - 200, total):
        np.testing.assert_array_equal(bio_stream.tokens(start, total), expected[start:total])


# --- guards -----------------------------------------------------------------------------------------


def test_an_empty_or_inverted_range_is_refused(tmp_path):
    """An empty range would return an empty instance and train on padding."""
    bio_stream = St.BioStream(make_renderer(), n_entities=20, exposures=2, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="empty or inverted"):
        bio_stream.tokens(5, 5)
    with pytest.raises(OLMoConfigurationError, match="empty or inverted"):
        bio_stream.tokens(9, 4)


def test_an_out_of_bounds_range_is_refused(tmp_path):
    """Reading past the end would silently repeat the last biography."""
    bio_stream = St.BioStream(make_renderer(), n_entities=20, exposures=2, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="out of bounds"):
        bio_stream.tokens(0, bio_stream.num_tokens + 1)
    with pytest.raises(OLMoConfigurationError, match="out of bounds"):
        bio_stream.tokens(-1, 10)


def test_a_slice_larger_than_its_table_is_refused(tmp_path):
    """
    The table is the ceiling: a slice asking for more entities would index off the end.

    The message names both numbers, because the fix is either a bigger table or a lower demand.
    """
    renderer = make_renderer(400)
    with pytest.raises(OLMoConfigurationError, match="but its table holds"):
        St.BioStream(renderer, n_entities=401, exposures=2, work_dir=tmp_path)


@pytest.mark.parametrize(
    "kwargs", [{"n_entities": 0}, {"exposures": 0}, {"chunk": 0}, {"seed": -1}]
)
def test_degenerate_sizes_are_refused(kwargs, tmp_path):
    """Each would produce an empty slice or an index with no entries."""
    call = {"n_entities": 20, "exposures": 2, "work_dir": tmp_path}
    call.update(kwargs)
    with pytest.raises(OLMoConfigurationError):
        St.BioStream(make_renderer(), **call)


def test_assignment_refuses_an_index_past_the_slice(tmp_path):
    """Otherwise the epoch arithmetic silently wraps into a nonexistent exposure."""
    bio_stream = St.BioStream(make_renderer(), n_entities=20, exposures=2, work_dir=tmp_path)
    with pytest.raises(OLMoConfigurationError, match="out of range"):
        bio_stream.assignment(np.array([bio_stream.n_bios]))


def test_the_fingerprint_tracks_everything_that_changes_the_stream(tmp_path):
    """Size, exposures, order seed and the renderer all have to move it."""
    renderer = make_renderer()
    base = St.BioStream(renderer, n_entities=40, exposures=2, work_dir=tmp_path).fingerprint()
    for kwargs in ({"n_entities": 41}, {"exposures": 3}, {"seed": 1}, {"chunk": 64}):
        call = {"n_entities": 40, "exposures": 2, "work_dir": tmp_path}
        call.update(kwargs)
        assert St.BioStream(renderer, **call).fingerprint() != base, kwargs
