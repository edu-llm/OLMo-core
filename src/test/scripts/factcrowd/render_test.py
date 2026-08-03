"""
What the vocabulary and the renderer guarantee: exact spans, one length, and enough throughput.

The span tests are the load-bearing ones. The bit-counter sums loss over exactly the tokens carrying
an attribute value, so a span that is off by one measures the wrong thing and reports a plausible
number -- and this programme has already shipped a corpus whose eval keys and prose disagreed about
where a value started. So spans are checked by *decoding the tokens they point at* and comparing
against the entity table, rather than by trusting an offset.

Equal template length is the other structural property. It is what lets biography ``i`` occupy tokens
``[i·L, (i+1)·L)``, which is what makes packing arithmetic instead of a 10-20 GB offset index.
"""

import time

import numpy as np
import pytest
from factcrowd.corpus import entities as E
from factcrowd.corpus import render as Rn
from factcrowd.corpus import values as V
from factcrowd.corpus import vocab as Vo

from olmo_core.exceptions import OLMoConfigurationError

DOMAIN_TOKENS = ("<facts>", "<mano>", "<brevo>", "<related>")
SEED = 1234


def assemble(n_entities: int = 2_000, bits_per_attribute: int = -1):
    """Build a matching (schema, vocabulary, table, renderer) set. ``bits<0`` means bioS."""
    if bits_per_attribute < 0:
        templates = Rn.BIOS_TEMPLATES
        literals = Rn.literal_words_of(templates)
        schema = V.bios_schema(reserved=tuple(literals) + Vo.SPECIALS + DOMAIN_TOKENS)
    else:
        templates = Rn.entropy_templates(V.ENTROPY_ATTRIBUTES, V.ENTROPY_WORDS_PER_VALUE)
        literals = Rn.literal_words_of(templates)
        schema = V.entropy_schema(
            bits_per_attribute, reserved=tuple(literals) + Vo.SPECIALS + DOMAIN_TOKENS
        )
    vocabulary = Vo.Vocabulary.build(
        schema.schema, literal_words=literals, domain_tokens=DOMAIN_TOKENS
    )
    table = E.EntityTable.build(schema.schema, n_entities, SEED)
    renderer = Rn.Renderer(
        table, schema, vocabulary, templates, domain_token=DOMAIN_TOKENS[0], seed=7
    )
    return schema, vocabulary, table, renderer


# --- the vocabulary ---------------------------------------------------------------------------------


def test_the_special_ids_are_fixed():
    """
    PAD is 0 so a zero-filled buffer decodes as padding, and PAD and EOS are distinct.

    Sharing an id would inflate OLMo-core's per-instance document count, which is derived by counting
    EOS occurrences, until it explodes.
    """
    _, vocabulary, _, _ = assemble(50)
    assert (vocabulary.pad_id, vocabulary.eos_id, vocabulary.bos_id) == (0, 1, 2)
    assert vocabulary.pad_id != vocabulary.eos_id
    assert vocabulary.decode([0, 1, 2]) == (Vo.PAD, Vo.EOS, Vo.BOS)


def test_every_pool_value_has_a_token_and_the_lookup_agrees():
    """The pool tables are what turn stored indices into tokens with no string handling at train time."""
    schema, vocabulary, _, _ = assemble(50)
    for pool in tuple(schema.schema.attributes) + tuple(schema.schema.names):
        ids = vocabulary.pool_token_ids[pool.name]
        assert len(ids) == len(pool)
        for index, value in enumerate(pool.values):
            assert int(ids[index]) == vocabulary.id_of(value)


def test_an_unseen_word_is_refused_rather_than_mapped_to_unknown():
    """
    A closed vocabulary means an unseen word is a bug in the renderer, not unusual text.

    Encoding it as ``<unk>`` would let a schema/renderer disagreement train silently.
    """
    _, vocabulary, _, _ = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="not in the vocabulary"):
        vocabulary.id_of("Zzzznotaword")


def test_a_literal_that_is_also_a_pool_value_is_refused():
    """
    A token serving as both prose and a fact is ambiguous, so recall becomes unmeasurable.

    This fired for real: the generated pool word "Born" collided with the template literal "Born",
    which is why the pool allocator takes a reserved set.
    """
    schema = V.bios_schema()
    collision = schema.schema.attributes[0].values[0]
    with pytest.raises(OLMoConfigurationError, match="already a word"):
        Vo.Vocabulary.build(schema.schema, literal_words=(collision,))


def test_a_repeated_domain_token_is_refused():
    """Two ids for one word would make the slice label ambiguous."""
    schema = V.bios_schema()
    with pytest.raises(OLMoConfigurationError, match="already a word"):
        Vo.Vocabulary.build(schema.schema, domain_tokens=("<facts>", "<facts>"))


def test_padded_size_rounds_up_and_never_down():
    """The unused rows never appear in the data, so they cost parameters and no correctness."""
    _, vocabulary, _, _ = assemble(50)
    padded = vocabulary.padded_size(128)
    assert padded >= vocabulary.size
    assert padded % 128 == 0
    assert padded - vocabulary.size < 128


def test_the_vocabulary_is_deterministic_and_its_fingerprint_tracks_content():
    """A checkpoint and a corpus are checked against each other by fingerprint."""
    _, first, _, _ = assemble(50)
    _, second, _, _ = assemble(50)
    assert first.fingerprint() == second.fingerprint()

    _, other, _, _ = assemble(50, bits_per_attribute=8)
    assert first.fingerprint() != other.fingerprint()


def test_decode_refuses_an_id_outside_the_vocabulary():
    """Otherwise a corrupt buffer decodes as an IndexError deep in a test helper."""
    _, vocabulary, _, _ = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="outside the vocabulary"):
        vocabulary.decode([vocabulary.size])


# --- one length, which is what makes packing arithmetic ----------------------------------------------


def test_all_bios_templates_render_to_one_length():
    """
    The property the ``Renderer`` enforces, and the reason its template set was rewritten.

    The first draft had literal counts from 9 to 14. Equal length is what lets biography ``i`` occupy
    tokens ``[i*L, (i+1)*L)``; unequal costs either heavy padding or a prefix-sum index over every
    document in a 1.29-billion-document stream.
    """
    literal_counts = {len(template.literals) for template in Rn.BIOS_TEMPLATES}
    assert literal_counts == {12}
    assert len(Rn.BIOS_TEMPLATES) >= 20


def test_templates_of_unequal_length_are_refused_with_the_lengths_named():
    """The message has to say which templates are wrong, because fixing it means editing them."""
    schema, vocabulary, table, _ = assemble(50)
    # A literal already in the vocabulary, so the length check is what fires rather than the
    # vocabulary's own refusal of an unseen word.
    ragged = list(Rn.BIOS_TEMPLATES[:21]) + [
        Rn.Template(parts=Rn.BIOS_TEMPLATES[0].parts + ("and",))
    ]
    with pytest.raises(OLMoConfigurationError, match="same token count") as excinfo:
        Rn.Renderer(table, schema, vocabulary, ragged, domain_token=DOMAIN_TOKENS[0])
    assert "template indices by length" in str(excinfo.value)


def test_the_entropy_axis_renders_one_length_at_every_demand():
    """
    The axis's defining property, measured end to end rather than argued from the schema.

    If token count moved with ``b``, the entropy axis would inherit the confound it exists to remove
    -- and it would do so silently, because the corpus would still be valid and the bits still exact.
    """
    lengths = set()
    for bits in (0, 4, 8, 16, 24, 32):
        _, _, _, renderer = assemble(200, bits_per_attribute=bits)
        lengths.add(renderer.tokens_per_bio)
    assert len(lengths) == 1, lengths


def test_tokens_per_bio_is_measured_not_assumed():
    """
    Pins the real figure, because the budget was written against an assumed 100.

    bioS renders to 25 tokens under this vocabulary and the entropy axis to 42. Both are far from
    100, and since token count sets the compute bill this number is load-bearing.
    """
    _, _, _, bios = assemble(200)
    _, _, _, entropy = assemble(200, bits_per_attribute=8)
    assert bios.tokens_per_bio == 25
    assert entropy.tokens_per_bio == 42


def test_bios_per_instance_is_arithmetic_and_bounds_the_waste():
    """20 biographies of 25 tokens leave 12 of 512 unused, 2.3%, which is paid for in FLOPs."""
    assert Rn.bios_per_instance(25, 512) == 20
    with pytest.raises(OLMoConfigurationError, match="does not fit|not one fits"):
        Rn.bios_per_instance(600, 512)
    with pytest.raises(OLMoConfigurationError, match="unused"):
        Rn.bios_per_instance(300, 512)


def test_instance_count_drops_a_partial_final_instance():
    """One instance is cheaper than making every instance a different shape."""
    assert Rn.instance_count(10, 200, 20) == 100
    assert Rn.instance_count(10, 3, 20) == 1


# --- spans, checked by decoding what they point at --------------------------------------------------


@pytest.mark.parametrize("bits_per_attribute", [-1, 8, 32])
def test_every_span_decodes_to_the_value_the_table_holds(bits_per_attribute):
    """
    The test the bit-counter's validity rests on, run on both axes.

    Checked by decoding the tokens the span points at, not by trusting the offset -- an off-by-one
    would otherwise measure a neighbouring token and report a plausible number.
    """
    schema, vocabulary, table, renderer = assemble(300, bits_per_attribute)
    for entity_id in range(0, 300, 7):
        for exposure in (0, 1, 99):
            token_ids, spans = renderer.render(entity_id, exposure)
            words = vocabulary.decode(token_ids)
            truth = table.attribute_values(entity_id)
            expected = {
                spec.name: tuple(truth[schema.pool_index[pool]] for pool in spec.pool_names)
                for spec in schema.values
            }
            assert {span.attribute: words[span.start : span.end] for span in spans} == expected


def test_the_name_appears_in_the_rendered_text():
    """
    The key has to be there, or a recall prompt has nothing to ask about.

    Kept separate from the value spans because names are excluded from bits and so from spans.
    """
    _, _, table, renderer = assemble(300)
    for entity_id in range(0, 300, 11):
        assert " ".join(table.name_parts(entity_id)) in renderer.text(entity_id, 0)


def test_spans_cover_every_attribute_exactly_once_and_do_not_overlap():
    """A value counted twice or missed would scale the bit-count by an integer factor."""
    schema, _, _, renderer = assemble(100)
    _, spans = renderer.render(0, 0)
    assert sorted(span.attribute for span in spans) == sorted(spec.name for spec in schema.values)
    ordered = sorted(spans, key=lambda span: span.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end <= later.start


def test_the_domain_token_opens_every_biography_and_eos_closes_it():
    """
    The domain token is mandatory in the mixture, and EOS is the document boundary the loader counts.
    """
    _, vocabulary, _, renderer = assemble(100)
    for exposure in range(5):
        token_ids, _ = renderer.render(3, exposure)
        assert vocabulary.decode(token_ids[:2]) == (DOMAIN_TOKENS[0], Vo.BOS)
        assert int(token_ids[-1]) == vocabulary.eos_id
        assert int((token_ids == vocabulary.eos_id).sum()) == 1


# --- phrasing variety -------------------------------------------------------------------------------


def test_one_entity_sees_many_phrasings_across_its_exposures():
    """
    The point of a template set. A single phrasing lets the model store a pattern, not a fact --
    our own corpus answered one question at 83% under one phrasing and 1.3% under another.
    """
    _, _, _, renderer = assemble(100)
    used = {renderer.template_index(0, exposure) for exposure in range(200)}
    assert len(used) == renderer.n_templates


def test_two_entities_at_the_same_exposure_do_not_share_a_phrasing_schedule():
    """Otherwise phrasing correlates with exposure index across the whole corpus."""
    _, _, _, renderer = assemble(100)
    first = [renderer.template_index(0, exposure) for exposure in range(50)]
    second = [renderer.template_index(1, exposure) for exposure in range(50)]
    assert first != second


def test_too_few_templates_is_refused_but_can_be_overridden_deliberately():
    """A single-template control is legitimate; reaching one by accident is not."""
    schema, vocabulary, table, _ = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="below the 20 minimum"):
        Rn.Renderer(table, schema, vocabulary, Rn.BIOS_TEMPLATES[:3], domain_token=DOMAIN_TOKENS[0])
    single = Rn.Renderer(
        table,
        schema,
        vocabulary,
        Rn.BIOS_TEMPLATES[:1],
        domain_token=DOMAIN_TOKENS[0],
        min_templates=1,
    )
    assert single.n_templates == 1


# --- the batched path must agree with the single one ------------------------------------------------


def test_render_instance_matches_rendering_one_at_a_time():
    """
    The throughput path and the reference path must produce identical bytes.

    ``render_instance`` vectorises template choice and writes through a view, so it is a different
    code path from ``render`` -- and the only one training will use.
    """
    _, _, _, renderer = assemble(500)
    entity_ids = np.array([7, 13, 400, 0, 499], dtype=np.uint64)
    exposures = np.array([0, 5, 199, 42, 1], dtype=np.uint64)

    width = renderer.tokens_per_bio
    batched = np.empty(entity_ids.size * width, dtype=np.uint32)
    batched_spans = renderer.render_instance(batched, entity_ids, exposures)

    for position, (entity_id, exposure) in enumerate(zip(entity_ids, exposures)):
        single, spans = renderer.render(int(entity_id), int(exposure))
        np.testing.assert_array_equal(batched[position * width : (position + 1) * width], single)
        assert batched_spans[position] == spans


def test_render_instance_refuses_a_short_buffer_or_mismatched_shapes():
    """Silently writing fewer biographies would leave stale tokens from a previous instance."""
    _, _, _, renderer = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="at least"):
        renderer.render_instance(
            np.empty(3, dtype=np.uint32),
            np.array([0, 1], dtype=np.uint64),
            np.array([0, 1], dtype=np.uint64),
        )
    with pytest.raises(OLMoConfigurationError, match="same shape"):
        renderer.render_instance(
            np.empty(1000, dtype=np.uint32),
            np.array([0, 1], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
        )


def test_render_into_refuses_a_wrong_dtype_buffer():
    """A signed or narrower buffer would truncate token ids without raising."""
    _, _, _, renderer = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="must be"):
        renderer.render_into(np.empty(1000, dtype=np.int32), 0, 0, 0)


# --- template validation ----------------------------------------------------------------------------


def test_a_template_omitting_an_attribute_is_refused():
    """
    Entities would then differ in how many facts they assert, so exposures stop being comparable.
    """
    schema, vocabulary, table, _ = assemble(50)
    stripped = tuple(part for part in Rn.BIOS_TEMPLATES[0].parts if part != "{major}")
    with pytest.raises(OLMoConfigurationError, match="omits"):
        Rn.Renderer(
            table,
            schema,
            vocabulary,
            [Rn.Template(parts=stripped)] * 22,
            domain_token=DOMAIN_TOKENS[0],
        )


def test_a_template_repeating_an_attribute_is_refused():
    """A value stated twice is exposed twice, so the exposure count would not be 200."""
    schema, vocabulary, table, _ = assemble(50)
    doubled = Rn.BIOS_TEMPLATES[0].parts + ("{major}",)
    with pytest.raises(OLMoConfigurationError, match="repeats"):
        Rn.Renderer(
            table,
            schema,
            vocabulary,
            [Rn.Template(parts=doubled)] * 22,
            domain_token=DOMAIN_TOKENS[0],
        )


def test_a_template_naming_an_unknown_slot_is_refused():
    """It would otherwise render as a literal word and silently enter the vocabulary."""
    schema, vocabulary, table, _ = assemble(50)
    wrong = Rn.BIOS_TEMPLATES[0].parts + ("{nonesuch}",)
    with pytest.raises(OLMoConfigurationError, match="unknown slots"):
        Rn.Renderer(
            table,
            schema,
            vocabulary,
            [Rn.Template(parts=wrong)] * 22,
            domain_token=DOMAIN_TOKENS[0],
        )


def test_a_table_from_a_different_schema_is_refused():
    """Its indices point into other pools, so every rendered fact would be wrong."""
    schema, vocabulary, _, _ = assemble(50)
    other_schema, _, other_table, _ = assemble(50, bits_per_attribute=8)
    assert schema.schema.fingerprint() != other_schema.schema.fingerprint()
    with pytest.raises(OLMoConfigurationError, match="different schema"):
        Rn.Renderer(
            other_table, schema, vocabulary, Rn.BIOS_TEMPLATES, domain_token=DOMAIN_TOKENS[0]
        )


# --- determinism and throughput ---------------------------------------------------------------------


def test_rendering_is_deterministic():
    """The stream is reproducible from a seed, which is what we publish instead of token shards."""
    _, _, _, first = assemble(500)
    _, _, _, second = assemble(500)
    for entity_id in (0, 17, 499):
        np.testing.assert_array_equal(first.render(entity_id, 3)[0], second.render(entity_id, 3)[0])


def test_splitmix64_is_deterministic_and_mixes():
    """
    Pure, so it is safe per biography, and 28x cheaper than constructing a Generator.

    A Generator per biography caps a worker near 13M tokens/s before any rendering happens.
    """
    values = np.arange(1000, dtype=np.uint64)
    np.testing.assert_array_equal(Rn.splitmix64(values), Rn.splitmix64(values))
    mixed = Rn.splitmix64(values)
    assert len(np.unique(mixed)) == 1000
    assert len(np.unique(mixed % np.uint64(22))) == 22


@pytest.mark.slow
def test_throughput_clears_what_a_training_node_consumes():
    """
    A worker must out-run the GPUs or the run is data-bound and the MFU number is meaningless.

    The requirement is ``8 * FLOPS * MFU / (6 * P)``, largest for the *smallest* model: 12.7M tokens/s
    for the 13M row on an eight-H100 node at 20% MFU. Measured single-threaded here and multiplied by
    the eight workers a node runs -- deliberately conservative, since it ignores that the first
    measurement includes warm-up.
    """
    _, _, _, renderer = assemble(20_000)
    per_instance = Rn.bios_per_instance(renderer.tokens_per_bio, 512)
    buffer = np.empty(512, dtype=np.uint32)
    entity_ids = np.arange(per_instance, dtype=np.uint64)
    exposures = np.zeros(per_instance, dtype=np.uint64)

    iterations = 4_000
    started = time.perf_counter()
    for step in range(iterations):
        renderer.render_instance(buffer, (entity_ids + step * per_instance) % 20_000, exposures)
    elapsed = time.perf_counter() - started

    tokens_per_second = iterations * per_instance * renderer.tokens_per_bio / elapsed
    assert tokens_per_second * 8 > 12.7e6, tokens_per_second
