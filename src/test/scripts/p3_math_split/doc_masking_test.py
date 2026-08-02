"""Intra-document attention masking, which is a speed feature with a correctness edge.

Packing puts several proofs in one 16,384-token sequence. Without document masking,
every token attends across the whole sequence — proofs attend to unrelated proofs,
which is both wrong and expensive: attention is 59% of FLOPs at this length.

OLMo-core supports it (`generate_doc_lengths=True` -> `doc_lens` -> `cu_doc_lens`),
and it finds boundaries by EOS. That last detail is the risk these tests exist for:
Qwen 2.5 uses the SAME id for pad and eos (151643), so a padded tail could be read as
thousands of one-token documents.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from olmo_core.data.utils import get_document_lengths  # noqa: E402
from tokenize_corpus import encode_rows_batched, pack_indices_by_length  # noqa: E402

EOS = 151643  # Qwen 2.5: also the pad id, which is the whole problem


def test_packed_documents_report_their_lengths():
    """Three documents of 3, 2 and 4 tokens, each terminated by EOS."""
    ids = torch.tensor([1, 2, EOS, 3, EOS, 4, 5, 6, EOS])
    assert get_document_lengths(ids, EOS).tolist() == [3, 2, 4]


def test_every_padding_token_reads_as_its_own_document():
    """Pins the wart rather than pretending it away.

    Qwen 2.5 uses one id for both pad and eos, so a padded tail is read as one
    document per padding token — measured, not assumed. It is not a correctness bug:
    those positions are masked out of the loss, and varlen attention isolates them so
    they cannot reach a real proof. It is a cost, paid per padding token, which is why
    the packer minimises the tail rather than the mask handling it.

    If this ever fails, someone gave pad its own id and the packing rationale below
    can be relaxed.
    """
    lens = get_document_lengths(torch.tensor([1, 2, EOS] + [EOS] * 8), EOS).tolist()
    assert lens == [3, 1, 1, 1, 1, 1, 1, 1, 1]

    distinct_pad = get_document_lengths(
        torch.tensor([1, 2, EOS] + [151900] * 8), EOS
    ).tolist()
    assert distinct_pad == [3, 8], "a distinct pad id would collapse the tail"


def test_the_packer_minimises_the_tail_rather_than_filling_in_order():
    """Best-fit-decreasing, because the tail is paid for twice: as wasted compute and
    as spurious one-token documents."""
    lengths = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    packed = pack_indices_by_length(lengths, 10)
    assert len(packed) == 5, "45 tokens have a five-bin optimum at capacity 10"
    assert sorted(i for row in packed for i in row) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in row) <= 10 for row in packed)


def test_the_dataset_is_configured_to_emit_document_lengths():
    """The flag is not an optimisation any more — without it, attention costs 1.53x."""
    src = (SCRIPTS / "train_platform.py").read_text()
    assert "generate_doc_lengths=True" in src, (
        "NumpyFSLDatasetConfig must set generate_doc_lengths=True, or intra-document "
        "masking never activates and every proof attends to its neighbours"
    )


def test_the_reason_is_written_down_next_to_the_flag():
    """A bare True invites someone to remove it as noise."""
    src = (SCRIPTS / "train_platform.py").read_text()
    i = src.index("generate_doc_lengths=True")
    window = src[max(0, i - 700) : i]
    assert "attention" in window.lower(), "explain the flag where it is set"


def test_tokenize_corpus_terminates_every_document_with_eos():
    """Document boundaries are EOS. If the packer omits it, two proofs merge into one
    document and attend to each other."""
    class TinyTokenizer:
        def __call__(self, texts, *, add_special_tokens, return_offsets_mapping):
            assert not add_special_tokens and return_offsets_mapping
            return {
                "input_ids": [[1, 2] for _ in texts],
                "offset_mapping": [[(0, 1), (1, 2)] for _ in texts],
            }

    encoded, _ = encode_rows_batched(
        TinyTokenizer(), [{"text": "xx", "mask_end": 1}], eos_id=EOS
    )
    assert encoded[0].tolist() == [1, 2, EOS]
