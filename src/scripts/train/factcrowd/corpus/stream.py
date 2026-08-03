"""
The fact slice as a token stream: stream order, the offset index, and token assembly.

Deliberately free of ``torch`` and of OLMo-core's data package, so all of it is testable on a laptop.
:mod:`factcrowd.corpus.source` is the four-method adapter that presents this as an OLMo-core
:class:`~olmo_core.data.composable.TokenSource`. The split is not stylistic: an earlier module put its
only real logic behind a ``torch`` import, the tests were skipped, and a call that raised
``TypeError`` for every input passed review and type-checking. Whatever can be checked without a GPU
stack belongs on this side of the line.

Nothing is materialised. 560 million biographies at the largest remaining cell is about 39 billion
tokens, or 155 GB as uint32; the entity table they come from is under 200 MB, and the stream is a pure
function of ``(table, seed, biography index)``. What gets published for reproducibility is the table
and the config.

**Two indexing problems, and only one needed solving.**

Which entity and exposure is biography ``i``? Arithmetic: ``i`` splits into an epoch and a position,
and the position runs through a permutation of ``[0, n_entities)`` keyed by the epoch. So each entity
appears exactly once per epoch and therefore exactly ``exposures`` times overall -- the 200-exposure
invariant is structural rather than checked -- while the order within an epoch differs per epoch. The
permutation earns its keep: without it entity ``e``'s neighbours would be ``e-1`` and ``e+1`` in all
200 of its exposures, and a model could learn the sequence instead of the facts.

Where in the stream does biography ``i`` start? Biographies vary from 21 to 152 tokens
(:mod:`factcrowd.corpus.render`), so this needs an index -- but not a per-document one. A cumulative
count every :data:`CHUNK` biographies plus a short scan is about 8 MB at the largest cell, against the
ten or more gigabytes a per-document table would take. Building it is one vectorised pass over
template choices, and it is cached under the work directory keyed by fingerprint.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .entities import name_codes
from .render import Renderer

__all__ = ["CHUNK", "BioStream"]


CHUNK = 512
"""
Biographies per entry of the token-offset index.

Sets the trade between index size and lookup scan. At 512 the largest remaining cell needs about 8 MB
and a lookup scans at most 512 lengths -- one vectorised call, microseconds. At 64 the scan is shorter
and the index is 64 MB; at 4096 the index is 1 MB and the scan begins to cost more than the rendering
it precedes.
"""

_TOKEN_DTYPE = np.uint32
_RENDER_BATCH = 64
"""Biographies rendered per call while filling a range. Amortises the vectorised template lookup."""


class BioStream:
    """
    Biographies rendered on demand, in a stable order, as one contiguous token stream.

    :param renderer: Supplies the biographies and their lengths.
    :param n_entities: Entities in the fact slice. Must not exceed the renderer's table.
    :param exposures: Times each entity appears. 200 for every cell in the grid.
    :param work_dir: Local directory for the cached offset index.
    :param seed: Seeds the per-epoch entity permutation. Independent of the table's seed and the
        renderer's, so the order can be reshuffled without changing a fact or a phrasing.
    :param chunk: Biographies per index entry. See :data:`CHUNK`.
    :param write_index: Whether this process may write the cache. False on non-primary ranks.

    :raises OLMoConfigurationError: If any size is out of range, or if ``n_entities`` exceeds the
        renderer's table.
    """

    def __init__(
        self,
        renderer: Renderer,
        *,
        n_entities: int,
        exposures: int,
        work_dir: Union[str, os.PathLike],
        seed: int = 0,
        chunk: int = CHUNK,
        write_index: bool = True,
    ) -> None:
        if n_entities <= 0 or exposures <= 0 or chunk <= 0:
            raise OLMoConfigurationError(
                f"'n_entities', 'exposures' and 'chunk' must be positive, got "
                f"{n_entities}, {exposures} and {chunk}"
            )
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")
        if n_entities > renderer.n_table_entities:
            raise OLMoConfigurationError(
                f"the slice wants {n_entities:,} entities but its table holds "
                f"{renderer.n_table_entities:,}. Generate a bigger table, or lower the cell's demand."
            )

        self._renderer = renderer
        self._n_entities = n_entities
        self._exposures = exposures
        self._seed = seed
        self._chunk = chunk
        self._work_dir = Path(work_dir)
        self._write_index = write_index
        self._n_bios = n_entities * exposures
        self._chunk_offsets = self._load_or_build_index()
        self._num_tokens = int(self._chunk_offsets[-1])

    @property
    def num_tokens(self) -> int:
        """Tokens in the slice, summed from the real biography lengths rather than estimated."""
        return self._num_tokens

    @property
    def n_bios(self) -> int:
        """Biographies in the slice: ``n_entities * exposures``."""
        return self._n_bios

    @property
    def n_entities(self) -> int:
        """Entities the slice covers."""
        return self._n_entities

    @property
    def exposures(self) -> int:
        """Times each entity appears."""
        return self._exposures

    def tokens(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        The tokens at ``[start_idx, end_idx)``, rendering whatever is needed to cover them.

        The range may begin and end mid-biography, which is what ``ConcatAndChunkInstanceSource``
        expects -- it chunks a token stream without regard for document boundaries. Roughly one
        biography per instance is split across an instance boundary as a result. Every token is still
        present exactly once, so exposure counting is unaffected; the split costs attention across that
        one biography, equally in every cell, so it cannot bias the comparison.

        :param start_idx: First token, inclusive.
        :param end_idx: One past the last token.

        :returns: Exactly ``end_idx - start_idx`` tokens.

        :raises OLMoConfigurationError: If the range is empty, inverted or out of bounds.
        """
        if end_idx <= start_idx:
            raise OLMoConfigurationError(
                f"token range [{start_idx}, {end_idx}) is empty or inverted"
            )
        if start_idx < 0 or end_idx > self._num_tokens:
            raise OLMoConfigurationError(
                f"token range [{start_idx}, {end_idx}) is out of bounds for a slice of "
                f"{self._num_tokens:,} tokens"
            )

        first_bio, bio_token_start = self.locate(start_idx)
        wanted = end_idx - start_idx
        discard = start_idx - bio_token_start
        pieces: List[np.ndarray] = []
        collected = -discard
        cursor = first_bio
        while collected < wanted:
            batch = min(_RENDER_BATCH, self._n_bios - cursor)
            if batch <= 0:
                raise OLMoConfigurationError(
                    f"ran out of biographies at index {cursor:,} while filling tokens "
                    f"[{start_idx:,}, {end_idx:,}); the offset index and the renderer disagree"
                )
            entity_ids, exposures = self.assignment(np.arange(cursor, cursor + batch))
            out = np.empty(batch * self._renderer.max_tokens_per_bio, dtype=_TOKEN_DTYPE)
            lengths, _ = self._renderer.render_run(out, entity_ids, exposures)
            pieces.append(out[: int(lengths.sum())])
            collected += int(lengths.sum())
            cursor += batch

        stream = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return stream[discard : discard + wanted]

    def fingerprint(self) -> str:
        """
        A digest of everything that determines the token stream.

        Covers the renderer's fingerprint -- which covers the schema, the vocabulary and the templates
        -- plus the slice's size and its order seed. Two streams with the same fingerprint are
        byte-identical, which is what lets a run be reproduced from a config rather than from shards.

        :returns: A hex digest.
        """
        digest = hashlib.sha256()
        for field in (
            "factcrowd.BioStream.v1",
            self._renderer.fingerprint(),
            str(self._n_entities),
            str(self._exposures),
            str(self._seed),
            str(self._chunk),
        ):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    def assignment(self, bio_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Which entity and exposure each biography index refers to.

        ``i`` splits into ``epoch = i // n_entities`` and ``position = i % n_entities``; the position
        then runs through a permutation of ``[0, n_entities)`` keyed by the epoch. Every entity
        appears exactly once per epoch, and so exactly ``exposures`` times overall, while its
        neighbours differ in every epoch.

        :param bio_indices: Biography indices.

        :returns: Entity ids and exposure indices, both uint64.

        :raises OLMoConfigurationError: If any index is out of range.
        """
        indices = np.asarray(bio_indices)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self._n_bios):
            raise OLMoConfigurationError(
                f"biography index out of range for a slice of {self._n_bios:,}: "
                f"[{int(indices.min())}, {int(indices.max())}]"
            )
        epochs = (indices // self._n_entities).astype(np.uint64)
        positions = (indices % self._n_entities).astype(np.uint64)

        entity_ids = np.empty(indices.size, dtype=np.uint64)
        # One keyed permutation per epoch, so a range straddling an epoch boundary still costs a
        # handful of vectorised calls rather than one per biography.
        for epoch in np.unique(epochs):
            mask = epochs == epoch
            entity_ids[mask] = name_codes(
                positions[mask], name_space=self._n_entities, seed=self._seed + int(epoch) + 1
            )
        return entity_ids, epochs

    def locate(self, token_index: int) -> Tuple[int, int]:
        """
        Which biography contains ``token_index``, and where that biography starts.

        A search over the chunk index followed by a scan of at most ``chunk`` lengths.

        :param token_index: A token position in the slice.

        :returns: The biography index and its first token's position.
        """
        chunk_index = int(np.searchsorted(self._chunk_offsets, token_index, side="right")) - 1
        chunk_index = max(0, min(chunk_index, self._chunk_offsets.size - 2))
        bio = chunk_index * self._chunk
        position = int(self._chunk_offsets[chunk_index])

        span = min(self._chunk, self._n_bios - bio)
        entity_ids, exposures = self.assignment(np.arange(bio, bio + span))
        cumulative = position + np.cumsum(self._renderer.lengths_of(entity_ids, exposures))
        within = int(np.searchsorted(cumulative, token_index, side="right"))
        if within:
            position = int(cumulative[within - 1])
        return bio + within, position

    @property
    def index_path(self) -> Path:
        """Where the cached offset index lives, keyed by fingerprint so a stale one is never read."""
        return self._work_dir / f"factcrowd-offsets-{self.fingerprint()[:16]}.npy"

    def _load_or_build_index(self) -> np.ndarray:
        """
        Read the cached chunk-offset index, or build and cache it.

        Building is one vectorised pass over template choices -- lengths never touch a token buffer --
        and the cache filename carries the fingerprint, so a changed schema, template set or seed gets
        a different file rather than a silently reused index.

        :returns: Cumulative token counts at every ``chunk``-th biography, plus a final total.
        """
        path = self.index_path
        if path.is_file():
            return np.load(path, mmap_mode="r")

        n_chunks = (self._n_bios + self._chunk - 1) // self._chunk
        offsets = np.zeros(n_chunks + 1, dtype=np.int64)
        block = max(self._chunk, (1 << 22) // self._chunk * self._chunk)
        for block_start in range(0, self._n_bios, block):
            span = min(block, self._n_bios - block_start)
            entity_ids, exposures = self.assignment(np.arange(block_start, block_start + span))
            lengths = self._renderer.lengths_of(entity_ids, exposures)
            padded = np.zeros(
                ((span + self._chunk - 1) // self._chunk) * self._chunk, dtype=np.int64
            )
            padded[:span] = lengths
            per_chunk = padded.reshape(-1, self._chunk).sum(axis=1)
            first = block_start // self._chunk
            offsets[first + 1 : first + 1 + per_chunk.size] = per_chunk
        # offsets[c] becomes the tokens before chunk c, and offsets[-1] the total -- which is exactly
        # what locate() searches and what num_tokens reports.
        np.cumsum(offsets, out=offsets)

        if self._write_index:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, offsets)
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "fingerprint": self.fingerprint(),
                        "n_entities": self._n_entities,
                        "exposures": self._exposures,
                        "n_bios": self._n_bios,
                        "num_tokens": int(offsets[-1]),
                        "chunk": self._chunk,
                        "mean_tokens_per_bio": int(offsets[-1]) / self._n_bios,
                    },
                    indent=2,
                )
                + "\n"
            )
        return offsets
