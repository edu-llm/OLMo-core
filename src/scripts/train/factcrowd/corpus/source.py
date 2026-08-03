"""
The OLMo-core adapter: :class:`~factcrowd.corpus.stream.BioStream` as a ``TokenSource``.

Four methods, and nothing else. All the logic -- stream order, the offset index, token assembly -- is
in :mod:`factcrowd.corpus.stream`, which imports no ``torch`` and is therefore tested on any machine.
This file is what needs the full install, and it is kept this thin on purpose: an earlier module put
its only real logic behind a ``torch`` import, its tests were skipped, and a call that raised
``TypeError`` for every input passed both review and type-checking.

Everything above a token stream is OLMo-core's and is not reimplemented here.
``ConcatAndChunkInstanceSource`` turns this into fixed-length instances,
``MixingInstanceSource`` combines it with the reasoning slices at fixed absolute token counts,
``ComposableDataLoader`` shuffles and batches, and the trainer owns checkpointing, the learning-rate
schedule and evaluation callbacks.
"""

from typing import Optional

from olmo_core.data.composable import TokenSource
from olmo_core.data.composable.token_source import TokenRange

from .render import Renderer
from .stream import CHUNK, BioStream

__all__ = ["BioTokenSource"]


class BioTokenSource(TokenSource):
    """
    A :class:`~olmo_core.data.composable.TokenSource` over generated biographies.

    :param renderer: Supplies the biographies and their lengths.
    :param n_entities: Entities in the fact slice.
    :param exposures: Times each entity appears.
    :param work_dir: Local directory for the cached offset index.
    :param seed: Seeds the per-epoch entity permutation.
    :param chunk: Biographies per index entry. See :data:`~factcrowd.corpus.stream.CHUNK`.
    :param label: Optional label for source visualisations.
    """

    def __init__(
        self,
        renderer: Renderer,
        *,
        n_entities: int,
        exposures: int,
        work_dir,
        seed: int = 0,
        chunk: int = CHUNK,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(work_dir=work_dir, label=label)
        self._stream = BioStream(
            renderer,
            n_entities=n_entities,
            exposures=exposures,
            work_dir=self.work_dir,
            seed=seed,
            chunk=chunk,
            write_index=self.fs_local_rank == 0,
        )

    @property
    def stream(self) -> BioStream:
        """The underlying stream, for tests and for reading its statistics."""
        return self._stream

    @property
    def num_tokens(self) -> int:
        """Tokens in the slice."""
        return self._stream.num_tokens

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        """
        Tokens at ``[start_idx, end_idx)``.

        :param start_idx: First token, inclusive.
        :param end_idx: One past the last token.

        :returns: The token range.
        """
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)
        return {"input_ids": self._stream.tokens(start_idx, end_idx)}

    @property
    def fingerprint(self) -> str:
        """
        A digest of everything that determines the stream.

        A property rather than a method, because ``SourceABC`` declares it as one -- and OLMo-core
        reads it when composing sources, so a method here would be compared as a bound object and two
        different corpora would fingerprint alike.
        """
        return self._stream.fingerprint()

    def children(self):
        """No sub-sources: this one generates rather than composes."""
        return []
