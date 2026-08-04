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
from .tasks import TaskStream

__all__ = ["BioTokenSource", "TaskTokenSource"]


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


class TaskTokenSource(TokenSource):
    """
    A :class:`~olmo_core.data.composable.TokenSource` over a generated reasoning slice.

    Thinner even than :class:`BioTokenSource`, because reasoning items are fixed width: locating a
    token is a division, so there is no offset index and nothing to cache.

    :param stream: The task stream to read from.
    :param work_dir: Local directory, required by the base class though nothing is cached here.
    :param label: Optional label for source visualisations.
    """

    def __init__(self, stream: TaskStream, *, work_dir, label: Optional[str] = None) -> None:
        super().__init__(work_dir=work_dir, label=label or f"reasoning:{stream.task.name}")
        self._stream = stream

    @property
    def stream(self) -> TaskStream:
        """The underlying stream, for tests and for reading its statistics."""
        return self._stream

    @property
    def num_tokens(self) -> int:
        """
        A whole number of items, so the *stream* never ends mid-item.

        That is a weaker guarantee than it looks, and weaker than this line used to claim. Rounding down
        protects the end of the stream only; the trainer asks
        :class:`~olmo_core.data.composable.ConcatAndChunkInstanceSource` for 512-token windows, and
        neither 24 nor 19 divides 512, so **3.1% of mano items and 3.5% of compare items are cut by an
        instance boundary** -- some of them mid-answer, leaving an instance that opens with an answer and
        no question. The streams are byte-identical across cells so the cuts are identical too, which
        makes this a uniform tax rather than a confound, but two things follow: ``answer_start`` is valid
        only *before* chunking, so an eval must locate answers itself rather than trust it; and a
        sequence length of 504 (or per-item padding with a label mask) would remove the tax entirely.
        """
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
        """A digest of the task and the slice size. A property, as ``SourceABC`` declares it."""
        return self._stream.fingerprint()

    def children(self):
        """No sub-sources: this one generates rather than composes."""
        return []
