"""Build primer vs control instance-source curricula for frontload-cl.

Schedules live here (not as separate ``curriculum/`` datasets). Both arms read the same
published shards under ``pretrain/frontload-cl-10b``; only post-warmup ordering differs.

Shared warmup (both arms)
-------------------------
~371M tokens of HQ main only (FineWeb-Edu main + FineWiki @ 5%), sized to the LR
warmup window. Same ``hq.split`` seed so both arms see the same warmup instances.
``CosWithWarmup`` in the train script is separate and identical for both arms.

Primer (after warmup)
---------------------
1. 100M contiguous SFT-like block.
2. Remaining HQ main + remaining 100M SFT-like, mixed uniformly.
3. Anneal 1B: FineWeb-Edu anneal + FineWiki @ 5% (no SFT-like).

Control (after warmup)
----------------------
1. Remaining HQ main + all 200M SFT-like, mixed uniformly.
2. Anneal 1B: same as primer.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from olmo_core.data import NumpyDatasetDType
from olmo_core.data.composable import (
    ConcatAndChunkInstanceSource,
    InstanceSource,
    MixingInstanceSource,
    NumpyDocumentSource,
    set_composable_seed,
)
from olmo_core.data.tokenizer import TokenizerConfig

from . import constants as C
from .corpus import Corpus, Refusal, Stage

log = logging.getLogger(__name__)

# ConcatAndChunk drops a trailing remainder (< seq_len). Mix targets that sit exactly on
# published pool sizes then fail under max_repetition_factor=1.0. A tiny bump absorbs that
# remainder without meaningfully changing the experiment budgets.
_POOL_EDGE_REPETITION = 1.05


def _require_sources(corpus: Corpus, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in corpus.paths_by_source or not corpus.paths_by_source[n]]
    if missing:
        have = ", ".join(sorted(corpus.paths_by_source)) or "none"
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"corpus missing source folder(s) {missing}; have: {have}",
        )


def _aligned_tokens(num_tokens: int, sequence_length: int) -> int:
    """Floor a token budget to a whole number of sequences."""
    return (num_tokens // sequence_length) * sequence_length


def _chunked(
    paths: List[str],
    *,
    label: str,
    tokenizer: TokenizerConfig,
    dtype: NumpyDatasetDType,
    sequence_length: int,
    work_dir: str,
) -> InstanceSource:
    doc = NumpyDocumentSource.Config(
        source_paths=paths,
        tokenizer=tokenizer,
        dtype=dtype,
        label=label,
    )
    return ConcatAndChunkInstanceSource.Config(
        sources=[doc],
        sequence_length=sequence_length,
        label=label,
    ).build(work_dir)


def _mix(
    specs: List[MixingInstanceSource.Spec],
    *,
    num_tokens: int,
    label: str,
    work_dir: str,
    seed: int,
    sequence_length: int,
) -> InstanceSource:
    return MixingInstanceSource(
        *specs,
        work_dir=work_dir,
        seed=seed,
        label=label,
        num_tokens=_aligned_tokens(num_tokens, sequence_length),
    )


def _split_finewiki(
    corpus: Corpus,
    *,
    sequence_length: int,
    work_dir: str,
) -> tuple[InstanceSource, InstanceSource]:
    """Disjoint FineWiki slices: 440M for pre-anneal HQ, 50M for anneal."""
    _require_sources(corpus, [C.SOURCE_FINEWIKI])
    wiki = _chunked(
        corpus.paths_by_source[C.SOURCE_FINEWIKI],
        label=C.SOURCE_FINEWIKI,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    ratio = C.HQ_FINEWIKI_MAIN / C.HQ_FINEWIKI_TOTAL
    return wiki.split(ratio, seed=C.DATA_SEED + 3)


def _hq_main_base(
    corpus: Corpus,
    *,
    wiki_main: InstanceSource,
    sequence_length: int,
    work_dir: str,
) -> InstanceSource:
    """FineWeb-Edu main + FineWiki main slice, mixed at 95/5."""
    _require_sources(corpus, [C.SOURCE_FINEWEB_MAIN])
    fw = _chunked(
        corpus.paths_by_source[C.SOURCE_FINEWEB_MAIN],
        label=C.SOURCE_FINEWEB_MAIN,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    return _mix(
        [
            MixingInstanceSource.Spec(
                source=fw,
                ratio=C.HQ_FINEWEB_RATIO,
                label=C.SOURCE_FINEWEB_MAIN,
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
            MixingInstanceSource.Spec(
                source=wiki_main,
                ratio=C.HQ_FINEWIKI_RATIO,
                label=C.SOURCE_FINEWIKI,
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
        ],
        num_tokens=C.HQ_PRE_ANNEAL,
        label="hq-main",
        work_dir=work_dir,
        seed=C.DATA_SEED,
        sequence_length=sequence_length,
    )


def _hq_anneal(
    corpus: Corpus,
    *,
    wiki_anneal: InstanceSource,
    sequence_length: int,
    work_dir: str,
) -> InstanceSource:
    _require_sources(corpus, [C.SOURCE_FINEWEB_ANNEAL])
    fw = _chunked(
        corpus.paths_by_source[C.SOURCE_FINEWEB_ANNEAL],
        label=C.SOURCE_FINEWEB_ANNEAL,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    return _mix(
        [
            MixingInstanceSource.Spec(
                source=fw,
                ratio=C.HQ_FINEWEB_RATIO,
                label=C.SOURCE_FINEWEB_ANNEAL,
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
            MixingInstanceSource.Spec(
                source=wiki_anneal,
                ratio=C.HQ_FINEWIKI_RATIO,
                label=C.SOURCE_FINEWIKI,
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
        ],
        num_tokens=C.HQ_ANNEAL,
        label="hq-anneal",
        work_dir=work_dir,
        seed=C.DATA_SEED + 1,
        sequence_length=sequence_length,
    )


def _sft_like(
    corpus: Corpus,
    *,
    sequence_length: int,
    work_dir: str,
    num_tokens: int = C.SFT_LIKE_TOTAL,
    label: str = "sft-like",
    seed: int = C.DATA_SEED + 2,
) -> InstanceSource:
    _require_sources(corpus, list(C.SFT_LIKE_SOURCES))
    specs = []
    for name in C.SFT_LIKE_SOURCES:
        src = _chunked(
            corpus.paths_by_source[name],
            label=name,
            tokenizer=corpus.tokenizer,
            dtype=corpus.dtype,
            sequence_length=sequence_length,
            work_dir=work_dir,
        )
        specs.append(
            MixingInstanceSource.Spec(
                source=src,
                ratio=C.SFT_LIKE_RATIOS[name],
                label=name,
                max_repetition_factor=_POOL_EDGE_REPETITION,
            )
        )
    return _mix(
        specs,
        num_tokens=num_tokens,
        label=label,
        work_dir=work_dir,
        seed=seed,
        sequence_length=sequence_length,
    )


def _split_hq_warmup(
    hq: InstanceSource,
    *,
    sequence_length: int,
) -> tuple[InstanceSource, InstanceSource]:
    """Carve the shared LR-warmup HQ window from ``hq-main``.

    Uses the actual mix size after seq-length flooring (not the nominal
    ``HQ_PRE_ANNEAL`` constant) and a fixed seed so primer and control get the
    same warmup instances and the same ``hq_rest``.
    """
    warmup_tokens = _aligned_tokens(C.WARMUP_TOKENS, sequence_length)
    warmup_ratio = warmup_tokens / hq.num_tokens
    if not (0.0 < warmup_ratio < 1.0):
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"warmup slice {warmup_tokens} does not fit in hq-main ({hq.num_tokens} tokens)",
        )
    return hq.split(warmup_ratio, seed=C.DATA_SEED + 10)


def build_primer_phases(
    corpus: Corpus,
    *,
    sequence_length: int = C.SEQ_LENGTH,
    work_dir: str,
) -> List[InstanceSource]:
    """Ordered phases: shared HQ warmup → SFT block → mixed rest → anneal."""
    set_composable_seed(C.DATA_SEED)
    wiki_main, wiki_anneal = _split_finewiki(
        corpus, sequence_length=sequence_length, work_dir=work_dir
    )
    hq = _hq_main_base(
        corpus,
        wiki_main=wiki_main,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    hq_warmup, hq_rest = _split_hq_warmup(hq, sequence_length=sequence_length)

    sft = _sft_like(corpus, sequence_length=sequence_length, work_dir=work_dir)
    primer_ratio = C.PRIMER_BLOCK / C.SFT_LIKE_TOTAL
    sft_block, sft_rest = sft.split(primer_ratio, seed=C.DATA_SEED + 11)

    main_tokens = hq_rest.num_tokens + sft_rest.num_tokens
    main = _mix(
        [
            MixingInstanceSource.Spec(
                source=hq_rest,
                ratio=hq_rest.num_tokens / main_tokens,
                label="hq-rest",
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
            MixingInstanceSource.Spec(
                source=sft_rest,
                ratio=sft_rest.num_tokens / main_tokens,
                label="sft-dispersed",
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
        ],
        num_tokens=main_tokens,
        label="main-post-primer",
        work_dir=work_dir,
        seed=C.DATA_SEED + 69,
        sequence_length=sequence_length,
    )
    anneal = _hq_anneal(
        corpus,
        wiki_anneal=wiki_anneal,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    phases = [hq_warmup, sft_block, main, anneal]
    _log_phases("primer", phases)
    return phases


def build_control_phases(
    corpus: Corpus,
    *,
    sequence_length: int = C.SEQ_LENGTH,
    work_dir: str,
) -> List[InstanceSource]:
    """Ordered phases: shared HQ warmup → flat HQ-rest+SFT → anneal."""
    set_composable_seed(C.DATA_SEED)
    wiki_main, wiki_anneal = _split_finewiki(
        corpus, sequence_length=sequence_length, work_dir=work_dir
    )
    hq = _hq_main_base(
        corpus,
        wiki_main=wiki_main,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    hq_warmup, hq_rest = _split_hq_warmup(hq, sequence_length=sequence_length)

    sft = _sft_like(corpus, sequence_length=sequence_length, work_dir=work_dir)
    post_tokens = hq_rest.num_tokens + sft.num_tokens
    post_warmup = _mix(
        [
            MixingInstanceSource.Spec(
                source=hq_rest,
                ratio=hq_rest.num_tokens / post_tokens,
                label="hq-rest",
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
            MixingInstanceSource.Spec(
                source=sft,
                ratio=sft.num_tokens / post_tokens,
                label="sft-like",
                max_repetition_factor=_POOL_EDGE_REPETITION,
            ),
        ],
        num_tokens=post_tokens,
        label="post-warmup-flat",
        work_dir=work_dir,
        seed=C.DATA_SEED + 420,
        sequence_length=sequence_length,
    )
    anneal = _hq_anneal(
        corpus,
        wiki_anneal=wiki_anneal,
        sequence_length=sequence_length,
        work_dir=work_dir,
    )
    phases = [hq_warmup, post_warmup, anneal]
    _log_phases("control", phases)
    return phases


def build_phases(
    arm: str,
    corpus: Corpus,
    *,
    sequence_length: int = C.SEQ_LENGTH,
    work_dir: str,
) -> List[InstanceSource]:
    if arm == "primer":
        return build_primer_phases(corpus, sequence_length=sequence_length, work_dir=work_dir)
    if arm == "control":
        return build_control_phases(corpus, sequence_length=sequence_length, work_dir=work_dir)
    raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown arm {arm!r}; use primer|control")


def _log_phases(arm: str, phases: List[InstanceSource]) -> None:
    total = sum(p.num_tokens for p in phases)
    log.info("%s curriculum: %d phases, %.3fB tokens total", arm, len(phases), total / 1e9)
    for i, phase in enumerate(phases):
        log.info("  phase[%d] %s: %.3fB tokens", i, phase.label or type(phase).__name__, phase.num_tokens / 1e9)


def visualize_arm(arm: str, corpus: Corpus, work_dir: str) -> None:
    for phase in build_phases(arm, corpus, work_dir=work_dir):
        phase.visualize()
