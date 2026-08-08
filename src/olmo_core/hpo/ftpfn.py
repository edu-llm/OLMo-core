"""
FT-PFN adapter: OLMo observations -> FT-PFN input, and a trusted surrogate wrapper.

Two responsibilities:

1. :func:`assemble_posterior_input` turns the controller's ``(unit_config, learning curve)``
   observations plus a set of query points into the tensors FT-PFN consumes, enforcing every
   part of the v0.0.1 input contract: unit-cube hyperparameters (<=10 dims), the fidelity
   coordinate ``t = tokens / target_tokens`` in ``(0, 1]``, the objective in ``[0, 1]``, finite
   values only, and at most 1000 distinct context curves. This is pure ``numpy`` and fully
   unit-tested without the optional dependency.

2. :class:`TrustedFTPFN` wraps the official ``ifbo.FTPFN`` posterior. It refuses to load an
   artifact whose checksum does not match :data:`olmo_core.hpo.artifacts.FTPFN_ARTIFACT_MD5`,
   avoiding the upstream unverified-download / ``weights_only=False`` path. ``ifbo`` and
   ``torch`` are imported lazily so importing this module never requires them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .artifacts import (
    FTPFN_ARTIFACT_MD5,
    FTPFN_ARTIFACT_SHA256,
    FTPFN_MAX_CONTEXT_CURVES,
    FTPFN_MAX_HP_DIMS,
    FTPFN_MODEL_VERSION,
)
from .objective import CENormalizer
from .types import CurvePoint

__all__ = [
    "ObservedCurve",
    "QueryPoint",
    "PosteriorInput",
    "Posterior",
    "assemble_posterior_input",
    "TrustedFTPFN",
]


@dataclass(frozen=True)
class ObservedCurve:
    """A context curve: one config's observed points, with a stable positive id."""

    curve_id: int
    unit_config: Tuple[float, ...]
    points: Tuple[CurvePoint, ...]


@dataclass(frozen=True)
class QueryPoint:
    """A point to forecast. ``curve_id`` is 0 for a config absent from the context."""

    curve_id: int
    unit_config: Tuple[float, ...]
    t: float


@dataclass(frozen=True)
class PosteriorInput:
    """Validated, unit-scaled FT-PFN input arrays."""

    context_ids: np.ndarray
    context_t: np.ndarray
    context_hp: np.ndarray
    context_y: np.ndarray
    query_ids: np.ndarray
    query_t: np.ndarray
    query_hp: np.ndarray


@runtime_checkable
class Posterior(Protocol):
    """Anything that can score query points' probability of improvement over a threshold."""

    def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
        """Return a ``(num_query,)`` array of P(y > threshold) in ``[0, 1]``."""
        ...


def _check_unit_hp(vec: Sequence[float], ndim: int) -> None:
    if len(vec) != ndim:
        raise ValueError(f"inconsistent hp dimensionality: expected {ndim}, got {len(vec)}")
    arr = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("hyperparameters must be finite")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("hyperparameters must lie in the unit cube [0, 1]")


def assemble_posterior_input(
    observed: Sequence[ObservedCurve],
    queries: Sequence[QueryPoint],
    *,
    target_tokens: int,
    normalizer: CENormalizer,
) -> PosteriorInput:
    """Assemble and validate FT-PFN input from observations and queries."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    # Determine hp dimensionality from the first available config.
    if observed:
        ndim = len(observed[0].unit_config)
    elif queries:
        ndim = len(queries[0].unit_config)
    else:
        raise ValueError("assemble_posterior_input needs at least one config")
    if ndim > FTPFN_MAX_HP_DIMS:
        raise ValueError(f"FT-PFN accepts at most {FTPFN_MAX_HP_DIMS} hp dims, got {ndim}")

    context_configs: dict[int, Tuple[float, ...]] = {}
    config_ids: dict[Tuple[float, ...], int] = {}
    for curve in observed:
        if not 1 <= curve.curve_id <= FTPFN_MAX_CONTEXT_CURVES:
            raise ValueError(
                f"context curve_id must be in [1, {FTPFN_MAX_CONTEXT_CURVES}], "
                f"got {curve.curve_id}"
            )
        config = tuple(float(value) for value in curve.unit_config)
        previous = context_configs.setdefault(curve.curve_id, config)
        if previous != config:
            raise ValueError(
                f"context curve_id {curve.curve_id} is bound to multiple configurations"
            )
        previous_id = config_ids.setdefault(config, curve.curve_id)
        if previous_id != curve.curve_id:
            raise ValueError(
                "real ifBO derives curve identity from hyperparameters, so one configuration "
                "cannot be bound to multiple context IDs"
            )

    distinct_context = set(context_configs)
    if len(distinct_context) > FTPFN_MAX_CONTEXT_CURVES:
        raise ValueError(
            f"FT-PFN context is capped at {FTPFN_MAX_CONTEXT_CURVES} curves, got "
            f"{len(distinct_context)}"
        )

    ctx_ids: List[int] = []
    ctx_t: List[float] = []
    ctx_hp: List[List[float]] = []
    ctx_y: List[float] = []
    for curve in observed:
        _check_unit_hp(curve.unit_config, ndim)
        for p in curve.points:
            t = p.tokens / target_tokens
            if not (0.0 < t <= 1.0):
                raise ValueError(f"fidelity t={t} outside (0, 1] for tokens={p.tokens}")
            ctx_ids.append(int(curve.curve_id))
            ctx_t.append(t)
            ctx_hp.append([float(v) for v in curve.unit_config])
            ctx_y.append(normalizer.to_ftpfn_y(p.ce))

    q_ids: List[int] = []
    q_t: List[float] = []
    q_hp: List[List[float]] = []
    for q in queries:
        _check_unit_hp(q.unit_config, ndim)
        if not 0 <= q.curve_id <= FTPFN_MAX_CONTEXT_CURVES:
            raise ValueError(
                f"query curve_id must be in [0, {FTPFN_MAX_CONTEXT_CURVES}], got {q.curve_id}"
            )
        if q.curve_id > 0:
            expected = context_configs.get(q.curve_id)
            if expected is None:
                raise ValueError(
                    f"query curve_id {q.curve_id} is absent from context; use reserved id 0"
                )
            if expected != tuple(float(value) for value in q.unit_config):
                raise ValueError(
                    f"query curve_id {q.curve_id} does not match its context configuration"
                )
        elif tuple(float(value) for value in q.unit_config) in config_ids:
            raise ValueError(
                "real ifBO would treat this id-0 query as a continuation because its "
                "configuration already appears in context"
            )
        if not (0.0 < q.t <= 1.0):
            raise ValueError(f"query fidelity t={q.t} outside (0, 1]")
        q_ids.append(int(q.curve_id))
        q_t.append(float(q.t))
        q_hp.append([float(v) for v in q.unit_config])

    return PosteriorInput(
        context_ids=np.asarray(ctx_ids, dtype=np.int64),
        context_t=np.asarray(ctx_t, dtype=np.float64),
        context_hp=np.asarray(ctx_hp, dtype=np.float64).reshape(-1, ndim),
        context_y=np.asarray(ctx_y, dtype=np.float64),
        query_ids=np.asarray(q_ids, dtype=np.int64),
        query_t=np.asarray(q_t, dtype=np.float64),
        query_hp=np.asarray(q_hp, dtype=np.float64).reshape(-1, ndim),
    )


class TrustedFTPFN:
    """Checksum-verified wrapper around the official ``ifbo.FTPFN`` posterior.

    :param artifact_path: Path to the extracted FT-PFN checkpoint.
    :param verify: If ``True`` (default), the file's MD5 must match the pinned artifact hash
        before it is loaded. This is the safe alternative to the upstream unverified download.
    :param device: Torch device string, defaults to CPU.
    """

    def __init__(self, artifact_path: str, *, verify: bool = True, device: str = "cpu") -> None:
        import os

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"FT-PFN artifact not found: {artifact_path}")
        if verify:
            self._verify_checksum(artifact_path)

        import torch  # lazy
        from ifbo.surrogate import FTPFN  # lazy; optional dependency

        self._device = torch.device(device)
        # The pinned upstream constructor always invokes its downloader and then loads whatever
        # appears under its target directory. Construct the pinned class without that side effect
        # so the exact file verified above is the file that determines this model's weights.
        self._model = FTPFN.__new__(FTPFN)
        torch.nn.Module.__init__(self._model)
        self._model.version = FTPFN_MODEL_VERSION
        self._model.target_path = Path(artifact_path).parent
        self._model.device = self._device
        self._model.model = torch.load(
            artifact_path,
            map_location=self._device,
            weights_only=False,
        )
        self._model.model.eval()
        self._torch = torch

    @staticmethod
    def _verify_checksum(path: str) -> None:
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
                sha256.update(chunk)
        md5_digest = md5.hexdigest()
        sha256_digest = sha256.hexdigest()
        if md5_digest != FTPFN_ARTIFACT_MD5 or sha256_digest != FTPFN_ARTIFACT_SHA256:
            raise ValueError(
                "FT-PFN artifact checksum mismatch: "
                f"MD5 {md5_digest} (expected {FTPFN_ARTIFACT_MD5}), "
                f"SHA-256 {sha256_digest} (expected {FTPFN_ARTIFACT_SHA256})"
            )

    def pi(
        self, x: PosteriorInput, threshold: float
    ) -> np.ndarray:  # pragma: no cover - needs ifbo
        if x.context_y.size == 0:
            # Installed ifBO 0.4.1 calls torch.stack([]) in tokenize(). A neutral prior keeps
            # cold-start allocation deterministic without pretending the model made a forecast.
            return np.full(x.query_hp.shape[0], 0.5, dtype=np.float64)

        from ifbo import Curve

        torch = self._torch
        # Build one context Curve per distinct id, and query Curves per query point.
        context: List[Curve] = []
        by_id: dict[int, list[int]] = {}
        for i, cid in enumerate(x.context_ids.tolist()):
            by_id.setdefault(cid, []).append(i)
        for cid, idxs in by_id.items():
            hp = torch.tensor(x.context_hp[idxs[0]], dtype=torch.float32, device=self._device)
            t = torch.tensor(x.context_t[idxs], dtype=torch.float32, device=self._device)
            y = torch.tensor(x.context_y[idxs], dtype=torch.float32, device=self._device)
            context.append(Curve(hyperparameters=hp, t=t, y=y))

        queries: List[Curve] = []
        for j in range(x.query_hp.shape[0]):
            hp = torch.tensor(x.query_hp[j], dtype=torch.float32, device=self._device)
            t = torch.tensor([x.query_t[j]], dtype=torch.float32, device=self._device)
            queries.append(Curve(hyperparameters=hp, t=t))

        results = self._model.predict(context=context, query=queries)
        best = torch.tensor(float(threshold), device=self._device)
        return np.array([float(r.pi(best).reshape(-1)[-1]) for r in results], dtype=np.float64)
