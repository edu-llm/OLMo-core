"""Immutable Skill-It inputs and the exact two-arm update math."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

RECIPE_PATH = Path(__file__).with_name("skillit_recipe.json")
RECIPE_SHA256 = "80842f2746f4e005f1a2771a32e4131f06dc22a28c15c89c5ce9b6688f6326e5"
SOURCE_COMMIT = "b435cbe9c352399fc4ab54b310f36d28f6c9746f"
OFFLINE_A_SOURCE_SHA256 = "e542e3e66f70c752110b51f60d1ee84f5f7860931dce5684e7a621f35dd74a21"
DERIVATIVE_FIT_SOURCE_SHA256 = "acb4754b46cd6a588dffce7e7ad0d9bd70b0188db010669a7cfccf8622da2bcc"

DATASET_ID = "pretrain/olmo-127b"
DATASET_VERSION = "v1"
DOMAINS = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)
FAMILIES = (
    "arc_challenge",
    "arc_easy",
    "mmlu_humanities",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_stem",
)
UPDATE_STEPS = (500, 875, 1250, 1625, 2000)
ETA = 0.2
W = 1.0


class SkillItContractError(RuntimeError):
    """A checked-in scientific input or arm selection changed."""


@dataclass(frozen=True)
class Arm:
    index: int
    arm_id: str
    a_mode: str
    wandb_project: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recipe(path: Path = RECIPE_PATH, *, verify_hash: bool = True) -> dict[str, Any]:
    if verify_hash and _sha256(path) != RECIPE_SHA256:
        raise SkillItContractError(f"immutable Skill-It recipe checksum mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    methodology = payload.get("methodology") or {}
    if methodology.get("source_commit") != SOURCE_COMMIT:
        raise SkillItContractError("unexpected methodology source commit")
    if methodology.get("completed_pilots_are_final") is not True:
        raise SkillItContractError("completed 60M pilots must remain final inputs")
    if methodology.get("offline_a_source_sha256") != OFFLINE_A_SOURCE_SHA256:
        raise SkillItContractError("offline A provenance hash changed")
    if methodology.get("derivative_fit_source_sha256") != DERIVATIVE_FIT_SOURCE_SHA256:
        raise SkillItContractError("derivative fit provenance hash changed")
    source = payload.get("data_source") or {}
    expected_source = {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "label_key": "source",
        "bucket": "edullm-data",
        "dtype": "uint32",
        "byte_order": "little",
        "header_bytes": 0,
    }
    if source != expected_source:
        raise SkillItContractError(f"unexpected immutable data source: {source!r}")
    if tuple(payload.get("domain_order") or ()) != DOMAINS:
        raise SkillItContractError("domain order changed")
    if tuple(payload.get("family_order") or ()) != FAMILIES:
        raise SkillItContractError("family order changed")
    skillit = payload.get("skillit") or {}
    if tuple(skillit.get("update_steps") or ()) != UPDATE_STEPS:
        raise SkillItContractError("Skill-It update schedule changed")
    if float(skillit.get("eta", -1)) != ETA or float(skillit.get("w", -1)) != W:
        raise SkillItContractError("Skill-It eta/w changed")
    offline = np.asarray(skillit.get("offline_a"), dtype=np.float64)
    if offline.shape != (len(DOMAINS), len(FAMILIES)) or not np.isfinite(offline).all():
        raise SkillItContractError(f"offline A has invalid shape/content: {offline.shape}")
    targets = skillit.get("derivative_fit") or {}
    if tuple(targets) != FAMILIES:
        raise SkillItContractError("derivative fit family order changed")
    for family in FAMILIES:
        target = targets[family]
        if tuple(target.get("t") or {}) != DOMAINS:
            raise SkillItContractError(f"{family}: derivative domain order changed")
    return payload


RECIPE = load_recipe()
ARMS = tuple(
    Arm(
        index=int(item["arm_index"]),
        arm_id=str(item["arm_id"]),
        a_mode=str(item["a_mode"]),
        wandb_project=str(item["wandb_project"]),
    )
    for item in RECIPE["arms"]
)
if tuple(arm.index for arm in ARMS) != (0, 1):
    raise SkillItContractError("arm indexes must be exactly 0 and 1")
if tuple((arm.arm_id, arm.a_mode, arm.wandb_project) for arm in ARMS) != (
    ("probe", "probe", "skillit-probe"),
    ("deriv", "derivative", "skillit-deriv"),
):
    raise SkillItContractError("Skill-It arm definitions changed")


def initial_weights() -> np.ndarray:
    weights = np.asarray(RECIPE["initial_weights"], dtype=np.float64)
    if weights.shape != (len(DOMAINS),) or np.any(weights < 0):
        raise SkillItContractError("invalid initial RegMix weights")
    return weights / weights.sum()


def offline_a() -> np.ndarray:
    return np.asarray(RECIPE["skillit"]["offline_a"], dtype=np.float64).copy()


def derivative_a(weights: Sequence[float]) -> np.ndarray:
    """A_ij(r) = max(0, -t_ij * (L_j(r) - c_j))."""
    r = np.asarray(weights, dtype=np.float64).reshape(-1)
    if r.shape != (len(DOMAINS),) or not np.isfinite(r).all():
        raise ValueError(f"weights must have shape {(len(DOMAINS),)}")
    targets: Mapping[str, Any] = RECIPE["skillit"]["derivative_fit"]
    out = np.zeros((len(DOMAINS), len(FAMILIES)), dtype=np.float64)
    for j, family in enumerate(FAMILIES):
        target = targets[family]
        c_j = float(target["c"])
        k_j = float(target["k"])
        t = np.asarray([float(target["t"][domain]) for domain in DOMAINS])
        loss_minus_c = k_j * math.exp(float(t @ r))
        out[:, j] = np.maximum(0.0, -t * loss_minus_c)
        if not math.isfinite(c_j + loss_minus_c):
            raise ValueError(f"{family}: non-finite mixing-law loss")
    return out


def adjacency(a_mode: str, weights: Sequence[float]) -> np.ndarray:
    if a_mode == "probe":
        return offline_a()
    if a_mode == "derivative":
        return derivative_a(weights)
    raise ValueError(f"unknown a_mode={a_mode!r}")


def update_weights(
    matrix: np.ndarray,
    losses: Sequence[float],
    *,
    eta: float = ETA,
    w: float = W,
) -> np.ndarray:
    """p_i(t+1) proportional to exp(eta * w * sum_j A_ij * L_j)."""
    a = np.asarray(matrix, dtype=np.float64)
    loss = np.asarray(losses, dtype=np.float64).reshape(-1)
    if a.shape != (len(DOMAINS), len(FAMILIES)):
        raise ValueError(f"A must have shape {(len(DOMAINS), len(FAMILIES))}")
    if loss.shape != (len(FAMILIES),) or not np.isfinite(loss).all():
        raise ValueError(f"losses must have shape {(len(FAMILIES),)} and be finite")
    logits = float(eta) * float(w) * (a @ loss)
    logits -= logits.max()
    unnormalized = np.exp(logits)
    return unnormalized / unnormalized.sum()


def family_losses(payload: Mapping[str, Any]) -> dict[str, float]:
    labels = payload.get("labels") or payload.get("task_loss_bpb") or {}
    label_map: Mapping[str, str] = RECIPE["family_labels"]
    missing = [label for label in label_map.values() if label not in labels]
    if missing:
        raise SkillItContractError(f"task-loss payload lacks Skill-It labels: {missing}")
    return {family: float(labels[label_map[family]]) for family in FAMILIES}


def arm_by_index(index: int) -> Arm:
    try:
        return ARMS[int(index)]
    except (IndexError, ValueError) as exc:
        raise SkillItContractError("arm index must be 0 (probe) or 1 (deriv)") from exc


__all__ = [
    "ARMS",
    "DATASET_ID",
    "DATASET_VERSION",
    "DOMAINS",
    "ETA",
    "FAMILIES",
    "RECIPE",
    "SkillItContractError",
    "UPDATE_STEPS",
    "W",
    "adjacency",
    "arm_by_index",
    "derivative_a",
    "family_losses",
    "initial_weights",
    "load_recipe",
    "offline_a",
    "update_weights",
]
