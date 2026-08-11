"""Hybrid HPO controller entry point for the eduLLM platform.

This is the CPU-only controller process. It must **not** initialize one global process group,
and it must **not** be launched under ``torchrun --nproc-per-node=N``: instead it spawns an
isolated subprocess per trial *segment*. Segments default to ``WORLD_SIZE=1`` for compatibility;
an explicit ``worker_world_size`` may wrap every search segment in its own single-node
``torchrun``. OLMo has no in-process pause API, so freeze/thaw is checkpoint + process exit + a
freshly built trainer that loads that checkpoint.

Like ``train_on_corpus.py`` this file lives in ``.edullm/`` (that is what the platform image copies
and runs) and keeps its heavy imports lazy so it can be loaded and its guards tested without
``torch`` fully initialized or ``edullm_data`` installed.

Submitting a run
----------------
GPU submission goes through the platform, never a direct AWS/boto3 call. Price and validate with
``edullm check --json`` first, keep the dtype token explicit in the committed command (see
``.edullm/run.yaml`` and the repository's ``AGENTS.md``), then ``edullm submit``. Nothing in this
file contacts a cloud provider.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import copy
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

# Capacity-block nodes clone the branch into /work without installing it. Prefer the cloned
# source tree over the image's older installed package so additive HPO modules are importable.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# Environment variables that betray an outer (torchrun/torchelastic) launcher.
_TORCHRUN_MARKERS = (
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
)


def assert_controller_is_cpu_only(env: Mapping[str, str]) -> None:
    """Fail closed unless this process is a single, non-torchrun controller.

    The controller owns scheduling and spawns its own single-GPU workers; if it is itself one rank
    of a ``torchrun`` job then every worker would collide on devices and checkpoints.
    """
    world_size = env.get("WORLD_SIZE", "1")
    if world_size != "1":
        raise RuntimeError(
            f"the HPO controller must be CPU-only and single-process, but WORLD_SIZE={world_size}. "
            "Do not launch it with `torchrun --nproc-per-node=N`; it spawns its own "
            "WORLD_SIZE=1 trial workers."
        )
    present = [m for m in _TORCHRUN_MARKERS if m in env]
    if present:
        raise RuntimeError(
            f"detected outer launcher markers {present}; run the controller as a plain "
            "`python .edullm/hpo_on_corpus.py ...` process"
        )


@dataclass(frozen=True)
class SegmentLaunch:
    """A fully specified single-GPU trial segment launch."""

    env: Dict[str, str]
    checkpoint_dir: str
    hard_stop_tokens: int


def plan_segment(
    *,
    gpu: int,
    master_port: int,
    trial_id: str,
    checkpoint_root: str,
    loaded_tokens: int,
    target_tokens: int,
    quantum: int,
    base_env: Optional[Mapping[str, str]] = None,
) -> SegmentLaunch:
    """Lay out one segment: a world-size-one env, a namespaced checkpoint dir, an absolute stop.

    Delegates to :mod:`olmo_core.hpo.worker` (imported lazily) for the shared, tested primitives.
    """
    from olmo_core.hpo.worker import (
        next_absolute_hard_stop,
        trial_namespace,
        world_size_one_env,
    )

    env = world_size_one_env(gpu=gpu, master_port=master_port, base_env=base_env)
    checkpoint_dir = trial_namespace(checkpoint_root, trial_id)
    hard_stop = next_absolute_hard_stop(loaded_tokens, target_tokens, quantum)
    return SegmentLaunch(env=env, checkpoint_dir=checkpoint_dir, hard_stop_tokens=hard_stop.value)


def build_worker_argv(
    *,
    run_id: str,
    trial_id: str,
    param_dtype: str,
    checkpoint_dir: str,
    hard_stop_tokens: int,
    segment_spec: Optional[str] = None,
    python: Optional[str] = None,
) -> List[str]:
    """The argv for one single-GPU trial-worker subprocess.

    The dtype is named explicitly so the platform's precision guard can price it at check time
    rather than letting a kernel discover an unsupported format on an admitted machine.
    """
    python = python or sys.executable
    argv = [
        python,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpo_on_corpus.py"),
        run_id,
        "--run-segment",
        "--trial-id",
        trial_id,
        "--checkpoint-dir",
        checkpoint_dir,
        "--hard-stop-tokens",
        str(hard_stop_tokens),
        "--param-dtype",
        param_dtype,
    ]
    if segment_spec is not None:
        argv.extend(["--segment-spec", segment_spec])
    return argv


def build_distributed_worker_argv(
    worker_argv: List[str],
    *,
    world_size: int,
    master_port: int,
) -> List[str]:
    """Wrap one segment worker in a single-node ``torchrun`` launch."""
    if world_size <= 1:
        raise ValueError("distributed worker world size must be greater than one")
    if len(worker_argv) < 2:
        raise ValueError("worker argv must name a Python executable and script")
    return [
        worker_argv[0],
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        *worker_argv[1:],
    ]


def build_finalist_worker_argv(
    worker_argv: List[str],
    *,
    world_size: int,
    master_port: int,
) -> List[str]:
    """Wrap one segment worker in the compatibility finalist ``torchrun`` path."""
    if world_size <= 1:
        raise ValueError("finalist world size must be greater than one")
    if len(worker_argv) < 2:
        raise ValueError("worker argv must name a Python executable and script")
    return [
        worker_argv[0],
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        *worker_argv[1:],
    ]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid FT-PFN HPO controller")
    parser.add_argument("run_id", nargs="?", default=os.environ.get("EDULLM_RUN_ID", ""))
    parser.add_argument("--dry-run", action="store_true", help="Validate topology and exit.")
    parser.add_argument("--param-dtype", default="bfloat16")
    parser.add_argument(
        "--run-segment", action="store_true", help="Internal: run one trial segment."
    )
    parser.add_argument("--worker-world-size", type=int, default=None)
    parser.add_argument(
        "--run-proxy-cohort",
        action="store_true",
        help="Run the preregistered paired first-rung proxy/reference cohort.",
    )
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--hard-stop-tokens", type=int, default=None)
    parser.add_argument("--segment-spec", default=None)
    parser.add_argument("--segment-spec-payload", default=None)
    parser.add_argument("--observation-path", default=None)
    parser.add_argument("--controller-spec", default=os.environ.get("EDULLM_HPO_SPEC"))
    parser.add_argument("--controller-state", default=None)
    parser.add_argument("--proxy-spec", default=None)
    parser.add_argument("--reference-spec", default=None)
    parser.add_argument("--cohort-output", default=None)
    parser.add_argument(
        "--checkpoint-root",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR"),
    )
    return parser.parse_args(argv)


def _load_object(reference: str) -> Any:
    try:
        module_name, attribute = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError("object references must use 'module:attribute' syntax") from exc
    return getattr(importlib.import_module(module_name), attribute)


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
            raise ValueError(f"controller spec contains an unresolved environment value: {value}")
        return expanded
    return value


def _run_configured_segment(
    *,
    config,
    worker,
    hard_stop_tokens: int,
    heldout_metric: str,
    param_dtype: str,
    transition: Optional[Mapping[str, Any]] = None,
    fidelity: Optional[Mapping[str, Any]] = None,
    model_parameterization: Optional[Mapping[str, Any]] = None,
):
    from olmo_core.config import DType
    from olmo_core.hpo.worker import (
        configure_hpo_experiment,
        execute_segment,
        validate_umup_model,
    )
    from olmo_core.train import (
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.train_module import validate_precision_support
    from olmo_core.utils import seed_all

    config.train_module.dp_config.param_dtype = DType(param_dtype)
    diagnostics = configure_hpo_experiment(
        config,
        worker=worker,
        hard_stop_tokens=hard_stop_tokens,
        heldout_metric=heldout_metric,
        fidelity=fidelity,
        model_parameterization=model_parameterization,
    )
    validate_precision_support(config)
    prepare_training_environment()
    try:
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        if model_parameterization is not None and model_parameterization.get("kind") == "umup":
            from olmo_core.hpo.umup import apply_umup_model

            apply_umup_model(model, n_layers=int(config.model.n_layers))
            validate_umup_model(model)
        train_module = config.train_module.build(model)
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(
            dataset,
            dp_process_group=train_module.dp_process_group,
        )
        trainer = config.trainer.build(train_module, data_loader)
        return execute_segment(
            trainer,
            diagnostics=diagnostics,
            spec=worker.segment_spec(hard_stop_tokens, transition=transition),
            actual_global_batch_size=int(config.data_loader.global_batch_size),
        )
    finally:
        teardown_training_environment()


def _arm_contract(arm, posthoc_variant: Optional[str]):
    """Resolve a preregistered arm or an explicitly named post-hoc variant."""

    from olmo_core.hpo.arms import Arm, ablation_matrix

    if posthoc_variant is None:
        return ablation_matrix()[arm]
    if arm is Arm.NO_CENTAUR and posthoc_variant == "proxy_removed_after_failed_admission":
        # Use the exact no-proxy training contract while retaining the no-Centaur
        # controller ablation requested after the proxy failed admission.
        return {
            **ablation_matrix()[Arm.NO_PROXY],
            "llm_ratio": 0.0,
            "llm_scope": "none",
        }
    raise ValueError(f"unsupported post-hoc arm variant: {posthoc_variant!r}")


def _build_controller_from_spec(spec: Dict[str, Any]):
    from olmo_core.hpo.arms import Arm
    from olmo_core.hpo.artifacts import ensure_ftpfn_artifact
    from olmo_core.hpo.bttackler import (
        BTTCalibrationProfile,
        BTTConfig,
        BTTDiagnoser,
        BTTMode,
    )
    from olmo_core.hpo.centaur import CentaurOverlay, RequiredModelAdvisor
    from olmo_core.hpo.controller import (
        ControllerConfig,
        HpoController,
        PopulationRestartMode,
    )
    from olmo_core.hpo.ftpfn import TrustedFTPFN
    from olmo_core.hpo.ifbo import IfBOCandidateGenerator
    from olmo_core.hpo.ipbt import IPBTConfig, IPBTController
    from olmo_core.hpo.objective import CENormalizer
    from olmo_core.hpo.simulate import OracleFTPFN, RandomProposer, SyntheticObjective
    from olmo_core.hpo.types import SearchDim, SearchSpace

    search_space = SearchSpace(
        tuple(
            SearchDim(
                name=str(value["name"]),
                low=float(value["low"]),
                high=float(value["high"]),
                log=bool(value.get("log", False)),
            )
            for value in spec["search_space"]
        )
    )
    normalizer = CENormalizer(**spec["normalizer"])
    seed = int(spec.get("seed", 0))
    posterior_spec = spec["posterior"]
    if posterior_spec["kind"] == "ftpfn":
        artifact_path = posterior_spec.get("artifact_path")
        if artifact_path is None:
            artifact_path = str(ensure_ftpfn_artifact(posterior_spec.get("cache_dir")))
        posterior = TrustedFTPFN(
            artifact_path,
            device=posterior_spec.get("device", "cpu"),
        )
    elif posterior_spec["kind"] == "synthetic":
        objective = SyntheticObjective(
            optimum=tuple(float(x) for x in posterior_spec["optimum"]),
            seed=seed,
        )
        posterior = OracleFTPFN(objective, normalizer)
    else:
        raise ValueError(f"unknown posterior kind: {posterior_spec['kind']}")

    algorithm = spec.get("algorithm", "brainlift")
    is_brainlift = algorithm in ("brainlift", "brainlift_test")
    arm_name = spec.get("arm")
    arm = None if arm_name is None else Arm(arm_name)
    contract = None
    if arm is not None:
        contract = _arm_contract(arm, spec.get("posthoc_variant"))
        expected_fidelity = (
            {"kind": "exact"}
            if int(contract["freeze_first_n_blocks"]) == 0
            else {"kind": "frozen_layer", "train_last_k": 8}
        )
        if spec.get("fidelity", {"kind": "exact"}) != expected_fidelity:
            raise ValueError(f"arm {arm.value} does not match its freezing contract")
        parameterization = spec.get("model_parameterization", {})
        if contract["model_parameterization"] == "umup_190m_same_depth":
            if (
                parameterization.get("kind") != "umup"
                or parameterization.get("source_architecture") != "olmo2_370M"
                or int(parameterization.get("depth", -1)) != 16
            ):
                raise ValueError(f"arm {arm.value} requires the same-depth u-muP proxy")
        elif contract["model_parameterization"] == "stock_olmo2_190m":
            if (
                parameterization.get("kind") != "standard"
                or parameterization.get("architecture") != "olmo2_190M"
                or int(parameterization.get("depth", -1)) != 12
            ):
                raise ValueError(f"arm {arm.value} requires conventional stock olmo2_190M")
        elif contract["model_parameterization"] == "stock_olmoe_1b_7b":
            if (
                parameterization.get("kind") != "standard"
                or parameterization.get("architecture") != "olmoe_1B_7B"
                or int(parameterization.get("depth", -1)) != 16
            ):
                raise ValueError(f"arm {arm.value} requires stock olmoe_1B_7B")
        else:
            raise ValueError(
                f"arm {arm.value} has unknown model contract "
                f"{contract['model_parameterization']!r}"
            )
    if algorithm == "brainlift" and posterior_spec["kind"] != "ftpfn":
        raise ValueError("production Brainlift mode requires the real FT-PFN posterior")
    proposer_spec = spec.get("proposer", {"kind": "ifbo"})
    if proposer_spec["kind"] == "ifbo":
        proposer = IfBOCandidateGenerator(search_space.ndim, seed=seed)
    elif proposer_spec["kind"] == "random" and not is_brainlift:
        proposer = RandomProposer(search_space.ndim, seed=seed)
    elif proposer_spec["kind"] == "cma" and algorithm == "cma_control":
        from olmo_core.hpo.centaur import CMAESProposer

        proposer = CMAESProposer(
            dim=search_space.ndim,
            seed=seed,
            sigma=float(proposer_spec.get("sigma", 0.2)),
            population_size=int(proposer_spec["population_size"]),
        )
    elif proposer_spec["kind"] == "cma":
        raise ValueError("CMA-ES is replaced by FT-PFN ifBO in the Brainlift algorithm")
    else:
        raise ValueError(
            f"proposer {proposer_spec['kind']!r} is invalid for algorithm {algorithm!r}"
        )

    btt_values = dict(spec.get("btt", {}))
    if "mode" in btt_values:
        btt_values["mode"] = BTTMode(btt_values["mode"])
    calibration_values = btt_values.pop("calibration", None)
    calibration = (
        None
        if calibration_values is None
        else BTTCalibrationProfile(
            profile_version=calibration_values["profile_version"],
            completed_run_ids=tuple(calibration_values["completed_run_ids"]),
            thresholds=calibration_values["thresholds"],
        )
    )
    btt = BTTDiagnoser(BTTConfig(**btt_values), calibration=calibration)

    centaur_spec = spec.get("centaur")
    centaur = None
    advisor = None
    action_centaur = None
    action_advisor = None
    if is_brainlift:
        if spec.get("ipbt") is None:
            raise ValueError("Brainlift mode requires the IPBT population shell")
        centaur_disabled = contract is not None and float(contract["llm_ratio"]) == 0.0
        if centaur_spec is None and not centaur_disabled:
            raise ValueError("Brainlift mode requires 5.6 Sol Centaur")
        if centaur_disabled and centaur_spec is not None:
            raise ValueError(f"the {arm.value} arm must disable Centaur entirely")
        if centaur_spec is not None and (
            centaur_spec.get("scope") != "multi_action"
            or centaur_spec.get("model") != "gpt-5.6-sol"
            or float(centaur_spec.get("ratio", -1.0)) != 0.3
            or int(centaur_spec.get("warmup", -1)) != 0
        ):
            raise ValueError(
                "Brainlift Centaur must use multi_action gpt-5.6-sol at ratio 0.3 with zero warmup"
            )
    if centaur_spec is not None:
        overlay = CentaurOverlay(
            warmup=int(centaur_spec["warmup"]),
            ratio=float(centaur_spec["ratio"]),
        )
        advisor_factory = _load_object(centaur_spec["advisor_factory"])
        built_advisor = advisor_factory(**centaur_spec.get("advisor_kwargs", {}))
        if centaur_spec.get("model") is not None:
            built_advisor = RequiredModelAdvisor(built_advisor, str(centaur_spec["model"]))
        if centaur_spec.get("scope", "config_only") == "multi_action":
            action_centaur = overlay
            action_advisor = built_advisor
        else:
            centaur = overlay
            advisor = built_advisor

    ipbt_spec = spec.get("ipbt")
    ipbt = None
    ipbt_meta_proposer = None
    if ipbt_spec is not None:
        ipbt = IPBTController(IPBTConfig(**ipbt_spec))
        ipbt_meta_proposer = IfBOCandidateGenerator(search_space.ndim, seed=seed + 1)

    controller_values = spec["controller"]
    restart_mode = PopulationRestartMode(
        controller_values.get(
            "restart_mode",
            (
                PopulationRestartMode.BTT_AGGREGATE.value
                if is_brainlift
                else PopulationRestartMode.IPBT_REFERENCE.value
            ),
        )
    )
    if is_brainlift and restart_mode is not PopulationRestartMode.BTT_AGGREGATE:
        raise ValueError("Brainlift mode requires BTT aggregate IPBT restarts")
    if algorithm == "ipbt_reference" and restart_mode is not PopulationRestartMode.IPBT_REFERENCE:
        raise ValueError("IPBT reference mode requires the reference restart tracker")
    controller = HpoController(
        search_space,
        normalizer,
        posterior,
        proposer,
        btt,
        ControllerConfig(
            target_tokens=int(controller_values["target_tokens"]),
            quantum=int(controller_values["quantum"]),
            n_fidelity_bins=int(controller_values["n_fidelity_bins"]),
            worker_count=int(controller_values["worker_count"]),
            budget_tokens=int(controller_values["budget_tokens"]),
            checkpoint_root=str(controller_values["checkpoint_root"]),
            seed=seed,
            failure_penalty=float(controller_values.get("failure_penalty", 0.0)),
            restart_mode=restart_mode,
            btt_restart_fraction=float(controller_values.get("btt_restart_fraction", 0.5)),
            ensure_full_fidelity=bool(spec.get("require_final_winner", False)),
        ),
        centaur=centaur,
        advisor=advisor,
        ipbt=ipbt,
        ipbt_meta_proposer=ipbt_meta_proposer,
        cma_replacement_proposer=(
            RandomProposer(search_space.ndim, seed=seed + 2) if algorithm == "cma_control" else None
        ),
        action_centaur=action_centaur,
        action_advisor=action_advisor,
    )
    return controller


def _aligned_global_batch_size(
    requested: float,
    *,
    quantum: int,
    multiple: int = 1,
) -> int:
    """Choose the nearest valid batch size that divides every fidelity quantum."""

    if not math.isfinite(requested) or requested <= 0.0 or quantum <= 0 or multiple <= 0:
        raise ValueError("requested batch size, quantum, and multiple must be positive")
    divisors = []
    for candidate in range(1, math.isqrt(quantum) + 1):
        if quantum % candidate == 0:
            paired = quantum // candidate
            if candidate % multiple == 0:
                divisors.append(candidate)
            if paired != candidate and paired % multiple == 0:
                divisors.append(paired)
    if not divisors:
        raise ValueError(f"quantum {quantum} has no divisor aligned to multiple {multiple}")
    return min(divisors, key=lambda value: (abs(value - requested), value))


def _segment_payload(
    allocation,
    *,
    controller_spec: Dict[str, Any],
    finalist_continuation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base_batch = int(controller_spec["base_global_batch_size"])
    realized_hps = dict(controller_spec.get("fixed_hps", {}))
    realized_hps.update(allocation.realized_hps)
    multiplier = float(realized_hps.get("global_batch_mult", 1.0))
    rank_microbatch_size = int(
        controller_spec.get("factory_kwargs", {}).get("rank_microbatch_size", 1)
    )
    global_batch_size = _aligned_global_batch_size(
        base_batch * multiplier,
        quantum=int(
            controller_spec["controller"].get(
                "quantum", controller_spec["controller"]["target_tokens"]
            )
        ),
        multiple=rank_microbatch_size,
    )
    target_tokens = int(controller_spec["controller"]["target_tokens"])
    factory_kwargs = copy.deepcopy(controller_spec.get("factory_kwargs", {}))
    if finalist_continuation is not None:
        factory_kwargs["rank_microbatch_size"] = int(finalist_continuation["rank_microbatch_size"])
    payload = {
        "experiment_factory": controller_spec["experiment_factory"],
        "factory_kwargs": factory_kwargs,
        "target_tokens": target_tokens,
        "global_batch_size": global_batch_size,
        "realized_hps": realized_hps,
        "checkpoint_root": controller_spec["controller"]["checkpoint_root"],
        "search_validation_callback": controller_spec["search_validation_callback"],
        "untouched_evaluator": controller_spec["untouched_evaluator"],
        "heldout_metric": controller_spec["heldout_metric"],
        "checkpoint_ref": allocation.checkpoint_ref,
        "transition": allocation.transition,
        "fidelity": controller_spec.get("fidelity", {"kind": "exact"}),
        "model_parameterization": controller_spec.get(
            "model_parameterization", {"kind": "standard"}
        ),
    }
    if controller_spec.get("curriculum_identity") is not None:
        payload["curriculum_identity"] = copy.deepcopy(controller_spec["curriculum_identity"])
    if finalist_continuation is not None:
        payload["finalist_continuation"] = dict(finalist_continuation)
    payload["config_hash"] = _segment_config_hash(payload)
    return payload


def _segment_config_hash(payload: Mapping[str, Any]) -> str:
    contract = {
        "realized_hps": payload["realized_hps"],
        "global_batch_size": payload["global_batch_size"],
        "target_tokens": payload["target_tokens"],
        "fidelity": payload.get("fidelity", {"kind": "exact"}),
        "model_parameterization": payload.get("model_parameterization", {"kind": "standard"}),
        "experiment_factory": payload["experiment_factory"],
        "factory_kwargs": payload.get("factory_kwargs", {}),
        "curriculum_identity": payload.get("curriculum_identity"),
        "transition": payload.get("transition"),
        "finalist_continuation": payload.get("finalist_continuation"),
    }
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_capacity_block_backend(controller_spec: Mapping[str, Any]):
    from olmo_core.hpo.capacity_block import (
        CapacityBlockBackend,
        CapacityBlockConfig,
        GhWorkflowGateway,
    )

    values = controller_spec.get("capacity_block", {})
    if not isinstance(values, Mapping):
        raise ValueError("capacity_block configuration must be an object")
    config = CapacityBlockConfig(
        branch=str(values.get("branch", "")),
        repository=str(values.get("repository", "edu-llm/OLMo-core")),
        platform_repository=str(values.get("platform_repository", "edu-llm/platform")),
        checkpoint_root=str(controller_spec["controller"]["checkpoint_root"]),
        max_workers=int(controller_spec.get("max_workers", 8)),
        worker_world_size=int(controller_spec.get("worker_world_size", 8)),
        wandb_project=str(values.get("wandb_project", "hpo-probe")),
        reservation_id=str(values.get("reservation_id", "")),
        region=str(values.get("region", "us-east-2")),
        outputs_bucket=str(values.get("outputs_bucket", "edullm-block-outputs-us-east-2")),
        poll_interval_seconds=float(values.get("poll_interval_seconds", 30.0)),
        observation_sync_attempts=int(values.get("observation_sync_attempts", 4)),
    )
    gateway = GhWorkflowGateway(repository=config.platform_repository)
    return CapacityBlockBackend(config, gateway)


def _refresh_capacity_workers(
    controller,
    backend,
    *,
    heartbeat: Optional[Callable[[], None]] = None,
) -> tuple[int, ...]:
    """Wait for capacity, update the controller width, and return bound slots."""

    nodes = backend.wait_for_idle_nodes(heartbeat=heartbeat)
    count = backend.worker_count(nodes)
    controller.config.worker_count = count
    return tuple(nodes[:count])


def _segment_worker_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Return topology metadata augmented by the explicit remote-worker contract."""

    env = dict(os.environ)
    world_size = getattr(args, "worker_world_size", None)
    if world_size is not None:
        if world_size <= 1:
            raise ValueError("--worker-world-size must be greater than one")
        env["EDULLM_WORKER_WORLD_SIZE"] = str(world_size)
    return env


def _dispatch_allocations(
    *,
    allocations,
    controller_spec: Dict[str, Any],
    run_id: str,
    param_dtype: str,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_seconds: float = 60.0,
    capacity_backend=None,
    capacity_nodes=None,
):
    launch_backend = str(controller_spec.get("launch_backend", "local"))
    if launch_backend not in {"local", "capacity_block"}:
        raise ValueError(f"unsupported HPO launch_backend {launch_backend!r}")

    worker_world_size = int(controller_spec.get("worker_world_size", 1))
    if worker_world_size < 1:
        raise ValueError("worker_world_size must be at least one")
    if launch_backend == "capacity_block" and worker_world_size != 8:
        raise ValueError("capacity-block HPO requires worker_world_size=8")
    finalist_continuation = None
    if launch_backend == "local":
        from olmo_core.hpo.worker import (
            distributed_worker_env,
            finalist_distributed_env,
            trial_namespace,
            world_size_one_env,
        )

        default_gpu_count = int(controller_spec["controller"]["worker_count"]) * worker_world_size
        gpu_ids = controller_spec.get(
            "gpu_ids",
            list(range(default_gpu_count)),
        )
        required_gpu_count = len(allocations) * worker_world_size
        if len(gpu_ids) < required_gpu_count:
            raise ValueError("controller spec provides fewer GPU ids than allocations")
        if worker_world_size > 1 and len(set(gpu_ids[:required_gpu_count])) != required_gpu_count:
            raise ValueError("distributed allocations require distinct GPU ids")
        finalist_continuation = (
            _eligible_finalist_continuation(
                allocations,
                controller_spec=controller_spec,
                gpu_ids=gpu_ids,
            )
            if worker_world_size == 1
            else None
        )
    spec_dir = Path(
        controller_spec.get(
            "segment_spec_dir",
            os.path.join(tempfile.gettempdir(), f"edullm-hpo-{run_id}"),
        )
    )
    spec_dir.mkdir(parents=True, exist_ok=True)
    launches = []
    capacity_trials = []
    for index, allocation in enumerate(allocations):
        payload = _segment_payload(
            allocation,
            controller_spec=controller_spec,
            finalist_continuation=finalist_continuation,
        )
        spec_path = spec_dir / f"decision-{allocation.decision_id}.json"
        spec_path.write_text(json.dumps(payload, sort_keys=True))
        if launch_backend == "capacity_block":
            from olmo_core.hpo.capacity_block import CapacityTrial

            capacity_trials.append(
                CapacityTrial(
                    trial_id=allocation.trial_id,
                    decision_id=int(allocation.decision_id),
                    target_fidelity=int(allocation.target_fidelity),
                    payload=payload,
                )
            )
            continue
        checkpoint_dir = trial_namespace(
            controller_spec["controller"]["checkpoint_root"],
            allocation.trial_id,
        )
        argv = build_worker_argv(
            run_id=run_id,
            trial_id=allocation.trial_id,
            param_dtype=param_dtype,
            checkpoint_dir=checkpoint_dir,
            hard_stop_tokens=allocation.target_fidelity,
            segment_spec=str(spec_path),
        )
        master_port = int(controller_spec.get("master_port_base", 29500)) + index
        if worker_world_size > 1:
            accelerator_count = worker_world_size
            argv = build_distributed_worker_argv(
                argv,
                world_size=accelerator_count,
                master_port=master_port,
            )
            first_gpu = index * accelerator_count
            env = distributed_worker_env(
                [int(gpu) for gpu in gpu_ids[first_gpu : first_gpu + accelerator_count]],
                world_size=accelerator_count,
            )
        elif finalist_continuation is None:
            env = world_size_one_env(int(gpu_ids[index]), master_port)
            accelerator_count = 1
        else:
            accelerator_count = int(finalist_continuation["world_size"])
            argv = build_finalist_worker_argv(
                argv,
                world_size=accelerator_count,
                master_port=master_port,
            )
            env = finalist_distributed_env(
                [int(gpu) for gpu in gpu_ids[:accelerator_count]],
                world_size=accelerator_count,
            )
        launches.append((allocation, argv, env, accelerator_count))

    if launch_backend == "capacity_block":
        backend = capacity_backend or _build_capacity_block_backend(controller_spec)
        return backend.run(
            capacity_trials,
            run_id=run_id,
            idle_nodes=capacity_nodes,
            heartbeat=heartbeat,
        )

    from olmo_core.hpo.types import WorkerObservation

    def run_one(launch):
        allocation, argv, env, accelerator_count = launch
        started = time.perf_counter()
        completed = subprocess.run(
            argv,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        accelerator_seconds = (time.perf_counter() - started) * accelerator_count
        if completed.returncode != 0:
            error_text = completed.stderr.lower()
            if "out of memory" in error_text or "outofmemoryerror" in error_text:
                return WorkerObservation(
                    trial_id=allocation.trial_id,
                    tokens=allocation.target_fidelity,
                    heldout_ce=float("nan"),
                    train_ce_history=(float("nan"),),
                    grad_norm_history=(float("nan"),),
                    activation_ratio=None,
                    numeric_failure=True,
                    checkpoint_ref=None,
                    accelerator_seconds=accelerator_seconds,
                )
            stdout_tail = completed.stdout.strip()[-8000:]
            stderr_tail = completed.stderr.strip()[-8000:]
            raise RuntimeError(
                f"trial {allocation.trial_id} exited {completed.returncode}; "
                f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
            )
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise RuntimeError(f"trial {allocation.trial_id} emitted no result JSON")
        heldout_ce = float("nan") if payload["heldout_ce"] is None else float(payload["heldout_ce"])
        return WorkerObservation(
            trial_id=str(payload["trial_id"]),
            tokens=int(payload["tokens"]),
            heldout_ce=heldout_ce,
            train_ce_history=tuple(float(x) for x in payload["train_ce_history"]),
            grad_norm_history=tuple(float(x) for x in payload["grad_norm_history"]),
            activation_ratio=(
                None if payload["activation_ratio"] is None else float(payload["activation_ratio"])
            ),
            numeric_failure=bool(payload["numeric_failure"]),
            checkpoint_ref=payload["checkpoint_ref"],
            accelerator_seconds=accelerator_seconds,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(launches)) as executor:
        futures = [executor.submit(run_one, launch) for launch in launches]
        pending = set(futures)
        while pending:
            _, pending = concurrent.futures.wait(
                pending,
                timeout=heartbeat_interval_seconds,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            if pending and heartbeat is not None:
                heartbeat()
        return [future.result() for future in futures]


def _eligible_finalist_continuation(
    allocations,
    *,
    controller_spec: Mapping[str, Any],
    gpu_ids,
) -> Optional[Dict[str, Any]]:
    """Resolve an opt-in distributed continuation without changing ordinary HPO dispatch."""
    configured = controller_spec.get("finalist_continuation")
    if not isinstance(configured, Mapping) or not bool(configured.get("enabled", False)):
        return None
    if len(allocations) != 1:
        return None
    allocation = allocations[0]
    kind = getattr(allocation, "kind", None)
    kind_value = getattr(kind, "value", kind)
    transition = getattr(allocation, "transition", None)
    plain_or_rescue = transition is None or transition == {"full_fidelity_rescue": True}
    if (
        kind_value != "resume"
        or getattr(allocation, "checkpoint_ref", None) is None
        or not plain_or_rescue
    ):
        return None
    world_size = int(configured.get("world_size", 8))
    rank_microbatch_size = int(configured.get("rank_microbatch_size", 0))
    if world_size <= 1 or world_size > len(gpu_ids):
        raise ValueError("finalist continuation world size exceeds the configured GPU set")
    if rank_microbatch_size <= 0:
        raise ValueError("finalist continuation rank_microbatch_size must be positive")
    realized_hps = dict(controller_spec.get("fixed_hps", {}))
    realized_hps.update(allocation.realized_hps)
    global_batch_size = _aligned_global_batch_size(
        int(controller_spec["base_global_batch_size"])
        * float(realized_hps.get("global_batch_mult", 1.0)),
        quantum=int(
            controller_spec["controller"].get(
                "quantum", controller_spec["controller"]["target_tokens"]
            )
        ),
        multiple=int(controller_spec.get("factory_kwargs", {}).get("rank_microbatch_size", 1)),
    )
    if global_batch_size % (world_size * rank_microbatch_size):
        # Preserve the lineage batch exactly. An indivisible winner safely remains on one GPU.
        return None
    return {
        "enabled": True,
        "world_size": world_size,
        "rank_microbatch_size": rank_microbatch_size,
    }


def _persist_controller_log(controller, path: str, remote_root: Optional[str] = None) -> None:
    text = controller.log.to_jsonl()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, target)
    if remote_root is not None:
        from olmo_core.io import file_exists, join_path, upload

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        remote = str(
            join_path(
                remote_root,
                "controller",
                f"decisions-{len(controller.log):08d}-{digest}.jsonl",
            )
        )
        if not file_exists(remote):
            upload(target, remote, save_overwrite=False)


def _remote_controller_snapshots(remote_root: str):
    from olmo_core.io import join_path, list_directory

    controller_root = str(join_path(remote_root, "controller"))
    try:
        paths = list_directory(controller_root, include_dirs=False)
        snapshots = []
        for path in paths:
            match = re.fullmatch(r"decisions-(\d+)-[0-9a-f]+\.jsonl", os.path.basename(path))
            if match:
                snapshots.append((int(match.group(1)), str(path)))
        return snapshots
    except FileNotFoundError:
        return []


def _read_controller_text(path: str) -> str:
    from cached_path import cached_path

    from olmo_core.io import is_url

    if is_url(path):
        return cached_path(path).read_text()
    return Path(path).read_text()


def _load_controller_log(local_path: str, remote_root: Optional[str]) -> Optional[str]:
    if Path(local_path).exists():
        return _read_controller_text(local_path)
    if remote_root is None:
        return None
    snapshots = _remote_controller_snapshots(remote_root)
    if not snapshots:
        return None
    _, latest = max(snapshots, key=lambda item: item[0])
    return _read_controller_text(latest)


def _run_exact_retrain(
    controller,
    spec: Dict[str, Any],
    *,
    run_id: str,
    param_dtype: str,
):
    if spec.get("fidelity", {}).get("kind", "exact") == "exact":
        return None
    exact_factory = spec.get("exact_experiment_factory", spec["experiment_factory"])
    from olmo_core.hpo.types import ActionKind, Allocation, ProposalSource

    winner_id, unit_config, _ = controller.best()
    exact_spec = copy.deepcopy(spec)
    exact_spec["fidelity"] = {"kind": "exact"}
    exact_spec["experiment_factory"] = exact_factory
    exact_spec["factory_kwargs"] = spec.get("exact_factory_kwargs", spec.get("factory_kwargs", {}))
    allocation = Allocation(
        decision_id=0,
        kind=ActionKind.START,
        trial_id=f"{winner_id}-exact-retrain",
        parent_trial_id=None,
        unit_config=tuple(unit_config),
        realized_hps=controller.search_space.from_unit(unit_config),
        current_fidelity=0,
        target_fidelity=int(controller.config.target_tokens),
        checkpoint_ref=None,
        horizon=0,
        threshold=0.0,
        mfpi_score=0.0,
        tie_break=(0.0, "exact_retrain"),
        source=ProposalSource.IFBO,
    )
    results = _dispatch_allocations(
        allocations=[allocation],
        controller_spec=exact_spec,
        run_id=run_id,
        param_dtype=param_dtype,
    )
    if len(results) != 1 or results[0].numeric_failure:
        raise RuntimeError("fresh exact winner retraining failed")
    return results[0]


def _run_untouched_evaluation(
    controller,
    spec: Dict[str, Any],
    *,
    exact_result=None,
) -> None:
    completed = getattr(controller, "final_evaluation_completed", None)
    if not callable(completed) or completed():
        return
    best = getattr(controller, "best", None)
    if not callable(best):
        return
    try:
        trial_id, unit_config, search_validation_ce = best()
    except RuntimeError:
        return
    factory_ref = spec.get("final_evaluator_factory")
    if factory_ref is None:
        raise RuntimeError(
            "a final-fidelity winner exists but final_evaluator_factory is not configured"
        )
    record = controller.state().trials[trial_id]
    checkpoint_ref = (
        record.latest_checkpoint_ref if exact_result is None else exact_result.checkpoint_ref
    )
    evaluator = _load_object(factory_ref)
    result = evaluator(
        trial_id=trial_id,
        unit_config=list(unit_config),
        search_validation_ce=search_validation_ce,
        checkpoint_ref=checkpoint_ref,
        **spec.get("final_evaluator_kwargs", {}),
    )
    payload = {
        "trial_id": trial_id,
        "unit_config": list(unit_config),
        "search_validation_ce": search_validation_ce,
        "checkpoint_ref": checkpoint_ref,
        "exact_retrain": (None if exact_result is None else asdict(exact_result)),
        "result": result,
    }
    controller.record_final_evaluation(payload)


def _enforce_required_winner(controller, spec: Mapping[str, Any]) -> None:
    if not bool(spec.get("require_final_winner", False)):
        return
    completed = getattr(controller, "final_evaluation_completed", None)
    if not callable(completed) or not completed():
        raise RuntimeError("comparison run completed without a full-fidelity winner")


def _persist_study_result(
    controller,
    spec: Mapping[str, Any],
    evidence_decision,
    *,
    exact_result=None,
) -> None:
    """Persist the winner, top five, and measured single-A100 time for one arm."""

    result_path = spec.get("study_result_path")
    if result_path is None:
        raise RuntimeError("three-arm study specs require study_result_path")
    top = controller.top_candidates(5)
    if not top:
        raise RuntimeError("cannot persist study result without a full-fidelity candidate")

    def candidate_payload(candidate):
        trial_id, unit_config, search_validation_ce = candidate
        return {
            "trial_id": trial_id,
            "unit_config": list(unit_config),
            "hyperparameters": controller.search_space.from_unit(unit_config),
            "search_validation_ce": search_validation_ce,
        }

    state = controller.state()
    exact_retrain = None if exact_result is None else asdict(exact_result)
    if exact_retrain is None and state.final_evaluations:
        exact_retrain = state.final_evaluations[-1].get("exact_retrain")
    search_seconds = float(state.accelerator_seconds_charged)
    retrain_seconds = 0.0 if exact_retrain is None else float(exact_retrain["accelerator_seconds"])
    retrain_tokens = 0 if exact_retrain is None else int(exact_retrain["tokens"])
    if (
        not math.isfinite(search_seconds)
        or search_seconds < 0.0
        or not math.isfinite(retrain_seconds)
        or retrain_seconds < 0.0
        or retrain_tokens < 0
    ):
        raise RuntimeError("study result contains invalid accelerator or token accounting")
    total_seconds = search_seconds + retrain_seconds
    payload = {
        "arm": spec["arm"],
        "winner": candidate_payload(top[0]),
        "top_five_full_fidelity": [candidate_payload(candidate) for candidate in top],
        "search_a100_seconds": search_seconds,
        "exact_retrain_a100_seconds": retrain_seconds,
        "total_a100_seconds": total_seconds,
        "total_a100_hours": total_seconds / 3600.0,
        "search_tokens_charged": state.tokens_charged,
        "exact_retrain_tokens": retrain_tokens,
        "total_tokens_charged": state.tokens_charged + retrain_tokens,
        "frozen_ranking_status": evidence_decision.value,
        "trusted_for_selection": evidence_decision.value == "prune_promote",
    }
    if spec.get("curriculum_identity") is not None:
        payload["curriculum_identity"] = copy.deepcopy(spec["curriculum_identity"])
        payload["heldout_metric"] = spec.get("heldout_metric")
    target = Path(str(result_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(temporary, target)
    return payload


def _probe_durable_roots(
    spec: Mapping[str, Any], checkpoint_root: Optional[str]
) -> tuple[str, ...]:
    from olmo_core.hpo.wandb_probe import durable_storage_roots

    extra_roots = [
        str(spec.get("controller_snapshot_root", "")),
        str(spec["controller"]["checkpoint_root"]),
    ]
    return durable_storage_roots(checkpoint_dir=checkpoint_root, extra_roots=extra_roots)


def _open_hpo_probe_session(
    *,
    run_id: str,
    job_type: str,
    spec: Mapping[str, Any],
    checkpoint_root: Optional[str],
    tags: Optional[List[str]] = None,
):
    from olmo_core.hpo.wandb_probe import HpoProbeSession

    return HpoProbeSession.open(
        run_id=run_id,
        job_type=job_type,
        durable_roots=_probe_durable_roots(spec, checkpoint_root),
        arm=spec.get("arm"),
        config={
            "arm": spec.get("arm"),
            "algorithm": spec.get("algorithm"),
            "controller_spec": spec.get("controller"),
            "model_parameterization": spec.get("model_parameterization"),
            "fidelity": spec.get("fidelity"),
            "curriculum_identity": spec.get("curriculum_identity"),
            "heldout_metric": spec.get("heldout_metric"),
        },
        tags=tags,
    )


def _finalize_hpo_probe_session(
    probe,
    *,
    controller,
    state_path: Optional[str],
    segment_spec_dir: Optional[Path],
    exit_code: int,
) -> None:
    if probe is None:
        return
    if controller is not None:
        probe.log_controller(controller)
    if state_path:
        probe.mirror_ephemeral_path(
            state_path,
            artifact_name="controller-state",
            artifact_type="hpo-controller-state",
        )
    if segment_spec_dir is not None and segment_spec_dir.exists():
        probe.mirror_ephemeral_directory(
            segment_spec_dir,
            artifact_name="segment-specs",
            artifact_type="hpo-segment-spec",
        )
    probe.close(exit_code=exit_code)


def _write_json_artifact(
    path: str,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist a JSON artifact locally or through OLMo's URL backend."""

    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    from olmo_core.io import is_url

    if is_url(path):
        from olmo_core.io import upload

        with tempfile.TemporaryDirectory(prefix="hpo-proxy-evidence-") as temp_dir:
            local = Path(temp_dir) / "proxy-evidence.json"
            local.write_text(text)
            upload(local, path, save_overwrite=overwrite)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, target)


def _cohort_contract(spec: Mapping[str, Any]):
    from olmo_core.hpo.proxy import ProxyEvidenceContract

    values = dict(spec["proxy_evidence_contract"])
    values["config_ids"] = tuple(values["config_ids"])
    return ProxyEvidenceContract(**values)


def _validate_paired_cohort_specs(proxy_spec: Mapping[str, Any], reference_spec: Mapping[str, Any]):
    from olmo_core.hpo.arms import Arm

    if proxy_spec.get("arm") != Arm.FULL_ACRONYM_SOUP.value:
        raise ValueError("the paired cohort proxy spec must be full_acronym_soup")
    if reference_spec.get("arm") != Arm.NO_PROXY.value:
        raise ValueError("the paired cohort reference spec must be no_proxy")
    proxy_contract = _cohort_contract(proxy_spec)
    reference_contract = _cohort_contract(reference_spec)
    if proxy_contract != reference_contract:
        raise ValueError("proxy and reference specs must use the identical cohort contract")
    if proxy_spec.get("fidelity") != {"kind": "frozen_layer", "train_last_k": 8}:
        raise ValueError("the paired proxy must use half-layer freezing")
    proxy_parameterization = proxy_spec.get("model_parameterization", {})
    if (
        proxy_parameterization.get("kind") != "umup"
        or proxy_parameterization.get("source_architecture") != "olmo2_370M"
        or int(proxy_parameterization.get("depth", -1)) != 16
        or proxy_parameterization.get("backend") != "unit-scaling"
    ):
        raise ValueError("the paired proxy must use same-depth u-muP")
    if reference_spec.get("fidelity") != {"kind": "exact"}:
        raise ValueError("the paired reference must be fully trainable")
    reference_parameterization = reference_spec.get("model_parameterization", {})
    if reference_parameterization != {
        "kind": "standard",
        "architecture": "olmo2_190M",
        "depth": 12,
        "backend": "none",
    }:
        raise ValueError("the paired reference must be conventional stock olmo2_190M")
    for key in (
        "seed",
        "search_space",
        "normalizer",
        "posterior",
        "proposer",
        "btt",
        "ipbt",
        "factory_kwargs",
        "base_global_batch_size",
        "search_validation_callback",
        "heldout_metric",
    ):
        if proxy_spec.get(key) != reference_spec.get(key):
            raise ValueError(f"paired cohort specs differ in shared field {key!r}")
    if int(proxy_spec["controller"]["quantum"]) != proxy_contract.first_rung_tokens:
        raise ValueError("proxy controller quantum does not match the cohort first rung")
    if int(reference_spec["controller"]["quantum"]) != proxy_contract.first_rung_tokens:
        raise ValueError("reference controller quantum does not match the cohort first rung")
    if len(proxy_spec["search_space"]) != proxy_contract.search_dimensions:
        raise ValueError("proxy search space does not match the cohort dimensionality")
    if len(reference_spec["search_space"]) != proxy_contract.search_dimensions:
        raise ValueError("reference search space does not match the cohort dimensionality")
    admission = proxy_spec.get("proxy_admission")
    if admission is None:
        raise ValueError("the proxy spec must pre-register an admission gate")
    return proxy_contract, dict(admission)


def _cohort_allocations(spec: Mapping[str, Any], *, side: str, contract):
    from olmo_core.hpo.proxy import preregistered_cohort
    from olmo_core.hpo.types import (
        ActionKind,
        Allocation,
        ProposalSource,
        SearchDim,
        SearchSpace,
    )

    search_space = SearchSpace(
        tuple(
            SearchDim(
                name=str(value["name"]),
                low=float(value["low"]),
                high=float(value["high"]),
                log=bool(value.get("log", False)),
            )
            for value in spec["search_space"]
        )
    )
    allocations = []
    for decision_id, (config_id, unit_config) in enumerate(preregistered_cohort(contract).items()):
        allocations.append(
            Allocation(
                decision_id=decision_id,
                kind=ActionKind.START,
                trial_id=f"proxy-cohort-{side}-{config_id}",
                parent_trial_id=None,
                unit_config=tuple(unit_config),
                realized_hps=search_space.from_unit(unit_config),
                current_fidelity=0,
                target_fidelity=contract.first_rung_tokens,
                checkpoint_ref=None,
                horizon=0,
                threshold=0.0,
                mfpi_score=0.0,
                tie_break=(float(decision_id), config_id),
                source=ProposalSource.IFBO,
            )
        )
    return allocations


def _run_cohort_side(
    spec: Dict[str, Any],
    *,
    side: str,
    contract,
    run_id: str,
    param_dtype: str,
) -> Dict[str, Dict[str, float]]:
    allocations = _cohort_allocations(spec, side=side, contract=contract)
    worker_count = int(spec["controller"]["worker_count"])
    results = []
    for start in range(0, len(allocations), worker_count):
        results.extend(
            _dispatch_allocations(
                allocations=allocations[start : start + worker_count],
                controller_spec=spec,
                run_id=f"{run_id}-proxy-cohort-{side}",
                param_dtype=param_dtype,
            )
        )
    observations: Dict[str, Dict[str, float]] = {}
    prefix = f"proxy-cohort-{side}-"
    for result in results:
        if result.numeric_failure or not math.isfinite(result.heldout_ce):
            raise RuntimeError(f"paired cohort {side} observation {result.trial_id} failed")
        if not result.trial_id.startswith(prefix):
            raise RuntimeError("paired cohort worker returned an unexpected trial id")
        config_id = result.trial_id[len(prefix) :]
        observations[config_id] = {
            "tokens": result.tokens,
            "ce": result.heldout_ce,
            "accelerator_seconds": result.accelerator_seconds,
        }
    return observations


def _validated_existing_cohort_artifact(
    path: str,
    *,
    contract,
    admission_values: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Load a complete local cohort artifact and rederive its admission decision."""

    from olmo_core.hpo.proxy import ProxyAdmission, evaluate_paired_proxy_observations
    from olmo_core.io import is_url

    if is_url(path):
        return None
    target = Path(path)
    if not target.is_file():
        return None
    artifact = json.loads(target.read_text())
    expected_contract = {
        **asdict(contract),
        "config_ids": list(contract.config_ids),
    }
    if (
        artifact.get("schema_version") != 1
        or artifact.get("contract") != expected_contract
        or artifact.get("admission") != dict(admission_values)
    ):
        raise ValueError("existing proxy evidence does not match the preregistered cohort")
    metrics = evaluate_paired_proxy_observations(
        contract,
        proxy_observations=artifact.get("proxy_observations", {}),
        reference_observations=artifact.get("reference_observations", {}),
    )
    expected_metrics = {
        **asdict(metrics),
        "proxy_kind": metrics.proxy_kind.value,
    }
    decision = ProxyAdmission(**admission_values).decide(metrics)
    if artifact.get("metrics") != expected_metrics or artifact.get("decision") != decision.value:
        raise ValueError("existing proxy evidence metrics or decision failed local recomputation")
    return artifact


def run_proxy_cohort(args: argparse.Namespace) -> int:
    """Execute and persist the preregistered common first-rung paired cohort."""

    if not args.proxy_spec or not args.reference_spec:
        raise ValueError("--run-proxy-cohort requires --proxy-spec and --reference-spec")
    proxy_spec = _expand_environment(json.loads(Path(args.proxy_spec).read_text()))
    reference_spec = _expand_environment(json.loads(Path(args.reference_spec).read_text()))
    contract, admission_values = _validate_paired_cohort_specs(proxy_spec, reference_spec)
    output_path = args.cohort_output or proxy_spec.get("proxy_evidence_path")
    if output_path is None:
        raise ValueError("the proxy cohort requires --cohort-output or proxy_evidence_path")
    from olmo_core.hpo.umup import require_official_umup_forward

    require_official_umup_forward()
    probe = _open_hpo_probe_session(
        run_id=args.run_id,
        job_type="proxy-cohort",
        spec=proxy_spec,
        checkpoint_root=args.checkpoint_root,
        tags=["proxy-cohort"],
    )
    exit_code = 0
    controller = None
    try:
        existing_artifact = _validated_existing_cohort_artifact(
            str(output_path),
            contract=contract,
            admission_values=admission_values,
        )
        if existing_artifact is not None:
            probe.record_proxy_cohort(existing_artifact, output_path=output_path)
            print(
                json.dumps(
                    {
                        "cohort_id": contract.cohort_id,
                        "decision": existing_artifact["decision"],
                        "output_path": str(output_path),
                        "reused_existing_evidence": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        reference_observations = _run_cohort_side(
            reference_spec,
            side="reference",
            contract=contract,
            run_id=args.run_id,
            param_dtype=args.param_dtype,
        )
        proxy_observations = _run_cohort_side(
            proxy_spec,
            side="proxy",
            contract=contract,
            run_id=args.run_id,
            param_dtype=args.param_dtype,
        )
        from olmo_core.hpo.proxy import (
            ProxyAdmission,
            evaluate_paired_proxy_observations,
        )

        metrics = evaluate_paired_proxy_observations(
            contract,
            proxy_observations=proxy_observations,
            reference_observations=reference_observations,
        )
        decision = ProxyAdmission(**admission_values).decide(metrics)
        artifact = {
            "schema_version": 1,
            "contract": {
                **asdict(contract),
                "config_ids": list(contract.config_ids),
            },
            "admission": admission_values,
            "proxy_observations": proxy_observations,
            "reference_observations": reference_observations,
            "metrics": {
                **asdict(metrics),
                "proxy_kind": metrics.proxy_kind.value,
            },
            "decision": decision.value,
        }
        _write_json_artifact(str(output_path), artifact)
        probe.record_proxy_cohort(artifact, output_path=output_path)
        print(
            json.dumps(
                {
                    "cohort_id": contract.cohort_id,
                    "decision": decision.value,
                    "output_path": str(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception:
        exit_code = 1
        raise
    finally:
        for side in ("reference", "proxy"):
            segment_spec_dir = Path(
                proxy_spec.get(
                    "segment_spec_dir",
                    os.path.join(
                        tempfile.gettempdir(), f"edullm-hpo-{args.run_id}-proxy-cohort-{side}"
                    ),
                )
            )
            probe.mirror_ephemeral_directory(
                segment_spec_dir,
                artifact_name=f"segment-specs-{side}",
                artifact_type="hpo-segment-spec",
            )
        _finalize_hpo_probe_session(
            probe,
            controller=controller,
            state_path=None,
            segment_spec_dir=None,
            exit_code=exit_code,
        )


def _validate_evidence_gates(spec: Dict[str, Any]):
    from olmo_core.hpo.proxy import AdmitDecision

    proxy_evidence = spec.get("proxy_evidence")
    proxy_evidence_path = spec.get("proxy_evidence_path")
    if proxy_evidence is not None and proxy_evidence_path is not None:
        raise ValueError("configure proxy_evidence or proxy_evidence_path, not both")
    if proxy_evidence_path is not None:
        try:
            proxy_evidence = json.loads(_read_controller_text(str(proxy_evidence_path)))
        except FileNotFoundError:
            proxy_evidence = None
    fidelity_kind = spec.get("fidelity", {}).get("kind", "exact")
    decision = AdmitDecision.PRUNE_PROMOTE
    contract = None
    if fidelity_kind == "frozen_layer":
        from olmo_core.hpo.proxy import ProxyEvidenceContract

        contract_values = spec.get("proxy_evidence_contract")
        if contract_values is None:
            raise ValueError("frozen_layer fidelity requires a pre-registered paired cohort")
        contract_values = dict(contract_values)
        contract_values["config_ids"] = tuple(contract_values["config_ids"])
        contract = ProxyEvidenceContract(**contract_values)
        if contract.first_rung_tokens != int(spec["controller"]["quantum"]):
            raise ValueError("proxy evidence cohort must be evaluated at the common first rung")
        if proxy_evidence is None:
            if spec.get("frozen_ranking_policy") != "reporting_only_until_admitted":
                raise ValueError("pending frozen evidence must remain reporting-only")
            decision = AdmitDecision.REPORTING_ONLY
    elif fidelity_kind != "exact":
        raise ValueError(f"unknown training fidelity kind: {fidelity_kind}")

    if proxy_evidence is not None:
        from olmo_core.hpo.proxy import (
            ProxyAdmission,
            evaluate_paired_proxy_observations,
        )

        if fidelity_kind != "frozen_layer" or contract is None:
            raise ValueError("proxy evidence is only valid for the frozen_layer fidelity")
        expected_contract = asdict(contract)
        expected_contract["config_ids"] = list(contract.config_ids)
        if proxy_evidence.get("schema_version") != 1:
            raise ValueError("proxy evidence has an unsupported schema version")
        if proxy_evidence.get("contract") != expected_contract:
            raise ValueError("proxy evidence does not match the pre-registered cohort contract")
        admission_values = spec.get("proxy_admission")
        if admission_values is None or proxy_evidence.get("admission") != admission_values:
            raise ValueError("proxy evidence does not match the pre-registered admission gate")
        try:
            metrics = evaluate_paired_proxy_observations(
                contract,
                proxy_observations=proxy_evidence["proxy_observations"],
                reference_observations=proxy_evidence["reference_observations"],
            )
        except KeyError as error:
            raise ValueError(f"proxy evidence is missing paired field {error.args[0]!r}") from error
        decision = ProxyAdmission(**admission_values).decide(metrics)
        if proxy_evidence.get("decision") != decision.value:
            raise ValueError("persisted proxy decision does not match recomputed evidence")
        if (
            decision is not AdmitDecision.PRUNE_PROMOTE
            and spec.get("frozen_ranking_policy") != "reporting_only_until_admitted"
        ):
            raise ValueError("proxy evidence did not pass the preregistered admission gate")

    return decision


def run_controller(args: argparse.Namespace) -> int:
    """Run the CPU controller process."""
    if not args.controller_spec:
        raise ValueError("--controller-spec is required in controller mode")
    spec = _expand_environment(json.loads(Path(args.controller_spec).read_text()))
    if args.checkpoint_root:
        required_prefix = args.checkpoint_root.rstrip("/") + "/"
        configured_root = str(spec["controller"]["checkpoint_root"])
        if not configured_root.startswith(required_prefix):
            raise ValueError("controller checkpoint_root must be under --checkpoint-root")
    evidence_decision = _validate_evidence_gates(spec)
    if (
        spec.get("fidelity", {}).get("kind") == "frozen_layer"
        and evidence_decision.value != "prune_promote"
    ):
        raise RuntimeError(
            "frozen-layer rankings are reporting-only until the pre-registered common "
            "first-rung cohort passes the proxy admission gate"
        )
    if spec.get("model_parameterization", {}).get("kind") == "umup":
        from olmo_core.hpo.umup import require_official_umup_forward

        require_official_umup_forward()
    capacity_backend = None
    capacity_nodes = None
    if spec.get("launch_backend", "local") == "capacity_block":
        capacity_backend = _build_capacity_block_backend(spec)
        capacity_nodes = capacity_backend.wait_for_idle_nodes()
        spec["controller"]["worker_count"] = capacity_backend.worker_count(capacity_nodes)
    controller = _build_controller_from_spec(spec)
    state_path = args.controller_state or spec.get(
        "controller_state_path",
        os.path.join(tempfile.gettempdir(), f"edullm-hpo-{args.run_id}.jsonl"),
    )
    segment_spec_dir = Path(
        spec.get(
            "segment_spec_dir",
            os.path.join(tempfile.gettempdir(), f"edullm-hpo-{args.run_id}"),
        )
    )
    remote_root = spec.get("controller_snapshot_root")
    probe = _open_hpo_probe_session(
        run_id=args.run_id,
        job_type="controller",
        spec=spec,
        checkpoint_root=args.checkpoint_root,
    )
    exit_code = 0
    study_payload = None
    study_result_path = spec.get("study_result_path")
    try:
        restored_log = _load_controller_log(state_path, remote_root)
        if restored_log is not None:
            controller.restore_log(restored_log)
        retry_tell = getattr(controller, "retry_pending_tell", None)
        if callable(retry_tell):
            retry_tell()
        pending_method = getattr(controller, "pending_allocations", None)
        pending_allocations = pending_method() if callable(pending_method) else []
        if pending_allocations:
            results = _dispatch_allocations(
                allocations=pending_allocations,
                controller_spec=spec,
                run_id=args.run_id,
                param_dtype=args.param_dtype,
                heartbeat=probe.log_heartbeat,
                capacity_backend=capacity_backend,
                capacity_nodes=capacity_nodes,
            )
            controller.ingest(results)
            _persist_controller_log(controller, state_path, remote_root)
            probe.log_controller(controller)
        max_rounds = int(spec.get("max_rounds", 10_000))
        for _ in range(max_rounds):
            if capacity_backend is not None:
                capacity_nodes = _refresh_capacity_workers(
                    controller,
                    capacity_backend,
                    heartbeat=probe.log_heartbeat,
                )
            try:
                allocations = controller.propose_round()
            except Exception:
                _persist_controller_log(controller, state_path, remote_root)
                probe.log_controller(controller)
                raise
            _persist_controller_log(controller, state_path, remote_root)
            probe.log_controller(controller)
            if not allocations:
                break
            results = _dispatch_allocations(
                allocations=allocations,
                controller_spec=spec,
                run_id=args.run_id,
                param_dtype=args.param_dtype,
                heartbeat=probe.log_heartbeat,
                capacity_backend=capacity_backend,
                capacity_nodes=capacity_nodes,
            )
            controller.ingest(results)
            _persist_controller_log(controller, state_path, remote_root)
            probe.log_controller(controller)
        final_completed_method = getattr(controller, "final_evaluation_completed", None)
        final_completed = (
            bool(final_completed_method()) if callable(final_completed_method) else False
        )
        exact_result = None
        if not final_completed:
            exact_result = _run_exact_retrain(
                controller,
                spec,
                run_id=args.run_id,
                param_dtype=args.param_dtype,
            )
            _run_untouched_evaluation(
                controller,
                spec,
                exact_result=exact_result,
            )
        _persist_controller_log(controller, state_path, remote_root)
        probe.log_controller(controller)
        _enforce_required_winner(controller, spec)
        if spec.get("arm") is not None:
            study_payload = _persist_study_result(
                controller,
                spec,
                evidence_decision,
                exact_result=exact_result,
            )
            if study_result_path is not None:
                probe.record_study_result(study_payload, study_result_path)
        return 0
    except Exception:
        exit_code = 1
        raise
    finally:
        _finalize_hpo_probe_session(
            probe,
            controller=controller,
            state_path=state_path,
            segment_spec_dir=segment_spec_dir,
            exit_code=exit_code,
        )


def run_segment(args: argparse.Namespace) -> int:
    """Run one isolated OLMo trial segment."""
    segment_sources = [
        bool(args.segment_spec),
        bool(args.segment_spec_payload),
    ]
    if sum(segment_sources) != 1:
        raise ValueError("segment mode requires exactly one segment-spec input")
    if not args.trial_id or not args.checkpoint_dir or args.hard_stop_tokens is None:
        raise ValueError("segment mode requires trial id, checkpoint directory, and hard stop")
    encoded_spec = args.segment_spec_payload
    if encoded_spec:
        try:
            decoded_spec = base64.b64decode(encoded_spec, altchars=b"-_", validate=True)
            spec = json.loads(decoded_spec.decode("utf-8"))
        except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("encoded segment spec is not valid JSON") from exc
    else:
        spec = json.loads(Path(args.segment_spec).read_text())
    model_parameterization = spec.get("model_parameterization", {"kind": "standard"})
    if model_parameterization.get("kind") == "umup":
        from olmo_core.hpo.umup import require_official_umup_forward

        require_official_umup_forward()
    factory = _load_object(spec["experiment_factory"])
    config = factory(**spec.get("factory_kwargs", {}))
    expected_curriculum_identity = spec.get("curriculum_identity")
    if (
        expected_curriculum_identity is not None
        and getattr(config, "curriculum_identity", None) != expected_curriculum_identity
    ):
        raise ValueError("resolved curriculum identity does not match the segment artifact")
    fidelity = spec.get("fidelity", {"kind": "exact"})

    from olmo_core.hpo.objective import EvaluatorGate
    from olmo_core.hpo.worker import WorkerConfig, trial_namespace

    worker = WorkerConfig(
        trial_id=args.trial_id,
        gpu=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
        target_tokens=int(spec["target_tokens"]),
        quantum=int(args.hard_stop_tokens),
        global_batch_size=int(spec["global_batch_size"]),
        realized_hps={key: float(value) for key, value in spec["realized_hps"].items()},
        checkpoint_root=str(spec["checkpoint_root"]),
        evaluator_gate=EvaluatorGate(
            search_validation=str(spec["search_validation_callback"]),
            untouched=str(spec["untouched_evaluator"]),
        ),
        curriculum_identity=expected_curriculum_identity,
    )
    expected_save_folder = trial_namespace(worker.checkpoint_root, worker.trial_id)
    if args.checkpoint_dir != expected_save_folder:
        raise ValueError(
            f"segment checkpoint namespace {args.checkpoint_dir!r} != {expected_save_folder!r}"
        )
    expected_hash = spec.get("config_hash")
    if expected_hash is not None and _segment_config_hash(spec) != expected_hash:
        raise ValueError("segment trial config hash mismatch")
    if spec.get("checkpoint_ref") is not None:
        config.trainer.load_path = spec["checkpoint_ref"]
        reset_optim = bool((spec.get("transition") or {}).get("optimizer_reset", False))
        config.trainer.load_optim_state = not reset_optim
    result = _run_configured_segment(
        config=config,
        worker=worker,
        hard_stop_tokens=int(args.hard_stop_tokens),
        heldout_metric=str(spec["heldout_metric"]),
        param_dtype=args.param_dtype,
        transition=spec.get("transition"),
        fidelity=fidelity,
        model_parameterization=model_parameterization,
    )
    payload = asdict(result)
    if not math.isfinite(payload["heldout_ce"]):
        payload["heldout_ce"] = None
    from olmo_core.hpo.worker import should_emit_worker_result

    if should_emit_worker_result(_segment_worker_environment(args)):
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        if args.observation_path:
            _write_json_artifact(args.observation_path, payload, overwrite=True)
            print(f"EDULLM_HPO_OBSERVATION={encoded}", flush=True)
        print(encoded, flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from olmo_core.hpo.runtime_secrets import load_runtime_secrets

    load_runtime_secrets()
    args = _parse_args(argv)
    if args.run_segment and args.run_proxy_cohort:
        raise ValueError("--run-segment and --run-proxy-cohort are mutually exclusive")
    # Trial segments are world-size-one unless explicitly marked as a distributed worker.
    if args.run_segment:
        from olmo_core.hpo.worker import assert_worker_topology

        assert_worker_topology(_segment_worker_environment(args))
    else:
        assert_controller_is_cpu_only(os.environ)
    if args.dry_run:
        return 0
    if args.run_segment:
        return run_segment(args)
    if args.run_proxy_cohort:
        return run_proxy_cohort(args)
    return run_controller(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
