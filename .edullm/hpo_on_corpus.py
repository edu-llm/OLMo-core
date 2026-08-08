"""Hybrid HPO controller entry point for the eduLLM platform.

This is the CPU-only controller process. It must **not** initialize one global process group,
and it must **not** be launched under ``torchrun --nproc-per-node=N``: instead it spawns one
isolated single-process subprocess per trial *segment*, each with ``WORLD_SIZE=1`` and its own
``CUDA_VISIBLE_DEVICES`` and ``MASTER_PORT``. OLMo has no in-process pause API, so freeze/thaw is
checkpoint + process exit + a freshly built trainer that loads that checkpoint.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid FT-PFN HPO controller")
    parser.add_argument("run_id", nargs="?", default=os.environ.get("EDULLM_RUN_ID", ""))
    parser.add_argument("--dry-run", action="store_true", help="Validate topology and exit.")
    parser.add_argument("--param-dtype", default="bfloat16")
    parser.add_argument(
        "--run-segment", action="store_true", help="Internal: run one trial segment."
    )
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--hard-stop-tokens", type=int, default=None)
    parser.add_argument("--segment-spec", default=None)
    parser.add_argument("--controller-spec", default=os.environ.get("EDULLM_HPO_SPEC"))
    parser.add_argument("--controller-state", default=None)
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
    )
    validate_precision_support(config)
    prepare_training_environment()
    try:
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        if fidelity is not None and fidelity.get("kind") == "umup":
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


def _build_controller_from_spec(spec: Dict[str, Any]):
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
        if centaur_spec is None:
            raise ValueError("Brainlift mode requires 5.6 Sol Centaur")
        if (
            centaur_spec.get("scope") != "multi_action"
            or centaur_spec.get("model") != "gpt-5.6-sol"
            or float(centaur_spec.get("ratio", -1.0)) != 0.3
            or int(centaur_spec.get("warmup", -1)) != 0
        ):
            raise ValueError(
                "Brainlift Centaur must use multi_action gpt-5.6-sol at ratio 0.3 "
                "with zero warmup"
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


def _segment_payload(
    allocation,
    *,
    controller_spec: Dict[str, Any],
) -> Dict[str, Any]:
    base_batch = int(controller_spec["base_global_batch_size"])
    realized_hps = dict(controller_spec.get("fixed_hps", {}))
    realized_hps.update(allocation.realized_hps)
    multiplier = float(realized_hps.get("global_batch_mult", 1.0))
    global_batch_size = int(round(base_batch * multiplier))
    target_tokens = int(controller_spec["controller"]["target_tokens"])
    payload = {
        "experiment_factory": controller_spec["experiment_factory"],
        "factory_kwargs": controller_spec.get("factory_kwargs", {}),
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
    }
    payload["config_hash"] = _segment_config_hash(payload)
    return payload


def _segment_config_hash(payload: Mapping[str, Any]) -> str:
    contract = {
        "realized_hps": payload["realized_hps"],
        "global_batch_size": payload["global_batch_size"],
        "target_tokens": payload["target_tokens"],
        "fidelity": payload.get("fidelity", {"kind": "exact"}),
        "experiment_factory": payload["experiment_factory"],
        "factory_kwargs": payload.get("factory_kwargs", {}),
        "transition": payload.get("transition"),
    }
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dispatch_allocations(
    *,
    allocations,
    controller_spec: Dict[str, Any],
    run_id: str,
    param_dtype: str,
):
    from olmo_core.hpo.types import WorkerObservation
    from olmo_core.hpo.worker import trial_namespace, world_size_one_env

    gpu_ids = controller_spec.get(
        "gpu_ids",
        list(range(int(controller_spec["controller"]["worker_count"]))),
    )
    if len(gpu_ids) < len(allocations):
        raise ValueError("controller spec provides fewer GPU ids than allocations")
    spec_dir = Path(
        controller_spec.get(
            "segment_spec_dir",
            os.path.join(tempfile.gettempdir(), f"edullm-hpo-{run_id}"),
        )
    )
    spec_dir.mkdir(parents=True, exist_ok=True)
    launches = []
    for index, allocation in enumerate(allocations):
        payload = _segment_payload(allocation, controller_spec=controller_spec)
        spec_path = spec_dir / f"decision-{allocation.decision_id}.json"
        spec_path.write_text(json.dumps(payload, sort_keys=True))
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
        env = world_size_one_env(
            int(gpu_ids[index]),
            int(controller_spec.get("master_port_base", 29500)) + index,
        )
        launches.append((allocation, argv, env))

    def run_one(launch):
        allocation, argv, env = launch
        completed = subprocess.run(
            argv,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
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
                )
            raise RuntimeError(
                f"trial {allocation.trial_id} exited {completed.returncode}: "
                f"{completed.stderr.strip()}"
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
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(launches)) as executor:
        return list(executor.map(run_one, launches))


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
    exact_factory = spec.get("exact_experiment_factory")
    if exact_factory is None:
        raise RuntimeError(
            "proxy fidelity requires exact_experiment_factory for fresh winner retraining"
        )
    from olmo_core.hpo.types import ActionKind, Allocation, ProposalSource

    winner_id, unit_config, _ = controller.best()
    exact_spec = copy.deepcopy(spec)
    exact_spec["fidelity"] = {"kind": "exact"}
    exact_spec["experiment_factory"] = exact_factory
    exact_spec["factory_kwargs"] = spec.get("exact_factory_kwargs", {})
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


def _validate_evidence_gates(spec: Dict[str, Any]) -> None:
    proxy_evidence = spec.get("proxy_evidence")
    fidelity_kind = spec.get("fidelity", {}).get("kind", "exact")
    if fidelity_kind != "exact" and proxy_evidence is None:
        raise ValueError(f"{fidelity_kind} fidelity requires preregistered admitted proxy evidence")
    if proxy_evidence is not None:
        from olmo_core.hpo.proxy import (
            AdmitDecision,
            ProxyAdmission,
            ProxyKind,
            ProxyMetrics,
        )

        metric_values = dict(proxy_evidence["metrics"])
        if fidelity_kind != "exact":
            expected_proxy_kind = {
                "frozen_layer": ProxyKind.FROZEN_LAYER,
                "umup": ProxyKind.UMUP,
            }[fidelity_kind]
            if metric_values.get("proxy_kind") != expected_proxy_kind.value:
                raise ValueError("proxy evidence kind does not match configured fidelity kind")
        if "proxy_kind" in metric_values:
            metric_values["proxy_kind"] = ProxyKind(metric_values["proxy_kind"])
        decision = ProxyAdmission(**proxy_evidence["admission"]).decide(
            ProxyMetrics(**metric_values)
        )
        if decision is not AdmitDecision.PRUNE_PROMOTE:
            raise ValueError("proxy evidence did not pass the preregistered admission gate")

    comparison = spec.get("budget_comparison")
    if fidelity_kind != "exact" and comparison is None:
        raise ValueError("proxy fidelity requires an equal-budget comparison")
    if comparison is not None:
        from olmo_core.hpo.arms import BudgetLedger, equal_budget

        candidate = BudgetLedger(**comparison["candidate"])
        control = BudgetLedger(**comparison["control"])
        if not equal_budget(
            candidate,
            control,
            rel_tol=float(comparison.get("rel_tol", 0.05)),
        ):
            raise ValueError("candidate and control arms do not have equal total budgets")


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
    _validate_evidence_gates(spec)
    controller = _build_controller_from_spec(spec)
    state_path = args.controller_state or spec.get(
        "controller_state_path",
        os.path.join(tempfile.gettempdir(), f"edullm-hpo-{args.run_id}.jsonl"),
    )
    remote_root = spec.get("controller_snapshot_root")
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
        )
        controller.ingest(results)
        _persist_controller_log(controller, state_path, remote_root)
    max_rounds = int(spec.get("max_rounds", 10_000))
    for _ in range(max_rounds):
        try:
            allocations = controller.propose_round()
        except Exception:
            _persist_controller_log(controller, state_path, remote_root)
            raise
        _persist_controller_log(controller, state_path, remote_root)
        if not allocations:
            break
        results = _dispatch_allocations(
            allocations=allocations,
            controller_spec=spec,
            run_id=args.run_id,
            param_dtype=args.param_dtype,
        )
        controller.ingest(results)
        _persist_controller_log(controller, state_path, remote_root)
    final_completed_method = getattr(controller, "final_evaluation_completed", None)
    final_completed = bool(final_completed_method()) if callable(final_completed_method) else False
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
    _enforce_required_winner(controller, spec)
    return 0


def run_segment(args: argparse.Namespace) -> int:
    """Run one isolated OLMo trial segment."""
    if not args.segment_spec:
        raise ValueError("--segment-spec is required in segment mode")
    if not args.trial_id or not args.checkpoint_dir or args.hard_stop_tokens is None:
        raise ValueError("segment mode requires trial id, checkpoint directory, and hard stop")
    spec = json.loads(Path(args.segment_spec).read_text())
    factory = _load_object(spec["experiment_factory"])
    config = factory(**spec.get("factory_kwargs", {}))
    fidelity = spec.get("fidelity", {"kind": "exact"})
    if fidelity.get("kind") == "umup":
        import unit_scaling  # noqa: F401

        configurator_ref = fidelity.get("configurator")
        if configurator_ref is None:
            raise ValueError("u-muP fidelity requires a configurator")
        configured = _load_object(configurator_ref)(
            config,
            width_factor=float(fidelity["width_factor"]),
            depth_factor=float(fidelity.get("depth_factor", 1.0)),
        )
        if configured is not None:
            config = configured

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
        reset_optim = bool(spec.get("transition", {}).get("optimizer_reset", False))
        config.trainer.load_optim_state = not reset_optim
    result = _run_configured_segment(
        config=config,
        worker=worker,
        hard_stop_tokens=int(args.hard_stop_tokens),
        heldout_metric=str(spec["heldout_metric"]),
        param_dtype=args.param_dtype,
        transition=spec.get("transition"),
        fidelity=fidelity,
    )
    payload = asdict(result)
    if not math.isfinite(payload["heldout_ce"]):
        payload["heldout_ce"] = None
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    # A trial-segment subprocess is world-size-one by construction; the controller is not.
    if args.run_segment:
        from olmo_core.hpo.worker import assert_single_process_topology

        assert_single_process_topology(os.environ)
    else:
        assert_controller_is_cpu_only(os.environ)
    if args.dry_run:
        return 0
    return run_segment(args) if args.run_segment else run_controller(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
