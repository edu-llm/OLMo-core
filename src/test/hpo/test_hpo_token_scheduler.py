from types import SimpleNamespace

import pytest
import torch
import unit_scaling as uu

from olmo_core.hpo.objective import EvaluatorGate
from olmo_core.hpo.worker import (
    HpoDiagnosticsCallback,
    WorkerConfig,
    build_hpo_scheduler,
    configure_hpo_experiment,
    validate_umup_model,
)
from olmo_core.optim.scheduler import WSD, SchedulerUnits
from olmo_core.train.callbacks import EvaluatorCallback
from olmo_core.train.common import Duration


def test_hpo_config_uses_token_units_and_fraction_schedules():
    sch = build_hpo_scheduler(
        {"lr": 1e-3, "warmup_fraction": 0.02, "decay_fraction": 0.2, "terminal_lr_ratio": 0.1}
    )
    assert isinstance(sch, WSD)
    assert sch.units is SchedulerUnits.tokens
    # Fraction-based, not absolute step/token counts.
    assert sch.warmup_fraction == pytest.approx(0.02)
    assert sch.decay_fraction == pytest.approx(0.2)
    assert sch.warmup is None
    assert sch.decay is None
    # Terminal LR is a ratio of peak LR.
    assert sch.decay_min_lr == pytest.approx(1e-4)


def test_scheduler_rejects_out_of_range_fractions():
    with pytest.raises(Exception):
        build_hpo_scheduler(
            {"lr": 1e-3, "warmup_fraction": 1.5, "decay_fraction": 0.2, "terminal_lr_ratio": 0.1}
        )


def test_configure_hpo_experiment_wires_runtime_guards_and_search_values():
    checkpointer = SimpleNamespace(
        save_interval=100,
        ephemeral_save_interval=10,
        fixed_steps=[100],
        max_checkpoints=3,
        save_async=True,
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            save_folder="/old",
            max_duration=Duration.steps(100),
            hard_stop=None,
            callbacks={
                "search_validation": EvaluatorCallback(eval_on_finish=True),
                "checkpointer": checkpointer,
            },
        ),
        data_loader=SimpleNamespace(global_batch_size=1024),
        train_module=SimpleNamespace(
            optim=SimpleNamespace(
                lr=1e-4,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999),
            ),
            scheduler=None,
            max_grad_norm=1.0,
        ),
    )
    worker = WorkerConfig(
        trial_id="t0",
        gpu=0,
        target_tokens=8192,
        quantum=2048,
        global_batch_size=1024,
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.2,
            "beta2_gap": 0.01,
            "eps": 1e-7,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 0.8,
        },
        checkpoint_root="/run/ckpt",
        evaluator_gate=EvaluatorGate(
            search_validation="search_validation", untouched="final_evaluation"
        ),
    )
    diagnostics = configure_hpo_experiment(
        config,
        worker=worker,
        hard_stop_tokens=3072,
        heldout_metric="eval/search_validation/val/CE loss",
    )
    assert isinstance(diagnostics, HpoDiagnosticsCallback)
    assert config.trainer.save_folder.endswith("trials/t0")
    assert config.trainer.max_duration == Duration.tokens(8192)
    assert config.trainer.hard_stop == Duration.tokens(3072)
    assert config.train_module.optim.lr == pytest.approx(1e-3)
    assert config.train_module.optim.betas == pytest.approx((0.9, 0.99))
    assert config.train_module.scheduler.units is SchedulerUnits.tokens
    assert config.train_module.max_grad_norm == pytest.approx(0.8)
    assert checkpointer.save_interval is None
    assert checkpointer.ephemeral_save_interval is None
    assert checkpointer.fixed_steps is None
    assert checkpointer.max_checkpoints is None
    assert checkpointer.save_async is False
    assert config.trainer.callbacks["hpo_diagnostics"] is diagnostics


def test_configure_applies_realized_global_batch_size_for_fresh_trial():
    checkpointer = SimpleNamespace(
        save_interval=100,
        ephemeral_save_interval=None,
        fixed_steps=None,
        max_checkpoints=3,
        save_async=True,
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(
            save_folder="/old",
            max_duration=Duration.steps(100),
            hard_stop=None,
            callbacks={
                "search_validation": EvaluatorCallback(eval_on_finish=True),
                "checkpointer": checkpointer,
            },
        ),
        data_loader=SimpleNamespace(global_batch_size=512),
        train_module=SimpleNamespace(
            optim=SimpleNamespace(
                lr=1e-4,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999),
            ),
            scheduler=None,
            max_grad_norm=1.0,
        ),
    )
    worker = WorkerConfig(
        trial_id="t0",
        gpu=0,
        target_tokens=8192,
        quantum=2048,
        global_batch_size=1024,
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.2,
            "beta2_gap": 0.01,
            "eps": 1e-7,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 0.8,
        },
        checkpoint_root="/run/ckpt",
        evaluator_gate=EvaluatorGate(
            search_validation="search_validation", untouched="final_evaluation"
        ),
    )
    configure_hpo_experiment(
        config,
        worker=worker,
        hard_stop_tokens=2048,
        heldout_metric="eval/search_validation/val/CE loss",
    )
    assert config.data_loader.global_batch_size == 1024


def test_configure_requires_segment_finish_search_evaluator():
    worker = WorkerConfig(
        trial_id="t0",
        gpu=0,
        target_tokens=8192,
        quantum=2048,
        global_batch_size=1024,
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.2,
            "beta2_gap": 0.01,
            "eps": 1e-7,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 0.8,
        },
        checkpoint_root="/run/ckpt",
        evaluator_gate=EvaluatorGate(
            search_validation="search_validation", untouched="final_evaluation"
        ),
    )

    def config_with(callback):
        return SimpleNamespace(
            trainer=SimpleNamespace(
                save_folder="/old",
                max_duration=Duration.steps(100),
                hard_stop=None,
                callbacks={
                    "search_validation": callback,
                    "checkpointer": SimpleNamespace(
                        save_interval=100,
                        ephemeral_save_interval=None,
                        fixed_steps=None,
                        max_checkpoints=3,
                        save_async=True,
                    ),
                },
            ),
            data_loader=SimpleNamespace(global_batch_size=1024),
            train_module=SimpleNamespace(
                optim=SimpleNamespace(
                    lr=1e-4,
                    weight_decay=0.01,
                    eps=1e-8,
                    betas=(0.9, 0.999),
                ),
                scheduler=None,
                max_grad_norm=1.0,
            ),
        )

    for callback in (object(), EvaluatorCallback(eval_on_finish=False)):
        with pytest.raises(RuntimeError):
            configure_hpo_experiment(
                config_with(callback),
                worker=worker,
                hard_stop_tokens=2048,
                heldout_metric="eval/search_validation/val/CE loss",
            )
    leaked = config_with(EvaluatorCallback(eval_on_finish=True))
    leaked.trainer.callbacks["final_evaluation"] = EvaluatorCallback(eval_on_finish=True)
    with pytest.raises(RuntimeError, match="untouched"):
        configure_hpo_experiment(
            leaked,
            worker=worker,
            hard_stop_tokens=2048,
            heldout_metric="eval/search_validation/val/CE loss",
        )


def test_configure_applies_frozen_layer_fidelity_to_model():
    config, worker = _fidelity_config_fixture()
    configure_hpo_experiment(
        config,
        worker=worker,
        hard_stop_tokens=2048,
        heldout_metric="eval/search_validation/val/CE loss",
        fidelity={"kind": "frozen_layer", "train_last_k": 4},
    )
    assert config.model.freeze_params == [
        "embeddings.*",
        "embedding_norm.*",
        "blocks.0.*",
        "blocks.1.*",
        "blocks.2.*",
        "blocks.3.*",
    ]


def test_configure_umup_parameterization_rejects_metadata_only_optimizer():
    config, worker = _fidelity_config_fixture()
    config.umup_backend = "unit-scaling"
    config.umup_parity_validated = True
    config.umup_metadata = {"proxy_depth": config.model.n_layers}
    with pytest.raises(RuntimeError, match="scaled AdamW"):
        configure_hpo_experiment(
            config,
            worker=worker,
            hard_stop_tokens=2048,
            heldout_metric="eval/search_validation/val/CE loss",
            fidelity={"kind": "frozen_layer", "train_last_k": 4},
            model_parameterization={"kind": "umup", "backend": "unit-scaling"},
        )

    config, worker = _fidelity_config_fixture()
    with pytest.raises(ValueError, match="fidelity"):
        configure_hpo_experiment(
            config,
            worker=worker,
            hard_stop_tokens=2048,
            heldout_metric="eval/search_validation/val/CE loss",
            fidelity={"kind": "umup"},
        )


def test_umup_model_validation_requires_official_parameter_metadata():
    with pytest.raises(RuntimeError, match="execution backend"):
        validate_umup_model(torch.nn.Linear(2, 2))
    model = uu.Linear(2, 2)
    model._umup_execution_backend = "unit-scaling-public-functional"
    validate_umup_model(model)


def _fidelity_config_fixture():
    checkpointer = SimpleNamespace(
        save_interval=100,
        ephemeral_save_interval=None,
        fixed_steps=None,
        max_checkpoints=3,
        save_async=True,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(n_layers=8, freeze_params=None),
        trainer=SimpleNamespace(
            save_folder="/old",
            max_duration=Duration.steps(100),
            hard_stop=None,
            callbacks={
                "search_validation": EvaluatorCallback(eval_on_finish=True),
                "checkpointer": checkpointer,
            },
        ),
        data_loader=SimpleNamespace(global_batch_size=1024),
        train_module=SimpleNamespace(
            optim=SimpleNamespace(
                lr=1e-4,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999),
            ),
            scheduler=None,
            max_grad_norm=1.0,
        ),
    )
    worker = WorkerConfig(
        trial_id="t0",
        gpu=0,
        target_tokens=8192,
        quantum=2048,
        global_batch_size=1024,
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.2,
            "beta2_gap": 0.01,
            "eps": 1e-7,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 0.8,
        },
        checkpoint_root="/run/ckpt",
        evaluator_gate=EvaluatorGate(
            search_validation="search_validation", untouched="final_evaluation"
        ),
    )
    return config, worker
