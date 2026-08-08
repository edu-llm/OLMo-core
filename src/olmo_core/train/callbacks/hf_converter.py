"""
Callback for converting checkpoints to HuggingFace format during and after training.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.config import DType
from olmo_core.distributed.utils import barrier, get_rank

from .callback import Callback
from .checkpointer import CheckpointerCallback

log = logging.getLogger(__name__)


@dataclass
class HFConverterCallback(Callback):
    """
    Converts saved checkpoints to HuggingFace format during and at the end of a training job.

    By default this runs once, after training completes, and uses
    :func:`olmo_core.nn.hf.convert_checkpoint_to_hf` to convert the final OLMo Core
    checkpoint to a HuggingFace-compatible format. Set :data:`convert_interval` to also
    convert while the run is still going.

    .. note::
        This callback requires the ``transformers`` library to be installed.

    .. warning::
        In distributed training, ALL ranks must participate in this callback because
        gathering the full model state dict from FSDP requires collective operations.
        Only rank 0 performs the actual HF conversion and saving.
    """

    priority: ClassVar[int] = -1  # Run after checkpointer callback.

    enabled: bool = True
    """
    Whether this callback is enabled. Set to ``False`` to disable HF conversion.
    """

    output_folder: Optional[str] = None
    """
    The folder to save the HuggingFace checkpoint to. If not specified, defaults to
    ``{checkpoint_path}-hf`` where ``checkpoint_path`` is the final checkpoint path.
    """

    convert_interval: Optional[int] = None
    """
    Steps between conversions while training is still running. ``None``, the default, keeps
    the original behaviour of converting only once training has finished.

    A conversion is not cheap and it is not concurrent: gathering the full model state dict
    is a collective that every rank participates in, and every rank other than zero then
    waits at a barrier while rank zero builds an HF model on CPU and writes it. Pick an
    interval against what a whole-fleet stall costs, not against how often a checkpoint
    lands.
    """

    raise_on_failure: bool = True
    """
    Whether a failed conversion should stop the run. ``True``, the default, is right for a
    job whose product is the converted model. Set it to ``False`` on a long training run that
    exports as a side effect, where the export depends on things the training does not -- a
    tokenizer fetched over the network, a write to a different prefix -- and losing the run
    to one of them costs incomparably more than losing the export.
    """

    experiment_config: Optional[Dict[str, Any]] = None
    """
    The experiment config to convert against. When this is ``None`` the config is read back
    from the checkpoint directory's ``config.json``.

    Supplying it matters for :data:`convert_interval`. A trainer saving asynchronously has
    not necessarily finished writing ``config.json`` for the step being converted, and the
    conversion only needs the model and tokenizer sections, which the process already holds.
    """

    dtype: Optional[DType] = DType.bfloat16
    """
    The dtype to save the HuggingFace model weights as. Defaults to bfloat16.
    """

    validate: bool = False
    """
    Whether to validate the converted model against the original model.
    Validation loads both models and compares their outputs.
    """

    debug: bool = False
    """
    Whether to output debug information during validation.
    Only has an effect if ``validate`` is ``True``.
    """

    tokenizer_id: Optional[str] = None
    """
    The HuggingFace tokenizer identifier to save with the model.
    If not specified, uses the tokenizer from the experiment config.
    """

    max_sequence_length: Optional[int] = None
    """
    The maximum sequence length for the model. If not specified, uses the tokenizer's
    default max length.
    """

    device: Optional[str] = None
    """
    The device to use for conversion. Defaults to CPU.
    """

    moe_capacity_factor: Optional[float] = None
    """
    The MoE capacity factor. Higher values can decrease validation false negatives
    but may cause OOM errors. Only relevant for MoE models.
    """

    def _get_checkpointer_callback(self) -> Optional[CheckpointerCallback]:
        for callback in self.trainer.callbacks.values():
            if isinstance(callback, CheckpointerCallback):
                return callback
        return None

    def _get_latest_checkpoint_path(self) -> Optional[str]:
        checkpointer = self._get_checkpointer_callback()
        if checkpointer is None:
            log.warning("CheckpointerCallback not found, cannot determine latest checkpoint path")
            return None

        if checkpointer._latest_checkpoint_path:
            return checkpointer._latest_checkpoint_path

        if checkpointer._checkpoints:
            return checkpointer._checkpoints[-1]

        return None

    def _output_path(self, checkpoint_path: str) -> str:
        """Where this conversion goes, which depends on whether there will be more than one.

        A run converting on an interval writes one directory per step under
        :data:`output_folder`, including the last one. A single destination would have each
        conversion overwrite the previous, and the point of converting during the run is that
        something downstream reads the intermediate results.
        """
        if self.output_folder is None:
            return checkpoint_path + "-hf"
        if self.convert_interval is None:
            return self.output_folder
        dirname = self.trainer.checkpointer.checkpoint_dirname(self.step)
        return f"{self.output_folder.rstrip('/')}/{dirname}"

    def _get_full_model_state_dict(self) -> Dict[str, Any]:
        """
        Get the full model state dict from the trainer's model.

        This is a collective operation - ALL ranks must call this method.
        The full state dict is gathered to rank 0.
        """
        model = self.trainer.train_module.model
        # full_state_dict=True gathers the complete model state to rank 0.
        # cpu_offload=True avoids GPU OOM for large models.
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        return dist_cp_sd.get_model_state_dict(model, options=sd_options)

    def post_step(self):
        # ON post_step RATHER THAN ON post_checkpoint_saved, WHICH IS THE HOOK THIS LOOKS
        # LIKE IT WANTS. A trainer configured with ``save_async=True`` fires
        # ``post_checkpoint_saved`` from the completion callback of a Future, on a worker
        # thread, at whatever moment the write finishes -- which is a different step on every
        # rank and is not the thread issuing the run's other collectives. The state dict
        # gather below is a collective, so hosting it there would have ranks entering it out
        # of order from a second thread. ``post_step`` runs on the main thread on every rank
        # in lockstep, which is what a collective needs.
        if self.convert_interval is None or not self.enabled:
            return
        if self.step % self.convert_interval != 0:
            return
        dirname = self.trainer.checkpointer.checkpoint_dirname(self.step)
        self._convert(f"{self.trainer.save_folder.rstrip('/')}/{dirname}")

    def post_train(self):
        if not self.enabled:
            log.info("HFConverterCallback is disabled, skipping conversion")
            barrier()
            return

        if self.convert_interval is not None and self.step % self.convert_interval == 0:
            # ``post_step`` converted these same weights moments ago, at this same step, to
            # the directory this would name. Doing it twice costs a second gather and a
            # second whole-fleet stall to write the bytes that are already there.
            log.info("Step %d was already converted to HuggingFace format", self.step)
            barrier()
            return

        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            log.warning("No checkpoint found, skipping HF conversion")
            barrier()
            return

        self._convert(checkpoint_path)

    def _convert(self, checkpoint_path: str):
        # NOTE: In distributed training with FSDP, getting the full model state dict requires
        # ALL ranks to participate in the collective operation. Only rank 0 performs the actual
        # HF conversion; all ranks synchronize at a barrier before returning.

        try:
            from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config
        except ImportError:
            log.error(
                "Failed to import HF conversion utilities. "
                "Make sure the 'transformers' library is installed."
            )
            barrier()
            return

        experiment_config: Optional[dict] = self.experiment_config
        if experiment_config is None and get_rank() == 0:
            try:
                experiment_config = load_config(checkpoint_path)
            except Exception as e:
                log.error(f"Failed to load config from checkpoint: {e}")

        # ALL ranks must participate in gathering the full state dict (FSDP collective).
        log.info("Gathering full model state dict (collective operation)...")
        try:
            model_state_dict = self._get_full_model_state_dict()
        except Exception as e:
            log.error(f"Failed to get model state dict: {e}")
            barrier()
            if self.raise_on_failure:
                raise
            return

        if get_rank() == 0:
            log.info(f"Converting checkpoint at '{checkpoint_path}' to HuggingFace format")

            if experiment_config is None:
                log.error("Experiment config not found in checkpoint, cannot convert to HF format")
                barrier()
                return

            # Copied because ``convert_checkpoint_to_hf`` deletes the deprecated keys out of
            # whatever it is given, and a callback converting on an interval hands it the same
            # config object every time.
            transformer_config_dict = copy.deepcopy(experiment_config.get("model"))
            tokenizer_config_dict = copy.deepcopy(
                experiment_config.get("dataset", {}).get("tokenizer")
            )

            if transformer_config_dict is None:
                log.error(
                    "Model config not found in experiment config, cannot convert to HF format"
                )
                barrier()
                return

            if tokenizer_config_dict is None:
                log.warning(
                    "Tokenizer config not found in experiment config, "
                    "conversion will proceed without tokenizer"
                )
                tokenizer_config_dict = {}

            output_path = self._output_path(checkpoint_path)

            device = torch.device(self.device) if self.device else None

            try:
                convert_checkpoint_to_hf(
                    original_checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    transformer_config_dict=transformer_config_dict,
                    tokenizer_config_dict=tokenizer_config_dict,
                    model_state_dict=model_state_dict,
                    dtype=self.dtype,
                    tokenizer_id=self.tokenizer_id,
                    max_sequence_length=self.max_sequence_length,
                    validate=self.validate,
                    debug=self.debug,
                    device=device,
                    moe_capacity_factor=self.moe_capacity_factor,
                )
                log.info(
                    f"Successfully converted checkpoint to HuggingFace format at '{output_path}'"
                )
            except Exception as e:
                log.error(f"Failed to convert checkpoint to HuggingFace format: {e}")
                barrier()
                if self.raise_on_failure:
                    raise
                return

        barrier()
