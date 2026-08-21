import logging
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.utils import get_world_size

from ..common import ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


def get_device_peak_flops_per_second(device_name: str) -> Optional[int]:
    """
    Get a CUDA device's peak BF16/FP16 tensor core throughput, for use as the denominator
    of MFU.

    The figure returned is the *dense* rate with *FP32 accumulation*, which is the only rate
    torch can reach: cuBLAS is called with ``CUBLAS_COMPUTE_32F`` for BF16 inputs and there
    is no BF16-accumulate path to opt into. Two independent factors of two stand between a
    datasheet headline and that rate:

    - **Sparsity.** Headline figures are quoted with 2:4 structured sparsity, which is twice
      the dense rate.
    - **Accumulation.** The consumer-class dies (Ampere GA10x, Ada AD10x) run FP16/BF16
      matrix math at half rate when the accumulator is FP32. The data-center dies (GA100,
      GH100, GB100) have no such penalty. Whether this correction applies to a *published
      figure* depends on which of the two rates that figure is, and NVIDIA is not consistent
      about it even between two SKUs of one die — compare the L40S and L40 below.

    :param device_name: The device name, as reported by :func:`torch.cuda.get_device_name`.

    :returns: The peak FLOP/s, or ``None`` if the device isn't recognized. Attributing an
        unrecognized device the peak of some other device is how a wrong MFU gets reported
        as if it were a right one.
    """
    dense_correction = 0.5  # listed specs are one-half lower without sparsity
    fp32_accumulate_correction = 0.5  # only for the consumer-class dies; see above

    if "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        if "NVL" in device_name:
            return int(1671e12 * dense_correction)
        elif "PCIe" in device_name:
            return int(1513e12 * dense_correction)
        else:  # for SXM and other variants
            return int(1979e12 * dense_correction)
    elif "B200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/hgx/
        return int(4.5e15 * dense_correction)
    elif "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return int(624e12 * dense_correction)
    # The three Ada names below are prefixes of one another, so the order of these branches
    # is what makes them mean anything: "L4" matches an L40S too.
    elif "L40S" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/l40s/, which lists BF16 as
        # "362.05 | 733*". Both corrections apply: 733 is with sparsity, and the pair is
        # quoted with FP16 accumulation, which is why it is twice the L40's pair below for
        # a die clocked 1.2% higher. Cross-check: the Ada whitepaper gives AD102 165.2
        # TFLOP/s dense with FP32 accumulate over 128 SMs at 2.52 GHz, i.e. 512
        # FLOP/clock/SM, and the L40S's 142 SMs at 2.52 GHz come to 183.2.
        return int(733e12 * dense_correction * fp32_accumulate_correction)
    elif "L40" in device_name:
        # data from https://resources.nvidia.com/en-us-l40/l40-datasheet, which lists BF16
        # as "181.05 | 362.1**". Only the sparsity correction applies: 512 FLOP/clock/SM
        # over 142 SMs at the L40's 2.49 GHz is 181.0, so the datasheet's dense figure is
        # already the FP32-accumulate one. The Ada whitepaper's L40 appendix agrees.
        return int(362.1e12 * dense_correction)
    elif "L4" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/l4/, which lists BF16 as "242
        # teraFLOPS*" with sparsity. Both corrections apply: 512 FLOP/clock/SM over AD104's
        # 58 SMs at 2.04 GHz is 60.6, so the whitepaper appendix's dense 121 is the
        # FP16-accumulate rate.
        return int(242e12 * dense_correction * fp32_accumulate_correction)
    elif "A10G" in device_name:
        # data from AWS's A10G datasheet, which lists BF16 as "70 TF | 140 TF*". Only the
        # sparsity correction applies: 512 FLOP/clock/SM over the A10G's 80 SMs at 1.71 GHz
        # is 70.1, so the dense 70 is already the FP32-accumulate rate. The A10G is a
        # different part from the A10, whose datasheet quotes "125 | 250" for a smaller,
        # slower die and does so with FP16 accumulation.
        return int(140e12 * dense_correction)
    else:
        return None


@dataclass
class SpeedMonitorCallback(Callback):
    """
    Monitors throughput.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = -2

    num_flops_per_token: Optional[int] = None
    num_params: Optional[int] = None
    device_peak_flops_per_second: Optional[int] = None

    _total_steps: int = 0
    _total_tokens: int = 0
    _total_flops: int = 0
    _start_time: float = 0.0
    _first_step: bool = True
    _step_last_logged: float = 0.0
    _batch_load_start: float = 0.0
    _batch_load_time: float = 0.0
    _step_tokens: int = 0
    _step_seq_len: int = 0
    _step_flops: int = 0
    _parallel_degree: int = 1
    _bps_avg: Optional[float] = None
    _tps_avg: Optional[float] = None
    _mfu_avg: Optional[float] = None

    def reset(self):
        self._first_step = True
        self._bps_avg = None

    @property
    def bps_avg(self) -> Optional[float]:
        return self._bps_avg

    @property
    def tps_avg(self) -> Optional[float]:
        return self._tps_avg

    @property
    def mfu_avg(self) -> Optional[float]:
        return self._mfu_avg

    def _get_num_flops_per_token(self, seq_len: int) -> Optional[int]:
        if self.num_flops_per_token is not None:
            return self.num_flops_per_token
        elif isinstance(self.trainer.train_module, TransformerTrainModule):
            return self.trainer.train_module.num_flops_per_token(seq_len)
        else:
            return None

    def pre_train(self):
        self._first_step = True

        if self.trainer.dp_process_group is not None:
            self._parallel_degree = get_world_size() // get_world_size(
                self.trainer.dp_process_group
            )

        if self.num_params is None and isinstance(
            self.trainer.train_module, TransformerTrainModule
        ):
            self.num_params = self.trainer.train_module.model.num_non_embedding_params

        if (
            self.device_peak_flops_per_second is None
            and self.trainer.device.type == "cuda"
            and isinstance(self.trainer.train_module, TransformerTrainModule)
        ):
            device_name = torch.cuda.get_device_name(self.trainer.device)

            tm = self.trainer.train_module
            using_half_precision = tm.autocast_precision == torch.bfloat16 or (
                tm.dp_config is not None and tm.dp_config.param_dtype == DType.bfloat16
            )
            if using_half_precision:
                self.device_peak_flops_per_second = get_device_peak_flops_per_second(device_name)
                if self.device_peak_flops_per_second is None:
                    log.warning(
                        f"Unrecognized CUDA device '{device_name}', so MFU won't be reported. "
                        "Set 'device_peak_flops_per_second' on this callback to the device's "
                        "dense peak FLOP/s with FP32 accumulation to get MFU back."
                    )
            log.info(
                f"Device: {device_name}, Device peak Flops/s: {self.device_peak_flops_per_second}"
            )

    def pre_load_batch(self):
        self._batch_load_start = time.perf_counter()

    def pre_step(self, batch: Dict[str, Any]):
        self._batch_load_time = time.perf_counter() - self._batch_load_start

        if self._first_step:
            # We don't record the first batch since the first one tends to take
            # unusually long.
            return

        self._total_steps += 1
        if "input_ids" in batch:
            tokens_in_batch = batch["input_ids"].numel()
            self._step_tokens = tokens_in_batch // self._parallel_degree
            self._step_seq_len = batch["input_ids"].shape[1]
            self._total_tokens += self._step_tokens

            self._step_flops = 0
            if (
                num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)
            ) is not None:
                self._step_flops = num_flops_per_token * self._step_tokens
                self._total_flops += self._step_flops

    def post_step(self):
        counter = time.perf_counter()
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time, reduce_type=ReduceType.max
        )

        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._total_flops = 0
            self._start_time = counter
            self._first_step = False
            self._step_last_logged = counter
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        if self._step_tokens and self._total_tokens:
            tps = self._step_tokens / step_time
            tps_avg = self._total_tokens / total_time
            self._tps_avg = tps_avg
            self.trainer.record_metric("throughput/device/TPS", tps)
            self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)

        if self.trainer.global_train_tokens_seen is not None:
            self.trainer.record_metric(
                "throughput/total tokens", self.trainer.global_train_tokens_seen
            )
            if self.num_params is not None:
                self.trainer.record_metric(
                    "throughput/chinchilla multiple",
                    self.trainer.global_train_tokens_seen / (20 * self.num_params),
                )

        flops_ps: Optional[float] = None
        flops_ps_avg: Optional[float] = None
        if self._step_flops and self._total_flops:
            flops_ps = self._step_flops / step_time
            flops_ps_avg = self._total_flops / total_time
            self.trainer.record_metric("throughput/device/flopsPS", flops_ps)
            self.trainer.record_metric("throughput/device/flopsPS (actual avg)", flops_ps_avg)
            self.trainer.record_metric(
                "throughput/total petaflops", self.trainer.global_train_petaflops
            )

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        self._bps_avg = bps_avg
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

        data_pct = 100 * self._batch_load_time / step_time
        self.trainer.record_metric(
            "throughput/device/data loading (%)", data_pct, reduce_type=ReduceType.max
        )

        if (
            self.device_peak_flops_per_second is not None
            and flops_ps is not None
            and flops_ps_avg is not None
        ):
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            # MFU is computed from FLOPs/sec. This stays correct even if sequence length changes.
            mfu = 100 * flops_ps / self.device_peak_flops_per_second
            mfu_avg = 100 * flops_ps_avg / self.device_peak_flops_per_second
            self._mfu_avg = mfu_avg
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)
