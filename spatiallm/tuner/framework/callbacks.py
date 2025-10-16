import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import PreTrainedModel, TrainerCallback
from typing_extensions import override

from . import logging
from .utils import has_length

if TYPE_CHECKING:
    from transformers import (
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    from ..hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )

logger = logging.get_logger(__name__)

TRAINER_LOG = "trainer_log.jsonl"


class MemoryCallback(TrainerCallback):
    """记录 GPU 显存占用到 TensorBoard 的回调"""

    def __init__(self, include_all_devices: bool = False):
        """
        初始化内存回调

        Args:
            include_all_devices: 是否记录所有 GPU 设备的内存使用情况
        """
        self.include_all_devices = include_all_devices

    def _get_memory_stats(self, device_id: Optional[int] = None) -> dict[str, float]:
        """获取指定设备的实际显存占用情况（类似nvidia-smi）"""
        try:
            if device_id is not None:
                device_idx = device_id
            else:
                device_idx = torch.cuda.current_device()

            device = torch.device(f"cuda:{device_idx}")
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            memory_total = (
                torch.cuda.get_device_properties(device).total_memory / 1024**3
            )

            memory_used = memory_reserved  # reserved memory更接近实际使用
            memory_free = memory_total - memory_reserved
            memory_usage_percent = (
                (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
            )

            prefix = f"gpu_{device_id}" if device_id is not None else "gpu"

            return {
                f"{prefix}/memory_used_GB": round(memory_used, 3),
                f"{prefix}/memory_free_GB": round(memory_free, 3),
                f"{prefix}/total_memory_GB": round(memory_total, 3),
                f"{prefix}/memory_usage_percent": round(memory_usage_percent, 2),
            }

        except Exception as e:
            logger.warning(f"Failed to get memory stats for device {device_id}: {e}")
            return {}

    def _log_memory_stats(self, trainer, logs: Optional[dict] = None):
        """记录内存统计信息"""
        if not torch.cuda.is_available():
            return

        try:
            memory_stats = {}

            if self.include_all_devices:
                # 记录所有 GPU 设备的内存使用情况
                for device_id in range(torch.cuda.device_count()):
                    device_stats = self._get_memory_stats(device_id)
                    memory_stats.update(device_stats)
            else:
                # 只记录当前设备的内存使用情况
                device_stats = self._get_memory_stats()
                memory_stats.update(device_stats)

            # 如果有 trainer，使用 trainer.log 记录到 TensorBoard
            if trainer is not None:
                trainer.log(memory_stats)

            # 如果有 logs 字典，也添加到其中（用于 on_log 回调）
            if logs is not None:
                logs.update(memory_stats)

        except Exception as e:
            logger.warning(f"Failed to log memory stats: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """在日志记录时添加内存信息"""
        if logs is None:
            return control

        # 只在有 CUDA 可用时记录内存信息
        if torch.cuda.is_available():
            self._log_memory_stats(None, logs)

        return control


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once(
                "Previous trainer log in this folder will be deleted."
            )
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        self._close_thread_pool()

    @override
    def on_substep_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=(
                round(self.cur_steps / self.max_steps * 100, 2)
                if self.max_steps != 0
                else 100
            ),
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(
                state.num_input_tokens_seen / (time.time() - self.start_time), 2
            )
            logs["total_tokens"] = state.num_input_tokens_seen

        logs = {k: v for k, v in logs.items() if v is not None}

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=(
                        round(self.cur_steps / self.max_steps * 100, 2)
                        if self.max_steps != 0
                        else 100
                    ),
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "spatiallm")

    @override
    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


def get_swanlab_callback(finetuning_args: "FinetuningArguments") -> "TrainerCallback":
    r"""Get the callback for logging to SwanLab."""
    import swanlab  # type: ignore
    from swanlab.integration.transformers import SwanLabCallback  # type: ignore

    if finetuning_args.swanlab_api_key is not None:
        swanlab.login(api_key=finetuning_args.swanlab_api_key)

    if finetuning_args.swanlab_lark_webhook_url is not None:
        from swanlab.plugin.notification import LarkCallback  # type: ignore

        lark_callback = LarkCallback(
            webhook_url=finetuning_args.swanlab_lark_webhook_url,
            secret=finetuning_args.swanlab_lark_secret,
        )
        swanlab.register_callbacks([lark_callback])

    class SwanLabCallbackExtension(SwanLabCallback):
        def setup(
            self,
            args: "TrainingArguments",
            state: "TrainerState",
            model: "PreTrainedModel",
            **kwargs,
        ):
            if not state.is_world_process_zero:
                return

            super().setup(args, state, model, **kwargs)
            try:
                if hasattr(self, "_swanlab"):
                    swanlab_public_config = self._swanlab.get_run().public.json()
                else:  # swanlab <= 0.4.9
                    swanlab_public_config = self._experiment.get_run().public.json()
            except Exception:
                swanlab_public_config = {}

            with open(
                os.path.join(args.output_dir, "swanlab_public_config.json"), "w"
            ) as f:
                f.write(json.dumps(swanlab_public_config, indent=2))

    swanlab_callback = SwanLabCallbackExtension(
        project=finetuning_args.swanlab_project,
        workspace=finetuning_args.swanlab_workspace,
        experiment_name=finetuning_args.swanlab_run_name,
        mode=finetuning_args.swanlab_mode,
        config={"Framework": "SpatialLM"},
        logdir=finetuning_args.swanlab_logdir,
        tags=["SpatialLM"],
    )
    return swanlab_callback
