# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import time
import torch
import torch.distributed as dist
from tqdm import tqdm

from swift.megatron.utils.flops import estimate_training_flops
from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_logger, is_last_rank
from .base import MegatronCallback
from .utils import get_logging_path, is_logging_jsonl_disabled

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.is_write_rank = is_last_rank()
        self.step_metrics_log_file = None

    @staticmethod
    def _append_text(path: str, text: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text)

    @staticmethod
    def _as_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _float_env(names, default):
        for name in names:
            value = os.environ.get(name)
            if value:
                try:
                    return float(value)
                except ValueError:
                    logger.warning(f'Invalid float env {name}={value!r}; ignoring it.')
        return default

    def _world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get('WORLD_SIZE', '1'))

    def _add_efficiency_metrics(self, logs):
        step_time_s = self._as_float(logs.get('step_time_s')) or self._as_float(logs.get('train_speed(s/it)'))
        total_tokens = self._as_float(logs.get('total_tokens'))
        supervised_tokens = self._as_float(logs.get('num_tokens'))
        tokens = total_tokens or supervised_tokens
        if not step_time_s or step_time_s <= 0 or not tokens or tokens <= 0:
            return
        logged_steps = self._as_float(logs.get('logged_steps')) or 1.0
        total_time_s = step_time_s * max(logged_steps, 1.0)
        logs['tokens_per_s'] = tokens / total_time_s
        seq_len_sum = self._as_float(logs.get('seq_len_sum'))
        num_sequences = self._as_float(logs.get('num_sequences'))
        if seq_len_sum and num_sequences and num_sequences > 0:
            logs['avg_seq_len'] = seq_len_sum / num_sequences
        device_peak_tflops = self._float_env(['SWIFT_MFU_DEVICE_TFLOPS', 'MFU_DEVICE_TFLOPS'], 989.0)
        world_size = max(self._world_size(), 1)
        estimate = estimate_training_flops(self.trainer.config, self.args, logs)
        if estimate is not None:
            model_flops = estimate.train_flops
            hardware_flops = estimate.hardware_flops
            logs['flops_estimator'] = estimate.estimator
            logs['attention_layers'] = estimate.attention_layers
            logs['linear_attention_layers'] = estimate.linear_attention_layers
            logs['dense_mlp_layers'] = estimate.dense_mlp_layers
            if estimate.moe_layers:
                logs['moe_layers'] = estimate.moe_layers
        else:
            model_params = self._float_env(['SWIFT_MFU_MODEL_PARAMS', 'MFU_MODEL_PARAMS', 'MODEL_PARAM_COUNT'], 27e9)
            model_flops = 6.0 * model_params * tokens
            hardware_flops = model_flops
            logs['flops_estimator'] = 'params_tokens_fallback'
        logs['model_tflops_per_gpu'] = model_flops / total_time_s / world_size / 1e12
        logs['hardware_tflops_per_gpu'] = hardware_flops / total_time_s / world_size / 1e12
        if device_peak_tflops > 0:
            logs['mfu'] = logs['model_tflops_per_gpu'] / device_peak_tflops
            logs['hfu'] = logs['hardware_tflops_per_gpu'] / device_peak_tflops

    @staticmethod
    def _drop_internal_metrics(logs):
        for key in ['_attention_seq_len_sq_sum', 'seq_len_sum']:
            logs.pop(key, None)

    @staticmethod
    def _ordered_logs(logs):
        priority_keys = [
            'loss',
            'grad_norm',
            'learning_rate',
            'step_time_s',
            'mfu',
            'hfu',
            'model_tflops_per_gpu',
            'hardware_tflops_per_gpu',
            'tokens_per_s',
            'total_tokens',
            'num_tokens',
            'avg_seq_len',
            'max_seq_len',
            'num_sequences',
            'train_speed(s/it)',
            'logged_steps',
            'timing/forward_backward_s',
            'timing/optimizer_step_s',
            'timing/optimizer_reduce_s',
            'timing/lr_scheduler_s',
            'timing/dataloader_next_s',
            'timing/batch_prepare_s',
            'timing/batch_fetch_s',
            'timing/zero_grad_s',
            'timing/metric_prepare_s',
            'timing/train_step_s',
            'iteration',
            'elapsed_time',
            'remaining_time',
            'memory(GiB)',
        ]
        ordered = {}
        for key in priority_keys:
            if key in logs:
                ordered[key] = logs[key]
        for key, value in logs.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = self.start_step
        if is_logging_jsonl_disabled():
            logger.info('logging_jsonl: disabled; step metrics are written to stdout/train log')
            self.jsonl_writer = None
            step_metrics_log_file = os.environ.get('SWIFT_STEP_METRICS_LOG_FILE')
            if step_metrics_log_file:
                self.step_metrics_log_file = os.path.abspath(os.path.expanduser(step_metrics_log_file))
                logger.info(f'step_metrics_log_file: {self.step_metrics_log_file}')
        else:
            logging_path = get_logging_path(self.args)
            logger.info(f'logging_path: {logging_path}')
            self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')

    def on_train_end(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None
        if self.training_bar is not None:
            self.training_bar.close()
        self.training_bar = None

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        raw_logs = logs
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        now = time.time()
        elapsed = now - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.last_log_step
        log_interval = now - self.last_log_time
        measured_step_time = self._as_float(logs.get('step_time_s'))
        train_speed = measured_step_time if measured_step_time is not None else (
            log_interval / n_steps if n_steps > 0 else 0.0)
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        self._add_efficiency_metrics(logs)
        self._drop_internal_metrics(logs)
        ordered_logs = self._ordered_logs({k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()})
        raw_logs.clear()
        raw_logs.update(ordered_logs)
        logs = raw_logs
        self.state.last_log_metrics = dict(logs)
        if 'loss' in logs:
            self.state.last_train_metrics = dict(logs)
        if n_steps > 0:
            self.last_log_time = now
            self.last_log_step = state.iteration
        if self.jsonl_writer is not None:
            self.jsonl_writer.append(logs)
        if self.is_write_rank:
            if self.jsonl_writer is None:
                metrics_json = json.dumps(logs, ensure_ascii=False)
                metrics_line = f'step_metrics_json: {metrics_json}'
                if self.step_metrics_log_file:
                    timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
                    self._append_text(self.step_metrics_log_file, f'{timestamp} {metrics_line}\n')
                else:
                    self.training_bar.write(metrics_line)
            else:
                self.training_bar.write(str(logs))
