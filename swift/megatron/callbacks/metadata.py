# Copyright (c) ModelScope Contributors. All rights reserved.
import datetime as dt
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from swift.utils import check_json_format, get_dist_setting, get_logger, is_last_rank
from .base import MegatronCallback
from .utils import (get_images_dir, get_log_dir, get_logging_path, get_run_metadata_path, get_run_summary_path,
                    get_swanlab_dir, get_tensorboard_dir, get_wandb_dir)

logger = get_logger()

_ENV_KEYS = [
    'CUDA_VISIBLE_DEVICES',
    'NPROC_PER_NODE',
    'NNODES',
    'NODE_RANK',
    'MASTER_ADDR',
    'MASTER_PORT',
    'RANK',
    'LOCAL_RANK',
    'WORLD_SIZE',
    'LOCAL_WORLD_SIZE',
    'LINEAR_CE_CHUNK_SIZE',
    'LINEAR_CE_DEBUG',
    'LOG_LEVEL',
    'REPORT_TO',
    'WANDB_PROJECT',
    'WANDB_ENTITY',
    'WANDB_DISABLED',
    'WANDB_LOG_RUN_ARTIFACTS',
    'WANDB_LOG_TRAIN_LOG',
    'NCCL_DEBUG',
    'CUDA_DEVICE_MAX_CONNECTIONS',
    'PYTORCH_CUDA_ALLOC_CONF',
    'TRITON_CACHE_DIR',
    'TORCH_EXTENSIONS_DIR',
    'HF_HOME',
    'HF_DATASETS_CACHE',
    'MODELSCOPE_CACHE',
    'SWIFT_RUN_NAME',
    'SWIFT_OUTPUT_DIR',
    'SWIFT_LOG_DIR',
    'SWIFT_LOG_FILE',
    'SWIFT_LAUNCH_COMMAND',
]

_PACKAGE_NAMES = [
    'ms-swift',
    'mcore-bridge',
    'megatron-core',
    'torch',
    'transformers',
    'transformer-engine',
    'flash-attn',
    'deepspeed',
    'peft',
    'triton',
    'wandb',
    'tensorboard',
]

_MODEL_CONFIG_FIELDS = [
    'num_layers',
    'hidden_size',
    'num_attention_heads',
    'num_query_groups',
    'ffn_hidden_size',
    'padded_vocab_size',
    'max_position_embeddings',
    'attention_backend',
    'experimental_attention_variant',
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'context_parallel_size',
    'sequence_parallel',
    'recompute_granularity',
    'recompute_method',
    'recompute_num_layers',
    'cross_entropy_loss_fusion',
    'transformer_impl',
    'params_dtype',
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run_git(path: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', '-C', str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_info(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    root = _run_git(path, 'rev-parse', '--show-toplevel')
    if not root:
        return None
    root_path = Path(root)
    status = _run_git(root_path, 'status', '--short') or ''
    status_lines = status.splitlines()
    return {
        'root': str(root_path),
        'commit': _run_git(root_path, 'rev-parse', 'HEAD'),
        'branch': _run_git(root_path, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'tag': _run_git(root_path, 'tag', '--points-at', 'HEAD'),
        'dirty': bool(status_lines),
        'status_count': len(status_lines),
        'status_short_head': status_lines[:200],
        'remote_origin': _run_git(root_path, 'remote', 'get-url', 'origin'),
    }


def _package_versions() -> Dict[str, Optional[str]]:
    versions = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _module_root(module_name: str) -> Optional[Path]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve()
    return None


def _env_snapshot() -> Dict[str, str]:
    return {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}


def _selected_model_config(config) -> Dict[str, Any]:
    data = {}
    for field in _MODEL_CONFIG_FIELDS:
        if hasattr(config, field):
            data[field] = getattr(config, field)
    return data


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(check_json_format(payload), f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        f.write('\n')


class MetadataCallback(MegatronCallback):
    """Writes run-level metadata separately from step metrics."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.is_write_rank = is_last_rank()
        self.log_dir = get_log_dir(self.args)
        self.metadata_path = get_run_metadata_path(self.args)
        self.summary_path = get_run_summary_path(self.args)

    def _build_metadata(self) -> Dict[str, Any]:
        cwd = Path.cwd().resolve()
        mcore_root = _module_root('mcore_bridge')
        metadata = {
            'schema_version': 1,
            'created_at_utc': _utc_now(),
            'hostname': socket.gethostname(),
            'platform': {
                'python': sys.version,
                'executable': sys.executable,
                'platform': platform.platform(),
            },
            'distributed': dict(zip(['rank', 'local_rank', 'world_size', 'local_world_size'], get_dist_setting())),
            'paths': {
                'cwd': str(cwd),
                'output_dir': self.args.output_dir,
                'log_dir': self.log_dir,
                'tensorboard_dir': get_tensorboard_dir(self.args),
                'wandb_dir': get_wandb_dir(self.args),
                'swanlab_dir': get_swanlab_dir(self.args),
                'images_dir': get_images_dir(self.args),
                'train_log': os.environ.get('SWIFT_LOG_FILE'),
                'logging_jsonl': get_logging_path(self.args),
                'run_metadata': self.metadata_path,
                'run_summary': self.summary_path,
                'mcore_bridge_root': str(mcore_root) if mcore_root else None,
            },
            'environment': _env_snapshot(),
            'packages': _package_versions(),
            'git': {
                'ms_swift': _git_info(cwd),
                'mcore_bridge': _git_info(mcore_root),
                'flash_linear_attention': _git_info(cwd / '.deps' / 'flash-linear-attention'),
            },
            'args': vars(self.args),
            'model_config': _selected_model_config(self.trainer.config),
        }
        return metadata

    def on_train_begin(self):
        if not self.is_write_rank:
            return
        _write_json(self.metadata_path, self._build_metadata())
        logger.info(f'run_metadata_path: {self.metadata_path}')

    def on_train_end(self):
        if not self.is_write_rank:
            return
        summary = {
            'schema_version': 1,
            'ended_at_utc': _utc_now(),
            'iteration': self.state.iteration,
            'train_iters': self.args.train_iters,
            'output_dir': self.args.output_dir,
            'last_model_checkpoint': getattr(self.state, 'last_model_checkpoint', None),
            'best_model_checkpoint': getattr(self.state, 'best_model_checkpoint', None),
            'best_metric': getattr(self.state, 'best_metric', None),
        }
        _write_json(self.summary_path, summary)
