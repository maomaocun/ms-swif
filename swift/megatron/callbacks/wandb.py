# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re

from swift.utils import check_json_format, get_logger, is_last_rank
from .base import MegatronCallback
from .utils import get_logging_path, get_run_metadata_path, get_run_summary_path, get_wandb_dir, rewrite_logs

logger = get_logger()


class WandbCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        if args.wandb_exp_name is None:
            args.wandb_exp_name = args.output_dir
        self.save_dir = get_wandb_dir(args)
        self.writer = None
        self.setup()

    def setup(self):
        import wandb
        args = self.args
        if is_last_rank():
            wandb.init(dir=self.save_dir, name=args.wandb_exp_name, project=args.wandb_project, config=self.config)
            self.writer = wandb

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        if is_last_rank():
            self.writer.log(logs, step=self.state.iteration)

    def on_train_begin(self):
        if is_last_rank() and self.writer is not None and getattr(self.writer, 'run', None) is not None:
            self.writer.run.summary['output_dir'] = self.args.output_dir
            if os.environ.get('SWIFT_LOG_FILE'):
                self.writer.run.summary['train_log'] = os.environ['SWIFT_LOG_FILE']

    @staticmethod
    def _enabled_env(name: str, default: str = '0') -> bool:
        return os.environ.get(name, default).lower() in {'1', 'true', 'yes', 'on'}

    @staticmethod
    def _artifact_name(name: str) -> str:
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('-')
        return name or 'megatron-run'

    def on_train_end(self):
        if not is_last_rank() or self.writer is None:
            return

        if self._enabled_env('WANDB_LOG_RUN_ARTIFACTS', '1'):
            artifact_name = self._artifact_name(self.args.wandb_exp_name or os.path.basename(self.args.output_dir))
            artifact = self.writer.Artifact(artifact_name, type='training-log')
            paths = [
                get_run_metadata_path(self.args),
                get_run_summary_path(self.args),
                get_logging_path(self.args),
            ]
            if self._enabled_env('WANDB_LOG_TRAIN_LOG', '0') and os.environ.get('SWIFT_LOG_FILE'):
                paths.append(os.environ['SWIFT_LOG_FILE'])
            added = 0
            for path in paths:
                if path and os.path.exists(path):
                    artifact.add_file(path)
                    added += 1
            if added:
                try:
                    self.writer.log_artifact(artifact)
                except Exception as exc:
                    logger.warning(f'Failed to log W&B artifact: {exc}')

        try:
            self.writer.finish()
        except Exception as exc:
            logger.warning(f'Failed to finish W&B run: {exc}')
