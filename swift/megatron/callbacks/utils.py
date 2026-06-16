# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def _env_enabled(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default).lower() in {'1', 'true', 'yes', 'on'}


def is_logging_jsonl_disabled() -> bool:
    return _env_enabled('SWIFT_DISABLE_LOGGING_JSONL')


def get_log_dir(args):
    log_dir = os.environ.get('SWIFT_LOG_DIR')
    if log_dir:
        return os.path.abspath(os.path.expanduser(log_dir))
    return os.path.abspath(os.path.expanduser(args.output_dir))


def get_logging_path(args):
    return os.path.join(get_log_dir(args), 'logging.jsonl')


def get_tensorboard_dir(args):
    tensorboard_dir = getattr(args, 'tensorboard_dir', None)
    if tensorboard_dir:
        return os.path.abspath(os.path.expanduser(tensorboard_dir))
    return os.path.join(get_log_dir(args), 'runs')


def get_wandb_dir(args):
    return os.path.join(get_log_dir(args), 'wandb')


def get_swanlab_dir(args):
    return os.path.join(get_log_dir(args), 'swanlab')


def get_images_dir(args):
    return os.path.join(get_log_dir(args), 'images')


def get_run_metadata_path(args):
    return os.path.join(get_log_dir(args), 'run_metadata.json')


def get_run_summary_path(args):
    return os.path.join(get_log_dir(args), 'run_summary.json')


def rewrite_logs(logs):
    new_logs = {}
    for k, v in logs.items():
        if isinstance(v, str):
            continue
        k = k.replace(' ', '_')
        if k == 'train_speed(s/it)':
            k = 'train_speed(s_it)'
        if k.startswith(('train/', 'eval/', 'test/')):
            pass
        elif k.startswith('eval_'):
            k = k[len('eval_'):]
            k = f'eval/{k}'
        elif k.startswith('test_'):
            k = k[len('test_'):]
            k = f'test/{k}'
        else:
            k = f'train/{k}'
        new_logs[k] = v
    return new_logs
