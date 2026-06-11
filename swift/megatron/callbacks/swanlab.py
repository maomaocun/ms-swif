# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils import check_json_format, is_last_rank
from .base import MegatronCallback
from .utils import get_swanlab_dir, rewrite_logs


class SwanlabCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        args = self.args
        self.config = check_json_format(vars(args))
        if args.swanlab_exp_name is None:
            args.swanlab_exp_name = args.output_dir
        self.save_dir = get_swanlab_dir(args)
        self.writer = None
        self.setup()

    def setup(self):
        import swanlab
        args = self.args
        if is_last_rank():
            swanlab.init(
                logdir=self.save_dir,
                experiment_name=args.swanlab_exp_name,
                project=args.swanlab_project,
                config=self.config)
            self.writer = swanlab

    def on_log(self, logs):
        logs = rewrite_logs(logs)
        if is_last_rank():
            self.writer.log(logs, step=self.state.iteration)
