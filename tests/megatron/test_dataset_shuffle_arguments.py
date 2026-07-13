import pytest
from types import SimpleNamespace

from swift.megatron.arguments.megatron_args import MegatronArguments, MegatronTunerMixin, RLHFMegatronArgumentsMixin


class _StopPostInit(Exception):
    pass


def _run_argument_normalization(monkeypatch, *, dataset_shuffle, train_dataloader_shuffle):
    monkeypatch.setattr(RLHFMegatronArgumentsMixin, '__post_init__', lambda self: None)
    monkeypatch.setattr(MegatronTunerMixin, '__post_init__', lambda self: None)
    args = SimpleNamespace(
        tuner_type='full',
        recompute_granularity='full',
        recompute_method='uniform',
        dataset_shuffle=dataset_shuffle,
        train_dataloader_shuffle=train_dataloader_shuffle,
        group_by_length=False,
        padding_free=True,
        _check_mcore_bridge=lambda: None,
        _set_default=lambda: (_ for _ in ()).throw(_StopPostInit),
    )
    with pytest.raises(_StopPostInit):
        MegatronArguments.__post_init__(args)
    return args


@pytest.mark.parametrize(
    ('dataset_shuffle', 'initial_value', 'expected'),
    [(False, True, False), (True, False, True), (None, False, False)],
)
def test_dataset_shuffle_controls_megatron_dataloader(monkeypatch, dataset_shuffle, initial_value, expected):
    args = _run_argument_normalization(
        monkeypatch,
        dataset_shuffle=dataset_shuffle,
        train_dataloader_shuffle=initial_value,
    )
    assert args.train_dataloader_shuffle is expected
