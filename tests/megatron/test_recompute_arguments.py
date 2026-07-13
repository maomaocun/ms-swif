import pytest
from types import SimpleNamespace

from swift.megatron.arguments.megatron_args import MegatronArguments, MegatronTunerMixin, RLHFMegatronArgumentsMixin


class _StopPostInit(Exception):
    pass


def _run_argument_normalization(monkeypatch, *, recompute_granularity, recompute_method, recompute_modules):
    monkeypatch.setattr(RLHFMegatronArgumentsMixin, '__post_init__', lambda self: None)
    monkeypatch.setattr(MegatronTunerMixin, '__post_init__', lambda self: None)
    args = SimpleNamespace(
        tuner_type='full',
        recompute_granularity=recompute_granularity,
        recompute_method=recompute_method,
        recompute_modules=recompute_modules,
        dataset_shuffle=None,
        train_dataloader_shuffle=True,
        group_by_length=False,
        padding_free=True,
        _check_mcore_bridge=lambda: None,
        _set_default=lambda: (_ for _ in ()).throw(_StopPostInit),
    )
    with pytest.raises(_StopPostInit):
        MegatronArguments.__post_init__(args)
    return args


@pytest.mark.parametrize('recompute_granularity', ['full', 'none'])
def test_non_selective_recompute_drops_selective_modules(monkeypatch, recompute_granularity):
    args = _run_argument_normalization(
        monkeypatch,
        recompute_granularity=recompute_granularity,
        recompute_method='uniform',
        recompute_modules=['core_attn'],
    )
    assert args.recompute_modules == []


def test_selective_recompute_preserves_modules(monkeypatch):
    args = _run_argument_normalization(
        monkeypatch,
        recompute_granularity='selective',
        recompute_method=None,
        recompute_modules=['core_attn'],
    )
    assert args.recompute_modules == ['core_attn']
