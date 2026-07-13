import torch
from types import SimpleNamespace

from swift.template.base import Template


def _make_padding_free_megatron_template():
    template = object.__new__(Template)
    template.processor = SimpleNamespace(pad_token_id=0)
    template.padding_side = 'right'
    template.mode = 'train'
    template.use_megatron = True
    template.padding_free = True
    template.sequence_parallel_size = 1
    return template


def test_get_pad_target_length_accepts_cached_dataset_values():
    batch = [
        {
            'pad_target_length': 8
        },
        {
            'pad_target_length': torch.tensor([12])
        },
        {
            '_extra_kwargs': {
                'pad_target_length': [16]
            }
        },
    ]
    assert Template._get_pad_target_length(batch) == 16


def test_resolve_padding_to_uses_absolute_target_and_parallel_multiple():
    assert Template._resolve_padding_to([10], 8, 17) == 24
    assert Template._resolve_padding_to([20], 8, 17) == 24
    assert Template._resolve_padding_to([10], 8, None) == 16


def test_padding_free_collator_pads_to_cached_target():
    template = _make_padding_free_megatron_template()
    batch = [{
        'input_ids': [10, 11, 12],
        'labels': [-100, 11, 12],
        'position_ids': [0, 1, 2],
        'length': 3,
        'pad_target_length': 8,
    }]

    result = template._data_collator(batch, padding_to=4)

    assert result['input_ids'].tolist() == [[10, 11, 12, 0, 0, 0, 0, 0]]
    assert result['labels'].tolist() == [[-100, 11, 12, -100, -100, -100, -100, -100]]
    assert result['position_ids'].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
