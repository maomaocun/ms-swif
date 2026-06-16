# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class FlopsEstimate:
    fwd_flops: float
    train_flops: float
    hardware_flops: float
    estimator: str
    attention_layers: int
    linear_attention_layers: int
    dense_mlp_layers: int
    moe_layers: int


def _get_attr(*objects: Any, names, default=None):
    if isinstance(names, str):
        names = (names, )
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            if isinstance(obj, dict):
                value = obj.get(name, None)
            else:
                value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _as_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _read_text_config(args) -> Dict[str, Any]:
    for attr in ('model_dir', 'model'):
        model_dir = getattr(args, attr, None)
        if not model_dir:
            continue
        config_path = os.path.join(os.path.expanduser(str(model_dir)), 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            continue
        text_config = config.get('text_config')
        return text_config if isinstance(text_config, dict) else config
    return {}


def _layer_type_counts(layer_types, num_layers: int) -> Optional[Tuple[int, int]]:
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        return None
    layer_types = list(layer_types)[:num_layers]
    linear_layers = sum(1 for layer_type in layer_types if str(layer_type) == 'linear_attention')
    attention_layers = sum(1 for layer_type in layer_types if str(layer_type) in {'full_attention', 'attention'})
    if linear_layers or attention_layers:
        return attention_layers, linear_layers
    return None


def _attention_layer_counts(config, args, text_config: Dict[str, Any], num_layers: int) -> Tuple[int, int]:
    layer_types = _get_attr(config, args, text_config, names=('layer_types', 'layers_block_type'))
    counts = _layer_type_counts(layer_types, num_layers)
    if counts is not None:
        return counts

    linear_attention_freq = _get_attr(config, args, text_config, names='linear_attention_freq')
    if isinstance(linear_attention_freq, (list, tuple)):
        pattern = list(linear_attention_freq)[:num_layers]
        linear_layers = sum(1 for item in pattern if int(item) > 0)
        return num_layers - linear_layers, linear_layers
    freq = _as_int(linear_attention_freq)
    if freq and freq > 0:
        linear_layers = sum(1 for i in range(num_layers) if (i + 1) % freq != 0)
        return num_layers - linear_layers, linear_layers

    full_attention_interval = _as_int(_get_attr(text_config, names='full_attention_interval'))
    if full_attention_interval and full_attention_interval > 0:
        attention_layers = sum(1 for i in range(num_layers) if (i + 1) % full_attention_interval == 0)
        return attention_layers, num_layers - attention_layers

    variant = _get_attr(config, args, text_config, names='experimental_attention_variant')
    if variant:
        return 0, num_layers
    return num_layers, 0


def _moe_layer_counts(config, args, num_layers: int) -> Tuple[int, int, bool]:
    num_experts = _get_attr(config, args, names=('num_moe_experts', 'num_experts'))
    if num_experts is None:
        return num_layers, 0, False

    moe_layer_freq = _get_attr(config, args, names='moe_layer_freq')
    if isinstance(moe_layer_freq, (list, tuple)):
        pattern = list(moe_layer_freq)[:num_layers]
        moe_layers = sum(1 for item in pattern if int(item) > 0)
        last_layer_is_moe = bool(pattern[-1]) if pattern else False
    else:
        freq = _as_int(moe_layer_freq, 1) or 1
        moe_layers = sum(1 for i in range(num_layers) if i % freq == 0)
        last_layer_is_moe = (num_layers - 1) % freq == 0
    return num_layers - moe_layers, moe_layers, last_layer_is_moe


def _recompute_multiplier(config, args) -> float:
    override = _as_float(os.environ.get('SWIFT_MFU_HARDWARE_FLOPS_MULTIPLIER'))
    if override and override > 0:
        return override
    recompute = _get_attr(config, args, names='recompute_granularity')
    if recompute == 'full':
        return 4.0
    return 3.0


def estimate_training_flops(config, args, logs: Dict[str, Any]) -> Optional[FlopsEstimate]:
    """Estimate per-log-window FLOPs from model config and observed sequence shapes.

    ``train_flops`` follows the Megatron convention: forward + backward dgrad/wgrad
    is 3x forward FLOPs. ``hardware_flops`` additionally accounts for full activation
    recompute when enabled, so it can be compared with device peak throughput as HFU.
    """
    total_tokens = _as_float(logs.get('total_tokens'))
    if not total_tokens or total_tokens <= 0:
        return None
    attention_seq_len_sq_sum = _as_float(logs.get('_attention_seq_len_sq_sum'), 0.0) or 0.0

    text_config = _read_text_config(args)
    num_layers = _as_int(_get_attr(config, args, text_config, names=('num_layers', 'num_hidden_layers')))
    hidden_size = _as_int(_get_attr(config, args, text_config, names='hidden_size'))
    num_attention_heads = _as_int(_get_attr(config, args, text_config, names='num_attention_heads'))
    ffn_hidden_size = _as_int(_get_attr(config, args, text_config, names=('ffn_hidden_size', 'intermediate_size')))
    vocab_size = _as_int(_get_attr(config, args, text_config, names=('padded_vocab_size', 'vocab_size')))
    if not all([num_layers, hidden_size, num_attention_heads, ffn_hidden_size, vocab_size]):
        return None

    mtp_num_layers = _as_int(_get_attr(args, config, names='mtp_num_layers'), 0)
    mtp_num_layers = mtp_num_layers or 0
    counted_layers = num_layers + mtp_num_layers
    attention_layers, linear_attention_layers = _attention_layer_counts(config, args, text_config, counted_layers)
    dense_mlp_layers, moe_layers, last_layer_is_moe = _moe_layer_counts(config, args, num_layers)
    if mtp_num_layers:
        moe_layers += mtp_num_layers if last_layer_is_moe else 0
        dense_mlp_layers += 0 if last_layer_is_moe else mtp_num_layers

    swiglu = _as_bool(_get_attr(config, args, text_config, names='swiglu'), True)
    ffn_expansion_factor = 3 if swiglu else 2
    linear_coeff = 0.0
    quad_coeff = 0.0

    kv_channels = _as_int(_get_attr(config, args, text_config, names=('kv_channels', 'head_dim')))
    if not kv_channels:
        kv_channels = max(hidden_size // max(num_attention_heads, 1), 1)
    num_query_groups = _as_int(
        _get_attr(config, args, text_config, names=('num_query_groups', 'num_key_value_heads')), num_attention_heads)
    num_query_groups = num_query_groups or num_attention_heads
    attention_output_gate = _as_bool(
        _get_attr(config, args, text_config, names=('attention_output_gate', 'attn_output_gate')), False)

    if attention_layers:
        if _as_bool(_get_attr(config, args, text_config, names='multi_latent_attention'), False):
            q_lora_rank = _get_attr(config, args, text_config, names='q_lora_rank')
            kv_lora_rank = _as_int(_get_attr(config, args, text_config, names='kv_lora_rank'), 0) or 0
            qk_head_dim = _as_int(_get_attr(config, args, text_config, names='qk_head_dim'), kv_channels) or kv_channels
            qk_pos_emb_head_dim = _as_int(
                _get_attr(config, args, text_config, names='qk_pos_emb_head_dim'), 0) or 0
            v_head_dim = _as_int(_get_attr(config, args, text_config, names='v_head_dim'), kv_channels) or kv_channels
            if q_lora_rank is None:
                q_term = hidden_size * num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim)
            else:
                q_lora_rank = _as_int(q_lora_rank, 0) or 0
                q_term = q_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)
            kv_term = kv_lora_rank * (hidden_size + num_attention_heads * (qk_head_dim + v_head_dim) + 1)
            kv_term += hidden_size * qk_pos_emb_head_dim
            out_proj = (num_attention_heads * v_head_dim) * hidden_size
            linear_coeff += 2.0 * (q_term + kv_term + out_proj) * attention_layers
            quad_coeff += num_attention_heads * (qk_head_dim + qk_pos_emb_head_dim + v_head_dim) * attention_layers
        else:
            query_projection_size = kv_channels * num_attention_heads
            key_projection_size = kv_channels * num_query_groups
            value_projection_size = kv_channels * num_query_groups
            gate_projection_size = query_projection_size if attention_output_gate else 0
            linear_coeff += 2.0 * (
                hidden_size
                * (query_projection_size + key_projection_size + value_projection_size + gate_projection_size)
                + query_projection_size * hidden_size) * attention_layers
            quad_coeff += 2.0 * query_projection_size * attention_layers

    if linear_attention_layers:
        variant = str(_get_attr(config, args, text_config, names='experimental_attention_variant') or '')
        if variant == 'gated_delta_net':
            qk_head_dim = _as_int(_get_attr(config, args, text_config, names='linear_key_head_dim'), 128) or 128
            v_head_dim = _as_int(_get_attr(config, args, text_config, names='linear_value_head_dim'), 128) or 128
            num_qk_heads = _as_int(_get_attr(config, args, text_config, names='linear_num_key_heads'), 16) or 16
            num_v_heads = _as_int(_get_attr(config, args, text_config, names='linear_num_value_heads'), 32) or 32
            conv_kernel_dim = _as_int(_get_attr(config, args, text_config, names='linear_conv_kernel_dim'), 4) or 4
            qk_dim = qk_head_dim * num_qk_heads
            v_dim = v_head_dim * num_v_heads
            linear_coeff += 2.0 * (
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                + num_v_heads * (v_head_dim**2) * 4
                + hidden_size * v_dim) * linear_attention_layers
        else:
            query_projection_size = kv_channels * num_attention_heads
            key_projection_size = kv_channels * num_query_groups
            value_projection_size = kv_channels * num_query_groups
            linear_coeff += 2.0 * (
                hidden_size * (query_projection_size + key_projection_size + value_projection_size)
                + query_projection_size * hidden_size) * linear_attention_layers

    if dense_mlp_layers:
        linear_coeff += 2.0 * hidden_size * ffn_hidden_size * ffn_expansion_factor * dense_mlp_layers
    if moe_layers:
        moe_ffn_hidden_size = _as_int(
            _get_attr(config, args, names='moe_ffn_hidden_size'), ffn_hidden_size) or ffn_hidden_size
        topk = _as_int(_get_attr(config, args, names='moe_router_topk'), 1) or 1
        shared_ffn = _as_int(_get_attr(config, args, names='moe_shared_expert_intermediate_size'), 0) or 0
        moe_latent_size = _as_int(_get_attr(config, args, names='moe_latent_size'))
        if moe_latent_size is None:
            moe_coeff = 2.0 * hidden_size * (
                moe_ffn_hidden_size * topk * ffn_expansion_factor + shared_ffn * ffn_expansion_factor)
        else:
            moe_coeff = 2.0 * (
                moe_ffn_hidden_size * topk * ffn_expansion_factor * moe_latent_size
                + 2 * hidden_size * moe_latent_size
                + hidden_size * shared_ffn * ffn_expansion_factor)
        linear_coeff += moe_coeff * moe_layers

    if mtp_num_layers:
        linear_coeff += 2.0 * mtp_num_layers * (3 * hidden_size + 2 * hidden_size * hidden_size)
    linear_coeff += 2.0 * hidden_size * vocab_size * (mtp_num_layers + 1)

    fwd_flops = linear_coeff * total_tokens + quad_coeff * attention_seq_len_sq_sum
    train_flops = 3.0 * fwd_flops
    hardware_flops = _recompute_multiplier(config, args) * fwd_flops
    estimator = 'megatron_config_v1'
    if linear_attention_layers:
        estimator += '+linear_attention'
    if moe_layers:
        estimator += '+moe'
    if _recompute_multiplier(config, args) > 3.0:
        estimator += '+full_recompute_hfu'
    return FlopsEstimate(
        fwd_flops=fwd_flops,
        train_flops=train_flops,
        hardware_flops=hardware_flops,
        estimator=estimator,
        attention_layers=attention_layers,
        linear_attention_layers=linear_attention_layers,
        dense_mlp_layers=dense_mlp_layers,
        moe_layers=moe_layers,
    )
