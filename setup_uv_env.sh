#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

run() {
  log "RUN: $*"
  "$@"
}

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
HF_VENV_DIR="${HF_VENV_DIR:-${SCRIPT_DIR}/.venv}"
MEGATRON_VENV_DIR="${MEGATRON_VENV_DIR:-${SCRIPT_DIR}/.venv-megatron}"
GLOBAL_VENV_DIR="${GLOBAL_VENV_DIR:-/mnt/cpfs/yangyicun/.venv}"

RECREATE="${RECREATE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
APPLY_CHUNKED_CE_PATCH="${APPLY_CHUNKED_CE_PATCH:-1}"
RUN_DRY_RUN="${RUN_DRY_RUN:-1}"

MCORE_BRIDGE_VERSION="${MCORE_BRIDGE_VERSION:-1.4.0}"
MEGATRON_CORE_VERSION="${MEGATRON_CORE_VERSION:-0.17.0}"
PEFT_VERSION="${PEFT_VERSION:-0.19.1}"
DEEPSPEED_VERSION="${DEEPSPEED_VERSION:-0.19.0}"

[[ -n "${UV_BIN}" ]] || die "uv not found. Install uv or set UV_BIN=/path/to/uv."

if [[ "${RECREATE}" == "1" ]]; then
  log "Removing existing venvs because RECREATE=1"
  rm -rf "${HF_VENV_DIR}" "${MEGATRON_VENV_DIR}"
fi

create_venv() {
  local venv_dir="$1"
  local prompt="$2"
  if [[ -x "${venv_dir}/bin/python" ]]; then
    log "Reuse existing venv: ${venv_dir}"
    return
  fi
  run "${UV_BIN}" venv --python "${PYTHON_BIN}" --system-site-packages --prompt "${prompt}" "${venv_dir}"
}

uv_pip() {
  local python_bin="$1"
  shift
  run "${UV_BIN}" pip install --python "${python_bin}" "$@"
}

python_has_module() {
  local python_bin="$1"
  local module="$2"
  "${python_bin}" - <<PY >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("${module}") else 1)
PY
}

python_has_distribution() {
  local python_bin="$1"
  local dist="$2"
  "${python_bin}" - <<PY >/dev/null 2>&1
import importlib.metadata as md
try:
    md.version("${dist}")
except md.PackageNotFoundError:
    raise SystemExit(1)
PY
}

install_hf_env() {
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    log "Skip HF/DeepSpeed dependency installation because SKIP_INSTALL=1"
    return
  fi

  local py="${HF_VENV_DIR}/bin/python"
  if python_has_module "${py}" torch && python_has_module "${py}" transformers && python_has_module "${py}" deepspeed; then
    log "HF/DeepSpeed dependencies already importable"
  else
    log "Installing HF/DeepSpeed dependencies into ${HF_VENV_DIR}"
    uv_pip "${py}" -r requirements.txt
    uv_pip "${py}" "deepspeed==${DEEPSPEED_VERSION}" tensorboard
  fi

  if [[ ! -x "${HF_VENV_DIR}/bin/swift" ]]; then
    uv_pip "${py}" -e .
  fi
}

install_megatron_env() {
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    log "Skip Megatron dependency installation because SKIP_INSTALL=1"
    return
  fi

  local py="${MEGATRON_VENV_DIR}/bin/python"
  if python_has_distribution "${py}" mcore-bridge && python_has_distribution "${py}" megatron-core; then
    log "Megatron dependencies already installed"
  else
    log "Installing Megatron dependencies into ${MEGATRON_VENV_DIR}"
    uv_pip "${py}" \
      "mcore-bridge==${MCORE_BRIDGE_VERSION}" \
      "megatron-core==${MEGATRON_CORE_VERSION}" \
      "peft==${PEFT_VERSION}"
  fi

  if [[ ! -x "${MEGATRON_VENV_DIR}/bin/megatron" || ! -x "${MEGATRON_VENV_DIR}/bin/swift" ]]; then
    uv_pip "${py}" --no-deps -e .
  fi
}

apply_chunked_ce_patch() {
  [[ "${APPLY_CHUNKED_CE_PATCH}" == "1" ]] || return

  local py="${MEGATRON_VENV_DIR}/bin/python"
  log "Applying/verifying chunked linear CE patch"
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/megatron_env.sh"
  "${py}" - <<'PY'
from __future__ import annotations

from pathlib import Path
import mcore_bridge
import py_compile

bridge_root = Path(mcore_bridge.__file__).resolve().parent
target = bridge_root / "model" / "gpt_model.py"
text = target.read_text()

if "_ChunkedLinearCrossEntropy" in text:
    print(f"chunked linear CE patch already present: {target}")
    py_compile.compile(str(target), doraise=True)
    raise SystemExit(0)

helper = r'''

def _parse_linear_ce_chunk_size() -> int:
    raw_value = os.environ.get('LINEAR_CE_CHUNK_SIZE', '').strip().lower()
    if raw_value in {'', '0', 'false', 'none', 'off'}:
        return 0
    multiplier = 1
    if raw_value.endswith('k'):
        multiplier = 1024
        raw_value = raw_value[:-1]
    elif raw_value.endswith('m'):
        multiplier = 1024 * 1024
        raw_value = raw_value[:-1]
    try:
        chunk_size = int(float(raw_value) * multiplier)
    except ValueError as exc:
        raise ValueError(
            f'LINEAR_CE_CHUNK_SIZE must be an integer token count, e.g. 2048 or 2k. Got: '
            f'{os.environ.get("LINEAR_CE_CHUNK_SIZE")!r}'
        ) from exc
    if chunk_size < 0:
        raise ValueError(f'LINEAR_CE_CHUNK_SIZE must be >= 0. Got: {chunk_size}')
    return chunk_size


def _tp_group_size(tp_group) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    if tp_group is None:
        return 1
    return torch.distributed.get_world_size(tp_group)


def _tp_all_reduce(tensor: torch.Tensor, op: torch.distributed.ReduceOp, tp_group) -> torch.Tensor:
    if _tp_group_size(tp_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=tp_group)
    return tensor


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Compute LM loss from hidden states in flattened-token chunks without full-sequence logits."""

    @staticmethod
    def forward(ctx, hidden_states, output_weight, labels, tp_group, vocab_start_index, chunk_size,
                reduce_grad_input):
        if labels.dim() != 2:
            raise ValueError(f'labels must be [batch, sequence], got shape: {tuple(labels.shape)}')
        if hidden_states.dim() != 3:
            raise ValueError(f'hidden_states must be [sequence, batch, hidden], got shape: {tuple(hidden_states.shape)}')
        seq_len, batch_size, hidden_size = hidden_states.shape
        if labels.shape != (batch_size, seq_len):
            raise ValueError(
                f'labels shape must match hidden states as [batch, sequence]. Got labels={tuple(labels.shape)}, '
                f'hidden_states={tuple(hidden_states.shape)}')
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be > 0. Got: {chunk_size}')

        labels_t = labels.transpose(0, 1).contiguous()
        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        target_flat = labels_t.view(-1)
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        losses_flat = torch.empty((seq_len * batch_size,), dtype=torch.float32, device=hidden_states.device)

        for chunk_start in range(0, hidden_flat.shape[0], chunk_size):
            chunk_end = min(hidden_flat.shape[0], chunk_start + chunk_size)
            target = target_flat[chunk_start:chunk_end]
            logits = torch.matmul(hidden_flat[chunk_start:chunk_end], output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group)

            target_mask = (target == -100) | (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            target_logits = torch.gather(logits, dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1)
            target_logits = target_logits.masked_fill(target_mask, 0.0)
            target_logits = _tp_all_reduce(target_logits, torch.distributed.ReduceOp.SUM, tp_group)

            chunk_loss = torch.log(global_sum) + global_max - target_logits
            chunk_loss = chunk_loss.masked_fill(target == -100, 0.0)
            losses_flat[chunk_start:chunk_end] = chunk_loss

        ctx.save_for_backward(hidden_states, output_weight, labels_t)
        ctx.tp_group = tp_group
        ctx.vocab_start_index = vocab_start_index
        ctx.chunk_size = chunk_size
        ctx.reduce_grad_input = reduce_grad_input
        return losses_flat.view(seq_len, batch_size).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, output_weight, labels_t = ctx.saved_tensors
        tp_group = ctx.tp_group
        vocab_start_index = ctx.vocab_start_index
        chunk_size = ctx.chunk_size
        reduce_grad_input = ctx.reduce_grad_input
        partition_vocab_size = output_weight.shape[0]
        vocab_end_index = vocab_start_index + partition_vocab_size
        seq_len, batch_size, hidden_size = hidden_states.shape

        hidden_flat = hidden_states.contiguous().view(seq_len * batch_size, hidden_size)
        target_flat = labels_t.view(-1)
        grad_output_flat = grad_output.transpose(0, 1).contiguous().view(-1).float()
        grad_hidden_flat = torch.zeros_like(hidden_flat) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(output_weight) if ctx.needs_input_grad[1] else None

        for chunk_start in range(0, hidden_flat.shape[0], chunk_size):
            chunk_end = min(hidden_flat.shape[0], chunk_start + chunk_size)
            hidden_chunk = hidden_flat[chunk_start:chunk_end]
            target = target_flat[chunk_start:chunk_end]
            logits = torch.matmul(hidden_chunk, output_weight.t()).float()

            local_max = logits.max(dim=-1).values
            global_max = _tp_all_reduce(local_max, torch.distributed.ReduceOp.MAX, tp_group)
            exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
            global_sum = _tp_all_reduce(exp_logits.sum(dim=-1), torch.distributed.ReduceOp.SUM, tp_group)
            grad_logits = exp_logits / global_sum.unsqueeze(-1)

            target_mask = (target == -100) | (target < vocab_start_index) | (target >= vocab_end_index)
            local_target = (target - vocab_start_index).masked_fill(target_mask, 0)
            subtract = (~target_mask).to(dtype=grad_logits.dtype).unsqueeze(-1)
            grad_logits.scatter_add_(dim=-1, index=local_target.unsqueeze(-1), src=-subtract)
            grad_logits = grad_logits.masked_fill((target == -100).unsqueeze(-1), 0.0)
            grad_logits.mul_(grad_output_flat[chunk_start:chunk_end].unsqueeze(-1))

            if grad_hidden_flat is not None:
                grad_hidden_chunk = torch.matmul(grad_logits, output_weight.float())
                if reduce_grad_input:
                    _tp_all_reduce(grad_hidden_chunk, torch.distributed.ReduceOp.SUM, tp_group)
                grad_hidden_flat[chunk_start:chunk_end] = grad_hidden_chunk.to(dtype=hidden_states.dtype)

            if grad_weight is not None:
                grad_weight_chunk = torch.matmul(grad_logits.t(), hidden_chunk.float())
                grad_weight.add_(grad_weight_chunk.to(dtype=grad_weight.dtype))

        grad_hidden = grad_hidden_flat.view(seq_len, batch_size, hidden_size) if grad_hidden_flat is not None else None
        return grad_hidden, grad_weight, None, None, None, None, None


def _chunked_linear_cross_entropy_loss(model, hidden_states, output_weight, labels, chunk_size):
    if getattr(model.config, 'context_parallel_size', 1) != 1:
        raise ValueError('LINEAR_CE_CHUNK_SIZE does not support context_parallel_size > 1.')
    if output_weight is None:
        output_weight = model.output_layer.weight
    if output_weight is None:
        raise ValueError('Unable to locate output layer weight for LINEAR_CE_CHUNK_SIZE.')

    if getattr(model.output_layer, 'sequence_parallel', False):
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True, group=model.pg_collection.tp)
        reduce_grad_input = False
    else:
        reduce_grad_input = _tp_group_size(model.pg_collection.tp) > 1

    vocab_start_index = torch.distributed.get_rank(model.pg_collection.tp) * output_weight.shape[0] \
        if _tp_group_size(model.pg_collection.tp) > 1 else 0
    return _ChunkedLinearCrossEntropy.apply(
        hidden_states, output_weight, labels, model.pg_collection.tp, vocab_start_index, chunk_size, reduce_grad_input)
'''

hook = r'''
        linear_ce_chunk_size = _parse_linear_ce_chunk_size()
        if (linear_ce_chunk_size > 0 and labels is not None and self.config.task_type == 'causal_lm'
                and not in_inference_mode):
            if runtime_gather_output:
                raise ValueError('LINEAR_CE_CHUNK_SIZE requires vocab-parallel output; runtime_gather_output must be false.')
            if getattr(self.config, 'use_mup', False):
                raise ValueError('LINEAR_CE_CHUNK_SIZE currently does not support MuP output scaling.')
            if not getattr(self, '_linear_ce_chunk_size_logged', False):
                logger.info(f'Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE={linear_ce_chunk_size}.')
                self._linear_ce_chunk_size_logged = True
            return _chunked_linear_cross_entropy_loss(self, hidden_states, output_weight, labels, linear_ce_chunk_size)

'''

helper_anchor = "mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')\n"
hook_anchor = "        if self.config.task_type == 'embedding':\n"

if helper_anchor not in text:
    raise RuntimeError(f"Cannot patch {target}: helper anchor not found")
if hook_anchor not in text:
    raise RuntimeError(f"Cannot patch {target}: hook anchor not found")

backup = target.with_suffix(target.suffix + ".bak-linear-ce")
if not backup.exists():
    backup.write_text(text)

text = text.replace(helper_anchor, helper_anchor + helper, 1)
text = text.replace(hook_anchor, hook + hook_anchor, 1)
target.write_text(text)
py_compile.compile(str(target), doraise=True)
print(f"patched chunked linear CE into: {target}")
PY
}

validate_env() {
  log "Validating environment"

  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/megatron_env.sh"

  command -v megatron >/dev/null || die "megatron command not found after sourcing megatron_env.sh"
  command -v swift >/dev/null || die "swift command not found after sourcing megatron_env.sh"

  python - <<'PY'
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
from pathlib import Path

required_modules = ["torch", "transformers", "deepspeed", "swift", "mcore_bridge"]
for name in required_modules:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing module: {name}")

from mcore_bridge.model import gpt_model
if not hasattr(gpt_model, "_ChunkedLinearCrossEntropy"):
    raise SystemExit("chunked linear CE patch is not present")

for dist in ["torch", "transformers", "deepspeed", "megatron-core", "mcore-bridge", "peft"]:
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist}: distribution metadata not found")

print(f"LINEAR_CE_CHUNK_SIZE default check: {os.environ.get('LINEAR_CE_CHUNK_SIZE', '<unset>')}")
print(f"gpt_model.py: {Path(gpt_model.__file__).resolve()}")
PY

  if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    if [[ -d "/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B" \
          && -f "${SCRIPT_DIR}/data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl" ]]; then
      log "Running 27B Megatron smoke dry-run"
      DRY_RUN=1 \
        SMOKE=1 \
        RUN_NAME=setup-dry-run \
        OUTPUT_ROOT=/tmp/ms-swift-setup-outputs \
        LOG_ROOT=/tmp/ms-swift-setup-logs \
        bash "${SCRIPT_DIR}/train_qwen36_27b_paper2arm_distill_megatron.sh" >/tmp/ms-swift-setup-dry-run.log
      log "Dry-run ok. Log: /tmp/ms-swift-setup-dry-run.log"
    else
      warn "Skip train dry-run because model or dataset path is missing."
    fi
  fi
}

log "ms-swift SFT environment setup"
log "SCRIPT_DIR=${SCRIPT_DIR}"
log "HF_VENV_DIR=${HF_VENV_DIR}"
log "MEGATRON_VENV_DIR=${MEGATRON_VENV_DIR}"
log "GLOBAL_VENV_DIR=${GLOBAL_VENV_DIR}"
log "RECREATE=${RECREATE} SKIP_INSTALL=${SKIP_INSTALL} APPLY_CHUNKED_CE_PATCH=${APPLY_CHUNKED_CE_PATCH}"

if [[ ! -d "${GLOBAL_VENV_DIR}" ]]; then
  warn "GLOBAL_VENV_DIR does not exist: ${GLOBAL_VENV_DIR}. Continuing with system site-packages only."
fi

create_venv "${HF_VENV_DIR}" "ms-swift"
create_venv "${MEGATRON_VENV_DIR}" "ms-swift-megatron"
install_hf_env
install_megatron_env
apply_chunked_ce_patch
validate_env

log "Environment is ready."
log "Megatron usage:"
log "  cd ${SCRIPT_DIR}"
log "  bash train_qwen36_27b_paper2arm_distill_megatron.sh"
log "Recreate from scratch:"
log "  RECREATE=1 bash setup_uv_env.sh"
