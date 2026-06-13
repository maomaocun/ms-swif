# Saved Image Training Environment

This directory contains small wrappers for reusing the saved DSW image that has
the `/usr/local/bin/python` environment, FlashAttention 3 Hopper build, FLA, and
Transformer Engine FP8 support installed.

Do not put Aliyun passwords or AccessKey secrets in these scripts. Use the
console, RAM role, `aliyun configure`, or short-lived environment variables.

## 1. Validate the saved image locally

After the image is saved and available in the container registry:

```bash
export IMAGE_URI='REGISTRY/namespace/image:tag'

docker login REGISTRY
docker pull "${IMAGE_URI}"
docker run --rm -it --gpus all --ipc=host --shm-size=256g \
  -v /mnt:/mnt \
  -w /mnt/workspace/ms-swif \
  "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh verify
```

The verification checks:

- CUDA device is Hopper/SM90 class.
- FlashAttention 3 Hopper kernel runs.
- FLA `chunk_gated_delta_rule` runs for the linear-attention path.
- Transformer Engine reports FP8 support.
- TE FP8 Linear and SwiGLU MLP forward/backward run.

## 2. Dry-run the 27B launch

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=256g \
  -v /mnt:/mnt \
  -w /mnt/workspace/ms-swif \
  -e MODEL_PATH=/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B \
  -e DATASET_PATH=/mnt/cpfs/path/to/train.jsonl \
  -e REPORT_TO=none \
  "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh dry-run
```

## 3. Smoke or full train

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=256g \
  -v /mnt:/mnt \
  -w /mnt/workspace/ms-swif \
  -e MODEL_PATH=/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B \
  -e DATASET_PATH=/mnt/cpfs/path/to/train.jsonl \
  -e OUTPUT_ROOT=/mnt/cpfs/yangyicun/data/agent_checkpoints/qwen36-27b-hopper \
  -e REPORT_TO=none \
  "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh smoke
```

Change the last argument from `smoke` to `train` for a full run.

Default Hopper settings in the wrapper:

- `MEGATRON_PYTHON=/usr/local/bin/python`
- `FP8_FORMAT=hybrid`
- `FP8_RECIPE=delayed`
- `FP8_PARAM_GATHER=false`
- local compile caches under `local_cache/image-run-qwen36-27b-*`

## 4. Submit DLC with the saved image

The existing DLC config can be reused while replacing only the image URI:

```bash
export IMAGE_URI='REGISTRY/namespace/image:tag'
DRY_RUN=1 bash training_script/submit_dlc_with_saved_image.sh
bash training_script/submit_dlc_with_saved_image.sh
```

The wrapper creates a temporary JSON config and sets `.dlc.worker_image` to
`IMAGE_URI`. It does not modify `dlc/config_qwen36_27b_dsv4pro_distill.json`.

If the saved image uses a different region registry endpoint, make sure the DLC
cluster can pull from that endpoint or copy the image to the same region as the
training cluster.
