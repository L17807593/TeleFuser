# LingBot-VLA v2 Base Model SDK

This example loads the official LingBot-VLA v2 6B base checkpoint through TeleFuser and returns its normalized
55-dimensional canonical action chunk. The RobotWin profile is used only to prepare the example observation; the
result is not converted to physical RobotWin actions.

## Inputs

- Three RGB cameras in the upstream RobotWin order: high, left wrist, right wrist.
- A raw 14-dimensional RobotWin state.
- A non-empty task string.

The SDK applies the bundled upstream RobotWin `bounds_99_woclip` statistics and maps the observation into
LingBot's 55-dimensional canonical state.

## Output

The pipeline returns `LingBotVlaV2CanonicalActionChunk` with:

- `canonical_normalized_actions`: `[H, 55]` base-model output.
- `horizon`: action chunk length, normally 50 for the official base config.
- `action_dim`: canonical action dimension, normally 55.
- `checkpoint_variant`: `base`.
- `policy_verified=False` and `verification_status="unverified_official_6b_base"`.

## Checkpoints

The VLA directory must contain `model.safetensors.index.json` and every referenced shard. The Qwen3-VL directory
supplies the visual-language backbone configuration and processor.

## Example

```bash
python examples/lingbot_vla_v2/lingbot_vla_v2_inference.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' \
  --output canonical_action_chunk.npz
```

The example saves canonical actions and checkpoint metadata in an `.npz` file. The base output must not be sent to
a robot without an embodiment-specific post-training checkpoint, action mapping, and policy validation.

## Minimal Single-GPU HTTP Service

The VLA-specific server loads one policy replica and serializes all inference calls on the selected GPU. It does not
use the shared media service, Ray, multi-GPU execution, dynamic batching, or robot control. Start it from the repository
with the isolated VLA environment:

```bash
.venv-vla/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_server.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8000
```

The process reports ready only after both the processor and policy have loaded:

```bash
curl http://127.0.0.1:8000/health
```

`POST /v1/vla/actions` accepts raw Base64 or a Base64 data URL for each camera. The state must contain exactly 14
finite values. For example:

```bash
.venv-vla/bin/python - <<'PY'
import base64
from pathlib import Path

import httpx


def encode(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


response = httpx.post(
    "http://127.0.0.1:8000/v1/vla/actions",
    json={
        "task": "pick up the red block",
        "state": [0.0] * 14,
        "camera_high": encode("/data/cam_high.png"),
        "camera_left_wrist": encode("/data/cam_left_wrist.png"),
        "camera_right_wrist": encode("/data/cam_right_wrist.png"),
        "seed": 7,
    },
    timeout=300.0,
)
response.raise_for_status()
print(response.json())
PY
```

The response contains `canonical_normalized_actions`, `horizon`, `action_dim`, `checkpoint_variant`,
`policy_verified`, and `verification_status`. A successful HTTP response confirms service and model execution only;
the normalized base-model output is not a physical robot command.

## Native TeleFuser Service

The native service uses the shared `PIPELINE_CONTRACT`, asynchronous task scheduler, pipeline pool, status API, runtime
metrics, and `TFClient`. It keeps the standalone endpoint above as a small debugging path.

The example resolves checkpoints under the existing `TF_MODEL_ZOO_PATH` layout:

- `lingbot/lingbot-vla-v2-6b`
- `Qwen3-VL-4B-Instruct`

Start one replica on one visible GPU:

```bash
TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action \
  --parallelism 1 \
  --host 127.0.0.1 \
  --port 18080
```

Submit `POST /v1/tasks/structured` with `task="vla_action"`, an `instruction`, the 14-dimensional `state`, and
the three Base64 camera fields. The creation response contains a task ID. Poll
`GET /v1/tasks/{task_id}/status`; a completed response contains the action payload under `result` and includes
`inference_time_s` and the optional `peak_memory_mb`.

The unified client handles image encoding, submission, polling, and result extraction:

```python
from telefuser.client import TFClient

client = TFClient("http://127.0.0.1:18080")
actions = client.predict_vla_actions(
    instruction="pick up the red block",
    state=[0.0] * 14,
    camera_high_path="/data/cam_high.png",
    camera_left_wrist_path="/data/cam_left_wrist.png",
    camera_right_wrist_path="/data/cam_right_wrist.png",
    seed=7,
)
print(actions["horizon"], actions["action_dim"])
```

For independent replicas, expose one GPU per replica through the existing pipeline pool:

```bash
CUDA_VISIBLE_DEVICES=0,1 TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action \
  --parallelism 2 \
  --num-replicas 2 \
  --port 18080
```

This is request-level replication, not tensor parallelism inside one policy replica. The response remains a normalized
base-model canonical action chunk and must not be treated as a physical robot command.

## TeleFuser Regression Baseline

The validation capture runs through the public loader and pipeline, then records preprocessing tensors, fixed initial
noise, every flow-matching `x_t` and velocity step, and the final canonical action. Run it twice before changing VLA
model code to establish and verify a strict local baseline:

```bash
.venv-vla/bin/python tools/validation/capture_lingbot_vla_v2_telefuser.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' \
  --seed 7 \
  --deterministic-moe \
  --output work_dirs/vla_regression/baseline_seed7.npz

# Repeat the same command with:
#   --output work_dirs/vla_regression/replay_seed7.npz

.venv-vla/bin/python tools/validation/run_lingbot_vla_v2_parity.py \
  --reference work_dirs/vla_regression/baseline_seed7.npz \
  --candidate work_dirs/vla_regression/replay_seed7.npz \
  --profile strict \
  --output work_dirs/vla_regression/strict_report.json
```

Each `.npz` has a same-name `.json` sidecar containing the checkpoint, processor, input, runtime, and tensor contract
metadata. The default checkpoint identity is a fast filename-and-size manifest. Add `--full-checkpoint-hash` when a
content hash of every checkpoint shard is required. Keep generated artifacts under `work_dirs`; do not commit them.

This is a TeleFuser regression check, not upstream parity. It detects changes to the current implementation but does
not establish equivalence with the official repository.

## Official Upstream Parity

The strict upstream baseline pins `Robbyant/lingbot-vla-v2` at commit
`be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`. Keep the checkout, uv environment, cache, and artifacts under
`work_dirs`; Git ignores them. Create the isolated runtime with:

```bash
mkdir -p work_dirs/.uv-cache-upstream work_dirs/.uv-tmp-upstream
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv venv work_dirs/.venv-lingbot-upstream --python .venv-vla/bin/python
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python -r tools/validation/requirements-lingbot-vla-v2-upstream.txt
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
git clone https://github.com/Robbyant/lingbot-vla-v2 work_dirs/lingbot-vla-v2-upstream
git -C work_dirs/lingbot-vla-v2-upstream checkout be27333c9b5f2663b0ec33f069dd7dfd67fa32b5
```

Generate the reference with `capture_lingbot_vla_v2_upstream.py` in the upstream uv environment and the candidate
with `capture_lingbot_vla_v2_telefuser.py` in `.venv-vla`. Pass identical model, processor, camera, task, state, seed,
and device arguments to both commands, add `--deterministic-moe`, and pass `--upstream-root` to the upstream command.
Then compare them with the strict comparator shown above. Generated artifacts belong in `work_dirs/vla_upstream_parity`.

This is a minimal inference-parity runtime, not a LeRobot training environment. The upstream setup itself combines
LeRobot 0.4.2 metadata constraints with versions outside those constraints, so LeRobot is installed with `--no-deps`;
the capture import and end-to-end run are the runtime checks.

The official code hard-codes FlashAttention during construction. The upstream capture replaces that selection only
inside its validation process so both sides use eager attention on the Python 3.10.12 / PyTorch 2.11 stack. Production
inference keeps the upstream Triton MoE path through `telefuser.ops`; strict capture uses `--deterministic-moe` because
the upstream kernel uses atomic accumulation and is not bitwise repeatable across separate processes. Artifact metadata
records both `attention_backend` and `moe_backend`, and the comparator rejects mixed-backend artifacts.
