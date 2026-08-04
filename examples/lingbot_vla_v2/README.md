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
