# LingBot-VLA v2 RobotWin SDK

This example loads the official LingBot-VLA v2 6B base checkpoint through TeleFuser and returns a structured
RobotWin action chunk. The current integration verifies the SDK contract without claiming policy quality: every
result is marked `policy_verified=False` and `verification_status="unverified_official_6b_base"`.

## Inputs

- Three RGB cameras in the upstream RobotWin order: high, left wrist, right wrist.
- A raw 14-dimensional RobotWin state.
- A non-empty task string.

The SDK applies the bundled upstream RobotWin `bounds_99_woclip` statistics and maps the observation into
LingBot's 55-dimensional canonical state.

## Output

The pipeline returns `LingBotVlaV2ActionChunk` with:

- `fields["action.arm.position"]`: `[H, 12]`.
- `fields["action.effector.position"]`: `[H, 2]`.
- `raw_actions` and `fields["action"]`: reconstructed `[H, 14]` RobotWin actions.
- `action_mask`: the 55-dimensional canonical RobotWin action mask.
- `horizon`: action chunk length, normally 50 for the official base config.
- `canonical_normalized_actions`: optional `[H, 55]` debugging output.

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
  --output action_chunk.npz
```

The example saves named arrays and verification metadata in an `.npz` file. Do not send the output to a robot until
the official 6B GPU smoke test and policy-level parity validation are complete.
