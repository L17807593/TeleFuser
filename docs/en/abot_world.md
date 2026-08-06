# ABot-World 0.5B-LF

TeleFuser provides a single-GPU, direct-browser integration for the public
ABot-World 0.5B-LF long-forcing checkpoint. The supported entry point is the
native HTTP controller:

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 \
  --port 7860
```

This integration does not use `telefuser stream-serve`, LiveKit, WebRTC, or a
TURN server. The browser speaks to the local HTTP controller. For an SSH
session, forward the port from the remote host:

```bash
ssh -N -p <ssh-port> -L 7860:127.0.0.1:7860 <user>@<server>
```

## Checkpoint And Image

The loader expects the unmodified checkpoint layout:

```text
ABot-World-0-5B-LF/
  diffusion_pytorch_model.safetensors
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
```

The browser's default sample image comes from the official ABot-World web
client asset at `../ABot-World/web_client/datasets/images/84b90ad568b693d2.png`.
Pass a different server-side image path from the page when needed.

## Pipeline Structure

`ABotWorldPipeline` is a TeleFuser `BasePipeline`. It uses the existing Wan
VAE and text stages plus the model-specific `ABotWorldDenoisingStage`.
`ABotWorldDiT` uses the public TeleFuser attention operations and the official
four-step x0-prediction causal sampler.

`ABotWorldInteractivePipeline` retains the prompt embedding, initial image
latent, self/cross KV caches, scheduler, RNG, and VAE temporal cache between
control blocks. The initial integration supports one GPU and one retained
causal session.

## Controls And Idle Behavior

WASD and arrow keys control movement. IJKL controls camera rotation. Connect
creates the image-conditioned session and displays the input preview; it does
not advance the DiT with an empty action state. A non-empty control snapshot
starts the next three-latent causal block. Releasing all keys stops new model
execution without discarding frames already queued for playback.

The browser consumes decoded frames in order at 12 FPS. The bounded FIFO
applies producer backpressure when playback is behind, so normal playback does
not drop generated blocks.

## KV And RoPE

The default causal window is 18 latent frames: six sink frames plus a
twelve-frame rolling tail. KV cache entries remain unrotated; RoPE is applied
when keys are read using bounded logical positions. Sink positions are `0..5`
and the rolling tail occupies the remaining local window, so the global session
frame number does not grow the RoPE index past the precomputed table.

This fixed logical position policy is an intentional difference from the
original non-sink ABot baseline and must be evaluated as part of any future
long-horizon quality claim.

## Tests

CPU contract tests cover model conversion, sink KV rolling, RoPE boundaries,
session cleanup, idle behavior, FIFO backpressure, and action layout:

```bash
python -m pytest tests/unit/models/test_wan22_video_vae.py \
  tests/unit/pipelines/abot_world -q
```

The opt-in GPU smoke uses the release checkpoint, the public `480x832` shape,
a fixed seed, and 30 control blocks:

```bash
CUDA_VISIBLE_DEVICES=0 \
ABOT_WORLD_MODEL_ROOT=/path/to/ABot-World-0-5B-LF \
ABOT_WORLD_TEST_IMAGE=/path/to/initial.png \
python -m pytest -m "gpu and slow" \
  tests/integration/test_abot_world_smoke.py -v -s
```

The smoke is a generation and cache contract test. It does not establish
visual quality, prompt fidelity, or parity over an unbounded session.

## Scope

The first integration is intentionally single-GPU and direct HTTP. LiveKit or
`stream-serve` support requires a separate transport integration and is not
part of this example.
