# ABot-World 0.5B-LF

This example exposes one local single-GPU entry point:

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 \
  --port 7860
```

The browser controls WASD/arrow movement and IJKL camera rotation. Connecting
creates the image-conditioned causal session but does not advance the DiT
until a non-empty control state is received. Generated blocks remain ordered
in a bounded FIFO and the producer waits when the browser is behind.
The six sink latents and rolling tail use fixed logical RoPE positions, so the
global session frame number does not index beyond the trained local window.

## Test tiers

CPU contract tests cover action-channel layout, checkpoint conversion, sink
KV rolling, RoPE boundary validation, session cleanup, and the direct runtime
idle/FIFO behavior:

```bash
pytest tests/unit/pipelines/abot_world
```

The 30-block GPU smoke is opt-in because it loads the release checkpoint:

```bash
ABOT_WORLD_MODEL_ROOT=/path/to/ABot-World-0-5B-LF \
ABOT_WORLD_TEST_IMAGE=/path/to/initial.png \
pytest -m "gpu and slow" tests/integration/test_abot_world_smoke.py -v
```

The smoke uses the public 480x832 shape, a fixed seed, and a fixed control
state. It checks that every block decodes frames and that the session's
emitted-frame counter matches the observed count. It is a generation contract
test, not a visual-quality or long-horizon parity claim.
