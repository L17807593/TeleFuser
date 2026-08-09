# SwiftVR Video Restoration

This example restores videos with the released `H-oliday/SwiftVR` checkpoint on
one CUDA GPU. The parity path uses BF16, dense Torch SDPA, the upstream fixed
timestep, and the original 24-frame causal chunk protocol.

The checkpoint root must contain:

```text
reae.safetensors
prompt_embedding.safetensors
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
```

Run offline restoration with:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /data/SwiftVR \
  --height 360 \
  --width 640 \
  --scale 3 \
  --output restored_1080p.mp4
```

Like the FlashVSR example, the file exposes `get_pipeline()` for model loading
and `run()` for loaded PIL frames. Its CLI accepts the same common video options:
`input_video`, `scale`, `height`, `width`, `gpu_num`, `model_root`, and `output`.
It defaults to FlashVSR's versioned `examples/data/dag.mp4` test video. The
SwiftVR-specific 24-frame causal protocol remains an internal pipeline default.
The command above resizes the low-resolution input to `640x360` and restores a
`1920x1080` video. A separate warmup session covers two full chunks and the
actual tail shape to absorb cold cuDNN plan selection. The reported processing
FPS then covers stateful inference and output transfer, while end-to-end FPS
also includes H.264 encoding.

The direct causal API accepts uint8 `[T,H,W,3]` tensors and returns BF16
`[1,T,3,H,W]` tensors in `[0,1]`:

```python
import torch

from examples.swiftvr.swiftvr_restore_h100 import get_pipeline

pipeline = get_pipeline(model_root="/data/SwiftVR")
session = pipeline.stream(resolution=(1920, 1080), clip_len=24, dit_overlap=1)
try:
    first = session.step(torch.zeros((24, 540, 960, 3), dtype=torch.uint8))
    tail = session.step(torch.zeros((5, 540, 960, 3), dtype=torch.uint8))
    flushed = session.flush()
finally:
    session.close()
```

## H100 performance

The following baseline was measured on one NVIDIA H100 80GB HBM3 with BF16,
eager Torch SDPA, and 24-frame chunks. The values are synchronized target-side
compute metrics; model loading, video encoding, network delivery, and client
playback are excluded.

| Output resolution | Compute FPS | TTFC | P50 / P95 chunk latency | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 47.28 | 15.01 s | 0.506 / 0.511 s | 27.08 GiB |
| 2560x1440 | 26.50 | 16.31 s | 0.895 / 0.960 s | 35.93 GiB |

The full example command above was also verified on one H100 with all 81 frames
of `dag.mp4`. The measured session produced `1920x1080` frames at 34.78 FPS;
including H.264 encoding, end-to-end throughput was 22.18 FPS versus the source
rate of 16 FPS. The output contains all 81 frames at 16 FPS. These figures
exclude one-time model loading, input decoding, and shape warmup.

TTFC includes cold cuDNN benchmarking and kernel-plan setup after model
loading. In the 1080p ten-minute stability run, SwiftVR processed 28,224 frames
in 1,176 chunks at 47.01 compute FPS. P50/P95/max chunk latency was
0.508/0.523/0.654 seconds, with no latency growth between the beginning and end
of the run. Session cleanup returned all device memory to the driver.

`SwiftVRPipeline.stream()` is the supported stateful interface. The example
does not expose `get_service()` because the current LiveKit protocol has no
video-input transport; presenting a partial service adapter would not make the
model usable through `telefuser stream-serve`.

See [the SwiftVR integration guide](../../docs/en/swiftvr.md) for checkpoint
loading, constraints, and measured results.
