# SwiftVR Video Restoration

This example restores videos with the released `H-oliday/SwiftVR` checkpoint on
one CUDA GPU. The parity path uses BF16, dense Torch SDPA, the upstream fixed
timestep, and the original 24-frame causal chunk protocol.

## Resources

| Resource | Link |
| --- | --- |
| SwiftVR source | [H-oliday/SwiftVR](https://github.com/H-oliday/SwiftVR) |
| Pretrained checkpoint | [H-oliday/SwiftVR on Hugging Face](https://huggingface.co/H-oliday/SwiftVR) |
| Project page | [h-oliday.github.io/SwiftVR](https://h-oliday.github.io/SwiftVR) |
| Paper | [arXiv:2606.09516](https://arxiv.org/abs/2606.09516) |

The TeleFuser integration was ported from official SwiftVR commit
`5ca168cef6ca7200f135fdfea85e5e13d12c5b53`. The checkpoint is downloaded from
Hugging Face and is compatible with the released model files below.

## Installation

Install PyTorch for the target CUDA version first, then install TeleFuser in
editable mode:

```bash
pip install -e ".[dev]"
```

Download the checkpoint with the Hugging Face CLI:

```bash
pip install -U huggingface_hub
huggingface-cli download H-oliday/SwiftVR --local-dir /data/SwiftVR
```

For gated or private repositories, authenticate first with `huggingface-cli login`.
The directory passed to `--model_root` must contain:

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

The CLI also accepts optional acceleration controls:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /data/SwiftVR \
  --attn_impl TORCH_SDPA \
  --compile_dit \
  --quantization tf-kernel-fp8
```

For multi-GPU DiT execution, `--gpu_num` enables Ulysses sequence parallelism.
The DiT's 40 attention heads support SP degrees that divide 40, such as 2, 4,
5, or 8. ReAE stays on GPU 0 to preserve the causal encoder/decoder state:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /data/SwiftVR \
  --gpu_num 2
```

For multi-GPU stage execution, pass three stage devices. Latents between ReAE
encode, DiT, and ReAE decode are handed off through `WorkerTensorChannel`:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /data/SwiftVR \
  --gpu_num 3 \
  --enable_stage_parallel \
  --stage_devices 0,1,2
```

Ulysses SP and stage parallelism are mutually exclusive.
`torch.compile` is currently a single-GPU optimization and cannot be combined
with SwiftVR Ulysses SP.

The direct causal API accepts uint8 `[T,H,W,3]` tensors and returns PIL RGB
frames. A partial chunk can return an empty list until enough causal context is
available:

```python
import torch

from examples.swiftvr.swiftvr_restore_h100 import get_pipeline

pipeline = get_pipeline(model_root="/data/SwiftVR")
session = pipeline.stream(resolution=(1920, 1080), clip_len=24, dit_overlap=1)
try:
    first_frames = session.step(torch.zeros((24, 540, 960, 3), dtype=torch.uint8))
    tail_frames = session.step(torch.zeros((5, 540, 960, 3), dtype=torch.uint8))
    flushed_frames = session.flush()
finally:
    session.close()
```

## H100 performance

The official README reports the following single-H100 results for causal
streaming with 24 frames:

| Resolution | Official FPS | Official average time | Official peak memory |
| --- | ---: | ---: | ---: |
| 2560x1440 | 31.32 | 0.766 s | 38.01 GB |
| 3840x2160 | 14.00 | 1.714 s | Not reported |

At 4K, the official README reports that every compared diffusion-based VR
baseline OOMs on one H100 while SwiftVR sustains 14 FPS.

The released implementation measured 31.05 FPS in a local run under the same
checkpoint and resolution. The fair core comparison below uses BF16, eager
Torch SDPA, `dit_overlap=0`, and synchronized GPU timing that excludes PIL
conversion and device-to-host output transfer.

| Output resolution | Official local probe | TeleFuser default | Compile | FP8Linear | Compile + FP8Linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2560x1440 | 31.05 FPS | 32.42 FPS | 40.3-40.5 FPS | 35.8-36.1 FPS | 45.2-45.7 FPS |

The default TeleFuser path is faster than the released implementation.
`torch.compile` has a one-time shape compilation cost of about 44
seconds on the benchmark host, so enable it for long-lived workers. The example
defaults to `dit_overlap=0` for parity with the published offline
result; direct `stream()` keeps upstream's default
`dit_overlap=1`.

The `tf-kernel-fp8` option uses the same public
`telefuser.ops.fp8_gemm.FP8Linear` W8A8 path as LiveAct. It wraps 360
SwiftVR DiT Linear layers with per-token activation quantization and cached FP8
weights. In a QHD BF16-to-FP8 output check, it produced PSNR 52.24 dB, MAE
0.00146, and maximum absolute error 0.015625. Treat it as an opt-in speed and
quality tradeoff and validate on the target content.

SageAttention remains selectable through `--attn_impl`, but it is not
the recommended H100 default for this model: SAGE_ATTN_2_8_8_SM90 measured
30.8-31.1 FPS eager and 38.1-38.4 FPS with compile, below the corresponding
Torch SDPA paths.

The pipeline returns `List[PIL.Image.Image]`. Its delivered throughput includes
PIL conversion and device-to-host transfer, so it is lower and more sensitive
to host scheduling than the core GPU number. The existing delivered-path
measurements are:

| Output resolution | Delivered PIL FPS | TTFC | P50 / P95 chunk latency | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 47.28 | 15.01 s | 0.506 / 0.511 s | 27.08 GiB |
| 2560x1440 | 26.50 | 16.31 s | 0.895 / 0.960 s | 35.93 GiB |

The optional three-GPU stage-parallel path uses `WorkerTensorChannel` for latent
handoff. A 240-input-frame steady run produced 237 output frames at 39.87-40.11
FPS with PIL conversion removed to isolate stage throughput. This is a pipeline
throughput result and is not directly comparable to single-GPU memory or
latency.

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

`SwiftVRPipeline.stream()` is the supported stateful interface. In stage-parallel
mode, one active stream session is supported per pipeline because the worker
stages retain causal ReAE/DiT state. The example does not expose `get_service()`
because the current LiveKit protocol has no video-input transport; presenting a
partial service adapter would not make the model usable through
`telefuser stream-serve`.

See [the SwiftVR integration guide](../../docs/en/swiftvr.md) for checkpoint
loading, constraints, and measured results.
