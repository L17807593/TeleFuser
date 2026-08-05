# MiniMax H3

These examples run the local MiniMax H3 Base release from its original FL2VA and Ref2VA partitions. They generate
24 FPS video with synchronized 32 kHz stereo audio. The local path supports 768p-class output; hosted Context-IR and
Regenerate-2K services are not implemented or implied.

## Requirements

- Linux with ffmpeg and ffprobe.
- One NVIDIA H100 80GB for sequential stage offload, or two/four H100 80GB GPUs for resident multi-GPU execution.
- Enough host memory for the approximately 63 GB encoder and 62 GB DiT partitions.
- The repository development environment and the unmodified model directory at
  `/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3`.

Run commands from the repository root. The examples load original checkpoint shards through `ModuleManager`. A
one-GPU run uses stage-level model CPU offload. Multi-GPU runs keep the stages resident: two GPUs use Ulysses2 for
the DiT, TP2 for the text encoder, and TP2 video-VAE tiling; four GPUs use DiT Ulysses2 x TP2, text TP4, and TP4
video-VAE tiling. The audio VAE remains on GPU 0. The encoder and DiT use BF16; both VAEs remain FP32, with the
reference FP16 autocast boundary applied only to CUDA video decode.

The source-controlled default inputs live in `examples/data/minimax-h3/`. They are the exact inputs frozen for the
official SGLang parity runs; `provenance.json` records their original URLs, byte sizes, and SHA-256 hashes.

## T2VA And FL2VA

Use the explicit mode names when demonstrating a particular task. T2VA has no reference input:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode t2va \
  --prompt "A cinematic coastal landscape with synchronized ambient sound." \
  --duration 5 \
  --output outputs/minimax_h3_t2va.mp4
```

First-frame FL2VA uses the bundled reference image when `--image` is omitted:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode first-frame \
  --prompt "Steam rises from the ramen while the family talks in the background." \
  --duration 8 \
  --output outputs/minimax_h3_first_frame.mp4
```

Last-frame-only FL2VA accepts either `--last-image` or the bundled image:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode last-frame \
  --prompt "The camera settles on a warm family dinner at the final frame." \
  --output outputs/minimax_h3_last_frame.mp4
```

First-and-last-frame FL2VA accepts two images. When both are omitted, the bundled image is used at both endpoints;
that default is useful for a contract smoke run, while meaningful motion requires distinct endpoint images.

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode first-last \
  --image /path/to/first.png \
  --last-image /path/to/last.png \
  --prompt "Move smoothly from the first composition to the last composition." \
  --output outputs/minimax_h3_first_last.mp4
```

For compatibility, omitting `--mode` infers T2VA, first-frame, last-frame, or first-last from `--image` and
`--last-image`. Explicit modes are preferable in reproducible commands.

## Ref2VA

With no material arguments, the simple Ref2VA script uses the bundled reference video followed by the bundled voice
reference:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --prompt "Preserve the source identity and motion, and use the reference voice for the dialogue." \
  --duration 5 \
  --output outputs/minimax_h3_ref2va.mp4
```

Custom material paths may be repeated:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --image /path/to/subject.png \
  --video /path/to/motion.mp4 \
  --audio /path/to/voice.wav \
  --prompt "Keep the subject identity and follow the reference motion." \
  --duration 5 \
  --output outputs/minimax_h3_ref2va_custom.mp4
```

The convenience CLI groups repeated arguments as images, videos, then audio. Use the JSON request runner whenever
heterogeneous ordering is semantic. It defaults to `examples/data/minimax-h3/ref2va.json`; relative material URIs
are resolved from the request file's directory.

```bash
python examples/minimax_h3/minimax_h3_request_h100.py \
  --request examples/data/minimax-h3/ref2va.json \
  --output outputs/minimax_h3_ordered_request.mp4
```

The JSON `conditions` array is passed in its original order. Each entry accepts `type` (`image`, `video`,
`video_audio`, or `audio`), `role: "reference"`, `uri`, and optional `start_time_seconds` for video inputs.
`video_audio` requires both tracks; `video` uses its original soundtrack when one is present. For example:

```json
{
  "task": "ref2va",
  "prompt": "Use <Image 1>, then <Audio 1>, then the motion and soundtrack from <Video 1>.",
  "conditions": [
    {"type": "image", "role": "reference", "uri": "subject.png"},
    {"type": "audio", "role": "reference", "uri": "voice.wav"},
    {
      "type": "video_audio",
      "role": "reference",
      "uri": "motion.mp4",
      "start_time_seconds": 1.5
    }
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
  "seed": 0,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "num_inference_steps": 50
}
```

Ref2VA may omit `target.duration_seconds` when exactly one audio-bearing condition supplies the duration. With
multiple audio-bearing conditions, duration must be explicit. Published limits are enforced before model execution:
at most 9 images, 3 videos, 3 audio-bearing inputs, and 12 files total; each audio/video clip must be 2-15 seconds,
total video and total audio duration must each be at most 15 seconds, and audio requires an image or video reference.

## Standard Python And Serve Entrypoints

All three executable modules expose `PPL_CONFIG`, `get_pipeline`, `run`, and `run_with_file`. The two fixed-partition
generation modules also expose `PIPELINE_MANIFEST` for serving. `get_pipeline(parallelism, model_root)` interprets
`parallelism` as the total GPU count and selects the corresponding profile below. `run` returns the in-memory
`MiniMaxH3Generation`; `run_with_file` writes the synchronized MP4 and returns its `output_path`.

```python
from examples.minimax_h3.minimax_h3_fl2va_h100 import get_pipeline, run_with_file

pipeline = get_pipeline(4, "/path/to/MiniMaxAI_MiniMax-H3")
try:
    artifact = run_with_file(
        pipeline,
        task="i2v",
        prompt="Steam rises from the ramen while the family talks.",
        first_image_path="examples/data/minimax-h3/fl2va-reference.png",
        output_path="outputs/minimax_h3_i2v.mp4",
    )
finally:
    pipeline.stop()
```

Serve T2VA, first-frame I2VA, and first-and-last-frame FL2VA from the shared FL2VA checkpoint partition:

```bash
telefuser serve examples/minimax_h3/minimax_h3_fl2va_h100.py --gpu-num 4 --port 8000
```

The service contract advertises `t2v`, `i2v`, and `fl2v`. Submit `first_image_path` for `i2v`, and both
`first_image_path` and `last_image_path` for `fl2v`. Output duration must be between 4 and 15 seconds:

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks/create \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"i2v\",\"prompt\":\"Animate this dinner scene with synchronized dialogue.\",\"first_image_path\":\"https://example.com/first.png\",\"resolution\":\"768p\",\"aspect_ratio\":\"16:9\",\"target_video_length\":5}"
```

Ref2VA uses its own checkpoint partition and advertises the standard `s2v` service task, which maps to the
model-specific Ref2VA task. Its required `conditions` parameter is the same ordered array accepted by the local
pipeline, so heterogeneous image, video, `video_audio`, and audio references are not reordered by the example.
Output duration must also be between 4 and 15 seconds:

```bash
telefuser serve examples/minimax_h3/minimax_h3_ref2va_h100.py --task s2v --gpu-num 4 --port 8001

curl -X POST http://127.0.0.1:8001/v1/tasks/create \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"s2v\",\"prompt\":\"Use <Video 1>, then <Audio 2>.\",\"conditions\":[{\"type\":\"video\",\"role\":\"reference\",\"uri\":\"https://example.com/motion.mp4\"},{\"type\":\"audio\",\"role\":\"reference\",\"uri\":\"https://example.com/voice.mp3\"}],\"resolution\":\"768p\",\"aspect_ratio\":\"16:9\",\"target_video_length\":5}"
```

Use `/v1/service/metadata` to inspect the active task contract. The JSON request runner remains the convenient local
entrypoint for request files and resolves relative material paths beside the JSON file.

## Generation And Parallel Options

The simple CLIs expose `--steps`, `--seed`, `--duration`, `--aspect-ratio`, `--flow-shift`, and
`--audio-flow-shift`. Supported explicit aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, and `9:16`;
`auto` follows the task policy or first FL2VA keyframe.

`--gpu-num` selects the total worker count. `--ulysses-degree` remains a compatibility alias for the same CLI option;
it no longer means that every selected GPU is necessarily an Ulysses rank.

| `--gpu-num` | DiT | Text encoder | Video VAE | Residency |
|---:|---|---|---|---|
| 1 | single GPU | single GPU | single GPU | sequential model CPU offload |
| 2 | Ulysses2 | TP2 | TP2 tiling | resident |
| 4 | Ulysses2 x TP2 | TP4 | TP4 tiling | resident |

The Ulysses degree must divide 56 attention heads. Scripts must run from their guarded entry points so worker
processes can spawn safely. H100 examples request packed FlashAttention 4 and fall back to packed PyTorch SDPA when
FlashAttention 4 is unavailable.

For multi-GPU resident profiles, `WorkerTensorChannel` transports text conditioning, visual condition rows, and the
final video latent directly between worker groups. CUDA intermediates therefore do not stage through the parent
process or CPU. The pipeline reports media, text, condition VAE, denoising, video/audio decode, allocator peak, and
DiT communication timings in `MiniMaxH3Generation.runtime_metrics`.

H3 also uses eager BF16 Triton paths for Q/K RMSNorm plus partial NeoX RoPE, indexed modulation, SwiGLU, and Ulysses
relayout when their input contracts match. Compatible `tf-kernel` builds may accelerate public RMSNorm, SwiGLU, and
RoPE operations. All public ops retain native PyTorch fallbacks for unsupported devices, dtypes, and compile mode.

FSDP2 remains available for an SP-only DiT profile and cannot be combined with DiT TP. The standard CLI can select
it explicitly for a two-GPU Ulysses run:

```bash
python examples/minimax_h3/minimax_h3_request_h100.py \
  --ulysses-degree 2 \
  --enable-fsdp \
  --output outputs/minimax_h3_ref2va_fsdp.mp4
```

The standard four-GPU profile already uses Ulysses2 x TP2 and therefore leaves FSDP disabled. Use
`load_minimax_h3_pipeline` directly to construct another supported combination; the product of Ulysses and TP degrees
must be 1, 2, or 4.

Ring attention, CFG parallelism, pipeline parallelism, sparse attention, quantization, and `torch.compile` are not
enabled for H3. Video-VAE parallelism is spatial tiling over the existing TP process group, not parameter tensor
parallelism. The dedicated service manifests expose the pipeline without adding framework-level configuration fields
or changing the shared request schema.

## Measured Four-GPU Profile

On the frozen 768p, five-second, 50-step T2VA request, after one warmup, the resident four-H100 profile measured
79.34 seconds wall time. The matched local SGLang SP2+TP2 run measured 79.37 seconds. Sampled peak GPU 0 memory was
62.7 GiB for TeleFuser and 67.8 GiB for SGLang. The TeleFuser DiT denoising phase took 76.41 seconds, including
4.12 seconds recorded in SP/TP communication.

These numbers establish parity for this request and environment, not a general performance guarantee. Direct stage
tensor transport removes CPU/parent staging but does not materially change the 50-step wall time because DiT compute
and per-layer TP/SP collectives dominate.
