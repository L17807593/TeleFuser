# MiniMax H3

These examples run the local MiniMax H3 Base release from its original FL2VA and Ref2VA partitions. They generate
24 FPS video with synchronized 32 kHz stereo audio. The local path supports 768p-class output; hosted Context-IR and
Regenerate-2K services are not implemented or implied.

## Requirements

- Linux with ffmpeg and ffprobe.
- One NVIDIA H100 80GB for the documented sequential-residency path.
- Enough host memory for the approximately 63 GB encoder and 62 GB DiT partitions.
- The repository development environment and the unmodified model directory at
  `/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3`.

Run commands from the repository root. The examples load original checkpoint shards through `ModuleManager`.
Encoder, DiT, visual VAE, and audio VAE use model-level CPU offload, so only one large component is resident on the
selected GPU at a time. The encoder and DiT use BF16; both VAEs remain FP32, with reference FP16 autocast applied
only to CUDA video decode.

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

## Generation And Parallel Options

The simple CLIs expose `--steps`, `--seed`, `--duration`, `--aspect-ratio`, `--flow-shift`, and
`--audio-flow-shift`. Supported explicit aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, and `9:16`;
`auto` follows the task policy or first FL2VA keyframe.

Use `--ulysses-degree 2` or `--ulysses-degree 4` to shard packed DiT attention. The encoder and VAEs remain
sequential on `--device` (normally `cuda:0`). The degree must divide 56 attention heads, and scripts must run from
their guarded entry points so worker processes can spawn safely.

FSDP can be combined with a multi-GPU Ulysses example:

```bash
python examples/minimax_h3/minimax_h3_request_h100.py \
  --ulysses-degree 2 \
  --enable-fsdp \
  --output outputs/minimax_h3_ref2va_fsdp.mp4
```

H3 tensor parallelism, Ring attention, CFG parallelism, pipeline parallelism, and visual-VAE spatial parallelism are
not enabled. The current service request schema also cannot preserve ordered heterogeneous Ref2VA materials; use
these local Python examples until a shared ordered-material service contract is separately approved.
