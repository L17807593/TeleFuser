# LTX-2.5 Distilled Examples

Single-H100 text-to-video (T2V) and image-to-video (I2V) generation with the distilled LTX-2.5 pipeline. The example
produces an MP4 containing generated video and synchronized 48 kHz stereo audio.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LTX-2.5 22B distilled model pack | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | N/A | Transformer, text encoder, video/audio VAEs, spatial upsampler, and duration head |

This example does not auto-download weights. Download the official repository while preserving its directory layout:

```bash
hf download Lightricks/LTX-2.5 --local-dir /path/to/LTX-2.5
```

The example requires the exact split LTX-2.5 checkpoint layout shown below; a consolidated checkpoint or Diffusers
directory is not accepted.

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-video | Supported | Generates video and audio from a text prompt |
| Image-to-video | Supported | Accepts one still image at a non-negative output-frame index |
| Multi-GPU inference | Unsupported | `get_pipeline()` rejects `parallelism != 1` |
| Video VAE | Supported | DiffVAE is the default; ConvVAE is selectable with `--video-vae conv` |
| CPU offload | Supported | `cpu` streams transformer blocks and releases modules between phases; `none` retains modules on the GPU |
| LoRA | Unsupported | The example does not expose a LoRA loader |
| Quantization | Unsupported | The example loads BF16 checkpoints |
| Feature cache | Unsupported | The example does not configure feature caching |
| Server API | Partial | Legacy `get_pipeline()` and `run_with_file()` entry points exist, but no explicit pipeline contract is declared |

## Requirements

- GPU: one NVIDIA H100; other GPU targets are not validated by this example
- Software: the standard TeleFuser environment and an `ffmpeg` executable on `PATH`
- DiffVAE: a matching NATTEN/libnatten build is required for the formal 1536x1024, 121-frame workload
- I2V input: a PIL-readable still image such as PNG or JPEG

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). For the formal
DiffVAE path, select the command matching the installed PyTorch and CUDA versions from the
[NATTEN installation guide](https://natten.org/install/), then verify that its CUDA kernel library is available:

```bash
python -c "import natten; print(natten.HAS_LIBNATTEN)"
```

The command must print `True`. Without NATTEN, the DiffVAE decoder uses the Triton/eager compatibility fallback,
which is not the formal performance or accuracy baseline.

## Model Directory

`--model-root` must point to the root of this exact split-checkpoint layout:

```text
/path/to/LTX-2.5/
|-- diffusion_models/
|   \-- ltx-2.5-22b-distilled-transformer-bf16.safetensors
|-- text_encoders/
|   \-- gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
|-- vae/
|   |-- ltx-2.5-video-vae-bf16.safetensors
|   |-- ltx-2.5-video-vae-conv-bf16.safetensors
|   \-- ltx-2.5-audio-vae-bf16.safetensors
|-- latent_upscale_models/
|   \-- ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
\-- model_patches/
    \-- ltx-2.5-duration-head-bf16.safetensors
```

The current checkpoint resolver validates all seven files at pipeline construction time, including both video VAE
checkpoints regardless of the `--video-vae` selection.

## Quick Start

Run the default T2V workload from the repository root:

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_i2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --prompt "A cinematic camera orbit around the subject." \
  --output-path work_dirs/ltx25-t2v.mp4
```

The command writes a 1536x1024, 121-frame, 24 FPS video with synchronized audio to
`work_dirs/ltx25-t2v.mp4`.

## Examples

### `ltx25_distilled_t2v_i2v_h100.py`

This is the single local entry point for both T2V and I2V inference.

#### Text-to-Video

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_i2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --prompt "Ocean waves roll beneath a cloudy sky as distant thunder echoes." \
  --output-path work_dirs/ltx25-t2v.mp4
```

#### Image-to-Video

Pass `--image-path` to enable image conditioning. This repository includes a small test asset for reproducing the
input contract:

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_i2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --image-path tests/assets/ltx25/official_guitar_man.png \
  --image-frame-index 0 \
  --image-strength 1.0 \
  --prompt "A man with short gray hair plays a red electric guitar." \
  --output-path work_dirs/ltx25-i2v.mp4
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--prompt` | Required | Text prompt used for video and audio generation |
| `--model-root` | Deployment-specific | Root of the required split model pack; pass it explicitly |
| `--output-path` | Required | Destination MP4; parent directories are created automatically |
| `--image-path` | None | Enables I2V with the supplied still image |
| `--image-frame-index` | `0` | Non-negative output-frame index for the image condition |
| `--image-strength` | `1.0` | Image-conditioning strength in the inclusive range `[0, 1]` |
| `--height` | `1024` | Output height; must be a positive multiple of 64 |
| `--width` | `1536` | Output width; must be a positive multiple of 64 |
| `--num-frames` | `121` | Output frame count; must satisfy `num_frames = 8k + 1` |
| `--frame-rate` | `24.0` | Positive output frame rate in FPS |
| `--seed` | `42` | Random seed |
| `--video-vae` | `diff` | Video decoder: `diff` or `conv` |
| `--offload` | `cpu` | Model residency policy: `cpu` or `none` |

Key behavior:

- The distilled pipeline runs fixed two-stage sampling and jointly generates video and audio.
- Output audio is written as stereo PCM at 48 kHz and muxed into the MP4 as AAC.
- The example supports exactly one GPU and one optional still-image condition.

## Configuration

### Video VAE

`--video-vae diff` selects DiffVAE and is the formal output path. It uses NATTEN when the compatible CUDA extension
is installed and otherwise falls back to the Triton/eager implementation. `--video-vae conv` selects ConvVAE as a
compatibility alternative.

### Model Residency

`--offload cpu` is the default. It streams transformer blocks between CPU and GPU, releases other modules at phase
boundaries, and lowers peak GPU residency at the cost of transfers. Use `--offload none` only when the GPU has enough
memory to retain modules between phases.

### Request Constraints

- `height` and `width` must be positive multiples of 64.
- `num_frames` must satisfy `num_frames = 8k + 1`; examples include 1, 9, 17, and 121.
- `frame_rate` must be positive.
- `image_frame_index` must be non-negative and `image_strength` must be in `[0, 1]`.

## Troubleshooting

### Missing Checkpoint

The pipeline reports the first missing component and its resolved path. Compare `--model-root` with the complete
layout above; both video VAE files are currently required even when only one decoder is selected.

### Output Has No Audio

TeleFuser keeps the generated video if audio muxing fails. Confirm that `ffmpeg` is installed and available on
`PATH`:

```bash
ffmpeg -version
```

### DiffVAE Uses the Compatibility Fallback

Confirm that NATTEN is importable and includes libnatten for the active PyTorch/CUDA environment:

```bash
python -c "import torch, natten; print(torch.__version__, torch.version.cuda, natten.HAS_LIBNATTEN)"
```

Use the [NATTEN installation matrix](https://natten.org/install/) to select a matching build when the final value is
`False`.

## Notes

- The formal workload is 1536x1024, 121 frames, 24 FPS, DiffVAE, and NATTEN on one H100.
- Lower resolutions and shorter valid frame counts are useful for smoke tests but are not the formal quality baseline.
