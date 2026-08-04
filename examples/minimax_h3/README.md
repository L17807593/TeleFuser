# MiniMax H3

These examples run the local MiniMax H3 Base release from its original FL2VA and Ref2VA partitions. They generate
24 FPS video together with 32 kHz stereo audio. The local path supports 768p-class output; hosted Context-IR and
Regenerate-2K services are not implemented or implied.

## Requirements

- Linux with ffmpeg and ffprobe.
- One NVIDIA H100 80GB for the documented sequential-residency path.
- Enough host memory for the original encoder and DiT checkpoint partitions.
- The repository development environment and the unmodified model directory at
  /hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3.

The examples strictly load original sharded checkpoint files through ModuleManager. Encoder, DiT, visual VAE, and
audio VAE use model-level CPU offload, so only one large component is resident on the selected GPU at a time. The
encoder and DiT use BF16; both VAEs are loaded in FP32, with reference FP16 autocast applied only to CUDA video decode.

## FL2VA

    python examples/minimax_h3/minimax_h3_fl2va_h100.py \
      --image /path/to/first.png \
      --prompt "A person turns toward the camera and speaks." \
      --duration 8 \
      --output outputs/minimax_h3_fl2va.mp4

Add --last-image /path/to/last.png for ordered first-and-last-frame mode. A single image may also use the last-frame
signature by calling the pipeline API with frame_index=-1.

Omit --image for text-only T2VA:

    python examples/minimax_h3/minimax_h3_fl2va_h100.py \
      --prompt "A cinematic coastal landscape with synchronized ambient sound." \
      --duration 5 \
      --output outputs/minimax_h3_t2va.mp4

Add `--ulysses-degree 2` or `--ulysses-degree 4` to execute the DiT on that many logical GPUs. The encoder and VAEs
remain sequential on `--device` (normally `cuda:0`), while the existing TeleFuser worker runtime launches one DiT
replica per device and shards the packed attention sequence. Always run the example from its guarded script entry
point so worker processes can spawn safely.

## Ref2VA

    python examples/minimax_h3/minimax_h3_ref2va_h100.py \
      --image /path/to/subject.png \
      --video /path/to/motion.mp4 \
      --prompt "Keep the subject identity and follow the reference motion." \
      --duration 5 \
      --output outputs/minimax_h3_ref2va.mp4

Ref2VA preserves the order supplied through the pipeline API. The CLI groups repeated --image, --video, and --audio
options by modality; use the Python API when heterogeneous interleaving is significant. Published limits are
enforced before model execution: at most 9 images, 3 videos, 3 audio-bearing inputs, and 12 files total; each
audio/video clip is 2-15 seconds with at most 15 seconds per modality. Audio requires an image or video reference.
