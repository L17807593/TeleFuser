# MiniMax H3

TeleFuser supports local MiniMax H3-Base generation from the original FL2VA and Ref2VA checkpoint partitions.
The pipeline produces 24 FPS video and synchronized 32 kHz stereo audio. It does not implement or imply the hosted
H3-Context-IR or H3-Regenerate-2K products.

## Supported Tasks

- T2VA through the FL2VA partition.
- First-frame, last-frame, and first-and-last-frame FL2VA conditioning.
- Ordered image, video, audio, and mixed Ref2VA conditioning.
- Dense packed attention through the public TeleFuser attention operation.
- Ulysses sequence parallelism with degrees 2 and 4 through the existing `ParallelConfig`, plus optional FSDP through
  the same runtime configuration.

Ref2VA applies the published input limits before model execution: at most nine images, three videos, three
audio-bearing inputs, and twelve files total. Each audio or video clip must be 2-15 seconds, total video duration and
total audio duration must each be at most 15 seconds, and audio requires an image or video reference.

## Checkpoint Layout

The examples expect this unmodified release layout:

    /hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3/
      FL2VA/
        transformer/model-*.safetensors
        text_encoder/model-*.safetensors
        video_vae/source/model.safetensors
        audio_vae/model.safetensors
        processor/
      Ref2VA/
        transformer/model-*.safetensors
        text_encoder/model-*.safetensors
        video_vae/source/model.safetensors
        audio_vae/model.safetensors
        processor/

All components load through ModuleManager with explicit H3 model classes and strict state-dict conversion. The
visual and audio VAEs remain FP32. The encoder and DiT use BF16 while preserving the reference FP32 DiT parameter,
timestep, scheduler, and RoPE boundaries. Visual VAE decode uses the reference FP16 CUDA autocast boundary.

Validate one component at a time to keep host-memory use bounded:

    python tools/validation/validate_minimax_h3_loading.py \
      --partition FL2VA --component transformer

## Examples

Run the examples from the repository root. The documented sequential-residency configuration needs one H100 80 GB
and enough host memory for the approximately 63 GB encoder and 62 GB DiT.

    python examples/minimax_h3/minimax_h3_fl2va_h100.py \
      --image /path/to/first.png \
      --prompt "A person turns toward the camera and speaks." \
      --duration 8 \
      --output outputs/minimax_h3_fl2va.mp4

Omit --image from the FL2VA command to run T2VA from the same original partition.

Use `--ulysses-degree 2` or `--ulysses-degree 4` to run the packed DiT through a TeleFuser `ParallelWorker`. The
encoder and VAEs retain sequential residency on `cuda:0`; the DiT is replicated across the selected logical devices
and sequence-sharded during attention. The degree must divide 56 attention heads. The example loader sets the
existing worker timeout to 1800 seconds because a release-length Ref2VA request can exceed the generic 600-second
`ParallelConfig` default.

    python examples/minimax_h3/minimax_h3_ref2va_h100.py \
      --image /path/to/subject.png \
      --video /path/to/motion.mp4 \
      --prompt "Keep the subject identity and follow the reference motion." \
      --duration 5 \
      --output outputs/minimax_h3_ref2va.mp4

See the MiniMax example README for CLI ordering details and the Python request contract.

## Service Compatibility Gate

The current TaskRequest schema cannot represent Ref2VA without information loss. It exposes fixed first-image,
last-image, reference-video, and audio paths, while H3 needs one ordered heterogeneous list with repeated modalities,
per-item roles, frame indices, and trim metadata. Flattening that list would lose ordering and video/audio
association.

No shared service field is added by this integration. The alternatives are a model-specific opaque extra field,
which weakens validation and discovery, or an approved shared ordered-material schema used by TaskRequest,
MediaGenerationService download handling, pipeline contracts, OpenAPI clients, caching, and service tests. The latter
is the recommended future change, but it requires separate approval. Proposed tests must cover JSON round trips,
heterogeneous order, repeated modalities, per-item metadata, URL and base64 localization, cache-key stability,
OpenAPI compatibility, and rejection before model execution.

Until that decision is approved, use the local Python API and examples. This limitation affects service request
submission only; it does not change the pipeline-owned request contract.

## Low-Step Smoke Validation

The frozen official requests, localized CDN bytes, provenance, manifests, raw arrays, and reports live in the ignored
`work_dirs/minimax_h3_parity` workspace. The smoke checks use 768p, 4 seconds, seed 0, and two requested inference
steps, which execute only one denoising update. Raw frames must reach cosine similarity 0.99 and PSNR 28 dB; raw
stereo waveforms must reach cosine similarity 0.94 and PSNR 30 dB. These smoke thresholds catch data-plumbing,
shape, and gross numerical regressions over that single update:

| Case | SGLang topology | Video cosine / PSNR | Audio cosine / PSNR |
|---|---|---:|---:|
| T2VA | single GPU, Torch SDPA | 0.99328 / 32.05 dB | 0.94545 / 33.29 dB |
| FL2VA | single GPU, Torch SDPA | 0.99560 / 30.37 dB | 0.97911 / 62.67 dB |
| Ref2VA | TP2 + Ulysses2, Torch SDPA | 0.99899 / 32.70 dB | 0.98160 / 31.44 dB |

These aggregate smoke metrics are not evidence of visual quality, prompt fidelity, conditioning fidelity, or parity
over a release-length diffusion trajectory. Production acceptance uses the official task duration and input media,
seed 0, the release default of 50 inference steps, intermediate boundary captures, per-frame and audio comparisons,
and explicit visual review.

SGLang Ref2VA single-GPU smoke inference and decode completed, but its controller could not allocate the additional
1.23 GiB needed to materialize 107 output frames while the model worker remained resident. The four-GPU result is
used as the reproducible raw Ref2VA smoke baseline; this is an SGLang output-lifecycle constraint, not a TeleFuser
model failure.

Every schema-v2 runner manifest records both the source and effective request hashes, all partition JSON config
hashes, dependency and GPU facts, parallel settings, and artifact checksums.

The original-checkpoint TeleFuser T2VA smoke check also ran with Ulysses2. Its final frame and waveform arrays are
bit-identical to the TeleFuser single-GPU smoke result. The complete request took 115.87 seconds; rank 0 reported 5.97
seconds in DiT forward, 0.59 seconds in Ulysses collectives, 70.10 GB peak allocated memory, and 72.52 GB peak
reserved memory. These measurements establish distributed consistency for this smoke request, not output quality,
and are not general performance guarantees.

## Reference And Differences

The implementation baseline is the clean local SGLang commit
e9366d7f79e0f45c4ff0b9247cebbedc0e2be8a0 and the original FL2VA/Ref2VA partitions. Intentional differences are:

- TeleFuser enforces the published Ref2VA count and duration limits even where the pinned SGLang admission path does
  not reject all of them.
- TeleFuser uses the Hugging Face Qwen3-VL encoder wrapper and unnormalized layer-50 hidden state; pinned SGLang uses
  its native foldable encoder. The computation contract is the same, but kernels and parallel reduction order differ.
- H3 packed attention currently uses Torch SDPA through telefuser.ops; other attention backends are rejected for
  packed sequence lengths.
- H3 DiT tensor parallelism is not enabled because its grouped QKV layout is not safely expressible through the
  existing sharding contract. Ring attention, CFG parallelism, pipeline parallelism, sparse attention, compile
  optimization, and visual-VAE spatial parallelism are also not enabled. Unsupported degrees fail explicitly.
- TeleFuser uses stage-level model CPU offload. It does not reproduce SGLang's validation-only layerwise DiT offload.
- The service schema remains unchanged because it cannot preserve ordered heterogeneous materials.

The ignored work directory work_dirs/minimax_h3_parity records the frozen reference commit, model hashes, requests,
fixtures, tensors, environment manifests, and comparison reports used for full-size validation.
