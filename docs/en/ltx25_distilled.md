# LTX-2.5 Distilled

The LTX-2.5 distilled work is isolated under `telefuser/models/ltx25` and
`telefuser/pipelines/ltx25_distilled`. It does not reuse the legacy LTX-2.3
model or pipeline modules.

## Runtime Baseline

The packed Gemma4 Unified checkpoint requires `transformers==5.14.1`. The
reference and TeleFuser runs must use the same PyTorch, CUDA, cuDNN, NATTEN,
and attention backend. The formal DiffVAE benchmark requires NATTEN; the
Triton neighborhood-attention fallback is a separately labelled compatibility
baseline.

The official split BF16 model pack contains the transformer, Gemma4, both
video VAE variants, audio VAE, spatial upsampler, and duration head. Use
`tools/validation/inspect_ltx25_checkpoints.py` to validate that layout
without loading its tensors.

## Public Pipeline

`LTX25DistilledPipeline.from_model_root()` exposes the isolated two-stage
T2V/I2V path. It accepts a prompt, seed, resolution, `8k + 1` frame count,
frame rate, and optional `LTX25ImageCondition` values. The pipeline moves
Gemma, the transformer, the upsampler, the video decoder, and the audio
decoder/vocoder through GPU memory sequentially, so these checkpoint groups
are not resident together. It returns decoded video chunks, decoded stereo
audio, and final video/audio VAE latents in `LTX25DistilledOutput`.

The default `video_vae="diff"` uses the NADiffusionVAE checkpoint. Pass
`video_vae="conv"` to select the official ConvVAE checkpoint; the selected
VAE is used consistently for image conditioning, latent-statistics
normalization around spatial upsampling, and video decoding. Both variants
return RGB video chunks in `[F, H, W, C]` with values in `[0, 1]`. Image
conditions are applied to the clean latent state before initial noising, which
preserves the official denoise-mask behavior.

## Example

Run `examples/ltx25_distilled/ltx25_distilled_t2v_i2v_h100.py` for the
single-GPU public T2V/I2V entry point. It writes an H.264 MP4 with the generated
stereo 48 kHz audio track and exposes the standard `get_pipeline`, `run`, and
`run_with_file` functions used by TeleFuser's file-output service contract.

## Golden Capture


## Frozen I2V Input

`tests/assets/ltx25/official_guitar_man.png` is the 928x512 RGB input directly
referenced by the official Lightricks LTX-2 model-card I2V example. Its SHA-256
is `e31cbbe4822ce07e1548121b436c0db3a067d1d78f2e75ab3e69375377b57274`.
The frozen request uses the matching prompt, `A man with short gray hair plays
a red electric guitar.`, at frame index 0 and strength 1.0. The test asset's
README records the direct source URL. The capture manifest records the input
hash and request fields, so the I2V precision/performance baselines do not rely
on a mutable network resource.

The formal single-GPU baseline follows the upstream default `CHUNKED_EAGER` DiffVAE recipe: deferred stage-4 context, four width chunks, and the `cutlass-fna` NATTEN backend. Its standard upstream tiling uses 128-frame tiles with 40-frame overlap and 1024/1536-pixel tiles with 160-pixel overlap for the formal T2V request. `CHUNKED_COMPILE` remains an experimental diagnostic mode and is not the formal baseline.

Capture the pinned upstream reference before evaluating a TeleFuser candidate:

```bash
PYTHONPATH=.:.venv/lib/python3.11/site-packages:work_dirs/LTX-2/packages/ltx-core/src:work_dirs/LTX-2/packages/ltx-pipelines/src \
  work_dirs/LTX-2/.venv-ltx/bin/python tools/validation/capture_ltx25_upstream.py \
  --model-root /path/to/LTX-2.5 \
  --upstream-root work_dirs/LTX-2 \
  --output-dir /tmp/ltx25-golden \
  --prompt "A small robot walks through a sunlit workshop." \
  --seed 42 --width 384 --height 256 --num-frames 9 --frame-rate 24 --offload cpu \
  --deterministic-audio
```

Use `--video-vae conv` for ConvVAE, or add `--image PATH --image-frame-index
0 --image-strength 1.0` for I2V. The manifest records checkpoint hashes,
runtime provenance, request inputs, image hash, stage-boundary tensors, and
per-step x0/noise/updated latents. It also records both raw decoded RGB and
audio waveform tensors. The upstream capture additionally writes `decoded.mp4`
through its native H.264/AAC muxer and records the container and video/audio
stream metadata, file size, and SHA-256 in the manifest's `container` section.

Three fresh upstream CPU-offload T2V DiffVAE fallback runs at 384x256 / 9
frames were captured with the same request. Prompt artifacts, Stage-1/2 x0 and
latent tensors, and decoded RGB were bit-identical across all three runs. The
default CUDA audio-vocoder convolution choice can vary across replays; the
largest observed pairwise difference was `max abs 0.013458251953125`, with
`NRMSE 0.14420734345912933`. Pass `--deterministic-audio` to both capture
tools for an exact waveform gate. This sets deterministic CUDA kernels only
around audio decoding and restores the prior process settings afterwards; it
does not change TeleFuser production inference behavior.

These small fallback captures are diagnostic artifacts only. They are not a
substitute for the 1536x1024 / 121-frame NATTEN formal accuracy and performance
gates.

## Component Validation

The isolated audio checkpoint loader validates all 58 decoder and 1,227
vocoder/BWE tensors before loading. With the saved Stage-2 small T2V latent,
the decoder mel tensor is exact. Under the capture-only deterministic audio
mode, the final waveform is also exact against upstream.

The isolated Conv VAE loader maps the checkpoint's one shared latent-statistics
pair to the encoder and decoder consumers, with no unexplained model keys. Its
raw decoder output is bit-identical to the upstream Conv decoder for the same
weights and saved I2V latent. The RGB Golden capture uses channel-last,
postprocessed `[0, 1]` values; candidate decoder output remains channel-first
and normalized to `[-1, 1]` until pipeline output conversion.

The isolated DiffVAE decoder applies the upstream-default `CHUNKED_EAGER` recipe: deferred stage-4 context, four width chunks, and `cutlass-fna` NATTEN attention. Its FP32 RoPE frequency buffers are retained across BF16 weight loading. `CHUNKED_COMPILE` is retained only as an experimental helper and does not select the default loader path. The isolated video encoder is bit-identical against upstream for the same BF16 video tensor and weights.

## Current Gate Status

The formal 1536x1024 / 121-frame T2V capture and the frozen 896x512 / 121-frame I2V capture both use the upstream eager tiling and pass the request, checkpoint, exact RNG, required artifact, decoded RGB, and decoded-audio gates. The T2V decoded RGB comparison recorded 61.90 dB PSNR and 0.999685 SSIM; the I2V comparison recorded 62.95 dB PSNR and 0.999671 SSIM.

The matched five-sample CPU-offload T2V performance comparison is not an accepted performance result: TeleFuser measured 87.60 s cold p50 versus 76.78 s upstream (+14.10%) and 86.65 s warm p50 versus 77.29 s upstream (+12.11%). Request and runtime identity checks passed; further optimization is required before claiming parity.

Upstream capture additionally records raw Gemma input and hidden-state diagnostics. These remain visible as golden-only diagnostics in comparison reports, while missing TeleFuser interface artifacts or unexpected candidate artifacts remain failures. Large BF16 comparisons calculate cosine and NRMSE with float64 reductions to avoid false failures from float32 accumulation.

Formal DiffVAE requires a NATTEN build with the CUDA `libnatten` extension; importing a Python-only NATTEN package is not sufficient. Verify the capability before a formal run:

```bash
python -c "import natten; print(natten.HAS_LIBNATTEN)"
```

For the pinned PyTorch 2.11.0 / CUDA 12.8 validation runtime, the matching wheel is installed with:

```bash
pip install "natten==0.21.6+torch2110cu128" -f https://whl.natten.org
```

## Performance Capture

After the formal accuracy gate passes, collect synchronized raw cold and warm samples separately for each offload mode. `--runs 5` records five independent cold construction-plus-generation samples and five warm samples, so both gates have a p50 rather than a single cold observation. Run the upstream and TeleFuser harnesses with the same model pack, prompt, seed, dimensions, NATTEN runtime, and `--offload` mode. Both benchmark tools measure generation inside `torch.inference_mode()` and write every synchronized timing and peak allocator fact to their JSON report.

Apply the formal decision only through the comparison gate:

```bash
python tools/validation/compare_ltx25_benchmarks.py \
  /tmp/ltx25-upstream-cpu.json /tmp/ltx25-telefuser-cpu.json \
  --output /tmp/ltx25-performance-cpu.json
```

It rejects mismatched request/runtime identity and fewer than five samples. A result within the 2% noise band is intentionally inconclusive, so collect more matched samples instead of reporting a performance pass. Repeat the full workflow for `offload=none` and the frozen I2V input.
