# SwiftVR

TeleFuser integrates the released `H-oliday/SwiftVR` video-restoration model as
a faithful, sequential, single-GPU pipeline. The default path preserves the
upstream BF16 model, one-step DiT, mask-free shifted-window attention, causal
ReAE state, and fixed chunk/flush behavior. Dense attention is dispatched
through `telefuser.ops.attention`; Torch SDPA remains the default, while other
TeleFuser dense attention backends can be selected explicitly.

## Provenance and checkpoint

The reference source is SwiftVR commit
`5ca168cef6ca7200f135fdfea85e5e13d12c5b53`. The local checkpoint is the model
revision `743ed2530c550764905400f38eb6cc41af5abc80` under `/data/SwiftVR`.

The implementation loads ReAE and the Diffusers transformer through the
existing `ModuleManager`. It keeps checkpoint keys unchanged. Both models are
loaded on CPU and moved ReAE-first to the target GPU, matching upstream
allocation and cuDNN plan selection. The prompt embedding remains on CPU until
the DiT condition cache is created, also matching upstream behavior.

## Offline usage

Install the project dependencies and run:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /data/SwiftVR \
  --height 360 --width 640 \
  --scale 3 \
  --output restored_1080p.mp4
```

The default input is FlashVSR's versioned `examples/data/dag.mp4` test video.
The CLI follows FlashVSR's common options: `input_video`, `scale`, `height`,
`width`, `gpu_num`, `model_root`, and `output`. Dimensions are internally padded
to multiples of 32 and cropped back exactly. The output preserves `4k+1` source
frames because that is the released temporal contract. BF16 and one GPU are the
supported parity path. `get_pipeline()` owns model loading, `run()` accepts
loaded PIL frames, and TeleFuser's shared video utilities own file I/O. `run()`
uses the stateful 24-frame session path. A throwaway session warms two full
chunks plus the actual tail shape before measuring the real request, then the
example reports both generation FPS and end-to-end FPS including H.264
encoding.

## Streaming behavior

`SwiftVRPipeline.stream()` creates independent ReAE encoder/decoder boundary
state, DiT overlap state, RoPE position state, and condition cache. `step()`
accepts arbitrary uint8 frame counts, buffers non-four-aligned tails, and
`flush()` pads only the final encoder group while preserving output frame
count. Pipeline calls and stream sessions return PIL RGB frames; partial chunks
that do not yet produce causal output return an empty list. GPU execution is
serialized by the shared pipeline lock in the default single-process path.

The optional stage-parallel path splits ReAE encode, DiT denoise, and ReAE
decode into `ParallelWorker` stages. The encode-to-DiT and DiT-to-decode latent
handoffs use `WorkerTensorChannel`, so CUDA tensors are transported by direct
worker-to-worker IPC when profiles fit the channel pool instead of being
materialized in the parent process. This path supports one active stream
session per pipeline because each worker owns its causal state.

Direct sessions are isolated and tested with interleaved inputs so ReAE
boundary state, DiT overlap, RoPE offsets, and frames cannot cross sessions.
`close()` releases all retained causal state. The example intentionally does
not expose `get_service()`: the stock LiveKit protocol has no inbound video
track or frame payload, so a local queue adapter would not provide a usable
`stream-serve` transport.

## H100 baseline

The published SwiftVR QHD result is 31.32 FPS on one H100 for 24 frames.
A local probe of the released implementation measured 31.05 FPS under the same
checkpoint and resolution. For an apples-to-apples core comparison, the
following synchronized GPU timings use BF16, 24-frame chunks, dit_overlap=0,
and exclude PIL conversion and device-to-host output transfer.

| Output resolution | Official SwiftVR | TeleFuser default | TeleFuser opt-in compile |
| --- | ---: | ---: | ---: |
| 2560x1440 | 31.05 FPS local probe (31.32 published) | 32.42 FPS | 40.3-40.5 FPS steady |

The TeleFuser default is therefore faster than the released implementation on
the parity path. The compile result uses torch.compile for the DiT blocks; the
first shape compile takes about 44 seconds on this host, so it is intended for
long-lived processes. The default CLI enables the parity setting dit_overlap=0
used by the published offline benchmark. The public stream() API still
defaults to dit_overlap=1, matching upstream streaming semantics.

The delivered List[PIL.Image.Image] path includes output conversion and
device-to-host transfer and is expected to be lower and more host-variable
than the core GPU number. The existing end-to-end example figures below report
that delivered path separately.

The optional stage-parallel path was measured on three H100 GPUs with
WorkerTensorChannel latent handoff enabled. A 240-input-frame steady run
produced 237 output frames at 39.87-40.11 FPS, with PIL conversion removed from
the benchmark to isolate stage throughput. This is a pipeline throughput
measurement and is not directly comparable to single-GPU memory or latency.

| Resolution | Delivered PIL FPS | TTFC | P50 / P95 chunk | Peak allocated | Retained session state |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920x1080 | 47.28 | 15.01 s | 0.506 / 0.511 s | 27.08 GiB | 235.5 MiB |
| 2560x1440 | 26.50 | 16.31 s | 0.895 / 0.960 s | 35.93 GiB | 411.8 MiB |

The complete example path was separately accepted on one H100 using all 81
frames of examples/data/dag.mp4, resized to 640x360 and restored at 3x. The
measured session produced the 1920x1080 frames in 2.33 seconds (34.78 FPS),
and H.264 encoding took 1.32 seconds, for 22.18 end-to-end FPS. The resulting
video contains 81 frames at 16 FPS. These example figures exclude one-time
model loading, input decoding, and shape warmup; they include output conversion
and device-to-host transfer, and the end-to-end figure additionally includes
video encoding.

TTFC includes cold cuDNN benchmarking and kernel-plan setup after model loading.
Retained memory in the table is the state of one direct causal session. No
multi-session service capacity is claimed because there is no supported
SwiftVR video transport in the shared streaming server.

## Current constraints

- Quantization, `torch.compile`, alternative dense attention backends, and
  stage-parallel execution are opt-in. The default remains BF16 eager Torch
  SDPA for parity.
- The stage-parallel path uses direct tensor channels for latent handoff, but it
  does not change SwiftVR's causal execution order or enable tensor/model
  parallelism inside the DiT blocks.
- Feature cache and sparse-attention substitution are still not enabled for
  SwiftVR.
- The supported online surface is the direct causal session API; the example
  does not expose a partial `stream-serve` adapter.
- RTX 5090 measurements require that hardware and are not represented by H100
  results.
