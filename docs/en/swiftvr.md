# SwiftVR

TeleFuser integrates the released `H-oliday/SwiftVR` video-restoration model as
a faithful, sequential, single-GPU pipeline. The default path preserves the
upstream BF16 model, one-step DiT, mask-free shifted-window attention, causal
ReAE state, and fixed chunk/flush behavior. Dense attention is dispatched
through `telefuser.ops.attention` with Torch SDPA.

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
count. GPU execution is serialized by the shared pipeline lock.

Direct sessions are isolated and tested with interleaved inputs so ReAE
boundary state, DiT overlap, RoPE offsets, and frames cannot cross sessions.
`close()` releases all retained causal state. The example intentionally does
not expose `get_service()`: the stock LiveKit protocol has no inbound video
track or frame payload, so a local queue adapter would not provide a usable
`stream-serve` transport.

## H100 baseline

The target-side compute measurements exclude model loading, video encoding,
LiveKit delivery, and client playback. Each measured chunk contains 24 output
frames after two warmup chunks.

| Resolution | Compute FPS | TTFC | P50 / P95 chunk | Peak allocated | Retained session state |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920x1080 | 47.28 | 15.01 s | 0.506 / 0.511 s | 27.08 GiB | 235.5 MiB |
| 2560x1440 | 26.50 | 16.31 s | 0.895 / 0.960 s | 35.93 GiB | 411.8 MiB |

The complete example path was separately accepted on one H100 using all 81
frames of `examples/data/dag.mp4`, resized to `640x360` and restored at 3x. The
measured session produced the `1920x1080` frames in 2.33 seconds (34.78 FPS),
and H.264 encoding took 1.32 seconds, for 22.18 end-to-end FPS. The resulting
video contains 81 frames at 16 FPS. These example figures exclude one-time
model loading, input decoding, and shape warmup; they include output conversion
and device-to-host transfer, and the end-to-end figure additionally includes
video encoding.

TTFC includes cold cuDNN benchmarking and kernel-plan setup after model loading.
Retained memory in the table is the state of one direct causal session. No
multi-session service capacity is claimed because there is no supported
SwiftVR video transport in the shared streaming server.

The 1080p long-stream run completed 1176 measured chunks and 28,224 frames over
600.46 seconds at 47.01 compute FPS. P50/P95/max chunk latency was
0.508/0.523/0.654 seconds. The first and last 100 chunks had P50 values of
0.50810 and 0.50796 seconds respectively, so no latency growth was observed.
Session cleanup completed and the benchmark process returned all device memory
to the driver.

## Current constraints

- No quantization, feature cache, sparse-attention substitution, or
  `torch.compile` path is enabled.
- No stage splitting or pipeline overlap is enabled; execution remains in the
  upstream order.
- The public example uses eager Torch SDPA and does not expose alternative
  attention backends.
- The supported online surface is the direct causal session API; the example
  does not expose a partial `stream-serve` adapter.
- RTX 5090 measurements require that hardware and are not represented by H100
  results.
