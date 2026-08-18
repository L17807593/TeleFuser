# LTX-2.5 Distilled Example

`ltx25_distilled_t2v_i2v_h100.py` runs the isolated single-H100 LTX-2.5
distilled pipeline and writes an MP4 with synchronized 48 kHz audio.

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_i2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --prompt "A cinematic camera orbit around the subject." \
  --output-path /tmp/ltx25-t2v.mp4
```

Add `--image-path`, `--image-frame-index`, and `--image-strength` for I2V.
The model root must contain the official split LTX-2.5 checkpoint layout. Use
`--video-vae conv` to select ConvVAE; the default is DiffVAE.
Use `--offload cpu` for sequential CPU release or `--offload none` to retain
models on GPU between phases when the H100 memory budget permits it.

The formal 1536x1024 / 121-frame DiffVAE workload requires NATTEN. The
Triton fallback remains a compatibility path and is not the formal performance
or accuracy baseline.
