import math
import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="qwen_t2i_fp8_h100",
    model_root=TF_MODEL_ZOO_PATH + "/Qwen-Image-2512",
    dit_filename=TF_MODEL_ZOO_PATH
    + "/Qwen-Image-2512-Lightning/qwen_image_2512_fp8_e4m3fn_scaled_8steps_v1.0.safetensors",
    vae_filename="vae/diffusion_pytorch_model.safetensors",
    negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=0,
    sample_solver="euler",
    cfg_scale=1.0,
    num_inference_steps=16,
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    offload_type: WeightOffloadType = WeightOffloadType.NO_CPU_OFFLOAD,
    offload_ratio: float = 1.0,
    prefetch_size: int = 1,
):
    vae_path = [f"{model_root}/{PPL_CONFIG['vae_filename']}"]
    text_encoder_path = [
        f"{model_root}/text_encoder/model-00001-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00002-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00003-of-00004.safetensors",
        f"{model_root}/text_encoder/model-00004-of-00004.safetensors",
    ]
    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(PPL_CONFIG["dit_filename"], device="cpu", torch_dtype=torch.float8_e4m3fn)
    mm.load_model(vae_path, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(text_encoder_path, device="cpu", torch_dtype=torch.bfloat16)
    pipeline = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = QwenImagePipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_config.offload_config.offload_type = offload_type
    pipe_config.dit_config.offload_config.offload_ratio = offload_ratio
    pipe_config.dit_config.offload_config.prefetch_size = prefetch_size
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
        pipe_config.enable_denoising_parallel = True
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: QwenImagePipeline,
    prompt,
    aspect_ratio: str = "1:1",
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
):
    height, width = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    images = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cpu",
        cfg_scale=PPL_CONFIG["cfg_scale"],
        exponential_shift_mu=math.log(3),
        shift_terminal=None,
        num_images_per_prompt=1,
    )
    return images


@click.command()
@click.option("--aspect_ratio", "-ar", default="1:1", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--gpu_num", default=1, help="Number of GPUs to use", type=int)
@click.option(
    "--prompt",
    default="A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.",
    help="Custom prompt text",
)
@click.option("--negative_prompt", default=PPL_CONFIG["negative_prompt"], help="Negative prompt")
@click.option("--output", default="image.jpg", help="Output image filename")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
@click.option(
    "--offload_type",
    type=click.Choice(["none", "async"]),
    default="none",
    show_default=True,
    help="DiT weight offload strategy",
)
@click.option(
    "--offload_ratio",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1.0,
    show_default=True,
    help="Fraction of DiT blocks offloaded for async mode",
)
@click.option(
    "--prefetch_size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of upcoming DiT blocks prefetched for async mode",
)
def main(
    aspect_ratio, gpu_num, prompt, negative_prompt, output, model_root, offload_type, offload_ratio, prefetch_size
):
    weight_offload_type = (
        WeightOffloadType.ASYNC_CPU_OFFLOAD if offload_type == "async" else WeightOffloadType.NO_CPU_OFFLOAD
    )
    pipeline = get_pipeline(gpu_num, model_root, weight_offload_type, offload_ratio, prefetch_size)

    torch.cuda.reset_peak_memory_stats()

    # Warm up
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)

    # Timing run
    torch.cuda.synchronize()
    s = time.time()
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)
    torch.cuda.synchronize()
    print(f"pipe cost {time.time() - s} s")
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    for i, image in enumerate(images):
        image.save(output.replace(".jpg", f"_{i}.jpg"))
    del pipeline


if __name__ == "__main__":
    main()
