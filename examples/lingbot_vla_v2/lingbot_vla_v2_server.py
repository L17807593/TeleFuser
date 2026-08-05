"""Start a minimal single-GPU LingBot-VLA v2 HTTP service."""

from __future__ import annotations

import click
import uvicorn

from telefuser.pipelines.lingbot_vla_v2.service import LingBotVlaV2ServiceConfig, create_lingbot_vla_v2_app


@click.command()
@click.option("--model-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--qwen3vl-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=click.IntRange(1, 65535))
@click.option("--max-image-mb", default=10, show_default=True, type=click.IntRange(1, 100))
def main(
    model_root: str,
    qwen3vl_root: str,
    device: str,
    host: str,
    port: int,
    max_image_mb: int,
) -> None:
    """Load one policy replica and serve normalized canonical actions."""
    config = LingBotVlaV2ServiceConfig(
        model_root=model_root,
        qwen3vl_root=qwen3vl_root,
        device=device,
        max_image_bytes=max_image_mb * 1024 * 1024,
    )
    app = create_lingbot_vla_v2_app(config)
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
