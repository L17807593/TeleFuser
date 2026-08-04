# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for MiniMax H3 example helpers."""

from telefuser.pipelines.minimax_h3.example_utils import (
    MINIMAX_H3_DEFAULT_FL2VA_IMAGE,
    MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
    MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
    MINIMAX_H3_DEFAULT_REQUEST,
    load_minimax_h3_pipeline,
    load_minimax_h3_request,
    partition_for_minimax_h3_request,
    save_generation,
)

__all__ = [
    "MINIMAX_H3_DEFAULT_FL2VA_IMAGE",
    "MINIMAX_H3_DEFAULT_REF2VA_AUDIO",
    "MINIMAX_H3_DEFAULT_REF2VA_VIDEO",
    "MINIMAX_H3_DEFAULT_REQUEST",
    "load_minimax_h3_pipeline",
    "load_minimax_h3_request",
    "partition_for_minimax_h3_request",
    "save_generation",
]
