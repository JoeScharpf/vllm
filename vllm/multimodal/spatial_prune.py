# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic spatial video/image token prune helpers (NPrune, Checkered).

Content-free baselines: no attention or similarity scores. Used by Cosmos3-Edge
video concurrency A/B via ``--video-pruning-method {nprune,checkered}``.
"""

from __future__ import annotations

import torch


NPRUNE_ALLOWED_STRIDES = (1, 2, 3, 4)


def nprune_keep_count(grid_h: int, grid_w: int, stride: int) -> int:
    """Tokens kept by the uniform lattice ``grid[::stride, ::stride]``."""
    return -(-grid_h // stride) * -(-grid_w // stride)


def nprune_select(
    grid_h: int,
    grid_w: int,
    stride: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep upper-left token of every ``stride x stride`` block (raster order)."""
    if stride not in NPRUNE_ALLOWED_STRIDES:
        raise ValueError(
            f"nprune stride must be one of {NPRUNE_ALLOWED_STRIDES}, got {stride}"
        )
    rows = torch.arange(0, grid_h, stride, device=device)
    cols = torch.arange(0, grid_w, stride, device=device)
    kept_idx = (rows.unsqueeze(1) * grid_w + cols.unsqueeze(0)).reshape(-1)
    kept_mask = torch.zeros(grid_h * grid_w, dtype=torch.bool, device=device)
    kept_mask[kept_idx] = True
    assert int(kept_mask.sum()) == nprune_keep_count(grid_h, grid_w, stride)
    return kept_idx, kept_mask


def checkered_keep_count(num_tokens: int) -> int:
    """Tokens kept by the checkerboard: ``ceil(num_tokens / 2)``."""
    return (num_tokens + 1) // 2


def checkered_select(
    grid_h: int,
    grid_w: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep cells where ``(row + col) % 2 == 0`` (upper-left phase)."""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid dims must be positive, got {grid_h}x{grid_w}")
    rows = torch.arange(grid_h, device=device)
    cols = torch.arange(grid_w, device=device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    kept_mask = (((row_grid + col_grid) % 2) == 0).reshape(-1)
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    assert int(kept_mask.sum()) == checkered_keep_count(grid_h * grid_w)
    return kept_idx, kept_mask


def spatial_tokens_per_frame(
    method: str,
    tokens_per_frame_base: int,
    grid_h: int,
    grid_w: int,
    *,
    nprune_stride: int = 2,
) -> int:
    """Per-frame soft-token count after spatial prune (same for every frame)."""
    if method == "nprune":
        return nprune_keep_count(grid_h, grid_w, nprune_stride)
    if method == "checkered":
        return checkered_keep_count(tokens_per_frame_base)
    raise ValueError(f"Unknown spatial prune method: {method}")


def apply_spatial_prune_to_video_embeds(
    emb: torch.Tensor,
    *,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    method: str,
    nprune_stride: int = 2,
) -> torch.Tensor:
    """Prune a ``(T*H*W, D)`` video embed tensor with a per-frame spatial mask."""
    tokens_per_frame = grid_h * grid_w
    if emb.shape[0] != num_frames * tokens_per_frame:
        raise ValueError(
            f"Expected {num_frames * tokens_per_frame} tokens "
            f"(T={num_frames}, H={grid_h}, W={grid_w}), got {emb.shape[0]}"
        )
    if method == "nprune":
        _, kept_mask = nprune_select(
            grid_h, grid_w, nprune_stride, device=emb.device
        )
    elif method == "checkered":
        _, kept_mask = checkered_select(grid_h, grid_w, device=emb.device)
    else:
        raise ValueError(f"Unknown spatial prune method: {method}")

    emb_thw = emb.view(num_frames, tokens_per_frame, emb.shape[-1])
    return emb_thw[:, kept_mask, :].reshape(-1, emb.shape[-1])
