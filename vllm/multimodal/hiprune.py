# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiPrune: training-free visual token pruning via hierarchical attention.

Implements the token selection method from "HiPrune: Training-Free Visual
Token Pruning via Hierarchical Attention in Vision-Language Models"
(https://arxiv.org/abs/2508.00553) for image inputs. Two encoder families
are supported:

- Encoders that pool patches into soft tokens (Gemma 4): see
  :func:`compute_soft_token_grid` / :func:`aggregate_patch_attention`.
- Encoders with a spatial patch merger (Qwen2.5-VL): see
  :func:`fold_merged_token_scores`.

The method exploits the hierarchical attention inside the vision encoder:

- **Anchor tokens**: highest-attention tokens in a *middle* encoder layer
  (object-centric).
- **Buffer tokens**: the four spatial neighbors of each anchor (local
  context).
- **Register tokens**: highest-attention tokens in the *last* encoder
  layer (deep layers repurpose low-information patches as global
  aggregation slots).

Everything else is pruned before the tokens reach the language model.

Split of responsibilities (mirrors EVS in ``vllm/multimodal/evs.py``):

- The multimodal processor calls :func:`compute_retained_tokens_count` to
  emit the reduced number of placeholder tokens. This count is a pure
  function of the token count and pruning ratio, so it is known before
  the vision encoder ever runs.
- The model calls :func:`hiprune_select` at encode time to decide *which*
  tokens fill those placeholders, using attention captured from the
  vision encoder. :func:`hiprune_select` keeps exactly
  ``compute_retained_tokens_count(...)`` tokens by construction, which is
  the invariant that keeps the prompt and the encoder output consistent.
"""

from collections.abc import Mapping

import torch

# Default HiPrune hyperparameters for Gemma 4's 16-layer vision encoder.
# The object layer (1-based) is the middle of the encoder; alpha is the
# paper's anchor-budget fraction. Neither is paper-validated for Gemma
# (the paper covers LLaVA and Qwen2.5-VL only).
GEMMA4_OBJECT_LAYER = 8
# Object layer (1-based) for Qwen2.5-VL's 32-layer vision encoder. This is
# the middle layer, matching the authors' released Qwen2.5-VL configuration
# (HIPRUNE_OBJECT_LAYER=16); the deep layer is the last (32nd) block.
QWEN2_5_VL_OBJECT_LAYER = 16
DEFAULT_ALPHA = 0.1


def get_hiprune_ratio(merged_kwargs: Mapping[str, object]) -> float | None:
    """Extract and validate the HiPrune retention ratio from mm kwargs.

    The ratio is the fraction of image tokens KEPT (e.g. 0.14 keeps 14%).
    ``None`` or ``1.0`` disables pruning. Passed per-request via
    ``mm_processor_kwargs={"hiprune_ratio": ...}`` (or the ``token_pruning``
    chat-completions field, which maps onto it).
    """
    val = merged_kwargs.get("hiprune_ratio")
    if val is None:
        return None
    ratio = float(val)  # type: ignore[arg-type]
    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"hiprune_ratio must be in (0, 1], got {ratio}. It is the "
            "fraction of image tokens to KEEP."
        )
    if ratio == 1.0:
        return None
    # Quantize to float32: the ratio travels to the model in a float32
    # tensor, and compute_retained_tokens_count rounds, so the processor
    # must use the exact same bits or the placeholder count can differ
    # from the model's kept-token count near half-integer boundaries.
    return float(torch.tensor(ratio, dtype=torch.float32).item())


def compute_retained_tokens_count(num_tokens: int, pruning_ratio: float) -> int:
    """Number of soft tokens kept for an image at the given retention.

    ``pruning_ratio`` here follows the request semantics: it is the
    *retention* ratio (fraction of tokens kept), e.g. ``0.14`` keeps 14%
    of the tokens.

    Called by both the multimodal processor (to size the placeholder
    sequence) and :func:`hiprune_select` (as the selection budget), so
    the two can never disagree.
    """
    return max(1, min(num_tokens, round(num_tokens * pruning_ratio)))


def compute_soft_token_grid(
    pixel_position_ids: torch.Tensor,
    pooling_kernel_size: int,
) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    """Derive the pooled soft-token grid from per-patch positions.

    The Gemma4 image processor pads ``pixel_position_ids`` up to a fixed
    patch budget with ``(-1, -1)`` entries; those must be excluded from
    both the grid derivation and the patch->soft-token mapping (padding
    positions would map to negative kernel indices).

    Args:
        pixel_position_ids: ``(num_patches, 2)`` of ``(x, y)`` patch
            coordinates, ``(-1, -1)`` for padding.
        pooling_kernel_size: The pooler's spatial kernel (3 for Gemma4).

    Returns:
        ``(valid, grid_w, grid_h, kernel_idx)`` where ``valid`` is a
        ``(num_patches,)`` bool mask of real patches and ``kernel_idx``
        maps each *valid* patch to its soft-token index, using the same
        arithmetic as ``Gemma4VisionPooler._avg_pool_by_positions``:
        ``(x // k) + (patch_w // k) * (y // k)``.
    """
    valid = ~((pixel_position_ids == -1).all(dim=-1))
    xs = pixel_position_ids[valid, 0]
    ys = pixel_position_ids[valid, 1]
    patch_w = int(xs.max()) + 1
    patch_h = int(ys.max()) + 1
    grid_w = patch_w // pooling_kernel_size
    grid_h = patch_h // pooling_kernel_size
    kernel_idx = (xs // pooling_kernel_size) + grid_w * (ys // pooling_kernel_size)
    return valid, grid_w, grid_h, kernel_idx


def aggregate_patch_attention(
    layer_attention: torch.Tensor,
    valid: torch.Tensor,
    kernel_idx: torch.Tensor,
    num_soft_tokens: int,
) -> torch.Tensor:
    """Aggregate one layer's patch attention into per-soft-token scores.

    Patch score = mean over heads, mean over valid queries (the "global
    attention" variant the HiPrune authors use for CLS-free encoders);
    soft-token score = sum over each pooling window. Padding patches are
    excluded on both axes: padding keys are attention-masked by the
    encoder anyway, but padding query rows are garbage and must not be
    averaged in. Computed in float32.

    Args:
        layer_attention: ``(num_heads, num_patches, num_patches)``
            post-softmax attention weights for one image.
        valid: ``(num_patches,)`` bool mask of real patches.
        kernel_idx: patch -> soft-token index for valid patches.
        num_soft_tokens: size of the pooled soft-token grid.

    Returns:
        ``(num_soft_tokens,)`` float32 scores (a distribution: sums to 1).
    """
    attn = layer_attention.float().mean(dim=0)  # (queries, keys)
    patch_scores = attn[valid][:, valid].mean(dim=0)  # (num_valid,)
    weights = torch.nn.functional.one_hot(kernel_idx.long(), num_soft_tokens)
    return weights.float().T @ patch_scores


def fold_merged_token_scores(
    patch_scores: torch.Tensor,
    spatial_merge_unit: int,
    reverse_indices: torch.Tensor,
) -> torch.Tensor:
    """Fold per-patch key scores into per-merged-token scores (Qwen2.5-VL).

    Mirrors the authors' Qwen2.5-VL reference aggregation: the vision
    tower processes patches in window-permuted order, where each
    consecutive group of ``spatial_merge_unit`` patches is one 2x2 merge
    unit; the per-patch scores are averaged within each unit and then
    un-permuted back to the original (raster) merged-token order with
    ``reverse_indices`` (the inverse of the tower's window permutation).

    Args:
        patch_scores: ``(num_patches,)`` per-key attention scores in the
            tower's window-permuted patch order (mean over heads and
            queries of the post-softmax attention).
        spatial_merge_unit: patches per merged token (4 for Qwen2.5-VL).
        reverse_indices: ``(num_patches // spatial_merge_unit,)`` inverse
            window permutation over merge units.

    Returns:
        ``(num_merged_tokens,)`` float32 scores in raster order.
    """
    unit_scores = patch_scores.float().view(-1, spatial_merge_unit).mean(dim=-1)
    return unit_scores[reverse_indices]


def hiprune_select(
    shallow_scores: torch.Tensor,
    deep_scores: torch.Tensor,
    num_tokens: int,
    grid_w: int,
    pruning_ratio: float,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The HiPrune anchor/buffer/register token selection.

    Exact arithmetic from the authors' released code (Qwen2.5-VL model
    file / LLaVA ``llava_arch.py``), separated into the three categories.
    Keeps exactly ``compute_retained_tokens_count(num_tokens,
    pruning_ratio)`` tokens: anchors + buffers fill part of the budget
    and registers fill the remainder; the ``deep -= selected`` trick
    guarantees registers never overlap the anchor/buffer set (attention
    scores are positive, so subtracting 1 puts already-selected tokens
    below every unselected one).

    Args:
        shallow_scores: ``(num_tokens,)`` soft-token scores from the
            object (middle) encoder layer.
        deep_scores: ``(num_tokens,)`` soft-token scores from the last
            encoder layer.
        num_tokens: total soft tokens for the image.
        grid_w: soft-token grid width (for spatial buffer neighbors).
        pruning_ratio: retention ratio in ``(0, 1]``.
        alpha: fraction of the budget allotted to anchors (each anchor
            also pulls in up to 4 buffer neighbors, hence the /5).

    Returns:
        ``(anchor_idx, buffer_idx, register_idx, kept_mask)`` where
        ``kept_mask`` is a ``(num_tokens,)`` bool mask with exactly the
        budgeted number of True entries.
    """
    deep = deep_scores.clone()  # the reference implementation mutates in-place

    budget = compute_retained_tokens_count(num_tokens, pruning_ratio)
    shallow_token_num = round((budget * alpha) / 5)

    anchor_idx = torch.topk(shallow_scores, k=shallow_token_num).indices
    shallow_all = torch.cat(
        [
            anchor_idx,
            anchor_idx - 1,
            anchor_idx + 1,
            anchor_idx - grid_w,
            anchor_idx + grid_w,
        ]
    )
    shallow_all = shallow_all.clamp(0, num_tokens - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    buffer_idx = shallow_all[~torch.isin(shallow_all, anchor_idx)]

    deep_token_num = budget - shallow_all.shape[0]
    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=deep.device)
    selected_mask.scatter_(0, shallow_all, 1)
    deep -= selected_mask.int()
    register_idx = torch.topk(deep, k=deep_token_num).indices

    kept_mask = selected_mask.clone()
    kept_mask[register_idx] = True
    return anchor_idx, buffer_idx, register_idx, kept_mask


def build_hiprune_metadata(
    anchor_idx: torch.Tensor,
    buffer_idx: torch.Tensor,
    register_idx: torch.Tensor,
    kept_mask: torch.Tensor,
    shallow_scores: torch.Tensor,
    deep_scores: torch.Tensor,
    grid_w: int,
    grid_h: int,
    retention: float,
    object_layer: int = GEMMA4_OBJECT_LAYER,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Assemble the JSON-safe per-image pruning metadata for API reporting.

    Mirrors the statistics of the reference Colab visualizer: the token
    category index sets plus each category's mean attention at the object
    (middle) and deep (last) encoder layers. Scores are distributions
    over soft tokens, so a category mean of ``1/num_tokens`` is the
    uniform baseline.
    """

    def _mean(scores: torch.Tensor, idx: torch.Tensor) -> float | None:
        return float(scores[idx].mean()) if idx.numel() else None

    pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    categories = {
        "anchor": anchor_idx,
        "buffer": buffer_idx,
        "register": register_idx,
        "kept": kept_idx,
        "pruned": pruned_idx,
    }
    return {
        "grid": [grid_w, grid_h],
        "num_tokens": int(kept_mask.shape[0]),
        "retention": retention,
        "object_layer": object_layer,
        "alpha": alpha,
        "pruned": pruned_idx.tolist(),
        "anchors": anchor_idx.tolist(),
        "buffers": buffer_idx.tolist(),
        "registers": register_idx.tolist(),
        "mean_attention": {
            "object_layer": {
                name: _mean(shallow_scores, idx) for name, idx in categories.items()
            },
            "deep_layer": {
                name: _mean(deep_scores, idx) for name, idx in categories.items()
            },
        },
    }
