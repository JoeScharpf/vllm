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

This module also implements **HyDART** (:func:`hydart_select`), a hybrid
variant selectable via ``HIPRUNE_METHOD=hydart``: anchors and buffers are
selected exactly as HiPrune does, but the remainder of the budget is
filled by a greedy maximal-marginal-relevance (MMR) loop over the visual
embeddings the language model consumes (DART-style duplication avoidance,
arXiv 2502.11494) instead of deep-layer registers. This removes the
second dense attention-score capture — the dominant selection cost on
large images — since only the object layer is needed.

**HiPrune++** (:func:`hiprune_pp_select`, ``HIPRUNE_METHOD=hiprune_pp``)
is the prompt-aware variant from the same paper (Appendix A, Algorithm
1): after the standard anchor/buffer/register selection it additionally
keeps the ``round(beta * budget)`` visual tokens most cosine-similar to
the prompt's mean text embedding, additive on top of the base budget.
The paper uses a paired CLIP text encoder where one exists; per the
paper's own fallback for encoders without one, this implementation
compares the LM-space visual embeddings against the average of the
language model's text embeddings of the prompt — uniformly for all
models. Because the prompt now influences selection, the API layer
attaches the user prompt as an mm kwarg (``hiprune_prompt``), which also
makes the multimodal cache hash prompt-dependent (correct: the same
image pruned under a different prompt keeps different tokens).

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

import os
from collections.abc import Mapping
from dataclasses import dataclass

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
# Object layer (1-based) for LLaVA-1.5's 24-layer CLIP vision encoder,
# from the paper's released LLaVA configuration. The deep layer is the
# feature-select layer (-2, i.e. the 23rd block — the last one vLLM
# loads, since the tower is truncated at the feature layer).
LLAVA_OBJECT_LAYER = 9
DEFAULT_ALPHA = 0.1

# HyDART MMR penalties (see hydart_select). lambda_seed penalizes
# resembling the anchor+buffer seed set (keep small: the object is already
# partially kept, resembling it should not be a death sentence);
# lambda_pick penalizes resembling already-picked diverse tokens (keep
# larger: near-duplicates add nothing).
DEFAULT_HYDART_LAMBDA_SEED = 0.1
DEFAULT_HYDART_LAMBDA_PICK = 0.5
# Above this token count the greedy loop switches to block picks (top-k
# per round) to bound the number of sequential GPU steps.
HYDART_BLOCK_THRESHOLD = 2048
HYDART_BLOCK_SIZE = 8

# HiPrune++ text-guidance proportion. Paper: "We set beta = 0.1 for all
# the models when evaluating Hiprune++."
DEFAULT_HIPRUNE_PP_BETA = 0.1

# DART pivot counts, from the official eval scripts
# (DART/Qwen2_5-VL/eval_scripts/lmms_eval.sh: pivot_image_token=4,
# pivot_text_token=4) and the LLM layer whose states drive selection
# (pruned_layer=2: the official code prunes when entering layer index 2,
# using the outputs of layer index 1 — i.e. after running 2 layers).
DEFAULT_DART_PIVOT_IMAGE = 4
DEFAULT_DART_PIVOT_TEXT = 4
DEFAULT_DART_LAYER = 2


def get_hiprune_method(merged_kwargs: Mapping[str, object] | None = None) -> str:
    """Selection method for ``--enable-hiprune`` servers.

    Per-request via ``mm_processor_kwargs={"hiprune_method": ...}``
    (attached by the API layer from the ``token_pruning_method`` chat
    field), falling back to the ``HIPRUNE_METHOD`` env var so
    env-configured servers and scripts keep working. One of ``hiprune``
    (default), ``hydart``, ``hiprune_pp`` or ``dart``.
    """
    val = merged_kwargs.get("hiprune_method") if merged_kwargs else None
    if val is None:
        val = os.environ.get("HIPRUNE_METHOD", "hiprune")
    method = str(val).lower()
    if method not in ("hiprune", "hydart", "hiprune_pp", "dart"):
        raise ValueError(
            "hiprune method must be 'hiprune', 'hydart', 'hiprune_pp' or "
            f"'dart', got {method!r}"
        )
    return method


def get_hiprune_pp_beta(merged_kwargs: Mapping[str, object] | None = None) -> float:
    """HiPrune++ text-guidance proportion.

    Per-request ``hiprune_beta`` mm kwarg, else ``HIPRUNE_PP_BETA`` env,
    else the paper default.
    """
    val = merged_kwargs.get("hiprune_beta") if merged_kwargs else None
    if val is None:
        val = os.environ.get("HIPRUNE_PP_BETA", DEFAULT_HIPRUNE_PP_BETA)
    beta = float(val)  # type: ignore[arg-type]
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"hiprune beta must be in [0, 1], got {beta}")
    return beta


def get_dart_pivots(
    merged_kwargs: Mapping[str, object] | None = None,
) -> tuple[int, int]:
    """(pivot_image, pivot_text) counts.

    Per-request ``hiprune_pivot_image`` / ``hiprune_pivot_text`` mm
    kwargs, else ``HIPRUNE_DART_PIVOT_IMAGE/TEXT`` env, else the paper
    defaults.
    """
    p_img_val = merged_kwargs.get("hiprune_pivot_image") if merged_kwargs else None
    if p_img_val is None:
        p_img_val = os.environ.get(
            "HIPRUNE_DART_PIVOT_IMAGE", DEFAULT_DART_PIVOT_IMAGE
        )
    p_txt_val = merged_kwargs.get("hiprune_pivot_text") if merged_kwargs else None
    if p_txt_val is None:
        p_txt_val = os.environ.get(
            "HIPRUNE_DART_PIVOT_TEXT", DEFAULT_DART_PIVOT_TEXT
        )
    p_img = int(p_img_val)  # type: ignore[arg-type]
    p_txt = int(p_txt_val)  # type: ignore[arg-type]
    if p_img < 1 or p_txt < 0:
        raise ValueError(
            "DART pivots must satisfy pivot_image >= 1 and pivot_text >= 0, "
            f"got ({p_img}, {p_txt})"
        )
    return p_img, p_txt


def get_dart_layer() -> int:
    """Number of LLM decoder layers run for DART scoring (paper: 2)."""
    layer = int(os.environ.get("HIPRUNE_DART_LAYER", DEFAULT_DART_LAYER))
    if layer < 1:
        raise ValueError(f"HIPRUNE_DART_LAYER must be >= 1, got {layer}")
    return layer


def get_hydart_lambdas(
    merged_kwargs: Mapping[str, object] | None = None,
) -> tuple[float, float]:
    """(lambda_seed, lambda_pick).

    Per-request ``hiprune_lambda_seed`` / ``hiprune_lambda_pick`` mm
    kwargs, else ``HYDART_LAMBDA_SEED/PICK`` env, else paper-Colab
    defaults.
    """
    seed_val = merged_kwargs.get("hiprune_lambda_seed") if merged_kwargs else None
    if seed_val is None:
        seed_val = os.environ.get("HYDART_LAMBDA_SEED", DEFAULT_HYDART_LAMBDA_SEED)
    pick_val = merged_kwargs.get("hiprune_lambda_pick") if merged_kwargs else None
    if pick_val is None:
        pick_val = os.environ.get("HYDART_LAMBDA_PICK", DEFAULT_HYDART_LAMBDA_PICK)
    return float(seed_val), float(pick_val)  # type: ignore[arg-type]


# All vLLM-side pruning keys that may appear in mm_processor_kwargs.
# Model processors strip these before calling the HF processor.
HIPRUNE_MM_KWARG_KEYS = (
    "hiprune_ratio",
    "hiprune_prompt",
    "hiprune_method",
    "hiprune_lambda_seed",
    "hiprune_lambda_pick",
    "hiprune_beta",
    "hiprune_pivot_image",
    "hiprune_pivot_text",
)

# Method <-> id mapping for the packed per-image config tensor (mm
# fields must be tensors, so the method travels as a float id).
HIPRUNE_METHOD_IDS: dict[str, int] = {
    "hiprune": 0,
    "hydart": 1,
    "hiprune_pp": 2,
    "dart": 3,
}
_HIPRUNE_ID_METHODS = {v: k for k, v in HIPRUNE_METHOD_IDS.items()}

# Row layout of the packed config: (method_id, lambda_seed, lambda_pick,
# beta, pivot_image, pivot_text).
HIPRUNE_CONFIG_WIDTH = 6


@dataclass(frozen=True)
class HipruneConfig:
    """Per-image pruning configuration, decoded from the packed row."""

    method: str
    lambda_seed: float
    lambda_pick: float
    beta: float
    pivot_image: int
    pivot_text: int


def pack_hiprune_config(
    merged_kwargs: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Encode the request's method + knobs as one float32 row.

    Attached per image by the multimodal processor (mirroring how
    ``hiprune_ratio`` travels) so the model forward can dispatch the
    selection method per image — a batch may span requests with
    different methods.
    """
    method = get_hiprune_method(merged_kwargs)
    lambda_seed, lambda_pick = get_hydart_lambdas(merged_kwargs)
    beta = get_hiprune_pp_beta(merged_kwargs)
    pivot_image, pivot_text = get_dart_pivots(merged_kwargs)
    return torch.tensor(
        [
            float(HIPRUNE_METHOD_IDS[method]),
            lambda_seed,
            lambda_pick,
            beta,
            float(pivot_image),
            float(pivot_text),
        ],
        dtype=torch.float32,
    )


def unpack_hiprune_config(row: torch.Tensor | None) -> HipruneConfig:
    """Decode a packed config row; ``None`` falls back to env/defaults."""
    if row is None:
        return unpack_hiprune_config(pack_hiprune_config())
    vals = row.float().tolist()
    if len(vals) != HIPRUNE_CONFIG_WIDTH:
        raise ValueError(
            f"hiprune_config row must have {HIPRUNE_CONFIG_WIDTH} entries, "
            f"got {len(vals)}"
        )
    method = _HIPRUNE_ID_METHODS.get(int(round(vals[0])))
    if method is None:
        raise ValueError(f"unknown hiprune method id {vals[0]!r}")
    return HipruneConfig(
        method=method,
        lambda_seed=vals[1],
        lambda_pick=vals[2],
        beta=vals[3],
        pivot_image=int(round(vals[4])),
        pivot_text=int(round(vals[5])),
    )


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


def get_hiprune_prompt(merged_kwargs: Mapping[str, object]) -> str | None:
    """Extract the HiPrune++ prompt text from mm kwargs.

    Attached by the API layer alongside ``hiprune_ratio`` when the server
    runs with ``HIPRUNE_METHOD=hiprune_pp`` (the text parts of the latest
    user message). May be ``None`` for image-only messages; the keep
    *count* never depends on the prompt (only on ratio and beta), so
    callers fall back to a zero text embedding — all similarities zero,
    the text slots are filled arbitrarily but the count invariant holds.
    """
    val = merged_kwargs.get("hiprune_prompt")
    if val is None:
        return None
    text = str(val)
    return text if text.strip() else None


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


def compute_hiprune_pp_budget(
    num_tokens: int,
    pruning_ratio: float,
    beta: float | None = None,
) -> tuple[int, int]:
    """HiPrune++ keep budget: ``(base_budget, text_token_count)``.

    The base budget is the plain HiPrune budget
    (:func:`compute_retained_tokens_count`); the text-guided tokens are
    *additive* on top of it, exactly as in the paper's Algorithm 1 where
    ``retained_idx = cat([a_idx, b_idx, r_idx, t_idx])`` after the base
    categories already fill ``N``. For the count, the paper's prose and
    pseudo-code disagree: the prose says "retain [beta * N] visual
    tokens ... where beta is the proportion of visual tokens selected by
    text-relevance", while the pseudo-code line reads
    ``t_sum = round(N * beta / 5)``. We follow the prose — the ``/5``
    mirrors the anchor line, where it exists because each anchor pulls
    in 4 buffer neighbors, a justification that does not apply to text
    tokens; and at the paper's 64-token LLaVA budget the ``/5`` form
    would add a single token, which cannot produce the reported gains.

    ``t_sum`` is clamped so the total never exceeds ``num_tokens``.
    Called by both the multimodal processor (placeholder sizing:
    ``base + t_sum``) and :func:`hiprune_pp_select`, so the two can
    never disagree.
    """
    if beta is None:
        beta = get_hiprune_pp_beta()
    base = compute_retained_tokens_count(num_tokens, pruning_ratio)
    t_sum = round(base * beta)
    return base, max(0, min(t_sum, num_tokens - base))


def dart_keep_count(
    num_tokens: int,
    pruning_ratio: float,
    pivot_image: int | None = None,
    pivot_text: int | None = None,
) -> int:
    """Number of image tokens DART keeps — deterministic, prompt-free.

    Exact arithmetic from the official implementation
    (``DART/Qwen2_5-VL/.../modeling_qwen2_5_vl_self.py``):
    ``TOKEN_TOPK = int(L * retention / (p_img + p_txt))`` and the final
    kept set is the image pivots plus ``(p_img + p_txt)`` disjoint
    anti-duplication picks of ``TOKEN_TOPK`` tokens each (text pivots are
    dropped from the set at the end, being text). Note the official
    ``reduction_ratio`` is the fraction *removed*; ``pruning_ratio`` here
    follows this codebase's request semantics (fraction KEPT), i.e.
    ``retention = 1 - reduction_ratio``.

    Clamped to ``[1, num_tokens]`` (the official code would crash on the
    rare boundary where the pick rounds exceed the candidate pool).

    Called by both the multimodal processor (placeholder sizing) and
    :func:`dart_select`, so the two can never disagree. The count never
    depends on the prompt: a missing/short prompt only changes *which*
    tokens fill the budget (see the top-up rule in :func:`dart_select`).
    """
    if pivot_image is None or pivot_text is None:
        env_img, env_txt = get_dart_pivots()
        pivot_image = env_img if pivot_image is None else pivot_image
        pivot_text = env_txt if pivot_text is None else pivot_text
    p_img_eff = min(pivot_image, num_tokens)
    token_topk = int(
        num_tokens * pruning_ratio / (pivot_image + pivot_text)
    )
    kept = p_img_eff + (pivot_image + pivot_text) * token_topk
    return max(1, min(num_tokens, kept))


def dart_select(
    hidden_states: torch.Tensor,
    key_l1_norms: torch.Tensor,
    num_image_tokens: int,
    pruning_ratio: float,
    pivot_image: int | None = None,
    pivot_text: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DART: duplication-aware selection from layer-K LLM states.

    Verbatim port of the official ``get_retained_image_token``
    (arXiv 2502.11494, EMNLP'25): pick the ``pivot_image`` image tokens
    and ``pivot_text`` text tokens with the largest attention-key L1
    norms as pivots, then for each pivot keep the ``TOKEN_TOPK``
    remaining image tokens *least* cosine-similar to it (anti-
    duplication), removing them from the candidate pool as they are
    picked. Image pivots are kept; text pivots only guide selection.

    Deviations from the official code, all deliberate:

    - **Pivot iteration order** is image pivots then text pivots, each
      ascending by index. The official code iterates a Python ``set``
      (arbitrary order); order slightly influences which pivot claims a
      contested token, so we fix a deterministic one.
    - **Top-up rule**: if the pivot rounds cannot fill the budget — no
      prompt (no text pivots), a prompt shorter than ``pivot_text``
      tokens, or the boundary clamp in :func:`dart_keep_count` — the
      remainder is filled by key-L1-norm rank over the remaining
      candidates. The official code has no such path (it would crash);
      this keeps the kept count equal to :func:`dart_keep_count` always,
      which the placeholder machinery requires.

    Args:
        hidden_states: ``(seq, hidden)`` — the aux scoring pass output
            for image tokens followed by prompt tokens, with the model's
            final norm applied (matching ``self.norm(layer_outputs[0])``
            in the official code). Any float dtype.
        key_l1_norms: ``(seq,)`` — per-token L1 norm of the layer-(K−1)
            post-RoPE attention keys, flattened across kv heads
            (``torch.norm(k, p=1, dim=-1)`` on ``(seq, kv_heads*head_dim)``).
        num_image_tokens: image tokens occupy ``[0, num_image_tokens)``;
            everything after is prompt text.
        pruning_ratio: retention ratio in ``(0, 1]``.
        pivot_image / pivot_text: pivot counts; ``None`` reads the env
            (``HIPRUNE_DART_PIVOT_IMAGE/TEXT``, paper defaults 4/4).

    Returns:
        ``(image_pivot_idx, text_pivot_idx, diverse_idx, kept_mask,
        pivot_similarity)``. ``text_pivot_idx`` is prompt-relative
        (0 = first prompt token). ``kept_mask`` is over image tokens
        with exactly ``dart_keep_count(...)`` True entries.
        ``pivot_similarity[i]`` is image token *i*'s max cosine
        similarity to any pivot (high = duplicated by a pivot; kept
        diverse tokens have low values), for metadata/tooltips.
    """
    if pivot_image is None or pivot_text is None:
        env_img, env_txt = get_dart_pivots()
        pivot_image = env_img if pivot_image is None else pivot_image
        pivot_text = env_txt if pivot_text is None else pivot_text

    seq_len = hidden_states.shape[0]
    num_text = seq_len - num_image_tokens
    device = hidden_states.device

    budget = dart_keep_count(
        num_image_tokens, pruning_ratio, pivot_image, pivot_text
    )
    token_topk = int(
        num_image_tokens * pruning_ratio / (pivot_image + pivot_text)
    )

    norms = key_l1_norms.float()
    p_img_eff = min(pivot_image, num_image_tokens)
    image_pivot_idx = torch.topk(norms[:num_image_tokens], k=p_img_eff).indices
    p_txt_eff = min(pivot_text, num_text)
    if p_txt_eff > 0:
        text_pivot_rel = torch.topk(norms[num_image_tokens:], k=p_txt_eff).indices
    else:
        text_pivot_rel = torch.empty(0, dtype=torch.long, device=device)

    # hidden_states already carries the final norm; cosine similarity
    # additionally normalizes, exactly like the official F.cosine_similarity.
    hs = torch.nn.functional.normalize(hidden_states.float(), dim=-1)

    pivot_order = torch.cat(
        [
            image_pivot_idx.sort().values,
            text_pivot_rel.sort().values + num_image_tokens,
        ]
    )
    pivot_similarity = (
        (hs[:num_image_tokens] @ hs[pivot_order].T).max(dim=-1).values
        if pivot_order.numel()
        else torch.zeros(num_image_tokens, device=device)
    )

    candidates = torch.ones(num_image_tokens, dtype=torch.bool, device=device)
    candidates[image_pivot_idx] = False

    diverse_total = budget - p_img_eff
    remaining = diverse_total
    diverse_chunks: list[torch.Tensor] = []
    for pivot in pivot_order:
        if remaining <= 0:
            break
        cand_idx = candidates.nonzero(as_tuple=True)[0]
        k = min(token_topk, remaining, int(cand_idx.numel()))
        if k <= 0:
            break
        # Official: cos_sim = -cosine_similarity(pivot, candidates);
        # topk of the negation keeps the LEAST similar (anti-duplication).
        sims = hs[cand_idx] @ hs[pivot]
        picked = cand_idx[torch.topk(-sims, k=k).indices]
        diverse_chunks.append(picked)
        candidates[picked] = False
        remaining -= k

    if remaining > 0:
        # Top-up (missing/short prompt or boundary clamp): fill by key
        # L1-norm rank, deterministic and prompt-free.
        cand_idx = candidates.nonzero(as_tuple=True)[0]
        k = min(remaining, int(cand_idx.numel()))
        if k > 0:
            picked = cand_idx[
                torch.topk(norms[:num_image_tokens][cand_idx], k=k).indices
            ]
            diverse_chunks.append(picked)
            candidates[picked] = False
            remaining -= k

    if diverse_chunks:
        diverse_idx = torch.cat(diverse_chunks)
    else:
        diverse_idx = torch.empty(0, dtype=torch.long, device=device)

    kept_mask = torch.zeros(num_image_tokens, dtype=torch.bool, device=device)
    kept_mask[image_pivot_idx] = True
    kept_mask[diverse_idx] = True
    assert int(kept_mask.sum()) == budget, (
        f"DART kept {int(kept_mask.sum())} tokens, budget is {budget}"
    )
    return image_pivot_idx, text_pivot_rel, diverse_idx, kept_mask, pivot_similarity


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


def hiprune_pp_select(
    shallow_scores: torch.Tensor,
    deep_scores: torch.Tensor,
    embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    num_tokens: int,
    grid_w: int,
    pruning_ratio: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """HiPrune++: HiPrune plus text-guided prompt tokens (Algorithm 1).

    Runs the exact HiPrune anchor/buffer/register selection
    (:func:`hiprune_select`), then additionally keeps the
    ``text_token_count`` (see :func:`compute_hiprune_pp_budget`) tokens
    whose LM-space embeddings are most cosine-similar to the prompt's
    mean text embedding, excluding tokens the base method already kept.
    Keeps exactly ``base_budget + text_token_count`` tokens.

    One deliberate deviation from the pseudo-code: Algorithm 1 masks
    already-chosen tokens by subtracting 1 from their similarity, which
    guarantees exclusion for positive attention scores but *not* for
    cosine similarity in ``[-1, 1]`` (a masked token at 0.9 would still
    beat an unmasked one at -0.2). We mask with ``-inf`` instead, which
    is the unambiguous intent.

    Args:
        shallow_scores: ``(num_tokens,)`` soft-token scores from the
            object (middle) encoder layer.
        deep_scores: ``(num_tokens,)`` soft-token scores from the last
            encoder layer.
        embeddings: ``(num_tokens, hidden)`` visual embeddings in the
            same (raster) token order as ``shallow_scores`` — the
            tensors the language model would consume. Any float dtype;
            similarity math runs in float32.
        text_embedding: ``(hidden,)`` mean of the prompt's text-token
            embeddings from the language model's embedding table (need
            not be pre-normalized).
        num_tokens: total soft tokens for the image.
        grid_w: soft-token grid width (for spatial buffer neighbors).
        pruning_ratio: retention ratio in ``(0, 1]``.
        alpha: fraction of the base budget allotted to anchors.
        beta: text-guidance proportion; ``None`` reads ``HIPRUNE_PP_BETA``
            from the environment (paper default 0.1).

    Returns:
        ``(anchor_idx, buffer_idx, register_idx, prompt_idx, kept_mask,
        text_similarity)`` where ``text_similarity`` is the
        ``(num_tokens,)`` float32 cosine similarity of every token to
        the mean text embedding (for metadata/tooltips).
    """
    anchor_idx, buffer_idx, register_idx, kept_mask = hiprune_select(
        shallow_scores, deep_scores, num_tokens, grid_w, pruning_ratio, alpha=alpha
    )

    _, t_sum = compute_hiprune_pp_budget(num_tokens, pruning_ratio, beta)

    emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    text = torch.nn.functional.normalize(text_embedding.float(), dim=-1)
    text_similarity = emb @ text

    if t_sum > 0:
        masked = text_similarity.clone()
        masked[kept_mask] = float("-inf")
        prompt_idx = torch.topk(masked, k=t_sum).indices
        kept_mask = kept_mask.clone()
        kept_mask[prompt_idx] = True
    else:
        prompt_idx = torch.empty(
            0, dtype=anchor_idx.dtype, device=kept_mask.device
        )

    return (
        anchor_idx,
        buffer_idx,
        register_idx,
        prompt_idx,
        kept_mask,
        text_similarity,
    )


def hydart_select(
    shallow_scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_tokens: int,
    grid_w: int,
    pruning_ratio: float,
    alpha: float = DEFAULT_ALPHA,
    lambda_seed: float = DEFAULT_HYDART_LAMBDA_SEED,
    lambda_pick: float = DEFAULT_HYDART_LAMBDA_PICK,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """HyDART: HiPrune anchors/buffers + greedy-MMR diverse fill.

    Anchors and buffers use the exact HiPrune arithmetic (see
    :func:`hiprune_select`); the remaining budget is filled one pick (or
    one block of picks) at a time, maximizing

        attn_hat - lambda_seed * r_seed - lambda_pick * r_pick

    where ``attn_hat`` is the object-layer attention min-max normalized to
    [0, 1], ``r_seed`` is each token's max cosine similarity to the
    anchor+buffer seed set (fixed), and ``r_pick`` is its max cosine
    similarity to the diverse tokens picked so far (updated after every
    pick). Similarities are clamped to [0, 1] so negative similarity is
    never a bonus. No deep-layer scores are needed, which removes the
    second dense attention capture HiPrune requires.

    Keeps exactly ``compute_retained_tokens_count(num_tokens,
    pruning_ratio)`` tokens, the same invariant as :func:`hiprune_select`
    (guaranteed by an assert on the seed-set size).

    Args:
        shallow_scores: ``(num_tokens,)`` soft-token scores from the
            object (middle) encoder layer.
        embeddings: ``(num_tokens, hidden)`` merged visual embeddings in
            the same (raster) token order as ``shallow_scores`` — the
            tensors the language model would consume. Any float dtype;
            similarity math runs in float32.
        num_tokens: total soft tokens for the image.
        grid_w: soft-token grid width (for spatial buffer neighbors).
        pruning_ratio: retention ratio in ``(0, 1]``.
        alpha: fraction of the budget allotted to anchors.
        lambda_seed: penalty weight for similarity to anchors+buffers.
        lambda_pick: penalty weight for similarity to prior diverse picks.
        block_size: picks per greedy round. ``None`` picks 1 at a time up
            to ``HYDART_BLOCK_THRESHOLD`` tokens and ``HYDART_BLOCK_SIZE``
            beyond, bounding sequential GPU steps on large images. Within
            a block, similarities are not updated (slightly coarser
            selection, same invariants).

    Returns:
        ``(anchor_idx, buffer_idx, diverse_idx, kept_mask, sim_stats)``.
        ``sim_stats[i]`` is: for diverse tokens, redundancy at selection
        time (low = the pick added novel content); for pruned tokens, the
        final max cosine similarity to the full kept set (high = well
        covered); 1.0 for anchors/buffers.
    """
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

    assert shallow_all.shape[0] <= budget, (
        f"anchors+buffers ({shallow_all.shape[0]}) exceed the keep budget "
        f"({budget}); lower alpha or raise the retention ratio"
    )

    if block_size is None:
        block_size = 1 if num_tokens <= HYDART_BLOCK_THRESHOLD else HYDART_BLOCK_SIZE

    emb = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    attn = shallow_scores.float()
    attn_hat = (attn - attn.min()) / (attn.max() - attn.min()).clamp_min(1e-12)

    if shallow_all.numel():
        r_seed = (emb @ emb[shallow_all].T).max(dim=-1).values.clamp_(0.0, 1.0)
    else:
        # Zero anchors (tiny budget * alpha): the whole budget goes to
        # diverse picks, mirroring hiprune_select's behavior where topk(0)
        # hands the full budget to registers.
        r_seed = torch.zeros(num_tokens, device=emb.device)
    r_pick = torch.zeros_like(r_seed)
    base_score = attn_hat - lambda_seed * r_seed

    blocked = torch.zeros(num_tokens, dtype=torch.bool, device=emb.device)
    blocked[shallow_all] = True

    diverse_num = budget - int(shallow_all.shape[0])
    pick_chunks: list[torch.Tensor] = []
    redundancy_chunks: list[torch.Tensor] = []
    remaining = diverse_num
    while remaining > 0:
        k = min(block_size, remaining)
        score = base_score - lambda_pick * r_pick
        score[blocked] = float("-inf")
        idx = torch.topk(score, k=k).indices
        pick_chunks.append(idx)
        # Redundancy at selection time (block granularity).
        redundancy_chunks.append(torch.maximum(r_seed[idx], r_pick[idx]))
        blocked[idx] = True
        sim_new = (emb @ emb[idx].T).max(dim=-1).values.clamp_(0.0, 1.0)
        r_pick = torch.maximum(r_pick, sim_new)
        remaining -= k
    if pick_chunks:
        diverse_idx = torch.cat(pick_chunks)
        diverse_redundancy = torch.cat(redundancy_chunks)
    else:
        diverse_idx = torch.empty(0, dtype=anchor_idx.dtype, device=emb.device)
        diverse_redundancy = torch.empty(0, device=emb.device)

    kept_mask = torch.zeros(num_tokens, dtype=torch.bool, device=emb.device)
    kept_mask[shallow_all] = True
    kept_mask[diverse_idx] = True

    sim_stats = torch.maximum(r_seed, r_pick)  # final max sim to kept set
    sim_stats[shallow_all] = 1.0
    if diverse_idx.numel():
        sim_stats[diverse_idx] = diverse_redundancy
    return anchor_idx, buffer_idx, diverse_idx, kept_mask, sim_stats


def build_hydart_metadata(
    anchor_idx: torch.Tensor,
    buffer_idx: torch.Tensor,
    diverse_idx: torch.Tensor,
    kept_mask: torch.Tensor,
    shallow_scores: torch.Tensor,
    sim_stats: torch.Tensor,
    grid_w: int,
    grid_h: int,
    retention: float,
    object_layer: int,
    alpha: float = DEFAULT_ALPHA,
    lambda_seed: float = DEFAULT_HYDART_LAMBDA_SEED,
    lambda_pick: float = DEFAULT_HYDART_LAMBDA_PICK,
) -> dict[str, object]:
    """JSON-safe per-image HyDART metadata for API reporting.

    Mirrors :func:`build_hiprune_metadata`, with the register category
    replaced by ``diverse`` and deep-layer attention replaced by cosine
    similarity statistics: diverse tokens report their redundancy at
    selection time (low = novel content), pruned tokens their final max
    similarity to the kept set (high = well covered by kept tokens).
    """

    def _mean(scores: torch.Tensor, idx: torch.Tensor) -> float | None:
        return float(scores[idx].mean()) if idx.numel() else None

    pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    categories = {
        "anchor": anchor_idx,
        "buffer": buffer_idx,
        "diverse": diverse_idx,
        "kept": kept_idx,
        "pruned": pruned_idx,
    }
    return {
        "method": "hydart",
        "grid": [grid_w, grid_h],
        "num_tokens": int(kept_mask.shape[0]),
        "retention": retention,
        "object_layer": object_layer,
        "alpha": alpha,
        "lambda_seed": lambda_seed,
        "lambda_pick": lambda_pick,
        "pruned": pruned_idx.tolist(),
        "anchors": anchor_idx.tolist(),
        "buffers": buffer_idx.tolist(),
        "diverse": diverse_idx.tolist(),
        "mean_attention": {
            "object_layer": {
                name: _mean(shallow_scores, idx) for name, idx in categories.items()
            },
        },
        "similarity": {
            "diverse_at_selection": _mean(sim_stats, diverse_idx),
            "pruned_vs_kept": _mean(sim_stats, pruned_idx),
        },
        # Per-token arrays for hover tooltips (index = soft-token index).
        "scores": {
            "object_layer": shallow_scores.float().tolist(),
            "similarity": sim_stats.float().tolist(),
        },
    }


def build_hiprune_pp_metadata(
    anchor_idx: torch.Tensor,
    buffer_idx: torch.Tensor,
    register_idx: torch.Tensor,
    prompt_idx: torch.Tensor,
    kept_mask: torch.Tensor,
    shallow_scores: torch.Tensor,
    deep_scores: torch.Tensor,
    text_similarity: torch.Tensor,
    grid_w: int,
    grid_h: int,
    retention: float,
    object_layer: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float | None = None,
) -> dict[str, object]:
    """JSON-safe per-image HiPrune++ metadata for API reporting.

    Extends :func:`build_hiprune_metadata` with the ``prompt`` token
    category (text-guided picks) and per-token cosine similarity to the
    prompt's mean text embedding, for hover tooltips.
    """
    if beta is None:
        beta = get_hiprune_pp_beta()

    def _mean(scores: torch.Tensor, idx: torch.Tensor) -> float | None:
        return float(scores[idx].mean()) if idx.numel() else None

    pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    categories = {
        "anchor": anchor_idx,
        "buffer": buffer_idx,
        "register": register_idx,
        "prompt": prompt_idx,
        "kept": kept_idx,
        "pruned": pruned_idx,
    }
    return {
        "method": "hiprune_pp",
        "grid": [grid_w, grid_h],
        "num_tokens": int(kept_mask.shape[0]),
        "retention": retention,
        "object_layer": object_layer,
        "alpha": alpha,
        "beta": beta,
        "pruned": pruned_idx.tolist(),
        "anchors": anchor_idx.tolist(),
        "buffers": buffer_idx.tolist(),
        "registers": register_idx.tolist(),
        "prompt_tokens": prompt_idx.tolist(),
        "mean_attention": {
            "object_layer": {
                name: _mean(shallow_scores, idx) for name, idx in categories.items()
            },
            "deep_layer": {
                name: _mean(deep_scores, idx) for name, idx in categories.items()
            },
        },
        "text_similarity_summary": {
            "prompt": _mean(text_similarity, prompt_idx),
            "kept": _mean(text_similarity, kept_idx),
            "pruned": _mean(text_similarity, pruned_idx),
        },
        # Per-token arrays for hover tooltips (index = soft-token index).
        "scores": {
            "object_layer": shallow_scores.float().tolist(),
            "deep_layer": deep_scores.float().tolist(),
            "text_similarity": text_similarity.float().tolist(),
        },
    }


def build_dart_metadata(
    image_pivot_idx: torch.Tensor,
    diverse_idx: torch.Tensor,
    kept_mask: torch.Tensor,
    key_l1_norms: torch.Tensor,
    pivot_similarity: torch.Tensor,
    grid_w: int,
    grid_h: int,
    retention: float,
    pivot_image: int,
    pivot_text: int,
    num_text_pivots: int,
    dart_layer: int,
) -> dict[str, object]:
    """JSON-safe per-image DART metadata for API reporting.

    Categories: ``pivot`` (image pivots, kept) and ``diverse``
    (anti-duplication picks). Per-token score arrays for tooltips: the
    layer-(K−1) attention-key L1 norm (the pivot-selection statistic)
    and each token's max cosine similarity to the pivots (high =
    duplicated by a pivot — likely pruned; low = novel — likely kept).
    ``num_text_pivots`` reports how many text pivots actually guided
    selection (0 when the prompt was missing/empty).
    """

    def _mean(scores: torch.Tensor, idx: torch.Tensor) -> float | None:
        return float(scores[idx].mean()) if idx.numel() else None

    num_tokens = int(kept_mask.shape[0])
    image_key_norms = key_l1_norms.float()[:num_tokens]
    pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    categories = {
        "pivot": image_pivot_idx,
        "diverse": diverse_idx,
        "kept": kept_idx,
        "pruned": pruned_idx,
    }
    return {
        "method": "dart",
        "grid": [grid_w, grid_h],
        "num_tokens": num_tokens,
        "retention": retention,
        "pivot_image": pivot_image,
        "pivot_text": pivot_text,
        "num_text_pivots": num_text_pivots,
        "dart_layer": dart_layer,
        "pruned": pruned_idx.tolist(),
        "pivots": image_pivot_idx.tolist(),
        "diverse": diverse_idx.tolist(),
        "key_norm_summary": {
            name: _mean(image_key_norms, idx) for name, idx in categories.items()
        },
        "similarity": {
            "kept_vs_pivots": _mean(pivot_similarity, kept_idx),
            "pruned_vs_pivots": _mean(pivot_similarity, pruned_idx),
        },
        # Per-token arrays for hover tooltips (index = soft-token index).
        "scores": {
            "key_norm": image_key_norms.tolist(),
            "pivot_similarity": pivot_similarity.float().tolist(),
        },
    }


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
        "method": "hiprune",
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
        # Per-token arrays for hover tooltips (index = soft-token index).
        "scores": {
            "object_layer": shallow_scores.float().tolist(),
            "deep_layer": deep_scores.float().tolist(),
        },
    }
