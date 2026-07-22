# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiPrune visual token selection (vllm/multimodal/hiprune.py)."""

import math

import pytest
import torch

from vllm.multimodal.hiprune import (
    LLAVA_OBJECT_LAYER,
    QWEN2_5_VL_OBJECT_LAYER,
    aggregate_patch_attention,
    build_dart_metadata,
    build_hiprune_metadata,
    build_hiprune_pp_metadata,
    build_hydart_metadata,
    compute_hiprune_pp_budget,
    compute_retained_tokens_count,
    compute_soft_token_grid,
    dart_keep_count,
    dart_select,
    fold_merged_token_scores,
    get_dart_layer,
    get_dart_pivots,
    get_hiprune_method,
    get_hiprune_pp_beta,
    get_hiprune_prompt,
    get_hiprune_ratio,
    get_hydart_lambdas,
    get_nprune_stride,
    hiprune_pp_select,
    hiprune_select,
    hydart_select,
    nprune_keep_count,
    nprune_select,
    build_nprune_metadata,
    checkered_keep_count,
    checkered_select,
    build_checkered_metadata,
    pack_hiprune_config,
    unpack_hiprune_config,
    HIPRUNE_CONFIG_WIDTH,
    HIPRUNE_METHOD_IDS,
    HIPRUNE_MM_KWARG_KEYS,
)

POOL_K = 3
PATCH_BUDGET = 2520  # Gemma4: max_soft_tokens (280) * pooling_kernel_size**2


def _make_position_ids(patch_w: int, patch_h: int) -> torch.Tensor:
    """Replicate Gemma4ImageProcessor position ids: real patches in
    row-major (x, y) order, padded to the fixed budget with (-1, -1)."""
    grid = torch.stack(
        torch.meshgrid(torch.arange(patch_w), torch.arange(patch_h), indexing="xy"),
        dim=-1,
    ).reshape(patch_w * patch_h, 2)
    pad = torch.full((PATCH_BUDGET - patch_w * patch_h, 2), -1)
    return torch.cat([grid, pad])


@pytest.mark.parametrize("num_tokens", [1, 12, 255, 260, 280])
@pytest.mark.parametrize("ratio", [0.05, 0.11, 0.14, 0.223, 0.5, 1.0])
def test_retained_count_bounds(num_tokens: int, ratio: float):
    kept = compute_retained_tokens_count(num_tokens, ratio)
    assert 1 <= kept <= num_tokens


@pytest.mark.parametrize("patch_w,patch_h", [(12, 9), (45, 39), (15, 51)])
def test_soft_token_grid(patch_w: int, patch_h: int):
    pos = _make_position_ids(patch_w, patch_h)
    valid, grid_w, grid_h, kernel_idx = compute_soft_token_grid(pos, POOL_K)

    assert valid.sum() == patch_w * patch_h
    assert (grid_w, grid_h) == (patch_w // POOL_K, patch_h // POOL_K)

    n_tokens = grid_w * grid_h
    assert kernel_idx.min() >= 0
    assert kernel_idx.max() == n_tokens - 1
    # Every soft token covers exactly k^2 patches.
    counts = torch.bincount(kernel_idx, minlength=n_tokens)
    assert (counts == POOL_K**2).all()


def test_aggregation_ignores_padding_rows():
    """Garbage attention rows at padding positions must not affect scores."""
    patch_w, patch_h = 12, 9
    n_real = patch_w * patch_h
    pos = _make_position_ids(patch_w, patch_h)
    valid, grid_w, grid_h, kernel_idx = compute_soft_token_grid(pos, POOL_K)
    n_tokens = grid_w * grid_h

    heads = 4
    attn = torch.zeros(heads, PATCH_BUDGET, PATCH_BUDGET)
    real_attn = torch.randn(heads, n_real, n_real).softmax(dim=-1)
    attn[:, :n_real, :n_real] = real_attn
    # Garbage rows for padding queries.
    attn[:, n_real:, :] = torch.rand(
        heads, PATCH_BUDGET - n_real, PATCH_BUDGET
    ).softmax(dim=-1)

    scores = aggregate_patch_attention(attn, valid, kernel_idx, n_tokens)
    assert scores.shape == (n_tokens,)
    # Still a probability distribution over soft tokens.
    assert torch.isclose(scores.sum(), torch.tensor(1.0), atol=1e-5)

    # Identical result when computed from the unpadded attention alone.
    ref = real_attn.float().mean(dim=0).mean(dim=0)
    ref_scores = (
        torch.nn.functional.one_hot(kernel_idx.long(), n_tokens).float().T @ ref
    )
    assert torch.allclose(scores, ref_scores, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("ratio", [0.11, 0.14, 0.223, 0.5])
def test_selection_invariants(seed: int, ratio: float):
    torch.manual_seed(seed)
    grid_w, grid_h = 20, 13  # a realistic Gemma4 soft-token grid
    n_tokens = grid_w * grid_h
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)

    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, n_tokens, grid_w, ratio
    )

    # Exact budget: the count the processor promised via placeholders.
    assert kept.sum().item() == compute_retained_tokens_count(n_tokens, ratio)

    # Categories are disjoint and jointly equal the kept set.
    all_idx = torch.cat([anchor, buffer, register])
    assert all_idx.unique().numel() == all_idx.numel()
    mask_from_cats = torch.zeros(n_tokens, dtype=torch.bool)
    mask_from_cats[all_idx] = True
    assert torch.equal(mask_from_cats, kept)

    # Buffers are spatial neighbors of anchors.
    if buffer.numel() > 0:
        neighbor_sets = torch.cat(
            [anchor - 1, anchor + 1, anchor - grid_w, anchor + grid_w]
        ).clamp(0, n_tokens - 1)
        assert torch.isin(buffer, neighbor_sets).all()


def test_selection_deterministic():
    torch.manual_seed(7)
    n_tokens, grid_w = 260, 20
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    first = hiprune_select(shallow, deep, n_tokens, grid_w, 0.14)
    second = hiprune_select(shallow, deep, n_tokens, grid_w, 0.14)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_full_retention_keeps_everything():
    torch.manual_seed(3)
    n_tokens, grid_w = 255, 15
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    _, _, _, kept = hiprune_select(shallow, deep, n_tokens, grid_w, 1.0)
    assert kept.all()


def test_metadata_contents():
    """Metadata must be JSON-safe and consistent with the selection."""
    import json

    torch.manual_seed(11)
    grid_w, grid_h = 15, 18
    n_tokens = grid_w * grid_h
    ratio = 0.14
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)

    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, n_tokens, grid_w, ratio
    )
    md = build_hiprune_metadata(
        anchor, buffer, register, kept, shallow, deep, grid_w, grid_h, ratio
    )

    # JSON-serializable (crosses the engine-core process boundary).
    json.dumps(md)

    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == n_tokens
    assert md["retention"] == ratio

    # Index sets match the selection exactly.
    assert md["anchors"] == anchor.tolist()
    assert md["buffers"] == buffer.tolist()
    assert md["registers"] == register.tolist()
    assert sorted(md["pruned"]) == (~kept).nonzero(as_tuple=True)[0].tolist()
    kept_count = compute_retained_tokens_count(n_tokens, ratio)
    assert len(md["pruned"]) == n_tokens - kept_count

    # Mean attentions match direct computation.
    ma = md["mean_attention"]
    assert ma["object_layer"]["anchor"] == pytest.approx(
        float(shallow[anchor].mean())
    )
    assert ma["deep_layer"]["register"] == pytest.approx(
        float(deep[register].mean())
    )
    # Anchors are the top of the object-layer distribution by construction.
    assert ma["object_layer"]["anchor"] > ma["object_layer"]["pruned"]


# --------------------------------------------------------------------------
# Qwen2.5-VL support
# --------------------------------------------------------------------------

MERGE_UNIT = 4  # Qwen2.5-VL spatial_merge_size**2


def test_metadata_object_layer_parameterized():
    torch.manual_seed(2)
    grid_w, grid_h = 8, 6
    n_tokens = grid_w * grid_h
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, n_tokens, grid_w, 0.25
    )
    md = build_hiprune_metadata(
        anchor,
        buffer,
        register,
        kept,
        shallow,
        deep,
        grid_w,
        grid_h,
        0.25,
        object_layer=QWEN2_5_VL_OBJECT_LAYER,
    )
    assert md["object_layer"] == QWEN2_5_VL_OBJECT_LAYER


@pytest.mark.parametrize("val,expected", [(None, None), (1.0, None), (0.14, 0.14)])
def test_get_hiprune_ratio_passthrough(val, expected):
    got = get_hiprune_ratio({"hiprune_ratio": val})
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
        # float32-quantized so processor and model round identically.
        assert got == float(torch.tensor(val, dtype=torch.float32).item())


@pytest.mark.parametrize("val", [0.0, -0.1, 1.5])
def test_get_hiprune_ratio_rejects_out_of_range(val):
    with pytest.raises(ValueError):
        get_hiprune_ratio({"hiprune_ratio": val})


def _random_window_permutation(num_units: int, generator: torch.Generator):
    window_index = torch.randperm(num_units, generator=generator)
    reverse_indices = torch.argsort(window_index)
    return window_index, reverse_indices


@pytest.mark.parametrize("seed", range(5))
def test_fold_merged_token_scores_unpermutes(seed: int):
    """Folding + un-permutation must recover raster-order token scores."""
    g = torch.Generator().manual_seed(seed)
    num_units = 60
    window_index, reverse_indices = _random_window_permutation(num_units, g)

    # Ground truth: one score per merged token in raster order.
    unit_truth = torch.rand(num_units, generator=g)
    # Patch scores in permuted order: each unit's 4 patches average to
    # the unit's truth value.
    noise = torch.rand(num_units, MERGE_UNIT, generator=g)
    noise = noise - noise.mean(dim=-1, keepdim=True)
    patch_scores = (unit_truth[window_index, None] + noise).reshape(-1)

    folded = fold_merged_token_scores(patch_scores, MERGE_UNIT, reverse_indices)
    assert torch.allclose(folded, unit_truth, atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ratio", [0.11, 0.25, 0.5])
def test_qwen_pipeline_matches_reference(seed: int, ratio: float):
    """Differential test against a direct transcription of the authors'
    Qwen2.5-VL reference: given the same per-layer attention matrices
    (in window-permuted patch order), our fold + select pipeline must
    keep exactly the same token set, split into the same categories.
    """
    g = torch.Generator().manual_seed(seed)
    grid_w, grid_h = 12, 9
    n_tokens = grid_w * grid_h
    n_patches = n_tokens * MERGE_UNIT
    num_heads = 4

    window_index, reverse_indices = _random_window_permutation(n_tokens, g)

    def rand_attn():
        # Dense unmasked post-softmax attention over permuted patches.
        return torch.rand(
            num_heads, n_patches, n_patches, generator=g
        ).softmax(dim=-1)

    attn_shallow = rand_attn()
    attn_deep = rand_attn()

    # ---- Reference transcription (modeling_qwen2_5_vl.py, HiPrune fork) --
    def ref_aggregate(attn):
        w = attn.mean(dim=0)  # average all heads
        w = w.mean(dim=0)  # average all queries -> per-key score
        w = w.view(w.shape[0] // MERGE_UNIT, -1).mean(dim=-1)
        return w[reverse_indices]

    ref_shallow = ref_aggregate(attn_shallow)
    ref_deep = ref_aggregate(attn_deep).clone()

    visual_token_num = round(n_tokens * ratio)
    shallow_token_num = round((visual_token_num * 0.1) / 5)
    shallow_idx = torch.topk(ref_shallow, k=shallow_token_num).indices
    shallow_all = torch.cat(
        [
            shallow_idx,
            shallow_idx - 1,
            shallow_idx + 1,
            shallow_idx - grid_w,
            shallow_idx + grid_w,
        ]
    ).clamp(0, n_tokens - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    deep_token_num = visual_token_num - shallow_all.shape[0]
    avail = torch.zeros(n_tokens, dtype=torch.bool)
    avail.scatter_(0, shallow_all, 1)
    ref_deep -= avail.int()
    deep_idx = torch.topk(ref_deep, k=deep_token_num).indices
    ref_mask = torch.zeros(n_tokens, dtype=torch.bool)
    ref_mask[torch.cat([shallow_all, deep_idx])] = True
    # ---------------------------------------------------------------------

    # Our pipeline: per-key scores (mean over heads and queries) fed
    # through fold_merged_token_scores, then hiprune_select.
    def our_scores(attn):
        per_key = attn.mean(dim=0).mean(dim=0)
        return fold_merged_token_scores(per_key, MERGE_UNIT, reverse_indices)

    shallow = our_scores(attn_shallow)
    deep = our_scores(attn_deep)
    assert torch.allclose(shallow, ref_aggregate(attn_shallow), atol=1e-7)

    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, n_tokens, grid_w, ratio
    )
    assert torch.equal(kept, ref_mask)
    # Anchor/buffer split partitions the reference's shallow set.
    assert set(torch.cat([anchor, buffer]).tolist()) == set(shallow_all.tolist())
    assert set(register.tolist()) == set(deep_idx.tolist())


@pytest.mark.parametrize("keep_every", [2, 3])
def test_recompute_mrope_keeps_original_positions_for_kept_tokens(keep_every):
    """After pruning, each kept image token must carry its ORIGINAL
    spatial mrope position (the EVS/HiPrune position semantics), and
    trailing text must resume after the image's position span.
    """
    from vllm.multimodal.evs import (
        compute_mrope_for_media,
        recompute_mrope_positions,
    )

    vision_start_id, image_id, video_id, text_id = 90, 91, 92, 1
    merge_size = 2
    size = (1, 8, 12)  # grid_thw -> 4x6 = 24 merged tokens
    n_tokens = (size[1] // merge_size) * (size[2] // merge_size)

    positions_full = compute_mrope_for_media(
        torch.tensor(size), merge_size
    )  # (n_tokens, 4)
    kept_mask = torch.zeros(n_tokens, dtype=torch.bool)
    kept_mask[::keep_every] = True
    mm_pos = positions_full[kept_mask].permute(1, 0).long()  # (4, kept)
    n_kept = int(kept_mask.sum())

    prefix_len, suffix_len = 5, 7
    input_ids = torch.tensor(
        [text_id] * (prefix_len - 1)
        + [vision_start_id]
        + [image_id] * n_kept
        + [text_id] * suffix_len
    )
    total = input_ids.numel()
    # Seed positions: plain text numbering (what the runner starts from).
    seed_positions = (
        torch.arange(total).view(1, -1).expand(3, -1).clone().long()
    )

    positions, _ = recompute_mrope_positions(
        input_ids,
        [mm_pos],
        seed_positions,
        num_computed_tokens=0,
        vision_start_token_id=vision_start_id,
        image_token_id=image_id,
        video_token_id=video_id,
    )

    base = prefix_len  # text positions 0..prefix_len-1, image starts after
    img_slice = positions[:, prefix_len : prefix_len + n_kept]
    expected = mm_pos[0:3] + base
    assert torch.equal(img_slice, expected)

    # Kept tokens' (h, w) equal their original raster coordinates.
    kept_idx = kept_mask.nonzero(as_tuple=True)[0]
    grid_w_merged = size[2] // merge_size
    assert torch.equal(
        img_slice[1] - base, (kept_idx // grid_w_merged).long()
    )
    assert torch.equal(
        img_slice[2] - base, (kept_idx % grid_w_merged).long()
    )

    # Trailing text resumes after the image's position span.
    text_start = positions[:, prefix_len + n_kept]
    expected_text_start = int(mm_pos[3, 0]) + base
    assert (text_start == expected_text_start).all()


def test_recompute_mrope_chunk_boundary_after_vision_start():
    """Chunked prefill can split a request exactly after the vision_start
    token, before any media token has been computed. The recompute must
    treat this as the start of the current media segment instead of
    searching for a (nonexistent) later vision_start and crashing.
    """
    from vllm.multimodal.evs import (
        compute_mrope_for_media,
        recompute_mrope_positions,
    )

    vision_start_id, image_id, video_id, text_id = 90, 91, 92, 1
    merge_size = 2
    size = (1, 8, 12)
    n_tokens = (size[1] // merge_size) * (size[2] // merge_size)

    mm_pos = (
        compute_mrope_for_media(torch.tensor(size), merge_size)
        .permute(1, 0)
        .long()
    )

    prefix_len, suffix_len = 5, 7
    input_ids = torch.tensor(
        [text_id] * (prefix_len - 1)
        + [vision_start_id]
        + [image_id] * n_tokens
        + [text_id] * suffix_len
    )
    total = input_ids.numel()
    seed_positions = torch.arange(total).view(1, -1).expand(3, -1).clone().long()

    # First chunk ended exactly at the vision_start boundary: prefix_len
    # tokens computed (the last one being vision_start), zero media tokens.
    positions, _ = recompute_mrope_positions(
        input_ids,
        [mm_pos],
        seed_positions,
        num_computed_tokens=prefix_len,
        vision_start_token_id=vision_start_id,
        image_token_id=image_id,
        video_token_id=video_id,
    )

    # Same result as computing in one shot from the beginning.
    expected, _ = recompute_mrope_positions(
        input_ids,
        [mm_pos],
        seed_positions.clone(),
        num_computed_tokens=0,
        vision_start_token_id=vision_start_id,
        image_token_id=image_id,
        video_token_id=video_id,
    )
    assert torch.equal(positions, expected)


# --------------------------------------------------------------------------
# LLaVA-1.5 support (CLIP tower: 24x24 grid + CLS token)
# --------------------------------------------------------------------------

LLAVA_GRID = 24
LLAVA_N_TOKENS = LLAVA_GRID * LLAVA_GRID  # 576
LLAVA_SEQ_LEN = LLAVA_N_TOKENS + 1  # + CLS at index 0


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ratio", [0.111, 0.223, 0.5])
def test_llava_pipeline_matches_reference(seed: int, ratio: float):
    """Differential test against the transformers-wrapper reference
    (llava_server.py / the authors' llava_arch.py): given the same
    post-softmax attention matrices over the full CLIP sequence (CLS at
    index 0), per-key aggregation + CLS-key stripping + selection must
    keep exactly the same token set, split into the same categories.
    """
    g = torch.Generator().manual_seed(seed)
    num_heads = 4

    def rand_attn():
        return torch.rand(
            num_heads, LLAVA_SEQ_LEN, LLAVA_SEQ_LEN, generator=g
        ).softmax(dim=-1)

    attn_shallow = rand_attn()
    attn_deep = rand_attn()

    # ---- Reference (llava_server.py): mean heads, mean queries, drop CLS
    ref_shallow = attn_shallow.mean(dim=0).mean(dim=0)[1:]
    ref_deep = attn_deep.mean(dim=0).mean(dim=0)[1:].clone()

    budget = compute_retained_tokens_count(LLAVA_N_TOKENS, ratio)
    shallow_token_num = round(budget * 0.1 / 5)
    anchor_ref = torch.topk(ref_shallow, k=shallow_token_num).indices
    shallow_all = torch.cat(
        [
            anchor_ref,
            anchor_ref - 1,
            anchor_ref + 1,
            anchor_ref - LLAVA_GRID,
            anchor_ref + LLAVA_GRID,
        ]
    ).clamp(0, LLAVA_N_TOKENS - 1)
    shallow_all = torch.unique(shallow_all, sorted=False)
    deep_token_num = budget - shallow_all.shape[0]
    avail = torch.zeros(LLAVA_N_TOKENS, dtype=torch.bool)
    avail.scatter_(0, shallow_all, 1)
    ref_deep -= avail.int()
    deep_idx = torch.topk(ref_deep, k=deep_token_num).indices
    ref_mask = torch.zeros(LLAVA_N_TOKENS, dtype=torch.bool)
    ref_mask[torch.cat([shallow_all, deep_idx])] = True
    # ---------------------------------------------------------------------

    # Our pipeline: per-key scores over the FULL sequence (what
    # compute_hiprune_key_scores returns), CLS key stripped by the model,
    # then hiprune_select.
    shallow = attn_shallow.mean(dim=0).mean(dim=0)[1:]
    deep = attn_deep.mean(dim=0).mean(dim=0)[1:]

    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, LLAVA_N_TOKENS, LLAVA_GRID, ratio
    )
    assert torch.equal(kept, ref_mask)
    assert set(torch.cat([anchor, buffer]).tolist()) == set(shallow_all.tolist())
    assert set(register.tolist()) == set(deep_idx.tolist())


@pytest.mark.parametrize("ratio", [0.111, 0.223])
def test_llava_budget_matches_placeholders(ratio: float):
    """The model keeps exactly the count the processor promised via the
    shrunken placeholder run (576-token LLaVA grid)."""
    torch.manual_seed(0)
    shallow = torch.rand(LLAVA_N_TOKENS).softmax(dim=0)
    deep = torch.rand(LLAVA_N_TOKENS).softmax(dim=0)
    _, _, _, kept = hiprune_select(shallow, deep, LLAVA_N_TOKENS, LLAVA_GRID, ratio)
    assert kept.sum().item() == compute_retained_tokens_count(LLAVA_N_TOKENS, ratio)

    emb = torch.randn(LLAVA_N_TOKENS, 32)
    _, _, _, kept_hd, _ = hydart_select(
        shallow, emb, LLAVA_N_TOKENS, LLAVA_GRID, ratio
    )
    assert kept_hd.sum().item() == compute_retained_tokens_count(
        LLAVA_N_TOKENS, ratio
    )


def test_llava_metadata_object_layer():
    torch.manual_seed(6)
    shallow = torch.rand(LLAVA_N_TOKENS).softmax(dim=0)
    deep = torch.rand(LLAVA_N_TOKENS).softmax(dim=0)
    anchor, buffer, register, kept = hiprune_select(
        shallow, deep, LLAVA_N_TOKENS, LLAVA_GRID, 0.223
    )
    md = build_hiprune_metadata(
        anchor,
        buffer,
        register,
        kept,
        shallow,
        deep,
        LLAVA_GRID,
        LLAVA_GRID,
        0.223,
        object_layer=LLAVA_OBJECT_LAYER,
    )
    assert md["object_layer"] == LLAVA_OBJECT_LAYER
    assert md["grid"] == [LLAVA_GRID, LLAVA_GRID]
    assert md["num_tokens"] == LLAVA_N_TOKENS


# --------------------------------------------------------------------------
# HyDART (HIPRUNE_METHOD=hydart)
# --------------------------------------------------------------------------


def _clustered_embeddings(
    n_tokens: int, dim: int, n_clusters: int, generator: torch.Generator
) -> torch.Tensor:
    """Embeddings with heavy duplication: tokens are noisy copies of a few
    cluster centers, the regime HyDART's diversity stage targets."""
    centers = torch.randn(n_clusters, dim, generator=generator)
    assignment = torch.randint(0, n_clusters, (n_tokens,), generator=generator)
    noise = 0.05 * torch.randn(n_tokens, dim, generator=generator)
    return centers[assignment] + noise


def _mean_pairwise_sim(emb: torch.Tensor, idx: torch.Tensor) -> float:
    e = torch.nn.functional.normalize(emb[idx].float(), dim=-1)
    sims = e @ e.T
    off_diag = sims[~torch.eye(len(idx), dtype=torch.bool)]
    return float(off_diag.mean())


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ratio", [0.11, 0.25, 0.5])
def test_hydart_invariants(seed: int, ratio: float):
    g = torch.Generator().manual_seed(seed)
    grid_w, grid_h = 12, 9
    n_tokens = grid_w * grid_h
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 6, g)

    anchor, buffer, diverse, kept, sim_stats = hydart_select(
        shallow, emb, n_tokens, grid_w, ratio
    )

    # Exact budget: the count the processor promised via placeholders.
    assert kept.sum().item() == compute_retained_tokens_count(n_tokens, ratio)

    # Categories are disjoint and jointly equal the kept set.
    all_idx = torch.cat([anchor, buffer, diverse])
    assert all_idx.unique().numel() == all_idx.numel()
    mask_from_cats = torch.zeros(n_tokens, dtype=torch.bool)
    mask_from_cats[all_idx] = True
    assert torch.equal(mask_from_cats, kept)

    # Anchors and buffers are byte-identical to HiPrune's seed stage.
    deep_dummy = torch.rand(n_tokens, generator=g).softmax(dim=0)
    hp_anchor, hp_buffer, _, _ = hiprune_select(
        shallow, deep_dummy, n_tokens, grid_w, ratio
    )
    assert torch.equal(anchor, hp_anchor)
    assert set(buffer.tolist()) == set(hp_buffer.tolist())

    # sim_stats semantics: seeds pinned at 1.0, everything in [0, 1].
    assert (sim_stats[anchor] == 1.0).all()
    assert (sim_stats[buffer] == 1.0).all()
    assert sim_stats.min() >= 0.0 and sim_stats.max() <= 1.0


def test_hydart_lambda_zero_is_attention_topk():
    """With both penalties off, diverse picks are exactly the attention
    top-k over non-seed tokens (HiPrune's register stage but with
    object-layer scores)."""
    g = torch.Generator().manual_seed(0)
    grid_w = 12
    n_tokens = grid_w * 9
    # Distinct scores so top-k is unambiguous under ties.
    shallow = torch.randperm(n_tokens, generator=g).float() / n_tokens
    emb = _clustered_embeddings(n_tokens, 32, 6, g)

    anchor, buffer, diverse, kept, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.3, lambda_seed=0.0, lambda_pick=0.0
    )
    seed_mask = torch.zeros(n_tokens, dtype=torch.bool)
    seed_mask[torch.cat([anchor, buffer])] = True
    masked = shallow.clone()
    masked[seed_mask] = float("-inf")
    expected = torch.topk(masked, k=diverse.numel()).indices
    assert set(diverse.tolist()) == set(expected.tolist())


def test_hydart_deterministic():
    g = torch.Generator().manual_seed(7)
    n_tokens, grid_w = 260, 20
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 6, g)
    first = hydart_select(shallow, emb, n_tokens, grid_w, 0.14)
    second = hydart_select(shallow, emb, n_tokens, grid_w, 0.14)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_hydart_full_retention_keeps_everything():
    g = torch.Generator().manual_seed(3)
    n_tokens, grid_w = 255, 15
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 6, g)
    _, _, _, kept, _ = hydart_select(shallow, emb, n_tokens, grid_w, 1.0)
    assert kept.all()


def test_hydart_zero_anchor_budget():
    """A budget small enough that round(budget * alpha / 5) == 0 must fill
    the whole budget with diverse picks, not crash on an empty seed set."""
    g = torch.Generator().manual_seed(1)
    grid_w = 6
    n_tokens = grid_w * 6
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 16, 4, g)

    ratio = 0.12  # budget 4 -> shallow_token_num = round(0.08) = 0
    anchor, buffer, diverse, kept, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, ratio
    )
    assert anchor.numel() == 0 and buffer.numel() == 0
    assert diverse.numel() == kept.sum().item()
    assert kept.sum().item() == compute_retained_tokens_count(n_tokens, ratio)


def test_hydart_over_budget_seed_asserts():
    """Anchors+buffers exceeding the budget must fail loudly instead of
    silently violating the placeholder count."""
    g = torch.Generator().manual_seed(2)
    grid_w = 10
    n_tokens = grid_w * 10
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 16, 4, g)
    with pytest.raises(AssertionError, match="exceed the keep budget"):
        hydart_select(shallow, emb, n_tokens, grid_w, 0.1, alpha=5.0)


def test_hydart_lambda_pick_reduces_redundancy():
    """Raising lambda_pick must not increase the mean pairwise cosine
    similarity of the diverse set (the redundancy it penalizes)."""
    g = torch.Generator().manual_seed(4)
    grid_w = 16
    n_tokens = grid_w * 12
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 5, g)

    _, _, div_lo, _, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.25, lambda_seed=0.0, lambda_pick=0.0
    )
    _, _, div_hi, _, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.25, lambda_seed=0.0, lambda_pick=2.0
    )
    assert _mean_pairwise_sim(emb, div_hi) <= _mean_pairwise_sim(emb, div_lo) + 1e-6


@pytest.mark.parametrize("block_size", [4, 16])
def test_hydart_block_greedy_invariants(block_size: int):
    """Block picks (large-image fast path) keep every invariant: exact
    budget, disjoint categories, no duplicate picks."""
    g = torch.Generator().manual_seed(5)
    grid_w = 20
    n_tokens = grid_w * 15
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 8, g)

    anchor, buffer, diverse, kept, sim_stats = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.3, block_size=block_size
    )
    assert kept.sum().item() == compute_retained_tokens_count(n_tokens, 0.3)
    all_idx = torch.cat([anchor, buffer, diverse])
    assert all_idx.unique().numel() == all_idx.numel()

    # With lambda_pick=0 the loop is one-shot, so block picks must match
    # single picks exactly (r_pick never enters the score).
    _, _, div_blocked, _, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.3, lambda_pick=0.0, block_size=block_size
    )
    _, _, div_single, _, _ = hydart_select(
        shallow, emb, n_tokens, grid_w, 0.3, lambda_pick=0.0, block_size=1
    )
    assert set(div_blocked.tolist()) == set(div_single.tolist())


def test_hydart_metadata_contents():
    """HyDART metadata must be JSON-safe and consistent with the selection."""
    import json

    g = torch.Generator().manual_seed(11)
    grid_w, grid_h = 15, 18
    n_tokens = grid_w * grid_h
    ratio = 0.14
    shallow = torch.rand(n_tokens, generator=g).softmax(dim=0)
    emb = _clustered_embeddings(n_tokens, 32, 6, g)

    anchor, buffer, diverse, kept, sim_stats = hydart_select(
        shallow, emb, n_tokens, grid_w, ratio
    )
    md = build_hydart_metadata(
        anchor,
        buffer,
        diverse,
        kept,
        shallow,
        sim_stats,
        grid_w,
        grid_h,
        ratio,
        object_layer=QWEN2_5_VL_OBJECT_LAYER,
        lambda_seed=0.1,
        lambda_pick=0.5,
    )
    json.dumps(md)

    assert md["method"] == "hydart"
    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == n_tokens
    assert md["retention"] == ratio
    assert md["object_layer"] == QWEN2_5_VL_OBJECT_LAYER
    assert md["lambda_seed"] == 0.1 and md["lambda_pick"] == 0.5
    assert md["anchors"] == anchor.tolist()
    assert md["buffers"] == buffer.tolist()
    assert md["diverse"] == diverse.tolist()
    assert sorted(md["pruned"]) == (~kept).nonzero(as_tuple=True)[0].tolist()
    kept_count = compute_retained_tokens_count(n_tokens, ratio)
    assert len(md["pruned"]) == n_tokens - kept_count

    ma = md["mean_attention"]["object_layer"]
    assert ma["anchor"] == pytest.approx(float(shallow[anchor].mean()))
    assert "deep_layer" not in md["mean_attention"]
    sim = md["similarity"]
    assert sim["diverse_at_selection"] == pytest.approx(
        float(sim_stats[diverse].mean())
    )
    assert 0.0 <= sim["pruned_vs_kept"] <= 1.0


def test_get_hiprune_method_env(monkeypatch):
    monkeypatch.delenv("HIPRUNE_METHOD", raising=False)
    assert get_hiprune_method() == "hiprune"
    monkeypatch.setenv("HIPRUNE_METHOD", "hydart")
    assert get_hiprune_method() == "hydart"
    monkeypatch.setenv("HIPRUNE_METHOD", "HyDART")
    assert get_hiprune_method() == "hydart"
    monkeypatch.setenv("HIPRUNE_METHOD", "hiprune_pp")
    assert get_hiprune_method() == "hiprune_pp"
    monkeypatch.setenv("HIPRUNE_METHOD", "dart")
    assert get_hiprune_method() == "dart"
    monkeypatch.setenv("HIPRUNE_METHOD", "nprune")
    assert get_hiprune_method() == "nprune"
    monkeypatch.setenv("HIPRUNE_METHOD", "bogus")
    with pytest.raises(ValueError):
        get_hiprune_method()


# --------------------------------------------------------------------------
# HiPrune++ (prompt-aware selection, paper Algorithm 1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 12, 255, 260, 576, 1280])
@pytest.mark.parametrize("ratio", [0.05, 0.11, 0.14, 0.223, 0.5, 0.95])
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.3, 1.0])
def test_hiprune_pp_budget_bounds(num_tokens: int, ratio: float, beta: float):
    base, t_sum = compute_hiprune_pp_budget(num_tokens, ratio, beta)
    assert base == compute_retained_tokens_count(num_tokens, ratio)
    assert t_sum >= 0
    # The paper's prose: retain [beta * N] text-guided visual tokens,
    # clamped so the total never exceeds the token count.
    assert t_sum == min(round(base * beta), num_tokens - base)
    assert base + t_sum <= num_tokens


def _pp_case(seed: int, n_tokens: int = 260, hidden: int = 64):
    torch.manual_seed(seed)
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    embeddings = torch.randn(n_tokens, hidden)
    text = torch.randn(hidden)
    return shallow, deep, embeddings, text


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ratio", [0.11, 0.14, 0.223, 0.5])
@pytest.mark.parametrize("beta", [0.1, 0.3])
def test_hiprune_pp_invariants(seed: int, ratio: float, beta: float):
    n_tokens, grid_w = 260, 20
    shallow, deep, embeddings, text = _pp_case(seed, n_tokens)

    anchor, buffer, register, prompt, kept, sim = hiprune_pp_select(
        shallow, deep, embeddings, text, n_tokens, grid_w, ratio, beta=beta
    )

    # Exact budget: base + text-guided extra, matching the placeholder
    # count the processors emit.
    base, t_sum = compute_hiprune_pp_budget(n_tokens, ratio, beta)
    assert kept.sum().item() == base + t_sum
    assert prompt.numel() == t_sum

    # The base categories are exactly plain HiPrune's selection.
    ref = hiprune_select(shallow, deep, n_tokens, grid_w, ratio)
    assert torch.equal(anchor, ref[0])
    assert torch.equal(buffer, ref[1])
    assert torch.equal(register, ref[2])

    # Categories are disjoint and jointly equal the kept set.
    all_idx = torch.cat([anchor, buffer, register, prompt])
    assert all_idx.unique().numel() == all_idx.numel()
    mask_from_cats = torch.zeros(n_tokens, dtype=torch.bool)
    mask_from_cats[all_idx] = True
    assert torch.equal(mask_from_cats, kept)

    # Prompt tokens are exactly the top-t_sum by text similarity among
    # tokens the base method did not keep (Algorithm 1 lines 26-30).
    emb_n = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    text_n = torch.nn.functional.normalize(text.float(), dim=-1)
    expected_sim = emb_n @ text_n
    assert torch.allclose(sim, expected_sim, atol=1e-6)
    masked = expected_sim.clone()
    masked[ref[3]] = float("-inf")
    expected_prompt = torch.topk(masked, k=t_sum).indices
    assert torch.equal(torch.sort(prompt).values, torch.sort(expected_prompt).values)


def test_hiprune_pp_deterministic():
    shallow, deep, embeddings, text = _pp_case(7)
    first = hiprune_pp_select(
        shallow, deep, embeddings, text, 260, 20, 0.14, beta=0.1
    )
    second = hiprune_pp_select(
        shallow, deep, embeddings, text, 260, 20, 0.14, beta=0.1
    )
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_hiprune_pp_zero_beta_matches_hiprune():
    shallow, deep, embeddings, text = _pp_case(11)
    anchor, buffer, register, prompt, kept, _ = hiprune_pp_select(
        shallow, deep, embeddings, text, 260, 20, 0.14, beta=0.0
    )
    ref = hiprune_select(shallow, deep, 260, 20, 0.14)
    assert prompt.numel() == 0
    assert torch.equal(kept, ref[3])


def test_hiprune_pp_full_retention_keeps_everything():
    shallow, deep, embeddings, text = _pp_case(3, n_tokens=255)
    _, _, _, prompt, kept, _ = hiprune_pp_select(
        shallow, deep, embeddings, text, 255, 15, 1.0, beta=0.1
    )
    # Base budget already covers every token; no room for text picks.
    assert kept.all()
    assert prompt.numel() == 0


def test_hiprune_pp_zero_text_embedding():
    """Image-only messages: zero text vector still keeps the exact count."""
    shallow, deep, embeddings, _ = _pp_case(5)
    text = torch.zeros(embeddings.shape[-1])
    _, _, _, prompt, kept, sim = hiprune_pp_select(
        shallow, deep, embeddings, text, 260, 20, 0.14, beta=0.1
    )
    base, t_sum = compute_hiprune_pp_budget(260, 0.14, 0.1)
    assert kept.sum().item() == base + t_sum
    assert prompt.numel() == t_sum
    assert (sim == 0).all()


def test_hiprune_pp_prompt_tokens_track_text():
    """Tokens aligned with the text embedding must win the prompt slots."""
    torch.manual_seed(9)
    n_tokens, grid_w, hidden = 260, 20, 64
    shallow = torch.rand(n_tokens).softmax(dim=0)
    deep = torch.rand(n_tokens).softmax(dim=0)
    text = torch.randn(hidden)
    # Orthogonalize all embeddings against text, then plant strong
    # alignment on a few tokens the base method does not keep.
    embeddings = torch.randn(n_tokens, hidden)
    text_n = torch.nn.functional.normalize(text, dim=-1)
    embeddings -= (embeddings @ text_n).unsqueeze(-1) * text_n
    ref_kept = hiprune_select(shallow, deep, n_tokens, grid_w, 0.14)[3]
    unkept = (~ref_kept).nonzero(as_tuple=True)[0]
    base, t_sum = compute_hiprune_pp_budget(n_tokens, 0.14, 0.3)
    planted = unkept[:t_sum]
    embeddings[planted] += 10.0 * text_n

    _, _, _, prompt, _, _ = hiprune_pp_select(
        shallow, deep, embeddings, text, n_tokens, grid_w, 0.14, beta=0.3
    )
    assert torch.equal(torch.sort(prompt).values, torch.sort(planted).values)


def test_hiprune_pp_metadata_contents():
    import json

    shallow, deep, embeddings, text = _pp_case(13)
    n_tokens, grid_w, grid_h, ratio, beta = 260, 20, 13, 0.14, 0.1
    anchor, buffer, register, prompt, kept, sim = hiprune_pp_select(
        shallow, deep, embeddings, text, n_tokens, grid_w, ratio, beta=beta
    )
    md = build_hiprune_pp_metadata(
        anchor,
        buffer,
        register,
        prompt,
        kept,
        shallow,
        deep,
        sim,
        grid_w,
        grid_h,
        ratio,
        object_layer=17,
        beta=beta,
    )
    json.dumps(md)  # must be JSON-safe

    assert md["method"] == "hiprune_pp"
    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == n_tokens
    assert md["beta"] == beta
    assert md["object_layer"] == 17
    assert sorted(md["prompt_tokens"]) == sorted(prompt.tolist())
    base, t_sum = compute_hiprune_pp_budget(n_tokens, ratio, beta)
    assert len(md["pruned"]) == n_tokens - (base + t_sum)
    # Categories partition kept + pruned = all tokens.
    all_reported = (
        md["anchors"]
        + md["buffers"]
        + md["registers"]
        + md["prompt_tokens"]
        + md["pruned"]
    )
    assert sorted(all_reported) == list(range(n_tokens))
    # Per-token arrays for tooltips.
    assert len(md["scores"]["object_layer"]) == n_tokens
    assert len(md["scores"]["deep_layer"]) == n_tokens
    assert len(md["scores"]["text_similarity"]) == n_tokens
    # Prompt tokens have the highest mean text similarity of the
    # reported categories (they were chosen by it).
    ts = md["text_similarity_summary"]
    assert ts["prompt"] >= ts["pruned"]


def test_get_hiprune_prompt_passthrough():
    assert get_hiprune_prompt({}) is None
    assert get_hiprune_prompt({"hiprune_prompt": None}) is None
    assert get_hiprune_prompt({"hiprune_prompt": "   "}) is None
    assert get_hiprune_prompt({"hiprune_prompt": "what color?"}) == "what color?"


def test_get_hiprune_pp_beta_env(monkeypatch):
    from vllm.multimodal.hiprune import get_hiprune_pp_beta

    monkeypatch.delenv("HIPRUNE_PP_BETA", raising=False)
    assert get_hiprune_pp_beta() == pytest.approx(0.1)
    monkeypatch.setenv("HIPRUNE_PP_BETA", "0.25")
    assert get_hiprune_pp_beta() == pytest.approx(0.25)


# --------------------------------------------------------------------------
# DART (duplication-aware selection from LLM layer-K states)
# --------------------------------------------------------------------------


def _dart_reference(
    hidden_states: torch.Tensor,
    key_l1_norms: torch.Tensor,
    num_image_tokens: int,
    ratio: float,
    p_img: int,
    p_txt: int,
) -> set[int]:
    """Line-for-line port of the official ``get_retained_image_token``
    (DART/Qwen2_5-VL/Qwen2_5VL_DART/modeling_qwen2_5_vl_self.py), on the
    flat aux-sequence layout (image tokens at [0, num_image_tokens), text
    after) and with the pivot iteration order fixed to sorted image
    pivots then sorted text pivots — the only free choice in the official
    code, where ``for item in list(indices_set)`` iterates a Python set.
    """
    image_token_start_index = 0
    image_token_length = num_image_tokens
    token_topk = int(image_token_length * ratio / (p_img + p_txt))

    norms_img = key_l1_norms[:num_image_tokens]
    norms_txt = key_l1_norms[num_image_tokens:]
    image_indices = sorted(
        (norms_img.topk(p_img).indices + image_token_start_index).tolist()
    )
    query_indices = sorted(
        (
            norms_txt.topk(p_txt).indices
            + image_token_start_index
            + image_token_length
        ).tolist()
    )
    indices_set = set(image_indices + query_indices)
    valid_indices = set(
        range(image_token_start_index, image_token_start_index + image_token_length)
    ) - set(image_indices)

    valid_indices_list = list(valid_indices)
    for item in image_indices + query_indices:  # deterministic order
        valid_vectors = hidden_states[valid_indices_list, :]
        cos_sim = -torch.nn.functional.cosine_similarity(
            hidden_states[item, :], valid_vectors, dim=-1
        )
        top_k_indices = cos_sim.topk(token_topk).indices
        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
        indices_set.update(top_k_real_indices)
        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)

    indices_set.difference_update(query_indices)
    return indices_set


def _dart_case(seed: int, n_img: int = 260, n_txt: int = 24, hidden: int = 64):
    torch.manual_seed(seed)
    hs = torch.randn(n_img + n_txt, hidden)
    key_l1 = torch.rand(n_img + n_txt) * 10
    return hs, key_l1


@pytest.mark.parametrize("num_tokens", [1, 12, 255, 260, 576, 1280])
@pytest.mark.parametrize("ratio", [0.05, 0.11, 0.14, 0.223, 0.5, 0.95, 1.0])
def test_dart_keep_count_bounds(num_tokens: int, ratio: float):
    kept = dart_keep_count(num_tokens, ratio, 4, 4)
    assert 1 <= kept <= num_tokens
    # Official formula whenever it lands in range.
    token_topk = int(num_tokens * ratio / 8)
    official = min(4, num_tokens) + 8 * token_topk
    assert kept == max(1, min(num_tokens, official))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("ratio", [0.11, 0.14, 0.25, 0.5])
def test_dart_matches_official_reference(seed: int, ratio: float):
    """dart_select must keep exactly the official implementation's set
    (same pivot order), validating the verbatim port."""
    n_img, n_txt = 260, 24
    hs, key_l1 = _dart_case(seed, n_img, n_txt)

    img_piv, txt_piv, diverse, kept, _ = dart_select(
        hs, key_l1, n_img, ratio, pivot_image=4, pivot_text=4
    )
    ref = _dart_reference(hs, key_l1, n_img, ratio, 4, 4)
    assert set(kept.nonzero(as_tuple=True)[0].tolist()) == ref


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("ratio", [0.11, 0.14, 0.25, 0.5])
@pytest.mark.parametrize("pivots", [(4, 4), (2, 6), (4, 0), (1, 1)])
def test_dart_invariants(seed: int, ratio: float, pivots: tuple[int, int]):
    p_img, p_txt = pivots
    n_img, n_txt = 260, 24
    hs, key_l1 = _dart_case(seed, n_img, n_txt)

    img_piv, txt_piv, diverse, kept, sim = dart_select(
        hs, key_l1, n_img, ratio, pivot_image=p_img, pivot_text=p_txt
    )

    # Exact deterministic budget (the placeholder invariant).
    assert kept.sum().item() == dart_keep_count(n_img, ratio, p_img, p_txt)

    # Image pivots + diverse partition the kept set, no duplicates.
    all_idx = torch.cat([img_piv, diverse])
    assert all_idx.unique().numel() == all_idx.numel()
    mask = torch.zeros(n_img, dtype=torch.bool)
    mask[all_idx] = True
    assert torch.equal(mask, kept)

    # Image pivots are the top-p_img by key L1 norm.
    expected_piv = torch.topk(key_l1[:n_img], k=min(p_img, n_img)).indices
    assert set(img_piv.tolist()) == set(expected_piv.tolist())

    # Text pivots are prompt-relative and within the prompt.
    assert txt_piv.numel() == min(p_txt, n_txt)
    if txt_piv.numel():
        assert txt_piv.max() < n_txt

    # Similarity stats cover every image token, in [-1, 1].
    assert sim.shape == (n_img,)
    assert (sim >= -1.0 - 1e-5).all() and (sim <= 1.0 + 1e-5).all()


def test_dart_deterministic():
    hs, key_l1 = _dart_case(7)
    first = dart_select(hs, key_l1, 260, 0.14, pivot_image=4, pivot_text=4)
    second = dart_select(hs, key_l1, 260, 0.14, pivot_image=4, pivot_text=4)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_dart_prompt_awareness():
    """Changing the text tokens must be able to change the kept set."""
    torch.manual_seed(21)
    n_img, n_txt, hidden = 260, 24, 64
    img_hs = torch.randn(n_img, hidden)
    key_img = torch.rand(n_img) * 10

    kept_sets = set()
    for text_seed in range(4):
        torch.manual_seed(100 + text_seed)
        txt_hs = torch.randn(n_txt, hidden)
        key_txt = torch.rand(n_txt) * 10
        hs = torch.cat([img_hs, txt_hs])
        key_l1 = torch.cat([key_img, key_txt])
        _, _, _, kept, _ = dart_select(
            hs, key_l1, n_img, 0.14, pivot_image=4, pivot_text=4
        )
        kept_sets.add(tuple(kept.nonzero(as_tuple=True)[0].tolist()))
    assert len(kept_sets) > 1


def test_dart_no_prompt_top_up():
    """Zero text tokens: image pivots alone under-fill the pivot rounds;
    the top-up must still hit the exact deterministic budget."""
    torch.manual_seed(3)
    n_img, hidden = 260, 64
    hs = torch.randn(n_img, hidden)  # no text rows at all
    key_l1 = torch.rand(n_img) * 10

    img_piv, txt_piv, diverse, kept, _ = dart_select(
        hs, key_l1, n_img, 0.14, pivot_image=4, pivot_text=4
    )
    assert txt_piv.numel() == 0
    # Budget still computed with pivot_text=4 (prompt-free invariant).
    assert kept.sum().item() == dart_keep_count(n_img, 0.14, 4, 4)


def test_dart_short_prompt_top_up():
    """Prompt shorter than pivot_text: fewer text pivots, same count."""
    hs, key_l1 = _dart_case(5, n_img=260, n_txt=2)
    img_piv, txt_piv, diverse, kept, _ = dart_select(
        hs, key_l1, 260, 0.14, pivot_image=4, pivot_text=4
    )
    assert txt_piv.numel() == 2
    assert kept.sum().item() == dart_keep_count(260, 0.14, 4, 4)


def test_dart_full_retention_keeps_everything():
    hs, key_l1 = _dart_case(9, n_img=256)
    _, _, _, kept, _ = dart_select(hs, key_l1, 256, 1.0, pivot_image=4, pivot_text=4)
    assert kept.all()


def test_dart_diverse_avoids_pivot_duplicates():
    """Tokens near-identical to the pivot must lose to novel tokens.

    Single image pivot, no text pivots, so the pivot's own
    anti-duplication round fills the whole budget and its clones are
    deterministically the *worst* candidates.
    """
    torch.manual_seed(17)
    n_img, hidden = 200, 64
    hs = torch.randn(n_img, hidden)
    key_l1 = torch.rand(n_img)
    # Make token 0 the sole pivot and tokens 1..40 its clones.
    key_l1[0] = 100.0
    hs[1:41] = hs[0] + 0.001 * torch.randn(40, hidden)

    _, _, diverse, kept, sim = dart_select(
        hs, key_l1, n_img, 0.25, pivot_image=1, pivot_text=0
    )
    # Budget = 1 + 50; 159 non-clone candidates exist, all less similar
    # to the pivot than any clone, so no clone survives.
    assert kept[0]  # the pivot itself
    assert kept[1:41].sum().item() == 0
    assert sim[1:41].min() > 0.99  # clones flagged as duplicated


def test_dart_metadata_contents():
    import json

    hs, key_l1 = _dart_case(13)
    n_img, grid_w, grid_h, ratio = 260, 20, 13, 0.14
    img_piv, txt_piv, diverse, kept, sim = dart_select(
        hs, key_l1, n_img, ratio, pivot_image=4, pivot_text=4
    )
    md = build_dart_metadata(
        img_piv,
        diverse,
        kept,
        key_l1,
        sim,
        grid_w,
        grid_h,
        ratio,
        pivot_image=4,
        pivot_text=4,
        num_text_pivots=int(txt_piv.numel()),
        dart_layer=2,
    )
    json.dumps(md)  # must be JSON-safe

    assert md["method"] == "dart"
    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == n_img
    assert md["pivot_image"] == 4 and md["pivot_text"] == 4
    assert md["num_text_pivots"] == 4
    assert md["dart_layer"] == 2
    assert sorted(md["pivots"]) == sorted(img_piv.tolist())
    assert sorted(md["diverse"]) == sorted(diverse.tolist())
    kept_count = dart_keep_count(n_img, ratio, 4, 4)
    assert len(md["pruned"]) == n_img - kept_count
    # Categories partition all tokens.
    all_reported = md["pivots"] + md["diverse"] + md["pruned"]
    assert sorted(all_reported) == list(range(n_img))
    # Per-token arrays for tooltips (image tokens only).
    assert len(md["scores"]["key_norm"]) == n_img
    assert len(md["scores"]["pivot_similarity"]) == n_img
    # Summary means are consistent with the per-token arrays.
    ps = torch.tensor(md["scores"]["pivot_similarity"])
    assert md["similarity"]["kept_vs_pivots"] == pytest.approx(
        float(ps[kept].mean()), abs=1e-6
    )
    assert md["similarity"]["pruned_vs_pivots"] == pytest.approx(
        float(ps[~kept].mean()), abs=1e-6
    )


def test_get_dart_pivots_env(monkeypatch):
    monkeypatch.delenv("HIPRUNE_DART_PIVOT_IMAGE", raising=False)
    monkeypatch.delenv("HIPRUNE_DART_PIVOT_TEXT", raising=False)
    assert get_dart_pivots() == (4, 4)
    monkeypatch.setenv("HIPRUNE_DART_PIVOT_IMAGE", "2")
    monkeypatch.setenv("HIPRUNE_DART_PIVOT_TEXT", "6")
    assert get_dart_pivots() == (2, 6)
    monkeypatch.setenv("HIPRUNE_DART_PIVOT_IMAGE", "0")
    with pytest.raises(ValueError):
        get_dart_pivots()


def test_get_dart_layer_env(monkeypatch):
    monkeypatch.delenv("HIPRUNE_DART_LAYER", raising=False)
    assert get_dart_layer() == 2
    monkeypatch.setenv("HIPRUNE_DART_LAYER", "3")
    assert get_dart_layer() == 3
    monkeypatch.setenv("HIPRUNE_DART_LAYER", "0")
    with pytest.raises(ValueError):
        get_dart_layer()


# --------------------------------------------------------------------- #
# Per-request method/knobs (mm kwargs over env) and the packed config
# --------------------------------------------------------------------- #


def test_get_hiprune_method_kwargs_precedence(monkeypatch):
    monkeypatch.delenv("HIPRUNE_METHOD", raising=False)
    assert get_hiprune_method({"hiprune_method": "dart"}) == "dart"
    assert get_hiprune_method({"hiprune_method": "HyDART"}) == "hydart"
    # kwargs without the key fall back to env.
    monkeypatch.setenv("HIPRUNE_METHOD", "hiprune_pp")
    assert get_hiprune_method({}) == "hiprune_pp"
    # kwargs win over env.
    assert get_hiprune_method({"hiprune_method": "hiprune"}) == "hiprune"
    with pytest.raises(ValueError):
        get_hiprune_method({"hiprune_method": "bogus"})


def test_knob_accessors_kwargs_precedence(monkeypatch):
    for var in (
        "HYDART_LAMBDA_SEED",
        "HYDART_LAMBDA_PICK",
        "HIPRUNE_PP_BETA",
        "HIPRUNE_DART_PIVOT_IMAGE",
        "HIPRUNE_DART_PIVOT_TEXT",
    ):
        monkeypatch.delenv(var, raising=False)

    assert get_hydart_lambdas({"hiprune_lambda_seed": 0.3}) == (0.3, 0.5)
    assert get_hydart_lambdas({"hiprune_lambda_pick": 0.9}) == (0.1, 0.9)
    assert get_hiprune_pp_beta({"hiprune_beta": 0.25}) == 0.25
    assert get_dart_pivots(
        {"hiprune_pivot_image": 2, "hiprune_pivot_text": 6}
    ) == (2, 6)

    # kwargs win over env.
    monkeypatch.setenv("HYDART_LAMBDA_SEED", "0.7")
    monkeypatch.setenv("HIPRUNE_PP_BETA", "0.4")
    monkeypatch.setenv("HIPRUNE_DART_PIVOT_IMAGE", "8")
    assert get_hydart_lambdas({"hiprune_lambda_seed": 0.2})[0] == 0.2
    assert get_hydart_lambdas({})[0] == 0.7
    assert get_hiprune_pp_beta({"hiprune_beta": 0.05}) == 0.05
    assert get_dart_pivots({"hiprune_pivot_image": 3})[0] == 3
    assert get_dart_pivots({})[0] == 8

    with pytest.raises(ValueError):
        get_hiprune_pp_beta({"hiprune_beta": 1.5})
    with pytest.raises(ValueError):
        get_dart_pivots({"hiprune_pivot_image": 0})


def test_pack_unpack_hiprune_config_roundtrip(monkeypatch):
    for var in (
        "HIPRUNE_METHOD",
        "HYDART_LAMBDA_SEED",
        "HYDART_LAMBDA_PICK",
        "HIPRUNE_PP_BETA",
        "HIPRUNE_DART_PIVOT_IMAGE",
        "HIPRUNE_DART_PIVOT_TEXT",
    ):
        monkeypatch.delenv(var, raising=False)

    kwargs = {
        "hiprune_method": "dart",
        "hiprune_lambda_seed": 0.2,
        "hiprune_lambda_pick": 0.7,
        "hiprune_beta": 0.15,
        "hiprune_pivot_image": 3,
        "hiprune_pivot_text": 5,
        "hiprune_stride": 1,
    }
    row = pack_hiprune_config(kwargs)
    assert row.shape == (HIPRUNE_CONFIG_WIDTH,)
    assert row.dtype == torch.float32

    cfg = unpack_hiprune_config(row)
    assert cfg.method == "dart"
    assert cfg.pivot_image == 3 and cfg.pivot_text == 5
    assert cfg.lambda_seed == pytest.approx(0.2, abs=1e-6)
    assert cfg.lambda_pick == pytest.approx(0.7, abs=1e-6)
    assert cfg.beta == pytest.approx(0.15, abs=1e-6)
    assert cfg.stride == 1

    # Every method id round-trips.
    for method in HIPRUNE_METHOD_IDS:
        cfg = unpack_hiprune_config(pack_hiprune_config({"hiprune_method": method}))
        assert cfg.method == method

    with pytest.raises(ValueError):
        unpack_hiprune_config(torch.zeros(3))
    with pytest.raises(ValueError):
        # Old 6-wide rows (pre-stride) are rejected loudly, not
        # silently defaulted — matters if processor and model worker
        # briefly run different commits during a deploy.
        unpack_hiprune_config(
            torch.tensor([0.0, 0.1, 0.5, 0.1, 4.0, 4.0], dtype=torch.float32)
        )
    with pytest.raises(ValueError):
        unpack_hiprune_config(
            torch.tensor([99.0, 0.1, 0.5, 0.1, 4.0, 4.0, 2.0], dtype=torch.float32)
        )


def test_unpack_none_falls_back_to_env(monkeypatch):
    for var in (
        "HYDART_LAMBDA_SEED",
        "HYDART_LAMBDA_PICK",
        "HIPRUNE_PP_BETA",
        "HIPRUNE_DART_PIVOT_IMAGE",
        "HIPRUNE_DART_PIVOT_TEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HIPRUNE_METHOD", "hydart")
    monkeypatch.setenv("HYDART_LAMBDA_SEED", "0.33")

    cfg = unpack_hiprune_config(None)
    assert cfg.method == "hydart"
    assert cfg.lambda_seed == pytest.approx(0.33, abs=1e-6)
    assert cfg.pivot_image == 4 and cfg.pivot_text == 4

    monkeypatch.delenv("HIPRUNE_METHOD", raising=False)
    assert unpack_hiprune_config(None).method == "hiprune"


def test_hiprune_mm_kwarg_keys_cover_pack_inputs():
    # pack_hiprune_config reads exactly these per-request keys; the strip
    # list model processors use must cover all of them (plus ratio and
    # prompt) or the HF processor would receive vLLM-side kwargs.
    assert set(HIPRUNE_MM_KWARG_KEYS) == {
        "hiprune_ratio",
        "hiprune_prompt",
        "hiprune_method",
        "hiprune_lambda_seed",
        "hiprune_lambda_pick",
        "hiprune_beta",
        "hiprune_pivot_image",
        "hiprune_pivot_text",
        "hiprune_stride",
    }


# --------------------------------------------------------------------------
# NPrune (naive uniform spatial pruning)
# --------------------------------------------------------------------------


def test_nprune_even_grid_exact_lattice():
    """On even grids stride 2 keeps exactly the upper-left corner of
    every 2x2 block — 25% retention, row-major indices."""
    grid_h, grid_w = 4, 6
    kept_idx, kept_mask = nprune_select(grid_h, grid_w, 2)

    expected = [
        r * grid_w + c
        for r in range(0, grid_h, 2)
        for c in range(0, grid_w, 2)
    ]
    assert kept_idx.tolist() == expected
    assert int(kept_mask.sum()) == len(expected) == grid_h * grid_w // 4
    assert kept_mask.shape[0] == grid_h * grid_w
    assert kept_mask[kept_idx].all()


@pytest.mark.parametrize(
    "grid_h,grid_w",
    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 5), (24, 24), (23, 31), (34, 46)],
)
@pytest.mark.parametrize("stride", [1, 2])
def test_nprune_count_invariant(grid_h: int, grid_w: int, stride: int):
    """kept_mask.sum() == nprune_keep_count for every grid — the
    processor/model placeholder invariant."""
    kept_idx, kept_mask = nprune_select(grid_h, grid_w, stride)
    expected = nprune_keep_count(grid_h, grid_w, stride)
    assert int(kept_mask.sum()) == kept_idx.numel() == expected
    # Exact ceil form, not ceil(N / stride^2).
    assert expected == math.ceil(grid_h / stride) * math.ceil(grid_w / stride)
    # Ascending, unique, in-range raster indices.
    assert (kept_idx[1:] > kept_idx[:-1]).all() if kept_idx.numel() > 1 else True
    assert kept_idx.min() >= 0 and kept_idx.max() < grid_h * grid_w
    # Every kept index sits on the lattice.
    rows = kept_idx // grid_w
    cols = kept_idx % grid_w
    assert (rows % stride == 0).all() and (cols % stride == 0).all()


def test_nprune_odd_grid_exact_not_ratio_derived():
    """3x3 at stride 2 keeps the 4 corners (44.4%), not ceil(9/4) = 3."""
    kept_idx, kept_mask = nprune_select(3, 3, 2)
    assert kept_idx.tolist() == [0, 2, 6, 8]
    assert nprune_keep_count(3, 3, 2) == 4


def test_nprune_stride_one_identity():
    """Stride 1 keeps every token (serving maps it to no-pruning; the
    selection itself must still be the identity)."""
    kept_idx, kept_mask = nprune_select(5, 7, 1)
    assert kept_mask.all()
    assert kept_idx.tolist() == list(range(35))
    assert nprune_keep_count(5, 7, 1) == 35


def test_nprune_deterministic():
    a_idx, a_mask = nprune_select(17, 29, 2)
    b_idx, b_mask = nprune_select(17, 29, 2)
    assert torch.equal(a_idx, b_idx) and torch.equal(a_mask, b_mask)


def test_nprune_gemma_grid_parity():
    """Grid dims from real Gemma pixel_position_ids feed the same count
    the selection produces (the processor/model parity path)."""
    for patch_w, patch_h in [(12, 9), (45, 39), (15, 51)]:
        pos = _make_position_ids(patch_w, patch_h)
        _, grid_w, grid_h, _ = compute_soft_token_grid(pos, POOL_K)
        kept_idx, kept_mask = nprune_select(grid_h, grid_w, 2)
        assert int(kept_mask.sum()) == nprune_keep_count(grid_h, grid_w, 2)
        assert kept_mask.shape[0] == grid_w * grid_h


def test_nprune_metadata_contents():
    grid_h, grid_w = 4, 6
    kept_idx, kept_mask = nprune_select(grid_h, grid_w, 2)
    md = build_nprune_metadata(kept_idx, kept_mask, grid_w, grid_h, 2)

    assert md["method"] == "nprune"
    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == grid_h * grid_w
    assert md["stride"] == 2
    # Actual retention (kept / num_tokens), exactly 0.25 on even grids.
    assert md["retention"] == pytest.approx(0.25)
    assert sorted(md["uniform"] + md["pruned"]) == list(range(grid_h * grid_w))
    assert set(md["uniform"]).isdisjoint(md["pruned"])
    # JSON-safe: plain ints/floats/lists only.
    import json

    json.dumps(md)

    # Odd grid: retention reports the actual kept fraction.
    kept_idx, kept_mask = nprune_select(3, 3, 2)
    md = build_nprune_metadata(kept_idx, kept_mask, 3, 3, 2)
    assert md["retention"] == pytest.approx(4 / 9)


def test_get_nprune_stride(monkeypatch):
    monkeypatch.delenv("HIPRUNE_NPRUNE_STRIDE", raising=False)
    assert get_nprune_stride() == 2
    assert get_nprune_stride({"hiprune_stride": 1}) == 1
    assert get_nprune_stride({"hiprune_stride": 2.0}) == 2
    # kwargs win over env.
    monkeypatch.setenv("HIPRUNE_NPRUNE_STRIDE", "1")
    assert get_nprune_stride() == 1
    assert get_nprune_stride({"hiprune_stride": 2}) == 2
    with pytest.raises(ValueError):
        get_nprune_stride({"hiprune_stride": 3})
    with pytest.raises(ValueError):
        get_nprune_stride({"hiprune_stride": 0})


def _checkered_expected(grid_h: int, grid_w: int) -> list[int]:
    """Independent reference: raster indices with (row + col) even."""
    return [
        r * grid_w + c
        for r in range(grid_h)
        for c in range(grid_w)
        if (r + c) % 2 == 0
    ]


def test_checkered_4x4_exact_pattern():
    """X.X. / .X.X / X.X. / .X.X — alternating on every row."""
    kept_idx, kept_mask = checkered_select(4, 4)
    assert kept_idx.tolist() == [0, 2, 5, 7, 8, 10, 13, 15]
    assert int(kept_mask.sum()) == 8


def test_checkered_3x3_exact_pattern():
    """X.X / .X. / X.X — 5 of 9 kept (the (0,0) phase gets the extra)."""
    kept_idx, kept_mask = checkered_select(3, 3)
    assert kept_idx.tolist() == [0, 2, 4, 6, 8]
    assert checkered_keep_count(9) == 5


def test_checkered_3x5_exact_pattern():
    """X.X.X / .X.X. / X.X.X — non-square odd grid."""
    kept_idx, kept_mask = checkered_select(3, 5)
    assert kept_idx.tolist() == [0, 2, 4, 6, 8, 10, 12, 14]
    assert checkered_keep_count(15) == 8


@pytest.mark.parametrize(
    "grid_h,grid_w",
    [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 5), (17, 16), (24, 24), (23, 31)],
)
def test_checkered_count_invariant(grid_h: int, grid_w: int):
    """kept == ceil(HW/2) — the processor/model placeholder invariant —
    and every kept cell satisfies (row + col) % 2 == 0."""
    kept_idx, kept_mask = checkered_select(grid_h, grid_w)
    n = grid_h * grid_w
    expected = checkered_keep_count(n)
    assert int(kept_mask.sum()) == kept_idx.numel() == expected == (n + 1) // 2
    assert kept_idx.tolist() == _checkered_expected(grid_h, grid_w)
    # Ascending, unique, in-range raster indices.
    assert (kept_idx[1:] > kept_idx[:-1]).all() if kept_idx.numel() > 1 else True
    rows = kept_idx // grid_w
    cols = kept_idx % grid_w
    assert ((rows + cols) % 2 == 0).all()
    # A checkerboard keeps at least one token on EVERY row (what
    # distinguishes it from a stride lattice, which skips rows) —
    # except on single-column grids, where odd rows have only the
    # odd-parity cell and correctly keep nothing.
    if grid_w >= 2:
        assert set(rows.tolist()) == set(range(grid_h))
    else:
        assert set(rows.tolist()) == set(range(0, grid_h, 2))


def test_checkered_count_is_shape_independent():
    """ceil(HW/2) depends only on the product — the property that lets
    every placeholder-sizing site use the count-only path."""
    for shapes in [[(2, 12), (3, 8), (4, 6), (24, 1)], [(3, 5), (5, 3), (15, 1)]]:
        counts = set()
        for grid_h, grid_w in shapes:
            _, kept_mask = checkered_select(grid_h, grid_w)
            counts.add(int(kept_mask.sum()))
        assert len(counts) == 1


def test_checkered_ratio_path_would_be_wrong():
    """The generic budget round(n * 0.5) rounds half-to-even and loses
    a token on odd counts — 9 tokens: ratio path 4, checkerboard 5.
    Documents why every sizing site needs the explicit branch."""
    assert compute_retained_tokens_count(9, 0.5) == 4
    assert checkered_keep_count(9) == 5


def test_checkered_deterministic():
    a_idx, a_mask = checkered_select(17, 29)
    b_idx, b_mask = checkered_select(17, 29)
    assert torch.equal(a_idx, b_idx) and torch.equal(a_mask, b_mask)


def test_checkered_invalid_grid_raises():
    with pytest.raises(ValueError):
        checkered_select(0, 5)
    with pytest.raises(ValueError):
        checkered_select(5, -1)


def test_checkered_gemma_grid_parity():
    """Grid dims from real Gemma pixel_position_ids feed the same count
    the selection produces (the processor/model parity path)."""
    for patch_w, patch_h in [(12, 9), (45, 39), (15, 51)]:
        pos = _make_position_ids(patch_w, patch_h)
        _, grid_w, grid_h, _ = compute_soft_token_grid(pos, POOL_K)
        kept_idx, kept_mask = checkered_select(grid_h, grid_w)
        assert int(kept_mask.sum()) == checkered_keep_count(grid_w * grid_h)
        assert kept_mask.shape[0] == grid_w * grid_h


def test_checkered_metadata_contents():
    grid_h, grid_w = 17, 16
    kept_idx, kept_mask = checkered_select(grid_h, grid_w)
    md = build_checkered_metadata(kept_idx, kept_mask, grid_w, grid_h)

    assert md["method"] == "checkered"
    assert md["grid"] == [grid_w, grid_h]
    assert md["num_tokens"] == 272
    assert len(md["uniform"]) == 136 and len(md["pruned"]) == 136
    # Actual retention, exactly 0.5 on even token counts.
    assert md["retention"] == pytest.approx(0.5)
    assert sorted(md["uniform"] + md["pruned"]) == list(range(272))
    assert set(md["uniform"]).isdisjoint(md["pruned"])
    # JSON-safe: plain ints/floats/lists only.
    import json

    json.dumps(md)

    # Odd token count: retention reports the actual kept fraction.
    kept_idx, kept_mask = checkered_select(3, 3)
    md = build_checkered_metadata(kept_idx, kept_mask, 3, 3)
    assert md["retention"] == pytest.approx(5 / 9)


def test_checkered_method_registration(monkeypatch):
    assert HIPRUNE_METHOD_IDS["checkered"] == 5
    assert get_hiprune_method({"hiprune_method": "checkered"}) == "checkered"
    monkeypatch.setenv("HIPRUNE_METHOD", "checkered")
    assert get_hiprune_method() == "checkered"
    # No knobs: the packed config row keeps its width.
    row = pack_hiprune_config({"hiprune_method": "checkered"})
    assert row.numel() == HIPRUNE_CONFIG_WIDTH
    assert unpack_hiprune_config(row).method == "checkered"
