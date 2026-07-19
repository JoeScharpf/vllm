# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HiPrune visual token selection (vllm/multimodal/hiprune.py)."""

import pytest
import torch

from vllm.multimodal.hiprune import (
    QWEN2_5_VL_OBJECT_LAYER,
    aggregate_patch_attention,
    build_hiprune_metadata,
    build_hydart_metadata,
    compute_retained_tokens_count,
    compute_soft_token_grid,
    fold_merged_token_scores,
    get_hiprune_method,
    get_hiprune_ratio,
    hiprune_select,
    hydart_select,
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
    monkeypatch.setenv("HIPRUNE_METHOD", "bogus")
    with pytest.raises(ValueError):
        get_hiprune_method()
