# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary LLM prefix pass for DART token selection.

DART (arXiv 2502.11494) selects visual tokens from the language model's
*own* layer-K states: attention-key L1 norms pick the pivots and hidden
states drive the anti-duplication cosine test (see
:func:`vllm.multimodal.hiprune.dart_select`). The official implementation
prunes mid-forward inside a HF ``transformers`` model; vLLM's paged KV
cache cannot change sequence length per layer, so this module instead
reproduces the states with a standalone pass over the first K decoder
layers, run at the multimodal-embedding stage over
``[image embeddings, prompt embeddings]``. Selection is then applied
*before* the real forward.

vLLM's decoder layers cannot be called directly here — their
``self.attn`` is the paged-attention op, which requires scheduler-built
metadata for the *batch being served*, not our auxiliary sequence. So
each layer's forward is mirrored functionally: every submodule (norms,
projections, rotary, MLP, MoE, per-layer inputs) is invoked exactly as
the layer's own ``forward`` does, with only the attention op replaced by
a plain masked ``scaled_dot_product_attention``.

Faithfulness notes (vs the official code):

- Key states are post-RoPE, kv-heads NOT repeated for GQA — the L1 norm
  runs over the ``(kv_heads * head_dim)`` flattened keys, matching the
  official reshape of the pre-``repeat_kv`` cached keys.
- Hidden states returned are the layer-(K−1) outputs passed through the
  model's final norm, matching ``self.norm(layer_outputs[0])``.
- Under tensor parallelism the key L1 norm would cover only the local
  rank's kv-head shard (a deviation); attention itself is exact since
  ``o_proj`` all-reduces. Our deployments are single-GPU.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Plain GQA attention over one flat sequence.

    Args:
        q: ``(seq, num_heads * head_dim)``.
        k / v: ``(seq, num_kv_heads * head_dim)``.
        attn_mask: ``(seq, seq)`` bool, True = may attend. ``None``
            falls back to pure causal.
    """
    seq = q.shape[0]
    q = q.view(seq, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k.view(seq, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v = v.view(seq, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    if num_kv_heads != num_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask if attn_mask is not None else None,
        is_causal=attn_mask is None,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1).reshape(seq, num_heads * head_dim)


def _causal_mask(
    seq: int,
    device: torch.device,
    bidirectional_prefix: int = 0,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """Boolean attention mask (True = may attend).

    ``bidirectional_prefix`` marks the leading image block as mutually
    visible (Gemma 4 vision-bidi); ``sliding_window`` additionally bounds
    |i - j| < window, matching the kernel-side clamp the real forward
    applies (``mm_prefix_clamp_sliding_window``). Returns ``None`` when
    pure causal suffices so SDPA can take its fast path.
    """
    if bidirectional_prefix <= 0 and sliding_window is None:
        return None
    idx = torch.arange(seq, device=device)
    allowed = idx.unsqueeze(1) >= idx.unsqueeze(0)  # j <= i
    if bidirectional_prefix > 0:
        img = idx < bidirectional_prefix
        allowed |= img.unsqueeze(1) & img.unsqueeze(0)
    if sliding_window is not None:
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        allowed &= dist < sliding_window
    return allowed


@torch.no_grad()
def dart_prefix_states_llama(
    layers: nn.ModuleList,
    final_norm: nn.Module,
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-K-layers pass for Llama-style stacks (Qwen2, LLaVA/Vicuna).

    Mirrors ``Qwen2DecoderLayer.forward`` / ``LlamaDecoderLayer.forward``
    including the fused-residual RMSNorm pattern, substituting the paged
    attention op with causal SDPA.

    Args:
        layers: the language model's decoder layers (only the first
            ``num_layers`` run).
        final_norm: the model's final RMSNorm (applied to the last
            layer's output, matching the official DART code).
        inputs_embeds: ``(seq, hidden)`` in model dtype.
        positions: ``(seq,)`` or ``(3, seq)`` for M-RoPE (Qwen2.5-VL).
        num_layers: K (official default 2).

    Returns:
        ``(hidden, key_l1)``: final-normed hidden states ``(seq, hidden)``
        and the last aux layer's post-RoPE key L1 norms ``(seq,)``.
    """
    hidden = inputs_embeds
    residual: torch.Tensor | None = None
    key_l1: torch.Tensor | None = None
    for layer in list(layers)[:num_layers]:
        attn = layer.self_attn
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)

        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        if getattr(attn, "qk_norm", False):
            seq = q.shape[0]
            q = attn.q_norm(q.view(seq, attn.num_heads, attn.head_dim)).view(
                seq, attn.q_size
            )
            k = attn.k_norm(k.view(seq, attn.num_kv_heads, attn.head_dim)).view(
                seq, attn.kv_size
            )
        q, k = attn.rotary_emb(positions, q, k)
        key_l1 = torch.norm(k.float(), p=1, dim=-1)
        attn_out = _sdpa(
            q,
            k,
            v,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.scaling,
            attn_mask=None,
        )
        hidden, _ = attn.o_proj(attn_out)

        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)

    assert key_l1 is not None
    hidden, _ = final_norm(hidden, residual)
    return hidden, key_l1


@torch.no_grad()
def dart_prefix_states_gemma(
    layers: nn.ModuleList,
    final_norm: nn.Module,
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    per_layer_inputs: torch.Tensor | None,
    num_layers: int,
    num_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-K-layers pass for Gemma 4 stacks.

    Mirrors ``Gemma4DecoderLayer.forward`` — Gemma's unfused residual
    pattern, q/k/v per-head norms, optional MoE block, per-layer-input
    (PLE) injection and the layer scalar — substituting the paged
    attention op with masked SDPA. The mask reproduces Gemma 4's
    vision-bidi semantics: image tokens attend bidirectionally within
    the image block, everything else is causal, and sliding-attention
    layers additionally bound |i − j| by the window (the
    ``mm_prefix_clamp_sliding_window`` clamp).

    Args:
        per_layer_inputs: ``(seq, num_layers_total, ple_dim)`` already
            projected (``project_per_layer_inputs``), or ``None`` for
            variants without PLE.
        num_image_tokens: length of the leading image block for the
            bidirectional mask.
    """
    hidden = inputs_embeds
    key_l1: torch.Tensor | None = None
    for layer_idx, layer in enumerate(list(layers)[:num_layers]):
        attn = layer.self_attn

        residual = hidden
        hidden = layer.input_layernorm(residual)

        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        seq = q.shape[0]
        q = attn.q_norm(q.view(seq, attn.num_heads, attn.head_dim)).view(
            seq, attn.q_size
        )
        k = attn.k_norm(k.view(seq, attn.num_kv_heads, attn.head_dim)).view(
            seq, attn.kv_size
        )
        v = attn.v_norm(v.view(seq, attn.num_kv_heads, attn.head_dim)).view(
            seq, attn.kv_size
        )
        q, k = attn.rotary_emb(positions, q, k)
        key_l1 = torch.norm(k.float(), p=1, dim=-1)

        # Gemma4Attention keeps its config; sliding layers bound the
        # attended distance by config.sliding_window.
        window = attn.config.sliding_window if attn.is_sliding else None
        mask = _causal_mask(
            seq,
            hidden.device,
            bidirectional_prefix=num_image_tokens,
            sliding_window=window,
        )
        attn_out = _sdpa(
            q,
            k,
            v,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.scaling,
            attn_mask=mask,
        )
        hidden, _ = attn.o_proj(attn_out)

        hidden = layer.post_attention_layernorm(hidden)
        hidden = hidden + residual
        residual = hidden

        mlp_in = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(mlp_in)

        if getattr(layer, "enable_moe_block", False):
            hidden_1 = layer.post_feedforward_layernorm_1(hidden)
            hidden_2 = layer.pre_feedforward_layernorm_2(residual)
            router_logits = layer.router(residual)
            hidden_2 = layer.moe(hidden_2, router_logits)
            hidden_2 = layer.post_feedforward_layernorm_2(hidden_2)
            hidden = hidden_1 + hidden_2

        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = hidden + residual

        if (
            per_layer_inputs is not None
            and layer.per_layer_input_gate is not None
        ):
            gate = layer.per_layer_input_gate(hidden)
            gate = F.gelu(gate, approximate="tanh")
            gated = gate * per_layer_inputs[:, layer_idx, :]
            contribution = layer.per_layer_projection(gated)
            contribution = layer.post_per_layer_input_norm(contribution)
            hidden = hidden + contribution

        hidden = hidden * layer.layer_scalar

    assert key_l1 is not None
    hidden = final_norm(hidden)
    return hidden, key_l1


def qwen2_5_vl_dart_positions(
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    num_text_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """M-RoPE positions ``(3, seq)`` for the aux sequence (image, text).

    Follows Qwen2.5-VL's multimodal rope: image tokens get (t, h, w)
    grid indices over the merged grid; text tokens get all three
    sections equal, continuing after the image block's max position.
    """
    t, h, w = grid_thw
    grid_h = h // spatial_merge_size
    grid_w = w // spatial_merge_size
    t_idx = (
        torch.arange(t, device=device)
        .view(t, 1)
        .expand(t, grid_h * grid_w)
        .reshape(-1)
    )
    h_idx = (
        torch.arange(grid_h, device=device)
        .view(1, grid_h, 1)
        .expand(t, grid_h, grid_w)
        .reshape(-1)
    )
    w_idx = (
        torch.arange(grid_w, device=device)
        .view(1, 1, grid_w)
        .expand(t, grid_h, grid_w)
        .reshape(-1)
    )
    image_pos = torch.stack([t_idx, h_idx, w_idx])  # (3, n_img)
    text_start = int(image_pos.max()) + 1
    text_pos = (
        torch.arange(text_start, text_start + num_text_tokens, device=device)
        .view(1, -1)
        .expand(3, -1)
    )
    return torch.cat([image_pos, text_pos], dim=1)
