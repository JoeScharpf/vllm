"""Render a HiPrune overlay from a vLLM chat response.

Reads the original image and the pruning data returned by the vLLM
server and draws it over the image: pruned 48x48 px cells are darkened
and kept cells are outlined by HiPrune category — anchors (red),
buffers (orange), registers (green).

Prefers the `token_pruning_metadata` response field (which carries the
grid and token categories); falls back to `pruned_token_indices` (kept
cells all green, grid reconstructed from the image size).

Usage:
    python3 visualize_pruned.py <image> <response.json> <output.png>
"""

import json
import sys

from PIL import Image, ImageDraw

POOL_K = 3      # Gemma pools 3x3 patches into one soft token
PATCH = 16      # ViT patch size in px
CELL = POOL_K * PATCH  # 48 px of resized image per soft token
MAX_SOFT_TOKENS = 280

ANCHOR_COLOR = (255, 60, 60, 255)
BUFFER_COLOR = (255, 170, 40, 255)
REGISTER_COLOR = (50, 255, 80, 255)
PRUNED_FILL = (0, 0, 0, 210)


def gemma_grid(width: int, height: int) -> tuple[int, int]:
    """Soft-token grid (grid_w, grid_h) for an image, per Gemma 4's
    processor: aspect-preserving resize to multiples of 48 px, capped at
    MAX_SOFT_TOKENS soft tokens (mild aspect flooring)."""
    import math

    scale = math.sqrt(MAX_SOFT_TOKENS * CELL * CELL / (width * height))
    scale = min(scale, 1.0) if width * height > MAX_SOFT_TOKENS * CELL * CELL else scale
    grid_w = max(1, int(width * scale // CELL))
    grid_h = max(1, int(height * scale // CELL))
    while grid_w * grid_h > MAX_SOFT_TOKENS:
        if grid_w >= grid_h:
            grid_w -= 1
        else:
            grid_h -= 1
    return grid_w, grid_h


def main() -> None:
    image_path, resp_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    img = Image.open(image_path).convert("RGB")
    with open(resp_path) as f:
        resp = json.load(f)

    metadata = (resp.get("token_pruning_metadata") or [None])[0]
    if metadata is not None:
        grid_w, grid_h = metadata["grid"]
        pruned = metadata["pruned"]
        categories = {
            "anchor": (metadata["anchors"], ANCHOR_COLOR),
            "buffer": (metadata["buffers"], BUFFER_COLOR),
            "register": (metadata["registers"], REGISTER_COLOR),
        }
    else:
        grid_w, grid_h = gemma_grid(*img.size)
        pruned = resp["pruned_token_indices"][0]
        kept = sorted(set(range(grid_w * grid_h)) - set(pruned))
        categories = {"kept": (kept, REGISTER_COLOR)}

    n_soft = grid_w * grid_h
    resized = img.resize((grid_w * CELL, grid_h * CELL), Image.BICUBIC)

    def cell_box(idx: int) -> list[int]:
        r, c = idx // grid_w, idx % grid_w
        return [c * CELL, r * CELL, c * CELL + CELL, r * CELL + CELL]

    overlay = resized.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for idx in pruned:
        draw.rectangle(cell_box(idx), fill=PRUNED_FILL)
    for _, (indices, color) in categories.items():
        for idx in indices:
            draw.rectangle(cell_box(idx), outline=color, width=2)

    gap = 24
    canvas = Image.new(
        "RGB", (resized.width * 2 + gap, resized.height + 56), (255, 255, 255)
    )
    canvas.paste(resized, (0, 56))
    canvas.paste(overlay, (resized.width + gap, 56))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), f"original (resized {resized.width}x{resized.height})", fill=(0, 0, 0))
    n_kept = n_soft - len(pruned)
    d.text(
        (resized.width + gap + 8, 10),
        f"HiPrune: {len(pruned)}/{n_soft} tokens pruned, {n_kept} kept",
        fill=(0, 0, 0),
    )
    if metadata is not None:
        d.text(
            (resized.width + gap + 8, 28),
            f"anchors {len(metadata['anchors'])} (red) | "
            f"buffers {len(metadata['buffers'])} (orange) | "
            f"registers {len(metadata['registers'])} (green)",
            fill=(0, 0, 0),
        )
    canvas.save(out_path)
    print(f"grid {grid_w}x{grid_h} = {n_soft} soft tokens; "
          f"{len(pruned)} pruned, {n_kept} kept -> {out_path}")
    if metadata is not None:
        ma = metadata["mean_attention"]
        print("mean attention (object layer):",
              {k: (round(v, 6) if v is not None else None)
               for k, v in ma["object_layer"].items()})
        print("mean attention (deep layer)  :",
              {k: (round(v, 6) if v is not None else None)
               for k, v in ma["deep_layer"].items()})


if __name__ == "__main__":
    main()
