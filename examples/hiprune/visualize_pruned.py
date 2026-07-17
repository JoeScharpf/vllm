"""Render a HiPrune overlay from a vLLM chat response.

Reads the original image and the `pruned_token_indices` returned by the
vLLM server, maps each pruned soft-token index onto Gemma 4's 48x48 px
grid, and saves a side-by-side figure: original | pruned overlay.

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
    pruned = resp["pruned_token_indices"][0]

    grid_w, grid_h = gemma_grid(*img.size)
    n_soft = grid_w * grid_h
    resized = img.resize((grid_w * CELL, grid_h * CELL), Image.BICUBIC)

    overlay = resized.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for idx in pruned:
        r, c = idx // grid_w, idx % grid_w
        x0, y0 = c * CELL, r * CELL
        draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], fill=(0, 0, 0, 210))
    # Outline kept cells so they pop.
    kept = sorted(set(range(n_soft)) - set(pruned))
    for idx in kept:
        r, c = idx // grid_w, idx % grid_w
        x0, y0 = c * CELL, r * CELL
        draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], outline=(50, 255, 80, 255), width=2)

    gap = 24
    canvas = Image.new(
        "RGB", (resized.width * 2 + gap, resized.height + 40), (255, 255, 255)
    )
    canvas.paste(resized, (0, 40))
    canvas.paste(overlay, (resized.width + gap, 40))
    d = ImageDraw.Draw(canvas)
    d.text((8, 12), f"original (resized {resized.width}x{resized.height})", fill=(0, 0, 0))
    d.text(
        (resized.width + gap + 8, 12),
        f"HiPrune: {len(pruned)}/{n_soft} tokens pruned, {len(kept)} kept (green)",
        fill=(0, 0, 0),
    )
    canvas.save(out_path)
    print(f"grid {grid_w}x{grid_h} = {n_soft} soft tokens; "
          f"{len(pruned)} pruned, {len(kept)} kept -> {out_path}")


if __name__ == "__main__":
    main()
