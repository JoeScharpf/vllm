"""Render a HiPrune overlay from a vLLM chat response.

Reads the original image and the pruning data returned by the vLLM
server and draws it over the image: pruned 48x48 px cells are darkened
and kept cells are outlined by HiPrune category — anchors (red),
buffers (orange), registers (green).

Prefers the `token_pruning_metadata` response field (which carries the
grid and token categories); falls back to `pruned_token_indices` (kept
cells all green, grid reconstructed from the image size).

Alongside the overlay (<output>.png), also writes readable artifacts
next to it:

- <output>.metadata.json — the pruning metadata, pretty-printed
- <output>.metadata.jsonl — one compact line per image (batch-friendly)
- <output>.report.txt — a human-readable summary of answer + statistics

Optional flags enrich the report:

- --baseline baseline.json — the same request sent without
  token_pruning; the report then shows both answers side by side
- --request request.json — the request body that was sent; the report
  then starts with the prompt and the request settings
- --timing timing.json — output of benchmark.py; the report then ends
  with a latency table (prefill/TTFT, decode tok/s, total per ratio)

Usage:
    python3 visualize_pruned.py <image> <response.json> <output.png> \
        [--baseline baseline.json] [--request request.json] \
        [--timing timing.json]
"""

import argparse
import json
from pathlib import Path

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


def extract_answer(resp: dict) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    return ((choices[0].get("message") or {}).get("content") or "").strip()


def extract_prompt(request: dict) -> str:
    """Pull the text parts out of the request messages (skipping images)."""
    parts: list[str] = []
    for msg in request.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [p.get("text", "") for p in content if p.get("type") == "text"]
    return " ".join(p for p in parts if p).strip()


def timing_lines(timing: dict) -> list[str]:
    """Format benchmark.py output as a report section."""
    lines = ["", "=== LATENCY (from benchmark.py, streaming) ==="]
    header = (f"{'retention':>10} {'prompt tok':>11} {'prefill/TTFT':>13} "
              f"{'decode tok/s':>13} {'total':>8}")
    lines += [header, "-" * len(header)]
    for r in timing.get("results", []):
        label = "baseline" if r["retention"] >= 1.0 else f"{r['retention']:.2f}"
        lines.append(
            f"{label:>10} {r['prompt_tokens']:>11} "
            f"{r['ttft_s']:>12.3f}s {r['decode_tok_s']:>13.1f} "
            f"{r['total_s']:>7.2f}s"
        )
    lines.append(
        "note: TTFT includes the vision encoder and network latency "
        "(identical across rows); compare TTFT and decode tok/s, not totals."
    )
    return lines


def write_reports(
    resp: dict,
    out_path: str,
    baseline: dict | None = None,
    request: dict | None = None,
    timing: dict | None = None,
) -> list[str]:
    """Write pretty/JSONL metadata and a plain-text report next to the
    overlay. Returns the list of files written."""
    stem = Path(out_path).with_suffix("")
    all_md = resp.get("token_pruning_metadata") or []
    written: list[str] = []

    if any(md is not None for md in all_md):
        pretty_path = f"{stem}.metadata.json"
        with open(pretty_path, "w") as f:
            json.dump(all_md, f, indent=2)
            f.write("\n")
        written.append(pretty_path)

        jsonl_path = f"{stem}.metadata.jsonl"
        with open(jsonl_path, "w") as f:
            for i, md in enumerate(all_md):
                f.write(json.dumps({"image": i, **(md or {})}) + "\n")
        written.append(jsonl_path)

    usage = resp.get("usage") or {}

    lines: list[str] = []
    if request is not None:
        lines.append(f"prompt      : {extract_prompt(request)}")
        settings = [f"model {request['model']}"] if "model" in request else []
        for key in ("max_tokens", "token_pruning", "temperature"):
            if key in request:
                settings.append(f"{key} {request[key]}")
        if settings:
            lines.append(f"settings    : {' | '.join(settings)}")
        lines.append("")
    if baseline is not None:
        lines += ["=== BASELINE ANSWER (no pruning) ===", extract_answer(baseline), ""]
        b_usage = baseline.get("usage") or {}
        if b_usage:
            lines.append(
                f"tokens      : prompt {b_usage.get('prompt_tokens')} | "
                f"completion {b_usage.get('completion_tokens')}"
            )
        lines += ["", "=== PRUNED ANSWER ==="]
        lines += [extract_answer(resp), ""]
    else:
        lines += [f"answer      : {extract_answer(resp)}", ""]
    if usage:
        lines.append(
            f"tokens      : prompt {usage.get('prompt_tokens')} | "
            f"completion {usage.get('completion_tokens')}"
        )
    for i, md in enumerate(all_md):
        if md is None:
            lines.append(f"image {i}     : not pruned")
            continue
        n = md["num_tokens"]
        n_pruned = len(md["pruned"])
        n_kept = n - n_pruned
        lines += [
            f"image {i}",
            f"  grid      : {md['grid'][0]} x {md['grid'][1]}  ({n} soft tokens)",
            f"  retention : {md['retention']:.4f}  "
            f"(object layer {md['object_layer']}, alpha {md['alpha']})",
            f"  kept      : {n_kept} ({100 * n_kept / n:.1f}%)   "
            f"pruned: {n_pruned} ({100 * n_pruned / n:.1f}%)",
            f"    anchors   : {len(md['anchors'])}  {md['anchors']}",
            f"    buffers   : {len(md['buffers'])}  {md['buffers']}",
            f"    registers : {len(md['registers'])}",
        ]
        for layer in ("object_layer", "deep_layer"):
            ma = md["mean_attention"][layer]
            cells = " | ".join(
                f"{k} {v:.4g}" for k, v in ma.items() if v is not None
            )
            lines.append(f"  mean attention ({layer.replace('_', ' ')}): {cells}")
    if timing is not None:
        lines += timing_lines(timing)
    report_path = f"{stem}.report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    written.append(report_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="original image sent in the request")
    parser.add_argument("response", help="pruned chat completion response JSON")
    parser.add_argument("output", help="overlay PNG to write")
    parser.add_argument(
        "baseline_pos", nargs="?", default=None,
        help="(deprecated positional) same as --baseline",
    )
    parser.add_argument("--baseline", help="baseline (unpruned) response JSON")
    parser.add_argument("--request", help="request body JSON that was sent")
    parser.add_argument("--timing", help="benchmark.py output JSON")
    args = parser.parse_args()

    image_path, resp_path, out_path = args.image, args.response, args.output
    baseline = None
    baseline_path = args.baseline or args.baseline_pos
    if baseline_path:
        with open(baseline_path) as f:
            baseline = json.load(f)
    request = None
    if args.request:
        with open(args.request) as f:
            request = json.load(f)
    timing = None
    if args.timing:
        with open(args.timing) as f:
            timing = json.load(f)

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

    for path in write_reports(resp, out_path, baseline, request, timing):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
