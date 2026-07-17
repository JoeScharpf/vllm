"""Streaming latency benchmark for HiPrune token pruning in vLLM.

Sends the same image + prompt at several `token_pruning` ratios using
streaming requests and measures, per ratio:

- prefill / TTFT: time from request start to the first content chunk
  (includes the vision encoder and any network latency, which is the
  same for every run)
- decode tok/s: completion tokens after the first divided by the time
  between the first and last chunk
- total: request start to stream end

A warm-up request runs first so model warm-up does not pollute the
first measurement. Results print as a Colab-style table and are saved
to a JSON file that visualize_pruned.py can fold into its report via
--timing.

Usage:
    python3 benchmark.py <image> --prompt "Describe this image." \
        --url http://localhost:8123 --ratios 1.0 0.5 0.3 0.14 \
        --max-tokens 100 --out timing.json

Only needs the standard library (urllib for HTTP).
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
import uuid
from pathlib import Path


def build_request(
    image_path: str, prompt: str, model: str, max_tokens: int,
    ratio: float | None,
) -> dict:
    suffix = Path(image_path).suffix.lower().lstrip(".") or "jpeg"
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    body: dict = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        # Greedy decoding so answers are deterministic and comparable
        # across ratios.
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Defeat the prefix cache so every run pays the full prefill cost;
        # otherwise repeats of a previously seen (image, ratio) pair report
        # near-zero TTFT.
        "cache_salt": uuid.uuid4().hex,
    }
    if ratio is not None and ratio < 1.0:
        body["token_pruning"] = ratio
    return body


def run_streaming(url: str, body: dict) -> dict:
    """Send one streaming chat completion and time the chunks."""
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    text_parts: list[str] = []
    usage: dict = {}
    t_start = time.perf_counter()
    t_first = None
    t_last = t_start
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                text_parts.append(delta)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    completion_tokens = usage.get("completion_tokens", 0)
    decode_time = t_last - t_first
    decode_tps = (
        (completion_tokens - 1) / decode_time
        if completion_tokens > 1 and decode_time > 0 else 0.0
    )
    return {
        "ttft_s": t_first - t_start,
        "decode_tok_s": decode_tps,
        "total_s": t_end - t_start,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "answer": "".join(text_parts).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="image file to send")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--url", default="http://localhost:8123")
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.14],
        help="token_pruning ratios; 1.0 = baseline (no pruning)",
    )
    parser.add_argument("--out", help="save results as JSON for --timing")
    args = parser.parse_args()

    print("warm-up request...")
    run_streaming(args.url, build_request(
        args.image, args.prompt, args.model, 8, None))

    results = []
    for ratio in args.ratios:
        label = "baseline" if ratio >= 1.0 else f"{ratio:.2f}"
        print(f"running retention {label}...")
        body = build_request(
            args.image, args.prompt, args.model, args.max_tokens,
            None if ratio >= 1.0 else ratio)
        r = run_streaming(args.url, body)
        r["retention"] = ratio
        results.append(r)

    header = (f"{'retention':>10} {'prompt tok':>11} {'prefill/TTFT':>13} "
              f"{'decode tok/s':>13} {'total':>8}   answer (start)")
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        label = "baseline" if r["retention"] >= 1.0 else f"{r['retention']:.2f}"
        first_line = r["answer"].split("\n")[0][:60]
        print(f"{label:>10} {r['prompt_tokens']:>11} "
              f"{r['ttft_s']:>12.3f}s {r['decode_tok_s']:>13.1f} "
              f"{r['total_s']:>7.2f}s   {first_line}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "prompt": args.prompt,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "results": results,
            }, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
