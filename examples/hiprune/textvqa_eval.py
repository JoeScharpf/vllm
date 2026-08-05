"""TextVQA benchmark for HiPrune/HyDART token pruning in vLLM.

TextVQA is the OCR stress test for token pruning: questions require
reading text in the image, so pruning away "redundant-looking" patches
that actually contain small text is directly punished.

Runs a fixed-seed subset of the TextVQA validation split (from the
lmms-lab/textvqa Hugging Face dataset; 10 human answers per question)
against a vLLM server at several `token_pruning` ratios and reports,
per ratio:

- VQA soft accuracy: per question, acc = min(1, matches / 3), where
  matches counts the human answers equal to the model answer after
  standard VQA normalization (lowercase, strip articles/punctuation)
- mean prompt tokens and a small serial TTFT pass

Decoding is greedy (temperature 0) with the standard short-answer
prompt suffix ("Answer the question using a single word or phrase.").

Outputs into --out-dir:
- textvqa_results.jsonl — one line per request
- textvqa_summary.json  — per-ratio aggregates
- textvqa_summary.txt   — human-readable table

Usage (next to the server):
    python3 textvqa_eval.py --url http://localhost:8124 \
        --model Qwen/Qwen2.5-VL-3B-Instruct --num-samples 1000 \
        --ratios 1.0 0.5 0.3 0.14 --concurrency 8 --out-dir ./textvqa_out

Requires: datasets, pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")
DIGIT_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
CONTRACTIONS = {
    "cant": "can't", "dont": "don't", "isnt": "isn't", "wont": "won't",
    "wouldnt": "wouldn't", "couldnt": "couldn't", "aint": "ain't",
}


def vqa_normalize(text: str) -> str:
    """The standard VQA answer normalization (abridged: punctuation,
    articles, digit words, common contractions)."""
    t = text.strip().lower()
    t = PUNCT.sub(" ", t)
    words = []
    for w in t.split():
        w = DIGIT_MAP.get(w, w)
        w = CONTRACTIONS.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(words)


def vqa_accuracy(model_answer: str, human_answers: list[str]) -> float:
    norm = vqa_normalize(model_answer)
    matches = sum(vqa_normalize(h) == norm for h in human_answers)
    return min(1.0, matches / 3.0)


def encode_image(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_textvqa_subset(num_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    print("loading lmms-lab/textvqa (validation split)...")
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:num_samples])

    samples = []
    for i in idxs:
        row = ds[i]
        samples.append({
            "id": str(row["question_id"]),
            "question": row["question"].strip(),
            "answers": [a.strip() for a in row["answers"]],
            "image_b64": encode_image(row["image"]),
        })
    rng.shuffle(samples)
    print(f"sampled {len(samples)} of {len(ds)} validation questions "
          f"(seed {seed})")
    return samples


def build_body(sample: dict, model: str, ratio: float | None,
               stream: bool = False) -> dict:
    body: dict = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{sample['image_b64']}"}},
                {"type": "text",
                 "text": f"{sample['question']}\nAnswer the question "
                         "using a single word or phrase."},
            ],
        }],
        "max_tokens": 20,
        "temperature": 0,
    }
    if ratio is not None and ratio < 1.0:
        body["token_pruning"] = ratio
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        body["cache_salt"] = uuid.uuid4().hex
    return body


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def measure_ttft(url: str, body: dict) -> float:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8124")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[1.0, 0.5, 0.3, 0.14])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timing-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="./textvqa_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_textvqa_subset(args.num_samples, args.seed)

    results_path = out_dir / "textvqa_results.jsonl"
    results_f = open(results_path, "w")
    summaries = []

    for ratio in args.ratios:
        label = "baseline" if ratio >= 1.0 else f"{ratio:.2f}"
        print(f"\n=== retention {label}: accuracy pass "
              f"({len(samples)} questions, concurrency {args.concurrency}) ===")

        def run_one(sample: dict, ratio: float = ratio) -> dict:
            body = build_body(sample, args.model,
                              None if ratio >= 1.0 else ratio)
            t0 = time.perf_counter()
            resp = post_json(args.url, body)
            elapsed = time.perf_counter() - t0
            raw = resp["choices"][0]["message"]["content"]
            acc = vqa_accuracy(raw, sample["answers"])
            return {
                "id": sample["id"],
                "ratio": ratio,
                "question": sample["question"],
                "human_answers": sample["answers"],
                "raw_answer": raw,
                "vqa_acc": acc,
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
                "latency_s": round(elapsed, 4),
            }

        t_pass = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(run_one, samples))
        pass_time = time.perf_counter() - t_pass
        for rec in records:
            results_f.write(json.dumps(rec) + "\n")
        results_f.flush()

        n = len(records)
        mean_acc = sum(r["vqa_acc"] for r in records) / n
        exact_zero = sum(r["vqa_acc"] == 0.0 for r in records)
        print(f"vqa accuracy {mean_acc:.3f} | "
              f"zero-credit answers {exact_zero}/{n} | "
              f"pass took {pass_time:.0f}s")

        print(f"=== retention {label}: timing pass "
              f"({args.timing_samples} serial streaming requests) ===")
        ttfts = []
        for sample in samples[:args.timing_samples]:
            body = build_body(sample, args.model,
                              None if ratio >= 1.0 else ratio, stream=True)
            ttfts.append(measure_ttft(args.url, body))
        prompt_tokens = [r["prompt_tokens"] for r in records
                         if r["prompt_tokens"] is not None]

        summaries.append({
            "ratio": ratio,
            "num_samples": n,
            "vqa_accuracy": mean_acc,
            "zero_credit": exact_zero,
            "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
            "mean_ttft_s": statistics.mean(ttfts),
            "median_ttft_s": statistics.median(ttfts),
            "accuracy_pass_seconds": pass_time,
        })

    results_f.close()
    summary_path = out_dir / "textvqa_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": "lmms-lab/textvqa (validation)",
            "model": args.model,
            "num_samples": len(samples),
            "seed": args.seed,
            "results": summaries,
        }, f, indent=2)

    header = (f"{'retention':>10} {'vqa acc':>8} {'zero-credit':>12} "
              f"{'prompt tok':>11} {'mean TTFT':>10}")
    lines = [
        f"TextVQA | {args.model} | {len(samples)} validation samples",
        "", header, "-" * len(header),
    ]
    for s in summaries:
        label = "baseline" if s["ratio"] >= 1.0 else f"{s['ratio']:.2f}"
        lines.append(
            f"{label:>10} {s['vqa_accuracy']:>8.3f} "
            f"{s['zero_credit']:>12} {s['mean_prompt_tokens']:>11.1f} "
            f"{s['mean_ttft_s']:>9.3f}s")
    table = "\n".join(lines)
    with open(out_dir / "textvqa_summary.txt", "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {results_path}, {summary_path}, "
          f"{out_dir / 'textvqa_summary.txt'}")


if __name__ == "__main__":
    main()
