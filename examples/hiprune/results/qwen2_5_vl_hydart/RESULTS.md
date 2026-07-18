# HyDART on Qwen2.5-VL-3B-Instruct — benchmark results

HyDART keeps HiPrune's anchor+buffer stage (object-layer attention) and
replaces the deep-layer register stage with greedy MMR over the merged
visual embeddings (cosine-similarity duplication avoidance, DART-style).
Enabled with `HIPRUNE_METHOD=hydart`; defaults `HYDART_LAMBDA_SEED=0.1`,
`HYDART_LAMBDA_PICK=0.5`.

All numbers measured on an RTX A6000 (safeai-gpu3), serving
`Qwen/Qwen2.5-VL-3B-Instruct` with `vllm serve --enable-hiprune`,
streaming requests with prefix caching disabled via `cache_salt`
(cold-start prefill every time), `temperature: 0`. The HiPrune columns
are the same-protocol run from `../qwen2_5_vl/RESULTS.md`.

## POPE accuracy vs speed

`pope_eval.py` on the same 396 balanced POPE samples (adversarial /
popular / random) as the HiPrune run. Raw log: `pope_summary.txt`.

| retention | HiPrune acc | HyDART acc | HiPrune F1 | HyDART F1 | invalid (HiP → HyD) |
|-----------|-------------|------------|------------|-----------|---------------------|
| baseline  | 0.886       | 0.884      | 0.877      | 0.875     | 0 → 0               |
| 0.50      | 0.859       | 0.856      | 0.848      | 0.846     | 10 → 9              |
| 0.30      | 0.833       | **0.841**  | 0.828      | **0.839** | 17 → 16             |
| 0.14      | 0.699       | **0.765**  | 0.710      | **0.780** | 57 → 35             |

- Baseline rows differ by 1 answer (0.886 vs 0.884): the ratio-1.0 path
  is identical code; the difference is batching nondeterminism.
- At moderate pruning HyDART matches HiPrune (0.50) or edges past it
  (0.30, +0.8 points).
- At aggressive pruning (0.14) HyDART is **+6.6 points** accuracy and
  produces 35 unparseable answers instead of 57 (of 396): replacing
  low-attention deep-layer registers with embedding-diverse tokens
  preserves substantially more usable signal when the budget is tight.

![accuracy comparison](pope_methods_accuracy.png)

![accuracy vs TTFT](pope_methods_tradeoff.png)

## Lambda mini-grid

POPE at 0.30 / 0.14 retention per (λ_seed, λ_pick) config, same 396
samples. Raw logs: `pope_summary_ls*_lp*.txt`.

| λ_seed | λ_pick | acc @ 0.30 | acc @ 0.14 | note |
|--------|--------|------------|------------|------|
| 0.0    | 0.0    | 0.838      | 0.753      | attention-only fill (no diversity) |
| 0.0    | 0.5    | 0.838      | 0.760      | diversity among picks only |
| 0.1    | 0.5    | **0.841**  | **0.765**  | **default** |
| 0.1    | 1.0    | 0.841      | 0.753      | over-penalizing redundancy hurts at 0.14 |

The default (0.1, 0.5) is the best cell at both ratios. The spread is
small at 0.30 (0.3 points) and meaningful at 0.14 (1.2 points over
attention-only) — and even the worst HyDART cell beats HiPrune's 0.699
at 0.14 by 5+ points, so most of the gain comes from swapping registers
for embedding-based selection, with the MMR penalties adding a smaller
refinement on top.

## Latency vs image size

`benchmark.py` streaming latency on the pyramids photo at three input
resolutions ("Describe this image in detail.", 100 completion tokens).
Raw logs: `timing_hydart_pyramids*.json`. TTFT speedup is HiPrune TTFT
divided by HyDART TTFT at the same ratio (higher = HyDART faster).

| image tokens (baseline) | retention | HyDART TTFT | HiPrune TTFT | speedup |
|------------------------|-----------|-------------|--------------|---------|
| 1,847                  | 0.30      | 0.40 s      | 0.34 s       | 0.86x   |
| 4,083                  | 0.30      | 0.78 s      | 0.84 s       | 1.07x   |
| 16,251                 | 0.30      | 5.29 s      | 6.69 s       | 1.27x   |
| 16,251                 | 0.14      | 4.94 s      | 6.52 s       | 1.32x   |

HyDART needs only one dense O(S²) score capture (the object layer)
instead of two, so its selection overhead is roughly half of HiPrune's.
The saving grows with image size — 1.3x faster TTFT than HiPrune on the
16k-token image — but the remaining single capture still dominates
cold-start TTFT versus no pruning at all (see the HiPrune results for
why: the dense pass costs more than the LLM-prefill saving on one-shot
requests). On small images the MMR loop's sequential picks make HyDART
slightly slower than HiPrune; above ~2048 tokens the loop switches to
block picks of 8 which keeps it bounded. As with HiPrune, the wins are
prompt/KV budget at every ratio, and TTFT for repeated (encoder-cached)
or token-heavy inputs.

## Answer fidelity spot check

`visualize_pruned.py` overlay for `token_pruning: 0.3` on the dog photo
(986 soft tokens → 296 kept: 6 anchors, 20 buffers, 270 diverse). Full
report: `dog_overlay_0.3.report.txt`.

- Answer: "A fluffy white puppy is sitting on a moss-covered log,
  looking directly at the camera with its ears perked up."
  (322 vs 1,012 baseline prompt tokens)
- Metadata: diverse tokens averaged 0.746 cosine redundancy at
  selection; pruned tokens ended at 0.791 max similarity to the kept
  set — i.e. what was dropped is well covered by what was kept.

![dog overlay](dog_overlay_0.3.png)

Left: original. Right: kept patches — anchors (red), buffers (orange),
diverse (blue); pruned patches dimmed.
