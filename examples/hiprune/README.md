# HiPrune demo: visualize pruned patches from a served response

`visualize_pruned.py` renders the pruning data returned by a vLLM server
running Gemma 4 with HiPrune (see the `token_pruning` request field) as
an overlay on the original image: pruned 48x48 px cells are darkened and
kept cells are outlined by HiPrune category — anchors (red), buffers
(orange), registers (green).

Responses carry two pruning fields:

- `pruned_token_indices` — per image, the soft-token indices dropped.
- `token_pruning_metadata` — per image, the full statistics: grid
  dimensions, anchor/buffer/register/pruned index sets, and mean
  attention per category at the object and deep encoder layers.

## Usage

1. Serve the model:

   ```bash
   VLLM_USE_V2_MODEL_RUNNER=0 vllm serve google/gemma-4-e4b-it --max-model-len 8192
   ```

2. Send a request with `token_pruning` and save the JSON response:

   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemma-4-e4b-it",
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 image.jpg)"'"}},
         {"type": "text", "text": "What breed of dog is this?"}
       ]}],
       "max_tokens": 60,
       "token_pruning": 0.14
     }' > response.json
   ```

3. Render the overlay:

   ```bash
   python3 visualize_pruned.py image.jpg response.json overlay.png
   ```

   Alongside `overlay.png` this also writes readable artifacts:

   - `overlay.metadata.json` — the pruning metadata, pretty-printed
   - `overlay.metadata.jsonl` — one compact line per image (batch-friendly)
   - `overlay.report.txt` — human-readable summary of the answer, token
     counts, category breakdown, and mean attention per category

   Optional flags enrich the report: pass `--baseline` with a response
   to the same request sent without `token_pruning` to show the
   baseline and pruned answers side by side, and `--request` with the
   request body JSON to prepend the prompt and request settings:

   ```bash
   python3 visualize_pruned.py image.jpg response.json overlay.png \
       --baseline baseline.json --request request.json
   ```

## Latency benchmark

`benchmark.py` sweeps `token_pruning` ratios over the same image and
prompt using streaming requests, measuring per ratio the prefill/TTFT
(includes the vision encoder), decode tok/s, and total time. It decodes
greedily (`temperature 0`) so answers are comparable across ratios, and
sets a random `cache_salt` per request so the prefix cache never hides
the prefill cost:

```bash
python3 benchmark.py image.jpg --prompt "Describe this image." \
    --url http://localhost:8000 --ratios 1.0 0.5 0.3 0.14 \
    --max-tokens 100 --out timing.json
```

Pass the resulting `timing.json` to `visualize_pruned.py --timing` to
append the latency table to the report.

## Example output

`pruned_overlay.png` was produced from a real serving run at
`token_pruning: 0.14`: 232 of 270 soft tokens pruned, 38 kept (1 anchor,
4 buffers, 33 registers), and the model still answered "Golden
Retriever".

![example overlay](pruned_overlay.png)
