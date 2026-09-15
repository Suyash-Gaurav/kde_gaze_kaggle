# Meeting notes — T4 KDE-Gaze KV probe

## Goal

Test the living-Gaussian / KDE-gaze claim on a real CoT model that fits a Kaggle T4,
without lying about HuggingFace compatibility.

Question the run is allowed to answer:

> On decode-time GSM8K traces, does a park+fovea+crawl *token* cache keep
> plan tokens and task accuracy better than H2O / SnapKV / StreamingLLM at
> the same token budget?

## Non-goals (this probe)

- 7B LongBench sweep
- FlashAttention / custom CUDA
- Replacing softmax with a virtual Gaussian mixture inside attention
- Claiming the original `Cache.update → G virtual tokens` design works
  with `model.generate` (it does not: RoPE, causal mask, and seq length
  all assume real token slots)

## Compatibility constraint

All methods **index-select real K/V** on a `DynamicCache` (or a tensor
mirror of it) after each prefill/decode step. Attention stays stock SDPA.

A gaze is a *policy over which token slots survive*, plus optional
mean-pool of a crawl span into one survivor slot at `round(μ)`.
It is not a second attention kernel.

## Model / data

- Primary: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, 4-bit NF4, SDPA
- Optional harder pass (only after 1.5B pipeline is green):
  `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` 4-bit
- GSM8K test, **fixed index list** `range(N)` with seed 42
- Greedy decode (`do_sample=False`)
- Pin `revision` strings in the run config dump

## Methods (matched token budget B)

| name | keep set |
|---|---|
| full | all tokens |
| stream | first `sink` + last `B-sink` |
| snap | prefill: last window scores the prefix; decode: same + recent |
| h2o | recent `R` + top cumulative last-token attention proxy |
| gaze-oracle | recent `R` + tokens in labeled plan span (never evict) + crawl reps |
| gaze-heur | recent `R` + residual-spike parks + crawl reps |

Budget `B = max(32, int(ratio * min(full_context, current_len)))` computed
from the **full-cache sequence length of that example**, not a global 2048
guess. Report both `ratio` and realized `B`.

## Protocol

1. Prefill prompt with full cache.
2. Decode token-by-token in a **custom greedy loop** (not `generate` + fake Cache).
3. After each token, apply the compressor to every layer with the **same
   keep index set** (shared across layers/heads for systems honesty on T4).
4. Log per example: gold, pred, exact-match, gen length, peak VRAM,
   fraction of plan-span tokens still kept in the last 32 steps, latency.

Plan span (oracle): token offsets of the first plan-like sentence in the
*generated* text (`let's`, `plan`, `step 1`, first 15% of CoT if no cue).
Labeled after the full-cache run, then reused so every compressor is
scored against the same span. Heuristic never sees the label.

## Metrics

- Exact match on normalized GSM8K number
- Tokens generated / finish rate (no OOM, hit EOS or max)
- Plan-keep rate (mediator)
- Peak allocated VRAM
- ms/token

Do **not** turn on `output_attentions` for the accuracy run.

## Default T4 schedule

Pass A (pipeline): `N=8`, ratios `(1.0, 0.25)`, methods `full, stream, gaze-heur`  
Pass B (compare): `N=20`, ratios `(0.15, 0.25, 0.40, 1.0)`, all methods  
Stop if Pass A gaze-heur crashes or matches stream on plan-keep.

## Go to 7B

Only if Pass B shows gaze-oracle plan-keep ≫ H2O on traces that are
**not** plan-first, or gaze-heur accuracy ≥ H2O at 0.15–0.25 with a
plan-keep gap. Otherwise the T4 1.5B result is the paper’s cheap negative.
