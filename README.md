# KDE-gaze KV probe (Kaggle T4)

Research-grade rewrite of the virtual-token draft. Compressors **index-select
real K/V**. Attention stays stock SDPA. `model.generate` is not used.

## Pass A (first Kaggle session)

```bash
pip install -q "transformers>=4.44" accelerate bitsandbytes datasets
python run_experiment.py \
  --num-examples 8 \
  --ratios 1.0,0.25 \
  --methods full,stream,h2o,gaze_heur \
  --output-dir runs/pass_a
```

Stop if `gaze_heur` errors or its `plan_keep` equals `stream`.

## Pass B

```bash
python run_experiment.py \
  --num-examples 20 \
  --ratios 0.15,0.25,0.40,1.0 \
  --methods all \
  --max-new 1024 \
  --output-dir runs/pass_b
```

## Local tests (no GPU, no weights)

```bash
python -m pytest test_compressors.py -q
```

## Why the original Cache subclass was deleted

Returning `G` virtual tokens from `Cache.update` breaks RoPE, causal masks,
and `get_seq_length`. H2O scores were never updated. Prefill exploded to
`S * G` tokens. This rewrite keeps original positions in a sidecar and passes
`position_ids=true_len` after every prune.

## Outputs

- `runs/*/config.json` — pinned hyperparameters
- `runs/*/results.jsonl` — per-example records
- `runs/*/summary.json` — accuracy, plan-keep, VRAM, finish rate
