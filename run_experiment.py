#!/usr/bin/env python3
"""Reproducible GSM8K KV-compression probe.

Pass A (default): N=8, ratios 1.0 and 0.25, subset of methods.
Pass B: raise --num-examples 20 --ratios 0.15,0.25,0.40,1.0 --methods all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

from compressors import build_compressor
from config import RunConfig
from generate_loop import greedy_decode, prompt_for
from metrics import char_span_to_token_indices, exact_match, plan_char_span


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=RunConfig.model_name)
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--ratios", default="1.0,0.25")
    p.add_argument("--methods", default="full,stream,h2o,gaze_heur")
    p.add_argument("--output-dir", default="runs/pass_a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if methods == ["all"]:
        methods = ["full", "stream", "h2o", "snap", "gaze_heur", "gaze_oracle"]
    return RunConfig(
        model_name=args.model,
        num_examples=args.num_examples,
        start_index=args.start_index,
        max_new_tokens=args.max_new,
        ratios=[float(x) for x in args.ratios.split(",")],
        methods=methods,
        output_dir=args.output_dir,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
    )


def load_model(cfg: RunConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = None
    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def load_rows(cfg: RunConfig):
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    end = min(cfg.start_index + cfg.num_examples, len(ds))
    return [ds[i] for i in range(cfg.start_index, end)]


def budget_for(ratio: float, seq_len: int, cfg: RunConfig) -> int:
    if ratio >= 1.0:
        return seq_len
    return max(cfg.min_budget, int(seq_len * ratio))


def run_full_then_label(model, tok, row, cfg: RunConfig):
    """Full-cache pass supplies the plan span used by gaze_oracle."""
    comp = build_compressor("full", budget=10**9, cfg=cfg)
    res = greedy_decode(
        model,
        tok,
        prompt_for(row["question"]),
        comp,
        max_new_tokens=cfg.max_new_tokens,
        max_prompt_tokens=cfg.max_prompt_tokens,
    )
    cs, ce = plan_char_span(res.text)
    plan_ids = char_span_to_token_indices(res.prompt_len, res.text, tok, cs, ce)
    return res, plan_ids


def eval_one(model, tok, row, method: str, ratio: float, cfg: RunConfig, plan_ids: Set[int], full_len: int) -> Dict:
    B = budget_for(ratio, full_len, cfg)
    comp = build_compressor(method, budget=B, cfg=cfg, oracle_parks=plan_ids if method == "gaze_oracle" else None)
    res = greedy_decode(
        model,
        tok,
        prompt_for(row["question"]),
        comp,
        max_new_tokens=cfg.max_new_tokens,
        max_prompt_tokens=cfg.max_prompt_tokens,
        plan_ids=plan_ids,
    )
    return {
        "method": method,
        "ratio": ratio,
        "budget": B,
        "exact_match": exact_match(res.text, row["answer"]) if not res.oom else False,
        "pred": res.text[-400:],
        "gold_tail": row["answer"][-200:],
        "gen_len": res.gen_len,
        "prompt_len": res.prompt_len,
        "latency_s": round(res.latency_s, 3),
        "peak_vram_gb": round(res.peak_vram_gb, 3),
        "plan_keep": round(res.plan_keep, 4),
        "keep_last_orig": res.keep_last_orig[-32:],
        "oom": res.oom,
        "error": res.error,
    }


def summarize(rows: List[Dict]) -> List[Dict]:
    keys = {}
    for r in rows:
        k = (r["method"], r["ratio"])
        keys.setdefault(k, []).append(r)
    out = []
    for (method, ratio), grp in keys.items():
        n = len(grp)
        out.append(
            {
                "method": method,
                "ratio": ratio,
                "n": n,
                "accuracy": sum(x["exact_match"] for x in grp) / n,
                "finish_rate": sum(not x["oom"] and not x["error"] for x in grp) / n,
                "mean_gen_len": sum(x["gen_len"] for x in grp) / n,
                "mean_plan_keep": sum(x["plan_keep"] for x in grp) / n,
                "mean_latency_s": sum(x["latency_s"] for x in grp) / n,
                "mean_peak_vram_gb": sum(x["peak_vram_gb"] for x in grp) / n,
            }
        )
    return out


def main() -> None:
    cfg = parse_args()
    seed_all(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.json")

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    model, tok = load_model(cfg)
    rows = load_rows(cfg)
    print(f"examples {cfg.start_index}:{cfg.start_index + len(rows)}")

    records: List[Dict] = []
    for i, row in enumerate(rows):
        print(f"\n=== example {cfg.start_index + i} ===")
        full_res, plan_ids = run_full_then_label(model, tok, row, cfg)
        full_len = full_res.prompt_len + full_res.gen_len
        records.append(
            {
                "example": cfg.start_index + i,
                "method": "full",
                "ratio": 1.0,
                "budget": full_len,
                "exact_match": exact_match(full_res.text, row["answer"]) if not full_res.oom else False,
                "pred": full_res.text[-400:],
                "gold_tail": row["answer"][-200:],
                "gen_len": full_res.gen_len,
                "prompt_len": full_res.prompt_len,
                "latency_s": round(full_res.latency_s, 3),
                "peak_vram_gb": round(full_res.peak_vram_gb, 3),
                "plan_keep": 1.0,
                "keep_last_orig": [],
                "oom": full_res.oom,
                "error": full_res.error,
                "plan_ids_n": len(plan_ids),
            }
        )
        print(f"  full gen_len={full_res.gen_len} em={records[-1]['exact_match']} err={full_res.error}")
        for ratio in cfg.ratios:
            for method in cfg.methods:
                if method == "full" and ratio >= 1.0:
                    continue
                rec = eval_one(model, tok, row, method, ratio, cfg, plan_ids, full_len)
                rec["example"] = cfg.start_index + i
                records.append(rec)
                print(
                    f"  {method:12s} r={ratio:.2f} B={rec['budget']:4d} "
                    f"em={rec['exact_match']} pk={rec['plan_keep']:.2f} "
                    f"len={rec['gen_len']} oom={rec['oom']} err={rec['error']}"
                )
        (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY")
    for s in summary:
        print(
            f"{s['method']:12s} r={s['ratio']:.2f} acc={s['accuracy']:.2f} "
            f"plan_keep={s['mean_plan_keep']:.2f} finish={s['finish_rate']:.2f} "
            f"vram={s['mean_peak_vram_gb']:.2f}GB"
        )


if __name__ == "__main__":
    main()
