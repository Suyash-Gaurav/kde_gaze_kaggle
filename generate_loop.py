"""Greedy decode with post-step KV surgery and true position_ids.

RoPE is baked into cached K at write time. After pruning we must still
feed the *original* next position, not the compressed length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from compressors import BaseCompressor, FullCompressor, apply_keep_to_caches
from metrics import plan_keep_rate


@dataclass
class DecodeResult:
    text: str
    token_ids: List[int]
    prompt_len: int
    gen_len: int
    latency_s: float
    peak_vram_gb: float
    plan_keep: float
    keep_last_orig: List[int]
    oom: bool
    error: Optional[str] = None


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _get_kv_lists(past):
    if past is None:
        raise RuntimeError("model returned no past_key_values")
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(past.key_cache), list(past.value_cache), past
    keys = [layer[0] for layer in past]
    vals = [layer[1] for layer in past]
    return keys, vals, None


def _set_kv_lists(past, keys, vals):
    if hasattr(past, "key_cache"):
        past.key_cache = keys
        past.value_cache = vals
        return past
    return tuple((k, v) for k, v in zip(keys, vals))


def prompt_for(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning. Put the final numeric answer after ####.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


@torch.no_grad()
def greedy_decode(
    model,
    tokenizer,
    prompt: str,
    compressor: BaseCompressor,
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
    plan_ids: Optional[set] = None,
) -> DecodeResult:
    device = next(model.parameters()).device
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    prompt_len = int(input_ids.shape[1])
    compressor.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    keep_last_orig: List[int] = []
    plan_keeps: List[float] = []
    generated: List[int] = []
    orig_pos = list(range(prompt_len))
    true_len = prompt_len

    try:
        pos = torch.arange(prompt_len, device=device).unsqueeze(0)
        out = model(
            input_ids=input_ids,
            position_ids=pos,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        keys, vals, past_obj = _get_kv_lists(past)
        keep = compressor.select(orig_pos, keys[0][0], is_prefill=True, true_len=true_len)
        if not isinstance(compressor, FullCompressor):
            apply_keep_to_caches(keys, vals, keep)
            orig_pos = [orig_pos[i] for i in keep]
            past = _set_kv_lists(past_obj or past, keys, vals)
        keep_last_orig = list(orig_pos)
        if plan_ids:
            plan_keeps.append(plan_keep_rate(orig_pos, plan_ids))

        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        eos = tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            tid = int(cur.item())
            generated.append(tid)
            if eos is not None and tid == eos:
                break
            position_ids = torch.tensor([[true_len]], device=device)
            out = model(
                input_ids=cur,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            true_len += 1
            orig_pos.append(true_len - 1)
            past = out.past_key_values
            keys, vals, past_obj = _get_kv_lists(past)
            keep = compressor.select(orig_pos, keys[0][0], is_prefill=False, true_len=true_len)
            if not isinstance(compressor, FullCompressor):
                apply_keep_to_caches(keys, vals, keep)
                orig_pos = [orig_pos[i] for i in keep]
                past = _set_kv_lists(past_obj or past, keys, vals)
            keep_last_orig = list(orig_pos)
            if plan_ids:
                plan_keeps.append(plan_keep_rate(orig_pos, plan_ids))
            cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return DecodeResult(
            text=text,
            token_ids=generated,
            prompt_len=prompt_len,
            gen_len=len(generated),
            latency_s=time.time() - t0,
            peak_vram_gb=_peak_gb(),
            plan_keep=float(sum(plan_keeps) / len(plan_keeps)) if plan_keeps else 0.0,
            keep_last_orig=keep_last_orig,
            oom=False,
        )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return DecodeResult(
            text="",
            token_ids=generated,
            prompt_len=prompt_len,
            gen_len=len(generated),
            latency_s=time.time() - t0,
            peak_vram_gb=_peak_gb(),
            plan_keep=0.0,
            keep_last_orig=keep_last_orig,
            oom=True,
            error="oom",
        )
    except Exception as exc:  # noqa: BLE001
        return DecodeResult(
            text="",
            token_ids=generated,
            prompt_len=prompt_len,
            gen_len=len(generated),
            latency_s=time.time() - t0,
            peak_vram_gb=_peak_gb(),
            plan_keep=0.0,
            keep_last_orig=keep_last_orig,
            oom=False,
            error=repr(exc),
        )
