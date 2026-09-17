"""Answer normalization and plan-span labeling."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

PLAN_CUES = (
    r"\blet'?s\b",
    r"\bplan\b",
    r"\bstep\s*1\b",
    r"\bfirst,?\b",
    r"\bwe need to\b",
    r"\bthe problem asks\b",
)


def normalize_number(text: str) -> Optional[str]:
    if text is None:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    text = text.replace(",", "").replace("$", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    token = nums[-1].rstrip(".")
    if token.count(".") == 1:
        left, right = token.split(".")
        if set(right) <= {"0"}:
            token = left
    return token


def exact_match(pred_text: str, gold_text: str) -> bool:
    p, g = normalize_number(pred_text), normalize_number(gold_text)
    return p is not None and p == g


def plan_char_span(generated: str) -> Tuple[int, int]:
    """Best-effort plan sentence in the generated CoT (char offsets)."""
    if not generated:
        return (0, 0)
    lower = generated.lower()
    hits = []
    for cue in PLAN_CUES:
        m = re.search(cue, lower)
        if m:
            hits.append(m.start())
    if hits:
        start = min(hits)
    else:
        start = 0
    # sentence-ish end
    end = generated.find("\n", start + 1)
    if end < 0:
        end = min(len(generated), start + max(40, len(generated) // 8))
    # fallback: first 15% of CoT
    if end <= start:
        end = max(1, int(0.15 * len(generated)))
        start = 0
    return start, end


def char_span_to_token_indices(
    prompt_len: int,
    generated: str,
    tokenizer,
    char_start: int,
    char_end: int,
) -> Set[int]:
    """Map a generated-text char span to absolute sequence token indices."""
    if tokenizer is None or not generated:
        n = max(1, (char_end - char_start) // 4)
        return set(range(prompt_len, prompt_len + n))
    enc = tokenizer(generated, add_special_tokens=False, return_offsets_mapping=True)
    ids: List[int] = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        ids.append(prompt_len + i)
    if not ids:
        frac = max(1, int(0.15 * max(len(enc["input_ids"]), 1)))
        ids = list(range(prompt_len, prompt_len + frac))
    return set(ids)


def plan_keep_rate(keep: Sequence[int], plan_ids: Set[int]) -> float:
    if not plan_ids:
        return 0.0
    kept = sum(1 for i in plan_ids if i in set(keep))
    return kept / len(plan_ids)



