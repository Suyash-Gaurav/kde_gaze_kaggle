"""Pinned experiment configuration. Dump this JSON with every run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_revision: str = "main"
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "test"
    # Fixed subset: first N test rows. Do not reshuffle between methods.
    num_examples: int = 8
    start_index: int = 0
    seed: int = 42

    max_new_tokens: int = 1024
    max_prompt_tokens: int = 384
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"

    methods: List[str] = field(
        default_factory=lambda: ["full", "stream", "h2o", "snap", "gaze_heur", "gaze_oracle"]
    )
    ratios: List[float] = field(default_factory=lambda: [1.0, 0.25])
    min_budget: int = 32
    sink_tokens: int = 4
    recent_tokens: int = 16
    snap_window: int = 32
    max_parks: int = 8
    max_crawls: int = 8
    park_residual: float = 1.05
    park_novelty: float = 0.25

    output_dir: str = "runs/default"

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))
