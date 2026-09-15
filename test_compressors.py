"""CPU unit tests — run before any Kaggle GPU session."""

from __future__ import annotations

import torch

from compressors import (
    GazeCompressor,
    H2OCompressor,
    SnapCompressor,
    StreamCompressor,
    apply_keep_to_caches,
    build_compressor,
)
from config import RunConfig
from metrics import exact_match, normalize_number, plan_char_span, plan_keep_rate


def _keys(s: int, d: int = 8, h: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(h, s, d, generator=g)


def test_stream_keeps_sink_and_tail():
    c = StreamCompressor(budget=6, recent=4, sink=2)
    orig = list(range(20))
    keep = c.select(orig, _keys(20), False, 20)
    kept = [orig[i] for i in keep]
    assert 0 in kept and 1 in kept
    assert 19 in kept
    assert len(keep) <= 6


def test_h2o_budget_and_recent():
    c = H2OCompressor(budget=6, recent=3)
    orig = list(range(12))
    keep = c.select(orig, _keys(12), False, 12)
    assert len(keep) <= 6
    kept = [orig[i] for i in keep]
    assert 11 in kept and 10 in kept


def test_snap_freezes_prefill_set():
    c = SnapCompressor(budget=6, recent=2, window=3)
    orig = list(range(10))
    k1 = c.select(orig, _keys(10), True, 10)
    frozen = list(c._frozen_orig)
    orig2 = orig + [10]
    keys2 = torch.cat([_keys(10), torch.randn(2, 1, 8)], dim=1)
    k2 = c.select(orig2, keys2, False, 11)
    assert c._frozen_orig == frozen
    assert len(k2) <= 6


def test_gaze_oracle_never_drops_plan():
    parks = {4, 5, 6}
    c = GazeCompressor(budget=8, recent=3, max_parks=4, oracle_parks=parks)
    orig = list(range(20))
    keep = c.select(orig, _keys(20), False, 20)
    kept = set(orig[i] for i in keep)
    assert parks <= kept


def test_gaze_heuristic_parks_are_immortal():
    c = GazeCompressor(budget=8, recent=3, max_parks=2, park_residual=0.0, park_novelty=0.0)
    orig = list(range(5))
    c.select(orig, _keys(5), False, 5)
    # force a park
    c.state.parks.add(2)
    orig = list(range(16))
    keep = c.select(orig, _keys(16), False, 16)
    assert 2 in {orig[i] for i in keep}


def test_apply_keep_shortens_seq():
    keys = [torch.randn(1, 2, 10, 4)]
    vals = [torch.randn(1, 2, 10, 4)]
    apply_keep_to_caches(keys, vals, [0, 1, 9])
    assert keys[0].shape[2] == 3
    assert vals[0].shape[2] == 3


def test_build_names():
    cfg = RunConfig()
    assert build_compressor("h2o", 32, cfg).name == "h2o"
    assert build_compressor("gaze_oracle", 32, cfg, oracle_parks={1}).name == "gaze_oracle"


def test_gsm8k_number_parse():
    assert normalize_number("blah #### 1,234") == "1234"
    assert exact_match("so the answer is #### 18", "18")
    assert not exact_match("#### 19", "18")


def test_plan_span_and_keep_rate():
    text = "Let's compute the total.\nThen add 3.\n#### 4"
    s, e = plan_char_span(text)
    assert s == 0
    assert e > s
    assert plan_keep_rate([0, 1, 8], {0, 1, 2}) == 2 / 3
