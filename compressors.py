"""Token-budget compressors on real K/V slots.

select() returns indices into the *current* cache (0..S-1).
orig_pos[j] maps slot j to the original token index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

import torch
from torch import Tensor


def _slots_for_orig(orig_pos: List[int], wanted: Sequence[int]) -> List[int]:
    inv = {p: i for i, p in enumerate(orig_pos)}
    return [inv[p] for p in wanted if p in inv]


def _recent_orig(orig_pos: List[int], n: int) -> List[int]:
    return sorted(orig_pos)[-n:] if orig_pos else []


@dataclass
class CompressorState:
    scores: dict = field(default_factory=dict)  # orig_pos -> float
    parks: Set[int] = field(default_factory=set)
    last_keep_orig: List[int] = field(default_factory=list)


class BaseCompressor:
    name = "base"

    def __init__(self, budget: int, recent: int = 16, sink: int = 4):
        self.budget = max(int(budget), 1)
        self.recent = recent
        self.sink = sink
        self.state = CompressorState()

    def reset(self) -> None:
        self.state = CompressorState()

    def select(
        self,
        orig_pos: List[int],
        keys: Tensor,
        is_prefill: bool,
        true_len: int,
    ) -> List[int]:
        raise NotImplementedError

    def observe_last_attn_proxy(self, orig_pos: List[int], keys: Tensor) -> None:
        if keys.numel() == 0 or keys.shape[1] == 0:
            return
        q = keys[:, -1, :]
        scale = keys.shape[-1] ** 0.5
        scores = torch.einsum("hd,hsd->hs", q, keys) / scale
        w = torch.softmax(scores.float(), dim=-1).mean(dim=0)
        for slot, orig in enumerate(orig_pos):
            self.state.scores[orig] = self.state.scores.get(orig, 0.0) + float(w[slot])


class FullCompressor(BaseCompressor):
    name = "full"

    def select(self, orig_pos, keys, is_prefill, true_len):
        keep = list(range(len(orig_pos)))
        self.state.last_keep_orig = list(orig_pos)
        return keep


class StreamCompressor(BaseCompressor):
    name = "stream"

    def select(self, orig_pos, keys, is_prefill, true_len):
        if len(orig_pos) <= self.budget:
            self.state.last_keep_orig = list(orig_pos)
            return list(range(len(orig_pos)))
        sink = list(range(min(self.sink, true_len)))
        recent = _recent_orig(orig_pos, self.budget - min(self.sink, self.budget))
        wanted = sink + recent
        keep = _slots_for_orig(orig_pos, wanted)
        if len(keep) < self.budget:
            for i in range(len(orig_pos) - 1, -1, -1):
                if i not in keep:
                    keep.append(i)
                if len(keep) >= self.budget:
                    break
        keep = sorted(keep)[: self.budget]
        self.state.last_keep_orig = [orig_pos[i] for i in keep]
        return keep


class H2OCompressor(BaseCompressor):
    name = "h2o"

    def select(self, orig_pos, keys, is_prefill, true_len):
        self.observe_last_attn_proxy(orig_pos, keys)
        if len(orig_pos) <= self.budget:
            self.state.last_keep_orig = list(orig_pos)
            return list(range(len(orig_pos)))
        n_recent = min(self.recent, self.budget)
        recent = _recent_orig(orig_pos, n_recent)
        n_hh = self.budget - n_recent
        candidates = [p for p in orig_pos if p not in recent]
        candidates.sort(key=lambda p: self.state.scores.get(p, 0.0), reverse=True)
        wanted = candidates[:n_hh] + recent
        keep = sorted(_slots_for_orig(orig_pos, wanted))[: self.budget]
        self.state.last_keep_orig = [orig_pos[i] for i in keep]
        return keep


class SnapCompressor(BaseCompressor):
    name = "snap"

    def __init__(self, budget: int, recent: int = 16, sink: int = 4, window: int = 32):
        super().__init__(budget, recent=recent, sink=sink)
        self.window = window
        self._frozen_orig: Optional[List[int]] = None

    def reset(self) -> None:
        super().reset()
        self._frozen_orig = None

    def select(self, orig_pos, keys, is_prefill, true_len):
        if len(orig_pos) <= self.budget:
            self.state.last_keep_orig = list(orig_pos)
            return list(range(len(orig_pos)))
        n_recent = min(self.recent, self.budget)
        recent = _recent_orig(orig_pos, n_recent)
        if is_prefill or self._frozen_orig is None:
            win = min(self.window, keys.shape[1])
            q = keys[:, -win:, :].mean(dim=1)
            scale = keys.shape[-1] ** 0.5
            scores = torch.einsum("hd,hsd->hs", q, keys) / scale
            w = torch.softmax(scores.float(), dim=-1).mean(0)
            scored = sorted(
                [(float(w[i]), orig_pos[i]) for i in range(len(orig_pos)) if orig_pos[i] not in recent],
                reverse=True,
            )
            self._frozen_orig = [p for _, p in scored[: self.budget - n_recent]]
        wanted = list(self._frozen_orig) + recent
        keep = sorted(_slots_for_orig(orig_pos, wanted))[: self.budget]
        self.state.last_keep_orig = [orig_pos[i] for i in keep]
        return keep


class GazeCompressor(BaseCompressor):
    name = "gaze_heur"

    def __init__(
        self,
        budget: int,
        recent: int = 16,
        sink: int = 4,
        max_parks: int = 8,
        max_crawls: int = 8,
        park_residual: float = 1.05,
        park_novelty: float = 0.25,
        oracle_parks: Optional[Set[int]] = None,
    ):
        super().__init__(budget, recent=recent, sink=sink)
        self.max_parks = max_parks
        self.max_crawls = max_crawls
        self.park_residual = park_residual
        self.park_novelty = park_novelty
        self.oracle_parks = set(oracle_parks or [])
        self.name = "gaze_oracle" if self.oracle_parks else "gaze_heur"

    def reset(self) -> None:
        self.state = CompressorState(parks=set(self.oracle_parks))

    def _heuristic_park(self, orig_t: int, keys: Tensor) -> bool:
        if orig_t in self.state.parks or len(self.state.parks) >= self.max_parks:
            return False
        if keys.shape[1] < 2:
            return False
        last = torch.nn.functional.normalize(keys[:, -1, :].mean(0), dim=0)
        prev = torch.nn.functional.normalize(keys[:, :-1, :].mean(dim=1).mean(0), dim=0)
        residual = torch.norm(last - prev).item()
        novelty = 1.0 - float(torch.clamp(last @ prev, -1, 1))
        return residual >= self.park_residual and novelty >= self.park_novelty

    def select(self, orig_pos, keys, is_prefill, true_len):
        orig_t = orig_pos[-1] if orig_pos else true_len - 1
        if self.oracle_parks:
            self.state.parks |= {p for p in self.oracle_parks if p < true_len}
        elif self._heuristic_park(orig_t, keys):
            self.state.parks.add(orig_t)

        if len(orig_pos) <= self.budget:
            self.state.last_keep_orig = list(orig_pos)
            return list(range(len(orig_pos)))

        n_recent = min(self.recent, self.budget)
        recent = _recent_orig(orig_pos, n_recent)
        parks = sorted(p for p in self.state.parks if p in set(orig_pos) and p not in recent)
        blocked = set(recent) | set(parks)
        rest = [p for p in orig_pos if p not in blocked]
        n_crawl = min(self.max_crawls, max(self.budget - len(recent) - len(parks), 0))
        if rest and n_crawl > 0:
            step = max(len(rest) / n_crawl, 1.0)
            crawls = [rest[min(int(j * step), len(rest) - 1)] for j in range(n_crawl)]
        else:
            crawls = []
        wanted = parks + crawls + recent
        keep = _slots_for_orig(orig_pos, wanted)
        keep = sorted(set(keep))[: self.budget]
        self.state.last_keep_orig = [orig_pos[i] for i in keep]
        return keep


def apply_keep_to_caches(key_cache: List[Tensor], value_cache: List[Tensor], keep: List[int]) -> None:
    if not keep or not key_cache:
        return
    device = key_cache[0].device
    idx = torch.tensor(keep, device=device, dtype=torch.long)
    for i, (k, v) in enumerate(zip(key_cache, value_cache)):
        if k.ndim != 4:
            raise ValueError(f"expected [B,H,S,D], got {tuple(k.shape)}")
        key_cache[i] = k.index_select(2, idx).contiguous()
        value_cache[i] = v.index_select(2, idx).contiguous()


def build_compressor(name: str, budget: int, cfg, oracle_parks: Optional[Set[int]] = None) -> BaseCompressor:
    name = name.lower().replace("-", "_")
    if name == "full":
        return FullCompressor(budget)
    if name == "stream":
        return StreamCompressor(budget, recent=cfg.recent_tokens, sink=cfg.sink_tokens)
    if name == "h2o":
        return H2OCompressor(budget, recent=cfg.recent_tokens, sink=cfg.sink_tokens)
    if name == "snap":
        return SnapCompressor(
            budget, recent=cfg.recent_tokens, sink=cfg.sink_tokens, window=cfg.snap_window
        )
    if name == "gaze_heur":
        return GazeCompressor(
            budget,
            recent=cfg.recent_tokens,
            max_parks=cfg.max_parks,
            max_crawls=cfg.max_crawls,
            park_residual=cfg.park_residual,
            park_novelty=cfg.park_novelty,
        )
    if name == "gaze_oracle":
        return GazeCompressor(
            budget,
            recent=cfg.recent_tokens,
            max_parks=cfg.max_parks,
            max_crawls=cfg.max_crawls,
            park_residual=cfg.park_residual,
            park_novelty=cfg.park_novelty,
            oracle_parks=oracle_parks or set(),
        )
    raise ValueError(f"unknown compressor {name}")
