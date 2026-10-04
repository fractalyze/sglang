"""Replay a fixed continuation through speculative decoding (benchmarking only).

With SGLANG_SIMULATE_ACC_REPLAY_PATH set, a request whose prompt appears in
that file decodes exactly the file's continuation for it:

- the first token after prefill is the continuation's first token;
- before each verify, the request's drafts are replaced by the continuation's
  next tokens, so the target verifies the replayed text;
- each verify accepts SGLANG_SIMULATE_ACC_REPLAY_LEN tokens (bonus included),
  all taken from the continuation.

Two configurations replaying one file therefore verify the same tokens in the
same number of rounds, and their decode times differ only by per-round cost.
Without the replay, a change to the target's numerics changes its greedy text,
and the drafter's acceptance on the new text moves decode time on its own.
Requests whose prompt is not in the file decode normally. Topk 1 only.

File format (JSON): {"continuations": [{"input_ids": [...], "output_ids": [...]}]}.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.environ import envs


def prompt_key(input_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, input_ids)).encode()).hexdigest()


class SpecReplay:
    """Per req-pool row: the replayed token at every sequence position, and whether the row replays."""

    def __init__(
        self,
        continuations: Dict[str, List[int]],
        accept_len: int,
        width: int,
        num_rows: int,
        device,
    ):
        if accept_len < 1:
            raise ValueError(f"replay accept length must be >= 1, got {accept_len}")
        self.continuations = continuations
        self.accept_len = accept_len
        self.width = width
        # Column `p` of a row is the token at sequence position `p` (prompt, then continuation).
        self.tokens = torch.zeros((num_rows, width), dtype=torch.int64, device=device)
        self.active = torch.zeros((num_rows,), dtype=torch.bool, device=device)

    @classmethod
    def from_file(cls, path: str, accept_len: int, num_draft_tokens: int, num_rows: int, device):
        with open(path) as f:
            entries = json.load(f)["continuations"]
        continuations = {prompt_key(e["input_ids"]): list(e["output_ids"]) for e in entries}
        # One verify window past the longest sequence; later positions clamp to the last column.
        width = max(len(e["input_ids"]) + len(e["output_ids"]) for e in entries) + num_draft_tokens + 1
        return cls(continuations, accept_len, width, num_rows, device)

    def register(self, rows: Sequence[int], prompts: Sequence[Sequence[int]]) -> None:
        """Binds each req-pool row to its prompt's continuation (or marks it not replaying)."""
        tokens = torch.zeros((len(rows), self.width), dtype=torch.int64)
        active = torch.zeros((len(rows),), dtype=torch.bool)
        for i, prompt in enumerate(prompts):
            cont = self.continuations.get(prompt_key(prompt))
            if cont is None:
                continue
            seq = list(prompt) + cont
            seq = seq[: self.width]
            tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.int64)
            # Positions past the continuation repeat its last token: the request stops before
            # emitting them, but a final verify window still reads them.
            tokens[i, len(seq) :] = seq[-1]
            active[i] = True
        idx = torch.tensor(list(rows), dtype=torch.int64, device=self.tokens.device)
        self.tokens.index_copy_(0, idx, tokens.to(self.tokens.device))
        self.active.index_copy_(0, idx, active.to(self.active.device))

    def _at(self, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.tokens[rows.long().unsqueeze(-1), positions.long().clamp(max=self.width - 1)]

    def first_tokens(self, next_tokens: torch.Tensor, rows: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Prefill's sampled token -> the continuation's first token, for replaying rows."""
        replay = self._at(rows, seq_lens.unsqueeze(-1)).squeeze(-1)
        return torch.where(self.active[rows.long()], replay.to(next_tokens.dtype), next_tokens)

    def drafts(self, draft_tokens: torch.Tensor, rows: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """Verify input [bs * n] (node 0 = last bonus, at position seq_len) -> drafts from the continuation."""
        bs = rows.shape[0]
        n = draft_tokens.numel() // bs
        offsets = torch.arange(n, device=draft_tokens.device)
        replay = self._at(rows, seq_lens.unsqueeze(-1) + offsets).to(draft_tokens.dtype)
        # Node 0 is already the replayed token (it was the previous round's replayed bonus).
        take = self.active[rows.long()].unsqueeze(-1) & (offsets > 0)
        return torch.where(take, replay, draft_tokens.view(bs, n)).reshape(-1)

    def accept(
        self,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        rows: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Replaying rows accept `accept_len` tokens: nodes 0..A-1, each predicting the continuation's next token."""
        bs, width = accept_index.shape
        n = predict.numel() // bs
        a = min(self.accept_len, width)
        active = self.active[rows.long()]
        cols = torch.arange(width, device=accept_index.device)
        base = torch.arange(bs, device=accept_index.device).unsqueeze(-1) * n
        sim_index = torch.where(cols < a, base + cols, torch.full_like(cols, -1)).to(accept_index.dtype)
        accept_index = torch.where(active.unsqueeze(-1), sim_index, accept_index)
        accept_lens = torch.where(active, torch.full_like(accept_lens, a), accept_lens)
        nodes = torch.arange(n, device=predict.device)
        replay = self._at(rows, seq_lens.unsqueeze(-1) + nodes + 1).to(predict.dtype)
        predict = torch.where(active.unsqueeze(-1), replay, predict.view(bs, n)).reshape(-1)
        return predict, accept_lens, accept_index


_REPLAY: Optional[SpecReplay] = None


def get_spec_replay(num_draft_tokens: int, num_rows: int, device) -> Optional[SpecReplay]:
    """The process's replay table, built on first use; None unless SGLANG_SIMULATE_ACC_REPLAY_PATH is set."""
    global _REPLAY
    path = envs.SGLANG_SIMULATE_ACC_REPLAY_PATH.get()
    if not path:
        return None
    if _REPLAY is None:
        _REPLAY = SpecReplay.from_file(
            path, envs.SGLANG_SIMULATE_ACC_REPLAY_LEN.get(), num_draft_tokens, num_rows, device
        )
    return _REPLAY
