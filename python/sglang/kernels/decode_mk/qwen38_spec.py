# Copyright 2026 Fractalyze Inc. All rights reserved.
"""Greedy speculative decoding of Qwen3.8-27B: its MTP head drafts, the
decode megakernel's verify step checks, as SGLang's EAGLE runs the MTP head
with one draft per step (topk 1).

A cycle starts from the next token x at position p, the first draft d₀, and
the MTP head's hidden state after drafting d₀:
  1. The head drafts d₁ … d_(K - 1), each from its own previous hidden state.
  2. One verify step runs x, d₀, …, d_(K - 1) at positions p … p + K, giving
     the model's greedy choices a₀ … a_K.
  3. The drafts are accepted while dᵢ = aᵢ: n of them. The cycle emits
     d₀ … d_(n - 1) and a_n, and x becomes a_n at position p + n + 1.
  4. The head runs over positions p … p + n from the model's hidden states
     there, rewriting its cache where drafting left its own guesses (SGLang's
     draft extend). Its last step drafts the next cycle's d₀.
The tokens are the model's own greedy choices, the verify step's logits being
bitwise those of decoding one token at a time, so the text is plain greedy
decoding's.

Each cycle replays two CUDA graphs: the drafts with the verify step, then the
extend for the n accepted; the host reads the 2K + 1 tokens between them.

Qwen38Generator.start and Qwen38Generator.cycle are the cycle's public API,
which generate() runs on and which fractalyze/sglang's decode_mk adapter
(python/sglang/srt/models/qwen3_5_decode_mk.py) drives after its own prefill.
A change to the cycle keeps their signatures and meaning, or changes that
adapter with them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from sglang.kernels.decode_mk.barrier import DEFAULT_TIMEOUT_NS
from sglang.kernels.decode_mk.qwen38_decode import Qwen38Decoder, Qwen38State, Qwen38Weights
from sglang.kernels.decode_mk.qwen38_layer import DIM, cos_sin_table
from sglang.kernels.decode_mk.qwen38_mtp import MtpWeights, Qwen38Mtp
from sglang.kernels.decode_mk.qwen38_verify import MAX_TOKENS, Qwen38Verifier, SlottedState
from sglang.kernels.decode_mk.thinker_attention import PagedCache

MAX_DRAFTS = MAX_TOKENS - 1


@dataclass
class Generation:
    """A greedy reply: its tokens, the first among them the prompt's last
    logits' choice; the seconds from that first token to the last token
    decoded, and how many were decoded, which the last cycle of speculative
    decoding can take past the reply's end; and, under speculative decoding,
    the tokens each cycle emitted."""
    tokens: list[int]
    decode_seconds: float
    decoded: int
    emitted: list[int] = field(default_factory=list)

    @property
    def decode_tok_s(self) -> float:
        """Tokens after the first over the time from the first to the last,
        as control/qwen38/bench.py rates SGLang's replies."""
        return (self.decoded - 1) / self.decode_seconds

    @property
    def accept_length(self) -> float:
        """Tokens emitted per verify step: accepted drafts plus the model's
        own choice, as SGLang's avg_spec_accept_length counts them."""
        return sum(self.emitted) / len(self.emitted)


def _capture(fn, warm: bool = True) -> torch.cuda.CUDAGraph:
    """`fn` as a CUDA graph, after one run on a side stream to warm it up
    unless `warm` is false."""
    if warm:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _paged_cache(positions: int, block_size: int = 16) -> PagedCache:
    return Qwen38State.zeros(4, positions, block_size).caches[0]


class Qwen38Generator:
    """Greedy decoding of Qwen3.8-27B on the megakernels, with up to
    `max_positions` positions a sequence, prompt and reply.

    The prompt runs through verify steps of MAX_TOKENS tokens, whose states
    are bitwise those of feeding it one token at a time. Without MTP weights
    the reply is plain decoding, one decode step a token (s2mk/qwen38_decode.py);
    with them, speculative decoding with `drafts` drafts a cycle."""

    def __init__(self, weights: Qwen38Weights, max_positions: int,
                 mtp: MtpWeights | None = None, drafts: int = MAX_DRAFTS,
                 timeout_ns: int = DEFAULT_TIMEOUT_NS, ctas: int | None = None,
                 splits: int | None = None) -> None:
        if mtp is not None and not 1 <= drafts <= MAX_DRAFTS:
            raise ValueError(f"drafts must be in [1, {MAX_DRAFTS}], got {drafts}")
        self.weights = weights
        self.max_positions = max_positions
        self.drafts = drafts if mtp is not None else 0
        cos_sin = cos_sin_table(max_positions)
        self.state = SlottedState.of(Qwen38State.zeros(len(weights.layers), max_positions),
                                     MAX_TOKENS + 1)
        self.verifier = Qwen38Verifier(weights, self.state, cos_sin, timeout_ns, ctas, splits)
        ctas, splits = self.verifier.num_ctas, self.verifier.splits
        device = weights.final_norm.device

        def scalar():
            return torch.zeros(1, dtype=torch.int32, device=device)

        # The loop's device state: the next token's position, the slot the
        # linear layers' states continue from, and the verify step's tokens
        # [x, d₀, …], whose drafts the head writes in place.
        self._pos, self._slot = scalar(), scalar()
        self._tokens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=device)
        self._answers = torch.zeros(MAX_TOKENS, dtype=torch.int64, device=device)
        self._prompt_pos, self._prompt_slot = scalar(), scalar()
        # _pos on the host, for cycle() to check; None until start().
        self._next_pos: int | None = None

        if mtp is None:
            self.mtp = None
            self._decoder_cos_sin = cos_sin
            self._ctas, self._splits = ctas, splits
            self._timeout_ns = timeout_ns
            return
        self.mtp = Qwen38Mtp(mtp, weights, _paged_cache(max_positions), cos_sin, timeout_ns,
                             ctas, splits)
        # Captured with the loop's state at position 0 and slot 0: the
        # warm-up runs write slots and positions a prompt overwrites first.
        self._draft_graph = _capture(self._draft_and_verify)
        self._extend_graphs = [_capture(lambda n=n: self._extend(n)) for n in range(drafts + 1)]

    # ------------------------------------------------------------ the cycle

    def start(self, hidden: torch.Tensor, token: int, pos: int, slot: int = 0) -> None:
        """Starts the loop at `token`, the next token, at position `pos`,
        after a prompt whose linear layers' states are in `slot`. With the
        head, whose cache already holds positions 0 … pos - 2, the head runs
        position pos - 1 from `hidden`, the model's final-normed hidden state
        there (bf16 [DIM]), with `token` after it, drafting the first cycle's
        d₀."""
        if not 0 < pos < self.max_positions:
            raise ValueError(f"position {pos} does not follow a prompt in "
                             f"{self.max_positions} positions")
        self._next_pos = pos
        self._pos.fill_(pos)
        self._slot.fill_(slot)
        self._tokens[0] = token
        if self.mtp is None:
            return
        self.mtp.token.fill_(token)
        self.mtp.pos.fill_(pos - 1)
        self.mtp.hidden_in.copy_(hidden)
        self.mtp.launch(with_logits=True)
        self._tokens[1] = self.mtp.logits.argmax()

    def cycle(self) -> list[int]:
        """One cycle from where start() or the last cycle left the loop: the
        accepted drafts and the model's own choice after them, 1 to
        `drafts` + 1 tokens. Leaves the loop at the last of them."""
        if self.mtp is None:
            raise ValueError("speculative decoding needs the MTP head's weights")
        if self._next_pos is None:
            raise ValueError("cycle() continues from start()")
        k = self.drafts
        if self._next_pos + k + 1 > self.max_positions:
            raise ValueError(f"a cycle at position {self._next_pos} verifies past "
                             f"{self.max_positions} positions")
        self._draft_graph.replay()
        read = torch.cat([self._tokens[1:k + 1].long(), self._answers[:k + 1]]).tolist()
        drafts, answers = read[:k], read[k:]
        n = 0
        while n < k and drafts[n] == answers[n]:
            n += 1
        self._extend_graphs[n].replay()
        self._next_pos += n + 1
        return drafts[:n] + [answers[n]]

    def _draft_and_verify(self) -> None:
        m, k = self.mtp, self.drafts
        for i in range(1, k):
            m.token.copy_(self._tokens[i:i + 1])
            m.hidden_in.copy_(m.hidden_out)
            torch.add(self._pos, i - 1, out=m.pos)
            m.launch()
            self._tokens[i + 1:i + 2].copy_(m.logits.argmax(dim=-1, keepdim=True))
        self.verifier.launch(self._tokens[:k + 1], self._pos, self._slot)
        torch.argmax(self.verifier.logits[:k + 1], dim=-1, out=self._answers[:k + 1])

    def _extend(self, n: int) -> None:
        """The head over positions p … p + n after accepting n drafts, then
        the next cycle's x and d₀, position and slot."""
        m = self.mtp
        following = torch.cat([self._tokens[1:n + 1], self._answers[n:n + 1].int()])
        for j in range(n + 1):
            m.token.copy_(following[j:j + 1])
            m.hidden_in.copy_(self.verifier.final_hidden[j])
            torch.add(self._pos, j, out=m.pos)
            m.launch(with_logits=j == n)
        self._tokens[0:1].copy_(self._answers[n:n + 1])
        self._tokens[1:2].copy_(m.logits.argmax(dim=-1, keepdim=True))
        self._pos.add_(n + 1)
        self._slot.copy_((self._slot + n + 1) % self.state.slots)

    # ---------------------------------------------------------- the prompt

    def _prefill(self, prompt: list[int]) -> int:
        """Feeds `prompt` from position 0 and the zero state in slot 0;
        returns its greedy next token. Leaves the loop at that token, and,
        with the head, the head's cache over the prompt and the first draft."""
        if not 0 < len(prompt) < self.max_positions:
            raise ValueError(f"a prompt of {len(prompt)} tokens does not fit "
                             f"{self.max_positions} positions")
        for s in self.state.conv + self.state.recurrent:
            s[0].zero_()
        hidden = torch.empty(len(prompt), DIM, dtype=torch.bfloat16,
                             device=self._tokens.device)
        tokens = torch.tensor(prompt, dtype=torch.int32, device=self._tokens.device)
        slot = 0
        for begin in range(0, len(prompt), MAX_TOKENS):
            n = min(MAX_TOKENS, len(prompt) - begin)
            self._prompt_pos.fill_(begin)
            self._prompt_slot.fill_(slot)
            self.verifier.launch(tokens[begin:begin + n], self._prompt_pos, self._prompt_slot)
            hidden[begin:begin + n] = self.verifier.final_hidden[:n]
            slot = (slot + n) % self.state.slots
        first = int(self.verifier.logits[n - 1].argmax())
        if self.mtp is not None:
            for i, token in enumerate(prompt[1:]):
                self.mtp.token.fill_(token)
                self.mtp.pos.fill_(i)
                self.mtp.hidden_in.copy_(hidden[i])
                self.mtp.launch(with_logits=False)
        self.start(hidden[-1], first, len(prompt), slot)
        torch.cuda.synchronize()
        return first

    # ------------------------------------------------------------ decoding

    def generate(self, prompt: list[int], max_tokens: int,
                 eos: set[int] = frozenset()) -> Generation:
        """The greedy reply to `prompt`: up to `max_tokens` tokens, stopping
        after the first in `eos`."""
        if len(prompt) + max_tokens + MAX_TOKENS > self.max_positions:
            raise ValueError(f"{len(prompt)} + {max_tokens} tokens do not fit "
                             f"{self.max_positions} positions")
        tokens = [self._prefill(prompt)]
        if self.mtp is None:
            return self._plain(tokens, max_tokens, eos)
        emitted = []
        start = time.perf_counter()
        while len(tokens) < max_tokens and tokens[-1] not in eos:
            accepted = self.cycle()
            emitted.append(len(accepted))
            tokens.extend(accepted)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return Generation(_cut(tokens, max_tokens, eos), seconds, len(tokens), emitted)

    def _plain(self, tokens: list[int], max_tokens: int, eos: set[int]) -> Generation:
        """Decode steps from the prompt's states, one token a step."""
        slot = int(self._slot)
        state = Qwen38State([self.state.linear(j, slot) for j in range(len(self.state.conv))],
                            self.state.caches)
        decoder = Qwen38Decoder(self.weights, state, self._decoder_cos_sin, self._timeout_ns,
                                self._ctas, self._splits)
        token = torch.zeros(1, dtype=torch.int32, device=self._pos.device)
        positions = torch.zeros(3, dtype=torch.int32, device=self._pos.device)

        def step():
            positions.copy_(self._pos.expand(3))
            decoder.launch(token, self._pos, positions)
            token.copy_(decoder.logits.argmax(dim=-1, keepdim=True))
            self._pos.add_(1)

        # A warm-up would advance the states in place.
        graph = _capture(step, warm=False)
        token.fill_(tokens[0])
        start = time.perf_counter()
        while len(tokens) < max_tokens and tokens[-1] not in eos:
            graph.replay()
            tokens.append(int(token))
        torch.cuda.synchronize()
        return Generation(tokens, time.perf_counter() - start, len(tokens))


def _cut(tokens: list[int], max_tokens: int, eos: set[int]) -> list[int]:
    """`tokens` up to max_tokens, and up to the first in `eos`."""
    tokens = tokens[:max_tokens]
    for i, token in enumerate(tokens):
        if token in eos:
            return tokens[:i + 1]
    return tokens
