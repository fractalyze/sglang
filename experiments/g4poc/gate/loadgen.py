"""Session load generator: replays a session file against /generate, non-streaming.

Poisson arrival (session layer): the offered load is a plan fixed before the run
(plan_open): `concurrency` sessions already mid-conversation at t=0, then Poisson
session arrivals at concurrency / expected_session_s. Within a session each turn is
sent think time after the previous reply arrives, as a user would. Both legs of a
pair replay the same plan, so they see the same offered load and (in scripted mode)
the same prompts. Slot arrival (in-flight layer): `concurrency` slots each replay a
deterministic session sequence back to back (slot_sessions).

Every session carries its own nonce in the system prompt: turns of one session
share a cached prefix, as multi-turn chat does, while no two sessions (and no two
pairs) do.
"""

import asyncio
import hashlib
import random
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import aiohttp
import msgspec

from gate import config, metrics
from workload import chat
from workload.schema import Session


# The generator's default think-time median (workload/generate.py).
INITIAL_SPREAD_S = 15.0
# Slots start within this many seconds of each other, so their first prefills do not all land at once.
SLOT_STAGGER_S = 2.0
# SGLang's HTTP server closes a keep-alive connection after 5 s idle (SGLANG_TIMEOUT_KEEP_ALIVE); aiohttp keeps pooled
# connections 15 s by default. With think time a session's connection idles past 5 s and the client can reuse a socket
# the server just closed (ServerDisconnectedError at send). Dropping idle connections first avoids the race.
SERVER_KEEPALIVE_S = 5.0
CLIENT_KEEPALIVE_S = 2.0


class SessionStart(msgspec.Struct, frozen=True):
    t_start: float
    session_index: int
    first_turn: int
    nonce: str


class RequestRecord(msgspec.Struct, kw_only=True):
    session_id: str
    nonce: str
    turn: int
    t_due: float
    t_send: float
    t_done: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    max_new_tokens: int = 0
    ok: bool = False
    # Why a request failed: the exception class (client transport, HTTP status), "abort" or "abandoned".
    error_type: str = ""
    error: str = ""
    finish_reason: str = ""
    output_ids: List[int] = []


def plan_open(sessions: Sequence[Session], load: config.SessionLoad, seed: str) -> List[SessionStart]:
    """The fixed offered load of one leg: who starts when, at which turn, under which nonce."""
    rng = random.Random(f"plan/{seed}/{load.name}")
    order = list(range(len(sessions)))
    rng.shuffle(order)
    starts: List[SessionStart] = []

    def take(t: float, mid_conversation: bool) -> None:
        j = len(starts)
        idx = order[j % len(order)]
        first = rng.randrange(len(sessions[idx].turns)) if mid_conversation else 0
        starts.append(SessionStart(round(t, 4), idx, first, f"{seed}-{j}"))

    # Steady state from t=0: sessions already part-way through, their first sends spread over one think time.
    spread = max(1.0, INITIAL_SPREAD_S * load.think_scale)
    for _ in range(load.concurrency):
        take(rng.uniform(0.0, spread), True)
    rate = load.concurrency / load.expected_session_s
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= load.warmup_s + load.window_s:
            return starts
        take(t, False)


def slot_sessions(sessions: Sequence[Session], load: config.SessionLoad, seed: str, slot: int) -> Iterator[SessionStart]:
    """Slot `slot`'s endless session sequence; its first session joins part-way through."""
    rng = random.Random(f"slots/{seed}/{load.name}/{slot}")
    j = 0
    while True:
        idx = rng.randrange(len(sessions))
        first = rng.randrange(len(sessions[idx].turns)) if j == 0 else 0
        yield SessionStart(0.0, idx, first, f"{seed}-s{slot}-{j}")
        j += 1


def plan_digest(plan: Sequence[SessionStart]) -> str:
    return hashlib.sha256(msgspec.json.encode(list(plan))).hexdigest()[:16]


class _Run:
    """One replay: the sessions, the plan and the records it fills."""

    def __init__(self, url: str, sessions: Sequence[Session], load: config.SessionLoad, tokenizer,
                 nonce_at: str, keep_outputs: bool):
        self.url = url
        self.sessions = sessions
        self.load = load
        self.tokenizer = tokenizer
        self.nonce_at = nonce_at
        self.keep_outputs = keep_outputs
        self.scripted = load.mode == "scripted"
        if load.mode not in ("scripted", "closed"):
            raise ValueError(f"mode {load.mode!r}")
        self.records: List[RequestRecord] = []
        self.t0 = 0.0
        self.t_end = load.warmup_s + load.window_s

    def now(self) -> float:
        return time.perf_counter() - self.t0

    async def _sleep_until(self, t: float) -> None:
        delay = t - self.now()
        if delay > 0:
            await asyncio.sleep(delay)

    def _payload(self, ids: List[int], max_new: int) -> Dict:
        sp = {"temperature": self.load.temperature, "max_new_tokens": max_new, "ignore_eos": self.scripted}
        if self.load.temperature > 0:
            # Gemma-4's generation_config sampling.
            sp.update(top_p=0.95, top_k=64)
        return {"input_ids": ids, "sampling_params": sp, "stream": False}

    async def _send(self, http: aiohttp.ClientSession, rec: RequestRecord, ids: List[int]) -> Optional[str]:
        rec.t_send = self.now()
        try:
            async with http.post(f"{self.url}/generate", json=self._payload(ids, rec.max_new_tokens)) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {str(body)[:200]}")
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as e:
            rec.t_done, rec.error_type, rec.error = self.now(), type(e).__name__, repr(e)[:300]
            return None
        rec.t_done = self.now()
        meta = body["meta_info"]
        rec.prompt_tokens = meta["prompt_tokens"]
        rec.cached_tokens = meta.get("cached_tokens", 0)
        rec.output_tokens = meta["completion_tokens"]
        fr = meta.get("finish_reason")
        rec.finish_reason = fr.get("type", "") if isinstance(fr, dict) else str(fr or "")
        rec.ok = rec.finish_reason != "abort"
        if not rec.ok:
            rec.error_type, rec.error = "abort", str(fr)[:300]
        if self.keep_outputs:
            rec.output_ids = list(body.get("output_ids") or [])
        return body.get("text", "")

    async def session(self, http: aiohttp.ClientSession, start: SessionStart) -> bool:
        """Replays one session; True when it stopped because its next turn falls after the window."""
        s = self.sessions[start.session_index]
        replies = [t.reply for t in s.turns]
        due = start.t_start
        for k in range(start.first_turn, len(s.turns)):
            turn = s.turns[k]
            if k > start.first_turn:
                due = self.now() + turn.think_s * self.load.think_scale
            if due >= self.t_end:
                return True
            await self._sleep_until(due)
            msgs = chat.messages_for_turn(s, k, replies, start.nonce, self.nonce_at)
            ids = await asyncio.get_running_loop().run_in_executor(None, chat.prompt_ids, self.tokenizer, msgs)
            max_new = turn.max_new_tokens if self.scripted else config.MAX_OUTPUT_TOKENS
            rec = RequestRecord(session_id=s.session_id, nonce=start.nonce, turn=k, t_due=due, t_send=0.0,
                                t_done=0.0, max_new_tokens=max_new)
            self.records.append(rec)
            text = await self._send(http, rec, ids)
            if text is None or not rec.ok:
                return False
            if not self.scripted:
                replies[k] = text
        return False

    async def slot(self, http: aiohttp.ClientSession, starts: Iterator[SessionStart], t_first: float) -> None:
        await self._sleep_until(t_first)
        for st in starts:
            if self.now() >= self.t_end:
                return
            # A session cut by the window end must not hand its slot to a fresh session: with think time that
            # would fire every slot's uncached first turn in the window's last think period.
            if await self.session(http, msgspec.structs.replace(st, t_start=self.now())):
                return


async def _scrape(url: str, run: _Run, out: List[Dict]) -> None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http:
            async with http.get(f"{url}/metrics") as resp:
                text = await resp.text()
        out.append({"t": run.now(), **metrics.prom_sample(text)})
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass


async def _sample_metrics(url: str, run: _Run, out: List[Dict], period_s: float = 2.0) -> None:
    while True:
        await _scrape(url, run, out)
        await asyncio.sleep(period_s)


def slot_first_sends(load: config.SessionLoad, seed: str, t_end: float) -> List[float]:
    """Each slot's first send time. With think time the slots stand for live chat sessions, so their first sends
    spread over one think time (as the poisson layer's do) instead of landing as one burst."""
    rng = random.Random(f"slots/{seed}/{load.name}")
    spread = max(SLOT_STAGGER_S, INITIAL_SPREAD_S * load.think_scale)
    stagger = min(spread, 0.1 * t_end)
    return [rng.uniform(0.0, stagger) for _ in range(load.concurrency)]


def _tasks(run: _Run, http: aiohttp.ClientSession, seed: str) -> Tuple[List, str]:
    load, sessions = run.load, run.sessions
    if load.arrival == "poisson":
        plan = plan_open(sessions, load, seed)
        return [asyncio.create_task(run.session(http, st)) for st in plan], plan_digest(plan)
    if load.arrival == "slots":
        firsts = slot_first_sends(load, seed, run.t_end)
        digest = hashlib.sha256(f"slots/{seed}/{msgspec.json.encode(load).decode()}".encode()).hexdigest()[:16]
        return [asyncio.create_task(run.slot(http, slot_sessions(sessions, load, seed, i), firsts[i]))
                for i in range(load.concurrency)], digest
    raise ValueError(f"arrival {load.arrival!r}")


async def replay(url: str, sessions: Sequence[Session], load: config.SessionLoad, tokenizer, seed: str,
                 nonce_at: str = "start", keep_outputs: bool = False, sample_metrics: bool = True) -> Dict:
    """Replays `load` (its plan drawn from `seed`); requests still open drain_timeout_s after the window are
    abandoned and recorded as errors."""
    run = _Run(url, sessions, load, tokenizer, nonce_at, keep_outputs)
    samples: List[Dict] = []
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=CLIENT_KEEPALIVE_S)
    timeout = aiohttp.ClientTimeout(total=load.window_s + load.warmup_s + load.drain_timeout_s)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
        run.t0 = time.perf_counter()
        sampler = asyncio.create_task(_sample_metrics(url, run, samples)) if sample_metrics else None
        tasks, digest = _tasks(run, http, seed)
        done, pending = await asyncio.wait(tasks, timeout=run.t_end + load.drain_timeout_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if sampler is not None:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            # Closes the window's counter delta even when the last reply beat the next periodic sample.
            await _scrape(url, run, samples)
        wall = run.now()
    abandoned = 0
    for r in run.records:
        if r.t_done == 0.0:
            r.t_done, r.error, abandoned = wall, "abandoned after drain timeout", abandoned + 1
            r.error_type = "abandoned"
    errors = [t.exception() for t in done if not t.cancelled() and t.exception() is not None]
    if errors:
        raise RuntimeError(f"{len(errors)} session tasks crashed; first: {errors[0]!r}")
    records = [msgspec.to_builtins(r) for r in run.records]
    return {"records": records, "wall_s": wall, "abandoned": abandoned, "plan_digest": digest,
            "metric_samples": samples, "failed": failed_records(records)}


# Failed requests whose details a run summary keeps (the rest are counted in n_failed).
FAILED_RECORDS_KEPT = 20
_FAILED_FIELDS = ("session_id", "turn", "t_due", "t_send", "t_done", "error_type", "error", "finish_reason",
                  "prompt_tokens", "output_tokens")


def failed_records(records: Sequence[Dict], limit: int = FAILED_RECORDS_KEPT) -> List[Dict]:
    """The first `limit` failed requests in send order, with why they failed."""
    failed = sorted((r for r in records if not r["ok"]), key=lambda r: r["t_send"])
    return [{k: r[k] for k in _FAILED_FIELDS} for r in failed[:limit]]
