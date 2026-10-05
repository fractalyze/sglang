import asyncio
import os
import sys
import tempfile

import msgspec
from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gate import checkpoint, config, loadgen, metrics, pd, rpquality, runner, server, stats  # noqa: E402
from workload import chat, generate, personas, schema, sources  # noqa: E402
from workload.schema import Message, Session, Turn  # noqa: E402


class CharTokenizer:
    """One token per character; a chat template whose history renders a model turn without the generation marker.

    Like Gemma-4's template, the generation prompt (model turn + empty thought channel) is not how the
    same turn renders once it is history, so a cached prefix ends where the previous reply began.
    """

    BOS, TURN, END, GEN = 1, 2, 3, 4

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in ids if i >= 10)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, return_dict=False):
        ids = [self.BOS]
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            ids += [self.TURN] + self.encode(role + "\n" + m["content"]) + [self.END]
        if add_generation_prompt:
            ids += [self.TURN] + self.encode("model\n") + [self.GEN]
        return ids


TOK = CharTokenizer()


def _session(sid="s0", n_turns=3, lang="en", history=2):
    hist = []
    for i in range(history):
        hist += [Message("user", f"hello {i}"), Message("assistant", f"reply {i}")]
    turns = [Turn(user=f"turn {k}", reply=f"scripted {k}", think_s=0.0 if k == 0 else 0.01, max_new_tokens=8)
             for k in range(n_turns)]
    return Session(session_id=sid, language=lang, system="You are X.", history=hist, turns=turns)


class SchemaTest(absltest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            schema.write(path, [_session("a"), _session("b")])
            got = schema.read(path)
        self.assertEqual([s.session_id for s in got], ["a", "b"])
        self.assertEqual(got[0], _session("a"))

    def test_rejects_history_ending_on_user(self):
        s = msgspec.structs.replace(_session(), history=[Message("user", "x")])
        with self.assertRaisesRegex(ValueError, "ends on a user"):
            schema.validate(s)

    def test_rejects_non_alternating_history(self):
        s = msgspec.structs.replace(_session(), history=[Message("assistant", "x"), Message("user", "y")])
        with self.assertRaisesRegex(ValueError, "alternating"):
            schema.validate(s)

    def test_rejects_empty_turns(self):
        with self.assertRaisesRegex(ValueError, "no turns"):
            schema.validate(msgspec.structs.replace(_session(), turns=[]))


class ChatTest(absltest.TestCase):
    def test_turn_messages_carry_previous_replies(self):
        s = _session(history=1)
        msgs = chat.messages_for_turn(s, 2, ["r0", "r1", "r2"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "assistant", "user",
                                                     "assistant", "user"])
        self.assertEqual(msgs[4]["content"], "r0")
        self.assertEqual(msgs[-1]["content"], "turn 2")

    def test_nonce_positions(self):
        s = _session()
        self.assertTrue(chat.system_text(s, "n1", "start").startswith("[session n1]"))
        self.assertTrue(chat.system_text(s, "n1", "after_system").endswith("[session n1]"))
        self.assertEqual(chat.system_text(s, "", "start"), s.system)
        with self.assertRaises(ValueError):
            chat.system_text(s, "n1", "middle")

    def test_next_turn_extends_previous_prompt_up_to_generation_marker(self):
        s = _session()
        replies = [t.reply for t in s.turns]
        p0 = chat.prompt_ids(TOK, chat.messages_for_turn(s, 0, replies, "n"))
        p1 = chat.prompt_ids(TOK, chat.messages_for_turn(s, 1, replies, "n"))
        self.assertEqual(p1[: len(p0) - 1], p0[:-1])

    def test_too_few_replies(self):
        with self.assertRaises(ValueError):
            chat.messages_for_turn(_session(), 2, ["r0"])


def _pool(langs=("en", "ko"), convs=40, pairs_per_conv=4, user_len=60, reply_len=400):
    out = {}
    for lang in langs:
        out[lang] = [sources.Pair(lang, f"{lang} user {c}.{p} " + "u" * user_len,
                                  f"{lang} reply {c}.{p}. " + ("word. " * (reply_len // 6)), c % 3 == 0, f"{lang}{c}")
                     for c in range(convs) for p in range(pairs_per_conv)]
    return out


class GenerateTest(parameterized.TestCase):
    def _cfg(self, **kw):
        base = dict(n_sessions=30, seed="t", lang_weights={"en": 0.5, "ko": 0.5}, first_prompt_median=3000,
                    first_prompt_min=1500, first_prompt_max=5000, max_input_tokens=6000, max_new_tokens=120,
                    turns_mean=4.0)
        base.update(kw)
        return generate.GenConfig(**base)

    def test_deterministic(self):
        a = generate.generate(self._cfg(), _pool(), TOK)
        b = generate.generate(self._cfg(), _pool(), TOK)
        self.assertEqual(a, b)
        c = generate.generate(self._cfg(seed="other"), _pool(), TOK)
        self.assertNotEqual(a, c)

    def test_sessions_respect_input_and_output_caps(self):
        cfg = self._cfg()
        for s in generate.generate(cfg, _pool(), TOK):
            schema.validate(s)
            self.assertIn(s.persona_id, [p.persona_id for p in personas.PERSONAS[s.language]])
            self.assertEqual(s.turns[0].think_s, 0.0)
            replies = [t.reply for t in s.turns]
            for k, t in enumerate(s.turns):
                self.assertLessEqual(t.max_new_tokens, cfg.max_new_tokens)
                self.assertGreaterEqual(t.max_new_tokens, cfg.min_reply_tokens)
                self.assertLessEqual(len(TOK.encode(t.reply)), cfg.max_new_tokens)
                n = len(chat.prompt_ids(TOK, chat.messages_for_turn(s, k, replies)))
                if k:
                    self.assertLessEqual(n, cfg.max_input_tokens + 32)
                    self.assertGreaterEqual(t.think_s, cfg.think_min_s)

    def test_first_prompt_tracks_target(self):
        cfg = self._cfg(first_prompt_sigma=0.0, first_prompt_median=3000)
        for s in generate.generate(cfg, _pool(), TOK):
            n = len(chat.prompt_ids(TOK, chat.messages_for_turn(s, 0, [t.reply for t in s.turns])))
            self.assertBetween(n, 2500, 3600)

    def test_languages_follow_weights(self):
        sessions = generate.generate(self._cfg(n_sessions=40, lang_weights={"en": 0.0, "ko": 1.0}), _pool(), TOK)
        self.assertEqual({s.language for s in sessions}, {"ko"})

    def test_missing_language_raises(self):
        with self.assertRaisesRegex(ValueError, "no language"):
            generate.generate(self._cfg(lang_weights={"fr": 1.0}), _pool(), TOK)

    @parameterized.parameters(("Short.", 50, "Short."),
                              ("One. Two. Three four five six", 20, "One. Two. Three"))
    def test_truncate_prefers_sentence_end(self, text, n, want_prefix):
        got = generate.truncate_tokens(text, n, TOK.encode, TOK.decode)
        self.assertLessEqual(len(got), n)
        self.assertTrue(got.startswith(want_prefix[: min(len(want_prefix), len(got))]))
        if len(text) > n:
            self.assertTrue(got.endswith(".") or len(got) == n)

    def test_every_language_has_personas(self):
        self.assertEqual(set(personas.LANGUAGES), set(generate.DEFAULT_LANG_WEIGHTS))
        for lang, ps in personas.PERSONAS.items():
            self.assertGreaterEqual(len(ps), 2, lang)


class SourcesTest(absltest.TestCase):
    def _row(self, lang="Korean", toxic=False, conv=None):
        conv = conv or [{"role": "user", "content": "역할극 하자. 너는 기사야."},
                        {"role": "assistant", "content": "좋아요, 저는 기사입니다. 무엇을 도와드릴까요?"},
                        {"role": "user", "content": "```code```"},
                        {"role": "assistant", "content": "코드는 여기 있습니다, 아주 긴 설명입니다."}]
        return {"conversation_hash": "h", "language": lang, "toxic": toxic, "conversation": conv}

    def test_keeps_chat_pairs_and_flags_roleplay(self):
        pairs = list(sources.pairs_from_rows([self._row()]))
        self.assertLen(pairs, 1)
        self.assertEqual(pairs[0].lang, "ko")
        self.assertTrue(pairs[0].rp)

    def test_skips_toxic_and_other_languages(self):
        self.assertEmpty(list(sources.pairs_from_rows([self._row(toxic=True), self._row(lang="Klingon")])))

    def test_cap_per_language(self):
        self.assertLen(list(sources.pairs_from_rows([self._row()] * 5, cap_per_lang=2)), 2)


# ---------------------------------------------------------------------------
# A fake /generate server with a prefix cache, for an end-to-end replay.
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self):
        self.seen = []
        self.requests = []

    def cached(self, ids):
        best = 0
        for s in self.seen:
            n = 0
            for a, b in zip(s, ids):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return best

    async def generate(self, request):
        from aiohttp import web

        body = await request.json()
        ids, sp = body["input_ids"], body["sampling_params"]
        cached = self.cached(ids)
        out = [7] * sp["max_new_tokens"]
        self.seen.append(ids + out)
        self.requests.append(body)
        await asyncio.sleep(0.001)
        return web.json_response({"text": "fake reply", "output_ids": out,
                                  "meta_info": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                                                "cached_tokens": cached, "finish_reason": {"type": "length"}}})

    async def metrics(self, request):
        from aiohttp import web

        n = len(self.requests)
        return web.Response(text=f"# HELP x\nsglang:num_requests_total{{a=\"1\"}} {n}\n"
                                 f"sglang:num_retracted_requests_total 0\nsglang:num_running_reqs 1\n")


async def _replay(sessions, load, seed="p0", **kw):
    from aiohttp import web

    fake = FakeServer()
    app = web.Application()
    app.router.add_post("/generate", fake.generate)
    app.router.add_get("/metrics", fake.metrics)
    srv_runner = web.AppRunner(app)
    await srv_runner.setup()
    site = web.TCPSite(srv_runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        rep = await loadgen.replay(f"http://127.0.0.1:{port}", sessions, load, TOK, seed, **kw)
    finally:
        await srv_runner.cleanup()
    return rep, fake


_FAST = config.SessionLoad(name="fast", concurrency=4, warmup_s=0.2, window_s=1.0, expected_session_s=0.5,
                           think_scale=1.0, drain_timeout_s=5.0)


class LoadgenTest(absltest.TestCase):
    def test_plan_is_deterministic_and_steady_from_start(self):
        sessions = [_session(f"s{i}", n_turns=4) for i in range(10)]
        load = config.SessionLoad(name="x", concurrency=8, warmup_s=10, window_s=100, expected_session_s=20)
        a, b = loadgen.plan_open(sessions, load, "seed"), loadgen.plan_open(sessions, load, "seed")
        self.assertEqual(a, b)
        self.assertNotEqual(a, loadgen.plan_open(sessions, load, "other"))
        self.assertTrue(all(st.first_turn == 0 for st in a[8:]))
        self.assertTrue(all(st.t_start < 110 for st in a))
        self.assertLen({st.nonce for st in a}, len(a))
        # Poisson arrivals at concurrency / expected_session_s over the horizon.
        self.assertBetween(len(a) - 8, 0.6 * 0.4 * 110, 1.4 * 0.4 * 110)

    def test_replay_reuses_prefix_within_session_only(self):
        sessions = [_session(f"s{i}", n_turns=3) for i in range(6)]
        rep, fake = asyncio.run(_replay(sessions, _FAST))
        recs = rep["records"]
        self.assertTrue(all(r["ok"] for r in recs))
        self.assertEqual(rep["abandoned"], 0)
        by_nonce = {}
        for r in recs:
            by_nonce.setdefault(r["nonce"], []).append(r)
        multi = [rs for rs in by_nonce.values() if len(rs) > 1]
        self.assertNotEmpty(multi)
        for rs in multi:
            rs.sort(key=lambda r: r["turn"])
            prev = rs[0]
            for r in rs[1:]:
                # The next turn hits the previous prompt minus the generation marker.
                self.assertGreaterEqual(r["cached_tokens"], prev["prompt_tokens"] - 1)
                prev = r
        # A session's first request shares only "<bos><turn>system\n[session <seed>-" with other sessions.
        firsts = [rs[0] for rs in by_nonce.values()]
        self.assertTrue(all(r["cached_tokens"] < 30 for r in firsts))
        self.assertTrue(all(b["sampling_params"]["ignore_eos"] for b in fake.requests))
        self.assertNotEmpty(rep["metric_samples"])

    def test_scripted_replays_identical_prompts(self):
        sessions = [_session(f"s{i}", n_turns=3) for i in range(6)]
        _, a = asyncio.run(_replay(sessions, _FAST, seed="same"))
        _, b = asyncio.run(_replay(sessions, _FAST, seed="same"))
        key = lambda body: tuple(body["input_ids"])  # noqa: E731
        common = {key(x) for x in a.requests} & {key(x) for x in b.requests}
        self.assertGreater(len(common), 0.8 * min(len(a.requests), len(b.requests)))

    def test_slots_keep_requests_in_flight(self):
        sessions = [_session(f"s{i}", n_turns=3) for i in range(6)]
        load = config.SessionLoad(name="slots", arrival="slots", concurrency=3, warmup_s=0.0, window_s=0.5,
                                  expected_session_s=0.0, think_scale=0.0, drain_timeout_s=5.0)
        rep, fake = asyncio.run(_replay(sessions, load))
        recs = rep["records"]
        self.assertGreater(len(recs), 6)
        occ = metrics.occupancy_in_window([(r["t_send"], r["t_done"]) for r in recs], 0.1, 0.5)
        self.assertGreaterEqual(occ["max"], 3)
        self.assertLessEqual(occ["max"], 3)
        rep2, _ = asyncio.run(_replay(sessions, load))
        self.assertEqual(rep["plan_digest"], rep2["plan_digest"])

    def test_closed_mode_appends_model_reply(self):
        sessions = [_session(f"s{i}", n_turns=2) for i in range(4)]
        load = msgspec.structs.replace(_FAST, mode="closed")
        _, fake = asyncio.run(_replay(sessions, load))
        texts = [TOK.decode(b["input_ids"]) for b in fake.requests]
        self.assertTrue(any("fake reply" in t for t in texts))
        self.assertTrue(all(not b["sampling_params"]["ignore_eos"] for b in fake.requests))
        self.assertTrue(all(b["sampling_params"]["max_new_tokens"] == config.MAX_OUTPUT_TOKENS
                            for b in fake.requests))


def _rec(t_send, e2e, ok=True, prompt=5000, cached=4000, out=200, due=None, nonce="n", turn=0, ids=None):
    return {"t_send": t_send, "t_done": t_send + e2e, "t_due": t_send if due is None else due, "ok": ok,
            "prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": out, "nonce": nonce, "turn": turn,
            "output_ids": ids or []}


class MetricsTest(parameterized.TestCase):
    @parameterized.parameters((50, 3.0), (90, 4.6), (0, 1.0), (100, 5.0), (25, 2.0))
    def test_percentile_interpolates(self, q, want):
        self.assertAlmostEqual(metrics.percentile([1, 2, 3, 4, 5], q), want)

    def test_summary_windows_latency_by_send_and_throughput_by_finish(self):
        recs = [_rec(1, 1), _rec(5, 2), _rec(9, 3), _rec(11, 1)]
        s = metrics.summarize(recs, warmup_s=4, window_s=6)
        self.assertEqual(s["n_sent"], 2)
        self.assertAlmostEqual(s["e2e_p50_s"], 2.5)
        # Finished inside [4, 10): t_done 7 only (12 is past the window).
        self.assertEqual(s["n_finished_in_window"], 1)
        self.assertAlmostEqual(s["output_tok_s_per_gpu"], 200 / 6)
        self.assertAlmostEqual(s["total_tok_s_per_gpu"], 5200 / 6)
        self.assertAlmostEqual(s["prefix_cache_hit_rate"], 0.8)

    def test_failed_requests_counted(self):
        s = metrics.summarize([_rec(5, 1, ok=False), _rec(5, 1)], 0, 10)
        self.assertEqual(s["n_failed"], 1)
        self.assertFalse(metrics.meets_slo(s, 10))

    def test_cost_formula(self):
        self.assertAlmostEqual(metrics.cost_per_mtok(0.70, 1000.0), 0.70 / 3.6)
        table = metrics.cost_table(1000.0)
        self.assertLen(table, len(config.GPU_PRICES_USD_PER_HR))
        self.assertEqual(metrics.cost_per_mtok(1.0, 0.0), float("inf"))

    def test_capacity_picks_most_inflight_point_meeting_each_slo(self):
        def pt(name, p90, inflight, tput):
            s = metrics.summarize([_rec(1, p90 * 0.5), _rec(2, p90)], 0, 10)
            s.update(inflight_mean=inflight, output_tok_s_per_gpu=tput, total_tok_s_per_gpu=tput * 20)
            return {"load": name, "summary": s, "retractions": {"requests": 0}}

        pts = [pt("a", 5, 8, 100), pt("b", 9, 16, 300), pt("c", 14, 32, 500), pt("d", 30, 64, 600)]
        cap = metrics.capacity_at_slos(pts, slos=(6, 10, 15))
        self.assertEqual(cap["capacity"]["6"]["load"], "a")
        self.assertEqual(cap["capacity"]["10"]["load"], "b")
        self.assertEqual(cap["capacity"]["15"]["load"], "c")
        self.assertAlmostEqual(cap["capacity"]["10"]["inflight_per_gpu"], 16)
        self.assertAlmostEqual(cap["capacity"]["10"]["goodput_output_tok_s_per_gpu"], 300)
        self.assertAlmostEqual(cap["capacity"]["10"]["usd_per_mtok_total"]["0.70"], 0.70 / (6000 * 3600) * 1e6)
        self.assertEqual([r["meets_slo"]["10"] for r in cap["points"]], [True, True, False, False])
        self.assertIsNone(metrics.capacity_at_slo([pt("d", 30, 64, 600)], 10))

    def test_occupancy_counts_overlap_with_window(self):
        occ = metrics.occupancy_in_window([(0, 4), (2, 6), (5, 20)], 2, 10)
        # Inside [2, 10): 2 + 4 + 5 seconds of open spans.
        self.assertAlmostEqual(occ["mean"], 11 / 8)
        self.assertEqual(occ["max"], 2)

    def test_inflight_and_sessions_layers(self):
        recs = [_rec(0, 2, nonce="a", turn=0), _rec(6, 2, nonce="a", turn=1), _rec(0, 10, nonce="b")]
        s = metrics.summarize(recs, 0, 10)
        self.assertAlmostEqual(s["inflight_mean"], (2 + 2 + 10) / 10)
        # Session a lives 0..8 including its think time, b 0..10.
        self.assertAlmostEqual(s["sessions_active_mean"], 1.8)
        self.assertAlmostEqual(s["per_request_out_tok_s_p50"], 100)

    def test_prometheus_parse_and_window_delta(self):
        text = ('# HELP a\nsglang:num_retracted_requests_total{tp="0"} 3\nsglang:num_retracted_requests_total{tp="1"} 2\n'
                'sglang:num_running_reqs 7\nsglang:time_to_first_token_seconds_bucket{le="1"} 9\n')
        v = metrics.parse_prom(text)
        self.assertEqual(v["sglang:num_retracted_requests_total"], 5)
        self.assertNotIn("sglang:time_to_first_token_seconds_bucket", v)
        samples = [{"t": 0, "sglang:num_retracted_requests_total": 1}, {"t": 5, "sglang:num_retracted_requests_total": 2},
                   {"t": 11, "sglang:num_retracted_requests_total": 9}]
        d = metrics.window_counter_delta(samples, 5, 10)
        self.assertEqual(d["sglang:num_retracted_requests_total"], 7)
        self.assertIsNone(metrics.window_counter_delta(samples, 5, 20))
        self.assertEqual(metrics.retractions(d, {"events": 0, "requests": 0})["requests"], 7)

    def test_retractions_from_log(self):
        log = ("KV cache pool is full. Retract requests. #retracted_reqs: 3, #new_tokens_gained: 10\n"
               "Decode batch ...\nKV cache pool is full. Retract requests. #retracted_reqs: 2, #new_tokens_gained: 4\n")
        self.assertEqual(metrics.retractions_from_log(log), {"events": 2, "requests": 5})
        self.assertEqual(metrics.retractions(None, {"events": 2, "requests": 5})["source"], "log")
        # Metrics on but the counter never incremented: zero retractions, not a fallback.
        none = metrics.retractions({"sglang:prompt_tokens_total": 10.0}, {"events": 0, "requests": 0})
        self.assertEqual((none["requests"], none["source"]), (0, "metrics"))


def _leg(e2e, out_tok=200, n=20, ids=None, retracted=0, window=10.0):
    recs = [_rec(1 + i * 0.4, e2e, out=out_tok, nonce=f"n{i}", ids=ids) for i in range(n)]
    load = {"name": "t", "warmup_s": 0.0, "window_s": window, "mode": "scripted"}
    return {"load": load, "replay": {"records": recs, "abandoned": 0, "plan_digest": "d"},
            "summary": metrics.summarize(recs, 0.0, window), "retractions": {"requests": retracted}}


class StatsTest(absltest.TestCase):
    def test_identical_legs_unit_gains(self):
        s = stats.summarize_pairs([_leg(2.0)] * 4, [_leg(2.0)] * 4)
        for m in stats.PAIRED_METRICS:
            self.assertAlmostEqual(s["overall"][m], 1.0)

    def test_faster_candidate_gains_above_one(self):
        s = stats.summarize_pairs([_leg(4.0)] * 4, [_leg(2.0)] * 4)
        self.assertAlmostEqual(s["overall"]["e2e_p90_gain"], 2.0)
        v = stats.timing_verdict(s, {}, "e2e_p90_gain")
        self.assertTrue(v["promote"])

    def test_throughput_regression_blocks(self):
        s = stats.summarize_pairs([_leg(4.0, out_tok=200)] * 4, [_leg(2.0, out_tok=100)] * 4)
        v = stats.timing_verdict(s, {}, "e2e_p90_gain")
        self.assertFalse(v["checks"]["output_tput_gain_no_regression"])
        self.assertFalse(v["promote"])

    def test_too_few_pairs(self):
        s = stats.summarize_pairs([_leg(4.0)] * 2, [_leg(2.0)] * 2)
        self.assertFalse(stats.timing_verdict(s, {})["checks"]["enough_pairs"])

    def test_unknown_metric(self):
        with self.assertRaises(ValueError):
            stats.timing_verdict(stats.summarize_pairs([_leg(1.0)] * 4, [_leg(1.0)] * 4), {}, "w8_composite")

    def test_retractions_and_hit_rate_reported(self):
        s = stats.summarize_pairs([_leg(2.0, retracted=3)] * 4, [_leg(2.0)] * 4)
        self.assertEqual(s["retractions"]["control"], [3] * 4)
        self.assertAlmostEqual(s["hit_rate"]["candidate"], 0.8)


class RunnerTest(absltest.TestCase):
    def _full_leg(self, **kw):
        leg = _leg(2.0, **kw)
        leg.update(telemetry={"foreign_seen": False, "thermal_or_hw_throttle_samples": 0},
                   weights_at_load={"ok": True, "checksum": "x"}, weights_at_end={"ok": True, "checksum": "x"},
                   host={"tripped": None, "peaks_by_phase": {"timed": {"peak_load1": 3}}},
                   decode_steps={"eager_steps": 5, "graph_steps": 100})
        return leg

    def test_integrity_passes_clean_leg_and_reports_eager_steps(self):
        integ = runner.leg_integrity(self._full_leg())
        self.assertTrue(integ["ok"], integ)
        self.assertEqual(integ["reported"]["eager_decode_steps"], 5)

    def test_integrity_fails_on_failed_request_or_client_lag(self):
        leg = self._full_leg()
        leg["summary"]["n_failed"] = 1
        self.assertFalse(runner.leg_integrity(leg)["checks"]["no_failed_requests"])
        leg = self._full_leg()
        leg["summary"]["client_lag_p99_s"] = 5.0
        self.assertFalse(runner.leg_integrity(leg)["checks"]["offered_load_kept"])

    def test_agreement_pairs_by_session_and_turn(self):
        a = _leg(2.0, ids=[1, 2, 3, 4])
        b = _leg(2.0, ids=[1, 2, 9, 9])
        res = runner.timed_output_agreement(a, b)
        self.assertEqual(res["n_streams"], 20)
        self.assertAlmostEqual(res["mean"], 0.5)

    def test_agreement_hard_only_for_scripted_aa(self):
        meta = {"load": {"mode": "scripted"}, "control": {"name": "base"}, "candidate": {"name": "base"}}
        low = [{"mean": 0.1, "n_streams": 3}]
        self.assertFalse(runner.agreement_check(meta, low, {})["ok"])
        meta["candidate"] = {"name": "trial"}
        self.assertTrue(runner.agreement_check(meta, low, {})["ok"])
        meta = {"load": {"mode": "closed"}, "control": {"name": "base"}, "candidate": {"name": "base"}}
        self.assertTrue(runner.agreement_check(meta, low, {})["ok"])

    def test_server_arg_diff_needs_declaration(self):
        a = {"server_info": {"kv_cache_dtype": "fp8_e4m3", "launch_command": "x --kv-cache-dtype fp8_e4m3"}}
        b = {"server_info": {"kv_cache_dtype": "auto", "launch_command": "x"}}
        self.assertNotEmpty(runner.server_arg_diff(a, b, [])["undeclared"])
        self.assertEmpty(runner.server_arg_diff(a, b, ["--kv-cache-dtype", "fp8_e4m3"])["undeclared"])

    def test_metrics_flag_added_once(self):
        self.assertEqual(runner.server_extra_args({"server_args": []}), ["--enable-metrics"])
        self.assertEqual(runner.server_extra_args({"server_args": ["--enable-metrics"]}), [])


class RefsTest(absltest.TestCase):
    def _write(self, d, name, refs):
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(msgspec.json.encode(refs).decode())
        return path

    def test_workstream_refs_merge_and_names_stay_unique(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, "a.json", {"base": {"commit": "x"}})
            b = self._write(d, "b.json", {"mem-x": {"commit": "y", "extends": "base"}})
            self.assertEqual(sorted(server.all_refs([a, b])), ["base", "mem-x"])
            dup = self._write(d, "c.json", {"base": {"commit": "z"}})
            with self.assertRaisesRegex(ValueError, "defined twice"):
                server.all_refs([a, dup])

    def test_repo_refs_load(self):
        refs = server.all_refs()
        self.assertIn("base", refs)
        for name in refs:
            self.assertEqual(server.load_ref(name)["name"], name)


class CheckpointTest(absltest.TestCase):
    def test_compare_against_pin(self):
        self.assertTrue(checkpoint.compare({"a": "1", "b": "2"}, {"a": "1", "b": "2"})["matches_pin"])
        bad = checkpoint.compare({"a": "1", "b": "3"}, {"a": "1", "b": "2"})
        self.assertFalse(bad["matches_pin"])
        self.assertEqual(bad["mismatched"], ["b"])
        self.assertEqual(checkpoint.compare({"a": "1"}, {"a": "1", "c": "9"})["mismatched"], ["c"])
        self.assertFalse(checkpoint.compare({}, {})["matches_pin"])


class RpQualityTest(parameterized.TestCase):
    @parameterized.parameters(
        ("안녕하세요, 저는 기사입니다. 무엇을 도와드릴까요?", "ko"),
        ("こんにちは、私は巫女です。お祭りへようこそ。", "ja"),
        ("客官请坐，今天想喝点什么茶？", "zh"),
        ("Здравствуйте, я библиотекарь. Чем могу помочь?", "ru"),
        ("Well, love, the tea is getting cold and the storm is coming.", "en"),
        ("Hola, mi arma, la cocina está lista y el pescado no ha llegado.", "es"),
        ("Bonjour, je suis peintre et la chambre est libre pour toi.", "fr"),
        ("Na, wat ist denn mit dem Auto los? Ich schaue mir das an.", "de"),
    )
    def test_detect_language(self, text, lang):
        self.assertEqual(rpquality.detect_language(text), lang)

    def test_item_score_needs_both_orders(self):
        self.assertEqual(rpquality.item_score("B", "A"), 1)
        self.assertEqual(rpquality.item_score("A", "B"), -1)
        self.assertEqual(rpquality.item_score("A", "A"), 0)
        self.assertEqual(rpquality.item_score(None, "A"), 0)

    def test_parse_judgement(self):
        self.assertEqual(rpquality.parse_judgement(" b\n"), "B")
        self.assertEqual(rpquality.parse_judgement("TIE"), "TIE")
        self.assertIsNone(rpquality.parse_judgement("neither"))

    def test_judge_verdict_fails_clear_loser(self):
        self.assertFalse(rpquality.judge_verdict([-1] * 30 + [0] * 10)["pass"])
        self.assertTrue(rpquality.judge_verdict([0] * 38 + [1, -1])["pass"])

    def test_consistency_verdict(self):
        base = [{"ref_nll": 0.5, "reply_language": "ko", "language": "ko"}] * 4
        ok = [{"ref_nll": 0.51, "reply_language": "ko", "language": "ko"}] * 4
        bad = [{"ref_nll": 0.6, "reply_language": "en", "language": "ko"}] * 4
        self.assertTrue(rpquality.consistency_verdict(base, ok)["pass"])
        v = rpquality.consistency_verdict(base, bad)
        self.assertFalse(v["checks"]["nll_rise_within_budget"])
        self.assertFalse(v["checks"]["language_adherence_not_lower"])

    def test_build_items_per_language(self):
        sessions = [_session(f"s{i}", lang=("en" if i % 2 else "ko")) for i in range(30)]
        items = rpquality.build_items(sessions, TOK)
        self.assertLen(items, 2 * rpquality.ITEMS_PER_LANGUAGE)
        self.assertEqual(items, rpquality.build_items(sessions, TOK))


async def _with_fake_server(fn):
    from aiohttp import web

    fake = FakeServer()
    app = web.Application()
    app.router.add_post("/generate", fake.generate)
    srv_runner = web.AppRunner(app)
    await srv_runner.setup()
    site = web.TCPSite(srv_runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn, f"http://127.0.0.1:{port}"), fake
    finally:
        await srv_runner.cleanup()


class PdMeasureTest(absltest.TestCase):
    def setUp(self):
        self.sessions = [_session(f"s{i}", n_turns=3, history=20) for i in range(5)]

    def test_prompts_have_exact_length_and_distinct_nonces(self):
        ps = pd.prompts_of_length(self.sessions, TOK, 200, 6, "seed")
        self.assertLen(ps, 6)
        self.assertTrue(all(len(p) == 200 for p in ps))
        self.assertLen({tuple(p) for p in ps}, 6)

    def test_prefill_point_counts_uncached_tokens(self):
        ps = pd.prompts_of_length(self.sessions, TOK, 200, 8, "pre")
        res, fake = asyncio.run(_with_fake_server(lambda url: pd.prefill_point(url, ps, in_flight=4)))
        self.assertEqual(res["n"], 8)
        self.assertTrue(all(b["sampling_params"]["max_new_tokens"] == 1 for b in fake.requests))
        self.assertLess(res["hit_rate"], 0.2)
        self.assertGreater(res["prefill_tok_s"], 0)

    def test_decode_point_runs_on_cached_prefixes(self):
        ps = pd.prompts_of_length(self.sessions, TOK, 200, 4, "dec")
        res, fake = asyncio.run(_with_fake_server(lambda url: pd.decode_point(url, ps)))
        self.assertEqual(res["batch"], 4)
        self.assertGreater(res["hit_rate"], 0.95)
        self.assertEqual(sum(b["sampling_params"]["max_new_tokens"] == pd.DECODE_TOKENS for b in fake.requests), 4)


class PdModelTest(absltest.TestCase):
    def test_kv_bytes_match_the_study_numbers(self):
        self.assertEqual(pd.SLIDING_KV_BYTES_PER_TOKEN, 102400)
        self.assertEqual(pd.FULL_KV_BYTES_PER_TOKEN, 10240)
        # 5K-token session: ~105 MB sliding window + ~51 MB full layers.
        self.assertAlmostEqual(pd.kv_transfer_bytes(5000) / 1e6, 104.8576 + 51.2, places=3)
        self.assertEqual(pd.kv_transfer_bytes(500), 500 * (102400 + 10240))

    def test_fleet_model_ratio_and_costs(self):
        wl = pd.Workload(mean_prompt_tokens=5000, hit_rate=0.9, mean_output_tokens=200)
        m = pd.fleet_model(wl, prefill_tok_s=10000, decode_tok_s=2000, colocated_out_tok_s=1000, link_gbps=100)
        self.assertAlmostEqual(m["prefill_gpu_s_per_request"], 0.05)
        self.assertAlmostEqual(m["decode_gpu_s_per_request"], 0.1)
        self.assertAlmostEqual(m["p_to_d_gpu_ratio"], 0.5)
        self.assertAlmostEqual(m["disagg_out_tok_s_per_gpu"], 200 / 0.15)
        self.assertAlmostEqual(m["kv_transfer"]["latency_s"], pd.kv_transfer_bytes(5000) * 8 / 100e9)
        self.assertLess(m["usd_per_mtok_output"]["disagg"]["0.70"], m["usd_per_mtok_output"]["colocated"]["0.70"])
        self.assertNotIn("latency_s", pd.fleet_model(wl, 10000, 2000, 1000)["kv_transfer"])

    def test_best_decode_respects_tpot_and_cache(self):
        pts = [{"decode_tok_s": 100, "tpot_p90_s": 0.01, "hit_rate": 0.99},
               {"decode_tok_s": 300, "tpot_p90_s": 0.02, "hit_rate": 0.99},
               {"decode_tok_s": 500, "tpot_p90_s": 0.05, "hit_rate": 0.99},
               {"decode_tok_s": 400, "tpot_p90_s": 0.02, "hit_rate": 0.5}]
        self.assertEqual(pd.best_decode(pts, 0.03)["decode_tok_s"], 300)
        self.assertIsNone(pd.best_decode(pts, 0.001))

    def test_fleet_size(self):
        wl = pd.Workload(mean_prompt_tokens=5000, hit_rate=0.9, mean_output_tokens=200)
        f = pd.fleet_size(wl, sessions=2200, think_s=18, e2e_s=4, prefill_tok_s=10000, decode_tok_s=2000,
                          colocated_out_tok_s=1000)
        self.assertAlmostEqual(f["requests_per_s"], 100)
        self.assertEqual(f["colocated_gpus"], 20)
        self.assertEqual(f["disagg_prefill_gpus"], 5)
        self.assertEqual(f["disagg_decode_gpus"], 10)


if __name__ == "__main__":
    absltest.main()
