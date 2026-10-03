"""W14 smoke test of gate/spec_replay.patch before any gate spec-run: two lifetimes of one ref.

  1. hook tree, no replay env: free-running outputs of 2 W1 and 8 W8 timed prompts;
  2. hook tree, replay env on those outputs: the same prompts must emit the same tokens
     in exactly ceil((n - 1) / A) verify rounds, and a prompt outside the file must
     still decode with its own acceptance.

  python -m trials.spec.replay_smoke --ref base4-spec-fp8head-fp8lmhead --out DIR
"""

import argparse
import asyncio
import json
import math
import os

from gate import client, config, gpu, hostwatch, prompts, server, specrule


def _decode(srv, ps, n):
    return asyncio.run(client.spec_acceptance(srv.url, ps, n, concurrency=len(ps)))


def _lifetime(ref, log_path, prompt_sets, env=None, attempts=6):
    """Decodes each (name, prompts, max_new, one_at_a_time) set in one server lifetime.

    Retried like a gate leg: bs2's co-tenant can take the GPU between the quiet check and weight load.
    """
    errors = []
    for attempt in range(attempts):
        gpu.wait_quiet(None, [])
        try:
            with server.Server(ref, f"{log_path}.{attempt}", overlay=specrule.OVERLAY, extra_env=env) as srv:
                return {name: ([_decode(srv, [p], n)[0] for p in ps] if single else _decode(srv, ps, n))
                        for name, ps, n, single in prompt_sets}
        except (RuntimeError, hostwatch.HostUnsafe) as e:
            errors.append(repr(e)[:300])
    raise RuntimeError(f"{log_path}: {errors}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = server.load_ref(args.ref)
    tok = prompts.load_tokenizer()
    corpus = prompts.load_corpus(tok)
    w1 = [prompts.timing_prompts(corpus, tok.bos_token_id, f"{config.W1.fixed_prompt_seed}/{r}", 1, 1024)[0]
          for r in range(2)]
    w8 = prompts.timing_prompts(corpus, tok.bos_token_id, f"{config.W8.fixed_prompt_seed}/0", 8, 1024)
    outside = prompts.timing_prompts(corpus, tok.bos_token_id, "replay-smoke-outside", 1, 1024)
    out = {"ref": args.ref, "accept_len": config.SPEC_REPLAY_ACCEPT_LEN}
    replay_path = os.path.join(args.out, "replay.json")
    with hostwatch.host_lock():
        sets = [("W1", w1, 256, True), ("W8", w8, 128, False), ("outside", outside, 256, False)]
        rec = _lifetime(ref, os.path.join(args.out, "record.log"), sets)
        entries = [{"input_ids": p, "output_ids": r["output_ids"]} for p, r in zip(w1 + w8, rec["W1"] + rec["W8"])]
        with open(replay_path, "w") as f:
            json.dump({"continuations": entries}, f)
        env = {"SGLANG_SIMULATE_ACC_REPLAY_PATH": replay_path,
               "SGLANG_SIMULATE_ACC_REPLAY_LEN": str(config.SPEC_REPLAY_ACCEPT_LEN)}
        rep = _lifetime(ref, os.path.join(args.out, "replay.log"), sets, env)
    a = config.SPEC_REPLAY_ACCEPT_LEN
    checks = {}
    for wl, n in (("W1", 256), ("W8", 128)):
        checks[f"{wl}_text_identical"] = [r["output_ids"] == q["output_ids"] for r, q in zip(rep[wl], rec[wl])]
        checks[f"{wl}_rounds"] = [r["verify_ct"] for r in rep[wl]]
        checks[f"{wl}_rounds_expected"] = math.ceil((n - 1) / a)
        checks[f"{wl}_record_tau"] = [q["completion_tokens"] / q["verify_ct"] for q in rec[wl]]
    checks["outside_identical_to_record"] = rep["outside"][0]["output_ids"] == rec["outside"][0]["output_ids"]
    checks["outside_tau"] = [x[0]["completion_tokens"] / x[0]["verify_ct"] for x in (rec["outside"], rep["outside"])]
    out["checks"] = checks
    with open(os.path.join(args.out, "smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
