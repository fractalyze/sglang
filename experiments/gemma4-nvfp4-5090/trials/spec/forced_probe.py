"""T-SPEC4 diagnosis: the gate's teacher-forced logprob pass at two concurrencies, one server lifetime.

T-SPEC4's candidate matched its control bit for bit on 19 of 22 forced prompts and differed on
every position of the 3 long ones. This reruns the gate's own forced pass at the gate's
concurrency and one request at a time, so a drafter effect (differs at both) can be told
apart from prefill batch composition (differs only when requests share batches):
  python -m trials.spec.forced_probe --ref base4-spec --out DIR [-- extra sglang flags]
"""

import argparse
import json
import os

from gate import fidelity, hostwatch, server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    reference = fidelity.load_json(fidelity.REFERENCE_PATH)
    out = {"ref": args.ref, "extra_args": args.extra}
    with hostwatch.host_lock(), server.Server(server.load_ref(args.ref), os.path.join(args.out, "server.log"),
                                              args.extra) as srv:
        out["commit"] = srv.commit
        for concurrency in (fidelity.REFERENCE_CONCURRENCY, 1):
            out[f"forced_c{concurrency}"] = fidelity.run_forced(srv.url, reference, concurrency)
    out["host_summary"] = srv.host_summary
    with open(os.path.join(args.out, "forced.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
