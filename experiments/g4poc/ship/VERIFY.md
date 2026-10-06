# SHIP: verification of the ship branch `jumanzii/g4poc-ship`

The ship branch is upstream `a9871012ac`, nine lever commits and `G4POC-DEPLOY.md`. Section 9 of that guide summarizes
these checks.

- **Tree checked:** ship head `e652f30b1d`, `python/` tree `bd956697e2`. The guide's commit on top changes only the
  guide.
- **Host:** build-server-3's RTX 5090, 2026-10-06 18:21-19:31 KST.
- **How:** each check is one unit of `ship/ship_unit.sh` under `$G4POC/unit.lock`, using the refs in
  `ship/refs.json`. Records are in `ship/runs/`; the run directories are under `/data/jooman/g4poc/runs` on bs3.

| unit | check | result | record |
|---|---|---|---|
| tests | each ship commit's own test files at that commit, then all of them at the head (GPU tests under the host lock) | all pass; head: 1,375 passed, 0 failed | `logs/ship-tests2.log` on bs3 |
| exact | identity, `final-hc-cp2048-lpm-glue-c1` (cfc12c0bac) vs `ship-inflight`: greedy, concurrency 1, 4 sessions x 3 turns | 12/12 identical, 12/12 same `cached_tokens` | `runs/exact-identity-inflight.json` |
| exact | identity, `final-mem-c1-c2a-glue-c1` (cfc12c0bac) vs `ship-chat` | 12/12, 12/12 | `runs/exact-identity-chat.json` |
| exact | HiCache exactness, `ship-inflight-ctl` vs `ship-inflight-smallpool` (16K device pool: later turns load back) | 12/12, 12/12. Each arm is also 12/12 against K2's runs of the same pair on cfc12c0bac (`exactmt-final-mem-c1-c2a-cp2048-lpm-glue-c1-20261006-161214-*`, `exactmt-final-hc-cp2048-lpm-glue-c1-smallpool-20261006-161303-*`) | `runs/exact-hicache.json` |
| exact | upstream equivalence, `upstream-base-inflight` (a9871012ac) vs `ship-defaults-inflight` (every switch unset, `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0`, empty MoE config dir) | 12/12, 12/12. Both server logs show the default MoE config and no channelwise FP8 config | `runs/exact-upstream.json` |
| c28 | A-B-B-A at 28 in flight, `final-hc-cp2048-lpm-glue-c1` vs `ship-inflight` | p90 gain 0.992, tok/s 0.992, p99 gain 1.023, control drift <= 0.5%, 0 failed | `runs/abba-c28.json` |
| c28 | GPU memory, 1 s whole-GPU samples during the A-B-B-A | plateau ~31,350 MiB. 4 samples over 31,642: 2 at the canary's hh:m3:10, 2 at 19:04:18-19 (32,097 MiB, source not recorded) | `runs/smi-c28-summary.json` |
| t30 | `ship-chat` under pthink30 at 72 (the seeded plan round 2 ran) | 63.3 live, 2.46 turns/s, p90 5.97 s, 432 tok/s, 0 failed, 0 retracted (round 2: 63.3, 2.47, 6.06 s, 432) | `runs/t30-ship-chat-C72-points.json` |

**HiCache start-check failures.** "Not enough host memory available" hit 4 of 14 HiCache starts in the 28G scope:
- once on `ship-inflight` (`sweep-ship-inflight-20261006-185106-*`, passed on the second try);
- three times on cfc12c0bac (`sweep-final-hc-cp2048-lpm-glue-c1-20261006-185819-*`, which failed all three tries).

`sweep_abba.sh` reran that pair (B2 then A2) whole.
