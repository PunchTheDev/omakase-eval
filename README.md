# oc-eval

Evaluation infrastructure for the OC orchestration competitions ([OC-R](../oc-router), [OC-H](../oc-harness)).
Zero runtime dependencies — everything below is stdlib Python, which keeps the
digest-pinned runtime image minimal and auditable.

## What lives here

| Module | Purpose |
|---|---|
| `actions` | The router↔harness contract: `Call(worker, role)` or `Answer(final)` |
| `engine` | Locked reference harness: bounded turn loop, budget enforcement |
| `suites` | Procedural task suites — every split reproducible from `(split, seed)` |
| `workers` | OpenAI-compatible pool client; pool configs pin every worker |
| `mockpool` | Deterministic dev pool with complementary skill profiles |
| `routers` | TinyRouter (~10K-param linear policy), manifest loader with sha/size guards |
| `stats` | Paired McNemar + bootstrap + published MDE |
| `score` | Composite axes, uplift vs. best-single, oracle capture, verdicts |
| `baselines` | Solo runs per worker, best-single bar, oracle ceiling |
| `frontier` | Append-only hash-chained results ledger |
| `showcase` | Champion stack vs. contenders — the /vs-labs data |

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # or just `pip install -e .`
.venv/bin/oc-eval mockpool --port 8100 &                      # dev worker pool
.venv/bin/oc-eval baselines --pool configs/pool.dev.json --out runs/baselines.dev.json
.venv/bin/oc-eval run --manifest ../oc-router/submission/manifest.json \
    --pool configs/pool.dev.json --baselines runs/baselines.dev.json \
    --frontier runs/frontier.jsonl
```

`run` exits 0 on a PASS verdict (significant accuracy gain vs. best single
worker, cost/latency within tolerance) and 1 otherwise — the same judgment the
maintainer's canonical rerun applies.

## Tests

```bash
.venv/bin/pytest -q
```

The e2e test trains a TinyRouter against the mock pool and asserts it beats the
best single worker with significance — the competition loop in miniature.
