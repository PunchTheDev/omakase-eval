"""oc-eval CLI — the maintainer's (and miners' self-score) entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import baselines as bl
from . import engine, frontier, mockpool, routers, score, suites
from .workers import Pool


def cmd_mockpool(args: argparse.Namespace) -> int:
    server = mockpool.serve(args.port)
    print(f"mock pool serving on 127.0.0.1:{args.port} (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    pool = Pool.from_config(args.pool)
    result = bl.compute(pool, args.split, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(result.to_json())
    print(f"best single worker: {result.best_single} "
          f"(acc {result.solo_axes[result.best_single]['accuracy']:.3f}); "
          f"oracle {result.oracle_accuracy:.3f} → wrote {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    pool = Pool.from_config(args.pool)
    base = bl.load(args.baselines)
    if (base.split, base.seed) != (args.split, args.seed):
        print("baselines were computed for a different (split, seed)", file=sys.stderr)
        return 2
    router = routers.load_router(args.manifest, os.path.dirname(args.manifest) or ".")
    tasks = suites.generate_split(args.split, args.seed)
    results = engine.run_split(router, tasks, pool, args.seed, args.split)
    verdict = score.judge(results, bl.deserialize_results(base.best_single_results), base.oracle_accuracy)

    blob = {
        "manifest_sha256": routers.sha256_file(args.manifest),
        "split": args.split,
        "seed": args.seed,
        "n_tasks": len(tasks),
        "verdict": verdict.to_dict(),
        "mde": _mde(len(tasks)),
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(blob, f, indent=1)
    if args.frontier:
        frontier.append(args.frontier, "run", blob)

    v = verdict
    print(f"accuracy {v.candidate.accuracy:.3f} vs best-single {v.baseline.accuracy:.3f} "
          f"(Δ {v.comparison.delta:+.3f}, p={v.comparison.p_value:.4f}, "
          f"oracle capture {v.oracle_capture if v.oracle_capture is None else round(v.oracle_capture, 3)})")
    print(f"verdict: {'PASS' if v.passed else 'FAIL'} — {v.reason}")
    return 0 if v.passed else 1


def cmd_verify_log(args: argparse.Namespace) -> int:
    ok, msg = frontier.verify(args.path)
    print(f"{args.path}: {msg}")
    return 0 if ok else 1


def _mde(n: int) -> float:
    from .stats import minimum_detectable_effect

    return round(minimum_detectable_effect(n), 4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oc-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("mockpool", help="serve the deterministic dev worker pool")
    s.add_argument("--port", type=int, default=8100)
    s.set_defaults(fn=cmd_mockpool)

    s = sub.add_parser("baselines", help="compute solo baselines + oracle for a split")
    s.add_argument("--pool", required=True)
    s.add_argument("--split", default="dev")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_baselines)

    s = sub.add_parser("run", help="evaluate a router manifest against baselines")
    s.add_argument("--manifest", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--baselines", required=True)
    s.add_argument("--split", default="dev")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out")
    s.add_argument("--frontier")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("verify-log", help="verify a frontier log's hash chain")
    s.add_argument("path")
    s.set_defaults(fn=cmd_verify_log)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
