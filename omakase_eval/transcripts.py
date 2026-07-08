"""Per-task evaluation transcripts — the runtime log a miner audits problem-by-problem.

Every scored task records its full call sequence (which worker, what role, the
response, tokens) plus the grading outcome. Transcripts are content-addressed:
the file name is a hash of its bytes, so a run blob referencing a transcript
sha pins exactly what was recorded — tamper-evident like the frontier log.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

from .engine import TaskResult
from .suites import Task, render_prompt


def _task_record(task: Task, result: TaskResult, run_seed: int) -> dict:
    return {
        "task_id": task.id,
        "suite": task.suite,
        "prompt": render_prompt(task, run_seed),
        "correct": result.correct,
        "answer": result.answer,
        "tokens": result.tokens,
        "cost": round(result.cost, 6),
        "latency_ms": round(result.latency_ms, 3),
        "steps": [
            {"worker": s.call.worker, "role": s.call.role, "response": s.response, "tokens": s.tokens}
            for s in result.steps
        ],
    }


def build(tasks: list[Task], results: list[TaskResult], run_seed: int, header: dict) -> dict:
    by_id = {t.id: t for t in tasks}
    records = [_task_record(by_id[r.task_id], r, run_seed) for r in results]
    return {"header": header, "tasks": records}


def summarize(transcript: dict) -> list[dict]:
    """Compact per-task rows for the run blob (drill-down loads the full transcript on demand)."""
    return [
        {"task_id": t["task_id"], "suite": t["suite"], "correct": t["correct"],
         "tokens": t["tokens"], "cost": t["cost"], "n_steps": len(t["steps"])}
        for t in transcript["tasks"]
    ]


def sha256(transcript: dict) -> str:
    blob = json.dumps(transcript, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def write(transcript: dict, directory: str) -> str:
    """Write content-addressed; return the sha256. Idempotent — same bytes, same file."""
    os.makedirs(directory, exist_ok=True)
    digest = sha256(transcript)
    path = os.path.join(directory, f"{digest}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(transcript, f, sort_keys=True, separators=(",", ":"))
    return digest


def read(directory: str, digest: str) -> dict | None:
    path = os.path.join(directory, f"{digest}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
