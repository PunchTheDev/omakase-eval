"""Deterministic mock worker pool for development and CI.

Each mock worker has a per-suite skill profile chosen so strengths are
complementary — the pool property routing needs (tinyrouter finding #1).
Correctness is a pure function of (worker, task): a worker either knows a task
or it doesn't, stable across runs, so paired statistics behave exactly as they
do against a real pinned pool at temperature 0.

The request must carry {"metadata": {"split","seed","task_id"}} (the engine
sends it); the server regenerates the task and answers correctly iff
hash(worker, task) falls under the worker's skill for that suite.
"""
from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import suites

SKILLS = {  # suite -> skill, deliberately complementary
    "reasoner-mock": {"reasoning": 0.92, "math": 0.74, "code_qa": 0.70},
    "coder-mock": {"reasoning": 0.70, "math": 0.66, "code_qa": 0.93},
    "math-mock": {"reasoning": 0.72, "math": 0.90, "code_qa": 0.68},
    "generalist-mock": {"reasoning": 0.80, "math": 0.76, "code_qa": 0.76},
    "small-mock": {"reasoning": 0.55, "math": 0.55, "code_qa": 0.55},
    # closed-lab stand-ins, showcase only — never in a competition pool config
    "lab-alpha-mock": {"reasoning": 0.88, "math": 0.86, "code_qa": 0.87},
    "lab-beta-mock": {"reasoning": 0.84, "math": 0.83, "code_qa": 0.86},
}


def knows(worker: str, task: suites.Task) -> bool:
    h = hashlib.sha256(f"{worker}|{task.id}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 < SKILLS[worker][task.suite]


def _wrong_answer(worker: str, task: suites.Task) -> str:
    h = hashlib.sha256(f"wrong|{worker}|{task.id}".encode()).digest()
    if task.options:
        distractors = [o for o in task.options if o != task.answer]
        return distractors[h[0] % len(distractors)]
    return str(int(task.answer) + 1 + h[1] % 7)


def respond(model: str, body: dict) -> str:
    meta = body.get("metadata") or {}
    tasks = {t.id: t for t in suites.generate_split(meta["split"], int(meta["seed"]))}
    task = tasks[meta["task_id"]]
    answer = task.answer if knows(model, task) else _wrong_answer(model, task)
    role = "verifier" if re.search(r"verif", body["messages"][0]["content"], re.I) else "worker"
    if role == "verifier":  # verifiers judge the draft in the user message against their own belief
        draft = body["messages"][1]["content"].strip().splitlines()[-1]
        return "CORRECT" if (task.answer in draft) == knows(model, task) else "REVISE"
    return answer


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            text = respond(body["model"], body)
            payload = {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"total_tokens": 40 + len(text.split())},
            }
            self.send_response(200)
        except Exception as e:  # noqa: BLE001 — surface the error to the client
            payload = {"error": str(e)}
            self.send_response(500)
        data = json.dumps(payload).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_: object) -> None:  # keep test output quiet
        pass


def serve(port: int = 8100) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    return server
