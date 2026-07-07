"""Deterministic mock worker pool for development and CI.

Each mock worker has a per-suite skill profile chosen so strengths are
complementary — the pool property routing needs (tinyrouter finding #1).
Correctness is a pure function of (worker, task): a worker either knows a task
or it doesn't, stable across runs, so paired statistics behave exactly as they
do against a real pinned pool at temperature 0.

The request must carry {"metadata": {"split","seed","task_id"}} (the engine
sends it); the server regenerates the task and answers correctly iff
hash(worker, task) falls under the worker's skill for that suite.

TRUST BOUNDARY: this mock *encodes ground truth*, so it is necessarily an
answer oracle — a hostile harness that probes it (crafted verifier drafts,
candidate sweeps) can recover answers. That is acceptable only because the mock
is a trusted maintainer-run dev/CI component. Production replaces it with a real
pinned LLM pool that has no answer key, reached over an attested egress
allow-list; the OC-H v2 contract (redacted views, central grading/metering) and
the Gate-3 answer-reconstruction ban are the defenses that carry to production.
Never point a competition's *scoring* pool at this server.
"""
from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import suites

SKILLS = {  # suite -> skill, deliberately complementary
    "reasoner-mock": {"reasoning": 0.86, "math": 0.62, "code_qa": 0.60},
    "coder-mock": {"reasoning": 0.62, "math": 0.58, "code_qa": 0.88},
    "math-mock": {"reasoning": 0.62, "math": 0.86, "code_qa": 0.60},
    "generalist-mock": {"reasoning": 0.72, "math": 0.70, "code_qa": 0.70},
    "small-mock": {"reasoning": 0.48, "math": 0.48, "code_qa": 0.48},
    # closed-lab stand-ins, showcase only — never in a competition pool config
    "lab-alpha-mock": {"reasoning": 0.82, "math": 0.82, "code_qa": 0.82},
    "lab-beta-mock": {"reasoning": 0.78, "math": 0.78, "code_qa": 0.78},
}

HEDGE_RATE = 0.5  # an ignorant worker hedges half the time — the confidence
# channel harnesses can exploit, standing in for logprobs/self-consistency


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
    if model not in SKILLS:
        raise ValueError(f"unknown worker {model!r}")
    for key in ("split", "seed", "task_id"):
        if key not in meta:
            raise ValueError(f"metadata missing {key!r}")
    task = suites.task_by_id(meta["split"], int(meta["seed"]), meta["task_id"])
    if knows(model, task):
        answer = task.answer
    else:
        answer = _wrong_answer(model, task)
        h = hashlib.sha256(f"hedge|{model}|{task.id}".encode()).digest()
        if int.from_bytes(h[:4], "big") / 2**32 < HEDGE_RATE:
            answer = f"possibly: {answer}"
    role = "verifier" if re.search(r"verif", body["messages"][0]["content"], re.I) else "worker"
    if role == "verifier":  # verifiers judge the draft in the user message against their own belief
        lines = body["messages"][1]["content"].strip().splitlines()
        draft = lines[-1] if lines else ""
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
