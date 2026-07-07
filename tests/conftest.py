import threading

import pytest

from oc_eval import mockpool
from oc_eval.workers import Pool, Worker

WORKER_IDS = list(mockpool.SKILLS)


@pytest.fixture(scope="session")
def pool():
    """A live mock pool on an ephemeral port, shared across the test session."""
    server = mockpool.serve(port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workers = [Worker(w, w, f"http://127.0.0.1:{port}", cost_per_1k=0.1 if w == "small-mock" else 1.0)
               for w in WORKER_IDS]
    yield Pool(workers, timeout_s=10)
    server.shutdown()
