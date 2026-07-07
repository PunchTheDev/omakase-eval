import json

from oc_eval import frontier


def test_chain_appends_and_verifies(tmp_path):
    log = tmp_path / "frontier.jsonl"
    frontier.append(str(log), "run", {"a": 1}, ts=1.0)
    frontier.append(str(log), "merge", {"b": 2}, ts=2.0)
    ok, msg = frontier.verify(str(log))
    assert ok, msg
    entries = frontier.read(str(log))
    assert entries[1]["prev"] == entries[0]["sha"]


def test_tampering_detected(tmp_path):
    log = tmp_path / "frontier.jsonl"
    frontier.append(str(log), "run", {"score": 0.5}, ts=1.0)
    frontier.append(str(log), "run", {"score": 0.6}, ts=2.0)
    lines = log.read_text().splitlines()
    doctored = json.loads(lines[0])
    doctored["payload"]["score"] = 0.99
    log.write_text(json.dumps(doctored, sort_keys=True, separators=(",", ":")) + "\n" + lines[1] + "\n")
    ok, msg = frontier.verify(str(log))
    assert not ok and "seq 0" in msg
