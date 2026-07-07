import json

import pytest

from oc_eval import routers


def _write_submission(tmp_path, weights_file="weights.json", arch="tiny-linear", corrupt=False):
    r = routers.TinyRouter(["a", "b"], [[0.1] * routers.DIM, [0.2] * routers.DIM])
    wpath = tmp_path / "weights.json"
    r.save(str(wpath))
    sha = routers.sha256_file(str(wpath))
    if corrupt:
        sha = "0" * 64
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arch": arch, "weights_file": weights_file, "weights_sha256": sha})
    )
    return tmp_path


def test_load_router_happy(tmp_path):
    sub = _write_submission(tmp_path)
    assert routers.load_router(str(sub / "manifest.json"), str(sub)).workers == ["a", "b"]


def test_load_router_rejects_sha_mismatch(tmp_path):
    sub = _write_submission(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="mismatch"):
        routers.load_router(str(sub / "manifest.json"), str(sub))


def test_load_router_rejects_path_traversal(tmp_path):
    sub = _write_submission(tmp_path, weights_file="../secret.json")
    with pytest.raises(ValueError, match="bare filename"):
        routers.load_router(str(sub / "manifest.json"), str(sub))


def test_load_router_rejects_unknown_arch(tmp_path):
    sub = _write_submission(tmp_path, arch="giant-transformer")
    with pytest.raises(ValueError, match="arch"):
        routers.load_router(str(sub / "manifest.json"), str(sub))
