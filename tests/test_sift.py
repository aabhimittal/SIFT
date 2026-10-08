import json
from types import SimpleNamespace

import numpy as np
import pytest

from sift.curate import VERDICTS, curate
from sift.data import Dataset, make_synthetic, make_validation
from sift.duplicates import dtw, find_duplicates
from sift.env import ReachEnv
from sift.evaluate import auroc, duplicate_pair_prf
from sift.influence import expert_relabel, run_influence
from sift.judge import ClaudeVLMJudge, CounterfactualJudge, render_png
from sift.model import MLPPolicy
from sift.report import render


@pytest.fixture(scope="module")
def small():
    return make_synthetic(n_clean=50, n_dup_clusters=3, dup_cluster_size=4, n_mislabeled=10, n_failed=8,
                          n_scene_reuse=6, seed=3)


@pytest.fixture(scope="module")
def full():
    return make_synthetic()


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLPPolicy(seed=1)
    X, Y = rng.normal(size=(7, 11)), rng.normal(size=(7, 2))
    _, g = m.loss_grad(X, Y)
    for k in rng.choice(m.n_params, 25, replace=False):
        e = np.zeros(m.n_params)
        e[k] = 1e-6
        fd = (m.loss(X, Y, m.theta + e) - m.loss(X, Y, m.theta - e)) / 2e-6
        assert abs(fd - g[k]) < 1e-6 + 1e-4 * abs(fd)


def test_expert_relabel_points_at_instructed_target(small):
    t = small[0]
    a = expert_relabel(t.obs[:1], small.cfg.v_max)[0]
    goal = t.targets[t.instr] - t.pos[0]
    assert np.dot(a, goal) > 0.99 * np.linalg.norm(a) * np.linalg.norm(goal)


def test_dtw_tolerates_time_warp():
    s = np.linspace(0, 1, 30)[:, None] * np.array([[1.0, 0.5]])
    slow = np.linspace(0, 1, 40)[:, None] * np.array([[1.0, 0.5]])
    other = np.linspace(0, 1, 30)[:, None] * np.array([[-1.0, 0.2]])
    assert dtw(s, s) == 0
    assert dtw(s, slow) < 0.02 < dtw(s, other)


def test_duplicates_recover_injected_clusters(small):
    res = find_duplicates(small)
    m = duplicate_pair_prf(small, res)
    assert m["recall"] == 1.0 and m["precision"] >= 0.9


def test_counterfactual_judge_finds_mislabels(full):
    r = CounterfactualJudge().judge(full)
    mm = full.tags() == "mislabeled"
    assert (r.mismatch & mm).sum() >= 0.85 * mm.sum()
    assert (r.mismatch & ~mm).sum() <= 0.15 * r.mismatch.sum()
    assert np.allclose(r.probs.sum(1), 1)


def test_influence_separates_bad_demos_and_curation_orders_tiers(full):
    env = ReachEnv(full.cfg)
    infl = run_influence(full, make_validation(cfg=full.cfg), env, seeds=(0, 1), log=lambda *a: None)
    tags = full.tags()
    assert auroc(-infl.mean, np.isin(tags, ["mislabeled", "failed"])) > 0.65
    cur = curate(infl, find_duplicates(full), CounterfactualJudge().judge(full))
    tiers = [VERDICTS.index(v) for v in cur.verdict[cur.order]]
    assert tiers == sorted(tiers)
    assert len(cur.subset(0.3)) == round(0.3 * len(full))


def test_dataset_roundtrip(tmp_path, small):
    p = tmp_path / "d.npz"
    small.save(p)
    back = Dataset.load(p)
    assert len(back) == len(small)
    assert np.array_equal(back[5].actions, small[5].actions) and back[5].tag == small[5].tag


def test_png_is_valid(small):
    png = render_png(small[0], size=64)
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png.endswith(b"IEND\xaeB`\x82")


def test_vlm_judge_parses_structured_output(small):
    t = small[0]
    payload = {"consistent": False, "confidence": 0.9, "best_instruction": "reach the blue target",
               "rationale": "path ends on the blue disk"}

    class FakeClient:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))
            self.calls = []

        def create(self, **kw):
            self.calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    client = FakeClient()
    judge = ClaudeVLMJudge(client=client)
    req = judge.build_request(t)
    assert req["messages"][0]["content"][0]["type"] == "image"
    assert req["output_config"]["format"]["type"] == "json_schema"
    res = judge.judge(small.subset([0, 1]))
    assert res.predicted[0] == 2 and client.calls


def test_report_embeds_results():
    fake = {"meta": {}, "trajectories": [], "note": "</script>"}
    html = render(fake)
    assert html.startswith("<!doctype html>") and "<\\/script>" in html and "/*__SIFT_DATA__*/" not in html
