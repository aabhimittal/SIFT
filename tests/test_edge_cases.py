"""Edge cases and failure modes. Each test pins a behaviour that was broken or
undefined before it was written (noted where relevant)."""

import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sift.cli import main
from sift.curate import Curation, curate, robust_outliers
from sift.data import UNKNOWN, Dataset, Trajectory, make_synthetic, make_validation
from sift.duplicates import DuplicateResult, find_duplicates, find_scene_duplicates
from sift.env import OBS_DIM, ReachEnv
from sift.export import export_subset, select
from sift.influence import InfluenceResult, rankdata, spearman, tracin
from sift.judge import ClaudeVLMJudge, CounterfactualJudge, JudgeResult
from sift.model import train_bc
from sift.pipeline import RunConfig, json_safe, run
from sift.report import load_results, render, write_report


@pytest.fixture(scope="module")
def tiny():
    return make_synthetic(n_clean=12, n_dup_clusters=1, dup_cluster_size=3, n_mislabeled=3, n_failed=2,
                          n_scene_reuse=2, seed=11)


TINY_RUN = dict(seeds=(0,), steps=150, n_checkpoints=3, n_eval=20, fractions=(0.5, 1.0), ablate_influence=False)


@pytest.fixture(scope="module")
def tiny_results(tiny):
    return run(RunConfig(**TINY_RUN), ds=tiny, log=lambda *a: None)


# --------------------------------------------------------------------------- data

def _traj(t, **kw):
    fields = dict(obs=t.obs, actions=t.actions, pos=t.pos, targets=t.targets, instr=t.instr)
    fields.update(kw)
    return Trajectory(**fields)


@pytest.mark.parametrize("bad, msg", [
    (dict(obs=np.zeros((0, OBS_DIM)), actions=np.zeros((0, 2)), pos=np.zeros((1, 2))), "no steps"),
    (dict(obs=np.zeros((30, 5))), "obs shape"),
    (dict(actions=np.zeros((30, 3))), "actions shape"),
    (dict(pos=np.zeros((30, 2))), "pos shape"),
    (dict(targets=np.zeros((2, 2))), "targets shape"),
    (dict(instr=3), "instruction index"),
    (dict(instr=-1), "instruction index"),
])
def test_trajectory_rejects_malformed_input(tiny, bad, msg):
    with pytest.raises(ValueError, match=msg):
        _traj(tiny[0], **bad)


def test_trajectory_rejects_nan(tiny):
    a = tiny[0].actions.copy()
    a[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _traj(tiny[0], actions=a)


def test_ragged_roundtrip_preserves_contents_and_fingerprint(tmp_path, tiny):
    # Previously: save() used np.stack and crashed on unequal lengths.
    trajs = [copy.deepcopy(t) for t in tiny.trajs]
    short = trajs[0]
    trajs[0] = _traj(short, obs=short.obs[:7], actions=short.actions[:7], pos=short.pos[:8])
    trajs[0].tag = short.tag
    ds = Dataset(trajs, tiny.cfg)
    ds.save(tmp_path / "r.npz")
    back = Dataset.load(tmp_path / "r.npz")
    assert [len(t) for t in back.trajs] == [len(t) for t in ds.trajs]
    assert back.fingerprint() == ds.fingerprint()
    assert np.array_equal(back[0].pos, ds[0].pos) and list(back.tags()) == list(ds.tags())


def test_legacy_stacked_format_still_loads(tmp_path, tiny):
    p = tmp_path / "legacy.npz"
    np.savez(p, obs=np.stack([t.obs for t in tiny.trajs]), actions=np.stack([t.actions for t in tiny.trajs]),
             pos=np.stack([t.pos for t in tiny.trajs]), targets=np.stack([t.targets for t in tiny.trajs]),
             instr=np.array([t.instr for t in tiny.trajs]), executed=np.array([t.executed for t in tiny.trajs]),
             tag=np.array([t.tag for t in tiny.trajs]), true_cluster=np.array([t.true_cluster for t in tiny.trajs]))
    assert Dataset.load(p).fingerprint() == tiny.fingerprint()


def test_load_without_ground_truth_marks_unknown(tmp_path, tiny):
    # Previously: KeyError on any file lacking the synthetic-only fields.
    p = tmp_path / "plain.npz"
    np.savez(p, obs=np.stack([t.obs for t in tiny.trajs]), actions=np.stack([t.actions for t in tiny.trajs]),
             instr=np.array([t.instr for t in tiny.trajs]))
    ds = Dataset.load(p)
    assert len(ds) == len(tiny) and set(ds.tags()) == {UNKNOWN} and not ds.has_ground_truth


def test_load_missing_required_array(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, obs=np.zeros((1, 3, OBS_DIM)))
    with pytest.raises(ValueError, match="actions"):
        Dataset.load(p)


def test_from_arrays_recovers_scene_and_path(tiny):
    ds = Dataset.from_arrays([t.obs for t in tiny.trajs], [t.actions for t in tiny.trajs],
                             [t.instr for t in tiny.trajs], tiny.cfg)
    for a, b in zip(ds.trajs, tiny.trajs):
        assert np.allclose(a.targets, b.targets) and np.allclose(a.pos, b.pos)
    with pytest.raises(ValueError, match="one entry per trajectory"):
        Dataset.from_arrays([tiny[0].obs], [], [0])


def test_empty_dataset_guards(tiny):
    with pytest.raises(ValueError):
        Dataset([], tiny.cfg).save("never.npz")
    with pytest.raises(ValueError, match="no trajectories"):
        tiny.arrays([])
    with pytest.raises(ValueError):
        find_duplicates(Dataset([], tiny.cfg))


# -------------------------------------------------------------------------- model

def test_train_bc_validates_inputs():
    X, Y = np.zeros((4, OBS_DIM)), np.zeros((4, 2))
    with pytest.raises(ValueError):
        train_bc(X[:0], Y[:0])
    with pytest.raises(ValueError):
        train_bc(X, Y[:3])
    with pytest.raises(ValueError):
        train_bc(X, Y, steps=0)


@pytest.mark.parametrize("steps, n", [(5, 10), (7, 3), (100, 10), (1, 1)])
def test_checkpoints_never_silently_vanish(steps, n):
    # Previously: steps < n_checkpoints gave zero checkpoints -> all-zero influence.
    X, Y = np.random.default_rng(0).random((8, OBS_DIM)), np.zeros((8, 2))
    ck = train_bc(X, Y, steps=steps, n_checkpoints=n)[1]
    assert len(ck) == min(n, steps) and ck[-1].step == steps
    assert len({c.step for c in ck}) == len(ck)


# ---------------------------------------------------------------------- influence

def test_rankdata_averages_ties():
    assert rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_spearman_edge_values():
    # Previously: a constant vector got distinct ranks and reported rho ~ 1.
    x = np.arange(6.0)
    assert np.isnan(spearman(np.ones(6), x))
    assert spearman(x, x) == pytest.approx(1.0) and spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.array([1.0]), np.array([2.0])))


def test_single_seed_statistics_are_defined():
    r = InfluenceResult(np.array([[1.0, -2.0, 3.0]]), np.ones((1, 3)), "rollout", True)
    assert np.all(r.std == 0) and np.isnan(r.mean_seed_spearman()) and r.topk_overlap() == 1.0


def test_tracin_guards(tiny):
    env = ReachEnv(tiny.cfg)
    model, ck = train_bc(*tiny.arrays()[:2], steps=60, n_checkpoints=2)
    val = make_validation(n=4, cfg=tiny.cfg)
    with pytest.raises(ValueError, match="checkpoints"):
        tracin(model, [], tiny, val, env)
    with pytest.raises(ValueError, match="validation mode"):
        tracin(model, ck, tiny, val, env, mode="test-set")


def test_projection_preserves_influence_ranking(tiny):
    env = ReachEnv(tiny.cfg)
    model, ck = train_bc(*tiny.arrays()[:2], steps=300, n_checkpoints=3)
    val = make_validation(n=8, cfg=tiny.cfg)
    exact, self_exact = tracin(model, ck, tiny, val, env)
    sk, self_sk = tracin(model, ck, tiny, val, env, project_dim=2048)
    assert spearman(exact, sk) > 0.9 and spearman(self_exact, self_sk) > 0.9
    with pytest.raises(ValueError):
        tracin(model, ck, tiny, val, env, project_dim=0)


# --------------------------------------------------------------------- duplicates

def test_majority_exact_copies_are_still_found(tiny):
    # Previously: median NN distance 0 -> threshold 0 -> 'd < 0' found nothing.
    ds = Dataset([copy.deepcopy(tiny[0]) for _ in range(6)] + tiny.trajs[1:4], tiny.cfg)
    res = find_duplicates(ds)
    assert len(set(res.cluster[:6])) == 1 and res.cluster[0] >= 0
    assert all(c == -1 or c != res.cluster[0] for c in res.cluster[6:])


def test_single_trajectory_has_no_duplicates(tiny):
    assert find_duplicates(tiny.subset([0])).cluster.tolist() == [-1]
    assert find_scene_duplicates(tiny.subset([0])).cluster.tolist() == [-1]
    with pytest.raises(ValueError):
        find_duplicates(tiny, threshold=-1.0)


def test_slowed_down_replay_is_a_duplicate_only_with_dtw(tiny):
    # Same path executed at half speed: different length, same behaviour.
    base = tiny[0]
    idx = np.clip(np.arange(2 * len(base)) // 2, 0, len(base) - 1)
    slow = _traj(base, obs=base.obs[idx], actions=base.actions[idx] / 2, pos=base.pos[np.append(idx, len(base))])
    ds = Dataset([base, slow] + tiny.trajs[5:12], tiny.cfg)
    assert find_duplicates(ds, threshold=0.05).cluster[0] == find_duplicates(ds, threshold=0.05).cluster[1] != -1


# --------------------------------------------------------------------------- judge

def test_counterfactual_judge_config_and_size_guards(tiny):
    for kw in (dict(folds=1), dict(ratio=1.0), dict(temperature=0)):
        with pytest.raises(ValueError):
            CounterfactualJudge(**kw)
    with pytest.raises(ValueError, match="folds"):
        CounterfactualJudge(steps=10).judge(tiny.subset([0, 1]))


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _resp(text=None, stop="end_turn"):
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop, content=content)


@pytest.mark.parametrize("reply", [
    _resp(stop="refusal"),
    _resp(None, stop="max_tokens"),
    _resp('{"consistent": false, "confid'),                                    # truncated JSON
    _resp('{"consistent": false, "confidence": 0.9, "best_instruction": "dance", "rationale": "x"}'),
    _resp('{"consistent": false, "confidence": "high", "best_instruction": "reach the red target", "rationale": "x"}'),
    _resp('{"consistent": false}'),                                            # missing keys
])
def test_vlm_judge_treats_bad_responses_as_no_evidence(tiny, reply):
    t = tiny[0]
    r = ClaudeVLMJudge(client=_FakeClient(reply)).judge_one(t)
    assert r["confidence"] == 0.0 and r["best_instruction"] == t.instruction
    res = ClaudeVLMJudge(client=_FakeClient(reply)).judge(tiny.subset([0]))
    assert not res.mismatch[0]


@pytest.mark.parametrize("conf, other, flagged", [(0.9, True, True), (0.9, False, False), (0.3, True, False)])
def test_vlm_judge_flags_only_confident_disagreement(tiny, conf, other, flagged):
    # Previously: a uniform (no-evidence) row tied, argmax picked index 0, and every
    # demo whose label was not instruction 0 was flagged.
    from sift.env import INSTRUCTIONS
    t = tiny[0]
    named = INSTRUCTIONS[(t.instr + 1) % 3] if other else t.instruction
    reply = _resp(json.dumps({"consistent": not other, "confidence": conf, "best_instruction": named, "rationale": ""}))
    assert ClaudeVLMJudge(client=_FakeClient(reply)).judge(tiny.subset([0])).mismatch[0] == flagged


def test_vlm_judge_lets_api_errors_propagate(tiny):
    with pytest.raises(RuntimeError):
        ClaudeVLMJudge(client=_FakeClient(RuntimeError("401"))).judge_one(tiny[0])


# -------------------------------------------------------------------------- curate

def test_robust_outliers_edge_cases():
    # Previously: MAD = 0 flagged anything a hair above the median.
    assert not robust_outliers(np.ones(10), 4).any()
    assert not robust_outliers(np.array([1.0, 2.0]), 4).any()
    x = np.array([1, 1.1, 0.9, 1, 1.05, 0.95, 50, np.nan])
    assert robust_outliers(x, 4).tolist() == [False] * 6 + [True, False]
    mostly_same = np.r_[np.ones(20), 9.0]
    assert robust_outliers(mostly_same, 4).tolist() == [False] * 20 + [True]


def _fake_inputs(mean, std, cluster, mismatch, oof=None):
    n = len(mean)
    per = np.stack([np.asarray(mean) - std, np.asarray(mean) + std])
    infl = InfluenceResult(per, np.ones((2, n)), "rollout", True)
    dups = DuplicateResult(np.asarray(cluster), [], 0.0, np.zeros(n))
    oof = np.ones(n) if oof is None else np.asarray(oof, float)
    judge = JudgeResult(np.ones(n), np.ones(n), np.zeros(n, int), np.ones((n, 3)) / 3, oof,
                        np.asarray(mismatch, bool), [""] * n)
    return infl, dups, judge


def test_curate_tiers_and_representatives():
    std = np.full(6, 0.5)
    # 0..2 one cluster; 2 is mislabeled (must not be the kept twin); 5 is confidently harmful
    infl, dups, judge = _fake_inputs([1.0, 3.0, 9.0, 0.2, 0.1, -5.0], std / np.sqrt(2) * np.sqrt(2),
                                     [0, 0, 0, -1, -1, -1], [0, 0, 1, 0, 0, 0])
    cur = curate(infl, dups, judge)
    assert cur.verdict.tolist() == ["redundant", "keep", "mismatch", "keep", "keep", "harmful"]
    assert cur.representative[0] == 1
    assert cur.order[-1] == 2 and cur.order[0] == 1


def test_cluster_with_one_eligible_member_has_no_redundant():
    infl, dups, judge = _fake_inputs([1.0, 2.0], np.zeros(2), [0, 0], [0, 1])
    assert curate(infl, dups, judge).verdict.tolist() == ["keep", "mismatch"]


def test_curate_rejects_mismatched_inputs():
    infl, dups, judge = _fake_inputs([1.0, 2.0], np.zeros(2), [0, 0], [0, 0])
    dups = DuplicateResult(np.array([0, 0, -1]), [], 0.0, np.zeros(3))
    with pytest.raises(ValueError, match="different numbers"):
        curate(infl, dups, judge)


@pytest.mark.parametrize("frac", [0, -0.1, 1.5])
def test_subset_fraction_bounds(frac):
    cur = Curation(np.array(["keep"] * 4), np.arange(4), np.zeros(4), np.full(4, -1))
    with pytest.raises(ValueError):
        cur.subset(frac)
    assert len(cur.subset(0.01)) == 1 and len(cur.subset(1.0)) == 4


# ------------------------------------------------------------------------ pipeline

@pytest.mark.parametrize("kw", [dict(seeds=()), dict(seeds=(0, 0)), dict(fractions=(0.0,)), dict(fractions=(1.2,)),
                                dict(steps=0), dict(val_mode="test"), dict(project_dim=0)])
def test_run_config_validation(kw):
    with pytest.raises(ValueError):
        RunConfig(**kw)


def test_fractions_are_sorted_and_deduplicated():
    assert RunConfig(fractions=(0.5, 0.1, 0.5)).fractions == (0.1, 0.5)


def test_json_safe():
    out = json_safe({"a": float("nan"), "b": [np.float32(np.inf), np.int64(3), np.bool_(True)], "c": (1.5,)})
    assert out == {"a": None, "b": [None, 3, True], "c": [1.5]}
    json.dumps(out, allow_nan=False)


def test_single_seed_run_produces_strict_json(tiny_results):
    # Previously: one seed -> NaN in results -> report page failed JSON.parse.
    text = json.dumps(tiny_results, allow_nan=False)
    assert tiny_results["meta"]["ground_truth"] is True and tiny_results["stability"]["spearman"] == [[1.0]]
    assert len(json.loads(text)["trajectories"]) == tiny_results["meta"]["n"]


def test_run_without_ground_truth(tiny):
    ds = Dataset.from_arrays([t.obs for t in tiny.trajs], [t.actions for t in tiny.trajs],
                             [t.instr for t in tiny.trajs], tiny.cfg)
    res = run(RunConfig(**TINY_RUN), ds=ds, log=lambda *a: None)
    assert res["meta"]["ground_truth"] is False and res["detection"] == {} and res["scaling"]["oracle"] is None
    json.dumps(res, allow_nan=False)


def test_run_rejects_tiny_dataset(tiny):
    with pytest.raises(ValueError, match="at least"):
        run(RunConfig(**TINY_RUN), ds=tiny.subset([0, 1]), log=lambda *a: None)


# -------------------------------------------------------------- report and export

def test_report_outputs(tmp_path, tiny_results):
    page = write_report(tiny_results, tmp_path)
    rows = list(csv.DictReader((tmp_path / "ranking.csv").open()))
    assert len(rows) == tiny_results["meta"]["n"] and [int(r["rank"]) for r in rows] == list(range(1, len(rows) + 1))
    assert load_results(tmp_path / "results.json")["meta"]["n"] == len(rows)
    assert "/*__SIFT_DATA__*/" not in page.read_text()


def test_render_never_emits_nan(tiny_results):
    bad = copy.deepcopy(tiny_results)
    bad["trajectories"][0]["oof_loss"] = float("nan")
    html = render(bad)
    blob = html.split('id="sift-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["trajectories"][0]["oof_loss"] is None


def test_load_results_rejects_foreign_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"hello": 1}')
    with pytest.raises(ValueError, match="not a SIFT results file"):
        load_results(p)


def test_select_rules(tiny_results):
    n = tiny_results["meta"]["n"]
    assert len(select(tiny_results, frac=0.5)) == round(0.5 * n)
    kept = select(tiny_results, verdicts=["keep"])
    assert all(t["verdict"] == "keep" for t in tiny_results["trajectories"] if t["id"] in set(kept))
    for kw in (dict(), dict(frac=0.5, verdicts=["keep"]), dict(frac=0.0), dict(verdicts=["great"])):
        with pytest.raises(ValueError):
            select(tiny_results, **kw)


def test_export_refuses_a_different_dataset_of_the_same_size(tiny, tiny_results):
    # Previously only the length was checked.
    other = make_synthetic(n_clean=12, n_dup_clusters=1, dup_cluster_size=3, n_mislabeled=3, n_failed=2,
                           n_scene_reuse=2, seed=12)
    assert len(other) == len(tiny)
    with pytest.raises(ValueError, match="fingerprint"):
        export_subset(other, tiny_results, frac=0.5)
    with pytest.raises(ValueError, match="trajectories"):
        export_subset(tiny.subset(range(5)), tiny_results, frac=0.5)
    assert len(export_subset(tiny, tiny_results, frac=0.5)) == round(0.5 * len(tiny))


# ------------------------------------------------------------------------------ CLI

def test_cli_generate_export_report(tmp_path, tiny, tiny_results, capsys):
    data, res = tmp_path / "d.npz", tmp_path / "out"
    tiny.save(data)
    write_report(tiny_results, res)
    main(["export", "--data", str(data), "--results", str(res / "results.json"), "--verdicts", "keep,redundant",
          "--out", str(tmp_path / "c.npz")])
    assert len(Dataset.load(tmp_path / "c.npz")) == sum(
        t["verdict"] in ("keep", "redundant") for t in tiny_results["trajectories"])
    main(["report", "--results", str(res / "results.json"), "--out", str(tmp_path / "again")])
    assert (tmp_path / "again" / "index.html").exists()
    main(["generate", "--out", str(tmp_path / "g.npz"), "--data-seed", "3"])
    assert len(Dataset.load(tmp_path / "g.npz")) > 0


def test_cli_reports_errors_cleanly(tmp_path, tiny, tiny_results, capsys):
    other = tmp_path / "other.npz"
    make_synthetic(seed=99).save(other)
    write_report(tiny_results, tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["export", "--data", str(other), "--results", str(tmp_path / "results.json"), "--frac", "0.5"])
    assert e.value.code == 2 and "error" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["demo", "--seeds", "0"], ["export", "--data", "a", "--results", "b", "--frac", "0"],
                                  ["export", "--data", "a", "--results", "b"], ["demo", "--steps", "-5"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
