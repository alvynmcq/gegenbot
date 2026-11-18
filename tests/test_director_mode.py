"""AI Director authority modes and how replay measures each part of its decision."""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agent.director import AIDirector, director_mode, limit_to_vetoes
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import FPLOptimizer
from src.replay import capture_snapshot, director_verdict, score_replay

SQUAD = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]


def test_mode_is_explicit_or_follows_the_replay_verdict(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    monkeypatch.setenv("DIRECTOR_MODE", "veto_only")
    assert director_mode(report) == "veto_only"
    monkeypatch.setenv("DIRECTOR_MODE", "auto")
    assert director_mode(report) == "full"  # no evidence yet
    report.write_text("not json")
    assert director_mode(report) == "full"
    report.write_text(json.dumps({"director_verdict": "no_clear_difference"}))
    assert director_mode(report) == "full"
    report.write_text(json.dumps({"director_verdict": "costs_points"}))
    assert director_mode(report) == "veto_only"


def test_verdict_needs_eight_weeks_and_an_interval_clear_of_zero():
    base = {"gameweeks": 8, "bootstrap_95": [-6.0, -0.5]}
    assert director_verdict(base) == "costs_points"
    assert director_verdict(dict(base, bootstrap_95=[0.5, 4.0])) == "adds_points"
    assert director_verdict(dict(base, bootstrap_95=[-2.0, 3.0])) == "no_clear_difference"
    assert director_verdict(dict(base, gameweeks=7)) == "collecting"


@pytest.fixture
def options(mock_bootstrap_data, mock_fixtures_data):
    players = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1, odds_data={})
    result = FPLOptimizer(players).optimize(SQUAD, bank_m=1.0, free_transfers=1, evaluate_chips=False,
                                            include_chip_candidates=False)
    return players, result


def _llm_decision(result, index, veto=()):
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": json.dumps({
        "selected_candidate_index": index, "veto_player_ids": list(veto),
        "veto_reason": "late fitness test", "rationale": "Two sentences. Here."})}}]}
    with patch("requests.post", return_value=response):
        return AIDirector(api_key="key").evaluate_and_decide(result)


def test_veto_only_keeps_the_solver_pick_and_the_directors_vetoes(options):
    _, result = options
    llm = _llm_decision(result, 1, veto=[7])
    solver = AIDirector(api_key="")._build_fallback_decision(result)
    final = limit_to_vetoes(llm, solver)
    assert final.selected_candidate_index == solver.selected_candidate_index
    assert final.source == "SOLVER_DIRECTOR_VETOES"
    assert 7 in final.veto_player_ids and final.veto_reason
    assert llm.chosen_move_name in final.rationale


def test_snapshot_records_the_directors_pick_its_option_and_the_pre_veto_pick(tmp_path, options):
    players, result = options
    llm = _llm_decision(result, 1)
    solver = AIDirector(api_key="")._build_fallback_decision(result)
    final = limit_to_vetoes(llm, solver)
    bootstrap = {"elements": [{"id": int(pid), "ep_next": "2.0"} for pid in players["id"]]}
    path = capture_snapshot(tmp_path, season="2026-2027", gameweek=11, deadline="2099-01-01T00:00:00Z",
                            players=players, bootstrap=bootstrap, optimization=result, decision=final,
                            model="m", director=llm, pre_veto=result, director_mode="veto_only",
                            captured_at=datetime.now(timezone.utc))
    snapshot = json.loads(path.read_text())
    assert {"ai_director", "ai_candidate", "pre_veto_optimizer", "optimizer_only"} <= snapshot["policies"].keys()
    assert snapshot["director_source"] == "SOLVER_DIRECTOR_VETOES"
    assert snapshot["director_mode"] == "veto_only" and snapshot["director_model"] == "m"


def _pick(captain, vice, bench_first=12):
    starters = [[1, 1], [3, 2], [4, 2], [5, 2], [8, 3], [9, 3], [10, 3], [11, 3], [13, 4], [14, 4], [15, 4]]
    return {"starters": starters, "bench": [[2, 1], [bench_first, 3], [6, 2], [7, 2]],
            "captain": captain, "vice": vice, "chip": None, "hit": 0}


def test_breakdown_splits_option_choice_overrides_and_vetoes():
    actual = {pid: {"minutes": 90, "points": 2} for pid in range(1, 16)}
    actual[9]["points"], actual[10]["points"] = 10, 6
    snapshot = {
        "version": 1, "season": "2026-2027", "gameweek": 9, "captured_at": "2026-10-31T10:00:00Z",
        "deadline": "2026-10-31T11:00:00Z", "director_source": "SOLVER_DIRECTOR_VETOES",
        "forecasts": {}, "policies": {
            "optimizer_only": _pick(8, 9),       # 30 + 2 = 32
            "ai_candidate": _pick(10, 9),        # option chosen: captain 10 -> 36
            "ai_director": _pick(9, 10),         # captain override to 9 -> 40
            "pre_veto_optimizer": _pick(11, 9),  # before news vetoes -> 32
        }}
    outcome = {"version": 1, "season": "2026-2027", "gameweek": 9, "finalized": True,
               "deadline": "2026-10-31T11:00:00Z", "collected_at": "2026-11-04T00:00:00Z",
               "players": {str(pid): row for pid, row in actual.items()}}
    report = score_replay([snapshot], {("2026-2027", 9): outcome})
    breakdown = report["director_breakdown"]
    assert report["ai_director_vs_optimizer"]["total_delta"] == 8
    assert breakdown["option_choice"]["total_delta"] == 4
    assert breakdown["lineup_overrides"]["total_delta"] == 4
    assert breakdown["news_vetoes"]["total_delta"] == 0
    assert breakdown["same_pick_rate"] == 0
    assert report["director_verdict"] == "collecting"
