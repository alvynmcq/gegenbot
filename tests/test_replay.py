"""Replay scoring and time boundaries use synthetic or explicitly frozen inputs."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from src.replay import fetch_outcomes, score_pick, score_replay


@pytest.fixture
def selection():
    return {
        "starters": [(1, 1), (3, 2), (4, 2), (5, 2), (8, 3), (9, 3), (10, 3), (11, 3),
                     (13, 4), (14, 4), (15, 4)],
        "bench": [(2, 1), (12, 3), (6, 2), (7, 2)],
        "captain": 8, "vice": 9, "chip": None, "hit": 4,
    }


@pytest.fixture
def actuals():
    return {pid: {"minutes": 90, "points": 2} for pid in range(1, 16)}


def test_scoring_enforces_formation_and_bench_order(selection, actuals):
    actuals[3] = {"minutes": 0, "points": 0}
    actuals[12]["points"] = 9
    actuals[6]["points"] = 4
    actuals[7]["points"] = 7
    assert score_pick(selection, actuals) == 22
    # A missing midfielder lets the first bench player enter alongside the defender.
    actuals[8] = {"minutes": 0, "points": 0}
    actuals[1] = {"minutes": 0, "points": 0}
    actuals[2]["points"] = 6
    assert score_pick(selection, actuals) == 33
    assert score_pick(dict(selection, chip="3xc"), actuals) == 35
    assert score_pick(dict(selection, chip="bboost"), actuals) == 40


def test_double_gameweek_totals_and_negative_captain_points(selection, actuals):
    actuals[8] = {"minutes": 180, "points": -1}
    actuals[9]["points"] = 10
    assert score_pick(selection, actuals) == 22
    assert score_pick(dict(selection, chip="3xc"), actuals) == 21
    del actuals[12]
    assert score_pick(selection, actuals) is None


@pytest.fixture
def snapshot(selection):
    return {
        "version": 1, "season": "2026-2027", "gameweek": 9, "captured_at": "2026-10-31T10:00:00Z",
        "deadline": "2026-10-31T11:00:00Z", "provenance": "gegenbot", "director_source": "LLM_DIRECTOR",
        "director_model": "example", "forecasts": {
            "8": {"gegenbot": 4, "official_ep_next": 3, "season_ppg": 2},
            "9": {"gegenbot": 5, "official_ep_next": None, "season_ppg": 4},
        },
        "policies": {"optimizer_only": selection, "roll": selection,
                     "ai_director": dict(selection, captain=9, vice=8)},
    }


@pytest.fixture
def outcome(actuals):
    actuals[9]["points"] = 10
    return {"version": 1, "season": "2026-2027", "gameweek": 9, "finalized": True,
            "deadline": "2026-10-31T11:00:00Z", "collected_at": "2026-11-04T00:00:00Z",
            "players": {str(pid): row for pid, row in actuals.items()}}


def test_paired_effect_excludes_fallback_and_matches_forecast_samples(snapshot, outcome):
    outcomes = {("2026-2027", 9): outcome}
    report = score_replay([snapshot], outcomes)
    assert report["ai_director_vs_optimizer"]["mean_delta"] == 8
    assert report["director_models"]["example"]["wins"] == 1
    assert report["forecast_metrics_common_players"]["gegenbot"]["mae"] == 2
    assert report["forecast_metrics_common_players"]["season_ppg"]["mae"] == 0
    assert report["forecast_coverage"]["gegenbot"] == 2
    assert report["forecast_coverage"]["official_ep_next"] == 1
    snapshot["director_source"] = "MATHEMATICAL_FALLBACK"
    assert score_replay([snapshot], outcomes)["ai_director_vs_optimizer"]["gameweeks"] == 0


def test_replay_rejects_duplicate_and_late_snapshots_and_unfinalized_outcomes(snapshot, outcome):
    key = ("2026-2027", 9)
    with pytest.raises(ValueError, match="Duplicate"):
        score_replay([snapshot, deepcopy(snapshot)], {key: outcome})
    with pytest.raises(ValueError, match="pre-deadline"):
        score_replay([dict(snapshot, captured_at=snapshot["deadline"])], {})
    with pytest.raises(ValueError, match="finalized"):
        score_replay([snapshot], {key: dict(outcome, finalized=False)})
    with pytest.raises(ValueError, match="finalized"):
        score_replay([snapshot], {key: dict(outcome, deadline="2026-10-31T09:00:00Z")})


def test_outcome_fetch_is_current_season_public_and_finalized_only(tmp_path, snapshot):
    bootstrap = {"events": [
        {"id": 1, "deadline_time": "2026-08-20T10:00:00Z"},
        {"id": 9, "deadline_time": snapshot["deadline"], "finished": True, "data_checked": True},
        {"id": 10, "deadline_time": "2026-11-06T18:30:00Z", "finished": True, "data_checked": False},
    ]}
    unfinished = dict(snapshot, gameweek=10, deadline="2026-11-06T18:30:00Z")
    previous = dict(snapshot, season="2025-2026")
    with patch("src.replay._public_json", side_effect=[
        bootstrap, {"elements": [{"id": 1, "stats": {"minutes": 90, "total_points": 5}}]},
    ]) as public:
        assert fetch_outcomes([snapshot, unfinished, previous], tmp_path) == 1
    assert [call.args[0] for call in public.call_args_list] == ["bootstrap-static", "event/9/live"]
    assert (tmp_path / "2026-2027/gw09.json").exists()
