"""Regression tests for feed provenance and fixture-aware forecasts."""
import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetcher import FPLCoreInsightsFetcher, forecast_season, fresh_projection_rows, verified_match_logs
from src.engine.metrics import calculate_player_metrics, verified_fixture_schedule
from src.engine.optimizer import FPLOptimizer


def test_forecast_snapshots_require_current_season_target_and_issue_time():
    now = dt.datetime(2026, 11, 6, 12, tzinfo=dt.timezone.utc)
    rows = pd.DataFrame([
        ("2025-2026", 11, "2026-11-06T10:00:00Z", 99),
        ("2026-2027", 10, "2026-11-06T10:00:00Z", 99),
        ("2026-2027", 11, "2026-11-03T10:00:00Z", 99),
        ("2026-2027", 11, "2026-11-07T10:00:00Z", 99),
        ("2026-2027", 11, "2026-11-05T10:00:00Z", 4),
        ("2026-2027", 11, "2026-11-06T10:00:00Z", 6),
    ], columns=["season", "target_gameweek", "as_of", "xp"])
    assert fresh_projection_rows(rows, "2026-2027", 11, now)["xp"].tolist() == [6]
    assert fresh_projection_rows(rows.drop(columns="as_of"), "2026-2027", 11, now) is None
    assert forecast_season([
        {"id": 22, "deadline_time": "2027-01-16T15:00:00Z"},
        {"id": 1, "deadline_time": "2026-08-22T12:00:00Z"},
    ]) == "2026-2027"


def test_stale_local_csv_cannot_shadow_fresh_remote_snapshot(tmp_path):
    stale = tmp_path / "projections.csv"
    stale.write_text("id,xp,season,target_gameweek,as_of\n1,99,2025-2026,11,2025-11-06T10:00:00Z\n")
    issued = dt.datetime.now(dt.timezone.utc).isoformat()
    response = MagicMock(status_code=200, text=(
        f"id,xp,season,target_gameweek,as_of\n1,6,2026-2027,11,{issued}\n"
    ))
    fetcher = FPLCoreInsightsFetcher(file_path=stale)
    with patch("src.data_fetcher.requests.get", return_value=response) as get:
        rows = fetcher.fetch_projections(season="2026-2027", gameweek=11)
    assert rows["xp"].tolist() == [6]
    get.assert_called_once()


def test_archive_rejects_reused_fixture_ids_and_requires_latest_completed_round():
    fixtures = [
        {"id": 1, "event": 9, "finished": True, "kickoff_time": "2026-10-24T15:00:00Z"},
        {"id": 2, "event": 10, "finished": True, "kickoff_time": "2026-10-31T15:00:00Z"},
        {"id": 3, "event": 11, "finished": True, "kickoff_time": "2026-11-07T15:00:00Z"},
    ]
    rows = pd.DataFrame([
        (12, 1, 9, "2026-10-24T15:00:00Z", 45),
        (12, 2, 10, "2025-10-31T15:00:00Z", 90),
        (12, 3, 11, "2026-11-07T15:00:00Z", 90),
    ], columns=["element", "fixture", "round", "kickoff_time", "minutes"])
    assert verified_match_logs(rows, fixtures, 11) is None
    rows.loc[1, "kickoff_time"] = fixtures[1]["kickoff_time"]
    verified = verified_match_logs(pd.concat([rows, rows]), fixtures, 11)
    assert verified["fixture"].tolist() == [1, 2]


def test_doubles_blanks_and_partial_external_horizon(mock_bootstrap_data):
    fixtures = [
        {"id": 1, "event": 1, "team_h": 1, "team_a": 2},
        {"id": 2, "event": 1, "team_h": 3, "team_a": 1},
        {"id": 3, "event": 2, "team_h": 2, "team_a": 4},
        {"id": 4, "event": 3, "team_h": 4, "team_a": 1},
    ]
    kwargs = dict(current_event=1, odds_data={}, schedule_complete=True)
    raw = calculate_player_metrics(mock_bootstrap_data, fixtures + fixtures[:1], **kwargs).set_index("id")
    assert raw.loc[1, "fixtures_in_gw"] == 2
    assert raw.loc[1, "xp"] == pytest.approx((4.5 * 0.6 + 4 * 0.4) * 2.05, abs=0.01)
    assert raw.loc[1, "xp_by_gameweek"][2] == 0
    external = pd.DataFrame({"id": [1, 4], "GW1": [9, 50]})
    scores = calculate_player_metrics(mock_bootstrap_data, fixtures, fplreview_df=external, **kwargs).set_index("id")
    assert scores.loc[1, "xp"] == 9
    assert scores.loc[1, "xp_by_gameweek"][3] == raw.loc[1, "xp_by_gameweek"][3]
    assert scores.loc[4, "xp"] == 0
    assert scores.loc[4, "fixtures_in_gw"] == 0
    assert scores.loc[4, "xp_source"] == "blank_gameweek"


def test_minutes_and_injury_affect_scores_once(mock_bootstrap_data, mock_fixtures_data):
    mock_bootstrap_data["elements"][0].update(status="d", chance_of_playing_next_round=50)
    mock_bootstrap_data["elements"][1].update(status="i", chance_of_playing_next_round=100)
    history = {1: {"recent_matches_count": 4, "rolling_minutes_avg": 30, "rolling_minutes_when_playing": 60}}
    scores = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1,
                                      vaastav_stats=history, odds_data={})
    first, second = scores.iloc[0], scores.iloc[1]
    assert first["expected_minutes_per_fixture"] == 30
    assert first["xp"] == pytest.approx(4.3 * 1.08 * 1.05 * (60 / 90) * 0.5, abs=0.01)
    assert second["xp"] == 0
    assert second["xp_by_gameweek"][2] > 0
    optimizer = FPLOptimizer(scores)
    assert optimizer.player_map[1]["discounted_xp"] == first["xp"]
    assert optimizer.player_map[1]["discounted_3gw"] == first["xp_3gw"]
    assert optimizer.player_map[2]["injury_multiplier"] == 0


def test_live_planning_stops_when_fixture_schedule_is_missing(mock_bootstrap_data, monkeypatch):
    from src.main import run_pipeline

    monkeypatch.setenv("FPL_TEAM_ID", "123")
    client = MagicMock()
    client.get_bootstrap_static.return_value = mock_bootstrap_data
    client.get_my_team.return_value = {
        "picks": [{"element": i, "selling_price": 50} for i in range(1, 16)],
        "transfers": {"bank": 0, "limit": 1, "made": 0},
    }
    client.get_fixtures.return_value = []
    assert run_pipeline(client, execute=True)["status"] == "execution_failed"
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()


def test_partial_schedule_cannot_turn_missing_data_into_a_blank():
    teams = {1, 2, 3}
    fixtures = [{"team_h": home, "team_a": away, "event": 1}
                for home in teams for away in teams if home != away]
    assert verified_fixture_schedule(fixtures, teams, 1)
    assert not verified_fixture_schedule(fixtures[:-1], teams, 1)
    assert not verified_fixture_schedule(fixtures[:-1] + fixtures[:1], teams, 1)
