"""Points model: fixture scaling, caching of finalized feeds, and metrics integration."""
import pandas as pd
import pytest

from src.engine import points_model as pm
from src.engine.metrics import calculate_player_metrics


def _row(element, gw, element_type, team, minutes=90, xg=0.0, xa=0.0, xgc=1.0, **extra):
    row = {"element": element, "gw": gw, "element_type": element_type, "team_id": team,
           "matches": 1, "minutes": minutes, "starts": int(minutes >= 60),
           "expected_goals_conceded": xgc if minutes else 0.0, "expected_goals": xg,
           "expected_assists": xa, "saves": 0, "bonus": 0, "defensive_contribution": 0,
           "yellow_cards": 0, "red_cards": 0}
    row.update(extra)
    return row


@pytest.fixture
def history():
    rows = []
    for gw in range(1, 9):
        # Team 1 striker; team 2 leaks goals, team 3 is a fortress.
        rows.append(_row(10, gw, 4, 1, xg=0.7, xa=0.1, xgc=1.0))
        rows.append(_row(11, gw, 1, 1, xgc=1.0, saves=3))
        rows.append(_row(20, gw, 1, 2, xgc=2.2, saves=4))
        rows.append(_row(21, gw, 4, 2, xg=0.3))
        rows.append(_row(30, gw, 1, 3, xgc=0.5, saves=1))
        rows.append(_row(31, gw, 4, 3, xg=0.5))
        rows.append(_row(12, gw, 3, 1, minutes=0))  # unused squad player
    return pd.DataFrame(rows)


def test_opponent_strength_blanks_doubles_and_minutes(history):
    state = pm.fit(history, 9)
    weak = pm.fixture_points(state, 10, opponent=2, is_home=True)
    strong = pm.fixture_points(state, 10, opponent=3, is_home=True)
    assert weak > strong > 0
    assert pm.fixture_points(state, 10, 2, is_home=True) > pm.fixture_points(state, 10, 2, is_home=False)
    assert pm.fixture_points(state, 12, 2, is_home=True) < 0.5  # never plays
    fixtures = [{"event": 9, "team_h": 1, "team_a": 2}, {"event": 10, "team_h": 1, "team_a": 2},
                {"event": 10, "team_h": 3, "team_a": 1}]
    out = pm.predict(state, [(10, 4, 1, 1.0), (21, 4, 2, 0.5)], fixtures, [9, 10, 11])
    assert out[10][11] == 0.0  # blank
    assert out[10][10] == pytest.approx(out[10][9] + pm.fixture_points(state, 10, 3, False), abs=0.01)
    full = pm.predict(state, [(21, 4, 2, 1.0)], fixtures, [9])[21][9]
    assert out[21][9] == pytest.approx(full / 2, abs=0.01)  # availability scales linearly


def test_only_past_gameweeks_are_used(history):
    future = pd.DataFrame([_row(10, 9, 4, 1, xg=5.0)])
    a = pm.fit(history, 9)
    b = pm.fit(pd.concat([history, future]), 9)
    assert pm.fixture_points(a, 10, 2, True) == pm.fixture_points(b, 10, 2, True)


def test_event_feeds_are_cached_and_unfinished_weeks_skipped(tmp_path):
    bootstrap = {
        "elements": [{"id": 1, "element_type": 4, "team": 7}],
        "events": [{"id": 1, "finished": True, "data_checked": True},
                   {"id": 2, "finished": True, "data_checked": False},
                   {"id": 3, "finished": False}],
    }
    feed = {"elements": [{"id": 1, "stats": {"minutes": 90, "starts": 1, "expected_goals": "0.45",
                                             "expected_goals_conceded": "1.10", "bonus": 2},
                          "explain": [{"fixture": 5, "stats": []}]},
                         {"id": 99, "stats": {"minutes": 90}, "explain": []}]}
    calls = []

    def fetch(gw):
        calls.append(gw)
        return feed

    first = pm.load_event_history(fetch, bootstrap, 3, tmp_path, "2026-2027")
    second = pm.load_event_history(fetch, bootstrap, 3, tmp_path, "2026-2027")
    assert calls == [1]
    assert (tmp_path / "2026-2027" / "gw01.json").exists()
    row = second.iloc[0].to_dict()
    assert len(second) == 1 and first.equals(second)
    assert (row["element"], row["team_id"], row["matches"], row["expected_goals"], row["bonus"]) == (1, 7, 1, 0.45, 2.0)


def _bootstrap():
    return {
        "elements": [
            {"id": 1, "web_name": "Doubtful", "element_type": 4, "team": 1, "now_cost": 80,
             "form": "6.0", "points_per_game": "5.0", "ep_next": "9.5", "status": "d",
             "chance_of_playing_next_round": 50, "selected_by_percent": "10"},
            {"id": 2, "web_name": "Fit", "element_type": 3, "team": 2, "now_cost": 60,
             "form": "6.0", "points_per_game": "5.0", "ep_next": "7.0", "status": "a",
             "selected_by_percent": "10"},
        ],
        "teams": [{"id": 1, "name": "One", "short_name": "ONE"}, {"id": 2, "name": "Two", "short_name": "TWO"}],
        "events": [{"id": 5, "is_next": True}],
    }


def test_metrics_use_model_for_later_weeks_and_keep_ep_next_by_default(monkeypatch):
    monkeypatch.setenv("HORIZON_LENGTH", "2")
    monkeypatch.delenv("FORECAST_NEXT_GW_SOURCE", raising=False)
    fixtures = [{"id": 1, "event": 5, "team_h": 1, "team_a": 2},
                {"id": 2, "event": 6, "team_h": 2, "team_a": 1}]
    model = {1: {5: 4.0, 6: 3.0}, 2: {5: 3.0, 6: 2.0}}
    df = calculate_player_metrics(_bootstrap(), fixtures, current_event=5, odds_data={},
                                  model_forecasts=model).set_index("id")
    fit = df.loc[2]
    assert fit["xp_source"] == "fpl_ep_next" and fit["xp"] == 7.0
    assert fit["xp_by_gameweek"][6] == 2.0 and fit["model_xp"] == 3.0
    doubtful = df.loc[1]
    # 50% available: model scaled by availability, ep_next capped by it.
    assert doubtful["model_xp"] == 2.0 and doubtful["xp"] == 2.0
    assert doubtful["xp_by_gameweek"][6] == 3.0 * 0.75

    monkeypatch.setenv("FORECAST_NEXT_GW_SOURCE", "model")
    df = calculate_player_metrics(_bootstrap(), fixtures, current_event=5, odds_data={},
                                  model_forecasts=model).set_index("id")
    assert df.loc[2, "xp_source"] == "points_model" and df.loc[2, "xp"] == 3.0
