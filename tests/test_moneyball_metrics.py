"""Tests for Moneyball predictive metrics, VORP, regression deltas, and AI Director tagging."""

import pandas as pd
import pytest

from src.agent.director import AIDirector
from src.data_fetcher import VaastavDataFetcher
from src.engine.metrics import (
    POSITION_FLOOR_COST,
    POSITION_FLOOR_XP,
    calculate_player_metrics,
)
from src.engine.optimizer import (
    CandidateSquad,
    OptimizationResult,
    PlayerPick,
    TransferMove,
)


@pytest.fixture
def mock_bootstrap_moneyball():
    """Generates synthetic bootstrap data for Moneyball evaluation."""
    teams = [
        {"id": 1, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "name": "Man City", "short_name": "MCI"},
    ]
    elements = [
        # 1. Elite Captain Anchor (Haaland)
        {
            "id": 1,
            "web_name": "Haaland",
            "element_type": 4,  # FWD
            "team": 2,
            "now_cost": 150,  # £15.0m
            "form": "8.0",
            "points_per_game": "8.5",
            "total_points": 100,
            "status": "a",
            "chance_of_playing_next_round": 100,
        },
        # 2. Undervalued Positive Regression Candidate (High xGI, low actual goals)
        {
            "id": 2,
            "web_name": "Eze",
            "element_type": 3,  # MID
            "team": 1,
            "now_cost": 70,  # £7.0m
            "form": "3.0",
            "points_per_game": "4.0",
            "total_points": 30,
            "status": "a",
            "chance_of_playing_next_round": 100,
        },
        # 3. High Efficiency Budget Enabler (£4.5m DEF with solid xP)
        {
            "id": 3,
            "web_name": "Konsa",
            "element_type": 2,  # DEF
            "team": 1,
            "now_cost": 45,  # £4.5m
            "form": "4.5",
            "points_per_game": "4.2",
            "total_points": 45,
            "status": "a",
            "chance_of_playing_next_round": 100,
        },
        # 4. Overvalued Hauler (Lucky goals, negative xGI delta)
        {
            "id": 4,
            "web_name": "LuckyStriker",
            "element_type": 4,  # FWD
            "team": 1,
            "now_cost": 65,  # £6.5m
            "form": "7.0",
            "points_per_game": "6.0",
            "total_points": 50,
            "status": "a",
            "chance_of_playing_next_round": 100,
        },
    ]
    return {
        "teams": teams,
        "elements": elements,
        "events": [{"id": 1, "is_current": True, "is_next": False, "finished": False}],
    }


def test_vaastav_moneyball_underlying_metrics():
    fetcher = VaastavDataFetcher()

    # Synthetic merged_gw dataframe
    gw_records = [
        # Player 2 (Eze): 4 matches, 3.5 total xGI, but only 1 actual goal involvement (xGI delta = +2.5)
        {"element": 2, "round": 1, "minutes": 90, "expected_goal_involvements": 0.9, "expected_goals_conceded": 1.0, "goals_scored": 0, "assists": 0, "goals_conceded": 1, "starts": 1, "bps": 18},
        {"element": 2, "round": 2, "minutes": 90, "expected_goal_involvements": 0.8, "expected_goals_conceded": 0.5, "goals_scored": 1, "assists": 0, "goals_conceded": 0, "starts": 1, "bps": 28},
        {"element": 2, "round": 3, "minutes": 90, "expected_goal_involvements": 1.0, "expected_goals_conceded": 1.2, "goals_scored": 0, "assists": 0, "goals_conceded": 1, "starts": 1, "bps": 15},
        {"element": 2, "round": 4, "minutes": 90, "expected_goal_involvements": 0.8, "expected_goals_conceded": 0.8, "goals_scored": 0, "assists": 0, "goals_conceded": 1, "starts": 1, "bps": 20},
    ]
    merged_gw_df = pd.DataFrame(gw_records)

    raw_records = [
        {"id": 2, "expected_goal_involvements_per_90": 0.85, "expected_goals_conceded_per_90": 0.88, "minutes": 360, "goals_scored": 1, "assists": 0, "goals_conceded": 3, "bps": 81}
    ]
    players_raw_df = pd.DataFrame(raw_records)

    metrics = fetcher.calculate_player_underlying_metrics(merged_gw_df=merged_gw_df, players_raw_df=players_raw_df, lookback_gws=4)

    p2 = metrics[2]
    assert p2["rolling_xgi_90"] == round(3.5 / 4.0, 2)
    assert p2["actual_gi"] == 1.0
    assert p2["xgi_delta"] == round(3.5 - 1.0, 2)  # +2.5 delta
    assert p2["rolling_minutes_avg"] == 90.0
    assert p2["starts_ratio"] == 1.0


def test_calculate_player_metrics_moneyball_enrichment(mock_bootstrap_moneyball):
    vaastav_stats = {
        1: {  # Haaland
            "rolling_xgi_90": 0.95,
            "rolling_xgc_90": 0.70,
            "rolling_minutes_avg": 90.0,
            "starts_ratio": 1.0,
            "recent_matches_count": 4,
            "xgi_delta": 0.2,
            "xgc_delta": -0.2,
            "rolling_bps_avg": 35.0,
        },
        2: {  # Eze - Positive regression candidate
            "rolling_xgi_90": 0.80,
            "rolling_xgc_90": 1.0,
            "rolling_minutes_avg": 90.0,
            "starts_ratio": 1.0,
            "recent_matches_count": 4,
            "xgi_delta": 1.8,  # Underperforming expected output by 1.8 returns
            "xgc_delta": 0.0,
            "rolling_bps_avg": 22.0,
        },
        3: {  # Konsa - High efficiency budget defender
            "rolling_xgi_90": 0.15,
            "rolling_xgc_90": 0.60,
            "rolling_minutes_avg": 90.0,
            "starts_ratio": 1.0,
            "recent_matches_count": 4,
            "xgi_delta": 0.1,
            "xgc_delta": 0.5,
            "rolling_bps_avg": 20.0,
        },
        4: {  # LuckyStriker - Overvalued hauler
            "rolling_xgi_90": 0.20,
            "rolling_xgc_90": 1.20,
            "rolling_minutes_avg": 85.0,
            "starts_ratio": 1.0,
            "recent_matches_count": 4,
            "xgi_delta": -1.8,  # Scored 1.8 more than xGI
            "xgc_delta": 0.0,
            "rolling_bps_avg": 25.0,
        }
    }

    df = calculate_player_metrics(
        mock_bootstrap_moneyball,
        current_event=1,
        vaastav_stats=vaastav_stats,
    )

    haaland = df[df["id"] == 1].iloc[0]
    eze = df[df["id"] == 2].iloc[0]
    konsa = df[df["id"] == 3].iloc[0]
    lucky = df[df["id"] == 4].iloc[0]

    # Verify Haaland is recognized as ELITE_ANCHOR
    assert haaland["moneyball_tag"] == "ELITE_ANCHOR"
    assert haaland["vorp"] > 0
    assert haaland["cost_m"] == 15.0

    # Verify Eze is recognized as UNDERVALUED_REGRESSION
    assert eze["moneyball_tag"] == "UNDERVALUED_REGRESSION"
    assert eze["xgi_delta"] == 1.8
    assert eze["moneyball_score"] > 0

    # Verify Konsa is recognized as HIGH_EFFICIENCY_ENABLER
    assert konsa["moneyball_tag"] == "HIGH_EFFICIENCY_ENABLER"
    assert konsa["vorp_per_m"] >= 1.50

    # Verify LuckyStriker is recognized as OVERVALUED_HAULER
    assert lucky["moneyball_tag"] == "OVERVALUED_HAULER"


def test_ai_director_formats_moneyball_context(mock_bootstrap_moneyball):
    director = AIDirector(api_key="mock_test_key")

    p_out = PlayerPick(
        id=4,
        web_name="LuckyStriker",
        position="FWD",
        element_type=4,
        team_name="Arsenal",
        team_code="ARS",
        cost_m=6.5,
        xp=4.0,
    )
    p_in = PlayerPick(
        id=2,
        web_name="Eze",
        position="MID",
        element_type=3,
        team_name="Arsenal",
        team_code="ARS",
        cost_m=7.0,
        xp=5.5,
        rolling_xgi_90=0.80,
        xgi_delta=1.8,
        moneyball_tag="UNDERVALUED_REGRESSION",
    )
    haaland = PlayerPick(
        id=1,
        web_name="Haaland",
        position="FWD",
        element_type=4,
        team_name="Man City",
        team_code="MCI",
        cost_m=15.0,
        xp=8.0,
        is_starter=True,
        is_captain=True,
        moneyball_tag="ELITE_ANCHOR",
    )

    candidate = CandidateSquad(
        name="Buy Eze (Moneyball Move)",
        transfers_count=1,
        transfers=[TransferMove(player_out=p_out, player_in=p_in)],
        starters=[haaland, p_in],
        bench=[],
        captain=haaland,
        vice_captain=p_in,
        formation="3-5-2",
        gross_xp=13.5,
        hit_cost=0,
        net_xp=13.5,
        total_cost_m=85.0,
        bank_remaining_m=2.0,
    )

    opt_result = OptimizationResult(
        best_lineup=candidate,
        candidates=[candidate],
        current_team_value_m=100.0,
        bank_m=2.0,
        free_transfers=1,
    )

    # Test prompt generation contains CHASE mode and Moneyball buy flags
    competitive_ctx = {
        "risk_mode": "CHASE",
        "risk_mode_note": "Trailing leader by 50 pts — need high-EV differentials.",
    }

    # Intercept HTTP call to check formatted payload prompt
    with pytest.MonkeyPatch.context() as m:
        captured_payload = {}
        def mock_post(url, json=None, **kwargs):
            nonlocal captured_payload
            captured_payload = json
            mock_resp = type("MockResp", (), {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "choices": [{
                        "message": {
                            "content": '{"selected_candidate_index": 0, "captain_name": "Haaland", "vice_captain_name": "Eze", "rationale": "Locked in Haaland captaincy with Eze as high-xGI differential."}'
                        }
                    }]
                }
            })()
            return mock_resp

        m.setattr("requests.post", mock_post)
        decision = director.evaluate_and_decide(opt_result, competitive_context=competitive_ctx)

        assert decision.selected_candidate_index == 0
        assert decision.captain_name == "Haaland"
        assert decision.source == "LLM_DIRECTOR"

        # Check prompt content sent to LLM
        prompt_text = json_str = captured_payload["messages"][1]["content"]
        assert "[MONEYBALL BUY]" in prompt_text
        assert "Eze" in prompt_text
        assert "CHASE" in prompt_text
