"""Tests for Bookmaker Odds and Implied Market Probability Engine."""

import pytest
import pandas as pd

from src.odds_tracker import (
    BookmakerOddsFetcher,
    calculate_clean_sheet_probability,
    calculate_goalscorer_probability,
    calculate_odds_xp,
    remove_vig,
)
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import PlayerPick, TransferMove, CandidateSquad, OptimizationResult
from src.agent.director import AIDirector


def test_remove_vig():
    # 3-way match odds with standard bookmaker overround (~105%)
    odds = [2.0, 3.4, 3.8]  # Implied raw: 0.50 + 0.294 + 0.263 = 1.057
    fair_probs = remove_vig(odds)

    assert len(fair_probs) == 3
    assert abs(sum(fair_probs) - 1.0) < 0.001
    assert fair_probs[0] > fair_probs[1] > fair_probs[2]


def test_calculate_clean_sheet_probability():
    # Strong team at home vs weak team
    cs_home = calculate_clean_sheet_probability(team_strength=5, opp_strength=2, is_home=True)
    # Weak team away vs strong team
    cs_away = calculate_clean_sheet_probability(team_strength=2, opp_strength=5, is_home=False)

    assert cs_home > cs_away
    assert cs_home >= 0.45
    assert cs_away <= 0.20


def test_calculate_goalscorer_and_odds_xp():
    # Elite striker with strong xGI
    fwd_goal_prob = calculate_goalscorer_probability(
        rolling_xgi_90=0.85,
        form=7.0,
        team_attack_strength=12,
        opp_def_strength=9,
        is_home=True,
        position="FWD",
    )
    assert fwd_goal_prob >= 0.40

    # Calculate expected points for FWD
    fwd_xp = calculate_odds_xp(
        position="FWD",
        cs_prob=0.30,
        goal_prob=fwd_goal_prob,
        assist_prob=0.25,
    )
    assert fwd_xp >= 4.0


def test_odds_integration_in_player_metrics():
    bootstrap = {
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS", "strength": 5, "strength_attack_home": 1300, "strength_defence_home": 1300},
            {"id": 2, "name": "Southampton", "short_name": "SOU", "strength": 2, "strength_attack_away": 950, "strength_defence_away": 950},
        ],
        "elements": [
            {
                "id": 1,
                "web_name": "Saka",
                "element_type": 3,
                "team": 1,
                "now_cost": 100,
                "form": "6.0",
                "points_per_game": "6.0",
                "total_points": 50,
                "status": "a",
                "chance_of_playing_next_round": 100,
            },
            {
                "id": 2,
                "web_name": "Gabriel",
                "element_type": 2,
                "team": 1,
                "now_cost": 60,
                "form": "5.0",
                "points_per_game": "5.0",
                "total_points": 45,
                "status": "a",
                "chance_of_playing_next_round": 100,
            },
        ],
        "events": [{"id": 1, "is_current": True, "is_next": False, "finished": False}],
    }

    df = calculate_player_metrics(bootstrap, current_event=1)

    saka = df[df["id"] == 1].iloc[0]
    gabriel = df[df["id"] == 2].iloc[0]

    assert "implied_cs_pct" in df.columns
    assert "implied_goal_pct" in df.columns
    assert saka["implied_goal_pct"] > 0
    assert gabriel["implied_cs_pct"] > 0


def test_ai_director_formats_market_odds():
    director = AIDirector(api_key="mock_test_key")

    p_in = PlayerPick(
        id=1,
        web_name="Haaland",
        position="FWD",
        element_type=4,
        team_name="Man City",
        team_code="MCI",
        cost_m=15.0,
        xp=9.5,
        implied_goal_pct=65.0,
    )
    p_out = PlayerPick(
        id=2,
        web_name="BenchFWD",
        position="FWD",
        element_type=4,
        team_name="Ipswich",
        team_code="IPS",
        cost_m=5.5,
        xp=2.0,
    )

    cand = CandidateSquad(
        name="Bring in Haaland",
        transfers_count=1,
        transfers=[TransferMove(player_out=p_out, player_in=p_in)],
        starters=[p_in],
        bench=[],
        captain=p_in,
        vice_captain=p_in,
        formation="3-4-3",
        gross_xp=9.5,
        hit_cost=0,
        net_xp=9.5,
        total_cost_m=90.0,
        bank_remaining_m=0.5,
    )

    opt_result = OptimizationResult(
        best_lineup=cand,
        candidates=[cand],
        current_team_value_m=100.0,
        bank_m=0.5,
        free_transfers=1,
    )

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
                            "content": '{"selected_candidate_index": 0, "captain_name": "Haaland", "vice_captain_name": "Haaland", "rationale": "High market goal probability makes Haaland an essential captaincy lock."}'
                        }
                    }]
                }
            })()
            return mock_resp

        m.setattr("requests.post", mock_post)
        director.evaluate_and_decide(opt_result)

        prompt_text = captured_payload["messages"][1]["content"]
        assert "[MARKET ODDS]: 65.0% anytime goal probability" in prompt_text
