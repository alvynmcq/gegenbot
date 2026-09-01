"""Tests for Set-Piece Hierarchy extraction and Price Change Target Predictions."""

import pandas as pd
import pytest

from src.agent.director import AIDirector
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import (
    CandidateSquad,
    OptimizationResult,
    PlayerPick,
    TransferMove,
)
from src.price_tracker import calculate_price_change_targets


@pytest.fixture
def mock_bootstrap_set_pieces():
    """Generates synthetic FPL bootstrap data with set piece and price movement information."""
    teams = [
        {"id": 1, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "name": "Aston Villa", "short_name": "AVL"},
    ]
    elements = [
        # 1. Penalty Taker + Corner Specialist (Saka) with Imminent Price Rise
        {
            "id": 10,
            "web_name": "Saka",
            "element_type": 3,  # MID
            "team": 1,
            "now_cost": 100,
            "form": "6.0",
            "points_per_game": "6.5",
            "total_points": 50,
            "status": "a",
            "chance_of_playing_next_round": 100,
            "penalties_order": 1,
            "direct_freekicks_order": 2,
            "corners_and_indirect_freekicks_order": 1,
            "transfers_in_event": 120000,
            "transfers_out_event": 10000,
            "selected_by_percent": "35.0",
        },
        # 2. Corner / Set Piece Defender (Digne)
        {
            "id": 20,
            "web_name": "Digne",
            "element_type": 2,  # DEF
            "team": 2,
            "now_cost": 45,
            "form": "4.0",
            "points_per_game": "4.0",
            "total_points": 35,
            "status": "a",
            "chance_of_playing_next_round": 100,
            "penalties_order": None,
            "direct_freekicks_order": 1,
            "corners_and_indirect_freekicks_order": 1,
            "transfers_in_event": 5000,
            "transfers_out_event": 4000,
            "selected_by_percent": "5.0",
        },
        # 3. Massively Sold Player with Imminent Price Fall
        {
            "id": 30,
            "web_name": "FallingStar",
            "element_type": 3,  # MID
            "team": 1,
            "now_cost": 85,
            "form": "1.0",
            "points_per_game": "2.0",
            "total_points": 10,
            "status": "a",
            "chance_of_playing_next_round": 100,
            "penalties_order": None,
            "direct_freekicks_order": None,
            "corners_and_indirect_freekicks_order": None,
            "transfers_in_event": 2000,
            "transfers_out_event": 80000,
            "selected_by_percent": "15.0",
        },
    ]
    return {
        "teams": teams,
        "elements": elements,
        "events": [{"id": 1, "is_current": True, "is_next": False, "finished": False}],
    }


def test_calculate_price_change_targets(mock_bootstrap_set_pieces):
    targets = calculate_price_change_targets(mock_bootstrap_set_pieces)

    # Saka had net +110k transfers -> RISE_IMMINENT
    saka_target = targets[10]
    assert saka_target["status"] == "RISE_IMMINENT"
    assert saka_target["target_pct"] > 0
    assert saka_target["net_transfers"] == 110000

    # FallingStar had net -78k transfers -> FALL_IMMINENT
    fall_target = targets[30]
    assert fall_target["status"] == "FALL_IMMINENT"
    assert fall_target["target_pct"] < 0

    # Digne had balanced net transfers -> STABLE
    digne_target = targets[20]
    assert digne_target["status"] == "STABLE"


def test_set_piece_hierarchy_enrichment(mock_bootstrap_set_pieces):
    df = calculate_player_metrics(mock_bootstrap_set_pieces, current_event=1)

    saka = df[df["id"] == 10].iloc[0]
    digne = df[df["id"] == 20].iloc[0]
    falling = df[df["id"] == 30].iloc[0]

    # Verify Saka set piece role and price rise status
    assert bool(saka["is_penalty_taker"]) is True
    assert bool(saka["is_set_piece_taker"]) is True
    assert saka["set_piece_role"] == "PENALTIES + CORNERS/FK"
    assert saka["imminent_price_change"] == "RISE_IMMINENT"

    # Verify Digne set piece role
    assert bool(digne["is_penalty_taker"]) is False
    assert bool(digne["is_set_piece_taker"]) is True
    assert digne["set_piece_role"] == "CORNERS + DIRECT_FK"

    # Verify FallingStar has no set pieces and imminent fall
    assert bool(falling["is_set_piece_taker"]) is False
    assert falling["set_piece_role"] is None or pd.isna(falling["set_piece_role"])
    assert falling["imminent_price_change"] == "FALL_IMMINENT"


def test_ai_director_formats_set_pieces_and_price_warnings():
    director = AIDirector(api_key="mock_test_key")

    p_out = PlayerPick(
        id=30,
        web_name="FallingStar",
        position="MID",
        element_type=3,
        team_name="Arsenal",
        team_code="ARS",
        cost_m=8.5,
        xp=2.0,
        imminent_price_change="FALL_IMMINENT",
    )
    p_in = PlayerPick(
        id=10,
        web_name="Saka",
        position="MID",
        element_type=3,
        team_name="Arsenal",
        team_code="ARS",
        cost_m=10.0,
        xp=7.0,
        set_piece_role="PENALTIES + CORNERS/FK",
        imminent_price_change="RISE_IMMINENT",
    )

    candidate = CandidateSquad(
        name="Upgrade to Saka (Price Rise Target)",
        transfers_count=1,
        transfers=[TransferMove(player_out=p_out, player_in=p_in)],
        starters=[p_in],
        bench=[],
        captain=p_in,
        vice_captain=p_in,
        formation="3-5-2",
        gross_xp=7.0,
        hit_cost=0,
        net_xp=7.0,
        total_cost_m=80.0,
        bank_remaining_m=1.0,
    )

    opt_result = OptimizationResult(
        best_lineup=candidate,
        candidates=[candidate],
        current_team_value_m=100.0,
        bank_m=1.0,
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
                            "content": '{"selected_candidate_index": 0, "captain_name": "Saka", "vice_captain_name": "Saka", "rationale": "Locked in Saka ahead of price rise with full penalty and corner duties."}'
                        }
                    }]
                }
            })()
            return mock_resp

        m.setattr("requests.post", mock_post)
        decision = director.evaluate_and_decide(opt_result)

        assert decision.selected_candidate_index == 0
        prompt_text = captured_payload["messages"][1]["content"]

        # Verify prompt received set piece role and price warning badges
        assert "[SET PIECES]: PENALTIES + CORNERS/FK" in prompt_text
        assert "[PRICE RISE IMMINENT]" in prompt_text
        assert "[PRICE FALL IMMINENT]" in prompt_text
