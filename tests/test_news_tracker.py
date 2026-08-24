"""Tests for NewsTracker and Live News Grounding in AI Decision Director."""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.director import AIDirector, DecisionOutput
from src.engine.optimizer import CandidateSquad, OptimizationResult, PlayerPick, TransferMove
from src.notifier.telegram import TelegramNotifier
from src.tracker.news_tracker import NewsTracker, PlayerNewsIntel


def _create_mock_player(
    p_id: int,
    name: str,
    pos: str = "MID",
    cost: float = 8.0,
    xp: float = 6.0,
    is_starter: bool = True,
    is_captain: bool = False,
    is_vc: bool = False,
) -> PlayerPick:
    return PlayerPick(
        id=p_id,
        web_name=name,
        position=pos,
        element_type=3 if pos == "MID" else 4,
        team_name="Arsenal",
        team_code="ARS",
        cost_m=cost,
        xp=xp,
        is_starter=is_starter,
        is_captain=is_captain,
        is_vice_captain=is_vc,
    )


def test_news_tracker_risk_level_determination():
    tracker = NewsTracker(enable_web_search=False)

    # 1. Healthy player
    assert tracker._determine_risk_level("a", None, "") == "CLEARED"
    assert tracker._determine_risk_level("a", 100, "") == "CLEARED"

    # 2. Ruled out / suspended
    assert tracker._determine_risk_level("i", 0, "Hamstring injury") == "RULED_OUT"
    assert tracker._determine_risk_level("s", None, "Suspended") == "RULED_OUT"
    assert tracker._determine_risk_level("a", 25, "Knock") == "RULED_OUT"

    # 3. 50% doubt
    assert tracker._determine_risk_level("d", 50, "Illness") == "DOUBT_50"
    assert tracker._determine_risk_level("d", None, "Doubtful") == "DOUBT_50"

    # 4. 75% doubt
    assert tracker._determine_risk_level("a", 75, "Tight groin") == "DOUBT_75"

    # 5. Active news text but status 'a'
    assert tracker._determine_risk_level("a", 100, "Joined team in training") == "MONITOR"


def test_news_tracker_build_player_intel():
    tracker = NewsTracker(enable_web_search=False)
    intel = tracker.build_player_intel(
        player_id=1,
        web_name="Saka",
        team_name="Arsenal",
        status="d",
        chance_of_playing_next_round=75,
        official_fpl_news="Knock - 75% chance of playing",
        news_added="2026-08-23 14:00:00",
        fetch_live_web=False,
    )

    assert intel.player_id == 1
    assert intel.web_name == "Saka"
    assert intel.risk_level == "DOUBT_75"
    assert "Saka (Arsenal) - Risk: DOUBT_75 | [75% chance]" in intel.summary()


def test_news_tracker_extract_focal_players_from_bootstrap():
    tracker = NewsTracker(enable_web_search=False)
    mock_bootstrap = {
        "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
        "elements": [
            {
                "id": 10,
                "web_name": "Saka",
                "team": 1,
                "status": "d",
                "chance_of_playing_next_round": 75,
                "news": "Knock",
                "news_added": "2026-08-20",
            },
            {
                "id": 11,
                "web_name": "Saliba",
                "team": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "news": "",
            },
            {
                "id": 12,
                "web_name": "Havertz",
                "team": 1,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "news": "",
            },
        ],
    }

    # Extract with focal ID 12 (Havertz) + Saka (flagged)
    intel_map = tracker.extract_focal_players_from_bootstrap(
        bootstrap_data=mock_bootstrap,
        focal_player_ids={12},
    )

    assert 10 in intel_map  # Saka because status != 'a'
    assert 12 in intel_map  # Havertz because focal player
    assert 11 not in intel_map  # Saliba is healthy and not in focal set


def test_ai_director_incorporates_news_intel():
    director = AIDirector(api_key="mock_key")

    saka = _create_mock_player(10, "Saka", is_captain=True)
    havertz = _create_mock_player(12, "Havertz", is_vc=True)
    bench_p = _create_mock_player(13, "Raya", pos="GKP", is_starter=False)

    cand = CandidateSquad(
        name="Candidate 1",
        transfers_count=0,
        transfers=[],
        starters=[saka, havertz],
        bench=[bench_p],
        captain=saka,
        vice_captain=havertz,
        formation="3-5-2",
        gross_xp=15.0,
        hit_cost=0,
        net_xp=15.0,
        total_cost_m=80.0,
        bank_remaining_m=1.0,
    )

    opt_result = OptimizationResult(
        candidates=[cand],
        current_team_value_m=100.0,
        bank_m=1.0,
        free_transfers=1,
    )

    news_intel = {
        10: PlayerNewsIntel(
            player_id=10,
            web_name="Saka",
            team_name="Arsenal",
            status="d",
            chance_of_playing_next_round=75,
            official_fpl_news="Muscle fatigue",
            risk_level="DOUBT_75",
        )
    }

    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "content": '{"selected_candidate_index": 0, "rationale": "Selected Candidate 1 due to high floor. Armband retained on Saka despite minor fatigue alert."}'
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_llm_response
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        decision = director.evaluate_and_decide(
            optimization_result=opt_result,
            news_intel=news_intel,
        )

        assert decision.source == "LLM_DIRECTOR"
        assert decision.selected_candidate_index == 0
        assert len(decision.news_alerts) == 1
        assert "Saka (Arsenal) - Risk: DOUBT_75" in decision.news_alerts[0]

        # Verify prompt payload sent to LLM included breaking news
        call_payload = mock_post.call_args[1]["json"]
        user_content = json_str = call_payload["messages"][1]["content"]
        assert "breaking_news_and_injuries" in user_content
        assert "Saka (Arsenal)" in user_content


def test_telegram_notifier_renders_news_alerts():
    saka = _create_mock_player(10, "Saka", is_captain=True)
    havertz = _create_mock_player(12, "Havertz", is_vc=True)
    bench_p = _create_mock_player(13, "Raya", pos="GKP", is_starter=False)

    cand = CandidateSquad(
        name="Roll Transfer",
        transfers_count=0,
        transfers=[],
        starters=[saka, havertz],
        bench=[bench_p],
        captain=saka,
        vice_captain=havertz,
        formation="3-5-2",
        gross_xp=15.0,
        hit_cost=0,
        net_xp=15.0,
        total_cost_m=80.0,
        bank_remaining_m=1.0,
    )

    decision = DecisionOutput(
        selected_candidate_index=0,
        selected_candidate=cand,
        chosen_move_name="Roll Transfer",
        transfers_description="No transfers (Roll/Bank transfer).",
        captain_name="Saka",
        vice_captain_name="Havertz",
        projected_net_xp=15.0,
        rationale="Rolled transfer to preserve flexibility.",
        source="LLM_DIRECTOR",
        news_alerts=["Saka (Arsenal) - Risk: DOUBT_75 | [75% chance] | FPL: Muscle fatigue"],
    )

    notifier = TelegramNotifier()
    alert_msg = notifier.build_pre_deadline_alert(decision=decision, gameweek=3, is_live_execution=False)

    assert "PRESS CONFERENCE & NEWS ALERTS:" in alert_msg
    assert "Saka (Arsenal) - Risk: DOUBT_75" in alert_msg
    assert "AI DIRECTOR RATIONALE:" in alert_msg
