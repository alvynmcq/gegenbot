"""Prevent spent chips from reappearing after news vetoes or before live writes."""
from unittest.mock import MagicMock, patch

import pytest

from src.chips import available_chips


def test_live_inventory_respects_chip_instances_and_windows():
    inventory = [
        {"name": "freehit", "status_for_entry": "played", "played_by_entry": [6], "start_event": 1, "stop_event": 19},
        {"name": "freehit", "status_for_entry": "available", "start_event": 20, "stop_event": 38},
    ]
    history = {"chips": [{"name": "freehit", "event": 6}]}
    assert available_chips(9, inventory, history) == []
    assert available_chips(20, inventory, history) == ["freehit"]


def test_active_chip_prevents_another_activation():
    assert available_chips(9, [
        {"name": "bboost", "status_for_entry": "active"},
        {"name": "freehit", "status_for_entry": "available"},
    ]) == []


def test_history_fallback_excludes_used_chip_and_unknown_inventory_fails_closed():
    assert "freehit" not in available_chips(9, history={"chips": [{"name": "freehit"}]})
    assert available_chips(9, history={}) is None
    assert available_chips(9, []) == []


@pytest.mark.parametrize("status,auto_enabled", [("played", "true"), ("available", "false")])
def test_news_resolve_cannot_restore_spent_or_disabled_chips(
    mock_bootstrap_data, mock_fixtures_data, monkeypatch, status, auto_enabled,
):
    import src.main as main
    from src.agent.director import AIDirector
    from src.api.client import FPLClient

    monkeypatch.setenv("FPL_TEAM_ID", "123456")
    monkeypatch.setenv("ENABLE_AUTO_CHIPS", auto_enabled)
    monkeypatch.setenv("FREE_HIT_MIN_XP_GAIN", "-1000")
    for key in ("WILDCARD_MIN_XP_GAIN", "BENCH_BOOST_MIN_BENCH_XP", "TRIPLE_CAPTAIN_MIN_XP"):
        monkeypatch.setenv(key, "9999")
    client = MagicMock(spec=FPLClient)
    client.get_bootstrap_static.return_value = mock_bootstrap_data
    client.get_fixtures.return_value = mock_fixtures_data
    client.get_entry_history.return_value = {"chips": [{"name": "freehit", "event": 1}] if status == "played" else []}
    client.get_my_team.return_value = {
        "picks": [{"element": i, "selling_price": 50} for i in [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]],
        "transfers": {"bank": 10, "limit": 1},
        "chips": [{"name": "freehit", "status_for_entry": status}],
    }
    fallback = AIDirector(api_key="")
    seen = []

    def decide(result, *args, **kwargs):
        seen.append(result)
        assert all(candidate.active_chip is None for candidate in result.candidates)
        if status == "played":
            assert "freehit" not in result.chip_evaluation.evaluations
        decision = fallback.evaluate_and_decide(result, *args, **kwargs)
        if len(seen) == 1:
            decision.veto_player_ids = [decision.selected_candidate.starters[1].id]
        return decision

    with patch.object(main, "AIDirector") as director, patch.object(main, "NewsTracker") as news, patch.object(main, "TelegramNotifier"):
        director.return_value.evaluate_and_decide.side_effect = decide
        news.return_value.extract_focal_players_from_bootstrap.return_value = {}
        result = main.run_pipeline(client, dry_run=True)
    assert len(seen) == 2
    assert result["active_chip"] is None
    assert result["mode"] == "dry_run"
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()


def test_live_execution_rechecks_chip_even_for_a_saved_plan(tmp_path):
    from src.execution import execute_plan
    client = MagicMock()
    client.get_my_team.return_value = {"chips": [{"name": "freehit", "status_for_entry": "played"}]}
    plan = {"stage": "ready", "team_id": 123, "gameweek": 9,
            "transfers": {"chip": "freehit"}, "lineup": {"chip": None, "picks": []}, "result": {}}
    result = execute_plan(client, tmp_path / "execution.json", plan)
    assert result["status"] == "execution_failed"
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()
