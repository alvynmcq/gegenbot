"""Exercise recovery using API doubles; no live account mutations."""
import copy
import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from src.execution import build_plan, execute_plan, execution_path
from src.state import atomic_write_json


@pytest.fixture
def plan():
    picks = [{"element": i, "position": i - 1, "is_captain": i == 2,
              "is_vice_captain": i == 3} for i in range(2, 17)]
    return {
        "stage": "ready", "team_id": 123, "gameweek": 2,
        "baseline": {"selling_prices": {str(i): 50 for i in range(1, 16)},
                     "bank": 10, "limit": 1, "made": 0},
        "transfers": {"entry": 123, "event": 2, "chip": None, "chips": None,
                      "transfers": [{"element_in": 16, "element_out": 1,
                                     "selling_price": 50, "purchase_price": 55}]},
        "lineup": {"chip": None, "picks": picks},
        "result": {"status": "success", "mode": "dry_run", "gameweek": 2},
    }


@pytest.fixture
def client(plan):
    api = MagicMock()
    live = {"picks": [{"element": i, "selling_price": 50} for i in range(1, 16)],
            "chips": [], "transfers": {"bank": 10, "limit": 1, "made": 0}}
    api.get_my_team.side_effect = lambda team: copy.deepcopy(live)
    api.get_bootstrap_static.return_value = {
        "events": [{"id": 2, "deadline_time": "2099-01-01T12:00:00Z"}],
        "elements": [{"id": 16, "now_cost": 55}],
    }

    def submit_transfers(payload):
        live["picks"] = [{"element": i} for i in range(2, 17)]

    def submit_lineup(team, payload):
        live["picks"] = payload["picks"]

    api.post_transfers.side_effect = submit_transfers
    api.post_lineup.side_effect = submit_lineup
    return api


def test_transfer_payload_uses_half_profit_selling_price():
    outgoing = SimpleNamespace(id=1, cost_m=8.2, selling_price_m=7.6)
    incoming = SimpleNamespace(id=2, cost_m=7.7)
    candidate = SimpleNamespace(active_chip=None, transfers_count=1,
        transfers=[SimpleNamespace(player_out=outgoing, player_in=incoming)], starters=[], bench=[])
    payload = build_plan(candidate, 123, 2, {})["transfers"]["transfers"][0]
    assert payload["selling_price"] == 76
    assert payload["purchase_price"] == 77


def test_transfer_failure_skips_lineup_and_prevents_replay(tmp_path, plan, client):
    client.post_transfers.side_effect = requests.Timeout("response lost")
    path = tmp_path / "execution.json"
    assert execute_plan(client, path, plan)["status"] == "execution_failed"
    client.post_lineup.assert_not_called()
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": [{"element": 1}]}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_not_called()


def test_lost_transfer_response_reconciles_squad_and_finishes_original_lineup(tmp_path, plan, client):
    client.post_transfers.side_effect = requests.Timeout()
    path = tmp_path / "execution.json"
    execute_plan(client, path, plan)
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"], "chips": []}
    assert execute_plan(client, path)["status"] == "success"
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_called_once_with(123, plan["lineup"])
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_called_once()


def test_lost_lineup_response_reconciles_captain_without_replay(tmp_path, plan, client):
    client.post_lineup.side_effect = requests.Timeout()
    path = tmp_path / "execution.json"
    execute_plan(client, path, plan)
    wrong = copy.deepcopy(plan["lineup"]["picks"])
    wrong[0]["is_captain"] = False
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": wrong}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"], "chips": []}
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_called_once()
    client.post_transfers.assert_called_once()


def test_reconciliation_requires_chip_confirmation(tmp_path, plan, client):
    plan["stage"] = "lineup_pending"
    plan["lineup"]["chip"] = "3xc"
    path = tmp_path / "execution.json"
    atomic_write_json(path, plan)
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"], "chips": []}
    client.get_entry_history.return_value = {"chips": []}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.get_entry_history.return_value = {"chips": [{"name": "3xc", "event": 2}]}
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_not_called()


def test_auth_failure_can_retry_before_any_mutation(tmp_path, plan, client):
    client.ensure_authenticated.return_value = False
    path = tmp_path / "execution.json"
    assert execute_plan(client, path, plan)["status"] == "execution_failed"
    assert json.loads(path.read_text())["stage"] == "ready"
    client.post_transfers.assert_not_called()
    client.ensure_authenticated.return_value = True
    assert execute_plan(client, path)["status"] == "success"


def test_resume_after_transfers_done_does_not_send_transfers(tmp_path, plan, client):
    plan["stage"] = "transfers_done"
    path = tmp_path / "execution.json"
    atomic_write_json(path, plan)
    client.get_my_team.side_effect = None
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"], "chips": []}
    assert execute_plan(client, path)["status"] == "success"
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_called_once()


def test_pipeline_restart_uses_saved_plan_before_fetching_new_projections(monkeypatch, plan):
    from src.main import run_pipeline
    monkeypatch.setenv("FPL_TEAM_ID", "123")
    deadline = "2099-01-01T12:00:00Z"
    plan["stage"] = "complete"
    plan["result"].update(status="success", mode="live_execution")
    atomic_write_json(execution_path(123, 2, deadline), plan)
    client = MagicMock()
    client.get_bootstrap_static.return_value = {"events": [{"id": 2, "is_next": True, "deadline_time": deadline}]}
    assert run_pipeline(client, execute=True)["mode"] == "live_execution"
    client.get_fixtures.assert_not_called()
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()


def test_daemon_retries_failed_runs_and_skips_successful_runs(monkeypatch):
    import src.main as main
    deadline = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=60)).isoformat()
    client = MagicMock()
    client.get_bootstrap_static.return_value = {"events": [{"id": 2, "is_next": True, "deadline_time": deadline}]}
    pipeline = MagicMock(side_effect=[{"status": "execution_failed"}, {"status": "success"}])
    monkeypatch.setattr(main, "run_pipeline", pipeline)
    sleeps = iter([None, None])

    def stop_after_three_loops(seconds):
        try:
            next(sleeps)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", stop_after_three_loops)
    main.run_daemon(client)
    assert pipeline.call_count == 2


def test_post_transport_failure_is_not_retried(tmp_path):
    from src.api.auth import FPLAuth
    from src.api.client import FPLClient, FPLClientError
    session = MagicMock()
    session.request.side_effect = requests.Timeout()
    client = FPLClient(auth=FPLAuth(token="test", state_file=tmp_path / "auth.json"), session=session)
    with pytest.raises(FPLClientError):
        client.post_transfers({})
    session.request.assert_called_once()


def test_concurrent_executions_submit_only_once(tmp_path, plan, client):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "execution.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute_plan, client, path, copy.deepcopy(plan)) for _ in range(2)]
        assert all(future.result()["status"] == "success" for future in futures)
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_called_once()


@pytest.mark.parametrize("change", ["deadline", "hits", "price", "bank", "captain", "auth_read", "stale_snapshot"])
def test_preflight_blocks_writes(tmp_path, plan, client, change):
    if change == "deadline":
        client.get_bootstrap_static.return_value["events"][0]["deadline_time"] = "2000-01-01T00:00:00Z"
    elif change in ("hits", "bank"):
        live = client.get_my_team(123)
        key = "limit" if change == "hits" else "bank"
        live["transfers"][key] = plan["baseline"][key] = 0
        client.get_my_team.side_effect = lambda team: live
    elif change == "price":
        client.get_bootstrap_static.return_value["elements"][0]["now_cost"] += 1
    elif change == "stale_snapshot":
        plan["baseline"]["made"] = 1
    elif change == "captain":
        plan["lineup"]["picks"][0]["is_vice_captain"] = True
    else:
        client.get_my_team.side_effect = requests.Timeout()
    assert execute_plan(client, tmp_path / "execution.json", plan)["status"] == "execution_failed"
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()


@pytest.mark.parametrize("stage", ["transfers", "lineup"])
def test_success_response_without_saved_changes_is_never_replayed(tmp_path, plan, client, stage):
    post = getattr(client, "post_" + stage)
    post.side_effect = None
    post.return_value = {"status": "ok"}
    path = tmp_path / "execution.json"
    assert execute_plan(client, path, plan)["status"] == "execution_failed"
    assert json.loads(path.read_text())["stage"] == stage + "_pending"
    assert execute_plan(client, path)["status"] == "execution_failed"
    post.assert_called_once()
    if stage == "transfers":
        client.post_lineup.assert_not_called()


def test_live_pipeline_refuses_public_squad_fallback(monkeypatch):
    from src.main import run_pipeline

    monkeypatch.setenv("FPL_TEAM_ID", "123")
    client = MagicMock()
    client.get_bootstrap_static.return_value = {
        "events": [{"id": 2, "is_next": True, "deadline_time": "2099-01-01T12:00:00Z"}],
    }
    client.get_my_team.side_effect = requests.Timeout()
    assert run_pipeline(client, execute=True)["status"] == "execution_failed"
    client.get_entry_picks.assert_not_called()
    client.get_fixtures.assert_not_called()
    client.post_transfers.assert_not_called()
    client.post_lineup.assert_not_called()


@pytest.mark.parametrize("status", [307, 401, 403, 429])
def test_write_is_never_replayed_after_http_rejection(tmp_path, status):
    from src.api.auth import FPLAuth
    from src.api.client import FPLClient, FPLClientError

    session = MagicMock()
    session.request.return_value.status_code = status
    auth = FPLAuth(token="test", state_file=tmp_path / "auth.json")
    auth.refresh_token = "test-refresh"
    client = FPLClient(auth=auth, session=session)
    client.refresh_access_token = MagicMock(return_value=True)
    with pytest.raises(FPLClientError):
        client.post_transfers({})
    client.refresh_access_token.assert_not_called()
    session.request.assert_called_once()
    assert session.request.call_args.kwargs["allow_redirects"] is False


def test_journal_identity_survives_a_deadline_change_within_the_season():
    first = execution_path(123, 4, "2026-09-01T12:00:00Z")
    assert first == execution_path(123, 4, "2026-09-02T12:00:00Z")
    assert first != execution_path(123, 4, "2027-09-01T12:00:00Z")


def test_deadline_is_rechecked_between_transfer_and_lineup(tmp_path, plan, client):
    fresh = client.get_bootstrap_static.return_value
    client.get_bootstrap_static.side_effect = [
        fresh, {"events": [{"id": 2, "deadline_time": "2000-01-01T12:00:00Z"}]},
    ]
    path = tmp_path / "execution.json"
    result = execute_plan(client, path, plan)
    assert result["status"] == "execution_failed"
    assert json.loads(path.read_text())["stage"] == "transfers_done"
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_not_called()
