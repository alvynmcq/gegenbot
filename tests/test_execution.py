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
    return {
        "stage": "ready", "team_id": 123, "gameweek": 2,
        "transfers": {"chip": None, "transfers": [{"element_in": 2, "element_out": 1}]},
        "lineup": {"chip": None, "picks": [
            {"element": 2, "position": 1, "is_captain": True, "is_vice_captain": False}]},
        "result": {"status": "success", "mode": "dry_run", "gameweek": 2},
    }


def test_transfer_payload_uses_half_profit_selling_price():
    outgoing = SimpleNamespace(id=1, cost_m=8.2, selling_price_m=7.6)
    incoming = SimpleNamespace(id=2, cost_m=7.7)
    candidate = SimpleNamespace(active_chip=None, transfers_count=1,
        transfers=[SimpleNamespace(player_out=outgoing, player_in=incoming)], starters=[], bench=[])
    payload = build_plan(candidate, 123, 2, {})["transfers"]["transfers"][0]
    assert payload["selling_price"] == 76
    assert payload["purchase_price"] == 77


def test_transfer_failure_skips_lineup_and_prevents_replay(tmp_path, plan):
    client = MagicMock()
    client.post_transfers.side_effect = requests.Timeout("response lost")
    path = tmp_path / "execution.json"
    assert execute_plan(client, path, plan)["status"] == "execution_failed"
    client.post_lineup.assert_not_called()
    client.get_my_team.return_value = {"picks": [{"element": 1}]}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_not_called()


def test_lost_transfer_response_reconciles_squad_and_finishes_original_lineup(tmp_path, plan):
    client = MagicMock()
    client.post_transfers.side_effect = requests.Timeout()
    path = tmp_path / "execution.json"
    execute_plan(client, path, plan)
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"]}
    assert execute_plan(client, path)["status"] == "success"
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_called_once_with(123, plan["lineup"])
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_called_once()


def test_lost_lineup_response_reconciles_captain_without_replay(tmp_path, plan):
    client = MagicMock()
    client.post_lineup.side_effect = requests.Timeout()
    path = tmp_path / "execution.json"
    execute_plan(client, path, plan)
    wrong = copy.deepcopy(plan["lineup"]["picks"])
    wrong[0]["is_captain"] = False
    client.get_my_team.return_value = {"picks": wrong}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"]}
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_called_once()
    client.post_transfers.assert_called_once()


def test_reconciliation_requires_chip_confirmation(tmp_path, plan):
    plan["stage"] = "lineup_pending"
    plan["lineup"]["chip"] = "3xc"
    path = tmp_path / "execution.json"
    atomic_write_json(path, plan)
    client = MagicMock()
    client.get_my_team.return_value = {"picks": plan["lineup"]["picks"]}
    client.get_entry_history.return_value = {"chips": []}
    assert execute_plan(client, path)["status"] == "execution_failed"
    client.get_entry_history.return_value = {"chips": [{"name": "3xc", "event": 2}]}
    assert execute_plan(client, path)["status"] == "success"
    client.post_lineup.assert_not_called()


def test_auth_failure_can_retry_before_any_mutation(tmp_path, plan):
    client = MagicMock()
    client.ensure_authenticated.return_value = False
    path = tmp_path / "execution.json"
    assert execute_plan(client, path, plan)["status"] == "execution_failed"
    assert json.loads(path.read_text())["stage"] == "ready"
    client.post_transfers.assert_not_called()
    client.ensure_authenticated.return_value = True
    assert execute_plan(client, path)["status"] == "success"


def test_resume_after_transfers_done_does_not_send_transfers(tmp_path, plan):
    plan["stage"] = "transfers_done"
    path = tmp_path / "execution.json"
    atomic_write_json(path, plan)
    client = MagicMock()
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


def test_concurrent_executions_submit_only_once(tmp_path, plan):
    from concurrent.futures import ThreadPoolExecutor
    client = MagicMock()
    path = tmp_path / "execution.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute_plan, client, path, copy.deepcopy(plan)) for _ in range(2)]
        assert all(future.result()["status"] == "success" for future in futures)
    client.post_transfers.assert_called_once()
    client.post_lineup.assert_called_once()
