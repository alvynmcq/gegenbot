"""Unit tests for persistent, self-healing FPL authentication and PingOne token refresh."""

import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.api.auth import FPLAuth
from src.api.client import FPLClient, FPLClientError


def test_auth_state_persistence_and_precedence(tmp_path, monkeypatch):
    """Test token state persistence to auth_state.json and precedence rules."""
    state_file = tmp_path / "auth_state.json"

    # Pre-populate state file
    state_file.write_text(json.dumps({
        "access_token": "cached_access_token_123",
        "refresh_token": "cached_refresh_token_456",
        "token_expiry": 1800000000.0,
    }))

    # Set conflicting environment variables
    monkeypatch.setenv("FPL_AUTH_TOKEN", "env_token_abc")
    monkeypatch.setenv("FPL_REFRESH_TOKEN", "env_refresh_xyz")

    # Priority 1: Loading from auth_state.json should take precedence over env vars
    auth = FPLAuth(state_file=state_file)
    assert auth.access_token == "cached_access_token_123"
    assert auth.refresh_token == "cached_refresh_token_456"
    assert auth.is_authenticated is True

    # Priority 2: When state_file is missing/empty, load from environment variables
    missing_state = tmp_path / "non_existent.json"
    auth_env = FPLAuth(state_file=missing_state)
    assert auth_env.access_token == "env_token_abc"
    assert auth_env.refresh_token == "env_refresh_xyz"

    # Test update_tokens saves to state_file
    auth_env.update_tokens(access_token="newly_acquired_token", refresh_token="new_refresh_token", expires_in=7200)
    assert missing_state.exists()
    saved_data = json.loads(missing_state.read_text())
    assert saved_data["access_token"] == "newly_acquired_token"
    assert saved_data["refresh_token"] == "new_refresh_token"


def test_pingone_token_refresh_success(tmp_path):
    """Test automated PingOne OAuth2 token refresh flow."""
    state_file = tmp_path / "auth_state.json"
    auth = FPLAuth(
        token="expired_token_1",
        refresh_token="valid_refresh_token_1",
        state_file=state_file,
    )
    client = FPLClient(auth=auth)

    mock_pingone_response = MagicMock()
    mock_pingone_response.status_code = 200
    mock_pingone_response.json.return_value = {
        "access_token": "refreshed_access_token_999",
        "refresh_token": "rotated_refresh_token_888",
        "expires_in": 7200,
        "token_type": "Bearer",
    }

    with patch("requests.post", return_value=mock_pingone_response) as mock_post:
        success = client.refresh_access_token()
        assert success is True
        assert client.auth.access_token == "refreshed_access_token_999"
        assert client.auth.refresh_token == "rotated_refresh_token_888"
        assert client.session.cookies.get("access_token") == "refreshed_access_token_999"

        # Verify PingOne request payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["data"]["grant_type"] == "refresh_token"
        assert call_kwargs["data"]["refresh_token"] == "valid_refresh_token_1"
        assert call_kwargs["data"]["client_id"] == client.PINGONE_CLIENT_ID

        # Verify state file was written
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert saved["access_token"] == "refreshed_access_token_999"
        assert saved["refresh_token"] == "rotated_refresh_token_888"


def test_http_401_interceptor_and_auto_retry(tmp_path):
    """Test HTTP 401 interceptor automatically triggers token refresh and replays the request."""
    state_file = tmp_path / "auth_state.json"
    auth = FPLAuth(
        token="stale_token",
        refresh_token="refresh_token_abc",
        state_file=state_file,
    )
    client = FPLClient(auth=auth)

    # Initial request returns 401, refreshed request returns 200
    resp_401 = MagicMock()
    resp_401.status_code = 401

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"picks": [{"element": 1, "position": 1}]}

    with patch.object(client.session, "request", side_effect=[resp_401, resp_200]) as mock_req:
        with patch.object(client, "refresh_access_token", return_value=True) as mock_refresh:
            result = client._request("GET", "my-team/12345/", authenticated=True)
            assert result == {"picks": [{"element": 1, "position": 1}]}
            assert mock_refresh.call_count == 1
            assert mock_req.call_count == 2


def test_http_401_interceptor_when_refresh_fails(tmp_path):
    """Test HTTP 401 raises FPLClientError with clear message when refresh fails."""
    auth = FPLAuth(
        token="bad_token",
        refresh_token="bad_refresh",
        state_file=tmp_path / "auth_state.json",
    )
    client = FPLClient(auth=auth)

    resp_401 = MagicMock()
    resp_401.status_code = 401

    with patch.object(client.session, "request", return_value=resp_401):
        with patch.object(client, "refresh_access_token", return_value=False):
            with pytest.raises(FPLClientError, match="Ensure FPL_AUTH_TOKEN is valid"):
                client._request("GET", "my-team/12345/", authenticated=True)


def test_unauthenticated_guard_when_no_refresh_available(tmp_path, monkeypatch):
    """Test graceful handling when no token or refresh credentials exist."""
    monkeypatch.delenv("FPL_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("FPL_EMAIL", raising=False)
    monkeypatch.delenv("FPL_PASSWORD", raising=False)
    monkeypatch.delenv("FPL_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("FPL_ACCESS_TOKEN", raising=False)

    auth = FPLAuth(token="", state_file=tmp_path / "empty.json")
    client = FPLClient(auth=auth)

    assert client.auth.can_refresh is False
    with pytest.raises(FPLClientError, match="Authentication token required"):
        client.get_my_team(12345)


def test_validate_auth_self_healing(tmp_path):
    """Test validate_auth heals invalid session via automated refresh before declaring failure."""
    state_file = tmp_path / "auth_state.json"
    auth = FPLAuth(
        token="expired_token",
        refresh_token="valid_refresh",
        state_file=state_file,
    )
    client = FPLClient(auth=auth)

    # First get_my_team raises 401 error, second attempt succeeds
    side_effects = [
        FPLClientError("Authentication failed (401): Ensure FPL_AUTH_TOKEN is valid."),
        {"picks": []},
    ]

    with patch.object(client, "get_my_team", side_effect=side_effects):
        with patch.object(client, "refresh_access_token", return_value=True) as mock_refresh:
            ok, msg = client.validate_auth(12345)
            assert ok is True
            assert "Authentication token refreshed and valid" in msg
            assert mock_refresh.call_count == 1


def test_http_404_fast_fail_no_retry():
    """Test HTTP 404 immediately raises FPLClientError without retrying or backoff."""
    client = FPLClient()
    resp_404 = MagicMock()
    resp_404.status_code = 404

    with patch.object(client.session, "request", return_value=resp_404) as mock_req:
        with pytest.raises(FPLClientError, match="Resource not found \\(404\\)"):
            client._request("GET", "entry/872480/event/2/picks/", max_retries=3)

        # Ensure it fast-failed on attempt 1 without wasting retries
        assert mock_req.call_count == 1


def test_expired_auth_state_falls_back_to_env(tmp_path, monkeypatch):
    """Test that an expired token in auth_state.json is discarded in favor of unexpired env token."""
    state_file = tmp_path / "auth_state.json"
    state_file.write_text(json.dumps({
        "access_token": "old_expired_token",
        "refresh_token": "old_refresh_token",
        "token_expiry": 1000.0,  # Far in the past
    }))

    monkeypatch.setenv("FPL_AUTH_TOKEN", "fresh_env_token")
    monkeypatch.setenv("FPL_REFRESH_TOKEN", "fresh_env_refresh")

    auth = FPLAuth(state_file=state_file)
    assert auth.access_token == "fresh_env_token"
    assert auth.refresh_token == "old_refresh_token"  # refresh token retained if not expired


def test_ensure_authenticated_auto_refresh(monkeypatch):
    """Test ensure_authenticated automatically triggers refresh when token is expired or missing."""
    auth = FPLAuth(token="", refresh_token="valid_refresh")
    client = FPLClient(auth=auth)

    refresh_called = []
    def mock_refresh():
        refresh_called.append(True)
        auth.access_token = "newly_refreshed_token"
        auth.token_expiry = time.time() + 3600
        return True

    monkeypatch.setattr(client, "refresh_access_token", mock_refresh)
    assert client.ensure_authenticated() is True
    assert len(refresh_called) == 1
    assert client.auth.access_token == "newly_refreshed_token"


def test_api_auth_sync_endpoint(tmp_path, monkeypatch):
    """Test dashboard /api/auth/sync route updates tokens and returns success."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FPL_AUTH_TOKEN=\nFPL_REFRESH_TOKEN=\n")
    from src.dashboard.app import create_app
    monkeypatch.setenv("DASHBOARD_SYNC_SECRET", "test-pairing-secret")
    app = create_app()
    test_client = app.test_client()

    res = test_client.post("/api/auth/sync", headers={"X-Gegenbot-Sync-Secret": "test-pairing-secret"}, json={
        "access_token": "mock_browser_access",
        "refresh_token": "mock_browser_refresh",
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "synchronized" in data["message"]





def _fake_jwt(exp: float) -> str:
    """Build an unsigned JWT-shaped string with the given expiry."""
    import base64
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{body}.sig"


def test_newer_env_refresh_token_beats_stale_state(tmp_path, monkeypatch):
    """A newer JWT refresh token in .env must win over an older one cached in auth_state.json."""
    old_rt, new_rt = _fake_jwt(1900000000), _fake_jwt(1950000000)
    state_file = tmp_path / "auth_state.json"
    state_file.write_text(json.dumps({"access_token": None, "refresh_token": old_rt}))
    monkeypatch.setenv("FPL_REFRESH_TOKEN", new_rt)

    assert FPLAuth(state_file=state_file).refresh_token == new_rt

    # And the reverse: a newer cached token is kept over an older env value
    state_file.write_text(json.dumps({"access_token": None, "refresh_token": new_rt}))
    monkeypatch.setenv("FPL_REFRESH_TOKEN", old_rt)
    assert FPLAuth(state_file=state_file).refresh_token == new_rt


def test_refresh_picks_up_token_synced_by_another_process(tmp_path):
    """The daemon must use a refresh token the dashboard wrote after the daemon started."""
    state_file = tmp_path / "auth_state.json"
    auth = FPLAuth(token="", refresh_token=_fake_jwt(1900000000), state_file=state_file)
    client = FPLClient(auth=auth, cache_dir=tmp_path)

    synced_rt = _fake_jwt(1950000000)
    state_file.write_text(json.dumps({"access_token": None, "refresh_token": synced_rt}))

    ok_resp = MagicMock(status_code=200)
    ok_resp.json.return_value = {"access_token": "new_access", "refresh_token": "rotated", "expires_in": 3600}
    with patch("requests.post", return_value=ok_resp) as mock_post:
        assert client.refresh_access_token() is True
        assert mock_post.call_args[1]["data"]["refresh_token"] == synced_rt
    assert json.loads(state_file.read_text())["refresh_token"] == "rotated"


def test_rejected_refresh_token_fails_cleanly(tmp_path, caplog):
    """invalid_grant returns False after one request and logs how to re-sync."""
    auth = FPLAuth(token="", refresh_token="revoked", state_file=tmp_path / "auth_state.json")
    client = FPLClient(auth=auth, cache_dir=tmp_path)

    bad_resp = MagicMock(status_code=400, text="{}")
    bad_resp.json.return_value = {"error": "invalid_grant", "error_description": "Refresh token does not exist"}
    with patch("requests.post", return_value=bad_resp) as mock_post:
        assert client.refresh_access_token() is False
        assert mock_post.call_count == 1
        assert mock_post.call_args[0][0] == "https://account.premierleague.com/as/token"
    assert "re-sync" in caplog.text
    assert client.auth.access_token == ""


def test_check_auth_reports_result(monkeypatch, capsys):
    from unittest.mock import MagicMock

    from src.main import run_check_auth

    client = MagicMock()
    monkeypatch.setenv("FPL_TEAM_ID", "123")
    client.validate_auth.return_value = (True, "Authentication token valid.")
    assert run_check_auth(client) == 0
    client.validate_auth.assert_called_with(123)
    assert capsys.readouterr().out.startswith("OK")

    client.validate_auth.return_value = (False, "refresh failed")
    assert run_check_auth(client) == 1
    assert capsys.readouterr().out.startswith("FAILED")

    monkeypatch.setenv("FPL_TEAM_ID", "")
    assert run_check_auth(client) == 1


def test_auth_sync_requires_secret_and_restricts_origins(monkeypatch):
    from src.dashboard.app import create_app
    monkeypatch.setenv("DASHBOARD_SYNC_SECRET", "pairing-secret")
    client = create_app().test_client()
    payload = {"access_token": "browser-token"}
    assert client.post("/api/auth/sync", json=payload).status_code == 403
    headers = {"X-Gegenbot-Sync-Secret": "pairing-secret", "Origin": "https://untrusted.example"}
    rejected = client.post("/api/auth/sync", json=payload, headers=headers)
    assert rejected.status_code == 403
    assert "Access-Control-Allow-Origin" not in rejected.headers
    preflight = client.options("/api/auth/sync", headers={"Origin": "https://fantasy.premierleague.com"})
    assert preflight.headers["Access-Control-Allow-Origin"] == "https://fantasy.premierleague.com"
    assert "X-Gegenbot-Sync-Secret" in preflight.headers["Access-Control-Allow-Headers"]


def test_auth_sync_works_without_writing_read_only_env(tmp_path, monkeypatch):
    from src.dashboard.app import create_app
    monkeypatch.setenv("DASHBOARD_SYNC_SECRET", "pairing-secret")
    env_file = tmp_path / ".env"
    original = "FPL_AUTH_TOKEN=old\nFPL_REFRESH_TOKEN=old-refresh\n"
    env_file.write_text(original)
    env_file.chmod(0o400)
    client = create_app().test_client()
    response = client.post("/api/auth/sync", json={"access_token": "new", "refresh_token": "new-refresh"},
                           headers={"X-Gegenbot-Sync-Secret": "pairing-secret"})
    assert response.status_code == 200
    assert env_file.read_text() == original
    saved = json.loads(FPLAuth.DEFAULT_STATE_PATH.read_text())
    assert saved["refresh_token"] == "new-refresh"
    assert FPLAuth.DEFAULT_STATE_PATH.stat().st_mode & 0o777 == 0o600


def test_auth_sync_reports_persistence_failure(monkeypatch):
    from src.dashboard.app import create_app
    monkeypatch.setenv("DASHBOARD_SYNC_SECRET", "pairing-secret")
    monkeypatch.setattr(FPLAuth, "save_state", lambda self: False)
    response = create_app().test_client().post("/api/auth/sync", json={"access_token": "new"},
                                             headers={"X-Gegenbot-Sync-Secret": "pairing-secret"})
    assert response.status_code == 500


@pytest.mark.parametrize("payload", [[], {"access_token": 123}, {"access_token": "new", "refresh_token": []}])
def test_auth_sync_rejects_invalid_payloads(monkeypatch, payload):
    from src.dashboard.app import create_app
    monkeypatch.setenv("DASHBOARD_SYNC_SECRET", "pairing-secret")
    response = create_app().test_client().post("/api/auth/sync", json=payload,
                                             headers={"X-Gegenbot-Sync-Secret": "pairing-secret"})
    assert response.status_code == 400


def test_generated_pairing_secret_survives_dashboard_restart(tmp_path):
    from src.dashboard.app import create_app
    create_app()
    secret_file = tmp_path / "data" / "dashboard_sync_secret.json"
    secret = json.loads(secret_file.read_text())["secret"]
    assert len(secret) >= 32
    assert secret_file.stat().st_mode & 0o777 == 0o600
    response = create_app().test_client().post("/api/auth/sync", json={"access_token": "new"},
                                             headers={"X-Gegenbot-Sync-Secret": secret})
    assert response.status_code == 200
