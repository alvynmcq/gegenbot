"""Unit tests for persistent, self-healing FPL authentication and PingOne token refresh."""

import json
from pathlib import Path
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


