"""FPL API Client for data retrieval and automated execution with self-healing authentication."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .auth import FPLAuth

logger = logging.getLogger(__name__)


class FPLClientError(Exception):
    """Base exception for FPL API Client errors."""
    pass


class FPLRateLimitError(FPLClientError):
    """Raised when rate limit is exceeded and retries exhausted."""
    pass


class FPLClient:
    """Client for Fantasy Premier League REST API with caching, PingOne OAuth token refresh, and retry backoff."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    PINGONE_TOKEN_URL = "https://auth.pingone.eu/68340de1-dfb9-412e-937c-20172986d129/as/token"
    PINGONE_CLIENT_ID = "1f243d70-a140-4035-8c41-341f5af5aa12"
    LOGIN_URL = "https://users.premierleague.com/accounts/login/"
    CACHE_DIR = Path("data")
    BOOTSTRAP_CACHE_FILE = CACHE_DIR / "bootstrap_cache.json"
    CACHE_TTL_SECONDS = 600  # 10 minutes

    def __init__(
        self,
        auth: Optional[FPLAuth] = None,
        session: Optional[requests.Session] = None,
        cache_dir: Optional[Path] = None,
        client_id: Optional[str] = None,
    ):
        self.auth = auth or FPLAuth()
        self.session = session or requests.Session()
        self.client_id = client_id or self.PINGONE_CLIENT_ID
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_cache_path = self.cache_dir / "bootstrap_cache.json"

        # Sync access token to session cookies
        self._sync_session_cookies()

    def _sync_session_cookies(self) -> None:
        """Synchronize current access token to session cookie jar under .premierleague.com domain."""
        if self.auth.access_token:
            clean_token = self.auth.access_token.replace("Bearer ", "").strip()
            self.session.cookies.set("access_token", clean_token, domain=".premierleague.com", path="/")

    def refresh_access_token(self) -> bool:
        """
        Refresh FPL OAuth access token using PingOne token endpoint or login fallback.
        Updates session headers, cookie jar, auth state, and persists to data/auth_state.json.
        """
        refresh_token = self.auth.refresh_token
        if refresh_token:
            logger.info("Attempting automated PingOne OAuth2 token refresh...")
            token_endpoints = [
                (self.PINGONE_TOKEN_URL, self.client_id),
                ("https://account.premierleague.com/as/token", "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"),
                ("https://auth.pingone.eu/68340de1-dfb9-412e-937c-20172986d129/as/token", "1f243d70-a140-4035-8c41-341f5af5aa12"),
            ]
            seen_targets = set()
            endpoints_to_try = []
            for u, cid in token_endpoints:
                if (u, cid) not in seen_targets:
                    seen_targets.add((u, cid))
                    endpoints_to_try.append((u, cid))

            for endpoint_url, cid in endpoints_to_try:
                payload = {
                    "grant_type": "refresh_token",
                    "client_id": cid,
                    "refresh_token": refresh_token,
                }
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                }
                try:
                    resp = requests.post(
                        endpoint_url,
                        data=payload,
                        headers=headers,
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        new_access = data.get("access_token")
                        new_refresh = data.get("refresh_token", refresh_token)
                        expires_in = data.get("expires_in", 7200)

                        if new_access:
                            self.auth.update_tokens(
                                access_token=new_access,
                                refresh_token=new_refresh,
                                expires_in=expires_in,
                            )
                            self._sync_session_cookies()
                            logger.info(f"Successfully refreshed OAuth2 access token via {endpoint_url} and updated state.")
                            return True
                        else:
                            logger.warning(f"OAuth endpoint {endpoint_url} response did not contain access_token.")
                    else:
                        logger.warning(
                            f"OAuth token refresh via {endpoint_url} failed with HTTP {resp.status_code}: {resp.text}"
                        )
                except Exception as e:
                    logger.warning(f"Error during token refresh via {endpoint_url}: {e}")

        # Fallback to direct credentials login if available
        if self.auth.email and self.auth.password:
            logger.info(f"Attempting direct credential login fallback for {self.auth.email}...")
            return self.login_with_credentials(self.auth.email, self.auth.password)

        logger.warning("No valid refresh token or login credentials available for token refresh.")
        return False

    def login_with_credentials(self, email: Optional[str] = None, password: Optional[str] = None) -> bool:
        """Authenticate directly against FPL users login endpoint as fallback."""
        user_email = email or self.auth.email
        user_pwd = password or self.auth.password
        if not user_email or not user_pwd:
            logger.warning("Email or password not provided for credential login.")
            return False

        payload = {
            "login": user_email,
            "password": user_pwd,
            "app": "plfpl-web",
            "redirect_uri": "https://fantasy.premierleague.com/",
        }
        headers = dict(self.auth.BASE_HEADERS)
        try:
            resp = self.session.post(self.LOGIN_URL, data=payload, headers=headers, timeout=15)
            cookies = resp.cookies.get_dict()
            token = cookies.get("access_token") or cookies.get("pl_profile")
            if token:
                self.auth.update_tokens(access_token=token, expires_in=7200)
                self._sync_session_cookies()
                logger.info("Direct credential login succeeded.")
                return True
            if resp.status_code in (200, 302):
                for cookie in self.session.cookies:
                    if cookie.name in ("access_token", "pl_profile"):
                        self.auth.update_tokens(access_token=cookie.value, expires_in=7200)
                        self._sync_session_cookies()
                        return True
            logger.warning(f"Direct credential login failed with HTTP {resp.status_code}.")
        except Exception as e:
            logger.warning(f"Exception during direct credential login: {e}")
        return False

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> Any:
        """Execute HTTP request with 429 rate limit backoff, 401 token refresh interceptor, and error handling."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        token_refreshed = False

        for attempt in range(1, max_retries + 1):
            headers = self.auth.get_headers(authenticated=authenticated)
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=15,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(
                        f"Rate limit 429 hit on {url}. Retrying after {retry_after}s (Attempt {attempt}/{max_retries})"
                    )
                    time.sleep(retry_after)
                    continue

                # 401 / 403 Interceptor & Auto-Retry
                if (response.status_code == 401 or response.status_code == 403) and authenticated:
                    if not token_refreshed and self.auth.can_refresh:
                        logger.info(f"Received HTTP {response.status_code} on {url}. Intercepting to refresh token...")
                        if self.refresh_access_token():
                            token_refreshed = True
                            # Replay the original request transparently with new headers
                            new_headers = self.auth.get_headers(authenticated=authenticated)
                            retry_resp = self.session.request(
                                method=method.upper(),
                                url=url,
                                headers=new_headers,
                                params=params,
                                json=json_data,
                                timeout=15,
                            )
                            if retry_resp.status_code not in (401, 403):
                                retry_resp.raise_for_status()
                                return retry_resp.json()
                            response = retry_resp

                    logger.warning(f"Authentication failed on {url} (HTTP {response.status_code}). Ensure FPL_AUTH_TOKEN is valid.")
                    raise FPLClientError(
                        f"Authentication failed ({response.status_code}): Ensure FPL_AUTH_TOKEN is valid."
                    )

                # Fast-fail non-retryable 4xx client errors (404 not found, 400 bad request, etc.)
                if response.status_code == 404:
                    logger.debug(f"Resource not found (404) for {url}")
                    raise FPLClientError(f"Resource not found (404) for {url}")

                if 400 <= response.status_code < 500:
                    logger.warning(f"Client error ({response.status_code}) on {url}: {response.text}")
                    raise FPLClientError(f"Client error ({response.status_code}) for {url}")

                response.raise_for_status()
                return response.json()

            except FPLClientError:
                raise
            except requests.exceptions.RequestException as exc:
                if attempt == max_retries:
                    logger.error(f"HTTP request failed permanently for {url}: {exc}")
                    raise FPLClientError(f"Request failed for {url}: {exc}") from exc
                sleep_time = backoff_factor ** attempt
                logger.warning(f"Request failed ({exc}), retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        raise FPLRateLimitError(f"Exceeded max retries for {url}")

    # ==========================================
    # Public GET Endpoints
    # ==========================================

    def get_bootstrap_static(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch bootstrap-static data with 10-minute file-based local cache."""
        now = time.time()
        if not force_refresh and self.bootstrap_cache_path.exists():
            try:
                mtime = self.bootstrap_cache_path.stat().st_mtime
                if (now - mtime) < self.CACHE_TTL_SECONDS:
                    logger.info("Loading bootstrap-static from local cache.")
                    with open(self.bootstrap_cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read bootstrap cache: {e}. Fetching fresh data.")

        logger.info("Fetching fresh bootstrap-static from FPL API.")
        data = self._request("GET", "bootstrap-static/", authenticated=False)

        try:
            with open(self.bootstrap_cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save bootstrap cache: {e}")

        return data

    def get_fixtures(self, event: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch fixtures, optionally filtered by gameweek event ID."""
        params = {"event": event} if event is not None else None
        return self._request("GET", "fixtures/", params=params, authenticated=False)

    def get_league_standings(self, league_id: int) -> Dict[str, Any]:
        """Fetch classic mini-league standings and manager metadata."""
        return self._request(
            "GET",
            f"leagues-classic/{league_id}/standings/",
            authenticated=False,
        )

    def get_entry_picks(self, entry_id: int, event_id: int) -> Dict[str, Any]:
        """Fetch manager picks for a specific gameweek event."""
        return self._request(
            "GET",
            f"entry/{entry_id}/event/{event_id}/picks/",
            authenticated=False,
        )

    def get_entry_history(self, entry_id: int) -> Dict[str, Any]:
        """Fetch manager team overall history and gameweek summary."""
        return self._request(
            "GET",
            f"entry/{entry_id}/history/",
            authenticated=False,
        )

    # ==========================================
    # Authenticated Operations
    # ==========================================

    def get_my_team(self, team_id: int) -> Dict[str, Any]:
        """Fetch user squad picks, bank balance, and free transfers available."""
        if not self.auth.is_authenticated:
            raise FPLClientError("Authentication token required to fetch my-team data.")
        return self._request(
            "GET",
            f"my-team/{team_id}/",
            authenticated=True,
        )

    def post_transfers(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit transfer operations to FPL API."""
        if not self.auth.is_authenticated:
            raise FPLClientError("Authentication token required to execute transfers.")
        return self._request(
            "POST",
            "transfers/",
            json_data=payload,
            authenticated=True,
        )

    def post_lineup(self, team_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit starting XI lineup, captain, vice-captain, and bench hierarchy."""
        if not self.auth.is_authenticated:
            raise FPLClientError("Authentication token required to submit lineup.")
        return self._request(
            "POST",
            f"my-team/{team_id}/",
            json_data=payload,
            authenticated=True,
        )

    def validate_auth(self, team_id: int) -> Tuple[bool, str]:
        """Verify if authentication token is active and valid for team_id, attempting refresh if needed."""
        if not self.auth.is_authenticated:
            if self.auth.can_refresh and self.refresh_access_token():
                logger.info("Authentication self-healed via token refresh.")
            else:
                return False, "FPL_AUTH_TOKEN is not configured."
        try:
            self.get_my_team(team_id)
            return True, "Authentication token valid."
        except FPLClientError as e:
            # Attempt a refresh and retry before declaring auth invalid
            if self.auth.can_refresh and self.refresh_access_token():
                try:
                    self.get_my_team(team_id)
                    return True, "Authentication token refreshed and valid."
                except Exception as retry_err:
                    return False, f"Authentication check failed after refresh: {retry_err}"
            return False, f"Authentication check failed: {e}"
        except Exception as e:
            return False, f"Unexpected error during auth check: {e}"
