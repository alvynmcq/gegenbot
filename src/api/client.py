"""FPL API Client for data retrieval and automated execution."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Client for Fantasy Premier League REST API with caching and retry backoff."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    CACHE_DIR = Path("data")
    BOOTSTRAP_CACHE_FILE = CACHE_DIR / "bootstrap_cache.json"
    CACHE_TTL_SECONDS = 600  # 10 minutes

    def __init__(
        self,
        auth: Optional[FPLAuth] = None,
        session: Optional[requests.Session] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.auth = auth or FPLAuth()
        self.session = session or requests.Session()
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_cache_path = self.cache_dir / "bootstrap_cache.json"

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
        """Execute HTTP request with 429 rate limit backoff and error handling."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = self.auth.get_headers(authenticated=authenticated)

        for attempt in range(1, max_retries + 1):
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

                if response.status_code == 401 or response.status_code == 403:
                    raise FPLClientError(
                        f"Authentication failed ({response.status_code}): Ensure FPL_AUTH_TOKEN is valid."
                    )

                response.raise_for_status()
                return response.json()

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
