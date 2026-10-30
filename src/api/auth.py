"""Authentication and token state management for official FPL API and PingOne OAuth."""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.state import atomic_write_json

from dotenv import dotenv_values, load_dotenv
load_dotenv(override=False)

logger = logging.getLogger(__name__)


def _parse_jwt_exp(token: Optional[str]) -> Optional[float]:
    """Extract expiry timestamp from JWT if present."""
    if not token or not isinstance(token, str):
        return None
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            exp = payload.get("exp")
            if exp and isinstance(exp, (int, float)):
                return float(exp)
    except Exception:
        pass
    return None


def _newest_refresh_token(*candidates: Optional[str]) -> Optional[str]:
    """Pick the newest refresh token from candidates listed in order of preference.

    PingOne rotates refresh tokens and revokes the old one, so using a stale copy fails.
    When every candidate is a JWT, the one with the latest expiry is the newest.
    Otherwise (e.g. opaque tokens) the first candidate wins.
    """
    tokens = []
    for t in candidates:
        t = (t or "").strip()
        if t and t not in tokens:
            tokens.append(t)
    if not tokens:
        return None
    expiries = [_parse_jwt_exp(t) for t in tokens]
    if all(expiries):
        return tokens[expiries.index(max(expiries))]
    return tokens[0]


class FPLAuth:
    """Manages authentication tokens, state persistence, and HTTP headers for FPL API."""

    DEFAULT_STATE_PATH = Path("data/auth_state.json")

    BASE_HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fantasy.premierleague.com/",
        "Origin": "https://fantasy.premierleague.com",
    }

    def __init__(
        self,
        token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        token_expiry: Optional[float] = None,
        state_file: Optional[Union[str, Path]] = None,
    ):
        self.state_file = (Path(state_file) if state_file else
                           Path(os.environ["STATE_DIR"]) / "auth_state.json"
                           if os.getenv("STATE_DIR") else self.DEFAULT_STATE_PATH)
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = refresh_token
        self.token_expiry: Optional[float] = token_expiry

        # If token is explicitly provided in constructor (e.g. FPLAuth(token="...")), prioritize it
        if token is not None:
            self.access_token = token.strip() if token else ""
            if refresh_token:
                self.refresh_token = refresh_token.strip()
        else:
            # 1. Check data/auth_state.json
            self.load_state()

            # 2. If state token is missing or expired, fall back to environment variables
            if not self.access_token:
                env_access = (os.getenv("FPL_ACCESS_TOKEN", "") or os.getenv("FPL_AUTH_TOKEN", "")).strip()
                if env_access:
                    env_exp = _parse_jwt_exp(env_access)
                    now = time.time()
                    if env_exp is None or env_exp > now:
                        self.access_token = env_access
                        if env_exp:
                            self.token_expiry = env_exp
                        logger.info("Using active FPL_AUTH_TOKEN / FPL_ACCESS_TOKEN from environment.")

            self.refresh_token = _newest_refresh_token(
                self.refresh_token, os.getenv("FPL_REFRESH_TOKEN", "")
            )

    @property
    def token(self) -> str:
        """Alias for access_token for backward compatibility."""
        return self.access_token or ""

    @token.setter
    def token(self, value: Optional[str]) -> None:
        self.access_token = value.strip() if value else ""

    def load_state(self) -> bool:
        """Load credentials and token state from auth_state.json if available and not expired."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            token = data.get("access_token") or data.get("token") or None
            token_expiry = data.get("token_expiry")

            now = time.time()
            jwt_exp = _parse_jwt_exp(token)
            effective_expiry = token_expiry or jwt_exp

            if effective_expiry and effective_expiry <= now:
                logger.info(f"Cached access token in {self.state_file} has expired. Discarding cached token.")
                self.access_token = None
            else:
                self.access_token = token
                self.token_expiry = effective_expiry

            self.refresh_token = data.get("refresh_token") or self.refresh_token
            return bool(self.access_token or self.refresh_token)
        except Exception as e:
            logger.warning(f"Failed to load auth state from {self.state_file}: {e}")
            return False

    def reload_refresh_token(self, env_path: Union[str, Path] = ".env") -> None:
        """Adopt a newer refresh token written by another process.

        The dashboard's /api/auth/sync writes auth_state.json, and a user may paste a
        fresh token into .env, while the daemon keeps its own copy in memory.
        """
        state_refresh = None
        if self.state_file.exists():
            try:
                state_refresh = json.loads(self.state_file.read_text(encoding="utf-8")).get("refresh_token")
            except Exception as e:
                logger.warning(f"Failed to read refresh token from {self.state_file}: {e}")

        env_refresh = None
        env_file = Path(env_path)
        if "PYTEST_CURRENT_TEST" not in os.environ and env_file.exists():
            env_refresh = dotenv_values(env_file).get("FPL_REFRESH_TOKEN")

        newest = _newest_refresh_token(state_refresh, self.refresh_token, env_refresh)
        if newest and newest != self.refresh_token:
            logger.info("Picked up a newer FPL refresh token from disk.")
            self.refresh_token = newest

    def save_state(self) -> bool:
        """Persist active token state to auth_state.json atomically."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_expiry": self.token_expiry,
                "token_type": "Bearer",
                "updated_at": time.time(),
            }
            atomic_write_json(self.state_file, payload)
            logger.info(f"Saved active authentication tokens to {self.state_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save auth state to {self.state_file}: {e}")
            return False

    def update_tokens(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[float] = None,
    ) -> None:
        """Update token values in memory and persist to auth_state.json."""
        self.access_token = access_token.strip() if access_token else ""
        if refresh_token:
            self.refresh_token = refresh_token.strip()
        if expires_in is not None:
            self.token_expiry = time.time() + float(expires_in)
        if not self.save_state():
            raise RuntimeError("Could not persist authentication state")

    def get_headers(self, authenticated: bool = False) -> Dict[str, str]:
        """Generate request headers with optional authentication token headers."""
        headers = dict(self.BASE_HEADERS)
        if authenticated and self.access_token:
            auth_val = self.access_token if self.access_token.startswith("Bearer ") else f"Bearer {self.access_token}"
            headers["Authorization"] = auth_val
            headers["X-API-Authorization"] = auth_val
        return headers

    @property
    def is_authenticated(self) -> bool:
        """Check if an authentication token is available."""
        return bool(self.access_token)

    @property
    def can_refresh(self) -> bool:
        """Check if a refresh token is available."""
        return bool(self.refresh_token)
