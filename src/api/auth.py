"""Authentication and token state management for official FPL API and PingOne OAuth."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


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
        email: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.state_file = Path(state_file) if state_file else self.DEFAULT_STATE_PATH
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = refresh_token
        self.token_expiry: Optional[float] = token_expiry
        self.email: Optional[str] = email or os.getenv("FPL_EMAIL", "").strip() or None
        self.password: Optional[str] = password or os.getenv("FPL_PASSWORD", "").strip() or None

        # If token is explicitly provided in constructor (e.g. FPLAuth(token="...")), prioritize it
        if token is not None:
            self.access_token = token.strip() if token else ""
            if refresh_token:
                self.refresh_token = refresh_token.strip()
        else:
            # 1. Check data/auth_state.json
            loaded_from_state = self.load_state()

            # 2. If not loaded from state or empty, check environment variables
            if not loaded_from_state or not self.access_token:
                env_access = (os.getenv("FPL_ACCESS_TOKEN", "") or os.getenv("FPL_AUTH_TOKEN", "")).strip()
                env_refresh = os.getenv("FPL_REFRESH_TOKEN", "").strip()
                if env_access:
                    self.access_token = env_access
                if env_refresh:
                    self.refresh_token = env_refresh

    @property
    def token(self) -> str:
        """Alias for access_token for backward compatibility."""
        return self.access_token or ""

    @token.setter
    def token(self, value: Optional[str]) -> None:
        self.access_token = value.strip() if value else ""

    def load_state(self) -> bool:
        """Load credentials and token state from auth_state.json if available."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.access_token = data.get("access_token") or data.get("token") or None
            self.refresh_token = data.get("refresh_token") or self.refresh_token
            self.token_expiry = data.get("token_expiry") or self.token_expiry
            logger.info(f"Loaded active token state from {self.state_file}")
            return bool(self.access_token or self.refresh_token)
        except Exception as e:
            logger.warning(f"Failed to load auth state from {self.state_file}: {e}")
            return False

    def save_state(self) -> bool:
        """Persist active token state to auth_state.json."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_expiry": self.token_expiry,
                "token_type": "Bearer",
                "updated_at": time.time(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
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
        self.save_state()

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
        """Check if refresh token or direct login credentials are available."""
        return bool(self.refresh_token or (self.email and self.password))
