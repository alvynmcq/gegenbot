"""Authentication and header management for official FPL API."""

import os
from typing import Dict, Optional


class FPLAuth:
    """Manages authentication tokens and HTTP headers for FPL API."""

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

    def __init__(self, token: Optional[str] = None):
        self.token = token if token is not None else os.getenv("FPL_AUTH_TOKEN", "").strip()

    def get_headers(self, authenticated: bool = False) -> Dict[str, str]:
        """Generate request headers with optional authentication token headers."""
        headers = dict(self.BASE_HEADERS)
        if authenticated and self.token:
            auth_value = self.token if self.token.startswith("Bearer ") else f"Bearer {self.token}"
            headers["Authorization"] = auth_value
            headers["X-API-Authorization"] = auth_value
        return headers

    @property
    def is_authenticated(self) -> bool:
        """Check if an authentication token is provided."""
        return bool(self.token)
