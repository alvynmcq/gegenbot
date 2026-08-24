"""Live News & Press Conference Tracker for FPL Players."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PlayerNewsIntel(BaseModel):
    """Structured injury, news, and press conference intelligence for an individual player."""
    player_id: int
    web_name: str
    team_name: str
    status: str = "a"
    chance_of_playing_next_round: Optional[int] = None
    official_fpl_news: str = ""
    news_added: Optional[str] = None
    live_search_snippets: List[str] = Field(default_factory=list)
    risk_level: str = "CLEARED"  # "CLEARED", "DOUBT_75", "DOUBT_50", "RULED_OUT", "MONITOR"

    def summary(self) -> str:
        """One-line concise summary of player news state."""
        parts = [f"{self.web_name} ({self.team_name}) - Risk: {self.risk_level}"]
        if self.chance_of_playing_next_round is not None:
            parts.append(f"[{self.chance_of_playing_next_round}% chance]")
        if self.official_fpl_news:
            parts.append(f"FPL: {self.official_fpl_news}")
        if self.live_search_snippets:
            parts.append(f"Latest: {self.live_search_snippets[0]}")
        return " | ".join(parts)


class NewsTracker:
    """Aggregates official FPL availability flags and live press conference updates."""

    def __init__(
        self,
        enable_web_search: Optional[bool] = None,
        tavily_api_key: Optional[str] = None,
        search_timeout: int = 5,
    ):
        if enable_web_search is None:
            self.enable_web_search = os.getenv("ENABLE_LIVE_NEWS", "true").lower() in ("true", "1", "yes")
        else:
            self.enable_web_search = enable_web_search

        self.tavily_api_key = (tavily_api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        self.search_timeout = search_timeout
        self._cache: Dict[str, List[str]] = {}

    def _determine_risk_level(self, status: str, chance: Optional[int], news_text: str) -> str:
        """Classify availability risk level based on FPL status and chance."""
        if status in ("i", "s", "u") or chance == 0:
            return "RULED_OUT"
        if chance is not None:
            if chance <= 25:
                return "RULED_OUT"
            if chance == 50:
                return "DOUBT_50"
            if chance == 75:
                return "DOUBT_75"
        if status == "d":
            return "DOUBT_50"
        if news_text:
            return "MONITOR"
        return "CLEARED"

    def _search_web_snippets(self, player_name: str, team_name: str) -> List[str]:
        """Perform a lightweight search for recent press conference / injury updates."""
        if not self.enable_web_search:
            return []

        cache_key = f"{player_name}_{team_name}".lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        snippets: List[str] = []

        # 1. Tavily Search API if configured
        if self.tavily_api_key:
            try:
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": f"{player_name} {team_name} injury update press conference FPL news",
                        "search_depth": "basic",
                        "max_results": 2,
                    },
                    timeout=self.search_timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for r in data.get("results", []):
                        raw_content = r.get("content", "")
                        if raw_content:
                            clean = re.sub(r"\s+", " ", raw_content).strip()
                            snippets.append(clean[:200])
            except Exception as e:
                logger.debug(f"Tavily search failed for {player_name}: {e}")

        # 2. Free DuckDuckGo Lite / HTML search fallback
        if not snippets:
            try:
                query = f"{player_name} {team_name} injury press conference FPL"
                url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                resp = requests.get(url, headers=headers, timeout=self.search_timeout)
                if resp.status_code == 200:
                    # Match snippet text in duckduckgo html
                    matches = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                    for m in matches[:2]:
                        clean = re.sub(r"<[^>]+>", "", m)
                        clean = re.sub(r"\s+", " ", clean).strip()
                        if clean:
                            snippets.append(clean[:200])
            except Exception as e:
                logger.debug(f"DuckDuckGo news search failed for {player_name}: {e}")

        self._cache[cache_key] = snippets
        return snippets

    def build_player_intel(
        self,
        player_id: int,
        web_name: str,
        team_name: str,
        status: str = "a",
        chance_of_playing_next_round: Optional[int] = None,
        official_fpl_news: str = "",
        news_added: Optional[str] = None,
        fetch_live_web: bool = True,
    ) -> PlayerNewsIntel:
        """Create a PlayerNewsIntel record with live search grounding if warranted."""
        risk = self._determine_risk_level(status, chance_of_playing_next_round, official_fpl_news)
        snippets: List[str] = []

        # Only query web search if player is flagged or is a high-priority target
        if fetch_live_web and (risk != "CLEARED" or official_fpl_news):
            snippets = self._search_web_snippets(web_name, team_name)

        return PlayerNewsIntel(
            player_id=player_id,
            web_name=web_name,
            team_name=team_name,
            status=status,
            chance_of_playing_next_round=chance_of_playing_next_round,
            official_fpl_news=official_fpl_news,
            news_added=news_added,
            live_search_snippets=snippets,
            risk_level=risk,
        )

    def extract_focal_players_from_bootstrap(
        self,
        bootstrap_data: Dict[str, Any],
        focal_player_ids: Optional[Set[int]] = None,
    ) -> Dict[int, PlayerNewsIntel]:
        """
        Scan bootstrap elements for player news.
        If focal_player_ids is provided, returns intel for those players plus any flagged players.
        """
        teams_map = {t["id"]: t.get("name", t.get("short_name", "Unknown")) for t in bootstrap_data.get("teams", [])}
        elements = bootstrap_data.get("elements", [])
        intel_map: Dict[int, PlayerNewsIntel] = {}

        for el in elements:
            p_id = el["id"]
            status = el.get("status", "a")
            news = el.get("news", "") or ""
            chance = el.get("chance_of_playing_next_round")
            news_added = el.get("news_added")
            team_name = teams_map.get(el.get("team"), "Unknown")
            web_name = el.get("web_name", "Unknown")

            is_focal = focal_player_ids is not None and p_id in focal_player_ids
            has_news_or_flag = status != "a" or bool(news) or (chance is not None and chance < 100)

            if is_focal or has_news_or_flag:
                intel = self.build_player_intel(
                    player_id=p_id,
                    web_name=web_name,
                    team_name=team_name,
                    status=status,
                    chance_of_playing_next_round=chance,
                    official_fpl_news=news,
                    news_added=news_added,
                    fetch_live_web=is_focal,
                )
                intel_map[p_id] = intel

        return intel_map
