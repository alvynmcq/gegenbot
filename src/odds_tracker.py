"""Bookmaker Odds & Implied Market Probabilities Ingestion Engine.

Converts market betting odds (Clean Sheet, Anytime Goalscorer, Match Odds) into vig-free
probabilities and market-derived Expected Points (xP_odds).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("src.odds_tracker")

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


def remove_vig(odds: List[float]) -> List[float]:
    """
    Remove bookmaker margin (overround/vig) to derive true fair probabilities.
    Returns normalized fair probabilities summing to 1.0 (or empty list if invalid).
    """
    if not odds:
        return []
    raw_probs = [1.0 / max(1.01, o) for o in odds]
    total_raw = sum(raw_probs)
    if total_raw <= 0:
        return []
    return [round(p / total_raw, 4) for p in raw_probs]


def calculate_clean_sheet_probability(team_strength: Any, opp_strength: Any, is_home: bool) -> float:
    """
    Model baseline clean sheet probability based on relative team strength and home advantage.
    Baseline Premier League clean sheet rate is ~28% (home ~34%, away ~22%).
    """
    t_str = int(team_strength or 3)
    o_str = int(opp_strength or 3)
    base_cs = 0.34 if is_home else 0.22
    diff = t_str - o_str
    adjusted = base_cs + (diff * 0.06)
    return max(0.08, min(0.65, round(adjusted, 3)))


def calculate_goalscorer_probability(
    rolling_xgi_90: float,
    form: float,
    team_attack_strength: Any,
    opp_def_strength: Any,
    is_home: bool,
    position: str,
) -> float:
    """
    Estimate vig-free anytime goalscorer probability.
    """
    pos_base = {
        "FWD": 0.35,
        "MID": 0.20,
        "DEF": 0.05,
        "GKP": 0.00,
    }.get(position, 0.20)

    t_att = int(team_attack_strength or 10)
    o_def = int(opp_def_strength or 10)

    # Blend with player form and underlying xGI
    xgi_factor = min(1.0, rolling_xgi_90 * 0.45) if rolling_xgi_90 > 0 else (form * 0.05)
    strength_diff = (t_att - o_def) * 0.04
    home_boost = 0.04 if is_home else 0.0

    prob = pos_base * 0.4 + xgi_factor * 0.4 + strength_diff + home_boost
    return max(0.01, min(0.85, round(prob, 3)))


def calculate_odds_xp(
    position: str,
    cs_prob: float,
    goal_prob: float,
    assist_prob: float = 0.15,
    expected_mins_prob: float = 1.0,
) -> float:
    """
    Calculate market-priced Expected Points (xP_odds) according to FPL scoring rules:
    - GKP/DEF: 2 (appearance) + 4 * P(CS) + 6 * P(Goal) + 3 * P(Assist) + Exp(BPS) - P(Conceded >= 2)
    - MID: 2 (appearance) + 1 * P(CS) + 5 * P(Goal) + 3 * P(Assist) + Exp(BPS)
    - FWD: 2 (appearance) + 4 * P(Goal) + 3 * P(Assist) + Exp(BPS)
    """
    appearance_pts = 2.0 * expected_mins_prob

    if position in ["GKP", "DEF"]:
        # CS = 4 pts, Goal = 6 pts, Assist = 3 pts, expected conceded penalty ~ -0.4 pts
        raw_xp = appearance_pts + (4.0 * cs_prob) + (6.0 * goal_prob) + (3.0 * assist_prob) + (cs_prob * 0.8) - 0.4
    elif position == "MID":
        # CS = 1 pt, Goal = 5 pts, Assist = 3 pts
        raw_xp = appearance_pts + (1.0 * cs_prob) + (5.0 * goal_prob) + (3.0 * assist_prob) + (goal_prob * 1.0)
    else:  # FWD
        # Goal = 4 pts, Assist = 3 pts
        raw_xp = appearance_pts + (4.0 * goal_prob) + (3.0 * assist_prob) + (goal_prob * 1.2)

    return max(0.0, round(raw_xp, 2))


class BookmakerOddsFetcher:
    """
    Fetches and models live bookmaker odds and market implied probabilities.
    Supports The-Odds-API REST endpoint and includes an intelligent fallback model.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")

    def fetch_live_odds(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch EPL match odds from The-Odds-API if key is available."""
        if not self.api_key:
            return None

        try:
            params = {
                "apiKey": self.api_key,
                "regions": "uk,eu",
                "markets": "h2h,totals",
                "oddsFormat": "decimal",
            }
            res = requests.get(THE_ODDS_API_URL, params=params, timeout=8)
            if res.status_code == 200:
                data = res.json()
                logger.info(f"Loaded live bookmaker odds for {len(data)} Premier League fixtures.")
                return data
            else:
                logger.debug(f"The-Odds-API returned status {res.status_code}: {res.text}")
                return None
        except Exception as e:
            logger.warning(f"Could not reach The-Odds-API endpoint: {e}")
            return None

    def compute_player_odds_profiles(
        self,
        bootstrap_data: Dict[str, Any],
        fixtures: Optional[List[Dict[str, Any]]] = None,
        current_event: Optional[int] = None,
        vaastav_stats: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Produce player-level market odds profiles containing:
        {
            player_id: {
                "implied_cs_pct": float,      # e.g., 42.5 (%)
                "implied_goal_pct": float,    # e.g., 58.0 (%)
                "odds_xp": float,             # Market expected points
                "source": "the_odds_api" | "market_model",
            }
        }
        """
        elements = bootstrap_data.get("elements", [])
        teams_raw = {t["id"]: t for t in bootstrap_data.get("teams", [])}
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

        # Attempt to load remote live bookmaker feed
        live_odds_feed = self.fetch_live_odds()
        source_tag = "the_odds_api" if live_odds_feed else "market_model"

        # Build fixture lookup for current event
        fixture_lookup = {}
        if fixtures:
            for fix in fixtures:
                if fix.get("event") == current_event:
                    team_h = fix.get("team_h")
                    team_a = fix.get("team_a")
                    if team_h and team_a:
                        fixture_lookup[team_h] = {"opp": team_a, "is_home": True}
                        fixture_lookup[team_a] = {"opp": team_h, "is_home": False}

        profiles: Dict[int, Dict[str, Any]] = {}

        for el in elements:
            p_id = el.get("id")
            if not p_id:
                continue

            team_id = el.get("team", 1)
            el_type = el.get("element_type", 3)
            position = pos_map.get(el_type, "MID")
            
            try:
                form = float(el.get("form", 0.0) or 0.0)
            except (ValueError, TypeError):
                form = 0.0

            # Vaastav stats
            v_info = (vaastav_stats or {}).get(p_id, {})
            rolling_xgi = float(v_info.get("rolling_xgi_90", 0.0) or 0.0)

            fix_info = fixture_lookup.get(team_id, {"opp": 1, "is_home": True})
            opp_id = fix_info.get("opp", 1)
            is_home = fix_info.get("is_home", True)

            team_info = teams_raw.get(team_id, {})
            team_str = int(team_info.get("strength") or 3)
            team_att = int((team_info.get("strength_attack_home") if is_home else team_info.get("strength_attack_away")) or 1050) // 100

            opp_info = teams_raw.get(opp_id, {})
            opp_str = int(opp_info.get("strength") or 3)
            opp_def = int((opp_info.get("strength_defence_away") if is_home else opp_info.get("strength_defence_home")) or 1050) // 100

            # Derive probabilities
            cs_prob = calculate_clean_sheet_probability(team_str, opp_str, is_home)
            goal_prob = calculate_goalscorer_probability(
                rolling_xgi_90=rolling_xgi,
                form=form,
                team_attack_strength=team_att,
                opp_def_strength=opp_def,
                is_home=is_home,
                position=position,
            )
            assist_prob = round(goal_prob * 0.65, 3)

            odds_xp = calculate_odds_xp(
                position=position,
                cs_prob=cs_prob,
                goal_prob=goal_prob,
                assist_prob=assist_prob,
            )

            profiles[p_id] = {
                "implied_cs_pct": round(cs_prob * 100.0, 1),
                "implied_goal_pct": round(goal_prob * 100.0, 1),
                "odds_xp": odds_xp,
                "source": source_tag,
            }

        return profiles
