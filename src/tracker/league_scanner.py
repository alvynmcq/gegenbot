"""Mini-league scanner for Effective Ownership (EO%), rival picks, and Threat Matrix."""

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.api.client import FPLClient

logger = logging.getLogger(__name__)

# In-memory picks cache: {(entry_id, gameweek): (timestamp, picks_data)}
_PICKS_CACHE: Dict[Tuple[int, int], Tuple[float, Dict[str, Any]]] = {}
_PICKS_CACHE_TTL = 3600  # 1 hour


class ThreatPlayer(BaseModel):
    """Player representation in threat matrix."""
    id: int
    web_name: str
    position: str
    team_code: str
    eo_percent: float
    owned_percent: float
    category: str  # "SHIELD", "VULNERABILITY", "DAGGER"
    risk_upside_note: str


class ThreatMatrix(BaseModel):
    """Categorized threat matrix for mini-league."""
    shields: List[ThreatPlayer] = Field(default_factory=list)        # High EO owned
    vulnerabilities: List[ThreatPlayer] = Field(default_factory=list) # High EO not owned
    daggers: List[ThreatPlayer] = Field(default_factory=list)        # Low EO differential owned


class RivalManager(BaseModel):
    """Summary of rival manager in mini-league."""
    entry_id: int
    player_name: str
    entry_name: str
    rank: int
    total_points: int
    event_transfers_cost: int = 0
    captain_name: Optional[str] = None
    active_chip: Optional[str] = None


class LeagueAnalysis(BaseModel):
    """Comprehensive mini-league scanning analysis."""
    league_id: int
    league_name: str
    gameweek: int
    total_managers: int
    rivals: List[RivalManager] = Field(default_factory=list)
    captain_distribution: Dict[str, int] = Field(default_factory=dict)
    threat_matrix: ThreatMatrix = Field(default_factory=ThreatMatrix)
    raw_eo: Dict[int, float] = Field(default_factory=dict)


class LeagueScanner:
    """Scans classic mini-leagues concurrently, computes EO% and generates tactical threat matrices."""

    def __init__(self, client: FPLClient, max_workers: int = 8):
        self.client = client
        self.max_workers = max_workers

    def _fetch_manager_picks(self, entry_id: int, gameweek: int) -> Optional[Dict[str, Any]]:
        """Fetch manager picks with in-memory TTL caching."""
        cache_key = (entry_id, gameweek)
        now = time.time()
        if cache_key in _PICKS_CACHE:
            ts, data = _PICKS_CACHE[cache_key]
            if (now - ts) < _PICKS_CACHE_TTL:
                return data

        try:
            picks_data = self.client.get_entry_picks(entry_id, gameweek)
            if picks_data:
                _PICKS_CACHE[cache_key] = (now, picks_data)
            return picks_data
        except Exception as exc:
            logger.debug(f"Failed to fetch picks for entry {entry_id} GW{gameweek}: {exc}")
            return None

    def scan_league(
        self,
        league_id: int,
        gameweek: int,
        my_team_ids: Optional[Set[int]] = None,
        bootstrap_data: Optional[Dict[str, Any]] = None,
    ) -> LeagueAnalysis:
        """
        Scan all managers in the specified mini-league concurrently for the target gameweek.
        """
        my_team_set = my_team_ids or set()

        if not bootstrap_data:
            bootstrap_data = self.client.get_bootstrap_static()

        elements = {el["id"]: el for el in bootstrap_data.get("elements", [])}
        teams = {t["id"]: t["short_name"] for t in bootstrap_data.get("teams", [])}
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

        # Fetch league standings
        try:
            standings_data = self.client.get_league_standings(league_id)
        except Exception as e:
            logger.warning(f"Could not fetch league standings for {league_id}: {e}")
            return LeagueAnalysis(
                league_id=league_id,
                league_name="Mini-League",
                gameweek=gameweek,
                total_managers=0,
            )

        league_name = standings_data.get("league", {}).get("name", f"League {league_id}")
        results = standings_data.get("standings", {}).get("results", [])

        if not results:
            logger.info(f"No managers found in league {league_id}")
            return LeagueAnalysis(
                league_id=league_id,
                league_name=league_name,
                gameweek=gameweek,
                total_managers=0,
            )

        total_managers = len(results)
        player_ownership_count: Dict[int, int] = {}
        player_eo_weighted: Dict[int, float] = {}
        captain_counts: Dict[str, int] = {}
        rivals_list: List[RivalManager] = []

        # Concurrent retrieval of rival squad picks
        manager_entries = [m.get("entry") for m in results if m.get("entry")]
        picks_results: Dict[int, Optional[Dict[str, Any]]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(manager_entries) or 1)) as executor:
            future_to_entry = {
                executor.submit(self._fetch_manager_picks, entry_id, gameweek): entry_id
                for entry_id in manager_entries
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                eid = future_to_entry[future]
                try:
                    picks_results[eid] = future.result()
                except Exception as exc:
                    logger.debug(f"Picks future failed for entry {eid}: {exc}")
                    picks_results[eid] = None

        for manager in results:
            entry_id = manager.get("entry")
            player_name = manager.get("player_name", "Manager")
            entry_name = manager.get("entry_name", "Team")
            rank = manager.get("rank", 0)
            total_points = manager.get("total", 0)

            rival_summary = RivalManager(
                entry_id=entry_id,
                player_name=player_name,
                entry_name=entry_name,
                rank=rank,
                total_points=total_points,
            )

            picks_data = picks_results.get(entry_id)
            if picks_data:
                entry_history = picks_data.get("entry_history", {})
                rival_summary.event_transfers_cost = entry_history.get("event_transfers_cost", 0)
                rival_summary.active_chip = picks_data.get("active_chip")

                picks = picks_data.get("picks", [])
                for pick in picks:
                    pid = pick["element"]
                    multiplier = pick.get("multiplier", 1)
                    is_captain = pick.get("is_captain", False)

                    player_ownership_count[pid] = player_ownership_count.get(pid, 0) + 1
                    player_eo_weighted[pid] = player_eo_weighted.get(pid, 0.0) + multiplier

                    if is_captain:
                        c_name = elements.get(pid, {}).get("web_name", f"Player {pid}")
                        rival_summary.captain_name = c_name
                        captain_counts[c_name] = captain_counts.get(c_name, 0) + 1

            rivals_list.append(rival_summary)

        # Compute percentages
        eo_percentages: Dict[int, float] = {}
        ownership_percentages: Dict[int, float] = {}
        for pid, weighted_val in player_eo_weighted.items():
            eo_percentages[pid] = round((weighted_val / total_managers) * 100.0, 1)
            ownership_percentages[pid] = round((player_ownership_count.get(pid, 0) / total_managers) * 100.0, 1)

        # Build Threat Matrix lists
        shields: List[ThreatPlayer] = []
        vulnerabilities: List[ThreatPlayer] = []
        daggers: List[ThreatPlayer] = []

        # Dynamic EO thresholds scaled to league sample size.
        # Smaller samples have lower signal so thresholds are relaxed.
        if total_managers >= 40:
            shield_threshold = 40.0
            vulnerability_threshold = 35.0
            dagger_threshold = 25.0
        elif total_managers >= 20:
            shield_threshold = 30.0
            vulnerability_threshold = 25.0
            dagger_threshold = 20.0
        else:
            shield_threshold = 20.0
            vulnerability_threshold = 15.0
            dagger_threshold = 15.0

        for pid, eo in eo_percentages.items():
            el = elements.get(pid, {})
            web_name = el.get("web_name", f"Player {pid}")
            pos = pos_map.get(el.get("element_type", 1), "MID")
            t_code = teams.get(el.get("team", 1), "UNK")
            own_pct = ownership_percentages.get(pid, 0.0)

            is_owned_by_me = pid in my_team_set

            if is_owned_by_me and eo >= shield_threshold:
                shields.append(ThreatPlayer(
                    id=pid,
                    web_name=web_name,
                    position=pos,
                    team_code=t_code,
                    eo_percent=eo,
                    owned_percent=own_pct,
                    category="SHIELD",
                    risk_upside_note=f"High league EO ({eo}%). Owned: protects current rank.",
                ))
            elif not is_owned_by_me and eo >= vulnerability_threshold:
                vulnerabilities.append(ThreatPlayer(
                    id=pid,
                    web_name=web_name,
                    position=pos,
                    team_code=t_code,
                    eo_percent=eo,
                    owned_percent=own_pct,
                    category="VULNERABILITY",
                    risk_upside_note=f"High rival EO ({eo}%). Unowned: major danger to rank if scoring.",
                ))
            elif is_owned_by_me and eo <= dagger_threshold:
                daggers.append(ThreatPlayer(
                    id=pid,
                    web_name=web_name,
                    position=pos,
                    team_code=t_code,
                    eo_percent=eo,
                    owned_percent=own_pct,
                    category="DAGGER",
                    risk_upside_note=f"Differential ({eo}% EO). Owned: massive rank gain opportunity.",
                ))

        shields.sort(key=lambda x: -x.eo_percent)
        vulnerabilities.sort(key=lambda x: -x.eo_percent)
        daggers.sort(key=lambda x: x.eo_percent)

        threat_matrix = ThreatMatrix(
            shields=shields[:6],
            vulnerabilities=vulnerabilities[:6],
            daggers=daggers[:6],
        )

        return LeagueAnalysis(
            league_id=league_id,
            league_name=league_name,
            gameweek=gameweek,
            total_managers=total_managers,
            rivals=rivals_list,
            captain_distribution=captain_counts,
            threat_matrix=threat_matrix,
            raw_eo=eo_percentages,
        )
