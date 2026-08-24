"""Expected Points (xP), form, and Fixture Difficulty Rating (FDR) metrics calculation."""

import logging
import os
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Position mapping
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD"
}


def calculate_team_fdr_next_n_fixtures(
    fixtures: List[Dict[str, Any]],
    current_event: int,
    n_gameweeks: int = 3
) -> Dict[int, Dict[str, float]]:
    """
    Calculate average FDR and next fixture home/away status for each team over the next N gameweeks.
    Returns: {team_id: {"avg_fdr": float, "next_is_home": bool, "next_fdr": float}}
    """
    team_fdr: Dict[int, List[float]] = {team_id: [] for team_id in range(1, 21)}
    team_next_home: Dict[int, bool] = {}
    team_next_fdr: Dict[int, float] = {}

    target_events = set(range(current_event, current_event + n_gameweeks))

    for fix in fixtures:
        event = fix.get("event")
        if event in target_events and not fix.get("finished", False):
            team_h = fix.get("team_h")
            team_a = fix.get("team_a")
            diff_h = fix.get("team_h_difficulty", 3)
            diff_a = fix.get("team_a_difficulty", 3)

            if team_h in team_fdr:
                team_fdr[team_h].append(diff_h)
                if event == current_event and team_h not in team_next_home:
                    team_next_home[team_h] = True
                    team_next_fdr[team_h] = diff_h

            if team_a in team_fdr:
                team_fdr[team_a].append(diff_a)
                if event == current_event and team_a not in team_next_home:
                    team_next_home[team_a] = False
                    team_next_fdr[team_a] = diff_a

    result: Dict[int, Dict[str, float]] = {}
    for team_id in range(1, 21):
        diffs = team_fdr.get(team_id, [])
        avg_fdr = sum(diffs) / len(diffs) if diffs else 3.0
        result[team_id] = {
            "avg_fdr": round(avg_fdr, 2),
            "next_is_home": team_next_home.get(team_id, False),
            "next_fdr": team_next_fdr.get(team_id, avg_fdr),
        }

    return result


def calculate_player_metrics(
    bootstrap_data: Dict[str, Any],
    fixtures: Optional[List[Dict[str, Any]]] = None,
    current_event: Optional[int] = None,
    fplreview_df: Optional[pd.DataFrame] = None,
    fplreview_xp_map: Optional[Dict[int, Any]] = None,
) -> pd.DataFrame:
    """
    Process raw FPL bootstrap data, fixtures, and optional external FPL Review projections
    into an enriched DataFrame with expected points (xP).
    Uses FPL Review xP when available, with a sensible fallback to official FPL ep_next or FDR baseline.
    """
    from src.data_fetcher import map_fplreview_to_elements

    elements = bootstrap_data.get("elements", [])
    teams_raw = bootstrap_data.get("teams", [])
    events_raw = bootstrap_data.get("events", [])

    # Identify current/next active gameweek event
    if current_event is None:
        for ev in events_raw:
            if ev.get("is_next") or ev.get("is_current"):
                current_event = ev.get("id")
                break
        if current_event is None:
            current_event = 1

    # Team mapping
    team_names = {t["id"]: t["name"] for t in teams_raw}
    team_short = {t["id"]: t["short_name"] for t in teams_raw}

    # FDR calculation
    fdr_lookup = {}
    if fixtures:
        fdr_lookup = calculate_team_fdr_next_n_fixtures(fixtures, current_event, n_gameweeks=3)

    # Resolve FPL Review mapped projections
    fplreview_map: Dict[int, Any] = {}
    if fplreview_xp_map is not None:
        fplreview_map = fplreview_xp_map
    elif fplreview_df is not None and not fplreview_df.empty:
        fplreview_map = map_fplreview_to_elements(fplreview_df, bootstrap_data, current_event=current_event)

    records = []
    for el in elements:
        player_id = el["id"]
        web_name = el.get("web_name", "Unknown")
        element_type = el.get("element_type", 1)
        position = POSITION_MAP.get(element_type, "MID")
        team_id = el.get("team", 1)
        team_name = team_names.get(team_id, "Unknown")
        team_code = team_short.get(team_id, "UNK")
        now_cost = el.get("now_cost", 50)  # in tenths of millions (e.g. 100 = 10.0m)
        cost_m = now_cost / 10.0

        # Form & PPG
        try:
            form = float(el.get("form", 0.0) or 0.0)
        except (ValueError, TypeError):
            form = 0.0

        try:
            ppg = float(el.get("points_per_game", 0.0) or 0.0)
        except (ValueError, TypeError):
            ppg = 0.0

        # Official FPL ep_next
        try:
            ep_next_raw = el.get("ep_next")
            ep_next_val = float(ep_next_raw) if ep_next_raw is not None else None
        except (ValueError, TypeError):
            ep_next_val = None

        total_points = el.get("total_points", 0)
        selected_by_percent = float(el.get("selected_by_percent", 0.0) or 0.0)
        status = el.get("status", "a")
        chance_next = el.get("chance_of_playing_next_round")

        # Availability multiplier
        if chance_next is not None:
            try:
                availability = float(chance_next) / 100.0
            except (ValueError, TypeError):
                availability = 1.0
        elif status == "a":
            availability = 1.0
        elif status == "d":
            availability = 0.5
        elif status in ["i", "s", "u"]:
            availability = 0.0
        else:
            availability = 0.75

        # Fixture Difficulty Multiplier
        team_fdr_info = fdr_lookup.get(team_id, {"avg_fdr": 3.0, "next_is_home": False, "next_fdr": 3.0})
        avg_fdr = team_fdr_info.get("avg_fdr", 3.0)
        next_fdr = team_fdr_info.get("next_fdr", 3.0)
        next_is_home = team_fdr_info.get("next_is_home", False)

        # Baseline expected points: 60% Form + 40% Season PPG
        if form > 0 and ppg > 0:
            base_xp = 0.60 * form + 0.40 * ppg
        elif form > 0:
            base_xp = form
        elif ppg > 0:
            base_xp = ppg
        else:
            # Fallback baseline by position
            pos_fallback = {"GKP": 2.5, "DEF": 2.5, "MID": 3.0, "FWD": 3.0}
            base_xp = pos_fallback.get(position, 2.5)

        # FDR Scaling: Each FDR point below 3.0 gives +8% boost, above 3.0 gives -8% penalty
        fdr_multiplier = max(0.6, 1.0 + (3.0 - next_fdr) * 0.08)
        if next_is_home:
            fdr_multiplier *= 1.05  # 5% home advantage boost

        heuristic_xp = round(max(0.0, base_xp * fdr_multiplier * availability), 2)

        # Default FPL fallback expected points: official ep_next (if valid) or FDR heuristic
        if ep_next_val is not None and ep_next_val > 0:
            default_fpl_xp = round(max(0.0, ep_next_val * availability), 2)
            default_source = "fpl_ep_next"
        else:
            default_fpl_xp = heuristic_xp
            default_source = "fpl_heuristic"

        # Check FPL Core Insights / External projection
        fplreview_val: Optional[float] = None
        fplreview_3gw_val: Optional[float] = None
        decay_factor = float(os.getenv("DECAY_FACTOR", "0.85"))
        decay_sum = 1.0 + decay_factor + (decay_factor ** 2)
        source_tag = "fpl_core_insights"

        if player_id in fplreview_map:
            entry = fplreview_map[player_id]
            if isinstance(entry, dict):
                fplreview_val = entry.get("fplreview_xp")
                fplreview_3gw_val = entry.get("fplreview_xp_3gw")
                source_tag = entry.get("source", "fpl_core_insights")
            elif isinstance(entry, (int, float)):
                fplreview_val = float(entry)
                fplreview_3gw_val = round(fplreview_val * decay_sum, 2)
                source_tag = "fplreview"

        if fplreview_val is not None:
            # Respect availability if fully unavailable (injured/suspended)
            xp = round(max(0.0, fplreview_val * availability), 2) if availability == 0.0 else fplreview_val
            xp_source = source_tag
        else:
            xp = default_fpl_xp
            xp_source = default_source

        # 3-gameweek projection xP
        if fplreview_3gw_val is not None:
            xp_3gw = fplreview_3gw_val
        else:
            fdr_3gw_mult = max(0.6, 1.0 + (3.0 - avg_fdr) * 0.08)
            xp_3gw = round(max(0.0, base_xp * fdr_3gw_mult * availability * decay_sum), 2)

        try:
            transfers_in_event = int(el.get("transfers_in_event", 0) or 0)
            transfers_out_event = int(el.get("transfers_out_event", 0) or 0)
            # Normalise to a -1..+1 momentum score; capped at ±100k transfers
            net_transfers = transfers_in_event - transfers_out_event
            price_momentum = round(max(-1.0, min(1.0, net_transfers / 100_000.0)), 3)
        except (ValueError, TypeError):
            price_momentum = 0.0

        records.append({
            "id": player_id,
            "web_name": web_name,
            "element_type": element_type,
            "position": position,
            "team_id": team_id,
            "team_name": team_name,
            "team_code": team_code,
            "now_cost": now_cost,
            "cost_m": cost_m,
            "form": form,
            "points_per_game": ppg,
            "total_points": total_points,
            "selected_by_percent": selected_by_percent,
            "status": status,
            "chance_of_playing_next_round": chance_next,
            "availability": availability,
            "ep_next": ep_next_val,
            "fdr_next": next_fdr,
            "fdr_avg_3gw": avg_fdr,
            "fplreview_xp": fplreview_val,
            "xp": xp,
            "xp_3gw": xp_3gw,
            "xp_source": xp_source,
            "price_momentum": price_momentum,
        })


    df = pd.DataFrame(records)
    return df
