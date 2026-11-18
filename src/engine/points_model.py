"""Fixture-level expected FPL points from recent underlying data.

The model works from a per-player, per-gameweek history (the shape of FPL's
``event/{gw}/live`` feed and vaastav's ``merged_gw.csv``) so the same code runs
live and in backtests. For each fixture it estimates:

* minutes: chance of appearing, chance of 60+ minutes and expected minutes,
  weighted towards recent team matches;
* attacking returns: xG and xA per 90, scaled by how much the opponent concedes
  and by home advantage;
* defensive returns: clean-sheet chance, goals conceded and saves from Poisson
  team expectations, and defensive contribution from a per-90 action rate;
* bonus and cards from per-90 rates.

Rates are shrunk towards positional averages so a cameo or a new signing does not
produce an extreme forecast.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.state import atomic_write_json

logger = logging.getLogger(__name__)

GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}  # defensive contribution actions needed for 2 points
RATE_COLUMNS = ("expected_goals", "expected_assists", "saves", "bonus",
                "defensive_contribution", "yellow_cards", "red_cards")
HISTORY_COLUMNS = ("element", "gw", "element_type", "team_id", "matches", "minutes", "starts",
                   "expected_goals_conceded", *RATE_COLUMNS)


@dataclass(frozen=True)
class ModelConfig:
    minutes_half_life: float = 3.0   # gameweeks
    rate_half_life: float = 10.0
    team_half_life: float = 8.0
    rate_prior_minutes: float = 600.0
    minutes_prior_matches: float = 1.0
    team_prior_matches: float = 4.0
    home_advantage: float = 0.10     # +/- share of expected goals


@dataclass
class ModelState:
    target_gw: int
    players: pd.DataFrame            # indexed by element
    team_attack: Dict[int, float]    # expected goals scored per match
    team_defence: Dict[int, float]   # expected goals conceded per match
    league_goals: float
    config: ModelConfig


def _weights(gws: pd.Series, target_gw: int, half_life: float) -> pd.Series:
    return 0.5 ** ((target_gw - 1 - gws) / half_life)


def _poisson_floor_mean(rate: float, divisor: int, terms: int = 25) -> float:
    """E[floor(X / divisor)] for X ~ Poisson(rate)."""
    if rate <= 0:
        return 0.0
    total, p = 0.0, math.exp(-rate)
    for k in range(terms):
        if k:
            p *= rate / k
        total += p * (k // divisor)
    return total


def _poisson_at_least(rate: float, threshold: int) -> float:
    if rate <= 0:
        return 0.0
    p, cumulative = math.exp(-rate), 0.0
    for k in range(threshold):
        if k:
            p *= rate / k
        cumulative += p
    return max(0.0, 1.0 - cumulative)


def fit(history: pd.DataFrame, target_gw: int, config: ModelConfig = ModelConfig()) -> ModelState:
    """Summarise history strictly before ``target_gw``."""
    h = history.loc[(history["gw"] < target_gw) & (history["matches"] > 0)].copy()
    if h.empty:
        raise ValueError("No completed matches before the target gameweek")
    h["per_match_minutes"] = h["minutes"] / h["matches"]

    # Team strength from the player who saw every minute (the most-used goalkeeper).
    team_rows = h.groupby(["team_id", "gw"]).agg(
        scored=("expected_goals", "sum"),
        conceded=("expected_goals_conceded", "max"),
        matches=("matches", "max"),
    ).reset_index()
    tw = _weights(team_rows["gw"], target_gw, config.team_half_life) * team_rows["matches"]
    league_goals = float((team_rows["scored"] * tw / team_rows["matches"]).sum() / tw.sum())
    prior = config.team_prior_matches
    team_attack, team_defence = {}, {}
    for team, rows in team_rows.assign(w=tw).groupby("team_id"):
        weight = rows["w"].sum()
        team_attack[int(team)] = float(((rows["scored"] / rows["matches"] * rows["w"]).sum()
                                        + prior * league_goals) / (weight + prior))
        team_defence[int(team)] = float(((rows["conceded"] / rows["matches"] * rows["w"]).sum()
                                         + prior * league_goals) / (weight + prior))

    # Minutes: per team match, recent form matters most.
    mw = _weights(h["gw"], target_gw, config.minutes_half_life) * h["matches"]
    h["played"] = (h["minutes"] > 0).astype(float)
    h["sixty"] = (h["per_match_minutes"] >= 60).astype(float)
    k = config.minutes_prior_matches
    minutes = pd.DataFrame({
        "w": mw,
        "p_play": h["played"] * mw,
        "p60": h["sixty"] * mw,
        "emin": h["per_match_minutes"] * mw,
        "element": h["element"],
    }).groupby("element").sum()
    players = pd.DataFrame(index=minutes.index)
    players["p_play"] = (minutes["p_play"] + 0.35 * k) / (minutes["w"] + k)
    players["p60"] = (minutes["p60"] + 0.25 * k) / (minutes["w"] + k)
    players["emin"] = (minutes["emin"] + 25.0 * k) / (minutes["w"] + k)

    # Per-90 rates, shrunk towards the positional average.
    rw = _weights(h["gw"], target_gw, config.rate_half_life)
    weighted = h[list(RATE_COLUMNS)].mul(rw, axis=0)
    weighted["minutes"] = h["minutes"] * rw
    weighted["element"] = h["element"]
    sums = weighted.groupby("element").sum()
    latest = h.sort_values("gw").groupby("element").last()
    players["element_type"] = latest["element_type"].astype(int)
    players["team_id"] = latest["team_id"].astype(int)
    by_position = sums.join(players["element_type"]).groupby("element_type").sum()
    for column in RATE_COLUMNS:
        position_rate = (by_position[column] / by_position["minutes"].clip(lower=1)).to_dict()
        prior_rate = players["element_type"].map(position_rate).fillna(0.0)
        players[f"{column}_p90"] = 90 * ((sums[column] + prior_rate * config.rate_prior_minutes)
                                         / (sums["minutes"] + config.rate_prior_minutes))
    return ModelState(target_gw, players, team_attack, team_defence, league_goals, config)


def fixture_points(state: ModelState, element: int, opponent: int, is_home: bool,
                   availability: float = 1.0, element_type: Optional[int] = None,
                   team_id: Optional[int] = None) -> float:
    """Expected points for one player in one fixture."""
    cfg = state.config
    if element in state.players.index:
        row = state.players.loc[element]
        position = int(element_type or row["element_type"])
        team = int(team_id or row["team_id"])
        p_play, p60, emin = row["p_play"], row["p60"], row["emin"]
        rates = {c: row[f"{c}_p90"] for c in RATE_COLUMNS}
    else:
        # Unseen player: a cautious squad-player prior.
        position, team = int(element_type or 3), int(team_id or 0)
        p_play, p60, emin = 0.2, 0.1, 10.0
        same = state.players.loc[state.players["element_type"] == position]
        rates = {c: float(same[f"{c}_p90"].median()) if not same.empty else 0.0 for c in RATE_COLUMNS}
    p_play, p60, emin = p_play * availability, p60 * availability, emin * availability

    league = state.league_goals
    home = 1 + cfg.home_advantage if is_home else 1 - cfg.home_advantage
    away = 1 - cfg.home_advantage if is_home else 1 + cfg.home_advantage
    attack_scale = state.team_defence.get(opponent, league) / league * home
    conceded = (state.team_defence.get(team, league) * state.team_attack.get(opponent, league)
                / league * away)

    share = emin / 90.0
    points = p_play + p60  # 1 point for appearing, a second for 60+ minutes
    points += share * attack_scale * (rates["expected_goals"] * GOAL_POINTS[position]
                                      + rates["expected_assists"] * 3)
    points += p60 * CLEAN_SHEET_POINTS[position] * math.exp(-conceded)
    if position in (1, 2):
        points -= p60 * _poisson_floor_mean(conceded, 2)
    if position == 1:
        # Saves scale with how many shots the opponent creates.
        save_rate = rates["saves"] * conceded / max(state.team_defence.get(team, league), 0.1)
        points += p_play * _poisson_floor_mean(save_rate * min(1.0, emin / max(p_play, 1e-6) / 90), 3)
    else:
        played_minutes = emin / max(p_play, 1e-6)
        dc_mean = rates["defensive_contribution"] * played_minutes / 90
        points += p_play * 2 * _poisson_at_least(dc_mean, DC_THRESHOLD[position])
    points += share * (rates["bonus"] - rates["yellow_cards"] - 3 * rates["red_cards"])
    return max(0.0, points)


def predict(state: ModelState, players: Iterable[Tuple[int, int, int, float]],
            fixtures: List[dict], events: Iterable[int]) -> Dict[int, Dict[int, float]]:
    """Expected points per player per event.

    ``players`` yields (element, element_type, team_id, availability);
    ``fixtures`` are FPL fixture dicts with ``event``, ``team_h`` and ``team_a``.
    """
    events = list(events)
    by_team: Dict[Tuple[int, int], List[Tuple[int, bool]]] = {}
    for fixture in fixtures:
        if fixture.get("event") in events and not fixture.get("finished"):
            by_team.setdefault((fixture["team_h"], fixture["event"]), []).append((fixture["team_a"], True))
            by_team.setdefault((fixture["team_a"], fixture["event"]), []).append((fixture["team_h"], False))
    out: Dict[int, Dict[int, float]] = {}
    for element, element_type, team_id, availability in players:
        out[element] = {
            event: round(sum(fixture_points(state, element, opponent, home, availability,
                                            element_type, team_id)
                             for opponent, home in by_team.get((team_id, event), [])), 3)
            for event in events
        }
    return out


def history_from_event_live(bootstrap: Dict[str, Any], gameweek_feeds: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """Convert FPL ``event/{gw}/live`` payloads into model history rows."""
    meta = {e["id"]: (e.get("element_type"), e.get("team")) for e in bootstrap.get("elements", [])}
    rows = []
    for gw, feed in gameweek_feeds.items():
        for element in feed.get("elements", []):
            pid = element.get("id")
            if pid not in meta or not meta[pid][0] or not meta[pid][1]:
                continue
            stats = element.get("stats", {})
            row = {"element": pid, "gw": int(gw), "element_type": meta[pid][0], "team_id": meta[pid][1],
                   "matches": len(element.get("explain") or [])}
            for column in ("minutes", "starts", "expected_goals_conceded", *RATE_COLUMNS):
                try:
                    value = float(stats.get(column) or 0.0)
                except (TypeError, ValueError):
                    value = 0.0
                row[column] = value if math.isfinite(value) else 0.0
            rows.append(row)
    return pd.DataFrame(rows, columns=list(HISTORY_COLUMNS))


def load_event_history(fetch: Callable[[int], Dict[str, Any]], bootstrap: Dict[str, Any],
                       target_gw: int, cache_dir: Path, season: str) -> pd.DataFrame:
    """Load every finalized gameweek before ``target_gw``, caching each feed on disk."""
    feeds = {}
    for event in bootstrap.get("events", []):
        gw = event.get("id")
        if type(gw) is not int or gw >= target_gw or not (event.get("finished") and event.get("data_checked")):
            continue
        path = Path(cache_dir) / season / f"gw{gw:02}.json"
        if path.exists():
            feeds[gw] = json.loads(path.read_text(encoding="utf-8"))
            continue
        feed = fetch(gw)
        if not isinstance(feed, dict) or not isinstance(feed.get("elements"), list):
            raise ValueError(f"Unexpected event/{gw}/live payload")
        atomic_write_json(path, feed)
        feeds[gw] = feed
    return history_from_event_live(bootstrap, feeds)


def forecast_players(bootstrap: Dict[str, Any], history: pd.DataFrame, fixtures: List[dict],
                     target_gw: int, horizon: int,
                     config: ModelConfig = ModelConfig()) -> Dict[int, Dict[int, float]]:
    """Full-availability forecasts for every listed player over the horizon."""
    state = fit(history, target_gw, config)
    players = [(e["id"], e["element_type"], e["team"], 1.0) for e in bootstrap.get("elements", [])
               if e.get("element_type") in (1, 2, 3, 4) and e.get("team")]
    return predict(state, players, fixtures, range(target_gw, target_gw + horizon))
