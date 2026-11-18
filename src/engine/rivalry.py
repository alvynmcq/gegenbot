"""Head-to-head mini-league maths: what matters is your score minus a target rival's.

For player i with your multiplier m and the rival's multiplier e, the difference
contributes (m - e) * points. Its expectation only depends on your own picks, so
rival ownership cannot change expected advantage; it changes the spread. The
spread of a player's points is close to Poisson-like: a 2025-26 backtest of the
points model gave variance ~= 3.5 * expected points at every forecast level.

To maximise the chance of finishing ahead, one point of weekly relative spread is
worth about ``gap / (weeks_remaining * weekly_spread)`` points: positive when
behind (seek variance), negative when ahead (avoid it), and small early on.
"""

from math import sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

POINTS_VARIANCE_PER_XP = 3.5
MAX_RISK_PRICE = 1.0


def player_variance(expected_points: float) -> float:
    return POINTS_VARIANCE_PER_XP * max(0.0, expected_points)


def lineup_multipliers(picks: Iterable[Mapping]) -> Dict[int, float]:
    """Normalise one manager's picks: starters 1, captain 2, bench 0 (chips ignored)."""
    out: Dict[int, float] = {}
    for pick in picks:
        pid = pick.get("element")
        if pid is None:
            continue
        if (pick.get("position") or 0) > 11:
            out[int(pid)] = 0.0
        else:
            out[int(pid)] = 2.0 if pick.get("is_captain") else 1.0
    return out


def choose_target(standings: List[Mapping], my_entry_id: Optional[int]) -> Optional[Tuple[int, float]]:
    """(rival entry, gap) where gap = rival total - my total.

    Behind: the leader. Leading: the nearest chaser (gap is then negative).
    """
    me = next((r for r in standings if r.get("entry") == my_entry_id), None)
    others = [r for r in standings if r.get("entry") != my_entry_id]
    if me is None or not others:
        return None
    best = max(others, key=lambda r: (r.get("total", 0), -(r.get("rank") or 0)))
    return int(best["entry"]), float(best.get("total", 0) - me.get("total", 0))


def relative_sd(mine: Mapping[int, float], rival: Mapping[int, float], points: Mapping[int, float]) -> float:
    """Spread of (my points - rival points) for one gameweek, assuming independent players."""
    variance = 0.0
    for pid in set(mine) | set(rival):
        diff = mine.get(pid, 0.0) - rival.get(pid, 0.0)
        if diff:
            variance += diff * diff * player_variance(points.get(pid, 0.0))
    return sqrt(variance)


def risk_price(gap: float, weeks_remaining: int, weekly_sd: float) -> float:
    """Points of expectation one point of weekly relative spread is worth (+ = seek it)."""
    if weeks_remaining <= 0 or weekly_sd <= 0:
        return 0.0
    price = gap / (weeks_remaining * weekly_sd)
    return max(-MAX_RISK_PRICE, min(MAX_RISK_PRICE, price))
