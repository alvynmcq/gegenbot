"""Multi-week transfer planning as one MILP.

The single-week solver picks the best squad for the next deadline and leaves the
future to a small beam search. This model chooses the squad, lineup and captain
for every week of the planning horizon at once, so it can see moves that only
pay off later: banking a free transfer for a double move, buying a player a
week before his good run, or taking a hit that sets up the next fortnight.

Its week-1 move is offered to the rolling planner as one more candidate, and the
plan's later transfers are kept so the planner can score that exact path.

Simplifications, all conservative:
- prices stay fixed over the horizon; a player bought in the plan sells for what he cost;
- an original holding, once sold, is not bought back (its sale price would differ);
- no chips inside the horizon (the planner prices chips separately);
- the player pool is pruned to the current squad plus the strongest and cheapest options.
"""
import logging
import os
from collections import Counter

import pulp

from src.engine.optimizer import _solve_problem, get_solver

logger = logging.getLogger(__name__)

POOL_BEST = 20       # per position, by weighted horizon points
POOL_CHEAP = 6       # per position, cheapest enablers
MAX_FREE = 5
QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
STARTING = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def _pool(planner, current):
    by_position = {position: [] for position in QUOTA}
    for pid, row in planner.players.items():
        by_position[row["element_type"]].append(pid)
    chosen = set(current)
    for position, ids in by_position.items():
        horizon = sorted(ids, key=lambda pid: (-sum(planner.weights[w] * planner._points(pid, w)
                                                    for w in planner.weeks), planner.buy[pid], pid))
        cheap = sorted(ids, key=lambda pid: (planner.buy[pid], -planner._points(pid, planner.gameweek), pid))
        chosen.update(horizon[:POOL_BEST])
        chosen.update(cheap[:POOL_CHEAP])
    return sorted(chosen)


def solve_horizon(planner, bench_weight=0.1, already_paid=0, veto=(), rival_term=None, time_limit=None):
    """Return the best multi-week plan as a dict, or None if it cannot be solved.

    ``rival_term(starter_vars, captain_vars)`` optionally adds the head-to-head
    variance price to week 1, as in the single-week solver.
    """
    weeks = planner.weeks
    first = weeks[0]
    current = set(planner.initial.squad)
    pool = _pool(planner, current)
    vetoed = set(veto)
    position = {pid: planner.players[pid]["element_type"] for pid in pool}
    team = {pid: planner.players[pid]["team_id"] for pid in pool}
    # Tenths of £1m. Originals carry their sale price because they are never rebought.
    price = {pid: planner.sell.get(pid, planner.buy[pid]) if pid in current else planner.buy[pid]
             for pid in pool}
    budget = planner.initial.bank + sum(price[pid] for pid in current)
    points = {(w, pid): planner._points(pid, w) for w in weeks for pid in pool}

    prob = pulp.LpProblem("FPL_Horizon", pulp.LpMaximize)
    squad, start, captain, buy = {}, {}, {}, {}
    for w in weeks:
        for pid in pool:
            squad[w, pid] = pulp.LpVariable(f"s_{w}_{pid}", cat=pulp.LpBinary)
            start[w, pid] = pulp.LpVariable(f"x_{w}_{pid}", cat=pulp.LpBinary)
            captain[w, pid] = pulp.LpVariable(f"c_{w}_{pid}", cat=pulp.LpBinary)
            buy[w, pid] = pulp.LpVariable(f"b_{w}_{pid}", cat=pulp.LpBinary)
    used = {w: pulp.LpVariable(f"used_{w}", lowBound=0, cat=pulp.LpInteger) for w in weeks}
    paid = {w: pulp.LpVariable(f"paid_{w}", lowBound=0, cat=pulp.LpInteger) for w in weeks}
    free = {w: pulp.LpVariable(f"free_{w}", lowBound=0, upBound=MAX_FREE, cat=pulp.LpInteger)
            for w in weeks[1:]}
    free_end = pulp.LpVariable("free_end", lowBound=0, upBound=MAX_FREE, cat=pulp.LpInteger)
    free[first] = planner.initial.free

    for index, w in enumerate(weeks):
        previous = weeks[index - 1] if index else None
        prob += pulp.lpSum(squad[w, p] for p in pool) == 15
        for pos, quota in QUOTA.items():
            members = [p for p in pool if position[p] == pos]
            prob += pulp.lpSum(squad[w, p] for p in members) == quota
            low, high = STARTING[pos]
            prob += pulp.lpSum(start[w, p] for p in members) >= low
            prob += pulp.lpSum(start[w, p] for p in members) <= high
        prob += pulp.lpSum(start[w, p] for p in pool) == 11
        prob += pulp.lpSum(captain[w, p] for p in pool) == 1
        for club, members in _group(pool, team).items():
            prob += pulp.lpSum(squad[w, p] for p in members) <= 3
        prob += pulp.lpSum(price[p] * squad[w, p] for p in pool) <= budget
        for p in pool:
            prob += start[w, p] <= squad[w, p]
            prob += captain[w, p] <= start[w, p]
            if position[p] in (1, 2) and not planner.allow_defensive_captain:
                prob += captain[w, p] == 0
            held_before = (1 if p in current else 0) if previous is None else squad[previous, p]
            prob += buy[w, p] >= squad[w, p] - held_before
            if p in current and previous is not None:
                prob += squad[w, p] <= squad[previous, p]  # no buying back an original
            if p in vetoed and p not in current:
                prob += squad[w, p] == 0
        transfers = pulp.lpSum(buy[w, p] for p in pool)
        prob += transfers == used[w] + paid[w]
        prob += used[w] <= free[w]
        later = free[weeks[index + 1]] if index + 1 < len(weeks) else free_end
        prob += later <= free[w] - used[w] + 1
        cap = planner.hit_cap - (already_paid if w == first else 0)
        prob += 4 * paid[w] <= max(0, cap)

    objective = []
    for w in weeks:
        weight = planner.weights[w]
        for p in pool:
            value = points[w, p]
            if value:
                objective.append(weight * value * (start[w, p] + captain[w, p]
                                                   + bench_weight * (squad[w, p] - start[w, p])))
        objective.append(-4 * weight * paid[w])
    last = weeks[-1]
    if last < 38:
        objective.append(planner.weights[last] * planner.terminal_free_value * (free_end - 1))
    extra = 0
    if rival_term is not None:
        extra = rival_term({p: start[first, p] for p in pool}, {p: captain[first, p] for p in pool}, pool)
    prob += pulp.lpSum(objective) + extra

    limit = time_limit if time_limit is not None else int(os.getenv("MULTI_PERIOD_TIME_LIMIT", "30"))
    _solve_problem(prob, get_solver(time_limit=limit))
    if prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        logger.info("Multi-week plan found no solution (status %s).", pulp.LpStatus.get(prob.status))
        return None

    rounds = []
    held = set(current)
    for w in weeks:
        now = {p for p in pool if squad[w, p].value() > 0.5}
        lead = next(p for p in pool if captain[w, p].value() > 0.5)
        hits = 4 * round(paid[w].value() or 0)
        rounds.append({"gameweek": w, "in": sorted(now - held), "out": sorted(held - now),
                       "hits": hits, "captain": lead,
                       "squad": sorted(now)})
        held = now
    return {
        "objective": round(pulp.value(prob.objective), 2),
        "optimal": prob.sol_status == pulp.LpSolutionOptimal,
        "free_transfers_end": round(free_end.value() or 0),
        "pool_size": len(pool),
        "rounds": rounds,
    }


def _group(pool, team):
    groups = {}
    for p in pool:
        groups.setdefault(team[p], []).append(p)
    return groups


def pair_moves(out_ids, in_ids, players):
    """Pair sales with purchases position by position; None if positions do not match."""
    if Counter(players[p]["element_type"] for p in out_ids) != Counter(players[p]["element_type"] for p in in_ids):
        return None
    pairs = []
    remaining = sorted(in_ids)
    for outgoing in sorted(out_ids):
        incoming = next(p for p in remaining if players[p]["element_type"] == players[outgoing]["element_type"])
        remaining.remove(incoming)
        pairs.append((outgoing, incoming))
    return pairs


def add_horizon_candidate(optimizer, planner, result, *, current_squad_ids, bank_m, free_transfers,
                          selling_prices=None, eo_weights=None, already_paid=0, veto=()):
    """Solve the multi-week plan and expose its first move as a candidate.

    If the week-1 squad matches an existing candidate, the plan is attached to it
    instead of duplicating it. Returns the candidate that carries the plan, or None.
    """
    def rival_term(starters, captains, pool):
        if not optimizer.variance_weight:
            return 0
        index = {pid: i for i, pid in enumerate(optimizer.df["id"].tolist())}
        xps = optimizer.df["xp"].tolist()
        return optimizer._rival_variance_term(
            {index[p]: v for p, v in starters.items()}, {index[p]: v for p, v in captains.items()},
            [index[p] for p in pool], eo_weights or {}, xps)

    plan = solve_horizon(planner, bench_weight=optimizer.bench_weight, already_paid=already_paid,
                         veto=veto, rival_term=rival_term)
    if plan is None:
        return None
    week_one = set(plan["rounds"][0]["squad"])
    for candidate in result.candidates:
        if candidate.active_chip:
            continue
        if {p.id for p in candidate.starters + candidate.bench} == week_one:
            candidate.horizon_plan = plan
            return candidate

    current = set(current_squad_ids)
    sale = selling_prices or {}
    total = round(sum(sale.get(pid, optimizer.player_map[pid]["cost_m"]) for pid in current) + bank_m, 2)
    candidate = optimizer._solve_for_k_transfers(
        current, total, k_transfers=len(week_one - current), free_transfers=free_transfers,
        selling_prices=selling_prices, eo_weights=eo_weights, fixed_squad=week_one,
        rolling_bonus_xp=float(os.getenv("ROLLING_BONUS_XP", "1.5")))
    if candidate is None:
        logger.info("Multi-week plan's first move could not be rebuilt as a candidate.")
        return None
    hit = f", -{candidate.hit_cost} hit" if candidate.hit_cost else ""
    candidate.name = (f"Option {len(result.candidates) + 1}: Multi-Week Plan "
                      f"({candidate.transfers_count} transfers now{hit})")
    candidate.horizon_plan = plan
    result.candidates.append(candidate)
    return candidate
