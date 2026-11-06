"""Bounded rolling transfer search and chip counterfactuals."""
import os
from collections import Counter
from dataclasses import dataclass, replace
from math import sqrt

from src.chips import available_chips

RISK_WEIGHTS = {"DEFEND": 0.45, "NEUTRAL": 0.22, "CHASE": 0.10}
REBUILD_CHIPS = {"wildcard", "freehit"}


@dataclass(frozen=True)
class SquadState:
    squad: frozenset
    retained: frozenset  # original holdings still eligible for the original sale price
    bank: int  # tenths of £1m
    free: int


@dataclass(frozen=True)
class Round:
    gameweek: int
    before: SquadState
    after: SquadState
    starters: tuple
    bench: tuple
    captain: int
    mean: float
    deviation: float
    hits: int = 0
    chip: str = ""


class RollingPlanner:
    """Compare legal first moves using a small beam of plausible continuations."""

    def __init__(self, optimizer, gameweek, weeks, squad, bank, free, selling_prices=None,
                 max_hit_points=0, risk_mode="NEUTRAL", transfers_made=0, transfer_limit=None):
        self.players = optimizer.player_map
        self.gameweek = gameweek
        self.weeks = tuple(weeks)
        self.weights = dict(zip(self.weeks, optimizer.decay_weights))
        self.allow_defensive_captain = optimizer.allow_defensive_captain
        self.risk_weight = RISK_WEIGHTS.get(risk_mode, RISK_WEIGHTS["NEUTRAL"])
        self.hit_cap = max_hit_points
        self.terminal_free_value = max(0.0, float(os.getenv("ROLLING_BONUS_XP", "1.5")))
        self.transfers_made = transfers_made
        squad = frozenset(squad)
        self.initial = SquadState(squad, squad, round(bank * 10), free)
        self.chip_free = free if transfer_limit is None else transfer_limit
        self.buy = {pid: round(row["cost_m"] * 10) for pid, row in self.players.items()}
        self.sell = {pid: round(price * 10) for pid, price in (selling_prices or {}).items()}
        self.lineup_cache = {}
        self.pools = {}
        for week in self.weeks[1:]:
            later = [w for w in self.weeks if w >= week]
            self.pools[week] = {}
            for position in (1, 2, 3, 4):
                ids = [pid for pid, row in self.players.items() if row["element_type"] == position]
                best = sorted(ids, key=lambda pid: (-sum(self.weights[w] * self._points(pid, w)
                                                         for w in later), self.buy[pid], pid))
                cheap = sorted(ids, key=lambda pid: (self.buy[pid], -self._points(pid, week), pid))
                immediate = sorted(ids, key=lambda pid: (-self._points(pid, week), self.buy[pid], pid))
                self.pools[week][position] = tuple(dict.fromkeys(best[:5] + cheap[:3] + immediate[:4]))

    def _points(self, pid, week):
        row = self.players[pid]
        if week == self.gameweek:
            return row["discounted_xp"]
        return row["xp_by_gameweek"].get(week, 0.0)

    def _risk(self, week, starters, captain, bench=(), triple=False):
        """Moment-based downside proxy with shared club and individual appearance risk."""
        independent = 0.0
        clubs = Counter()
        appearances = Counter(starters + bench)
        appearances[captain] += 2 if triple else 1
        for pid, multiplier in appearances.items():
            row = self.players[pid]
            exposure = self._points(pid, week) * multiplier
            chance = row["injury_multiplier"] if week == self.gameweek else min(
                1.0, row["availability"] + 0.25 * (week - self.gameweek))
            # Keep the relative variance bounded for speculative low-minute picks.
            miss_variance = (1 - chance) / max(0.25, chance)
            role_variance = {"LOW": 0.25, "MEDIUM": 0.12}.get(row.get("minutes_reliability"), 0.04)
            independent += exposure ** 2 * (0.16 + miss_variance + role_variance)
            clubs[row["team_id"]] += exposure
        return sqrt(independent + sum((0.20 * exposure) ** 2 for exposure in clubs.values()))

    def _lineup(self, squad, week):
        key = (squad, week)
        if key in self.lineup_cache:
            return self.lineup_cache[key]
        positions = {position: [] for position in (1, 2, 3, 4)}
        for pid in sorted(squad, key=lambda pid: (-self._points(pid, week), pid)):
            positions[self.players[pid]["element_type"]].append(pid)
        best = None
        for defenders in (3, 4, 5):
            for forwards in (1, 2, 3):
                mids = 10 - defenders - forwards
                if not 2 <= mids <= 5:
                    continue
                starters = tuple(positions[1][:1] + positions[2][:defenders]
                                 + positions[3][:mids] + positions[4][:forwards])
                attackers = [pid for pid in starters if self.players[pid]["element_type"] in (3, 4)
                             and self._points(pid, week) > 0]
                captain_pool = starters if self.allow_defensive_captain else attackers or starters
                captain = max(captain_pool, key=lambda pid: (self._points(pid, week), -pid))
                mean = sum(self._points(pid, week) for pid in starters) + self._points(captain, week)
                risk = self._risk(week, starters, captain)
                value = mean - self.risk_weight * risk
                if best is None or value > best[0]:
                    best = (value, starters, tuple(sorted(squad.difference(starters))), captain, mean, risk)
        self.lineup_cache[key] = best
        return best

    def _transfer(self, state, outgoing, incoming):
        sale = self.sell.get(outgoing, self.buy[outgoing]) if outgoing in state.retained else self.buy[outgoing]
        bank = state.bank + sale - self.buy[incoming]
        if bank < 0:
            return None
        squad = (state.squad - {outgoing}) | {incoming}
        if max(Counter(self.players[pid]["team_id"] for pid in squad).values()) > 3:
            return None
        return SquadState(squad, state.retained - {outgoing}, bank, state.free)

    def _moves(self, state, week, origin):
        for outgoing in sorted(state.squad & origin.squad):
            position = self.players[outgoing]["element_type"]
            for incoming in self.pools[week][position]:
                if incoming in state.squad or incoming in origin.squad:
                    continue
                moved = self._transfer(state, outgoing, incoming)
                if moved:
                    yield moved

    def _round(self, week, before, squad, count=0, chip=""):
        _, starters, bench, captain, mean, risk = self._lineup(squad.squad, week)
        hits = 0 if chip in REBUILD_CHIPS else 4 * max(0, count - before.free)
        if chip == "freehit":
            after = before
        elif chip == "wildcard":
            after = squad
        else:
            after = replace(squad, free=min(5, max(0, before.free - count) + 1))
        return Round(week, before, after, starters, bench, captain, mean - hits, risk, hits, chip)

    def _rank(self, path):
        mean = sum(self.weights[step.gameweek] * step.mean for step in path)
        risk = sqrt(sum((self.weights[step.gameweek] * step.deviation) ** 2 for step in path))
        return mean - self.risk_weight * risk + self._saved_transfer_value(path)

    def _saved_transfer_value(self, path):
        final = path[-1]
        if final.gameweek >= 38:
            return 0.0
        return self.weights[final.gameweek] * self.terminal_free_value * max(0, final.after.free - 1)

    def _future(self, first, weeks):
        beam = [first]
        for week in weeks:
            options = []
            for path in beam:
                before = path[-1].after
                options.append(path + (self._round(week, before, before),))
                for count, states in self._transfer_layers(before, week):
                    options.extend(path + (self._round(week, before, state, count),) for state in states)
            options.sort(key=lambda path: (self._rank(path), path[-1].after.free, path[-1].after.bank), reverse=True)
            beam = []
            seen = set()
            for path in options:
                if path[-1].after not in seen:
                    beam.append(path)
                    seen.add(path[-1].after)
                if len(beam) == 3:
                    break
        return beam[0]

    def _transfer_layers(self, before, week):
        frontier = [before]
        for count in (1, 2):
            if 4 * max(0, count - before.free) > self.hit_cap:
                break
            states = {state: None for previous in frontier for state in self._moves(previous, week, before)}
            frontier = sorted(states, key=lambda state: (
                self._lineup(state.squad, week)[0], state.bank, tuple(sorted(state.squad))), reverse=True)[:4]
            yield count, frontier

    def plan(self, candidate):
        chip = candidate.active_chip or ""
        starters = tuple(p.id for p in candidate.starters)
        bench = tuple(p.id for p in candidate.bench)
        squad = frozenset(starters + bench)
        captain = candidate.captain.id
        current = SquadState(squad, self.initial.retained & squad,
                             round(candidate.bank_remaining_m * 10), self.initial.free)
        if chip == "freehit":
            current = self.initial
        if chip in REBUILD_CHIPS:
            after = replace(current, free=self.chip_free)
        else:
            after = replace(current, free=min(5, max(0, self.initial.free - candidate.transfers_count) + 1))
        deviation = self._risk(self.gameweek, starters, captain, bench if chip == "bboost" else (),
                               triple=chip == "3xc")
        first = Round(self.gameweek, self.initial, after, starters, bench, captain,
                      candidate.net_xp, deviation, candidate.hit_cost, chip)
        return self._future((first,), self.weeks[1:])

    def _rebuild(self, state, week, chip):
        # Greedy rebuild counterfactual, scored over the remaining weeks for Wildcard.
        later = [w for w in self.weeks if w >= week] if chip == "wildcard" else [week]

        def score(squad):
            return sum(self.weights[w] * self._lineup(squad, w)[0] for w in later)

        origin = state
        for _ in range(5):
            best = max(self._moves(state, week, origin),
                       key=lambda s: (score(s.squad), s.bank, tuple(sorted(s.squad))), default=state)
            if score(best.squad) <= score(state.squad):
                break
            state = best
        return state

    def wait_value(self, baseline, chip, inventory, target_gameweeks=()):
        """Best within-window improvement from using the same available chip later."""
        if not isinstance(inventory, list):
            return None, None
        expiry = max((c.get("stop_event", 38) for c in inventory
                      if c.get("name") == chip and c.get("status_for_entry") == "available"
                      and not c.get("played_by_entry")
                      and c.get("start_event", 1) <= self.gameweek <= c.get("stop_event", 38)), default=0)
        best, target = 0.0, None
        for index, step in enumerate(baseline[1:], start=1):
            week = step.gameweek
            if (week > expiry or chip not in available_chips(week, inventory)
                    or (target_gameweeks and week not in target_gameweeks)):
                continue
            if chip in REBUILD_CHIPS:
                rebuild = self._rebuild(step.before, week, chip)
                minimum = 2 if chip == "wildcard" else 1
                if len(rebuild.squad - step.before.squad) < minimum:
                    continue
                played = self._round(week, step.before, rebuild, chip=chip)
                path = self._future(baseline[:index] + (played,), self.weeks[index + 1:])
            else:
                bonus = (sum(self._points(pid, week) for pid in step.bench) if chip == "bboost"
                         else self._points(step.captain, week))
                risk = self._risk(week, step.starters, step.captain,
                                  step.bench if chip == "bboost" else (), triple=chip == "3xc")
                path = baseline[:index] + (replace(step, mean=step.mean + bonus, deviation=risk),) + baseline[index + 1:]
            gain = self._rank(path) - self._rank(baseline)
            if gain > best:
                best, target = gain, week
        return round(best, 2), target

    def summary(self, path):
        return {
            "discounted_net_xp": round(sum(self.weights[step.gameweek] * step.mean for step in path), 2),
            "downside_sd": round(sqrt(sum((self.weights[step.gameweek] * step.deviation) ** 2 for step in path)), 2),
            "risk_weight": self.risk_weight,
            "terminal_transfer_value": round(self._saved_transfer_value(path), 2),
            "rounds": [{
                "gameweek": step.gameweek,
                "chip": step.chip or None,
                "out": sorted(step.before.squad - set(step.starters + step.bench)),
                "in": sorted(set(step.starters + step.bench) - step.before.squad),
                "net_xp": round(step.mean, 2),
                "hit_points": step.hits,
                "bank_next_m": step.after.bank / 10,
                "free_transfers_next": step.after.free,
                "captain_id": step.captain,
            } for step in path],
        }

    def annotate(self, result, inventory, offer_chips, forced_chip=None, veto=()):
        """Score first moves, then compare spending each eligible chip with waiting."""
        already_paid = 4 * max(0, self.transfers_made - self.chip_free)
        candidates = [
            candidate for candidate in result.candidates
            if (not candidate.transfers_count or already_paid + candidate.hit_cost <= self.hit_cap)
            and all(move.player_in.id not in veto for move in candidate.transfers)
        ]
        if not candidates:
            return
        result.candidates = candidates
        paths = [self.plan(candidate) for candidate in candidates]
        for candidate, path in zip(candidates, paths):
            self.record(candidate, path)
        baseline = max(paths, key=self._rank)
        chip_result = result.chip_evaluation
        if chip_result is None:
            return
        qualified = []
        for chip, evaluation in chip_result.evaluations.items():
            candidate = evaluation.squad_candidate
            if candidate is None:
                continue
            forced = chip == forced_chip
            eligible_now = (
                (chip != "freehit" or not self.transfers_made)
                and (chip != "wildcard" or candidate.transfers_count + self.transfers_made >= 2)
                and (chip != "freehit" or candidate.transfers_count > 0)
                and (chip in REBUILD_CHIPS or not candidate.transfers_count
                     or already_paid + candidate.hit_cost <= self.hit_cap)
                and all(move.player_in.id not in veto for move in candidate.transfers)
            )
            path = self.plan(candidate)
            now = round(self._rank(path) - self._rank(baseline), 2)
            if chip == "wildcard":
                allowed = not evaluation.target_gameweeks or self.gameweek in evaluation.target_gameweeks
                evaluation.threshold_met = allowed and now >= evaluation.threshold
                evaluation.reason = (f"Rolling Wildcard gain {now:+.2f} vs {evaluation.threshold:.2f} threshold."
                                     if allowed else "Outside the configured Wildcard target weeks.")
            later, week = self.wait_value(baseline, chip, inventory, evaluation.target_gameweeks)
            margin = round(now - later, 2) if later is not None else None
            evaluation.opportunity = {
                "play_now_gain": now,
                "best_later_gain": later,
                "best_later_gameweek": week,
                "play_now_edge": margin,
                "eligible_now": eligible_now,
            }
            self.record(candidate, path, opportunity_cost=later or 0.0)
            if (eligible_now and evaluation.threshold_met and margin is not None and margin > 0):
                qualified.append((margin, chip, candidate))
            if forced and chip not in REBUILD_CHIPS and eligible_now:
                chosen = candidate.model_copy(deep=True)
                chosen.name = f"Forced {evaluation.display_name}"
                result.candidates = [chosen]

        if qualified:
            _, chip, candidate = max(qualified, key=lambda item: item[:2])
            chip_result.recommended_chip = chip
            chip_result.recommendation_reason = "Playing now beats the best eligible later use in the forecast horizon."
            if offer_chips:
                chosen = candidate.model_copy(deep=True)
                chosen.name = f"Option {len(candidates) + 1}: {chip_result.evaluations[chip].display_name}"
                result.candidates.append(chosen)
        else:
            chip_result.recommended_chip = None
            chip_result.recommendation_reason = "No automatic chip clears both its threshold and the value of waiting."

    def record(self, candidate, path=None, opportunity_cost=0.0):
        path = path if path is not None else self.plan(candidate)
        candidate.rolling_plan = self.summary(path)
        candidate.rolling_plan["chip_opportunity_cost"] = opportunity_cost
        candidate.strategic_value_score = round(self._rank(path) - opportunity_cost, 2)
