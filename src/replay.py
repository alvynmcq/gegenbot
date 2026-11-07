"""Freeze pre-deadline decisions and score them later, without a live account."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
from math import isfinite, sqrt
from pathlib import Path
from random import Random
import re
from statistics import mean

import requests

from src.data_fetcher import forecast_season
from src.state import atomic_write_json, state_lock

API = "https://fantasy.premierleague.com/api"
VERSION = 1
FORECAST_MODELS = ("gegenbot", "official_ep_next", "season_ppg")


def _timestamp(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Replay timestamps must include a timezone")
    return timestamp


def _key(record):
    season, week = record["season"], record["gameweek"]
    if (not re.fullmatch(r"\d{4}-\d{4}", season) or int(season[:4]) + 1 != int(season[5:])
            or type(week) is not int or not 1 <= week <= 38):
        raise ValueError("Invalid replay season or gameweek")
    return season, week


def _path(root, key):
    season, week = key
    return Path(root) / season / f"gw{week:02}.json"


def _number(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (ValueError, TypeError):
        return None


def freeze_forecasts(players, bootstrap):
    """Freeze the three pre-deadline estimates on the same player IDs."""
    official = {element["id"]: _number(element.get("ep_next")) for element in bootstrap["elements"]}
    forecasts = {}
    for row in players.itertuples(index=False):
        forecasts[str(row.id)] = {
            "gegenbot": _number(row.xp),
            "official_ep_next": official.get(row.id),
            "season_ppg": round(row.points_per_game * row.fixtures_in_gw, 2),
        }
    return forecasts


def freeze_pick(candidate, previous_hit):
    chip = candidate.active_chip
    return {
        "starters": [[p.id, p.element_type] for p in candidate.starters],
        "bench": [[p.id, p.element_type] for p in candidate.bench],
        "captain": candidate.captain.id,
        "vice": candidate.vice_captain.id,
        "chip": chip,
        "hit": 0 if chip in ("wildcard", "freehit") else previous_hit + candidate.hit_cost,
        "projected_net_xp": round(candidate.net_xp - (0 if chip in ("wildcard", "freehit") else previous_hit), 2),
    }


def capture_snapshot(root, *, season, gameweek, deadline, players, bootstrap, optimization,
                     decision=None, model=None, previous_hit=0, forced_chip=None, veto_count=0,
                     captured_at=None, provenance="gegenbot"):
    """Keep the first frozen decision for a season and gameweek."""
    from src.engine.optimizer import optimizer_choice_index

    captured_at = captured_at or datetime.now(timezone.utc)
    if captured_at >= _timestamp(deadline):
        return None
    key = _key({"season": season, "gameweek": gameweek})
    path = _path(root, key)
    with state_lock(path):
        if path.exists():
            return path
        options = optimization.candidates
        baseline = options[optimizer_choice_index(optimization)]
        immediate = max(options, key=lambda c: c.net_xp)
        policies = {"optimizer_only": freeze_pick(baseline, previous_hit),
                    "immediate_xp": freeze_pick(immediate, previous_hit)}
        roll = next((c for c in options if not c.transfers_count and not c.active_chip), None)
        if roll is not None:
            policies["roll"] = freeze_pick(roll, previous_hit)
        source = decision.source if decision else None
        if source == "LLM_DIRECTOR" and not forced_chip:
            policies["ai_director"] = freeze_pick(decision.selected_candidate, previous_hit)
        snapshot = {
            "version": VERSION,
            "season": season,
            "gameweek": gameweek,
            "deadline": deadline,
            "captured_at": captured_at.isoformat(),
            "provenance": provenance,
            "forecasts": freeze_forecasts(players, bootstrap),
            "policies": policies,
            "director_source": source,
            "director_model": model if source == "LLM_DIRECTOR" else None,
            "forced_chip": forced_chip,
            "veto_count": veto_count,
        }
        if datetime.now(timezone.utc) >= _timestamp(deadline) and provenance == "gegenbot":
            return None
        atomic_write_json(path, snapshot)
        return path


def _valid_formation(players):
    positions = Counter(position for _, position in players)
    return (positions[1] == 1 and 3 <= positions[2] <= 5 and 2 <= positions[3] <= 5
            and 1 <= positions[4] <= 3)


def score_pick(pick, actuals):
    """Score a legal selection using finalized official gameweek totals."""
    starters = [tuple(player) for player in pick["starters"]]
    bench = [tuple(player) for player in pick["bench"]]
    squad = starters + bench
    ids = {pid for pid, _ in squad}
    captain, vice = pick["captain"], pick["vice"]
    starter_ids = {pid for pid, _ in starters}
    chip, hits = pick["chip"], pick["hit"]
    if (len(starters) != 11 or len(bench) != 4 or len(ids) != 15 or not _valid_formation(starters)
            or Counter(pos for _, pos in squad) != {1: 2, 2: 5, 3: 5, 4: 3}
            or captain == vice or not {captain, vice} <= starter_ids
            or chip not in (None, "3xc", "bboost", "wildcard", "freehit")
            or type(hits) is not int or hits < 0 or hits % 4):
        raise ValueError("Invalid lineup in replay snapshot")
    if not ids <= actuals.keys():
        return None

    def played(pid):
        row = actuals[pid]
        return row["minutes"] > 0 or row.get("cards", 0) > 0

    if chip == "bboost":
        scoring = squad
    else:
        scoring = starters.copy()
        reserve_keeper = next(p for p in bench if p[1] == 1)
        keeper_index = next(i for i, p in enumerate(scoring) if p[1] == 1)
        if not played(scoring[keeper_index][0]) and played(reserve_keeper[0]):
            scoring[keeper_index] = reserve_keeper
        missing = [p for p in scoring if p[1] != 1 and not played(p[0])]
        reserves = [p for p in bench if p[1] != 1 and played(p[0])]
        # Try legal replacements in bench priority order, allowing a defender to
        # cover a missing defender when using an earlier attacker elsewhere.
        best = ()
        best_priority = ()
        for size in range(1, min(len(missing), len(reserves)) + 1):
            for added in combinations(reserves, size):
                priority = tuple(int(player in added) for player in reserves)
                if priority <= best_priority:
                    continue
                for removed in combinations(missing, size):
                    lineup = [p for p in scoring if p not in removed] + list(added)
                    if _valid_formation(lineup):
                        best, best_priority = lineup, priority
                        break
        if best:
            scoring = best

    points = sum(actuals[pid]["points"] for pid, _ in scoring if played(pid))
    armband = captain if actuals[captain]["minutes"] > 0 else vice
    if actuals[armband]["minutes"] > 0:
        points += actuals[armband]["points"] * (2 if chip == "3xc" else 1)
    return points - hits


def _metrics(rows):
    errors = [predicted - actual for predicted, actual in rows]
    bands = defaultdict(list)
    for predicted, actual in rows:
        band = "<2" if predicted < 2 else "2–<4" if predicted < 4 else "4–<6" if predicted < 6 else "6+"
        bands[band].append((predicted, actual))
    return {
        "samples": len(rows),
        "mae": round(mean(map(abs, errors)), 3),
        "rmse": round(sqrt(mean(error ** 2 for error in errors)), 3),
        "bias": round(mean(errors), 3),
        "calibration": {
            band: {"samples": len(pairs), "predicted": round(mean(p for p, _ in pairs), 3),
                   "actual": round(mean(a for _, a in pairs), 3)}
            for band, pairs in sorted(bands.items())
        },
    }


def _paired(deltas):
    if not deltas:
        return {"gameweeks": 0, "mean_delta": None, "bootstrap_95": None}
    interval = None
    if len(deltas) >= 8:
        rng = Random(0)
        draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        interval = [round(draws[50], 3), round(draws[1949], 3)]
    return {
        "gameweeks": len(deltas), "mean_delta": round(mean(deltas), 3),
        "total_delta": sum(deltas), "wins": sum(d > 0 for d in deltas),
        "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
        "bootstrap_95": interval,
    }


def _validate_snapshot(snapshot):
    key = _key(snapshot)
    if snapshot["version"] != VERSION or _timestamp(snapshot["captured_at"]) >= _timestamp(snapshot["deadline"]):
        raise ValueError(f"Invalid pre-deadline snapshot for {key}")
    return key


def score_replay(snapshots, outcomes):
    forecasts = defaultdict(list)
    coverage = Counter()
    reports, seen, ai_deltas, roll_deltas, immediate_deltas = [], set(), [], [], []
    by_model = defaultdict(list)
    for snapshot in sorted(snapshots, key=_key):
        key = _validate_snapshot(snapshot)
        if key in seen:
            raise ValueError(f"Duplicate snapshot for {key}")
        seen.add(key)
        outcome = outcomes.get(key)
        if outcome is None:
            reports.append({"season": key[0], "gameweek": key[1], "status": "missing_outcome"})
            continue
        official_deadline = _timestamp(outcome["deadline"])
        if (_key(outcome) != key or outcome["version"] != VERSION or outcome["finalized"] is not True
                or _timestamp(outcome["collected_at"]) <= official_deadline
                or _timestamp(snapshot["captured_at"]) >= official_deadline):
            raise ValueError(f"Invalid finalized outcome for {key}")
        actuals = {int(pid): row for pid, row in outcome["players"].items()}
        if any(type(row["minutes"]) is not int or row["minutes"] < 0
               or type(row["points"]) is not int for row in actuals.values()):
            raise ValueError(f"Invalid player outcomes for {key}")
        for pid, predictions in snapshot["forecasts"].items():
            for model in FORECAST_MODELS:
                if _number(predictions.get(model)) is not None and int(pid) in actuals:
                    coverage[model] += 1
            values = [_number(predictions.get(model)) for model in FORECAST_MODELS]
            if None in values or int(pid) not in actuals:
                continue
            for model, prediction in zip(FORECAST_MODELS, values):
                forecasts[model].append((prediction, actuals[int(pid)]["points"]))
        scores = {name: score_pick(pick, actuals) for name, pick in snapshot["policies"].items()}
        ai = scores.get("ai_director")
        baseline = scores.get("optimizer_only")
        roll = scores.get("roll")
        immediate = scores.get("immediate_xp")
        row = {"season": key[0], "gameweek": key[1], "status": "scored",
               "provenance": snapshot.get("provenance"), "scores": scores,
               "captured_at": snapshot["captured_at"], "deadline": outcome["deadline"],
               "director_source": snapshot.get("director_source"), "veto_count": snapshot.get("veto_count", 0)}
        if (ai is not None and baseline is not None and snapshot.get("director_source") == "LLM_DIRECTOR"
                and not snapshot.get("forced_chip")):
            row["ai_delta"] = ai - baseline
            ai_deltas.append(ai - baseline)
            by_model[snapshot.get("director_model") or "unknown"].append(ai - baseline)
        if baseline is not None and roll is not None:
            roll_deltas.append(baseline - roll)
        if baseline is not None and immediate is not None:
            immediate_deltas.append(baseline - immediate)
        reports.append(row)
    return {
        "rounds": reports,
        "forecast_metrics_common_players": {model: _metrics(rows) for model, rows in forecasts.items()},
        "forecast_coverage": dict(coverage),
        "optimizer_vs_roll": _paired(roll_deltas),
        "optimizer_vs_immediate_xp": _paired(immediate_deltas),
        "ai_director_vs_optimizer": _paired(ai_deltas),
        "director_models": {model: _paired(deltas) for model, deltas in sorted(by_model.items())},
    }


def _public_json(endpoint):
    response = requests.get(f"{API}/{endpoint}/", timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_outcomes(snapshots, destination):
    bootstrap = _public_json("bootstrap-static")
    season = forecast_season(bootstrap["events"])
    finalized = {event["id"]: event["deadline_time"] for event in bootstrap["events"]
                 if event.get("finished") and event.get("data_checked")}
    count = 0
    for snapshot in snapshots:
        key = _validate_snapshot(snapshot)
        if key[0] != season or key[1] not in finalized:
            continue
        deadline = finalized[key[1]]
        if _timestamp(snapshot["captured_at"]) >= _timestamp(deadline):
            raise ValueError(f"Snapshot is later than the official deadline for {key}")
        path = _path(destination, key)
        if path.exists():
            continue
        elements = _public_json(f"event/{key[1]}/live")["elements"]
        players = {str(row["id"]): {
            "minutes": row["stats"]["minutes"],
            "points": row["stats"]["total_points"],
            "cards": row["stats"].get("yellow_cards", 0) + row["stats"].get("red_cards", 0),
        } for row in elements}
        atomic_write_json(path, {
            "version": VERSION, "season": season, "gameweek": key[1], "finalized": True,
            "deadline": deadline, "collected_at": datetime.now(timezone.utc).isoformat(),
            "players": players,
        })
        count += 1
    return count


def reconstruct_snapshot(bundle, destination):
    """Run a historical optimizer against an explicitly supplied archived starting state."""
    from src.engine.metrics import calculate_player_metrics, verified_fixture_schedule
    from src.engine.optimizer import FPLOptimizer
    from src.engine.strategy import RollingPlanner

    bootstrap, fixtures = bundle["bootstrap"], bundle["fixtures"]
    gameweek = bundle["gameweek"]
    captured_at = _timestamp(bundle["captured_at"])
    events = bootstrap["events"]
    season = forecast_season(events)
    target = next(event for event in events if event["id"] == gameweek)
    deadline = _timestamp(target["deadline_time"])
    if (not target.get("is_next") or target.get("finished") or captured_at >= deadline
            or season is None or not bundle.get("provenance")):
        raise ValueError("Historical bundle must contain a dated pre-deadline snapshot for the next gameweek")
    team_ids = {team["id"] for team in bootstrap["teams"]}
    if not verified_fixture_schedule(fixtures, team_ids, gameweek):
        raise ValueError("Historical bundle requires a complete archived fixture schedule")
    if any(f.get("event") is not None and f["event"] >= gameweek
           and (f.get("started") or f.get("finished") or f.get("team_h_score") is not None
                or f.get("team_a_score") is not None) for f in fixtures):
        raise ValueError("Historical fixture snapshot contains target or future results")

    players = calculate_player_metrics(bootstrap, fixtures, current_event=gameweek,
                                       odds_data={}, schedule_complete=True)
    optimizer = FPLOptimizer(players)
    squad = bundle["squad"]
    bank = float(bundle["bank_m"])
    free = bundle["free_transfers"]
    prices = {int(pid): float(value) for pid, value in bundle["selling_prices"].items()}
    if (len(squad) != 15 or len(set(squad)) != 15 or set(prices) != set(squad)
            or not set(squad) <= optimizer.player_map.keys()
            or not isfinite(bank) or bank < 0 or any(not isfinite(p) or p <= 0 for p in prices.values())
            or type(free) is not int or not 0 <= free <= 5):
        raise ValueError("Historical bundle needs 15 players, their sale prices, bank, and free transfers")
    hit_cap = bundle.get("max_hit_points", 0)
    if type(hit_cap) is not int or hit_cap < 0:
        raise ValueError("Invalid historical hit cap")
    result = optimizer.optimize(squad, bank_m=bank, free_transfers=free, selling_prices=prices,
                                current_gw=gameweek, evaluate_chips=False, include_chip_candidates=False)
    result.candidates = [c for c in result.candidates if c.hit_cost <= hit_cap]
    scheduled = {fixture.get("event") for fixture in fixtures if not fixture.get("finished")}
    weeks = [gameweek]
    for week in range(gameweek + 1, gameweek + min(4, optimizer.horizon_length)):
        if week not in scheduled:
            break
        weeks.append(week)
    if len(weeks) > 1:
        planner = RollingPlanner(optimizer, gameweek, weeks, squad, bank, free,
                                 selling_prices=prices, max_hit_points=hit_cap)
        planner.annotate(result, inventory=[], offer_chips=False)
    return capture_snapshot(
        destination, season=season, gameweek=gameweek, deadline=target["deadline_time"],
        players=players, bootstrap=bootstrap, optimization=result,
        captured_at=captured_at, provenance=f"reconstructed: {bundle['provenance']}",
    )


def _read_directory(path):
    return [json.loads(file.read_text()) for file in sorted(path.glob("*/*.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("fetch-outcomes", "score"):
        sub = commands.add_parser(command)
        sub.add_argument("snapshots", type=Path)
        sub.add_argument("outcomes", type=Path)
        if command == "score":
            sub.add_argument("--output", type=Path)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("bundle", type=Path)
    reconstruct.add_argument("snapshots", type=Path)
    args = parser.parse_args()
    if args.command == "reconstruct":
        print(reconstruct_snapshot(json.loads(args.bundle.read_text()), args.snapshots))
        return
    snapshots = _read_directory(args.snapshots)
    if not snapshots:
        parser.error("No replay snapshots found")
    if args.command == "fetch-outcomes":
        print(f"Wrote {fetch_outcomes(snapshots, args.outcomes)} finalized outcomes.")
        return
    outcomes = {}
    for row in _read_directory(args.outcomes):
        key = _key(row)
        if key in outcomes:
            parser.error(f"Duplicate outcomes for {key}")
        outcomes[key] = row
    report = score_replay(snapshots, outcomes)
    if args.output:
        atomic_write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
