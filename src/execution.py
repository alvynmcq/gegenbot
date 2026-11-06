"""Durable live execution with reconciliation after uncertain API responses."""
import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path

from src.state import atomic_write_json
from src.chips import available_chips

logger = logging.getLogger(__name__)


class ExecutionBlocked(RuntimeError):
    """A safety check failed with a message safe to display."""


def execution_path(team_id: int, gameweek: int, deadline: str) -> Path:
    directory = Path(os.getenv("STATE_DIR", "data")) / "executions"
    prefix = f"{team_id}-{gameweek}-"
    legacy_key = hashlib.sha256(deadline.encode()).hexdigest()[:12]
    legacy = directory / f"{prefix}{legacy_key}.json"
    if legacy.exists() or not deadline:
        return legacy
    scheduled = datetime.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    season = scheduled.year - (scheduled.month < 7)
    return directory / f"{prefix}season-{season}.json"


def build_plan(candidate, team_id: int, gameweek: int, result: dict, baseline=None) -> dict:
    chip = candidate.active_chip
    tx_chip = chip if chip in ("wildcard", "freehit") else None
    transfers = None
    if candidate.transfers_count or tx_chip:
        transfers = {
            "chips": tx_chip, "chip": tx_chip, "entry": team_id, "event": gameweek,
            "transfers": [{
                "element_in": t.player_in.id, "element_out": t.player_out.id,
                "purchase_price": round(t.player_in.cost_m * 10),
                "selling_price": round((t.player_out.selling_price_m
                    if t.player_out.selling_price_m is not None else t.player_out.cost_m) * 10),
            } for t in candidate.transfers],
        }
    starters = sorted(candidate.starters, key=lambda p: p.element_type)
    picks = [{"element": p.id, "position": index,
              "is_captain": p.is_captain if index <= 11 else False,
              "is_vice_captain": p.is_vice_captain if index <= 11 else False}
             for index, p in enumerate(starters + candidate.bench, 1)]
    return {"stage": "ready", "team_id": team_id, "gameweek": gameweek,
            "transfers": transfers, "baseline": baseline,
            "lineup": {"chip": chip if chip in ("bboost", "3xc") else None, "picks": picks},
            "result": result}


def _chips_with_status(live, gameweek, *statuses):
    return {c["name"] for c in live["chips"]
            if c["status_for_entry"] in statuses
            and c.get("start_event", 1) <= gameweek <= c.get("stop_event", 38)}


def _chip_confirmed(client, plan, live, chip):
    active = _chips_with_status(live, plan["gameweek"], "active")
    if _chips_with_status(live, plan["gameweek"], "pending"):
        return False
    if not chip or active:
        return active == ({chip} if chip else set())
    history = client.get_entry_history(plan["team_id"])
    return any(c.get("name") == chip and c.get("event") == plan["gameweek"]
               for c in history.get("chips", []))


def _nonnegative_int(value):
    if type(value) is not int or value < 0:
        raise ExecutionBlocked("Invalid FPL numeric field")
    return value


def _squad_ids(picks):
    ids = {_nonnegative_int(p["element"]) for p in picks}
    if len(picks) != 15 or len(ids) != 15 or 0 in ids:
        raise ExecutionBlocked("Expected 15 distinct squad players")
    return ids


def planning_snapshot(live):
    """Capture the authenticated inputs used to choose transfers."""
    picks = live["picks"]
    _squad_ids(picks)
    transfers = live["transfers"]
    limit = transfers["limit"]
    return {
        "selling_prices": {str(p["element"]): _nonnegative_int(p["selling_price"]) for p in picks},
        "bank": _nonnegative_int(transfers["bank"]),
        "limit": None if limit is None else _nonnegative_int(limit),
        "made": _nonnegative_int(transfers["made"]),
    }


def _picks_signature(picks):
    return sorted((p["element"], p["position"], p["is_captain"],
                   p["is_vice_captain"]) for p in picks)


def _validate_lineup(picks):
    _squad_ids(picks)
    if sorted(_nonnegative_int(p["position"]) for p in picks) != list(range(1, 16)):
        raise ExecutionBlocked("Invalid lineup positions")
    captain = [p for p in picks if p["is_captain"]]
    vice = [p for p in picks if p["is_vice_captain"]]
    if (len(captain) != 1 or len(vice) != 1 or captain == vice
            or captain[0]["position"] > 11 or vice[0]["position"] > 11):
        raise ExecutionBlocked("Invalid captain or vice-captain")


def _fresh_deadline(client, plan):
    bootstrap = client.get_bootstrap_static(force_refresh=True)
    event = next(e for e in bootstrap["events"] if e["id"] == plan["gameweek"])
    deadline = datetime.datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    planned = plan["result"].get("deadline_time")
    if planned and deadline != datetime.datetime.fromisoformat(planned.replace("Z", "+00:00")):
        raise ExecutionBlocked("FPL deadline changed since planning")
    if event.get("finished") or deadline.tzinfo is None:
        raise ExecutionBlocked("Invalid execution deadline")
    return bootstrap, deadline


def _check_deadline(deadline):
    margin = max(1, int(os.getenv("FPL_DEADLINE_MARGIN_MINUTES", "5")))
    if datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=margin) >= deadline:
        raise ExecutionBlocked("Execution deadline safety window reached")


def _validate_transfers(plan, live, bootstrap, desired, chip):
    snapshot = planning_snapshot(live)
    if snapshot != plan.get("baseline"):
        raise ExecutionBlocked("Squad or finances changed since planning")
    transfers = plan["transfers"]
    if (transfers["entry"] != plan["team_id"] or transfers["event"] != plan["gameweek"]
            or transfers["chips"] != chip):
        raise ExecutionBlocked("Transfer payload does not match plan")
    moves = transfers["transfers"]
    actual = _squad_ids(live["picks"])
    outgoing = {t["element_out"] for t in moves}
    incoming = {t["element_in"] for t in moves}
    if (len(outgoing) != len(moves) or len(incoming) != len(moves)
            or not outgoing <= actual or incoming & actual or (actual - outgoing) | incoming != desired):
        raise ExecutionBlocked("Transfers do not produce the planned squad")
    if chip is None:
        limit = snapshot["limit"]
        if limit is None:
            raise ExecutionBlocked("Free-transfer allowance unavailable")
        hits = max(0, snapshot["made"] + len(moves) - limit) * 4
        cap = _nonnegative_int(int(os.getenv("FPL_MAX_HIT_POINTS", "0")))
        if hits > cap:
            raise ExecutionBlocked("Gameweek points-hit cap exceeded")
    prices = {p["id"]: p["now_cost"] for p in bootstrap["elements"]}
    bank = snapshot["bank"]
    for move in moves:
        sell = snapshot["selling_prices"][str(move["element_out"])]
        buy = _nonnegative_int(prices[move["element_in"]])
        if sell != move["selling_price"] or buy != move["purchase_price"]:
            raise ExecutionBlocked("Transfer price changed since planning")
        bank += sell - buy
    if bank < 0:
        raise ExecutionBlocked("Transfer budget exceeded")


def execute_plan(client, path: Path, new_plan=None) -> dict:
    """Resume the persisted plan; never replay a POST with an uncertain outcome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = json.loads(path.read_text()) if path.exists() else new_plan
        if plan is None:
            raise ValueError("No execution plan available")
        result = plan["result"]
        if plan["stage"] == "complete":
            return result
        try:
            if not client.ensure_authenticated():
                raise ExecutionBlocked("Authentication failed before execution")
            team_id = plan["team_id"]
            lineup = plan["lineup"]
            _validate_lineup(lineup["picks"])
            desired = _squad_ids(lineup["picks"])
            tx = plan["transfers"]
            tx_chip = tx["chip"] if tx is not None else None
            lineup_chip = lineup["chip"]
            if (tx_chip not in (None, "wildcard", "freehit")
                    or lineup_chip not in (None, "bboost", "3xc") or (tx_chip and lineup_chip)):
                raise ExecutionBlocked("Incompatible chip payloads")

            if plan["stage"] == "ready" and tx is not None:
                bootstrap, deadline = _fresh_deadline(client, plan)
                live = client.get_my_team(team_id)
                activation = tx_chip or lineup_chip
                if (_chips_with_status(live, plan["gameweek"], "active", "pending")
                        or activation and activation not in available_chips(plan["gameweek"], live["chips"])):
                    raise ExecutionBlocked("Planned chip cannot be applied to the live squad")
                _validate_transfers(plan, live, bootstrap, desired, tx_chip)
                _check_deadline(deadline)
                plan["stage"] = "transfers_pending"
                atomic_write_json(path, plan)
                client.post_transfers(tx)

            if plan["stage"] == "transfers_pending":
                live = client.get_my_team(team_id)
                if (_squad_ids(live["picks"]) != desired
                        or not _chip_confirmed(client, plan, live, tx_chip)):
                    raise ExecutionBlocked("Transfer outcome unconfirmed")
                plan["stage"] = "transfers_done"
                atomic_write_json(path, plan)

            if plan["stage"] in ("ready", "transfers_done"):
                _, deadline = _fresh_deadline(client, plan)
                live = client.get_my_team(team_id)
                if (_squad_ids(live["picks"]) != desired
                        or not _chip_confirmed(client, plan, live, tx_chip)
                        or lineup_chip and lineup_chip not in available_chips(plan["gameweek"], live["chips"])):
                    raise ExecutionBlocked("Live squad or chip differs from the lineup plan")
                _check_deadline(deadline)
                plan["stage"] = "lineup_pending"
                atomic_write_json(path, plan)
                client.post_lineup(team_id, lineup)

            if plan["stage"] == "lineup_pending":
                live = client.get_my_team(team_id)
                if (_picks_signature(live["picks"]) != _picks_signature(lineup["picks"])
                        or not _chip_confirmed(client, plan, live, lineup_chip or tx_chip)):
                    raise ExecutionBlocked("Lineup outcome unconfirmed")
                plan["stage"] = "complete"

            if plan["stage"] != "complete":
                raise ExecutionBlocked("Unknown execution stage")
            result.update(status="success", mode="live_execution")
            result.pop("execution_error", None)
        except Exception as exc:
            # Error messages from HTTP clients may contain credentials or private payloads.
            logger.error("Live execution paused at %s (%s)", plan["stage"], type(exc).__name__)
            reason = str(exc) if isinstance(exc, ExecutionBlocked) else "Check the live team before further actions"
            result.update(status="execution_failed", mode="execution_failed",
                          execution_error=f"Execution paused at {plan['stage']}. {reason}.")
        result["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_write_json(path, plan)
        return result
