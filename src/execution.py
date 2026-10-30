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


def execution_path(team_id: int, gameweek: int, deadline: str) -> Path:
    season_key = hashlib.sha256(deadline.encode()).hexdigest()[:12]
    return Path(os.getenv("STATE_DIR", "data")) / "executions" / f"{team_id}-{gameweek}-{season_key}.json"


def build_plan(candidate, team_id: int, gameweek: int, result: dict) -> dict:
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
            "transfers": transfers,
            "lineup": {"chip": chip if chip in ("bboost", "3xc") else None, "picks": picks},
            "result": result}


def _chip_confirmed(client, plan: dict, chip) -> bool:
    if not chip:
        return True
    history = client.get_entry_history(plan["team_id"])
    return any(c.get("name") == chip and c.get("event") == plan["gameweek"]
               for c in history.get("chips", []))


def _check_chip_available(client, plan, chip):
    if not chip:
        return
    live = client.get_my_team(plan["team_id"])
    inventory = live.get("chips") if isinstance(live, dict) else None
    history = client.get_entry_history(plan["team_id"]) if not isinstance(inventory, list) else None
    eligible = available_chips(plan["gameweek"], inventory, history)
    if eligible is None or chip not in eligible:
        raise RuntimeError(f"Chip {chip} is unavailable; refusing activation")


def _picks_signature(picks):
    return sorted((p["element"], p["position"], bool(p.get("is_captain")),
                   bool(p.get("is_vice_captain"))) for p in picks)


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
                raise RuntimeError("Authentication failed before execution")
            if plan["stage"] == "transfers_pending":
                live = client.get_my_team(plan["team_id"])
                desired = {p["element"] for p in plan["lineup"]["picks"]}
                actual = {p["element"] for p in live["picks"]}
                if actual != desired or not _chip_confirmed(client, plan, plan["transfers"]["chip"]):
                    raise RuntimeError("Transfer outcome unconfirmed; review the team manually before retrying")
                plan["stage"] = "transfers_done"
                atomic_write_json(path, plan)
            if plan["stage"] == "lineup_pending":
                live = client.get_my_team(plan["team_id"])
                if (_picks_signature(live["picks"]) != _picks_signature(plan["lineup"]["picks"])
                        or not _chip_confirmed(client, plan, plan["lineup"]["chip"])):
                    raise RuntimeError("Lineup outcome unconfirmed; review the team manually before retrying")
                plan["stage"] = "complete"
            if plan["stage"] == "ready":
                if plan["transfers"] is not None:
                    _check_chip_available(client, plan, plan["transfers"]["chip"])
                    plan["stage"] = "transfers_pending"
                    atomic_write_json(path, plan)
                    client.post_transfers(plan["transfers"])
                plan["stage"] = "transfers_done"
                atomic_write_json(path, plan)
            if plan["stage"] == "transfers_done":
                _check_chip_available(client, plan, plan["lineup"]["chip"])
                plan["stage"] = "lineup_pending"
                atomic_write_json(path, plan)
                client.post_lineup(plan["team_id"], plan["lineup"])
                plan["stage"] = "complete"
            result.update(status="success", mode="live_execution")
            result.pop("execution_error", None)
        except Exception as exc:
            # Error messages from HTTP clients may contain credentials or private payloads.
            logger.error("Live execution paused at %s (%s)", plan["stage"], type(exc).__name__)
            result.update(status="execution_failed", mode="execution_failed",
                          execution_error=f"Execution paused at {plan['stage']}; verify the live team before further actions.")
        result["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_write_json(path, plan)
        return result
