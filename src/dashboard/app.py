"""Lightweight Flask web dashboard for autonomous FPL visual analysis."""

import hmac
import secrets
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(override=False)

from src.state import atomic_write_json, state_lock

from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)

# In-memory entry cache with 5-minute TTL to prevent API spam
_ENTRY_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_ENTRY_CACHE_TTL = 300  # seconds

def fetch_entry_details(team_id: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch entry details from the official FPL API: https://fantasy.premierleague.com/api/entry/{team_id}/
    Caches successful responses in memory for 5 minutes.
    """
    now = time.time()
    if team_id in _ENTRY_CACHE:
        cached_time, cached_data = _ENTRY_CACHE[team_id]
        if (now - cached_time) < _ENTRY_CACHE_TTL:
            return cached_data

    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            _ENTRY_CACHE[team_id] = (now, data)
            return data
        logger.warning(f"FPL API returned HTTP {response.status_code} for entry {team_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch entry details for team_id {team_id}: {e}")
    return None


def extract_leagues(
    entry_data: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate classic leagues into two lists:
    - global_leagues: leagues where league_type == 's' (Overall, Country/Regional, Fan League, 2nd Chance, etc.)
    - mini_leagues: leagues where league_type == 'x' (Private mini-leagues with friends/workmates)

    Computes rank trajectories (Green up-arrow / Red down-arrow comparing entry_rank vs entry_last_rank).
    """
    if not entry_data or not isinstance(entry_data, dict):
        return [], []

    classic_leagues = entry_data.get("leagues", {}).get("classic", [])
    if not classic_leagues and "classic" in entry_data:
        classic_leagues = entry_data.get("classic", [])

    global_leagues: List[Dict[str, Any]] = []
    mini_leagues: List[Dict[str, Any]] = []

    for item in classic_leagues:
        l_type = item.get("league_type", "")
        name = item.get("name", "Unknown League")
        rank = item.get("entry_rank") if item.get("entry_rank") is not None else item.get("rank")
        last_rank = item.get("entry_last_rank")
        total_players = item.get("total_players")

        # Calculate rank movement and direction
        movement = 0
        direction = "same"
        if rank is not None and last_rank is not None and last_rank > 0 and rank > 0:
            movement = last_rank - rank
            if movement > 0:
                direction = "up"
            elif movement < 0:
                direction = "down"
            else:
                direction = "same"
        elif last_rank is None or last_rank == 0:
            direction = "new"

        league_dict = {
            "id": item.get("id"),
            "name": name,
            "league_type": l_type,
            "entry_rank": rank,
            "entry_last_rank": last_rank,
            "rank": rank,
            "last_rank": last_rank,
            "total_players": total_players,
            "movement": movement,
            "direction": direction,
            "raw": item,
        }

        if l_type == "s":
            global_leagues.append(league_dict)
        elif l_type == "x":
            mini_leagues.append(league_dict)

    return global_leagues, mini_leagues


def build_global_summary(
    global_leagues: List[Dict[str, Any]],
    entry_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build quick summary cards for Overall Rank, Country, and Favorite Club.
    """
    overall_league = None
    country_league = None
    club_league = None

    region_name = (
        entry_data.get("player_region_name", "").strip().lower()
        if entry_data
        else ""
    )

    for gl in global_leagues:
        gl_name = gl["name"].strip().lower()
        # Check Overall
        if gl_name == "overall" or gl.get("id") == 275:
            overall_league = gl
        # Check Country
        elif region_name and region_name in gl_name:
            if not country_league:
                country_league = gl
        # Check other leagues as candidate country/club
        elif not country_league and gl_name in [
            "england", "scotland", "wales", "northern ireland", "ireland",
            "united states", "australia", "canada", "norway", "sweden", "india", "nigeria"
        ]:
            country_league = gl

    # Second pass for candidate club league
    for gl in global_leagues:
        if gl == overall_league or gl == country_league:
            continue
        gl_name = gl["name"].strip().lower()
        if "gameweek" not in gl_name and "second chance" not in gl_name and "cup" not in gl_name:
            if not club_league:
                club_league = gl
                break

    # Fallback overall if not explicitly named Overall
    if not overall_league and entry_data and entry_data.get("summary_overall_rank"):
        overall_league = {
            "name": "Overall",
            "rank": entry_data.get("summary_overall_rank"),
            "entry_rank": entry_data.get("summary_overall_rank"),
            "last_rank": None,
            "entry_last_rank": None,
            "total_players": None,
            "movement": 0,
            "direction": "same",
        }

    return {
        "overall": overall_league,
        "country": country_league,
        "club": club_league,
    }


def load_decision_data(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load latest decision state strictly from local disk."""
    path = data_path or Path(os.getenv("STATE_DIR", "data")) / "latest_decision.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return {}


def create_app(data_file_path: Optional[Path] = None) -> Flask:
    """Flask application factory."""
    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    state_file = data_file_path or Path(os.getenv("STATE_DIR", "data")) / "latest_decision.json"
    sync_secret = os.getenv("DASHBOARD_SYNC_SECRET", "")
    if not sync_secret:
        secret_file = Path(os.getenv("STATE_DIR", "data")) / "dashboard_sync_secret.json"
        with state_lock(secret_file):
            if secret_file.exists():
                sync_secret = json.loads(secret_file.read_text(encoding="utf-8"))["secret"]
            else:
                sync_secret = secrets.token_urlsafe(32)
                atomic_write_json(secret_file, {"secret": sync_secret})
    if not isinstance(sync_secret, str) or not sync_secret:
        raise ValueError("Dashboard pairing secret must be a nonempty string")

    # Template filters
    @app.template_filter("comma")
    def comma_filter(val: Any) -> str:
        if val is None or val == "":
            return "-"
        try:
            return f"{int(val):,}"
        except (ValueError, TypeError):
            return str(val)

    @app.template_filter("abs")
    def abs_filter(val: Any) -> Any:
        try:
            return abs(val)
        except (ValueError, TypeError):
            return val

    @app.route("/")
    def index():
        data = load_decision_data(state_file)

        # Categorize starters by position for pitch view
        starters_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        if data and "decision" in data and "selected_candidate" in data["decision"]:
            starters = data["decision"]["selected_candidate"].get("starters", [])
            for p in starters:
                pos = p.get("position", "MID")
                if pos in starters_by_pos:
                    starters_by_pos[pos].append(p)

        # Retrieve team entry details and parse leagues
        entry_data: Optional[Dict[str, Any]] = None

        # Check if entry_data already attached to decision state
        if data and "entry_data" in data and isinstance(data["entry_data"], dict):
            entry_data = data["entry_data"]
        elif data and "entry" in data and isinstance(data["entry"], dict):
            entry_data = data["entry"]

        # If not present in file, fetch from FPL API using team_id
        if not entry_data:
            team_id_raw = (
                os.getenv("FPL_TEAM_ID")
                or os.getenv("TEAM_ID")
                or (str(data.get("team_id", "")) if data else "")
                or (str(data.get("entry_id", "")) if data else "")
            )
            if team_id_raw:
                try:
                    team_id = int(str(team_id_raw).strip())
                    entry_data = fetch_entry_details(team_id)
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Invalid team_id '{team_id_raw}': {exc}")

        # Separate classic leagues into global and mini leagues
        global_leagues, mini_leagues = extract_leagues(entry_data)
        global_summary = build_global_summary(global_leagues, entry_data)

        return render_template(
            "index.html",
            data=data,
            starters_by_pos=starters_by_pos,
            entry_data=entry_data or {},
            global_leagues=global_leagues,
            mini_leagues=mini_leagues,
            global_summary=global_summary,
        )

    @app.route("/api/decision")
    def api_decision():
        """REST endpoint returning current decision state."""
        data = load_decision_data(state_file)
        return jsonify(data)

    @app.after_request
    def auth_sync_cors(response):
        if request.path == "/api/auth/sync":
            origin = request.headers.get("Origin")
            allowed = {"https://fantasy.premierleague.com", request.host_url.rstrip("/")}
            if origin in allowed:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Gegenbot-Sync-Secret"
        return response

    @app.route("/api/auth/sync", methods=["POST", "OPTIONS"])
    def api_auth_sync():
        """Accept browser tokens only with the pairing secret and a trusted origin."""
        origin = request.headers.get("Origin")
        if origin and origin not in {"https://fantasy.premierleague.com", request.host_url.rstrip("/")}:
            return jsonify(status="error", message="Origin not allowed"), 403
        if request.method == "OPTIONS":
            return jsonify(status="ok")
        secret = sync_secret
        provided = request.headers.get("X-Gegenbot-Sync-Secret", "")
        if not hmac.compare_digest(provided.encode(), secret.encode()):
            return jsonify(status="error", message="Invalid pairing secret"), 403
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(status="error", message="Expected a JSON object"), 400
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token", "")
        if not isinstance(access_token, str) or not access_token.strip() or not isinstance(refresh_token, str):
            return jsonify(status="error", message="Expected access_token and optional refresh_token strings"), 400
        try:
            from src.api.auth import FPLAuth, _parse_jwt_exp
            exp = _parse_jwt_exp(access_token)
            if exp is not None and exp <= time.time():
                return jsonify(status="error", message="Access token has expired"), 400
            auth = FPLAuth()
            with state_lock(auth.state_file):
                auth = FPLAuth()
                auth.update_tokens(access_token=access_token, refresh_token=refresh_token or None,
                                   expires_in=(exp - time.time()) if exp else 7200)
            return jsonify(status="success", message="Gegenbot authentication successfully synchronized!", expires_at=exp)
        except Exception:
            logger.error("Failed to persist synchronized authentication tokens")
            return jsonify(status="error", message="Could not save authentication state"), 500

    @app.route("/healthz")
    def health():
        return jsonify({"status": "healthy"})

    return app


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    app = create_app()
    app.run(host=host, port=port, debug=False)
