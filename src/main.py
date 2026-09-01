"""Main CLI Orchestrator and Background Daemon for Autonomous FPL Engine."""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.agent.director import AIDirector, DecisionOutput
from src.api.auth import FPLAuth
from src.api.client import FPLClient
from src.data_fetcher import FPLCoreInsightsFetcher, FPLReviewFetcher, VaastavDataFetcher
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import FPLOptimizer, OptimizationResult
from src.notifier.telegram import TelegramNotifier
from src.tracker.league_scanner import LeagueAnalysis, LeagueScanner
from src.tracker.news_tracker import NewsTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fpl-orchestrator")


def load_environment():
    """Load environment variables from .env file with override enabled."""
    load_dotenv(override=True)


load_environment()


def get_active_gameweek(events: List[Dict[str, Any]]) -> tuple[int, str, bool]:
    """Identify the next actionable gameweek and its deadline."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # 1. Look for event marked is_next (the upcoming editable gameweek)
    for ev in events:
        if ev.get("is_next"):
            return ev.get("id", 1), ev.get("deadline_time", ""), True

    # 2. Check if is_current is still before its deadline
    for ev in events:
        if ev.get("is_current") and not ev.get("finished"):
            deadline_str = ev.get("deadline_time", "")
            if deadline_str:
                try:
                    deadline_dt = datetime.datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                    if deadline_dt > now_utc:
                        return ev.get("id", 1), deadline_str, False
                except Exception:
                    pass

    # 3. Fallback to first unfinished event with a future deadline
    for ev in events:
        if not ev.get("finished", False):
            return ev.get("id", 1), ev.get("deadline_time", ""), True

    return 1, "", True


def get_latest_live_gameweek(events: List[Dict[str, Any]]) -> Optional[int]:
    """
    Identify the most recent gameweek whose deadline has passed (picks are publicly available).
    Returns None if no gameweek deadline has passed yet in the season (pre-season).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    latest_gw: Optional[int] = None

    for ev in events:
        deadline_str = ev.get("deadline_time", "")
        if deadline_str:
            try:
                deadline_dt = datetime.datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                if deadline_dt <= now_utc:
                    latest_gw = ev.get("id")
            except Exception:
                pass

    if latest_gw is None:
        for ev in events:
            if ev.get("is_previous") or (ev.get("is_current") and ev.get("finished")):
                latest_gw = ev.get("id")

    return latest_gw


def _build_competitive_context(
    entry_history: Dict[str, Any],
    rivals: Optional[List[Dict[str, Any]]] = None,
    my_entry_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Derive DEFEND / NEUTRAL / CHASE risk mode from entry history and league standings.
    Returns a dict ready to inject into the LLM prompt.
    """
    current_season = entry_history.get("current", [])
    if not current_season:
        return {}

    # Most recent GW entry is last in the list
    latest = current_season[-1]
    my_total_points = latest.get("total_points", 0)
    overall_rank = latest.get("overall_rank")
    rank_in_mini_league: Optional[int] = None
    points_behind_leader: Optional[int] = None
    points_ahead_last: Optional[int] = None

    if rivals:
        sorted_rivals = sorted(rivals, key=lambda r: -r.get("total", 0))
        leader_points = sorted_rivals[0].get("total", 0) if sorted_rivals else my_total_points

        # 1. Direct match by entry ID if present in standings
        my_rival_match = next((r for r in sorted_rivals if my_entry_id and r.get("entry") == my_entry_id), None)
        if my_rival_match:
            rank_in_mini_league = my_rival_match.get("rank")
            my_league_points = my_rival_match.get("total", my_total_points)
            points_behind_leader = max(0, leader_points - my_league_points)
            curr_rank = my_rival_match.get("rank", 1)
            next_below = next((r for r in sorted_rivals if r.get("rank", 0) > curr_rank), None)
            points_ahead_last = (my_league_points - next_below.get("total", 0)) if next_below else 0
        else:
            # 2. Derive relative rank by points comparison
            matched = False
            for idx, r in enumerate(sorted_rivals):
                if r.get("total", 0) <= my_total_points:
                    rank_in_mini_league = idx + 1
                    matched = True
                    if idx + 1 < len(sorted_rivals):
                        points_ahead_last = my_total_points - sorted_rivals[idx + 1].get("total", 0)
                    else:
                        points_ahead_last = 0
                    break
            if not matched:
                rank_in_mini_league = len(sorted_rivals) + 1
                points_ahead_last = 0
            points_behind_leader = max(0, leader_points - my_total_points)

    # Determine risk mode
    gws_remaining = max(1, 38 - len(current_season))
    if points_behind_leader is not None:
        if points_behind_leader <= 5:
            risk_mode = "DEFEND"
        elif points_behind_leader >= 35 or (points_behind_leader >= 20 and gws_remaining <= 15) or (points_behind_leader >= 10 and gws_remaining <= 6):
            risk_mode = "CHASE"
        else:
            risk_mode = "NEUTRAL"
    else:
        risk_mode = "NEUTRAL"

    return {
        "overall_rank": overall_rank,
        "my_total_points": my_total_points,
        "rank_in_mini_league": rank_in_mini_league,
        "points_behind_leader": points_behind_leader,
        "points_ahead_next_rival_below": points_ahead_last,
        "gameweeks_remaining": gws_remaining,
        "risk_mode": risk_mode,
        "risk_mode_note": {
            "DEFEND": "Within 5 pts of league leader — protect rank, avoid hits and differentials.",
            "CHASE": "Significant deficit to leader — be aggressive, take calculated hits, play high-EV differentials.",
            "NEUTRAL": "Mid-table — balance risk vs reward, take free hits, avoid unnecessary -4s.",
        }.get(risk_mode, ""),
    }


def _build_chip_season_plan(
    entry_history: Dict[str, Any],
    events: List[Dict[str, Any]],
    current_gw: int,
) -> Dict[str, Any]:
    """
    Build a season-long chip plan from available chips and upcoming DGW/BGW events.
    Returned as a dict injected into the director prompt.
    """
    chips_raw = entry_history.get("chips", [])
    used_chip_names = {c.get("name") for c in chips_raw}
    all_chips = {"wildcard", "wildcard2", "bboost", "3xc", "freehit"}
    chips_remaining = list(all_chips - used_chip_names)

    # Detect DGW/BGW from events (events with more/fewer fixtures than usual)
    upcoming_dgw: List[int] = []
    upcoming_bgw: List[int] = []
    for ev in events:
        ev_id = ev.get("id", 0)
        if ev_id <= current_gw:
            continue
        # chip_plays is available in some FPL seasons as a heuristic; use average as proxy
        # FPL API marks BGW events with very low average_entry_score sometimes, but the most
        # reliable approach is checking the event fixture count from the fixtures endpoint.
        # Here we use the 'most_selected' field absence as a BGW hint (unavailable in BGWs)
        chip_plays = ev.get("chip_plays", [])
        if not chip_plays and ev.get("id"):
            upcoming_bgw.append(ev_id)

    # Build plain-English guidance for each chip still available
    recommendations: Dict[str, str] = {}
    if "wildcard" in chips_remaining or "wildcard2" in chips_remaining:
        recommendations["Wildcard"] = (
            "Best played before a DGW to maximise double-fixture players. "
            "Can also rescue a badly-structured squad mid-season."
        )
    if "freehit" in chips_remaining:
        recommendations["Free Hit"] = (
            "Optimal for a BGW with many blanking key assets — temp full squad rebuild, reverts next GW."
        )
    if "bboost" in chips_remaining:
        recommendations["Bench Boost"] = (
            "Save for a high-fixture DGW where your bench has strong double-gameweek coverage."
        )
    if "3xc" in chips_remaining:
        recommendations["Triple Captain"] = (
            "Best in a DGW where a top premium asset plays twice — typically a striker vs weak opposition."
        )

    return {
        "chips_remaining": chips_remaining,
        "chips_used": list(used_chip_names),
        "upcoming_dgw_gameweeks": upcoming_dgw,
        "upcoming_bgw_gameweeks": upcoming_bgw,
        "chip_guidance": recommendations,
    }


def _append_decision_history(
    decision: "DecisionOutput",
    gameweek: int,
    opt_result: "OptimizationResult",
) -> None:
    """
    Append this GW's decision to data/decisions_history.json for autonomous performance tracking.
    Only writes predicted xP — actual score should be patched post-deadline by a scheduled job.
    """
    history_file = Path("data/decisions_history.json")
    history_file.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Avoid duplicate entries for the same GW
    history = [h for h in history if h.get("gameweek") != gameweek]

    chosen = decision.selected_candidate
    history.append({
        "gameweek": gameweek,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "decision_source": decision.source,
        "chosen_move": decision.chosen_move_name,
        "captain": decision.captain_name,
        "vice_captain": decision.vice_captain_name,
        "transfers_count": chosen.transfers_count,
        "hit_cost": chosen.hit_cost,
        "predicted_net_xp": decision.projected_net_xp,
        "strategic_value_score": chosen.strategic_value_score,
        "rationale": decision.rationale,
        "actual_gw_score": None,  # Patched by post-deadline job
        "actual_overall_rank_change": None,
    })

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Decision appended to {history_file} (GW{gameweek})")



def run_pipeline(
    client: FPLClient,
    dry_run: bool = True,
    execute: bool = False,
    custom_squad_ids: Optional[List[int]] = None,
    custom_bank_m: float = 0.0,
    custom_ft: int = 1,
) -> Dict[str, Any]:
    """
    Execute full optimization, decision, and dispatch pipeline.
    """
    logger.info("=" * 60)
    logger.info(f"Starting FPL Tactical Pipeline (Mode: {'DRY-RUN' if not execute else 'LIVE EXECUTION'})")
    logger.info("=" * 60)

    # 1. Fetch bootstrap and fixtures
    bootstrap = client.get_bootstrap_static()
    events = bootstrap.get("events", [])
    gw_id, deadline_time, is_next = get_active_gameweek(events)
    logger.info(f"Target Gameweek: GW{gw_id} | Deadline: {deadline_time}")

    fixtures = []
    try:
        fixtures = client.get_fixtures()
    except Exception as e:
        logger.warning(f"Could not fetch fixtures: {e}. Using baseline FDR ratings.")

    # 2. Ingest external datasets (olbauday/FPL-Core-Insights & vaastav/Fantasy-Premier-League)
    logger.info("Fetching FPL Core Insights dataset (olbauday/FPL-Core-Insights)...")
    core_fetcher = FPLCoreInsightsFetcher()
    core_insights_df = core_fetcher.fetch_projections()
    if core_insights_df is not None and not core_insights_df.empty:
        logger.info(f"Loaded {len(core_insights_df)} player records from FPL Core Insights data feed.")
    else:
        logger.warning(
            "External FPL Core Insights unavailable. Seamlessly falling back to default FPL expected points (ep_next / baseline)."
        )

    # Ingest Vaastav dataset for underlying stats (xGI/90, xGC/90, minutes tracking)
    vaastav_stats: Optional[Dict[int, Dict[str, Any]]] = None
    enable_vaastav = os.getenv("VAASTAV_DATA_ENABLED", "true").lower() in ("true", "1", "yes")
    if enable_vaastav:
        try:
            logger.info("Fetching Vaastav FPL dataset (vaastav/Fantasy-Premier-League)...")
            vaastav_fetcher = VaastavDataFetcher()
            vaastav_stats = vaastav_fetcher.calculate_player_underlying_metrics()
            if vaastav_stats:
                logger.info(f"Loaded underlying metrics for {len(vaastav_stats)} players from Vaastav dataset.")
        except Exception as e:
            logger.warning(f"Could not load Vaastav FPL dataset: {e}. Continuing with standard projections.")

    # 3. Enrich player metrics with xP, underlying stats, and market odds
    logger.info("Computing expected points (xP), underlying form, FDR weighting, and market implied odds...")
    players_df = calculate_player_metrics(
        bootstrap,
        fixtures,
        current_event=gw_id,
        fplreview_df=core_insights_df,
        vaastav_stats=vaastav_stats,
    )

    # 3. Retrieve current squad, selling prices, and bank
    team_id_str = os.getenv("FPL_TEAM_ID", "").strip()
    current_squad_ids: List[int] = []
    selling_prices: Optional[Dict[int, float]] = None
    bank_m: float = custom_bank_m
    free_transfers: int = custom_ft

    if custom_squad_ids:
        current_squad_ids = custom_squad_ids
    elif client.auth.is_authenticated and team_id_str:
        try:
            team_id = int(team_id_str)
            auth_ok, auth_msg = client.validate_auth(team_id)
            if not auth_ok:
                logger.warning(f"FPL Authentication warning: {auth_msg}")

            logger.info(f"Fetching live squad for Team ID {team_id}...")
            my_team_data = client.get_my_team(team_id)
            picks = my_team_data.get("picks", [])
            current_squad_ids = [p["element"] for p in picks]
            selling_prices = {
                p["element"]: float(p.get("selling_price", p.get("now_cost", 50))) / 10.0
                for p in picks
            }
            transfers_info = my_team_data.get("transfers", {})
            bank_m = round(transfers_info.get("bank", 0) / 10.0, 1)
            free_transfers = transfers_info.get("limit", 1)
            logger.info(
                f"Live squad retrieved: 15 players | Bank: £{bank_m}m | Free Transfers: {free_transfers}"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch live team: {e}. Initializing default squad from top performers.")

    optimizer = FPLOptimizer(players_df)

    if not current_squad_ids:
        logger.info("Selecting valid optimal starting squad from player pool...")
        current_squad_ids = optimizer.select_initial_squad(budget_m=100.0)
        bank_m = 0.5
        free_transfers = 1

    # 3b. Fetch entry history for competitive context + chip season plan
    entry_history_data: Dict[str, Any] = {}
    competitive_context: Dict[str, Any] = {}
    chip_season_plan: Dict[str, Any] = {}
    if team_id_str:
        try:
            raw_history = client.get_entry_history(int(team_id_str))
            # Guard: only use if the API returned a proper dict
            if isinstance(raw_history, dict):
                entry_history_data = raw_history
                competitive_context = _build_competitive_context(
                    entry_history_data,
                    rivals=None,  # Populated after league scan below
                )
                chip_season_plan = _build_chip_season_plan(
                    entry_history_data,
                    events=events,
                    current_gw=gw_id,
                )
                logger.info(
                    f"Competitive context: Rank #{competitive_context.get('overall_rank', 'N/A')} | "
                    f"Risk mode: {competitive_context.get('risk_mode', 'NEUTRAL')} | "
                    f"Chips remaining: {chip_season_plan.get('chips_remaining', [])}"
                )
        except Exception as e:
            logger.warning(f"Could not fetch entry history for context: {e}")


    # 4. Mini-League Threat Matrix Scanning (Pre-Optimization for EO-Aware Solving)
    league_analysis: Optional[LeagueAnalysis] = None
    eo_weights: Optional[Dict[int, float]] = None
    league_id_str = (
        os.getenv("LEAGUE_ID")
        or os.getenv("FPL_LEAGUE_ID")
        or os.getenv("MINI_LEAGUE_ID")
        or ""
    ).strip()

    league_id: Optional[int] = None
    if league_id_str:
        try:
            league_id = int(league_id_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid LEAGUE_ID string '{league_id_str}' in environment.")

    # Auto-discovery fallback if league_id is not explicitly configured
    if league_id is None and team_id_str:
        try:
            res = client.session.get(f"{client.base_url}entry/{team_id_str}/", timeout=8)
            if res.status_code == 200:
                e_data = res.json()
                for l_item in e_data.get("leagues", {}).get("classic", []):
                    if l_item.get("league_type") == "x":
                        league_id = int(l_item["id"])
                        logger.info(f"Auto-discovered private mini-league '{l_item.get('name')}' (ID: {league_id}) from user profile.")
                        break
        except Exception as e:
            logger.debug(f"Mini-league auto-discovery fallback skipped: {e}")

    if league_id:
        try:
            # Before target GW deadline, target GW picks are unpublished (return 404).
            # Scan the latest completed/live gameweek to assess rival squads.
            scan_gw = get_latest_live_gameweek(events)
            if scan_gw is not None:
                logger.info(f"Scanning Mini-League {league_id} using GW{scan_gw} picks for Effective Ownership...")
            else:
                logger.info(f"Scanning Mini-League {league_id} standings (Pre-GW1, no picks available yet)...")

            scanner = LeagueScanner(client)
            league_analysis = scanner.scan_league(
                league_id=league_id,
                gameweek=scan_gw,
                my_team_ids=set(current_squad_ids),
                bootstrap_data=bootstrap,
            )
            eo_weights = league_analysis.raw_eo
            logger.info(
                f"Mini-league '{league_analysis.league_name}' (ID: {league_id}) scanned: {league_analysis.total_managers} managers | "
                f"Shields: {len(league_analysis.threat_matrix.shields)} | "
                f"Vulnerabilities: {len(league_analysis.threat_matrix.vulnerabilities)} | "
                f"Daggers: {len(league_analysis.threat_matrix.daggers)}"
            )
        except Exception as e:
            logger.warning(f"Mini-league scan failed: {e}")

    # Enrich competitive context with mini-league rank once rivals are available
    if league_analysis and league_analysis.rivals and entry_history_data:
        try:
            rival_standings = [
                {"entry": r.entry_id, "rank": r.rank, "total": r.total_points}
                for r in league_analysis.rivals
            ]
            my_entry_id = int(team_id_str) if team_id_str else None
            competitive_context = _build_competitive_context(
                entry_history_data,
                rivals=rival_standings,
                my_entry_id=my_entry_id,
            )
            logger.info(
                f"Mini-league rank: #{competitive_context.get('rank_in_mini_league', 'N/A')} | "
                f"Behind leader: {competitive_context.get('points_behind_leader', 'N/A')} pts | "
                f"Risk mode: {competitive_context.get('risk_mode', 'NEUTRAL')}"
            )
        except Exception as e:
            logger.warning(f"Could not enrich competitive context with league standings: {e}")

    # 5. PuLP MILP Optimization & Automated Chip Evaluation
    logger.info("Running PuLP MILP solver (Multi-Period Horizon, Real Selling Prices & EO Shielding)...")
    opt_result = optimizer.optimize(
        current_squad_ids=current_squad_ids,
        bank_m=bank_m,
        free_transfers=free_transfers,
        selling_prices=selling_prices,
        eo_weights=eo_weights,
        current_gw=gw_id,
        evaluate_chips=True,
    )

    logger.info(f"Generated {len(opt_result.candidates)} candidate options:")
    for idx, cand in enumerate(opt_result.candidates, start=1):
        logger.info(
            f"  [{idx}] {cand.name} ➔ Net xP: {cand.net_xp:.2f} | 3-GW xP: {cand.multi_gw_xp:.2f} | Formation: {cand.formation} | Captain: {cand.captain.web_name if cand.captain else 'N/A'}"
        )

    # Automated Chip Strategy Evaluation
    enable_auto_chips = os.getenv("ENABLE_AUTO_CHIPS", "false").lower() in ("true", "1", "yes")
    recommended_chip: Optional[str] = None
    triggered_chip_eval: Optional[Dict[str, Any]] = None

    if opt_result.chip_evaluation:
        logger.info("=" * 60)
        logger.info("AUTOMATED CHIP STRATEGY EVALUATION:")
        for c_name, c_eval in opt_result.chip_evaluation.evaluations.items():
            status_text = "THRESHOLD MET" if c_eval.threshold_met else "BELOW THRESHOLD"
            logger.info(
                f"  • {c_eval.display_name}: Projected {c_eval.projected_xp:.2f} xP "
                f"(Gain: +{c_eval.xp_gain:.2f} pts vs +{c_eval.threshold:.1f} pts threshold) -> {status_text}"
            )

        if enable_auto_chips and opt_result.chip_evaluation.recommended_chip:
            recommended_chip = opt_result.chip_evaluation.recommended_chip
            c_eval = opt_result.chip_evaluation.evaluations[recommended_chip]
            triggered_chip_eval = c_eval.model_dump()
            logger.info(
                f"⚡ AUTOMATED CHIP ACTIVATED: {c_eval.display_name} "
                f"(Gain: +{c_eval.xp_gain:.2f} xP | Total: {c_eval.projected_xp:.2f} pts)"
            )
            logger.info(f"   Rationale: {c_eval.reason}")
        elif not enable_auto_chips and opt_result.chip_evaluation.recommended_chip:
            rec = opt_result.chip_evaluation.recommended_chip
            logger.info(f"💡 Chip recommendation available ({rec}), but ENABLE_AUTO_CHIPS is false.")

    # 6. Gather News Intelligence & AI Decision Director Evaluation
    logger.info("Gathering live press conference & news intelligence...")
    news_tracker = NewsTracker(search_timeout=3)
    focal_player_ids: Set[int] = set()
    # Prioritize captains and transfer-in targets from top options
    for cand in opt_result.candidates[:5]:
        if cand.captain:
            focal_player_ids.add(cand.captain.id)
        if cand.vice_captain:
            focal_player_ids.add(cand.vice_captain.id)
        for t in cand.transfers:
            focal_player_ids.add(t.player_in.id)

    news_intel = news_tracker.extract_focal_players_from_bootstrap(
        bootstrap_data=bootstrap,
        focal_player_ids=focal_player_ids,
    )

    logger.info("Consulting AI Decision Director...")
    director = AIDirector()
    decision = director.evaluate_and_decide(
        opt_result,
        league_analysis,
        news_intel,
        competitive_context=competitive_context or None,
        chip_season_plan=chip_season_plan or None,
    )

    # Check for AI Director Late Breaking News Veto & Instant Re-Solve
    if decision.veto_player_ids:
        logger.warning(
            f"⚠️ AI Director issued late news emergency veto on player IDs: {decision.veto_player_ids}. "
            f"Reason: {decision.veto_reason or 'Late injury/rotation doubt'}. Re-running MILP solver..."
        )
        for pid in decision.veto_player_ids:
            if pid in optimizer.player_map:
                optimizer.player_map[pid]["injury_multiplier"] = 0.0
                optimizer.player_map[pid]["discounted_xp"] = 0.0
                optimizer.player_map[pid]["xp"] = 0.0
            optimizer.df.loc[optimizer.df["id"] == pid, "injury_multiplier"] = 0.0
            optimizer.df.loc[optimizer.df["id"] == pid, "discounted_xp"] = 0.0
            optimizer.df.loc[optimizer.df["id"] == pid, "xp"] = 0.0

        opt_result = optimizer.optimize(
            current_squad_ids=current_squad_ids,
            bank_m=bank_m,
            free_transfers=free_transfers,
            selling_prices=selling_prices,
            eo_weights=eo_weights,
            current_gw=gw_id,
            evaluate_chips=True,
        )
        logger.info(f"Re-solved! Generated {len(opt_result.candidates)} clean candidate options without vetoed players.")

        decision = director.evaluate_and_decide(
            opt_result,
            league_analysis,
            news_intel,
            competitive_context=competitive_context or None,
            chip_season_plan=chip_season_plan or None,
        )

    # Determine the actual active chip:
    # 1) If the chosen candidate already carries a chip (e.g. Wildcard / Free Hit rebuild candidate), use it.
    # 2) If a lineup chip (bboost, 3xc) was recommended & enabled, apply it to the chosen candidate.
    active_chip: Optional[str] = decision.selected_candidate.active_chip
    if enable_auto_chips and recommended_chip in ("bboost", "3xc") and not active_chip:
        active_chip = recommended_chip
        decision.selected_candidate.active_chip = active_chip
        if active_chip == "bboost":
            bench_xp = sum(p.xp for p in decision.selected_candidate.bench)
            decision.projected_net_xp = round(decision.selected_candidate.net_xp + bench_xp, 2)
            if "Bench Boost" not in decision.chosen_move_name:
                decision.chosen_move_name += " + Bench Boost"
        elif active_chip == "3xc":
            c_xp = decision.selected_candidate.captain.xp if decision.selected_candidate.captain else 0.0
            decision.projected_net_xp = round(decision.selected_candidate.net_xp + c_xp, 2)
            if "Triple Captain" not in decision.chosen_move_name:
                decision.chosen_move_name += " + Triple Captain"

    logger.info("=" * 60)
    logger.info(f"FINAL DECISION: {decision.chosen_move_name} (Source: {decision.source})")
    if active_chip:
        logger.info(f"Active Chip: {active_chip.upper()}")
    logger.info(f"Tactical Moves: {decision.transfers_description}")
    logger.info(f"Armband: {decision.captain_name} (C) | Vice: {decision.vice_captain_name} (VC)")
    logger.info(f"Projected Net xP: {decision.projected_net_xp:.2f}")
    logger.info(f"Director Rationale: \"{decision.rationale}\"")
    if decision.news_alerts:
        logger.info("Press Conference / News Alerts:")
        for alert in decision.news_alerts:
            logger.info(f"  🚨 {alert}")
    logger.info("=" * 60)

    # 7. Live Execution (if requested)
    if execute and client.auth.is_authenticated and team_id_str:
        team_id = int(team_id_str)
        cand = decision.selected_candidate

        # 7a. Transfers submission
        if cand.transfers_count > 0 or cand.active_chip in ("wildcard", "freehit"):
            chip_tx = cand.active_chip if cand.active_chip in ("wildcard", "freehit") else None
            logger.info(
                f"Submitting {cand.transfers_count} live transfer(s) to FPL API "
                f"(Chip: {chip_tx or 'None'})..."
            )
            transfers_payload = {
                "chips": chip_tx,
                "chip": chip_tx,
                "entry": team_id,
                "event": gw_id,
                "transfers": [
                    {
                        "element_in": t.player_in.id,
                        "element_out": t.player_out.id,
                        "purchase_price": int(t.player_in.cost_m * 10),
                        "selling_price": int(t.player_out.cost_m * 10),
                    }
                    for t in cand.transfers
                ],
            }
            try:
                tx_resp = client.post_transfers(transfers_payload)
                logger.info(f"Transfers successfully submitted: {tx_resp}")
            except Exception as e:
                logger.error(f"Live transfer submission failed: {e}")

        # 7b. Lineup & Captaincy submission
        chip_lineup = cand.active_chip if cand.active_chip in ("bboost", "3xc") else None
        logger.info(
            f"Submitting live lineup & captaincy to FPL API "
            f"(Chip: {chip_lineup or 'None'})..."
        )
        # 11 starters + 4 bench
        picks_payload = []
        # Starters 1-11
        for idx, p in enumerate(cand.starters, start=1):
            picks_payload.append({
                "element": p.id,
                "position": idx,
                "is_captain": p.is_captain,
                "is_vice_captain": p.is_vice_captain,
            })
        # Bench 12-15
        for idx, p in enumerate(cand.bench, start=12):
            picks_payload.append({
                "element": p.id,
                "position": idx,
                "is_captain": False,
                "is_vice_captain": False,
            })

        lineup_payload = {
            "chip": chip_lineup,
            "picks": picks_payload,
        }
        try:
            lineup_resp = client.post_lineup(team_id, lineup_payload)
            logger.info(f"Lineup successfully submitted: {lineup_resp}")
        except Exception as e:
            logger.error(f"Live lineup submission failed: {e}")

    # 8. Persist State to data/latest_decision.json
    state_dir = Path("data")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "latest_decision.json"

    output_payload = {
        "status": "success",
        "mode": "live_execution" if execute else "dry_run",
        "gameweek": gw_id,
        "deadline_time": deadline_time,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "decision": decision.model_dump(),
        "optimization": opt_result.model_dump(),
        "active_chip": active_chip,
        "chip_triggered": triggered_chip_eval,
        "league_analysis": league_analysis.model_dump() if league_analysis else None,
        "news_intelligence": [v.model_dump() for v in news_intel.values()] if news_intel else [],
    }

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"State persisted to {state_file}")

    # 8b. Append to decision history log for autonomous performance tracking
    _append_decision_history(decision, gw_id, opt_result)

    # 9. Send Telegram Alerts
    notifier = TelegramNotifier()
    if active_chip and triggered_chip_eval and triggered_chip_eval.get("chip_name") == active_chip:
        notifier.notify_chip_triggered(
            chip_name=active_chip,
            display_name=triggered_chip_eval["display_name"],
            projected_xp=triggered_chip_eval["projected_xp"],
            xp_gain=triggered_chip_eval["xp_gain"],
            reason=triggered_chip_eval["reason"],
            gameweek=gw_id,
            is_live_execution=execute,
        )

    notifier.notify_pre_deadline(
        decision=decision,
        gameweek=gw_id,
        is_live_execution=execute,
    )

    return output_payload


def run_post_deadline_intel(client: FPLClient, target_gw: Optional[int] = None):
    """Run post-deadline mini-league scanner and send intelligence briefing to Telegram."""
    logger.info("Running post-deadline mini-league intelligence scan...")
    bootstrap = client.get_bootstrap_static()
    events = bootstrap.get("events", [])
    gw_id = target_gw or get_latest_live_gameweek(events)
    if not gw_id:
        gw_id, _, _ = get_active_gameweek(events)

    league_id_str = os.getenv("LEAGUE_ID", "").strip()
    if not league_id_str:
        logger.warning("LEAGUE_ID not configured in .env. Skipping post-deadline scan.")
        return

    league_id = int(league_id_str)
    scanner = LeagueScanner(client)

    # Load my squad if available in latest_decision.json
    state_file = Path("data/latest_decision.json")
    my_team_set: Set[int] = set()
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                starters = data.get("decision", {}).get("selected_candidate", {}).get("starters", [])
                bench = data.get("decision", {}).get("selected_candidate", {}).get("bench", [])
                my_team_set = {p["id"] for p in starters + bench}
        except Exception:
            pass

    analysis = scanner.scan_league(
        league_id=league_id,
        gameweek=gw_id,
        my_team_ids=my_team_set,
        bootstrap_data=bootstrap,
    )

    # Update state file with new league analysis
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["league_analysis"] = analysis.model_dump()
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not update state file: {e}")

    # Dispatch to Telegram
    notifier = TelegramNotifier()
    notifier.notify_post_deadline(analysis)
    logger.info("Post-deadline intelligence briefing completed.")


def run_daemon(client: FPLClient):
    """
    Continuous background daemon scheduler.
    Triggers --execute at T-90m before deadline and --post-deadline at T+5m after deadline.
    """
    logger.info("Starting Autonomous FPL Daemon Scheduler...")
    executed_gameweeks: Set[int] = set()
    post_scanned_gameweeks: Set[int] = set()

    while True:
        try:
            bootstrap = client.get_bootstrap_static(force_refresh=True)
            events = bootstrap.get("events", [])
            gw_id, deadline_str, is_next = get_active_gameweek(events)

            if not deadline_str:
                logger.info(f"No active deadline found. Sleeping for 15 minutes...")
                time.sleep(900)
                continue

            # Parse deadline UTC datetime
            # Format usually: "2026-08-25T17:30:00Z"
            deadline_dt = datetime.datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            delta = deadline_dt - now_utc
            delta_minutes = delta.total_seconds() / 60.0

            logger.info(
                f"[DAEMON] GW{gw_id} Deadline: {deadline_str} | Time Remaining: {delta_minutes:.1f} mins "
                f"| Pre-Executed: {gw_id in executed_gameweeks} | Post-Scanned: {gw_id in post_scanned_gameweeks}"
            )

            # Trigger T-90m Pre-Deadline Execution
            if 0 < delta_minutes <= 90 and gw_id not in executed_gameweeks:
                logger.info(f"⚡ T-90m threshold reached for GW{gw_id}! Triggering squad optimization & execution...")
                run_pipeline(client, dry_run=False, execute=True)
                executed_gameweeks.add(gw_id)

            # Trigger T+5m Post-Deadline Scan
            if -180 <= delta_minutes <= -5 and gw_id not in post_scanned_gameweeks:
                logger.info(f"📡 Gameweek {gw_id} underway (+5m passed deadline). Triggering post-deadline scanner...")
                run_post_deadline_intel(client, target_gw=gw_id)
                post_scanned_gameweeks.add(gw_id)

            # Adaptive sleep: sleep faster near deadline, slower when days away
            if 0 < delta_minutes <= 120:
                sleep_sec = 60  # 1 minute near deadline
            elif 0 < delta_minutes <= 1440:
                sleep_sec = 300  # 5 minutes within 24 hours
            else:
                sleep_sec = 1800  # 30 minutes when far away

            time.sleep(sleep_sec)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"Daemon loop encountered error: {e}. Retrying in 60s...")
            time.sleep(60)


def main():
    load_environment()

    parser = argparse.ArgumentParser(description="Autonomous Fantasy Premier League Engine")
    parser.add_argument("--dry-run", action="store_true", help="Run full optimization simulation without live API submission")
    parser.add_argument("--execute", action="store_true", help="Execute live transfers & lineup on FPL API and alert Telegram")
    parser.add_argument("--post-deadline", action="store_true", help="Scan mini-league rivals and send post-deadline intelligence briefing")
    parser.add_argument("--daemon", action="store_true", help="Run autonomous background scheduler loop (T-90m execution, T+5m scanner)")

    args = parser.parse_args()

    client = FPLClient()

    if args.daemon:
        run_daemon(client)
    elif args.post_deadline:
        run_post_deadline_intel(client)
    elif args.execute:
        run_pipeline(client, dry_run=False, execute=True)
    else:
        # Default to dry-run
        run_pipeline(client, dry_run=True, execute=False)


if __name__ == "__main__":
    main()
