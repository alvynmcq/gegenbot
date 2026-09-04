"""AI Decision Director for evaluating MILP candidates with Game Theory, Threat Matrix & Live News Intelligence."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
import requests
from pydantic import BaseModel, Field

from src.engine.optimizer import CandidateSquad, OptimizationResult
from src.tracker.league_scanner import LeagueAnalysis
from src.tracker.news_tracker import PlayerNewsIntel

logger = logging.getLogger(__name__)


class DecisionOutput(BaseModel):
    """Final decision produced by AI Director (or deterministic fallback)."""
    selected_candidate_index: int
    selected_candidate: CandidateSquad
    chosen_move_name: str
    transfers_description: str
    captain_name: str
    vice_captain_name: str
    projected_net_xp: float
    rationale: str
    source: str  # "LLM_DIRECTOR" or "MATHEMATICAL_FALLBACK"
    news_alerts: List[str] = Field(default_factory=list)
    captain_override: Optional[str] = None
    vice_captain_override: Optional[str] = None
    veto_player_ids: List[int] = Field(default_factory=list)
    veto_reason: Optional[str] = None


class AIDirector:
    """Evaluates optimizer candidate moves against mini-league threat dynamics and live press conference news."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "").strip()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        env_timeout = os.getenv("LLM_TIMEOUT", "").strip()
        self.timeout = timeout if timeout is not None else (int(env_timeout) if env_timeout.isdigit() else 45)
        self.max_retries = max_retries

    def _extract_candidate_news_alerts(
        self,
        candidate: CandidateSquad,
        news_intel: Optional[Union[Dict[int, PlayerNewsIntel], List[PlayerNewsIntel]]] = None,
    ) -> List[str]:
        """Collect relevant news alerts for players involved in a chosen candidate."""
        if not news_intel:
            return []

        intel_by_id: Dict[int, PlayerNewsIntel] = {}
        if isinstance(news_intel, dict):
            intel_by_id = news_intel
        elif isinstance(news_intel, list):
            intel_by_id = {item.player_id: item for item in news_intel if hasattr(item, "player_id")}

        alerts: List[str] = []
        # Key focal players in chosen candidate
        focal_players = list(candidate.starters)
        if candidate.captain:
            focal_players.append(candidate.captain)
        if candidate.vice_captain:
            focal_players.append(candidate.vice_captain)
        for t in candidate.transfers:
            focal_players.append(t.player_in)

        seen_ids = set()
        for p in focal_players:
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)
            if p.id in intel_by_id:
                info = intel_by_id[p.id]
                if info.risk_level != "CLEARED" or info.official_fpl_news or info.live_search_snippets:
                    alerts.append(info.summary())

        return alerts

    def _build_fallback_decision(
        self,
        optimization_result: OptimizationResult,
        reason: str = "Automated mathematical optimization",
        news_intel: Optional[Union[Dict[int, PlayerNewsIntel], List[PlayerNewsIntel]]] = None,
    ) -> DecisionOutput:
        """Fallback to the candidate with highest strategic value score and net xP."""
        candidates = optimization_result.candidates
        if not candidates:
            raise ValueError("No candidates available for fallback decision.")

        best_candidate_idx = 0
        best_score = -999.0
        for idx, cand in enumerate(candidates):
            score = cand.strategic_value_score if cand.strategic_value_score != 0.0 else cand.net_xp
            if score > best_score:
                best_score = score
                best_candidate_idx = idx

        chosen = candidates[best_candidate_idx]
        c_name = chosen.captain.web_name if chosen.captain else "Unknown"
        vc_name = chosen.vice_captain.web_name if chosen.vice_captain else "Unknown"

        if chosen.transfers_count == 0:
            tx_desc = "No transfers (Roll/Bank transfer)."
        else:
            tx_desc = ", ".join([f"{t.player_out.web_name} ➔ {t.player_in.web_name}" for t in chosen.transfers])

        rationale = (
            f"Selected {chosen.name} based on optimal MILP projection of {chosen.net_xp} net xP. "
            f"Handing armband to {c_name} with {vc_name} as vice-captain to maximize expected points."
        )

        logger.info(f"Fallback decision made: {chosen.name}")
        news_alerts = self._extract_candidate_news_alerts(chosen, news_intel)

        return DecisionOutput(
            selected_candidate_index=best_candidate_idx,
            selected_candidate=chosen,
            chosen_move_name=chosen.name,
            transfers_description=tx_desc,
            captain_name=c_name,
            vice_captain_name=vc_name,
            projected_net_xp=chosen.net_xp,
            rationale=rationale,
            source="MATHEMATICAL_FALLBACK",
            news_alerts=news_alerts,
        )

    def evaluate_and_decide(
        self,
        optimization_result: OptimizationResult,
        league_analysis: Optional[LeagueAnalysis] = None,
        news_intel: Optional[Union[Dict[int, PlayerNewsIntel], List[PlayerNewsIntel]]] = None,
        competitive_context: Optional[Dict[str, Any]] = None,
        chip_season_plan: Optional[Dict[str, Any]] = None,
    ) -> DecisionOutput:
        """
        Send prompt to LLM containing solver candidates, mini-league threat matrix,
        live press conference / injury intelligence, competitive rank context, and chip season plan.
        Fallback to PuLP top net xP candidate on error or missing API key.
        """
        candidates = optimization_result.candidates
        if not candidates:
            raise ValueError("Optimization result contains 0 candidates.")

        if not self.api_key:
            logger.info("LLM_API_KEY not configured. Using deterministic mathematical MILP solver.")
            return self._build_fallback_decision(optimization_result, "No API key configured", news_intel)

        # Format candidates context for LLM
        candidate_summaries = []
        for idx, cand in enumerate(candidates):
            c_name = cand.captain.web_name if cand.captain else "N/A"
            vc_name = cand.vice_captain.web_name if cand.vice_captain else "N/A"
            tx_list = [f"OUT: {t.player_out.web_name} (£{t.player_out.cost_m}m) ➔ IN: {t.player_in.web_name} (£{t.player_in.cost_m}m)" for t in cand.transfers]

            # xP confidence flags, underlying Vaastav stats, and Moneyball metrics for transfer targets
            xp_flags = []
            for t in cand.transfers:
                conf = t.player_in.xp_confidence
                if conf != "HIGH":
                    xp_flags.append(f"{t.player_in.web_name} xP confidence: {conf}")
                if t.player_in.imminent_price_change == "RISE_IMMINENT":
                    xp_flags.append(f"{t.player_in.web_name} [PRICE RISE IMMINENT]: target threshold met (buying captures +£0.1m rise)")
                elif abs(t.player_in.price_momentum) >= 0.1:
                    direction = "rising" if t.player_in.price_momentum > 0 else "falling"
                    xp_flags.append(f"{t.player_in.web_name} price {direction} (momentum: {t.player_in.price_momentum:+.2f})")
                if t.player_out.imminent_price_change == "FALL_IMMINENT":
                    xp_flags.append(f"{t.player_out.web_name} [PRICE FALL IMMINENT]: selling avoids -£0.1m loss")
                if t.player_in.set_piece_role:
                    xp_flags.append(f"{t.player_in.web_name} [SET PIECES]: {t.player_in.set_piece_role}")
                if t.player_in.rolling_xgi_90 is not None and t.player_in.rolling_xgi_90 >= 0.40:
                    xp_flags.append(f"{t.player_in.web_name} strong underlying attacking threat ({t.player_in.rolling_xgi_90:.2f} xGI/90)")
                if t.player_in.moneyball_tag == "UNDERVALUED_REGRESSION":
                    delta_str = f"{t.player_in.xgi_delta:+.2f}" if t.player_in.xgi_delta is not None else "+0.8"
                    xp_flags.append(f"{t.player_in.web_name} [MONEYBALL BUY]: underperforming xGI by {delta_str} (positive mean reversion candidate)")
                elif t.player_in.moneyball_tag == "HIGH_EFFICIENCY_ENABLER":
                    vorp_str = f"{t.player_in.vorp_per_m:.2f}" if t.player_in.vorp_per_m is not None else "1.8"
                    xp_flags.append(f"{t.player_in.web_name} [MONEYBALL ENABLER]: elite budget yield ({vorp_str} VORP/£m)")
                elif t.player_in.moneyball_tag == "OVERVALUED_HAULER":
                    xp_flags.append(f"{t.player_in.web_name} [MONEYBALL CAUTION]: overperforming underlying xGI")
                if t.player_in.implied_goal_pct is not None and t.player_in.implied_goal_pct >= 40.0:
                    xp_flags.append(f"{t.player_in.web_name} [MARKET ODDS]: {t.player_in.implied_goal_pct:.1f}% anytime goal probability")
                elif t.player_in.implied_cs_pct is not None and t.player_in.implied_cs_pct >= 38.0 and t.player_in.position in ["GKP", "DEF"]:
                    xp_flags.append(f"{t.player_in.web_name} [MARKET ODDS]: {t.player_in.implied_cs_pct:.1f}% clean sheet probability")
                if t.player_in.minutes_reliability == "LOW":
                    xp_flags.append(f"{t.player_in.web_name} minutes risk (recent starts/mins below threshold)")

            bench_summary = [f"{p.web_name} (Sub #{p.bench_order})" for p in cand.bench]
            starter_names = [p.web_name for p in cand.starters]
            candidate_summaries.append({
                "candidate_index": idx,
                "option_name": cand.name,
                "transfers_count": cand.transfers_count,
                "transfers": tx_list if tx_list else ["Roll / No transfers"],
                "formation": cand.formation,
                "starters": starter_names,
                "captain": c_name,
                "vice_captain": vc_name,
                "vice_captain_strategy": cand.vice_captain_strategy,
                "bench_order": bench_summary,
                "gross_xp": cand.gross_xp,
                "hit_cost": cand.hit_cost,
                "net_xp": cand.net_xp,
                "multi_gw_xp": cand.multi_gw_xp,
                "strategic_value_score": cand.strategic_value_score,
                "bank_remaining_m": cand.bank_remaining_m,
                "xp_data_flags": xp_flags if xp_flags else [],
            })

        # Format Threat Matrix
        threats_summary = {}
        if league_analysis and league_analysis.total_managers > 0:
            threats_summary = {
                "league_name": league_analysis.league_name,
                "total_rival_managers": league_analysis.total_managers,
                "rival_captains": league_analysis.captain_distribution,
                "shields_owned_high_eo": [f"{p.web_name} ({p.eo_percent}% EO)" for p in league_analysis.threat_matrix.shields],
                "vulnerabilities_unowned_high_eo": [f"{p.web_name} ({p.eo_percent}% EO)" for p in league_analysis.threat_matrix.vulnerabilities],
                "daggers_owned_differentials": [f"{p.web_name} ({p.eo_percent}% EO)" for p in league_analysis.threat_matrix.daggers],
            }

        # Format Live News & Press Conference Intel
        news_list: List[str] = []
        if news_intel:
            items = news_intel.values() if isinstance(news_intel, dict) else news_intel
            for item in items:
                if hasattr(item, "summary"):
                    if item.risk_level != "CLEARED" or item.official_fpl_news or item.live_search_snippets:
                        news_list.append(item.summary())
                elif isinstance(item, dict):
                    news_list.append(str(item))

        max_idx = len(candidates) - 1
        risk_mode = (competitive_context or {}).get("risk_mode", "NEUTRAL")
        risk_note = (competitive_context or {}).get("risk_mode_note", "")

        manager_decryption_rubric = {
            "high_rotation_or_cameo_risk": "Phrases like 'felt something in training', 'assess in the morning', 'touch and go', 'illness', 'managing his load', 'European game coming'. Action: Avoid captaincy; ensure reliable vice-captain; favor alternative candidates.",
            "cleared_solid_starter": "Phrases like 'trained fully all week', 'ready to go', 'cleared by medical team', 'no concerns'. Action: Full confidence.",
            "emergency_veto_rule": "If breaking news reveals a target in the solver's top candidates is definitively OUT or benched, specify veto_player_ids to trigger an instant mathematical re-solve."
        }

        forced_chip = (chip_season_plan or {}).get("forced_chip")
        chip_directive = ""
        if forced_chip:
            chip_directive = (
                f"MANDATORY CHIP DIRECTIVE: The '{forced_chip.upper()}' chip has been user-forced for this gameweek. "
                f"You MUST select the candidate move corresponding to the {forced_chip.upper()} strategy. "
            )

        prompt = {
            "instruction": (
                "You are an elite Director of Football and veteran Fantasy Premier League strategist. "
                "Evaluate the mathematical MILP candidates, mini-league threat dynamics, Moneyball statistical metrics, and breaking press conference / injury news. "
                f"Select the single best move option (candidate_index: 0 to {max_idx}). "
                f"CRITICAL: The current competitive risk mode is '{risk_mode}'. {risk_note} "
                f"{chip_directive}"
                "If risk_mode is DEFEND: strongly prefer rolling, avoid hits, protect shields. "
                "If risk_mode is CHASE: actively consider high-EV Moneyball differentials (marked [MONEYBALL BUY] with high xGI and low EO%) to close point deficits against mini-league leaders. "
                "If risk_mode is NEUTRAL: balance xP optimisation with risk management. "
                "Decide whether to accept or override the solver's Captain / Vice-Captain based on manager press conference quotes. "
                "If breaking news reveals that a transfer target is unexpectedly injured or benched, you may issue a 'veto_player_ids' to trigger a clean re-solve. "
                "Provide your tactical rationale in EXACTLY two concise, impactful sentences balancing "
                "expected points (xP), injury safety from latest news, defensive shielding vs differential upside, "
                "and your competitive position. "
                "Respond in strictly valid JSON format matching the schema."
            ),
            "manager_decryption_rubric": manager_decryption_rubric,
            "competitive_context": competitive_context or {},
            "chip_season_plan": chip_season_plan or {},
            "candidates": candidate_summaries,
            "league_threats": threats_summary,
            "breaking_news_and_injuries": news_list[:15],
            "schema": {
                "selected_candidate_index": f"int (0 to {max_idx})",
                "captain_override": "optional str (web_name of starter to captain, or null)",
                "vice_captain_override": "optional str (web_name of starter to vice-captain, or null)",
                "veto_player_ids": "optional list[int] (player IDs to veto due to late breaking news)",
                "veto_reason": "optional str",
                "rationale": "str (concise 2-sentence tactical rationale)"
            }
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional FPL Tactical Decision Director. Output valid JSON only."},
                {"role": "user", "content": json.dumps(prompt, indent=2)}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        parsed = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                resp.raise_for_status()
                resp_data = resp.json()
                content_str = resp_data["choices"][0]["message"]["content"]
                parsed = json.loads(content_str)
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
                if attempt < self.max_retries:
                    sleep_time = attempt * 2
                    logger.warning(
                        f"LLM Director attempt {attempt}/{self.max_retries} failed ({net_err}). "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"LLM Director call timed out / connection failed after {self.max_retries} attempts ({net_err}). Engaging automated mathematical fallback.")
                    return self._build_fallback_decision(optimization_result, f"LLM timeout/network error: {net_err}", news_intel)
            except requests.exceptions.HTTPError as http_err:
                status_code = resp.status_code if 'resp' in locals() and resp is not None else 0
                if status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    sleep_time = attempt * 2
                    logger.warning(
                        f"LLM Director attempt {attempt}/{self.max_retries} returned HTTP {status_code}. "
                        f"Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"LLM Director HTTP error {status_code} ({http_err}). Engaging automated mathematical fallback.")
                    return self._build_fallback_decision(optimization_result, f"LLM HTTP error: {http_err}", news_intel)
            except Exception as exc:
                logger.warning(f"LLM Director call failed ({exc}). Engaging automated mathematical fallback.")
                return self._build_fallback_decision(optimization_result, f"LLM error: {exc}", news_intel)

        if not parsed:
            return self._build_fallback_decision(optimization_result, "Empty or invalid LLM response", news_intel)

        chosen_idx = int(parsed.get("selected_candidate_index", 0))
        if chosen_idx < 0 or chosen_idx >= len(candidates):
            chosen_idx = 0
        rationale_text = str(parsed.get("rationale") or "").strip()

        captain_override = parsed.get("captain_override")
        vice_captain_override = parsed.get("vice_captain_override")
        raw_veto_ids = parsed.get("veto_player_ids") or []
        veto_player_ids = [int(pid) for pid in raw_veto_ids if str(pid).isdigit()]
        veto_reason = parsed.get("veto_reason")

        chosen = candidates[chosen_idx]

        # Process Captain Override if valid starter in chosen squad
        if captain_override and isinstance(captain_override, str):
            target_norm = captain_override.strip().lower()
            for p in chosen.starters:
                if p.web_name.lower() == target_norm:
                    for s in chosen.starters:
                        s.is_captain = (s.id == p.id)
                    chosen.captain = p
                    logger.info(f"AI Director overrode captain to: {p.web_name}")
                    break

        # Process Vice-Captain Override if valid starter in chosen squad
        if vice_captain_override and isinstance(vice_captain_override, str):
            target_norm = vice_captain_override.strip().lower()
            for p in chosen.starters:
                if p.web_name.lower() == target_norm and not p.is_captain:
                    for s in chosen.starters:
                        s.is_vice_captain = (s.id == p.id)
                    chosen.vice_captain = p
                    logger.info(f"AI Director overrode vice-captain to: {p.web_name}")
                    break

        c_name = chosen.captain.web_name if chosen.captain else "Unknown"
        vc_name = chosen.vice_captain.web_name if chosen.vice_captain else "Unknown"

        if chosen.transfers_count == 0:
            tx_desc = "No transfers (Roll/Bank transfer)."
        else:
            tx_desc = ", ".join([f"{t.player_out.web_name} ➔ {t.player_in.web_name}" for t in chosen.transfers])

        if not rationale_text:
            rationale_text = (
                f"Selected {chosen.name} to maximize expected points ({chosen.net_xp} net xP). "
                f"Captaincy assigned to {c_name} to capitalize on favorable fixture metrics."
            )

        logger.info(f"AI Director decision received: {chosen.name}")
        news_alerts = self._extract_candidate_news_alerts(chosen, news_intel)

        return DecisionOutput(
            selected_candidate_index=chosen_idx,
            selected_candidate=chosen,
            chosen_move_name=chosen.name,
            transfers_description=tx_desc,
            captain_name=c_name,
            vice_captain_name=vc_name,
            projected_net_xp=chosen.net_xp,
            rationale=rationale_text,
            source="LLM_DIRECTOR",
            news_alerts=news_alerts,
            captain_override=captain_override,
            vice_captain_override=vice_captain_override,
            veto_player_ids=veto_player_ids,
            veto_reason=veto_reason,
        )

