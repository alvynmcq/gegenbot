"""AI Decision Director for evaluating MILP candidates with Game Theory, Threat Matrix & Live News Intelligence."""

import json
import logging
import os
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


class AIDirector:
    """Evaluates optimizer candidate moves against mini-league threat dynamics and live press conference news."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 15,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "").strip()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

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

        # Sort candidates by strategic value score descending (accounting for FT banking and hit hurdle)
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

        logger.info(f"Fallback decision made: Option {best_candidate_idx + 1} ({chosen.name})")

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
    ) -> DecisionOutput:
        """
        Send prompt to LLM containing solver candidates, mini-league threat matrix,
        and live press conference / injury intelligence.
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
            bench_summary = [f"{p.web_name} (Sub #{p.bench_order})" for p in cand.bench]
            candidate_summaries.append({
                "candidate_index": idx,
                "option_name": cand.name,
                "transfers_count": cand.transfers_count,
                "transfers": tx_list if tx_list else ["Roll / No transfers"],
                "formation": cand.formation,
                "captain": c_name,
                "vice_captain": vc_name,
                "bench_order": bench_summary,
                "gross_xp": cand.gross_xp,
                "hit_cost": cand.hit_cost,
                "net_xp": cand.net_xp,
                "multi_gw_xp": cand.multi_gw_xp,
                "strategic_value_score": cand.strategic_value_score,
                "bank_remaining_m": cand.bank_remaining_m,
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
        prompt = {
            "instruction": (
                "You are an elite Director of Football and veteran Fantasy Premier League strategist. "
                "Evaluate the mathematical MILP candidates, mini-league threat dynamics, and breaking press conference / injury news. "
                f"Select the single best move option (candidate_index: 0 to {max_idx}). "
                "Carefully consider alternative moves if the top-ranked transfer target carries late injury, rotation, or minutes risk from press conferences. "
                "Provide your tactical rationale in EXACTLY two concise, impactful sentences balancing "
                "expected points (xP), injury safety from latest news, and defensive shielding vs differential upside. "
                "Respond in strictly valid JSON format matching the schema."
            ),
            "candidates": candidate_summaries,
            "league_threats": threats_summary,
            "breaking_news_and_injuries": news_list[:15],
            "schema": {
                "selected_candidate_index": f"int (0 to {max_idx})",
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

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            resp.raise_for_status()
            resp_data = resp.json()
            content_str = resp_data["choices"][0]["message"]["content"]
            parsed = json.loads(content_str)

            chosen_idx = int(parsed.get("selected_candidate_index", 0))
            if chosen_idx < 0 or chosen_idx >= len(candidates):
                chosen_idx = 0
            rationale_text = parsed.get("rationale", "").strip()

            chosen = candidates[chosen_idx]
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

            logger.info(f"AI Director decision received: Option {chosen_idx + 1} ({chosen.name})")

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
            )

        except Exception as exc:
            logger.warning(f"LLM Director call failed ({exc}). Engaging automated mathematical fallback.")
            return self._build_fallback_decision(optimization_result, f"LLM error: {exc}", news_intel)
