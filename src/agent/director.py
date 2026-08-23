"""AI Decision Director for evaluating MILP candidates with Game Theory & Threat Matrix."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel, Field

from src.engine.optimizer import CandidateSquad, OptimizationResult
from src.tracker.league_scanner import LeagueAnalysis

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


class AIDirector:
    """Evaluates optimizer candidate moves against mini-league threat dynamics using LLM."""

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

    def _build_fallback_decision(
        self,
        optimization_result: OptimizationResult,
        reason: str = "Automated mathematical optimization",
    ) -> DecisionOutput:
        """Fallback to the candidate with highest net xP."""
        candidates = optimization_result.candidates
        if not candidates:
            raise ValueError("No candidates available for fallback decision.")

        # Sort candidates by net xP descending
        best_candidate_idx = 0
        best_net_xp = -999.0
        for idx, cand in enumerate(candidates):
            if cand.net_xp > best_net_xp:
                best_net_xp = cand.net_xp
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
        )

    def evaluate_and_decide(
        self,
        optimization_result: OptimizationResult,
        league_analysis: Optional[LeagueAnalysis] = None,
    ) -> DecisionOutput:
        """
        Send prompt to LLM containing solver candidates & league threat matrix.
        Fallback to PuLP top net xP candidate on error or missing API key.
        """
        candidates = optimization_result.candidates
        if not candidates:
            raise ValueError("Optimization result contains 0 candidates.")

        if not self.api_key:
            logger.info("LLM_API_KEY not configured. Using deterministic mathematical MILP solver.")
            return self._build_fallback_decision(optimization_result, "No API key configured")

        # Format candidates context for LLM
        candidate_summaries = []
        for idx, cand in enumerate(candidates):
            c_name = cand.captain.web_name if cand.captain else "N/A"
            vc_name = cand.vice_captain.web_name if cand.vice_captain else "N/A"
            tx_list = [f"OUT: {t.player_out.web_name} (£{t.player_out.cost_m}m) ➔ IN: {t.player_in.web_name} (£{t.player_in.cost_m}m)" for t in cand.transfers]
            candidate_summaries.append({
                "candidate_index": idx,
                "option_name": cand.name,
                "transfers_count": cand.transfers_count,
                "transfers": tx_list if tx_list else ["Roll / No transfers"],
                "formation": cand.formation,
                "captain": c_name,
                "vice_captain": vc_name,
                "gross_xp": cand.gross_xp,
                "hit_cost": cand.hit_cost,
                "net_xp": cand.net_xp,
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

        prompt = {
            "instruction": (
                "You are an elite Director of Football and veteran Fantasy Premier League strategist. "
                "Evaluate the following mathematical MILP candidates and mini-league threat dynamics. "
                "Select the single best move option (candidate_index: 0, 1, or 2). "
                "Provide your tactical rationale in EXACTLY two concise, impactful sentences balancing "
                "expected points (xP) with defensive shielding or differential attacking upside. "
                "Respond in strictly valid JSON format matching the schema."
            ),
            "candidates": candidate_summaries,
            "league_threats": threats_summary,
            "schema": {
                "selected_candidate_index": "int (0, 1, or 2)",
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
            )

        except Exception as exc:
            logger.warning(f"LLM Director call failed ({exc}). Engaging automated mathematical fallback.")
            return self._build_fallback_decision(optimization_result, f"LLM error: {exc}")
