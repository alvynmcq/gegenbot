"""MILP Squad and Lineup Optimizer for Fantasy Premier League using PuLP.
Features:
- Real FPL selling price accounting (half-profit rule)
- Multi-gameweek horizon lookahead (immediate GW + 3-GW fixture swing)
- Rolling transfer valuation (strategic free transfer banking bonus)
- Game-theoretic Effective Ownership (EO%) shielding integration
- Injury discounting and strict bench priority ordering
- Double Gameweek (DGW) & Blank Gameweek (BGW) chip guards
- Vice-Captain 100% availability safety rule
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import pulp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def get_solver(msg: bool = False, time_limit: Optional[int] = None) -> pulp.LpSolver:
    """
    Return the optimal MILP solver backend:
    1. Gurobi (if SOLVER_BACKEND=gurobi or gurobipy is installed and licensed)
    2. HiGHS (highspy direct API or HiGHS_CMD binary) — state-of-the-art open-source default
    3. PULP_CBC_CMD (graceful fallback)
    """
    backend = os.getenv("SOLVER_BACKEND", "highs").lower().strip()

    # 1. Attempt Gurobi if explicitly requested or if gurobipy is available
    if backend == "gurobi":
        try:
            if hasattr(pulp, "GUROBI"):
                solver = pulp.GUROBI(msg=msg, timeLimit=time_limit)
                if solver.available():
                    return solver
            if hasattr(pulp, "GUROBI_CMD"):
                solver = pulp.GUROBI_CMD(msg=msg, timeLimit=time_limit)
                if solver.available():
                    return solver
        except Exception as exc:
            logger.debug(f"Gurobi solver requested but not available ({exc}). Falling back to HiGHS.")

    # 2. Attempt HiGHS direct Python API (Default open-source solver)
    try:
        if hasattr(pulp, "getSolver"):
            solver = pulp.getSolver("HiGHS", msg=msg, timeLimit=time_limit)
            if solver.available():
                return solver
    except Exception as exc:
        logger.debug(f"HiGHS solver via getSolver not available: {exc}")

    # 3. Attempt HiGHS_CMD binary solver
    try:
        if hasattr(pulp, "HiGHS_CMD"):
            solver = pulp.HiGHS_CMD(msg=msg, timeLimit=time_limit)
            if solver.available():
                return solver
    except Exception as exc:
        logger.debug(f"HiGHS_CMD binary solver not available: {exc}")

    # 4. Graceful fallback to PULP_CBC_CMD
    logger.debug("Using PULP_CBC_CMD solver fallback.")
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)


def _solve_problem(prob: pulp.LpProblem, primary_solver: Optional[pulp.LpSolver] = None) -> int:
    """Solve LpProblem with primary solver (HiGHS) and automatic fallback to CBC on exception."""
    solver = primary_solver or get_solver(msg=False)
    try:
        status = prob.solve(solver)
        return status
    except Exception as exc:
        logger.warning(f"Solver {solver} failed with exception ({exc}). Retrying with PULP_CBC_CMD.")
        fallback = pulp.PULP_CBC_CMD(msg=False)
        return prob.solve(fallback)


def get_player_injury_multiplier(
    status: Optional[str] = "a",
    chance: Optional[int] = None,
) -> float:
    """
    Calculate availability/injury multiplier based on FPL status and chance of playing.
    - status == 'a' or chance == 100 or chance is None: multiplier = 1.0
    - chance == 75: multiplier = 0.80 (or INJURY_XP_MULTIPLIER_75)
    - chance == 50: multiplier = 0.40 (or INJURY_XP_MULTIPLIER_50)
    - chance == 25: multiplier = 0.10 (or INJURY_XP_MULTIPLIER_25)
    - chance == 0 or status in ['i', 's', 'u']: multiplier = 0.0 (or INJURY_XP_MULTIPLIER_0)
    """
    stat = (status or "a").strip().lower()

    if stat in ["i", "s", "u"]:
        return float(os.getenv("INJURY_XP_MULTIPLIER_0", "0.0"))

    if chance is not None:
        try:
            c = int(chance)
            if c <= 0:
                return float(os.getenv("INJURY_XP_MULTIPLIER_0", "0.0"))
            elif c <= 25:
                return float(os.getenv("INJURY_XP_MULTIPLIER_25", "0.10"))
            elif c <= 50:
                return float(os.getenv("INJURY_XP_MULTIPLIER_50", "0.40"))
            elif c <= 75:
                return float(os.getenv("INJURY_XP_MULTIPLIER_75", "0.80"))
            else:
                return 1.0
        except (ValueError, TypeError):
            pass

    if stat == "d":
        return float(os.getenv("INJURY_XP_MULTIPLIER_50", "0.40"))

    return 1.0


class PlayerPick(BaseModel):
    """Represents a player in a candidate squad."""
    id: int
    web_name: str
    position: str
    element_type: int
    team_name: str
    team_code: str
    cost_m: float
    selling_price_m: Optional[float] = None
    xp: float
    raw_xp: Optional[float] = None
    discounted_xp: Optional[float] = None
    xp_3gw: Optional[float] = None
    injury_multiplier: float = 1.0
    status: Optional[str] = "a"
    chance_of_playing_next_round: Optional[int] = None
    fplreview_xp: Optional[float] = None
    xp_source: Optional[str] = None
    xp_confidence: str = "HIGH"  # "HIGH", "MEDIUM", "LOW" — based on source agreement
    price_momentum: float = 0.0  # -1.0 (falling) to +1.0 (rising); from event transfer volume
    rolling_xgi_90: Optional[float] = None  # Expected Goal Involvements per 90 from Vaastav dataset
    rolling_xgc_90: Optional[float] = None  # Expected Goals Conceded per 90 from Vaastav dataset
    minutes_reliability: Optional[str] = None  # "HIGH", "MEDIUM", "LOW" based on recent start/sub frequency
    starts_ratio: Optional[float] = None  # Ratio of recent matches started
    is_starter: bool = False
    is_captain: bool = False
    is_vice_captain: bool = False
    bench_order: Optional[int] = None  # 0 for Sub GK (Pick 12), 1-3 for outfield bench (Picks 13-15)
    fixtures_in_gw: int = 1


class TransferMove(BaseModel):
    """Represents a single transfer out and in."""
    player_out: PlayerPick
    player_in: PlayerPick


class CandidateSquad(BaseModel):
    """Full squad selection with transfers, lineup, captaincy, and net expected points."""
    name: str  # e.g., "Roll Transfer (0 Transfers)", "Best 1-Transfer Move", "Wildcard Squad"
    transfers_count: int
    transfers: List[TransferMove] = Field(default_factory=list)
    starters: List[PlayerPick] = Field(default_factory=list)
    bench: List[PlayerPick] = Field(default_factory=list)
    captain: Optional[PlayerPick] = None
    vice_captain: Optional[PlayerPick] = None
    formation: str  # e.g. "3-4-3"
    gross_xp: float
    hit_cost: int
    net_xp: float
    multi_gw_xp: float = 0.0
    strategic_value_score: float = 0.0  # Net xP + Rolling transfer bonus + Multi-GW outlook
    total_cost_m: float
    bank_remaining_m: float
    active_chip: Optional[str] = None  # e.g. "wildcard", "freehit", "bboost", "3xc"
    vice_captain_strategy: str = ""  # e.g. "Safety Net (Haaland VC backs differential Mbeumo C)"


class ChipEvaluation(BaseModel):
    """Evaluation result for an individual FPL chip."""
    chip_name: str  # "wildcard", "freehit", "bboost", "3xc"
    display_name: str  # "Wildcard", "Free Hit", "Bench Boost", "Triple Captain"
    projected_xp: float
    baseline_xp: float
    xp_gain: float
    threshold: float
    threshold_met: bool
    squad_candidate: Optional[CandidateSquad] = None
    reason: str = ""


class ChipEvaluationResult(BaseModel):
    """Aggregated evaluations for all FPL chips."""
    evaluations: Dict[str, ChipEvaluation] = Field(default_factory=dict)
    recommended_chip: Optional[str] = None  # None, "wildcard", "freehit", "bboost", "3xc"
    recommendation_reason: Optional[str] = None


class OptimizationResult(BaseModel):
    """Result containing top candidates generated by optimizer."""
    candidates: List[CandidateSquad] = Field(default_factory=list)
    current_team_value_m: float
    bank_m: float
    free_transfers: int
    chip_evaluation: Optional[ChipEvaluationResult] = None


def _parse_target_gws(env_var_name: str) -> List[int]:
    """Parse comma-separated gameweek targets from environment variable."""
    raw = os.getenv(env_var_name, "").strip()
    if not raw:
        return []
    targets: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            targets.append(int(part))
    return targets


def calculate_decay_weights(horizon_length: int = 3, decay_factor: float = 0.85) -> Tuple[List[float], float]:
    """
    Calculate exponential decay weights [gamma^0, gamma^1, ..., gamma^(N-1)] and their sum.
    e.g. For horizon=3, decay=0.85: [1.0, 0.85, 0.7225], sum = 2.5725.
    """
    n = max(1, horizon_length)
    weights = [round(decay_factor ** t, 4) for t in range(n)]
    return weights, round(sum(weights), 4)


class FPLOptimizer:
    """Advanced MILP solver for FPL squad optimization with Multi-Period Horizon and Game Theory."""

    def __init__(
        self,
        players_df: pd.DataFrame,
        decay_factor: Optional[float] = None,
        horizon_length: Optional[int] = None,
        bench_weight: Optional[float] = None,
    ):
        """
        Initialize optimizer with player DataFrame and apply injury/rotation status discounting.
        Incorporates configurable horizon decay factor (gamma), horizon lookahead length (N),
        and bench strength auto-sub weighting factor.
        Expected columns: id, web_name, element_type, position, team_id, team_name, team_code, cost_m, xp
        Optional columns: xp_3gw, status, chance_of_playing_next_round, fixtures_in_gw
        """
        self.df = players_df.copy().reset_index(drop=True)
        self.decay_factor = float(decay_factor if decay_factor is not None else os.getenv("DECAY_FACTOR", "0.85"))
        self.horizon_length = int(horizon_length if horizon_length is not None else os.getenv("HORIZON_LENGTH", "3"))
        self.bench_weight = float(
            bench_weight if bench_weight is not None else os.getenv("BENCH_WEIGHT_FACTOR", os.getenv("SUB_FACTOR", "0.10"))
        )
        self.decay_weights, self.decay_sum = calculate_decay_weights(self.horizon_length, self.decay_factor)

        if "raw_xp" not in self.df.columns:
            self.df["raw_xp"] = self.df["xp"]

        if "xp_3gw" not in self.df.columns:
            self.df["xp_3gw"] = self.df["xp"] * self.decay_sum

        if "fixtures_in_gw" not in self.df.columns:
            self.df["fixtures_in_gw"] = 1

        multipliers = []
        discounted_xps = []
        discounted_3gw = []
        for _, row in self.df.iterrows():
            stat = row.get("status", "a")
            chance = row.get("chance_of_playing_next_round")
            mult = get_player_injury_multiplier(stat, chance)
            multipliers.append(mult)
            raw = float(row.get("raw_xp", row.get("xp", 0.0)))
            disc = round(raw * mult, 2)
            discounted_xps.append(disc)
            raw_3gw = float(row.get("xp_3gw", raw * self.decay_sum))
            discounted_3gw.append(round(raw_3gw * mult, 2))

        self.df["injury_multiplier"] = multipliers
        self.df["discounted_xp"] = discounted_xps
        self.df["discounted_3gw"] = discounted_3gw
        self.df["xp"] = discounted_xps

        self.player_map: Dict[int, Dict[str, Any]] = {
            row["id"]: row.to_dict() for _, row in self.df.iterrows()
        }

    def _build_player_pick(
        self,
        player_id: int,
        is_starter: bool = False,
        is_captain: bool = False,
        is_vice_captain: bool = False,
        bench_order: Optional[int] = None,
        selling_price_m: Optional[float] = None,
    ) -> PlayerPick:
        info = self.player_map[player_id]
        rev_xp = info.get("fplreview_xp")
        raw_xp = float(info.get("raw_xp", info["xp"]))
        disc_xp = float(info.get("discounted_xp", info["xp"]))
        xp_3gw_val = float(info.get("discounted_3gw", disc_xp * self.decay_sum))
        mult = float(info.get("injury_multiplier", 1.0))
        chance_val = info.get("chance_of_playing_next_round")
        chance_int = int(chance_val) if chance_val is not None and not pd.isna(chance_val) else None
        n_fix = int(info.get("fixtures_in_gw", 1))

        # xP confidence: compare external projection vs FPL ep_next
        ep_next_val = info.get("ep_next")
        xp_confidence = "HIGH"
        if rev_xp is not None and not pd.isna(rev_xp) and ep_next_val is not None and not pd.isna(ep_next_val):
            ep_next_f = float(ep_next_val)
            rev_xp_f = float(rev_xp)
            if ep_next_f > 0 and rev_xp_f > 0:
                divergence = abs(rev_xp_f - ep_next_f) / max(ep_next_f, rev_xp_f)
                if divergence > 0.50:
                    xp_confidence = "LOW"
                elif divergence > 0.30:
                    xp_confidence = "MEDIUM"
        elif rev_xp is None or (rev_xp is not None and pd.isna(rev_xp)):
            # No external projection — relying on FPL ep_next alone
            xp_confidence = "MEDIUM"

        price_momentum_val = float(info.get("price_momentum", 0.0))
        rolling_xgi_val = float(info["rolling_xgi_90"]) if "rolling_xgi_90" in info and not pd.isna(info["rolling_xgi_90"]) else None
        rolling_xgc_val = float(info["rolling_xgc_90"]) if "rolling_xgc_90" in info and not pd.isna(info["rolling_xgc_90"]) else None
        minutes_reliability_val = str(info["minutes_reliability"]) if "minutes_reliability" in info and info["minutes_reliability"] else None
        starts_ratio_val = float(info["starts_ratio"]) if "starts_ratio" in info and not pd.isna(info["starts_ratio"]) else None

        return PlayerPick(
            id=player_id,
            web_name=info["web_name"],
            position=info["position"],
            element_type=int(info["element_type"]),
            team_name=info["team_name"],
            team_code=info["team_code"],
            cost_m=float(info["cost_m"]),
            selling_price_m=selling_price_m,
            xp=disc_xp,
            raw_xp=raw_xp,
            discounted_xp=disc_xp,
            xp_3gw=xp_3gw_val,
            injury_multiplier=mult,
            status=info.get("status", "a"),
            chance_of_playing_next_round=chance_int,
            fplreview_xp=float(rev_xp) if rev_xp is not None and not pd.isna(rev_xp) else None,
            xp_source=info.get("xp_source"),
            xp_confidence=xp_confidence,
            price_momentum=price_momentum_val,
            rolling_xgi_90=rolling_xgi_val,
            rolling_xgc_90=rolling_xgc_val,
            minutes_reliability=minutes_reliability_val,
            starts_ratio=starts_ratio_val,
            is_starter=is_starter,
            is_captain=is_captain,
            is_vice_captain=is_vice_captain,
            bench_order=bench_order,
            fixtures_in_gw=n_fix,
        )

    def select_initial_squad(self, budget_m: float = 100.0) -> List[int]:
        """Generate a valid, optimal 15-player initial squad respecting all FPL rules and budget."""
        prob = pulp.LpProblem("FPL_Initial_Squad", pulp.LpMaximize)
        n_players = len(self.df)
        indices = list(range(n_players))

        squad_vars = pulp.LpVariable.dicts("init_squad", indices, cat=pulp.LpBinary)
        costs = self.df["cost_m"].tolist()
        xps = self.df["xp"].tolist()
        elem_types = self.df["element_type"].tolist()
        team_ids = self.df["team_id"].tolist()
        player_ids = self.df["id"].tolist()

        prob += pulp.lpSum([squad_vars[i] for i in indices]) == 15
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 1]) == 2
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 2]) == 5
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 3]) == 5
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 4]) == 3

        for t in set(team_ids):
            prob += pulp.lpSum([squad_vars[i] for i in indices if team_ids[i] == t]) <= 3

        prob += pulp.lpSum([costs[i] * squad_vars[i] for i in indices]) <= budget_m
        prob += pulp.lpSum([squad_vars[i] * xps[i] for i in indices])

        status = _solve_problem(prob)

        if status == pulp.LpStatusOptimal:
            selected_ids = [player_ids[i] for i in indices if pulp.value(squad_vars[i]) > 0.5]
            return selected_ids

        # Fallback greedy selection
        selected_ids: List[int] = []
        team_counts: Dict[int, int] = {}
        for pos_type, quota in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            candidates = self.df[self.df["element_type"] == pos_type].sort_values("xp", ascending=False)
            count = 0
            for _, row in candidates.iterrows():
                pid = int(row["id"])
                tid = int(row["team_id"])
                if team_counts.get(tid, 0) < 3:
                    selected_ids.append(pid)
                    team_counts[tid] = team_counts.get(tid, 0) + 1
                    count += 1
                    if count == quota:
                        break
        return selected_ids

    def _assign_lineup_and_bench(
        self,
        selected_squad_ids: Set[int],
        selected_starter_indices: List[int],
        indices: List[int],
        player_ids: List[int],
        elem_types: List[int],
        captain_idx: int,
        selling_prices: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[PlayerPick], List[PlayerPick], PlayerPick, PlayerPick, str]:
        """
        Build Starters (Picks 1-11), Bench (Picks 12-15), Captain, Vice-Captain, and Formation.
        """
        sp_map = selling_prices or {}

        # 1. Build Starters
        starters: List[PlayerPick] = []
        def_count = 0
        mid_count = 0
        fwd_count = 0
        for i in selected_starter_indices:
            p_id = player_ids[i]
            pos_type = elem_types[i]
            if pos_type == 2:
                def_count += 1
            elif pos_type == 3:
                mid_count += 1
            elif pos_type == 4:
                fwd_count += 1
            starters.append(self._build_player_pick(
                p_id,
                is_starter=True,
                selling_price_m=sp_map.get(p_id),
            ))

        formation = f"{def_count}-{mid_count}-{fwd_count}"

        # Captain
        captain_pid = player_ids[captain_idx]
        captain_pick: Optional[PlayerPick] = None
        for p in starters:
            if p.id == captain_pid:
                p.is_captain = True
                captain_pick = p
                break
        if captain_pick is None:
            captain_pick = starters[0]
            captain_pick.is_captain = True

        # 2. Vice-Captain with strict 100% availability rule:
        outfield_starters = [p for p in starters if p.element_type != 1 and p.id != captain_pick.id]
        safe_candidates = [
            p for p in outfield_starters
            if p.injury_multiplier == 1.0 and (p.chance_of_playing_next_round is None or p.chance_of_playing_next_round == 100)
        ]

        if safe_candidates:
            vc_target = max(safe_candidates, key=lambda p: (p.xp, p.raw_xp or 0.0))
        elif outfield_starters:
            vc_target = max(outfield_starters, key=lambda p: (p.injury_multiplier, p.xp))
        else:
            vc_target = starters[0]

        for p in starters:
            if p.id == vc_target.id:
                p.is_vice_captain = True
                vc_pick = p
                break
        else:
            vc_pick = starters[0]

        # 3. Bench Priority Ordering (Picks 12 to 15):
        bench_indices = [i for i in indices if player_ids[i] in selected_squad_ids and i not in selected_starter_indices]
        bench_gkp_idx = next(i for i in bench_indices if elem_types[i] == 1)
        bench_outfield_indices = [i for i in bench_indices if elem_types[i] != 1]

        def outfield_bench_sort_key(idx: int):
            row = self.player_map[player_ids[idx]]
            disc_xp = float(row.get("discounted_xp", row["xp"]))
            fdr = float(row.get("fdr_next", 3.0))
            form = float(row.get("form", 0.0) or 0.0)
            cost = float(row.get("cost_m", 5.0))
            return (-disc_xp, -fdr, -form, -cost)

        bench_outfield_indices.sort(key=outfield_bench_sort_key)

        bench: List[PlayerPick] = []
        bench.append(self._build_player_pick(
            player_ids[bench_gkp_idx],
            is_starter=False,
            bench_order=0,
            selling_price_m=sp_map.get(player_ids[bench_gkp_idx]),
        ))
        for order_idx, b_idx in enumerate(bench_outfield_indices, start=1):
            p_id = player_ids[b_idx]
            bench.append(self._build_player_pick(
                p_id,
                is_starter=False,
                bench_order=order_idx,
                selling_price_m=sp_map.get(p_id),
            ))

        return starters, bench, captain_pick, vc_pick, formation

    def _solve_optimal_squad_from_scratch(
        self,
        total_budget_m: float,
        current_squad_ids: Optional[Set[int]] = None,
        chip_name: Optional[str] = None,
        squad_title: str = "Optimal Squad From Scratch",
        eo_weights: Optional[Dict[int, float]] = None,
    ) -> Optional[CandidateSquad]:
        """
        Solve MILP for a brand new 15-man squad from scratch (Wildcard / Free Hit) with 0 hits.
        """
        prob = pulp.LpProblem("FPL_Optimizer_Scratch", pulp.LpMaximize)

        n_players = len(self.df)
        indices = list(range(n_players))

        squad_vars = pulp.LpVariable.dicts("squad", indices, cat=pulp.LpBinary)
        starter_vars = pulp.LpVariable.dicts("starter", indices, cat=pulp.LpBinary)
        captain_vars = pulp.LpVariable.dicts("captain", indices, cat=pulp.LpBinary)

        costs = self.df["cost_m"].tolist()
        xps = self.df["xp"].tolist()
        elem_types = self.df["element_type"].tolist()
        team_ids = self.df["team_id"].tolist()
        player_ids = self.df["id"].tolist()
        multipliers = self.df["injury_multiplier"].tolist()

        # 1. Total squad size == 15
        prob += pulp.lpSum([squad_vars[i] for i in indices]) == 15, "TotalSquad15"

        # 2. Position quotas
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 1]) == 2, "SquadGK2"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 2]) == 5, "SquadDEF5"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 3]) == 5, "SquadMID5"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 4]) == 3, "SquadFWD3"

        # 3. Max 3 per team
        for t in set(team_ids):
            prob += pulp.lpSum([squad_vars[i] for i in indices if team_ids[i] == t]) <= 3, f"Max3Team_{t}"

        # 4. Budget
        prob += pulp.lpSum([costs[i] * squad_vars[i] for i in indices]) <= total_budget_m, "BudgetLimit"

        # 5. Starters constraints
        for i in indices:
            prob += starter_vars[i] <= squad_vars[i], f"StarterInSquad_{i}"
            if multipliers[i] == 0.0:
                prob += starter_vars[i] == 0, f"NoZeroMultiplierStarter_{i}"

        prob += pulp.lpSum([starter_vars[i] for i in indices]) == 11, "Starters11"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 1]) == 1, "StarterGK1"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 2]) >= 3, "StarterDEFMin3"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 2]) <= 5, "StarterDEFMax5"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 3]) >= 2, "StarterMIDMin2"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 3]) <= 5, "StarterMIDMax5"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 4]) >= 1, "StarterFWDMin1"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 4]) <= 3, "StarterFWDMax3"

        # 6. Captain
        for i in indices:
            prob += captain_vars[i] <= starter_vars[i], f"CaptainInStarter_{i}"

        prob += pulp.lpSum([captain_vars[i] for i in indices]) == 1, "OneCaptain"

        # 7. Objective with optional EO shield weighting and sub factor bench weighting
        eo_map = eo_weights or {}
        eo_boosts = [(eo_map.get(player_ids[i], 0.0) / 100.0) * 0.02 * xps[i] for i in indices]

        prob += pulp.lpSum([
            starter_vars[i] * (xps[i] + eo_boosts[i])
            + captain_vars[i] * xps[i]
            + (squad_vars[i] - starter_vars[i]) * self.bench_weight * xps[i]
            for i in indices
        ]), "TotalXP"

        status = _solve_problem(prob)

        if status != pulp.LpStatusOptimal:
            logger.warning("PuLP solver failed to find optimal solution from scratch.")
            return None

        selected_squad_indices = [i for i in indices if pulp.value(squad_vars[i]) > 0.5]
        selected_starter_indices = [i for i in indices if pulp.value(starter_vars[i]) > 0.5]
        captain_idx = next(i for i in indices if pulp.value(captain_vars[i]) > 0.5)

        selected_squad_ids = {player_ids[i] for i in selected_squad_indices}
        curr_set = current_squad_ids or set()
        transfers_out_ids = curr_set - selected_squad_ids
        transfers_in_ids = selected_squad_ids - curr_set

        actual_transfers_count = len(transfers_out_ids)

        transfers_out_sorted = sorted(list(transfers_out_ids), key=lambda pid: self.player_map[pid]["cost_m"])
        transfers_in_sorted = sorted(list(transfers_in_ids), key=lambda pid: self.player_map[pid]["cost_m"])
        transfers_list: List[TransferMove] = []
        for out_id, in_id in zip(transfers_out_sorted, transfers_in_sorted):
            transfers_list.append(TransferMove(
                player_out=self._build_player_pick(out_id),
                player_in=self._build_player_pick(in_id),
            ))

        starters, bench, captain_pick, vc_pick, formation = self._assign_lineup_and_bench(
            selected_squad_ids=selected_squad_ids,
            selected_starter_indices=selected_starter_indices,
            indices=indices,
            player_ids=player_ids,
            elem_types=elem_types,
            captain_idx=captain_idx,
        )

        starters_xp_sum = sum(p.xp for p in starters)
        gross_xp = round(starters_xp_sum + captain_pick.xp, 2)
        hit_cost = 0
        net_xp = gross_xp

        # Multi-GW decayed outlook sum (3-GW)
        multi_gw_sum = round(sum(p.xp_3gw or (p.xp * self.decay_sum) for p in starters) + (captain_pick.xp_3gw or (captain_pick.xp * self.decay_sum)), 2)

        total_cost_m = round(sum(self.player_map[pid]["cost_m"] for pid in selected_squad_ids), 2)
        bank_remaining_m = round(total_budget_m - total_cost_m, 2)

        return CandidateSquad(
            name=squad_title,
            transfers_count=actual_transfers_count,
            transfers=transfers_list,
            starters=starters,
            bench=bench,
            captain=captain_pick,
            vice_captain=vc_pick,
            formation=formation,
            gross_xp=gross_xp,
            hit_cost=hit_cost,
            net_xp=net_xp,
            multi_gw_xp=multi_gw_sum,
            strategic_value_score=net_xp,
            total_cost_m=total_cost_m,
            bank_remaining_m=bank_remaining_m,
            active_chip=chip_name,
        )

    def _solve_for_k_transfers(
        self,
        current_squad_ids: Set[int],
        total_budget_m: float,
        k_transfers: int,
        free_transfers: int = 1,
        selling_prices: Optional[Dict[int, float]] = None,
        eo_weights: Optional[Dict[int, float]] = None,
        multi_gw_weight: float = 0.30,
        rolling_bonus_xp: float = 1.50,
        excluded_transfers_in: Optional[List[Set[int]]] = None,
    ) -> Optional[CandidateSquad]:
        """
        Formulate and solve MILP optimization with exact FPL selling price math,
        multi-period horizon lookahead, and rolling transfer valuation.
        Supports integer cuts (no-good cuts) to exclude specific incoming transfer combinations.
        """
        prob = pulp.LpProblem(f"FPL_Optimizer_K{k_transfers}", pulp.LpMaximize)

        n_players = len(self.df)
        indices = list(range(n_players))

        squad_vars = pulp.LpVariable.dicts("squad", indices, cat=pulp.LpBinary)
        starter_vars = pulp.LpVariable.dicts("starter", indices, cat=pulp.LpBinary)
        captain_vars = pulp.LpVariable.dicts("captain", indices, cat=pulp.LpBinary)

        costs = self.df["cost_m"].tolist()
        xps = self.df["xp"].tolist()
        xps_3gw = self.df["discounted_3gw"].tolist()
        elem_types = self.df["element_type"].tolist()
        team_ids = self.df["team_id"].tolist()
        player_ids = self.df["id"].tolist()
        multipliers = self.df["injury_multiplier"].tolist()

        sp_map = selling_prices or {}

        # 1. Total squad size == 15
        prob += pulp.lpSum([squad_vars[i] for i in indices]) == 15, "TotalSquad15"

        # 2. Position quotas
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 1]) == 2, "SquadGK2"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 2]) == 5, "SquadDEF5"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 3]) == 5, "SquadMID5"
        prob += pulp.lpSum([squad_vars[i] for i in indices if elem_types[i] == 4]) == 3, "SquadFWD3"

        # 3. Max 3 per team
        for t in set(team_ids):
            prob += pulp.lpSum([squad_vars[i] for i in indices if team_ids[i] == t]) <= 3, f"Max3Team_{t}"

        # 4. Exact FPL Budget Math with Real Selling Prices:
        effective_costs = [
            sp_map.get(player_ids[i], costs[i]) if player_ids[i] in current_squad_ids else costs[i]
            for i in indices
        ]
        prob += pulp.lpSum([effective_costs[i] * squad_vars[i] for i in indices]) <= total_budget_m, "BudgetLimit"

        # 5. Starters constraints
        for i in indices:
            prob += starter_vars[i] <= squad_vars[i], f"StarterInSquad_{i}"

        # Forbid 0.0 multiplier players in Starting XI
        zero_mult_indices = [i for i in indices if multipliers[i] == 0.0]
        for i in zero_mult_indices:
            prob += starter_vars[i] == 0, f"NoZeroMultStarter_{i}"

        prob += pulp.lpSum([starter_vars[i] for i in indices]) == 11, "Starters11"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 1]) == 1, "StarterGK1"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 2]) >= 3, "StarterDEFMin3"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 2]) <= 5, "StarterDEFMax5"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 3]) >= 2, "StarterMIDMin2"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 3]) <= 5, "StarterMIDMax5"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 4]) >= 1, "StarterFWDMin1"
        prob += pulp.lpSum([starter_vars[i] for i in indices if elem_types[i] == 4]) <= 3, "StarterFWDMax3"

        # 6. Captain
        for i in indices:
            prob += captain_vars[i] <= starter_vars[i], f"CaptainInStarter_{i}"

        prob += pulp.lpSum([captain_vars[i] for i in indices]) == 1, "OneCaptain"

        # 7. Transfer Count Constraint
        current_indices = [i for i in indices if player_ids[i] in current_squad_ids]
        retained = pulp.lpSum([squad_vars[i] for i in current_indices])
        prob += (15 - retained) == k_transfers, f"ExactTransfers_{k_transfers}"

        # 8. Exclusion cuts for generating diverse alternative candidate moves
        if excluded_transfers_in:
            for cut_idx, excl_set in enumerate(excluded_transfers_in):
                if not excl_set:
                    continue
                excl_indices = [i for i in indices if player_ids[i] in excl_set]
                if excl_indices:
                    prob += (
                        pulp.lpSum([squad_vars[i] for i in excl_indices]) <= len(excl_indices) - 1,
                        f"ExclTransfersInCut_{cut_idx}",
                    )

        # 9. Objective function:
        eo_map = eo_weights or {}
        eo_boosts = [(eo_map.get(player_ids[i], 0.0) / 100.0) * 0.02 * xps[i] for i in indices]

        blended_xps = [
            round((1.0 - multi_gw_weight) * xps[i] + multi_gw_weight * (xps_3gw[i] / self.decay_sum), 3)
            for i in indices
        ]

        prob += pulp.lpSum([
            starter_vars[i] * (blended_xps[i] + eo_boosts[i])
            + captain_vars[i] * blended_xps[i]
            + (squad_vars[i] - starter_vars[i]) * self.bench_weight * blended_xps[i]
            for i in indices
        ]), "TotalXP"

        status = _solve_problem(prob)

        if status != pulp.LpStatusOptimal:
            prob.constraints[f"ExactTransfers_{k_transfers}"] = (15 - retained) <= k_transfers
            status = _solve_problem(prob)

            if status != pulp.LpStatusOptimal:
                for i in zero_mult_indices:
                    cname = f"NoZeroMultStarter_{i}"
                    if cname in prob.constraints:
                        del prob.constraints[cname]
                status = _solve_problem(prob)

            if status != pulp.LpStatusOptimal:
                logger.debug(f"PuLP solver found no feasible alternative for K={k_transfers}")
                return None

        selected_squad_indices = [i for i in indices if pulp.value(squad_vars[i]) > 0.5]
        selected_starter_indices = [i for i in indices if pulp.value(starter_vars[i]) > 0.5]
        captain_idx = next(i for i in indices if pulp.value(captain_vars[i]) > 0.5)

        selected_squad_ids = {player_ids[i] for i in selected_squad_indices}
        transfers_out_ids = current_squad_ids - selected_squad_ids
        transfers_in_ids = selected_squad_ids - current_squad_ids

        actual_transfers_count = len(transfers_out_ids)

        transfers_out_sorted = sorted(list(transfers_out_ids), key=lambda pid: self.player_map[pid]["cost_m"])
        transfers_in_sorted = sorted(list(transfers_in_ids), key=lambda pid: self.player_map[pid]["cost_m"])
        transfers_list: List[TransferMove] = []
        for out_id, in_id in zip(transfers_out_sorted, transfers_in_sorted):
            transfers_list.append(TransferMove(
                player_out=self._build_player_pick(out_id, selling_price_m=sp_map.get(out_id)),
                player_in=self._build_player_pick(in_id),
            ))

        starters, bench, captain_pick, vc_pick, formation = self._assign_lineup_and_bench(
            selected_squad_ids=selected_squad_ids,
            selected_starter_indices=selected_starter_indices,
            indices=indices,
            player_ids=player_ids,
            elem_types=elem_types,
            captain_idx=captain_idx,
            selling_prices=selling_prices,
        )

        starters_xp_sum = sum(p.xp for p in starters)
        gross_xp = round(starters_xp_sum + captain_pick.xp, 2)
        paid_transfers = max(0, actual_transfers_count - free_transfers)
        hit_cost = paid_transfers * 4
        net_xp = round(gross_xp - hit_cost, 2)

        multi_gw_sum = round(sum(p.xp_3gw or (p.xp * self.decay_sum) for p in starters) + (captain_pick.xp_3gw or (captain_pick.xp * self.decay_sum)), 2)

        roll_bonus = rolling_bonus_xp if actual_transfers_count == 0 else 0.0
        strategic_value_score = round(net_xp + roll_bonus + (multi_gw_sum / self.decay_sum) * 0.15, 2)

        total_cost_m = round(sum(effective_costs[i] for i in selected_squad_indices), 2)
        bank_remaining_m = round(total_budget_m - total_cost_m, 2)

        if actual_transfers_count == 0:
            name = f"Option 1: Roll / Bank Transfer (0 Moves · +{rolling_bonus_xp:.1f} FT Strategic Value)"
        elif actual_transfers_count == 1:
            name = f"Best 1-Transfer Move ({'Free' if free_transfers >= 1 else '-4 Hit'})"
        else:
            name = f"Best {actual_transfers_count}-Transfer Move (-{hit_cost} Hit)"

        # VC strategy label: helps LLM understand the tactical pairing intent
        vc_strategy = ""
        if captain_pick and vc_pick:
            c_xp = captain_pick.xp
            vc_xp = vc_pick.xp
            # Use selected_by_percent from player_map as global ownership proxy
            c_ownership = float(self.player_map.get(captain_pick.id, {}).get("selected_by_percent", 20.0))
            vc_ownership = float(self.player_map.get(vc_pick.id, {}).get("selected_by_percent", 20.0))
            c_is_differential = c_ownership < 15.0
            vc_is_template = vc_ownership >= 30.0
            if c_is_differential and vc_is_template:
                vc_strategy = (
                    f"Safety Net: {vc_pick.web_name} ({vc_ownership:.0f}% global sel.) backs "
                    f"differential captain {captain_pick.web_name} ({c_ownership:.0f}% sel.)"
                )
            elif not c_is_differential and not (vc_ownership >= 30.0):
                vc_strategy = (
                    f"Differential Pair: {vc_pick.web_name} ({vc_ownership:.0f}% sel.) extends "
                    f"template captain {captain_pick.web_name} ({c_ownership:.0f}% sel.) for double upside"
                )
            else:
                vc_strategy = (
                    f"Balanced: {captain_pick.web_name} (C, {c_xp:.1f} xP) | "
                    f"{vc_pick.web_name} (VC, {vc_xp:.1f} xP)"
                )

        return CandidateSquad(
            name=name,
            transfers_count=actual_transfers_count,
            transfers=transfers_list,
            starters=starters,
            bench=bench,
            captain=captain_pick,
            vice_captain=vc_pick,
            formation=formation,
            gross_xp=gross_xp,
            hit_cost=hit_cost,
            net_xp=net_xp,
            multi_gw_xp=multi_gw_sum,
            strategic_value_score=strategic_value_score,
            total_cost_m=total_cost_m,
            bank_remaining_m=bank_remaining_m,
            vice_captain_strategy=vc_strategy,
        )

    # ==========================================
    # Automated Chip Evaluation Methods
    # ==========================================

    def evaluate_wildcard(
        self,
        current_squad_ids: List[int],
        total_budget_m: float,
        baseline_candidate: CandidateSquad,
        min_gain: float = 18.0,
        current_gw: Optional[int] = None,
        target_gws: Optional[List[int]] = None,
        eo_weights: Optional[Dict[int, float]] = None,
    ) -> ChipEvaluation:
        """
        Evaluate Wildcard chip: solves optimal 15-man squad from scratch with zero hit penalties.
        Compares wildcard_xp - standard_xp against WILDCARD_MIN_XP_GAIN.
        """
        wc_squad = self._solve_optimal_squad_from_scratch(
            total_budget_m=total_budget_m,
            current_squad_ids=set(current_squad_ids),
            chip_name="wildcard",
            squad_title="Wildcard Squad (Free Rebuild)",
            eo_weights=eo_weights,
        )

        if not wc_squad:
            return ChipEvaluation(
                chip_name="wildcard",
                display_name="Wildcard",
                projected_xp=0.0,
                baseline_xp=baseline_candidate.net_xp,
                xp_gain=0.0,
                threshold=min_gain,
                threshold_met=False,
                reason="Wildcard optimization failed to produce feasible squad.",
            )

        projected_xp = wc_squad.net_xp
        baseline_xp = baseline_candidate.net_xp
        xp_gain = round(projected_xp - baseline_xp, 2)

        target_list = target_gws if target_gws is not None else _parse_target_gws("TARGET_WILDCARD_GW")
        gw_match = True
        if target_list and current_gw is not None:
            gw_match = current_gw in target_list

        threshold_met = (xp_gain >= min_gain) and gw_match
        reason = (
            f"Projected xP gain (+{xp_gain:.2f} pts) vs standard solve (+{min_gain:.1f} pts threshold)"
            if threshold_met
            else f"Gain (+{xp_gain:.2f} pts) below threshold (+{min_gain:.1f} pts)"
        )
        if target_list and current_gw is not None and not gw_match:
            reason += f" (GW{current_gw} not in target GWs {target_list})"

        return ChipEvaluation(
            chip_name="wildcard",
            display_name="Wildcard",
            projected_xp=projected_xp,
            baseline_xp=baseline_xp,
            xp_gain=xp_gain,
            threshold=min_gain,
            threshold_met=threshold_met,
            squad_candidate=wc_squad,
            reason=reason,
        )

    def evaluate_free_hit(
        self,
        current_squad_ids: List[int],
        total_budget_m: float,
        baseline_candidate: CandidateSquad,
        min_gain: float = 14.0,
        current_gw: Optional[int] = None,
        target_gws: Optional[List[int]] = None,
        eo_weights: Optional[Dict[int, float]] = None,
    ) -> ChipEvaluation:
        """
        Evaluate Free Hit chip: solves optimal 15-man squad for 1 GW only without persisting transfers.
        Compares free_hit_xp - standard_xp against FREE_HIT_MIN_XP_GAIN (or if active squad has <= 7 starters).
        """
        fh_squad = self._solve_optimal_squad_from_scratch(
            total_budget_m=total_budget_m,
            current_squad_ids=set(current_squad_ids),
            chip_name="freehit",
            squad_title="Free Hit Squad (1-GW Optimization)",
            eo_weights=eo_weights,
        )

        if not fh_squad:
            return ChipEvaluation(
                chip_name="freehit",
                display_name="Free Hit",
                projected_xp=0.0,
                baseline_xp=baseline_candidate.net_xp,
                xp_gain=0.0,
                threshold=min_gain,
                threshold_met=False,
                reason="Free Hit optimization failed to produce feasible squad.",
            )

        projected_xp = fh_squad.net_xp
        baseline_xp = baseline_candidate.net_xp
        xp_gain = round(projected_xp - baseline_xp, 2)

        target_list = target_gws if target_gws is not None else _parse_target_gws("TARGET_FREE_HIT_GW")
        gw_match = True
        if target_list and current_gw is not None:
            gw_match = current_gw in target_list

        threshold_met = (xp_gain >= min_gain) and gw_match
        reason = (
            f"Projected xP gain (+{xp_gain:.2f} pts) vs standard solve (+{min_gain:.1f} pts threshold)"
            if threshold_met
            else f"Gain (+{xp_gain:.2f} pts) below threshold (+{min_gain:.1f} pts)"
        )
        if target_list and current_gw is not None and not gw_match:
            reason += f" (GW{current_gw} not in target GWs {target_list})"

        return ChipEvaluation(
            chip_name="freehit",
            display_name="Free Hit",
            projected_xp=projected_xp,
            baseline_xp=baseline_xp,
            xp_gain=xp_gain,
            threshold=min_gain,
            threshold_met=threshold_met,
            squad_candidate=fh_squad,
            reason=reason,
        )

    def evaluate_bench_boost(
        self,
        baseline_candidate: CandidateSquad,
        min_bench_xp: float = 16.0,
        current_gw: Optional[int] = None,
        target_gws: Optional[List[int]] = None,
    ) -> ChipEvaluation:
        """
        Evaluate Bench Boost chip: compares bench sum xP against BENCH_BOOST_MIN_BENCH_XP.
        """
        bench_xp = sum(p.xp for p in baseline_candidate.bench)
        baseline_xp = baseline_candidate.net_xp
        projected_xp = round(baseline_xp + bench_xp, 2)
        xp_gain = round(bench_xp, 2)

        target_list = target_gws if target_gws is not None else _parse_target_gws("TARGET_BENCH_BOOST_GW")
        gw_match = True
        if target_list and current_gw is not None:
            gw_match = current_gw in target_list

        threshold_met = (bench_xp >= min_bench_xp) and gw_match
        reason = (
            f"Bench projected xP ({bench_xp:.2f} pts) meets threshold ({min_bench_xp:.1f} pts)"
            if threshold_met
            else f"Bench projected xP ({bench_xp:.2f} pts) below threshold ({min_bench_xp:.1f} pts)"
        )
        if target_list and current_gw is not None and not gw_match:
            reason += f" (GW{current_gw} not in target GWs {target_list})"

        bb_squad = baseline_candidate.model_copy(deep=True)
        bb_squad.active_chip = "bboost"
        bb_squad.name = f"{baseline_candidate.name} + Bench Boost"
        bb_squad.net_xp = projected_xp

        return ChipEvaluation(
            chip_name="bboost",
            display_name="Bench Boost",
            projected_xp=projected_xp,
            baseline_xp=baseline_xp,
            xp_gain=xp_gain,
            threshold=min_bench_xp,
            threshold_met=threshold_met,
            squad_candidate=bb_squad,
            reason=reason,
        )

    def evaluate_triple_captain(
        self,
        baseline_candidate: CandidateSquad,
        min_captain_xp: float = 11.5,
        current_gw: Optional[int] = None,
        target_gws: Optional[List[int]] = None,
    ) -> ChipEvaluation:
        """
        Evaluate Triple Captain chip: compares captain projected xP against TRIPLE_CAPTAIN_MIN_XP.
        """
        captain_pick = baseline_candidate.captain
        captain_xp = captain_pick.xp if captain_pick else 0.0
        baseline_xp = baseline_candidate.net_xp
        projected_xp = round(baseline_xp + captain_xp, 2)
        xp_gain = round(captain_xp, 2)

        target_list = target_gws if target_gws is not None else _parse_target_gws("TARGET_TRIPLE_CAPTAIN_GW")
        gw_match = True
        if target_list and current_gw is not None:
            gw_match = current_gw in target_list

        threshold_met = (captain_xp >= min_captain_xp) and gw_match
        c_name = captain_pick.web_name if captain_pick else "Captain"
        
        reason = (
            f"Captain {c_name} projected xP ({captain_xp:.2f} pts) meets threshold ({min_captain_xp:.1f} pts)"
            if threshold_met
            else f"Captain {c_name} projected xP ({captain_xp:.2f} pts) below threshold ({min_captain_xp:.1f} pts)"
        )
        if target_list and current_gw is not None and not gw_match:
            reason += f" (GW{current_gw} not in target GWs {target_list})"

        tc_squad = baseline_candidate.model_copy(deep=True)
        tc_squad.active_chip = "3xc"
        tc_squad.name = f"{baseline_candidate.name} + Triple Captain ({c_name})"
        tc_squad.net_xp = projected_xp

        return ChipEvaluation(
            chip_name="3xc",
            display_name="Triple Captain",
            projected_xp=projected_xp,
            baseline_xp=baseline_xp,
            xp_gain=xp_gain,
            threshold=min_captain_xp,
            threshold_met=threshold_met,
            squad_candidate=tc_squad,
            reason=reason,
        )

    def evaluate_all_chips(
        self,
        current_squad_ids: List[int],
        total_budget_m: float,
        baseline_candidate: CandidateSquad,
        current_gw: Optional[int] = None,
        available_chips: Optional[List[str]] = None,
        eo_weights: Optional[Dict[int, float]] = None,
    ) -> ChipEvaluationResult:
        """
        Evaluate all chips against environment thresholds and select the best recommendation.
        """
        wc_min = float(os.getenv("WILDCARD_MIN_XP_GAIN", "18.0"))
        fh_min = float(os.getenv("FREE_HIT_MIN_XP_GAIN", "14.0"))
        bb_min = float(os.getenv("BENCH_BOOST_MIN_BENCH_XP", "16.0"))
        tc_min = float(os.getenv("TRIPLE_CAPTAIN_MIN_XP", "11.5"))

        evals: Dict[str, ChipEvaluation] = {
            "wildcard": self.evaluate_wildcard(current_squad_ids, total_budget_m, baseline_candidate, min_gain=wc_min, current_gw=current_gw, eo_weights=eo_weights),
            "freehit": self.evaluate_free_hit(current_squad_ids, total_budget_m, baseline_candidate, min_gain=fh_min, current_gw=current_gw, eo_weights=eo_weights),
            "bboost": self.evaluate_bench_boost(baseline_candidate, min_bench_xp=bb_min, current_gw=current_gw),
            "3xc": self.evaluate_triple_captain(baseline_candidate, min_captain_xp=tc_min, current_gw=current_gw),
        }

        if available_chips is not None:
            evals = {k: v for k, v in evals.items() if k in available_chips}

        qualified = [e for e in evals.values() if e.threshold_met]
        if not qualified:
            return ChipEvaluationResult(
                evaluations=evals,
                recommended_chip=None,
                recommendation_reason="No chip thresholds met for this gameweek.",
            )

        qualified.sort(key=lambda e: -e.xp_gain)
        best_chip = qualified[0]

        return ChipEvaluationResult(
            evaluations=evals,
            recommended_chip=best_chip.chip_name,
            recommendation_reason=f"Recommended {best_chip.display_name}: {best_chip.reason}",
        )

    def optimize(
        self,
        current_squad_ids: List[int],
        bank_m: float = 0.0,
        free_transfers: int = 1,
        selling_prices: Optional[Dict[int, float]] = None,
        eo_weights: Optional[Dict[int, float]] = None,
        current_gw: Optional[int] = None,
        evaluate_chips: bool = True,
    ) -> OptimizationResult:
        """
        Generate 7 strategic candidate moves with exact FPL selling math,
        multi-period horizon lookahead, integer exclusion cuts, and chip evaluation.
        """
        current_set = set(current_squad_ids)
        sp_map = selling_prices or {}

        # Exact squad selling value (FPL liquidation value)
        current_team_cost = sum(
            sp_map.get(pid, self.player_map.get(pid, {}).get("cost_m", 5.0))
            for pid in current_set
        )
        total_budget_m = round(current_team_cost + bank_m, 2)

        candidates: List[CandidateSquad] = []
        rolling_bonus_xp = float(os.getenv("ROLLING_BONUS_XP", "1.5"))

        # Candidate 1: 0 Transfers (Roll / Bank FT with +1.5 xP Strategic Value)
        cand_0 = self._solve_for_k_transfers(
            current_set,
            total_budget_m,
            k_transfers=0,
            free_transfers=free_transfers,
            selling_prices=selling_prices,
            eo_weights=eo_weights,
            rolling_bonus_xp=rolling_bonus_xp,
        )
        if cand_0:
            cand_0.name = f"Option 1: Roll / Bank Transfer (0 Moves · +{rolling_bonus_xp:.1f} FT Strategic Value)"
            candidates.append(cand_0)

        # Candidates 2, 3, 4: Top 3 1-Transfer Moves (Optimal + 2 Alternatives)
        excluded_in_1: List[Set[int]] = []
        for rank in range(1, 4):
            cand_1 = self._solve_for_k_transfers(
                current_set,
                total_budget_m,
                k_transfers=1,
                free_transfers=free_transfers,
                selling_prices=selling_prices,
                eo_weights=eo_weights,
                rolling_bonus_xp=rolling_bonus_xp,
                excluded_transfers_in=excluded_in_1,
            )
            if cand_1 and cand_1.transfers:
                cand_in_ids = {t.player_in.id for t in cand_1.transfers}
                excluded_in_1.append(cand_in_ids)
                opt_num = len(candidates) + 1
                prefix = "Optimal 1-Transfer Move" if rank == 1 else f"Alternative 1-Transfer Move (Rank #{rank})"
                cost_label = "Free" if free_transfers >= 1 else "-4 Hit"
                cand_1.name = f"Option {opt_num}: {prefix} ({cost_label})"
                candidates.append(cand_1)

        # Candidates 5, 6: Top 2 2-Transfer Moves (Optimal + 1 Alternative)
        hit_min_gain = float(os.getenv("HIT_MIN_NET_XP_GAIN", "5.0"))
        excluded_in_2: List[Set[int]] = []
        baseline_xp = cand_0.net_xp if cand_0 else 0.0

        for rank in range(1, 3):
            cand_2 = self._solve_for_k_transfers(
                current_set,
                total_budget_m,
                k_transfers=2,
                free_transfers=free_transfers,
                selling_prices=selling_prices,
                eo_weights=eo_weights,
                rolling_bonus_xp=rolling_bonus_xp,
                excluded_transfers_in=excluded_in_2,
            )
            if cand_2 and cand_2.transfers:
                cand_in_ids = {t.player_in.id for t in cand_2.transfers}
                excluded_in_2.append(cand_in_ids)
                opt_num = len(candidates) + 1
                prefix = "Optimal 2-Transfer Move" if rank == 1 else f"Alternative 2-Transfer Move (Rank #{rank})"
                if cand_2.hit_cost > 0:
                    net_gain = round(cand_2.net_xp - baseline_xp, 2)
                    if net_gain >= hit_min_gain:
                        cand_2.name = f"Option {opt_num}: {prefix} (-{cand_2.hit_cost} Hit · +{net_gain:.2f} Net xP Gain vs Roll)"
                    else:
                        cand_2.name = f"Option {opt_num}: {prefix} (-{cand_2.hit_cost} Hit · Sub-optimal: +{net_gain:.2f} xP below +{hit_min_gain:.1f} xP Hurdle)"
                        cand_2.strategic_value_score = round(cand_2.strategic_value_score - 2.0, 2)
                else:
                    cand_2.name = f"Option {opt_num}: {prefix} (Free)"
                candidates.append(cand_2)

        # Chip Evaluation & Option 7 Selection
        chip_result: Optional[ChipEvaluationResult] = None
        if evaluate_chips and candidates:
            best_candidate = max(candidates, key=lambda c: c.net_xp)
            chip_result = self.evaluate_all_chips(
                current_squad_ids=current_squad_ids,
                total_budget_m=total_budget_m,
                baseline_candidate=best_candidate,
                current_gw=current_gw,
                eo_weights=eo_weights,
            )

        # Option 7: Recommended Active Chip Candidate or Rank #3 2-Transfer Move
        opt_7_added = False
        if chip_result and chip_result.recommended_chip:
            rec_chip_eval = chip_result.evaluations.get(chip_result.recommended_chip)
            if rec_chip_eval and rec_chip_eval.squad_candidate:
                chip_cand = rec_chip_eval.squad_candidate.model_copy(deep=True)
                opt_num = len(candidates) + 1
                chip_cand.name = f"Option {opt_num}: Active Chip Recommendation ({rec_chip_eval.display_name} · {rec_chip_eval.reason})"
                candidates.append(chip_cand)
                opt_7_added = True

        if not opt_7_added:
            # Fallback for Option 7: 3rd 2-transfer move (or 4th 1-transfer move)
            cand_2_rank3 = self._solve_for_k_transfers(
                current_set,
                total_budget_m,
                k_transfers=2,
                free_transfers=free_transfers,
                selling_prices=selling_prices,
                eo_weights=eo_weights,
                rolling_bonus_xp=rolling_bonus_xp,
                excluded_transfers_in=excluded_in_2,
            )
            if cand_2_rank3 and cand_2_rank3.transfers:
                opt_num = len(candidates) + 1
                cand_2_rank3.name = f"Option {opt_num}: Alternative 2-Transfer Move (Rank #3)"
                candidates.append(cand_2_rank3)
            else:
                # Extra 1-transfer alternative if 2-transfer combination exhausted
                cand_1_rank4 = self._solve_for_k_transfers(
                    current_set,
                    total_budget_m,
                    k_transfers=1,
                    free_transfers=free_transfers,
                    selling_prices=selling_prices,
                    eo_weights=eo_weights,
                    rolling_bonus_xp=rolling_bonus_xp,
                    excluded_transfers_in=excluded_in_1,
                )
                if cand_1_rank4 and cand_1_rank4.transfers:
                    opt_num = len(candidates) + 1
                    cand_1_rank4.name = f"Option {opt_num}: Alternative 1-Transfer Move (Rank #4)"
                    candidates.append(cand_1_rank4)

        return OptimizationResult(
            candidates=candidates,
            current_team_value_m=round(current_team_cost, 2),
            bank_m=round(bank_m, 2),
            free_transfers=free_transfers,
            chip_evaluation=chip_result,
        )
