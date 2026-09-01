"""Data ingestion and mapping helper for external FPL Core Insights and expected points datasets."""

import io
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


TRANSLIT_MAP = {
    "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ß": "ss",
    "đ": "d", "Đ": "D",
    "ł": "l", "Ł": "L",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "TH",
}


def _normalize_name(name: str) -> str:
    """Normalize player name for robust fuzzy/case-insensitive matching."""
    if not name:
        return ""
    s = str(name)
    for k, v in TRANSLIT_MAP.items():
        s = s.replace(k, v)
    # Strip accents / diacritics
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Convert to lowercase and strip punctuation/extra whitespace
    clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", ascii_name).strip().lower()
    return re.sub(r"\s+", " ", clean_name)


class FPLCoreInsightsFetcher:
    """
    Fetches, parses, and maps FPL Core Insights (olbauday/FPL-Core-Insights)
    dataset with direct official FPL element ID alignment.
    Supports local CSV fallback and legacy FPL Review projection schemas.
    """

    DEFAULT_CSV_PATH = Path("data/playerstats.csv")
    LOCAL_CANDIDATE_PATHS = [
        Path("data/playerstats.csv"),
        Path("data/fpl_core_insights.csv"),
        Path("data/fplreview.csv"),
        Path("data/projections.csv"),
        Path("playerstats.csv"),
        Path("fplreview.csv"),
        Path("projections.csv"),
    ]
    PRIMARY_GITHUB_URL = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2026-2027/playerstats.csv"
    FALLBACK_GITHUB_URL = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/playerstats.csv"

    def __init__(
        self,
        url: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        timeout: int = 10,
    ):
        self.url = (
            url
            or os.getenv("FPL_CORE_INSIGHTS_URL")
            or os.getenv("FPLREVIEW_PROJECTIONS_URL")
            or os.getenv("FPLREVIEW_CSV_URL")
            or self.PRIMARY_GITHUB_URL
        )
        self.file_path = (
            file_path
            or os.getenv("FPL_CORE_INSIGHTS_CSV_PATH")
            or os.getenv("FPLREVIEW_CSV_PATH")
            or self.DEFAULT_CSV_PATH
        )
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        self.timeout = timeout

    def fetch_projections(
        self,
        csv_content: Optional[str] = None,
        force_url: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and parse projections CSV from string content, local file, or remote URL.
        Local CSV files take priority over remote scraping.
        """
        # 1. Direct CSV content provided (e.g. in unit tests or memory cache)
        if csv_content is not None:
            try:
                trimmed = csv_content.strip()
                if trimmed.startswith("<!DOCTYPE") or trimmed.startswith("<html"):
                    logger.warning("Provided CSV content contains HTML markup. Ignoring.")
                    return None
                df = pd.read_csv(io.StringIO(csv_content))
                logger.info(f"Loaded {len(df)} projection records from provided CSV content.")
                return df
            except Exception as e:
                logger.warning(f"Failed to parse provided CSV content: {e}")
                return None

        # 2. Local CSV file check (Prioritize local projection files)
        candidate_files: List[Path] = []
        if self.file_path:
            candidate_files.append(self.file_path)
        for cand in self.LOCAL_CANDIDATE_PATHS:
            if cand not in candidate_files:
                candidate_files.append(cand)

        for candidate in candidate_files:
            if candidate.exists() and candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
                        header_preview = f.read(512).strip()
                    if header_preview.startswith("<!DOCTYPE") or header_preview.startswith("<html"):
                        logger.warning(f"Local file {candidate} contains HTML instead of CSV projections. Skipping.")
                        continue
                    df = pd.read_csv(candidate)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} projection records from local file: {candidate}")
                        return df
                except Exception as e:
                    logger.warning(f"Failed to read local FPL projections CSV ({candidate}): {e}")

        # 3. Remote HTTP Fetch from GitHub / configured endpoint
        target_urls = []
        if force_url:
            target_urls.append(force_url)
        elif self.url:
            target_urls.append(self.url)
            if self.FALLBACK_GITHUB_URL not in target_urls:
                target_urls.append(self.FALLBACK_GITHUB_URL)

        headers = {
            "User-Agent": "Gegenbot-FPL-Engine/2.0",
            "Accept": "text/csv, application/json, text/plain, */*",
        }

        for target_url in target_urls:
            logger.info(f"Fetching FPL Core Insights dataset from remote endpoint: {target_url}")
            try:
                resp = requests.get(target_url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    text_content = resp.text.strip()
                    if text_content and not text_content.startswith("<!DOCTYPE") and not text_content.startswith("<html"):
                        try:
                            df = pd.read_csv(io.StringIO(text_content))
                            if not df.empty:
                                logger.info(f"Successfully fetched and parsed {len(df)} records from {target_url}")
                                return df
                        except Exception as parse_err:
                            logger.warning(f"Could not parse response from {target_url} as CSV: {parse_err}")
                else:
                    logger.warning(f"Remote fetch from {target_url} returned HTTP {resp.status_code}.")
            except requests.exceptions.Timeout:
                logger.warning(f"Remote fetch from {target_url} timed out after {self.timeout}s.")
            except Exception as exc:
                logger.warning(f"Error fetching from {target_url}: {exc}")

        return None

    def map_to_bootstrap(
        self,
        df: pd.DataFrame,
        bootstrap_data: Dict[str, Any],
        current_event: int = 1,
        decay_factor: Optional[float] = None,
        horizon_length: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Map dataset rows to official FPL element IDs.
        Supports both FPL-Core-Insights (direct `id` match + xGI/CBIT) and legacy FPL Review schemas.
        """
        if df is None or df.empty:
            return {}

        n_horizon = int(horizon_length if horizon_length is not None else os.getenv("HORIZON_LENGTH", "3"))
        gamma = float(decay_factor if decay_factor is not None else os.getenv("DECAY_FACTOR", "0.85"))
        decay_weights = [round(gamma ** t, 4) for t in range(max(1, n_horizon))]
        decay_sum = sum(decay_weights)

        elements = bootstrap_data.get("elements", [])
        teams = bootstrap_data.get("teams", [])

        # Build official lookup dicts
        team_id_to_name = {t["id"]: t["name"].lower() for t in teams}
        team_id_to_short = {t["id"]: t["short_name"].lower() for t in teams}

        # Normalize column names in projections DataFrame
        col_map = {col: str(col).strip() for col in df.columns}
        cols_lower = {str(col).strip().lower(): col for col in df.columns}

        # Identify candidate columns for ID, Name, Team, and Gameweek points
        id_col = next((c for c in df.columns if str(c).strip().lower() in ["id", "fpl_id", "element_id", "code"]), None)
        name_col = next((c for c in df.columns if str(c).strip().lower() in ["name", "player", "web_name", "full_name"]), None)
        team_col = next((c for c in df.columns if str(c).strip().lower() in ["team", "club", "team_name", "team_short"]), None)

        # Identify GW columns for current and future events
        gw_col_patterns = [
            re.compile(rf"^(gw)?{current_event}(_pts|_xp|_points)?$", re.IGNORECASE),
            re.compile(rf"^pts_{current_event}$", re.IGNORECASE),
            re.compile(rf"^ep_next$", re.IGNORECASE),
            re.compile(rf"^xp$", re.IGNORECASE),
            re.compile(rf"^expected_points$", re.IGNORECASE),
            re.compile(rf"^points$", re.IGNORECASE),
        ]
        gw_col = None
        for pattern in gw_col_patterns:
            for col in df.columns:
                if pattern.match(str(col).strip()):
                    gw_col = col
                    break
            if gw_col:
                break

        if gw_col is None:
            # Fallback to first numeric column that might represent projections
            for col in df.columns:
                if col not in [id_col, name_col, team_col] and pd.api.types.is_numeric_dtype(df[col]):
                    gw_col = col
                    break

        if gw_col is None and "ep_next" not in cols_lower:
            logger.warning("Projections DataFrame does not contain recognizable ID, Name, or xP columns.")
            return {}

        # Dynamically identify future gameweek columns up to horizon_length
        future_gw_cols: List[Tuple[int, str]] = []
        for t in range(1, max(1, n_horizon)):
            target_gw = current_event + t
            target_col = None
            for pattern in [
                re.compile(rf"^(gw)?{target_gw}(_pts|_xp|_points)?$", re.IGNORECASE),
                re.compile(rf"^pts_{target_gw}$", re.IGNORECASE),
            ]:
                for col in df.columns:
                    if pattern.match(str(col).strip()):
                        target_col = col
                        break
                if target_col:
                    break
            if target_col:
                future_gw_cols.append((t, target_col))

        # Build lookup tables for official FPL elements
        official_by_id = {el["id"]: el for el in elements}
        official_by_norm_name: Dict[str, List[Dict[str, Any]]] = {}
        official_by_team_and_name: Dict[str, Dict[str, Any]] = {}

        for el in elements:
            el_id = el["id"]
            web_norm = _normalize_name(el.get("web_name", ""))
            first_norm = _normalize_name(el.get("first_name", ""))
            second_norm = _normalize_name(el.get("second_name", ""))
            full_norm = f"{first_norm} {second_norm}".strip()
            team_id = el.get("team", 1)
            t_short = team_id_to_short.get(team_id, "")
            t_name = team_id_to_name.get(team_id, "")

            official_by_id[el_id] = el
            if web_norm:
                official_by_norm_name.setdefault(web_norm, []).append(el)
            if full_norm and full_norm != web_norm:
                official_by_norm_name.setdefault(full_norm, []).append(el)

            if t_short and web_norm:
                official_by_team_and_name[f"{t_short}:{web_norm}"] = el
            if t_name and web_norm:
                official_by_team_and_name[f"{t_name}:{web_norm}"] = el
            if t_short and full_norm:
                official_by_team_and_name[f"{t_short}:{full_norm}"] = el
            if t_name and full_norm:
                official_by_team_and_name[f"{t_name}:{full_norm}"] = el

        mapped_projections: Dict[int, Dict[str, Any]] = {}

        for _, row in df.iterrows():
            xp_val: Optional[float] = None

            if gw_col and gw_col in row:
                try:
                    raw_val = float(row[gw_col])
                    if not pd.isna(raw_val):
                        xp_val = round(max(0.0, raw_val), 2)
                except (ValueError, TypeError):
                    pass

            if xp_val is None and "ep_next" in cols_lower:
                try:
                    raw_ep = float(row[cols_lower["ep_next"]])
                    if not pd.isna(raw_ep):
                        xp_val = round(max(0.0, raw_ep), 2)
                except (ValueError, TypeError):
                    pass

            if xp_val is None:
                continue

            # Calculate discounted multi-week projection with dynamic decay weighting
            multi_gw_sum = xp_val
            has_future_cols = False
            for t_idx, col_name in future_gw_cols:
                if col_name in row:
                    try:
                        p_future = float(row[col_name])
                        if not pd.isna(p_future):
                            multi_gw_sum += decay_weights[t_idx] * max(0.0, p_future)
                            has_future_cols = True
                    except (ValueError, TypeError):
                        pass

            if has_future_cols:
                xp_3gw_val = round(multi_gw_sum, 2)
            else:
                xp_3gw_val = round(xp_val * decay_sum, 2)

            matched_element_id: Optional[int] = None

            # 1. Match by numeric ID
            if id_col and id_col in row and not pd.isna(row[id_col]):
                try:
                    cand_id = int(float(row[id_col]))
                    if cand_id in official_by_id:
                        matched_element_id = cand_id
                except (ValueError, TypeError):
                    pass

            # 2. Match by Name + Team
            if matched_element_id is None and name_col and name_col in row:
                raw_name = str(row[name_col])
                norm_name = _normalize_name(raw_name)
                raw_team = str(row[team_col]).lower().strip() if team_col and team_col in row and not pd.isna(row[team_col]) else ""

                if raw_team:
                    key = f"{raw_team}:{norm_name}"
                    if key in official_by_team_and_name:
                        matched_element_id = official_by_team_and_name[key]["id"]

                if matched_element_id is None and norm_name in official_by_norm_name:
                    candidates_list = official_by_norm_name[norm_name]
                    if len(candidates_list) == 1:
                        matched_element_id = candidates_list[0]["id"]
                    elif raw_team:
                        for c in candidates_list:
                            c_tid = c.get("team", 1)
                            if team_id_to_short.get(c_tid) == raw_team or team_id_to_name.get(c_tid) == raw_team:
                                matched_element_id = c["id"]
                                break

            if matched_element_id is not None:
                xg_90 = float(row.get(cols_lower.get("expected_goals_per_90", ""), 0.0) or 0.0) if "expected_goals_per_90" in cols_lower else 0.0
                xa_90 = float(row.get(cols_lower.get("expected_assists_per_90", ""), 0.0) or 0.0) if "expected_assists_per_90" in cols_lower else 0.0
                xgi_90 = float(row.get(cols_lower.get("expected_goal_involvements_per_90", ""), 0.0) or 0.0) if "expected_goal_involvements_per_90" in cols_lower else (xg_90 + xa_90)
                def_contrib = float(row.get(cols_lower.get("defensive_contribution_per_90", ""), 0.0) or 0.0) if "defensive_contribution_per_90" in cols_lower else 0.0

                source = "fpl_core_insights" if "expected_goals_per_90" in cols_lower or "defensive_contribution_per_90" in cols_lower else "fplreview"

                mapped_projections[matched_element_id] = {
                    "fplreview_xp": xp_val,
                    "fplreview_xp_3gw": xp_3gw_val,
                    "xg_90": xg_90,
                    "xa_90": xa_90,
                    "xgi_90": xgi_90,
                    "def_contrib": def_contrib,
                    "source": source,
                }

        logger.info(f"Mapped {len(mapped_projections)} projections to official FPL element IDs.")
        return mapped_projections


# Retain FPLReviewFetcher alias for full backward compatibility
FPLReviewFetcher = FPLCoreInsightsFetcher


def calculate_fallback_xp(
    element: Dict[str, Any],
    next_fdr: float = 3.0,
    next_is_home: bool = False,
    avg_fdr: float = 3.0,
    availability: float = 1.0,
    decay_factor: Optional[float] = None,
    horizon_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate fallback baseline expected points for an individual player using
    ep_next, player form, points per game, and FDR rating without interrupting the pipeline.
    """
    n_horizon = int(horizon_length if horizon_length is not None else os.getenv("HORIZON_LENGTH", "3"))
    gamma = float(decay_factor if decay_factor is not None else os.getenv("DECAY_FACTOR", "0.85"))
    decay_weights = [round(gamma ** t, 4) for t in range(max(1, n_horizon))]
    decay_sum = sum(decay_weights)

    try:
        form = float(element.get("form", 0.0) or 0.0)
    except (ValueError, TypeError):
        form = 0.0

    try:
        ppg = float(element.get("points_per_game", 0.0) or 0.0)
    except (ValueError, TypeError):
        ppg = 0.0

    try:
        ep_next_raw = element.get("ep_next")
        ep_next_val = float(ep_next_raw) if ep_next_raw is not None else None
    except (ValueError, TypeError):
        ep_next_val = None

    pos_fallback = {1: 2.5, 2: 2.5, 3: 3.0, 4: 3.0}
    elem_type = element.get("element_type", 3)
    pos_base = pos_fallback.get(elem_type, 2.5)

    if form > 0 and ppg > 0:
        base_xp = 0.60 * form + 0.40 * ppg
    elif form > 0:
        base_xp = form
    elif ppg > 0:
        base_xp = ppg
    else:
        base_xp = pos_base

    fdr_mult = max(0.6, 1.0 + (3.0 - next_fdr) * 0.08)
    if next_is_home:
        fdr_mult *= 1.05

    heuristic_xp = round(max(0.0, base_xp * fdr_mult * availability), 2)

    if ep_next_val is not None and ep_next_val > 0:
        xp = round(max(0.0, ep_next_val * availability), 2)
        source = "fpl_ep_next"
    else:
        xp = heuristic_xp
        source = "fpl_heuristic"

    fdr_3gw_mult = max(0.6, 1.0 + (3.0 - avg_fdr) * 0.08)
    xp_3gw = round(max(0.0, base_xp * fdr_3gw_mult * availability * decay_sum), 2)

    return {
        "xp": xp,
        "xp_3gw": xp_3gw,
        "xp_source": source,
        "base_xp": round(base_xp, 2),
    }


def fetch_fplreview_projections(
    url: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    csv_content: Optional[str] = None,
    timeout: int = 10,
) -> Optional[pd.DataFrame]:
    """Helper function to fetch projections DataFrame."""
    fetcher = FPLCoreInsightsFetcher(url=url, file_path=file_path, timeout=timeout)
    return fetcher.fetch_projections(csv_content=csv_content)


fetch_core_insights_projections = fetch_fplreview_projections


def map_fplreview_to_elements(
    fplreview_df: pd.DataFrame,
    bootstrap_data: Dict[str, Any],
    current_event: int = 1,
    decay_factor: Optional[float] = None,
    horizon_length: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Helper function to map projections to official element IDs."""
    fetcher = FPLCoreInsightsFetcher()
    return fetcher.map_to_bootstrap(
        fplreview_df,
        bootstrap_data,
        current_event=current_event,
        decay_factor=decay_factor,
        horizon_length=horizon_length,
    )


map_core_insights_to_elements = map_fplreview_to_elements


class VaastavDataFetcher:
    """
    Fetches, parses, and aggregates match logs and underlying statistics
    from the vaastav/Fantasy-Premier-League repository.
    Supplies rolling xGI/90, xGC/90, actual minutes, and start reliability.
    """
    DEFAULT_SEASON = "2026-27"
    FALLBACK_SEASONS = ["2026-27", "2025-26", "2024-25"]
    BASE_GITHUB_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

    def __init__(
        self,
        season: Optional[str] = None,
        base_url: Optional[str] = None,
        local_dir: Optional[Union[str, Path]] = None,
        timeout: int = 10,
    ):
        self.season = season or os.getenv("VAASTAV_SEASON", self.DEFAULT_SEASON)
        self.base_url = (base_url or os.getenv("VAASTAV_DATA_URL") or f"{self.BASE_GITHUB_RAW}/{self.season}").rstrip("/")
        self.local_dir = Path(local_dir or os.getenv("VAASTAV_LOCAL_DIR", "data"))
        self.timeout = timeout

    def fetch_merged_gw(self) -> Optional[pd.DataFrame]:
        """Fetch merged_gw.csv from local cache or remote GitHub repository."""
        candidate_paths = [
            self.local_dir / "vaastav_merged_gw.csv",
            self.local_dir / "merged_gw.csv",
            Path("data/vaastav_merged_gw.csv"),
            Path("data/merged_gw.csv"),
        ]
        for p in candidate_paths:
            if p.exists() and p.is_file():
                try:
                    df = pd.read_csv(p)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} records from local Vaastav merged_gw file: {p}")
                        return df
                except Exception as e:
                    logger.warning(f"Failed to read local Vaastav merged_gw CSV ({p}): {e}")

        target_urls = [f"{self.base_url}/gws/merged_gw.csv"]
        for s in self.FALLBACK_SEASONS:
            fallback_url = f"{self.BASE_GITHUB_RAW}/{s}/gws/merged_gw.csv"
            if fallback_url not in target_urls:
                target_urls.append(fallback_url)

        headers = {"User-Agent": "Gegenbot-FPL-Engine/2.0", "Accept": "text/csv, text/plain, */*"}
        for url in target_urls:
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200 and resp.text.strip() and not resp.text.strip().startswith("<!DOCTYPE") and not resp.text.strip().startswith("<html"):
                    df = pd.read_csv(io.StringIO(resp.text))
                    if not df.empty:
                        logger.info(f"Successfully fetched {len(df)} Vaastav gameweek records from {url}")
                        try:
                            self.local_dir.mkdir(parents=True, exist_ok=True)
                            df.to_csv(self.local_dir / "vaastav_merged_gw.csv", index=False)
                        except Exception:
                            pass
                        return df
            except Exception as e:
                logger.debug(f"Could not fetch Vaastav merged_gw from {url}: {e}")

        return None

    def fetch_players_raw(self) -> Optional[pd.DataFrame]:
        """Fetch players_raw.csv from local cache or remote GitHub repository."""
        candidate_paths = [
            self.local_dir / "vaastav_players_raw.csv",
            self.local_dir / "players_raw.csv",
            Path("data/vaastav_players_raw.csv"),
            Path("data/players_raw.csv"),
        ]
        for p in candidate_paths:
            if p.exists() and p.is_file():
                try:
                    df = pd.read_csv(p)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} records from local Vaastav players_raw file: {p}")
                        return df
                except Exception as e:
                    logger.warning(f"Failed to read local Vaastav players_raw CSV ({p}): {e}")

        target_urls = [f"{self.base_url}/players_raw.csv"]
        for s in self.FALLBACK_SEASONS:
            fallback_url = f"{self.BASE_GITHUB_RAW}/{s}/players_raw.csv"
            if fallback_url not in target_urls:
                target_urls.append(fallback_url)

        headers = {"User-Agent": "Gegenbot-FPL-Engine/2.0", "Accept": "text/csv, text/plain, */*"}
        for url in target_urls:
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200 and resp.text.strip() and not resp.text.strip().startswith("<!DOCTYPE") and not resp.text.strip().startswith("<html"):
                    df = pd.read_csv(io.StringIO(resp.text))
                    if not df.empty:
                        logger.info(f"Successfully fetched {len(df)} Vaastav players_raw records from {url}")
                        try:
                            self.local_dir.mkdir(parents=True, exist_ok=True)
                            df.to_csv(self.local_dir / "vaastav_players_raw.csv", index=False)
                        except Exception:
                            pass
                        return df
            except Exception as e:
                logger.debug(f"Could not fetch Vaastav players_raw from {url}: {e}")

        return None

    def calculate_player_underlying_metrics(
        self,
        merged_gw_df: Optional[pd.DataFrame] = None,
        players_raw_df: Optional[pd.DataFrame] = None,
        lookback_gws: int = 4,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate rolling player-level underlying form, minutes reliability, and Moneyball regression metrics:
        - rolling_xgi_90: Expected Goal Involvements per 90 in recent GWs
        - rolling_xgc_90: Expected Goals Conceded per 90 in recent GWs
        - rolling_minutes_avg: Average minutes in recent GWs
        - starts_ratio: Ratio of starts in recent appearances
        - season_xgi_90: Overall season xGI/90 from players_raw
        - xgi_delta: Difference between Expected GI and Actual GI (xGI - (Goals + Assists))
        - xgc_delta: Difference between Expected GC and Actual GC (xGC - Goals Conceded)
        - rolling_bps_avg: Average BPS in recent matches
        - def_contrib_90: Defensive contribution / CBI per 90
        """
        if merged_gw_df is None:
            merged_gw_df = self.fetch_merged_gw()
        if players_raw_df is None:
            players_raw_df = self.fetch_players_raw()

        results: Dict[int, Dict[str, Any]] = {}

        # 1. Process players_raw for baseline per-90 metrics
        if players_raw_df is not None and not players_raw_df.empty:
            id_col = "id" if "id" in players_raw_df.columns else None
            for _, row in players_raw_df.iterrows():
                try:
                    p_id = int(row[id_col]) if id_col else None
                    if p_id is None:
                        continue
                    xgi_90 = float(row.get("expected_goal_involvements_per_90", 0.0) or 0.0)
                    xg_90 = float(row.get("expected_goals_per_90", 0.0) or 0.0)
                    xa_90 = float(row.get("expected_assists_per_90", 0.0) or 0.0)
                    xgc_90 = float(row.get("expected_goals_conceded_per_90", 0.0) or 0.0)
                    def_contrib_90 = float(row.get("defensive_contribution_per_90", 0.0) or 0.0)

                    tot_goals = float(row.get("goals_scored", 0.0) or 0.0)
                    tot_assists = float(row.get("assists", 0.0) or 0.0)
                    tot_xgi_season = float(row.get("expected_goal_involvements", 0.0) or (float(row.get("expected_goals", 0.0) or 0.0) + float(row.get("expected_assists", 0.0) or 0.0)))
                    season_xgi_delta = round(tot_xgi_season - (tot_goals + tot_assists), 2)

                    tot_gc = float(row.get("goals_conceded", 0.0) or 0.0)
                    tot_xgc_season = float(row.get("expected_goals_conceded", 0.0) or 0.0)
                    season_xgc_delta = round(tot_xgc_season - tot_gc, 2)

                    results[p_id] = {
                        "season_xgi_90": round(xgi_90 or (xg_90 + xa_90), 2),
                        "season_xg_90": round(xg_90, 2),
                        "season_xa_90": round(xa_90, 2),
                        "season_xgc_90": round(xgc_90, 2),
                        "def_contrib_90": round(def_contrib_90, 2),
                        "rolling_xgi_90": round(xgi_90 or (xg_90 + xa_90), 2),
                        "rolling_xgc_90": round(xgc_90, 2),
                        "rolling_minutes_avg": float(row.get("minutes", 0) or 0),
                        "starts_ratio": 1.0,
                        "recent_matches_count": 0,
                        "xgi_delta": season_xgi_delta,
                        "xgc_delta": season_xgc_delta,
                        "actual_gi": round(tot_goals + tot_assists, 2),
                        "actual_gc": round(tot_gc, 2),
                        "rolling_bps_avg": float(row.get("bps", 0.0) or 0.0),
                    }
                except Exception:
                    continue

        # 2. Process merged_gw for recent rolling match stats
        if merged_gw_df is not None and not merged_gw_df.empty:
            element_col = "element" if "element" in merged_gw_df.columns else ("id" if "id" in merged_gw_df.columns else None)
            round_col = "round" if "round" in merged_gw_df.columns else ("GW" if "GW" in merged_gw_df.columns else None)

            if element_col and round_col:
                max_round = merged_gw_df[round_col].max()
                min_round = max(1, max_round - lookback_gws + 1)
                recent_df = merged_gw_df[merged_gw_df[round_col] >= min_round]

                for element_id, group in recent_df.groupby(element_col):
                    try:
                        p_id = int(element_id)
                        total_mins = float(group["minutes"].sum() if "minutes" in group.columns else 0)
                        total_xgi = float(group["expected_goal_involvements"].sum() if "expected_goal_involvements" in group.columns else 0.0)
                        total_xgc = float(group["expected_goals_conceded"].sum() if "expected_goals_conceded" in group.columns else 0.0)
                        total_starts = float(group["starts"].sum() if "starts" in group.columns else (group["minutes"] >= 60).sum())
                        matches_count = len(group)

                        goals = float(group["goals_scored"].sum() if "goals_scored" in group.columns else 0.0)
                        assists = float(group["assists"].sum() if "assists" in group.columns else 0.0)
                        actual_gi = goals + assists
                        actual_gc = float(group["goals_conceded"].sum() if "goals_conceded" in group.columns else 0.0)

                        rolling_xgi_90 = round((total_xgi / (total_mins / 90.0)), 2) if total_mins >= 45 else (results.get(p_id, {}).get("season_xgi_90", 0.0))
                        rolling_xgc_90 = round((total_xgc / (total_mins / 90.0)), 2) if total_mins >= 45 else (results.get(p_id, {}).get("season_xgc_90", 0.0))
                        avg_mins = round(total_mins / matches_count, 1) if matches_count > 0 else 0.0
                        starts_ratio = round(total_starts / matches_count, 2) if matches_count > 0 else 1.0

                        # Regression delta over recent window: positive means underperformed expected (buy candidate)
                        xgi_delta = round(total_xgi - actual_gi, 2)
                        xgc_delta = round(total_xgc - actual_gc, 2)

                        bps_sum = float(group["bps"].sum() if "bps" in group.columns else 0.0)
                        avg_bps = round(bps_sum / matches_count, 1) if matches_count > 0 else 0.0

                        if p_id not in results:
                            results[p_id] = {}
                        results[p_id].update({
                            "rolling_xgi_90": rolling_xgi_90,
                            "rolling_xgc_90": rolling_xgc_90,
                            "rolling_minutes_avg": avg_mins,
                            "starts_ratio": starts_ratio,
                            "recent_matches_count": matches_count,
                            "xgi_delta": xgi_delta,
                            "xgc_delta": xgc_delta,
                            "actual_gi": round(actual_gi, 2),
                            "actual_gc": round(actual_gc, 2),
                            "rolling_bps_avg": avg_bps,
                        })
                    except Exception:
                        continue

        logger.info(f"Computed Vaastav underlying form metrics for {len(results)} players.")
        return results

