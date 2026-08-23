"""Data ingestion and mapping helper for external FPL Review Expected Points (xP) projections."""

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


class FPLReviewFetcher:
    """Fetches, parses, and maps free FPL Review projections to official FPL element IDs."""

    DEFAULT_CSV_PATH = Path("data/fplreview.csv")
    LOCAL_CANDIDATE_PATHS = [
        Path("data/fplreview.csv"),
        Path("data/projections.csv"),
        Path("fplreview.csv"),
        Path("projections.csv"),
    ]
    DEFAULT_PROJECTIONS_URL = "https://fplreview.com/free-planner/"

    def __init__(
        self,
        url: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        timeout: int = 10,
    ):
        self.url = url or os.getenv("FPLREVIEW_PROJECTIONS_URL") or os.getenv("FPLREVIEW_CSV_URL") or self.DEFAULT_PROJECTIONS_URL
        self.file_path = file_path or os.getenv("FPLREVIEW_CSV_PATH") or self.DEFAULT_CSV_PATH
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        self.timeout = timeout

    def fetch_projections(
        self,
        csv_content: Optional[str] = None,
        force_url: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and parse FPL Review projections CSV from string content, local file, or remote URL.
        Local CSV files (data/fplreview.csv or data/projections.csv) take priority over remote scraping.
        Returns a DataFrame if successful, or None on failure/timeout without interrupting the pipeline.
        """
        # 1. Direct CSV content provided (e.g. in unit tests or memory cache)
        if csv_content is not None:
            try:
                # Check for HTML error payload passed as string
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

        # 3. Remote HTTP Fetch (Fallback when no valid local CSV found)
        target_url = force_url or self.url
        if target_url:
            logger.info(f"Fetching FPL Review projections from remote endpoint: {target_url}")
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/csv, application/json, text/plain, */*",
            }
            try:
                resp = requests.get(target_url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    text_content = resp.text.strip()
                    # Check if response is valid CSV (not HTML 404/error page)
                    if text_content and not text_content.startswith("<!DOCTYPE") and not text_content.startswith("<html"):
                        try:
                            df = pd.read_csv(io.StringIO(text_content))
                            if not df.empty:
                                logger.info(f"Successfully fetched and parsed {len(df)} projections from {target_url}")
                                return df
                        except Exception as parse_err:
                            logger.warning(f"Could not parse response from {target_url} as CSV: {parse_err}")
                    else:
                        logger.warning(f"Remote endpoint {target_url} returned HTML or non-CSV payload.")
                else:
                    logger.warning(f"Remote projections fetch failed with HTTP status {resp.status_code}.")
            except requests.exceptions.Timeout:
                logger.warning(f"Remote fetch from {target_url} timed out after {self.timeout}s.")
            except requests.exceptions.RequestException as exc:
                logger.warning(f"Network error while fetching FPL Review projections: {exc}")
            except Exception as exc:
                logger.warning(f"Unexpected error during FPL Review projections fetch: {exc}")

        return None

    def map_to_bootstrap(
        self,
        fplreview_df: pd.DataFrame,
        bootstrap_data: Dict[str, Any],
        current_event: int = 1,
        decay_factor: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        Map FPL Review projection rows to official FPL element IDs.
        Calculates multi-gameweek discounted expected points:
            xP_effective = xP_t + (gamma * xP_{t+1}) + (gamma^2 * xP_{t+2})
        Returns: {element_id: {"fplreview_xp": float, "fplreview_xp_3gw": float}}
        """
        if fplreview_df is None or fplreview_df.empty:
            return {}

        gamma = float(decay_factor if decay_factor is not None else os.getenv("DECAY_FACTOR", "0.85"))

        elements = bootstrap_data.get("elements", [])
        teams = bootstrap_data.get("teams", [])

        # Build official lookup dicts
        team_id_to_name = {t["id"]: t["name"].lower() for t in teams}
        team_id_to_short = {t["id"]: t["short_name"].lower() for t in teams}

        # Normalize column names in projections DataFrame
        col_map = {col: str(col).strip() for col in fplreview_df.columns}
        df = fplreview_df.rename(columns=col_map)

        # Identify key columns in projections DataFrame
        id_col = None
        name_col = None
        team_col = None
        pos_col = None

        for col in df.columns:
            low = str(col).lower()
            if low in ["id", "fpl_id", "element", "code"]:
                id_col = col
            elif low in ["name", "player", "web_name", "full_name"]:
                name_col = col
            elif low in ["team", "team_name", "club"]:
                team_col = col
            elif low in ["pos", "position", "element_type", "type"]:
                pos_col = col

        # Identify gameweek projection column
        gw_col = None
        candidates = [
            f"{current_event}_Pts",
            f"{current_event}_pts",
            f"{current_event}_xP",
            f"{current_event}_xp",
            f"GW{current_event}_Pts",
            f"GW{current_event}",
            f"GW_{current_event}",
            f"{current_event}_Points",
            f"{current_event}_EV",
            "fplreview_xp",
            "xP",
            "xp",
            "EV",
            "ev",
            "Pts",
            "pts",
            "points",
            "Points",
            "ep_next",
            "proj_points",
        ]
        for cand in candidates:
            if cand in df.columns:
                gw_col = cand
                break

        # If no standard name found, look for any column matching '{gw}_' pattern or numeric
        if not gw_col:
            for col in df.columns:
                if re.match(rf"^{current_event}_", str(col), re.IGNORECASE) or re.match(rf"^gw{current_event}", str(col), re.IGNORECASE):
                    gw_col = col
                    break

        if not gw_col:
            logger.warning("Could not identify Gameweek xP projection column in FPL Review data.")
            return {}

        # Also identify next 3 gameweeks for discounted 3GW projection
        gw_plus_1_col = next((c for c in df.columns if re.match(rf"^(gw)?{current_event + 1}(_pts|_xp|_points)?$", str(c), re.IGNORECASE)), None)
        gw_plus_2_col = next((c for c in df.columns if re.match(rf"^(gw)?{current_event + 2}(_pts|_xp|_points)?$", str(c), re.IGNORECASE)), None)

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

            # Index by normalized name variants
            for n in filter(None, [web_norm, second_norm, full_norm]):
                official_by_norm_name.setdefault(n, []).append(el)
                if t_short:
                    official_by_team_and_name[f"{t_short}:{n}"] = el
                if t_name:
                    official_by_team_and_name[f"{t_name}:{n}"] = el

        mapped_projections: Dict[int, Dict[str, float]] = {}

        for _, row in df.iterrows():
            # Extract raw points projection for current gameweek
            try:
                raw_xp = float(row[gw_col])
                if pd.isna(raw_xp):
                    continue
                xp_val = round(max(0.0, raw_xp), 2)
            except (ValueError, TypeError):
                continue

            # Calculate discounted multi-week projection: xP_t + (gamma * xP_{t+1}) + (gamma^2 * xP_{t+2})
            multi_gw_sum = xp_val
            has_future_cols = False
            if gw_plus_1_col and gw_plus_1_col in row:
                try:
                    p1 = float(row[gw_plus_1_col])
                    if not pd.isna(p1):
                        multi_gw_sum += gamma * max(0.0, p1)
                        has_future_cols = True
                except (ValueError, TypeError):
                    pass
            if gw_plus_2_col and gw_plus_2_col in row:
                try:
                    p2 = float(row[gw_plus_2_col])
                    if not pd.isna(p2):
                        multi_gw_sum += (gamma ** 2) * max(0.0, p2)
                        has_future_cols = True
                except (ValueError, TypeError):
                    pass

            if has_future_cols:
                xp_3gw_val = round(multi_gw_sum, 2)
            else:
                # Horizon decay over 3-gameweek baseline lookahead
                decay_sum = 1.0 + gamma + (gamma ** 2)
                xp_3gw_val = round(xp_val * decay_sum, 2)

            matched_element_id: Optional[int] = None

            # 1. Match by numeric ID
            if id_col and id_col in row and not pd.isna(row[id_col]):
                try:
                    cand_id = int(row[id_col])
                    if cand_id in official_by_id:
                        matched_element_id = cand_id
                except (ValueError, TypeError):
                    pass

            # 2. Match by Name + Team
            if matched_element_id is None and name_col and name_col in row:
                raw_name = str(row[name_col])
                norm_name = _normalize_name(raw_name)
                raw_team = str(row[team_col]).lower().strip() if team_col and team_col in row and not pd.isna(row[team_col]) else ""

                # Try exact team + name match
                if raw_team:
                    key = f"{raw_team}:{norm_name}"
                    if key in official_by_team_and_name:
                        matched_element_id = official_by_team_and_name[key]["id"]

                # Try fuzzy/name-only match if unambiguous
                if matched_element_id is None and norm_name in official_by_norm_name:
                    candidates_list = official_by_norm_name[norm_name]
                    if len(candidates_list) == 1:
                        matched_element_id = candidates_list[0]["id"]
                    elif raw_team:
                        # Disambiguate by team name or team short name
                        for c in candidates_list:
                            c_tid = c.get("team", 1)
                            if team_id_to_short.get(c_tid) == raw_team or team_id_to_name.get(c_tid) == raw_team:
                                matched_element_id = c["id"]
                                break

            if matched_element_id is not None:
                mapped_projections[matched_element_id] = {
                    "fplreview_xp": xp_val,
                    "fplreview_xp_3gw": xp_3gw_val,
                }

        logger.info(f"Mapped {len(mapped_projections)} FPL Review projections to official FPL element IDs.")
        return mapped_projections


def calculate_fallback_xp(
    element: Dict[str, Any],
    next_fdr: float = 3.0,
    next_is_home: bool = False,
    avg_fdr: float = 3.0,
    availability: float = 1.0,
    decay_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate fallback baseline expected points for an individual player using
    ep_next, player form, points per game, and FDR rating without interrupting the pipeline.
    """
    gamma = float(decay_factor if decay_factor is not None else os.getenv("DECAY_FACTOR", "0.85"))
    decay_sum = 1.0 + gamma + (gamma ** 2)

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
    """Helper function to fetch FPL Review projections DataFrame."""
    fetcher = FPLReviewFetcher(url=url, file_path=file_path, timeout=timeout)
    return fetcher.fetch_projections(csv_content=csv_content)


def map_fplreview_to_elements(
    fplreview_df: pd.DataFrame,
    bootstrap_data: Dict[str, Any],
    current_event: int = 1,
    decay_factor: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:
    """Helper function to map FPL Review projections to official element IDs."""
    fetcher = FPLReviewFetcher()
    return fetcher.map_to_bootstrap(
        fplreview_df,
        bootstrap_data,
        current_event=current_event,
        decay_factor=decay_factor,
    )

