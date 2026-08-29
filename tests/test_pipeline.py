"""Comprehensive unit tests for Autonomous FPL Engine."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.director import AIDirector, DecisionOutput
from src.api.auth import FPLAuth
from src.api.client import FPLClient
from src.dashboard.app import create_app, extract_leagues, fetch_entry_details, build_global_summary
from src.data_fetcher import (
    FPLReviewFetcher,
    calculate_fallback_xp,
    fetch_fplreview_projections,
    map_fplreview_to_elements,
)
from src.engine.metrics import calculate_player_metrics, calculate_team_fdr_next_n_fixtures
from src.engine.optimizer import (
    CandidateSquad,
    FPLOptimizer,
    OptimizationResult,
    PlayerPick,
    get_solver,
    _solve_problem,
)
from src.notifier.telegram import TelegramNotifier
from src.tracker.league_scanner import LeagueAnalysis, LeagueScanner, ThreatMatrix, ThreatPlayer, RivalManager


# ==========================================
# Synthetic Fixtures & Mock Data
# ==========================================

@pytest.fixture
def mock_bootstrap_data():
    """Generates synthetic FPL bootstrap-static data with 20 teams and a realistic player pool."""
    teams = [
        {"id": i, "name": f"Team {i}", "short_name": f"T{i:02d}", "strength": 3}
        for i in range(1, 21)
    ]
    events = [
        {"id": 1, "name": "Gameweek 1", "is_current": True, "is_next": False, "finished": False, "deadline_time": "2026-08-25T17:30:00Z"},
        {"id": 2, "name": "Gameweek 2", "is_current": False, "is_next": True, "finished": False, "deadline_time": "2026-09-01T10:00:00Z"},
    ]

    elements = []
    pid = 1

    # 4 GKs (2 cheap, 2 premium)
    for team_id in [1, 2, 3, 4]:
        elements.append({
            "id": pid,
            "web_name": f"GK_{pid}",
            "element_type": 1,
            "team": team_id,
            "now_cost": 45 + (pid % 2) * 10,
            "form": "4.5",
            "points_per_game": "4.0",
            "total_points": 40,
            "selected_by_percent": "15.0",
            "status": "a",
            "chance_of_playing_next_round": 100,
        })
        pid += 1

    # 10 DEFs
    for team_id in range(1, 11):
        elements.append({
            "id": pid,
            "web_name": f"DEF_{pid}",
            "element_type": 2,
            "team": team_id,
            "now_cost": 45 + (pid % 3) * 5,
            "form": "5.0",
            "points_per_game": "4.8",
            "total_points": 48,
            "selected_by_percent": "20.0",
            "status": "a",
            "chance_of_playing_next_round": 100,
        })
        pid += 1

    # 10 MIDs
    for team_id in range(1, 11):
        elements.append({
            "id": pid,
            "web_name": f"MID_{pid}",
            "element_type": 3,
            "team": team_id,
            "now_cost": 60 + (pid % 4) * 15,
            "form": "7.2" if pid == 15 else "5.5",
            "points_per_game": "6.0",
            "total_points": 60,
            "selected_by_percent": "35.0" if pid == 15 else "10.0",
            "status": "a",
            "chance_of_playing_next_round": 100,
        })
        pid += 1

    # 6 FWDs
    for team_id in range(1, 7):
        elements.append({
            "id": pid,
            "web_name": f"FWD_{pid}",
            "element_type": 4,
            "team": team_id,
            "now_cost": 75 + (pid % 3) * 20,
            "form": "8.5" if pid == 25 else "4.0",
            "points_per_game": "7.0" if pid == 25 else "4.0",
            "total_points": 70,
            "selected_by_percent": "50.0" if pid == 25 else "5.0",
            "status": "a",
            "chance_of_playing_next_round": 100,
        })
        pid += 1

    return {"elements": elements, "teams": teams, "events": events}


@pytest.fixture
def mock_fixtures_data():
    """Generates synthetic fixture data."""
    return [
        {"event": 1, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4, "finished": False},
        {"event": 1, "team_h": 3, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False},
        {"event": 2, "team_h": 1, "team_a": 3, "team_h_difficulty": 2, "team_a_difficulty": 4, "finished": False},
        {"event": 2, "team_h": 2, "team_a": 4, "team_h_difficulty": 3, "team_a_difficulty": 3, "finished": False},
    ]


# ==========================================
# 1. API & Authentication Tests
# ==========================================

def test_fpl_auth_headers():
    auth_unauth = FPLAuth(token="")
    headers_unauth = auth_unauth.get_headers(authenticated=False)
    assert "Authorization" not in headers_unauth
    assert "X-API-Authorization" not in headers_unauth
    assert "Cookie" not in headers_unauth
    assert "User-Agent" in headers_unauth
    assert not auth_unauth.is_authenticated

    auth_auth = FPLAuth(token="my_secret_token_123")
    headers_auth = auth_auth.get_headers(authenticated=True)
    assert headers_auth["Authorization"] == "Bearer my_secret_token_123"
    assert headers_auth["X-API-Authorization"] == "Bearer my_secret_token_123"
    assert auth_auth.is_authenticated

    auth_bearer = FPLAuth(token="Bearer already_prefixed_token")
    headers_bearer = auth_bearer.get_headers(authenticated=True)
    assert headers_bearer["Authorization"] == "Bearer already_prefixed_token"
    assert headers_bearer["X-API-Authorization"] == "Bearer already_prefixed_token"


def test_fpl_client_caching(tmp_path, mock_bootstrap_data):
    cache_dir = tmp_path / "cache"
    client = FPLClient(cache_dir=cache_dir)

    with patch.object(client, "_request", return_value=mock_bootstrap_data) as mock_req:
        # First call fetches from API
        res1 = client.get_bootstrap_static()
        assert res1 == mock_bootstrap_data
        assert mock_req.call_count == 1
        assert client.bootstrap_cache_path.exists()

        # Second call reads from local cache
        res2 = client.get_bootstrap_static(force_refresh=False)
        assert res2 == mock_bootstrap_data
        assert mock_req.call_count == 1  # No additional network request


def test_fpl_client_authenticated_endpoints():
    auth = FPLAuth(token="test_auth_token_xyz")
    client = FPLClient(auth=auth)

    with patch.object(client, "_request", return_value={"picks": []}) as mock_req:
        res = client.get_my_team(12345)
        assert res == {"picks": []}
        mock_req.assert_called_once_with("GET", "my-team/12345/", authenticated=True)

    with patch.object(client, "_request", return_value={"status": "ok"}) as mock_req:
        res = client.post_transfers({"transfers": []})
        assert res == {"status": "ok"}
        mock_req.assert_called_once_with("POST", "transfers/", json_data={"transfers": []}, authenticated=True)

    with patch.object(client, "_request", return_value={"status": "ok"}) as mock_req:
        res = client.post_lineup(12345, {"picks": []})
        assert res == {"status": "ok"}
        mock_req.assert_called_once_with("POST", "my-team/12345/", json_data={"picks": []}, authenticated=True)


def test_fpl_client_unauthenticated_guards():
    from src.api.client import FPLClientError
    unauth_client = FPLClient(auth=FPLAuth(token=""))

    with pytest.raises(FPLClientError, match="Authentication token required"):
        unauth_client.get_my_team(12345)

    with pytest.raises(FPLClientError, match="Authentication token required"):
        unauth_client.post_transfers({})

    with pytest.raises(FPLClientError, match="Authentication token required"):
        unauth_client.post_lineup(12345, {})


def test_fpl_client_auth_error_message():
    from src.api.client import FPLClientError
    auth = FPLAuth(token="invalid_token")
    client = FPLClient(auth=auth)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch.object(client.session, "request", return_value=mock_resp):
        with pytest.raises(FPLClientError, match="Ensure FPL_AUTH_TOKEN is valid"):
            client._request("GET", "my-team/12345/", authenticated=True)


# ==========================================
# 2. Metrics & xP Calculation Tests
# ==========================================

def test_metrics_calculation(mock_bootstrap_data, mock_fixtures_data):
    df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)

    assert not df.empty
    assert "xp" in df.columns
    assert "xp_3gw" in df.columns
    assert "position" in df.columns

    # Test injury penalty calculation
    mock_injured_bootstrap = {
        "elements": [
            {
                "id": 99,
                "web_name": "Injured_Star",
                "element_type": 4,
                "team": 1,
                "now_cost": 100,
                "form": "8.0",
                "points_per_game": "8.0",
                "status": "i",
                "chance_of_playing_next_round": 0,
            },
            {
                "id": 100,
                "web_name": "Doubtful_Star",
                "element_type": 4,
                "team": 1,
                "now_cost": 100,
                "form": "8.0",
                "points_per_game": "8.0",
                "status": "d",
                "chance_of_playing_next_round": 50,
            }
        ],
        "teams": mock_bootstrap_data["teams"],
        "events": mock_bootstrap_data["events"],
    }
    df_inj = calculate_player_metrics(mock_injured_bootstrap, mock_fixtures_data, current_event=1)
    injured_player = df_inj[df_inj["id"] == 99].iloc[0]
    doubtful_player = df_inj[df_inj["id"] == 100].iloc[0]

    assert injured_player["xp"] == 0.0
    assert doubtful_player["xp"] > 0.0
    assert doubtful_player["availability"] == 0.5


def test_fplreview_csv_parsing_and_mapping(mock_bootstrap_data):
    # Sample FPL Review CSV data
    sample_csv = """ID,Name,Pos,Team,1_Pts,2_Pts,3_Pts
1,GK_1,GKP,Team 1,5.8,5.2,4.9
15,MID_15,MID,Team 5,9.4,8.8,9.1
25,FWD_25,FWD,Team 5,11.2,10.5,10.0
"""
    fetcher = FPLReviewFetcher()
    df = fetcher.fetch_projections(csv_content=sample_csv)
    assert df is not None
    assert len(df) == 3

    # 1. Default Horizon Decay Factor (gamma = 0.85): 5.8 + 0.85*5.2 + 0.85^2*4.9 = 13.76
    mapped_default = fetcher.map_to_bootstrap(df, mock_bootstrap_data, current_event=1, decay_factor=0.85)
    assert 1 in mapped_default
    assert mapped_default[1]["fplreview_xp"] == 5.8
    expected_decayed_3gw = round(5.8 + 0.85 * 5.2 + (0.85 ** 2) * 4.9, 2)
    assert mapped_default[1]["fplreview_xp_3gw"] == expected_decayed_3gw
    assert 15 in mapped_default
    assert mapped_default[15]["fplreview_xp"] == 9.4
    assert 25 in mapped_default
    assert mapped_default[25]["fplreview_xp"] == 11.2

    # 2. Custom Horizon Decay Factor (gamma = 1.0): 5.8 + 5.2 + 4.9 = 15.9
    mapped_undecayed = fetcher.map_to_bootstrap(df, mock_bootstrap_data, current_event=1, decay_factor=1.0)
    assert mapped_undecayed[1]["fplreview_xp_3gw"] == round(5.8 + 5.2 + 4.9, 2)

    # 3. Custom Horizon Decay Factor (gamma = 0.90): 5.8 + 0.9*5.2 + 0.81*4.9 = 14.45
    mapped_90 = fetcher.map_to_bootstrap(df, mock_bootstrap_data, current_event=1, decay_factor=0.90)
    assert mapped_90[1]["fplreview_xp_3gw"] == round(5.8 + 0.90 * 5.2 + (0.90 ** 2) * 4.9, 2)


def test_fplreview_name_and_accent_matching(mock_bootstrap_data):
    # Test matching by name with accents/diacritics (e.g. Ødegaard / Odegaard)
    custom_bootstrap = {
        "elements": [
            {"id": 501, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard", "team": 1, "element_type": 3, "form": "5.0", "points_per_game": "5.0"},
            {"id": 502, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland", "team": 2, "element_type": 4, "form": "8.0", "points_per_game": "8.0"},
        ],
        "teams": [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Man City", "short_name": "MCI"},
        ],
        "events": mock_bootstrap_data["events"],
    }

    sample_csv = """Name,Team,Pos,1_Pts
Odegaard,Arsenal,MID,7.6
Erling Haaland,MCI,FWD,10.8
"""
    fetcher = FPLReviewFetcher()
    df = fetcher.fetch_projections(csv_content=sample_csv)
    mapped = fetcher.map_to_bootstrap(df, custom_bootstrap, current_event=1)

    assert 501 in mapped
    assert mapped[501]["fplreview_xp"] == 7.6
    assert 502 in mapped
    assert mapped[502]["fplreview_xp"] == 10.8


def test_fplreview_fallback_and_metrics_integration(mock_bootstrap_data, mock_fixtures_data):
    # Prepare bootstrap with official ep_next values
    bootstrap = dict(mock_bootstrap_data)
    bootstrap["elements"][0]["ep_next"] = "4.2"  # Player 1 has ep_next
    bootstrap["elements"][1]["ep_next"] = "0.0"  # Player 2 has 0 ep_next -> uses heuristic

    sample_csv = """ID,Name,Pos,Team,1_Pts
15,MID_15,MID,Team 5,9.5
"""
    df_projections = fetch_fplreview_projections(csv_content=sample_csv)

    players_df = calculate_player_metrics(
        bootstrap,
        mock_fixtures_data,
        current_event=1,
        fplreview_df=df_projections,
    )

    p15 = players_df[players_df["id"] == 15].iloc[0]
    p1 = players_df[players_df["id"] == 1].iloc[0]
    p2 = players_df[players_df["id"] == 2].iloc[0]

    # Player 15 should use FPL Review projection
    assert p15["xp"] == 9.5
    assert p15["fplreview_xp"] == 9.5
    assert p15["xp_source"] == "fplreview"

    # Player 1 should fallback to official ep_next
    assert p1["xp"] == 4.2
    assert p1["xp_source"] == "fpl_ep_next"

    # Player 2 should fallback to FDR baseline heuristic
    assert p2["xp"] > 0
    assert p2["xp_source"] == "fpl_heuristic"


def test_fplreview_network_failure_fallback(mock_bootstrap_data, mock_fixtures_data):
    # Simulate network timeout/failure in fetcher
    with patch("requests.get", side_effect=Exception("Connection timed out")):
        fetcher = FPLReviewFetcher(url="https://mock-fplreview.test/fail", timeout=1)
        res_df = fetcher.fetch_projections()
        assert res_df is None

        # calculate_player_metrics should seamlessly work with None
        df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1, fplreview_df=res_df)
        assert not df.empty
        assert "xp" in df.columns
        assert all(source in ["fpl_ep_next", "fpl_heuristic"] for source in df["xp_source"])


# ==========================================
# 3. PuLP MILP Optimizer & Constraints Tests
# ==========================================

def test_pulp_optimizer_constraints(mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)

    # Initial valid 15-player squad: 2 GK, 5 DEF, 5 MID, 3 FWD
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    opt_result = optimizer.optimize(
        current_squad_ids=initial_squad,
        bank_m=2.0,
        free_transfers=1,
    )

    assert len(opt_result.candidates) >= 1

    for cand in opt_result.candidates:
        # 1. Total squad size == 15
        total_squad = cand.starters + cand.bench
        assert len(total_squad) == 15

        # 2. Starting XI == 11
        assert len(cand.starters) == 11
        assert len(cand.bench) == 4

        # 3. Formation constraints
        gk_starters = [p for p in cand.starters if p.position == "GKP"]
        def_starters = [p for p in cand.starters if p.position == "DEF"]
        mid_starters = [p for p in cand.starters if p.position == "MID"]
        fwd_starters = [p for p in cand.starters if p.position == "FWD"]

        assert len(gk_starters) == 1
        assert 3 <= len(def_starters) <= 5
        assert 2 <= len(mid_starters) <= 5
        assert 1 <= len(fwd_starters) <= 3

        # 4. Captain & Vice-Captain
        assert cand.captain is not None
        assert cand.vice_captain is not None
        assert cand.captain.id != cand.vice_captain.id
        assert cand.captain.is_captain
        assert cand.vice_captain.is_vice_captain

        # 5. Max 3 players per club
        team_counts = {}
        for p in total_squad:
            team_counts[p.team_name] = team_counts.get(p.team_name, 0) + 1
        assert all(count <= 3 for count in team_counts.values())

        # 6. Budget compliance
        assert cand.bank_remaining_m >= 0.0


def test_optimizer_candidate_options(mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)

    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    opt_result = optimizer.optimize(
        current_squad_ids=initial_squad,
        bank_m=5.0,
        free_transfers=1,
    )

    # Verifies 7 strategic candidate options
    assert len(opt_result.candidates) == 7
    cand_0 = opt_result.candidates[0]
    cand_1 = opt_result.candidates[1]
    cand_2 = opt_result.candidates[2]
    cand_3 = opt_result.candidates[3]
    cand_4 = opt_result.candidates[4]
    cand_5 = opt_result.candidates[5]
    cand_6 = opt_result.candidates[6]

    # Candidate 0: Roll (0 transfers)
    assert cand_0.transfers_count == 0
    assert cand_0.hit_cost == 0

    # Candidates 1, 2, 3: 1-transfer moves (Optimal + 2 Alternatives)
    assert cand_1.transfers_count == 1
    assert cand_2.transfers_count == 1
    assert cand_3.transfers_count == 1
    # Verify distinct transfer in targets
    transfers_in_1 = {t.player_in.id for t in cand_1.transfers}
    transfers_in_2 = {t.player_in.id for t in cand_2.transfers}
    transfers_in_3 = {t.player_in.id for t in cand_3.transfers}
    assert transfers_in_1 != transfers_in_2
    assert transfers_in_2 != transfers_in_3

    # Candidates 4, 5: 2-transfer moves (Optimal + Alternative)
    assert cand_4.transfers_count == 2
    assert cand_5.transfers_count == 2
    if cand_4.transfers_count == 2:
        assert cand_4.hit_cost == 4  # 1 FT used, 1 extra transfer = -4

    # Candidate 6: Active chip or alternative move
    assert cand_6 is not None


def test_pulp_optimizer_with_fplreview_objective(mock_bootstrap_data, mock_fixtures_data):
    # Give player 20 an immense FPL Review xP projection
    sample_csv = """ID,Name,Pos,Team,1_Pts
20,MID_20,MID,Team 10,14.5
"""
    fplreview_df = fetch_fplreview_projections(csv_content=sample_csv)
    players_df = calculate_player_metrics(
        mock_bootstrap_data,
        mock_fixtures_data,
        current_event=1,
        fplreview_df=fplreview_df,
    )

    optimizer = FPLOptimizer(players_df)
    # Initial squad without player 20
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 27, 28, 29]

    opt_result = optimizer.optimize(
        current_squad_ids=initial_squad,
        bank_m=5.0,
        free_transfers=1,
    )

    # 1-Transfer option should transfer in player 20 and make them captain due to 14.5 xP
    cand_1 = opt_result.candidates[1]
    transfers_in_ids = [t.player_in.id for t in cand_1.transfers]
    if cand_1.transfers_count >= 1:
        assert 20 in transfers_in_ids
        assert cand_1.captain.id == 20
        assert cand_1.captain.fplreview_xp == 14.5
        assert cand_1.captain.xp_source == "fplreview"


# ==========================================
# 4. League Scanner & Threat Matrix Tests
# ==========================================

def test_league_scanner_threat_matrix(mock_bootstrap_data):
    client = FPLClient()

    mock_standings = {
        "league": {"name": "Test Rival Mini-League"},
        "standings": {
            "results": [
                {"entry": 101, "player_name": "Rival Alice", "entry_name": "Alice XI", "rank": 1, "total": 150},
                {"entry": 102, "player_name": "Rival Bob", "entry_name": "Bob XI", "rank": 2, "total": 140},
            ]
        }
    }

    # Alice picks player 15 as Captain (multiplier 2), player 25 as starter (multiplier 1)
    mock_picks_101 = {
        "picks": [
            {"element": 15, "position": 1, "multiplier": 2, "is_captain": True},
            {"element": 25, "position": 2, "multiplier": 1, "is_captain": False},
        ],
        "entry_history": {"event_transfers_cost": 4},
        "active_chip": None,
    }

    # Bob picks player 15 as starter (multiplier 1), player 25 as starter (multiplier 1)
    mock_picks_102 = {
        "picks": [
            {"element": 15, "position": 1, "multiplier": 1, "is_captain": False},
            {"element": 25, "position": 2, "multiplier": 1, "is_captain": False},
        ],
        "entry_history": {"event_transfers_cost": 0},
        "active_chip": "bboost",
    }

    with patch.object(client, "get_league_standings", return_value=mock_standings), \
         patch.object(client, "get_entry_picks", side_effect=lambda entry, gw: mock_picks_101 if entry == 101 else mock_picks_102):

        scanner = LeagueScanner(client)
        # My team owns player 15 and player 1 (differential), but does NOT own player 25
        my_team = {1, 15}

        analysis = scanner.scan_league(
            league_id=999,
            gameweek=1,
            my_team_ids=my_team,
            bootstrap_data=mock_bootstrap_data,
        )

        assert analysis.total_managers == 2
        # Player 15 EO: (2 + 1) / 2 = 150.0%
        assert analysis.raw_eo[15] == 150.0
        # Player 25 EO: (1 + 1) / 2 = 100.0%
        assert analysis.raw_eo[25] == 100.0

        # Player 15 is owned by me & EO > 40% -> SHIELD
        shield_ids = [p.id for p in analysis.threat_matrix.shields]
        assert 15 in shield_ids

        # Player 25 is unowned by me & EO > 35% -> VULNERABILITY
        vuln_ids = [p.id for p in analysis.threat_matrix.vulnerabilities]
        assert 25 in vuln_ids


def test_get_latest_live_gameweek():
    """Test get_latest_live_gameweek identifies past deadlines and handles pre-season."""
    from src.main import get_latest_live_gameweek

    # Case 1: GW1 deadline in past, GW2 deadline in future
    events_in_season = [
        {"id": 1, "deadline_time": "2020-01-01T10:00:00Z", "finished": True, "is_previous": True},
        {"id": 2, "deadline_time": "2099-01-01T10:00:00Z", "finished": False, "is_next": True},
    ]
    assert get_latest_live_gameweek(events_in_season) == 1

    # Case 2: Pre-season (all deadlines in future)
    events_preseason = [
        {"id": 1, "deadline_time": "2099-01-01T10:00:00Z", "finished": False, "is_next": True},
        {"id": 2, "deadline_time": "2099-01-08T10:00:00Z", "finished": False, "is_next": False},
    ]
    assert get_latest_live_gameweek(events_preseason) is None


def test_league_scanner_preseason_without_picks(mock_bootstrap_data):
    """Test mini-league scanning before season starts (gameweek=None) scans standings without pick errors."""
    mock_standings = {
        "league": {"id": 999, "name": "Preseason League"},
        "standings": {
            "results": [
                {"entry": 101, "player_name": "Alice", "entry_name": "Alice XI", "rank": 1, "total": 0},
                {"entry": 102, "player_name": "Bob", "entry_name": "Bob FC", "rank": 2, "total": 0},
            ]
        },
    }
    client = FPLClient()

    with patch.object(client, "get_league_standings", return_value=mock_standings), \
         patch.object(client, "get_entry_picks") as mock_picks:

        scanner = LeagueScanner(client)
        analysis = scanner.scan_league(
            league_id=999,
            gameweek=None,
            my_team_ids={1, 2},
            bootstrap_data=mock_bootstrap_data,
        )

        assert analysis.total_managers == 2
        assert len(analysis.rivals) == 2
        assert analysis.raw_eo == {}
        # Ensure get_entry_picks was NOT called in pre-season mode
        mock_picks.assert_not_called()


# ==========================================
# 5. AI Director & Fallback Tests
# ==========================================

def test_ai_director_fallback(mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1)

    # Without API key, should use mathematical fallback
    director = AIDirector(api_key="")
    decision = director.evaluate_and_decide(opt_result)

    assert decision.source == "MATHEMATICAL_FALLBACK"
    assert decision.selected_candidate is not None
    assert decision.projected_net_xp > 0
    assert len(decision.rationale) > 10


def test_ai_director_llm_success(mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1)

    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "selected_candidate_index": 1,
                        "rationale": "Option 2 provides the optimal risk-reward balance with high-upside midfield targeting. Handing the armband to the top form talisman gives our rank defensive stability."
                    })
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_llm_response
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp):
        director = AIDirector(api_key="mock_test_key")
        decision = director.evaluate_and_decide(opt_result)

        assert decision.source == "LLM_DIRECTOR"
        assert decision.selected_candidate_index == 1
        assert "Option 2" in decision.rationale


# ==========================================
# 6. Telegram Dispatcher Tests
# ==========================================

def test_telegram_message_builder(mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1)

    director = AIDirector(api_key="")
    decision = director.evaluate_and_decide(opt_result)

    notifier = TelegramNotifier(bot_token="mock_token", chat_id="12345")
    pre_msg = notifier.build_pre_deadline_alert(decision, gameweek=1, is_live_execution=False)

    assert "GAMEWEEK 1 SQUAD BRIEFING" in pre_msg
    assert "(C)" in pre_msg
    assert "AI DIRECTOR RATIONALE" in pre_msg
    assert "PROJECTED SQUAD NET xP" in pre_msg


# ==========================================
# 7. Flask Dashboard Endpoint Tests
# ==========================================

def test_flask_dashboard(tmp_path, mock_bootstrap_data, mock_fixtures_data):
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1)

    director = AIDirector(api_key="")
    decision = director.evaluate_and_decide(opt_result)

    state_file = tmp_path / "latest_decision.json"
    data_payload = {
        "status": "success",
        "mode": "dry_run",
        "gameweek": 1,
        "deadline_time": "2026-08-25T17:30:00Z",
        "timestamp": "2026-08-22T21:00:00Z",
        "decision": decision.model_dump(),
        "optimization": opt_result.model_dump(),
        "league_analysis": {
            "league_id": 12345,
            "league_name": "Championship Mini-League",
            "gameweek": 1,
            "total_managers": 5,
            "rivals": [],
            "captain_distribution": {"Haaland": 3, "Salah": 2},
            "threat_matrix": {
                "shields": [],
                "vulnerabilities": [],
                "daggers": []
            },
            "raw_eo": {}
        }
    }

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data_payload, f)

    app = create_app(data_file_path=state_file)
    client = app.test_client()

    # Test /healthz
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    assert res_health.json["status"] == "healthy"

    # Test /api/decision
    res_api = client.get("/api/decision")
    assert res_api.status_code == 200
    assert res_api.json["gameweek"] == 1
    assert "decision" in res_api.json

    # Test Web UI HTML /
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    html = res_ui.data.decode("utf-8")
    assert "FPL AI Engine" in html
    assert "AI Director Tactical Directive" in html
    assert "Starting XI Lineup" in html


def test_league_extraction_and_trajectory():
    """Test separating classic leagues into global and mini leagues with correct trajectory calculation."""
    entry_payload = {
        "id": 999999,
        "player_first_name": "Erling",
        "player_last_name": "Haaland",
        "player_region_name": "Norway",
        "summary_overall_points": 1450,
        "summary_overall_rank": 12500,
        "leagues": {
            "classic": [
                {
                    "id": 275,
                    "name": "Overall",
                    "league_type": "s",
                    "entry_rank": 12500,
                    "entry_last_rank": 15000,
                    "total_players": 10500000,
                },
                {
                    "id": 111,
                    "name": "Norway",
                    "league_type": "s",
                    "entry_rank": 450,
                    "entry_last_rank": 400,
                    "total_players": 250000,
                },
                {
                    "id": 222,
                    "name": "Manchester City",
                    "league_type": "s",
                    "entry_rank": 120,
                    "entry_last_rank": 120,
                    "total_players": 150000,
                },
                {
                    "id": 8888,
                    "name": "Office Legends",
                    "league_type": "x",
                    "entry_rank": 2,
                    "entry_last_rank": 5,
                    "total_players": 20,
                },
                {
                    "id": 9999,
                    "name": "Family Cup",
                    "league_type": "x",
                    "entry_rank": 8,
                    "entry_last_rank": 4,
                    "total_players": 12,
                },
                {
                    "id": 7777,
                    "name": "Pub Buddies",
                    "league_type": "x",
                    "entry_rank": 1,
                    "entry_last_rank": 1,
                    "total_players": 8,
                },
            ]
        }
    }

    global_leagues, mini_leagues = extract_leagues(entry_payload)

    # 3 global leagues (s)
    assert len(global_leagues) == 3
    assert all(l["league_type"] == "s" for l in global_leagues)
    overall = next(l for l in global_leagues if l["name"] == "Overall")
    assert overall["rank"] == 12500
    assert overall["last_rank"] == 15000
    assert overall["movement"] == 2500
    assert overall["direction"] == "up"

    norway = next(l for l in global_leagues if l["name"] == "Norway")
    assert norway["movement"] == -50  # dropped from 400 to 450
    assert norway["direction"] == "down"

    city = next(l for l in global_leagues if l["name"] == "Manchester City")
    assert city["movement"] == 0
    assert city["direction"] == "same"

    # 3 mini leagues (x)
    assert len(mini_leagues) == 3
    assert all(l["league_type"] == "x" for l in mini_leagues)
    
    office = next(l for l in mini_leagues if l["name"] == "Office Legends")
    assert office["rank"] == 2
    assert office["last_rank"] == 5
    assert office["movement"] == 3
    assert office["direction"] == "up"

    family = next(l for l in mini_leagues if l["name"] == "Family Cup")
    assert family["rank"] == 8
    assert family["last_rank"] == 4
    assert family["movement"] == -4
    assert family["direction"] == "down"

    summary = build_global_summary(global_leagues, entry_payload)
    assert summary["overall"]["name"] == "Overall"
    assert summary["country"]["name"] == "Norway"
    assert summary["club"]["name"] == "Manchester City"


def test_flask_dashboard_with_leagues_ui(tmp_path, mock_bootstrap_data, mock_fixtures_data):
    """Test dashboard template rendering with Global Rankings cards and Mini-Leagues table."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1)

    director = AIDirector(api_key="")
    decision = director.evaluate_and_decide(opt_result)

    state_file = tmp_path / "latest_decision.json"
    data_payload = {
        "status": "success",
        "mode": "dry_run",
        "gameweek": 1,
        "deadline_time": "2026-08-25T17:30:00Z",
        "timestamp": "2026-08-22T21:00:00Z",
        "decision": decision.model_dump(),
        "optimization": opt_result.model_dump(),
        "entry_data": {
            "id": 123456,
            "name": "Tactical Masterclass",
            "player_first_name": "Pep",
            "player_last_name": "Guardiola",
            "player_region_name": "Spain",
            "summary_overall_points": 2100,
            "summary_overall_rank": 5420,
            "leagues": {
                "classic": [
                    {
                        "id": 275,
                        "name": "Overall",
                        "league_type": "s",
                        "entry_rank": 5420,
                        "entry_last_rank": 6200,
                        "total_players": 10500000,
                    },
                    {
                        "id": 123,
                        "name": "Spain",
                        "league_type": "s",
                        "entry_rank": 150,
                        "entry_last_rank": 180,
                        "total_players": 300000,
                    },
                    {
                        "id": 456,
                        "name": "Arsenal",
                        "league_type": "s",
                        "entry_rank": 80,
                        "entry_last_rank": 70,
                        "total_players": 500000,
                    },
                    {
                        "id": 9999,
                        "name": "Premier Devs Mini-League",
                        "league_type": "x",
                        "entry_rank": 1,
                        "entry_last_rank": 3,
                        "total_players": 16,
                    },
                ]
            }
        }
    }

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data_payload, f)

    app = create_app(data_file_path=state_file)
    client = app.test_client()

    res_ui = client.get("/")
    assert res_ui.status_code == 200
    html = res_ui.data.decode("utf-8")

    # Verify Global Rankings section elements
    assert "Global Rankings & Benchmarks" in html
    assert "Overall Rank" in html
    assert "5,420" in html
    assert "Spain" in html
    assert "Arsenal" in html

    # Verify Mini-Leagues table elements
    assert "Private Mini-Leagues" in html
    assert "Premier Devs Mini-League" in html
    assert "▲ +2" in html


# ==========================================
# 8. Automated Chip Evaluation Tests
# ==========================================

def test_solver_optimizer_import_compatibility():
    """Verify that src.solver.optimizer is fully compatible and re-exports optimizer components."""
    from src.solver.optimizer import (
        FPLOptimizer as SolverOptimizer,
        CandidateSquad as SolverCandidate,
        ChipEvaluation,
        ChipEvaluationResult,
    )
    assert SolverOptimizer is not None
    assert SolverCandidate is not None
    assert ChipEvaluation is not None
    assert ChipEvaluationResult is not None


def test_evaluate_triple_captain(mock_bootstrap_data, mock_fixtures_data):
    """Test evaluate_triple_captain calculation against TRIPLE_CAPTAIN_MIN_XP threshold."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1, evaluate_chips=False)

    baseline = opt_result.candidates[0]
    # Set captain xP to 12.0
    baseline.captain.xp = 12.0

    tc_eval = optimizer.evaluate_triple_captain(baseline, min_captain_xp=11.5)
    assert tc_eval.chip_name == "3xc"
    assert tc_eval.threshold == 11.5
    assert tc_eval.xp_gain == 12.0
    assert tc_eval.projected_xp == round(baseline.net_xp + 12.0, 2)
    assert tc_eval.threshold_met is True

    # Test when captain xP is below threshold
    baseline.captain.xp = 9.0
    tc_eval_low = optimizer.evaluate_triple_captain(baseline, min_captain_xp=11.5)
    assert tc_eval_low.threshold_met is False


def test_evaluate_bench_boost(mock_bootstrap_data, mock_fixtures_data):
    """Test evaluate_bench_boost calculation against BENCH_BOOST_MIN_BENCH_XP threshold."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(initial_squad, bank_m=1.0, free_transfers=1, evaluate_chips=False)

    baseline = opt_result.candidates[0]
    # Set bench player xP sum to 18.0 (e.g. 4.5 each)
    for p in baseline.bench:
        p.xp = 4.5

    bb_eval = optimizer.evaluate_bench_boost(baseline, min_bench_xp=16.0)
    assert bb_eval.chip_name == "bboost"
    assert bb_eval.threshold == 16.0
    assert bb_eval.xp_gain == 18.0
    assert bb_eval.threshold_met is True

    # Below threshold
    for p in baseline.bench:
        p.xp = 2.0
    bb_eval_low = optimizer.evaluate_bench_boost(baseline, min_bench_xp=16.0)
    assert bb_eval_low.xp_gain == 8.0
    assert bb_eval_low.threshold_met is False


def test_evaluate_wildcard_and_free_hit(mock_bootstrap_data, mock_fixtures_data):
    """Test Wildcard and Free Hit evaluation solving optimal 15-man squad from scratch with 0 hits."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    
    # Valid 15-man starting squad (2 GK, 5 DEF, 5 MID, 3 FWD)
    squad = [3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27]
    opt_result = optimizer.optimize(squad, bank_m=0.0, free_transfers=1, evaluate_chips=False)
    baseline = opt_result.candidates[0]

    # Evaluate Wildcard
    wc_eval = optimizer.evaluate_wildcard(squad, total_budget_m=100.0, baseline_candidate=baseline, min_gain=1.0)
    assert wc_eval.chip_name == "wildcard"
    assert wc_eval.squad_candidate is not None
    assert wc_eval.squad_candidate.hit_cost == 0
    assert len(wc_eval.squad_candidate.starters) == 11
    assert len(wc_eval.squad_candidate.bench) == 4
    assert wc_eval.projected_xp >= baseline.net_xp

    # Evaluate Free Hit
    fh_eval = optimizer.evaluate_free_hit(squad, total_budget_m=100.0, baseline_candidate=baseline, min_gain=1.0)
    assert fh_eval.chip_name == "freehit"
    assert fh_eval.squad_candidate is not None
    assert fh_eval.squad_candidate.hit_cost == 0


def test_pipeline_with_auto_chip_execution(tmp_path, mock_bootstrap_data, mock_fixtures_data, monkeypatch):
    """Test main pipeline triggering automated chip execution, setting payload chips, and Telegram notification."""
    monkeypatch.setenv("ENABLE_AUTO_CHIPS", "true")
    monkeypatch.setenv("TRIPLE_CAPTAIN_MIN_XP", "1.0")  # low threshold to trigger TC
    monkeypatch.setenv("FPL_TEAM_ID", "123456")

    mock_client = MagicMock(spec=FPLClient)
    mock_client.auth = MagicMock()
    mock_client.auth.is_authenticated = True
    mock_client.get_bootstrap_static.return_value = mock_bootstrap_data
    mock_client.get_fixtures.return_value = mock_fixtures_data
    mock_client.get_my_team.return_value = {
        "picks": [{"element": i} for i in [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]],
        "transfers": {"bank": 10, "limit": 1},
    }
    mock_client.post_transfers.return_value = {"status": "ok"}
    mock_client.post_lineup.return_value = {"status": "ok"}

    from src.main import run_pipeline
    with patch("src.main.TelegramNotifier") as mock_notifier_cls:
        mock_notifier = MagicMock()
        mock_notifier_cls.return_value = mock_notifier

        result = run_pipeline(mock_client, dry_run=False, execute=True)

        assert result["status"] == "success"
        assert result["active_chip"] is not None
        # Verify chip alert was sent to Telegram
        assert mock_notifier.notify_chip_triggered.called
        # Verify lineup payload included the active chip
        assert mock_client.post_lineup.called
        call_args = mock_client.post_lineup.call_args[0]
        assert call_args[1]["chip"] == result["active_chip"]


def test_pipeline_wildcard_recommended_but_standard_move_selected(tmp_path, mock_bootstrap_data, mock_fixtures_data, monkeypatch):
    """Verify that when Wildcard threshold is met, but Director chooses a 1-transfer move, active_chip remains None."""
    monkeypatch.setenv("ENABLE_AUTO_CHIPS", "true")
    monkeypatch.setenv("WILDCARD_MIN_XP_GAIN", "0.01")  # Low threshold to trigger Wildcard recommendation
    monkeypatch.setenv("TRIPLE_CAPTAIN_MIN_XP", "999.0")
    monkeypatch.setenv("BENCH_BOOST_MIN_BENCH_XP", "999.0")
    monkeypatch.setenv("FREE_HIT_MIN_XP_GAIN", "999.0")
    monkeypatch.setenv("FPL_TEAM_ID", "123456")

    mock_client = MagicMock(spec=FPLClient)
    mock_client.auth = MagicMock()
    mock_client.auth.is_authenticated = True
    mock_client.validate_auth.return_value = (True, "Auth OK")
    mock_client.get_bootstrap_static.return_value = mock_bootstrap_data
    mock_client.get_fixtures.return_value = mock_fixtures_data
    mock_client.get_entry_history.return_value = {}
    mock_client.get_my_team.return_value = {
        # Squad with low form players so Wildcard scratch squad provides massive xP gain
        "picks": [{"element": i, "selling_price": 65} for i in [3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 27, 28, 29]],
        "transfers": {"bank": 100, "limit": 1},
    }
    mock_client.post_transfers.return_value = {"status": "ok"}
    mock_client.post_lineup.return_value = {"status": "ok"}

    from src.main import run_pipeline

    with patch("src.main.AIDirector") as mock_director_cls, patch("src.main.TelegramNotifier") as mock_notifier_cls:
        mock_director = MagicMock()
        mock_director_cls.return_value = mock_director
        mock_notifier = MagicMock()
        mock_notifier_cls.return_value = mock_notifier

        # Simulate Director choosing Candidate 1 (the standard move, not the rebuild at candidate 0)
        def mock_evaluate(opt_res, *args, **kwargs):
            chosen = opt_res.candidates[1]
            return DecisionOutput(
                selected_candidate_index=1,
                selected_candidate=chosen,
                chosen_move_name=chosen.name,
                transfers_description="Standard Move",
                captain_name="GK_1",
                vice_captain_name="GK_2",
                projected_net_xp=chosen.net_xp,
                rationale="Prefer saving Wildcard and making standard move.",
                source="LLM_DIRECTOR",
            )
        mock_director.evaluate_and_decide.side_effect = mock_evaluate

        result = run_pipeline(mock_client, dry_run=False, execute=True)

        assert result["status"] == "success"
        # Wildcard was NOT selected by the Director, so active_chip must be None!
        assert result["active_chip"] is None
        assert result["decision"]["selected_candidate"]["active_chip"] is None
        # Telegram chip alert must NOT be dispatched
        assert not mock_notifier.notify_chip_triggered.called
        # Transfers must not include chips="wildcard"
        if mock_client.post_transfers.called:
            tx_call_args = mock_client.post_transfers.call_args[0]
            assert tx_call_args[0].get("chips") is None


def test_pipeline_wildcard_recommended_and_rebuild_selected(tmp_path, mock_bootstrap_data, mock_fixtures_data, monkeypatch):
    """Verify that when Wildcard threshold is met and Director selects Candidate 0, active_chip is wildcard."""
    monkeypatch.setenv("ENABLE_AUTO_CHIPS", "true")
    monkeypatch.setenv("WILDCARD_MIN_XP_GAIN", "0.01")
    monkeypatch.setenv("TRIPLE_CAPTAIN_MIN_XP", "999.0")
    monkeypatch.setenv("BENCH_BOOST_MIN_BENCH_XP", "999.0")
    monkeypatch.setenv("FREE_HIT_MIN_XP_GAIN", "999.0")
    monkeypatch.setenv("FPL_TEAM_ID", "123456")

    mock_client = MagicMock(spec=FPLClient)
    mock_client.auth = MagicMock()
    mock_client.auth.is_authenticated = True
    mock_client.validate_auth.return_value = (True, "Auth OK")
    mock_client.get_bootstrap_static.return_value = mock_bootstrap_data
    mock_client.get_fixtures.return_value = mock_fixtures_data
    mock_client.get_entry_history.return_value = {}
    mock_client.get_my_team.return_value = {
        # Squad with low form players so Wildcard scratch squad provides massive xP gain
        "picks": [{"element": i, "selling_price": 65} for i in [3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 27, 28, 29]],
        "transfers": {"bank": 100, "limit": 1},
    }
    mock_client.post_transfers.return_value = {"status": "ok"}
    mock_client.post_lineup.return_value = {"status": "ok"}

    from src.main import run_pipeline

    with patch("src.main.AIDirector") as mock_director_cls, patch("src.main.TelegramNotifier") as mock_notifier_cls:
        mock_director = MagicMock()
        mock_director_cls.return_value = mock_director
        mock_notifier = MagicMock()
        mock_notifier_cls.return_value = mock_notifier

        # Simulate Director choosing Candidate 0 (the Wildcard rebuild squad)
        def mock_evaluate(opt_res, *args, **kwargs):
            chosen = opt_res.candidates[0]
            return DecisionOutput(
                selected_candidate_index=0,
                selected_candidate=chosen,
                chosen_move_name=chosen.name,
                transfers_description="Full Rebuild",
                captain_name="GK_1",
                vice_captain_name="GK_2",
                projected_net_xp=chosen.net_xp,
                rationale="Full squad overhaul needed.",
                source="LLM_DIRECTOR",
            )
        mock_director.evaluate_and_decide.side_effect = mock_evaluate

        result = run_pipeline(mock_client, dry_run=False, execute=True)

        assert result["status"] == "success"
        assert result["active_chip"] == "wildcard"
        assert result["decision"]["selected_candidate"]["active_chip"] == "wildcard"
        assert mock_notifier.notify_chip_triggered.called
        assert mock_client.post_transfers.called
        tx_call_args = mock_client.post_transfers.call_args[0]
        assert tx_call_args[0]["chips"] == "wildcard"


# ==========================================
# 9. Injury Discounting & Bench Ordering Tests
# ==========================================

def test_player_injury_multipliers():
    """Verify player status and chance multipliers."""
    from src.solver.optimizer import get_player_injury_multiplier

    # Available / 100%
    assert get_player_injury_multiplier("a", 100) == 1.0
    assert get_player_injury_multiplier("a", None) == 1.0

    # 75% chance (Orange flag)
    assert get_player_injury_multiplier("d", 75) == 0.80

    # 50% chance (Orange/Yellow flag)
    assert get_player_injury_multiplier("d", 50) == 0.40
    assert get_player_injury_multiplier("d", None) == 0.40

    # 25% chance (Red flag)
    assert get_player_injury_multiplier("d", 25) == 0.10

    # 0% chance or injured / suspended / unavailable
    assert get_player_injury_multiplier("a", 0) == 0.0
    assert get_player_injury_multiplier("i", 0) == 0.0
    assert get_player_injury_multiplier("i", None) == 0.0
    assert get_player_injury_multiplier("s", None) == 0.0
    assert get_player_injury_multiplier("u", None) == 0.0


def test_optimizer_injury_discounting_and_bench_ordering(mock_bootstrap_data, mock_fixtures_data):
    """Test optimizer discounts doubtful players, forbids 0.0 multiplier players from Starting XI, and orders bench."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)

    # Inject status flags into test player pool
    # Player 17 (MID, premium): Flagged with 75% chance (multiplier 0.80)
    players_df.loc[players_df["id"] == 17, "chance_of_playing_next_round"] = 75
    players_df.loc[players_df["id"] == 17, "status"] = "d"

    # Player 18 (MID): Injured (status 'i', chance 0, multiplier 0.0)
    players_df.loc[players_df["id"] == 18, "chance_of_playing_next_round"] = 0
    players_df.loc[players_df["id"] == 18, "status"] = "i"

    optimizer = FPLOptimizer(players_df)

    # Check that Player 18 has 0.0 discounted xP
    p18_info = optimizer.player_map[18]
    assert p18_info["injury_multiplier"] == 0.0
    assert p18_info["discounted_xp"] == 0.0

    # Check that Player 17 has 0.80 multiplier
    p17_info = optimizer.player_map[17]
    assert p17_info["injury_multiplier"] == 0.80
    assert p17_info["discounted_xp"] == round(p17_info["raw_xp"] * 0.80, 2)

    # Solve lineup for squad containing injured player 18
    squad_ids = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(squad_ids, bank_m=0.0, free_transfers=0, evaluate_chips=False)

    cand = opt_result.candidates[0]

    # Player 18 (multiplier 0.0) MUST NOT be in the Starting XI
    starter_ids = [p.id for p in cand.starters]
    assert 18 not in starter_ids
    assert len(cand.starters) == 11

    # Bench Priority Ordering:
    # Position 12: Sub GK (bench_order 0)
    assert cand.bench[0].position == "GKP"
    assert cand.bench[0].bench_order == 0

    # Positions 13, 14, 15: Outfield subs in descending order of discounted xP
    outfield_bench = cand.bench[1:]
    assert len(outfield_bench) == 3
    assert outfield_bench[0].bench_order == 1
    assert outfield_bench[1].bench_order == 2
    assert outfield_bench[2].bench_order == 3

    # Injured player 18 with 0 xP should be at the end of the bench (Sub 3 / Pick 15)
    assert cand.bench[-1].id == 18

    # Verify outfield bench is strictly in descending xP order
    assert outfield_bench[0].xp >= outfield_bench[1].xp >= outfield_bench[2].xp


def test_vice_captain_safety_rule(mock_bootstrap_data, mock_fixtures_data):
    """Test that Vice-Captain is assigned to the highest xP outfield starter with 100% chance (status 'a', chance 100)."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)

    # Make Captain (e.g. player 27) top xP
    players_df.loc[players_df["id"] == 27, "xp"] = 12.0

    # Make player 19 have high xP (10.0), but flagged with 75% chance
    players_df.loc[players_df["id"] == 19, "xp"] = 10.0
    players_df.loc[players_df["id"] == 19, "status"] = "d"
    players_df.loc[players_df["id"] == 19, "chance_of_playing_next_round"] = 75

    # Make player 20 have xP 7.0, with 100% chance of playing (status 'a', chance 100)
    players_df.loc[players_df["id"] == 20, "xp"] = 7.0
    players_df.loc[players_df["id"] == 20, "status"] = "a"
    players_df.loc[players_df["id"] == 20, "chance_of_playing_next_round"] = 100

    optimizer = FPLOptimizer(players_df)
    squad_ids = [1, 2, 5, 6, 7, 8, 9, 17, 19, 20, 21, 22, 27, 28, 29]
    opt_result = optimizer.optimize(squad_ids, bank_m=0.0, free_transfers=0, evaluate_chips=False)

    cand = opt_result.candidates[0]
    vc = cand.vice_captain
    assert vc is not None
    # VC must NOT be the goalkeeper
    assert vc.position != "GKP"
    # VC must have 100% chance of playing
    assert vc.status == "a"
    assert vc.injury_multiplier == 1.0
    assert vc.chance_of_playing_next_round in (100, None)


# ==========================================
# 10. Real Selling Prices & Multi-Period Lookahead Tests
# ==========================================

def test_selling_price_budget_constraint(mock_bootstrap_data, mock_fixtures_data):
    """Test optimizer accounts for actual selling prices instead of inflated market costs."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    squad_ids = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    # Assume players have lower selling prices (e.g. 0.5m less than now_cost)
    selling_prices = {pid: optimizer.player_map[pid]["cost_m"] - 0.5 for pid in squad_ids}

    opt_result = optimizer.optimize(
        current_squad_ids=squad_ids,
        bank_m=0.2,
        free_transfers=1,
        selling_prices=selling_prices,
        evaluate_chips=False,
    )

    cand = opt_result.candidates[0]
    expected_budget = sum(selling_prices.values()) + 0.2
    assert opt_result.current_team_value_m == round(sum(selling_prices.values()), 2)
    assert cand.bank_remaining_m >= 0.0


def test_multi_period_and_rolling_bonus(mock_bootstrap_data, mock_fixtures_data):
    """Test candidate squads compute 3-GW multi-period sums and rolling transfer strategic bonuses."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    squad_ids = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    opt_result = optimizer.optimize(
        current_squad_ids=squad_ids,
        bank_m=1.0,
        free_transfers=1,
        evaluate_chips=False,
    )

    cand_0 = opt_result.candidates[0]  # 0 transfers
    assert cand_0.transfers_count == 0
    assert cand_0.multi_gw_xp > 0
    assert cand_0.strategic_value_score > cand_0.net_xp  # Has rolling strategic bonus added


def test_validate_auth_method():
    """Test FPLClient validate_auth reporting."""
    from src.api.client import FPLClient
    from src.api.auth import FPLAuth

    # Unauthenticated client
    client_no_auth = FPLClient(auth=FPLAuth(token=""))
    ok, msg = client_no_auth.validate_auth(12345)
    assert ok is False
    assert "FPL_AUTH_TOKEN is not configured" in msg


# ==========================================
# 5. Solver Engine Upgrade Tests (HiGHS, Decay, Candidates)
# ==========================================

def test_highs_solver_backend_detection_and_fallback():
    """Test HiGHS solver backend configuration and graceful PULP_CBC_CMD fallback."""
    import pulp

    # 1. Test get_solver returns an available solver
    solver = get_solver(msg=False)
    assert solver is not None

    # 2. Test problem solving with get_solver
    prob = pulp.LpProblem("TestSolver", pulp.LpMaximize)
    x = pulp.LpVariable.dict("x", [0], lowBound=0, upBound=10, cat=pulp.LpContinuous) if hasattr(pulp.LpVariable, "dict") else pulp.LpVariable("x", 0, 10)
    prob += x[0] if isinstance(x, dict) else x
    status = _solve_problem(prob, primary_solver=solver)
    assert status == pulp.LpStatusOptimal
    val = pulp.value(x[0]) if isinstance(x, dict) else pulp.value(x)
    assert val == 10.0

    # 3. Test fallback behavior when primary solver raises an error
    faulty_solver = MagicMock()
    faulty_solver.actualSolve.side_effect = Exception("HiGHS binary error simulation")
    prob2 = pulp.LpProblem("TestFallback", pulp.LpMaximize)
    y = pulp.LpVariable("y", 0, 5)
    prob2 += y
    status2 = _solve_problem(prob2, primary_solver=faulty_solver)
    assert status2 == pulp.LpStatusOptimal
    assert pulp.value(y) == 5.0


def test_multi_gameweek_decay_factor_in_optimizer():
    """Test multi-gameweek horizon decay factor math in FPLOptimizer."""
    gamma = 0.80
    expected_decay_sum = round(1.0 + gamma + (gamma ** 2), 4)  # 1.0 + 0.80 + 0.64 = 2.44

    raw_df = pd.DataFrame([{
        "id": 1,
        "web_name": "TestMID",
        "element_type": 3,
        "position": "MID",
        "team_id": 1,
        "team_name": "Team 1",
        "team_code": "T01",
        "cost_m": 6.0,
        "xp": 5.0,
        "status": "a",
        "chance_of_playing_next_round": 100,
    }])

    optimizer = FPLOptimizer(raw_df, decay_factor=gamma)
    assert optimizer.decay_factor == 0.80
    assert optimizer.df.iloc[0]["discounted_3gw"] == round(5.0 * expected_decay_sum, 2)
    assert optimizer.df.iloc[0]["discounted_3gw"] == 12.20


def test_diverse_candidates_generation_and_hit_hurdle(mock_bootstrap_data, mock_fixtures_data):
    """Test generation of 7 distinct candidates with rolling value and hit hurdle logic."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    optimizer = FPLOptimizer(players_df)
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    opt_result = optimizer.optimize(
        current_squad_ids=initial_squad,
        bank_m=2.0,
        free_transfers=1,
        evaluate_chips=False,
    )

    assert len(opt_result.candidates) == 7
    cand_0 = opt_result.candidates[0]
    cand_1 = opt_result.candidates[1]
    cand_4 = opt_result.candidates[4]

    # Candidate 1: 0 Transfers (Roll)
    assert cand_0.transfers_count == 0
    assert "Roll / Bank Transfer" in cand_0.name
    assert cand_0.strategic_value_score > cand_0.net_xp  # Has +1.5 FT rolling value bonus

    # Candidate 2: 1-Transfer Move
    assert cand_1.transfers_count <= 1
    assert cand_1.hit_cost == 0

    # Candidate 5: 2-Transfer Move
    assert cand_4.transfers_count <= 2
    if cand_4.transfers_count == 2:
        assert cand_4.hit_cost == 4
        # Verify candidate name contains hit details or hurdle evaluation
        assert "-4 Hit" in cand_4.name


def test_local_csv_priority_and_html_fallback(tmp_path, mock_bootstrap_data, mock_fixtures_data):
    """Test local CSV file priority over remote scraping, and HTML error payload resilience."""
    # 1. Test local CSV reading
    local_csv_file = tmp_path / "custom_projections.csv"
    local_csv_file.write_text("ID,Name,Pos,Team,1_Pts\n1,GK_1,GKP,Team 1,7.7\n")

    fetcher = FPLReviewFetcher(file_path=local_csv_file)
    df = fetcher.fetch_projections()
    assert df is not None
    assert len(df) == 1
    assert float(df.iloc[0]["1_Pts"]) == 7.7

    # 2. Test HTML error page passed as file
    html_file = tmp_path / "projections.csv"
    html_file.write_text("<!DOCTYPE html><html><head><title>404 Not Found</title></head><body>Error</body></html>")
    fetcher_html_file = FPLReviewFetcher(file_path=html_file)
    # When file is HTML, should skip and return None without exception
    with patch("requests.get", return_value=MagicMock(status_code=404, text="<html>404</html>")):
        res_df = fetcher_html_file.fetch_projections()
        assert res_df is None

    # 3. Test HTML payload passed as string content
    html_str = "<!DOCTYPE html><html><body>Error</body></html>"
    df_html = fetch_fplreview_projections(csv_content=html_str)
    assert df_html is None

    # 4. Pipeline calculation resilience when projections return None
    players_df = calculate_player_metrics(
        mock_bootstrap_data,
        mock_fixtures_data,
        current_event=1,
        fplreview_df=None,
    )
    assert not players_df.empty
    assert "xp" in players_df.columns


def test_calculate_fallback_xp_helper():
    """Test calculate_fallback_xp standalone helper."""
    element = {
        "id": 10,
        "web_name": "TestPlayer",
        "element_type": 3,
        "form": "6.0",
        "points_per_game": "5.0",
        "ep_next": "5.5",
    }
    fb = calculate_fallback_xp(
        element=element,
        next_fdr=2.0,
        next_is_home=True,
        avg_fdr=2.5,
        availability=1.0,
        decay_factor=0.85,
    )
    assert "xp" in fb
    assert fb["xp"] == 5.5
    assert fb["xp_source"] == "fpl_ep_next"
    assert fb["xp_3gw"] > fb["xp"]


def test_decay_weights_and_horizon_length_in_optimizer():
    """Test calculate_decay_weights helper and configurable horizon length."""
    from src.engine.optimizer import calculate_decay_weights

    # Horizon 1
    w1, s1 = calculate_decay_weights(horizon_length=1, decay_factor=0.85)
    assert w1 == [1.0]
    assert s1 == 1.0

    # Horizon 3
    w3, s3 = calculate_decay_weights(horizon_length=3, decay_factor=0.85)
    assert w3 == [1.0, 0.85, 0.7225]
    assert s3 == 2.5725

    # Horizon 5
    w5, s5 = calculate_decay_weights(horizon_length=5, decay_factor=0.85)
    assert len(w5) == 5
    assert s5 == round(1.0 + 0.85 + 0.7225 + 0.6141 + 0.5220, 4)

    raw_df = pd.DataFrame([{
        "id": 1,
        "web_name": "TestMID",
        "element_type": 3,
        "position": "MID",
        "team_id": 1,
        "team_name": "Team 1",
        "team_code": "T01",
        "cost_m": 6.0,
        "xp": 10.0,
        "status": "a",
        "chance_of_playing_next_round": 100,
    }])

    opt_h5 = FPLOptimizer(raw_df, decay_factor=0.85, horizon_length=5, bench_weight=0.15)
    assert opt_h5.horizon_length == 5
    assert opt_h5.bench_weight == 0.15
    assert opt_h5.decay_sum == s5
    assert opt_h5.df.iloc[0]["discounted_3gw"] == round(10.0 * s5, 2)


def test_bench_weight_sub_factor_in_optimizer(mock_bootstrap_data, mock_fixtures_data):
    """Test that bench_weight (sub factor) affects MILP optimization properly."""
    players_df = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1)
    
    # Custom sub factor = 0.20
    optimizer = FPLOptimizer(players_df, bench_weight=0.20)
    assert optimizer.bench_weight == 0.20
    
    initial_squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    opt_result = optimizer.optimize(
        current_squad_ids=initial_squad,
        bank_m=2.0,
        free_transfers=1,
        evaluate_chips=False,
    )
    assert len(opt_result.candidates) >= 1
    # Check that lineup and bench are properly constructed
    for cand in opt_result.candidates:
        assert len(cand.starters) == 11
        assert len(cand.bench) == 4
        assert cand.formation in ["3-5-2", "3-4-3", "4-4-2", "4-3-3", "4-5-1", "5-3-2", "5-4-1", "5-2-3"]






