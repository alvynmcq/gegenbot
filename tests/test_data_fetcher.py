"""Tests for FPL Core Insights data fetcher and projection mapping."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.data_fetcher import FPLCoreInsightsFetcher, FPLReviewFetcher
from src.engine.metrics import calculate_player_metrics


def test_fpl_core_insights_direct_id_mapping(mock_bootstrap_data):
    csv_content = """id,web_name,team,ep_next,form,points_per_game,expected_goals_per_90,expected_assists_per_90,defensive_contribution_per_90
1,GK_1,1,4.5,4.0,4.2,0.0,0.0,0.0
2,GK_2,2,3.8,3.5,3.9,0.0,0.0,0.0
7,Calafiori,1,5.2,5.0,5.1,0.20,0.15,4.94
"""
    fetcher = FPLCoreInsightsFetcher()
    df = fetcher.fetch_projections(csv_content=csv_content)
    assert df is not None
    assert len(df) == 3

    mapped = fetcher.map_to_bootstrap(df, mock_bootstrap_data, current_event=1)
    assert 1 in mapped
    assert mapped[1]["fplreview_xp"] == 4.5
    assert mapped[1]["source"] == "fpl_core_insights"
    assert mapped[7]["fplreview_xp"] == 5.2
    assert mapped[7]["xgi_90"] == 0.35
    assert mapped[7]["def_contrib"] == 4.94


def test_fpl_core_insights_enrichment_in_calculate_metrics(mock_bootstrap_data, mock_fixtures_data):
    csv_content = """id,web_name,team,ep_next,form,points_per_game,expected_goals_per_90,expected_assists_per_90
1,GK_1,1,6.0,5.0,5.5,0.0,0.0
"""
    fetcher = FPLCoreInsightsFetcher()
    df = fetcher.fetch_projections(csv_content=csv_content)

    players_df = calculate_player_metrics(
        mock_bootstrap_data,
        mock_fixtures_data,
        current_event=1,
        fplreview_df=df,
    )

    p1 = players_df[players_df["id"] == 1].iloc[0]
    assert p1["xp"] == 6.0
    assert p1["xp_source"] == "fpl_core_insights"


def test_fpl_core_insights_remote_fallback_urls():
    fetcher = FPLCoreInsightsFetcher(file_path=Path("nonexistent.csv"))

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "id,web_name,ep_next\n1,Saka,7.5\n"
        mock_get.return_value = mock_resp

        df = fetcher.fetch_projections()
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["web_name"] == "Saka"


def test_fpl_core_insights_html_rejection():
    fetcher = FPLCoreInsightsFetcher(file_path=Path("nonexistent.csv"))
    html_content = "<!DOCTYPE html><html><body>Error 404</body></html>"

    df = fetcher.fetch_projections(csv_content=html_content)
    assert df is None


def test_vaastav_fetcher_rolling_metrics():
    from src.data_fetcher import VaastavDataFetcher

    players_raw_df = pd.DataFrame([
        {
            "id": 1,
            "web_name": "Saka",
            "expected_goals_per_90": 0.35,
            "expected_assists_per_90": 0.25,
            "expected_goal_involvements_per_90": 0.60,
            "expected_goals_conceded_per_90": 0.80,
            "minutes": 180,
        },
        {
            "id": 2,
            "web_name": "Haaland",
            "expected_goals_per_90": 0.85,
            "expected_assists_per_90": 0.15,
            "expected_goal_involvements_per_90": 1.00,
            "expected_goals_conceded_per_90": 0.70,
            "minutes": 180,
        },
    ])

    merged_gw_df = pd.DataFrame([
        # GW1
        {"element": 1, "round": 1, "minutes": 90, "expected_goal_involvements": 0.80, "expected_goals_conceded": 0.50, "starts": 1},
        {"element": 2, "round": 1, "minutes": 90, "expected_goal_involvements": 1.20, "expected_goals_conceded": 0.40, "starts": 1},
        # GW2
        {"element": 1, "round": 2, "minutes": 90, "expected_goal_involvements": 0.40, "expected_goals_conceded": 0.60, "starts": 1},
        {"element": 2, "round": 2, "minutes": 90, "expected_goal_involvements": 0.80, "expected_goals_conceded": 0.50, "starts": 1},
    ])

    fetcher = VaastavDataFetcher()
    stats = fetcher.calculate_player_underlying_metrics(merged_gw_df=merged_gw_df, players_raw_df=players_raw_df)

    assert 1 in stats
    assert 2 in stats
    assert stats[1]["rolling_xgi_90"] == 0.60  # (1.20 / (180/90)) = 0.60
    assert stats[2]["rolling_xgi_90"] == 1.00  # (2.00 / (180/90)) = 1.00
    assert stats[1]["starts_ratio"] == 1.0
    assert stats[1]["rolling_minutes_avg"] == 90.0


def test_vaastav_enrichment_in_calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data):
    vaastav_stats = {
        1: {
            "rolling_xgi_90": 0.75,
            "rolling_xgc_90": 0.40,
            "rolling_minutes_avg": 90.0,
            "starts_ratio": 1.0,
            "recent_matches_count": 3,
        },
        2: {
            "rolling_xgi_90": 0.05,
            "rolling_xgc_90": 1.80,
            "rolling_minutes_avg": 30.0,
            "starts_ratio": 0.33,
            "recent_matches_count": 3,
        }
    }

    players_df = calculate_player_metrics(
        mock_bootstrap_data,
        mock_fixtures_data,
        current_event=1,
        vaastav_stats=vaastav_stats,
    )

    p1 = players_df[players_df["id"] == 1].iloc[0]
    p2 = players_df[players_df["id"] == 2].iloc[0]

    assert p1["rolling_xgi_90"] == 0.75
    assert p1["minutes_reliability"] == "HIGH"

    assert p2["rolling_xgi_90"] == 0.05
    assert p2["minutes_reliability"] == "LOW"
