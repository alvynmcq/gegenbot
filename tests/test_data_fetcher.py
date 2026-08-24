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
