"""Shared pytest fixtures for Gegenbot test suite."""

import pytest


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
