"""Head-to-head risk: rival ownership changes spread, never expectation."""
import itertools
from types import SimpleNamespace

import pytest

from src.engine import rivalry
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import FPLOptimizer
from src.engine.strategy import RollingPlanner
from tests.test_strategy import make_planner


def test_target_is_leader_when_behind_and_chaser_when_leading():
    standings = [{"entry": 1, "rank": 1, "total": 700}, {"entry": 2, "rank": 2, "total": 680},
                 {"entry": 3, "rank": 3, "total": 650}]
    assert rivalry.choose_target(standings, 3) == (1, 50.0)
    assert rivalry.choose_target(standings, 1) == (2, -20.0)
    assert rivalry.choose_target(standings, 99) is None


def test_lineup_multipliers_and_price():
    picks = [{"element": 5, "position": 1, "is_captain": True}, {"element": 6, "position": 7},
             {"element": 7, "position": 13}]
    assert rivalry.lineup_multipliers(picks) == {5: 2.0, 6: 1.0, 7: 0.0}
    assert rivalry.risk_price(30, 27, 10) == pytest.approx(30 / 270)  # early: small
    assert rivalry.risk_price(30, 3, 10) == 1.0                        # late: capped
    assert rivalry.risk_price(-30, 3, 10) == -1.0                      # leading: avoid spread
    assert rivalry.relative_sd({1: 1}, {1: 1}, {1: 5.0}) == 0.0        # identical picks cancel


def test_milp_variance_term_is_exact_for_all_binary_choices():
    # (m - e)^2 - e^2 must equal starter*(1 - 2e) + captain*(3 - 2e) with m = starter + captain.
    for starter, captain, e in itertools.product((0, 1), (0, 1), (0, 0.5, 1, 2)):
        if captain > starter:
            continue
        m = starter + captain
        assert (m - e) ** 2 - e ** 2 == pytest.approx(starter * (1 - 2 * e) + captain * (3 - 2 * e))


def _captain_planner(gap):
    players = make_planner().players
    players[13]["discounted_xp"] = 4.0
    players[14]["discounted_xp"] = 3.9
    rival = {pid: 1.0 for pid in (1, 3, 4, 5, 8, 9, 10, 11, 15)}
    rival[13] = 2.0  # the rival captains the top forward
    optimizer = SimpleNamespace(player_map=players, decay_weights=[1.0, 0.85, 0.7225],
                                allow_defensive_captain=False)
    return RollingPlanner(optimizer, 10, (10, 11, 12), range(1, 16), 0, 1,
                          rival=rival, rival_gap=gap, weeks_remaining=3)


def test_planner_prices_spread_by_situation_and_uses_the_armband():
    behind, ahead = _captain_planner(60), _captain_planner(-60)
    assert behind.risk_price > 0 > ahead.risk_price
    assert behind.variance_weight > 0 > ahead.variance_weight
    # Behind: avoid matching the rival's captain. Ahead: shadow it.
    assert behind._lineup(behind.initial.squad, 10)[3] == 14
    assert ahead._lineup(ahead.initial.squad, 10)[3] == 13


def test_no_rival_keeps_mode_weights():
    planner = make_planner("DEFEND")
    assert planner.rival is None and planner.risk_weight == 0.45 and planner.variance_weight == 0.0


def test_solver_differentiates_from_or_shadows_the_rival_captain(mock_bootstrap_data, mock_fixtures_data):
    players = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1, odds_data={})
    squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]

    def captain(weight, rival=None):
        optimizer = FPLOptimizer(players)
        optimizer.variance_weight = weight
        result = optimizer.optimize(squad, bank_m=1.0, free_transfers=1, eo_weights=rival,
                                    evaluate_chips=False, include_chip_candidates=False)
        return result.candidates[0].captain.id

    plain = captain(0.0)
    rival = {plain: 200.0}
    assert captain(0.0, rival) == plain          # without a price, ownership is irrelevant
    assert captain(-0.5, rival) == plain         # ahead: shadow the rival's captain
    assert captain(0.5, rival) != plain          # far behind: differentiate
