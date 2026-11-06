"""Rolling plans use legal moves and compare the same chip across its window."""
from types import SimpleNamespace

from src.engine.strategy import RollingPlanner


def make_planner(risk_mode="NEUTRAL", **kwargs):
    positions = [1] * 2 + [2] * 5 + [3] * 5 + [4] * 3 + [3, 3, 4]
    players = {
        pid: {
            "cost_m": 5.0, "element_type": position, "team_id": pid,
            "discounted_xp": 2.0, "xp_by_gameweek": {11: 2.0, 12: 2.0},
            "availability": 1.0, "injury_multiplier": 1.0,
        }
        for pid, position in enumerate(positions, start=1)
    }
    players[16].update(discounted_xp=3.0, xp_by_gameweek={11: 0.0, 12: 0.0})
    players[17].update(discounted_xp=0.0, xp_by_gameweek={11: 14.0, 12: 14.0})
    players[18].update(discounted_xp=0.0, xp_by_gameweek={11: 12.0, 12: 12.0})
    optimizer = SimpleNamespace(player_map=players, decay_weights=[1.0, 0.85, 0.7225],
                                allow_defensive_captain=False)
    return RollingPlanner(optimizer, 10, (10, 11, 12), range(1, 16), 0, 1,
                          risk_mode=risk_mode, **kwargs)


def candidate(first_mid=8, chip=None, hit=0):
    starters = [1, 3, 4, 5, first_mid, 9, 10, 11, 13, 14, 15]
    bench = [2, 6, 7, 12]
    return SimpleNamespace(starters=[SimpleNamespace(id=i) for i in starters],
                           bench=[SimpleNamespace(id=i) for i in bench],
                           captain=SimpleNamespace(id=first_mid),
                           transfers_count=int(first_mid != 8), bank_remaining_m=0,
                           active_chip=chip, hit_cost=hit,
                           net_xp=24 + (2 if first_mid == 16 else 0) - hit)


def test_rolling_now_can_unlock_two_free_moves_next_week():
    planner = make_planner()
    rolled = planner.plan(candidate())
    spent = planner.plan(candidate(16))
    assert rolled[0].after.free == 2
    assert spent[0].after.free == 1
    assert rolled[1].after.squad - rolled[1].before.squad == {17, 18}
    assert rolled[1].hits == 0
    assert planner._rank(rolled) > planner._rank(spent)
    assert all(step.hits == 0 and step.after.bank >= 0 for step in rolled + spent)


def test_selling_price_and_club_limit_are_enforced():
    planner = make_planner(selling_prices={8: 4.9})
    assert planner._transfer(planner.initial, 8, 17) is None
    assert planner._transfer(planner.initial, 9, 17).bank == 0
    for pid in (1, 3, 4, 17):
        planner.players[pid]["team_id"] = 42
    assert planner._transfer(planner.initial, 9, 17) is None


def test_free_hit_restores_squad_and_preserves_saved_transfers():
    planner = make_planner()
    freehit = planner.plan(candidate(16, chip="freehit"))
    wildcard = planner.plan(candidate(16, chip="wildcard"))
    assert freehit[0].after.squad == planner.initial.squad
    assert freehit[0].after.free == wildcard[0].after.free == 1
    assert 16 in wildcard[0].after.squad
    assert 16 not in freehit[1].before.squad


def test_same_chip_opportunity_respects_expiry():
    planner = make_planner()
    baseline = planner.plan(candidate())
    chip = {"name": "3xc", "status_for_entry": "available", "start_event": 1, "stop_event": 12}
    gain, week = planner.wait_value(baseline, "3xc", [chip])
    assert gain > 8
    assert week == 11
    assert planner.wait_value(baseline, "3xc", [chip], [12])[1] == 12
    # A new instance in the next window is not the same chip.
    next_instance = dict(chip, start_event=11, stop_event=20)
    assert planner.wait_value(baseline, "3xc", [dict(chip, stop_event=10), next_instance]) == (0.0, None)
    assert planner.wait_value(baseline, "3xc", None) == (None, None)


def test_defend_prices_shared_club_and_appearance_risk_more_than_chase():
    defend = make_planner("DEFEND")
    chase = make_planner("CHASE")
    starters = (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)
    diversified = defend._risk(10, starters, 8)
    defend.players[9]["team_id"] = defend.players[8]["team_id"]
    defend.players[8]["injury_multiplier"] = 0.5
    assert defend._risk(10, starters, 8) > diversified
    path = chase.plan(candidate())
    assert defend._rank(path) < chase._rank(path)


def test_automatic_chip_is_held_when_waiting_has_more_value():
    planner = make_planner()
    ordinary = candidate()
    ordinary.transfers = []
    triple = candidate(chip="3xc")
    triple.transfers = []
    triple.net_xp += 2
    evaluation = SimpleNamespace(squad_candidate=triple, threshold_met=True, target_gameweeks=[])
    result = SimpleNamespace(candidates=[ordinary], chip_evaluation=SimpleNamespace(
        evaluations={"3xc": evaluation}, recommended_chip="3xc", recommendation_reason=None,
    ))
    inventory = [{"name": "3xc", "status_for_entry": "available", "start_event": 1, "stop_event": 12}]
    planner.annotate(result, inventory, offer_chips=True)
    assert result.candidates == [ordinary]
    assert result.chip_evaluation.recommended_chip is None
    assert evaluation.opportunity["best_later_gameweek"] == 11
    assert evaluation.opportunity["play_now_edge"] < 0
