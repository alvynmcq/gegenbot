"""Multi-week transfer MILP: banking, hit caps, plan following, and candidate wiring."""
from src.engine.horizon import add_horizon_candidate, pair_moves, solve_horizon
from src.engine.metrics import calculate_player_metrics
from src.engine.optimizer import FPLOptimizer
from src.engine.strategy import RollingPlanner
from tests.test_strategy import candidate, make_planner


def _ins(plan):
    return [set(r["in"]) for r in plan["rounds"]]


def test_plan_buys_for_later_weeks_within_free_transfers():
    planner = make_planner()  # 17 and 18 score nothing now, 14 and 12 later
    plan = solve_horizon(planner, bench_weight=0.0)
    ins = _ins(plan)
    assert set().union(*ins) == {17, 18}
    assert all(r["hits"] == 0 for r in plan["rounds"])
    # One free transfer now, at most two next week if the first is banked.
    assert len(ins[0]) <= 1 and len(ins[0]) + len(ins[1]) <= 2
    assert plan["optimal"]


def test_hits_only_when_allowed():
    planner = make_planner(max_hit_points=8)
    planner.initial = planner.initial.__class__(planner.initial.squad, planner.initial.retained,
                                                planner.initial.bank, 0)
    plan = solve_horizon(planner, bench_weight=0.0)
    assert sum(r["hits"] for r in plan["rounds"]) <= 8
    assert all(r["hits"] <= 8 for r in plan["rounds"])
    capped = make_planner()
    capped.initial = planner.initial
    assert all(r["hits"] == 0 for r in solve_horizon(capped, bench_weight=0.0)["rounds"])


def test_vetoed_players_are_never_bought():
    plan = solve_horizon(make_planner(), bench_weight=0.0, veto={17})
    assert 17 not in set().union(*_ins(plan))


def test_planner_follows_a_legal_plan_and_ignores_an_illegal_one():
    planner = make_planner()
    rounds = [{"gameweek": 10, "in": [], "out": []},
              {"gameweek": 11, "in": [17, 18], "out": [8, 15]},
              {"gameweek": 12, "in": [], "out": []}]
    roll = candidate()
    first = planner.plan(roll)[:1]
    followed = planner._follow(first, {"rounds": rounds})
    assert followed[1].after.squad - followed[1].before.squad == {17, 18}
    assert followed[1].hits == 0
    rounds[1]["out"] = [8, 99]
    assert planner._follow(first, {"rounds": rounds}) is None
    roll.horizon_plan = {"rounds": rounds}
    assert planner._rank(planner.plan(roll)) == planner._rank(planner._future(first, planner.weeks[1:]))


def test_pair_moves_matches_positions():
    players = {1: {"element_type": 3}, 2: {"element_type": 4}, 3: {"element_type": 4}, 4: {"element_type": 3}}
    assert pair_moves([1, 2], [3, 4], players) == [(1, 4), (2, 3)]
    assert pair_moves([1], [3], players) is None


def test_plan_is_offered_as_a_candidate(mock_bootstrap_data, mock_fixtures_data, monkeypatch):
    monkeypatch.setenv("HORIZON_LENGTH", "2")
    players = calculate_player_metrics(mock_bootstrap_data, mock_fixtures_data, current_event=1, odds_data={})
    optimizer = FPLOptimizer(players)
    squad = [1, 2, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 27, 28, 29]
    result = optimizer.optimize(squad, bank_m=1.0, free_transfers=1, evaluate_chips=False,
                                include_chip_candidates=False)
    before = len(result.candidates)
    planner = RollingPlanner(optimizer, 1, (1, 2), squad, 1.0, 1)
    carrier = add_horizon_candidate(optimizer, planner, result, current_squad_ids=squad, bank_m=1.0,
                                    free_transfers=1)
    assert carrier is not None and carrier.horizon_plan["rounds"][0]["gameweek"] == 1
    week_one = set(carrier.horizon_plan["rounds"][0]["squad"])
    assert {p.id for p in carrier.starters + carrier.bench} == week_one
    # Either a new option, or attached to the identical existing one.
    assert len(result.candidates) in (before, before + 1)
    assert sum(c.horizon_plan is not None for c in result.candidates) == 1
    planner.annotate(result, inventory=[], offer_chips=False)
    assert carrier.rolling_plan is not None
