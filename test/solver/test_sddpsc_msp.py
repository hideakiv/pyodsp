"""MultistageProgram(method="sddpsc"): the pipeline builds both of an inner
node's models, and the leaves' transition chains, from one stage builder.

The same unit store as test_sddpsc_multistage.py, hand-wired there: buy one
unit (binary) at stage 0 and at stage 1, the level carrying over
deterministically, then sell up to the level held at a random price.
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.dec.sddpsc.node import SddpScNodeInner
from pyodsp.model.msp import MultistageProgram

SOLVER = "appsi_highs"
SELL = [
    {"name": "low", "probability": 0.5, "price": 1.5},
    {"name": "high", "probability": 0.5, "price": 2.5},
]


def store(tmp_path, buy, sell=SELL, transitions=None, **kw):
    """buy: the stage-1 realizations, each with its own buy price."""
    kw.setdefault("log_level", logging.CRITICAL)
    kw.setdefault("stage_bound", -5.0)
    msp = MultistageProgram(
        "store",
        sense="min",
        solver=SOLVER,
        output_dir=tmp_path,
        max_iteration=100,
        sample_frequency=10,
        sample_size=10,
        method="sddpsc",
        **kw,
    )

    @msp.stage(state=["x"])
    def stage(m, state, node):
        if node.is_last:
            m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 2))
            m.sell = pyo.Constraint(expr=m.y <= state.x)
            return -node["price"] * m.y
        m.u = pyo.Var(within=pyo.Binary)
        m.x = pyo.Var(bounds=(0, 2))
        m.step = pyo.Constraint(expr=m.x == state.x + m.u)
        return (1.0 if node.is_first else node["buy"]) * m.u

    msp.set_initial_state(x=0.0)
    msp.set_stage_realizations([buy, sell], transitions=transitions)
    return msp


ONE = [{"name": "a", "probability": 1.0, "buy": 0.25}]
TWO = [
    {"name": "a", "probability": 0.5, "buy": 0.0},
    {"name": "b", "probability": 0.5, "buy": 0.25},
]


def test_three_stages(tmp_path):
    msp = store(tmp_path, ONE)

    result = msp.solve()

    # buy both units: 1 + 0.25 - 2 * 2
    assert result.bound == pytest.approx(-2.75, abs=1e-4)
    assert result.first_stage["x"] == pytest.approx(1.0)
    assert isinstance(msp.build().nodes[1][0], SddpScNodeInner)


def test_two_independent_nodes_per_stage(tmp_path):
    # 1 + mean(0, 0.25) - 4
    assert store(tmp_path, TWO).solve().bound == pytest.approx(-2.875, abs=1e-4)


def test_markov_transitions(tmp_path):
    # after the cheap node high prices are likelier: units worth 2.3 there,
    # 1.7 after the other -- 1 + 0.5 * (0 - 4.6) + 0.5 * (0.25 - 3.4)
    msp = store(
        tmp_path,
        TWO,
        transitions=[[[0.5, 0.5]], [[0.2, 0.8], [0.8, 0.2]]],
    )
    assert msp.solve().bound == pytest.approx(-2.875, abs=1e-4)


def test_a_shorter_lookback_still_bounds_the_optimum(tmp_path):
    # lookback=1: the last stage's chain starts at stage 1, its incoming
    # level free, so the reachable set it sees is larger -- the bound may
    # be weaker but must stay valid
    bound = store(tmp_path, ONE, lookback=1).solve().bound
    assert bound <= -2.75 + 1e-4


def test_each_leaf_chain_replays_the_earlier_stages(tmp_path):
    built = store(tmp_path, ONE).build()
    last = built.models["2-0"]
    assert [b.local_name for b in last.component_objects(pyo.Block)] == [
        "stage_0",
        "stage_1",
        "stage_2",
    ]
    # the chain's earlier costs are dropped: only stage 2's is priced
    assert "u" not in str(last.objective.expr)

    shorter = store(tmp_path, ONE, lookback=1).build().models["2-0"]
    assert [b.local_name for b in shorter.component_objects(pyo.Block)] == [
        "stage_1",
        "stage_2",
    ]


def test_random_transitions_are_refused(tmp_path):
    msp = MultistageProgram(
        "bad", stage_bound=0.0, output_dir=tmp_path, method="sddpsc"
    )

    @msp.stage(state=["x"])
    def stage(m, state, node):
        m.x = pyo.Var(bounds=(0, node.get("cap", 2.0)))
        return m.x

    msp.set_initial_state(x=0.0)
    msp.set_realizations(
        [{"name": "a", "probability": 0.5, "cap": 1.0},
         {"name": "b", "probability": 0.5, "cap": 3.0}],
        stages=3,
    )
    with pytest.raises(ValueError, match="share one transition"):
        msp.build()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "nope"}, "method must be"),
        ({"method": "sddpsc", "mpi": True}, "MPI"),
        ({"lookback": 2}, "only applies"),
    ],
)
def test_bad_options_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MultistageProgram("x", **kwargs)


def test_sddpsc_needs_a_stage_bound(tmp_path):
    msp = store(tmp_path, ONE, stage_bound=None)
    with pytest.raises(ValueError, match="stage_bound"):
        msp.build()


def test_lookback_must_reach_the_previous_stage(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        store(tmp_path, ONE, lookback=0).build()


def test_the_policy_replays_a_scenario_path(tmp_path):
    result = store(tmp_path, TWO).solve()

    # stage 0 buys (1), the dear stage-1 node still buys (0.25), and the
    # high price sells both units
    path = result.simulate(["0-0", "1-1", "2-1"])

    assert [round(s.stage_cost, 6) for s in path] == [1.0, 0.25, -5.0]
    assert [s.next_state for s in path] == [[1.0], [2.0], []]
    # stage 0's own solve prices the whole future at what the run learned
    assert path[0].total_cost == pytest.approx(result.bound, abs=1e-4)
    # the last stage reports its own decision, not the chain behind it
    assert path[2].solution == {"y": pytest.approx(2.0)}
    assert "sddpsc" in result.summary()


def test_the_policy_answers_any_incoming_state(tmp_path):
    policy = store(tmp_path, ONE).solve().policy()

    # nothing held yet at stage 1: buy the one unit it can
    assert policy.evaluate("1-0", [0.0]).next_state == [1.0]
    # one unit held at the last stage: sell exactly that
    assert policy.evaluate("2-0", [1.0]).stage_cost == pytest.approx(-1.5)
    with pytest.raises(ValueError, match="couples on 1"):
        policy.evaluate("2-0", [1.0, 2.0])


def test_the_policy_can_be_rebuilt_from_a_saved_run(tmp_path):
    from pyodsp.dec.sddpsc.policy import SddpScPolicy

    result = store(tmp_path, TWO).solve()
    path = ["0-0", "1-1", "2-1"]
    live = [s.stage_cost for s in result.simulate(path)]

    policy = SddpScPolicy.from_saved(store(tmp_path, TWO).build().nodes, tmp_path)

    assert [s.stage_cost for s in policy.simulate(path)] == pytest.approx(live)
    assert policy.evaluate("0-0").total_cost == pytest.approx(result.bound, abs=1e-4)
