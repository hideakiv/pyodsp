"""sddpsc on a genuinely multistage (3-stage) lattice.

A unit store: each of stages 1 and 2 may buy one unit (binary u_t), the
level x_t = x_{t-1} + u_t carries over deterministically from x_0 = 0, and
stage 3 sells up to the level held at a random price. Only the costs are
random, the restriction sddpsc targets.

    stage 1: buy at 1
    stage 2: buy at 0.25
    stage 3: sell y <= x_2 at 1.5 or 2.5, equally likely (mean 2)

Each unit is worth 2 at the end, so buying both is optimal:
1 + 0.25 - 2 * 2 = -2.75. A lattice whose stage-2 cut ignores stage 3
sees only buying costs and settles on buying nothing, at 0.

Each stage-t leaf embeds the transition chain from stage 1 as free
variables, coupling only on x_{t-1} (see pyodsp.dec.sddpsc).
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.sddpsc.alg_root_bm import SddpScAlgRootBm
from pyodsp.dec.sddpsc.alg_leaf_pyomo import SddpScAlgLeafPyomo
from pyodsp.dec.sddpsc.node import SddpScNodeInner
from pyodsp.dec.sddpsc.run import SddpScRun

SOLVER = "appsi_highs"
PRICE_1 = 1.0
PRICE_2 = 0.25
SELL_PRICES = [1.5, 2.5]
OPTIMUM = PRICE_1 + PRICE_2 - 2 * sum(SELL_PRICES) / len(SELL_PRICES)
# the worst any stage-2-onward cost can be: selling two units at the top price
BOUND = -2 * max(SELL_PRICES)


def chain(model, stages):
    """Stages 1..stages of the transition, as plain variables."""
    model.u = pyo.Var(range(1, stages + 1), within=pyo.Binary)
    model.x = pyo.Var(range(1, stages + 1), bounds=(0, 2))

    def step(m, t):
        prev = 0.0 if t == 1 else m.x[t - 1]
        return m.x[t] == prev + m.u[t]

    model.step = pyo.Constraint(range(1, stages + 1), rule=step)


def leaf_alg(model, coupling_up, **kwargs):
    return SddpScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig(SOLVER), coupling_up),
        SolverConfig("ipopt"),
        **kwargs,
    )


def make_inner(idx, price):
    """A stage-2 node buying at `price`."""
    # root role: x_1 arrives as a parameter
    own = pyo.ConcreteModel()
    own.x_in = pyo.Param(mutable=True, initialize=0.0)
    own.u = pyo.Var(within=pyo.Binary)
    own.x = pyo.Var(bounds=(0, 2))
    own.step = pyo.Constraint(expr=own.x == own.x_in + own.u)
    own.obj = pyo.Objective(expr=price * own.u)
    inner_root = SddpScAlgRootBm(
        PyomoSolver(own, SolverConfig(SOLVER), [own.x]),
        state_params=[own.x_in],
    )
    # leaf role: the chain from stage 1, coupling on x_1
    lm = pyo.ConcreteModel()
    chain(lm, 2)
    lm.obj = pyo.Objective(expr=price * lm.u[2])
    inner_leaf = leaf_alg(lm, [lm.x[1]], future_vars=[lm.x[2]])
    inner = SddpScNodeInner(
        idx, inner_root, inner_leaf, log_level_root=0, log_level_leaf=0
    )
    inner.set_bound(BOUND)
    return inner


def build_nodes(stage2=((PRICE_2, (0.5, 0.5)),)):
    """stage2: one (buy price, probabilities over SELL_PRICES) per
    stage-2 node, each reached from stage 1 with equal probability."""
    # stage 1
    root_model = pyo.ConcreteModel()
    chain(root_model, 1)
    root_model.obj = pyo.Objective(expr=PRICE_1 * root_model.u[1])
    root = DecNodeRoot(
        0,
        SddpScAlgRootBm(
            PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.x[1]])
        ),
        log_level_root=0,
    )

    inners = []
    for k, (price, _) in enumerate(stage2):
        inner = make_inner(1 + k, price)
        root.add_child(inner.get_idx(), multiplier=1 / len(stage2))
        inners.append(inner)
    root.set_groups([[inner.get_idx() for inner in inners]])

    # stage 3: the chain from stage 1, coupling on x_2
    leaves = []
    for i, price in enumerate(SELL_PRICES):
        m = pyo.ConcreteModel()
        chain(m, 2)
        m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 2))
        m.sell = pyo.Constraint(expr=m.y <= m.x[2])
        m.obj = pyo.Objective(expr=-price * m.y)
        leaf = DecNodeLeaf(
            1 + len(stage2) + i, leaf_alg(m, [m.x[2]]), log_level_leaf=0
        )
        leaf.set_bound(BOUND)
        leaves.append(leaf)
    for inner, (_, probabilities) in zip(inners, stage2):
        for leaf, p in zip(leaves, probabilities):
            inner.add_child(leaf.get_idx(), multiplier=p)
        inner.set_groups([[leaf.get_idx() for leaf in leaves]])

    return [[root], inners, leaves], root


def optimum(stage2):
    """Buy at every stage -- each unit is worth at least 1.5 at the end,
    more than any buy price here -- and sell both units."""
    value = PRICE_1
    for price, probabilities in stage2:
        sell = sum(p * q for p, q in zip(probabilities, SELL_PRICES))
        value += (price - 2 * sell) / len(stage2)
    return value


def test_optimum_is_what_the_docstring_says():
    assert OPTIMUM == pytest.approx(-2.75)


def run_lattice(tmp_path, stage2=((PRICE_2, (0.5, 0.5)),), max_iteration=60):
    nodes, root = build_nodes(stage2)
    run = SddpScRun(
        nodes,
        tmp_path,
        level=logging.WARNING,
        max_iteration=max_iteration,
        sample_frequency=10,
        sample_size=10,
    )
    run.run()
    return run, root


def test_three_stage_lattice_prices_the_last_stage(tmp_path):
    run, root = run_lattice(tmp_path)

    # ignoring stage 3 this converges to 0, buying nothing
    assert run.graph.bound == pytest.approx(OPTIMUM, abs=1e-4)
    assert [round(v) for v in root.alg_root.solution] == [1]


def test_simulated_paths_pay_every_stage(tmp_path):
    run, _ = run_lattice(tmp_path)

    # buying both units, a path costs 1 + 0.25 - 2 * (its sell price)
    expected = {PRICE_1 + PRICE_2 - 2 * price for price in SELL_PRICES}
    assert {round(v, 6) for v in run.graph.simulation_samples} <= expected


def test_an_inner_node_needs_a_leaf_with_a_future():
    model = pyo.ConcreteModel()
    chain(model, 2)
    model.obj = pyo.Objective(expr=model.u[2])
    root_model = pyo.ConcreteModel()
    root_model.x = pyo.Var()
    root_model.obj = pyo.Objective(expr=root_model.x)
    root_alg = SddpScAlgRootBm(
        PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.x])
    )

    with pytest.raises(ValueError, match="future_vars"):
        SddpScNodeInner(1, root_alg, leaf_alg(model, [model.x[1]]))


# Two stage-2 nodes sharing the stage-3 children. Stagewise independent:
# the same probabilities from both, so a cut one earns is valid for both.
INDEPENDENT = ((0.0, (0.5, 0.5)), (0.25, (0.5, 0.5)))
# Markov: high prices are likelier after the cheap stage-2 node, so the
# two price different futures and must not share cuts.
MARKOV = ((0.0, (0.2, 0.8)), (0.25, (0.8, 0.2)))


@pytest.mark.parametrize(
    "stage2, expected",
    [(INDEPENDENT, -2.875), (MARKOV, -2.875)],
    ids=["independent", "markov"],
)
def test_two_nodes_per_stage_converge(tmp_path, stage2, expected):
    assert optimum(stage2) == pytest.approx(expected)

    run, root = run_lattice(tmp_path, stage2, max_iteration=100)

    assert run.graph.bound == pytest.approx(expected, abs=1e-4)
    assert [round(v) for v in root.alg_root.solution] == [1]


class FakeUp:
    """An up message as add_cuts reads it."""

    def __init__(self, c, origin, multipliers, rhs=0.0):
        from pyodsp.dec.bdsc.message import BdScUpMessage
        from pyodsp.alg.bm.cuts import OptimalityCut

        cut = OptimalityCut(coeffs={0: 0.0}, rhs=rhs, info={}, objective_value=rhs)
        self.message = BdScUpMessage(cut, c, 0.0, 0.0)
        self.message.set_origin(origin, multipliers)


def built_master(idx, multipliers):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 2))
    model.obj = pyo.Objective(expr=model.x)
    alg = SddpScAlgRootBm(PyomoSolver(model, SolverConfig(SOLVER), [model.x]))
    alg.set_logger(idx, 1, 0)
    alg.build([list(multipliers)], dict(multipliers), {k: BOUND for k in multipliers})
    return alg


def replies(c, origin, multipliers, children=(10, 11)):
    return {k: FakeUp(c, origin, multipliers).message for k in children}


def test_only_the_origin_takes_the_rho_step():
    probabilities = {10: 0.5, 11: 0.5}
    origin, sibling = built_master(1, probabilities), built_master(2, probabilities)
    rho = sibling.rho

    for master in (origin, sibling):
        master.add_cuts(replies(1.0, origin=1, multipliers=probabilities))

    assert origin.rho == pytest.approx(rho + 1.0)
    assert sibling.rho == pytest.approx(rho)


def test_a_sibling_with_the_same_future_takes_the_cut():
    probabilities = {10: 0.5, 11: 0.5}
    sibling = built_master(2, probabilities)

    sibling.add_cuts(replies(0.0, origin=1, multipliers=probabilities))

    assert sibling.bm.cpm.get_num_cuts() == 1


def test_a_sibling_with_a_different_future_leaves_the_replies_alone():
    sibling = built_master(2, {10: 0.8, 11: 0.2})
    rho = sibling.rho

    sibling.add_cuts(replies(0.0, origin=1, multipliers={10: 0.2, 11: 0.8}))
    sibling.add_cuts(replies(1.0, origin=1, multipliers={10: 0.2, 11: 0.8}))

    assert sibling.bm.cpm.get_num_cuts() == 0
    assert sibling.rho == pytest.approx(rho)


def test_replies_without_an_origin_are_the_masters_own():
    probabilities = {10: 0.5, 11: 0.5}
    master = built_master(1, probabilities)
    rho = master.rho

    master.add_cuts(replies(1.0, origin=None, multipliers=None))

    assert master.rho == pytest.approx(rho + 1.0)
