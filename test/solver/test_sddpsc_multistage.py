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


def build_nodes():
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
    root.add_child(1, multiplier=1.0)
    root.set_groups([[1]])

    # stage 2, root role: x_1 arrives as a parameter
    own = pyo.ConcreteModel()
    own.x_in = pyo.Param(mutable=True, initialize=0.0)
    own.u = pyo.Var(within=pyo.Binary)
    own.x = pyo.Var(bounds=(0, 2))
    own.step = pyo.Constraint(expr=own.x == own.x_in + own.u)
    own.obj = pyo.Objective(expr=PRICE_2 * own.u)
    inner_root = SddpScAlgRootBm(
        PyomoSolver(own, SolverConfig(SOLVER), [own.x]),
        state_params=[own.x_in],
    )
    # stage 2, leaf role: the chain from stage 1, coupling on x_1
    lm = pyo.ConcreteModel()
    chain(lm, 2)
    lm.obj = pyo.Objective(expr=PRICE_2 * lm.u[2])
    inner_leaf = leaf_alg(lm, [lm.x[1]], future_vars=[lm.x[2]])
    inner = SddpScNodeInner(
        1, inner_root, inner_leaf, log_level_root=0, log_level_leaf=0
    )
    inner.set_bound(BOUND)

    # stage 3: the chain from stage 1, coupling on x_2
    leaves = []
    for i, price in enumerate(SELL_PRICES):
        m = pyo.ConcreteModel()
        chain(m, 2)
        m.y = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, 2))
        m.sell = pyo.Constraint(expr=m.y <= m.x[2])
        m.obj = pyo.Objective(expr=-price * m.y)
        leaf = DecNodeLeaf(2 + i, leaf_alg(m, [m.x[2]]), log_level_leaf=0)
        leaf.set_bound(BOUND)
        inner.add_child(2 + i, multiplier=1 / len(SELL_PRICES))
        leaves.append(leaf)
    inner.set_groups([[leaf.get_idx() for leaf in leaves]])

    return [[root], [inner], leaves], root


def test_optimum_is_what_the_docstring_says():
    assert OPTIMUM == pytest.approx(-2.75)


def run_lattice(tmp_path):
    nodes, root = build_nodes()
    run = SddpScRun(
        nodes,
        tmp_path,
        level=logging.WARNING,
        max_iteration=60,
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


@pytest.mark.xfail(
    strict=True,
    reason="an inner node's root role never sees its ancestors' cost: "
    "pass_dn_message sets _parent_objective on the leaf role's model only",
)
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
