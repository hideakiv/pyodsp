"""BD/BDSC/DD only accept minimize problems (see alg_root_bm.py /
alg_leaf_pyomo.py __init__ guards) — a maximize problem must be converted
to an equivalent minimize one first, via pyomo_utils.negate_objective_sense,
with the final reported objective negated back. This module verifies both
halves: that the guards reject maximize outright, and that the negate-step
pattern reproduces hand-computed maximize optima end-to-end.

alg/bm/ itself still handles maximize internally (CuttingPlaneMethod's
sign-flip) — that's what makes DD's and BDSC's own dual-sense-inverted
master problems work even though the *user-facing* sense is always
minimize now. This module only exercises the user-facing restriction.
"""

from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.solver.pyomo_utils import negate_objective_sense, negate_saved_objective_csv
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.run import BdScRun


def _make_maximize_solver():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)
    return PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])


def test_bd_root_rejects_maximize():
    with pytest.raises(ValueError, match="only accepts minimize"):
        BdAlgRootBm(_make_maximize_solver())


def test_bd_leaf_rejects_maximize():
    with pytest.raises(ValueError, match="only accepts minimize"):
        BdAlgLeafPyomo(_make_maximize_solver())


def test_dd_root_rejects_maximize():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    with pytest.raises(ValueError, match="only accepts minimize"):
        DdAlgRootBm(model, False, SolverConfig("appsi_highs"), {0: [model.x]})


def test_dd_leaf_rejects_maximize():
    with pytest.raises(ValueError, match="only accepts minimize"):
        DdAlgLeafPyomo(_make_maximize_solver())


def test_bdsc_root_rejects_maximize():
    with pytest.raises(ValueError, match="only accepts minimize"):
        BdScAlgRootBm(_make_maximize_solver())


def test_bdsc_leaf_rejects_maximize():
    with pytest.raises(ValueError, match="only accepts minimize"):
        BdScAlgLeafPyomo(_make_maximize_solver(), SolverConfig("ipopt"))


# True problem: maximize -x + theta1 + theta2, x in [0, 10]; leaf i
# (i=1,2): maximize y_i s.t. y_i <= 5 - x, y_i in [0, 100].
# g(x) = max(0, 5-x); f(x) = -x + 2*g(x); optimum at x*=0, f*=10.
def _bd_create_root():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.obj = pyo.Objective(expr=-model.x, sense=pyo.maximize)
    negate_objective_sense(model)  # now: minimize x
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    node = DecNodeRoot(0, BdAlgRootBm(solver))
    node.add_child(1, multiplier=1.0)
    node.add_child(2, multiplier=1.0)
    return node


def _bd_create_leaf(idx):
    model = pyo.ConcreteModel()
    model.prev_x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.y = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.con = pyo.Constraint(expr=model.y <= 5 - model.prev_x)
    model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)
    negate_objective_sense(model)  # now: minimize -y
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.prev_x])
    node = DecNodeLeaf(idx, BdAlgLeafPyomo(solver))
    node.set_bound(-100)  # negated from the true (maximize) bound of 100
    return node


def test_benders_maximize_via_negate_step_matches_hand_computed_optimum(tmp_path):
    root = _bd_create_root()
    nodes = [root, _bd_create_leaf(1), _bd_create_leaf(2)]
    BdRun(nodes, tmp_path, max_iteration=100).run()

    assert root.alg_root.get_vars()[0].value == pytest.approx(0.0, abs=1e-4)
    # the algorithm solved the negated (minimize) problem; negate the
    # reported bound back to recover the true maximize objective.
    assert -root.alg_root.bm.obj_bound[-1] == pytest.approx(10.0, abs=1e-4)


def test_benders_maximize_saved_bound_needs_negate_saved_objective_csv(tmp_path):
    root = _bd_create_root()
    nodes = [root, _bd_create_leaf(1), _bd_create_leaf(2)]
    BdRun(nodes, tmp_path, max_iteration=100).run()

    node_dir = tmp_path / "node0"
    saved = pd.read_csv(node_dir / "bm.csv", index_col=0)
    # run() saves whatever the (negated) model actually produced — the raw
    # file is in the internal convention, not the true maximize one.
    assert saved["obj_bound"].iloc[-1] == pytest.approx(-10.0, abs=1e-4)

    negate_saved_objective_csv(node_dir)

    fixed = pd.read_csv(node_dir / "bm.csv", index_col=0)
    assert fixed["obj_bound"].iloc[-1] == pytest.approx(10.0, abs=1e-4)


COST = {1: 4, 2: 1, 3: 6}


def _create_dd_master(mode):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.x3 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1], 2: [block.x2], 3: [block.x3]}
    block.c1 = pyo.Constraint(expr=3 * block.x1 + 2 * block.x2 + 4 * block.x3 == 17)

    solver_name = "ipopt" if mode == "proximal" else "appsi_highs"
    root_alg = DdAlgRootBm(block, True, SolverConfig(solver_name), vars_dn, mode=mode)
    return DecNodeRoot(0, root_alg)


def _create_dd_sub(idx):
    block = pyo.ConcreteModel()
    block.x = pyo.Var(bounds=(1, 2))
    block.obj = pyo.Objective(expr=COST[idx] * block.x, sense=pyo.maximize)
    negate_objective_sense(block)  # now: minimize -COST[idx]*x
    solver = PyomoSolver(block, SolverConfig("appsi_highs"), [block.x])
    return DecNodeLeaf(idx, DdAlgLeafPyomo(solver))


@pytest.mark.parametrize("mode", [None, "proximal"])
def test_dual_decomposition_maximize_via_negate_step_matches_hand_computed_optimum(
    tmp_path, mode
):
    # True problem: maximize 4*x1 + x2 + 6*x3 s.t. 3*x1+2*x2+4*x3=17, 1<=xi<=2.
    # Ratio test (coeff/constraint-coeff): x3=1.5 > x1=1.333 > x2=0.5, so
    # x1, x3 sit at their upper bound (2) and x2 absorbs the remaining slack.
    # Optimal objective = 4*2 + 1*1.5 + 6*2 = 21.5 (the coupling constraint is
    # only satisfied in aggregate at the optimal dual price — a degenerate
    # subproblem may report any feasible x2 in [1, 2], so only the bound is
    # asserted here, not the individual leaf solutions).
    master = _create_dd_master(mode)
    subs = [_create_dd_sub(i) for i in (1, 2, 3)]
    for sub in subs:
        master.add_child(sub.get_idx())

    DdRun([master] + subs, tmp_path, max_iteration=200).run()

    # DD's master is a Lagrangian dual and is unaffected by the negate step
    # (see module docstring); only the leaves were negated, so the reported
    # bound must be negated back to recover the true maximize objective.
    assert -master.alg_root.bm.obj_bound[-1] == pytest.approx(21.5, abs=1e-3)


def _first_stage(model, r):
    lb = 1 / 4 - 1 / 32 / (1 + r / 2)
    model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(lb, 1))
    model.obj_expr1 = 3 * model.x


def _second_stage(model, s, r):
    delta = 1 / 32 / (1 + r / 2)
    h = delta * s if s <= r / 2 else 1 / 4 - delta * (s - r / 2)
    model.y = pyo.Var(within=pyo.Binary)
    model.c1 = pyo.Constraint(expr=-model.y / 2 >= h - model.x)
    model.obj_expr2 = -2 * model.y


def test_bdsc_maximize_via_negate_step_matches_caroe_schultz_optimum(tmp_path):
    # Caroe & Schultz (1999) instance (examples/bdsc/cs.py), known minimize
    # optimum 0.2031. Framing it as the "true" problem the user actually
    # wants to *maximize* is the negation: maximize -(obj_expr1+obj_expr2).
    # Negating that back to minimize reproduces cs.py's original model
    # exactly, so the algorithm (which only ever sees this — the one
    # already-working combination of senses) still applies unmodified; the
    # true maximize optimum is -0.2031.
    r = 2

    def create_root():
        model = pyo.ConcreteModel()
        _first_stage(model, r)
        model.obj = pyo.Objective(expr=-model.obj_expr1, sense=pyo.maximize)
        negate_objective_sense(model)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
        return DecNodeRoot(0, BdScAlgRootBm(solver))

    def create_leaf(i):
        model = pyo.ConcreteModel()
        _first_stage(model, r)
        _second_stage(model, s=i, r=r)
        model.obj = pyo.Objective(expr=-model.obj_expr2, sense=pyo.maximize)
        negate_objective_sense(model)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
        leaf = BdScAlgLeafPyomo(solver, SolverConfig("ipopt"))
        node = DecNodeLeaf(i, leaf, log_level_leaf=0)
        # true (maximize-convention) bound is 2.01 (-obj_expr2 <= 2.01,
        # since cs.py established obj_expr2 >= -2.01); negate it the same
        # way the model was negated, matching cs.py's own -2.01.
        node.set_bound(-2.01)
        return node

    root_node = create_root()
    nodes = [root_node]
    group = []
    for i in range(r):
        leaf_node = create_leaf(i + 1)
        nodes.append(leaf_node)
        root_node.add_child(i + 1, multiplier=1 / r)
        group.append(i + 1)
    root_node.set_groups([group])

    BdScRun(nodes, tmp_path).run()

    assert -root_node.alg_root.bm.obj_bound[-1] == pytest.approx(-0.2031, abs=1e-3)
