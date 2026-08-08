"""End-to-end maximize coverage for the bundle method's sign-flip handling.

CuttingPlaneMethod internally treats every problem as a minimization,
negating cuts/theta/objective values when the true sense is maximize (see
alg/bm/cp.py). These tests solve small hand-computed maximize problems
through the real decomposition wiring (Tree/BdRun for Benders, HubAndSpoke/
DdRun for dual decomposition, covering both BundleMethod and
ProximalBundleMethod) to catch sign errors that mocked unit tests would miss.
"""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.run import DdRun


def test_benders_maximize_matches_hand_computed_optimum(tmp_path):
    # Root: maximize -x + theta1 + theta2, x in [0, 10].
    # Leaf i (i=1,2): maximize y_i s.t. y_i <= 5 - x, y_i in [0, 100].
    # g(x) = max(0, 5-x); f(x) = -x + 2*g(x); optimum at x*=0, f*=10.
    def create_root():
        model = pyo.ConcreteModel()
        model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
        model.obj = pyo.Objective(expr=-model.x, sense=pyo.maximize)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
        node = DecNodeRoot(0, BdAlgRootBm(solver))
        node.add_child(1, multiplier=1.0)
        node.add_child(2, multiplier=1.0)
        return node

    def create_leaf(idx):
        model = pyo.ConcreteModel()
        model.prev_x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
        model.y = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
        model.con = pyo.Constraint(expr=model.y <= 5 - model.prev_x)
        model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.prev_x])
        node = DecNodeLeaf(idx, BdAlgLeafPyomo(solver))
        node.set_bound(100)
        return node

    root = create_root()
    nodes = [root, create_leaf(1), create_leaf(2)]
    BdRun(nodes, tmp_path, max_iteration=100).run()

    assert root.alg_root.get_vars()[0].value == pytest.approx(0.0, abs=1e-4)
    assert root.alg_root.bm.obj_bound[-1] == pytest.approx(10.0, abs=1e-4)


COST = {1: 4, 2: 1, 3: 6}


def _create_dd_master(mode):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.x3 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1], 2: [block.x2], 3: [block.x3]}
    block.c1 = pyo.Constraint(expr=3 * block.x1 + 2 * block.x2 + 4 * block.x3 == 17)

    solver_name = "ipopt" if mode == "proximal" else "appsi_highs"
    root_alg = DdAlgRootBm(
        block, False, SolverConfig(solver_name), vars_dn, mode=mode
    )
    return DecNodeRoot(0, root_alg)


def _create_dd_sub(idx):
    block = pyo.ConcreteModel()
    block.x = pyo.Var(bounds=(1, 2))
    block.obj = pyo.Objective(expr=COST[idx] * block.x, sense=pyo.maximize)
    solver = PyomoSolver(block, SolverConfig("appsi_highs"), [block.x])
    return DecNodeLeaf(idx, DdAlgLeafPyomo(solver))


@pytest.mark.parametrize("mode", [None, "proximal"])
def test_dual_decomposition_maximize_matches_hand_computed_optimum(tmp_path, mode):
    # maximize 4*x1 + x2 + 6*x3 s.t. 3*x1+2*x2+4*x3=17, 1<=xi<=2.
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

    assert master.alg_root.bm.obj_bound[-1] == pytest.approx(21.5, abs=1e-3)
