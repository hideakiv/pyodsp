"""Maximize problems, end to end, with no conversion step by the caller.

BD/BDSC/SDDP/DD are minimize-only internally. PyomoSolver converts a
maximize model on construction — before it captures original_objective,
so that attribute and CuttingPlaneMethod's cached sign both describe the
converted problem — and remembers the flip so results come back in the
units the model was written in.

Each optimum below is hand-computed in the *user's* maximize convention
and asserted directly, with nothing negated by the test.

The internal masters are the exception and stay as they are: dual
decomposition's Lagrangian master and BDSC's pricing master are maximize
problems on purpose, opt out of the conversion, and are turned into
minimizations by CuttingPlaneMethod's sign flip as before.
"""

from pathlib import Path

import pandas as pd
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
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.run import BdScRun

CONSTRUCTORS = [
    lambda s: BdAlgRootBm(s),
    lambda s: BdAlgLeafPyomo(s),
    lambda s: DdAlgLeafPyomo(s),
    lambda s: BdScAlgRootBm(s),
    lambda s: BdScAlgLeafPyomo(s, SolverConfig("ipopt")),
]
CONSTRUCTOR_IDS = ["bd_root", "bd_leaf", "dd_leaf", "bdsc_root", "bdsc_leaf"]


def _make_maximize_solver(convert_maximize=True):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)
    return PyomoSolver(
        model,
        SolverConfig("appsi_highs"),
        [model.x],
        convert_maximize=convert_maximize,
    )


# -- every algorithm now accepts a maximize model ---------------------------


@pytest.mark.parametrize("construct", CONSTRUCTORS, ids=CONSTRUCTOR_IDS)
def test_algorithms_accept_a_maximize_model_and_remember_the_flip(construct):
    alg = construct(_make_maximize_solver())

    assert alg.is_minimize() is True
    assert alg.get_sense_multiplier() == -1.0


@pytest.mark.parametrize("construct", CONSTRUCTORS, ids=CONSTRUCTOR_IDS)
def test_algorithms_still_reject_an_unconverted_maximize_model(construct):
    # convert_maximize=False is reserved for the internal masters, so an
    # algorithm handed such a solver indicates a bug rather than a user
    # error — but it must not be solved as though it were a minimization.
    with pytest.raises(ValueError, match="needs a minimize model"):
        construct(_make_maximize_solver(convert_maximize=False))


# -- Benders ----------------------------------------------------------------

# True problem: maximize -x + theta1 + theta2, x in [0, 10]; leaf i
# (i=1,2): maximize y_i s.t. y_i <= 5 - x, y_i in [0, 100].
# g(x) = max(0, 5-x); f(x) = -x + 2*g(x); optimum at x*=0, f*=10.


def _bd_create_root():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.obj = pyo.Objective(expr=-model.x, sense=pyo.maximize)
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
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.prev_x])
    node = DecNodeLeaf(idx, BdAlgLeafPyomo(solver))
    # In the model's own (maximize) units: y never exceeds 100.
    node.set_bound(100)
    return node


def test_benders_maximize_matches_hand_computed_optimum(tmp_path):
    root = _bd_create_root()
    nodes = [root, _bd_create_leaf(1), _bd_create_leaf(2)]
    BdRun(nodes, tmp_path, max_iteration=100).run()

    assert root.alg_root.get_vars()[0].value == pytest.approx(0.0, abs=1e-4)
    assert root.alg_root.bm.get_objective_bound() == pytest.approx(10.0, abs=1e-4)


def test_benders_maximize_saves_its_trajectory_in_the_users_units(tmp_path):
    root = _bd_create_root()
    nodes = [root, _bd_create_leaf(1), _bd_create_leaf(2)]
    BdRun(nodes, tmp_path, max_iteration=100).run()

    saved = pd.read_csv(Path(tmp_path) / "node0" / "bm.csv", index_col=0)

    # No post-hoc correction: the file is written in the maximize
    # convention the models were declared in.
    assert saved["obj_bound"].iloc[-1] == pytest.approx(10.0, abs=1e-4)


def test_a_leaf_written_in_the_other_sense_is_rejected(tmp_path):
    # The senses used to be compared after conversion, where they were
    # trivially equal. The check now compares what the user wrote.
    root = _bd_create_root()

    model = pyo.ConcreteModel()
    model.prev_x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    model.y = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.con = pyo.Constraint(expr=model.y <= 5 - model.prev_x)
    model.obj = pyo.Objective(expr=-model.y, sense=pyo.minimize)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.prev_x])
    minimize_leaf = DecNodeLeaf(1, BdAlgLeafPyomo(solver))

    nodes = [root, minimize_leaf, _bd_create_leaf(2)]

    with pytest.raises(ValueError, match="Inconsistent optimization sense"):
        BdRun(nodes, tmp_path, max_iteration=100).run()


# -- dual decomposition -----------------------------------------------------

COST = {1: 4, 2: 1, 3: 6}


def _create_dd_master(mode):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.x3 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1], 2: [block.x2], 3: [block.x3]}
    block.c1 = pyo.Constraint(expr=3 * block.x1 + 2 * block.x2 + 4 * block.x3 == 17)

    solver_name = "ipopt" if mode == "proximal" else "appsi_highs"
    root_alg = DdAlgRootBm(block, SolverConfig(solver_name), vars_dn, mode=mode)
    return DecNodeRoot(0, root_alg)


def _create_dd_sub(idx):
    block = pyo.ConcreteModel()
    block.x = pyo.Var(bounds=(1, 2))
    block.obj = pyo.Objective(expr=COST[idx] * block.x, sense=pyo.maximize)
    solver = PyomoSolver(block, SolverConfig("appsi_highs"), [block.x])
    return DecNodeLeaf(idx, DdAlgLeafPyomo(solver))


def _run_dd(tmp_path, mode):
    master = _create_dd_master(mode)
    subs = [_create_dd_sub(i) for i in (1, 2, 3)]
    for sub in subs:
        master.add_child(sub.get_idx())
    DdRun([master] + subs, tmp_path, max_iteration=200).run()
    return master


@pytest.mark.parametrize("mode", [None, "proximal"])
def test_dual_decomposition_maximize_matches_hand_computed_optimum(tmp_path, mode):
    # True problem: maximize 4*x1 + x2 + 6*x3 s.t. 3*x1+2*x2+4*x3=17, 1<=xi<=2.
    # Ratio test (coeff/constraint-coeff): x3=1.5 > x1=1.333 > x2=0.5, so
    # x1, x3 sit at their upper bound (2) and x2 absorbs the remaining slack.
    # Optimal objective = 4*2 + 1*1.5 + 6*2 = 21.5 (the coupling constraint is
    # only satisfied in aggregate at the optimal dual price — a degenerate
    # subproblem may report any feasible x2 in [1, 2], so only the bound is
    # asserted here, not the individual leaf solutions).
    master = _run_dd(tmp_path, mode)

    # This root has no user model of its own — its master is the Lagrangian
    # dual, synthesized from the coupling constraints — so the sense it
    # reports in is the one its scenarios told it.
    assert master.alg_root.get_sense_multiplier() == -1.0
    assert master.alg_root.bm.get_objective_bound() == pytest.approx(21.5, abs=1e-3)


def test_dual_decomposition_maximize_saves_in_the_users_units(tmp_path):
    _run_dd(tmp_path, None)

    saved = pd.read_csv(Path(tmp_path) / "node0" / "bm.csv", index_col=0)
    assert saved["obj_bound"].iloc[-1] == pytest.approx(21.5, abs=1e-3)


def test_dual_decomposition_rejects_scenarios_written_in_mixed_senses(tmp_path):
    """The one root that cannot check against a model of its own.

    A Benders root compares a child's sense with its own master's. Dual
    decomposition's master is synthesized from the coupling constraints
    and is a maximize problem no matter what the user wrote, so the
    scenarios are the only authority — the first sets the sense and the
    rest must agree, or which units results come back in would depend on
    iteration order.
    """
    master = _create_dd_master(None)

    minimize_sub = pyo.ConcreteModel()
    minimize_sub.x = pyo.Var(bounds=(1, 2))
    minimize_sub.obj = pyo.Objective(expr=-COST[2] * minimize_sub.x, sense=pyo.minimize)
    odd_one_out = DecNodeLeaf(
        2,
        DdAlgLeafPyomo(
            PyomoSolver(minimize_sub, SolverConfig("appsi_highs"), [minimize_sub.x])
        ),
    )

    subs = [_create_dd_sub(1), odd_one_out, _create_dd_sub(3)]
    for sub in subs:
        master.add_child(sub.get_idx())

    with pytest.raises(ValueError, match="Inconsistent optimization sense"):
        DdRun([master] + subs, tmp_path, max_iteration=50).run()


# -- Benders with scaled cuts -----------------------------------------------


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


def test_bdsc_maximize_matches_caroe_schultz_optimum(tmp_path):
    # Caroe & Schultz (1999) instance (examples/bdsc/cs.py), whose minimize
    # optimum is 0.2031. Stated here as the maximize problem the user
    # actually wants — maximize -(obj_expr1 + obj_expr2) — whose optimum is
    # therefore -0.2031.
    r = 2

    def create_root():
        model = pyo.ConcreteModel()
        _first_stage(model, r)
        model.obj = pyo.Objective(expr=-model.obj_expr1, sense=pyo.maximize)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
        return DecNodeRoot(0, BdScAlgRootBm(solver))

    def create_leaf(i):
        model = pyo.ConcreteModel()
        _first_stage(model, r)
        _second_stage(model, s=i, r=r)
        model.obj = pyo.Objective(expr=-model.obj_expr2, sense=pyo.maximize)
        solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
        leaf = BdScAlgLeafPyomo(solver, SolverConfig("ipopt"))
        node = DecNodeLeaf(i, leaf, log_level_leaf=0)
        # In the model's own (maximize) units: -obj_expr2 = 2*y <= 2.01.
        node.set_bound(2.01)
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

    assert root_node.alg_root.bm.get_objective_bound() == pytest.approx(
        -0.2031, abs=1e-3
    )
