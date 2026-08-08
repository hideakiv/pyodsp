"""Carrying columns across trial points in bdsc.

Each cgmp cut records a column (x, theta) the cgsp produced, and bounds
alpha below by that column's value. The trial point y and the penalty rho
appear only in the cgmp's objective (MasterCreator.create), never in these
cuts — so a new trial point alone does not invalidate them. What does is
anything that shrinks the cgsp and can make a recorded column infeasible:
a cut the master added, or a higher floor under theta. A cut the master
merely purged relaxes the cgsp, so those columns survive.
"""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage
from pyodsp.dec.bdsc.run import BdScRun
from pyodsp.alg.bm.cuts import CutList, OptimalityCut

SOLVER = "appsi_highs"
FLOOR = -1e9


def make_leaf(max_iteration=20):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    leaf = BdScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig(SOLVER), [model.x]),
        SolverConfig("ipopt"),
        max_iteration,
    )
    leaf.set_logger(idx=0, depth=1, level=0)
    leaf.build()
    return leaf


def cut(rhs, coeff=1.0):
    return OptimalityCut(
        coeffs={0: coeff}, rhs=rhs, info={}, objective_value=rhs
    )


def message(solution=None, cut_list=None):
    return BdScDnMessage(
        solution=solution if solution is not None else [5.0],
        rho=1.0,
        cut_list=cut_list,
        subobj_bounds=[FLOOR],
        objective=0.0,
    )


def cgmp_cut_count(leaf):
    return leaf.cgmp.cpm.get_num_cuts()


def generate_columns(leaf, cut_list=None, solution=None):
    leaf.pass_dn_message(message(solution=solution, cut_list=cut_list))
    leaf.get_up_message()
    return list(leaf._cgmp_cuts)


def test_columns_are_generated_and_recorded():
    leaf = make_leaf()

    columns = generate_columns(leaf)

    assert columns, "column generation produced nothing to carry over"


def test_columns_carry_over_to_a_new_trial_point():
    leaf = make_leaf()
    columns = generate_columns(leaf)

    # a new trial point with the master's cuts unchanged
    leaf.pass_dn_message(message(solution=[7.0], cut_list=None))

    assert leaf._cgmp_cuts == columns
    assert cgmp_cut_count(leaf) == len(columns)


def test_columns_are_dropped_when_the_master_adds_a_cut():
    leaf = make_leaf()
    generate_columns(leaf, cut_list=[CutList([cut(-5.0)])])

    # the master's set gained a cut: the cgsp shrinks, so a recorded column
    # may no longer be feasible
    leaf.pass_dn_message(
        message(solution=[7.0], cut_list=[CutList([cut(-5.0), cut(-3.0, coeff=2.0)])])
    )

    assert leaf._cgmp_cuts == []
    assert cgmp_cut_count(leaf) == 0


def test_columns_survive_when_the_master_only_purges():
    leaf = make_leaf()
    both = [CutList([cut(-5.0), cut(-3.0, coeff=2.0)])]
    columns = generate_columns(leaf, cut_list=both)

    # the master dropped one cut and added none: the cgsp only relaxes
    leaf.pass_dn_message(message(solution=[7.0], cut_list=[CutList([cut(-5.0)])]))

    assert leaf._cgmp_cuts == columns
    assert cgmp_cut_count(leaf) == len(columns)


def test_a_changed_subproblem_bound_is_refused():
    # the floor under theta is fixed for the run; raising it would shrink the
    # cgsp and quietly invalidate every column collected so far
    leaf = make_leaf()
    generate_columns(leaf)

    changed = BdScDnMessage(
        solution=[7.0], rho=1.0, cut_list=None, subobj_bounds=[-10.0], objective=0.0
    )

    with pytest.raises(ValueError, match="subproblem bound changed"):
        leaf.pass_dn_message(changed)


# --- end to end: reuse must not change the answer -------------------------

R = 2
LOWER_BOUND = 1 / 4 - 1 / 32 / (1 + R / 2)
DELTA = 1 / 32 / (1 + R / 2)
CS_OPTIMUM = 3 * (3 / 4 - 1 / 32 / (1 + R / 2)) - 2


def build_caroe_schultz():
    """The instance from examples/bdsc/cs.py, whose optimum is known."""
    root_model = pyo.ConcreteModel()
    root_model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(LOWER_BOUND, 1))
    root_model.obj = pyo.Objective(expr=3 * root_model.x, sense=pyo.minimize)
    root_alg = BdScAlgRootBm(
        PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.x])
    )
    root = DecNodeRoot(0, root_alg, log_level_root=0)

    nodes, group = [root], []
    for s in range(1, R + 1):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(LOWER_BOUND, 1))
        model.y = pyo.Var(within=pyo.Binary)
        h = DELTA * s if s <= R / 2 else 1 / 4 - DELTA * (s - R / 2)
        model.c1 = pyo.Constraint(expr=-model.y / 2 >= h - model.x)
        model.obj = pyo.Objective(expr=-2 * model.y, sense=pyo.minimize)
        alg = BdScAlgLeafPyomo(
            PyomoSolver(model, SolverConfig(SOLVER), [model.x]), SolverConfig("ipopt")
        )
        leaf = DecNodeLeaf(s, alg, log_level_leaf=0)
        leaf.set_bound(-2.01)
        nodes.append(leaf)
        root.add_child(s, multiplier=1 / R)
        group.append(s)
    root.set_groups([group])
    return nodes, root_alg


def run_caroe_schultz(tmp_path, monkeypatch, reuse=True):
    if not reuse:
        # the previous behaviour: every cgmp starts with no columns at all
        original = BdScAlgLeafPyomo._create_master

        def without_reuse(self, solution, rho):
            self._cgmp_cuts = []
            original(self, solution, rho)

        monkeypatch.setattr(BdScAlgLeafPyomo, "_create_master", without_reuse)

    nodes, root_alg = build_caroe_schultz()
    BdScRun(nodes, Path(tmp_path), level=50).run()
    return root_alg.bm.obj_bound[-1]


def test_reuse_reaches_the_known_optimum(tmp_path, monkeypatch):
    bound = run_caroe_schultz(tmp_path, monkeypatch, reuse=True)

    assert bound == pytest.approx(CS_OPTIMUM, abs=1e-5)


def test_reuse_agrees_with_starting_every_master_empty(tmp_path, monkeypatch):
    with_reuse = run_caroe_schultz(tmp_path / "a", monkeypatch, reuse=True)
    without_reuse = run_caroe_schultz(tmp_path / "b", monkeypatch, reuse=False)

    assert with_reuse == pytest.approx(without_reuse, abs=1e-5)
