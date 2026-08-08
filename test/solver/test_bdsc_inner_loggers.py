"""A bdsc leaf runs two inner solvers, the cgsp and the cgmp.

Neither is a node of its own, so both report at the depth of the leaf that
runs them and are told apart by a suffix on the id — rather than at a
hardcoded depth, and rather than under a fresh name per trial point, which
left a new logger (and handler) behind on every call.
"""

import logging

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage

SOLVER = "appsi_highs"


def make_leaf(idx=7, depth=3):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    leaf = BdScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig(SOLVER), [model.x]), SolverConfig("ipopt"), 20
    )
    leaf.set_logger(idx=idx, depth=depth, level=0)
    leaf.build()
    return leaf


def message(solution=None):
    return BdScDnMessage(
        solution=solution if solution is not None else [5.0],
        rho=1.0,
        cut_list=None,
        subobj_bounds=[-1e9],
        objective=0.0,
    )


def test_cgsp_logger_reports_at_the_leafs_depth():
    leaf = make_leaf(idx=7, depth=3)

    assert leaf.cgsp.logger.depth == 3
    assert leaf.cgsp.logger.node_id == "7_cgsp"


def test_cgmp_logger_reports_at_the_leafs_depth():
    leaf = make_leaf(idx=7, depth=3)

    leaf.pass_dn_message(message())

    assert leaf.cgmp.logger.depth == 3
    assert leaf.cgmp.logger.node_id == "7_cgmp"


def test_successive_masters_share_one_logger():
    leaf = make_leaf(idx=7, depth=3)

    leaf.pass_dn_message(message())
    first = leaf.cgmp.logger.logger
    leaf.pass_dn_message(message(solution=[9.0]))
    second = leaf.cgmp.logger.logger

    assert first is second
    assert len(second.handlers) == 1


def test_inner_loggers_do_not_collide_with_each_other():
    leaf = make_leaf(idx=7, depth=3)
    leaf.pass_dn_message(message())

    assert leaf.cgsp.logger.logger is not leaf.cgmp.logger.logger


def test_logger_names_are_registered_once_per_component():
    make_leaf(idx=11, depth=2).pass_dn_message(message())
    make_leaf(idx=11, depth=2).pass_dn_message(message())

    # rebuilding the leaf must not accumulate registry entries either
    named = [
        name
        for name in logging.root.manager.loggerDict
        if name.endswith("11_cgsp") or name.endswith("11_cgmp")
    ]
    assert sorted(named) == [
        "Bundle Method 11_cgsp",
        "Proximal Bundle Method 11_cgmp",
    ]
