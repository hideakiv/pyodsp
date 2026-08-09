import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage
from pyodsp.alg.bm.cuts import OptimalityCut


def make_leaf(max_iteration=20):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))  # coupling var
    model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])

    # the column-generation master problem gets a quadratic proximal term
    # (ProximalBundleMethod), which appsi_highs cannot solve; ipopt can.
    leaf = BdScAlgLeafPyomo(solver, SolverConfig("ipopt"), max_iteration)
    leaf.set_logger(idx=0, depth=1, level=20)
    leaf.build()
    return leaf, model


def test_a_maximize_model_is_accepted_and_reported_in_its_own_sense():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])

    leaf = BdScAlgLeafPyomo(solver, SolverConfig("ipopt"))

    assert leaf.is_minimize() is True
    assert leaf.get_sense_multiplier() == -1.0


def test_construction_rejects_an_unconverted_maximize_model():
    # convert_maximize=False is reserved for the internal masters; an
    # algorithm handed such a solver is a bug, not a user error.
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)
    solver = PyomoSolver(
        model, SolverConfig("appsi_highs"), [model.x], convert_maximize=False
    )

    with pytest.raises(ValueError, match="needs a minimize model"):
        BdScAlgLeafPyomo(solver, SolverConfig("ipopt"))


def test_get_up_message_converges_and_returns_optimality_cut():
    leaf, model = make_leaf()

    leaf.pass_dn_message(
        BdScDnMessage(
            solution=[5.0], rho=1.0, cut_list=None, subobj_bounds=[-1e9], objective=0.0
        )
    )

    up_message = leaf.get_up_message()

    assert isinstance(up_message.get_cut(), OptimalityCut)
    # subproblem is `min s s.t. s >= x`, so with x fixed at 5 the true
    # optimum is s=5; the column-generation loop should recover that value.
    assert up_message.get_objective() == pytest.approx(5.0, abs=1e-4)
