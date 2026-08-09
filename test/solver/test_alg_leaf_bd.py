import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.message import BdDnMessage
from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut


def make_leaf(x_bounds=(0, 100)):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))  # coupling var
    model.s = pyo.Var(domain=pyo.Reals, bounds=x_bounds)
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    leaf = BdAlgLeafPyomo(solver)
    leaf.build()
    return leaf, model


def test_optimal_subproblem_returns_optimality_cut():
    leaf, model = make_leaf()

    leaf.pass_dn_message(BdDnMessage(solution=[5.0], objective=1.0))
    up_message = leaf.get_up_message()

    cut = up_message.get_cut()
    assert isinstance(cut, OptimalityCut)
    assert cut.objective_value == 5.0
    assert up_message.get_objective() == 6.0  # parent(1.0) + original(5.0)


def test_infeasible_subproblem_returns_feasibility_cut():
    leaf, model = make_leaf(x_bounds=(0, 100))

    # x fixed to 200 forces s >= 200, but s is bounded to [0, 100] -> infeasible
    leaf.pass_dn_message(BdDnMessage(solution=[200.0], objective=0.0))
    up_message = leaf.get_up_message()

    cut = up_message.get_cut()
    assert isinstance(cut, FeasibilityCut)
    assert up_message.get_objective() is None


def test_get_subgradient_inner_raises_on_unknown_status():
    leaf, model = make_leaf()

    class FakeSolver:
        def is_optimal(self):
            return False

        def is_infeasible(self):
            return False

    leaf.solver = FakeSolver()
    with pytest.raises(ValueError, match="Unknown solver status"):
        leaf._get_subgradient_inner()
