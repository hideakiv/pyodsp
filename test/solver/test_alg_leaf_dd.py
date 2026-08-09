import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.message import DdInitDnMessage, DdDnMessage
from pyodsp.alg.bm.cuts import OptimalityCut, FeasibilityCut


def make_leaf(bounds=(-10, 10)):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=bounds)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.minimize)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    leaf = DdAlgLeafPyomo(solver)
    leaf.build()  # deactivates the original objective
    leaf.pass_init_dn_message(
        DdInitDnMessage(coupling_matrix=[{0: 1.0}])
    )
    return leaf, model


def test_optimal_subproblem_returns_optimality_cut():
    leaf, model = make_leaf(bounds=(-10, 10))

    # dual solution = 3.0 -> _mod_obj = x + 3*x = 4x, minimized over [-10, 10] -> x=-10
    leaf.pass_dn_message(DdDnMessage(solution=[3.0]))
    up_message = leaf.get_up_message()

    cut = up_message.get_cut()
    assert isinstance(cut, OptimalityCut)
    assert cut.objective_value == -40.0
    assert cut.coeffs == {0: 10.0}
    assert cut.rhs == -10.0
    assert cut.info["solution"] == [-10.0]


def test_unbounded_subproblem_returns_feasibility_cut():
    leaf, model = make_leaf(bounds=(0, None))

    # dual solution = -3.0 -> _mod_obj = x + (-3)*x = -2x, minimized over
    # [0, inf) with a negative coefficient -> unbounded below.
    leaf.pass_dn_message(DdDnMessage(solution=[-3.0]))
    up_message = leaf.get_up_message()

    cut = up_message.get_cut()
    assert isinstance(cut, FeasibilityCut)


def test_get_solution_or_ray_raises_on_unknown_status():
    leaf, model = make_leaf()

    class FakeSolver:
        def solve(self):
            pass

        def is_optimal(self):
            return False

        def is_unbounded(self):
            return False

    leaf.solver = FakeSolver()
    with pytest.raises(ValueError, match="Unknown solver status"):
        leaf.get_solution_or_ray()


def test_get_final_up_message_none_when_no_final_dn_message_received():
    leaf, model = make_leaf()

    message = leaf.get_final_up_message()

    assert message.get_objective() is None


def test_get_final_up_message_after_final_dn_message():
    from pyodsp.dec.dd.message import DdFinalDnMessage

    leaf, model = make_leaf(bounds=(-10, 10))
    # in production, pass_dn_message runs at least once during the main loop
    # (creating _mod_obj) before a final message is ever passed.
    leaf.pass_dn_message(DdDnMessage(solution=[3.0]))

    leaf.pass_final_dn_message(DdFinalDnMessage(solution=[3.0]))
    message = leaf.get_final_up_message()

    assert message.get_objective() == 3.0
