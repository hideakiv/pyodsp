import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.run import BdScRun


def make_run(tmp_path):
    root_model = pyo.ConcreteModel()
    root_model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    root_model.obj = pyo.Objective(expr=root_model.x, sense=pyo.minimize)
    root_alg = BdScAlgRootBm(
        PyomoSolver(root_model, SolverConfig("appsi_highs"), [root_model.x])
    )
    root = DecNodeRoot(0, root_alg, log_level_root=0)

    leaf_model = pyo.ConcreteModel()
    leaf_model.x = pyo.Var(domain=pyo.Reals, bounds=(0, 10))
    leaf_model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    leaf_model.obj = pyo.Objective(expr=leaf_model.s, sense=pyo.minimize)
    leaf_model.con = pyo.Constraint(expr=leaf_model.s >= leaf_model.x)
    leaf_alg = BdScAlgLeafPyomo(
        PyomoSolver(leaf_model, SolverConfig("appsi_highs"), [leaf_model.x]),
        SolverConfig("ipopt"),
    )
    leaf = DecNodeLeaf(1, leaf_alg, log_level_leaf=0)
    leaf.set_bound(0)
    root.add_child(1, multiplier=1.0)
    root.set_groups([[1]])

    return BdScRun([root, leaf], tmp_path, level=50, max_iteration=2)


def test_run_rejects_an_initial_solution(tmp_path):
    # a dn message here carries the master's rho, subproblem bounds and cut
    # set alongside the solution, so there is nothing coherent to send before
    # the master has stepped — reject it rather than build a broken message
    run = make_run(tmp_path)

    with pytest.raises(NotImplementedError, match="does not support an initial"):
        run.run([1.0])


def test_run_without_an_initial_solution_proceeds(tmp_path):
    run = make_run(tmp_path)

    run.run()

    assert (tmp_path / "node0" / "cuts.csv").exists()
