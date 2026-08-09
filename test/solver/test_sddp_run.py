import logging

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.sddp.run import SddpRun
from pyodsp.dec.sddp.run_mpi import SddpRunMpi

SOLVER = "appsi_highs"


def build_nodes():
    root_model = pyo.ConcreteModel()
    root_model.next_inventory = pyo.Var(bounds=(0, 10))
    root_model.obj = pyo.Objective(
        expr=1 * root_model.next_inventory, sense=pyo.minimize
    )
    root_alg = BdAlgRootBm(
        PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.next_inventory])
    )
    root = DecNodeRoot(0, root_alg, log_level_root=0)
    root.add_child(1, multiplier=1.0)

    leaf_model = pyo.ConcreteModel()
    leaf_model.prev_inventory = pyo.Var()
    leaf_model.buy = pyo.Var(bounds=(0, 10))
    leaf_model.demand = pyo.Constraint(
        expr=leaf_model.prev_inventory + leaf_model.buy >= 2
    )
    leaf_model.obj = pyo.Objective(expr=3 * leaf_model.buy, sense=pyo.minimize)
    leaf_alg = BdAlgLeafPyomo(
        PyomoSolver(leaf_model, SolverConfig(SOLVER), [leaf_model.prev_inventory])
    )
    leaf = DecNodeLeaf(1, leaf_alg)
    leaf.set_bound(0)

    return [[root], [leaf]]


def make_run(tmp_path, cls=SddpRun):
    return cls(
        build_nodes(),
        tmp_path,
        level=logging.WARNING,
        max_iteration=10,
        sample_frequency=5,
        sample_size=5,
    )


def test_run_rejects_an_initial_solution(tmp_path):
    # each iteration's forward pass starts from the root's own solve, so a
    # caller-supplied initial solution has nowhere to enter — it used to be
    # accepted and then silently discarded by Lattice.run
    run = make_run(tmp_path)

    with pytest.raises(NotImplementedError, match="does not support an initial"):
        run.run([1.0])


def test_mpi_run_rejects_an_initial_solution(tmp_path):
    # rejected before any collective, so this is safe on a single rank
    run = make_run(tmp_path, cls=SddpRunMpi)

    with pytest.raises(NotImplementedError, match="does not support an initial"):
        run.run([1.0])


def test_run_without_an_initial_solution_proceeds(tmp_path):
    run = make_run(tmp_path)

    run.run()

    assert (tmp_path / "node0" / "cuts.csv").exists()


def test_lattice_run_takes_no_initial_solution(tmp_path):
    # the graph itself no longer accepts an argument it would ignore
    run = make_run(tmp_path)

    with pytest.raises(TypeError):
        run.graph.run([1.0])


def build_maximize_nodes():
    """The same two-stage program stated as a maximization.

    Minimizing x + 3*max(0, 2-x) over x in [0, 10] has optimum 2 at x = 2;
    negating both stages gives a maximize problem whose optimum is -2.
    SDDP has no sense handling of its own — PyomoSolver converts these
    models when the solvers are built — so this checks that the whole
    lattice path, including Lattice's convergence test, still lands on the
    right answer and reports it in the sense the models were written in.
    """
    root_model = pyo.ConcreteModel()
    root_model.next_inventory = pyo.Var(bounds=(0, 10))
    root_model.obj = pyo.Objective(
        expr=-1 * root_model.next_inventory, sense=pyo.maximize
    )
    root_alg = BdAlgRootBm(
        PyomoSolver(root_model, SolverConfig(SOLVER), [root_model.next_inventory])
    )
    root = DecNodeRoot(0, root_alg, log_level_root=0)
    root.add_child(1, multiplier=1.0)

    leaf_model = pyo.ConcreteModel()
    leaf_model.prev_inventory = pyo.Var()
    leaf_model.buy = pyo.Var(bounds=(0, 10))
    leaf_model.demand = pyo.Constraint(
        expr=leaf_model.prev_inventory + leaf_model.buy >= 2
    )
    leaf_model.obj = pyo.Objective(expr=-3 * leaf_model.buy, sense=pyo.maximize)
    leaf_alg = BdAlgLeafPyomo(
        PyomoSolver(leaf_model, SolverConfig(SOLVER), [leaf_model.prev_inventory])
    )
    leaf = DecNodeLeaf(1, leaf_alg)
    # in the model's own (maximize) units: -3*buy is never above 0
    leaf.set_bound(0)

    return [[root], [leaf]]


def test_a_maximize_program_converges_and_reports_in_its_own_units(tmp_path):
    nodes = build_maximize_nodes()
    SddpRun(
        nodes,
        tmp_path,
        level=logging.WARNING,
        max_iteration=20,
        sample_frequency=5,
        sample_size=5,
    ).run()

    root_alg = nodes[0][0].alg_root
    assert root_alg.get_sense_multiplier() == -1.0
    assert root_alg.bm.get_objective_bound() == pytest.approx(-2.0, abs=1e-6)
    assert root_alg.get_vars()[0].value == pytest.approx(2.0, abs=1e-6)
