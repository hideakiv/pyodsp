from pathlib import Path
from mpi4py import MPI
import pyomo.environ as pyo

from aircon import first_stage, mid_stage, last_stage
from utils import assert_approximately_equal

import pyodsp

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf, DecNodeInner
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.sddp.run_mpi import SddpRunMpi
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

"""
mpiexec -n 4 python sddp_mpi.py

Builds the IDENTICAL lattice on every rank (unlike bd/dd's *_mpi examples,
which split distinct nodes across ranks) — only rank 0's alg_root objects
may purge; every other rank's are purgeable=False replicas used solely for
the parallel Monte Carlo simulation (see LatticeMpi).
"""


def main(solver="appsi_highs", agg=False):
    # pyodsp emits log records but installs no handler; opt in to see progress.
    pyodsp.configure_logging()
    rank = MPI.COMM_WORLD.Get_rank()
    purgeable = rank == 0
    demand = [[1], [1, 3], [1, 3]]

    node00 = create_root(0, demand[0][0], solver, purgeable, agg)
    node10 = create_inner(1, demand[1][0], solver, purgeable, agg)
    node11 = create_inner(2, demand[1][1], solver, purgeable, agg)
    node20 = create_leaf(3, demand[2][0], solver)
    node21 = create_leaf(4, demand[2][1], solver)

    nodes = [[node00], [node10, node11], [node20, node21]]

    sddp_run = SddpRunMpi(nodes, Path("output/aircon/sddp_mpi"))
    sddp_run.run()

    if rank == 0:
        assert_approximately_equal(nodes[0][0].alg_root.bm.obj_bound[-1], 6.25)


def create_root(idx, demand, solver_name, purgeable, agg=False):
    model = pyo.ConcreteModel()
    first_stage(model, demand)
    coupling_dn = [model.next_inventory]
    config = SolverConfig(solver_name=solver_name)
    solver_root = PyomoSolver(model, config, coupling_dn)
    alg_root = BdAlgRootBm(solver_root, purgeable=purgeable)
    node = DecNodeRoot(idx, alg_root)
    node.add_child(1, multiplier=0.5)
    node.add_child(2, multiplier=0.5)
    if agg:
        node.set_groups([[1, 2]])
    return node


def create_inner(idx, demand, solver_name, purgeable, agg=False):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    mid_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.prev_inventory]
    coupling_dn = [model.next_inventory]
    config = SolverConfig(solver_name=solver_name)
    solver_root = PyomoSolver(model, config, coupling_dn)
    alg_root = BdAlgRootBm(solver_root, max_iteration=1, purgeable=purgeable)
    solver_leaf = PyomoSolver(model, config, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    node = DecNodeInner(idx, alg_root, alg_leaf, log_level_root=0)
    node.set_bound(0)
    node.add_child(3, multiplier=0.5)
    node.add_child(4, multiplier=0.5)
    if agg:
        node.set_groups([[3, 4]])
    return node


def create_leaf(idx, demand, solver_name):
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    last_stage(model, model.prev_inventory, demand)
    model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
    coupling_up = [model.prev_inventory]
    config = SolverConfig(solver_name=solver_name)
    solver_leaf = PyomoSolver(model, config, coupling_up)
    alg_leaf = BdAlgLeafPyomo(solver_leaf)
    node = DecNodeLeaf(idx, alg_leaf)
    node.set_bound(0)
    return node


if __name__ == "__main__":
    main(agg=True)
