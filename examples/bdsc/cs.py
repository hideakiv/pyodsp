"""
Caroe and Schultz (1999) instance

"""

from pathlib import Path

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.run import BdScRun

from utils import assert_approximately_equal


def first_stage(model: pyo.ConcreteModel, r: int):
    lb = 1 / 4 - 1 / 32 / (1 + r / 2)
    model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(lb, 1))
    model.obj_expr1 = 3 * model.x


def second_stage(model: pyo.ConcreteModel, s: int, r: int):
    delta = 1 / 32 / (1 + r / 2)
    if s <= r / 2:
        h = delta * s
    else:
        h = 1 / 4 - delta * (s - r / 2)
    model.y = pyo.Var(within=pyo.Binary)

    model.c1 = pyo.Constraint(expr=-model.y / 2 >= h - model.x)
    model.obj_expr2 = -2 * model.y


def create_root_node(r: int, solver="appsi_highs"):
    model = pyo.ConcreteModel()

    first_stage(model, r)

    model.obj = pyo.Objective(expr=model.obj_expr1, sense=pyo.minimize)

    coupling_dn = [model.x]
    config = SolverConfig(solver_name=solver)
    first_stage_solver = PyomoSolver(model, config, coupling_dn)
    first_stage_alg = BdScAlgRootBm(first_stage_solver)
    root_node = DecNodeRoot(0, first_stage_alg)
    return root_node


def create_leaf_node(i: int, r: int, solver="appsi_highs"):
    model = pyo.ConcreteModel()

    first_stage(model, r)
    second_stage(model, s=i, r=r)
    model.obj = pyo.Objective(expr=model.obj_expr2, sense=pyo.minimize)

    coupling_up = [model.x]
    config = SolverConfig(solver_name=solver)
    second_stage_solver = PyomoSolver(model, config, coupling_up)
    master_config = SolverConfig(solver_name="ipopt")
    second_stage_alg = BdScAlgLeafPyomo(second_stage_solver, master_config)
    leaf_node = DecNodeLeaf(i, second_stage_alg, log_level_leaf=0)
    leaf_node.set_bound(-2.01)
    return leaf_node


def main():

    r = 2

    nodes = []
    group = []
    root_node = create_root_node(r)
    nodes.append(root_node)
    for i in range(r):
        leaf_node = create_leaf_node(i + 1, r)
        nodes.append(leaf_node)
        root_node.add_child(i + 1, multiplier=1 / r)
        group.append(i + 1)
    root_node.set_groups([group])

    bd_run = BdScRun(nodes, Path("output/bdsc/cs"))
    bd_run.run()

    # optimal = 3 * (3 / 4 - 1 / 32 / (1 + r / 2)) - 2
    optimal = 0.2031

    assert_approximately_equal(root_node.alg_root.bm.obj_bound[-1], optimal)


if __name__ == "__main__":
    main()
