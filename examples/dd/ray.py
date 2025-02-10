from pathlib import Path

import pyomo.environ as pyo

from pyodec.solver.pyomo_solver import PyomoSolver

from pyodec.dec.dd.node_root import DdRootNode
from pyodec.dec.dd.alg_root_bm import DdAlgRootBm
from pyodec.dec.dd.node_leaf import DdLeafNode
from pyodec.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodec.dec.dd.run import DdRun


def create_master() -> DdRootNode:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.y1 = pyo.Var(within=pyo.Reals)
    block.y2 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1, block.x2], 2: [block.y1, block.y2]}

    block.c1 = pyo.Constraint(
        expr=-1 * block.x1 + 5 * block.x2 + 7 * block.y1 - 6 * block.y2 == 1
    )

    root_alg = DdAlgRootBm(block, True, "appsi_highs", vars_dn)
    root_node = DdRootNode(0, root_alg)
    return root_node


cost = {1: [1, 2], 2: [3, 4]}


def create_sub(i) -> DdLeafNode:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    vars_up = [block.x1, block.x2]

    block.obj = pyo.Objective(
        expr=cost[i][0] * block.x1 + cost[i][1] * block.x2, sense=pyo.minimize
    )
    if i == 1:
        block.c1 = pyo.Constraint(expr=block.x1 + block.x2 >= 1)
        block.c2 = pyo.Constraint(expr=-block.x1 + 2 * block.x2 >= 0)
        block.c3 = pyo.Constraint(expr=block.x2 >= 1)
    elif i == 2:
        block.c1 = pyo.Constraint(expr=3 * block.x1 - block.x2 >= 2)
        block.c2 = pyo.Constraint(expr=block.x1 + 2 * block.x2 >= 3)

    sub_solver = PyomoSolver(block, "appsi_highs", vars_up)
    sub_alg = DdAlgLeafPyomo(sub_solver)
    leaf_node = DdLeafNode(i, sub_alg, 0)
    return leaf_node


if __name__ == "__main__":
    master = create_master()
    sub_1 = create_sub(1)
    sub_2 = create_sub(2)

    master.add_child(1)
    master.add_child(2)

    dd_run = DdRun([master, sub_1, sub_2], Path("output/dd/ray"))
    dd_run.run()
