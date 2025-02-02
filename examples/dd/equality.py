import pyomo.environ as pyo

from pyodec.dec.dd.node_root import DdRootNode
from pyodec.dec.dd.node_leaf import DdLeafNode
from pyodec.dec.dd.solver_leaf import DdSolverLeaf
from pyodec.dec.dd.run import DdRun


def create_master() -> DdRootNode:
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)
    block.x3 = pyo.Var(within=pyo.Reals)
    vars_dn = {1: [block.x1], 2: [block.x2], 3: [block.x3]}

    block.c1 = pyo.Constraint(expr=3 * block.x1 + 2 * block.x2 + 4 * block.x3 == 17)

    root_node = DdRootNode(0, block, True, "appsi_highs", vars_dn)
    return root_node


cost = {1: -4, 2: -1, 3: -6}


def create_sub(i) -> DdLeafNode:
    block = pyo.ConcreteModel()
    block.x = pyo.Var(bounds=(1, 2))
    vars_up = [block.x]

    block.obj = pyo.Objective(expr=cost[i] * block.x, sense=pyo.minimize)

    sub_solver = DdSolverLeaf(block, "appsi_highs")
    leaf_node = DdLeafNode(i, sub_solver, 0, vars_up)
    return leaf_node


if __name__ == "__main__":
    master = create_master()
    sub_1 = create_sub(1)
    sub_2 = create_sub(2)
    sub_3 = create_sub(3)

    master.add_child(1)
    master.add_child(2)
    master.add_child(3)

    dd_run = DdRun([master, sub_1, sub_2, sub_3])
    dd_run.run()
