import pyomo.environ as pyo

from pyodec.dec.bd.node_root import BdRootNode
from pyodec.dec.bd.solver_root import BdSolverRoot
from pyodec.dec.bd.node_leaf import BdLeafNode
from pyodec.dec.bd.solver_leaf import BdSolverLeaf
from pyodec.dec.bd.run import BdRun


def create_root_node():
    model1 = pyo.ConcreteModel()

    model1.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model1.x2 = pyo.Var(within=pyo.NonNegativeReals)

    model1.c1 = pyo.Constraint(expr=model1.x1 + model1.x2 <= 120)
    model1.c2 = pyo.Constraint(expr=model1.x1 >= 40)
    model1.c3 = pyo.Constraint(expr=model1.x2 >= 20)

    model1.obj = pyo.Objective(
        expr=100 * model1.x1 + 150 * model1.x2, sense=pyo.minimize
    )

    first_stage_solver = BdSolverRoot(model1, "appsi_highs")
    coupling_dn = [model1.x1, model1.x2]
    root_node = BdRootNode(0, first_stage_solver, coupling_dn)
    return root_node


d1 = {1: 500, 2: 300}
d2 = {1: 100, 2: 300}
q1 = {1: -24, 2: -28}
q2 = {1: -28, 2: -32}
p = {1: 0.4, 2: 0.6}


def create_leaf_node(i):
    block = pyo.ConcreteModel()
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)

    block.y1 = pyo.Var(within=pyo.NonNegativeReals)
    block.y2 = pyo.Var(within=pyo.NonNegativeReals)

    block.c1 = pyo.Constraint(expr=6 * block.y1 + 10 * block.y2 <= 60 * block.x1)
    block.c2 = pyo.Constraint(expr=8 * block.y1 + 5 * block.y2 <= 80 * block.x2)
    block.c3 = pyo.Constraint(expr=block.y1 <= d1[i])
    block.c4 = pyo.Constraint(expr=block.y2 <= d2[i])

    block.obj = pyo.Objective(
        expr=q1[i] * block.y1 + q2[i] * block.y2, sense=pyo.minimize
    )

    second_stage_solver = BdSolverLeaf(block, "appsi_highs")
    coupling_up = [block.x1, block.x2]
    leaf_node = BdLeafNode(i, second_stage_solver, -30000, 0, coupling_up)
    return leaf_node


if __name__ == "__main__":
    root_node = create_root_node()
    leaf_node_1 = create_leaf_node(1)
    leaf_node_2 = create_leaf_node(2)

    root_node.add_child(1, multiplier=p[1])
    root_node.add_child(2, multiplier=p[2])

    root_node.set_groups([[1, 2]])

    bd_run = BdRun([root_node, leaf_node_1, leaf_node_2])
    bd_run.run()
