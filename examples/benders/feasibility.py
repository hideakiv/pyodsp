import pyomo.environ as pyo

from pyodec.dec.bd.node_root import BdRootNode
from pyodec.dec.bd.solver_root import BdSolverRoot
from pyodec.dec.bd.node_leaf import BdLeafNode
from pyodec.dec.bd.solver_leaf import BdSolverLeaf
from pyodec.dec.bd.run import BdRun

model1 = pyo.ConcreteModel()

model1.x1 = pyo.Var(within=pyo.NonNegativeReals)
model1.x2 = pyo.Var(within=pyo.NonNegativeReals)

model1.obj = pyo.Objective(expr=3 * model1.x1 + 2 * model1.x2, sense=pyo.minimize)

first_stage_solver = BdSolverRoot(model1, "appsi_highs")
coupling_dn = [model1.x1, model1.x2]
root_node = BdRootNode(0, first_stage_solver, coupling_dn)

model2 = {1: pyo.ConcreteModel(), 2: pyo.ConcreteModel()}
xi1 = {1: 6, 2: 4}
xi2 = {1: 8, 2: 4}
p = {1: 0.5, 2: 0.5}
leaf_nodes = {}

for i, block in model2.items():
    block.x1 = pyo.Var(within=pyo.Reals)
    block.x2 = pyo.Var(within=pyo.Reals)

    block.y1 = pyo.Var(within=pyo.NonNegativeReals)
    block.y2 = pyo.Var(within=pyo.NonNegativeReals)

    block.c1 = pyo.Constraint(expr=3 * block.y1 + 2 * block.y2 <= block.x1)
    block.c2 = pyo.Constraint(expr=2 * block.y1 + 5 * block.y2 <= block.x2)
    block.c3 = pyo.Constraint(expr=block.y1 >= 0.8 * xi1[i])
    block.c4 = pyo.Constraint(expr=block.y1 <= xi1[i])
    block.c5 = pyo.Constraint(expr=block.y2 >= 0.8 * xi2[i])
    block.c6 = pyo.Constraint(expr=block.y2 >= xi2[i])

    block.obj = pyo.Objective(expr=-15 * block.y1 - 12 * block.y2, sense=pyo.minimize)

    second_stage_solver = BdSolverLeaf(block, "appsi_highs")
    coupling_up = [block.x1, block.x2]
    leaf_node = BdLeafNode(i, second_stage_solver, 0.0, 0, coupling_up)
    leaf_nodes[i] = leaf_node
    root_node.add_child(i, multiplier=p[i])

root_node.build([[1, 2]])

bd_run = BdRun([root_node, *leaf_nodes.values()])
bd_run.run()
