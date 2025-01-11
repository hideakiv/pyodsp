import pyomo.environ as pyo

from pyodec.core.subsolver.pyomo_subsolver import PyomoSubSolver
from pyodec.dec.bd.node_root import BdRootNode
from pyodec.dec.bd.node_leaf import BdLeafNode
from pyodec.dec.bd.run import BdRun

model1 = pyo.ConcreteModel()

model1.x1 = pyo.Var(within=pyo.NonNegativeReals)
model1.x2 = pyo.Var(within=pyo.NonNegativeReals)

model1.c1 = pyo.Constraint(expr=model1.x1 + model1.x2 <= 120)
model1.c2 = pyo.Constraint(expr=model1.x1 >= 40)
model1.c3 = pyo.Constraint(expr=model1.x2 >= 20)

model1.obj = pyo.Objective(expr=100 * model1.x1 + 150 * model1.x2, sense=pyo.minimize)

first_stage_solver = PyomoSubSolver(model1, "appsi_highs")
coupling_dn = [model1.x1, model1.x2]
root_node = BdRootNode(0, first_stage_solver, coupling_dn)

model2 = {1: pyo.ConcreteModel(), 2: pyo.ConcreteModel()}
d1 = {1: 500, 2: 300}
d2 = {1: 100, 2: 300}
q1 = {1: -24, 2: -28}
q2 = {1: -28, 2: -32}
p = {1: 0.4, 2: 0.6}
leaf_nodes = {}

for i, block in model2.items():
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

    second_stage_solver = PyomoSubSolver(block, "appsi_highs", use_dual=True)
    coupling_up = [block.x1, block.x2]
    leaf_node = BdLeafNode(i, second_stage_solver, 0, coupling_up, multiplier=p[i])
    leaf_nodes[i] = leaf_node
    root_node.add_child(i)

root_node.build(num_cuts=1)

bd_run = BdRun([root_node, *leaf_nodes.values()])
bd_run.run()
